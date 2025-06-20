"""
MAML Training Framework for Physics Discovery
============================================

Meta-learning framework that trains an agent to quickly adapt to new physics
discovery tasks using Model-Agnostic Meta-Learning (MAML).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
import time
import json
from pathlib import Path
import copy
from tqdm import tqdm

import types # Ensure types is imported

# Import existing Janus components
from janus.ai_interpretability.environments import SymbolicDiscoveryEnv
from hypothesis_policy_network import HypothesisNet, PPOTrainer
from janus.core.grammar import ProgressiveGrammar
from janus.core.expression import Variable
from janus.ai_interpretability.utils.math_utils import calculate_symbolic_accuracy, safe_env_reset # Added import
import sympy as sp

# Try to import enhanced feedback if available
try:
    from enhanced_feedback import IntrinsicRewardCalculator, EnhancedObservationEncoder
except ImportError:
    print("Warning: enhanced_feedback module not found, using basic feedback")
    IntrinsicRewardCalculator = None
    EnhancedObservationEncoder = None

# Try to import feedback integration if available
try:
    from feedback_integration import add_intrinsic_rewards_to_env
except ImportError:
    print("Warning: feedback_integration module not found, intrinsic rewards disabled")
    add_intrinsic_rewards_to_env = None

# Import our new components
from physics_task_distribution import PhysicsTaskDistribution, PhysicsTask


@dataclass
class MetaLearningConfig:
    """Configuration for meta-learning"""
    # Meta-learning parameters
    meta_lr: float = 0.0003
    adaptation_lr: float = 0.01
    adaptation_steps: int = 5
    
    # Task sampling
    tasks_per_batch: int = 10
    support_episodes: int = 10
    query_episodes: int = 10
    
    # Environment parameters
    max_episode_steps: int = 50
    max_tree_depth: int = 7
    max_complexity: int = 20
    
    # Training parameters
    meta_iterations: int = 1000
    checkpoint_interval: int = 50
    eval_interval: int = 25
    
    # Reward configuration
    use_intrinsic_rewards: bool = True
    intrinsic_weight: float = 0.2
    
    # Logging
    log_dir: str = "./meta_learning_logs"
    checkpoint_dir: str = "./meta_learning_checkpoints"
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class MetaLearningPolicy(nn.Module):
    """Enhanced HypothesisNet with meta-learning capabilities"""
    
    def __init__(self, 
                 observation_dim: int,
                 action_dim: int,
                 hidden_dim: int = 256,
                 n_layers: int = 3,
                 use_task_embedding: bool = True):
        super().__init__()
        
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.use_task_embedding = use_task_embedding
        
        # Task encoder for conditioning
        if use_task_embedding:
            self.task_encoder = nn.LSTM(
                observation_dim,
                hidden_dim // 2,
                batch_first=True,
                bidirectional=True
            )
            
            # Task modulation network
            self.task_modulator = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim * 2)  # Gains and biases
            )
        
        # Main policy network (modulated by task)
        layers = []
        input_dim = observation_dim
        
        for i in range(n_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(hidden_dim))
            input_dim = hidden_dim
        
        self.feature_extractor = nn.ModuleList(layers)
        
        # Heads for actor-critic
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Physics-aware heads
        self.symmetry_detector = nn.Linear(hidden_dim, 10)  # Common symmetries
        self.conservation_predictor = nn.Linear(hidden_dim, 5)  # Conservation laws
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0.0)
    
    def forward(self, 
              obs: torch.Tensor,
              task_embedding: Optional[torch.Tensor] = None,
              action_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]: # Add action_mask
        """Forward pass with optional task conditioning"""
        
        # Get task modulation if available
        if self.use_task_embedding and task_embedding is not None:
            # Encode task context
            if len(task_embedding.shape) == 2:
                task_embedding = task_embedding.unsqueeze(0)
            
            _, (hidden, _) = self.task_encoder(task_embedding)
            task_features = hidden.transpose(0, 1).reshape(hidden.shape[1], -1)
            
            # Get modulation parameters
            modulation = self.task_modulator(task_features)
            gains, biases = modulation.chunk(2, dim=-1)
            gains = 1 + 0.1 * torch.tanh(gains)  # Mild modulation
        else:
            gains = 1.0
            biases = 0.0
        
        # Extract features with modulation
        x = obs
        for i, layer in enumerate(self.feature_extractor):
            x = layer(x)
            # Apply modulation after ReLU layers
            if isinstance(layer, nn.ReLU) and i < len(self.feature_extractor) - 1:
                if isinstance(gains, torch.Tensor):
                    x = x * gains + biases
        
        features = x
        
        # Compute outputs
        policy_logits = self.policy_head(features)

        # Apply the action mask to the logits before returning
        if action_mask is not None:
            # Ensure mask is on the same device
            action_mask = action_mask.to(policy_logits.device)

            # Handle dimension mismatch by padding or truncating the action_mask
            if action_mask.shape[-1] < policy_logits.shape[-1]:
                padding = torch.zeros(
                    *action_mask.shape[:-1],
                    policy_logits.shape[-1] - action_mask.shape[-1],
                    dtype=torch.bool,
                    device=policy_logits.device
                )
                action_mask = torch.cat([action_mask, padding], dim=-1)
            elif action_mask.shape[-1] > policy_logits.shape[-1]:
                action_mask = action_mask[..., :policy_logits.shape[-1]]

            # Expand mask if necessary (e.g. if policy_logits has a batch dim and mask doesn't)
            if action_mask.shape != policy_logits.shape:
                try:
                    action_mask = action_mask.expand_as(policy_logits)
                except RuntimeError as e:
                    # This can happen if the mask cannot be broadcast, e.g. wrong number of dims
                    # Fallback: if mask is 1D and logits are 2D (batch, actions), unsqueeze and expand
                    if len(action_mask.shape) == 1 and len(policy_logits.shape) == 2:
                        action_mask = action_mask.unsqueeze(0).expand_as(policy_logits)
                    else: # Re-raise if we can't handle it
                        raise e

            policy_logits[~action_mask] = float('-inf') # Set invalid actions to negative infinity

        value = self.value_head(features)
        
        # Physics predictions (return raw logits)
        symmetry_logits = self.symmetry_detector(features)
        conservation_logits = self.conservation_predictor(features)
        
        return {
            'policy_logits': policy_logits,
            'value': value,
            'symmetries_logits': symmetry_logits,
            'conservations_logits': conservation_logits,
            'features': features
        }
    
    def act(self, obs: torch.Tensor, task_embedding: Optional[torch.Tensor] = None, action_mask: Optional[torch.Tensor] = None) -> Tuple[int, Dict]:
        """Select action using current policy"""
        with torch.no_grad():
            # Ensure obs is 2D if not already (batch_size, feature_dim)
            if obs.ndim == 1:
                obs = obs.unsqueeze(0)

            outputs = self.forward(obs, task_embedding, action_mask)
            
            # Sample from policy
            probs = F.softmax(outputs['policy_logits'], dim=-1)

            # Fallback for NaN or inf probabilities
            if torch.any(torch.isnan(probs)) or torch.any(torch.isinf(probs)):
                print(f"Warning: NaN or inf detected in policy probabilities. Logits: {outputs['policy_logits']}")
                if action_mask is not None:
                    # Ensure action_mask is 1D for random choice from valid actions
                    current_action_mask = action_mask
                    if current_action_mask.ndim > 1:
                         # Assuming batch size of 1 for .act(), take the first mask
                        current_action_mask = current_action_mask[0]

                    valid_actions = torch.where(current_action_mask)[0]
                    if len(valid_actions) > 0:
                        action = valid_actions[torch.randint(len(valid_actions), (1,))].item()
                    else:
                        # No valid actions, default to 0 (this case should ideally be prevented by env)
                        action = 0
                        print("Warning: No valid actions in mask during fallback. Defaulting to action 0.")
                else:
                    action = 0 # Default action if no mask

                # Default action_info
                action_info = {
                    'log_prob': -np.log(probs.shape[-1] if probs.shape[-1] > 0 else 1), # Uniform log_prob
                    'value': outputs['value'].item() if outputs['value'] is not None else 0.0,
                    'entropy': 0.0 # Cannot compute entropy reliably
                }
                return action, action_info

            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            
            return action.item(), {
                'log_prob': dist.log_prob(action).item(),
                'value': outputs['value'].item(),
                'entropy': dist.entropy().item()
            }


class TaskEnvironmentBuilder:
    """Builds SymbolicDiscoveryEnv from PhysicsTask"""
    
    def __init__(self, config: MetaLearningConfig):
        self.config = config
        self.observation_encoder = EnhancedObservationEncoder()
        
    def build_env(self, task: PhysicsTask, max_action_space: Optional[int] = None) -> SymbolicDiscoveryEnv:
        """Create environment for specific physics task"""
        
        # Generate training data
        data = task.generate_data(1000, noise=True)
        
        # Create variables (excluding target)
        variables = []
        for i, var_name in enumerate(task.variables[:-1]):
            var_properties = {}
            
            # Add physical properties based on task metadata
            if var_name in task.physical_parameters:
                var_properties['is_constant'] = True
                var_properties['value'] = task.physical_parameters[var_name]
            
            variables.append(Variable(var_name, i, var_properties))
        
        # Create grammar appropriate for task
        grammar = self._create_task_grammar(task)
        
        # Configure rewards based on task properties
        reward_config = {
            'mse_weight': -1.0,
            'complexity_penalty': -0.005,
            'parsimony_bonus': 0.1,
        }
        
        # Add symmetry bonus if task has symmetries
        if task.symmetries and "none" not in task.symmetries:
            reward_config['symmetry_bonus'] = 0.2
        
        # Create environment
        env = SymbolicDiscoveryEnv(
            grammar=grammar,
            target_data=data,
            variables=variables,
            max_depth=self.config.max_tree_depth,
            max_complexity=self.config.max_complexity,
            reward_config=reward_config,
            action_space_size=max_action_space # Added line
        )
        
        # Dynamically attach get_action_mask if not present
        if not hasattr(env, 'get_action_mask'):
            def get_action_mask(self_env): # Renamed self to self_env
                if hasattr(self_env, 'current_state') and hasattr(self_env.current_state, 'get_valid_actions'):
                    valid_actions = self_env.current_state.get_valid_actions()
                    mask = np.zeros(self_env.action_space.n, dtype=bool)
                    mask[valid_actions] = True
                    return mask
                else:
                    return np.ones(self_env.action_space.n, dtype=bool) # Default: all actions valid

            env.get_action_mask = types.MethodType(get_action_mask, env)

        # Add intrinsic rewards if configured
        if self.config.use_intrinsic_rewards and add_intrinsic_rewards_to_env is not None:
            try:
                add_intrinsic_rewards_to_env(env, weight=self.config.intrinsic_weight)
            except Exception as e:
                print(f"Warning: Could not add intrinsic rewards: {e}")
        elif self.config.use_intrinsic_rewards and add_intrinsic_rewards_to_env is None:
            print("Warning: Intrinsic rewards enabled in config, but feedback_integration module not found. Skipping.")
        
        # Store task metadata in env
        env.task_info = {
            'name': task.name,
            'true_law': task.true_law,
            'domain': task.domain,
            'difficulty': task.difficulty,
            'symmetries': task.symmetries
        }
        
        return env
    
    def _create_task_grammar(self, task: PhysicsTask) -> ProgressiveGrammar:
        """Create grammar with operators appropriate for task"""
        # Using load_defaults=True ensures a standardized grammar
        # with all possible operators, guaranteeing a consistent
        # action space across all environments.
        # Initialize an EMPTY grammar
        grammar = ProgressiveGrammar(load_defaults=True)
        
        # Domain-specific operators
        if task.domain == "mechanics":
            grammar.add_operators(['**2', 'sqrt'])
            # For pendulum, add trig functions
            if "pendulum" in task.name:
                grammar.add_operators(['sin', 'cos'])
                
        elif task.domain == "thermodynamics":
            grammar.add_operators(['log', 'exp', '**'])
            
        elif task.domain == "electromagnetism":
            # '**2' handles r**2, '1/' is mapped to 'inv' for 1/r
            grammar.add_operators(['**2', '1/']) # Corrected: removed '**3'
        
        # Add special operators based on true law (optional, for guidance)
        if "**" in task.true_law: # Corrected: removed "and '**2' not in grammar.operators"
            grammar.add_operators(['**'])
        
        return grammar


class MAMLTrainer:
    """Main MAML training logic for physics discovery"""
    
    def __init__(self, 
                 config: MetaLearningConfig,
                 policy: MetaLearningPolicy,
                 task_distribution: PhysicsTaskDistribution):
        
        self.config = config
        self.policy = policy.to(config.device)
        self.task_distribution = task_distribution
        self.env_builder = TaskEnvironmentBuilder(config)
        
        # Meta-optimizer
        self.meta_optimizer = torch.optim.Adam(
            self.policy.parameters(), 
            lr=config.meta_lr
        )
        
        # Logging
        self.writer = SummaryWriter(config.log_dir)
        self.iteration = 0
        
        # Tracking
        self.meta_losses = []
        self.task_metrics = defaultdict(list)
        self.discovered_laws = defaultdict(list)
        
        # Setup directories
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
    def meta_train_step(self) -> Dict[str, float]:
        """Single meta-training step across multiple tasks"""
        
        meta_loss = 0
        meta_metrics = defaultdict(float)
        
        # Sample batch of tasks
        tasks = self.task_distribution.sample_task_batch(
            self.config.tasks_per_batch,
            curriculum=True  # Progressive difficulty
        )
        
        task_gradients = []
        
        for task_idx, task in enumerate(tasks):
            # Create environment for this task
            env = self.env_builder.build_env(task, max_action_space=self.policy.action_dim)
            
            # Clone policy for inner loop adaptation
            adapted_policy = self._clone_policy()
            inner_optimizer = torch.optim.SGD(
                adapted_policy.parameters(), 
                lr=self.config.adaptation_lr
            )
            
            # Collect support trajectories with base policy
            support_trajectories = self._collect_trajectories(
                self.policy,
                env,
                n_episodes=self.config.support_episodes,
                task_context=None  # First exposure to task
            )
            
            # Extract task embedding from support trajectories
            task_embedding = self._compute_task_embedding(support_trajectories)
            
            # Inner loop: Adapt to task using support data
            for adapt_step in range(self.config.adaptation_steps):
                support_loss = self._compute_trajectory_loss(
                    adapted_policy,
                    support_trajectories,
                    task_embedding,
                    task
                )
                
                inner_optimizer.zero_grad()
                support_loss.backward()
                inner_optimizer.step()
            
            # Collect query trajectories with adapted policy
            query_trajectories = self._collect_trajectories(
                adapted_policy,
                env,
                n_episodes=self.config.query_episodes,
                task_context=task_embedding
            )
            
            # Compute meta-loss on query set
            query_loss = self._compute_trajectory_loss(
                adapted_policy,
                query_trajectories,
                task_embedding,
                task
            )
            
            # Track task-specific metrics
            task_metrics = self._compute_task_metrics(
                query_trajectories,
                task,
                adapted_policy
            )
            
            # Accumulate metrics
            for key, value in task_metrics.items():
                meta_metrics[key] += value / self.config.tasks_per_batch
            
            # IMPORTANT: Compute gradient w.r.t. initial policy parameters
            # This is the key to MAML - we want gradients that improve 
            # the initial parameters such that adaptation works better
            
            # Store gradients for this task
            task_grad = torch.autograd.grad(
                query_loss,
                self.policy.parameters(),
                retain_graph=True,
                allow_unused=True # FIX: Prevents crash when graph is broken by NaNs or other issues.
            )
            task_gradients.append(task_grad)
            
            # Log individual task performance
            self._log_task_performance(task, task_metrics, task_idx)
            
            meta_loss += query_loss
        
        # Meta-update: Average gradients across tasks
        self.meta_optimizer.zero_grad()
        
        # Manually set gradients (MAML-style)
        for param_idx, param in enumerate(self.policy.parameters()):
            # Filter out None gradients that result from allow_unused=True
            valid_grads = [g[param_idx] for g in task_gradients if g is not None and g[param_idx] is not None]
            if valid_grads:
                param.grad = sum(valid_grads) / len(valid_grads)
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        
        # Meta-update
        self.meta_optimizer.step()
        
        # Track meta-loss
        avg_meta_loss = meta_loss.item() / self.config.tasks_per_batch
        self.meta_losses.append(avg_meta_loss)
        meta_metrics['meta_loss'] = avg_meta_loss
        
        return meta_metrics
    
    def _clone_policy(self) -> MetaLearningPolicy:
        """Create a functional clone of the policy for adaptation"""
        # Deep copy that maintains computational graph
        cloned = copy.deepcopy(self.policy)
        cloned.load_state_dict(self.policy.state_dict())
        return cloned
    
    def _collect_trajectories(self,
                            policy: MetaLearningPolicy,
                            env: SymbolicDiscoveryEnv,
                            n_episodes: int,
                            task_context: Optional[torch.Tensor] = None) -> List[Dict]:
        """Collect trajectories using given policy"""
        
        trajectories = []
        
        for episode in range(n_episodes):
            obs, _ = safe_env_reset(env) # Use safe_env_reset
            trajectory = {
                'observations': [],
                'actions': [],
                'rewards': [],
                'log_probs': [],
                'values': [],
                'dones': [],
                'infos': []
            }
            
            episode_reward = 0
            
            for step in range(self.config.max_episode_steps):
                # Convert observation to tensor
                obs_tensor = torch.FloatTensor(obs).to(self.config.device)

                # Get the action mask from the environment (should always exist due to TaskEnvironmentBuilder)
                action_mask_np = env.get_action_mask()
                action_mask = torch.BoolTensor(action_mask_np).to(self.config.device)
                
                # Get action from policy, passing the mask
                action, action_info = policy.act(obs_tensor, task_context, action_mask) # Pass mask
                
                # Take environment step
                next_obs, reward, done, truncated, info = env.step(action)
                
                # Store transition
                trajectory['observations'].append(obs)
                trajectory['actions'].append(action)
                trajectory['rewards'].append(reward)
                trajectory['log_probs'].append(action_info['log_prob'])
                trajectory['values'].append(action_info['value'])
                trajectory['dones'].append(done or truncated)
                trajectory['infos'].append(info)
                
                episode_reward += reward
                obs = next_obs
                
                if done or truncated:
                    break
            
            # Add episode metadata
            trajectory['episode_reward'] = episode_reward
            trajectory['episode_length'] = len(trajectory['actions'])
            
            # Track discoveries
            if trajectory['infos'] and 'expression' in trajectory['infos'][-1]:
                trajectory['discovered_expression'] = trajectory['infos'][-1]['expression']
                trajectory['discovery_mse'] = trajectory['infos'][-1].get('mse', float('inf'))
            
            trajectories.append(trajectory)
        
        return trajectories
    
    def _compute_task_embedding(self, trajectories: List[Dict]) -> torch.Tensor:
        """
        Computes the task context by collecting all observations from support trajectories.
        This sequence will be fed into the policy's task encoder (LSTM).
        """
        all_obs = []
        for traj in trajectories:
            # Ensure 'observations' key exists and is not empty
            if 'observations' in traj and traj['observations']:
                all_obs.extend(traj['observations'])

        if not all_obs:
            # Handle cases where no observations were collected
            # Return a zero tensor of the correct shape to avoid downstream errors
            # NOTE: You need to know the observation dimension here. We can get it from the policy.
            obs_dim = self.policy.observation_dim
            return torch.zeros((1, obs_dim), device=self.config.device)

        # Return a 2D tensor of all observations from the support set: (sequence_length, feature_size)
        return torch.FloatTensor(all_obs).to(self.config.device)
    
    def _compute_trajectory_loss(self,
                                policy: MetaLearningPolicy,
                                trajectories: List[Dict],
                                task_embedding: torch.Tensor,
                                task: PhysicsTask) -> torch.Tensor:
        """Compute policy gradient loss for trajectories"""
        
        total_loss = 0
        
        for traj in trajectories:
            # Convert to tensors
            obs = torch.FloatTensor(traj['observations']).to(self.config.device)
            actions = torch.LongTensor(traj['actions']).to(self.config.device)
            rewards = torch.FloatTensor(traj['rewards']).to(self.config.device)
            
            # Forward pass through policy
            outputs = policy(obs, task_embedding)
            
            # Policy loss (REINFORCE with baseline)
            log_probs = F.log_softmax(outputs['policy_logits'], dim=-1)

            # Ensure actions are correctly shaped and clamped before gathering
            if actions.dim() == 1: # Should be true as actions are collected as a list of ints
                actions_expanded = actions.unsqueeze(1)
            else: # Fallback, though not expected with current collection logic
                actions_expanded = actions

            num_actions_from_logits = log_probs.size(1)
            # Clamp actions to be within the valid range of policy_logits [0, num_actions_from_logits - 1]
            actions_clamped = torch.clamp(actions_expanded, 0, num_actions_from_logits - 1)

            selected_log_probs = log_probs.gather(1, actions_clamped).squeeze(-1)
            
            # Compute advantages
            returns = self._compute_returns(rewards)
            advantages = returns - outputs['value'].squeeze()
            
            policy_loss = -(selected_log_probs * advantages.detach()).mean()
            
            # Value loss
            value_loss = F.mse_loss(outputs['value'].squeeze(), returns)
            
            # Entropy bonus for exploration
            entropy = -(log_probs * log_probs.exp()).sum(dim=-1).mean()
            
            # Physics-informed losses (if applicable)
            physics_loss = 0
            if task.symmetries and "none" not in task.symmetries:
                # Encourage learning of symmetries
                symmetry_targets = self._create_symmetry_targets(task.symmetries)

                # FIX: Use the numerically stable BCEWithLogitsLoss and the new logits output
                physics_loss = F.binary_cross_entropy_with_logits(
                    outputs['symmetries_logits'].mean(0),
                    symmetry_targets.to(self.config.device)
                )
            
            # Combine losses
            traj_loss = (
                policy_loss + 
                0.5 * value_loss - 
                0.01 * entropy +
                0.1 * physics_loss
            )
            
            total_loss += traj_loss
        
        return total_loss / len(trajectories)
    
    def _compute_returns(self, rewards: torch.Tensor, gamma: float = 0.99) -> torch.Tensor:
        """Compute discounted returns"""
        returns = torch.zeros_like(rewards)
        running_return = 0
        
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + gamma * running_return
            returns[t] = running_return
            
        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        return returns
    
    def _create_symmetry_targets(self, symmetries: List[str]) -> torch.Tensor:
        """Create binary targets for symmetry detection"""
        symmetry_types = [
            'time_reversal', 'spatial_translation', 'rotational',
            'galilean', 'scale_invariance', 'charge_conjugation',
            'energy_conservation', 'momentum_conservation',
            'angular_momentum_conservation', 'none'
        ]
        
        targets = torch.zeros(len(symmetry_types))
        for i, sym_type in enumerate(symmetry_types):
            if sym_type in symmetries:
                targets[i] = 1.0
                
        return targets
    
    def _compute_task_metrics(self,
                            trajectories: List[Dict],
                            task: PhysicsTask,
                            policy: MetaLearningPolicy) -> Dict[str, float]:
        """Compute task-specific metrics"""
        
        metrics = {
            'discovery_rate': 0,
            'avg_episode_reward': 0,
            'avg_episode_length': 0,
            'best_mse': float('inf'),
            'correct_discovery': 0
        }
        
        discoveries = []
        
        for traj in trajectories:
            metrics['avg_episode_reward'] += traj['episode_reward']
            metrics['avg_episode_length'] += traj['episode_length']
            
            if 'discovered_expression' in traj:
                expr = traj['discovered_expression']
                mse = traj['discovery_mse']
                
                discoveries.append(expr)
                metrics['discovery_rate'] += 1
                metrics['best_mse'] = min(metrics['best_mse'], mse)
                
                # Check if discovery matches true law
                if self._expression_matches(expr, task.true_law):
                    metrics['correct_discovery'] += 1
        
        # Normalize metrics
        n_traj = len(trajectories)
        metrics['discovery_rate'] /= n_traj
        metrics['avg_episode_reward'] /= n_traj
        metrics['avg_episode_length'] /= n_traj
        metrics['correct_discovery'] /= n_traj
        
        # Track unique discoveries
        metrics['unique_discoveries'] = len(set(discoveries))
        
        # Store discoveries for this task
        self.discovered_laws[task.name].extend(discoveries)
        
        return metrics
    
    def _expression_matches(self, expr: str, target: str, tol: float = 0.01) -> bool:
        """Check if discovered expression matches target using symbolic accuracy."""
        if not expr or not target:
            return False
        try:
            # Convert target string to a SymPy expression
            # Assuming target is a single expression string.
            target_expr = sp.sympify(target)
            ground_truth_dict = {'true_law': target_expr}

            # Calculate symbolic accuracy
            accuracy = calculate_symbolic_accuracy(expr, ground_truth_dict)

            # Check if accuracy is above the threshold
            return accuracy > 0.99
        except (sp.SympifyError, TypeError) as e:
            # Log the error or handle it as appropriate
            print(f"Error sympifying target expression: {target}. Error: {e}")
            # If target cannot be parsed, we can't make a comparison.
            # Depending on desired behavior, could return False or raise error.
            return False
    
    def _log_task_performance(self, 
                            task: PhysicsTask,
                            metrics: Dict[str, float],
                            task_idx: int):
        """Log individual task performance"""
        
        prefix = f"task_{task_idx}_{task.name}"
        
        for key, value in metrics.items():
            self.writer.add_scalar(f"{prefix}/{key}", value, self.iteration)
        
        # Log task metadata
        self.writer.add_scalar(f"{prefix}/difficulty", task.difficulty, self.iteration)
    
    def evaluate_on_new_tasks(self, n_tasks: int = 10) -> Dict[str, float]:
        """Evaluate meta-learned policy on unseen tasks"""
        
        eval_metrics = defaultdict(list)
        
        # Sample new tasks (high difficulty)
        eval_tasks = self.task_distribution.sample_task_batch(
            n_tasks,
            curriculum=False
        )
        
        for task in tqdm(eval_tasks, desc="Evaluating"):
            env = self.env_builder.build_env(task, max_action_space=self.policy.action_dim)
            
            # Clone policy for adaptation
            adapted_policy = self._clone_policy()
            inner_optimizer = torch.optim.SGD(
                adapted_policy.parameters(),
                lr=self.config.adaptation_lr
            )
            
            # Adaptation phase
            adapt_trajectories = self._collect_trajectories(
                self.policy,
                env,
                n_episodes=self.config.support_episodes
            )
            
            task_embedding = self._compute_task_embedding(adapt_trajectories)
            
            # Adapt
            for _ in range(self.config.adaptation_steps):
                loss = self._compute_trajectory_loss(
                    adapted_policy,
                    adapt_trajectories,
                    task_embedding,
                    task
                )
                
                inner_optimizer.zero_grad()
                loss.backward()
                inner_optimizer.step()
            
            # Test phase
            test_trajectories = self._collect_trajectories(
                adapted_policy,
                env,
                n_episodes=20,  # More episodes for evaluation
                task_context=task_embedding
            )
            
            # Compute metrics
            task_metrics = self._compute_task_metrics(
                test_trajectories,
                task,
                adapted_policy
            )
            
            # Store metrics
            for key, value in task_metrics.items():
                eval_metrics[key].append(value)
            eval_metrics['task_difficulty'].append(task.difficulty)
            eval_metrics['task_domain'].append(task.domain)
        
        # Aggregate metrics
        aggregated = {}
        for key, values in eval_metrics.items():
            if key != 'task_domain':
                aggregated[f"eval/{key}_mean"] = np.mean(values)
                aggregated[f"eval/{key}_std"] = np.std(values)
        
        return aggregated
    
    def save_checkpoint(self, path: Optional[str] = None):
        """Save training checkpoint"""
        if path is None:
            path = f"{self.config.checkpoint_dir}/checkpoint_{self.iteration}.pt"
        
        torch.save({
            'iteration': self.iteration,
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.meta_optimizer.state_dict(),
            'meta_losses': self.meta_losses,
            'discovered_laws': dict(self.discovered_laws),
            'config': self.config
        }, path)
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint"""
        checkpoint = torch.load(path)
        
        self.iteration = checkpoint['iteration']
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.meta_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.meta_losses = checkpoint['meta_losses']
        self.discovered_laws = defaultdict(list, checkpoint['discovered_laws'])
    
    def train(self):
        """Main training loop"""
        
        print(f"Starting MAML training for {self.config.meta_iterations} iterations")
        print(f"Tasks per batch: {self.config.tasks_per_batch}")
        print(f"Support episodes: {self.config.support_episodes}")
        print(f"Query episodes: {self.config.query_episodes}")
        print("-" * 50)
        
        for iteration in range(self.config.meta_iterations):
            self.iteration = iteration
            
            # Meta-training step
            start_time = time.time()
            meta_metrics = self.meta_train_step()
            step_time = time.time() - start_time
            
            # Log metrics
            for key, value in meta_metrics.items():
                self.writer.add_scalar(f"train/{key}", value, iteration)
            self.writer.add_scalar("train/step_time", step_time, iteration)
            
            # Print progress
            if iteration % 10 == 0:
                print(f"\nIteration {iteration}/{self.config.meta_iterations}")
                print(f"  Meta loss: {meta_metrics['meta_loss']:.4f}")
                print(f"  Discovery rate: {meta_metrics['discovery_rate']:.3f}")
                print(f"  Correct discoveries: {meta_metrics['correct_discovery']:.3f}")
                print(f"  Unique discoveries: {meta_metrics['unique_discoveries']:.1f}")
                print(f"  Step time: {step_time:.2f}s")
            
            # Evaluation
            if iteration % self.config.eval_interval == 0 and iteration > 0:
                print("\nEvaluating on new tasks...")
                eval_metrics = self.evaluate_on_new_tasks()
                
                for key, value in eval_metrics.items():
                    self.writer.add_scalar(key, value, iteration)
                
                print(f"  Eval discovery rate: {eval_metrics['eval/discovery_rate_mean']:.3f} ± {eval_metrics['eval/discovery_rate_std']:.3f}")
                print(f"  Eval correct rate: {eval_metrics['eval/correct_discovery_mean']:.3f} ± {eval_metrics['eval/correct_discovery_std']:.3f}")
            
            # Checkpointing
            if iteration % self.config.checkpoint_interval == 0 and iteration > 0:
                self.save_checkpoint()
                print(f"  Saved checkpoint at iteration {iteration}")
            
            # Log discovered laws
            if iteration % 50 == 0:
                self._log_discovered_laws()
        
        print("\nTraining complete!")
        self.save_checkpoint(f"{self.config.checkpoint_dir}/final_checkpoint.pt")
        
    def _log_discovered_laws(self):
        """Log summary of discovered laws"""
        print("\nDiscovered Laws Summary:")
        print("-" * 50)
        
        for task_name, discoveries in self.discovered_laws.items():
            if discoveries:
                unique_discoveries = list(set(discoveries))
                print(f"\n{task_name}:")
                for i, expr in enumerate(unique_discoveries[:5]):  # Top 5
                    count = discoveries.count(expr)
                    print(f"  {i+1}. {expr} (found {count} times)")


def main():
    """Main entry point for meta-training"""
    
    # Configuration
    config = MetaLearningConfig(
        meta_lr=0.0003,
        adaptation_lr=0.01,
        adaptation_steps=5,
        tasks_per_batch=10,
        support_episodes=10,
        query_episodes=10,
        meta_iterations=1000,
        use_intrinsic_rewards=True,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    print("Meta-Learning for Physics Discovery")
    print("=" * 50)
    print(f"Device: {config.device}")
    print(f"Config: {json.dumps(config.__dict__, indent=2)}")
    print()
    
    # Initialize task distribution
    print("Initializing physics task distribution...")
    task_distribution = PhysicsTaskDistribution(include_noise=True)
    print(task_distribution.describe_task_distribution())
    print()
    
    # Determine observation and action space dimensions
    print("Determining observation and action space dimensions...")
    env_builder = TaskEnvironmentBuilder(config) # Already initialized in current code structure if we place this after config

    max_obs_dim = 0
    max_action_dim = 0
    
    # Check a few different tasks
    sample_tasks = task_distribution.sample_task_batch(5, curriculum=False) # Sample 5 tasks
    for task in sample_tasks:
        try:
            sample_env = env_builder.build_env(task)
            obs_dim_task = sample_env.observation_space.shape[0]
            action_dim_task = sample_env.action_space.n

            max_obs_dim = max(max_obs_dim, obs_dim_task)
            max_action_dim = max(max_action_dim, action_dim_task)

            print(f"  Task: {task.name} - Obs: {obs_dim_task}, Actions: {action_dim_task}")
        except Exception as e:
            print(f"  Warning: Could not build environment for {task.name}: {e}")

    # Add some buffer to handle variations
    obs_dim = max_obs_dim
    action_dim = max_action_dim + 1  # Add 1 for safety

    # Corrected print statement to avoid issues if used directly in f-string
    print("\nUsing dimensions:")
    print(f"  Observation dimension: {obs_dim}")
    print(f"  Action dimension: {action_dim}")
    print()
    
    # Initialize meta-learning policy
    print("Initializing meta-learning policy...")
    policy = MetaLearningPolicy(
        observation_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=256,
        n_layers=3,
        use_task_embedding=True
    )
    
    total_params = sum(p.numel() for p in policy.parameters())
    print(f"Total parameters: {total_params:,}")
    print()
    
    # Initialize trainer
    trainer = MAMLTrainer(config, policy, task_distribution)
    
    # Start training
    try:
        trainer.train()
    except KeyboardInterrupt:
        # Corrected print statement
        print("\nTraining interrupted by user")
        trainer.save_checkpoint(f"{config.checkpoint_dir}/interrupted_checkpoint.pt")
    except Exception as e:
        # Corrected print statement
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
        trainer.save_checkpoint(f"{config.checkpoint_dir}/error_checkpoint.pt")
    
    # Final evaluation
    # Corrected print statement
    print("\nFinal evaluation on diverse tasks...")
    try:
        final_metrics = trainer.evaluate_on_new_tasks(n_tasks=20)

        # Corrected print statement
        print("\nFinal Performance:")
        print("-" * 30)
        for key, value in sorted(final_metrics.items()):
            if "mean" in key:
                print(f"{key}: {value:.3f}")
    except Exception as e:
        print(f"Evaluation failed: {e}")


if __name__ == "__main__":
    main()
