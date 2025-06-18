"""
DEPRECATED: This module has been replaced by maml_training_framework.py

This file is kept for reference only. Please use:
- maml_training_framework.py for MAML implementation
- physics_task_distribution.py for task distribution

Migration guide:
- Replace: from meta_physics_discovery import PhysicsTask
  With:    from physics_task_distribution import PhysicsTask

- Replace: from meta_physics_discovery import MAMLPhysicsDiscovery
  With:    from maml_training_framework import MAMLTrainer
"""

raise DeprecationWarning(
    "meta-physics-discovery.py is deprecated. "
    "Use maml_training_framework.py instead."
)

"""
Meta-Learning Framework for Physics Discovery
============================================

A comprehensive system for learning to discover physics laws across
multiple domains using meta-reinforcement learning.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import OrderedDict
import copy

from symbolic_discovery_env import SymbolicDiscoveryEnv
from hypothesis_policy_network import HypothesisNet


@dataclass
class PhysicsTask:
    """Represents a physics discovery task"""
    name: str
    data_generator: callable
    true_law: str
    variables: List[str]
    symmetries: List[str]
    difficulty: float
    domain: str  # mechanics, E&M, thermodynamics, etc.


class PhysicsTaskDistribution:
    """Distribution over physics discovery tasks"""
    
    def __init__(self):
        self.tasks = self._create_task_library()
        self.task_families = self._organize_by_family()
        
    def _create_task_library(self) -> List[PhysicsTask]:
        """Create diverse physics tasks"""
        tasks = []
        
        # Classical Mechanics
        tasks.extend([
            PhysicsTask(
                name="harmonic_oscillator",
                data_generator=lambda n: self._generate_harmonic_data(n),
                true_law="0.5 * k * x**2",
                variables=["x", "k"],
                symmetries=["time_reversal", "energy_conservation"],
                difficulty=0.2,
                domain="mechanics"
            ),
            PhysicsTask(
                name="pendulum_small_angle",
                data_generator=lambda n: self._generate_pendulum_data(n, small_angle=True),
                true_law="-g/L * theta",
                variables=["theta", "g", "L"],
                symmetries=["time_reversal"],
                difficulty=0.3,
                domain="mechanics"
            ),
            PhysicsTask(
                name="kepler_orbit",
                data_generator=lambda n: self._generate_kepler_data(n),
                true_law="G*M/r**2",
                variables=["r", "G", "M"],
                symmetries=["rotational", "energy_conservation"],
                difficulty=0.6,
                domain="mechanics"
            ),
            PhysicsTask(
                name="damped_oscillator",
                data_generator=lambda n: self._generate_damped_oscillator_data(n),
                true_law="-k*x - b*v",
                variables=["x", "v", "k", "b"],
                symmetries=["none"],
                difficulty=0.5,
                domain="mechanics"
            ),
        ])
        
        # Thermodynamics
        tasks.extend([
            PhysicsTask(
                name="ideal_gas",
                data_generator=lambda n: self._generate_ideal_gas_data(n),
                true_law="P*V/(n*T)",
                variables=["P", "V", "n", "T"],
                symmetries=["scale_invariance"],
                difficulty=0.3,
                domain="thermodynamics"
            ),
            PhysicsTask(
                name="heat_conduction",
                data_generator=lambda n: self._generate_heat_conduction_data(n),
                true_law="-k * dT/dx",
                variables=["T", "x", "k"],
                symmetries=["translation"],
                difficulty=0.4,
                domain="thermodynamics"
            ),
        ])
        
        # Electromagnetism
        tasks.extend([
            PhysicsTask(
                name="coulomb_law",
                data_generator=lambda n: self._generate_coulomb_data(n),
                true_law="k*q1*q2/r**2",
                variables=["r", "q1", "q2", "k"],
                symmetries=["rotational", "charge_conjugation"],
                difficulty=0.4,
                domain="electromagnetism"
            ),
            PhysicsTask(
                name="rc_circuit",
                data_generator=lambda n: self._generate_rc_circuit_data(n),
                true_law="V * exp(-t/(R*C))",
                variables=["t", "V", "R", "C"],
                symmetries=["none"],
                difficulty=0.5,
                domain="electromagnetism"
            ),
        ])
        
        # Conservation Laws
        tasks.extend([
            PhysicsTask(
                name="elastic_collision",
                data_generator=lambda n: self._generate_collision_data(n),
                true_law="0.5*m1*v1**2 + 0.5*m2*v2**2",
                variables=["m1", "v1", "m2", "v2"],
                symmetries=["galilean", "time_reversal", "momentum_conservation"],
                difficulty=0.7,
                domain="mechanics"
            ),
        ])
        
        return tasks
    
    def _organize_by_family(self) -> Dict[str, List[PhysicsTask]]:
        """Organize tasks by physical domain"""
        families = {}
        for task in self.tasks:
            if task.domain not in families:
                families[task.domain] = []
            families[task.domain].append(task)
        return families
    
    def sample_task(self, 
                   difficulty_range: Optional[Tuple[float, float]] = None,
                   domain: Optional[str] = None) -> PhysicsTask:
        """Sample a task with optional constraints"""
        
        candidates = self.tasks
        
        if difficulty_range:
            candidates = [t for t in candidates 
                         if difficulty_range[0] <= t.difficulty <= difficulty_range[1]]
        
        if domain:
            candidates = [t for t in candidates if t.domain == domain]
        
        return np.random.choice(candidates)
    
    def sample_task_batch(self, 
                         n_tasks: int,
                         curriculum: bool = True) -> List[PhysicsTask]:
        """Sample batch of tasks, optionally with curriculum"""
        
        if curriculum:
            # Start with easier tasks
            difficulties = np.linspace(0.2, 0.8, n_tasks)
            tasks = []
            for diff in difficulties:
                task = self.sample_task(
                    difficulty_range=(diff - 0.1, diff + 0.1)
                )
                tasks.append(task)
            return tasks
        else:
            return [self.sample_task() for _ in range(n_tasks)]
    
    # Data generation methods
    def _generate_harmonic_data(self, n_samples: int) -> np.ndarray:
        """Generate harmonic oscillator data"""
        k = np.random.uniform(0.5, 2.0)
        x = np.random.uniform(-2, 2, n_samples)
        energy = 0.5 * k * x**2
        return np.column_stack([x, np.full(n_samples, k), energy])
    
    def _generate_pendulum_data(self, n_samples: int, small_angle: bool = True) -> np.ndarray:
        """Generate pendulum data"""
        g = 9.81
        L = np.random.uniform(0.5, 2.0)
        
        if small_angle:
            theta = np.random.uniform(-0.3, 0.3, n_samples)
            angular_accel = -g/L * theta
        else:
            theta = np.random.uniform(-np.pi/2, np.pi/2, n_samples)
            angular_accel = -g/L * np.sin(theta)
        
        return np.column_stack([
            theta, 
            np.full(n_samples, g), 
            np.full(n_samples, L),
            angular_accel
        ])
    
    def _generate_kepler_data(self, n_samples: int) -> np.ndarray:
        """Generate orbital mechanics data"""
        G = 6.67e-11
        M = 5.97e24  # Earth mass
        r = np.random.uniform(6.4e6, 4e7, n_samples)  # Near Earth orbits
        force = G * M / r**2
        return np.column_stack([r, np.full(n_samples, G), np.full(n_samples, M), force])


class MetaLearningPolicy(nn.Module):
    """Meta-learning policy that can quickly adapt to new physics tasks"""
    
    def __init__(self, 
                 observation_dim: int,
                 action_dim: int,
                 hidden_dim: int = 256,
                 meta_hidden_dim: int = 128):
        super().__init__()
        
        # Task encoder - learns to recognize physics domains
        self.task_encoder = nn.LSTM(
            observation_dim, 
            meta_hidden_dim,
            batch_first=True
        )
        
        # Modulation network - adapts policy based on task
        self.modulation_net = nn.Sequential(
            nn.Linear(meta_hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2)  # Gains and biases
        )
        
        # Base policy network (will be modulated)
        self.base_policy = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Action head
        self.action_head = nn.Linear(hidden_dim, action_dim)
        
        # Value head for PPO
        self.value_head = nn.Linear(hidden_dim, 1)
        
        # Physics-specific heads
        self.symmetry_detector = nn.Linear(hidden_dim, 10)  # Common symmetries
        self.conservation_detector = nn.Linear(hidden_dim, 5)  # Conservation laws
        
    def forward(self, 
                obs: torch.Tensor,
                task_context: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass with optional task adaptation"""
        
        # Encode task context if provided
        if task_context is not None:
            _, (task_hidden, _) = self.task_encoder(task_context)
            task_embedding = task_hidden.squeeze(0)
            
            # Get modulation parameters
            modulation = self.modulation_net(task_embedding)
            gains, biases = modulation.chunk(2, dim=-1)
            gains = 1 + torch.tanh(gains)  # Multiplicative adaptation
        else:
            gains = 1.0
            biases = 0.0
        
        # Base policy with modulation
        features = self.base_policy(obs)
        features = features * gains + biases
        
        # Compute outputs
        action_logits = self.action_head(features)
        value = self.value_head(features)
        
        # Physics-specific predictions
        symmetries = torch.sigmoid(self.symmetry_detector(features))
        conservations = torch.sigmoid(self.conservation_detector(features))
        
        return {
            'action_logits': action_logits,
            'value': value,
            'symmetries': symmetries,
            'conservations': conservations,
            'features': features
        }
    
    def adapt(self, 
              support_trajectories: List[Dict],
              adaptation_steps: int = 5,
              adaptation_lr: float = 0.01) -> 'MetaLearningPolicy':
        """Fast adaptation on support set (MAML-style)"""
        
        # Clone policy for adaptation
        adapted_policy = copy.deepcopy(self)
        optimizer = torch.optim.Adam(adapted_policy.parameters(), lr=adaptation_lr)
        
        for _ in range(adaptation_steps):
            total_loss = 0
            
            for traj in support_trajectories:
                obs = torch.FloatTensor(traj['observations'])
                actions = torch.LongTensor(traj['actions'])
                rewards = torch.FloatTensor(traj['rewards'])
                
                # Forward pass
                outputs = adapted_policy(obs)
                
                # Compute losses
                action_loss = F.cross_entropy(
                    outputs['action_logits'], 
                    actions
                )
                
                # Physics-informed losses
                if 'discovered_symmetries' in traj:
                    symmetry_targets = torch.FloatTensor(traj['discovered_symmetries'])
                    symmetry_loss = F.binary_cross_entropy(
                        outputs['symmetries'],
                        symmetry_targets
                    )
                else:
                    symmetry_loss = 0
                
                total_loss += action_loss + 0.1 * symmetry_loss
            
            # Update adapted policy
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        
        return adapted_policy


class MAMLPhysicsDiscovery:
    """Model-Agnostic Meta-Learning for Physics Discovery"""
    
    def __init__(self,
                 meta_policy: MetaLearningPolicy,
                 task_distribution: PhysicsTaskDistribution,
                 meta_lr: float = 0.001,
                 adaptation_lr: float = 0.01,
                 adaptation_steps: int = 5):
        
        self.meta_policy = meta_policy
        self.task_distribution = task_distribution
        self.meta_optimizer = torch.optim.Adam(
            meta_policy.parameters(), 
            lr=meta_lr
        )
        
        self.adaptation_lr = adaptation_lr
        self.adaptation_steps = adaptation_steps
        
        # Tracking
        self.meta_losses = []
        self.task_performances = {}
        
    def meta_train_step(self, 
                       n_tasks: int = 10,
                       n_support: int = 10,
                       n_query: int = 10) -> Dict[str, float]:
        """Single meta-training step"""
        
        meta_loss = 0
        meta_metrics = {
            'discovery_rate': 0,
            'avg_episodes_to_discovery': 0,
            'symmetry_detection_acc': 0
        }
        
        # Sample batch of tasks
        tasks = self.task_distribution.sample_task_batch(n_tasks)
        
        for task in tasks:
            # Create environment for this task
            env = self._create_task_environment(task)
            
            # Collect support trajectories
            support_trajs = self._collect_trajectories(
                self.meta_policy, 
                env, 
                n_episodes=n_support
            )
            
            # Adapt policy to task
            adapted_policy = self.meta_policy.adapt(
                support_trajs,
                adaptation_steps=self.adaptation_steps,
                adaptation_lr=self.adaptation_lr
            )
            
            # Collect query trajectories with adapted policy
            query_trajs = self._collect_trajectories(
                adapted_policy,
                env,
                n_episodes=n_query
            )
            
            # Compute meta-loss on query set
            task_loss, task_metrics = self._compute_task_loss(
                adapted_policy,
                query_trajs,
                task
            )
            
            meta_loss += task_loss
            
            # Accumulate metrics
            for key, value in task_metrics.items():
                meta_metrics[key] += value / n_tasks
        
        # Meta-update
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()
        
        # Track performance
        self.meta_losses.append(meta_loss.item())
        
        return meta_metrics
    
    def _create_task_environment(self, task: PhysicsTask) -> SymbolicDiscoveryEnv:
        """Create environment for specific physics task"""
        
        # Generate data for task
        data = task.data_generator(1000)
        
        # Create variables
        from progressive_grammar_system import Variable
        variables = []
        for i, var_name in enumerate(task.variables[:-1]):  # Last is target
            variables.append(Variable(var_name, i, {}))
        
        # Create grammar with task-specific operators
        grammar = self._create_task_grammar(task)
        
        # Create environment
        env = SymbolicDiscoveryEnv(
            grammar=grammar,
            target_data=data,
            variables=variables,
            max_depth=7,
            max_complexity=20,
            reward_config={
                'mse_weight': -1.0,
                'complexity_penalty': -0.01,
                'symmetry_bonus': 0.2 if task.symmetries else 0.0
            }
        )
        
        return env
    
    def _create_task_grammar(self, task: PhysicsTask):
        """Create grammar appropriate for task domain"""
        
        from progressive_grammar_system import ProgressiveGrammar
        grammar = ProgressiveGrammar()
        
        # Add domain-specific operators
        if task.domain == "mechanics":
            grammar.add_operators(['+', '-', '*', '/', '**2', 'sqrt'])
        elif task.domain == "thermodynamics":
            grammar.add_operators(['+', '-', '*', '/', 'log', 'exp'])
        elif task.domain == "electromagnetism":
            grammar.add_operators(['+', '-', '*', '/', '**2', 'exp', '1/'])
        
        # Add task-specific functions if needed
        if "sin" in task.true_law or "cos" in task.true_law:
            grammar.add_operators(['sin', 'cos'])
        
        return grammar
    
    def _collect_trajectories(self,
                            policy: MetaLearningPolicy,
                            env: SymbolicDiscoveryEnv,
                            n_episodes: int) -> List[Dict]:
        """Collect trajectories using policy"""
        
        trajectories = []
        
        for _ in range(n_episodes):
            obs = env.reset()
            trajectory = {
                'observations': [],
                'actions': [],
                'rewards': [],
                'discovered_expressions': [],
                'task_context': []
            }
            
            done = False
            while not done:
                # Add observation
                trajectory['observations'].append(obs)
                
                # Get action from policy
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                    outputs = policy(obs_tensor)
                    action_probs = F.softmax(outputs['action_logits'], dim=-1)
                    action = torch.multinomial(action_probs, 1).item()
                
                # Take action
                next_obs, reward, done, _, info = env.step(action)
                
                # Record trajectory
                trajectory['actions'].append(action)
                trajectory['rewards'].append(reward)
                
                if 'expression' in info:
                    trajectory['discovered_expressions'].append(info['expression'])
                
                obs = next_obs
            
            trajectories.append(trajectory)
        
        return trajectories
    
    def _compute_task_loss(self,
                          policy: MetaLearningPolicy,
                          trajectories: List[Dict],
                          task: PhysicsTask) -> Tuple[torch.Tensor, Dict]:
        """Compute loss and metrics for task"""
        
        total_loss = 0
        metrics = {
            'discovery_rate': 0,
            'avg_episodes_to_discovery': 0,
            'symmetry_detection_acc': 0
        }
        
        discoveries = []
        episodes_to_discovery = []
        
        for i, traj in enumerate(trajectories):
            # Standard RL loss
            obs = torch.FloatTensor(traj['observations'])
            actions = torch.LongTensor(traj['actions'])
            rewards = torch.FloatTensor(traj['rewards'])
            
            outputs = policy(obs)
            
            # Policy gradient loss
            action_log_probs = F.log_softmax(outputs['action_logits'], dim=-1)
            selected_log_probs = action_log_probs.gather(1, actions.unsqueeze(1))
            
            # Compute returns
            returns = self._compute_returns(rewards)
            
            policy_loss = -(selected_log_probs.squeeze() * returns).mean()
            
            # Value loss
            value_loss = F.mse_loss(outputs['value'].squeeze(), returns)
            
            # Physics-specific losses
            if task.symmetries:
                # Create symmetry targets based on task
                symmetry_targets = self._create_symmetry_targets(task.symmetries)
                symmetry_loss = F.binary_cross_entropy(
                    outputs['symmetries'].mean(0),
                    symmetry_targets
                )
            else:
                symmetry_loss = 0
            
            total_loss += policy_loss + 0.5 * value_loss + 0.1 * symmetry_loss
            
            # Track metrics
            if traj['discovered_expressions']:
                discoveries.extend(traj['discovered_expressions'])
                episodes_to_discovery.append(i + 1)
        
        # Compute metrics
        if discoveries:
            # Check if any discovery matches true law
            discovery_rate = sum(
                self._expression_matches(expr, task.true_law) 
                for expr in discoveries
            ) / len(trajectories)
            metrics['discovery_rate'] = discovery_rate
            
            if episodes_to_discovery:
                metrics['avg_episodes_to_discovery'] = np.mean(episodes_to_discovery)
        
        return total_loss / len(trajectories), metrics
    
    def _compute_returns(self, rewards: torch.Tensor, gamma: float = 0.99) -> torch.Tensor:
        """Compute discounted returns"""
        returns = torch.zeros_like(rewards)
        running_return = 0
        
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + gamma * running_return
            returns[t] = running_return
        
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
    
    def _expression_matches(self, expr: str, target: str, tol: float = 0.01) -> bool:
        """Check if discovered expression matches target"""
        
        # Simple string matching for now
        # In practice, would evaluate on test data
        import re
        
        # Normalize expressions
        expr_normalized = re.sub(r'\s+', '', expr)
        target_normalized = re.sub(r'\s+', '', target)
        
        return expr_normalized == target_normalized
    
    def evaluate_on_new_task(self, 
                           task: PhysicsTask,
                           n_adaptation_episodes: int = 10,
                           n_test_episodes: int = 50) -> Dict[str, float]:
        """Evaluate meta-learned policy on completely new task"""
        
        # Create environment
        env = self._create_task_environment(task)
        
        # Collect adaptation data
        adaptation_trajs = self._collect_trajectories(
            self.meta_policy,
            env,
            n_episodes=n_adaptation_episodes
        )
        
        # Adapt to new task
        adapted_policy = self.meta_policy.adapt(
            adaptation_trajs,
            adaptation_steps=self.adaptation_steps,
            adaptation_lr=self.adaptation_lr
        )
        
        # Test adapted policy
        test_trajs = self._collect_trajectories(
            adapted_policy,
            env,
            n_episodes=n_test_episodes
        )
        
        # Compute metrics
        discoveries = []
        episodes_to_first_discovery = None
        
        for i, traj in enumerate(test_trajs):
            if traj['discovered_expressions']:
                discoveries.extend(traj['discovered_expressions'])
                if episodes_to_first_discovery is None:
                    # Check if any matches true law
                    for expr in traj['discovered_expressions']:
                        if self._expression_matches(expr, task.true_law):
                            episodes_to_first_discovery = i + 1
                            break
        
        metrics = {
            'discovery_rate': len([e for e in discoveries if self._expression_matches(e, task.true_law)]) / n_test_episodes,
            'unique_discoveries': len(set(discoveries)),
            'episodes_to_discovery': episodes_to_first_discovery or n_test_episodes,
            'task_difficulty': task.difficulty
        }
        
        return metrics


class PhysicsDiscoveryAnalyzer:
    """Analyze what the meta-learner has learned"""
    
    def __init__(self, meta_policy: MetaLearningPolicy):
        self.meta_policy = meta_policy
        
    def analyze_learned_features(self, task_distribution: PhysicsTaskDistribution):
        """Analyze internal representations"""
        
        # Sample diverse tasks
        tasks = task_distribution.sample_task_batch(20, curriculum=False)
        
        task_embeddings = []
        task_labels = []
        
        for task in tasks:
            # Get task embedding
            env = create_task_environment(task)
            trajs = collect_trajectories(self.meta_policy, env, n_episodes=5)
            
            # Extract task context
            context = self._extract_task_context(trajs)
            _, (task_hidden, _) = self.meta_policy.task_encoder(context)
            
            task_embeddings.append(task_hidden.squeeze().detach().numpy())
            task_labels.append(task.domain)
        
        # Analyze clustering by domain
        from sklearn.manifold import TSNE
        embeddings_2d = TSNE(n_components=2).fit_transform(task_embeddings)
        
        # Plot (would visualize in practice)
        print("Task embeddings clustered by domain:")
        for domain in set(task_labels):
            domain_points = embeddings_2d[[i for i, l in enumerate(task_labels) if l == domain]]
            print(f"  {domain}: center at {domain_points.mean(axis=0)}")
    
    def extract_physics_priors(self) -> Dict[str, Any]:
        """Extract learned physics priors from meta-policy"""
        
        # Analyze symmetry detector
        symmetry_weights = self.meta_policy.symmetry_detector.weight.detach().numpy()
        
        # Find most important features for each symmetry
        important_features = {}
        symmetry_types = ['time_reversal', 'spatial_translation', 'rotational', etc.]
        
        for i, sym in enumerate(symmetry_types):
            feature_importance = np.abs(symmetry_weights[i])
            top_features = np.argsort(feature_importance)[-10:]
            important_features[sym] = top_features
        
        return {
            'symmetry_detection_features': important_features,
            'learned_modulations': self._analyze_modulations()
        }


# Training Script
def train_meta_physics_discoverer():
    """Complete training pipeline"""
    
    # Initialize
    task_distribution = PhysicsTaskDistribution()
    
    # Create meta-policy
    meta_policy = MetaLearningPolicy(
        observation_dim=256,  # After encoding
        action_dim=50,  # Grammar actions
        hidden_dim=512,
        meta_hidden_dim=256
    )
    
    # Create meta-learner
    meta_learner = MAMLPhysicsDiscovery(
        meta_policy=meta_policy,
        task_distribution=task_distribution,
        meta_lr=0.0003,
        adaptation_lr=0.01,
        adaptation_steps=5
    )
    
    # Training loop
    n_meta_iterations = 1000
    
    for iteration in range(n_meta_iterations):
        # Meta-training step
        metrics = meta_learner.meta_train_step(
            n_tasks=10,
            n_support=10,
            n_query=10
        )
        
        if iteration % 10 == 0:
            print(f"\nIteration {iteration}")
            print(f"  Discovery rate: {metrics['discovery_rate']:.3f}")
            print(f"  Avg episodes to discovery: {metrics['avg_episodes_to_discovery']:.1f}")
            
        # Periodic evaluation on held-out tasks
        if iteration % 50 == 0:
            # Test on new physics problem
            test_task = PhysicsTask(
                name="double_pendulum_energy",
                data_generator=lambda n: generate_double_pendulum_data(n),
                true_law="m1*g*h1 + m2*g*h2 + 0.5*m1*v1**2 + 0.5*m2*v2**2",
                variables=["h1", "h2", "v1", "v2", "m1", "m2", "g"],
                symmetries=["energy_conservation"],
                difficulty=0.8,
                domain="mechanics"
            )
            
            eval_metrics = meta_learner.evaluate_on_new_task(
                test_task,
                n_adaptation_episodes=20,
                n_test_episodes=100
            )
            
            print(f"\n  Evaluation on {test_task.name}:")
            print(f"    Discovery rate after adaptation: {eval_metrics['discovery_rate']:.3f}")
            print(f"    Episodes to discovery: {eval_metrics['episodes_to_discovery']}")
    
    return meta_learner


if __name__ == "__main__":
    print("Meta-Learning for Physics Discovery")
    print("=" * 50)
    
    # Train meta-learner
    meta_learner = train_meta_physics_discoverer()
    
    # Analyze what was learned
    analyzer = PhysicsDiscoveryAnalyzer(meta_learner.meta_policy)
    analyzer.analyze_learned_features(meta_learner.task_distribution)
    
    # Extract learned priors
    physics_priors = analyzer.extract_physics_priors()
    print("\nLearned Physics Priors:")
    print(physics_priors)
