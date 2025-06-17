"""
Multi-Agent Self-Play Training System for Janus
===============================================

Extends the existing PPOTrainer with league-based self-play, 
adversarial hypothesis generation, and cooperative discovery.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Deque
from collections import deque
from dataclasses import dataclass
import pickle
from copy import deepcopy

from hypothesis_policy_network import HypothesisNet, PPOTrainer, RolloutBuffer
from symbolic_discovery_env import SymbolicDiscoveryEnv


@dataclass
class AgentSnapshot:
    """Snapshot of an agent at a particular training iteration."""
    policy_state_dict: Dict
    iteration: int
    performance_stats: Dict
    discovered_laws: List[str]
    elo_rating: float = 1500.0


class LeaguePlayManager:
    """Manages a league of agents for self-play training."""
    
    def __init__(self, 
                 base_policy: HypothesisNet,
                 max_league_size: int = 50,
                 snapshot_interval: int = 10000):
        
        self.base_policy = base_policy
        self.max_league_size = max_league_size
        self.snapshot_interval = snapshot_interval
        
        # League components
        self.main_agents: List[AgentSnapshot] = []
        self.exploiter_agents: List[AgentSnapshot] = []
        self.league_agents: List[AgentSnapshot] = []
        
        # Performance tracking
        self.matchup_history: Dict[Tuple[int, int], List[float]] = {}
        self.discovery_cache: Dict[str, float] = {}
        
    def should_snapshot(self, iteration: int, performance: float) -> bool:
        """Determine if current agent should be added to league."""
        if iteration % self.snapshot_interval == 0:
            return True
        
        # Add if significantly better than league average
        if self.league_agents:
            avg_perf = np.mean([a.performance_stats.get('avg_reward', 0) 
                               for a in self.league_agents])
            if performance > avg_perf * 1.2:
                return True
        
        return False
    
    def add_to_league(self, 
                     policy: HypothesisNet,
                     iteration: int,
                     stats: Dict,
                     discoveries: List[str]):
        """Add a snapshot to the league."""
        
        snapshot = AgentSnapshot(
            policy_state_dict=deepcopy(policy.state_dict()),
            iteration=iteration,
            performance_stats=stats,
            discovered_laws=discoveries
        )
        
        self.league_agents.append(snapshot)
        
        # Maintain league size
        if len(self.league_agents) > self.max_league_size:
            # Remove lowest performing agent
            self.league_agents.sort(key=lambda x: x.elo_rating)
            self.league_agents.pop(0)
    
    def sample_opponent(self, 
                       strategy: str = 'prioritized_quality_diversity') -> AgentSnapshot:
        """Sample an opponent from the league."""
        
        if not self.league_agents:
            # Return a random policy if league is empty
            random_policy = deepcopy(self.base_policy)
            random_policy.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
            return AgentSnapshot(
                policy_state_dict=random_policy.state_dict(),
                iteration=0,
                performance_stats={},
                discovered_laws=[]
            )
        
        if strategy == 'uniform':
            return np.random.choice(self.league_agents)
        
        elif strategy == 'prioritized_quality_diversity':
            # Balance between strong opponents and diverse discoveries
            quality_scores = np.array([a.elo_rating for a in self.league_agents])
            
            # Calculate diversity based on discovered laws
            diversity_scores = []
            for agent in self.league_agents:
                unique_discoveries = set(agent.discovered_laws)
                diversity = len(unique_discoveries) / (len(agent.discovered_laws) + 1)
                diversity_scores.append(diversity)
            
            diversity_scores = np.array(diversity_scores)
            
            # Combined score
            combined_scores = 0.7 * (quality_scores / quality_scores.max()) + \
                            0.3 * (diversity_scores / diversity_scores.max())
            
            # Softmax sampling
            probs = np.exp(combined_scores) / np.sum(np.exp(combined_scores))
            return np.random.choice(self.league_agents, p=probs)
        
        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}")


class AdversarialDiscoveryEnv(SymbolicDiscoveryEnv):
    """
    Extended environment with adversarial and cooperative modes.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.mode = 'standard'  # 'standard', 'adversarial', 'cooperative'
        self.opponent_policy = None
        self.partner_policy = None
        
        # Adversarial components
        self.hypothesis_history: Deque[str] = deque(maxlen=100)
        self.counter_examples: List[np.ndarray] = []
        
    def set_mode(self, mode: str, policy: Optional[HypothesisNet] = None):
        """Set environment mode and associated policy."""
        self.mode = mode
        
        if mode == 'adversarial':
            self.opponent_policy = policy
        elif mode == 'cooperative':
            self.partner_policy = policy
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Extended step with mode-specific behavior."""
        
        # Standard step
        obs, reward, terminated, truncated, info = super().step(action)
        
        if self.mode == 'adversarial' and terminated:
            # Opponent tries to find counter-examples
            reward = self._adversarial_evaluation(info)
            
        elif self.mode == 'cooperative' and self.partner_policy:
            # Partner provides hints
            hint_reward = self._cooperative_hint_bonus(obs)
            reward += hint_reward
        
        return obs, reward, terminated, truncated, info
    
    def _adversarial_evaluation(self, info: Dict) -> float:
        """Adversarial agent tries to break discovered law."""
        
        if 'expression' not in info:
            return self.reward_config['timeout_penalty']
        
        expr_str = info['expression']
        
        # Check if we've seen this before
        if expr_str in self.hypothesis_history:
            return -0.1  # Penalty for rediscovery
        
        self.hypothesis_history.append(expr_str)
        
        # Generate adversarial examples
        n_adversarial = 10
        adversarial_data = []
        
        for _ in range(n_adversarial):
            # Perturb existing data
            idx = np.random.randint(len(self.target_data))
            perturbed = self.target_data[idx].copy()
            
            # Smart perturbation based on expression structure
            if 'sin' in expr_str or 'cos' in expr_str:
                # Phase shift for periodic functions
                perturbed[0] += np.pi / 4
            else:
                # Gradient-based perturbation
                perturbed += np.random.randn(len(perturbed)) * 0.1
            
            adversarial_data.append(perturbed)
        
        # Evaluate on adversarial examples
        adversarial_mse = self._evaluate_on_data(
            info['expression'], 
            np.array(adversarial_data)
        )
        
        # Reward robustness
        robustness_bonus = -np.log(adversarial_mse + 1e-10) * 0.5
        
        return info.get('reward', 0) + robustness_bonus
    
    def _cooperative_hint_bonus(self, obs: np.ndarray) -> float:
        """Partner provides structural hints."""
        
        if self.partner_policy is None:
            return 0.0
        
        # Get partner's action distribution
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            mask_tensor = torch.BoolTensor(self.get_action_mask()).unsqueeze(0)
            
            partner_output = self.partner_policy(obs_tensor, mask_tensor)
            action_probs = torch.softmax(partner_output['action_logits'], dim=-1)
        
        # Entropy-based hint quality
        entropy = -torch.sum(action_probs * torch.log(action_probs + 1e-10))
        
        # Low entropy = confident hint
        hint_quality = 1.0 / (1.0 + entropy.item())
        
        return hint_quality * 0.1
    
    def _evaluate_on_data(self, expr_str: str, data: np.ndarray) -> float:
        """Evaluate expression on given data."""
        try:
            expr = self.grammar.create_expression('var', [expr_str])
            predictions = []
            
            for i in range(len(data)):
                subs = {var.symbolic: data[i, var.index] for var in self.variables}
                pred = float(expr.symbolic.subs(subs))
                predictions.append(pred)
            
            targets = data[:, -1]
            mse = np.mean((np.array(predictions) - targets) ** 2)
            return mse
            
        except:
            return float('inf')


class MultiAgentPPOTrainer(PPOTrainer):
    """
    Extended PPO trainer with self-play capabilities.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Self-play components
        self.league_manager = LeaguePlayManager(self.policy)
        self.training_iteration = 0
        
        # Metrics
        self.vs_league_winrate: Deque[float] = deque(maxlen=100)
        self.discovery_diversity: Deque[float] = deque(maxlen=100)
        
    def collect_selfplay_rollouts(self, 
                                 n_steps: int,
                                 opponent_sampling: str = 'prioritized_quality_diversity'
                                 ) -> Dict[str, List]:
        """Collect rollouts against league opponents."""
        
        rollouts = {
            'main': RolloutBuffer(),
            'vs_league': RolloutBuffer(),
            'cooperative': RolloutBuffer()
        }
        
        steps_per_mode = n_steps // 3
        
        # 1. Standard exploration
        self.env.set_mode('standard')
        self._collect_rollout_batch(rollouts['main'], steps_per_mode)
        
        # 2. Adversarial self-play
        opponent = self.league_manager.sample_opponent(opponent_sampling)
        opponent_policy = deepcopy(self.policy)
        opponent_policy.load_state_dict(opponent.policy_state_dict)
        
        self.env.set_mode('adversarial', opponent_policy)
        self._collect_rollout_batch(rollouts['vs_league'], steps_per_mode)
        
        # 3. Cooperative discovery
        partner = self.league_manager.sample_opponent('uniform')
        partner_policy = deepcopy(self.policy)
        partner_policy.load_state_dict(partner.policy_state_dict)
        
        self.env.set_mode('cooperative', partner_policy)
        self._collect_rollout_batch(rollouts['cooperative'], steps_per_mode)
        
        # Compute returns for all buffers
        for buffer in rollouts.values():
            buffer.compute_returns_and_advantages(
                self.policy, 
                self.gamma, 
                self.gae_lambda
            )
        
        return rollouts
    
    def _collect_rollout_batch(self, buffer: RolloutBuffer, n_steps: int):
        """Collect rollouts for a specific buffer."""
        
        obs, _ = self.env.reset()
        
        for _ in range(n_steps):
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            mask_tensor = torch.BoolTensor(self.env.get_action_mask()).unsqueeze(0)
            
            with torch.no_grad():
                action, log_prob, value = self.policy.get_action(
                    obs_tensor, 
                    mask_tensor
                )
            
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            
            buffer.add(
                obs=obs,
                action=action,
                reward=reward,
                value=value.item(),
                log_prob=log_prob.item(),
                done=terminated or truncated,
                action_mask=self.env.get_action_mask()
            )
            
            if terminated or truncated:
                # Track discoveries
                if 'expression' in info:
                    self._update_discovery_metrics(info['expression'])
                
                obs, _ = self.env.reset()
            else:
                obs = next_obs
    
    def train_selfplay(self,
                      total_timesteps: int,
                      rollout_length: int = 2048,
                      n_epochs: int = 10,
                      league_update_interval: int = 10000):
        """Main self-play training loop."""
        
        n_updates = total_timesteps // rollout_length
        
        for update in range(n_updates):
            self.training_iteration += rollout_length
            
            # Collect diverse rollouts
            rollout_data = self.collect_selfplay_rollouts(rollout_length)
            
            # Train on all data
            all_losses = []
            
            for rollout_name, buffer in rollout_data.items():
                data = buffer.get()
                
                # Weight different rollout types
                weight = {
                    'main': 1.0,
                    'vs_league': 0.8,
                    'cooperative': 0.6
                }.get(rollout_name, 1.0)
                
                for epoch in range(n_epochs):
                    indices = np.random.permutation(len(data['observations']))
                    
                    for start_idx in range(0, len(indices), 64):
                        batch_indices = indices[start_idx:start_idx + 64]
                        batch = {k: v[batch_indices] for k, v in data.items()}
                        
                        loss_info = self.train_step(batch)
                        loss_info['loss'] *= weight
                        all_losses.append(loss_info)
            
            # Update league
            if self.training_iteration % league_update_interval == 0:
                current_stats = {
                    'avg_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
                    'avg_complexity': np.mean(self.episode_complexities) if self.episode_complexities else 0,
                    'discovery_diversity': np.mean(self.discovery_diversity) if self.discovery_diversity else 0
                }
                
                discoveries = list(set([
                    info['expression'] 
                    for info in self.env._evaluation_cache.values() 
                    if 'expression' in info
                ]))
                
                self.league_manager.add_to_league(
                    self.policy,
                    self.training_iteration,
                    current_stats,
                    discoveries
                )
            
            # Logging
            if update % 10 == 0:
                self._log_selfplay_metrics(all_losses)
    
    def _update_discovery_metrics(self, expression: str):
        """Track diversity of discoveries."""
        
        # Simple diversity metric based on expression structure
        expr_features = set()
        
        for op in ['sin', 'cos', 'log', 'exp', '**', 'sqrt']:
            if op in expression:
                expr_features.add(op)
        
        # Count unique variables used
        for var in self.env.variables:
            if var.name in expression:
                expr_features.add(f'var_{var.name}')
        
        diversity_score = len(expr_features) / 10.0  # Normalize
        self.discovery_diversity.append(diversity_score)
    
    def _log_selfplay_metrics(self, losses: List[Dict]):
        """Log self-play specific metrics."""
        
        print(f"\n{'='*60}")
        print(f"Self-Play Update {self.training_iteration}")
        print(f"{'='*60}")
        
        print(f"League Size: {len(self.league_manager.league_agents)}")
        print(f"Avg Reward: {np.mean(self.episode_rewards) if self.episode_rewards else 0:.3f}")
        print(f"Discovery Diversity: {np.mean(self.discovery_diversity) if self.discovery_diversity else 0:.3f}")
        print(f"Avg Loss: {np.mean([l['loss'] for l in losses]):.3f}")
        
        # Top discoveries
        if hasattr(self.env, '_evaluation_cache') and self.env._evaluation_cache:
            discoveries = sorted(
                [(info['expression'], info['mse']) 
                 for info in self.env._evaluation_cache.values() 
                 if 'expression' in info and 'mse' in info],
                key=lambda x: x[1]
            )[:3]
            
            print("\nTop Discoveries:")
            for expr, mse in discoveries:
                print(f"  {expr} (MSE: {mse:.3e})")


# Example usage
if __name__ == "__main__":
    from progressive_grammar_system import ProgressiveGrammar, Variable
    
    # Setup
    grammar = ProgressiveGrammar()
    variables = [
        Variable("x", 0, {"smoothness": 0.9}),
        Variable("v", 1, {"conservation_score": 0.8})
    ]
    
    # Generate synthetic data
    n_samples = 1000
    x_data = np.random.uniform(-2, 2, n_samples)
    v_data = np.random.uniform(-3, 3, n_samples)
    energy = 0.5 * v_data**2 + 0.5 * x_data**2  # Harmonic oscillator
    data = np.column_stack([x_data, v_data, energy])
    
    # Create adversarial environment
    env = AdversarialDiscoveryEnv(
        grammar=grammar,
        target_data=data,
        variables=variables,
        max_depth=7,
        max_complexity=15
    )
    
    # Create policy
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    policy = HypothesisNet(
        observation_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=256,
        encoder_type='transformer'
    )
    
    # Create self-play trainer
    trainer = MultiAgentPPOTrainer(policy, env)
    
    print("Starting self-play training...")
    trainer.train_selfplay(
        total_timesteps=100000,
        rollout_length=2048,
        n_epochs=5,
        league_update_interval=10000
    )
