"""
Distributed Training Infrastructure for Janus
============================================

Scales training across multiple GPUs/nodes using Ray, with
asynchronous evaluation and parallel hypothesis testing.
"""

import ray
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.util.placement_group import placement_group

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import time
from pathlib import Path
from collections import OrderedDict

from hypothesis_policy_network import HypothesisNet, TreeEncoder, TransformerEncoder
from janus.ai_interpretability.environments import SymbolicDiscoveryEnv
from progressive_grammar_system import ProgressiveGrammar, Variable


class RLlibHypothesisNet(TorchModelV2, nn.Module):
    """Wrapper to make HypothesisNet compatible with RLlib."""
    
    def __init__(self, 
                 obs_space, 
                 action_space, 
                 num_outputs, 
                 model_config, 
                 name):
        
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)
        
        # Extract custom config
        custom_config = model_config.get("custom_model_config", {})
        
        # Create underlying HypothesisNet
        self.hypothesis_net = HypothesisNet(
            observation_dim=obs_space.shape[0],
            action_dim=action_space.n,
            hidden_dim=custom_config.get("hidden_dim", 256),
            encoder_type=custom_config.get("encoder_type", "transformer"),
            grammar=custom_config.get("grammar", None)
        )
        
        self._last_tree_repr = None
    
    @override(TorchModelV2)
    def forward(self, 
                input_dict: Dict[str, Any], 
                state: List[Any], 
                seq_lens: Any) -> Tuple[torch.Tensor, List[Any]]:
        
        obs = input_dict["obs"]
        
        # Get action mask if available
        action_mask = input_dict.get("action_mask", None)
        
        # Forward through HypothesisNet
        outputs = self.hypothesis_net(obs, action_mask)
        
        # Store tree representation for value function
        self._last_tree_repr = outputs['tree_representation']
        
        # Return logits for RLlib
        return outputs['action_logits'], state
    
    @override(TorchModelV2)
    def value_function(self) -> torch.Tensor:
        """Value function based on last tree representation."""
        if self._last_tree_repr is None:
            return torch.tensor(0.0)
        
        return self.hypothesis_net.value_net(self._last_tree_repr).squeeze()


@ray.remote(num_gpus=0.1)
class AsyncExpressionEvaluator:
    """Asynchronous expression evaluator for parallel hypothesis testing."""
    
    def __init__(self, grammar: ProgressiveGrammar):
        self.grammar = grammar
        self.cache = OrderedDict()
        self.max_cache_size = 4096  # Max number of expressions to cache
    
    def evaluate_batch(self, 
                      expressions: List[str],
                      data: np.ndarray,
                      variables: List[Variable]) -> List[Dict[str, float]]:
        """Evaluate multiple expressions in parallel."""
        
        results = []
        
        for expr_str in expressions:
            if expr_str in self.cache:
                self.cache.move_to_end(expr_str)  # Mark as recently used
                results.append(self.cache[expr_str])
                continue
            
            # If cache is full and expr_str is new, remove the oldest item
            if len(self.cache) >= self.max_cache_size:
                self.cache.popitem(last=False)

            try:
                # Parse expression
                expr = self.grammar.create_expression('var', [expr_str])
                
                # Evaluate on data
                predictions = []
                for i in range(len(data)):
                    subs = {var.symbolic: data[i, var.index] for var in variables}
                    pred = float(expr.symbolic.subs(subs))
                    predictions.append(pred)
                
                # Calculate metrics
                targets = data[:, -1]
                mse = np.mean((np.array(predictions) - targets) ** 2)
                
                # Physics-specific metrics
                variance_ratio = np.var(predictions) / np.var(targets)
                
                result = {
                    'mse': mse,
                    'complexity': expr.complexity,
                    'variance_ratio': variance_ratio,
                    'valid': True
                }
                
            except Exception as e:
                result = {
                    'mse': float('inf'),
                    'complexity': float('inf'),
                    'variance_ratio': 0.0,
                    'valid': False,
                    'error': str(e)
                }
            
            self.cache[expr_str] = result
            results.append(result)
        
        return results


@ray.remote(num_cpus=2, num_gpus=0.5)
class DistributedExperimentWorker:
    """Worker for running distributed experiments."""
    
    def __init__(self, 
                 worker_id: int,
                 grammar: ProgressiveGrammar,
                 base_config: Dict[str, Any]):
        
        self.worker_id = worker_id
        self.grammar = grammar
        self.config = base_config
        
        # Create local environment
        self.env = self._create_env()
        
        # Evaluator pool
        self.evaluator = AsyncExpressionEvaluator.remote(grammar)
        
    def _create_env(self) -> SymbolicDiscoveryEnv:
        """Create environment with worker-specific configuration."""
        
        # Worker-specific modifications
        config = self.config.copy()
        
        # Vary hyperparameters across workers
        config['max_depth'] = config['max_depth'] + (self.worker_id % 3) - 1
        config['reward_config']['mse_weight'] *= (1 + 0.1 * (self.worker_id % 5))
        
        return SymbolicDiscoveryEnv(**config)
    
    def run_experiment(self, 
                      policy_weights: Dict,
                      n_episodes: int = 10) -> Dict[str, Any]:
        """Run experiment with given policy weights."""
        
        # Create and load policy
        policy = HypothesisNet(
            observation_dim=self.env.observation_space.shape[0],
            action_dim=self.env.action_space.n,
            hidden_dim=256
        )
        policy.load_state_dict(policy_weights)
        policy.eval()
        
        # Collect episodes
        episodes_data = []
        discovered_expressions = []
        
        for _ in range(n_episodes):
            obs, _ = self.env.reset()
            episode_reward = 0
            
            while True:
                # Get action
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                    mask_tensor = torch.BoolTensor(self.env.get_action_mask()).unsqueeze(0)
                    
                    action, _, _ = policy.get_action(obs_tensor, mask_tensor)
                
                # Step
                obs, reward, terminated, truncated, info = self.env.step(action)
                episode_reward += reward
                
                if terminated or truncated:
                    if 'expression' in info:
                        discovered_expressions.append(info['expression'])
                    
                    episodes_data.append({
                        'reward': episode_reward,
                        'complexity': info.get('complexity', 0),
                        'mse': info.get('mse', float('inf'))
                    })
                    break
        
        # Batch evaluate discovered expressions
        if discovered_expressions:
            eval_results = ray.get(
                self.evaluator.evaluate_batch.remote(
                    discovered_expressions,
                    self.config['target_data'],
                    self.config['variables']
                )
            )
        else:
            eval_results = []
        
        return {
            'worker_id': self.worker_id,
            'episodes': episodes_data,
            'discoveries': discovered_expressions,
            'eval_results': eval_results,
            'avg_reward': np.mean([e['reward'] for e in episodes_data]),
            'best_mse': min([r['mse'] for r in eval_results], default=float('inf'))
        }


class DistributedJanusTrainer:
    """Main distributed training orchestrator."""
    
    def __init__(self,
                 grammar: ProgressiveGrammar,
                 env_config: Dict[str, Any],
                 num_workers: int = 8,
                 num_gpus: int = 4):
        
        self.grammar = grammar
        self.env_config = env_config
        self.num_workers = num_workers
        self.num_gpus = num_gpus
        
        # Initialize Ray
        if not ray.is_initialized():
            ray.init(num_cpus=num_workers * 2, num_gpus=num_gpus)
        
        # Register custom model
        ModelCatalog.register_custom_model("hypothesis_net", RLlibHypothesisNet)
        
        # Create workers
        self.workers = [
            DistributedExperimentWorker.remote(i, grammar, env_config)
            for i in range(num_workers)
        ]
        
        # Shared discovery repository
        self.discovery_repo = DiscoveryRepository.remote()
        
    def train_with_pbt(self,
                      num_iterations: int = 100,
                      checkpoint_dir: str = "./checkpoints"):
        """Train using Population Based Training."""
        
        # PBT scheduler
        pbt_scheduler = PopulationBasedTraining(
            time_attr="training_iteration",
            perturbation_interval=10,
            hyperparam_mutations={
                "lr": tune.uniform(1e-5, 1e-3),
                "clip_param": tune.uniform(0.1, 0.3),
                "entropy_coeff": tune.uniform(0.0, 0.01),
                "train_batch_size": [2048, 4096, 8192]
            }
        )
        
        # RLlib config
        config = PPOConfig()
        config.training(
            lr=3e-4,
            clip_param=0.2,
            entropy_coeff=0.01,
            train_batch_size=4096,
            sgd_minibatch_size=256,
            num_sgd_iter=10
        )
        config.resources(
            num_gpus=0.5,
            num_cpus_per_worker=2
        )
        config.rollouts(
            num_rollout_workers=self.num_workers,
            rollout_fragment_length=512
        )
        config.environment(
            env=SymbolicDiscoveryEnv,
            env_config=self.env_config
        )
        config.model(
            custom_model="hypothesis_net",
            custom_model_config={
                "hidden_dim": 256,
                "encoder_type": "transformer",
                "grammar": self.grammar
            }
        )
        
        # Run tuning
        analysis = tune.run(
            "PPO",
            config=config.to_dict(),
            stop={"training_iteration": num_iterations},
            num_samples=4,  # Population size
            scheduler=pbt_scheduler,
            checkpoint_at_end=True,
            local_dir=checkpoint_dir,
            verbose=1
        )
        
        # Get best trial
        best_trial = analysis.get_best_trial("episode_reward_mean", "max")
        
        return best_trial
    
    def parallel_hypothesis_search(self,
                                 policy_weights: Dict,
                                 num_rounds: int = 10,
                                 episodes_per_round: int = 50):
        """Run parallel hypothesis search across workers."""
        
        all_discoveries = []
        
        for round_idx in range(num_rounds):
            print(f"\nRound {round_idx + 1}/{num_rounds}")
            
            # Distribute work across workers
            futures = []
            episodes_per_worker = episodes_per_round // self.num_workers
            
            for worker in self.workers:
                future = worker.run_experiment.remote(
                    policy_weights,
                    n_episodes=episodes_per_worker
                )
                futures.append(future)
            
            # Collect results
            results = ray.get(futures)
            
            # Aggregate discoveries
            round_discoveries = []
            for result in results:
                round_discoveries.extend(result['discoveries'])
                
                # Log worker performance
                print(f"  Worker {result['worker_id']}: "
                      f"Avg Reward = {result['avg_reward']:.3f}, "
                      f"Best MSE = {result['best_mse']:.3e}")
            
            all_discoveries.extend(round_discoveries)
            
            # Update shared repository
            ray.get(self.discovery_repo.add_discoveries.remote(round_discoveries))
            
            # Get repository statistics
            repo_stats = ray.get(self.discovery_repo.get_statistics.remote())
            print(f"  Repository: {repo_stats['total_unique']} unique discoveries")
            print(f"  Top discovery: {repo_stats['best_discovery']}")
        
        return all_discoveries
    
    def adaptive_curriculum_search(self,
                                 initial_policy_weights: Dict,
                                 num_stages: int = 5):
        """Adaptive curriculum with distributed evaluation."""
        
        policy_weights = initial_policy_weights
        
        for stage in range(num_stages):
            print(f"\n{'='*60}")
            print(f"Curriculum Stage {stage + 1}/{num_stages}")
            print(f"{'='*60}")
            
            # Update environment difficulty
            self._update_curriculum_stage(stage)
            
            # Run distributed search
            discoveries = self.parallel_hypothesis_search(
                policy_weights,
                num_rounds=5,
                episodes_per_round=100
            )
            
            # Evaluate success rate
            success_rate = self._evaluate_stage_success(discoveries)
            print(f"Stage success rate: {success_rate:.2%}")
            
            # Fine-tune policy if needed
            if success_rate < 0.7:
                print("Fine-tuning policy for current difficulty...")
                policy_weights = self._finetune_policy(policy_weights, stage)
        
        return policy_weights
    
    def _update_curriculum_stage(self, stage: int):
        """Update environment configuration for curriculum stage."""
        
        # Progressive difficulty
        difficulty_configs = [
            {'max_depth': 3, 'max_complexity': 5},
            {'max_depth': 5, 'max_complexity': 10},
            {'max_depth': 7, 'max_complexity': 15},
            {'max_depth': 10, 'max_complexity': 25},
            {'max_depth': 12, 'max_complexity': 40}
        ]
        
        if stage < len(difficulty_configs):
            config = difficulty_configs[stage]
            
            # Update all workers
            for worker in self.workers:
                ray.get(worker.env.update_config.remote(config))
    
    def _evaluate_stage_success(self, discoveries: List[str]) -> float:
        """Evaluate success rate for current stage."""
        
        if not discoveries:
            return 0.0
        
        # Get evaluation results
        eval_results = ray.get(
            self.evaluator.evaluate_batch.remote(
                discoveries,
                self.env_config['target_data'],
                self.env_config['variables']
            )
        )
        
        # Success = MSE below threshold and reasonable complexity
        successes = sum(
            1 for r in eval_results 
            if r['mse'] < 0.1 and r['complexity'] < 20
        )
        
        return successes / len(eval_results)
    
    def _finetune_policy(self, 
                        policy_weights: Dict,
                        stage: int) -> Dict:
        """Fine-tune policy for current stage."""
        
        # Quick fine-tuning with higher learning rate
        config = PPOConfig()
        config.training(lr=1e-3, train_batch_size=1024)
        config.rollouts(num_rollout_workers=2)
        
        # Run short training
        trainer = config.build(env=SymbolicDiscoveryEnv)
        trainer.set_weights({"default_policy": policy_weights})
        
        for _ in range(10):
            trainer.train()
        
        return trainer.get_weights()["default_policy"]


@ray.remote
class DiscoveryRepository:
    """Centralized repository for discovered expressions."""
    
    def __init__(self):
        self.discoveries = {}
        self.metadata = {}
    
    def add_discoveries(self, expressions: List[str]):
        """Add new discoveries with metadata."""
        
        timestamp = time.time()
        
        for expr in expressions:
            if expr not in self.discoveries:
                self.discoveries[expr] = {
                    'first_discovered': timestamp,
                    'count': 0
                }
            
            self.discoveries[expr]['count'] += 1
            self.discoveries[expr]['last_seen'] = timestamp
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get repository statistics."""
        
        if not self.discoveries:
            return {
                'total_unique': 0,
                'total_occurrences': 0,
                'best_discovery': None
            }
        
        total_occurrences = sum(d['count'] for d in self.discoveries.values())
        
        # Find most common discovery
        best = max(self.discoveries.items(), key=lambda x: x[1]['count'])
        
        return {
            'total_unique': len(self.discoveries),
            'total_occurrences': total_occurrences,
            'best_discovery': best[0],
            'best_count': best[1]['count']
        }
    
    def get_top_discoveries(self, n: int = 10) -> List[Tuple[str, Dict]]:
        """Get top N discoveries by occurrence count."""
        
        sorted_discoveries = sorted(
            self.discoveries.items(),
            key=lambda x: x[1]['count'],
            reverse=True
        )
        
        return sorted_discoveries[:n]


# Utility functions for distributed training
def create_placement_group_for_training(num_workers: int, 
                                      gpus_per_worker: float = 0.5) -> Any:
    """Create Ray placement group for efficient resource allocation."""
    
    bundles = []
    
    # Driver bundle
    bundles.append({"CPU": 2, "GPU": 0.5})
    
    # Worker bundles
    for _ in range(num_workers):
        bundles.append({
            "CPU": 2,
            "GPU": gpus_per_worker
        })
    
    pg = placement_group(bundles, strategy="SPREAD")
    ray.get(pg.ready())
    
    return pg


def distributed_hyperparameter_search(grammar: ProgressiveGrammar,
                                    env_config: Dict[str, Any],
                                    search_space: Dict[str, Any],
                                    num_trials: int = 20):
    """Run distributed hyperparameter search."""
    
    # Define search space
    config = {
        "env": SymbolicDiscoveryEnv,
        "env_config": env_config,
        "framework": "torch",
        "model": {
            "custom_model": "hypothesis_net",
            "custom_model_config": {
                "hidden_dim": tune.choice([128, 256, 512]),
                "encoder_type": tune.choice(["transformer", "treelstm"]),
                "grammar": grammar
            }
        },
        "lr": tune.loguniform(1e-5, 1e-3),
        "entropy_coeff": tune.uniform(0.0, 0.1),
        "train_batch_size": tune.choice([2048, 4096, 8192]),
        "num_sgd_iter": tune.choice([5, 10, 15])
    }
    
    # Add custom search parameters
    config.update(search_space)
    
    # Run search
    analysis = tune.run(
        "PPO",
        config=config,
        num_samples=num_trials,
        stop={"training_iteration": 50},
        metric="episode_reward_mean",
        mode="max",
        resources_per_trial={"cpu": 4, "gpu": 0.5}
    )
    
    return analysis.best_config


# Example usage
if __name__ == "__main__":
    # Setup
    grammar = ProgressiveGrammar()
    variables = [
        Variable("x", 0, {"smoothness": 0.9}),
        Variable("v", 1, {"conservation_score": 0.8})
    ]
    
    # Generate data
    n_samples = 1000
    x_data = np.linspace(-2, 2, n_samples)
    v_data = -2 * x_data  # Linear relationship
    energy = x_data**2 + v_data**2  # Quadratic
    data = np.column_stack([x_data, v_data, energy])
    
    # Environment config
    env_config = {
        'grammar': grammar,
        'target_data': data,
        'variables': variables,
        'max_depth': 5,
        'max_complexity': 15
    }
    
    # Create trainer
    trainer = DistributedJanusTrainer(
        grammar=grammar,
        env_config=env_config,
        num_workers=4,
        num_gpus=2
    )
    
    print("Starting distributed training...")
    
    # Train with PBT
    best_trial = trainer.train_with_pbt(num_iterations=50)
    
    print(f"\nBest trial config: {best_trial.config}")
    print(f"Best reward: {best_trial.last_result['episode_reward_mean']}")
    
    # Shutdown Ray
    ray.shutdown()
