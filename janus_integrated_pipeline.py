"""
Integrated Advanced Training Pipeline for Janus
==============================================

Combines all advanced training components into a unified system
with automatic optimization and adaptation.
"""

import numpy as np
import torch
import ray
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import yaml
import wandb
from dataclasses import dataclass
import time

# Import all custom components
from progressive_grammar_system import ProgressiveGrammar, Variable
from symbolic_discovery_env import SymbolicDiscoveryEnv, CurriculumManager
from hypothesis_policy_network import HypothesisNet
from physics_discovery_extensions import ConservationDetector, SymbolicRegressor
from experiment_runner import ExperimentRunner, ExperimentConfig

# Import new components (from previous artifacts)
# from multiagent_selfplay import MultiAgentPPOTrainer, LeaguePlayManager, AdversarialDiscoveryEnv
# from distributed_training import DistributedJanusTrainer, DistributedExperimentWorker
# from emergent_monitor import EmergentBehaviorTracker


@dataclass
class JanusConfig:
    """Master configuration for Janus training."""
    
    # Environment
    env_type: str = "physics_discovery"
    max_depth: int = 10
    max_complexity: int = 30
    target_phenomena: str = "harmonic_oscillator"  # or "pendulum", "kepler", etc.
    
    # Training
    training_mode: str = "advanced"  # "basic", "selfplay", "distributed", "advanced"
    total_timesteps: int = 1_000_000
    n_agents: int = 4
    use_curriculum: bool = True
    
    # Self-play
    league_size: int = 50
    opponent_sampling: str = "prioritized_quality_diversity"
    
    # Distributed
    num_workers: int = 8
    num_gpus: int = 4
    use_pbt: bool = True
    
    # Monitoring
    track_emergence: bool = True
    wandb_project: str = "janus-physics-discovery"
    checkpoint_freq: int = 10000
    
    # Paths
    data_dir: str = "./data"
    checkpoint_dir: str = "./checkpoints"
    results_dir: str = "./results"
    
    @classmethod
    def from_yaml(cls, path: str) -> 'JanusConfig':
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)


class AdvancedJanusTrainer:
    """
    Master trainer that orchestrates all advanced training components.
    """
    
    def __init__(self, config: JanusConfig):
        self.config = config
        
        # Setup directories
        self._setup_directories()
        
        # Initialize components
        self.grammar = ProgressiveGrammar()
        self.variables = []
        self.env = None
        self.trainer = None
        
        # Advanced components
        self.league_manager = None
        self.distributed_trainer = None
        self.emergent_tracker = None
        self.curriculum_manager = None
        
        # Initialize based on mode
        self._initialize_mode()
        
    def _setup_directories(self):
        """Create necessary directories."""
        for dir_path in [self.config.data_dir, 
                        self.config.checkpoint_dir, 
                        self.config.results_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def _initialize_mode(self):
        """Initialize components based on training mode."""
        
        if self.config.training_mode in ["selfplay", "advanced"]:
            from multiagent_selfplay import LeaguePlayManager
            self.league_manager = LeaguePlayManager(
                base_policy=None,  # Will be set later
                max_league_size=self.config.league_size
            )
        
        if self.config.training_mode in ["distributed", "advanced"]:
            if not ray.is_initialized():
                ray.init(num_cpus=self.config.num_workers * 2, 
                        num_gpus=self.config.num_gpus)
        
        if self.config.track_emergence:
            from emergent_monitor import EmergentBehaviorTracker
            self.emergent_tracker = EmergentBehaviorTracker(
                save_dir=Path(self.config.results_dir) / "emergence"
            )
        
        # Initialize W&B
        if self.config.wandb_project:
            wandb.init(
                project=self.config.wandb_project,
                config=self.config.__dict__,
                name=f"janus_{self.config.training_mode}_{int(time.time())}"
            )
    
    def prepare_data(self, 
                    data_path: Optional[str] = None,
                    generate_synthetic: bool = True) -> np.ndarray:
        """Prepare training data."""
        
        if data_path and Path(data_path).exists():
            # Load real data
            data = np.load(data_path)
            print(f"Loaded data from {data_path}: shape {data.shape}")
        
        elif generate_synthetic:
            # Generate synthetic physics data
            data = self._generate_synthetic_data()
            print(f"Generated synthetic {self.config.target_phenomena} data")
        
        else:
            raise ValueError("No data provided")
        
        # Discover variables from data
        print("\nDiscovering variables...")
        self.variables = self.grammar.discover_variables(data)
        print(f"Discovered {len(self.variables)} variables:")
        for var in self.variables:
            print(f"  - {var.name}: {var.properties}")
        
        return data
    
    def _generate_synthetic_data(self) -> np.ndarray:
        """Generate synthetic physics data based on target phenomena."""
        
        n_samples = 2000
        
        if self.config.target_phenomena == "harmonic_oscillator":
            t = np.linspace(0, 20, n_samples)
            x = np.sin(2 * np.pi * 0.5 * t) + 0.05 * np.random.randn(n_samples)
            v = 2 * np.pi * 0.5 * np.cos(2 * np.pi * 0.5 * t) + 0.05 * np.random.randn(n_samples)
            energy = 0.5 * (x**2 + v**2)
            data = np.column_stack([x, v, energy])
            
        elif self.config.target_phenomena == "pendulum":
            t = np.linspace(0, 20, n_samples)
            theta = 0.2 * np.sin(np.sqrt(9.81) * t) + 0.02 * np.random.randn(n_samples)
            omega = 0.2 * np.sqrt(9.81) * np.cos(np.sqrt(9.81) * t) + 0.02 * np.random.randn(n_samples)
            energy = 0.5 * omega**2 + 9.81 * (1 - np.cos(theta))
            data = np.column_stack([theta, omega, energy])
            
        elif self.config.target_phenomena == "kepler":
            # Circular orbit
            t = np.linspace(0, 10, n_samples)
            r = 1.0 + 0.01 * np.random.randn(n_samples)
            theta = t + 0.01 * np.random.randn(n_samples)
            vr = 0.01 * np.random.randn(n_samples)
            vtheta = 1.0 / r + 0.01 * np.random.randn(n_samples)
            energy = 0.5 * (vr**2 + r**2 * vtheta**2) - 1.0 / r
            angular_momentum = r**2 * vtheta
            data = np.column_stack([r, theta, vr, vtheta, energy, angular_momentum])
            
        else:
            raise ValueError(f"Unknown phenomena: {self.config.target_phenomena}")
        
        # Save generated data
        save_path = Path(self.config.data_dir) / f"{self.config.target_phenomena}_synthetic.npy"
        np.save(save_path, data)
        
        return data
    
    def create_environment(self, data: np.ndarray) -> SymbolicDiscoveryEnv:
        """Create the discovery environment."""
        
        env_config = {
            'grammar': self.grammar,
            'target_data': data,
            'variables': self.variables,
            'max_depth': self.config.max_depth,
            'max_complexity': self.config.max_complexity,
            'reward_config': {
                'completion_bonus': 0.1,
                'mse_weight': -1.0,
                'complexity_penalty': -0.01,
                'depth_penalty': -0.001
            }
        }
        
        # Create base environment
        if self.config.training_mode == "selfplay":
            from multiagent_selfplay import AdversarialDiscoveryEnv
            env = AdversarialDiscoveryEnv(**env_config)
        else:
            env = SymbolicDiscoveryEnv(**env_config)
        
        # Wrap with curriculum if enabled
        if self.config.use_curriculum:
            self.curriculum_manager = CurriculumManager(env)
            env = self.curriculum_manager.get_current_env()
        
        return env
    
    def create_trainer(self):
        """Create the appropriate trainer based on mode."""
        
        # Create policy network
        policy = HypothesisNet(
            observation_dim=self.env.observation_space.shape[0],
            action_dim=self.env.action_space.n,
            hidden_dim=256,
            encoder_type='transformer',
            grammar=self.grammar
        )
        
        if self.config.training_mode == "basic":
            from hypothesis_policy_network import PPOTrainer
            trainer = PPOTrainer(policy, self.env)
            
        elif self.config.training_mode == "selfplay":
            from multiagent_selfplay import MultiAgentPPOTrainer
            trainer = MultiAgentPPOTrainer(policy, self.env)
            self.league_manager.base_policy = policy
            
        elif self.config.training_mode == "distributed":
            from distributed_training import DistributedJanusTrainer
            trainer = DistributedJanusTrainer(
                grammar=self.grammar,
                env_config={
                    'target_data': self.env.target_data,
                    'variables': self.variables,
                    'max_depth': self.config.max_depth,
                    'max_complexity': self.config.max_complexity
                },
                num_workers=self.config.num_workers,
                num_gpus=self.config.num_gpus
            )
            
        elif self.config.training_mode == "advanced":
            # Combine all advanced features
            from multiagent_selfplay import MultiAgentPPOTrainer
            trainer = MultiAgentPPOTrainer(policy, self.env)
            self.league_manager.base_policy = policy
            
            # Add distributed components if enough resources
            if self.config.num_gpus > 1:
                self._setup_distributed_components()
        
        else:
            raise ValueError(f"Unknown training mode: {self.config.training_mode}")
        
        # Integrate emergent tracking
        if self.emergent_tracker:
            from emergent_monitor import integrate_emergent_tracking
            integrate_emergent_tracking(trainer, self.emergent_tracker)
        
        return trainer
    
    def _setup_distributed_components(self):
        """Setup distributed training components."""
        from distributed_training import AsyncExpressionEvaluator, DistributedExperimentWorker
        
        # Create evaluator pool
        self.evaluators = [
            AsyncExpressionEvaluator.remote(self.grammar)
            for _ in range(4)
        ]
        
        # Create experiment workers
        self.experiment_workers = [
            DistributedExperimentWorker.remote(
                i, 
                self.grammar,
                {
                    'target_data': self.env.target_data,
                    'variables': self.variables,
                    'max_depth': self.config.max_depth,
                    'max_complexity': self.config.max_complexity
                }
            )
            for i in range(self.config.num_workers)
        ]
    
    def train(self):
        """Main training loop."""
        
        print(f"\nStarting {self.config.training_mode} training...")
        print(f"Total timesteps: {self.config.total_timesteps}")
        
        if self.config.training_mode == "basic":
            self._train_basic()
            
        elif self.config.training_mode == "selfplay":
            self._train_selfplay()
            
        elif self.config.training_mode == "distributed":
            self._train_distributed()
            
        elif self.config.training_mode == "advanced":
            self._train_advanced()
        
        # Final analysis
        self._final_analysis()
    
    def _train_basic(self):
        """Basic training loop."""
        self.trainer.train(
            total_timesteps=self.config.total_timesteps,
            rollout_length=2048,
            n_epochs=10,
            log_interval=10
        )
    
    def _train_selfplay(self):
        """Self-play training loop."""
        self.trainer.train_selfplay(
            total_timesteps=self.config.total_timesteps,
            rollout_length=2048,
            n_epochs=10,
            league_update_interval=10000
        )
    
    def _train_distributed(self):
        """Distributed training loop."""
        if self.config.use_pbt:
            # Population-based training
            best_trial = self.trainer.train_with_pbt(
                num_iterations=self.config.total_timesteps // 10000
            )
            print(f"Best trial: {best_trial}")
        else:
            # Adaptive curriculum search
            policy_weights = self.trainer.adaptive_curriculum_search(
                initial_policy_weights=self.trainer.policy.state_dict(),
                num_stages=5
            )
    
    def _train_advanced(self):
        """Advanced training combining all features."""
        
        timesteps_per_phase = self.config.total_timesteps // 4
        
        # Phase 1: Basic exploration with curriculum
        print("\nPhase 1: Basic Exploration")
        self.trainer.train(
            total_timesteps=timesteps_per_phase,
            rollout_length=1024,
            n_epochs=5
        )
        
        # Update curriculum
        if self.curriculum_manager:
            success_rate = np.mean([r > 0 for r in self.trainer.episode_rewards])
            self.curriculum_manager.update_curriculum(success_rate > 0.5)
        
        # Phase 2: Self-play refinement
        print("\nPhase 2: Self-Play Refinement")
        self.trainer.train_selfplay(
            total_timesteps=timesteps_per_phase,
            rollout_length=2048,
            n_epochs=10
        )
        
        # Phase 3: Distributed exploration (if available)
        if hasattr(self, 'experiment_workers'):
            print("\nPhase 3: Distributed Exploration")
            policy_weights = self.trainer.policy.state_dict()
            
            discoveries = self.trainer.parallel_hypothesis_search(
                policy_weights,
                num_rounds=10,
                episodes_per_round=100
            )
            
            print(f"Discovered {len(set(discoveries))} unique expressions")
        
        # Phase 4: Final optimization
        print("\nPhase 4: Final Optimization")
        self.trainer.train(
            total_timesteps=timesteps_per_phase,
            rollout_length=4096,
            n_epochs=15
        )
    
    def _final_analysis(self):
        """Perform final analysis and save results."""
        
        print("\n" + "="*60)
        print("FINAL ANALYSIS")
        print("="*60)
        
        # Run comprehensive evaluation
        experiment_config = ExperimentConfig(
            name=f"final_eval_{self.config.target_phenomena}",
            environment_type=self.config.target_phenomena,
            algorithm='janus_full',
            env_params={},
            noise_level=0.0,
            max_experiments=1000,
            n_runs=10
        )
        
        runner = ExperimentRunner(
            base_dir=self.config.results_dir,
            use_wandb=bool(self.config.wandb_project)
        )
        
        # Generate final report
        if self.emergent_tracker:
            report = self.emergent_tracker.generate_final_report()
            
            print("\nEmergent Behavior Summary:")
            print(f"  Total Discoveries: {report['total_discoveries']}")
            print(f"  Discovery Clusters: {report['discovery_clusters']}")
            print(f"  Phase Transitions: {report['phase_transitions']}")
            
            if report['top_discoveries']:
                print("\nTop Discoveries:")
                for i, disc in enumerate(report['top_discoveries'][:5]):
                    print(f"  {i+1}. {disc['expression']}")
                    print(f"     Influence: {disc['influence']:.3f}, "
                          f"Complexity: {disc['complexity']}, "
                          f"Accuracy: {disc['accuracy']:.3f}")
        
        # Save final model
        checkpoint_path = Path(self.config.checkpoint_dir) / "final_model.pt"
        torch.save({
            'policy_state_dict': self.trainer.policy.state_dict(),
            'grammar_state': self.grammar.export_grammar_state(),
            'config': self.config.__dict__,
            'variables': [(v.name, v.index, v.properties) for v in self.variables]
        }, checkpoint_path)
        
        print(f"\nModel saved to {checkpoint_path}")
        
        # Close W&B
        if self.config.wandb_project:
            wandb.finish()
    
    def run_experiment_suite(self):
        """Run full experiment suite for validation."""
        
        print("\nRunning experiment suite...")
        
        # Phase 1: Known law rediscovery
        from experiment_runner import run_phase1_validation
        phase1_results = run_phase1_validation()
        
        # Phase 2: Robustness testing
        from experiment_runner import run_phase2_robustness
        phase2_results = run_phase2_robustness()
        
        # Visualize results
        from experiment_visualizer import ExperimentVisualizer
        visualizer = ExperimentVisualizer(results_dir=self.config.results_dir)
        
        # Create comprehensive plots
        visualizer.plot_sample_efficiency_curves(phase1_results)
        visualizer.plot_noise_resilience(phase2_results)
        
        # Generate HTML report
        visualizer.create_summary_report(
            pd.concat([phase1_results, phase2_results]),
            output_path=Path(self.config.results_dir) / "experiment_report.html"
        )
        
        print(f"Experiment report saved to {self.config.results_dir}/experiment_report.html")


def main():
    """Main entry point for advanced Janus training."""
    
    # Load configuration
    config_path = "config/advanced_training.yaml"
    if Path(config_path).exists():
        config = JanusConfig.from_yaml(config_path)
    else:
        # Use default configuration
        config = JanusConfig(
            training_mode="advanced",
            target_phenomena="harmonic_oscillator",
            total_timesteps=500_000,
            use_curriculum=True,
            track_emergence=True
        )
    
    # Create trainer
    trainer = AdvancedJanusTrainer(config)
    
    # Prepare data
    data = trainer.prepare_data(generate_synthetic=True)
    
    # Create environment
    trainer.env = trainer.create_environment(data)
    
    # Create trainer
    trainer.trainer = trainer.create_trainer()
    
    # Run training
    trainer.train()
    
    # Run validation experiments
    if config.training_mode == "advanced":
        trainer.run_experiment_suite()
    
    print("\nTraining complete!")
    
    # Cleanup
    if ray.is_initialized():
        ray.shutdown()


if __name__ == "__main__":
    main()
