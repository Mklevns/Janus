"""
Integrated Advanced Training Pipeline for Janus
==============================================

Combines all advanced training components into a unified system
with automatic optimization and adaptation.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any, Type
from pathlib import Path
import yaml
from janus.ai_interpretability.utils.math_utils import validate_inputs, safe_import
# from dataclasses import dataclass, field # No longer needed for JanusConfig
import time
from pydantic import BaseModel, Field, model_validator # BaseModel still needed for other configs
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Dict, List, Optional, Any # Ensure Any is available
from janus.config.models import JanusConfig, SyntheticDataParamsConfig, RewardConfig, CurriculumStageConfig, RayConfig
from janus.config.loader import ConfigLoader # Added ConfigLoader

# Handle optional imports using safe_import
ray = safe_import("ray", "ray")
HAS_RAY = ray is not None
if not HAS_RAY:
    print("⚠️  Ray not installed. Distributed features will be disabled.")

wandb = safe_import("wandb", "wandb")
HAS_WANDB = wandb is not None
if not HAS_WANDB:
    print("⚠️  W&B not installed. Experiment tracking will be disabled.")

# Import all custom components
from janus.core.grammar import ProgressiveGrammar
from janus.core.expression import Variable
from janus.ai_interpretability.environments import SymbolicDiscoveryEnv, CurriculumManager # Ensure SymbolicDiscoveryEnv is imported
from enhanced_feedback import EnhancedSymbolicDiscoveryEnv, IntrinsicRewardCalculator, AdaptiveTrainingController
from hypothesis_policy_network import HypothesisNet
from physics_discovery_extensions import ConservationDetector, SymbolicRegressor
from experiment_runner import ExperimentRunner, ExperimentConfig

# Import new components (from previous artifacts)
# from multiagent_selfplay import MultiAgentPPOTrainer, LeaguePlayManager, AdversarialDiscoveryEnv
# from distributed_training import DistributedJanusTrainer, DistributedExperimentWorker
# from emergent_monitor import EmergentBehaviorTracker


class AdvancedJanusTrainer:
    """
    Master trainer that orchestrates all advanced training components.
    """
    
    @validate_inputs
    def __init__(self, config: JanusConfig):
        self.config: JanusConfig = config
        
        # Setup directories
        self._setup_directories()
        
        # Initialize components
        self.grammar = ProgressiveGrammar()
        self.variables = []
        self.env = None # type: ignore # Will be SymbolicDiscoveryEnv or similar
        self.trainer = None # type: ignore # Will be PPOTrainer or similar
        
        # Advanced components
        self.league_manager = None # type: ignore # Optional, from multiagent_selfplay
        self.distributed_trainer = None # type: ignore # Optional, from distributed_training
        self.emergent_tracker = None # type: ignore # Optional, from emergent_monitor
        self.curriculum_manager = None # type: ignore # Optional
        
        # Initialize based on mode
        self._initialize_mode()

        # Experiment runner for validation suite, configured with strict_mode
        # This runner is used if self.run_experiment_suite() is called.
        self.validation_experiment_runner = ExperimentRunner(
            base_dir=self.config.results_dir, # Use existing results_dir
            use_wandb=bool(self.config.wandb_project),
            strict_mode=self.config.strict_mode # Pass strict_mode here
        )
        
    def _setup_directories(self):
        """Create necessary directories."""
        for dir_path in [self.config.data_dir, 
                        self.config.checkpoint_dir, 
                        self.config.results_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def _initialize_mode(self):
        """Initialize components based on training mode."""
        
        if self.config.training_mode in ["selfplay", "advanced"]:
            try:
                from multiagent_selfplay import LeaguePlayManager
                self.league_manager = LeaguePlayManager(
                    base_policy=None,  # Will be set later
                    max_league_size=self.config.league_size
                )
            except ImportError:
                print("⚠️  Multi-agent self-play components not available")
                self.league_manager = None
        
        if self.config.training_mode in ["distributed", "advanced"]:
            if HAS_RAY and ray: # Check both HAS_RAY and if ray module is not None
                if not ray.is_initialized():
                    ray.init(num_cpus=self.config.num_workers * 2, 
                            num_gpus=self.config.num_gpus)
            elif not HAS_RAY: # Only print if HAS_RAY is False (already printed by safe_import)
                # The initial print from safe_import already covers this.
                # print("⚠️  Ray not available. Distributed features disabled.")
                pass
            else: # HAS_RAY is True but ray is None (should not happen with current safe_import)
                 print("Unexpected: HAS_RAY is True but 'ray' module is None.")
                if self.config.training_mode == "distributed":
                    print("❌ Cannot run in distributed mode without Ray")
                    raise RuntimeError("Ray required for distributed training")
        
        if self.config.track_emergence:
            try:
                from emergent_monitor import EmergentBehaviorTracker
                self.emergent_tracker = EmergentBehaviorTracker(
                    save_dir=Path(self.config.results_dir) / "emergence"
                )
            except ImportError:
                print("⚠️  Emergent behavior tracking not available")
                self.emergent_tracker = None
        
        # Initialize W&B
        if self.config.wandb_project and HAS_WANDB and wandb: # Check both HAS_WANDB and if wandb module is not None
            wandb.init(
                project=self.config.wandb_project,
                config=self.config.model_dump(), # Use model_dump() for Pydantic models
                name=f"janus_{self.config.training_mode}_{int(time.time())}"
            )
        elif self.config.wandb_project and not HAS_WANDB:
            # The initial print from safe_import already covers this.
            # print("⚠️  W&B tracking requested but wandb not installed")
            pass
        elif self.config.wandb_project and HAS_WANDB and not wandb: # Should not happen
            print("Unexpected: HAS_WANDB is True but 'wandb' module is None.")

    
    @validate_inputs
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
    
    @validate_inputs
    def create_environment(self, data: np.ndarray) -> SymbolicDiscoveryEnv: # Added SymbolicDiscoveryEnv hint
        """Create the discovery environment."""
        
        # Use reward config from JanusConfig if available
        reward_config = self.config.reward_config or {
            'completion_bonus': 0.1,
            'mse_weight': -1.0,
            'complexity_penalty': -0.01,
            'depth_penalty': -0.001
        }
        
        env_config = {
            'grammar': self.grammar,
            'target_data': data,
            'variables': self.variables,
            'max_depth': self.config.max_depth,
            'max_complexity': self.config.max_complexity,
            'reward_config': self.config.reward_config.model_dump() if self.config.reward_config else reward_config
        }
        
        # Create base environment
        if self.config.training_mode == "advanced":
            try:
                env = EnhancedSymbolicDiscoveryEnv(**env_config)
            except Exception:
                print("⚠️  Enhanced environment not available, using standard environment")
                env = SymbolicDiscoveryEnv(**env_config)
        elif self.config.training_mode == "selfplay":
            try:
                from multiagent_selfplay import AdversarialDiscoveryEnv
                env = AdversarialDiscoveryEnv(**env_config)
            except ImportError:
                print("⚠️  AdversarialDiscoveryEnv not available, using standard environment")
                env = SymbolicDiscoveryEnv(**env_config)
        else:
            env = SymbolicDiscoveryEnv(**env_config)
        
        # Wrap with curriculum if enabled
        if self.config.use_curriculum:
            self.curriculum_manager = CurriculumManager(env)
            env = self.curriculum_manager.get_current_env()
        
        return env
    
    # Not decorating create_trainer as it's more of an internal setup method
    # and its validation depends heavily on prior state (self.env).
    def create_trainer(self):
        """Create the appropriate trainer based on mode."""
        
        # Determine observation dimension
        obs_dim = self.env.observation_space.shape[0]
        if isinstance(self.env, EnhancedSymbolicDiscoveryEnv):
            obs_dim = self.env.observation_encoder.enhance_observation(
                np.zeros(obs_dim),
                self.env.current_state,
                self.grammar
            ).shape[0]

        # Create policy network
        policy = HypothesisNet(
            observation_dim=obs_dim,
            action_dim=self.env.action_space.n,
            hidden_dim=256,
            encoder_type='transformer',
            grammar=self.grammar
        )
        
        if self.config.training_mode == "basic":
            from hypothesis_policy_network import PPOTrainer
            trainer = PPOTrainer(policy, self.env)
            
        elif self.config.training_mode == "selfplay":
            try:
                from multiagent_selfplay import MultiAgentPPOTrainer
                trainer = MultiAgentPPOTrainer(policy, self.env)
                if self.league_manager:
                    self.league_manager.base_policy = policy
            except ImportError:
                print("⚠️  Falling back to basic trainer (multiagent module not found)")
                from hypothesis_policy_network import PPOTrainer
                trainer = PPOTrainer(policy, self.env)
            
        elif self.config.training_mode == "distributed":
            if HAS_RAY:
                try:
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
                except ImportError:
                    print("⚠️  Distributed training module not found")
                    raise
            else:
                raise RuntimeError("Cannot create distributed trainer without Ray")
            
        elif self.config.training_mode == "advanced":
            # Try to use advanced features, fall back gracefully
            try:
                from multiagent_selfplay import MultiAgentPPOTrainer
                trainer = MultiAgentPPOTrainer(policy, self.env)
                if self.league_manager:
                    self.league_manager.base_policy = policy
            except ImportError:
                print("⚠️  Using basic trainer for advanced mode")
                from hypothesis_policy_network import PPOTrainer
                trainer = PPOTrainer(policy, self.env)
            
            # Add distributed components if enough resources
            if self.config.num_gpus > 1 and HAS_RAY and ray: # Check ray module
                self._setup_distributed_components()
        
        else:
            raise ValueError(f"Unknown training mode: {self.config.training_mode}")
        
        # Integrate emergent tracking
        if self.emergent_tracker:
            try:
                from emergent_monitor import integrate_emergent_tracking
                integrate_emergent_tracking(trainer, self.emergent_tracker)
            except ImportError:
                print("⚠️  Could not integrate emergent tracking")
        
        return trainer
    
    # Internal helper, not decorating
    def _setup_distributed_components(self):
        """Setup distributed training components."""
        try:
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
        except ImportError:
            print("⚠️  Distributed components not available")
            self.evaluators = []
            self.experiment_workers = []
    
    @validate_inputs
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
        # Use parameters from the configuration for consistency and control.
        self.trainer.train(
            total_timesteps=self.config.total_timesteps,
            rollout_length=self.config.ppo_rollout_length,
            n_epochs=self.config.ppo_n_epochs,
            log_interval=self.config.log_interval
        )
    
    def _train_selfplay(self):
        """Self-play training loop."""
        if hasattr(self.trainer, 'train_selfplay'):
            self.trainer.train_selfplay(
                total_timesteps=self.config.total_timesteps,
                rollout_length=2048,
                n_epochs=10,
                league_update_interval=10000
            )
        else:
            print("⚠️  Self-play not available, using basic training")
            self._train_basic()
    
    def _train_distributed(self):
        """Distributed training loop."""
        if hasattr(self.trainer, 'train_with_pbt'):
            if self.config.use_pbt:
                # Population-based training
                best_trial = self.trainer.train_with_pbt(
                    num_iterations=self.config.total_timesteps // 10000
                )
                print(f"Best trial: {best_trial}")
            else:
                # Adaptive curriculum search
                if hasattr(self.trainer, 'adaptive_curriculum_search'):
                    policy_weights = self.trainer.adaptive_curriculum_search(
                        initial_policy_weights=self.trainer.policy.state_dict(),
                        num_stages=5
                    )
                else:
                    print("⚠️  Adaptive curriculum not available")
                    self._train_basic()
        else:
            print("⚠️  Distributed training not available, using basic training")
            self._train_basic()
    
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
        if hasattr(self.trainer, 'train_selfplay'):
            self.trainer.train_selfplay(
                total_timesteps=timesteps_per_phase,
                rollout_length=2048,
                n_epochs=10
            )
        else:
            print("⚠️  Self-play not available, continuing with basic training")
            self.trainer.train(
                total_timesteps=timesteps_per_phase,
                rollout_length=2048,
                n_epochs=10
            )
        
        # Phase 3: Distributed exploration (if available)
        if hasattr(self, 'experiment_workers') and self.experiment_workers:
            print("\nPhase 3: Distributed Exploration")
            policy_weights = self.trainer.policy.state_dict()
            
            try:
                discoveries = self.trainer.parallel_hypothesis_search(
                    policy_weights,
                    num_rounds=10,
                    episodes_per_round=100
                )
                
                print(f"Discovered {len(set(discoveries))} unique expressions")
            except AttributeError:
                print("⚠️  Parallel hypothesis search not available in trainer")
        else:
            print("\nPhase 3: Distributed Exploration (skipped - no workers available)")
        
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
        # The ExperimentRunner for validation is now self.validation_experiment_runner
        # We need to create ExperimentConfig instances for it.
        # This part might be better handled by run_experiment_suite or specific validation methods.
        # For now, let's comment out direct ExperimentConfig creation here as it's complex
        # and better suited for the experiment_runner script's validation functions.

        # experiment_config = ExperimentConfig.from_janus_config(
        # name=f"final_eval_{self.config.target_phenomena}",
        # experiment_type='physics_discovery_example',
        # janus_config=self.config, # Pass the full JanusConfig
        # algorithm_name='janus_full', # Example
        # n_runs=1 # Typically final eval is one detailed run
        # )
        #
        # if self.validation_experiment_runner:
        # self.validation_experiment_runner.run_single_experiment(experiment_config)
        # else:
        # print("⚠️ Validation experiment runner not initialized, skipping final evaluation run through it.")

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
            'config': self.config.model_dump(), # Use model_dump for Pydantic models
            'variables': [(v.name, v.index, v.properties) for v in self.variables]
        }, checkpoint_path)
        
        print(f"\nModel saved to {checkpoint_path}")
        
        # Close W&B
        if self.config.wandb_project and HAS_WANDB and wandb: # Check wandb module
            wandb.finish()
    
    def run_experiment_suite(self):
        """Run full experiment suite for validation."""
        
        print("\nRunning experiment suite...")
        
        # Phase 1: Known law rediscovery
        # These functions are defined in experiment_runner.py and create their own ExperimentRunner.
        # To pass strict_mode, we'd need to modify those functions or how they are called.
        # For now, this trainer's self.validation_experiment_runner (with strict_mode)
        # isn't directly used by these imported functions.
        # This will be handled in launch_advanced_training.py for args.mode == 'validate'

        phase1_results_df = None
        phase2_results_df = None
        try:
            from experiment_runner import run_phase1_validation, run_phase2_robustness
            # If these functions are to use the strict_mode from *this* trainer's config,
            # they would need to accept it as an argument.
            # For now, they will run with their default ExperimentRunner settings.
            # If strict_mode is critical for these validation runs when triggered from AdvancedJanusTrainer,
            # then run_phase1_validation etc. need a strict_mode param.
            # This is addressed by how launch_advanced_training.py calls them.
            if self.config.validation_phases and "phase1" in self.config.validation_phases:
                print("\nRunning Phase 1 Validation (as part of AdvancedJanusTrainer suite)...")
                phase1_results_df = run_phase1_validation(strict_mode_override=self.config.strict_mode)
            if self.config.validation_phases and "phase2" in self.config.validation_phases:
                print("\nRunning Phase 2 Robustness (as part of AdvancedJanusTrainer suite)...")
                phase2_results_df = run_phase2_robustness(strict_mode_override=self.config.strict_mode)

        except ImportError as e:
            print(f"⚠️  Could not run validation suites: {e}")
        
        # Visualize results
        if phase1_results_df is not None or phase2_results_df is not None:
            try:
                from janus.ai_interpretability.utils.visualization import ExperimentVisualizer
                visualizer = ExperimentVisualizer(results_dir=str(Path(self.config.results_dir) / "trainer_suite_viz"))
                
                all_validation_results = []
                if phase1_results_df is not None:
                    all_validation_results.append(phase1_results_df)
                if phase2_results_df is not None:
                    all_validation_results.append(phase2_results_df)

                if all_validation_results:
                    try:
                        import pandas as pd
                        final_df_for_report = pd.concat(all_validation_results)
                        if not final_df_for_report.empty:
                             visualizer.create_summary_report(
                                final_df_for_report,
                                output_path=Path(self.config.results_dir) / "trainer_validation_report.html"
                            )
                             print(f"Trainer validation report saved to {self.config.results_dir}/trainer_validation_report.html")
                        else:
                            print("⚠️  No data in final_df_for_report for visualization.")
                    except ImportError:
                        print("⚠️  Pandas not installed. Skipping HTML report generation for trainer suite.")
                    except Exception as e_concat: # Catch errors during concat or report generation
                        print(f"⚠️  Error generating combined report for trainer suite: {e_concat}")

            except ImportError:
                print("⚠️  ExperimentVisualizer not available for trainer suite.")
        else:
            print("⚠️  No validation results from trainer suite to visualize.")


def main():
    """Main entry point for advanced Janus training."""
    
    # Load configuration using ConfigLoader
    # Assumes advanced_training.yaml was renamed to default.yaml in config/
    config_path = "config/default.yaml"
    
    try:
        loader = ConfigLoader(primary_config_path=config_path)
        config = loader.load_resolved_config() # This returns a JanusConfig object
        print(f"✓ Configuration loaded successfully from {config_path} and environment variables.")
    except FileNotFoundError:
        print(f"❌ Error: Configuration file {config_path} not found. Exiting.")
        return
    except ValueError as e: # Catch Pydantic validation errors from loader
        print(f"❌ Error: Configuration validation failed: {e}. Exiting.")
        return
    except Exception as e:
        print(f"❌ Error: An unexpected error occurred during configuration loading: {e}. Exiting.")
        return

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
    if HAS_RAY and ray and ray.is_initialized(): # Check ray module
        ray.shutdown()


if __name__ == "__main__":
    main()
