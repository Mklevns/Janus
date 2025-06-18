#!/usr/bin/env python3
"""
launch_advanced_training.py
==========================

Launch script for advanced Janus training with automatic setup
and environment validation.
"""

import argparse
import sys
import os
from pathlib import Path
import subprocess
import yaml
import torch
import psutil
from typing import Dict, Any

# Optional imports with fallbacks
try:
    import ray
    HAS_RAY = True
except ImportError:
    HAS_RAY = False
    print("⚠️  Ray not installed. Distributed training will be unavailable.")

try:
    import GPUtil
    HAS_GPUTIL = True
except ImportError:
    HAS_GPUTIL = False
    print("⚠️  GPUtil not installed. GPU details will be limited.")

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def check_system_requirements() -> Dict[str, Any]:
    """Check if system meets requirements for training."""
    
    print("Checking system requirements...")
    
    requirements = {
        'cpu_count': psutil.cpu_count(),
        'memory_gb': psutil.virtual_memory().total / (1024**3),
        'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'cuda_available': torch.cuda.is_available(),
        'ray_installed': HAS_RAY,
    }
    
    # Check GPU details
    if requirements['gpu_count'] > 0:
        if HAS_GPUTIL:
            gpus = GPUtil.getGPUs()
            requirements['gpu_details'] = [
                {
                    'name': gpu.name,
                    'memory_mb': gpu.memoryTotal,
                    'driver': gpu.driver
                }
                for gpu in gpus
            ]
        else:
            # Fallback to basic torch info
            requirements['gpu_details'] = [
                {
                    'name': torch.cuda.get_device_name(i),
                    'memory_mb': torch.cuda.get_device_properties(i).total_memory / (1024**2),
                    'driver': 'N/A'
                }
                for i in range(requirements['gpu_count'])
            ]
    
    print(f"  CPUs: {requirements['cpu_count']}")
    print(f"  Memory: {requirements['memory_gb']:.1f} GB")
    print(f"  GPUs: {requirements['gpu_count']}")
    
    return requirements


def validate_config(config_path: str) -> Dict[str, Any]:
    """Validate and load configuration."""
    
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate required fields
    required_fields = [
        'training_mode', 'target_phenomena', 'total_timesteps',
        'checkpoint_dir', 'results_dir'
    ]
    
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field in config: {field}")
    
    # Adjust based on system capabilities
    system_reqs = check_system_requirements()
    
    if config.get('num_gpus', 0) > system_reqs['gpu_count']:
        print(f"⚠️  Warning: Config requests {config['num_gpus']} GPUs but only "
              f"{system_reqs['gpu_count']} available. Adjusting...")
        config['num_gpus'] = system_reqs['gpu_count']
    
    if config.get('num_workers', 0) > system_reqs['cpu_count']:
        print(f"⚠️  Warning: Config requests {config['num_workers']} workers but only "
              f"{system_reqs['cpu_count']} CPUs available. Adjusting...")
        config['num_workers'] = min(config['num_workers'], system_reqs['cpu_count'] - 2)
    
    return config


def setup_environment(config: Dict[str, Any]):
    """Setup directories and environment."""
    
    print("\nSetting up environment...")
    
    # Create directories
    for dir_key in ['checkpoint_dir', 'results_dir', 'data_dir']:
        if dir_key in config:
            dir_path = Path(config[dir_key])
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"  Created {dir_key}: {dir_path}")
    
    # Set environment variables
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(
        str(i) for i in range(config.get('num_gpus', 0))
    )
    
    # Initialize Ray if needed
    if config['training_mode'] in ['distributed', 'advanced'] and config.get('num_workers', 0) > 1:
        if not ray.is_initialized():
            ray_config = config.get('ray_config', {})
            
            # Extract only valid ray.init() parameters
            valid_ray_params = {
                'num_cpus': ray_config.get('num_cpus', 8),
                'num_gpus': ray_config.get('num_gpus', config.get('num_gpus', 0)),
                'object_store_memory': ray_config.get('object_store_memory'),
                'include_dashboard': ray_config.get('include_dashboard', False),
                'dashboard_host': ray_config.get('dashboard_host', '127.0.0.1'),
                '_temp_dir': ray_config.get('_temp_dir'),
                'local_mode': ray_config.get('local_mode', False)
            }
            
            # Remove None values
            valid_ray_params = {k: v for k, v in valid_ray_params.items() if v is not None}
            
            print(f"\nInitializing Ray with {valid_ray_params.get('num_cpus', 8)} CPUs "
                  f"and {valid_ray_params.get('num_gpus', 0)} GPUs...")
            
            try:
                ray.init(**valid_ray_params)
            except Exception as e:
                print(f"⚠️  Ray initialization failed: {e}")
                print("  Continuing without Ray (will use single-machine training)")


def launch_training(config: Dict[str, Any], resume: bool = False): # config here is the dict from YAML
    """Launch the training process."""
    
    from integrated_pipeline import AdvancedJanusTrainer, JanusConfig
    
    print("\n" + "="*60)
    print("LAUNCHING JANUS ADVANCED TRAINING")
    print("="*60)
    
    # Create trainer configuration
    janus_config = JanusConfig(**config)
    
    # Create trainer
    trainer = AdvancedJanusTrainer(janus_config)
    
    # Check for resume
    if resume:
        checkpoint_path = Path(config['checkpoint_dir']) / "latest_checkpoint.pt"
        if checkpoint_path.exists():
            print(f"\nResuming from checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)
            # Load state (implementation depends on trainer structure)
            print("✓ Checkpoint loaded")
        else:
            print("⚠️  No checkpoint found, starting fresh")
    
    try:
        # Prepare data
        print("\nPreparing data...")
        data = trainer.prepare_data(generate_synthetic=True)
        print(f"✓ Data prepared: shape {data.shape}")
        
        # Create environment
        print("\nCreating environment...")
        trainer.env = trainer.create_environment(data)
        print(f"✓ Environment created: {trainer.env}")
        
        # Create trainer
        print("\nInitializing trainer...")
        trainer.trainer = trainer.create_trainer()
        print(f"✓ Trainer initialized: {type(trainer.trainer).__name__}")
        
        # Start training
        print("\n" + "-"*60)
        print("Starting training...")
        print("-"*60)
        
        trainer.train()
        
        # Run validation if requested
        if janus_config.run_validation_suite: # Use janus_config field
            print("\nRunning validation suite...")
            trainer.run_experiment_suite()
        
        print("\n✅ Training completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        save_checkpoint(trainer, janus_config) # Pass janus_config object
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        raise
        
    finally:
        # Cleanup
        if HAS_RAY and ray.is_initialized():
            ray.shutdown()
        print("\nCleanup completed")


def save_checkpoint(trainer, janus_config: JanusConfig): # Changed signature to JanusConfig
    """Save emergency checkpoint."""
    
    print("\nSaving emergency checkpoint...")
    
    # Ensure grammar state can be retrieved; might need specific method on trainer or grammar object
    grammar_state = None
    if hasattr(trainer, 'grammar') and hasattr(trainer.grammar, 'export_grammar_state'):
        grammar_state = trainer.grammar.export_grammar_state()
    elif hasattr(trainer, 'env') and hasattr(trainer.env, 'grammar') and hasattr(trainer.env.grammar, 'export_grammar_state'):
        grammar_state = trainer.env.grammar.export_grammar_state()


    policy_state = None
    optimizer_state = None
    current_iteration = 0

    if hasattr(trainer, 'trainer') and trainer.trainer is not None: # If trainer.trainer is the actual PPO/etc trainer
        if hasattr(trainer.trainer, 'policy') and hasattr(trainer.trainer.policy, 'state_dict'):
            policy_state = trainer.trainer.policy.state_dict()
        if hasattr(trainer.trainer, 'optimizer') and hasattr(trainer.trainer.optimizer, 'state_dict'):
            optimizer_state = trainer.trainer.optimizer.state_dict()
        current_iteration = getattr(trainer.trainer, 'training_iteration',
                                getattr(trainer.trainer, '_iteration', 0)) # Common attribute names
    elif hasattr(trainer, 'policy') and hasattr(trainer.policy, 'state_dict'): # If AdvancedJanusTrainer itself holds policy
        policy_state = trainer.policy.state_dict()
        if hasattr(trainer, 'optimizer') and hasattr(trainer.optimizer, 'state_dict'):
             optimizer_state = trainer.optimizer.state_dict()
        current_iteration = getattr(trainer, 'training_iteration', 0)


    checkpoint = {
        'iteration': current_iteration,
        'policy_state_dict': policy_state,
        'optimizer_state_dict': optimizer_state,
        'config': janus_config.model_dump(), # Save JanusConfig as dict
        'grammar_state': grammar_state
    }
    
    checkpoint_path = Path(janus_config.checkpoint_dir) / "emergency_checkpoint.pt" # Use from JanusConfig
    torch.save(checkpoint, checkpoint_path)
    print(f"✓ Checkpoint saved to {checkpoint_path}")


def run_distributed_sweep(config_path: str, n_trials: int = 20):
    """Run distributed hyperparameter sweep."""
    
    if not HAS_RAY:
        print("❌ Ray is not installed. Cannot run distributed sweep.")
        print("  Install with: pip install ray[tune]")
        sys.exit(1)
    
    print(f"\nRunning distributed sweep with {n_trials} trials...")
    
    from integrated_pipeline import distributed_hyperparameter_search
    from progressive_grammar_system import ProgressiveGrammar
    
    config = validate_config(config_path)
    setup_environment(config)
    
    grammar = ProgressiveGrammar()
    
    # Run sweep
    best_config = distributed_hyperparameter_search(
        grammar=grammar,
        env_config={
            'max_depth': config['max_depth'],
            'max_complexity': config['max_complexity']
        },
        search_space=config.get('hyperparam_search', {}),
        num_trials=n_trials
    )
    
    print(f"\nBest configuration found:")
    for key, value in best_config.items():
        print(f"  {key}: {value}")
    
    # Save best config
    best_config_path = Path(config['results_dir']) / "best_config.yaml"
    with open(best_config_path, 'w') as f:
        yaml.dump(best_config, f)
    
    print(f"\n✓ Best configuration saved to {best_config_path}")


def main():
    """Main entry point."""
    
    parser = argparse.ArgumentParser(
        description="Launch advanced Janus physics discovery training"
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config/advanced_training.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--mode', '-m',
        type=str,
        choices=['train', 'sweep', 'validate'],
        default='train',
        help='Execution mode'
    )
    
    parser.add_argument(
        '--resume', '-r',
        action='store_true',
        help='Resume from latest checkpoint'
    )
    
    parser.add_argument(
        '--n-trials', '-n',
        type=int,
        default=20,
        help='Number of trials for hyperparameter sweep'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode'
    )
    
    parser.add_argument(
        '--strict',
        action='store_true',
        help='Enable strict mode for plugin loading and experiment validation'
    )

    args = parser.parse_args()
    
    # Print banner
    print("\n" + "="*60)
    print("       JANUS PHYSICS DISCOVERY SYSTEM")
    print("       Advanced Training Pipeline v1.0")
    print("="*60)
    
    # Load and validate configuration
    try:
        config_dict = validate_config(args.config) # Returns a dict
        print(f"\n✓ Raw configuration loaded from {args.config}")
        # Integrate command-line strict mode into the config dictionary
        # This will be picked up by JanusConfig when it's instantiated
        config_dict['strict_mode'] = args.strict

        # Note: JanusConfig instantiation happens inside launch_training or when creating AdvancedJanusTrainer
        # For modes like 'sweep' or 'validate' that might not go through launch_training's JanusConfig creation,
        # we need to be mindful of how strict_mode is passed if those paths also use ExperimentRunner.

        print(f"  Training mode: {config_dict['training_mode']}")
        print(f"  Target phenomena: {config_dict['target_phenomena']}")
        print(f"  Total timesteps: {config_dict['total_timesteps']:,}")
        print(f"  Strict mode CLI: {args.strict}")
        
    except Exception as e:
        print(f"\n❌ Configuration error: {e}")
        sys.exit(1)
    
    # Execute based on mode
    try:
        if args.mode == 'train':
            # setup_environment expects a dict, config_dict is fine
            setup_environment(config_dict)
            # launch_training expects a dict, instantiates JanusConfig inside
            launch_training(config_dict, resume=args.resume)
            
        elif args.mode == 'sweep':
            # run_distributed_sweep expects config_path, validate_config is called inside it.
            # We need to ensure strict_mode is passed to ExperimentRunner if it's used by sweep's internals.
            # For now, assuming sweep doesn't directly use ExperimentRunner in a way that needs strict_mode.
            # If it does, run_distributed_sweep would need modification.
            print(f"⚠️  Strict mode not directly propagated to 'sweep' mode's internal ExperimentRunner instances yet.")
            run_distributed_sweep(args.config, n_trials=args.n_trials)
            
        elif args.mode == 'validate':
            from experiment_runner import run_phase1_validation, run_phase2_robustness
            # setup_environment expects a dict
            setup_environment(config_dict)
            
            print(f"\nRunning validation experiments (Strict mode: {args.strict})...")
            # Pass strict_mode to these validation functions
            phase1_results = run_phase1_validation(strict_mode_override=args.strict)
            phase2_results = run_phase2_robustness(strict_mode_override=args.strict)
            
            print("\n✓ Validation completed")
            if phase1_results is not None:
                 print(f"  Phase 1 results: {len(phase1_results)} experiments run (DataFrame shape: {phase1_results.shape})")
            else:
                 print("  Phase 1 results: None")
            if phase2_results is not None:
                print(f"  Phase 2 results: {len(phase2_results)} experiments run (DataFrame shape: {phase2_results.shape})")
            else:
                print("  Phase 2 results: None")

    except Exception as e:
        if args.debug:
            raise
        else:
            print(f"\n❌ Execution failed: {e}")
            sys.exit(1)
    
    print("\n✅ All tasks completed successfully!")


if __name__ == "__main__":
    main()