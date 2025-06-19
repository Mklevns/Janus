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
from typing import Dict, Any, Optional # Added Optional for save_checkpoint
from janus.ai_interpretability.utils.math_utils import validate_inputs, safe_import
from janus.config.models import JanusConfig # For type hinting
from janus.config.loader import ConfigLoader # Added ConfigLoader

# Optional imports with fallbacks using safe_import
ray = safe_import("ray", "ray")
HAS_RAY = ray is not None
# The warning for Ray is printed by safe_import if not found,
# but we can add a specific message about disabled features.
if not HAS_RAY:
    print("⚠️  Ray features for distributed training will be unavailable.")

GPUtil = safe_import("GPUtil", "GPUtil")
HAS_GPUTIL = GPUtil is not None
# Warning for GPUtil is printed by safe_import if not found.
if not HAS_GPUTIL:
    print("⚠️  GPUtil features for detailed GPU monitoring will be unavailable.")

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
        if HAS_GPUTIL and GPUtil: # Check GPUtil module is not None
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


@validate_inputs
def validate_config(config_path: str) -> JanusConfig:
    """Validate and load configuration."""
    
    loader = ConfigLoader(primary_config_path=config_path)
    janus_config_obj = loader.load_resolved_config() # Env vars already applied by loader

    # Pydantic model validation handles required fields and types.
    # Custom validation for things like mode is in JanusConfig itself.

    # Adjust based on system capabilities
    system_reqs = check_system_requirements()
    
    if hasattr(janus_config_obj, 'algorithm') and janus_config_obj.algorithm is not None:
        if janus_config_obj.algorithm.num_gpus > system_reqs['gpu_count']:
            print(f"⚠️  Warning: Config requests {janus_config_obj.algorithm.num_gpus} GPUs but only "
                  f"{system_reqs['gpu_count']} available. Adjusting...")
            janus_config_obj.algorithm.num_gpus = system_reqs['gpu_count']

        # Ensure num_workers is at least 1 if it's set, and not more than available CPUs-2
        # (or simply system_reqs['cpu_count'] if only 1 CPU is available in total)
        current_num_workers = janus_config_obj.algorithm.num_workers
        max_permissible_workers = system_reqs['cpu_count'] - 2 if system_reqs['cpu_count'] > 2 else 1
        max_permissible_workers = max(1, max_permissible_workers) # Must be at least 1

        if current_num_workers > max_permissible_workers:
            print(f"⚠️  Warning: Config requests {current_num_workers} workers but only "
                  f"{max_permissible_workers} permissible (based on {system_reqs['cpu_count']} CPUs). Adjusting...")
            janus_config_obj.algorithm.num_workers = max_permissible_workers
        elif current_num_workers <= 0: # Ensure at least 1 worker if specified (or default to 1 in model)
            print(f"⚠️  Warning: Config requests {current_num_workers} workers. Setting to 1.")
            janus_config_obj.algorithm.num_workers = 1


    return janus_config_obj


@validate_inputs
def setup_environment(config: JanusConfig): # Changed type hint
    """Setup directories and environment."""
    
    print("\nSetting up environment...")
    
    # Create directories (paths now come from config object's sub-models)
    # Assuming ExperimentConfig holds these paths
    if config.experiment:
        for dir_path_str in [config.experiment.checkpoint_dir, config.experiment.results_dir, config.experiment.data_dir]:
            if dir_path_str: # Check if path is not None
                dir_path = Path(dir_path_str)
                dir_path.mkdir(parents=True, exist_ok=True)
                print(f"  Ensured directory exists: {dir_path}")

    # Set environment variables (num_gpus from algorithm config)
    num_gpus_to_set = 0
    if hasattr(config, 'algorithm') and config.algorithm is not None:
        num_gpus_to_set = config.algorithm.num_gpus
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(
        str(i) for i in range(num_gpus_to_set)
    )
    
    # Initialize Ray if needed
    # training_mode from training config, num_workers from algorithm config
    if (hasattr(config, 'training') and config.training is not None and
        hasattr(config, 'algorithm') and config.algorithm is not None and
        config.training.training_mode in ['distributed', 'advanced'] and
        config.algorithm.num_workers > 1):
        if HAS_RAY and ray and not ray.is_initialized(): # Check ray module
            ray_cfg_model = config.algorithm.ray_config # This is RayConfig Pydantic model
            
            # Extract only valid ray.init() parameters from RayConfig model
            valid_ray_params = {
                'num_cpus': ray_cfg_model.num_cpus,
                'num_gpus': ray_cfg_model.num_gpus if ray_cfg_model.num_gpus is not None else config.algorithm.num_gpus,
                'object_store_memory': ray_cfg_model.object_store_memory,
                'include_dashboard': ray_cfg_model.include_dashboard,
                'dashboard_host': ray_cfg_model.dashboard_host,
                '_temp_dir': ray_cfg_model.temp_dir, # Uses alias _temp_dir for loading, actual attribute is temp_dir
                'local_mode': ray_cfg_model.local_mode
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


@validate_inputs
def launch_training(config: JanusConfig, resume: bool = False): # config is now JanusConfig object
    """Launch the training process."""
    
    from integrated_pipeline import AdvancedJanusTrainer
    
    print("\n" + "="*60)
    print("LAUNCHING JANUS ADVANCED TRAINING")
    print("="*60)
    
    # config is already a JanusConfig object, no need to re-instantiate
    janus_config = config
    
    # Create trainer
    trainer = AdvancedJanusTrainer(janus_config)
    
    # Check for resume
    if resume:
        # checkpoint_dir from experiment config
        checkpoint_dir = janus_config.experiment.checkpoint_dir if janus_config.experiment else "./checkpoints"
        checkpoint_path = Path(checkpoint_dir) / "latest_checkpoint.pt"
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
        if HAS_RAY and ray and ray.is_initialized(): # Check ray module
            ray.shutdown()
        print("\nCleanup completed")


@validate_inputs
def save_checkpoint(trainer: Any, janus_config: JanusConfig): # Type hint for trainer as Any
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
        'config': janus_config.model_dump(mode='json'), # Save JanusConfig as dict
        'grammar_state': grammar_state
    }
    
    # checkpoint_dir from experiment config
    checkpoint_dir = janus_config.experiment.checkpoint_dir if janus_config.experiment else "./checkpoints"
    checkpoint_path = Path(checkpoint_dir) / "emergency_checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)
    print(f"✓ Checkpoint saved to {checkpoint_path}")


def run_distributed_sweep(config_path: str, n_trials: int = 20):
    """Run distributed hyperparameter sweep."""
    
    if not HAS_RAY or not ray: # Check ray module
        print("❌ Ray is not installed or failed to import. Cannot run distributed sweep.")
        print("  Install with: pip install ray[tune]")
        sys.exit(1)
    
    print(f"\nRunning distributed sweep with {n_trials} trials...")
    
    from integrated_pipeline import distributed_hyperparameter_search
    from janus.core.grammar import ProgressiveGrammar
    
    janus_config_obj = validate_config(config_path) # Now returns JanusConfig object
    setup_environment(janus_config_obj) # Expects JanusConfig object
    
    grammar = ProgressiveGrammar()
    
    env_config_params = {}
    if hasattr(janus_config_obj, 'environment') and janus_config_obj.environment is not None:
        env_config_params['max_depth'] = janus_config_obj.environment.max_depth
        env_config_params['max_complexity'] = janus_config_obj.environment.max_complexity
    else: # Fallback or raise error if essential
        print("⚠️ Warning: Environment config not found in JanusConfig for sweep's env_config. Using defaults.")
        env_config_params['max_depth'] = 10 # Default placeholder
        env_config_params['max_complexity'] = 30 # Default placeholder

    search_space_params = {}
    if hasattr(janus_config_obj, 'algorithm') and janus_config_obj.algorithm is not None:
        # Assuming hyperparam_search is a dict within algorithm.hyperparameters
        # Or if AlgorithmConfig had `hyperparam_search: Optional[Dict[str, Any]] = None`
        # search_space_params = janus_config_obj.algorithm.hyperparam_search or {}
        search_space_params = janus_config_obj.algorithm.hyperparameters.get('hyperparam_search', {})

    # Run sweep
    best_config_dict = distributed_hyperparameter_search( # Assuming this function returns a dict
        grammar=grammar,
        env_config=env_config_params,
        search_space=search_space_params,
        num_trials=n_trials
    )
    
    print(f"\nBest configuration found by sweep (dictionary):")
    for key, value in best_config_dict.items():
        print(f"  {key}: {value}")
    
    # Save best config
    results_dir_path = "./results" # Default if not found
    if hasattr(janus_config_obj, 'experiment') and janus_config_obj.experiment is not None:
        results_dir_path = janus_config_obj.experiment.results_dir

    best_config_path = Path(results_dir_path) / "best_config_from_sweep.yaml"
    with open(best_config_path, 'w') as f:
        yaml.dump(best_config_dict, f)
    
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
        # validate_config now returns a JanusConfig object
        loaded_janus_config = validate_config(args.config)
        print(f"\n✓ Configuration loaded and validated from {args.config}")

        # Apply CLI strict_mode override to the loaded JanusConfig object
        if hasattr(loaded_janus_config, 'experiment') and loaded_janus_config.experiment is not None:
            loaded_janus_config.experiment.strict_mode = args.strict
            print(f"  Applied CLI strict_mode: {args.strict} to loaded_janus_config.experiment.strict_mode")
        else:
            # This case might occur if JanusConfig structure changes or experiment is None
            print(f"⚠️ Warning: Could not set strict_mode from CLI args as 'experiment' attribute is missing or None in JanusConfig.")


        # Print details from the JanusConfig object
        print(f"  Training mode: {loaded_janus_config.training.training_mode if hasattr(loaded_janus_config, 'training') else 'N/A'}")
        print(f"  Target phenomena: {loaded_janus_config.experiment.target_phenomena if hasattr(loaded_janus_config, 'experiment') else 'N/A'}")
        print(f"  Total timesteps: {loaded_janus_config.training.total_timesteps if hasattr(loaded_janus_config, 'training') else 'N/A'}")
        print(f"  Strict mode from config: {loaded_janus_config.experiment.strict_mode if hasattr(loaded_janus_config, 'experiment') else 'N/A'}")
        
    except Exception as e:
        print(f"\n❌ Configuration error: {e}")
        sys.exit(1)
    
    # Execute based on mode
    try:
        if args.mode == 'train':
            # setup_environment and launch_training now expect JanusConfig object
            setup_environment(loaded_janus_config)
            launch_training(loaded_janus_config, resume=args.resume)
            
        elif args.mode == 'sweep':
            # run_distributed_sweep expects config_path, validate_config is called inside it.
            # It will need similar updates to use ConfigLoader and JanusConfig object internally.
            # For now, this part is NOT updated as per subtask instructions.
            print(f"⚠️  Strict mode from CLI (args.strict={args.strict}) not directly propagated to 'sweep' mode's internal ExperimentRunner instances yet.")
            print(f"   Sweep mode will use strict_mode from its loaded YAML or its defaults.")
            run_distributed_sweep(args.config, n_trials=args.n_trials)
            
        elif args.mode == 'validate':
            from experiment_runner import run_phase1_validation, run_phase2_robustness
            # setup_environment expects JanusConfig object
            setup_environment(loaded_janus_config)
            
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