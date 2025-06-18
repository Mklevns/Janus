# Fast test configuration for Phase 1 (refactored)
from experiment_runner import ExperimentRunner, ExperimentConfig
from integrated_pipeline import JanusConfig, SyntheticDataParamsConfig # Import JanusConfig

def run_fast_test():
    runner = ExperimentRunner(use_wandb=False) # Assumes ExperimentRunner is initialized as before

    # Create JanusConfig for this specific test
    janus_cfg_fast_test = JanusConfig(
        target_phenomena='harmonic_oscillator',
        env_specific_params={'k': 1.0, 'm': 1.0}, # Physics environment parameters
        synthetic_data_params=SyntheticDataParamsConfig(
            noise_level=0.0,
            n_samples=10, # Corresponds to n_trajectories in ExperimentConfig context
            time_range=[0, 5] # Example: duration 5. If sampling_rate=0.1, this means 50 points.
        ),
        # Genetic algorithm parameters from JanusConfig
        genetic_population_size=50,
        genetic_generations=20,
        max_complexity=8, # For simpler expressions by Genetic Algo / SDE

        # Training loop / evaluation parameters from JanusConfig
        # ExperimentConfig.max_experiments maps to num_evaluation_cycles
        num_evaluation_cycles=1, # For a fast test, this might mean just one main evaluation or a very short run.
                                 # If 'genetic' algo runs for 'genetic_generations', this might not be directly used by it.
                                 # This mapping needs to be clear. For genetic, max_experiments might be generations.
                                 # Let's assume for genetic, it's controlled by genetic_generations.
                                 # So, num_evaluation_cycles might be more for iterative algos like PPO.
                                 # For this test, we rely on genetic_generations for control.

        # Ensure other JanusConfig fields are at their defaults or set as needed for this test
        # For example, if the 'genetic' algorithm in ExperimentRunner's factory uses other
        # JanusConfig fields, they should be set here.
    )

    # Create ExperimentConfig using from_janus_config
    # The 'algorithm' field in ExperimentConfig is set via algorithm_name here.
    exp_config_fast_test = ExperimentConfig.from_janus_config(
        name="fast_test_genetic_refactored",
        experiment_type='physics_discovery_example', # This must match a registered experiment plugin
        janus_config=janus_cfg_fast_test,
        algorithm_name='genetic', # Specify the algorithm key for the registry
        n_runs=1,
        seed=42, # Example seed for this specific run
        # algo_params_override can be used for any params not in JanusConfig or for specific overrides
        # For this genetic test, specific genetic params are already in JanusConfig.
        algo_params_override={}
    )

    # Override n_trajectories and trajectory_length on ExperimentConfig if they are not
    # perfectly derived by from_janus_config or if a different interpretation is needed.
    # Here, n_samples from synthetic_data_params is used for n_trajectories by from_janus_config.
    # Trajectory_length derivation in from_janus_config from time_range might need adjustment
    # based on how it's used by the environment's data generation.
    # For this test, let's explicitly set them to match the original test's intent.
    exp_config_fast_test.n_trajectories = 10
    exp_config_fast_test.trajectory_length = 50 # 50 data points per trajectory.
                                             # If sampling_rate=0.1, this means duration of 5 per trajectory.
                                             # This matches time_range=[0,5] used in SyntheticDataParamsConfig.

    print("Running fast genetic algorithm test (refactored)...")
    result = runner.run_single_experiment(exp_config_fast_test, run_id=0)

    print(f"\nCompleted in {result.wall_time_seconds:.1f} seconds")
    print(f"Discovered: {result.discovered_law}")
    print(f"Accuracy: {result.symbolic_accuracy:.2%}")

    # Expected: Should discover something like 0.5*v**2 + 0.5*x**2

if __name__ == "__main__":
    run_fast_test()
