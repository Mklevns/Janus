# Fast test configuration for Phase 1
from experiment_runner import ExperimentRunner, ExperimentConfig

def run_fast_test():
    runner = ExperimentRunner(use_wandb=False)
    
    # Just test genetic algorithm - it's faster and more stable
    config = ExperimentConfig(
        name="fast_test_genetic",
        environment_type='harmonic_oscillator',
        algorithm='genetic',
        env_params={'k': 1.0, 'm': 1.0},
        algo_params={
            'population_size': 50,  # Smaller population
            'generations': 20,      # Fewer generations
            'max_complexity': 8     # Simpler expressions
        },
        noise_level=0.0,
        n_trajectories=10,
        trajectory_length=50,
        n_runs=1
    )
    
    print("Running fast genetic algorithm test...")
    result = runner.run_single_experiment(config, run_id=0)
    
    print(f"\nCompleted in {result.wall_time_seconds:.1f} seconds")
    print(f"Discovered: {result.discovered_law}")
    print(f"Accuracy: {result.symbolic_accuracy:.2%}")
    
    # Expected: Should discover something like 0.5*v**2 + 0.5*x**2
    
if __name__ == "__main__":
    run_fast_test()
