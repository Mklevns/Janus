#!/usr/bin/env python3
"""
Phase 1 Validation: Known Law Rediscovery
=========================================

This script runs the complete Phase 1 validation suite for Janus.
Simply run: python run_phase1_experiments.py
"""

import torch
print("CUDA available:", torch.cuda.is_available())
print("Default device:", torch.device('cuda'))
print("Device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Device name0:", torch.cuda.get_device_name(0))
print()  # blank line for readability

import os
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging
import traceback

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import all components
from progressive_grammar_system import ProgressiveGrammar, Variable
from symbolic_discovery_env import SymbolicDiscoveryEnv, CurriculumManager
from hypothesis_policy_network import HypothesisNet, PPOTrainer
from physics_discovery_extensions import SymbolicRegressor, ConservationDetector
from experiment_runner import (
    ExperimentRunner, ExperimentConfig, ExperimentResult,
    HarmonicOscillatorEnv, PendulumEnv, KeplerEnv
)
from experiment_visualizer import ExperimentVisualizer, perform_statistical_tests
from utils import calculate_symbolic_accuracy, _count_operations

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('phase1_validation.log', encoding='utf-8'),  # Add encoding
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class Phase1Validator:
    """Orchestrates Phase 1 validation experiments."""

    def __init__(self, output_dir: str = "./phase1_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def create_phase1_configs(self) -> list:
        """Create all Phase 1 experiment configurations."""
        configs = []

        # Parameters for all Phase 1 experiments
        base_params = {
            'noise_level': 0.0,  # No noise for Phase 1
            'n_trajectories': 20,
            'trajectory_length': 100,
            'sampling_rate': 0.1,
            'n_runs': 5,
            'max_experiments': 2000
        }

        # 1. Harmonic Oscillator experiments
        for algo in ['janus_full', 'genetic', 'random']:
            config = ExperimentConfig(
                name=f"P1_HarmonicOscillator_{algo}",
                environment_type='harmonic_oscillator',
                algorithm=algo,
                env_params={'k': 1.0, 'm': 1.0},
                algo_params={
                    'max_depth': 5,
                    'max_complexity': 10,
                    'env_params': {
                        'max_depth': 5,
                        'max_complexity': 10,
                        'reward_config': {
                        'completion_bonus': 0.1,
                        'mse_weight': -1.0,
                        'complexity_penalty': -0.01,
                        'validity_bonus': 0.05,
                        'depth_penalty': -0.001,
                        'timeout_penalty': -1.0
                    }
                    },
                    'policy_params': {
                        'hidden_dim': 256,
                        'encoder_type': 'transformer'
                    }
                },
                noise_level=base_params['noise_level'],
                n_trajectories=int(base_params['n_trajectories']),
                trajectory_length=int(base_params['trajectory_length']),
                sampling_rate=base_params['sampling_rate'],
                n_runs=int(base_params['n_runs']),
                max_experiments=int(base_params['max_experiments'])
            )
            configs.append(config)

        # 2. Pendulum experiments (small angle)
        for algo in ['janus_full', 'genetic', 'random']:
            config = ExperimentConfig(
                name=f"P1_Pendulum_SmallAngle_{algo}",
                environment_type='pendulum',
                algorithm=algo,
                env_params={'g': 9.81, 'l': 1.0, 'm': 1.0, 'small_angle': True},
                algo_params={
                    'max_depth': 6,
                    'max_complexity': 12,
                    'env_params': {
                        'max_depth': 6,
                        'max_complexity': 12
                    }
                },
                noise_level=base_params['noise_level'],
                n_trajectories=int(base_params['n_trajectories']),
                trajectory_length=int(base_params['trajectory_length']),
                sampling_rate=base_params['sampling_rate'],
                n_runs=int(base_params['n_runs']),
                max_experiments=int(base_params['max_experiments'])
            )
            configs.append(config)

        # 3. Kepler orbit experiments
        for algo in ['janus_full', 'genetic']:  # Skip random for complex system
            config = ExperimentConfig(
                name=f"P1_Kepler_Orbit_{algo}",
                environment_type='kepler',
                algorithm=algo,
                env_params={'G': 1.0, 'M': 1.0},
                algo_params={
                    'max_depth': 8,
                    'max_complexity': 20,
                    'env_params': {
                        'max_depth': 8,
                        'max_complexity': 20
                    }
                },
                n_trajectories=10,  # Fewer trajectories for complex system
                trajectory_length=int(base_params['trajectory_length']),
                sampling_rate=base_params['sampling_rate'],
                n_runs=int(base_params['n_runs']),
                max_experiments=int(base_params['max_experiments']),
                noise_level=base_params['noise_level']
            )
            configs.append(config)

        return configs

    def run_single_janus_experiment(self, config: ExperimentConfig, run_id: int) -> ExperimentResult:
        """Run a single Janus experiment with proper setup."""
        logger.info(f"Starting Janus experiment: {config.name} (Run {run_id + 1}/{config.n_runs})")

        # Set random seeds
        np.random.seed(config.seed + run_id)
        import torch
        torch.manual_seed(config.seed + run_id)

        # Create physics environment
        env_class = {
            'harmonic_oscillator': HarmonicOscillatorEnv,
            'pendulum': PendulumEnv,
            'kepler': KeplerEnv
        }[config.environment_type]

        physics_env = env_class(config.env_params, config.noise_level)

        # Generate training data
        logger.info("Generating training trajectories...")
        trajectories = []
        for i in range(config.n_trajectories):
            # Generate appropriate initial conditions
            if config.environment_type == 'harmonic_oscillator':
                init_cond = np.random.randn(2) * np.array([1.0, 2.0])
            elif config.environment_type == 'pendulum':
                max_angle = np.pi/12 if config.env_params.get('small_angle', False) else np.pi/2
                init_cond = np.array([
                    np.random.uniform(-max_angle, max_angle),
                    np.random.uniform(-1, 1)
                ])
            elif config.environment_type == 'kepler':
                # Circular to elliptical orbits
                r0 = np.random.uniform(0.5, 2.0)
                v0 = np.sqrt(config.env_params['G'] * config.env_params['M'] / r0)
                v0 *= np.random.uniform(0.8, 1.2)  # Vary from circular
                init_cond = np.array([r0, 0.0, 0.0, v0/r0])

            t_span = np.arange(0, config.trajectory_length * config.sampling_rate,
                              config.sampling_rate)
            trajectory = physics_env.generate_trajectory(init_cond, t_span)
            trajectories.append(trajectory)

        env_data = np.vstack(trajectories)
        logger.info(f"Generated {env_data.shape[0]} observations with {env_data.shape[1]} features")

        # Variable discovery
        logger.info("Discovering variables...")
        grammar = ProgressiveGrammar()
        # Use only the base state variables for discovery, not derived quantities
        num_state_vars = len(physics_env.state_vars)
        discovered_vars = grammar.discover_variables(env_data[:, :num_state_vars])

        logger.info(f"Discovered {len(discovered_vars)} variables:")
        for var in discovered_vars:
            logger.info(f"  {var.name}: {var.properties}")

        # Create Janus components
        if config.algorithm == 'janus_full':
            # Create RL environment
            discovery_env = SymbolicDiscoveryEnv(
                grammar=grammar,
                target_data=env_data,
                variables=discovered_vars,
                **config.algo_params.get('env_params', {})
            )

            # Create policy network
            policy = HypothesisNet(
                observation_dim=discovery_env.observation_space.shape[0],
                action_dim=discovery_env.action_space.n,
                grammar=grammar,
                **config.algo_params.get('policy_params', {})
            )

            # Create trainer
            trainer = PPOTrainer(policy, discovery_env)

            # Train with curriculum
            curriculum = CurriculumManager(discovery_env)

            logger.info("Starting PPO training...")
            start_time = time.time()

            # Training loop
            best_expression_str = None
            best_complexity = 0
            best_mse = float('inf')
            sample_curve = []

            total_timesteps = 50000
            for timestep in range(0, total_timesteps, 1000):
                # Get current environment from curriculum
                current_env = curriculum.get_current_env()
                trainer.env = current_env

                # Train
                trainer.train(
                    total_timesteps=1000,
                    rollout_length=256,
                    n_epochs=3,
                    batch_size=64,
                    log_interval=10 if timestep % 10000 == 0 else 100
                )

                # Evaluate current best from the environment's cache
                if trainer.episode_mse:
                    # Get the MSE of the most recently completed episode
                    current_mse = trainer.episode_mse[-1]
                    sample_curve.append((timestep, np.mean(list(trainer.episode_mse))))

                    # Check if this is the best result so far
                    if current_mse < best_mse:
                        best_mse = current_mse
                        # Extract the corresponding expression from the environment's cache
                        # Note: This relies on the environment's internal state after an episode.
                        if hasattr(trainer.env, '_evaluation_cache') and trainer.env._evaluation_cache:
                            best_expression_str = trainer.env._evaluation_cache.get('expression')
                            best_complexity = trainer.env._evaluation_cache.get('complexity')

                    # Update curriculum
                    success = current_mse < 0.01  # Threshold for success
                    curriculum.update_curriculum(success)

                # Early stopping
                if best_mse < 1e-6:
                    logger.info(f"Converged early at timestep {timestep}")
                    break

            # Also run conservation detector
            logger.info("Searching for conservation laws...")
            detector = ConservationDetector(grammar)
            conserved_laws = detector.find_conserved_quantities(
                env_data,
                discovered_vars,
                max_complexity=config.algo_params['max_complexity']
            )

            # The primary result is the best law found by the RL agent
            result = ExperimentResult(
                config=config,
                run_id=run_id,
                discovered_law=best_expression_str,
                predictive_mse=best_mse,
                law_complexity=best_complexity,
                sample_efficiency_curve=sample_curve,
                n_experiments_to_convergence=len(sample_curve) * 50 if best_mse < 1e-6 else total_timesteps,
                wall_time_seconds=time.time() - start_time,
                trajectory_data=env_data
            )

            # Calculate symbolic accuracy using the discovered law
            if result.discovered_law:
                result.symbolic_accuracy = calculate_symbolic_accuracy(
                    result.discovered_law,
                    physics_env.ground_truth_laws
                )

        elif config.algorithm == 'genetic':
            # Run genetic baseline
            logger.info("Running genetic programming baseline...")
            start_time = time.time()

            regressor = SymbolicRegressor(grammar)
            X = env_data[:, :-1]  # Features
            y = env_data[:, -1]   # Target (energy)

            best_expr = regressor.fit(
                X, y,
                discovered_vars,
                max_complexity=config.algo_params['max_complexity']
            )

            result = ExperimentResult(
                config=config,
                run_id=run_id,
                discovered_law=str(best_expr.symbolic) if best_expr else None,
                law_complexity=best_expr.complexity if best_expr else 0,
                wall_time_seconds=time.time() - start_time
            )

            # Calculate accuracy
            if best_expr:
                result.symbolic_accuracy = calculate_symbolic_accuracy(
                    str(best_expr.symbolic),
                    physics_env.ground_truth_laws
                )

                # Calculate MSE
                predictions = []
                for i in range(X.shape[0]):
                    subs = {var.symbolic: X[i, var.index] for var in discovered_vars}
                    try:
                        pred = float(best_expr.symbolic.subs(subs))
                        predictions.append(pred)
                    except:
                        predictions.append(0)

                result.predictive_mse = np.mean((np.array(predictions) - y)**2)

        else:  # random
            logger.info("Running random baseline...")
            result = self._run_random_baseline(
                grammar, env_data, discovered_vars, config, run_id
            )

        logger.info(f"Completed {config.name}: Accuracy={result.symbolic_accuracy:.2%}, "
                   f"MSE={result.predictive_mse:.2e}")

        return result

    def _run_random_baseline(self, grammar, env_data, variables, config, run_id):
        """Run random action baseline."""
        start_time = time.time()

        # Create environment
        discovery_env = SymbolicDiscoveryEnv(
            grammar=grammar,
            target_data=env_data,
            variables=variables,
            **config.algo_params.get('env_params', {})
        )

        best_expression = None
        best_mse = float('inf')

        # Run random episodes
        for episode in range(100):  # Fewer episodes for random
            obs, _ = discovery_env.reset()
            done = False

            while not done:
                # Random valid action
                action_mask = discovery_env.get_action_mask()
                valid_actions = np.where(action_mask)[0]

                if len(valid_actions) == 0:
                    break

                action = np.random.choice(valid_actions)
                obs, reward, terminated, truncated, _ = discovery_env.step(action)
                done = terminated or truncated

            # Check if we found a good expression
            if discovery_env._evaluation_cache.get('mse', float('inf')) < best_mse:
                best_mse = discovery_env._evaluation_cache['mse']
                best_expression = discovery_env._evaluation_cache.get('expression')

        return ExperimentResult(
            config=config,
            run_id=run_id,
            discovered_law=best_expression,
            predictive_mse=best_mse,
            n_experiments_to_convergence=100 * 50,
            wall_time_seconds=time.time() - start_time
        )

    def run_all_phase1_experiments(self):
        """Run complete Phase 1 validation suite."""
        logger.info("="*60)
        logger.info("PHASE 1 VALIDATION: KNOWN LAW REDISCOVERY")
        logger.info("="*60)

        # Create configurations
        configs = self.create_phase1_configs()
        logger.info(f"Created {len(configs)} experiment configurations")

        # Run experiments
        runner = ExperimentRunner(base_dir=str(self.output_dir))
        all_results = []

        for config in configs:
            logger.info(f"\nRunning experiment: {config.name}")

            config_results = []
            for run_id in range(config.n_runs):
                try:
                    if config.algorithm == 'janus_full':
                        result = self.run_single_janus_experiment(config, run_id)
                    else:
                        # For 'genetic' or 'random', use the ExperimentRunner's method
                        # which already uses the utils function.
                        result = runner.run_single_experiment(config, run_id)

                    config_results.append(result)
                    all_results.append(result)

                    # Save intermediate results
                    runner._save_result(result)

                except Exception as e:
                    logger.error(f"Error in {config.name} run {run_id}: {str(e)}")
                    logger.error(traceback.format_exc())

        # Convert to DataFrame
        results_df = runner._results_to_dataframe(all_results)
        results_df.to_csv(self.output_dir / f"phase1_results_{self.timestamp}.csv", index=False)

        # Generate analysis
        logger.info("\nGenerating analysis and visualizations...")
        self.analyze_phase1_results(results_df)

        return results_df

    def analyze_phase1_results(self, results_df):
        """Analyze and visualize Phase 1 results."""
        visualizer = ExperimentVisualizer(str(self.output_dir))

        # 1. Summary statistics
        logger.info("\n" + "="*60)
        logger.info("PHASE 1 RESULTS SUMMARY")
        logger.info("="*60)

        # Success rates by algorithm and environment
        success_rates = results_df.groupby(['algorithm', 'environment']).agg({
            'symbolic_accuracy': lambda x: (x > 0.9).mean()
        }).round(2)

        logger.info("\nSuccess Rates (Accuracy > 90%):")
        logger.info(success_rates)

        # Average metrics
        avg_metrics = results_df.groupby('algorithm').agg({
            'symbolic_accuracy': 'mean',
            'predictive_mse': 'mean',
            'n_experiments': 'mean',
            'wall_time': 'mean'
        }).round(3)

        logger.info("\nAverage Performance Metrics:")
        logger.info(avg_metrics)

        # 2. Statistical tests
        stats_results = perform_statistical_tests(results_df)

        logger.info("\nStatistical Significance Tests:")
        for test_name, results in stats_results.items():
            if isinstance(results, dict) and 'p_value' in results:
                logger.info(f"{test_name}: p={results['p_value']:.4f} "
                          f"({'significant' if results.get('significant') else 'not significant'})")

        # 3. Generate plots
        logger.info("\nGenerating visualizations...")

        # Success rate heatmap
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(10, 6))
        pivot = results_df.pivot_table(
            values='symbolic_accuracy',
            index='environment',
            columns='algorithm',
            aggfunc=lambda x: (x > 0.9).mean()
        )

        sns.heatmap(pivot, annot=True, cmap='RdYlGn', vmin=0, vmax=1,
                   cbar_kws={'label': 'Success Rate'})
        plt.title('Phase 1: Law Rediscovery Success Rates')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'phase1_success_rates.png', dpi=300)
        plt.close()

        # Sample efficiency comparison
        plt.figure(figsize=(10, 6))
        for algo in results_df['algorithm'].unique():
            algo_data = results_df[results_df['algorithm'] == algo]

            envs = algo_data['environment'].values
            experiments = algo_data['n_experiments'].values

            plt.scatter(envs, experiments, label=algo, s=100, alpha=0.7)

        plt.yscale('log')
        plt.ylabel('Experiments to Convergence')
        plt.xlabel('Environment')
        plt.title('Phase 1: Sample Efficiency Comparison')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'phase1_sample_efficiency.png', dpi=300)
        plt.close()

        # Generate HTML report
        visualizer.create_summary_report(
            results_df,
            str(self.output_dir / f'phase1_report_{self.timestamp}.html')
        )

        logger.info(f"\nAll results saved to: {self.output_dir}")

        # Print key findings
        logger.info("\n" + "="*60)
        logger.info("KEY FINDINGS")
        logger.info("="*60)

        # Best performing algorithm
        best_algo = avg_metrics['symbolic_accuracy'].idxmax()
        logger.info(f"Best performing algorithm: {best_algo} "
                   f"(avg accuracy: {avg_metrics.loc[best_algo, 'symbolic_accuracy']:.2%})")

        # Efficiency gain
        if 'janus_full' in avg_metrics.index and 'genetic' in avg_metrics.index:
            efficiency_gain = (avg_metrics.loc['genetic', 'n_experiments'] /
                             avg_metrics.loc['janus_full', 'n_experiments'])
            logger.info(f"Janus efficiency gain over genetic: {efficiency_gain:.1f}x faster")

        # Perfect rediscoveries
        perfect_discoveries = results_df[results_df['symbolic_accuracy'] == 1.0]
        logger.info(f"Perfect rediscoveries: {len(perfect_discoveries)}/{len(results_df)} "
                   f"({len(perfect_discoveries)/len(results_df):.0%})")


def main():
    """Main entry point for Phase 1 validation."""
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║          JANUS PHASE 1 VALIDATION: LAW REDISCOVERY        ║
    ╠═══════════════════════════════════════════════════════════╣
    ║  This will test Janus on known physics problems:          ║
    ║  • Harmonic Oscillator (F = -kx)                         ║
    ║  • Simple Pendulum (small angle)                         ║
    ║  • Kepler Orbits (gravitational systems)                 ║
    ║                                                           ║
    ║  Expected runtime: 2-4 hours                              ║
    ╚═══════════════════════════════════════════════════════════╝
    """)

    # Confirm start
    response = input("\nStart Phase 1 validation? (y/n): ")
    if response.lower() != 'y':
        print("Validation cancelled.")
        return

    # Run validation
    validator = Phase1Validator()
    results = validator.run_all_phase1_experiments()

    print("\n" + "="*60)
    print("PHASE 1 VALIDATION COMPLETE!")
    print("="*60)
    print(f"\nResults saved to: {validator.output_dir}")
    print("\nNext steps:")
    print("1. Review the generated report and plots")
    print("2. If success rate > 90%, proceed to Phase 2 (robustness)")
    print("3. If success rate < 90%, debug and tune hyperparameters")


if __name__ == "__main__":
    main()
