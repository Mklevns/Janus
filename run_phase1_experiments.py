#!/usr/bin/env python3
"""
Phase 1 Validation: Known Law Rediscovery (Refactored)
=========================================

This script runs the complete Phase 1 validation suite for Janus.
Simply run: python run_phase1_experiments_refactored.py
"""

import torch
print("CUDA available:", torch.cuda.is_available())
print("Default device:", torch.device('cuda'))
print("Device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Device name0:", torch.cuda.get_device_name(0))
print()

import os
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging
import traceback
from typing import List # Added for type hinting

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from janus.core.grammar import ProgressiveGrammar
from janus.core.expression import Variable
from janus.ai_interpretability.environments import SymbolicDiscoveryEnv, CurriculumManager
from hypothesis_policy_network import HypothesisNet, PPOTrainer
from physics_discovery_extensions import SymbolicRegressor, ConservationDetector
from experiment_runner import (
    ExperimentRunner, ExperimentConfig, ExperimentResult,
    HarmonicOscillatorEnv, PendulumEnv, KeplerEnv
)
from janus.ai_interpretability.utils.visualization import ExperimentVisualizer, perform_statistical_tests
from utils import calculate_symbolic_accuracy, _count_operations
from integrated_pipeline import JanusConfig, SyntheticDataParamsConfig, RewardConfig
from janus.ai_interpretability.utils.math_utils import safe_env_reset # Added import

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('phase1_validation_refactored.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class Phase1Validator:
    def __init__(self, output_dir: str = "./phase1_results_refactored"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def create_phase1_configs(self) -> List[ExperimentConfig]:
        configs: List[ExperimentConfig] = []

        base_n_runs = 2
        base_num_eval_cycles = 200 # Reduced further for faster CI like tests
        base_sampling_rate = 0.1
        base_trajectory_length = 100
        base_n_trajectories = 10 # Reduced for faster tests

        def algo_name_to_seed_offset(algo_name: str) -> int:
            if "janus" in algo_name: return 0
            if "genetic" in algo_name: return 1000
            if "random" in algo_name: return 2000
            return 3000

        time_range_default = [0, int(base_trajectory_length * base_sampling_rate)]

        for algo_name in ['janus_full', 'genetic', 'random']:
            env_specific_params_ho = {'k': 1.0, 'm': 1.0}
            reward_cfg_ho = RewardConfig(completion_bonus=0.1, mse_weight=-0.1, complexity_penalty=-0.001, depth_penalty=-0.0001)

            janus_cfg_ho = JanusConfig(
                target_phenomena='harmonic_oscillator',
                env_specific_params=env_specific_params_ho,
                max_depth=5, max_complexity=10,
                synthetic_data_params=SyntheticDataParamsConfig(noise_level=0.0, n_samples=base_n_trajectories, time_range=time_range_default),
                num_evaluation_cycles=base_num_eval_cycles,
                reward_config=reward_cfg_ho,
                policy_hidden_dim=256, policy_encoder_type='transformer',
                genetic_population_size=50 if algo_name == 'genetic' else JanusConfig().genetic_population_size, # Smaller for test
                genetic_generations=20 if algo_name == 'genetic' else JanusConfig().genetic_generations, # Smaller for test
                total_timesteps=50000, # Default PPO total steps for this script's custom loop
                ppo_rollout_length=256, ppo_n_epochs=3, ppo_batch_size=64, log_interval=100, # Adjusted log interval
            )

            exp_config_ho = ExperimentConfig.from_janus_config(
                name=f"P1_HarmonicOscillator_{algo_name}", experiment_type='physics_discovery_example',
                janus_config=janus_cfg_ho, algorithm_name=algo_name,
                n_runs=base_n_runs, seed=42 + algo_name_to_seed_offset(algo_name)
            )
            exp_config_ho.trajectory_length = base_trajectory_length
            exp_config_ho.sampling_rate = base_sampling_rate
            exp_config_ho.n_trajectories = base_n_trajectories
            configs.append(exp_config_ho)

        for algo_name in ['janus_full', 'genetic', 'random']:
            env_specific_params_pendulum = {'g': 9.81, 'l': 1.0, 'm': 1.0, 'small_angle': True}
            janus_cfg_pendulum = JanusConfig(
                target_phenomena='pendulum', env_specific_params=env_specific_params_pendulum,
                max_depth=6, max_complexity=12,
                synthetic_data_params=SyntheticDataParamsConfig(noise_level=0.0, n_samples=base_n_trajectories, time_range=time_range_default),
                num_evaluation_cycles=base_num_eval_cycles,
                total_timesteps=50000, ppo_rollout_length=256, ppo_n_epochs=3, ppo_batch_size=64, log_interval=100,
            )
            exp_config_pendulum = ExperimentConfig.from_janus_config(
                name=f"P1_Pendulum_SmallAngle_{algo_name}", experiment_type='physics_discovery_example',
                janus_config=janus_cfg_pendulum, algorithm_name=algo_name,
                n_runs=base_n_runs, seed=142 + algo_name_to_seed_offset(algo_name),
            )
            exp_config_pendulum.trajectory_length = base_trajectory_length
            exp_config_pendulum.sampling_rate = base_sampling_rate
            exp_config_pendulum.n_trajectories = base_n_trajectories
            configs.append(exp_config_pendulum)

        for algo_name in ['janus_full', 'genetic']:
            env_specific_params_kepler = {'G': 1.0, 'M': 1.0}
            janus_cfg_kepler = JanusConfig(
                target_phenomena='kepler', env_specific_params=env_specific_params_kepler,
                max_depth=8, max_complexity=20,
                synthetic_data_params=SyntheticDataParamsConfig(noise_level=0.0, n_samples=10, time_range=time_range_default),
                num_evaluation_cycles=base_num_eval_cycles,
                total_timesteps=50000, ppo_rollout_length=256, ppo_n_epochs=3, ppo_batch_size=64, log_interval=100,
            )
            exp_config_kepler = ExperimentConfig.from_janus_config(
                name=f"P1_Kepler_Orbit_{algo_name}", experiment_type='physics_discovery_example',
                janus_config=janus_cfg_kepler, algorithm_name=algo_name,
                n_runs=base_n_runs, seed=242 + algo_name_to_seed_offset(algo_name)
            )
            exp_config_kepler.trajectory_length = base_trajectory_length
            exp_config_kepler.sampling_rate = base_sampling_rate
            exp_config_kepler.n_trajectories = 10
            configs.append(exp_config_kepler)

        return configs

    def run_single_janus_experiment(self, exp_config: ExperimentConfig, run_id: int) -> ExperimentResult:
        logger.info(f"Starting Janus experiment: {exp_config.name} (Run {run_id + 1}/{exp_config.n_runs})")
        janus_cfg = exp_config.janus_config

        np.random.seed(exp_config.seed + run_id)
        torch.manual_seed(exp_config.seed + run_id)

        env_class = {'harmonic_oscillator': HarmonicOscillatorEnv, 'pendulum': PendulumEnv, 'kepler': KeplerEnv}[exp_config.environment_type]
        physics_env = env_class(janus_cfg.env_specific_params, exp_config.noise_level)

        trajectories = []
        for _ in range(exp_config.n_trajectories):
            if exp_config.environment_type == 'harmonic_oscillator':
                init_cond = np.random.randn(2) * np.array([janus_cfg.env_specific_params.get('x_scale', 1.0), janus_cfg.env_specific_params.get('v_scale', 2.0)])
            elif exp_config.environment_type == 'pendulum':
                max_angle = np.pi/12 if janus_cfg.env_specific_params.get('small_angle', False) else np.pi/2
                init_cond = np.array([np.random.uniform(-max_angle, max_angle), np.random.uniform(-1, 1)])
            elif exp_config.environment_type == 'kepler':
                r0 = np.random.uniform(0.5, 2.0)
                v0_num = janus_cfg.env_specific_params.get('G',1.0) * janus_cfg.env_specific_params.get('M',1.0)
                v0 = np.sqrt(v0_num / r0) if r0 > 0 and v0_num >= 0 else 1.0
                v0 *= np.random.uniform(0.8, 1.2)
                init_cond = np.array([r0, 0.0, 0.0, v0/r0 if r0 > 0 else 1.0])
            t_span = np.arange(0, exp_config.trajectory_length * exp_config.sampling_rate, exp_config.sampling_rate)
            trajectory = physics_env.generate_trajectory(init_cond, t_span)
            trajectories.append(trajectory)
        env_data = np.vstack(trajectories)
        logger.info(f"Generated {env_data.shape[0]} obs with {env_data.shape[1]} feats")

        grammar = ProgressiveGrammar()
        num_state_vars = len(physics_env.state_vars)
        discovered_vars = grammar.discover_variables(env_data[:, :num_state_vars])
        logger.info(f"Discovered {len(discovered_vars)} variables.")

        if exp_config.algorithm == 'janus_full':
            sde_params = {'max_depth': janus_cfg.max_depth, 'max_complexity': janus_cfg.max_complexity,
                          'reward_config': janus_cfg.reward_config.model_dump(),
                          'target_variable_index': exp_config.target_variable_index}
            if exp_config.algo_params and 'env_params' in exp_config.algo_params: sde_params.update(exp_config.algo_params['env_params'])
            discovery_env = SymbolicDiscoveryEnv(grammar=grammar, target_data=env_data, variables=discovered_vars, **sde_params)

            policy_creation_params = {'hidden_dim': janus_cfg.policy_hidden_dim, 'encoder_type': janus_cfg.policy_encoder_type, 'grammar': grammar}
            if exp_config.algo_params and 'policy_params' in exp_config.algo_params: policy_creation_params.update(exp_config.algo_params['policy_params'])
            policy = HypothesisNet(observation_dim=discovery_env.observation_space.shape[0], action_dim=discovery_env.action_space.n, **policy_creation_params)

            trainer = PPOTrainer(policy, discovery_env)
            curriculum = CurriculumManager(discovery_env)
            logger.info("Starting PPO training...")
            start_time = time.time()
            best_expression_str, best_complexity, best_mse, sample_curve = None, 0, float('inf'), []

            loop_total_timesteps = janus_cfg.total_timesteps # Using total_timesteps from JanusConfig for this loop
            steps_per_log_cycle = janus_cfg.timesteps_per_eval_cycle

            ppo_training_args = {
                'rollout_length': janus_cfg.ppo_rollout_length, 'n_epochs': janus_cfg.ppo_n_epochs,
                'batch_size': janus_cfg.ppo_batch_size, 'log_interval': janus_cfg.log_interval
            }
            if hasattr(trainer, 'optimizer') and hasattr(trainer.optimizer, 'param_groups'):
                 trainer.optimizer.param_groups[0]['lr'] = janus_cfg.ppo_learning_rate

            for current_step_total in range(0, loop_total_timesteps, steps_per_log_cycle):
                trainer.env = curriculum.get_current_env()
                trainer.train(total_timesteps=steps_per_log_cycle, **ppo_training_args)
                if trainer.episode_mse:
                    current_cycle_mse = trainer.episode_mse[-1] if trainer.episode_mse else float('inf')
                    current_avg_mse = np.mean(list(trainer.episode_mse)) if trainer.episode_mse else float('inf')
                    sample_curve.append((current_step_total + steps_per_log_cycle, current_avg_mse))
                    if current_cycle_mse < best_mse:
                        best_mse = current_cycle_mse
                        if hasattr(trainer.env, '_evaluation_cache') and trainer.env._evaluation_cache:
                            best_expression_str = trainer.env._evaluation_cache.get('expression')
                            best_complexity = trainer.env._evaluation_cache.get('complexity')
                    curriculum.update_curriculum(current_cycle_mse < 0.01)
                if best_mse < 1e-6: break

            if janus_cfg.enable_conservation_detection:
                detector = ConservationDetector(grammar)
                detector.find_conserved_quantities(env_data, discovered_vars, max_complexity=janus_cfg.max_complexity)

            result = ExperimentResult(
                config=exp_config, run_id=run_id, discovered_law=best_expression_str, predictive_mse=best_mse,
                law_complexity=best_complexity, sample_efficiency_curve=sample_curve,
                n_experiments_to_convergence=len(sample_curve) * steps_per_log_cycle if best_mse < 1e-6 else loop_total_timesteps, # Adjusted convergence definition
                wall_time_seconds=time.time() - start_time, trajectory_data=env_data
            )
            if result.discovered_law:
                result.symbolic_accuracy = calculate_symbolic_accuracy(result.discovered_law, physics_env.ground_truth_laws)

        elif exp_config.algorithm == 'genetic':
            logger.info("Running genetic programming baseline...")
            start_time = time.time()
            regressor = SymbolicRegressor(grammar)
            X, y = env_data[:, :-1], env_data[:, -1]
            best_expr = regressor.fit(X, y, discovered_vars, max_complexity=janus_cfg.max_complexity,
                                      population_size=janus_cfg.genetic_population_size, generations=janus_cfg.genetic_generations)
            result = ExperimentResult(
                config=exp_config, run_id=run_id, discovered_law=str(best_expr.symbolic) if best_expr else None,
                law_complexity=best_expr.complexity if best_expr else 0, wall_time_seconds=time.time() - start_time
            )
            if best_expr:
                result.symbolic_accuracy = calculate_symbolic_accuracy(str(best_expr.symbolic), physics_env.ground_truth_laws)
                try:
                    predictions = [float(best_expr.symbolic.subs({v.symbolic: X[i, v.index] for v in discovered_vars})) for i in range(X.shape[0])]
                    result.predictive_mse = np.mean((np.array(predictions) - y)**2)
                except: result.predictive_mse = float('inf')
        else: # random
            logger.info("Running random baseline...")
            start_time = time.time()
            sde_params_random = {'max_depth': janus_cfg.max_depth, 'max_complexity': janus_cfg.max_complexity,
                                 'reward_config': janus_cfg.reward_config.model_dump(),
                                 'target_variable_index': exp_config.target_variable_index}
            discovery_env_random = SymbolicDiscoveryEnv(grammar=grammar, target_data=env_data, variables=discovered_vars, **sde_params_random)
            best_expr_random, best_mse_random = None, float('inf')
            for _ in range(100): # Fewer episodes for random
                obs, _ = safe_env_reset(discovery_env_random) # Use safe_env_reset
                done = False
                while not done:
                    action_mask = discovery_env_random.get_action_mask()
                    valid_actions = np.where(action_mask)[0]
                    if not valid_actions.size: break
                    action = np.random.choice(valid_actions)
                    obs, reward, terminated, truncated, _ = discovery_env_random.step(action)
                    done = terminated or truncated
                if discovery_env_random._evaluation_cache.get('mse', float('inf')) < best_mse_random:
                    best_mse_random = discovery_env_random._evaluation_cache['mse']
                    best_expr_random = discovery_env_random._evaluation_cache.get('expression')
            result = ExperimentResult(
                config=exp_config, run_id=run_id, discovered_law=best_expr_random, predictive_mse=best_mse_random,
                n_experiments_to_convergence=100 * getattr(discovery_env_random, 'max_steps', 50), # Approximation
                wall_time_seconds=time.time() - start_time
            )
        logger.info(f"Completed {exp_config.name}: Acc={result.symbolic_accuracy:.2%}, MSE={result.predictive_mse:.2e}")
        return result

    def run_all_phase1_experiments(self):
        logger.info("="*60 + "\nPHASE 1 VALIDATION: KNOWN LAW REDISCOVERY (Refactored)\n" + "="*60)
        configs = self.create_phase1_configs()
        logger.info(f"Created {len(configs)} experiment configurations")
        runner = ExperimentRunner(base_dir=str(self.output_dir))
        all_results = []
        for exp_conf_item in configs:
            logger.info(f"\nRunning experiment: {exp_conf_item.name}")
            for run_idx in range(exp_conf_item.n_runs):
                try:
                    current_result_item = self.run_single_janus_experiment(exp_conf_item, run_idx) if exp_conf_item.algorithm == 'janus_full' \
                                          else runner.run_single_experiment(exp_conf_item, run_idx)
                    all_results.append(current_result_item)
                    runner._save_result(current_result_item)
                except Exception as e:
                    logger.error(f"Error in {exp_conf_item.name} run {run_idx}: {str(e)}\n{traceback.format_exc()}")
        results_df = runner._results_to_dataframe(all_results)
        results_df.to_csv(self.output_dir / f"phase1_results_{self.timestamp}.csv", index=False)
        logger.info("\nGenerating analysis and visualizations...")
        self.analyze_phase1_results(results_df)
        return results_df

    def analyze_phase1_results(self, results_df):
        visualizer = ExperimentVisualizer(str(self.output_dir))
        logger.info("\n" + "="*60 + "\nPHASE 1 RESULTS SUMMARY (Refactored)\n" + "="*60)

        group_col_env_name = 'environment_type'
        if group_col_env_name not in results_df.columns:
            logger.warning(f"Column '{group_col_env_name}' not found. Using 'experiment_name' derived.")
            if 'experiment_name' in results_df.columns:
                results_df[group_col_env_name] = results_df['experiment_name'].apply(lambda x: x.split('_')[1] if len(x.split('_')) > 2 else 'unknown')
            else: group_col_env_name = None

        if group_col_env_name:
            success_rates = results_df.groupby(['algorithm', group_col_env_name]).agg(
                success_rate=('symbolic_accuracy', lambda x: (x > 0.9).mean())
            ).round(2)
            logger.info("\nSuccess Rates (Accuracy > 90%):\n" + str(success_rates))
        else: logger.warning("Skipping success rate calculation.")

        avg_metrics_agg_map = {
            'symbolic_accuracy': 'mean', 'predictive_mse': 'mean',
            'n_experiments_to_convergence': 'mean', 'wall_time_seconds': 'mean'
        }
        valid_agg_metrics_map = {k: v for k,v in avg_metrics_agg_map.items() if k in results_df.columns}

        if valid_agg_metrics_map:
            avg_metrics = results_df.groupby('algorithm').agg(valid_agg_metrics_map).round(3)
            logger.info("\nAverage Performance Metrics:\n" + str(avg_metrics))
        else:
            logger.warning("No valid columns for average metrics. DF cols: %s", results_df.columns)
            avg_metrics = pd.DataFrame()

        stats_results = perform_statistical_tests(results_df)
        logger.info("\nStatistical Significance Tests:")
        for test_name, res in stats_results.items():
            if isinstance(res, dict) and 'p_value' in res: logger.info(f"{test_name}: p={res['p_value']:.4f} ({'sig' if res.get('significant') else 'not sig'})")

        logger.info("\nGenerating visualizations...")
        import matplotlib.pyplot as plt
        import seaborn as sns
        if group_col_env_name and 'symbolic_accuracy' in results_df.columns:
            pivot_table_values = results_df.pivot_table(values='symbolic_accuracy', index=group_col_env_name, columns='algorithm', aggfunc=lambda x: (x > 0.9).mean())
            plt.figure(figsize=(10,6))
            sns.heatmap(pivot_table_values, annot=True, cmap='RdYlGn', vmin=0, vmax=1, cbar_kws={'label': 'Success Rate'})
            plt.title('Phase 1: Law Rediscovery Success Rates'); plt.tight_layout()
            plt.savefig(self.output_dir / 'phase1_success_rates.png', dpi=300); plt.close()
        else: logger.warning("Skipping success rate heatmap.")

        plt.figure(figsize=(10, 6))
        if group_col_env_name and 'n_experiments_to_convergence' in results_df.columns:
            for algo_name_plot in results_df['algorithm'].unique():
                algo_data_plot = results_df[results_df['algorithm'] == algo_name_plot]
                plt.scatter(algo_data_plot[group_col_env_name].values, algo_data_plot['n_experiments_to_convergence'].values, label=algo_name_plot, s=100, alpha=0.7)
            plt.yscale('log'); plt.ylabel('Experiments to Convergence'); plt.xlabel('Environment Type')
            plt.title('Phase 1: Sample Efficiency Comparison'); plt.legend(); plt.xticks(rotation=45); plt.tight_layout()
            plt.savefig(self.output_dir / 'phase1_sample_efficiency.png', dpi=300); plt.close()
        else: logger.warning("Skipping sample efficiency plot.")

        visualizer.create_summary_report(results_df, str(self.output_dir / f'phase1_report_{self.timestamp}.html'))
        logger.info(f"\nAll results saved to: {self.output_dir}")
        logger.info("\n" + "="*60 + "\nKEY FINDINGS (Refactored)\n" + "="*60)
        if not avg_metrics.empty and 'symbolic_accuracy' in avg_metrics.columns:
            best_algo_name = avg_metrics['symbolic_accuracy'].idxmax()
            logger.info(f"Best performing algorithm: {best_algo_name} (avg accuracy: {avg_metrics.loc[best_algo_name, 'symbolic_accuracy']:.2%})")
        if not avg_metrics.empty and 'janus_full' in avg_metrics.index and 'genetic' in avg_metrics.index and \
           'n_experiments_to_convergence' in avg_metrics.columns and avg_metrics.loc['janus_full', 'n_experiments_to_convergence'] > 0:
            efficiency_gain_val = avg_metrics.loc['genetic', 'n_experiments_to_convergence'] / avg_metrics.loc['janus_full', 'n_experiments_to_convergence']
            logger.info(f"Janus efficiency gain over genetic: {efficiency_gain_val:.1f}x faster")
        if 'symbolic_accuracy' in results_df.columns:
            perfect_discoveries = results_df[results_df['symbolic_accuracy'] == 1.0]
            logger.info(f"Perfect rediscoveries: {len(perfect_discoveries)}/{len(results_df)} ({len(perfect_discoveries)/len(results_df) if len(results_df) > 0 else 0:.0%})")

def main():
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
    # response = input("\nStart Phase 1 validation? (y/n): ") # Auto-run for now
    # if response.lower() != 'y':
    #     print("Validation cancelled.")
    #     return
    print("\nStarting Phase 1 validation automatically...")
    validator = Phase1Validator()
    results = validator.run_all_phase1_experiments()
    print("\n" + "="*60 + "\nPHASE 1 VALIDATION COMPLETE!\n" + "="*60)
    print(f"\nResults saved to: {validator.output_dir}")
    print("\nNext steps:\n1. Review report & plots.\n2. If >90% success, proceed to Phase 2.\n3. Else, debug/tune.")

if __name__ == "__main__":
    main()
