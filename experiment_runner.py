"""
ExperimentRunner: Automated Validation Framework for Janus
==========================================================

Comprehensive framework for running, tracking, and analyzing physics discovery experiments.
Supports all phases of the validation protocol with proper statistical rigor.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field, asdict
import json
import pickle
from pathlib import Path
import time
from datetime import datetime
import sympy as sp
from scipy.integrate import odeint
import torch
import wandb
from tqdm import tqdm
import hashlib

from utils import calculate_symbolic_accuracy

@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    name: str
    environment_type: str  # 'harmonic_oscillator', 'pendulum', 'kepler', etc.
    algorithm: str  # 'janus_full', 'janus_no_rl', 'genetic', 'pysr', 'random'

    # Environment parameters
    env_params: Dict[str, Any] = field(default_factory=dict)
    noise_level: float = 0.0
    hidden_confounders: bool = False

    # Algorithm parameters
    algo_params: Dict[str, Any] = field(default_factory=dict)
    max_experiments: int = 1000
    max_time_seconds: int = 3600

    # Data collection
    n_trajectories: int = 10
    trajectory_length: int = 100
    sampling_rate: float = 0.1

    # Reproducibility
    seed: int = 42
    n_runs: int = 5  # Number of independent runs for statistics

    # Target variable specification for discovery
    target_variable_index: Optional[int] = None

    def get_hash(self) -> str:
        """Get unique hash for this configuration."""
        config_str = json.dumps(asdict(self), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]


@dataclass
class ExperimentResult:
    """Results from a single experiment run."""
    config: ExperimentConfig
    run_id: int

    # Discovery metrics
    discovered_law: Optional[str] = None
    symbolic_accuracy: float = 0.0  # Similarity to ground truth
    predictive_mse: float = float('inf')
    law_complexity: int = 0

    # Efficiency metrics
    n_experiments_to_convergence: int = 0
    wall_time_seconds: float = 0.0
    sample_efficiency_curve: List[Tuple[int, float]] = field(default_factory=list)

    # Robustness metrics
    noise_resilience_score: float = 0.0
    generalization_score: float = 0.0

    # Component contributions (for ablations)
    component_metrics: Dict[str, float] = field(default_factory=dict)

    # Raw data
    trajectory_data: Optional[np.ndarray] = None
    experiment_log: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary for saving."""
        result = asdict(self)
        # Convert numpy arrays to lists for JSON serialization
        if self.trajectory_data is not None:
            result['trajectory_data'] = self.trajectory_data.tolist()
        return result


class PhysicsEnvironment:
    """Base class for physics simulation environments."""

    def __init__(self, params: Dict[str, Any], noise_level: float = 0.0):
        self.params = params
        self.noise_level = noise_level
        self.state_vars = []
        self.ground_truth_laws = {}

    def generate_trajectory(self,
                          initial_conditions: np.ndarray,
                          t_span: np.ndarray) -> np.ndarray:
        """Generate a trajectory from initial conditions."""
        raise NotImplementedError

    def add_observation_noise(self, trajectory: np.ndarray) -> np.ndarray:
        """Add realistic observation noise."""
        if self.noise_level > 0:
            noise = np.random.randn(*trajectory.shape) * self.noise_level
            # Scale noise by signal magnitude
            signal_std = np.std(trajectory, axis=0)
            scaled_noise = noise * signal_std
            return trajectory + scaled_noise
        return trajectory

    def get_ground_truth_laws(self) -> Dict[str, sp.Expr]:
        """Return ground truth conservation laws and equations of motion."""
        return self.ground_truth_laws


class HarmonicOscillatorEnv(PhysicsEnvironment):
    """Simple harmonic oscillator environment."""

    def __init__(self, params: Dict[str, Any], noise_level: float = 0.0):
        super().__init__(params, noise_level)
        self.k = params.get('k', 1.0)  # Spring constant
        self.m = params.get('m', 1.0)  # Mass
        self.state_vars = ['x', 'v']

        # Define ground truth laws
        x, v = sp.symbols('x v')
        self.ground_truth_laws = {
            'energy_conservation': 0.5 * self.m * v**2 + 0.5 * self.k * x**2,
            'equation_of_motion': -self.k * x / self.m
        }

    def dynamics(self, state: np.ndarray, t: float) -> np.ndarray:
        """ODE for harmonic oscillator."""
        x, v = state
        dxdt = v
        dvdt = -self.k * x / self.m
        return np.array([dxdt, dvdt])

    def generate_trajectory(self,
                          initial_conditions: np.ndarray,
                          t_span: np.ndarray) -> np.ndarray:
        """Generate trajectory using ODE solver."""
        trajectory = odeint(self.dynamics, initial_conditions, t_span)

        # Add derived quantities
        x, v = trajectory[:, 0], trajectory[:, 1]
        energy = 0.5 * self.m * v**2 + 0.5 * self.k * x**2

        # Combine into full observation matrix
        full_trajectory = np.column_stack([x, v, energy])

        return self.add_observation_noise(full_trajectory)


class PendulumEnv(PhysicsEnvironment):
    """Pendulum environment with configurable angle range."""

    def __init__(self, params: Dict[str, Any], noise_level: float = 0.0):
        super().__init__(params, noise_level)
        self.g = params.get('g', 9.81)
        self.l = params.get('l', 1.0)
        self.m = params.get('m', 1.0)
        self.small_angle = params.get('small_angle', False)
        self.state_vars = ['theta', 'omega']

        # Ground truth laws
        theta, omega = sp.symbols('theta omega')
        if self.small_angle:
            self.ground_truth_laws = {
                'energy_conservation': 0.5 * self.m * self.l**2 * omega**2 +
                                     0.5 * self.m * self.g * self.l * theta**2,
                'equation_of_motion': -self.g * theta / self.l
            }
        else:
            self.ground_truth_laws = {
                'energy_conservation': 0.5 * self.m * self.l**2 * omega**2 +
                                     self.m * self.g * self.l * (1 - sp.cos(theta)),
                'equation_of_motion': -self.g * sp.sin(theta) / self.l
            }

    def dynamics(self, state: np.ndarray, t: float) -> np.ndarray:
        """ODE for pendulum."""
        theta, omega = state
        if self.small_angle:
            domega_dt = -self.g * theta / self.l
        else:
            domega_dt = -self.g * np.sin(theta) / self.l
        return np.array([omega, domega_dt])

    def generate_trajectory(self,
                          initial_conditions: np.ndarray,
                          t_span: np.ndarray) -> np.ndarray:
        """Generate pendulum trajectory."""
        trajectory = odeint(self.dynamics, initial_conditions, t_span)

        theta, omega = trajectory[:, 0], trajectory[:, 1]
        if self.small_angle:
            energy = 0.5 * self.m * self.l**2 * omega**2 + \
                    0.5 * self.m * self.g * self.l * theta**2
        else:
            energy = 0.5 * self.m * self.l**2 * omega**2 + \
                    self.m * self.g * self.l * (1 - np.cos(theta))

        full_trajectory = np.column_stack([theta, omega, energy])
        return self.add_observation_noise(full_trajectory)


class KeplerEnv(PhysicsEnvironment):
    """Two-body gravitational system."""

    def __init__(self, params: Dict[str, Any], noise_level: float = 0.0):
        super().__init__(params, noise_level)
        self.G = params.get('G', 1.0)
        self.M = params.get('M', 1.0)  # Central mass
        self.state_vars = ['r', 'theta', 'vr', 'vtheta']

        # Ground truth (in polar coordinates)
        r, theta, vr, vtheta = sp.symbols('r theta vr vtheta')
        self.ground_truth_laws = {
            'energy_conservation': 0.5 * (vr**2 + r**2 * vtheta**2) - self.G * self.M / r,
            'angular_momentum': r**2 * vtheta,
            'equation_of_motion_r': r * vtheta**2 - self.G * self.M / r**2
        }

    def dynamics(self, state: np.ndarray, t: float) -> np.ndarray:
        """ODE for Kepler problem in polar coordinates."""
        r, theta, vr, vtheta = state

        dr_dt = vr
        dtheta_dt = vtheta
        dvr_dt = r * vtheta**2 - self.G * self.M / r**2
        dvtheta_dt = -2 * vr * vtheta / r

        return np.array([dr_dt, dtheta_dt, dvr_dt, dvtheta_dt])

    def generate_trajectory(self,
                          initial_conditions: np.ndarray,
                          t_span: np.ndarray) -> np.ndarray:
        """Generate orbital trajectory."""
        trajectory = odeint(self.dynamics, initial_conditions, t_span)

        r, theta, vr, vtheta = trajectory.T
        energy = 0.5 * (vr**2 + r**2 * vtheta**2) - self.G * self.M / r
        angular_momentum = r**2 * vtheta

        full_trajectory = np.column_stack([r, theta, vr, vtheta, energy, angular_momentum])
        return self.add_observation_noise(full_trajectory)


class ExperimentRunner:
    """Main class for running and managing experiments."""

    def __init__(self,
                 base_dir: str = "./experiments",
                 use_wandb: bool = True):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.use_wandb = use_wandb

        # Environment registry
        self.env_registry = {
            'harmonic_oscillator': HarmonicOscillatorEnv,
            'pendulum': PendulumEnv,
            'kepler': KeplerEnv
        }

        # Algorithm registry (to be populated)
        self.algo_registry = {}
        self._register_algorithms()

    def _register_algorithms(self):
        """Register available algorithms."""
        # Import algorithm classes
        from symbolic_discovery_env import SymbolicDiscoveryEnv, CurriculumManager
        from hypothesis_policy_network import HypothesisNet, PPOTrainer
        from progressive_grammar_system import ProgressiveGrammar
        from physics_discovery_extensions import SymbolicRegressor

        # Janus full system
        def create_janus_full(env_data, variables, config):
            grammar = ProgressiveGrammar()
            env_creation_params = config.algo_params.get('env_params', {})
            # Add target_variable_index to env parameters
            env_creation_params['target_variable_index'] = config.target_variable_index

            discovery_env = SymbolicDiscoveryEnv(
                grammar=grammar,
                target_data=env_data,
                variables=variables,
                **env_creation_params
            )

            policy = HypothesisNet(
                observation_dim=discovery_env.observation_space.shape[0],
                action_dim=discovery_env.action_space.n,
                **config.algo_params.get('policy_params', {})
            )

            trainer = PPOTrainer(policy, discovery_env)
            return trainer

        # Genetic programming baseline
        def create_genetic(env_data, variables, config):
            grammar = ProgressiveGrammar()
            regressor = SymbolicRegressor(grammar)
            return regressor

        self.algo_registry['janus_full'] = create_janus_full
        self.algo_registry['genetic'] = create_genetic
        # Add more algorithms as implemented
        self.algo_registry['random']     = lambda env_data, variables, config: None

    def setup_experiment(self, config: ExperimentConfig) -> Dict[str, Any]:
        """Set up an experiment with given configuration."""
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

        # Create environment
        env_class = self.env_registry[config.environment_type]
        physics_env = env_class(config.env_params, config.noise_level)

        # Generate training data
        trajectories = []
        for _ in range(config.n_trajectories):
            # Random initial conditions
            if config.environment_type == 'harmonic_oscillator':
                init_cond = np.random.randn(2) * np.array([1.0, 2.0])
            elif config.environment_type == 'pendulum':
                max_angle = np.pi/12 if config.env_params.get('small_angle', False) else np.pi
                init_cond = np.random.rand(2) * np.array([max_angle, 1.0])
            elif config.environment_type == 'kepler':
                init_cond = np.array([1.0, 0.0, 0.0, 1.0])  # Circular orbit

            t_span = np.arange(0, config.trajectory_length * config.sampling_rate,
                              config.sampling_rate)
            trajectory = physics_env.generate_trajectory(init_cond, t_span)
            trajectories.append(trajectory)

        env_data = np.vstack(trajectories)

        # Create variables (would be discovered by Janus)
        from progressive_grammar_system import Variable
        variables = [
            Variable(name, idx, {})
            for idx, name in enumerate(physics_env.state_vars)
        ]

        # Create algorithm
        algorithm = self.algo_registry[config.algorithm](env_data, variables, config)

        return {
            'physics_env': physics_env,
            'env_data': env_data,
            'variables': variables,
            'algorithm': algorithm,
            'ground_truth': physics_env.get_ground_truth_laws()
        }

    def run_single_experiment(self,
                            config: ExperimentConfig,
                            run_id: int = 0) -> ExperimentResult:
        """Run a single experiment and return results."""
        print(f"\n{'='*60}")
        print(f"Running: {config.name} (Run {run_id + 1}/{config.n_runs})")
        print(f"{'='*60}")

        # Setup
        setup = self.setup_experiment(config)
        start_time = time.time()

        # Initialize result
        result = ExperimentResult(config=config, run_id=run_id)
        result.trajectory_data = setup['env_data']

        # Run algorithm
        if config.algorithm.startswith('janus'):
            result = self._run_janus_experiment(setup, config, result)
        elif config.algorithm == 'genetic':
            result = self._run_genetic_experiment(setup, config, result)
        # Add more algorithm runners

        # Calculate final metrics
        result.wall_time_seconds = time.time() - start_time
        result.symbolic_accuracy = calculate_symbolic_accuracy(
            result.discovered_law,
            setup['ground_truth']
        )

        return result

    def _run_janus_experiment(self,
                            setup: Dict,
                            config: ExperimentConfig,
                            result: ExperimentResult) -> ExperimentResult:
        """Run Janus algorithm."""
        trainer = setup['algorithm']

        # Track learning curve
        sample_efficiency_curve = []
        best_mse = float('inf')
        best_expression = None

        # Training loop with periodic evaluation
        timesteps_per_eval = 1000
        total_timesteps = config.max_experiments * 50  # Approximate

        for timestep in range(0, total_timesteps, timesteps_per_eval):
            # Train
            trainer.train(
                total_timesteps=timesteps_per_eval,
                rollout_length=512,
                n_epochs=3,
                log_interval=100
            )

            # Evaluate best discovered expression
            if hasattr(trainer, 'env') and hasattr(trainer.env, '_evaluation_cache') and trainer.env._evaluation_cache:
                for eval_entry in trainer.env._evaluation_cache:
                    current_mse = eval_entry['mse']
                    current_expression = eval_entry['expression']
                    if current_mse < best_mse:
                        best_mse = current_mse
                        best_expression = current_expression # Store best expression

                # Log current best mse for sample efficiency curve
                # We use the overall best_mse found so far, not just from the current cache
                sample_efficiency_curve.append((timestep, best_mse))
            elif trainer.episode_mse: # Fallback if cache not available
                current_mse_from_episode = np.mean(trainer.episode_mse)
                if current_mse_from_episode < best_mse:
                    best_mse = current_mse_from_episode
                    # Expression is not available in this fallback case
                sample_efficiency_curve.append((timestep, current_mse_from_episode))

        result.discovered_law = best_expression # Store best_expression in results
        result.predictive_mse = best_mse
        result.sample_efficiency_curve = sample_efficiency_curve
        result.n_experiments_to_convergence = len(sample_efficiency_curve) * 10

        return result

    def _run_genetic_experiment(self,
                              setup: Dict,
                              config: ExperimentConfig,
                              result: ExperimentResult) -> ExperimentResult:
        """Run genetic programming baseline."""
        regressor = setup['algorithm']

        # Extract features and targets
        target_idx_to_use = config.target_variable_index
        if target_idx_to_use is None:
            target_idx_to_use = -1 # Default to last column

        y = setup['env_data'][:, target_idx_to_use]
        X = np.delete(setup['env_data'], target_idx_to_use, axis=1)


        # Adjust variable indices if target is not the last column
        # This is a simplification; a robust solution would map original indices to new X indices
        # For now, we assume variables are correctly aligned or genetic algorithm handles it
        # based on the number of columns in X.

        # Fit
        best_expr = regressor.fit(
            X, y,
            setup['variables'],
            max_complexity=config.algo_params.get('max_complexity', 15)
        )

        result.discovered_law = str(best_expr.symbolic)
        result.law_complexity = best_expr.complexity

        # Calculate MSE
        predictions = []
        for i in range(X.shape[0]):
            subs = {var.symbolic: X[i, var.index] for var in setup['variables']}
            try:
                pred = float(best_expr.symbolic.subs(subs))
                predictions.append(pred)
            except:
                predictions.append(0)

        result.predictive_mse = np.mean((np.array(predictions) - y)**2)

        return result

    def run_experiment_suite(self,
                           configs: List[ExperimentConfig],
                           parallel: bool = False) -> pd.DataFrame:
        """Run a suite of experiments and return aggregated results."""
        all_results = []

        for config in tqdm(configs, desc="Experiments"):
            config_results = []

            for run_id in range(config.n_runs):
                result = self.run_single_experiment(config, run_id)
                config_results.append(result)

                # Save individual result
                self._save_result(result)

            all_results.extend(config_results)

        # Create results dataframe
        df = self._results_to_dataframe(all_results)

        # Save aggregated results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_csv(self.base_dir / f"results_{timestamp}.csv", index=False)

        return df

    def _save_result(self, result: ExperimentResult):
        """Save individual result."""
        result_dir = self.base_dir / result.config.get_hash()
        result_dir.mkdir(exist_ok=True)

        filename = f"run_{result.run_id}.pkl"
        with open(result_dir / filename, 'wb') as f:
            pickle.dump(result, f)

        # Also save human-readable summary
        summary = {
            'config_name': result.config.name,
            'discovered_law': result.discovered_law,
            'symbolic_accuracy': result.symbolic_accuracy,
            'predictive_mse': result.predictive_mse,
            'n_experiments': result.n_experiments_to_convergence,
            'wall_time': result.wall_time_seconds
        }

        with open(result_dir / f"summary_{result.run_id}.json", 'w') as f:
            json.dump(summary, f, indent=2)

    def _results_to_dataframe(self, results: List[ExperimentResult]) -> pd.DataFrame:
        """Convert results to pandas DataFrame for analysis."""
        rows = []
        for result in results:
            row = {
                'experiment': result.config.name,
                'algorithm': result.config.algorithm,
                'environment': result.config.environment_type,
                'noise_level': result.config.noise_level,
                'run_id': result.run_id,
                'symbolic_accuracy': result.symbolic_accuracy,
                'predictive_mse': result.predictive_mse,
                'law_complexity': result.law_complexity,
                'n_experiments': result.n_experiments_to_convergence,
                'wall_time': result.wall_time_seconds,
                'discovered_law': result.discovered_law
            }
            rows.append(row)

        return pd.DataFrame(rows)

    def analyze_results(self, df: pd.DataFrame):
        """Generate analysis plots and statistics."""
        # Create analysis directory
        analysis_dir = self.base_dir / "analysis"
        analysis_dir.mkdir(exist_ok=True)

        # 1. Sample efficiency comparison
        plt.figure(figsize=(10, 6))
        for algo in df['algorithm'].unique():
            algo_df = df[df['algorithm'] == algo]
            mean_experiments = algo_df.groupby('noise_level')['n_experiments'].mean()
            std_experiments = algo_df.groupby('noise_level')['n_experiments'].std()

            plt.errorbar(mean_experiments.index, mean_experiments.values,
                        yerr=std_experiments.values, label=algo, marker='o')

        plt.xlabel('Noise Level')
        plt.ylabel('Experiments to Convergence')
        plt.title('Sample Efficiency vs. Noise')
        plt.legend()
        plt.yscale('log')
        plt.savefig(analysis_dir / 'sample_efficiency.png')
        plt.close()

        # 2. Accuracy comparison
        plt.figure(figsize=(10, 6))
        accuracy_pivot = df.pivot_table(
            values='symbolic_accuracy',
            index='environment',
            columns='algorithm',
            aggfunc='mean'
        )

        accuracy_pivot.plot(kind='bar')
        plt.ylabel('Symbolic Accuracy')
        plt.title('Discovery Accuracy by Environment')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(analysis_dir / 'accuracy_comparison.png')
        plt.close()

        # 3. Statistical summary
        summary_stats = df.groupby(['algorithm', 'environment']).agg({
            'symbolic_accuracy': ['mean', 'std'],
            'predictive_mse': ['mean', 'std'],
            'n_experiments': ['mean', 'std'],
            'wall_time': ['mean', 'std']
        }).round(3)

        summary_stats.to_csv(analysis_dir / 'summary_statistics.csv')
        print("\n=== Summary Statistics ===")
        print(summary_stats)

        return summary_stats


# Phase 1 implementation
def run_phase1_validation():
    """Run Phase 1: Known law rediscovery."""
    runner = ExperimentRunner()

    configs = []

    # Harmonic oscillator experiments
    for algo in ['janus_full', 'genetic']:
        config = ExperimentConfig(
            name=f"HO_rediscovery_{algo}",
            environment_type='harmonic_oscillator',
            algorithm=algo,
            env_params={'k': 1.0, 'm': 1.0},
            noise_level=0.0,
            max_experiments=1000,
            n_runs=5
        )
        configs.append(config)

    # Pendulum experiments
    for algo in ['janus_full', 'genetic']:
        config = ExperimentConfig(
            name=f"Pendulum_rediscovery_{algo}",
            environment_type='pendulum',
            algorithm=algo,
            env_params={'g': 9.81, 'l': 1.0, 'm': 1.0, 'small_angle': True},
            noise_level=0.0,
            max_experiments=1000,
            n_runs=5
        )
        configs.append(config)

    # Run all experiments
    results_df = runner.run_experiment_suite(configs)

    # Analyze
    runner.analyze_results(results_df)

    return results_df


# Phase 2 implementation
def run_phase2_robustness():
    """Run Phase 2: Efficiency and robustness benchmarking."""
    runner = ExperimentRunner()

    configs = []
    noise_levels = [0.0, 0.05, 0.1, 0.2]

    for noise in noise_levels:
        for algo in ['janus_full', 'genetic', 'random']:
            config = ExperimentConfig(
                name=f"HO_noise_{noise}_{algo}",
                environment_type='harmonic_oscillator',
                algorithm=algo,
                env_params={'k': 1.0, 'm': 1.0},
                noise_level=noise,
                max_experiments=2000,
                n_runs=10
            )
            configs.append(config)

    results_df = runner.run_experiment_suite(configs)
    runner.analyze_results(results_df)

    return results_df


if __name__ == "__main__":
    print("ExperimentRunner Test")
    print("====================")

    # Test basic functionality
    config = ExperimentConfig(
        name="test_harmonic_oscillator",
        environment_type='harmonic_oscillator',
        algorithm='genetic',
        env_params={'k': 1.0, 'm': 1.0},
        noise_level=0.05,
        n_runs=1
    )

    runner = ExperimentRunner(use_wandb=False)
    result = runner.run_single_experiment(config, run_id=0)

    print(f"\nDiscovered law: {result.discovered_law}")
    print(f"Symbolic accuracy: {result.symbolic_accuracy:.3f}")
    print(f"Predictive MSE: {result.predictive_mse:.3e}")
    print(f"Wall time: {result.wall_time_seconds:.1f}s")
