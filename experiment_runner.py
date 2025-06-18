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
import abc # Added for BaseExperiment if it uses it, and for the new class
import logging # Added for logging within PhysicsDiscoveryExperiment methods
import importlib.metadata # For plugin discovery

from base_experiment import BaseExperiment # Added
from utils import calculate_symbolic_accuracy

@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    name: str
    experiment_type: str # New field: Specifies the experiment plugin to use
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

        # Environment registry (for physics environments used by experiments)
        self.env_registry = {
            'harmonic_oscillator': HarmonicOscillatorEnv,
            'pendulum': PendulumEnv,
            'kepler': KeplerEnv
            # This registry might also become plugin-based in the future
        }

        # Algorithm registry (for algorithms used by experiments)
        self.algo_registry: Dict[str, Callable] = {}
        self._register_algorithms() # Populates self.algo_registry

        # Experiment plugin registry
        self.experiment_plugins: Dict[str, Callable[..., BaseExperiment]] = {}
        self._discover_experiments() # Populates self.experiment_plugins

    def _discover_experiments(self):
        """Discovers and registers experiment plugins from entry points."""
        logging.info("Discovering 'janus.experiments' plugins...")
        try:
            # Modern approach for Python 3.8+
            # For Python 3.10+ one could use:
            # eps = importlib.metadata.entry_points(group='janus.experiments')
            # For 3.8/3.9, the following is more robust if group-specific selection isn't available directly
            all_eps = importlib.metadata.entry_points()
            if hasattr(all_eps, 'select'): # For Python 3.10+ and some backports
                eps = all_eps.select(group='janus.experiments')
            elif 'janus.experiments' in all_eps: # For older 3.8/3.9
                eps = all_eps['janus.experiments']
            else:
                eps = []
        except Exception as e:
            logging.warning(f"Could not query entry points for 'janus.experiments' due to: {e}. Manual registration or installation might be needed.")
            eps = []

        if not eps:
            logging.warning("No 'janus.experiments' plugins found or loaded. ExperimentRunner may not find specific experiment types.")
            # Optionally, register default known experiments here if desired
            # For example, if PhysicsDiscoveryExperiment is core and should always be available:
            # if 'physics_discovery_example' not in self.experiment_plugins and hasattr(sys.modules[__name__], 'PhysicsDiscoveryExperiment'):
            #     self.experiment_plugins['physics_discovery_example'] = getattr(sys.modules[__name__], 'PhysicsDiscoveryExperiment')
            #     logging.info("Registered fallback/default 'physics_discovery_example'.")
            return

        for entry_point in eps:
            try:
                loaded_class = entry_point.load()
                # Check if it's a class and a subclass of BaseExperiment
                if isinstance(loaded_class, type) and issubclass(loaded_class, BaseExperiment):
                    if entry_point.name in self.experiment_plugins:
                        logging.warning(f"Duplicate experiment plugin name '{entry_point.name}'. Overwriting {self.experiment_plugins[entry_point.name]} with {loaded_class}.")
                    self.experiment_plugins[entry_point.name] = loaded_class
                    logging.info(f"Discovered and registered experiment plugin: '{entry_point.name}' -> {loaded_class.__module__}.{loaded_class.__name__}")
                else:
                    logging.warning(f"Plugin '{entry_point.name}' ({entry_point.value}) is not a valid class or does not inherit from BaseExperiment. Skipping.")
            except Exception as e:
                logging.error(f"Failed to load experiment plugin '{entry_point.name}' from value '{entry_point.value}'. Error: {e}", exc_info=True)

        if not self.experiment_plugins:
            logging.warning("No 'janus.experiments' plugins were successfully loaded after processing entry points.")
        else:
            logging.info(f"Successfully loaded experiment plugins: {list(self.experiment_plugins.keys())}")

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

    # def setup_experiment(self, config: ExperimentConfig) -> Dict[str, Any]: # MOVED to PhysicsDiscoveryExperiment
    #     """Set up an experiment with given configuration."""
    #     # np.random.seed(config.seed)
    #     # torch.manual_seed(config.seed)
    #     # ... (rest of the logic is now in PhysicsDiscoveryExperiment.setup())

    def run_single_experiment(self,
                            config: ExperimentConfig,
                            run_id: int = 0) -> ExperimentResult:
        """
        Run a single experiment using the BaseExperiment structure and return results.
        This version uses the plugin system to select the experiment class.
        """
        logging.info(f"ExperimentRunner: Starting run_single_experiment for '{config.name}' (Run {run_id + 1}/{config.n_runs}). Experiment Type: '{config.experiment_type}'")

        if not hasattr(config, 'experiment_type') or not config.experiment_type:
            logging.error(f"Experiment configuration '{config.name}' is missing the 'experiment_type' field.")
            err_result = ExperimentResult(config=config, run_id=run_id, discovered_law="Error: experiment_type not specified in config.")
            err_result.symbolic_accuracy = 0.0
            err_result.predictive_mse = float('inf')
            return err_result

        experiment_class = self.experiment_plugins.get(config.experiment_type)

        if experiment_class is None:
            logging.error(f"Experiment type '{config.experiment_type}' for config '{config.name}' not found in discovered plugins. Available plugins: {list(self.experiment_plugins.keys())}")
            err_result = ExperimentResult(config=config, run_id=run_id, discovered_law=f"Error: Unknown experiment_type '{config.experiment_type}'.")
            err_result.symbolic_accuracy = 0.0
            err_result.predictive_mse = float('inf')
            return err_result

        logging.info(f"Instantiating experiment of type '{config.experiment_type}' using class {experiment_class.__module__}.{experiment_class.__name__}.")

        try:
            # Instantiate the chosen experiment class.
            # All experiment plugins are expected to have a constructor compatible with:
            # (config: ExperimentConfig, algo_registry: Dict, env_registry: Dict)
            experiment_instance = experiment_class(
                config=config,
                algo_registry=self.algo_registry,
                env_registry=self.env_registry
            )
        except Exception as e:
            logging.error(f"Failed to instantiate experiment class '{experiment_class.__name__}' for type '{config.experiment_type}'. Error: {e}", exc_info=True)
            err_result = ExperimentResult(config=config, run_id=run_id, discovered_law=f"Error: Failed to instantiate experiment '{config.experiment_type}'.")
            err_result.symbolic_accuracy = 0.0
            err_result.predictive_mse = float('inf')
            return err_result

        # Execute the experiment (calls setup, run, teardown internally via BaseExperiment's execute method)
        result = experiment_instance.execute(run_id=run_id)

        if result is None:
            # This case should ideally be handled by experiment_instance.execute() always returning an ExperimentResult.
            logging.error(f"Experiment '{config.name}' (Run {run_id + 1}, Type '{config.experiment_type}') critically failed and did not return a result object from execute().")
            # Create a minimal error result to avoid downstream errors if execute() somehow returns None
            result = ExperimentResult(config=config, run_id=run_id, discovered_law="Critical Execution Error: execute() returned None.")
            result.symbolic_accuracy = 0.0
            result.predictive_mse = float('inf')
            result.wall_time_seconds = 0.0
        else:
            logging.info(f"ExperimentRunner: Completed run_single_experiment for '{config.name}' (Run {run_id + 1}). Law: '{result.discovered_law}', Accuracy: {result.symbolic_accuracy:.4f}, Time: {result.wall_time_seconds:.2f}s.")

        return result

    # _run_janus_experiment and _run_genetic_experiment have been moved into PhysicsDiscoveryExperiment.run()
    # For now, they remain here but are not directly called by the new run_single_experiment.
    # PhysicsDiscoveryExperiment.run() currently has a placeholder.
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
            experiment_type='physics_discovery_example', # Added field
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
            experiment_type='physics_discovery_example', # Added field
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
                experiment_type='physics_discovery_example', # Added field
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
        name="test_harmonic_oscillator_from_main",
        experiment_type='physics_discovery_example', # Added: Specify which plugin to use
        environment_type='harmonic_oscillator',
        algorithm='genetic', # This will be used by PhysicsDiscoveryExperiment's run method
        env_params={'k': 1.0, 'm': 1.0},
        noise_level=0.05,
        n_runs=1,
        # algo_params, trajectory_length etc. will use defaults or can be added here
    )

    runner = ExperimentRunner(use_wandb=False) # Initializes plugins

    # Check if the example plugin is available before trying to run
    if config.experiment_type in runner.experiment_plugins:
        logging.info(f"Attempting to run experiment '{config.name}' of type '{config.experiment_type}' from __main__.")
        result = runner.run_single_experiment(config, run_id=0)

        if result:
            print(f"\n--- Experiment '{config.name}' Results ---")
            print(f"Discovered law: {result.discovered_law}")
            print(f"Symbolic accuracy: {result.symbolic_accuracy:.3f}")
            print(f"Predictive MSE: {result.predictive_mse:.3e}")
            print(f"Wall time: {result.wall_time_seconds:.1f}s")
            print(f"Run ID: {result.run_id}")
            print(f"Config used: {result.config}")
        else:
            print(f"Experiment '{config.name}' did not return a result object.")
    else:
        logging.error(f"Experiment type '{config.experiment_type}' (named '{config.name}') not found in discovered plugins.")
        logging.error("This is expected if the package has not been installed (e.g., via 'pip install .').")
        logging.error(f"Available plugins: {list(runner.experiment_plugins.keys())}")
        print(f"\nSkipping __main__ execution of '{config.name}' as plugin '{config.experiment_type}' is not available.")
        print("This might be because the package is not installed in editable mode or at all.")
        print("Try running 'pip install -e .' from the repository root.")


    # New way (conceptual) - This is now handled inside run_single_experiment
    # experiment_class_loaded = runner.experiment_plugins.get(config.experiment_type)
    # if experiment_class_loaded:
    #    experiment = experiment_class_loaded(config, runner.algo_registry, runner.env_registry)
    # result = experiment.execute() # execute will store result, run_single_experiment will retrieve it.


    # For now, let's comment out the direct execution in main,
    # as the refactoring is significant and will be tested step-by-step.
    print("Refactoring in progress. Direct execution in __main__ is temporarily commented out.")
    # print(f"\nDiscovered law: {result.discovered_law}") # Temporarily commented
    # print(f"Symbolic accuracy: {result.symbolic_accuracy:.3f}") # Temporarily commented
    # print(f"Predictive MSE: {result.predictive_mse:.3e}") # Temporarily commented
    # print(f"Wall time: {result.wall_time_seconds:.1f}s") # Temporarily commented

# Placeholder for the new PhysicsDiscoveryExperiment class
# We will fill this in subsequent steps.
class PhysicsDiscoveryExperiment(BaseExperiment):
    def __init__(self,
                 config: ExperimentConfig,
                 algo_registry: Dict[str, Callable],
                 env_registry: Dict[str, Callable]):
        super().__init__() # Initialize BaseExperiment
        self.config = config
        self.algo_registry = algo_registry
        self.env_registry = env_registry

        # Attributes to be populated by setup
        self.physics_env: Optional[PhysicsEnvironment] = None
        self.env_data: Optional[np.ndarray] = None
        self.variables: Optional[List[Any]] = None # Variable type from progressive_grammar_system
        self.algorithm: Optional[Any] = None # Type depends on algorithm
        self.ground_truth_laws: Optional[Dict[str, sp.Expr]] = None

        # Result object, to be populated by execute/run
        self.experiment_result: Optional[ExperimentResult] = None
        self._start_time: float = 0.0

    def setup(self):
        """Sets up the experiment environment, data, and algorithm."""
        logging.info(f"[{self.config.name}] Initializing setup: Seed {self.config.seed}, Env '{self.config.environment_type}', Algo '{self.config.algorithm}'.")
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

        # 1. Create Environment
        logging.debug(f"[{self.config.name}] Creating environment '{self.config.environment_type}'.")
        if self.config.environment_type not in self.env_registry:
            logging.error(f"Environment type '{self.config.environment_type}' not in registry. Available: {list(self.env_registry.keys())}")
            raise ValueError(f"Environment type '{self.config.environment_type}' not found in registry.")
        env_class = self.env_registry[self.config.environment_type]
        self.physics_env = env_class(self.config.env_params, self.config.noise_level)
        logging.info(f"[{self.config.name}] Environment '{self.config.environment_type}' created: {self.physics_env}")

        # 2. Generate Training Data
        logging.debug(f"[{self.config.name}] Generating data: {self.config.n_trajectories} trajectories, {self.config.trajectory_length} length.")
        trajectories = []
        for i in range(self.config.n_trajectories):
            # Determine initial conditions based on environment type
            if self.config.environment_type == 'harmonic_oscillator':
                init_cond = np.random.randn(2) * np.array([1.0, 2.0])  # x0, v0
            elif self.config.environment_type == 'pendulum':
                max_angle = np.pi / 12 if self.config.env_params.get('small_angle', False) else np.pi
                init_cond = np.array([np.random.uniform(-max_angle, max_angle), np.random.uniform(-1.0, 1.0)]) # theta, omega
            elif self.config.environment_type == 'kepler':
                ecc = np.random.uniform(0.0, 0.6)
                sma = np.random.uniform(0.5, 1.5)
                r_p = sma * (1 - ecc)
                if self.physics_env and hasattr(self.physics_env, 'G') and hasattr(self.physics_env, 'M') and (self.physics_env.G * self.physics_env.M > 0):
                    v_p_linear = np.sqrt(self.physics_env.G * self.physics_env.M * (2 / r_p - 1 / sma))
                    v_theta_angular = v_p_linear / r_p
                else: # Fallback if G or M are zero or not set, avoid division by zero / sqrt of negative
                    v_theta_angular = np.random.uniform(0.5, 1.5) / r_p if r_p > 0 else 1.0
                init_cond = np.array([r_p, 0.0, 0.0, v_theta_angular])  # r, theta_angle, vr, v_theta_angular_speed
            else:
                num_state_vars = len(self.physics_env.state_vars) if self.physics_env and self.physics_env.state_vars else 2
                init_cond = np.random.rand(num_state_vars) * 2 - 1  # Random values in [-1, 1]

            t_span = np.arange(0, self.config.trajectory_length * self.config.sampling_rate, self.config.sampling_rate)
            trajectory = self.physics_env.generate_trajectory(init_cond, t_span)
            trajectories.append(trajectory)

        if not trajectories:
            logging.error(f"[{self.config.name}] No trajectories generated.")
            raise ValueError("No trajectories generated. Check environment parameters and generation logic.")
        self.env_data = np.vstack(trajectories)
        logging.info(f"[{self.config.name}] Training data generated with shape {self.env_data.shape}.")

        # 3. Create Variables for Algorithm
        # Ensure progressive_grammar_system.Variable is available
        try:
            from progressive_grammar_system import Variable
        except ImportError:
            logging.error("Failed to import 'Variable' from 'progressive_grammar_system'. Ensure it's installed and accessible.")
            raise

        self.variables = [Variable(name, idx, {}) for idx, name in enumerate(self.physics_env.state_vars)]
        logging.debug(f"[{self.config.name}] State variables for algorithm: {self.variables}")

        # 4. Create Algorithm Instance
        logging.debug(f"[{self.config.name}] Creating algorithm '{self.config.algorithm}'.")
        if self.config.algorithm not in self.algo_registry:
            logging.error(f"Algorithm '{self.config.algorithm}' not in registry. Available: {list(self.algo_registry.keys())}")
            raise ValueError(f"Algorithm '{self.config.algorithm}' not found in registry.")

        algo_factory = self.algo_registry[self.config.algorithm]
        self.algorithm = algo_factory(self.env_data, self.variables, self.config) # Pass full config
        logging.info(f"[{self.config.name}] Algorithm '{self.config.algorithm}' created: {self.algorithm}")

        # 5. Get Ground Truth Laws
        self.ground_truth_laws = self.physics_env.get_ground_truth_laws()
        logging.debug(f"[{self.config.name}] Ground truth laws obtained: {list(self.ground_truth_laws.keys()) if self.ground_truth_laws else 'None'}.")
        logging.info(f"[{self.config.name}] Setup phase complete.")

    def run(self, run_id: int) -> ExperimentResult:
        """
        Runs the core logic of the experiment (algorithm execution) based on the
        algorithm specified in the config.
        """
        logging.info(f"[{self.config.name}] Starting experiment run phase (run_id: {run_id}). Algorithm: {self.config.algorithm}")

        current_run_result = ExperimentResult(config=self.config, run_id=run_id)
        current_run_result.trajectory_data = self.env_data # Populated during setup

        # Ensure algorithm was created in setup
        if self.algorithm is None:
            logging.error(f"[{self.config.name}] Algorithm was not initialized during setup. Aborting run.")
            current_run_result.discovered_law = "Error: Algorithm not initialized"
            current_run_result.symbolic_accuracy = 0.0
            current_run_result.predictive_mse = float('inf')
            return current_run_result

        # --- Algorithm-specific execution ---
        if self.config.algorithm.startswith('janus'):
            # Logic from _run_janus_experiment
            trainer = self.algorithm # self.algorithm is the trainer instance from setup

            sample_efficiency_curve = []
            best_mse = float('inf')
            best_expression = None # Stores the symbolic expression string

            # Training loop with periodic evaluation
            # Use max_experiments from config, assuming it means number of evaluations or similar proxy
            # The original total_timesteps = config.max_experiments * 50 was an approximation.
            # Let's refine this: if max_experiments is "number of PPO rollouts/updates",
            # and each eval happens after N updates.
            # For now, using config.max_experiments as number of training iterations/epochs for the PPO trainer
            # This needs to align with how PPOTrainer is implemented.
            # Assuming trainer.train() handles its own internal loop and timesteps_per_eval is for our eval points

            # Using max_experiments as a guide for number of "major training cycles" or evaluations
            # This part might need adjustment based on PPOTrainer's train method behavior
            num_evaluations = self.config.algo_params.get('num_evaluations', 50) # Default to 50 evaluations
            timesteps_per_eval_cycle = self.config.algo_params.get('timesteps_per_eval_cycle', 1000) # PPO steps per cycle

            logging.info(f"[{self.config.name}] Janus training: {num_evaluations} evaluations, {timesteps_per_eval_cycle} PPO steps per cycle.")

            for i in range(num_evaluations):
                # Train the PPO agent for a certain number of timesteps
                # The PPOTrainer's train method might take total_timesteps, n_epochs, etc.
                # This needs to match the PPOTrainer's API.
                # Assuming trainer.train() is a blocking call for one cycle of training.
                # Parameters like rollout_length, n_epochs are from original _run_janus_experiment
                # and should be part of algo_params in config.
                ppo_train_params = self.config.algo_params.get('ppo_train_params', {
                    'total_timesteps': timesteps_per_eval_cycle, # Steps for this training call
                    'rollout_length': 512,
                    'n_epochs': 3,
                    'log_interval': 100
                })
                trainer.train(**ppo_train_params)

                current_timestep_marker = (i + 1) * timesteps_per_eval_cycle

                # Evaluate best discovered expression from the discovery environment's cache
                if hasattr(trainer, 'env') and hasattr(trainer.env, '_evaluation_cache') and trainer.env._evaluation_cache:
                    # Iterate through new entries in cache, or just get the current best
                    # This logic assumes _evaluation_cache stores {expression_str: mse} or similar
                    # The original code iterated trainer.env._evaluation_cache which was a list of dicts
                    for eval_entry in trainer.env._evaluation_cache: # Assuming this is still a list
                        mse = eval_entry.get('mse', float('inf'))
                        expr_str = eval_entry.get('expression') # Assuming it's a string
                        if expr_str and mse < best_mse:
                            best_mse = mse
                            best_expression = expr_str
                    sample_efficiency_curve.append((current_timestep_marker, best_mse))
                elif hasattr(trainer, 'episode_mse') and trainer.episode_mse: # Fallback
                    # This fallback might not provide the expression string
                    current_episode_mse_mean = np.mean(trainer.episode_mse)
                    if current_episode_mse_mean < best_mse:
                        best_mse = current_episode_mse_mean
                        # best_expression remains the last known best or None
                    sample_efficiency_curve.append((current_timestep_marker, best_mse)) # Log with overall best_mse
                else:
                    # If no evaluation mechanism, log a warning or default value
                    logging.warning(f"[{self.config.name}] No evaluation cache or episode_mse found on Janus trainer. Cannot track MSE progress.")
                    sample_efficiency_curve.append((current_timestep_marker, float('inf')))


            current_run_result.discovered_law = str(best_expression) if best_expression else None
            current_run_result.predictive_mse = best_mse if best_mse != float('inf') else float('inf')
            current_run_result.sample_efficiency_curve = sample_efficiency_curve
            # n_experiments_to_convergence is tricky. Original: len(sample_efficiency_curve) * 10
            # This depends on what "experiment" means. If it's an eval, then it's len(sample_efficiency_curve)
            current_run_result.n_experiments_to_convergence = len(sample_efficiency_curve) * self.config.algo_params.get('eval_multiplier_for_convergence', 1)


        elif self.config.algorithm == 'genetic':
            # Logic from _run_genetic_experiment
            regressor = self.algorithm # self.algorithm is the SymbolicRegressor instance

            target_idx_to_use = self.config.target_variable_index
            if target_idx_to_use is None: # Default to the last column if not specified
                target_idx_to_use = -1

            if self.env_data is None or self.env_data.shape[1] <= abs(target_idx_to_use):
                 logging.error(f"[{self.config.name}] Environment data is missing or target index is out of bounds.")
                 current_run_result.discovered_law = "Error: Invalid data or target index for genetic algorithm."
                 return current_run_result

            y = self.env_data[:, target_idx_to_use]
            X = np.delete(self.env_data, target_idx_to_use, axis=1)

            # Ensure variables passed to regressor.fit are consistent with X's columns
            # The self.variables were created based on original state_vars. If target is removed,
            # variable mapping might need adjustment if regressor.fit expects it.
            # Assuming SymbolicRegressor.fit can handle variables list that might include the target,
            # or it internally selects based on X's shape. Or, variables should be filtered.
            # For now, passing self.variables as is, assuming regressor handles it.

            logging.info(f"[{self.config.name}] Genetic training: X shape {X.shape}, y shape {y.shape}. Variables: {self.variables}")

            best_expr_obj = regressor.fit( # Assuming fit returns an object with .symbolic and .complexity
                X, y,
                self.variables, # These are Variable objects from progressive_grammar_system
                max_complexity=self.config.algo_params.get('max_complexity', 15)
            )

            if best_expr_obj and hasattr(best_expr_obj, 'symbolic'):
                current_run_result.discovered_law = str(best_expr_obj.symbolic)
                current_run_result.law_complexity = best_expr_obj.complexity if hasattr(best_expr_obj, 'complexity') else len(str(best_expr_obj.symbolic))

                # Calculate MSE for genetic algorithm's best expression
                # This requires substituting values into the symbolic expression
                predictions = []
                # Ensure self.variables map correctly to columns of X for substitution
                # If target was removed, and variables were not filtered, this needs care.
                # Assuming var.index corresponds to original column indices.
                # We need a mapping from original index to X's column index if target wasn't last.

                # Simplified: if variables are Variable objects with 'symbolic' and 'index'
                # and regressor.fit ensures best_expr_obj.symbolic uses these.
                # The original code for MSE calculation:
                # predictions = []
                # for i in range(X.shape[0]):
                #     subs = {var.symbolic: X[i, var.index] for var in setup['variables']} # var.index needs to be correct for X
                #     try:
                #         pred = float(best_expr_obj.symbolic.subs(subs))
                #         predictions.append(pred)
                #     except: # Broad except, consider specific exceptions
                #         predictions.append(np.nan) # Use NaN for failed predictions
                # current_run_result.predictive_mse = np.nanmean((np.array(predictions) - y)**2) if predictions else float('inf')
                # For now, assuming regressor.fit might also return MSE or a way to predict:
                if hasattr(best_expr_obj, 'mse') and best_expr_obj.mse is not None:
                    current_run_result.predictive_mse = best_expr_obj.mse
                elif hasattr(regressor, 'predict'): # If regressor has a predict method
                    try:
                        predictions = regressor.predict(X) # Assuming predict takes X
                        current_run_result.predictive_mse = np.mean((predictions - y)**2)
                    except Exception as e:
                        logging.error(f"[{self.config.name}] Error during prediction with genetic model: {e}")
                        current_run_result.predictive_mse = float('inf')
                else: # Fallback to manual substitution if needed (more complex due to variable mapping)
                    logging.warning(f"[{self.config.name}] Genetic algorithm did not provide direct MSE or predict method. MSE might be inaccurate.")
                    current_run_result.predictive_mse = float('inf') # Placeholder if no direct MSE
            else:
                current_run_result.discovered_law = "Error: Genetic algorithm failed to find an expression."
                current_run_result.predictive_mse = float('inf')
                current_run_result.law_complexity = 0

            # n_experiments_to_convergence for genetic is often just 1 (as it's one fit call)
            # Or could be generations if the regressor tracks that.
            current_run_result.n_experiments_to_convergence = 1 # Default for batch genetic algorithm

        elif self.config.algorithm == 'random':
            logging.info(f"[{self.config.name}] Running Random Search algorithm...")
            # Implement random search: generate a number of random expressions
            num_random_expressions = self.config.algo_params.get('num_random_expressions', 100)
            best_random_expr_str = None
            # This would require a grammar and a way to sample from it, then evaluate MSE.
            # For now, this is a placeholder.
            current_run_result.discovered_law = "random_placeholder_expression"
            current_run_result.predictive_mse = np.random.rand() * 100
            current_run_result.law_complexity = len(current_run_result.discovered_law)
            current_run_result.n_experiments_to_convergence = num_random_expressions
            logging.warning(f"[{self.config.name}] Random Search algorithm is a basic placeholder.")

        else:
            logging.error(f"[{self.config.name}] Unknown algorithm type '{self.config.algorithm}' for run method.")
            current_run_result.discovered_law = f"Error: Unknown algorithm type '{self.config.algorithm}'"
            current_run_result.predictive_mse = float('inf')
            current_run_result.symbolic_accuracy = 0.0
            return current_run_result

        # --- Final calculations (if not done by algo-specific logic) ---
        # Symbolic accuracy calculation
        if self.ground_truth_laws and current_run_result.discovered_law:
            current_run_result.symbolic_accuracy = calculate_symbolic_accuracy(
                current_run_result.discovered_law,
                self.ground_truth_laws
            )
        else:
            current_run_result.symbolic_accuracy = 0.0
            if not self.ground_truth_laws:
                 logging.debug(f"[{self.config.name}] No ground truth laws provided for symbolic accuracy calculation.")
            if not current_run_result.discovered_law:
                 logging.debug(f"[{self.config.name}] No law discovered for symbolic accuracy calculation.")


        logging.info(f"[{self.config.name}] Run phase finished. Law: '{current_run_result.discovered_law}', Accuracy: {current_run_result.symbolic_accuracy:.4f}, MSE: {current_run_result.predictive_mse:.4e}")
        return current_run_result

    def teardown(self):
        """Cleans up any resources used by the experiment."""
        logging.info(f"[{self.config.name}] Tearing down experiment.")
        # Example: if self.algorithm and hasattr(self.algorithm, 'close'): self.algorithm.close()
        self.physics_env = None
        self.env_data = None
        self.variables = None
        self.algorithm = None
        self.ground_truth_laws = None
        logging.info(f"[{self.config.name}] Teardown complete.")

    def execute(self, run_id: int = 0) -> ExperimentResult:
        """
        Orchestrates the experiment: setup, run, teardown.
        Ensures teardown is called and calculates wall time.
        Overrides BaseExperiment.execute to manage self.experiment_result and _start_time.
        """
        self._start_time = time.time()
        # Initialize the result object that will be populated
        self.experiment_result = ExperimentResult(config=self.config, run_id=run_id)
        # Populate trajectory data early if available from setup,
        # though run() will also set it on its returned result.
        if self.env_data is not None:
             self.experiment_result.trajectory_data = self.env_data

        try:
            logging.info(f"[{self.config.name}] EXECUTE: Starting setup (run {run_id})...")
            self.setup() # Calls PhysicsDiscoveryExperiment.setup()
            logging.info(f"[{self.config.name}] EXECUTE: Setup complete (run {run_id}).")

            logging.info(f"[{self.config.name}] EXECUTE: Starting run (run {run_id})...")
            # self.run() is the method defined in this class (PhysicsDiscoveryExperiment)
            # It returns an ExperimentResult instance specific to this run.
            run_specific_result = self.run(run_id=run_id)

            # Merge run_specific_result into self.experiment_result
            # This ensures all fields from run_specific_result are captured.
            self.experiment_result.discovered_law = run_specific_result.discovered_law
            self.experiment_result.symbolic_accuracy = run_specific_result.symbolic_accuracy
            self.experiment_result.predictive_mse = run_specific_result.predictive_mse
            self.experiment_result.law_complexity = run_specific_result.law_complexity
            self.experiment_result.n_experiments_to_convergence = run_specific_result.n_experiments_to_convergence
            self.experiment_result.sample_efficiency_curve = run_specific_result.sample_efficiency_curve
            self.experiment_result.component_metrics = run_specific_result.component_metrics
            # trajectory_data should already be set if setup was successful and run used it.
            if run_specific_result.trajectory_data is not None:
                self.experiment_result.trajectory_data = run_specific_result.trajectory_data


            logging.info(f"[{self.config.name}] EXECUTE: Run complete (run {run_id}). Discovered: {self.experiment_result.discovered_law}")

        except Exception as e:
            logging.error(f"[{self.config.name}] EXECUTE: Exception during setup or run (run {run_id}): {e}", exc_info=True)
            if self.experiment_result: # Ensure it exists
                self.experiment_result.discovered_law = f"Error: {str(e)}"
                self.experiment_result.symbolic_accuracy = 0.0
                self.experiment_result.predictive_mse = float('inf')
            # Optionally re-raise if BaseExperiment's execute shouldn't silence by default for the framework
            # raise
        finally:
            current_wall_time = time.time() - self._start_time
            if self.experiment_result: # Ensure it exists
                self.experiment_result.wall_time_seconds = current_wall_time

            logging.info(f"[{self.config.name}] EXECUTE: Starting teardown (run {run_id}). Wall time: {current_wall_time:.2f}s.")
            self.teardown() # Calls PhysicsDiscoveryExperiment.teardown()
            logging.info(f"[{self.config.name}] EXECUTE: Teardown complete (run {run_id}).")

        if not self.experiment_result:
            # This should ideally not happen if initialization is correct
            logging.error(f"[{self.config.name}] EXECUTE: experiment_result is None at the end of execute. Creating a default error result.")
            self.experiment_result = ExperimentResult(config=self.config, run_id=run_id, discovered_law="Critical error: result object not created.")
            self.experiment_result.wall_time_seconds = time.time() - self._start_time

        return self.experiment_result


if __name__ == "__main__":
    print("ExperimentRunner Test")
    print("====================")

    # Test basic functionality
    config = ExperimentConfig(
        name="test_harmonic_oscillator_from_main",
        experiment_type='physics_discovery_example', # Added: Specify which plugin to use
        environment_type='harmonic_oscillator',
        algorithm='genetic', # This will be used by PhysicsDiscoveryExperiment's run method
        env_params={'k': 1.0, 'm': 1.0},
        noise_level=0.05,
        n_runs=1
        # algo_params, trajectory_length etc. will use defaults or can be added here
    )

    # Configure logging for the __main__ example
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

    runner = ExperimentRunner(use_wandb=False) # Initializes plugins, including discovery

    # Check if the example plugin is available before trying to run
    if config.experiment_type in runner.experiment_plugins:
        logging.info(f"Attempting to run experiment '{config.name}' of type '{config.experiment_type}' from __main__.")
        try:
            result = runner.run_single_experiment(config, run_id=0)

            if result:
                print(f"\n--- Experiment '{config.name}' Results ---")
                print(f"Discovered law: {result.discovered_law}")
                print(f"Symbolic accuracy: {result.symbolic_accuracy:.3f}")
                print(f"Predictive MSE: {result.predictive_mse:.3e}")
                print(f"Wall time: {result.wall_time_seconds:.1f}s")
                print(f"Run ID: {result.run_id}")
                # print(f"Config used: {result.config}") # Config can be verbose
            else:
                # This case should ideally be handled by run_single_experiment returning an error ExperimentResult
                print(f"Experiment '{config.name}' did not return a result object from run_single_experiment.")
        except Exception as e:
            logging.error(f"An error occurred during __main__ execution of experiment '{config.name}': {e}", exc_info=True)
            print(f"An error occurred while running the experiment: {e}")
    else:
        logging.error(f"Experiment type '{config.experiment_type}' (for experiment named '{config.name}') not found in discovered plugins.")
        logging.error("This can happen if the package is not installed (e.g., via 'pip install .') or if entry points are not correctly set up/discovered.")
        logging.error(f"Available plugins: {list(runner.experiment_plugins.keys())}")
        print(f"\nSkipping __main__ execution of '{config.name}' as plugin '{config.experiment_type}' is not available.")
        print("This might be because the package is not installed in editable mode ('pip install -e .') or at all.")
        print("If you have just added the entry point to setup.py, re-installing the package is necessary.")

    # New way (conceptual) - This is now handled inside run_single_experiment
    # experiment_class_loaded = runner.experiment_plugins.get(config.experiment_type)
    # if experiment_class_loaded:
    #    experiment = experiment_class_loaded(config, runner.algo_registry, runner.env_registry)
    # result = experiment.execute() # execute will store result, run_single_experiment will retrieve it.


    # For now, let's comment out the direct execution in main,
    # as the refactoring is significant and will be tested step-by-step.
    print("Refactoring in progress. Direct execution in __main__ is temporarily commented out.")
    # print(f"\nDiscovered law: {result.discovered_law}") # Temporarily commented
    # print(f"Symbolic accuracy: {result.symbolic_accuracy:.3f}") # Temporarily commented
    # print(f"Predictive MSE: {result.predictive_mse:.3e}") # Temporarily commented
    # print(f"Wall time: {result.wall_time_seconds:.1f}s") # Temporarily commented
    # print(f"Symbolic accuracy: {result.symbolic_accuracy:.3f}") # Temporarily commented
    # print(f"Predictive MSE: {result.predictive_mse:.3e}") # Temporarily commented
    # print(f"Wall time: {result.wall_time_seconds:.1f}s") # Temporarily commented
