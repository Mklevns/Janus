"""
ExperimentRunner: Automated Validation Framework for Janus (Refactored for StrictMode)
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
import abc
import logging
import importlib.metadata
import sys # For sys.exit

from custom_exceptions import MissingDependencyError, PluginNotFoundError, InvalidConfigError, DataGenerationError
from base_experiment import BaseExperiment
from math_utils import calculate_symbolic_accuracy
from robust_hypothesis_extraction import HypothesisTracker, JanusTrainingIntegration
from conservation_reward_fix import ConservationBiasedReward
from symmetry_detection_fix import PhysicsSymmetryDetector
from live_monitor import TrainingLogger, LiveMonitor
# Attempt to import EmergentBehaviorTracker, handle if not found
try:
    from emergent_monitor import EmergentBehaviorTracker
    _EMERGENT_MONITOR_AVAILABLE = True
except ImportError:
    _EMERGENT_MONITOR_AVAILABLE = False
    logging.warning("EmergentBehaviorTracker not found in emergent_monitor. Phase transition tracking will be disabled.")
    # Define a dummy class if not available to prevent errors during initialization
    class EmergentBehaviorTracker:
        def __init__(self, *args, **kwargs):
            logging.warning("Using dummy EmergentBehaviorTracker as the real one is not available.")
            self.phase_detector = type('DummyPhaseDetector', (object,), {'phase_transitions': []})()
        def log_discovery(self, *args, **kwargs): pass

from sympy import lambdify, symbols
from progressive_grammar_system import Expression as SymbolicExpression
from config_models import JanusConfig, SyntheticDataParamsConfig
from pydantic import BaseModel


@dataclass
class ExperimentConfig:
    """Encapsulates all settings for a single experimental run. (Refactored)"""
    name: str
    experiment_type: str
    janus_config: JanusConfig

    environment_type: str
    algorithm: str
    noise_level: float

    algo_params: Optional[Dict[str, Any]] = field(default_factory=dict)
    max_experiments: int = 1000
    max_time_seconds: int = 3600

    n_trajectories: int = 10
    trajectory_length: int = 100
    sampling_rate: float = 0.1

    seed: int = 42
    n_runs: int = 5
    target_variable_index: Optional[int] = None

    def __post_init__(self):
        self.environment_type = self.janus_config.target_phenomena
        if self.janus_config.synthetic_data_params:
            self.noise_level = self.janus_config.synthetic_data_params.noise_level
            self.n_trajectories = self.janus_config.synthetic_data_params.n_samples
        self.max_experiments = self.janus_config.num_evaluation_cycles

    @classmethod
    def from_janus_config(cls, name: str, experiment_type: str, janus_config: JanusConfig,
                          algorithm_name: str, n_runs: int = 1, seed: int = 42,
                          algo_params_override: Optional[Dict[str, Any]] = None,
                          target_variable_index: Optional[int] = None) -> 'ExperimentConfig':
        traj_len = 100
        n_traj = 10
        sampling = 0.1
        noise = 0.0

        if janus_config.synthetic_data_params:
            noise = janus_config.synthetic_data_params.noise_level
            n_traj = janus_config.synthetic_data_params.n_samples
            if janus_config.synthetic_data_params.time_range and len(janus_config.synthetic_data_params.time_range) == 2:
                duration = janus_config.synthetic_data_params.time_range[1] - janus_config.synthetic_data_params.time_range[0]
                if sampling > 0: # ensure sampling rate is positive
                    traj_len = int(duration / sampling) if duration / sampling >=1 else 1


        return cls(
            name=name, experiment_type=experiment_type, janus_config=janus_config,
            environment_type=janus_config.target_phenomena, algorithm=algorithm_name,
            noise_level=noise, algo_params=algo_params_override or {},
            max_experiments=janus_config.num_evaluation_cycles, max_time_seconds=3600,
            n_trajectories=n_traj, trajectory_length=traj_len, sampling_rate=sampling,
            seed=seed, n_runs=n_runs, target_variable_index=target_variable_index
        )

    def get_hash(self) -> str:
        temp_dict = asdict(self)
        if 'janus_config' in temp_dict and self.janus_config is not None:
            temp_dict['janus_config'] = self.janus_config.model_dump(mode='json')
        else:
            temp_dict.pop('janus_config', None)
        config_str = json.dumps(temp_dict, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]


@dataclass
class ExperimentResult:
    config: ExperimentConfig
    run_id: int
    discovered_law: Optional[str] = None
    symbolic_accuracy: float = 0.0
    predictive_mse: float = float('inf')
    law_complexity: int = 0
    n_experiments_to_convergence: int = 0
    wall_time_seconds: float = 0.0
    sample_efficiency_curve: List[Tuple[int, float]] = field(default_factory=list)
    noise_resilience_score: float = 0.0
    generalization_score: float = 0.0
    component_metrics: Dict[str, float] = field(default_factory=dict)
    trajectory_data: Optional[np.ndarray] = None
    experiment_log: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict:
        result = asdict(self)
        if self.trajectory_data is not None:
            result['trajectory_data'] = self.trajectory_data.tolist()
        if 'config' in result and result['config'].get('janus_config') is not None:
            if isinstance(self.config.janus_config, BaseModel):
                 result['config']['janus_config'] = self.config.janus_config.model_dump(mode='json')
            elif isinstance(self.config.janus_config, dict):
                 result['config']['janus_config'] = self.config.janus_config
        return result


class PhysicsEnvironment(abc.ABC): # Added abc.ABC
    def __init__(self, params: Dict[str, Any], noise_level: float = 0.0):
        self.params = params
        self.noise_level = noise_level
        self.state_vars: List[str] = [] # Ensure type hint
        self.ground_truth_laws: Dict[str, sp.Expr] = {} # Ensure type hint

    @abc.abstractmethod # Added
    def generate_trajectory(self, initial_conditions: np.ndarray, t_span: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def add_observation_noise(self, trajectory: np.ndarray) -> np.ndarray:
        if self.noise_level > 0:
            noise = np.random.randn(*trajectory.shape) * self.noise_level
            signal_std = np.std(trajectory, axis=0)
            # Add a small epsilon to signal_std to prevent division by zero or scaling by zero if std is 0
            scaled_noise = noise * (signal_std + 1e-9)
            return trajectory + scaled_noise
        return trajectory

    def get_ground_truth_laws(self) -> Dict[str, sp.Expr]:
        return self.ground_truth_laws

    # Added for PhysicsDiscoveryExperiment compatibility if needed
    def get_ground_truth_conserved_quantities(self, trajectory_data: np.ndarray, variables: List[Any]) -> Dict[str, Any]:
        # This is a placeholder. Each specific environment should implement this
        # to return a dictionary of its ground truth conserved quantities
        # based on the provided trajectory data.
        # Example: {'conserved_energy': true_energy_values, 'conserved_momentum': true_momentum_values}
        # where true_energy_values could be an array matching trajectory_data's length.
        # For now, returning an empty dict or a structure indicating data not processed.
        logging.warning(f"get_ground_truth_conserved_quantities not implemented for {type(self).__name__}. Returning raw data.")
        # Fallback: Try to extract based on known law names if they match columns
        processed_gt = {}
        if self.ground_truth_laws:
            for idx, var_info in enumerate(variables): # Assuming variables match columns
                for law_name_key in self.ground_truth_laws.keys():
                     # This is a heuristic and might not be correct.
                     # Assumes a column in trajectory_data might directly represent a conserved quantity.
                    if var_info.name == law_name_key or f"conserved_{var_info.name}" == law_name_key :
                        if idx < trajectory_data.shape[1]:
                             processed_gt[law_name_key] = trajectory_data[:, idx]
        if not processed_gt: # If no direct match
            processed_gt['raw_gt_data'] = trajectory_data # Or some other indicator
        return processed_gt


class HarmonicOscillatorEnv(PhysicsEnvironment):
    def __init__(self, params: Dict[str, Any], noise_level: float = 0.0):
        super().__init__(params, noise_level)
        self.k = params.get('k', 1.0); self.m = params.get('m', 1.0)
        self.state_vars = ['x', 'v']; x_sym, v_sym = sp.symbols('x v')
        self.ground_truth_laws = {'energy_conservation': 0.5 * self.m * v_sym**2 + 0.5 * self.k * x_sym**2,
                                  'equation_of_motion': -self.k * x_sym / self.m}
    def dynamics(self, state: np.ndarray, t: float) -> np.ndarray:
        x, v = state; return np.array([v, -self.k * x / self.m])
    def generate_trajectory(self, initial_conditions: np.ndarray, t_span: np.ndarray) -> np.ndarray:
        traj = odeint(self.dynamics, initial_conditions, t_span)
        x, v = traj.T; energy = 0.5*self.m*v**2 + 0.5*self.k*x**2
        return self.add_observation_noise(np.column_stack([x, v, energy]))

class PendulumEnv(PhysicsEnvironment):
    def __init__(self, params: Dict[str, Any], noise_level: float = 0.0):
        super().__init__(params, noise_level)
        self.g = params.get('g', 9.81); self.l = params.get('l', 1.0); self.m = params.get('m', 1.0)
        self.small_angle = params.get('small_angle', False); self.state_vars = ['theta', 'omega']
        theta_s, omega_s = sp.symbols('theta omega')
        if self.small_angle:
            self.ground_truth_laws = {'energy_conservation': 0.5*self.m*self.l**2*omega_s**2 + 0.5*self.m*self.g*self.l*theta_s**2,
                                      'equation_of_motion': -self.g*theta_s/self.l}
        else:
            self.ground_truth_laws = {'energy_conservation': 0.5*self.m*self.l**2*omega_s**2 + self.m*self.g*self.l*(1-sp.cos(theta_s)),
                                      'equation_of_motion': -self.g*sp.sin(theta_s)/self.l}
    def dynamics(self, state: np.ndarray, t: float) -> np.ndarray:
        theta, omega = state
        domega_dt = -self.g*theta/self.l if self.small_angle else -self.g*np.sin(theta)/self.l
        return np.array([omega, domega_dt])
    def generate_trajectory(self, initial_conditions: np.ndarray, t_span: np.ndarray) -> np.ndarray:
        traj = odeint(self.dynamics, initial_conditions, t_span)
        theta, omega = traj.T
        energy = (0.5*self.m*self.l**2*omega**2 + 0.5*self.m*self.g*self.l*theta**2) if self.small_angle \
                 else (0.5*self.m*self.l**2*omega**2 + self.m*self.g*self.l*(1-np.cos(theta)))
        return self.add_observation_noise(np.column_stack([theta, omega, energy]))

class KeplerEnv(PhysicsEnvironment):
    def __init__(self, params: Dict[str, Any], noise_level: float = 0.0):
        super().__init__(params, noise_level)
        self.G = params.get('G', 1.0); self.M = params.get('M', 1.0)
        self.state_vars = ['r', 'theta', 'vr', 'vtheta']
        r_s, _, vr_s, vtheta_s = sp.symbols('r theta vr vtheta') # theta_s unused in laws
        self.ground_truth_laws = {'energy_conservation': 0.5*(vr_s**2 + (r_s*vtheta_s)**2) - self.G*self.M/r_s,
                                  'angular_momentum': r_s**2 * vtheta_s,
                                  'equation_of_motion_r': r_s*vtheta_s**2 - self.G*self.M/r_s**2}
    def dynamics(self, state: np.ndarray, t: float) -> np.ndarray:
        r, _, vr, vtheta = state # theta not used in dynamics eqns directly
        dr_dt = vr; dtheta_dt = vtheta
        dvr_dt = r*vtheta**2 - self.G*self.M/r**2
        dvtheta_dt = -2*vr*vtheta/r if r > 1e-6 else 0 # Avoid division by zero if r is tiny
        return np.array([dr_dt, dtheta_dt, dvr_dt, dvtheta_dt])
    def generate_trajectory(self, initial_conditions: np.ndarray, t_span: np.ndarray) -> np.ndarray:
        traj = odeint(self.dynamics, initial_conditions, t_span)
        r, theta, vr, vtheta = traj.T
        energy = 0.5*(vr**2 + (r*vtheta)**2) - self.G*self.M/r
        angular_momentum = r**2 * vtheta
        return self.add_observation_noise(np.column_stack([r, theta, vr, vtheta, energy, angular_momentum]))


class ExperimentRunner:
    def __init__(self,
                 base_dir: str = "./experiments",
                 use_wandb: bool = True,
                 strict_mode: bool = False): # Added strict_mode
        self.base_dir = Path(base_dir); self.base_dir.mkdir(exist_ok=True)
        self.use_wandb = use_wandb
        self.strict_mode = strict_mode # Store strict_mode
        self.env_registry = {'harmonic_oscillator': HarmonicOscillatorEnv, 'pendulum': PendulumEnv, 'kepler': KeplerEnv}
        self.algo_registry: Dict[str, Callable] = {}
        self._register_algorithms()
        self.experiment_plugins: Dict[str, Callable[..., BaseExperiment]] = {}
        self._discover_experiments()

    def _discover_experiments(self):
        logging.info("Discovering 'janus.experiments' plugins...")
        try:
            all_eps = importlib.metadata.entry_points()
            eps = all_eps.select(group='janus.experiments') if hasattr(all_eps, 'select') else all_eps.get('janus.experiments', [])
        except Exception as e:
            logging.warning(f"Could not query entry points: {e}. Manual registration might be needed.")
            eps = []
        if not eps: logging.warning("No 'janus.experiments' plugins found."); return
        for entry_point in eps:
            try:
                loaded_class = entry_point.load()
                if isinstance(loaded_class, type) and issubclass(loaded_class, BaseExperiment):
                    if entry_point.name in self.experiment_plugins: logging.warning(f"Duplicate plugin '{entry_point.name}'. Overwriting.")
                    self.experiment_plugins[entry_point.name] = loaded_class
                    logging.info(f"Registered experiment plugin: '{entry_point.name}'")
                else: logging.warning(f"Plugin '{entry_point.name}' not valid. Skipping.")
            except Exception as e:
                logging.error(f"Failed to load plugin '{entry_point.name}'. Error: {e}", exc_info=True)
                if self.strict_mode: # Strict mode check
                    logging.critical(f"Strict mode: Exiting due to failure in loading plugin '{entry_point.name}'.")
                    sys.exit(1)
        if not self.experiment_plugins: logging.warning("No plugins loaded after processing.")
        else: logging.info(f"Loaded plugins: {list(self.experiment_plugins.keys())}")

    def _register_algorithms(self):
        from symbolic_discovery_env import SymbolicDiscoveryEnv
        from hypothesis_policy_network import HypothesisNet, PPOTrainer
        from progressive_grammar_system import ProgressiveGrammar
        from physics_discovery_extensions import SymbolicRegressor
        def create_janus_full(env_data: np.ndarray, variables: List[Any], exp_config: ExperimentConfig) -> PPOTrainer:
            grammar = ProgressiveGrammar(); janus_cfg = exp_config.janus_config
            sde_params = {'max_depth': janus_cfg.max_depth, 'max_complexity': janus_cfg.max_complexity,
                          'reward_config': janus_cfg.reward_config.model_dump(),
                          'target_variable_index': exp_config.target_variable_index}
            if exp_config.algo_params and 'env_params' in exp_config.algo_params: sde_params.update(exp_config.algo_params['env_params'])
            discovery_env = SymbolicDiscoveryEnv(grammar=grammar, target_data=env_data, variables=variables, **sde_params)
            policy_params = {'hidden_dim': janus_cfg.policy_hidden_dim, 'encoder_type': janus_cfg.policy_encoder_type, 'grammar': grammar}
            if exp_config.algo_params and 'policy_params' in exp_config.algo_params: policy_params.update(exp_config.algo_params['policy_params'])
            policy = HypothesisNet(obs_dim=discovery_env.observation_space.shape[0], act_dim=discovery_env.action_space.n, **policy_params) # Renamed parameters
            return PPOTrainer(policy, discovery_env)
        def create_genetic(env_data: np.ndarray, variables: List[Any], exp_config: ExperimentConfig) -> SymbolicRegressor:
            grammar = ProgressiveGrammar(); janus_cfg = exp_config.janus_config
            reg_params = {'population_size': janus_cfg.genetic_population_size, 'generations': janus_cfg.genetic_generations,
                           'max_complexity': janus_cfg.max_complexity}
            if exp_config.algo_params:
                if 'regressor_params' in exp_config.algo_params: reg_params.update(exp_config.algo_params['regressor_params'])
                else: reg_params.update(exp_config.algo_params) # direct override
            return SymbolicRegressor(grammar=grammar, **reg_params)
        self.algo_registry['janus_full'] = create_janus_full
        self.algo_registry['genetic'] = create_genetic
        self.algo_registry['random'] = lambda _1, _2, _3: None # Ensure consistent signature

    def run_single_experiment(self, config: ExperimentConfig, run_id: int = 0) -> ExperimentResult:
        logging.info(f"Runner: Starting exp '{config.name}' (Run {run_id+1}/{config.n_runs}). Type: '{config.experiment_type}'")
        if not hasattr(config, 'experiment_type') or not config.experiment_type:
            msg = f"Config '{config.name}' missing 'experiment_type'."
            logging.error(msg)
            if self.strict_mode: logging.critical(f"Strict mode: Exiting. {msg}"); sys.exit(1)
            raise InvalidConfigError(msg)
        experiment_class = self.experiment_plugins.get(config.experiment_type)
        if experiment_class is None:
            msg = f"Experiment type '{config.experiment_type}' for '{config.name}' not found. Available: {list(self.experiment_plugins.keys())}"
            logging.error(msg)
            if self.strict_mode: logging.critical(f"Strict mode: Exiting. {msg}"); sys.exit(1)
            raise PluginNotFoundError(msg)
        logging.info(f"Instantiating experiment '{config.experiment_type}' from {experiment_class.__module__}")
        try:
            experiment_instance = experiment_class(config=config, algo_registry=self.algo_registry, env_registry=self.env_registry)
        except Exception as e:
            logging.error(f"Failed to instantiate '{experiment_class.__name__}' for '{config.experiment_type}'. Error: {e}", exc_info=True)
            if self.strict_mode: logging.critical(f"Strict mode: Exiting. Instantiation error for '{config.experiment_type}': {e}"); sys.exit(1)
            raise InvalidConfigError(f"Failed to instantiate '{config.experiment_type}': {e}") from e # Changed to InvalidConfigError for consistency
        result = experiment_instance.execute(run_id=run_id)
        if result is None: # Should not happen if execute() guarantees a result
            logging.error(f"Experiment '{config.name}' execute() returned None.")
            result = ExperimentResult(config=config, run_id=run_id, discovered_law="Critical Error: execute() returned None.")
        logging.info(f"Runner: Completed exp '{config.name}' (Run {run_id+1}). Law: '{result.discovered_law}', Acc: {result.symbolic_accuracy:.4f}")
        return result

    # _run_janus_experiment and _run_genetic_experiment are legacy helpers, assuming plugins handle their logic.
    # If they were still primary execution paths, they'd need similar JanusConfig integration.

    def run_experiment_suite(self, configs: List[ExperimentConfig], parallel: bool = False) -> pd.DataFrame:
        all_results: List[ExperimentResult] = []
        if parallel: logging.warning("Parallel execution not implemented. Running sequentially.")
        for config_item in tqdm(configs, desc="All Experiment Configs"):
            for i in range(config_item.n_runs):
                single_run_result = self.run_single_experiment(config_item, run_id=i)
                all_results.append(single_run_result)
                self._save_result(single_run_result)
        results_df = self._results_to_dataframe(all_results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_df.to_csv(self.base_dir / f"all_results_{timestamp}.csv", index=False)
        logging.info(f"Aggregated results saved to all_results_{timestamp}.csv")
        return results_df

    def _save_result(self, result: ExperimentResult):
        config_hash = result.config.get_hash()
        result_dir = self.base_dir / config_hash
        result_dir.mkdir(parents=True, exist_ok=True)
        pickle_fn = f"run_{result.run_id}_{result.config.name.replace(' ', '_')}.pkl" # Sanitize name
        with open(result_dir / pickle_fn, 'wb') as f: pickle.dump(result, f)
        summary_fn = f"summary_run_{result.run_id}_{result.config.name.replace(' ', '_')}.json"
        summary_data = {'config_name': result.config.name, 'run_id': result.run_id,
                        'discovered_law': result.discovered_law, 'symbolic_accuracy': result.symbolic_accuracy,
                        'predictive_mse': result.predictive_mse, 'wall_time_seconds': result.wall_time_seconds,
                        'config_hash': config_hash}
        with open(result_dir / summary_fn, 'w') as f: json.dump(summary_data, f, indent=2)
        logging.debug(f"Saved result for '{result.config.name}' run {result.run_id} to {result_dir}")

    def _results_to_dataframe(self, results: List[ExperimentResult]) -> pd.DataFrame:
        rows = [{'experiment_name': res.config.name, 'algorithm': res.config.algorithm,
                 'environment_type': res.config.environment_type, 'noise_level': res.config.noise_level,
                 'run_id': res.run_id, 'symbolic_accuracy': res.symbolic_accuracy,
                 'predictive_mse': res.predictive_mse, 'law_complexity': res.law_complexity,
                 'n_experiments_to_convergence': res.n_experiments_to_convergence,
                 'wall_time_seconds': res.wall_time_seconds, 'discovered_law': res.discovered_law,
                 'config_hash': res.config.get_hash()} for res in results]
        return pd.DataFrame(rows)

    def analyze_results(self, df: pd.DataFrame): # Simplified, full analysis in ExperimentVisualizer
        analysis_dir = self.base_dir / "analysis"; analysis_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Basic analysis summary. Full visualizations in {analysis_dir} if visualizer is run.")
        # Key metrics summary
        summary_stats = df.groupby(['algorithm', 'environment_type']).agg(
            mean_accuracy=('symbolic_accuracy', 'mean'),
            mean_mse=('predictive_mse', 'mean'),
            mean_time=('wall_time_seconds', 'mean')
        ).round(3)
        print("\n=== Summary Statistics (Mean Performance) ===")
        print(summary_stats)
        summary_stats.to_csv(analysis_dir / 'summary_performance_statistics.csv')


def run_phase1_validation(strict_mode_override: bool = False): # Added strict_mode_override
    logging.basicConfig(level=logging.INFO)
    runner = ExperimentRunner(base_dir="./experiments_phase1", strict_mode=strict_mode_override) # Pass strict_mode
    configs: List[ExperimentConfig] = []
    algorithms = ['janus_full', 'genetic']
    environments = ['harmonic_oscillator', 'pendulum']
    for env_type_str in environments:
        for algo_str in algorithms:
            env_specific_params_dict = {}
            if env_type_str == 'pendulum': env_specific_params_dict = {'g': 9.81, 'l': 1.0, 'm': 1.0, 'small_angle': True}
            elif env_type_str == 'harmonic_oscillator': env_specific_params_dict = {'k': 1.0, 'm': 1.0}
            janus_cfg_instance = JanusConfig(
                target_phenomena=env_type_str, env_specific_params=env_specific_params_dict,
                synthetic_data_params=SyntheticDataParamsConfig(noise_level=0.0, n_samples=20, time_range=[0,10]),
                num_evaluation_cycles=100, policy_hidden_dim=128, genetic_population_size=50,
            )
            exp_config = ExperimentConfig.from_janus_config(
                name=f"{env_type_str}_rediscovery_{algo_str}", experiment_type='physics_discovery_example',
                janus_config=janus_cfg_instance, algorithm_name=algo_str, n_runs=2, seed=42
            )
            configs.append(exp_config)
    if not configs: logging.warning("Phase 1: No configs."); return None
    if not runner.experiment_plugins.get('physics_discovery_example'):
        logging.error("Phase 1: 'physics_discovery_example' plugin not found."); return None
    results_df = runner.run_experiment_suite(configs)
    if results_df is not None and not results_df.empty: runner.analyze_results(results_df)
    else: logging.warning("Phase 1: No results.")
    return results_df

def run_phase2_robustness(strict_mode_override: bool = False): # Added strict_mode_override
    logging.basicConfig(level=logging.INFO)
    runner = ExperimentRunner(base_dir="./experiments_phase2", strict_mode=strict_mode_override) # Pass strict_mode
    configs: List[ExperimentConfig] = []
    noise_levels_list = [0.0, 0.01, 0.05]; algorithms_list = ['janus_full', 'genetic']
    env_type_for_robustness = 'harmonic_oscillator'
    for noise_val in noise_levels_list:
        for algo_name_str in algorithms_list:
            janus_cfg_instance_robust = JanusConfig(
                target_phenomena=env_type_for_robustness, env_specific_params={'k': 1.0, 'm': 1.0},
                synthetic_data_params=SyntheticDataParamsConfig(noise_level=noise_val, n_samples=20, time_range=[0,10]),
                num_evaluation_cycles=100,
            )
            exp_config_robust = ExperimentConfig.from_janus_config(
                name=f"{env_type_for_robustness}_noise_{noise_val*100:.0f}pct_{algo_name_str}",
                experiment_type='physics_discovery_example', janus_config=janus_cfg_instance_robust,
                algorithm_name=algo_name_str, n_runs=2, seed=42
            )
            configs.append(exp_config_robust)
    if not configs: logging.warning("Phase 2: No configs."); return None
    if not runner.experiment_plugins.get('physics_discovery_example'):
        logging.error("Phase 2: 'physics_discovery_example' plugin not found."); return None
    results_df = runner.run_experiment_suite(configs)
    if results_df is not None and not results_df.empty: runner.analyze_results(results_df)
    else: logging.warning("Phase 2: No results.")
    return results_df

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(levelname)s-[%(module)s:%(funcName)s:%(lineno)d]-%(message)s')
    print("ExperimentRunner __main__ Test Section"); print("="*30)
    test_janus_config = JanusConfig(
        target_phenomena='harmonic_oscillator', env_specific_params={'k': 1.0, 'm': 1.0},
        synthetic_data_params=SyntheticDataParamsConfig(noise_level=0.01, n_samples=10, time_range=[0,5]),
        num_evaluation_cycles=50, genetic_population_size=50, genetic_generations=20, max_complexity=8,
    )
    example_exp_config = ExperimentConfig.from_janus_config(
        name="main_test_harmonic_genetic", experiment_type='physics_discovery_example',
        janus_config=test_janus_config, algorithm_name='genetic', n_runs=1, seed=123
    )
    main_strict_mode = os.getenv("JANUS_STRICT_MODE", "false").lower() == "true"
    runner_main = ExperimentRunner(use_wandb=False, base_dir="./experiments_main_test", strict_mode=main_strict_mode)
    if example_exp_config.experiment_type not in runner_main.experiment_plugins:
        logging.error(f"__main__: Plugin '{example_exp_config.experiment_type}' not found. Check registration & installation.")
        logging.error(f"Available: {list(runner_main.experiment_plugins.keys())}")
    else:
        logging.info(f"__main__: Running '{example_exp_config.name}'...")
        try:
            result_main_test = runner_main.run_single_experiment(example_exp_config, run_id=0)
            if result_main_test: print(f"\n--- Result: {result_main_test.discovered_law}, Acc: {result_main_test.symbolic_accuracy:.4f} ---")
            else: print(f"__main__: Experiment '{example_exp_config.name}' no result.")
        except Exception as e_main: logging.error(f"__main__: Critical error: {e_main}", exc_info=True)
    print("\n__main__ Test Section Finished.")

# PhysicsDiscoveryExperiment and other classes (HarmonicOscillatorEnv, etc.) remain largely unchanged
# as their parameterization is now handled via JanusConfig passed through ExperimentConfig.
# The key changes are in ExperimentRunner and how ExperimentConfig is created and used.

class PhysicsDiscoveryExperiment(BaseExperiment):
    def __init__(self, config: ExperimentConfig, algo_registry: Dict[str, Callable], env_registry: Dict[str, Callable]):
        super().__init__()
        self.config = config
        self.janus_config = config.janus_config
        self.algo_registry = algo_registry
        self.env_registry = env_registry
        results_base_path = Path(self.janus_config.results_dir)
        experiment_instance_name = f"{self.config.name}_seed{self.config.seed}_{int(time.time())}"
        base_output_dir = results_base_path / experiment_instance_name
        self.tracker_save_dir = base_output_dir / "hypothesis_tracking"
        self.log_file_path = base_output_dir / "training_logs.jsonl"
        self.tracker_save_dir.mkdir(parents=True, exist_ok=True)
        self.conservation_reward = ConservationBiasedReward(
            conservation_types=self.janus_config.conservation_types,
            weight_factor=self.janus_config.conservation_weight_factor
        )
        self.symmetry_detector = PhysicsSymmetryDetector(
            tolerance=self.janus_config.symmetry_tolerance,
            confidence_threshold=self.janus_config.symmetry_confidence_threshold
        )
        self.hypothesis_tracker = HypothesisTracker(
            save_directory=str(self.tracker_save_dir),
            autosave_interval=self.janus_config.tracker_autosave_interval
        )
        self.training_integration = JanusTrainingIntegration(self.hypothesis_tracker)
        self.training_logger = TrainingLogger(
            backends=self.janus_config.logger_backends, log_file_path=str(self.log_file_path),
            redis_host=self.janus_config.redis_host, redis_port=self.janus_config.redis_port,
            redis_channel=f"{self.janus_config.redis_channel_base}_{self.config.name}_s{self.config.seed}"
        )
        monitor_data_source = "redis" if "redis" in self.janus_config.logger_backends else "file"
        self.live_monitor = LiveMonitor(
            data_source=monitor_data_source, log_file_path=str(self.log_file_path),
            redis_host=self.janus_config.redis_host, redis_port=self.janus_config.redis_port,
            redis_channel=f"{self.janus_config.redis_channel_base}_{self.config.name}_s{self.config.seed}"
        )
        self.physics_env: Optional[PhysicsEnvironment] = None
        self.env_data: Optional[np.ndarray] = None
        self.ground_truth_trajectory_processed: Optional[Dict[str,Any]] = None
        self.variables: Optional[List[Any]] = None
        self.sympy_vars: Optional[List[sp.Symbol]] = None
        self.algorithm: Optional[Any] = None
        self.ground_truth_laws: Optional[Dict[str, sp.Expr]] = None
        self.experiment_result: Optional[ExperimentResult] = None
        self._start_time: float = 0.0

        if _EMERGENT_MONITOR_AVAILABLE:
            self.emergent_tracker = EmergentBehaviorTracker(save_dir=str(self.tracker_save_dir / "emergent_analysis"))
        else:
            # Use the dummy version if the real one couldn't be imported
            self.emergent_tracker = EmergentBehaviorTracker(save_dir=str(self.tracker_save_dir / "emergent_analysis"))
        self._last_logged_phase_idx_to_monitor = 0

    def _flush_new_phase_transitions_to_live_monitor(self):
        if not hasattr(self, 'emergent_tracker') or not _EMERGENT_MONITOR_AVAILABLE:
            return
        if not hasattr(self, 'live_monitor') or not self.live_monitor:
            return

        # Ensure phase_detector exists, even if it's a dummy
        if not hasattr(self.emergent_tracker, 'phase_detector'):
            return

        all_transitions = self.emergent_tracker.phase_detector.phase_transitions
        new_transitions_to_log = all_transitions[self._last_logged_phase_idx_to_monitor:]

        for transition_event in new_transitions_to_log:
            # log_phase_transition expects Dict[str, Any] with 'timestamp' and 'type'
            # transition_event is typically {'timestamp': ..., 'type': ..., 'metrics': ...}
            # Forwarding the whole event is fine as log_phase_transition uses .get()
            self.live_monitor.log_phase_transition(transition_event)

        self._last_logged_phase_idx_to_monitor = len(all_transitions)

    def setup(self):
        logging.info(f"[{self.config.name}] Setup: Seed {self.config.seed}, Env '{self.config.environment_type}', Algo '{self.config.algorithm}'.")
        np.random.seed(self.config.seed); torch.manual_seed(self.config.seed)
        env_class = self.env_registry.get(self.config.environment_type)
        if not env_class: raise ValueError(f"Env type '{self.config.environment_type}' not in registry.")
        self.physics_env = env_class(self.janus_config.env_specific_params, self.config.noise_level)
        trajectories = []
        for _ in range(self.config.n_trajectories):
            init_cond = self._get_initial_conditions()
            t_span = np.arange(0, self.config.trajectory_length * self.config.sampling_rate, self.config.sampling_rate)
            if self.physics_env: trajectory = self.physics_env.generate_trajectory(init_cond, t_span); trajectories.append(trajectory)
        if not trajectories: raise ValueError("No trajectories generated.")
        self.env_data = np.vstack(trajectories)
        logging.info(f"[{self.config.name}] Data shape {self.env_data.shape}.")
        try: from progressive_grammar_system import Variable
        except ImportError: logging.error("Failed to import 'Variable'"); raise
        if self.physics_env and self.physics_env.state_vars:
            self.variables = [Variable(name, idx, {}) for idx, name in enumerate(self.physics_env.state_vars)]
        else:
            self.variables = [Variable(f'var{idx}', idx, {}) for idx in range(self.env_data.shape[1])]
            logging.warning(f"[{self.config.name}] Using generic variable names.")
        algo_factory = self.algo_registry.get(self.config.algorithm)
        if not algo_factory: raise ValueError(f"Algo '{self.config.algorithm}' not in registry.")
        self.algorithm = algo_factory(self.env_data, self.variables, self.config)
        if self.physics_env: self.ground_truth_laws = self.physics_env.get_ground_truth_laws()
        if self.physics_env and hasattr(self.physics_env, 'get_ground_truth_conserved_quantities'):
            self.ground_truth_trajectory_processed = self.physics_env.get_ground_truth_conserved_quantities(self.env_data, self.variables)
        else: self.ground_truth_trajectory_processed = {'raw_data': self.env_data}
        if self.variables: self.sympy_vars = [symbols(v.name) for v in self.variables]
        logging.info(f"[{self.config.name}] Setup complete.")
        self.live_monitor.start_monitoring() # Start monitor after setup
        logging.info(f"[{self.config.name}] Live monitor started.")


    def _get_initial_conditions(self) -> np.ndarray:
        env_type = self.config.environment_type
        env_specific_params = self.janus_config.env_specific_params
        if env_type == 'harmonic_oscillator':
            return np.random.randn(2) * np.array([env_specific_params.get('x_scale', 1.0), env_specific_params.get('v_scale', 2.0)])
        elif env_type == 'pendulum':
            max_angle = np.pi / 2 if env_specific_params.get('small_angle', False) else np.pi
            return np.array([np.random.uniform(-abs(max_angle), abs(max_angle)), np.random.uniform(-1.0, 1.0)])
        elif env_type == 'kepler':
            ecc = np.random.uniform(0.0, 0.7); sma = np.random.uniform(0.5, 2.0); r_periapsis = sma * (1 - ecc)
            G = env_specific_params.get('G', 1.0); M = env_specific_params.get('M', 1.0)
            v_periapsis_tangential = np.sqrt(G * M * (2/r_periapsis - 1/sma)) if (G*M > 0 and r_periapsis > 0 and sma > 0) else 1.0
            return np.array([r_periapsis, 0.0, 0.0, v_periapsis_tangential / r_periapsis if r_periapsis > 0 else 1.0])
        else:
            num_state_vars = len(self.physics_env.state_vars) if self.physics_env and self.physics_env.state_vars else 2
            return np.random.rand(num_state_vars) * 2 - 1

    def _remap_expression(self, expr_str: str, var_mapping: Dict[str, str]) -> str:
        """Remaps an expression string from temporary variable names to original variable names."""
        result = expr_str
        # Sort variables by length of the key (e.g., 'x10' before 'x1') to prevent partial replacements.
        sorted_vars = sorted(var_mapping.items(), key=lambda item: len(item[0]), reverse=True)
        for new_var, orig_var in sorted_vars:
            # This simple replacement should be fine for 'x0', 'x1', etc. as they are distinct tokens.
            # If original variable names could be substrings of temp names or vice-versa in complex ways,
            # or if temp names are not distinct tokens (e.g. 'x' and 'x_temp'),
            # a more robust regex-based replacement (e.g. using word boundaries) would be needed.
            # For now, assuming 'x0', 'x1' are safe.
            result = result.replace(new_var, orig_var)
        return result

    def run(self, run_id: int) -> ExperimentResult:
        current_run_result = ExperimentResult(config=self.config, run_id=run_id)
        if self.env_data is not None: current_run_result.trajectory_data = self.env_data
        if self.algorithm is None:
            current_run_result.discovered_law = "Error: Algo not init"; return current_run_result

        algo_name = self.config.algorithm; janus_cfg = self.janus_config
        try:
            if algo_name.startswith('janus'):
                trainer = self.algorithm; num_evals = janus_cfg.num_evaluation_cycles; steps_per_eval = janus_cfg.timesteps_per_eval_cycle
                ppo_params = {'rollout_length': janus_cfg.ppo_rollout_length, 'n_epochs': janus_cfg.ppo_n_epochs,
                              'batch_size': janus_cfg.ppo_batch_size, 'learning_rate': janus_cfg.ppo_learning_rate,
                              'gamma': janus_cfg.ppo_gamma, 'gae_lambda': janus_cfg.ppo_gae_lambda}
                if self.config.algo_params and 'ppo_train_params' in self.config.algo_params:
                    ppo_params.update(self.config.algo_params['ppo_train_params'])
                best_mse_overall, best_expr_overall, efficiency_curve, current_total_steps = float('inf'), None, [], 0
                for i in range(num_evals):
                    trainer.train(total_timesteps=steps_per_eval, **ppo_params); current_total_steps += steps_per_eval
                    if hasattr(trainer, 'env') and hasattr(trainer.env, '_evaluation_cache') and self.training_integration:
                        for eval_entry in trainer.env._evaluation_cache:
                            hyp_data = eval_entry.get('expression_obj', {'expression_str': eval_entry.get('expression')})
                            eval_results_tracker = {'performance_score': -eval_entry.get('mse', float('inf')),
                                                    'conservation_score': eval_entry.get('conservation_score', 0.0),
                                                    'symmetry_score': eval_entry.get('symmetry_score', 0.0),
                                                    'trajectory_fit': eval_entry.get('mse', float('inf')),
                                                    'functional_form': str(eval_entry.get('expression'))}
                            self.training_integration.on_hypothesis_evaluated(hyp_data, eval_results_tracker, current_total_steps, i)
                        if hasattr(trainer.env, 'clear_evaluation_cache'): trainer.env.clear_evaluation_cache()

                    current_best_mse_for_log = float('inf') # For logging efficiency curve
                    if self.training_integration:
                        best_hyp = self.training_integration.get_best_discovered_law(criterion='overall')
                        if best_hyp:
                            hyp_data = best_hyp['hypothesis_data']
                            eval_res = best_hyp['evaluation_results']
                            current_best_mse_for_log = eval_res.get('trajectory_fit', float('inf')) # Update for efficiency curve

                            if _EMERGENT_MONITOR_AVAILABLE and hasattr(self, 'emergent_tracker'):
                                current_mse = eval_res.get('trajectory_fit', eval_res.get('mse', float('inf')))
                                discovery_info = {
                                    'complexity': eval_res.get('complexity', len(str(hyp_data.get('expression_str', hyp_data)))),
                                    'mse': current_mse,
                                    'accuracy': 1.0 / (1.0 + current_mse) if current_mse != float('inf') and current_mse >= 0 else 0.0,
                                    'novelty': eval_res.get('novelty_score', 0.0) # If available
                                    # Any other metrics from eval_res can be added here
                                }
                                self.emergent_tracker.log_discovery(
                                    expression=str(hyp_data.get('expression_str', hyp_data)),
                                    info=discovery_info,
                                    discovery_path=[] # Placeholder for discovery path
                                )
                        # Update best_expr_overall and best_mse_overall for efficiency curve logging
                        # This part was slightly adjusted to ensure these variables are updated for the curve
                        if best_hyp:
                             best_expr_overall = best_hyp['hypothesis_data'] # Already assigned above, but make sure it's in scope
                             best_mse_overall = current_best_mse_for_log       # Use the mse from the current best hypothesis

                    efficiency_curve.append((current_total_steps, best_mse_overall if 'best_mse_overall' in locals() else current_best_mse_for_log))
                    self._flush_new_phase_transitions_to_live_monitor() # Call flush at the end of each eval cycle

                if self.training_integration:
                    final_best_hyp = self.training_integration.get_best_discovered_law(criterion='overall')
                    if final_best_hyp:
                        hyp_data_final = final_best_hyp['hypothesis_data']
                        current_run_result.discovered_law = str(hyp_data_final.get('expression_str', hyp_data_final))
                        eval_res_final = final_best_hyp['evaluation_results']
                        current_run_result.predictive_mse = eval_res_final.get('trajectory_fit', float('inf'))
                        current_run_result.law_complexity = eval_res_final.get('complexity', len(str(current_run_result.discovered_law)))
                        current_run_result.component_metrics['conservation_score'] = eval_res_final.get('conservation_score',0.0) # Ensure default
                        current_run_result.component_metrics['symmetry_score'] = eval_res_final.get('symmetry_score',0.0) # Ensure default
                current_run_result.sample_efficiency_curve = efficiency_curve
                current_run_result.n_experiments_to_convergence = num_evals
            elif algo_name == 'genetic':
                regressor = self.algorithm
                target_idx = self.config.target_variable_index if self.config.target_variable_index is not None else -1

                if self.env_data is None: raise ValueError("env_data is None for genetic")

                n_vars_original = self.env_data.shape[1]

                y = self.env_data[:, target_idx]
                X = np.delete(self.env_data, target_idx, axis=1)

                variable_indices: List[int] = []
                if target_idx == -1: # Target is the last column implicitly
                    variable_indices = list(range(n_vars_original - 1))
                else:
                    for i in range(n_vars_original):
                        if i < target_idx:
                            variable_indices.append(i)
                        elif i > target_idx:
                            variable_indices.append(i - 1) # Adjust index for the new X matrix

                original_var_names = [f"x{i}" for i in range(n_vars_original)]
                var_mapping: Dict[str, str] = {}
                # Corrected original_var_names to use actual variable names if available
                # This assumes self.variables contains Variable objects with a 'name' attribute
                # and corresponds to the original columns of self.env_data
                if self.variables and len(self.variables) == n_vars_original:
                    original_var_names = [v.name for v in self.variables]

                for new_idx, orig_idx_in_full_data in enumerate(variable_indices):
                     # Map from new temporary name (e.g., "x0") to original name (e.g., "pos")
                     # The 'orig_idx_in_full_data' is the index in the original self.env_data / self.variables
                     # The 'new_idx' is the index in the X array.
                    var_mapping[f"x{new_idx}"] = original_var_names[orig_idx_in_full_data]

                fit_params = {'generations': janus_cfg.genetic_generations,
                              'population_size': janus_cfg.genetic_population_size,
                              'max_complexity': janus_cfg.max_complexity}
                if self.config.algo_params and 'fit_params' in self.config.algo_params:
                    fit_params.update(self.config.algo_params['fit_params'])

                # Assuming regressor.fit will be updated to accept var_mapping
                # and will not need self.variables directly if var_mapping is provided.
                best_expr_obj = regressor.fit(X, y, var_mapping=var_mapping, **fit_params)

                if best_expr_obj and hasattr(best_expr_obj, 'symbolic'):
                    raw_expr = str(best_expr_obj.symbolic)
                    # Assuming self._remap_expression will be implemented in this class
                    current_run_result.discovered_law = self._remap_expression(raw_expr, var_mapping)
                    current_run_result.law_complexity = getattr(best_expr_obj, 'complexity', len(raw_expr))
                    if hasattr(best_expr_obj, 'mse') and best_expr_obj.mse is not None:
                        current_run_result.predictive_mse = best_expr_obj.mse
                    elif hasattr(regressor, 'predict'):
                        predictions = regressor.predict(X) # Predict might also need var_mapping or remapped features
                        current_run_result.predictive_mse = np.mean((predictions - y)**2)
                else:
                    current_run_result.discovered_law = "Error: Genetic failed."
                    current_run_result.predictive_mse = float('inf')

                if _EMERGENT_MONITOR_AVAILABLE and hasattr(self, 'emergent_tracker') and \
                   current_run_result.discovered_law and not current_run_result.discovered_law.startswith("Error:"):
                    current_mse_ga = current_run_result.predictive_mse
                    info_dict = {
                        'complexity': current_run_result.law_complexity,
                        'mse': current_mse_ga,
                        'accuracy': 1.0 / (1.0 + current_mse_ga) if current_mse_ga != float('inf') and current_mse_ga >= 0 else 0.0
                        # Add novelty if available/calculable for GA
                    }
                    self.emergent_tracker.log_discovery(
                        expression=current_run_result.discovered_law,
                        info=info_dict,
                        discovery_path=[] # Placeholder
                    )
                self._flush_new_phase_transitions_to_live_monitor() # Call flush after GA discovery attempt

                current_run_result.n_experiments_to_convergence = getattr(regressor, 'generations', 1)
            elif algo_name == 'random':
                num_random_expr = getattr(janus_cfg, 'random_search_iterations', janus_cfg.num_evaluation_cycles)
                # Simplified random search logic for brevity
                # No specific log_discovery for random search for now, unless meaningful metrics are generated
                current_run_result.discovered_law = "random_placeholder"; current_run_result.predictive_mse = np.random.rand()*5
                current_run_result.n_experiments_to_convergence = num_random_expr
                self._flush_new_phase_transitions_to_live_monitor() # Call flush after random search loop
            else: current_run_result.discovered_law = f"Error: Unknown algo '{algo_name}'"

            if self.ground_truth_laws and current_run_result.discovered_law and not current_run_result.discovered_law.startswith("Error:"):
                current_run_result.symbolic_accuracy = calculate_symbolic_accuracy(current_run_result.discovered_law, self.ground_truth_laws)
        except Exception as e:
            logging.error(f"[{self.config.name}] Algo exec error ('{algo_name}'): {e}", exc_info=True)
            current_run_result.discovered_law = f"Error: {str(e)[:100]}"
        logging.info(f"[{self.config.name}] Run done. Law: '{current_run_result.discovered_law}', Acc: {current_run_result.symbolic_accuracy:.4f}")
        return current_run_result

    def teardown(self):
        logging.info(f"[{self.config.name}] Tearing down experiment.")
        self.physics_env = None; self.env_data = None; self.variables = None
        self.algorithm = None; self.ground_truth_laws = None
        logging.info(f"[{self.config.name}] Teardown complete.")

    def execute(self, run_id: int = 0) -> ExperimentResult:
        self._start_time = time.time()
        self.experiment_result = ExperimentResult(config=self.config, run_id=run_id)
        try:
            logging.info(f"[{self.config.name}] EXECUTE: Setup (run {run_id})...")
            self.setup()
            if self.env_data is not None: self.experiment_result.trajectory_data = self.env_data
            logging.info(f"[{self.config.name}] EXECUTE: Run (run {run_id})...")
            run_specific_result = self.run(run_id=run_id)
            if run_specific_result: # Merge results
                self.experiment_result.discovered_law = run_specific_result.discovered_law
                self.experiment_result.symbolic_accuracy = run_specific_result.symbolic_accuracy
                self.experiment_result.predictive_mse = run_specific_result.predictive_mse
                self.experiment_result.law_complexity = run_specific_result.law_complexity
                self.experiment_result.n_experiments_to_convergence = run_specific_result.n_experiments_to_convergence
                self.experiment_result.sample_efficiency_curve = run_specific_result.sample_efficiency_curve
                self.experiment_result.component_metrics = run_specific_result.component_metrics
                if run_specific_result.trajectory_data is not None: self.experiment_result.trajectory_data = run_specific_result.trajectory_data
            else: self.experiment_result.discovered_law = "Error: run() returned None"
            logging.info(f"[{self.config.name}] EXECUTE: Run complete. Law: '{self.experiment_result.discovered_law}'")
        except Exception as e:
            logging.error(f"[{self.config.name}] EXECUTE: Exception (run {run_id}): {e}", exc_info=True)
            if self.experiment_result: self.experiment_result.discovered_law = f"Critical Error: {str(e)[:150]}"
            if self.config.janus_config.strict_mode: # Check strict_mode from JanusConfig
                logging.critical(f"Strict mode: Exiting due to critical error during execute: {e}")
                sys.exit(1)
            # Not re-raising if not strict mode, allowing teardown
        finally:
            if self.experiment_result: self.experiment_result.wall_time_seconds = time.time() - self._start_time
            logging.info(f"[{self.config.name}] EXECUTE: Teardown (run {run_id})...")
            self.cleanup(); self.teardown()
            logging.info(f"[{self.config.name}] EXECUTE: Teardown complete (run {run_id}).")
        return self.experiment_result if self.experiment_result else ExperimentResult(config=self.config, run_id=run_id, discovered_law="Critical error: result not formed.")

    def get_best_discovered_law(self) -> Optional[Dict[str, Any]]:
        if not self.training_integration: return None
        return self.training_integration.get_best_discovered_law(criterion='overall')

    def get_training_statistics(self) -> Optional[Dict[str, Any]]:
        if not self.hypothesis_tracker: return None
        return self.hypothesis_tracker.get_training_statistics()

    def export_results(self, output_file_base: str = "experiment_export"):
        if not self.hypothesis_tracker: return
        export_dir = Path(self.hypothesis_tracker.save_directory) / "exports"
        export_dir.mkdir(parents=True, exist_ok=True)
        self.hypothesis_tracker.export_best_hypotheses(str(export_dir / f"{output_file_base}_best_overall.json"), criterion='overall', n=10, format='json')
        self.hypothesis_tracker.export_best_hypotheses(str(export_dir / f"{output_file_base}_best_conservation.json"), criterion='conservation', n=10, format='json')
        logging.info(f"[{self.config.name}] Results exported to {export_dir}")

    def cleanup(self):
        logging.info(f"[{self.config.name}] Performing cleanup...")
        if self.hypothesis_tracker and hasattr(self.hypothesis_tracker, 'save_state'): self.hypothesis_tracker.save_state()
        if self.live_monitor and hasattr(self.live_monitor, 'stop_monitoring'): self.live_monitor.stop_monitoring()
        if self.training_logger and hasattr(self.training_logger, 'close'): self.training_logger.close()
        logging.info(f"[{self.config.name}] Cleanup done.")
