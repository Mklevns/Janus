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

from custom_exceptions import MissingDependencyError, PluginNotFoundError, InvalidConfigError, DataGenerationError
from base_experiment import BaseExperiment # Added
from math_utils import calculate_symbolic_accuracy
from robust_hypothesis_extraction import HypothesisTracker, JanusTrainingIntegration
from conservation_reward_fix import ConservationBiasedReward
from symmetry_detection_fix import PhysicsSymmetryDetector
from live_monitor import TrainingLogger, LiveMonitor # Assuming live_monitor.py now has these
from sympy import lambdify, symbols # For converting sympy expr to callable
from progressive_grammar_system import Expression as SymbolicExpression # Alias to avoid clash if 'Expression' is used elsewhere
# Pathlib is already imported.

@dataclass
class ExperimentConfig:
    """Encapsulates all settings for a single experimental run.

    This class defines the parameters for an experiment, including the type of
    environment, the algorithm to be used, data collection settings, and
    parameters for reproducibility.

    Attributes:
        name: A human-readable name for the experiment.
        experiment_type: Specifies the experiment plugin to use (e.g.,
            'physics_discovery_example').
        environment_type: The type of physics environment to simulate (e.g.,
            'harmonic_oscillator', 'pendulum').
        algorithm: The discovery algorithm to employ (e.g., 'janus_full',
            'genetic').
        env_params: A dictionary of parameters specific to the chosen
            environment (e.g., spring constant 'k' for a harmonic oscillator).
        noise_level: The standard deviation of Gaussian noise added to
            observations.
        hidden_confounders: If True, simulates scenarios with unobserved
            variables that might affect the system's dynamics.
        algo_params: A dictionary of parameters specific to the chosen
            algorithm (e.g., learning rate for a neural network).
        max_experiments: The maximum number of discovery attempts or iterations
            the algorithm is allowed.
        max_time_seconds: The maximum wall-clock time allocated for the
            experiment.
        n_trajectories: The number of distinct trajectories to generate from the
            environment.
        trajectory_length: The number of time steps in each generated
            trajectory.
        sampling_rate: The time interval between consecutive data points in a
            trajectory.
        seed: The random seed used for all stochastic processes to ensure
            reproducibility.
        n_runs: The number of independent times this experiment configuration
            will be run to gather statistical data.
        target_variable_index: Optional integer index specifying which variable
            in the dataset is the target for discovery. If None, a default
            may be assumed by the algorithm or environment.
    """
    name: str
    experiment_type: str
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
        """Generates a unique hash for this experiment configuration.

        The hash is computed based on a sorted JSON representation of the
        configuration, ensuring that identical configurations produce the same
        hash. This is useful for organizing and retrieving experiment results.

        Returns:
            A string representing the MD5 hash of the configuration, truncated
            to 8 characters.
        """
        config_str = json.dumps(asdict(self), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]


@dataclass
class ExperimentResult:
    """Stores the outcomes and data from a single execution of an experiment.

    This class aggregates all relevant information produced during one run of
    an experiment defined by an `ExperimentConfig`.

    Attributes:
        config: The `ExperimentConfig` object that defined this experiment run.
        run_id: An integer identifying this specific run, especially if the same
            config is run multiple times (part of `n_runs`).
        discovered_law: A string representation of the symbolic law or equation
            discovered by the algorithm. None if no law was found.
        symbolic_accuracy: A float score (typically between 0.0 and 1.0)
            indicating the similarity of the discovered law to the ground truth
            law, if known.
        predictive_mse: The mean squared error of the discovered law's
            predictions on a test dataset or the training data.
        law_complexity: An integer measure of the complexity of the discovered
            law (e.g., number of nodes in its symbolic tree).
        n_experiments_to_convergence: The number of iterations, evaluations, or
            discovery attempts the algorithm took to arrive at the final law.
        wall_time_seconds: The total wall-clock time taken to complete the
            experiment run.
        sample_efficiency_curve: A list of tuples, where each tuple (samples,
            metric) tracks the performance metric (e.g., MSE) against the
            number of samples or experiments used.
        noise_resilience_score: A score indicating how well the discovery
            process performed in the presence of noise.
        generalization_score: A score indicating how well the discovered law
            performs on unseen data or conditions.
        component_metrics: A dictionary storing metrics related to specific
            components or ablation studies within the algorithm.
        trajectory_data: A NumPy array containing the raw trajectory data
            generated or used by the experiment. May be None if data is very
            large and not stored by default.
        experiment_log: A list of dictionaries or structured log entries
            capturing key events or intermediate results during the experiment.
    """
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
        """Converts the ExperimentResult to a dictionary for serialization.

        This method is primarily used for saving the results to formats like
        JSON or Pickle. It handles the conversion of NumPy arrays in
        `trajectory_data` to lists to ensure JSON compatibility.

        Returns:
            A dictionary representation of the ExperimentResult instance.
        """
        result = asdict(self)
        # Convert numpy arrays to lists for JSON serialization
        if self.trajectory_data is not None:
            result['trajectory_data'] = self.trajectory_data.tolist()
        return result


class PhysicsEnvironment:
    """Abstract base class for defining physics simulation environments.

    This class provides a common interface for various physical systems that
    can be used in discovery experiments. Subclasses must implement methods
    for generating trajectories and defining ground truth laws.

    Attributes:
        params (Dict[str, Any]): A dictionary of parameters that define the
            specific instance of the physical system (e.g., mass, spring
            constant).
        noise_level (float): The level of Gaussian noise to be added to
            observations. A value of 0.0 means no noise.
        state_vars (List[str]): A list of strings representing the names of
            the state variables in the system (e.g., ['x', 'v']).
        ground_truth_laws (Dict[str, sp.Expr]): A dictionary where keys are
            names of physical laws (e.g., 'energy_conservation') and values
            are their SymPy expression representations.
    """

    def __init__(self, params: Dict[str, Any], noise_level: float = 0.0):
        """Initializes the physics environment.

        Args:
            params: A dictionary of parameters defining the physical system.
            noise_level: The standard deviation of Gaussian noise to add to
                observations. Defaults to 0.0 (no noise).
        """
        self.params = params
        self.noise_level = noise_level
        self.state_vars = []
        self.ground_truth_laws = {}

    def generate_trajectory(self,
                          initial_conditions: np.ndarray,
                          t_span: np.ndarray) -> np.ndarray:
        """Generates a single trajectory of the system.

        This method must be implemented by subclasses to simulate the
        system's evolution over time from a given set of initial conditions.

        Args:
            initial_conditions: A NumPy array representing the starting state
                of the system.
            t_span: A NumPy array of time points at which to record the
                system's state.

        Returns:
            A NumPy array where each row is the state of the system at a
            corresponding time point in `t_span`.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError

    def add_observation_noise(self, trajectory: np.ndarray) -> np.ndarray:
        """Adds realistic observation noise to a generated trajectory.

        The noise added is Gaussian, scaled by the `noise_level` attribute
        and the standard deviation of each signal component.

        Args:
            trajectory: A NumPy array representing the clean trajectory data.

        Returns:
            A NumPy array of the same shape as `trajectory`, but with added
            noise. Returns the original trajectory if `noise_level` is 0.
        """
        if self.noise_level > 0:
            noise = np.random.randn(*trajectory.shape) * self.noise_level
            # Scale noise by signal magnitude
            signal_std = np.std(trajectory, axis=0)
            scaled_noise = noise * signal_std
            return trajectory + scaled_noise
        return trajectory

    def get_ground_truth_laws(self) -> Dict[str, sp.Expr]:
        """Returns the ground truth physical laws of the environment.

        These laws are typically used for validating discovered expressions.

        Returns:
            A dictionary where keys are law names (str) and values are SymPy
            expressions (sp.Expr) representing the laws.
        """
        return self.ground_truth_laws


class HarmonicOscillatorEnv(PhysicsEnvironment):
    """Represents a simple harmonic oscillator environment.

    This environment simulates the motion of a mass attached to a spring,
    obeying Hooke's Law. It can be used to test discovery algorithms on a
    well-understood physical system.

    Attributes:
        k (float): The spring constant.
        m (float): The mass of the oscillator.
        state_vars (List[str]): Typically ['x', 'v'] for position and velocity.
        ground_truth_laws (Dict[str, sp.Expr]): Includes laws for energy
            conservation and the equation of motion.
    """

    def __init__(self, params: Dict[str, Any], noise_level: float = 0.0):
        """Initializes the HarmonicOscillatorEnv.

        Args:
            params: A dictionary containing parameters 'k' (spring constant)
                and 'm' (mass). Defaults to 1.0 for both if not provided.
            noise_level: The standard deviation of Gaussian noise.
        """
        super().__init__(params, noise_level)
        self.k = params.get('k', 1.0)  # Spring constant
        self.m = params.get('m', 1.0)  # Mass
        self.state_vars = ['x', 'v']

        # Define ground truth laws
        x, v = sp.symbols('x v')
        self.ground_truth_laws = {
            'energy_conservation': 0.5 * self.m * v**2 + 0.5 * self.k * x**2,
            'equation_of_motion': -self.k * x / self.m  # This is acceleration
        }

    def dynamics(self, state: np.ndarray, t: float) -> np.ndarray:
        """Defines the ordinary differential equations for the harmonic oscillator.

        Args:
            state: A NumPy array `[x, v]` representing the current position and
                velocity.
            t: The current time (not used in this autonomous system, but
                required by `odeint`).

        Returns:
            A NumPy array `[dxdt, dvdt]` representing the time derivatives of
            position and velocity.
        """
        x, v = state
        dxdt = v
        dvdt = -self.k * x / self.m  # Acceleration = F/m = -kx/m
        return np.array([dxdt, dvdt])

    def generate_trajectory(self,
                          initial_conditions: np.ndarray,
                          t_span: np.ndarray) -> np.ndarray:
        """Generates a trajectory for the harmonic oscillator using an ODE solver.

        Also calculates and appends the total energy as an additional observed
        quantity.

        Args:
            initial_conditions: A NumPy array `[x0, v0]` for the initial state.
            t_span: A NumPy array of time points for the simulation.

        Returns:
            A NumPy array where each row is `[x, v, energy]` at a time point,
            potentially with added noise.
        """
        trajectory = odeint(self.dynamics, initial_conditions, t_span)

        # Add derived quantities
        x, v = trajectory[:, 0], trajectory[:, 1]
        energy = 0.5 * self.m * v**2 + 0.5 * self.k * x**2

        # Combine into full observation matrix
        full_trajectory = np.column_stack([x, v, energy])

        return self.add_observation_noise(full_trajectory)


class PendulumEnv(PhysicsEnvironment):
    """Represents a pendulum environment.

    This environment simulates the motion of a simple pendulum. It can be
    configured for small-angle approximation or the full non-linear dynamics.

    Attributes:
        g (float): Acceleration due to gravity.
        l (float): Length of the pendulum.
        m (float): Mass of the pendulum bob.
        small_angle (bool): If True, uses the small-angle approximation
            (sin(theta) approx theta).
        state_vars (List[str]): Typically ['theta', 'omega'] for angular
            position and angular velocity.
        ground_truth_laws (Dict[str, sp.Expr]): Includes laws for energy
            conservation and the equation of motion, adapted for small-angle
            if applicable.
    """

    def __init__(self, params: Dict[str, Any], noise_level: float = 0.0):
        """Initializes the PendulumEnv.

        Args:
            params: A dictionary containing 'g' (gravity), 'l' (length),
                'm' (mass), and optionally 'small_angle' (bool, defaults to
                False).
            noise_level: The standard deviation of Gaussian noise.
        """
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
                'equation_of_motion': -self.g * theta / self.l  # Angular acceleration
            }
        else:
            self.ground_truth_laws = {
                'energy_conservation': 0.5 * self.m * self.l**2 * omega**2 +
                                     self.m * self.g * self.l * (1 - sp.cos(theta)),
                'equation_of_motion': -self.g * sp.sin(theta) / self.l  # Angular acceleration
            }

    def dynamics(self, state: np.ndarray, t: float) -> np.ndarray:
        """Defines the ODEs for the pendulum.

        Args:
            state: A NumPy array `[theta, omega]` representing current angular
                position and angular velocity.
            t: Current time (not used).

        Returns:
            A NumPy array `[dtheta_dt, domega_dt]` representing time derivatives.
        """
        theta, omega = state
        if self.small_angle:
            domega_dt = -self.g * theta / self.l
        else:
            domega_dt = -self.g * np.sin(theta) / self.l
        return np.array([omega, domega_dt])

    def generate_trajectory(self,
                          initial_conditions: np.ndarray,
                          t_span: np.ndarray) -> np.ndarray:
        """Generates a trajectory for the pendulum.

        Calculates and appends total energy as an additional observed quantity.

        Args:
            initial_conditions: NumPy array `[theta0, omega0]` for initial state.
            t_span: NumPy array of time points.

        Returns:
            NumPy array where each row is `[theta, omega, energy]`, potentially
            with added noise.
        """
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
    """Represents a two-body gravitational system (Kepler problem).

    This environment simulates the motion of a smaller mass orbiting a larger
    central mass under gravity, described in polar coordinates.

    Attributes:
        G (float): Gravitational constant.
        M (float): Mass of the central body.
        state_vars (List[str]): Typically ['r', 'theta', 'vr', 'vtheta'] for
            radial distance, angle, radial velocity, and angular speed.
        ground_truth_laws (Dict[str, sp.Expr]): Includes energy conservation,
            angular momentum conservation, and the radial equation of motion.
    """

    def __init__(self, params: Dict[str, Any], noise_level: float = 0.0):
        """Initializes the KeplerEnv.

        Args:
            params: Dictionary containing 'G' (gravitational constant) and
                'M' (central mass). Defaults to 1.0 for both.
            noise_level: Standard deviation of Gaussian noise.
        """
        super().__init__(params, noise_level)
        self.G = params.get('G', 1.0)
        self.M = params.get('M', 1.0)  # Central mass
        self.state_vars = ['r', 'theta', 'vr', 'vtheta'] # r, angle, dr/dt, dtheta/dt

        # Ground truth (in polar coordinates)
        r, theta, vr, vtheta = sp.symbols('r theta vr vtheta')
        self.ground_truth_laws = {
            'energy_conservation': 0.5 * (vr**2 + (r * vtheta)**2) - self.G * self.M / r, # Note: (r*vtheta) is tangential velocity
            'angular_momentum': r**2 * vtheta, # Per unit mass (specific angular momentum)
            'equation_of_motion_r': r * vtheta**2 - self.G * self.M / r**2 # Radial acceleration (d^2r/dt^2)
        }

    def dynamics(self, state: np.ndarray, t: float) -> np.ndarray:
        """Defines the ODEs for the Kepler problem in polar coordinates.

        Args:
            state: NumPy array `[r, theta, vr, vtheta]` representing current
                radial distance, angle, radial velocity, and angular speed.
            t: Current time (not used).

        Returns:
            NumPy array `[dr_dt, dtheta_dt, dvr_dt, dvtheta_dt]` of time
            derivatives.
        """
        r, theta, vr, vtheta = state

        dr_dt = vr
        dtheta_dt = vtheta # This is omega (angular velocity)
        dvr_dt = r * vtheta**2 - self.G * self.M / r**2 # Radial acceleration
        dvtheta_dt = -2 * vr * vtheta / r          # Tangential acceleration component for angular velocity

        return np.array([dr_dt, dtheta_dt, dvr_dt, dvtheta_dt])

    def generate_trajectory(self,
                          initial_conditions: np.ndarray,
                          t_span: np.ndarray) -> np.ndarray:
        """Generates an orbital trajectory for the Kepler problem.

        Calculates and appends total energy and angular momentum (per unit mass)
        as additional observed quantities.

        Args:
            initial_conditions: NumPy array `[r0, theta0, vr0, vtheta0]` for
                the initial state.
            t_span: NumPy array of time points.

        Returns:
            NumPy array where each row is `[r, theta, vr, vtheta, energy,
            angular_momentum]`, potentially with added noise.
        """
        trajectory = odeint(self.dynamics, initial_conditions, t_span)

        r, theta, vr, vtheta = trajectory.T
        # Note: vtheta from odeint is d(angle)/dt. Tangential velocity is r * vtheta.
        energy = 0.5 * (vr**2 + (r * vtheta)**2) - self.G * self.M / r
        angular_momentum = r**2 * vtheta # Specific angular momentum

        full_trajectory = np.column_stack([r, theta, vr, vtheta, energy, angular_momentum])
        return self.add_observation_noise(full_trajectory)


class ExperimentRunner:
    """Orchestrates the execution and management of physics discovery experiments.

    This class handles the setup of experiments based on configurations,
    manages different types of physics environments and discovery algorithms,
    runs experiments (potentially in suites), saves results, and provides
    basic analysis capabilities. It utilizes a plugin system for discovering
    and loading experiment types and algorithms.

    Attributes:
        base_dir (Path): The root directory where experiment results and
            configurations are stored.
        use_wandb (bool): If True, attempts to use Weights & Biases for logging.
        env_registry (Dict[str, Callable]): A registry mapping environment names
            (e.g., 'harmonic_oscillator') to their corresponding class
            constructors (e.g., HarmonicOscillatorEnv).
        algo_registry (Dict[str, Callable]): A registry mapping algorithm names
            (e.g., 'janus_full') to factory functions that create algorithm
            instances.
        experiment_plugins (Dict[str, Callable[..., BaseExperiment]]): A registry
            mapping experiment type names (e.g., 'physics_discovery_example')
            to the experiment class constructors, discovered via plugins.
    """

    def __init__(self,
                 base_dir: str = "./experiments",
                 use_wandb: bool = True):
        """Initializes the ExperimentRunner.

        Args:
            base_dir: The directory to store experiment results. Defaults to
                "./experiments".
            use_wandb: Whether to enable Weights & Biases integration.
                Defaults to True.
        """
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
        """Discovers and registers experiment plugins from 'janus.experiments' entry points.

        This method uses `importlib.metadata` to find installed packages that
        declare 'janus.experiments' entry points. Valid plugins (subclasses of
        `BaseExperiment`) are loaded and added to the `experiment_plugins`
        registry.
        """
        logging.info("Discovering 'janus.experiments' plugins...")
        try:
            # Modern approach for Python 3.8+
            # For Python 3.10+ one could use:
            # eps = importlib.metadata.entry_points(group='janus.experiments')
            # For 3.8/3.9, the following is more robust if group-specific selection isn't available directly
            all_eps = importlib.metadata.entry_points()
            if hasattr(all_eps, 'select'):  # For Python 3.10+ and some backports
                eps = all_eps.select(group='janus.experiments')
            elif 'janus.experiments' in all_eps:  # For older 3.8/3.9
                eps = all_eps['janus.experiments']
            else:
                eps = []
        except Exception as e:
            logging.warning(f"Could not query entry points for 'janus.experiments' due to: {e}. Manual registration or installation might be needed.")
            eps = []

        if not eps:
            logging.warning("No 'janus.experiments' plugins found or loaded. ExperimentRunner may not find specific experiment types.")
            return

        for entry_point in eps:
            try:
                loaded_class = entry_point.load()
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
        """Registers known discovery algorithms into the `algo_registry`.

        This method imports necessary algorithm components and creates factory
        functions for each algorithm. These factories are then stored in
        `self.algo_registry` keyed by algorithm name (e.g., 'janus_full',
        'genetic'). The factory functions are designed to instantiate the
        algorithm with environment data, variables, and the experiment
        configuration.
        """
        # Import algorithm classes
        from symbolic_discovery_env import SymbolicDiscoveryEnv, CurriculumManager # type: ignore
        from hypothesis_policy_network import HypothesisNet, PPOTrainer # type: ignore
        from progressive_grammar_system import ProgressiveGrammar
        from physics_discovery_extensions import SymbolicRegressor

        # Janus full system
        def create_janus_full(env_data: np.ndarray,
                              variables: List[Any],
                              config: ExperimentConfig) -> PPOTrainer:
            """Factory function to create a Janus PPO-based trainer."""
            grammar = ProgressiveGrammar()
            env_creation_params = config.algo_params.get('env_params', {})
            env_creation_params['target_variable_index'] = config.target_variable_index

            discovery_env = SymbolicDiscoveryEnv(
                grammar=grammar,
                target_data=env_data,
                variables=variables,
                **env_creation_params
            )
            policy_params = config.algo_params.get('policy_params', {})
            policy = HypothesisNet(
                observation_dim=discovery_env.observation_space.shape[0],
                action_dim=discovery_env.action_space.n,
                **policy_params
            )
            trainer = PPOTrainer(policy, discovery_env)
            return trainer

        # Genetic programming baseline
        def create_genetic(env_data: np.ndarray,
                           variables: List[Any],
                           config: ExperimentConfig) -> SymbolicRegressor:
            """Factory function to create a Symbolic Regressor (genetic algorithm)."""
            grammar = ProgressiveGrammar()
            # algo_params for SymbolicRegressor might include population size, generations, etc.
            regressor_params = config.algo_params.get('regressor_params', {})
            regressor = SymbolicRegressor(grammar=grammar, **regressor_params)
            return regressor

        self.algo_registry['janus_full'] = create_janus_full
        self.algo_registry['genetic'] = create_genetic
        # A placeholder for a random search algorithm
        self.algo_registry['random'] = lambda env_data, variables, config: None

    def run_single_experiment(self,
                            config: ExperimentConfig,
                            run_id: int = 0) -> ExperimentResult:
        """Runs a single experiment as defined by the configuration.

        This method selects the appropriate experiment plugin based on
        `config.experiment_type`, instantiates it, and then executes it.
        It handles errors during plugin loading or instantiation and ensures
        that an `ExperimentResult` object is always returned.

        Args:
            config: The `ExperimentConfig` object detailing the experiment.
            run_id: An integer identifier for this specific run, useful when
                the same configuration is executed multiple times.

        Returns:
            An `ExperimentResult` object containing the outcomes of the
            experiment. If critical errors occur (e.g., plugin not found,
            instantiation failure), the `discovered_law` field in the result
            will contain an error message, and metrics like accuracy and MSE
            will be set to default failure values.
        """
        logging.info(f"ExperimentRunner: Starting run_single_experiment for '{config.name}' (Run {run_id + 1}/{config.n_runs}). Experiment Type: '{config.experiment_type}'")

        if not hasattr(config, 'experiment_type') or not config.experiment_type:
            logging.error(f"Experiment configuration '{config.name}' is missing the 'experiment_type' field.")
            # RAISE InvalidConfigError
            raise InvalidConfigError(f"Configuration '{config.name}' is missing the 'experiment_type' field.")

        experiment_class = self.experiment_plugins.get(config.experiment_type)

        if experiment_class is None:
            logging.error(f"Experiment type '{config.experiment_type}' for config '{config.name}' not found in discovered plugins. Available plugins: {list(self.experiment_plugins.keys())}")
            # RAISE PluginNotFoundError
            raise PluginNotFoundError(f"Experiment type '{config.experiment_type}' for config '{config.name}' not found. Available: {list(self.experiment_plugins.keys())}")

        logging.info(f"Instantiating experiment of type '{config.experiment_type}' using class {experiment_class.__module__}.{experiment_class.__name__}.")

        try:
            experiment_instance = experiment_class(
                config=config,
                algo_registry=self.algo_registry,
                env_registry=self.env_registry
            )
        except (ValueError, TypeError) as e: # Catch common configuration or type errors during instantiation
            logging.error(f"Failed to instantiate experiment class '{experiment_class.__name__}' for type '{config.experiment_type}' due to invalid parameters or configuration. Error: {e}", exc_info=True)
            # RAISE InvalidConfigError
            raise InvalidConfigError(f"Failed to instantiate experiment '{config.experiment_type}' due to invalid parameters: {e}") from e
        except Exception as e: # Catch other unexpected errors during instantiation
            logging.error(f"Failed to instantiate experiment class '{experiment_class.__name__}' for type '{config.experiment_type}'. Error: {e}", exc_info=True)
            # Consider if this should be a more generic error or PluginOperationFailedError
            raise RuntimeError(f"An unexpected error occurred while instantiating experiment '{config.experiment_type}': {e}") from e

        result = experiment_instance.execute(run_id=run_id)

        if result is None:
            logging.error(f"Experiment '{config.name}' (Run {run_id + 1}, Type '{config.experiment_type}') critically failed and did not return a result object from execute().")
            result = ExperimentResult(config=config, run_id=run_id, discovered_law="Critical Execution Error: execute() returned None.")
            result.symbolic_accuracy = 0.0
            result.predictive_mse = float('inf')
            result.wall_time_seconds = 0.0 # Ensure time is set
        else:
            logging.info(f"ExperimentRunner: Completed run_single_experiment for '{config.name}' (Run {run_id + 1}). Law: '{result.discovered_law}', Accuracy: {result.symbolic_accuracy:.4f}, Time: {result.wall_time_seconds:.2f}s.")

        return result

    def _run_janus_experiment(self,
                            setup: Dict[str, Any],
                            config: ExperimentConfig,
                            result: ExperimentResult) -> ExperimentResult:
        """Internal helper to run the Janus algorithm (legacy, now in plugin).

        This method encapsulates the logic for running the Janus PPO-based
        symbolic discovery algorithm. It's maintained for potential direct use
        or reference but is typically superseded by the `run` method within a
        specific experiment plugin (e.g., `PhysicsDiscoveryExperiment`).

        Args:
            setup: A dictionary containing pre-initialized components like the
                'algorithm' (PPOTrainer instance).
            config: The `ExperimentConfig` for this run.
            result: The `ExperimentResult` object to populate.

        Returns:
            The populated `ExperimentResult` object.
        """
        trainer: PPOTrainer = setup['algorithm']
        sample_efficiency_curve = []
        best_mse = float('inf')
        best_expression = None

        # Training loop with periodic evaluation
        # Default PPO training parameters, can be overridden by config.algo_params
        ppo_train_defaults = {
            'total_timesteps': config.algo_params.get('timesteps_per_eval', 1000),
            'rollout_length': config.algo_params.get('rollout_length', 512),
            'n_epochs': config.algo_params.get('n_epochs', 3),
            'log_interval': config.algo_params.get('log_interval', 100)
        }
        # Approximate total timesteps or number of evaluation cycles
        num_eval_cycles = config.max_experiments # Assuming max_experiments is number of eval cycles

        for i in range(num_eval_cycles):
            trainer.train(**ppo_train_defaults)
            current_total_timesteps = (i + 1) * ppo_train_defaults['total_timesteps']

            # Evaluate best discovered expression from the trainer's environment
            if hasattr(trainer, 'env') and hasattr(trainer.env, '_evaluation_cache') and trainer.env._evaluation_cache:
                # Assuming _evaluation_cache is a list of dicts like {'expression': str, 'mse': float}
                for eval_entry in trainer.env._evaluation_cache: # This might need adjustment based on actual cache structure
                    current_mse = eval_entry.get('mse', float('inf'))
                    current_expression = eval_entry.get('expression')
                    if current_expression and current_mse < best_mse:
                        best_mse = current_mse
                        best_expression = current_expression
                sample_efficiency_curve.append((current_total_timesteps, best_mse))
            elif hasattr(trainer, 'episode_mse') and trainer.episode_mse: # Fallback if no direct cache
                current_episode_mse_mean = np.mean(trainer.episode_mse)
                if current_episode_mse_mean < best_mse:
                    best_mse = current_episode_mse_mean
                    # Note: best_expression might not be updated in this fallback
                sample_efficiency_curve.append((current_total_timesteps, best_mse))
            else:
                sample_efficiency_curve.append((current_total_timesteps, float('inf')))


        result.discovered_law = str(best_expression) if best_expression else None
        result.predictive_mse = best_mse
        result.sample_efficiency_curve = sample_efficiency_curve
        # n_experiments_to_convergence can be interpreted as number of evaluation cycles
        result.n_experiments_to_convergence = num_eval_cycles
        return result

    def _run_genetic_experiment(self,
                              setup: Dict[str, Any],
                              config: ExperimentConfig,
                              result: ExperimentResult) -> ExperimentResult:
        """Internal helper to run genetic programming (legacy, now in plugin).

        Encapsulates logic for the genetic programming baseline using
        `SymbolicRegressor`. Similar to `_run_janus_experiment`, this is
        maintained for reference and typically superseded by plugin implementations.

        Args:
            setup: Dictionary with pre-initialized 'algorithm' (SymbolicRegressor)
                and 'env_data', 'variables'.
            config: The `ExperimentConfig`.
            result: The `ExperimentResult` object to populate.

        Returns:
            The populated `ExperimentResult` object.
        """
        regressor: SymbolicRegressor = setup['algorithm']
        env_data: np.ndarray = setup['env_data']
        variables: List[Any] = setup['variables'] # List of Variable objects

        target_idx = config.target_variable_index if config.target_variable_index is not None else -1
        y = env_data[:, target_idx]
        X = np.delete(env_data, target_idx, axis=1)

        # Variables for regressor might need adjustment if target column is removed
        # Assuming SymbolicRegressor or its ProgressiveGrammar handles variable mapping.
        # Or, 'variables' in setup should be pre-filtered.
        # For now, passing all variables.

        best_expr_obj = regressor.fit(
            X, y,
            variables, # Pass the list of Variable objects
            max_complexity=config.algo_params.get('max_complexity', 15)
        )

        if best_expr_obj and hasattr(best_expr_obj, 'symbolic'):
            result.discovered_law = str(best_expr_obj.symbolic)
            result.law_complexity = best_expr_obj.complexity if hasattr(best_expr_obj, 'complexity') else 0 # Or len(str)

            # Calculate MSE
            # This part is complex if not directly provided by regressor, due to variable mapping
            if hasattr(best_expr_obj, 'mse') and best_expr_obj.mse is not None:
                 result.predictive_mse = best_expr_obj.mse
            elif hasattr(regressor, 'predict'):
                try:
                    predictions = regressor.predict(X)
                    result.predictive_mse = np.mean((np.array(predictions) - y)**2)
                except Exception:
                    result.predictive_mse = float('inf') # Prediction failed
            else: # Fallback: manual substitution (can be error-prone)
                predictions = []
                # Create a mapping from variable name to its index in X
                # This assumes variables in 'setup' are still the original ones.
                # A safer way is to get variable names from regressor or X's columns.
                # For simplicity, this part is omitted here but crucial for correctness.
                # Placeholder for manual MSE calculation:
                result.predictive_mse = float('inf')
        else:
            result.discovered_law = None
            result.predictive_mse = float('inf')
            result.law_complexity = 0

        return result

    def run_experiment_suite(self,
                           configs: List[ExperimentConfig],
                           parallel: bool = False) -> pd.DataFrame:
        """Runs a suite of experiments as defined by a list of configurations.

        Iterates through each `ExperimentConfig` in the provided list, runs it
        for the specified number of `n_runs`, and collects all results.
        Individual results are saved, and an aggregated DataFrame is compiled
        and saved.

        Args:
            configs: A list of `ExperimentConfig` objects.
            parallel: If True, attempts to run experiments in parallel (currently
                not implemented). Defaults to False.

        Returns:
            A pandas DataFrame summarizing the results from all experiment runs.
        """
        all_results: List[ExperimentResult] = []

        if parallel:
            logging.warning("Parallel experiment execution is not yet implemented. Running sequentially.")

        for config_item in tqdm(configs, desc="All Experiment Configs"):
            config_run_results: List[ExperimentResult] = []
            for i in range(config_item.n_runs):
                # Pass 'i' as run_id
                single_run_result = self.run_single_experiment(config_item, run_id=i)
                config_run_results.append(single_run_result)
                self._save_result(single_run_result) # Save each individual run
            all_results.extend(config_run_results)

        results_df = self._results_to_dataframe(all_results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_df.to_csv(self.base_dir / f"all_results_{timestamp}.csv", index=False)
        logging.info(f"Aggregated results saved to all_results_{timestamp}.csv")

        return results_df

    def _save_result(self, result: ExperimentResult):
        """Saves a single ExperimentResult to disk.

        The result is saved as a Pickle file and a human-readable JSON summary.
        Results are organized into directories based on the hash of their
        `ExperimentConfig`.

        Args:
            result: The `ExperimentResult` object to save.
        """
        config_hash = result.config.get_hash()
        result_dir = self.base_dir / config_hash
        result_dir.mkdir(parents=True, exist_ok=True) # Ensure parent dirs exist

        # Save full result as Pickle
        pickle_filename = f"run_{result.run_id}_{result.config.name}.pkl"
        with open(result_dir / pickle_filename, 'wb') as f:
            pickle.dump(result, f)

        # Save summary as JSON
        summary_filename = f"summary_run_{result.run_id}_{result.config.name}.json"
        summary_data = {
            'config_name': result.config.name,
            'run_id': result.run_id,
            'discovered_law': result.discovered_law,
            'symbolic_accuracy': result.symbolic_accuracy,
            'predictive_mse': result.predictive_mse,
            'n_experiments_to_convergence': result.n_experiments_to_convergence,
            'wall_time_seconds': result.wall_time_seconds,
            'config_hash': config_hash
        }
        with open(result_dir / summary_filename, 'w') as f:
            json.dump(summary_data, f, indent=2)
        logging.debug(f"Saved result for run {result.run_id} of '{result.config.name}' to {result_dir}")

    def _results_to_dataframe(self, results: List[ExperimentResult]) -> pd.DataFrame:
        """Converts a list of ExperimentResult objects into a pandas DataFrame.

        This is useful for analysis and reporting of experiment suite outcomes.

        Args:
            results: A list of `ExperimentResult` objects.

        Returns:
            A pandas DataFrame where each row corresponds to an
            `ExperimentResult`.
        """
        rows = []
        for res in results:
            row_data = {
                'experiment_name': res.config.name,
                'algorithm': res.config.algorithm,
                'environment_type': res.config.environment_type,
                'noise_level': res.config.noise_level,
                'run_id': res.run_id,
                'symbolic_accuracy': res.symbolic_accuracy,
                'predictive_mse': res.predictive_mse,
                'law_complexity': res.law_complexity,
                'n_experiments_to_convergence': res.n_experiments_to_convergence,
                'wall_time_seconds': res.wall_time_seconds,
                'discovered_law': res.discovered_law,
                'config_hash': res.config.get_hash()
                # Add other config fields if needed for detailed analysis, e.g., res.config.seed
            }
            # Include key env_params and algo_params if they are consistent or simple
            # For complex params, consider serializing them or summarizing
            # Example: row_data.update({f'env_{k}': v for k,v in res.config.env_params.items()})
            rows.append(row_data)
        return pd.DataFrame(rows)

    def analyze_results(self, df: pd.DataFrame):
        """Performs basic analysis on a DataFrame of experiment results.

        Generates and saves plots for sample efficiency and accuracy comparison.
        Also prints and saves a CSV of summary statistics.

        Args:
            df: A pandas DataFrame, typically generated by
                `_results_to_dataframe`.

        Returns:
            A pandas DataFrame containing summary statistics.
        """
        analysis_dir = self.base_dir / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)

        # 1. Sample efficiency comparison (Experiments to Convergence vs. Noise)
        plt.figure(figsize=(10, 6))
        for algo in df['algorithm'].unique():
            algo_df = df[df['algorithm'] == algo]
            # Group by 'noise_level' and then 'algorithm' to calculate mean and std for 'n_experiments_to_convergence'
            # Filter out NaN or inf values to prevent plotting issues
            plot_df = df[np.isfinite(df['n_experiments_to_convergence'])].copy()
            if not plot_df.empty:
                # Ensure noise_level is numeric for proper sorting and plotting
                plot_df['noise_level'] = pd.to_numeric(plot_df['noise_level'], errors='coerce')
                plot_df.dropna(subset=['noise_level'], inplace=True)

                grouped = plot_df.groupby(['noise_level', 'algorithm'])['n_experiments_to_convergence']
                mean_experiments = grouped.mean().unstack()
                std_experiments = grouped.std().unstack()

                for algo_col in mean_experiments.columns:
                    plt.errorbar(mean_experiments.index, mean_experiments[algo_col],
                                 yerr=std_experiments[algo_col] if algo_col in std_experiments.columns else None,
                                 label=algo_col, marker='o', capsize=3)
                plt.xlabel('Noise Level')
                plt.ylabel('Mean Experiments to Convergence')
                plt.title('Sample Efficiency vs. Noise Level')
                plt.legend()
                plt.yscale('log') # Keep log scale if appropriate for data range
                plt.grid(True, which="both", ls="-", alpha=0.5)
                plt.savefig(analysis_dir / 'sample_efficiency_vs_noise.png')
            else:
                logging.warning("No valid data for sample efficiency plot after filtering.")
        plt.close()


        # 2. Symbolic Accuracy comparison by Environment and Algorithm
        plt.figure(figsize=(12, 7))
        # Filter out NaN or inf values for 'symbolic_accuracy'
        plot_df_acc = df[np.isfinite(df['symbolic_accuracy'])].copy()
        if not plot_df_acc.empty:
            # Using seaborn for potentially nicer grouped bar plots
            sns.barplot(x='environment_type', y='symbolic_accuracy', hue='algorithm', data=plot_df_acc, errorbar='sd', capsize=.1)
            plt.ylabel('Mean Symbolic Accuracy')
            plt.xlabel('Environment Type')
            plt.title('Symbolic Accuracy by Environment and Algorithm')
            plt.xticks(rotation=30, ha='right')
            plt.legend(title='Algorithm')
            plt.tight_layout()
            plt.savefig(analysis_dir / 'accuracy_by_environment_algorithm.png')
        else:
            logging.warning("No valid data for symbolic accuracy plot after filtering.")
        plt.close()

        # 3. Statistical summary table
        # Define aggregations for summary statistics
        agg_funcs = {
            'symbolic_accuracy': ['mean', 'std', 'count'],
            'predictive_mse': ['mean', 'std'],
            'n_experiments_to_convergence': ['mean', 'std'],
            'wall_time_seconds': ['mean', 'std']
        }
        # Group by algorithm, environment, and noise level for a more detailed summary
        detailed_summary_stats = df.groupby(['algorithm', 'environment_type', 'noise_level']).agg(agg_funcs).round(3)
        detailed_summary_stats.to_csv(analysis_dir / 'detailed_summary_statistics.csv')
        print("\n=== Detailed Summary Statistics ===")
        print(detailed_summary_stats)

        # Simpler summary (as before, but good to have both)
        simple_summary_stats = df.groupby(['algorithm', 'environment_type']).agg({
            'symbolic_accuracy': ['mean', 'std'],
            'predictive_mse': ['mean', 'std'],
        }).round(3)
        simple_summary_stats.to_csv(analysis_dir / 'simple_summary_statistics.csv')
        print("\n=== Simple Summary Statistics (Aggregated over noise) ===")
        print(simple_summary_stats)

        logging.info(f"Analysis plots and statistics saved to {analysis_dir}")
        return detailed_summary_stats # Return the more detailed one


# Phase 1 implementation (Example)
def run_phase1_validation():
    """Example: Run Phase 1 - Known law rediscovery across environments."""
    logging.basicConfig(level=logging.INFO) # Ensure logging is configured
    runner = ExperimentRunner(base_dir="./experiments_phase1")
    configs = []
    algorithms = ['janus_full', 'genetic'] # Algorithms to test
    environments = ['harmonic_oscillator', 'pendulum'] # Environments for rediscovery

    for env_type in environments:
        for algo in algorithms:
            env_specific_params = {}
            if env_type == 'pendulum':
                env_specific_params = {'g': 9.81, 'l': 1.0, 'm': 1.0, 'small_angle': True}
            elif env_type == 'harmonic_oscillator':
                env_specific_params = {'k': 1.0, 'm': 1.0}

            config = ExperimentConfig(
                name=f"{env_type}_rediscovery_{algo}",
                experiment_type='physics_discovery_example', # Assuming this is a registered plugin
                environment_type=env_type,
                algorithm=algo,
                env_params=env_specific_params,
                noise_level=0.0, # Ideal conditions for rediscovery
                max_experiments=100, # Reduced for quicker example run
                n_runs=2 # Reduced for quicker example run
            )
            configs.append(config)

    if not configs:
        logging.warning("Phase 1: No configurations generated. Check environment and algorithm lists.")
        return None
    if not runner.experiment_plugins.get('physics_discovery_example'):
        logging.error("Phase 1: 'physics_discovery_example' plugin not found. Cannot run validation.")
        return None

    results_df = runner.run_experiment_suite(configs)
    if results_df is not None and not results_df.empty:
        runner.analyze_results(results_df)
    else:
        logging.warning("Phase 1: No results generated from experiment suite.")
    return results_df


# Phase 2 implementation (Example)
def run_phase2_robustness():
    """Example: Run Phase 2 - Robustness to noise benchmark."""
    logging.basicConfig(level=logging.INFO) # Ensure logging is configured
    runner = ExperimentRunner(base_dir="./experiments_phase2")
    configs = []
    noise_levels = [0.0, 0.01, 0.05] # Example noise levels
    algorithms = ['janus_full', 'genetic'] # Assuming 'random' might be too slow or less informative here
    # Using a single, well-understood environment for noise robustness
    environment_type = 'harmonic_oscillator'

    for noise in noise_levels:
        for algo in algorithms:
            config = ExperimentConfig(
                name=f"{environment_type}_noise_{noise*100:.0f}pct_{algo}",
                experiment_type='physics_discovery_example',
                environment_type=environment_type,
                algorithm=algo,
                env_params={'k': 1.0, 'm': 1.0},
                noise_level=noise,
                max_experiments=100, # Reduced for quicker example
                n_runs=2 # Reduced for quicker example
            )
            configs.append(config)

    if not configs:
        logging.warning("Phase 2: No configurations generated.")
        return None
    if not runner.experiment_plugins.get('physics_discovery_example'):
        logging.error("Phase 2: 'physics_discovery_example' plugin not found. Cannot run validation.")
        return None

    results_df = runner.run_experiment_suite(configs)
    if results_df is not None and not results_df.empty:
        runner.analyze_results(results_df)
    else:
        logging.warning("Phase 2: No results generated from experiment suite.")
    return results_df


if __name__ == "__main__":
    # Basic configuration for logging, useful for seeing output from ExperimentRunner
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - [%(module)s:%(funcName)s:%(lineno)d] - %(message)s')

    print("ExperimentRunner __main__ Test Section")
    print("======================================")

    # Example: Test a single experiment run directly if needed
    # This helps in debugging a specific configuration or plugin.
    example_config = ExperimentConfig(
        name="main_test_harmonic_oscillator",
        experiment_type='physics_discovery_example', # CRITICAL: This plugin must be registered
        environment_type='harmonic_oscillator',
        algorithm='genetic',
        env_params={'k': 1.0, 'm': 1.0},
        noise_level=0.01,
        max_experiments=50, # Small value for a quick test
        n_runs=1 # Single run for this direct test
    )

    runner_main = ExperimentRunner(use_wandb=False, base_dir="./experiments_main_test")

    if example_config.experiment_type not in runner_main.experiment_plugins:
        logging.error(f"__main__: Experiment type '{example_config.experiment_type}' not found in plugins.")
        logging.error("Please ensure 'physics_discovery_example' is correctly registered via setup.py entry_points.")
        logging.error("And that the package has been installed (e.g., 'pip install -e .').")
        logging.error(f"Available plugins: {list(runner_main.experiment_plugins.keys())}")
    else:
        logging.info(f"__main__: Attempting to run experiment '{example_config.name}'...")
        try:
            result = runner_main.run_single_experiment(example_config, run_id=0)
            if result:
                print(f"\n--- __main__ Experiment '{example_config.name}' Results ---")
                print(f"  Discovered law: {result.discovered_law}")
                print(f"  Symbolic accuracy: {result.symbolic_accuracy:.4f}")
                print(f"  Predictive MSE: {result.predictive_mse:.4e}")
                print(f"  Wall time: {result.wall_time_seconds:.2f}s")
            else:
                # run_single_experiment should ideally always return a result, even for errors
                print(f"__main__: Experiment '{example_config.name}' returned None or an unexpected result.")
        except Exception as e:
            logging.error(f"__main__: Critical error during run_single_experiment: {e}", exc_info=True)
            print(f"__main__: An exception occurred: {e}")

    # Example of running predefined validation phases (optional, can be lengthy)
    # print("\nStarting Phase 1 Validation (Known Law Rediscovery)...")
    # phase1_results = run_phase1_validation()
    # if phase1_results is not None:
    #     print("Phase 1 Validation completed. Results summary:")
    #     print(phase1_results.groupby(['experiment_name', 'algorithm'])['symbolic_accuracy'].mean())
    # else:
    #     print("Phase 1 Validation did not produce results or was skipped.")

    # print("\nStarting Phase 2 Validation (Robustness Benchmark)...")
    # phase2_results = run_phase2_robustness()
    # if phase2_results is not None:
    #     print("Phase 2 Validation completed. Results summary (mean accuracy by noise):")
    #     print(phase2_results.groupby(['noise_level', 'algorithm'])['symbolic_accuracy'].mean())
    # else:
    #     print("Phase 2 Validation did not produce results or was skipped.")

    print("\n__main__ Test Section Finished.")


class PhysicsDiscoveryExperiment(BaseExperiment):
    """An example implementation of BaseExperiment for physics discovery tasks.

    This class orchestrates the setup, execution, and teardown of a specific
    type of physics discovery experiment. It integrates with the ExperimentRunner's
    registries for environments and algorithms.

    Attributes:
        config (ExperimentConfig): Configuration object for the experiment.
        algo_registry (Dict[str, Callable]): Registry of available algorithms.
        env_registry (Dict[str, Callable]): Registry of available environments.
        physics_env (Optional[PhysicsEnvironment]): The instantiated physics
            environment for the experiment.
        env_data (Optional[np.ndarray]): Data generated from the physics environment.
        variables (Optional[List[Any]]): List of variables for the symbolic system,
            compatible with `progressive_grammar_system`.
        algorithm (Optional[Any]): The instantiated discovery algorithm.
        ground_truth_laws (Optional[Dict[str, sp.Expr]]): Ground truth equations
            for the environment.
        experiment_result (Optional[ExperimentResult]): Holds the results of the
            experiment execution.
        _start_time (float): Internal timer for wall-clock time measurement.
    """
    def __init__(self,
                 config: ExperimentConfig,
                 algo_registry: Dict[str, Callable],
                 env_registry: Dict[str, Callable]):
        """Initializes the PhysicsDiscoveryExperiment.

        Args:
            config: The configuration object for this experiment.
            algo_registry: A dictionary mapping algorithm names to their
                factory functions.
            env_registry: A dictionary mapping environment names to their
                class constructors.
        """
        super().__init__() # Initialize BaseExperiment
        self.config = config
        self.algo_registry = algo_registry
        self.env_registry = env_registry

        # Paths for logging and tracking
        base_output_dir = Path(self.config.algo_params.get('base_output_dir', './experiments_output')) / self.config.name / f"run_{int(time.time())}"
        self.tracker_save_dir = base_output_dir / "hypothesis_tracking"
        self.log_file_path = base_output_dir / "training_logs.jsonl" # Changed from .json to .jsonl for line-by-line

        # Ensure directories exist
        self.tracker_save_dir.mkdir(parents=True, exist_ok=True)
        if self.log_file_path.parent:
             self.log_file_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize new components as per "Complete Integration Example"
        self.conservation_reward = ConservationBiasedReward(
            conservation_types=self.config.algo_params.get('conservation_types', ['energy', 'momentum']),
            weight_factor=self.config.algo_params.get('conservation_weight_factor', 0.3)
        )

        self.symmetry_detector = PhysicsSymmetryDetector(
            tolerance=self.config.algo_params.get('symmetry_tolerance', 1e-4),
            confidence_threshold=self.config.algo_params.get('symmetry_confidence', 0.7)
        )

        self.hypothesis_tracker = HypothesisTracker(
            save_directory=str(self.tracker_save_dir),
            autosave_interval=self.config.algo_params.get('tracker_autosave_interval', 100)
        )

        self.training_integration = JanusTrainingIntegration(self.hypothesis_tracker)

        # Configure logger backends, allow for 'redis' if specified in config
        logger_backends = self.config.algo_params.get('logger_backends', ["file"])
        self.training_logger = TrainingLogger(
            backends=logger_backends, # e.g. ["file", "redis"]
            log_file_path=str(self.log_file_path),
            redis_host=self.config.algo_params.get('redis_host', 'localhost'),
            redis_port=self.config.algo_params.get('redis_port', 6379),
            redis_channel=f"janus_{self.config.name}_metrics"
        )

        # Configure LiveMonitor data source
        monitor_data_source = "redis" if "redis" in logger_backends else "file"
        self.live_monitor = LiveMonitor(
            data_source=monitor_data_source,
            log_file_path=str(self.log_file_path), # Monitor tails this file if source is 'file'
            redis_host=self.config.algo_params.get('redis_host', 'localhost'),
            redis_port=self.config.algo_params.get('redis_port', 6379),
            redis_channel=f"janus_{self.config.name}_metrics" # Same channel as logger publishes to
        )

        # Attributes to be populated by setup()
        self.physics_env: Optional[PhysicsEnvironment] = None
        self.env_data: Optional[np.ndarray] = None # Full trajectory data for reference
        self.ground_truth_trajectory_processed: Optional[Dict[str,Any]] = None # For conservation checks
        self.variables: Optional[List[Any]] = None # progressive_grammar_system.Variable objects
        self.sympy_vars: Optional[List[sp.Symbol]] = None # Sympy symbols for lambdify
        self.env_data: Optional[np.ndarray] = None
        self.variables: Optional[List[Any]] = None
        self.algorithm: Optional[Any] = None
        self.ground_truth_laws: Optional[Dict[str, sp.Expr]] = None

        # Result object, to be populated by execute() -> run()
        self.experiment_result: Optional[ExperimentResult] = None
        self._start_time: float = 0.0 # For timing the experiment run

    def setup(self):
        """Sets up the experiment environment, data generation, and algorithm instantiation.

        This method performs the following steps:
        1. Sets random seeds for reproducibility.
        2. Instantiates the physics environment based on `config.environment_type`.
        3. Generates trajectory data from the environment.
        4. Defines variables for the symbolic discovery process.
        5. Instantiates the discovery algorithm based on `config.algorithm`.
        6. Retrieves ground truth laws from the environment.

        Raises:
            ValueError: If specified environment or algorithm is not found in
                their respective registries, or if data generation fails.
            ImportError: If essential components like `Variable` from
                `progressive_grammar_system` cannot be imported.
        """
        logging.info(f"[{self.config.name}] Initializing setup: Seed {self.config.seed}, Env '{self.config.environment_type}', Algo '{self.config.algorithm}'.")
        np.random.seed(self.config.seed)
        if 'torch' in globals(): # Check if torch is imported
            torch.manual_seed(self.config.seed)

        # 1. Create Environment
        logging.debug(f"[{self.config.name}] Creating environment '{self.config.environment_type}'.")
        env_class = self.env_registry.get(self.config.environment_type)
        if not env_class:
            logging.error(f"Environment type '{self.config.environment_type}' not in registry. Available: {list(self.env_registry.keys())}")
            raise ValueError(f"Environment type '{self.config.environment_type}' not found in registry.")
        self.physics_env = env_class(self.config.env_params, self.config.noise_level)
        logging.info(f"[{self.config.name}] Environment '{self.config.environment_type}' created: {self.physics_env}")

        # 2. Generate Training Data
        logging.debug(f"[{self.config.name}] Generating data: {self.config.n_trajectories} trajectories, {self.config.trajectory_length} length.")
        trajectories = []
        for _ in range(self.config.n_trajectories):
            # Initial conditions can be customized per environment
            init_cond = self._get_initial_conditions()
            t_span = np.arange(0, self.config.trajectory_length * self.config.sampling_rate, self.config.sampling_rate)
            if self.physics_env: # Should be true due to check above
                trajectory = self.physics_env.generate_trajectory(init_cond, t_span)
                trajectories.append(trajectory)

        if not trajectories:
            logging.error(f"[{self.config.name}] No trajectories generated.")
            raise ValueError("No trajectories generated. Check environment parameters and generation logic.")
        self.env_data = np.vstack(trajectories)
        logging.info(f"[{self.config.name}] Training data generated with shape {self.env_data.shape}.")

        # 3. Create Variables for Algorithm
        try:
            from progressive_grammar_system import Variable # type: ignore
        except ImportError:
            logging.error("Failed to import 'Variable' from 'progressive_grammar_system'. Ensure it's installed and accessible.")
            raise
        if self.physics_env and self.physics_env.state_vars:
            self.variables = [Variable(name, idx, {}) for idx, name in enumerate(self.physics_env.state_vars)]
        else:
            # Fallback if state_vars are not defined, though this indicates an issue with the env
            num_cols = self.env_data.shape[1]
            self.variables = [Variable(f'var{idx}', idx, {}) for idx in range(num_cols)]
            logging.warning(f"[{self.config.name}] PhysicsEnv.state_vars not defined. Using generic variable names based on data columns.")

        logging.debug(f"[{self.config.name}] State variables for algorithm: {[str(v) for v in self.variables]}")


        # 4. Create Algorithm Instance
        logging.debug(f"[{self.config.name}] Creating algorithm '{self.config.algorithm}'.")
        algo_factory = self.algo_registry.get(self.config.algorithm)
        if not algo_factory:
            logging.error(f"Algorithm '{self.config.algorithm}' not in registry. Available: {list(self.algo_registry.keys())}")
            raise ValueError(f"Algorithm '{self.config.algorithm}' not found in registry.")
        # Pass self.config to algo_factory for algorithm-specific parameters
        self.algorithm = algo_factory(self.env_data, self.variables, self.config)
        logging.info(f"[{self.config.name}] Algorithm '{self.config.algorithm}' created: {type(self.algorithm).__name__}")

        # Initialize Hypothesis Tracker and Training Integration
        # save_directory can be made more configurable if needed
        tracker_save_dir = Path(self.config.base_dir if hasattr(self.config, 'base_dir') else "./experiments") / self.config.name / "hypothesis_tracking"
        self.hypothesis_tracker = HypothesisTracker(
            save_directory=str(tracker_save_dir),
            autosave_interval=self.config.algo_params.get('tracker_autosave_interval', 100)
        )
        self.training_integration = JanusTrainingIntegration(self.hypothesis_tracker)
        logging.info(f"[{self.config.name}] HypothesisTracker initialized. Save dir: {tracker_save_dir}")

        # 5. Get Ground Truth Laws
        if self.physics_env:
            self.ground_truth_laws = self.physics_env.get_ground_truth_laws()
            logging.debug(f"[{self.config.name}] Ground truth laws obtained: {list(self.ground_truth_laws.keys()) if self.ground_truth_laws else 'None'}.")

        # Prepare ground_truth_trajectory_processed and sympy_vars
        if self.physics_env and hasattr(self.physics_env, 'get_ground_truth_conserved_quantities'):
            # Assumes env can provide a dict like {'conserved_energy': val, 'conserved_momentum': val}
            # This might need to be based on the full self.env_data
            self.ground_truth_trajectory_processed = self.physics_env.get_ground_truth_conserved_quantities(self.env_data, self.variables)
        else:
            # Fallback: This might be just the raw trajectory data, and conservation_reward_fix
            # would need to infer or calculate conserved quantities from it.
            # For simplicity, let's assume env_data itself can be used by compute_conservation_bonus
            # if it expects values like energy, momentum directly.
            # The compute_conservation_bonus in conservation_reward_fix.py expects dicts.
            # Let's assume self.env_data is a numpy array. We need to structure it.
            # This part is complex and depends on how conserved quantities are defined and extracted.
            # Placeholder:
            self.ground_truth_trajectory_processed = {'raw_data': self.env_data}
            # Also create sympy_vars for lambdify
        if self.variables:
            self.sympy_vars = [symbols(v.name) for v in self.variables]

        logging.info(f"[{self.config.name}] Setup phase complete.")
        self.live_monitor.start_monitoring()
        logging.info(f"[{self.config.name}] Live monitor started.")

    def _get_initial_conditions(self) -> np.ndarray:
        """Helper to determine initial conditions based on environment type."""
        if not self.physics_env: # Should not happen if setup order is correct
            raise RuntimeError("Physics environment not initialized in _get_initial_conditions.")

        env_type = self.config.environment_type
        if env_type == 'harmonic_oscillator':
            # Example: random position and velocity, scaled
            return np.random.randn(2) * np.array([self.config.env_params.get('x_scale', 1.0),
                                                  self.config.env_params.get('v_scale', 2.0)])
        elif env_type == 'pendulum':
            max_angle = np.pi / 2 if self.config.env_params.get('small_angle', False) else np.pi
            # Ensure max_angle is positive if using uniform distribution this way
            max_angle_abs = abs(max_angle)
            return np.array([np.random.uniform(-max_angle_abs, max_angle_abs),
                             np.random.uniform(-1.0, 1.0)]) # theta, omega
        elif env_type == 'kepler':
            # Example: random eccentricity and semi-major axis to define orbit
            ecc = np.random.uniform(0.0, 0.7) # Limit eccentricity
            sma = np.random.uniform(0.5, 2.0) # Semi-major axis
            r_periapsis = sma * (1 - ecc) # Periapsis distance

            # Need G, M from env to calculate velocity at periapsis
            G = self.physics_env.params.get('G', 1.0)
            M = self.physics_env.params.get('M', 1.0)
            if G * M <= 0 or r_periapsis <=0 or sma <= 0: # Avoid math errors
                 v_periapsis_tangential = 1.0 # Fallback velocity
            else:
                # Velocity at periapsis (tangential, specific to polar coords where v_r=0 at periapsis)
                v_periapsis_tangential = np.sqrt(G * M * (2 / r_periapsis - 1 / sma))

            # Initial state: [r, theta_angle, vr, v_theta_angular_speed]
            # Start at periapsis (theta=0), radial velocity is zero.
            return np.array([r_periapsis, 0.0, 0.0, v_periapsis_tangential / r_periapsis if r_periapsis > 0 else 1.0])
        else:
            # Generic fallback: random initial conditions for number of state vars
            num_state_vars = len(self.physics_env.state_vars) if self.physics_env.state_vars else 2
            return np.random.rand(num_state_vars) * 2 - 1 # Random values in [-1, 1]


    def run(self, run_id: int) -> ExperimentResult:
        """Executes the core symbolic discovery algorithm.

        This method delegates to algorithm-specific logic based on
        `self.config.algorithm`. It handles training the algorithm,
        extracting the discovered law, and calculating performance metrics
        like predictive MSE and symbolic accuracy.

        Args:
            run_id: The identifier for the current run.

        Returns:
            An `ExperimentResult` object populated with the outcomes of this
            run. If the algorithm fails or is not supported, an error state
            is reflected in the result.
        """
        logging.info(f"[{self.config.name}] Starting experiment run phase (run_id: {run_id}). Algorithm: {self.config.algorithm}")
        current_run_result = ExperimentResult(config=self.config, run_id=run_id)
        if self.env_data is not None:
             current_run_result.trajectory_data = self.env_data # From setup

        if self.algorithm is None: # Should be caught by setup, but defensive check
            logging.error(f"[{self.config.name}] Algorithm not initialized. Aborting run.")
            current_run_result.discovered_law = "Error: Algorithm not initialized"
            current_run_result.symbolic_accuracy = 0.0
            current_run_result.predictive_mse = float('inf')
            return current_run_result

        # --- Algorithm-specific execution ---
        algo_name = self.config.algorithm
        try:
            if algo_name.startswith('janus'): # Example: Janus PPO-based
                trainer = self.algorithm # Assuming self.algorithm is the PPO trainer
                num_evals = self.config.algo_params.get('num_evaluations', 50)
                steps_per_eval = self.config.algo_params.get('timesteps_per_eval_cycle', 1000)
                ppo_params = self.config.algo_params.get('ppo_train_params', {})

                best_mse_overall = float('inf')
                best_expr_overall = None
                efficiency_curve = []
                current_total_steps = 0 # Initialize current_total_steps

                for i in range(num_evals):
                    trainer.train(total_timesteps=steps_per_eval, **ppo_params)
                    current_total_steps += steps_per_eval # Accumulate steps

                    current_step_in_loop = i # Or map to global step if available

                    # New hypothesis recording logic
                    if hasattr(trainer, 'env') and hasattr(trainer.env, '_evaluation_cache') and self.training_integration:
                        # Ideally, SymbolicDiscoveryEnv would call on_hypothesis_evaluated directly.
                        # For now, we process its cache here as a bridge.
                        # This cache should ideally be cleared by the env after each cycle if we process it like this.
                        for eval_entry in trainer.env._evaluation_cache: # This cache might contain more than just the last cycle
                            hyp_data = eval_entry.get('expression_obj', {'expression_str': eval_entry.get('expression')}) # Adapt as per actual cache structure

                            # We need to assemble evaluation_results for on_hypothesis_evaluated
                            # The old cache had 'mse'. We need performance_score, conservation_score, etc.
                            # This part requires that the environment's evaluation cache is richer,
                            # or these scores are computed here.
                            # For now, let's assume a basic structure. This will be refined in Step 6.
                            eval_results_for_tracker = {
                                'performance_score': -eval_entry.get('mse', float('inf')), # Assuming performance is inverse of MSE
                                'conservation_score': eval_entry.get('conservation_score', 0.0), # Needs to be populated by env
                                'symmetry_score': eval_entry.get('symmetry_score', 0.0), # Needs to be populated by env
                                'trajectory_fit': eval_entry.get('mse', float('inf')),
                                'functional_form': str(eval_entry.get('expression'))
                            }

                            self.training_integration.on_hypothesis_evaluated(
                                hypothesis_data=hyp_data,
                                evaluation_results=eval_results_for_tracker,
                                step=current_total_steps, # Global step
                                episode=current_step_in_loop # Current eval cycle as episode
                            )
                        if hasattr(trainer.env, 'clear_evaluation_cache'): # Ideal: trainer.env.clear_evaluation_cache()
                            pass # trainer.env.clear_evaluation_cache() # Call if available


                    # Update best_expr_overall and best_mse_overall using the tracker
                    if self.training_integration:
                        best_hyp = self.training_integration.get_best_discovered_law(criterion='overall')
                        if best_hyp:
                            best_expr_overall = best_hyp['hypothesis_data'] # Or specific field like 'expression_str'
                            # Assuming performance_score in tracker is primary metric, might be -MSE
                            eval_res = best_hyp['evaluation_results']
                            best_mse_overall = eval_res.get('trajectory_fit', -eval_res.get('performance_score', float('inf')))

                    efficiency_curve.append((current_total_steps, best_mse_overall))
                    # The rest of the loop continues...

                # After the loop, populate current_run_result from the tracker's best
                if self.training_integration:
                    final_best_hyp_after_loop = self.training_integration.get_best_discovered_law(criterion='overall')
                    if final_best_hyp_after_loop:
                        current_run_result.discovered_law = str(final_best_hyp_after_loop['hypothesis_data'].get('expression_str', final_best_hyp_after_loop['hypothesis_data']))
                        eval_res_final = final_best_hyp_after_loop['evaluation_results']
                        current_run_result.predictive_mse = eval_res_final.get('trajectory_fit', -eval_res_final.get('performance_score', float('inf')))
                        current_run_result.law_complexity = len(str(current_run_result.discovered_law)) # Simplified
                        current_run_result.component_metrics['conservation_score'] = eval_res_final.get('conservation_score')
                        current_run_result.component_metrics['symmetry_score'] = eval_res_final.get('symmetry_score')
                    else: # Fallback if tracker is empty
                        current_run_result.discovered_law = str(best_expr_overall) if best_expr_overall else None
                        current_run_result.predictive_mse = best_mse_overall

                current_run_result.sample_efficiency_curve = efficiency_curve
                current_run_result.n_experiments_to_convergence = num_evals # Or steps, depending on definition

            elif algo_name == 'genetic': # Example: SymbolicRegressor
                regressor = self.algorithm
                target_idx = self.config.target_variable_index if self.config.target_variable_index is not None else -1
                if self.env_data is None: raise ValueError("env_data is None for genetic algorithm")
                y = self.env_data[:, target_idx]
                X = np.delete(self.env_data, target_idx, axis=1)

                # Variables might need filtering if target column was removed and they weren't already
                # Assuming self.variables are suitable for X
                fit_params = self.config.algo_params.get('fit_params', {}) # e.g. generations, pop_size
                best_expr_obj = regressor.fit(X, y, self.variables, **fit_params)

                if best_expr_obj and hasattr(best_expr_obj, 'symbolic'):
                    current_run_result.discovered_law = str(best_expr_obj.symbolic)
                    current_run_result.law_complexity = getattr(best_expr_obj, 'complexity', len(str(best_expr_obj.symbolic)))
                    # MSE calculation:
                    if hasattr(best_expr_obj, 'mse') and best_expr_obj.mse is not None:
                        current_run_result.predictive_mse = best_expr_obj.mse
                    elif hasattr(regressor, 'predict'):
                        predictions = regressor.predict(X)
                        current_run_result.predictive_mse = np.mean((predictions - y)**2)
                    else: # Fallback for MSE needed
                        current_run_result.predictive_mse = float('inf')
                else:
                    current_run_result.discovered_law = "Error: Genetic algorithm failed."
                    current_run_result.predictive_mse = float('inf')
                current_run_result.n_experiments_to_convergence = getattr(regressor, 'generations', 1) # Example

            elif algo_name == 'random':
                # Placeholder for random search
                current_run_result.discovered_law = "random_expr_placeholder"
                current_run_result.predictive_mse = np.random.rand() * 10 # Dummy MSE
                current_run_result.law_complexity = 10
                current_run_result.n_experiments_to_convergence = self.config.algo_params.get('num_random_expressions', 100)
                logging.warning(f"[{self.config.name}] Random search is a placeholder.")

            else:
                logging.error(f"[{self.config.name}] Unknown algorithm '{algo_name}' in run method.")
                current_run_result.discovered_law = f"Error: Unknown algorithm '{algo_name}'"
                return current_run_result # Early exit for unknown algorithm

            # Calculate symbolic accuracy
            if self.ground_truth_laws and current_run_result.discovered_law:
                current_run_result.symbolic_accuracy = calculate_symbolic_accuracy(
                    current_run_result.discovered_law, self.ground_truth_laws
                )
            else:
                current_run_result.symbolic_accuracy = 0.0
                logging.debug(f"[{self.config.name}] Symbolic accuracy not computed (no ground truth or no law discovered).")

        except Exception as e:
            logging.error(f"[{self.config.name}] Exception during algorithm execution ('{algo_name}'): {e}", exc_info=True)
            current_run_result.discovered_law = f"Error during {algo_name}: {str(e)[:100]}" # Truncate long errors
            current_run_result.symbolic_accuracy = 0.0
            current_run_result.predictive_mse = float('inf')

        logging.info(f"[{self.config.name}] Run phase finished. Law: '{current_run_result.discovered_law}', Accuracy: {current_run_result.symbolic_accuracy:.4f}, MSE: {current_run_result.predictive_mse:.4e}")
        return current_run_result

    def teardown(self):
        """Cleans up resources used by the experiment.

        This method is called after the experiment execution (even if errors
        occurred) to release resources, close files, or terminate any child
        processes. For this example, it primarily logs the completion and
        resets internal state variables.
        """
        logging.info(f"[{self.config.name}] Tearing down experiment.")
        # Example: if self.algorithm and hasattr(self.algorithm, 'close'): self.algorithm.close()
        self.physics_env = None
        self.env_data = None
        self.variables = None
        self.algorithm = None
        self.ground_truth_laws = None
        logging.info(f"[{self.config.name}] Teardown complete.")

    def execute(self, run_id: int = 0) -> ExperimentResult:
        """Orchestrates the full lifecycle of the experiment: setup, run, and teardown.

        This method overrides `BaseExperiment.execute`. It measures the total
        wall-clock time for the experiment and ensures that `setup`, `run`, and
        `teardown` are called in sequence. It manages an `ExperimentResult`
        object that is populated by the `run` method.

        Args:
            run_id: The identifier for this specific execution run.

        Returns:
            An `ExperimentResult` object containing all outcomes and data from
            the experiment. If critical failures occur, the result object will
            reflect an error state.
        """
        self._start_time = time.time()
        # Initialize a result object for this execution.
        # It will be further populated by the 'run' method.
        self.experiment_result = ExperimentResult(config=self.config, run_id=run_id)

        try:
            logging.info(f"[{self.config.name}] EXECUTE: Starting setup (run {run_id})...")
            self.setup()
            logging.info(f"[{self.config.name}] EXECUTE: Setup complete (run {run_id}).")

            # If setup was successful, env_data might be available to store early
            if self.env_data is not None:
                self.experiment_result.trajectory_data = self.env_data

            logging.info(f"[{self.config.name}] EXECUTE: Starting run (run {run_id})...")
            # The 'run' method is responsible for creating and returning its own ExperimentResult,
            # which we then adopt as the main result of this execution.
            run_specific_result = self.run(run_id=run_id)

            # Merge results from run_specific_result into self.experiment_result
            # This ensures that even if run_specific_result is a new instance,
            # its data is captured in the experiment_result managed by 'execute'.
            if run_specific_result: # run_specific_result should always be an ExperimentResult
                self.experiment_result.discovered_law = run_specific_result.discovered_law
                self.experiment_result.symbolic_accuracy = run_specific_result.symbolic_accuracy
                self.experiment_result.predictive_mse = run_specific_result.predictive_mse
                self.experiment_result.law_complexity = run_specific_result.law_complexity
                self.experiment_result.n_experiments_to_convergence = run_specific_result.n_experiments_to_convergence
                self.experiment_result.sample_efficiency_curve = run_specific_result.sample_efficiency_curve
                self.experiment_result.component_metrics = run_specific_result.component_metrics
                if run_specific_result.trajectory_data is not None: # Ensure trajectory data is kept if run updated it
                    self.experiment_result.trajectory_data = run_specific_result.trajectory_data
            else: # Should not happen if run() is implemented correctly
                 logging.error(f"[{self.config.name}] EXECUTE: run() method returned None. This is unexpected.")
                 self.experiment_result.discovered_law = "Error: run() returned None"


            logging.info(f"[{self.config.name}] EXECUTE: Run complete (run {run_id}). Discovered: '{self.experiment_result.discovered_law}'")

        except Exception as e:
            logging.error(f"[{self.config.name}] EXECUTE: Exception during setup or run (run {run_id}): {e}", exc_info=True)
            if self.experiment_result: # Ensure it exists
                self.experiment_result.discovered_law = f"Critical Error: {str(e)[:150]}" # Truncate long errors
                self.experiment_result.symbolic_accuracy = 0.0
                self.experiment_result.predictive_mse = float('inf')
            # Depending on desired behavior, could re-raise e here
        finally:
            current_wall_time = time.time() - self._start_time
            if self.experiment_result:
                self.experiment_result.wall_time_seconds = current_wall_time

            logging.info(f"[{self.config.name}] EXECUTE: Starting teardown and cleanup (run {run_id}). Wall time: {current_wall_time:.2f}s.")
            self.cleanup() # Call new cleanup method
            self.teardown() # Call original teardown
            logging.info(f"[{self.config.name}] EXECUTE: Teardown and cleanup complete (run {run_id}).")

        # Final check to ensure an ExperimentResult object is always returned.
        if not self.experiment_result:
            logging.critical(f"[{self.config.name}] EXECUTE: self.experiment_result is None at the end. This indicates a major issue.")
            # Create a minimal error result if it's somehow still None
            self.experiment_result = ExperimentResult(config=self.config, run_id=run_id,
                                                      discovered_law="Critical error: experiment_result not formed.")
            self.experiment_result.wall_time_seconds = time.time() - self._start_time

        return self.experiment_result


if __name__ == "__main__":

    This class orchestrates the setup, execution, and teardown of a specific
    type of physics discovery experiment. It integrates with the ExperimentRunner's
    registries for environments and algorithms.

    Attributes:
        config (ExperimentConfig): Configuration object for the experiment.
        algo_registry (Dict[str, Callable]): Registry of available algorithms.
        env_registry (Dict[str, Callable]): Registry of available environments.
        physics_env (Optional[PhysicsEnvironment]): The instantiated physics
            environment for the experiment.
        env_data (Optional[np.ndarray]): Data generated from the physics environment.
        variables (Optional[List[Any]]): List of variables for the symbolic system,
            compatible with `progressive_grammar_system`.
        algorithm (Optional[Any]): The instantiated discovery algorithm.
        ground_truth_laws (Optional[Dict[str, sp.Expr]]): Ground truth equations
            for the environment.
        experiment_result (Optional[ExperimentResult]): Holds the results of the
            experiment execution.
        _start_time (float): Internal timer for wall-clock time measurement.
    """
    def __init__(self,
                 config: ExperimentConfig,
                 algo_registry: Dict[str, Callable],
                 env_registry: Dict[str, Callable]):
        """Initializes the PhysicsDiscoveryExperiment.

        Args:
            config: The configuration object for this experiment.
            algo_registry: A dictionary mapping algorithm names to their
                factory functions.
            env_registry: A dictionary mapping environment names to their
                class constructors.
        """
        super().__init__() # Initialize BaseExperiment
        self.config = config
        self.algo_registry = algo_registry
        self.env_registry = env_registry
        self.hypothesis_tracker: Optional[HypothesisTracker] = None
        self.training_integration: Optional[JanusTrainingIntegration] = None
        # Logger and monitor will be added in a later step for Gap 3 integration
        self.training_logger: Optional[Any] = None # Placeholder for TrainingLogger
        self.live_monitor: Optional[Any] = None # Placeholder for LiveMonitor


        # Attributes to be populated by setup()
        self.physics_env: Optional[PhysicsEnvironment] = None
        self.env_data: Optional[np.ndarray] = None
        self.variables: Optional[List[Any]] = None
        self.algorithm: Optional[Any] = None
        self.ground_truth_laws: Optional[Dict[str, sp.Expr]] = None

        # Result object, to be populated by execute() -> run()
        self.experiment_result: Optional[ExperimentResult] = None
        self._start_time: float = 0.0 # For timing the experiment run

    def setup(self):
        """Sets up the experiment environment, data generation, and algorithm instantiation.

        This method performs the following steps:
        1. Sets random seeds for reproducibility.
        2. Instantiates the physics environment based on `config.environment_type`.
        3. Generates trajectory data from the environment.
        4. Defines variables for the symbolic discovery process.
        5. Instantiates the discovery algorithm based on `config.algorithm`.
        6. Retrieves ground truth laws from the environment.

        Raises:
            ValueError: If specified environment or algorithm is not found in
                their respective registries, or if data generation fails.
            ImportError: If essential components like `Variable` from
                `progressive_grammar_system` cannot be imported.
        """
        logging.info(f"[{self.config.name}] Initializing setup: Seed {self.config.seed}, Env '{self.config.environment_type}', Algo '{self.config.algorithm}'.")
        np.random.seed(self.config.seed)
        if 'torch' in globals(): # Check if torch is imported
            torch.manual_seed(self.config.seed)

        # 1. Create Environment
        logging.debug(f"[{self.config.name}] Creating environment '{self.config.environment_type}'.")
        env_class = self.env_registry.get(self.config.environment_type)
        if not env_class:
            logging.error(f"Environment type '{self.config.environment_type}' not in registry. Available: {list(self.env_registry.keys())}")
            # RAISE PluginNotFoundError
            raise PluginNotFoundError(f"Environment type '{self.config.environment_type}' not found in registry. Available: {list(self.env_registry.keys())}")
        self.physics_env = env_class(self.config.env_params, self.config.noise_level)
        logging.info(f"[{self.config.name}] Environment '{self.config.environment_type}' created: {self.physics_env}")

        # 2. Generate Training Data
        logging.debug(f"[{self.config.name}] Generating data: {self.config.n_trajectories} trajectories, {self.config.trajectory_length} length.")
        trajectories = []
        for _ in range(self.config.n_trajectories):
            # Initial conditions can be customized per environment
            init_cond = self._get_initial_conditions()
            t_span = np.arange(0, self.config.trajectory_length * self.config.sampling_rate, self.config.sampling_rate)
            if self.physics_env: # Should be true due to check above
                trajectory = self.physics_env.generate_trajectory(init_cond, t_span)
                trajectories.append(trajectory)

        if not trajectories:
            logging.error(f"[{self.config.name}] No trajectories generated.")
            # RAISE DataGenerationError
            raise DataGenerationError(f"No trajectories generated for experiment '{self.config.name}'. Check environment parameters and generation logic.")
        self.env_data = np.vstack(trajectories)
        logging.info(f"[{self.config.name}] Training data generated with shape {self.env_data.shape}.")

        # 3. Create Variables for Algorithm
        try:
            from progressive_grammar_system import Variable # type: ignore
        except ImportError as e:
            logging.error("Failed to import 'Variable' from 'progressive_grammar_system'. Ensure it's installed and accessible.")
            # RAISE MissingDependencyError
            raise MissingDependencyError("The 'Variable' class from 'progressive_grammar_system' could not be imported. Please ensure the package is installed correctly.") from e
        if self.physics_env and self.physics_env.state_vars:
            self.variables = [Variable(name, idx, {}) for idx, name in enumerate(self.physics_env.state_vars)]
        else:
            # Fallback if state_vars are not defined, though this indicates an issue with the env
            num_cols = self.env_data.shape[1]
            self.variables = [Variable(f'var{idx}', idx, {}) for idx in range(num_cols)]
            logging.warning(f"[{self.config.name}] PhysicsEnv.state_vars not defined. Using generic variable names based on data columns.")

        logging.debug(f"[{self.config.name}] State variables for algorithm: {[str(v) for v in self.variables]}")


        # 4. Create Algorithm Instance
        logging.debug(f"[{self.config.name}] Creating algorithm '{self.config.algorithm}'.")
        algo_factory = self.algo_registry.get(self.config.algorithm)
        if not algo_factory:
            logging.error(f"Algorithm '{self.config.algorithm}' not in registry. Available: {list(self.algo_registry.keys())}")
            # RAISE PluginNotFoundError
            raise PluginNotFoundError(f"Algorithm '{self.config.algorithm}' not found in registry. Available: {list(self.algo_registry.keys())}")
        try:
            self.algorithm = algo_factory(self.env_data, self.variables, self.config)
        except (TypeError, ValueError) as e: # Catch issues if algo_factory fails due to bad config from self.config
            logging.error(f"Failed to create algorithm '{self.config.algorithm}' due to configuration or parameter error: {e}", exc_info=True)
            raise InvalidConfigError(f"Failed to create algorithm '{self.config.algorithm}' with the provided configuration: {e}") from e
        logging.info(f"[{self.config.name}] Algorithm '{self.config.algorithm}' created: {type(self.algorithm).__name__}")

        # Initialize Hypothesis Tracker and Training Integration
        # save_directory can be made more configurable if needed
        tracker_save_dir = Path(self.config.base_dir if hasattr(self.config, 'base_dir') else "./experiments") / self.config.name / "hypothesis_tracking"
        self.hypothesis_tracker = HypothesisTracker(
            save_directory=str(tracker_save_dir),
            autosave_interval=self.config.algo_params.get('tracker_autosave_interval', 100)
        )
        self.training_integration = JanusTrainingIntegration(self.hypothesis_tracker)
        logging.info(f"[{self.config.name}] HypothesisTracker initialized. Save dir: {tracker_save_dir}")

        # 5. Get Ground Truth Laws
        if self.physics_env:
            self.ground_truth_laws = self.physics_env.get_ground_truth_laws()
            logging.debug(f"[{self.config.name}] Ground truth laws obtained: {list(self.ground_truth_laws.keys()) if self.ground_truth_laws else 'None'}.")
        logging.info(f"[{self.config.name}] Setup phase complete.")

    def _get_initial_conditions(self) -> np.ndarray:
        """Helper to determine initial conditions based on environment type."""
        if not self.physics_env: # Should not happen if setup order is correct
            raise RuntimeError("Physics environment not initialized in _get_initial_conditions.")

        env_type = self.config.environment_type
        if env_type == 'harmonic_oscillator':
            # Example: random position and velocity, scaled
            return np.random.randn(2) * np.array([self.config.env_params.get('x_scale', 1.0),
                                                  self.config.env_params.get('v_scale', 2.0)])
        elif env_type == 'pendulum':
            max_angle = np.pi / 2 if self.config.env_params.get('small_angle', False) else np.pi
            # Ensure max_angle is positive if using uniform distribution this way
            max_angle_abs = abs(max_angle)
            return np.array([np.random.uniform(-max_angle_abs, max_angle_abs),
                             np.random.uniform(-1.0, 1.0)]) # theta, omega
        elif env_type == 'kepler':
            # Example: random eccentricity and semi-major axis to define orbit
            ecc = np.random.uniform(0.0, 0.7) # Limit eccentricity
            sma = np.random.uniform(0.5, 2.0) # Semi-major axis
            r_periapsis = sma * (1 - ecc) # Periapsis distance

            # Need G, M from env to calculate velocity at periapsis
            G = self.physics_env.params.get('G', 1.0)
            M = self.physics_env.params.get('M', 1.0)
            if G * M <= 0 or r_periapsis <=0 or sma <= 0: # Avoid math errors
                 v_periapsis_tangential = 1.0 # Fallback velocity
            else:
                # Velocity at periapsis (tangential, specific to polar coords where v_r=0 at periapsis)
                v_periapsis_tangential = np.sqrt(G * M * (2 / r_periapsis - 1 / sma))

            # Initial state: [r, theta_angle, vr, v_theta_angular_speed]
            # Start at periapsis (theta=0), radial velocity is zero.
            return np.array([r_periapsis, 0.0, 0.0, v_periapsis_tangential / r_periapsis if r_periapsis > 0 else 1.0])
        else:
            # Generic fallback: random initial conditions for number of state vars
            num_state_vars = len(self.physics_env.state_vars) if self.physics_env.state_vars else 2
            return np.random.rand(num_state_vars) * 2 - 1 # Random values in [-1, 1]


    def run(self, run_id: int) -> ExperimentResult:
        """Executes the core symbolic discovery algorithm.

        This method delegates to algorithm-specific logic based on
        `self.config.algorithm`. It handles training the algorithm,
        extracting the discovered law, and calculating performance metrics
        like predictive MSE and symbolic accuracy.

        Args:
            run_id: The identifier for the current run.

        Returns:
            An `ExperimentResult` object populated with the outcomes of this
            run. If the algorithm fails or is not supported, an error state
            is reflected in the result.
        """
        logging.info(f"[{self.config.name}] Starting experiment run phase (run_id: {run_id}). Algorithm: {self.config.algorithm}")
        current_run_result = ExperimentResult(config=self.config, run_id=run_id)
        if self.env_data is not None:
             current_run_result.trajectory_data = self.env_data # From setup

        if self.algorithm is None: # Should be caught by setup, but defensive check
            logging.error(f"[{self.config.name}] Algorithm not initialized. Aborting run.")
            current_run_result.discovered_law = "Error: Algorithm not initialized"
            current_run_result.symbolic_accuracy = 0.0
            current_run_result.predictive_mse = float('inf')
            return current_run_result

        # --- Algorithm-specific execution ---
        algo_name = self.config.algorithm
        try:
            if algo_name.startswith('janus'): # Example: Janus PPO-based
                trainer = self.algorithm
                num_eval_cycles = self.config.algo_params.get('num_evaluation_cycles', 50) # Renamed
                steps_per_cycle = self.config.algo_params.get('timesteps_per_eval_cycle', 1000)
                ppo_params = self.config.algo_params.get('ppo_train_params', {})

                efficiency_curve = [] # Remains for tracking overall best MSE per cycle
                current_total_steps = 0
                # best_mse_overall will be derived from tracker for efficiency_curve later

                for cycle_idx in range(num_eval_cycles): # Renamed loop variable
                    trainer.train(total_timesteps=steps_per_cycle, **ppo_params)
                    current_total_steps += steps_per_cycle

                    evaluated_hypotheses_in_cycle = []
                    if hasattr(trainer, 'env') and hasattr(trainer.env, '_evaluation_cache'):
                        evaluated_hypotheses_in_cycle = list(trainer.env._evaluation_cache)
                        if hasattr(trainer.env, 'clear_evaluation_cache'):
                            trainer.env.clear_evaluation_cache()

                    for hyp_eval_entry in evaluated_hypotheses_in_cycle:
                        hypothesis_object = hyp_eval_entry.get('expression_obj')
                        if not isinstance(hypothesis_object, SymbolicExpression): # Check type
                            hyp_str = hyp_eval_entry.get('expression', '')
                            # Attempt to parse if it's a string and grammar is available
                            # This is a fallback, ideally SymbolicDiscoveryEnv provides SymbolicExpression objects
                            if hasattr(self, 'grammar_system_instance_if_needed'): # Placeholder for actual grammar access
                                try:
                                    # Assuming a parse method on a grammar instance
                                    # hypothesis_object = self.grammar_system_instance_if_needed.parse(hyp_str)
                                    pass # Actual parsing logic depends on grammar system
                                except Exception as e:
                                    logging.warning(f"[{self.config.name}] Could not parse hypothesis string '{hyp_str}': {e}")
                                    # If parsing fails or not possible, skip
                                    if not isinstance(hypothesis_object, SymbolicExpression):
                                        logging.warning(f"[{self.config.name}] Skipping hypothesis due to missing or unparsable object: {hyp_str}")
                                        continue
                            else:
                                logging.warning(f"[{self.config.name}] Skipping hypothesis with no valid object: {hyp_eval_entry.get('expression')}")
                                continue

                        law_params = {}
                        if hasattr(hypothesis_object, 'get_parameters'):
                            law_params = hypothesis_object.get_parameters()
                        elif hasattr(hypothesis_object, 'params'):
                            law_params = hypothesis_object.params

                        callable_law_function = None
                        if self.sympy_vars and hasattr(hypothesis_object, 'symbolic') and hypothesis_object.symbolic:
                            try:
                                callable_law_function = lambdify(self.sympy_vars, hypothesis_object.symbolic, modules=['numpy', 'sympy'])
                            except Exception as e:
                                logging.error(f"[{self.config.name}] Error lambdifying expression {hypothesis_object.symbolic}: {e}")

                        # --- Conservation Bonus Calculation (Sub-task 6.4) ---
                        conservation_bonus = 0.0
                        if self.env_data is not None and self.ground_truth_trajectory_processed is not None:
                            # Placeholder for predicted_traj. A real implementation would evaluate
                            # the hypothesis_object over a trajectory to get its predicted conserved quantities.
                            # This placeholder passes a simplified dict. The actual values derived from
                            # the hypothesis (e.g. its own calculated energy/momentum) are needed here.
                            # This structure must align with what conservation_reward_fix.py expects.
                            placeholder_pred_traj_dict = {
                                f'conserved_{c_type}': None for c_type in self.conservation_reward.conservation_types
                            }
                            # Example: if hypothesis_object could evaluate its own energy:
                            # placeholder_pred_traj_dict['conserved_energy'] = hypothesis_object.calculate_energy(self.env_data, law_params)
                            # This kind of evaluation is deferred.

                            try:
                                conservation_bonus = self.conservation_reward.compute_conservation_bonus(
                                    predicted_traj=placeholder_pred_traj_dict,
                                    ground_truth_traj=self.ground_truth_trajectory_processed,
                                    hypothesis_params=law_params
                                )
                                logging.debug(f"[{self.config.name}] Cycle {cycle_idx}: Hypothesis {hypothesis_object.symbolic}, Conservation Bonus: {conservation_bonus:.4f}")
                            except Exception as e:
                                logging.error(f"[{self.config.name}] Error computing conservation bonus for {hypothesis_object.symbolic}: {e}")

                        # --- Symmetry Detection (Sub-task 6.5) ---
                        symmetry_results = {}
                        symmetry_score = 0.0
                        if callable_law_function and self.env_data is not None:
                            try:
                                # Use a sample of env_data for symmetry checks if env_data is large
                                sample_size = min(100, self.env_data.shape[0])
                                sample_indices = np.random.choice(self.env_data.shape[0], size=sample_size, replace=False)
                                sample_traj_for_symmetry = self.env_data[sample_indices]

                                # Ensure sample_traj_for_symmetry has enough dimensions for symmetry detector if it expects specific shapes
                                # The current symmetry_detection_fix.py _check_velocity_parity expects state_vector [pos, vel, ...]
                                # This implies self.env_data columns need to align with this.
                                # This alignment is a deeper issue related to environment data structure.
                                # For now, we pass the sample as is.

                                symmetry_results = self.symmetry_detector.detect_all_symmetries(
                                    law_function=callable_law_function,
                                    trajectory=sample_traj_for_symmetry,
                                    params=law_params
                                )
                                symmetry_score = self.symmetry_detector.symmetry_guided_score(
                                    law_function=callable_law_function,
                                    trajectory=sample_traj_for_symmetry,
                                    params=law_params,
                                    expected_symmetries=self.config.algo_params.get('expected_symmetries', ['velocity_parity', 'time_reversal'])
                                )
                                logging.debug(f"[{self.config.name}] Cycle {cycle_idx}: Hypothesis {hypothesis_object.symbolic}, Symmetry Score: {symmetry_score:.4f}, Results: {symmetry_results}")
                            except Exception as e:
                                logging.error(f"[{self.config.name}] Error during symmetry detection for {hypothesis_object.symbolic}: {e}")

                        # --- Hypothesis Recording and Logging (Sub-task 6.6) ---
                        base_performance_score = -hyp_eval_entry.get('mse', float('inf')) # Higher is better
                        trajectory_fit_error = hyp_eval_entry.get('mse', float('inf')) # Lower is better
                        current_complexity = getattr(hypothesis_object, 'complexity', len(str(hypothesis_object.symbolic)))

                        if self.training_integration:
                            self.training_integration.on_hypothesis_evaluated(
                                hypothesis_data={ # Store rich hypothesis data
                                    'expression_str': str(hypothesis_object.symbolic),
                                    'sympy_expr': str(hypothesis_object.symbolic), # Assuming .symbolic is sympy compatible string
                                    'object_type': type(hypothesis_object).__name__,
                                    # Potentially store a serialized version of hypothesis_object if feasible and small
                                },
                                evaluation_results={
                                    'performance_score': base_performance_score, # Overall score, higher is better
                                    'conservation_score': conservation_bonus,    # Higher is better
                                    'symmetry_score': symmetry_score,            # Higher is better
                                    'trajectory_fit': trajectory_fit_error,    # Error metric, lower is better
                                    'functional_form': str(hypothesis_object.symbolic),
                                    'complexity': current_complexity
                                },
                                step=current_total_steps,
                                episode=cycle_idx,
                                training_context={'environment': self.config.environment_type, 'algorithm_cycle': cycle_idx}
                            )

                        if self.training_logger:
                            best_overall_from_tracker = self.training_integration.get_best_discovered_law(criterion='overall') if self.training_integration else None
                            best_overall_score_for_log = best_overall_from_tracker['evaluation_results']['performance_score'] if best_overall_from_tracker else -float('inf')

                            # Placeholder for entropy production - this would require more complex calculation
                            entropy_production_val = 0.0

                            self.training_logger.log_metrics(
                                step=current_total_steps,
                                episode=cycle_idx,
                                reward_components_sum=base_performance_score + conservation_bonus + symmetry_score, # Example composite reward
                                base_performance=base_performance_score,
                                conservation_bonus=conservation_bonus,
                                symmetry_score=symmetry_score,
                                current_hypothesis_trajectory_fit=trajectory_fit_error,
                                current_hypothesis_complexity=current_complexity,
                                discovered_law_params=law_params,
                                best_overall_score_tracker=best_overall_score_for_log,
                                entropy_production=entropy_production_val
                                # Add more specific metrics from symmetry_results if needed, e.g. symmetry_results.get('translation_score',0)
                            )

                    # --- End of loop for evaluated_hypotheses_in_cycle ---

                    # Update efficiency curve based on overall best from tracker after this cycle
                    if self.training_integration:
                        best_hyp_for_curve = self.training_integration.get_best_discovered_law(criterion='overall')
                        if best_hyp_for_curve:
                            eval_res_curve = best_hyp_for_curve['evaluation_results']
                            # Use trajectory_fit (lower is better) for efficiency curve tracking MSE
                            mse_for_curve = eval_res_curve.get('trajectory_fit', float('inf'))
                            efficiency_curve.append((current_total_steps, mse_for_curve))
                        elif efficiency_curve:
                            efficiency_curve.append((current_total_steps, efficiency_curve[-1][1]))
                        else:
                             efficiency_curve.append((current_total_steps, float('inf')))
                    else:
                        efficiency_curve.append((current_total_steps, float('inf')))

                # Final result population using the tracker
                if self.training_integration:
                    final_best_hyp = self.training_integration.get_best_discovered_law(criterion='overall')
                    if final_best_hyp:
                        hyp_data_final = final_best_hyp['hypothesis_data']
                        # Extract expression string, preferring 'expression_str' or 'sympy_expr' if available
                        if isinstance(hyp_data_final, dict):
                            current_run_result.discovered_law = str(hyp_data_final.get('expression_str', hyp_data_final.get('sympy_expr', str(hyp_data_final))))
                        else: # Fallback if hyp_data_final is not a dict (e.g., the object itself)
                            current_run_result.discovered_law = str(hyp_data_final)

                        eval_res_final = final_best_hyp['evaluation_results']
                        current_run_result.predictive_mse = eval_res_final.get('trajectory_fit', -eval_res_final.get('performance_score', float('inf')))
                        current_run_result.law_complexity = eval_res_final.get('complexity', len(str(current_run_result.discovered_law)))
                        current_run_result.component_metrics['conservation_score'] = eval_res_final.get('conservation_score')
                        current_run_result.component_metrics['symmetry_score'] = eval_res_final.get('symmetry_score')
                    else: # If tracker is empty or has no 'overall' best
                        current_run_result.discovered_law = None
                        current_run_result.predictive_mse = float('inf') # Keep as inf if no law found
                else: # Fallback if no training_integration
                    current_run_result.discovered_law = None
                    current_run_result.predictive_mse = float('inf')

                current_run_result.sample_efficiency_curve = efficiency_curve
                current_run_result.n_experiments_to_convergence = num_eval_cycles

            elif algo_name == 'genetic': # Example: SymbolicRegressor
                regressor = self.algorithm
                target_idx = self.config.target_variable_index if self.config.target_variable_index is not None else -1
                if self.env_data is None: raise ValueError("env_data is None for genetic algorithm")
                y = self.env_data[:, target_idx]
                X = np.delete(self.env_data, target_idx, axis=1)

                # Variables might need filtering if target column was removed and they weren't already
                # Assuming self.variables are suitable for X
                fit_params = self.config.algo_params.get('fit_params', {}) # e.g. generations, pop_size
                best_expr_obj = regressor.fit(X, y, self.variables, **fit_params)

                if best_expr_obj and hasattr(best_expr_obj, 'symbolic'):
                    current_run_result.discovered_law = str(best_expr_obj.symbolic)
                    current_run_result.law_complexity = getattr(best_expr_obj, 'complexity', len(str(best_expr_obj.symbolic)))
                    # MSE calculation:
                    if hasattr(best_expr_obj, 'mse') and best_expr_obj.mse is not None:
                        current_run_result.predictive_mse = best_expr_obj.mse
                    elif hasattr(regressor, 'predict'):
                        predictions = regressor.predict(X)
                        current_run_result.predictive_mse = np.mean((predictions - y)**2)
                    else: # Fallback for MSE needed
                        current_run_result.predictive_mse = float('inf')
                else:
                    current_run_result.discovered_law = "Error: Genetic algorithm failed."
                    current_run_result.predictive_mse = float('inf')
                current_run_result.n_experiments_to_convergence = getattr(regressor, 'generations', 1) # Example

            elif algo_name == 'random':
                # Placeholder for random search
                current_run_result.discovered_law = "random_expr_placeholder"
                current_run_result.predictive_mse = np.random.rand() * 10 # Dummy MSE
                current_run_result.law_complexity = 10
                current_run_result.n_experiments_to_convergence = self.config.algo_params.get('num_random_expressions', 100)
                logging.warning(f"[{self.config.name}] Random search is a placeholder.")

            else:
                logging.error(f"[{self.config.name}] Unknown algorithm '{algo_name}' in run method.")
                current_run_result.discovered_law = f"Error: Unknown algorithm '{algo_name}'"
                return current_run_result # Early exit for unknown algorithm

            # Calculate symbolic accuracy
            if self.ground_truth_laws and current_run_result.discovered_law:
                current_run_result.symbolic_accuracy = calculate_symbolic_accuracy(
                    current_run_result.discovered_law, self.ground_truth_laws
                )
            else:
                current_run_result.symbolic_accuracy = 0.0
                logging.debug(f"[{self.config.name}] Symbolic accuracy not computed (no ground truth or no law discovered).")

        except Exception as e:
            logging.error(f"[{self.config.name}] Exception during algorithm execution ('{algo_name}'): {e}", exc_info=True)
            current_run_result.discovered_law = f"Error during {algo_name}: {str(e)[:100]}" # Truncate long errors
            current_run_result.symbolic_accuracy = 0.0
            current_run_result.predictive_mse = float('inf')

        logging.info(f"[{self.config.name}] Run phase finished. Law: '{current_run_result.discovered_law}', Accuracy: {current_run_result.symbolic_accuracy:.4f}, MSE: {current_run_result.predictive_mse:.4e}")
        return current_run_result

    def teardown(self):
        """Cleans up resources used by the experiment.

        This method is called after the experiment execution (even if errors
        occurred) to release resources, close files, or terminate any child
        processes. For this example, it primarily logs the completion and
        resets internal state variables.
        """
        logging.info(f"[{self.config.name}] Tearing down experiment.")
        # Example: if self.algorithm and hasattr(self.algorithm, 'close'): self.algorithm.close()
        self.physics_env = None
        self.env_data = None
        self.variables = None
        self.algorithm = None
        self.ground_truth_laws = None
        logging.info(f"[{self.config.name}] Teardown complete.")

    def execute(self, run_id: int = 0) -> ExperimentResult:
        """Orchestrates the full lifecycle of the experiment: setup, run, and teardown.

        This method overrides `BaseExperiment.execute`. It measures the total
        wall-clock time for the experiment and ensures that `setup`, `run`, and
        `teardown` are called in sequence. It manages an `ExperimentResult`
        object that is populated by the `run` method.

        Args:
            run_id: The identifier for this specific execution run.

        Returns:
            An `ExperimentResult` object containing all outcomes and data from
            the experiment. If critical failures occur, the result object will
            reflect an error state.
        """
        self._start_time = time.time()
        # Initialize a result object for this execution.
        # It will be further populated by the 'run' method.
        self.experiment_result = ExperimentResult(config=self.config, run_id=run_id)

        try:
            logging.info(f"[{self.config.name}] EXECUTE: Starting setup (run {run_id})...")
            self.setup()
            logging.info(f"[{self.config.name}] EXECUTE: Setup complete (run {run_id}).")

            # If setup was successful, env_data might be available to store early
            if self.env_data is not None:
                self.experiment_result.trajectory_data = self.env_data

            logging.info(f"[{self.config.name}] EXECUTE: Starting run (run {run_id})...")
            run_specific_result = self.run(run_id=run_id) # This method is designed to return a result with error info for algo failures

            # Merge results from run_specific_result into self.experiment_result
            if run_specific_result:
                self.experiment_result.discovered_law = run_specific_result.discovered_law
                self.experiment_result.symbolic_accuracy = run_specific_result.symbolic_accuracy
                self.experiment_result.predictive_mse = run_specific_result.predictive_mse
                self.experiment_result.law_complexity = run_specific_result.law_complexity
                self.experiment_result.n_experiments_to_convergence = run_specific_result.n_experiments_to_convergence
                self.experiment_result.sample_efficiency_curve = run_specific_result.sample_efficiency_curve
                self.experiment_result.component_metrics = run_specific_result.component_metrics
                if run_specific_result.trajectory_data is not None:
                    self.experiment_result.trajectory_data = run_specific_result.trajectory_data
            else:
                 logging.error(f"[{self.config.name}] EXECUTE: run() method returned None. This is unexpected.")
                 self.experiment_result.discovered_law = "Error: run() returned None"

            logging.info(f"[{self.config.name}] EXECUTE: Run complete (run {run_id}). Discovered: '{self.experiment_result.discovered_law}'")

        except (PluginNotFoundError, InvalidConfigError, DataGenerationError, MissingDependencyError) as e:
            # These are setup-related errors, re-raise them directly
            logging.error(f"[{self.config.name}] EXECUTE: Setup failed for run {run_id} with a critical configuration or dependency error: {e}", exc_info=True)
            raise # Re-raise the specific custom exception
        except Exception as e:
            # Catch other unexpected errors during setup or if run() itself raises an unexpected exception
            logging.error(f"[{self.config.name}] EXECUTE: Unexpected exception during setup or run phase for run {run_id}: {e}", exc_info=True)
            if self.experiment_result:
                self.experiment_result.discovered_law = f"Unexpected Critical Error: {str(e)[:150]}"
                self.experiment_result.symbolic_accuracy = 0.0
                self.experiment_result.predictive_mse = float('inf')
        finally:
            current_wall_time = time.time() - self._start_time
            if self.experiment_result:
                self.experiment_result.wall_time_seconds = current_wall_time

            logging.info(f"[{self.config.name}] EXECUTE: Starting teardown and cleanup (run {run_id}). Wall time: {current_wall_time:.2f}s.")
            self.cleanup() # Call new cleanup method
            self.teardown() # Call original teardown
            logging.info(f"[{self.config.name}] EXECUTE: Teardown and cleanup complete (run {run_id}).")

        # Final check to ensure an ExperimentResult object is always returned.
        if not self.experiment_result:
            logging.critical(f"[{self.config.name}] EXECUTE: self.experiment_result is None at the end. This indicates a major issue.")
            # Create a minimal error result if it's somehow still None
            self.experiment_result = ExperimentResult(config=self.config, run_id=run_id,
                                                      discovered_law="Critical error: experiment_result not formed.")
            self.experiment_result.wall_time_seconds = time.time() - self._start_time

        return self.experiment_result

    def get_best_discovered_law(self) -> Optional[Dict[str, Any]]:
        """Gets the best discovered law from the hypothesis tracker."""
        if not self.training_integration:
            logging.warning(f"[{self.config.name}] Training integration not initialized. Cannot get best law.")
            return None
        return self.training_integration.get_best_discovered_law(criterion='overall')

    def get_training_statistics(self) -> Optional[Dict[str, Any]]:
        """Gets training statistics from the hypothesis tracker."""
        if not self.hypothesis_tracker:
            logging.warning(f"[{self.config.name}] Hypothesis tracker not initialized. Cannot get stats.")
            return None
        return self.hypothesis_tracker.get_training_statistics()

    def export_results(self, output_file_base: str = "experiment_export"):
        """Exports key results from the hypothesis tracker."""
        if not self.hypothesis_tracker:
            logging.warning(f"[{self.config.name}] Hypothesis tracker not initialized. Cannot export.")
            return

        export_dir = Path(self.hypothesis_tracker.save_directory) / "exports" # Ensure self.tracker_save_dir is used if self.hypothesis_tracker.save_directory is not set early
        if hasattr(self.hypothesis_tracker, 'save_directory') and self.hypothesis_tracker.save_directory:
            export_dir = Path(self.hypothesis_tracker.save_directory) / "exports"
        elif hasattr(self, 'tracker_save_dir'): # Fallback to tracker_save_dir from __init__
             export_dir = self.tracker_save_dir / "exports"
        else: # Absolute fallback
            logging.warning(f"[{self.config.name}] save_directory for hypothesis_tracker not found, using default ./exports")
            export_dir = Path("./exports")

        export_dir.mkdir(parents=True, exist_ok=True)

        self.hypothesis_tracker.export_best_hypotheses(
            str(export_dir / f"{output_file_base}_best_overall.json"), criterion='overall', n=10, format='json'
        )
        self.hypothesis_tracker.export_best_hypotheses(
            str(export_dir / f"{output_file_base}_best_conservation.json"), criterion='conservation', n=10, format='json'
        )
        # Add more exports if needed
        logging.info(f"[{self.config.name}] Results exported to {export_dir}")

    def cleanup(self):
        """Cleans up resources, like saving final tracker state or stopping monitor."""
        logging.info(f"[{self.config.name}] Performing cleanup...")
        if self.hypothesis_tracker and hasattr(self.hypothesis_tracker, 'save_state'): # Check for save_state method
            self.hypothesis_tracker.save_state()
            logging.info(f"[{self.config.name}] Hypothesis tracker state saved.")

        if self.live_monitor and hasattr(self.live_monitor, 'stop_monitoring'): # Check attribute and method
            self.live_monitor.stop_monitoring()
            logging.info(f"[{self.config.name}] Live monitor stopped.")

        if self.training_logger and hasattr(self.training_logger, 'close'): # Check attribute and method
            self.training_logger.close()
            logging.info(f"[{self.config.name}] Training logger closed.")


if __name__ == "__main__":
    # Basic configuration for logging, useful for seeing output from ExperimentRunner
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - [%(module)s:%(funcName)s:%(lineno)d] - %(message)s')

    print("ExperimentRunner __main__ Test Section")
    print("======================================")

    # Example: Test a single experiment run directly if needed
    # This helps in debugging a specific configuration or plugin.
    example_config = ExperimentConfig(
        name="main_test_harmonic_oscillator",
        experiment_type='physics_discovery_example', # CRITICAL: This plugin must be registered
        environment_type='harmonic_oscillator',
        algorithm='genetic',
        env_params={'k': 1.0, 'm': 1.0},
        noise_level=0.01,
        max_experiments=50, # Small value for a quick test
        n_runs=1 # Single run for this direct test
    )

    runner_main = ExperimentRunner(use_wandb=False, base_dir="./experiments_main_test")

    if example_config.experiment_type not in runner_main.experiment_plugins:
        logging.error(f"__main__: Experiment type '{example_config.experiment_type}' not found in plugins.")
        logging.error("Please ensure 'physics_discovery_example' is correctly registered via setup.py entry_points.")
        logging.error("And that the package has been installed (e.g., 'pip install -e .').")
        logging.error(f"Available plugins: {list(runner_main.experiment_plugins.keys())}")
    else:
        logging.info(f"__main__: Attempting to run experiment '{example_config.name}'...")
        try:
            result = runner_main.run_single_experiment(example_config, run_id=0)
            if result:
                print(f"\n--- __main__ Experiment '{example_config.name}' Results ---")
                print(f"  Discovered law: {result.discovered_law}")
                print(f"  Symbolic accuracy: {result.symbolic_accuracy:.4f}")
                print(f"  Predictive MSE: {result.predictive_mse:.4e}")
                print(f"  Wall time: {result.wall_time_seconds:.2f}s")
            else:
                # run_single_experiment should ideally always return a result, even for errors
                print(f"__main__: Experiment '{example_config.name}' returned None or an unexpected result.")
        except Exception as e:
            logging.error(f"__main__: Critical error during run_single_experiment: {e}", exc_info=True)
            print(f"__main__: An exception occurred: {e}")

    # Example of running predefined validation phases (optional, can be lengthy)
    # print("\nStarting Phase 1 Validation (Known Law Rediscovery)...")
    # phase1_results = run_phase1_validation()
    # if phase1_results is not None:
    #     print("Phase 1 Validation completed. Results summary:")
    #     print(phase1_results.groupby(['experiment_name', 'algorithm'])['symbolic_accuracy'].mean())
    # else:
    #     print("Phase 1 Validation did not produce results or was skipped.")

    # print("\nStarting Phase 2 Validation (Robustness Benchmark)...")
    # phase2_results = run_phase2_robustness()
    # if phase2_results is not None:
    #     print("Phase 2 Validation completed. Results summary (mean accuracy by noise):")
    #     print(phase2_results.groupby(['noise_level', 'algorithm'])['symbolic_accuracy'].mean())
    # else:
    #     print("Phase 2 Validation did not produce results or was skipped.")

    print("\n__main__ Test Section Finished.")
