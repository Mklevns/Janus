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
from math_utils import calculate_symbolic_accuracy

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
            err_result = ExperimentResult(config=config, run_id=run_id, discovered_law="Error: experiment_type not specified in config.")
            # Ensure default error values are set
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
                trainer = self.algorithm # Assuming self.algorithm is the PPO trainer
                num_evals = self.config.algo_params.get('num_evaluations', 50)
                steps_per_eval = self.config.algo_params.get('timesteps_per_eval_cycle', 1000)
                ppo_params = self.config.algo_params.get('ppo_train_params', {})

                best_mse_overall = float('inf')
                best_expr_overall = None
                efficiency_curve = []

                for i in range(num_evals):
                    trainer.train(total_timesteps=steps_per_eval, **ppo_params)
                    current_total_steps = (i + 1) * steps_per_eval
                    # Evaluation logic (simplified, assumes trainer.env has a cache)
                    if hasattr(trainer, 'env') and hasattr(trainer.env, '_evaluation_cache'):
                        # Logic to get best expression and MSE from cache
                        # This is highly dependent on SymbolicDiscoveryEnv's implementation
                        # For now, let's assume a method get_best_solution() exists or similar
                        # current_expr, current_mse_cached = trainer.env.get_best_solution_from_cache()
                        # This is a placeholder - actual eval needs robust implementation
                        # Using a simplified approach based on previous logic:
                        temp_best_mse_in_cache = float('inf')
                        temp_best_expr_in_cache = None
                        if trainer.env._evaluation_cache: # If cache is list of dicts
                             for entry in trainer.env._evaluation_cache:
                                if entry.get('mse', float('inf')) < temp_best_mse_in_cache:
                                    temp_best_mse_in_cache = entry['mse']
                                    temp_best_expr_in_cache = entry.get('expression')

                        if temp_best_expr_in_cache and temp_best_mse_in_cache < best_mse_overall:
                            best_mse_overall = temp_best_mse_in_cache
                            best_expr_overall = temp_best_expr_in_cache
                    efficiency_curve.append((current_total_steps, best_mse_overall))

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
            if self.experiment_result: # Ensure it exists
                self.experiment_result.wall_time_seconds = current_wall_time

            logging.info(f"[{self.config.name}] EXECUTE: Starting teardown (run {run_id}). Wall time: {current_wall_time:.2f}s.")
            self.teardown()
            logging.info(f"[{self.config.name}] EXECUTE: Teardown complete (run {run_id}).")

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
                trainer = self.algorithm # Assuming self.algorithm is the PPO trainer
                num_evals = self.config.algo_params.get('num_evaluations', 50)
                steps_per_eval = self.config.algo_params.get('timesteps_per_eval_cycle', 1000)
                ppo_params = self.config.algo_params.get('ppo_train_params', {})

                best_mse_overall = float('inf')
                best_expr_overall = None
                efficiency_curve = []

                for i in range(num_evals):
                    trainer.train(total_timesteps=steps_per_eval, **ppo_params)
                    current_total_steps = (i + 1) * steps_per_eval
                    # Evaluation logic (simplified, assumes trainer.env has a cache)
                    if hasattr(trainer, 'env') and hasattr(trainer.env, '_evaluation_cache'):
                        # Logic to get best expression and MSE from cache
                        # This is highly dependent on SymbolicDiscoveryEnv's implementation
                        # For now, let's assume a method get_best_solution() exists or similar
                        # current_expr, current_mse_cached = trainer.env.get_best_solution_from_cache()
                        # This is a placeholder - actual eval needs robust implementation
                        # Using a simplified approach based on previous logic:
                        temp_best_mse_in_cache = float('inf')
                        temp_best_expr_in_cache = None
                        if trainer.env._evaluation_cache: # If cache is list of dicts
                             for entry in trainer.env._evaluation_cache:
                                if entry.get('mse', float('inf')) < temp_best_mse_in_cache:
                                    temp_best_mse_in_cache = entry['mse']
                                    temp_best_expr_in_cache = entry.get('expression')

                        if temp_best_expr_in_cache and temp_best_mse_in_cache < best_mse_overall:
                            best_mse_overall = temp_best_mse_in_cache
                            best_expr_overall = temp_best_expr_in_cache
                    efficiency_curve.append((current_total_steps, best_mse_overall))

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
            if self.experiment_result: # Ensure it exists
                self.experiment_result.wall_time_seconds = current_wall_time

            logging.info(f"[{self.config.name}] EXECUTE: Starting teardown (run {run_id}). Wall time: {current_wall_time:.2f}s.")
            self.teardown()
            logging.info(f"[{self.config.name}] EXECUTE: Teardown complete (run {run_id}).")

        # Final check to ensure an ExperimentResult object is always returned.
        if not self.experiment_result:
            logging.critical(f"[{self.config.name}] EXECUTE: self.experiment_result is None at the end. This indicates a major issue.")
            # Create a minimal error result if it's somehow still None
            self.experiment_result = ExperimentResult(config=self.config, run_id=run_id,
                                                      discovered_law="Critical error: experiment_result not formed.")
            self.experiment_result.wall_time_seconds = time.time() - self._start_time

        return self.experiment_result


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
