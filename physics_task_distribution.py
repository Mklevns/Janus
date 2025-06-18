"""
Physics Task Distribution for Meta-Learning
==========================================

Comprehensive task distribution with physically accurate data generators
for meta-learning physics discovery.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import scipy.integrate as integrate
from collections import defaultdict


@dataclass
class PhysicsTask:
    """Represents a physics discovery task with full metadata"""
    name: str
    data_generator: Callable[[int], np.ndarray]
    true_law: str
    variables: List[str]
    variable_ranges: Dict[str, Tuple[float, float]]
    symmetries: List[str]
    conserved_quantities: List[str]
    difficulty: float
    domain: str
    physical_parameters: Dict[str, Any] = field(default_factory=dict)
    noise_level: float = 0.01
    description: str = ""
    
    def generate_data(self, n_samples: int, noise: bool = True) -> np.ndarray:
        """Generate data with optional noise"""
        data = self.data_generator(n_samples)
        
        if noise and self.noise_level > 0:
            # Add Gaussian noise to observations (not parameters)
            n_vars = len(self.variables)
            noise_mask = np.ones(n_vars, dtype=bool)
            # Don't add noise to constant parameters
            for i, var in enumerate(self.variables[:-1]):  # Last is usually target
                if var in self.physical_parameters:
                    noise_mask[i] = False
            
            noise_array = np.random.normal(0, self.noise_level, data.shape)
            noise_array[:, ~noise_mask] = 0
            data = data + noise_array * np.abs(data)  # Multiplicative noise
        
        return data


class PhysicsEnvironmentBase(ABC):
    """Base class for physics environments with data generation"""
    
    def __init__(self, 
                 dt: float = 0.01,
                 t_max: float = 10.0,
                 param_ranges: Optional[Dict[str, Tuple[float, float]]] = None):
        self.dt = dt
        self.t_max = t_max
        self.param_ranges = param_ranges or self.get_default_param_ranges()
        
    @abstractmethod
    def get_default_param_ranges(self) -> Dict[str, Tuple[float, float]]:
        """Get default parameter ranges"""
        pass
    
    @abstractmethod
    def dynamics(self, state: np.ndarray, t: float, params: Dict[str, float]) -> np.ndarray:
        """Define system dynamics dx/dt = f(x, t, params)"""
        pass
    
    @abstractmethod
    def compute_observables(self, trajectory: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        """Compute observable quantities from state trajectory"""
        pass
    
    def sample_parameters(self) -> Dict[str, float]:
        """Sample random parameters"""
        params = {}
        for param, (low, high) in self.param_ranges.items():
            if low == high:
                params[param] = low
            else:
                params[param] = np.random.uniform(low, high)
        return params
    
    def generate_trajectory(self, params: Optional[Dict[str, float]] = None) -> np.ndarray:
        """Generate a single trajectory"""
        if params is None:
            params = self.sample_parameters()
        
        # Initial conditions
        x0 = self.sample_initial_conditions(params)
        
        # Time points
        t = np.arange(0, self.t_max, self.dt)
        
        # Integrate dynamics
        solution = integrate.odeint(
            lambda x, t: self.dynamics(x, t, params),
            x0, t
        )
        
        # Compute observables
        observables = self.compute_observables(solution, params)
        
        return observables
    
    @abstractmethod
    def sample_initial_conditions(self, params: Dict[str, float]) -> np.ndarray:
        """Sample initial conditions"""
        pass
    
    def generate_dataset(self, n_samples: int) -> np.ndarray:
        """Generate dataset with multiple trajectories"""
        all_data = []
        
        for _ in range(n_samples // 100):  # Generate multiple trajectories
            params = self.sample_parameters()
            trajectory = self.generate_trajectory(params)
            
            # Sample points from trajectory
            indices = np.random.choice(len(trajectory), size=min(100, len(trajectory)), replace=False)
            all_data.append(trajectory[indices])
        
        return np.vstack(all_data)[:n_samples]


# Concrete Physics Environments

class HarmonicOscillatorEnv(PhysicsEnvironmentBase):
    """Simple harmonic oscillator: F = -kx"""
    
    def get_default_param_ranges(self) -> Dict[str, Tuple[float, float]]:
        return {
            'k': (0.5, 3.0),  # Spring constant
            'm': (0.5, 2.0),  # Mass
        }
    
    def dynamics(self, state: np.ndarray, t: float, params: Dict[str, float]) -> np.ndarray:
        x, v = state
        k, m = params['k'], params['m']
        return np.array([v, -k/m * x])
    
    def sample_initial_conditions(self, params: Dict[str, float]) -> np.ndarray:
        x0 = np.random.uniform(-2, 2)
        v0 = np.random.uniform(-2, 2)
        return np.array([x0, v0])
    
    def compute_observables(self, trajectory: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        """Return [x, v, k, m, E]"""
        x, v = trajectory[:, 0], trajectory[:, 1]
        k, m = params['k'], params['m']
        
        # Total energy (conserved)
        E = 0.5 * m * v**2 + 0.5 * k * x**2
        
        return np.column_stack([
            x, v,
            np.full_like(x, k),
            np.full_like(x, m),
            E
        ])


class PendulumEnv(PhysicsEnvironmentBase):
    """Pendulum with optional small angle approximation"""
    
    def __init__(self, small_angle: bool = False, **kwargs):
        self.small_angle = small_angle
        super().__init__(**kwargs)
    
    def get_default_param_ranges(self) -> Dict[str, Tuple[float, float]]:
        return {
            'L': (0.5, 2.0),  # Length
            'g': (9.81, 9.81),  # Gravity (fixed)
            'b': (0.0, 0.5) if not self.small_angle else (0.0, 0.0),  # Damping
        }
    
    def dynamics(self, state: np.ndarray, t: float, params: Dict[str, float]) -> np.ndarray:
        theta, omega = state
        L, g, b = params['L'], params['g'], params['b']
        
        if self.small_angle:
            # Linear approximation
            theta_dd = -g/L * theta - b * omega
        else:
            # Full nonlinear dynamics
            theta_dd = -g/L * np.sin(theta) - b * omega
        
        return np.array([omega, theta_dd])
    
    def sample_initial_conditions(self, params: Dict[str, float]) -> np.ndarray:
        if self.small_angle:
            theta0 = np.random.uniform(-0.3, 0.3)  # Small angles
        else:
            theta0 = np.random.uniform(-np.pi/2, np.pi/2)
        omega0 = np.random.uniform(-1, 1)
        return np.array([theta0, omega0])
    
    def compute_observables(self, trajectory: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        """Return [theta, omega, L, g, angular_accel]"""
        theta, omega = trajectory[:, 0], trajectory[:, 1]
        L, g = params['L'], params['g']
        
        if self.small_angle:
            angular_accel = -g/L * theta
        else:
            angular_accel = -g/L * np.sin(theta)
        
        return np.column_stack([
            theta, omega,
            np.full_like(theta, L),
            np.full_like(theta, g),
            angular_accel
        ])


class KeplerOrbitEnv(PhysicsEnvironmentBase):
    """Orbital mechanics - circular orbits for simplicity"""
    
    def get_default_param_ranges(self) -> Dict[str, Tuple[float, float]]:
        return {
            'G': (6.67e-11, 6.67e-11),  # Gravitational constant
            'M': (5e24, 2e25),  # Central mass (Earth to Neptune range)
        }
    
    def dynamics(self, state: np.ndarray, t: float, params: Dict[str, float]) -> np.ndarray:
        # State: [x, y, vx, vy]
        x, y, vx, vy = state
        G, M = params['G'], params['M']
        
        r = np.sqrt(x**2 + y**2)
        if r < 1e6:  # Avoid singularity
            r = 1e6
        
        # Gravitational acceleration
        ax = -G * M * x / r**3
        ay = -G * M * y / r**3
        
        return np.array([vx, vy, ax, ay])
    
    def sample_initial_conditions(self, params: Dict[str, float]) -> np.ndarray:
        # Sample circular orbit
        r0 = np.random.uniform(6.4e6, 4e7)  # Near Earth to geosynchronous
        theta0 = np.random.uniform(0, 2*np.pi)
        
        x0 = r0 * np.cos(theta0)
        y0 = r0 * np.sin(theta0)
        
        # Circular velocity
        v_circular = np.sqrt(params['G'] * params['M'] / r0)
        vx0 = -v_circular * np.sin(theta0)
        vy0 = v_circular * np.cos(theta0)
        
        # Add small perturbation
        vx0 *= np.random.uniform(0.9, 1.1)
        vy0 *= np.random.uniform(0.9, 1.1)
        
        return np.array([x0, y0, vx0, vy0])
    
    def compute_observables(self, trajectory: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        """Return [r, v, G, M, F_grav]"""
        x, y, vx, vy = trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], trajectory[:, 3]
        G, M = params['G'], params['M']
        
        r = np.sqrt(x**2 + y**2)
        v = np.sqrt(vx**2 + vy**2)
        F_grav = G * M / r**2
        
        return np.column_stack([
            r, v,
            np.full_like(r, G),
            np.full_like(r, M),
            F_grav
        ])


class DampedOscillatorEnv(PhysicsEnvironmentBase):
    """Damped harmonic oscillator"""
    
    def get_default_param_ranges(self) -> Dict[str, Tuple[float, float]]:
        return {
            'k': (1.0, 3.0),  # Spring constant
            'm': (0.5, 2.0),  # Mass
            'b': (0.1, 1.0),  # Damping coefficient
        }
    
    def dynamics(self, state: np.ndarray, t: float, params: Dict[str, float]) -> np.ndarray:
        x, v = state
        k, m, b = params['k'], params['m'], params['b']
        return np.array([v, (-k*x - b*v) / m])
    
    def sample_initial_conditions(self, params: Dict[str, float]) -> np.ndarray:
        x0 = np.random.uniform(-2, 2)
        v0 = np.random.uniform(-2, 2)
        return np.array([x0, v0])
    
    def compute_observables(self, trajectory: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        """Return [x, v, k, b, force]"""
        x, v = trajectory[:, 0], trajectory[:, 1]
        k, b = params['k'], params['b']
        
        force = -k*x - b*v
        
        return np.column_stack([
            x, v,
            np.full_like(x, k),
            np.full_like(x, b),
            force
        ])


class IdealGasEnv(PhysicsEnvironmentBase):
    """Ideal gas law - generates thermodynamic states"""
    
    def get_default_param_ranges(self) -> Dict[str, Tuple[float, float]]:
        return {
            'R': (8.314, 8.314),  # Gas constant
            'n': (0.1, 2.0),  # Moles
        }
    
    def dynamics(self, state: np.ndarray, t: float, params: Dict[str, float]) -> np.ndarray:
        # For ideal gas, we'll simulate adiabatic process
        P, V, T = state
        n, R = params['n'], params['R']
        gamma = 1.4  # For diatomic gas
        
        # Adiabatic expansion/compression
        dV_dt = 0.1 * np.sin(t)  # Oscillating volume
        dP_dt = -gamma * P / V * dV_dt
        dT_dt = T / P * dP_dt + T / V * dV_dt
        
        return np.array([dP_dt, dV_dt, dT_dt])
    
    def sample_initial_conditions(self, params: Dict[str, float]) -> np.ndarray:
        # Sample valid thermodynamic state
        P0 = np.random.uniform(0.5e5, 2e5)  # 0.5-2 atm
        V0 = np.random.uniform(0.01, 0.1)  # 10-100 L
        T0 = P0 * V0 / (params['n'] * params['R'])
        return np.array([P0, V0, T0])
    
    def compute_observables(self, trajectory: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        """Return [P, V, n, T, R_calculated]"""
        P, V, T = trajectory[:, 0], trajectory[:, 1], trajectory[:, 2]
        n, R = params['n'], params['R']
        
        # R_calculated should equal R (ideal gas law)
        R_calculated = P * V / (n * T)
        
        return np.column_stack([
            P, V,
            np.full_like(P, n),
            T,
            R_calculated
        ])


class CoulombLawEnv(PhysicsEnvironmentBase):
    """Coulomb's law - electrostatic interactions"""
    
    def get_default_param_ranges(self) -> Dict[str, Tuple[float, float]]:
        return {
            'k_e': (8.99e9, 8.99e9),  # Coulomb constant
            'q1': (-1e-6, 1e-6),  # Charge 1 (μC)
            'q2': (-1e-6, 1e-6),  # Charge 2 (μC)
        }
    
    def dynamics(self, state: np.ndarray, t: float, params: Dict[str, float]) -> np.ndarray:
        # Two charges: state = [x1, y1, x2, y2, vx1, vy1, vx2, vy2]
        # For simplicity, fix charge 2 at origin, only charge 1 moves
        x, y, vx, vy = state
        k_e, q1, q2 = params['k_e'], params['q1'], params['q2']
        
        r = np.sqrt(x**2 + y**2)
        if r < 0.01:  # Avoid singularity
            r = 0.01
        
        # Coulomb force
        F = k_e * q1 * q2 / r**2
        ax = F * x / (r * abs(q1))  # Assuming unit mass
        ay = F * y / (r * abs(q1))
        
        return np.array([vx, vy, ax, ay])
    
    def sample_initial_conditions(self, params: Dict[str, float]) -> np.ndarray:
        # Start at random position
        r0 = np.random.uniform(0.1, 1.0)
        theta0 = np.random.uniform(0, 2*np.pi)
        
        x0 = r0 * np.cos(theta0)
        y0 = r0 * np.sin(theta0)
        
        # Random initial velocity
        v0 = np.random.uniform(0, 0.5)
        phi0 = np.random.uniform(0, 2*np.pi)
        
        vx0 = v0 * np.cos(phi0)
        vy0 = v0 * np.sin(phi0)
        
        return np.array([x0, y0, vx0, vy0])
    
    def compute_observables(self, trajectory: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        """Return [r, q1, q2, k_e, F_coulomb]"""
        x, y = trajectory[:, 0], trajectory[:, 1]
        k_e, q1, q2 = params['k_e'], params['q1'], params['q2']
        
        r = np.sqrt(x**2 + y**2)
        F_coulomb = k_e * abs(q1 * q2) / r**2
        
        return np.column_stack([
            r,
            np.full_like(r, q1),
            np.full_like(r, q2),
            np.full_like(r, k_e),
            F_coulomb
        ])


class ElasticCollisionEnv(PhysicsEnvironmentBase):
    """1D elastic collision - conservation of momentum and energy"""
    
    def get_default_param_ranges(self) -> Dict[str, Tuple[float, float]]:
        return {
            'm1': (0.5, 2.0),
            'm2': (0.5, 2.0),
        }
    
    def dynamics(self, state: np.ndarray, t: float, params: Dict[str, float]) -> np.ndarray:
        # Simple 1D collision
        x1, x2, v1, v2 = state
        
        # No forces except during collision
        return np.array([v1, v2, 0, 0])
    
    def sample_initial_conditions(self, params: Dict[str, float]) -> np.ndarray:
        # Set up collision scenario
        x1 = -1.0
        x2 = 1.0
        v1 = np.random.uniform(0.5, 2.0)  # Moving right
        v2 = np.random.uniform(-2.0, -0.5)  # Moving left
        return np.array([x1, x2, v1, v2])
    
    def compute_observables(self, trajectory: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        """Return [m1, v1, m2, v2, total_KE, total_momentum]"""
        x1, x2, v1, v2 = trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], trajectory[:, 3]
        m1, m2 = params['m1'], params['m2']
        
        # Check for collision and compute post-collision velocities
        for i in range(1, len(x1)):
            if x1[i-1] < x2[i-1] and x1[i] >= x2[i]:  # Collision detected
                # Elastic collision formulas
                v1_new = ((m1 - m2) * v1[i-1] + 2 * m2 * v2[i-1]) / (m1 + m2)
                v2_new = ((m2 - m1) * v2[i-1] + 2 * m1 * v1[i-1]) / (m1 + m2)
                v1[i:] = v1_new
                v2[i:] = v2_new
        
        # Conserved quantities
        total_KE = 0.5 * m1 * v1**2 + 0.5 * m2 * v2**2
        total_momentum = m1 * v1 + m2 * v2
        
        return np.column_stack([
            np.full_like(v1, m1),
            v1,
            np.full_like(v2, m2),
            v2,
            total_KE,
            total_momentum
        ])


class PhysicsTaskDistribution:
    """Distribution over physics discovery tasks with robust data generation"""
    
    def __init__(self, include_noise: bool = True):
        self.include_noise = include_noise
        self.tasks = self._create_task_library()
        self.task_families = self._organize_by_family()
        self.task_by_name = {task.name: task for task in self.tasks}
        
    def _create_task_library(self) -> List[PhysicsTask]:
        """Create comprehensive physics task library"""
        tasks = []
        
        # Classical Mechanics Tasks
        
        # Harmonic Oscillator
        ho_env = HarmonicOscillatorEnv()
        tasks.append(PhysicsTask(
            name="harmonic_oscillator_energy",
            data_generator=lambda n: ho_env.generate_dataset(n),
            true_law="0.5 * m * v**2 + 0.5 * k * x**2",
            variables=["x", "v", "k", "m", "E"],
            variable_ranges={
                "x": (-2, 2), "v": (-2, 2), 
                "k": (0.5, 3.0), "m": (0.5, 2.0),
                "E": (0, 10)
            },
            symmetries=["time_reversal", "energy_conservation"],
            conserved_quantities=["energy"],
            difficulty=0.2,
            domain="mechanics",
            description="Discover energy conservation in harmonic oscillator",
            noise_level=0.01
        ))
        
        # Pendulum (small angle)
        pend_small_env = PendulumEnv(small_angle=True)
        tasks.append(PhysicsTask(
            name="pendulum_small_angle",
            data_generator=lambda n: pend_small_env.generate_dataset(n),
            true_law="-g/L * theta",
            variables=["theta", "omega", "L", "g", "angular_accel"],
            variable_ranges={
                "theta": (-0.3, 0.3), "omega": (-1, 1),
                "L": (0.5, 2.0), "g": (9.81, 9.81),
                "angular_accel": (-10, 10)
            },
            symmetries=["time_reversal"],
            conserved_quantities=["energy"],
            difficulty=0.3,
            domain="mechanics",
            description="Linear approximation of pendulum motion"
        ))
        
        # Pendulum (full nonlinear)
        pend_full_env = PendulumEnv(small_angle=False)
        tasks.append(PhysicsTask(
            name="pendulum_nonlinear",
            data_generator=lambda n: pend_full_env.generate_dataset(n),
            true_law="-g/L * sin(theta)",
            variables=["theta", "omega", "L", "g", "angular_accel"],
            variable_ranges={
                "theta": (-np.pi/2, np.pi/2), "omega": (-1, 1),
                "L": (0.5, 2.0), "g": (9.81, 9.81),
                "angular_accel": (-20, 20)
            },
            symmetries=["none"],
            conserved_quantities=["energy"],
            difficulty=0.5,
            domain="mechanics",
            description="Full nonlinear pendulum dynamics"
        ))
        
        # Kepler's Law
        kepler_env = KeplerOrbitEnv()
        tasks.append(PhysicsTask(
            name="kepler_orbit",
            data_generator=lambda n: kepler_env.generate_dataset(n),
            true_law="G*M/r**2",
            variables=["r", "v", "G", "M", "F_grav"],
            variable_ranges={
                "r": (6.4e6, 4e7), "v": (1e3, 1e4),
                "G": (6.67e-11, 6.67e-11), "M": (5e24, 2e25),
                "F_grav": (0, 10)
            },
            symmetries=["rotational", "energy_conservation", "angular_momentum_conservation"],
            conserved_quantities=["energy", "angular_momentum"],
            difficulty=0.6,
            domain="mechanics",
            description="Gravitational force in orbital mechanics",
            physical_parameters={"G": 6.67e-11}
        ))
        
        # Damped Oscillator
        damped_env = DampedOscillatorEnv()
        tasks.append(PhysicsTask(
            name="damped_oscillator",
            data_generator=lambda n: damped_env.generate_dataset(n),
            true_law="-k*x - b*v",
            variables=["x", "v", "k", "b", "force"],
            variable_ranges={
                "x": (-2, 2), "v": (-2, 2),
                "k": (1.0, 3.0), "b": (0.1, 1.0),
                "force": (-10, 10)
            },
            symmetries=["none"],  # Damping breaks time reversal
            conserved_quantities=["none"],
            difficulty=0.5,
            domain="mechanics",
            description="Force law for damped harmonic oscillator"
        ))
        
        # Thermodynamics Tasks
        
        # Ideal Gas Law
        gas_env = IdealGasEnv()
        tasks.append(PhysicsTask(
            name="ideal_gas",
            data_generator=lambda n: gas_env.generate_dataset(n),
            true_law="P*V/(n*T)",
            variables=["P", "V", "n", "T", "R"],
            variable_ranges={
                "P": (0.5e5, 2e5), "V": (0.01, 0.1),
                "n": (0.1, 2.0), "T": (200, 400),
                "R": (8.314, 8.314)
            },
            symmetries=["scale_invariance"],
            conserved_quantities=["none"],
            difficulty=0.3,
            domain="thermodynamics",
            description="Ideal gas law relationship",
            physical_parameters={"R": 8.314}
        ))
        
        # Electromagnetism Tasks
        
        # Coulomb's Law
        coulomb_env = CoulombLawEnv()
        tasks.append(PhysicsTask(
            name="coulomb_law",
            data_generator=lambda n: coulomb_env.generate_dataset(n),
            true_law="k_e*q1*q2/r**2",
            variables=["r", "q1", "q2", "k_e", "F_coulomb"],
            variable_ranges={
                "r": (0.01, 1.0), "q1": (-1e-6, 1e-6),
                "q2": (-1e-6, 1e-6), "k_e": (8.99e9, 8.99e9),
                "F_coulomb": (0, 1000)
            },
            symmetries=["rotational", "charge_conjugation"],
            conserved_quantities=["none"],
            difficulty=0.4,
            domain="electromagnetism",
            description="Electrostatic force between charges",
            physical_parameters={"k_e": 8.99e9}
        ))
        
        # Conservation Laws
        
        # Elastic Collision
        collision_env = ElasticCollisionEnv()
        tasks.append(PhysicsTask(
            name="elastic_collision",
            data_generator=lambda n: collision_env.generate_dataset(n),
            true_law="0.5*m1*v1**2 + 0.5*m2*v2**2",
            variables=["m1", "v1", "m2", "v2", "total_KE", "total_momentum"],
            variable_ranges={
                "m1": (0.5, 2.0), "v1": (-2, 2),
                "m2": (0.5, 2.0), "v2": (-2, 2),
                "total_KE": (0, 10), "total_momentum": (-5, 5)
            },
            symmetries=["galilean", "time_reversal"],
            conserved_quantities=["energy", "momentum"],
            difficulty=0.7,
            domain="mechanics",
            description="Conservation laws in elastic collision"
        ))
        
        # Advanced/Composite Tasks
        
        # Double Pendulum Energy (for evaluation)
        tasks.append(PhysicsTask(
            name="double_pendulum_energy",
            data_generator=lambda n: self._generate_double_pendulum_data(n),
            true_law="m1*g*L1*(1-cos(theta1)) + m2*g*(L1*(1-cos(theta1)) + L2*(1-cos(theta2))) + 0.5*m1*(L1*omega1)**2 + 0.5*m2*(L1**2*omega1**2 + L2**2*omega2**2 + 2*L1*L2*omega1*omega2*cos(theta1-theta2))",
            variables=["theta1", "theta2", "omega1", "omega2", "m1", "m2", "L1", "L2", "g", "E"],
            variable_ranges={
                "theta1": (-np.pi, np.pi), "theta2": (-np.pi, np.pi),
                "omega1": (-5, 5), "omega2": (-5, 5),
                "m1": (0.5, 2.0), "m2": (0.5, 2.0),
                "L1": (0.5, 1.5), "L2": (0.5, 1.5),
                "g": (9.81, 9.81), "E": (0, 100)
            },
            symmetries=["energy_conservation"],
            conserved_quantities=["energy"],
            difficulty=0.9,
            domain="mechanics",
            description="Total energy in double pendulum system"
        ))
        
        return tasks
    
    def _generate_double_pendulum_data(self, n_samples: int) -> np.ndarray:
        """Generate double pendulum data (simplified for demonstration)"""
        # This would require a more complex implementation
        # For now, return synthetic data that follows the energy conservation law
        data = []
        
        for _ in range(n_samples):
            # Sample parameters
            m1 = np.random.uniform(0.5, 2.0)
            m2 = np.random.uniform(0.5, 2.0)
            L1 = np.random.uniform(0.5, 1.5)
            L2 = np.random.uniform(0.5, 1.5)
            g = 9.81
            
            # Sample state that conserves energy
            E_total = np.random.uniform(10, 50)
            
            # Sample angles
            theta1 = np.random.uniform(-np.pi/2, np.pi/2)
            theta2 = np.random.uniform(-np.pi/2, np.pi/2)
            
            # Calculate potential energy
            PE = m1*g*L1*(1-np.cos(theta1)) + m2*g*(L1*(1-np.cos(theta1)) + L2*(1-np.cos(theta2)))
            
            # Remaining energy is kinetic
            KE = E_total - PE
            if KE < 0:
                KE = 0.1
                E_total = PE + KE
            
            # Distribute kinetic energy between velocities (simplified)
            omega1 = np.sqrt(2*KE/(m1*L1**2 + m2*L1**2)) * np.random.choice([-1, 1])
            omega2 = np.sqrt(2*KE/(m2*L2**2)) * np.random.choice([-1, 1]) * 0.5
            
            data.append([theta1, theta2, omega1, omega2, m1, m2, L1, L2, g, E_total])
        
        return np.array(data)
    
    def _organize_by_family(self) -> Dict[str, List[PhysicsTask]]:
        """Organize tasks by physical domain"""
        families = defaultdict(list)
        for task in self.tasks:
            families[task.domain].append(task)
        return dict(families)
    
    def sample_task(self, 
                   difficulty_range: Optional[Tuple[float, float]] = None,
                   domain: Optional[str] = None,
                   require_symmetry: Optional[str] = None) -> PhysicsTask:
        """Sample a task with optional constraints"""
        
        candidates = self.tasks
        
        if difficulty_range:
            candidates = [t for t in candidates 
                         if difficulty_range[0] <= t.difficulty <= difficulty_range[1]]
        
        if domain:
            candidates = [t for t in candidates if t.domain == domain]
            
        if require_symmetry:
            candidates = [t for t in candidates if require_symmetry in t.symmetries]
        
        if not candidates:
            raise ValueError(f"No tasks match the specified criteria")
        
        return np.random.choice(candidates)
    
    def sample_task_batch(self, 
                         n_tasks: int,
                         curriculum: bool = True,
                         balanced_domains: bool = False) -> List[PhysicsTask]:
        """Sample batch of tasks with various strategies"""
        
        if curriculum:
            # Progressive difficulty
            difficulties = np.linspace(0.2, 0.8, n_tasks)
            tasks = []
            for diff in difficulties:
                task = self.sample_task(
                    difficulty_range=(max(0.1, diff - 0.15), min(0.9, diff + 0.15))
                )
                tasks.append(task)
                
        elif balanced_domains:
            # Balance across domains
            tasks = []
            domains = list(self.task_families.keys())
            for i in range(n_tasks):
                domain = domains[i % len(domains)]
                task = self.sample_task(domain=domain)
                tasks.append(task)
                
        else:
            # Random sampling
            tasks = [self.sample_task() for _ in range(n_tasks)]
        
        return tasks
    
    def get_task_by_name(self, name: str) -> PhysicsTask:
        """Get specific task by name"""
        if name not in self.task_by_name:
            raise ValueError(f"Unknown task: {name}")
        return self.task_by_name[name]
    
    def get_related_tasks(self, task: PhysicsTask, n_tasks: int = 5) -> List[PhysicsTask]:
        """Get tasks related to a given task"""
        # Related = same domain + similar difficulty
        candidates = [
            t for t in self.tasks 
            if t.domain == task.domain 
            and t.name != task.name
            and abs(t.difficulty - task.difficulty) < 0.3
        ]
        
        if len(candidates) <= n_tasks:
            return candidates
        
        return np.random.choice(candidates, size=n_tasks, replace=False).tolist()
    
    def describe_task_distribution(self) -> str:
        """Get summary of task distribution"""
        summary = []
        summary.append(f"Total tasks: {len(self.tasks)}")
        summary.append("\nTasks by domain:")
        for domain, tasks in self.task_families.items():
            summary.append(f"  {domain}: {len(tasks)} tasks")
            diff_range = (min(t.difficulty for t in tasks), 
                         max(t.difficulty for t in tasks))
            summary.append(f"    Difficulty range: {diff_range[0]:.1f} - {diff_range[1]:.1f}")
        
        summary.append("\nConserved quantities:")
        conserved_counts = defaultdict(int)
        for task in self.tasks:
            for quantity in task.conserved_quantities:
                conserved_counts[quantity] += 1
        for quantity, count in conserved_counts.items():
            summary.append(f"  {quantity}: {count} tasks")
        
        return "\n".join(summary)


# Example usage and testing
if __name__ == "__main__":
    # Create task distribution
    task_dist = PhysicsTaskDistribution(include_noise=True)
    
    # Print distribution summary
    print("Physics Task Distribution")
    print("=" * 50)
    print(task_dist.describe_task_distribution())
    
    # Test data generation for a specific task
    print("\n\nTesting Harmonic Oscillator Task:")
    print("-" * 30)
    ho_task = task_dist.get_task_by_name("harmonic_oscillator_energy")
    data = ho_task.generate_data(100, noise=True)
    print(f"Generated data shape: {data.shape}")
    print(f"Variables: {ho_task.variables}")
    print(f"Data sample:\n{data[:5]}")
    
    # Test curriculum sampling
    print("\n\nCurriculum Sampling (5 tasks):")
    print("-" * 30)
    curriculum_tasks = task_dist.sample_task_batch(5, curriculum=True)
    for i, task in enumerate(curriculum_tasks):
        print(f"{i+1}. {task.name} (difficulty: {task.difficulty:.1f}, domain: {task.domain})")
    
    # Test domain-balanced sampling
    print("\n\nDomain-Balanced Sampling (6 tasks):")
    print("-" * 30)
    balanced_tasks = task_dist.sample_task_batch(6, curriculum=False, balanced_domains=True)
    for i, task in enumerate(balanced_tasks):
        print(f"{i+1}. {task.name} (domain: {task.domain})")
    
    # Test related task finding
    print("\n\nRelated Tasks to 'pendulum_small_angle':")
    print("-" * 30)
    base_task = task_dist.get_task_by_name("pendulum_small_angle")
    related = task_dist.get_related_tasks(base_task, n_tasks=3)
    for task in related:
        print(f"  - {task.name} (difficulty: {task.difficulty:.1f})")
