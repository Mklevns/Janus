# =============================================================================
# janus/experiments/physics/generic.py
"""Generic physics discovery experiment for custom environments."""

import numpy as np
from typing import Type, Optional

from janus.experiments.base import BaseExperiment
from janus.experiments.registry import register_experiment
from janus.config.models import ExperimentResult
from janus.physics.environments.base import PhysicsEnvironment
from janus.core.grammar import ProgressiveGrammar
from janus.core.expression import Variable


@register_experiment(
    name="generic_physics_discovery",
    category="physics",
    aliases=["custom_physics"],
    description="Generic physics discovery for custom environments",
    tags=["generic", "custom", "flexible"]
)
class GenericPhysicsDiscovery(BaseExperiment):
    """
    Generic experiment for discovering physics laws in custom environments.
    
    This can be used with any PhysicsEnvironment implementation.
    """
    
    def __init__(self, 
                 config: ExperimentConfig,
                 env_class: Optional[Type[PhysicsEnvironment]] = None,
                 **kwargs):
        super().__init__(config, **kwargs)
        self.env_class = env_class or config.env_class
        
    def setup(self):
        """Setup custom environment."""
        if not self.env_class:
            raise ValueError("env_class must be provided for generic physics discovery")
            
        # Create environment instance
        self.env = self.env_class(
            params=self.config.env_params,
            noise_level=self.config.noise_level
        )
        
        # Generate data
        self.data = self._generate_data()
        
        # Create variables from environment
        self.variables = self._create_variables_from_env()
        
        # Setup grammar and algorithm
        self.grammar = ProgressiveGrammar()
        self._setup_algorithm()
        
    def _generate_data(self) -> np.ndarray:
        """Generate trajectory data from environment."""
        trajectories = []
        
        for _ in range(self.config.n_trajectories):
            # Random initial conditions
            n_vars = len(self.env.state_vars)
            initial = np.random.randn(n_vars) * 2.0
            
            # Generate trajectory
            t_span = np.linspace(0, self.config.trajectory_length,
                               int(self.config.trajectory_length / self.config.sampling_rate))
            
            trajectory = self.env.generate_trajectory(initial, t_span)
            trajectories.append(trajectory)
            
        return np.vstack(trajectories)
        
    def _create_variables_from_env(self) -> List[Variable]:
        """Create variables based on environment state variables."""
        variables = []
        
        for i, var_name in enumerate(self.env.state_vars):
            # Try to infer properties from variable name
            properties = self._infer_variable_properties(var_name)
            variables.append(Variable(var_name, i, properties))
            
        return variables
        
    def _infer_variable_properties(self, var_name: str) -> Dict[str, Any]:
        """Infer variable properties from name."""
        properties = {}
        
        # Common patterns
        if any(p in var_name.lower() for p in ['x', 'y', 'z', 'pos', 'position']):
            properties['type'] = 'position'
            properties['units'] = 'm'
        elif any(v in var_name.lower() for v in ['v', 'vel', 'velocity', 'dot']):
            properties['type'] = 'velocity'
            properties['units'] = 'm/s'
        elif any(a in var_name.lower() for a in ['a', 'acc', 'acceleration']):
            properties['type'] = 'acceleration'
            properties['units'] = 'm/s^2'
        elif any(t in var_name.lower() for t in ['theta', 'phi', 'angle']):
            properties['type'] = 'angle'
            properties['units'] = 'rad'
        elif any(w in var_name.lower() for w in ['omega', 'angular']):
            properties['type'] = 'angular_velocity'
            properties['units'] = 'rad/s'
            
        return properties
        
    def run(self, run_id: int = 0) -> ExperimentResult:
        """Run discovery on custom environment."""
        # Implementation similar to HarmonicOscillatorDiscovery
        # but more generic to handle any environment
        
        discovered_expr = self.algorithm.discover(
            data=self.data,
            variables=self.variables,
            max_iterations=self.config.max_iterations
        )
        
        # Check against ground truth if available
        accuracy = 1.0
        if hasattr(self.env, 'get_ground_truth_laws'):
            ground_truth = self.env.get_ground_truth_laws()
            accuracy = self._compute_accuracy(discovered_expr, ground_truth)
            
        result = ExperimentResult(
            config=self.config,
            run_id=run_id,
            discovered_law=str(discovered_expr),
            symbolic_accuracy=accuracy,
            law_complexity=len(str(discovered_expr))
        )
        
        return result
        
    def teardown(self):
        """Clean up resources."""
        self.data = None
        self.env = None
