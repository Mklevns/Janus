# =============================================================================
# janus/experiments/physics/harmonic_oscillator.py
"""Harmonic oscillator discovery experiment."""

import numpy as np
from typing import List, Dict, Any

from janus.experiments.base import BaseExperiment
from janus.experiments.registry import register_experiment
from janus.config.models import ExperimentResult
from janus.physics.environments.harmonic import HarmonicOscillatorEnv
from janus.physics.algorithms import create_algorithm
from janus.core.grammar import ProgressiveGrammar
from janus.core.expression import Variable


@register_experiment(
    name="harmonic_oscillator_discovery",
    category="physics",
    aliases=["ho_discovery", "spring_discovery"],
    description="Discover F = -kx for harmonic oscillator",
    tags=["classical", "mechanics", "fundamental"],
    supported_algorithms=["genetic", "reinforcement", "hybrid"]
)
class HarmonicOscillatorDiscovery(BaseExperiment):
    """
    Experiment to discover Hooke's law from spring oscillator data.
    
    Expected discovery: F = -kx or energy E = 0.5*k*x^2 + 0.5*m*v^2
    """
    
    def setup(self):
        """Setup harmonic oscillator environment."""
        # Create environment
        self.env = HarmonicOscillatorEnv(
            params={
                'k': self.config.env_params.get('k', 1.0),
                'm': self.config.env_params.get('m', 1.0)
            },
            noise_level=self.config.noise_level
        )
        
        # Generate trajectories
        self.logger.info(f"Generating {self.config.n_trajectories} trajectories...")
        self.trajectories = []
        
        for i in range(self.config.n_trajectories):
            # Random initial conditions
            x0 = np.random.uniform(-2, 2)
            v0 = np.random.uniform(-2, 2)
            initial_conditions = np.array([x0, v0])
            
            # Time span
            t_span = np.linspace(0, self.config.trajectory_length, 
                               int(self.config.trajectory_length / self.config.sampling_rate))
            
            # Generate trajectory
            trajectory = self.env.generate_trajectory(initial_conditions, t_span)
            self.trajectories.append(trajectory)
            
        # Stack all trajectories
        self.data = np.vstack(self.trajectories)
        self.logger.info(f"Generated data shape: {self.data.shape}")
        
        # Create variables
        self.variables = [
            Variable('x', 0, properties={'units': 'm', 'type': 'position'}),
            Variable('v', 1, properties={'units': 'm/s', 'type': 'velocity'})
        ]
        
        # Create grammar
        self.grammar = ProgressiveGrammar()
        
        # Create algorithm
        self.algorithm = create_algorithm(
            self.config.algorithm,
            grammar=self.grammar,
            data=self.data,
            variables=self.variables,
            config=self.config.algorithm_params
        )
        
    def run(self, run_id: int = 0) -> ExperimentResult:
        """Run discovery algorithm."""
        self.logger.info(f"Starting discovery with {self.config.algorithm} algorithm...")
        
        # Set random seed for reproducibility
        np.random.seed(self.config.seed + run_id)
        
        # Run discovery
        if self.config.algorithm == 'genetic':
            result = self._run_genetic_discovery()
        elif self.config.algorithm == 'reinforcement':
            result = self._run_rl_discovery()
        else:
            raise ValueError(f"Unknown algorithm: {self.config.algorithm}")
            
        return result
        
    def _run_genetic_discovery(self) -> ExperimentResult:
        """Run genetic algorithm discovery."""
        from janus.physics.algorithms.genetic import SymbolicRegressor
        
        # Fit regressor
        best_expr = self.algorithm.fit(
            X=self.data[:, :-1],  # Features (x, v)
            y=self.data[:, -1],   # Target (next state or force)
            max_complexity=self.config.max_complexity
        )
        
        # Create result
        result = ExperimentResult(
            config=self.config,
            run_id=self.context.run_id,
            discovered_law=str(best_expr.symbolic),
            symbolic_accuracy=1.0 - best_expr.mse,
            law_complexity=best_expr.complexity
        )
        
        # Check if we discovered the true law
        true_law_discovered = self._check_true_law(best_expr)
        result.metadata['true_law_discovered'] = true_law_discovered
        
        return result
        
    def _run_rl_discovery(self) -> ExperimentResult:
        """Run reinforcement learning discovery."""
        from janus.core.symbolic_discovery import SymbolicDiscoveryEnv
        from janus.core.hypothesis_network import HypothesisNet, PPOTrainer
        
        # Create RL environment
        discovery_env = SymbolicDiscoveryEnv(
            grammar=self.grammar,
            target_data=self.data,
            variables=self.variables,
            max_depth=self.config.max_depth,
            max_complexity=self.config.max_complexity
        )
        
        # Create policy network
        policy = HypothesisNet(
            observation_dim=discovery_env.observation_space.shape[0],
            action_dim=discovery_env.action_space.n,
            hidden_dim=self.config.algorithm_params.get('hidden_dim', 256)
        )
        
        # Create trainer
        trainer = PPOTrainer(policy, discovery_env)
        
        # Train
        trainer.train(
            n_episodes=self.config.algorithm_params.get('n_episodes', 1000),
            checkpoint_callback=lambda: self.checkpoint(f"rl_checkpoint_ep{trainer.episode}")
        )
        
        # Get best discovered expression
        best_expr = trainer.best_expression
        
        # Create result
        result = ExperimentResult(
            config=self.config,
            run_id=self.context.run_id,
            discovered_law=str(best_expr.symbolic) if best_expr else "None",
            symbolic_accuracy=trainer.best_reward,
            law_complexity=best_expr.complexity if best_expr else 0,
            n_experiments_to_convergence=trainer.episodes_to_convergence
        )
        
        return result
        
    def _check_true_law(self, expr) -> bool:
        """Check if discovered expression matches true law."""
        expr_str = str(expr.symbolic).replace(' ', '')
        
        # Known forms of harmonic oscillator laws
        true_laws = [
            '-k*x',           # Force law
            '-1.0*x',         # Normalized force
            '0.5*k*x**2',     # Potential energy
            '0.5*x**2',       # Normalized potential
            '0.5*v**2+0.5*x**2'  # Total energy (normalized)
        ]
        
        # Check if expression matches any true form
        for true_law in true_laws:
            if self._expressions_equivalent(expr_str, true_law):
                return True
                
        return False
        
    def _expressions_equivalent(self, expr1: str, expr2: str) -> bool:
        """Check if two expressions are symbolically equivalent."""
        try:
            import sympy as sp
            # Parse expressions
            e1 = sp.parse_expr(expr1)
            e2 = sp.parse_expr(expr2)
            # Check equivalence
            return sp.simplify(e1 - e2) == 0
        except:
            # Fallback to string comparison
            return expr1 == expr2
            
    def teardown(self):
        """Clean up resources."""
        # Clean up algorithm resources
        if hasattr(self.algorithm, 'cleanup'):
            self.algorithm.cleanup()
            
        # Clear large data arrays
        self.data = None
        self.trajectories = None
        
        self.logger.info("Cleanup complete")
