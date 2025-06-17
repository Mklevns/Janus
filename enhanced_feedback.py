"""
Enhanced Feedback Loop System for Janus
=======================================

Implements tighter integration between components for more effective discovery.
"""

import numpy as np
import torch
import torch.nn as nn
import sympy as sp
from typing import Dict, List, Tuple, Optional, Any, Deque
from collections import deque
from dataclasses import dataclass
import time

from symbolic_discovery_env import SymbolicDiscoveryEnv, ExpressionNode
from hypothesis_policy_network import HypothesisNet, PPOTrainer
from progressive_grammar_system import Expression


class ConservationBiasedReward:
    """
    Calculates a reward bonus by checking if an expression is numerically conserved
    across trajectory data and respects common physical symmetries.
    """
    def __init__(self, variance_threshold: float = 1e-5):
        self.variance_threshold = variance_threshold

    def compute_conservation_bonus(self, expression: str, data: np.ndarray, variables: List[Any]) -> float:
        """
        Computes a bonus score based on conservation and symmetry principles.

        Args:
            expression: The symbolic expression string.
            data: The training data, with columns corresponding to variables.
            variables: A list of variable objects, used to map symbols to data columns.

        Returns:
            A float between 0 and 1 representing the conservation bonus.
        """
        try:
            # Use a local dict for sympy parsing to avoid namespace conflicts
            sym_expr = sp.sympify(expression, locals={var.name: sp.Symbol(var.name) for var in variables})
        except (sp.SympifyError, SyntaxError):
            return 0.0  # Expression is not valid

        # Get a score for how numerically constant the expression is over the data
        variance_score = self._compute_trajectory_variance_score(sym_expr, data, variables)

        # Get a score for how well the expression respects physical symmetries
        symmetry_score = self._test_symmetries(sym_expr, variables)

        # The final bonus requires both numerical conservation and physical plausibility
        return variance_score * symmetry_score

    def _compute_trajectory_variance_score(self, sym_expr: sp.Expr, data: np.ndarray, variables: List[Any]) -> float:
        """
        Evaluates the expression over the trajectory and returns a score based on low variance.
        """
        if sym_expr.is_number:
            return 0.0  # A constant number is trivially conserved but not a law

        # Map symbolic variables to their corresponding data columns
        sym_to_idx = {v.symbolic: v.index for v in variables}
        
        # Ensure all variables in the expression are in our variable list
        present_vars = [s for s in sym_expr.free_symbols if s in sym_to_idx]
        if not present_vars:
            return 0.0

        try:
            # Use sympy's lambdify for fast numerical evaluation
            func = sp.lambdify(present_vars, sym_expr, 'numpy')
            
            # Prepare arguments for the function in the correct order
            eval_data = [data[:, sym_to_idx[s]] for s in present_vars]

            values = func(*eval_data)
            
            if not np.all(np.isfinite(values)):
                return 0.0 # Invalid numerical results (e.g., from log(-1) or 1/0)

            # Calculate normalized variance to make the score scale-invariant
            mean_val = np.mean(values)
            variance = np.var(values)
            
            # Avoid division by zero for expressions that evaluate to a constant near zero
            if np.isclose(mean_val, 0):
                normalized_variance = variance 
            else:
                normalized_variance = variance / (mean_val**2 + 1e-9)
            
            # The score is high when normalized variance is low
            return 1.0 / (1.0 + 100 * normalized_variance) # The '100' is a sensitivity factor

        except Exception:
            return 0.0  # Evaluation failed

    def _test_symmetries(self, sym_expr: sp.Expr, variables: List[Any]) -> float:
        """
        Tests the symbolic expression for common physical symmetries (e.g., time-reversal).
        """
        num_tests = 0
        score = 0.0

        # Heuristically identify position and velocity variables by name
        vel_vars = [v.symbolic for v in variables if v.name.startswith(('v', 'omega'))]
        
        # Test 1: Time-Reversal Invariance for Energy-like quantities
        # A valid energy conservation law should be an even function of velocities (e.g., v^2).
        if vel_vars:
            num_tests += 1
            # Substitute each velocity variable `v` with `-v`
            t_reversed_expr = sym_expr.subs({v: -v for v in vel_vars})
            
            # If the expression is unchanged, it respects this symmetry
            if sp.simplify(sym_expr - t_reversed_expr) == 0:
                score += 1.0
        
        # More symmetry tests (e.g., translational, rotational) could be added here
        
        return (score / num_tests) if num_tests > 0 else 1.0 # If no tests apply, don't penalize

class IntrinsicRewardCalculator:
    """Calculate intrinsic rewards based on novelty and discovery value."""
    
    def __init__(self, 
                 novelty_weight: float = 0.3,
                 diversity_weight: float = 0.2,
                 complexity_growth_weight: float = 0.1,
                 conservation_weight: float = 0.4): # Add new weight
        
        self.novelty_weight = novelty_weight
        self.diversity_weight = diversity_weight
        self.complexity_growth_weight = complexity_growth_weight
        self.conservation_weight = conservation_weight # Initialize new weight
        self.conservation_calculator = ConservationBiasedReward() # Instantiate new module
        
        # History tracking
        self.expression_history: Deque[str] = deque(maxlen=1000)
        self.complexity_history: Deque[int] = deque(maxlen=100)
        self.discovery_embeddings: Deque[np.ndarray] = deque(maxlen=500)
        
        # Novelty detector (simplified version)
        self.expression_cache = {}
        
    def calculate_intrinsic_reward(self,
                                 expression: str,
                                 complexity: int,
                                 extrinsic_reward: float,
                                 embedding: Optional[np.ndarray], # Keep this, but ensure it's Optional if it can be None
                                 data: np.ndarray, # Add data
                                 variables: List[Any] # Add variables
                                 ) -> float:
        """Calculate combined intrinsic and extrinsic reward."""
        
        # 1. Novelty reward
        novelty_reward = self._calculate_novelty_reward(expression)
        
        # 2. Diversity reward
        diversity_reward = self._calculate_diversity_reward(expression, embedding)
        
        # 3. Complexity growth reward
        complexity_reward = self._calculate_complexity_growth_reward(complexity)

        # 4. Conservation Bonus
        conservation_bonus = self.conservation_calculator.compute_conservation_bonus(
            expression, data, variables
        )
        
        # Combine rewards
        intrinsic_reward = (
            self.novelty_weight * novelty_reward +
            self.diversity_weight * diversity_reward +
            self.complexity_growth_weight * complexity_reward +
            self.conservation_weight * conservation_bonus # Add weighted conservation bonus
        )
        
        # Update history
        self.expression_history.append(expression)
        self.complexity_history.append(complexity)
        if embedding is not None:
            self.discovery_embeddings.append(embedding)
        
        # Return combined reward
        return extrinsic_reward + intrinsic_reward
    
    def _calculate_novelty_reward(self, expression: str) -> float:
        """Reward for discovering new expressions."""
        
        # Simple novelty: have we seen this exact expression?
        if expression not in self.expression_cache:
            self.expression_cache[expression] = 1
            base_novelty = 1.0
        else:
            # Decay reward for repeated discoveries
            self.expression_cache[expression] += 1
            base_novelty = 1.0 / self.expression_cache[expression]
        
        # Structural novelty: new operators or combinations
        operators = ['+', '-', '*', '/', '**', 'sin', 'cos', 'log', 'exp']
        used_ops = [op for op in operators if op in expression]
        
        # Bonus for using rare operators
        op_novelty = 0.0
        rare_ops = ['sin', 'cos', 'log', 'exp', '**']
        for op in rare_ops:
            if op in used_ops:
                op_novelty += 0.2
        
        return base_novelty + op_novelty
    
    def _calculate_diversity_reward(self, 
                                  expression: str,
                                  embedding: Optional[np.ndarray]) -> float:
        """Reward for increasing diversity of discoveries."""
        
        if embedding is None or len(self.discovery_embeddings) < 2:
            return 0.0
        
        # Calculate distance to nearest neighbors
        distances = []
        for past_embedding in list(self.discovery_embeddings)[-20:]:
            dist = np.linalg.norm(embedding - past_embedding)
            distances.append(dist)
        
        if distances:
            # Reward expressions that are far from recent discoveries
            min_dist = np.min(distances)
            mean_dist = np.mean(distances)
            
            # Normalize to [0, 1]
            diversity_score = np.tanh(mean_dist / 2.0)
            return diversity_score
        
        return 0.0
    
    def _calculate_complexity_growth_reward(self, complexity: int) -> float:
        """Reward for appropriate complexity growth."""
        
        if len(self.complexity_history) < 2:
            return 0.0
        
        # Calculate complexity trend
        recent_complexities = list(self.complexity_history)[-10:]
        mean_complexity = np.mean(recent_complexities)
        
        # Reward moderate complexity growth
        if complexity > mean_complexity:
            # Growing complexity (good if not too fast)
            growth_rate = (complexity - mean_complexity) / (mean_complexity + 1)
            if growth_rate < 0.5:  # Not growing too fast
                return 0.5 * growth_rate
            else:
                return 0.0  # Penalize huge jumps
        else:
            # Reward finding simpler equivalent expressions
            if complexity < mean_complexity * 0.8:
                return 0.3
        
        return 0.0
    
    def get_exploration_bonus(self) -> float:
        """Get current exploration bonus based on discovery stagnation."""
        
        if len(self.expression_history) < 50:
            return 0.0
        
        # Check for stagnation (repeated discoveries)
        recent = list(self.expression_history)[-50:]
        unique_recent = len(set(recent))
        stagnation_ratio = unique_recent / len(recent)
        
        # Higher bonus when stagnating
        if stagnation_ratio < 0.3:
            return 0.5  # High stagnation
        elif stagnation_ratio < 0.5:
            return 0.2  # Moderate stagnation
        else:
            return 0.0  # Good diversity