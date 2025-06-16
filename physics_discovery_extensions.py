"""
Physics Discovery Extensions for Progressive Grammar
===================================================

Advanced functionality for discovering physical laws, including conservation
detection, symmetry analysis, and dimensional analysis.
"""

import numpy as np
import sympy as sp
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass
import networkx as nx
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error


@dataclass
class PhysicalLaw:
    """Represents a discovered physical law with metadata."""
    expression: 'Expression'
    law_type: str  # 'conservation', 'equation_of_motion', 'constraint'
    accuracy: float
    domain_of_validity: Dict[str, Tuple[float, float]]
    symmetries: List[str]
    
    def __str__(self):
        return f"{self.law_type}: {self.expression.symbolic} (accuracy: {self.accuracy:.3f})"


class ConservationDetector:
    """Detects conserved quantities in physical systems."""
    
    def __init__(self, grammar: 'ProgressiveGrammar'):
        self.grammar = grammar
        self.tolerance = 1e-6
        
    def find_conserved_quantities(self, 
                                trajectories: np.ndarray,
                                variables: List['Variable'],
                                max_complexity: int = 10) -> List[PhysicalLaw]:
        """
        Search for expressions that remain constant over trajectories.
        Uses genetic programming with information-theoretic fitness.
        """
        conserved_laws = []
        
        # Generate candidate expressions
        candidates = self._generate_candidates(variables, max_complexity)
        
        for candidate in candidates:
            # Evaluate conservation
            is_conserved, variance = self._test_conservation(
                candidate, 
                trajectories, 
                variables
            )
            
            if is_conserved:
                law = PhysicalLaw(
                    expression=candidate,
                    law_type='conservation',
                    accuracy=1.0 - variance,
                    domain_of_validity=self._compute_domain(trajectories, variables),
                    symmetries=self._detect_symmetries(candidate)
                )
                conserved_laws.append(law)
        
        return conserved_laws
    
    def _generate_candidates(self, 
                           variables: List['Variable'],
                           max_complexity: int) -> List['Expression']:
        """Generate candidate conservation expressions."""
        candidates = []
        
        # Start with simple expressions
        for var in variables:
            candidates.append(var)
        
        # Build more complex expressions iteratively
        for complexity in range(2, max_complexity + 1):
            new_candidates = []
            
            # Binary operations
            for op in ['+', '-', '*', '/', '**']:
                for expr1 in candidates:
                    for expr2 in candidates:
                        if expr1.complexity + expr2.complexity + 1 == complexity:
                            new_expr = self.grammar.create_expression(
                                op, [expr1, expr2]
                            )
                            if new_expr:
                                new_candidates.append(new_expr)
            
            candidates.extend(new_candidates)
        
        return candidates
    
    def _test_conservation(self,
                         expression: 'Expression',
                         trajectories: np.ndarray,
                         variables: List['Variable']) -> Tuple[bool, float]:
        """Test if expression is conserved over trajectories."""
        values = []
        
        for t in range(trajectories.shape[0]):
            # Create variable substitution dictionary
            subs = {
                var.symbolic: trajectories[t, var.index] 
                for var in variables
            }
            
            try:
                # Evaluate expression
                value = float(expression.symbolic.subs(subs))
                values.append(value)
            except:
                return False, float('inf')
        
        # Check if variance is below threshold
        values = np.array(values)
        normalized_variance = np.var(values) / (np.mean(values)**2 + 1e-10)
        
        return normalized_variance < self.tolerance, normalized_variance
    
    def _compute_domain(self,
                       trajectories: np.ndarray,
                       variables: List['Variable']) -> Dict[str, Tuple[float, float]]:
        """Compute domain of validity for discovered law."""
        domain = {}
        for var in variables:
            values = trajectories[:, var.index]
            domain[var.name] = (np.min(values), np.max(values))
        return domain
    
    def _detect_symmetries(self, expression: 'Expression') -> List[str]:
        """Detect symmetries in the expression."""
        symmetries = []
        
        # Check for time reversal symmetry
        if self._is_even_in_velocities(expression):
            symmetries.append('time_reversal')
        
        # Check for scaling symmetry
        if self._has_scaling_symmetry(expression):
            symmetries.append('scaling')
        
        return symmetries
    
    def _is_even_in_velocities(self, expression: 'Expression') -> bool:
        """Check if expression is even in velocity-like variables."""
        # Simplified check - in practice would need more sophisticated analysis
        return True  # Placeholder
    
    def _has_scaling_symmetry(self, expression: 'Expression') -> bool:
        """Check if expression has scaling symmetry."""
        # Check if expression is homogeneous
        return True  # Placeholder


class SymbolicRegressor:
    """Performs symbolic regression to fit data to mathematical expressions."""
    
    def __init__(self, grammar: 'ProgressiveGrammar'):
        self.grammar = grammar
        self.population_size = 100
        self.generations = 50
        self.mutation_rate = 0.1
        self.crossover_rate = 0.7
        
    def fit(self,
           X: np.ndarray,
           y: np.ndarray,
           variables: List['Variable'],
           max_complexity: int = 15) -> 'Expression':
        """
        Fit data using genetic programming for symbolic regression.
        X: input data (n_samples, n_features)
        y: target values (n_samples,)
        """
        # Initialize population
        population = self._initialize_population(variables, max_complexity)
        
        for generation in range(self.generations):
            # Evaluate fitness
            fitness_scores = [
                self._evaluate_fitness(expr, X, y, variables) 
                for expr in population
            ]
            
            # Selection
            selected = self._tournament_selection(population, fitness_scores)
            
            # Create next generation
            next_population = []
            
            while len(next_population) < self.population_size:
                if np.random.random() < self.crossover_rate:
                    # Crossover
                    parent1 = np.random.choice(selected)
                    parent2 = np.random.choice(selected)
                    child = self._crossover(parent1, parent2)
                else:
                    # Direct reproduction
                    child = np.random.choice(selected)
                
                # Mutation
                if np.random.random() < self.mutation_rate:
                    child = self._mutate(child, variables)
                
                if child and child.complexity <= max_complexity:
                    next_population.append(child)
            
            population = next_population
        
        # Return best expression
        final_fitness = [
            self._evaluate_fitness(expr, X, y, variables) 
            for expr in population
        ]
        best_idx = np.argmax(final_fitness)
        
        return population[best_idx]
    
    def _initialize_population(self,
                             variables: List['Variable'],
                             max_complexity: int) -> List['Expression']:
        """Initialize random population of expressions."""
        population = []
        
        # Add simple expressions
        for var in variables:
            population.append(var)
        
        # Generate random expressions
        while len(population) < self.population_size:
            expr = self._random_expression(variables, max_complexity)
            if expr:
                population.append(expr)
        
        return population
    
    def _random_expression(self,
                         variables: List['Variable'],
                         max_complexity: int) -> Optional['Expression']:
        """Generate random expression."""
        if max_complexity <= 1:
            # Return variable or constant
            if np.random.random() < 0.7:
                return np.random.choice(variables)
            else:
                return self.grammar.create_expression(
                    'const', 
                    [np.random.randn()]
                )
        
        # Choose operator
        op_type = np.random.choice(['binary', 'unary'])
        
        if op_type == 'binary':
            op = np.random.choice(list(self.grammar.primitives['binary_ops']))
            left = self._random_expression(variables, max_complexity // 2)
            right = self._random_expression(variables, max_complexity // 2)
            if left and right:
                return self.grammar.create_expression(op, [left, right])
        else:
            op = np.random.choice(list(self.grammar.primitives['unary_ops']))
            operand = self._random_expression(variables, max_complexity - 1)
            if operand:
                return self.grammar.create_expression(op, [operand])
        
        return None
    
    def _evaluate_fitness(self,
                      expression: 'Expression',
                      X: np.ndarray,
                      y: np.ndarray,
                      variables: List['Variable']
    ) -> float:
        
        predictions = []
        
        # 1) Build predictions, bailing out if any evaluation fails
        for i in range(X.shape[0]):
            # Create substitution dictionary
            subs = {
                var.symbolic: X[i, var.index]
                for var in variables
            }
            try:
                # evaluate and coerce to float
                pred = float(expression.symbolic.subs(subs))
            except Exception:
                # invalid expression (e.g. domain error)
                return -float('inf')
            predictions.append(pred)
        
        # 2) Convert to array and guard against NaNs
        preds = np.array(predictions, dtype=float)
        if np.isnan(preds).any():
            return -float('inf')
        
        # 3) Compute MSE; guard against unexpected errors
        try:
            mse = mean_squared_error(y, preds)
        except Exception:
            return -float('inf')
        
        # 4) Complexity penalty
        complexity_penalty = 0.01 * expression.complexity
        
        # 5) Return fitness (higher is better, so negative MSE minus penalty)
        return -mse - complexity_penalty
    
    def _tournament_selection(self,
                            population: List['Expression'],
                            fitness_scores: List[float],
                            tournament_size: int = 3) -> List['Expression']:
        """Tournament selection for genetic algorithm."""
        selected = []
        
        for _ in range(len(population)):
            # Random tournament
            indices = np.random.choice(
                len(population), 
                tournament_size, 
                replace=False
            )
            
            # Select best from tournament
            tournament_fitness = [fitness_scores[i] for i in indices]
            winner_idx = indices[np.argmax(tournament_fitness)]
            selected.append(population[winner_idx])
        
        return selected
    
    def _crossover(self,
                  parent1: 'Expression',
                  parent2: 'Expression') -> Optional['Expression']:
        """Crossover two expressions."""
        # Simplified crossover - swap random subtrees
        # In practice, would need more sophisticated implementation
        if np.random.random() < 0.5:
            return parent1
        else:
            return parent2
    
    def _mutate(self,
               expression: 'Expression',
               variables: List['Variable']) -> 'Expression':
        """Mutate expression."""
        # Simplified mutation - return slightly modified expression
        # In practice, would modify subtrees
        return expression
