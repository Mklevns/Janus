# mklevns/janus/Mklevns-Janus-377dbdd2e196e36c324f61780a3b40b78803255b/physics_discovery_extensions.py
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
from symmetry_detection_fix import PhysicsSymmetryDetector
import networkx as nx
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
import pickle
from copy import deepcopy
from progressive_grammar_system import Expression, Variable, ProgressiveGrammar

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
        """Generate candidate conservation expressions using grammar guidance."""

        expressions_by_complexity: Dict[int, List['Expression']] = {1: []}
        unique: Dict[str, 'Expression'] = {}

        # Include variables and constants as base expressions
        base_expressions: List['Expression'] = []
        for var in variables:
            expressions_by_complexity[1].append(var)
            unique[str(var.symbolic)] = var
            base_expressions.append(var)

        for c_val in self.grammar.primitives.get('constants', {}).values():
            const_expr = self.grammar.create_expression('const', [c_val])
            if const_expr and str(const_expr.symbolic) not in unique:
                expressions_by_complexity[1].append(const_expr)
                unique[str(const_expr.symbolic)] = const_expr
                base_expressions.append(const_expr)

        all_candidates = list(base_expressions)

        # Iteratively build more complex expressions
        for complexity in range(2, max_complexity + 1):
            level_candidates: List['Expression'] = []

            # Unary operations
            for op in self.grammar.primitives.get('unary_ops', []):
                for sub_c in range(1, complexity):
                    for expr in expressions_by_complexity.get(sub_c, []):
                        if expr.complexity + 1 == complexity:
                            new_expr = self.grammar.create_expression(op, [expr])
                            if new_expr and str(new_expr.symbolic) not in unique:
                                unique[str(new_expr.symbolic)] = new_expr
                                level_candidates.append(new_expr)

            # Binary operations
            for op in self.grammar.primitives.get('binary_ops', []):
                for c1 in range(1, complexity):
                    c2 = complexity - 1 - c1
                    if c2 < 1:
                        continue
                    for expr1 in expressions_by_complexity.get(c1, []):
                        for expr2 in expressions_by_complexity.get(c2, []):
                            new_expr = self.grammar.create_expression(op, [expr1, expr2])
                            if new_expr and str(new_expr.symbolic) not in unique:
                                unique[str(new_expr.symbolic)] = new_expr
                                level_candidates.append(new_expr)

            if level_candidates:
                expressions_by_complexity[complexity] = level_candidates
                all_candidates.extend(level_candidates)

        return all_candidates
    
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
                    # _crossover now returns two children
                    child1, child2 = self._crossover(parent1, parent2)

                    children_to_process = []
                    if child1:
                        children_to_process.append(child1)
                    if child2:
                        children_to_process.append(child2)
                else:
                    # Direct reproduction
                    # Create a list to have a consistent processing path
                    children_to_process = [np.random.choice(selected)]
                
                for child_candidate in children_to_process:
                    if len(next_population) >= self.population_size:
                        break # Stop if population is full

                    mutated_child = child_candidate # Initialize with the candidate
                    # Mutation
                    if np.random.random() < self.mutation_rate:
                        # _mutate returns a new mutated expression or the original if mutation failed
                        mutated_child = self._mutate(child_candidate, variables)

                    if mutated_child and mutated_child.complexity <= max_complexity:
                        next_population.append(mutated_child)
            
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

    def _get_all_subexpressions(self, expression: 'Expression') -> List['Expression']:
        """Recursively collect all subexpressions from an expression tree."""
        nodes = []
        if isinstance(expression, Expression):
            nodes.append(expression)
            for op in expression.operands:
                nodes.extend(self._get_all_subexpressions(op))
        return nodes
    
    def _crossover(self,
                  parent1: 'Expression',
                  parent2: 'Expression') -> Tuple[Optional['Expression'], Optional['Expression']]:
        """Crossover two expressions by swapping subtrees."""
        # Make copies to avoid modifying the original parents
        p1_copy = pickle.loads(pickle.dumps(parent1))
        p2_copy = pickle.loads(pickle.dumps(parent2))

        # Get all possible crossover points (subexpressions)
        p1_nodes = self._get_all_subexpressions(p1_copy)
        p2_nodes = self._get_all_subexpressions(p2_copy)

        if not p1_nodes or not p2_nodes:
            # If crossover is not possible (e.g., one parent is a terminal),
            # return copies of the original parents.
            return p1_copy, p2_copy

        # Select random subtrees to swap
        # Ensure that we are selecting actual subexpressions, not terminals if they cannot be swapped meaningfully.
        # For simplicity, we assume _get_all_subexpressions returns swappable nodes.
        crossover_point1 = np.random.choice(p1_nodes)
        crossover_point2 = np.random.choice(p2_nodes)

        # Swap the operator and operands attributes of the selected nodes.
        # This effectively swaps the subtrees rooted at these nodes.
        crossover_point1.operator, crossover_point2.operator = (
            crossover_point2.operator,
            crossover_point1.operator,
        )
        crossover_point1.operands, crossover_point2.operands = (
            crossover_point2.operands,
            crossover_point1.operands,
        )

        # Call __post_init__ on the modified crossover points first
        if hasattr(crossover_point1, '__post_init__'):
            crossover_point1.__post_init__()
        if hasattr(crossover_point2, '__post_init__'):
            crossover_point2.__post_init__()

        # Recalculate complexity and symbolic form for both modified expressions
        p1_copy.__post_init__()
        p2_copy.__post_init__()

        return p1_copy, p2_copy
    
    def _mutate(self,
               expression: 'Expression',
               variables: List['Variable']) -> 'Expression':
        """Mutate an expression by modifying a random node."""
        expr_copy = pickle.loads(pickle.dumps(expression))
        nodes = self._get_all_subexpressions(expr_copy)

        if not nodes:
            return expr_copy

        # Select a random node to mutate
        node_to_mutate = np.random.choice(nodes)

        # Apply a random mutation
        mutation_type = np.random.choice(['operator', 'operand'])

        if mutation_type == 'operator' and node_to_mutate.operator not in ['var', 'const']:
            all_ops = list(self.grammar.primitives['binary_ops'] | self.grammar.primitives['unary_ops'])
            # Ensure arity matches
            current_arity = len(node_to_mutate.operands)
            possible_new_ops = [
                op
                for op in all_ops
                if (
                    op in self.grammar.primitives['binary_ops'] and current_arity == 2
                )
                or (
                    op in self.grammar.primitives['unary_ops'] and current_arity == 1
                )
            ]
            if possible_new_ops:
                node_to_mutate.operator = np.random.choice(possible_new_ops)

        elif mutation_type == 'operand' and node_to_mutate.operands:
            if node_to_mutate.operator == 'const':
                # If it's a const node, only replace its value with another number.
                node_to_mutate.operands[0] = np.random.randn()
            else:
                op_idx = np.random.randint(0, len(node_to_mutate.operands))
                # Replace an operand with a new random terminal (variable or constant)
                if np.random.random() < 0.5:
                    node_to_mutate.operands[op_idx] = np.random.choice(variables)
                else:
                    node_to_mutate.operands[op_idx] = np.random.randn()

        # Recalculate properties
        expr_copy.__post_init__()

        return expr_copy