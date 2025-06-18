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
from optimized_candidate_generation import OptimizedCandidateGenerator # Added
import random # Added
import psutil # Added
import os # Added
import time # Added
import logging # Added
import cProfile # Added
import pstats # Added
from io import StringIO # Added

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
        # Initialize the optimized generator
        self._optimizer = OptimizedCandidateGenerator(
            self.grammar, 
            enable_parallel=True,  # Use multiple cores
            cache_size=10000      # Adjust based on available memory
        )
    
    def find_conserved_quantities(self, 
                                trajectories: np.ndarray,
                                variables: List['Variable'],
                                max_complexity: int = 10) -> List[PhysicalLaw]:
        """
        Search for expressions that remain constant over trajectories.
        Now uses optimized generation for massive speedup.
        """
        conserved_laws = []
        
        # Use optimized generator instead of _generate_candidates
        candidates = self._optimizer.generate_candidates(
            variables, 
            max_complexity,
            fitness_threshold=0.5  # Optional: prune low-promise expressions
        )
        
        # Rest of the method remains the same
        for candidate in candidates:
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
    
    # Remove or comment out the old _generate_candidates method
    # def _generate_candidates(self,
    #                        variables: List['Variable'],
    #                        max_complexity: int) -> List['Expression']:
    #     """Generate candidate conservation expressions using grammar guidance."""

    #     expressions_by_complexity: Dict[int, List['Expression']] = {1: []}
    #     unique: Dict[str, 'Expression'] = {}

    #     # Include variables and constants as base expressions
    #     base_expressions: List['Expression'] = []
    #     for var in variables:
    #         expressions_by_complexity[1].append(var)
    #         unique[str(var.symbolic)] = var
    #         base_expressions.append(var)

    #     for c_val in self.grammar.primitives.get('constants', {}).values():
    #         const_expr = self.grammar.create_expression('const', [c_val])
    #         if const_expr and str(const_expr.symbolic) not in unique:
    #             expressions_by_complexity[1].append(const_expr)
    #             unique[str(const_expr.symbolic)] = const_expr
    #             base_expressions.append(const_expr)

    #     all_candidates = list(base_expressions)

    #     # Iteratively build more complex expressions
    #     for complexity in range(2, max_complexity + 1):
    #         level_candidates: List['Expression'] = []

    #         # Unary operations
    #         for op in self.grammar.primitives.get('unary_ops', []):
    #             for sub_c in range(1, complexity):
    #                 for expr in expressions_by_complexity.get(sub_c, []):
    #                     if expr.complexity + 1 == complexity:
    #                         new_expr = self.grammar.create_expression(op, [expr])
    #                         if new_expr and str(new_expr.symbolic) not in unique:
    #                             unique[str(new_expr.symbolic)] = new_expr
    #                             level_candidates.append(new_expr)

    #         # Binary operations
    #         for op in self.grammar.primitives.get('binary_ops', []):
    #             for c1 in range(1, complexity):
    #                 c2 = complexity - 1 - c1
    #                 if c2 < 1:
    #                     continue
    #                 for expr1 in expressions_by_complexity.get(c1, []):
    #                     for expr2 in expressions_by_complexity.get(c2, []):
    #                         new_expr = self.grammar.create_expression(op, [expr1, expr2])
    #                         if new_expr and str(new_expr.symbolic) not in unique:
    #                             unique[str(new_expr.symbolic)] = new_expr
    #                             level_candidates.append(new_expr)

    #         if level_candidates:
    #             expressions_by_complexity[complexity] = level_candidates
    #             all_candidates.extend(level_candidates)

    #     return all_candidates
    
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
    """Symbolic regression for discovering equations of motion."""
    
    def __init__(self, grammar: 'ProgressiveGrammar', **kwargs):
        self.grammar = grammar
        self.population_size = kwargs.get('population_size', 100)
        self.generations = kwargs.get('generations', 50)
        self.max_complexity = kwargs.get('max_complexity', 10)
        self.mutation_rate = kwargs.get('mutation_rate', 0.1) # Keep existing or allow override
        self.crossover_rate = kwargs.get('crossover_rate', 0.7) # Keep existing or allow override
        
        # Use optimized generator
        self._optimizer = OptimizedCandidateGenerator(
            self.grammar, # grammar was self.grammar
            enable_parallel=True
        )
    
    def fit(self,
           X: np.ndarray,
           y: np.ndarray,
           variables: List['Variable'], # variables was var_mapping, this is the original signature
           max_complexity: int = 15) -> 'Expression': # max_complexity was part of **fit_params
        """
        Fit data using genetic programming for symbolic regression.
        X: input data (n_samples, n_features)
        y: target values (n_samples,)
        """
        # Initialize population
        # The _initialize_population in the issue description uses self.max_complexity (from __init__)
        # not the max_complexity passed to fit. Sticking to issue description.
        population = self._initialize_population(variables) 
        
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
                    child1, child2 = self._crossover(parent1, parent2)

                    children_to_process = []
                    if child1:
                        children_to_process.append(child1)
                    if child2:
                        children_to_process.append(child2)
                else:
                    children_to_process = [np.random.choice(selected)]
                
                for child_candidate in children_to_process:
                    if len(next_population) >= self.population_size:
                        break

                    mutated_child = child_candidate 
                    if np.random.random() < self.mutation_rate:
                        mutated_child = self._mutate(child_candidate, variables)

                    # The max_complexity for filtering here should be the one from __init__
                    # to be consistent with _initialize_population.
                    if mutated_child and mutated_child.complexity <= self.max_complexity:
                        next_population.append(mutated_child)
            
            population = next_population
        
        final_fitness = [
            self._evaluate_fitness(expr, X, y, variables) 
            for expr in population
        ]
        best_idx = np.argmax(final_fitness)
        
        return population[best_idx]

    def _initialize_population(self, variables: List['Variable']) -> List['Expression']:
        """Initialize population with optimized generation."""
        # Generate diverse initial population efficiently
        all_candidates = self._optimizer.generate_candidates(
            variables, 
            max_complexity=5,  # Start with simpler expressions
            fitness_threshold=None  # Keep all for diversity
        )
        
        # Sample if we have too many
        if len(all_candidates) > self.population_size:
            # import random # Already imported at the top
            population = random.sample(all_candidates, self.population_size)
        else:
            population = all_candidates
            
            # Fill remaining slots with mutations
            while len(population) < self.population_size and population: # Add check for non-empty population before random.choice
                parent = random.choice(population)
                mutated = self._mutate(parent, variables) # Pass variables to mutate
                if mutated:
                    population.append(mutated)
        
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
        
        for i in range(X.shape[0]):
            subs = {
                var.symbolic: X[i, var.index] 
                for var in variables
            }
            try:
                pred = float(expression.symbolic.subs(subs))
            except Exception:
                return -float('inf')
            predictions.append(pred)
        
        preds = np.array(predictions, dtype=float)
        if np.isnan(preds).any():
            return -float('inf')
        
        try:
            mse = mean_squared_error(y, preds)
        except Exception:
            return -float('inf')
        
        complexity_penalty = 0.01 * expression.complexity
        return -mse - complexity_penalty
    
    def _tournament_selection(self,
                            population: List['Expression'],
                            fitness_scores: List[float],
                            tournament_size: int = 3) -> List['Expression']:
        selected = []
        
        for _ in range(len(population)):
            indices = np.random.choice(
                len(population), 
                tournament_size, 
                replace=False
            )
            
            tournament_fitness = [fitness_scores[i] for i in indices]
            winner_idx = indices[np.argmax(tournament_fitness)]
            selected.append(population[winner_idx])
        
        return selected

    def _get_all_subexpressions(self, expression: 'Expression') -> List['Expression']:
        nodes = []
        if isinstance(expression, Expression):
            nodes.append(expression)
            for op in expression.operands:
                nodes.extend(self._get_all_subexpressions(op))
        return nodes
    
    def _crossover(self,
                  parent1: 'Expression',
                  parent2: 'Expression') -> Tuple[Optional['Expression'], Optional['Expression']]:
        p1_copy = pickle.loads(pickle.dumps(parent1))
        p2_copy = pickle.loads(pickle.dumps(parent2))

        p1_nodes = self._get_all_subexpressions(p1_copy)
        p2_nodes = self._get_all_subexpressions(p2_copy)

        if not p1_nodes or not p2_nodes:
            return p1_copy, p2_copy

        crossover_point1 = np.random.choice(p1_nodes)
        crossover_point2 = np.random.choice(p2_nodes)

        crossover_point1.operator, crossover_point2.operator = (
            crossover_point2.operator,
            crossover_point1.operator,
        )
        crossover_point1.operands, crossover_point2.operands = (
            crossover_point2.operands,
            crossover_point1.operands,
        )

        if hasattr(crossover_point1, '__post_init__'):
            crossover_point1.__post_init__()
        if hasattr(crossover_point2, '__post_init__'):
            crossover_point2.__post_init__()

        p1_copy.__post_init__()
        p2_copy.__post_init__()

        return p1_copy, p2_copy
    
    def _mutate(self,
               expression: 'Expression',
               variables: List['Variable']) -> 'Expression':
        expr_copy = pickle.loads(pickle.dumps(expression))
        nodes = self._get_all_subexpressions(expr_copy)

        if not nodes:
            return expr_copy

        node_to_mutate = np.random.choice(nodes)
        mutation_type = np.random.choice(['operator', 'operand'])

        if mutation_type == 'operator' and node_to_mutate.operator not in ['var', 'const']:
            all_ops = list(self.grammar.primitives['binary_ops'] | self.grammar.primitives['unary_ops'])
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
            # This logic for 'const' was in the original, keeping it.
            # The issue description's _initialize_population (which calls _mutate)
            # doesn't seem to create 'const' type Expressions that would flow here.
            # However, _random_expression can create 'const' expressions.
            if node_to_mutate.operator == 'const': 
                node_to_mutate.operands[0] = np.random.randn()
            else:
                op_idx = np.random.randint(0, len(node_to_mutate.operands))
                if np.random.random() < 0.5:
                    node_to_mutate.operands[op_idx] = np.random.choice(variables)
                else:
                    # Create a new 'const' expression for the operand
                    # const_val = np.random.randn()
                    # Assuming ProgressiveGrammar has a way to make a simple constant expression
                    # Or, if operands can be raw values for 'const' type, this might be simpler.
                    # For now, directly assigning the value, assuming it's handled by __post_init__
                    # or that operands can be basic types for some operators.
                    # The original _mutate had `np.random.randn()`, which is a float, not an Expression.
                    # Reverting to the simpler form from original _mutate:
                    node_to_mutate.operands[op_idx] = np.random.randn()


        expr_copy.__post_init__()
        return expr_copy

# --- New functions and classes to be added ---

def get_memory_usage_mb():
    """Get current memory usage in MB."""
    # import psutil # Already imported
    # import os # Already imported
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def monitor_generation_performance(detector: ConservationDetector, 
                                 variables: List['Variable'],
                                 max_complexity: int):
    """Monitor and log generation performance."""
    # import time # Already imported
    # import logging # Already imported
    
    logger = logging.getLogger(__name__)
    
    start_time = time.time()
    start_memory = get_memory_usage_mb()
    
    # Run generation
    # Accessing _optimizer directly as per the issue description
    candidates = detector._optimizer.generate_candidates(
        variables, max_complexity
    )
    
    elapsed = time.time() - start_time
    memory_used = get_memory_usage_mb() - start_memory
    
    # Log performance metrics
    logger.info(f"Candidate generation completed:")
    logger.info(f"  - Time: {elapsed:.2f}s")
    logger.info(f"  - Candidates: {len(candidates)}")
    if elapsed > 0: # Avoid division by zero
        logger.info(f"  - Rate: {len(candidates)/elapsed:.1f} expr/s")
    else:
        logger.info(f"  - Rate: N/A (elapsed time was zero)")
    logger.info(f"  - Memory: {memory_used:.1f} MB")
    
    # Log complexity distribution
    complexity_dist = {}
    for expr in candidates:
        c = expr.complexity
        complexity_dist[c] = complexity_dist.get(c, 0) + 1
    
    logger.info(f"  - Distribution: {complexity_dist}")
    
    return candidates

class GenerationProfiles:
    """Pre-configured generation profiles for different scenarios."""
    
    @staticmethod
    def fast_exploration():
        """Fast exploration for interactive use."""
        return {
            'enable_parallel': False,  # Avoid overhead for small problems
            'beam_width': 100,
            'max_combinations_per_level': 50,
            'cache_size': 1000
        }
    
    @staticmethod
    def thorough_search():
        """Thorough search for batch processing."""
        return {
            'enable_parallel': True,
            'beam_width': 5000,
            'max_combinations_per_level': 500,
            'cache_size': 50000
        }
    
    @staticmethod
    def memory_constrained():
        """For systems with limited memory."""
        return {
            'enable_parallel': False,
            'beam_width': 500,
            'max_combinations_per_level': 100,
            'cache_size': 5000
        }

def create_optimized_detector(profile='balanced', grammar: Optional[ProgressiveGrammar] = None) -> ConservationDetector: # Added grammar argument
    """Create detector with optimized generation."""
    # grammar = ProgressiveGrammar() # Use passed grammar or create new
    if grammar is None:
        grammar = ProgressiveGrammar() # type: ignore
    
    # Select profile
    if profile == 'fast':
        config = GenerationProfiles.fast_exploration()
    elif profile == 'thorough':
        config = GenerationProfiles.thorough_search()
    elif profile == 'memory':
        config = GenerationProfiles.memory_constrained()
    else:  # balanced
        config = {
            'enable_parallel': True,
            'beam_width': 1000,
            'cache_size': 10000
        }
    
    # Create optimized generator
    optimizer = OptimizedCandidateGenerator(grammar, **config) # type: ignore
    
    # Create detector
    detector = ConservationDetector(grammar) # type: ignore
    detector._optimizer = optimizer # type: ignore
    
    return detector


def profile_generation(detector: ConservationDetector, variables: List['Variable'], max_complexity: int):
    """Profile the generation process."""
    # import cProfile # Already imported
    # import pstats # Already imported
    # from io import StringIO # Already imported
    
    profiler = cProfile.Profile()
    
    # Profile the generation
    profiler.enable()
    # Accessing _optimizer directly as per the issue description
    candidates = detector._optimizer.generate_candidates(
        variables, max_complexity
    )
    profiler.disable()
    
    # Get statistics
    s = StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(20)  # Top 20 functions
    
    print(f"\nGenerated {len(candidates)} candidates")
    print("\nTop time-consuming functions:")
    print(s.getvalue())
    
    return candidates
