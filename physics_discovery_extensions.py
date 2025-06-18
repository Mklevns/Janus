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
from functools import lru_cache
from itertools import product, combinations_with_replacement
from progressive_grammar_system import Expression, Variable, ProgressiveGrammar
from optimized_candidate_generation import OptimizedCandidateGenerator
import random
import psutil
import os
import time
import logging
import cProfile
import pstats
from io import StringIO

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
            cache_size=10000       # Adjust based on available memory
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

        # Use optimized generator instead of the old, slow _generate_candidates
        candidates = self._optimizer.generate_candidates(
            variables,
            max_complexity,
            fitness_threshold=0.5  # Optional: prune low-promise expressions
        )

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

    def _test_conservation(self,
                           expression: 'Expression',
                           trajectories: np.ndarray,
                           variables: List['Variable']) -> Tuple[bool, float]:
        """Test if expression is conserved over trajectories."""
        values = []

        for t in range(trajectories.shape[0]):
            subs = {var.symbolic: trajectories[t, var.index] for var in variables}
            try:
                value = float(expression.symbolic.subs(subs))
                values.append(value)
            except (TypeError, ValueError):
                return False, float('inf')

        values = np.array(values)
        if len(values) == 0:
            return False, float('inf')
            
        mean_val = np.mean(values)
        if np.isclose(mean_val, 0):
            normalized_variance = np.var(values)
        else:
            normalized_variance = np.var(values) / (mean_val**2 + 1e-10)

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
        if self._is_even_in_velocities(expression):
            symmetries.append('time_reversal')
        if self._has_scaling_symmetry(expression):
            symmetries.append('scaling')
        return symmetries

    def _is_even_in_velocities(self, expression: 'Expression') -> bool:
        """Check if expression is even in velocity-like variables."""
        return True  # Placeholder

    def _has_scaling_symmetry(self, expression: 'Expression') -> bool:
        """Check if expression has scaling symmetry."""
        return True  # Placeholder


class SymbolicRegressor:
    """Symbolic regression for discovering equations of motion."""

    def __init__(self, grammar: 'ProgressiveGrammar', **kwargs):
        self.grammar = grammar
        self.population_size = kwargs.get('population_size', 100)
        self.generations = kwargs.get('generations', 50)
        self.max_complexity = kwargs.get('max_complexity', 10)
        self.mutation_rate = kwargs.get('mutation_rate', 0.1)
        self.crossover_rate = kwargs.get('crossover_rate', 0.7)

        # Use optimized generator
        self._optimizer = OptimizedCandidateGenerator(
            self.grammar,
            enable_parallel=True
        )

    def fit(self,
           X: np.ndarray,
           y: np.ndarray,
           variables: List['Variable'],
           max_complexity: int = 15) -> 'Expression':
        """
        Fit data using genetic programming for symbolic regression.
        """
        # Set max_complexity for this fit run, overriding the instance default if provided
        self.max_complexity = max_complexity
        population = self._initialize_population(variables)

        for generation in range(self.generations):
            fitness_scores = [self._evaluate_fitness(expr, X, y, variables) for expr in population]
            selected = self._tournament_selection(population, fitness_scores)
            next_population = []

            while len(next_population) < self.population_size:
                if np.random.random() < self.crossover_rate and len(selected) >= 2:
                    parent1, parent2 = random.sample(selected, 2)
                    children = self._crossover(parent1, parent2)
                else:
                    children = [deepcopy(random.choice(selected))]

                for child in children:
                    if np.random.random() < self.mutation_rate:
                        child = self._mutate(child, variables)
                    if child and child.complexity <= self.max_complexity:
                        next_population.append(child)
                    if len(next_population) >= self.population_size:
                        break
            
            population = next_population

        final_fitness = [self._evaluate_fitness(expr, X, y, variables) for expr in population]
        best_idx = np.argmax(final_fitness)
        return population[best_idx]

    def _initialize_population(self, variables: List['Variable']) -> List['Expression']:
        """Initialize population with optimized generation."""
        all_candidates = self._optimizer.generate_candidates(
            variables,
            max_complexity=5,  # Start with simpler expressions
        )

        if len(all_candidates) > self.population_size:
            population = random.sample(all_candidates, self.population_size)
        else:
            population = all_candidates
            while len(population) < self.population_size and population:
                parent = random.choice(population)
                mutated = self._mutate(parent, variables)
                if mutated:
                    population.append(mutated)
        return population

    def _evaluate_fitness(self,
                      expression: 'Expression',
                      X: np.ndarray,
                      y: np.ndarray,
                      variables: List['Variable']) -> float:
        predictions = []
        for i in range(X.shape[0]):
            subs = {var.symbolic: X[i, j] for j, var in enumerate(variables)}
            try:
                pred = float(expression.symbolic.subs(subs))
                if not np.isfinite(pred):
                    return -np.inf
                predictions.append(pred)
            except Exception:
                return -np.inf

        preds = np.array(predictions)
        mse = mean_squared_error(y, preds)
        complexity_penalty = 0.01 * expression.complexity
        return -mse - complexity_penalty

    def _tournament_selection(self,
                            population: List['Expression'],
                            fitness_scores: List[float],
                            tournament_size: int = 3) -> List['Expression']:
        selected = []
        for _ in range(len(population)):
            indices = np.random.choice(len(population), tournament_size, replace=False)
            winner_idx = max(indices, key=lambda i: fitness_scores[i])
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
        p1_copy = deepcopy(parent1)
        p2_copy = deepcopy(parent2)

        p1_nodes = self._get_all_subexpressions(p1_copy)
        p2_nodes = self._get_all_subexpressions(p2_copy)

        if not p1_nodes or not p2_nodes:
            return p1_copy, p2_copy

        crossover_point1 = random.choice(p1_nodes)
        crossover_point2 = random.choice(p2_nodes)

        # The swap needs to happen at the parent level
        def find_and_replace(root, target, replacement):
            if root is target:
                return replacement
            if isinstance(root, Expression):
                for i, op in enumerate(root.operands):
                    if op is target:
                        root.operands[i] = replacement
                        return root
                    else:
                        find_and_replace(op, target, replacement)
            return root
        
        find_and_replace(p1_copy, crossover_point1, deepcopy(crossover_point2))
        find_and_replace(p2_copy, crossover_point2, deepcopy(crossover_point1))
        
        if hasattr(p1_copy, '__post_init__'): p1_copy.__post_init__()
        if hasattr(p2_copy, '__post_init__'): p2_copy.__post_init__()

        return p1_copy, p2_copy

    def _mutate(self,
               expression: 'Expression',
               variables: List['Variable']) -> 'Expression':
        expr_copy = deepcopy(expression)
        nodes = self._get_all_subexpressions(expr_copy)

        if not nodes:
            return expr_copy

        node_to_mutate = random.choice(nodes)
        
        # Replace the node with a new random expression
        parent_node = self._find_parent(expr_copy, node_to_mutate)
        new_random_expr = self._random_expression(variables, max_complexity=3)

        if parent_node and new_random_expr:
            for i, child in enumerate(parent_node.operands):
                if child is node_to_mutate:
                    parent_node.operands[i] = new_random_expr
                    break
        elif not parent_node: # It's the root
             expr_copy = new_random_expr

        if hasattr(expr_copy, '__post_init__'): expr_copy.__post_init__()
        return expr_copy

    def _random_expression(self,
                         variables: List['Variable'],
                         max_complexity: int) -> Optional['Expression']:
        """Generate a random expression up to a given complexity."""
        if max_complexity <= 1:
            return random.choice(variables)
        
        op_type = random.choice(['unary', 'binary'])
        
        if op_type == 'unary':
            op = random.choice(list(self.grammar.primitives['unary_ops']))
            operand = self._random_expression(variables, max_complexity - 1)
            return self.grammar.create_expression(op, [operand]) if operand else None
        else:
            op = random.choice(list(self.grammar.primitives['binary_ops']))
            c1 = random.randint(1, max_complexity - 2)
            c2 = max_complexity - 1 - c1
            left = self._random_expression(variables, c1)
            right = self._random_expression(variables, c2)
            return self.grammar.create_expression(op, [left, right]) if left and right else None

    def _find_parent(self, root, target):
        if not isinstance(root, Expression): return None
        for child in root.operands:
            if child is target: return root
            parent = self._find_parent(child, target)
            if parent: return parent
        return None

# --- New functions and classes to be added ---

def get_memory_usage_mb():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def monitor_generation_performance(detector: ConservationDetector,
                                   variables: List['Variable'],
                                   max_complexity: int):
    """Monitor and log generation performance."""
    logger = logging.getLogger(__name__)

    start_time = time.time()
    start_memory = get_memory_usage_mb()

    candidates = detector._optimizer.generate_candidates(variables, max_complexity)

    elapsed = time.time() - start_time
    memory_used = get_memory_usage_mb() - start_memory

    logger.info(f"Candidate generation completed:")
    logger.info(f"  - Time: {elapsed:.2f}s")
    logger.info(f"  - Candidates: {len(candidates)}")
    if elapsed > 0:
        logger.info(f"  - Rate: {len(candidates)/elapsed:.1f} expr/s")
    else:
        logger.info(f"  - Rate: N/A (elapsed time was zero)")
    logger.info(f"  - Memory: {memory_used:.1f} MB")

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
        return {'enable_parallel': False, 'beam_width': 100, 'max_combinations_per_level': 50, 'cache_size': 1000}
    @staticmethod
    def thorough_search():
        return {'enable_parallel': True, 'beam_width': 5000, 'max_combinations_per_level': 500, 'cache_size': 50000}
    @staticmethod
    def memory_constrained():
        return {'enable_parallel': False, 'beam_width': 500, 'max_combinations_per_level': 100, 'cache_size': 5000}

def create_optimized_detector(profile='balanced', grammar: Optional[ProgressiveGrammar] = None) -> ConservationDetector:
    """Create detector with optimized generation."""
    if grammar is None:
        grammar = ProgressiveGrammar()

    profiles = {
        'fast': GenerationProfiles.fast_exploration(),
        'thorough': GenerationProfiles.thorough_search(),
        'memory': GenerationProfiles.memory_constrained(),
        'balanced': {'enable_parallel': True, 'beam_width': 1000, 'cache_size': 10000}
    }
    config = profiles.get(profile, profiles['balanced'])
    
    optimizer = OptimizedCandidateGenerator(grammar, **config)
    detector = ConservationDetector(grammar)
    detector._optimizer = optimizer
    return detector

def profile_generation(detector: ConservationDetector, variables: List['Variable'], max_complexity: int):
    """Profile the generation process."""
    profiler = cProfile.Profile()
    profiler.enable()
    candidates = detector._optimizer.generate_candidates(variables, max_complexity)
    profiler.disable()

    s = StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(20)
    
    print(f"\nGenerated {len(candidates)} candidates")
    print("\nTop time-consuming functions:")
    print(s.getvalue())
    return candidates
