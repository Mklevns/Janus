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
from copy import deepcopy # Keep for cases where Expression.clone() is not applicable
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
                    # Ensure the selected choice is cloned if it's an Expression
                    choice = random.choice(selected)
                    children = [choice.clone() if isinstance(choice, Expression) else deepcopy(choice)]


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
            # If not enough unique candidates, clone and mutate to fill population
            population = [cand.clone() if isinstance(cand, Expression) else deepcopy(cand) for cand in all_candidates]
            while len(population) < self.population_size and population: # Ensure population is not empty
                parent_idx = random.randrange(len(population)) # Avoid error if population becomes empty
                parent = population[parent_idx]
                # Ensure parent is an Expression before mutating
                if isinstance(parent, Expression):
                    mutated = self._mutate(parent, variables) # parent is already a clone or fresh
                    if mutated and mutated.complexity <= self.max_complexity : # Check complexity of new mutant
                        population.append(mutated)
                elif len(all_candidates) > 0 : # Fallback if parent is not an expression, try another candidate
                    new_parent_candidate = random.choice(all_candidates)
                    mutated = self._mutate(new_parent_candidate.clone() if isinstance(new_parent_candidate, Expression) else deepcopy(new_parent_candidate), variables)
                    if mutated and mutated.complexity <= self.max_complexity:
                         population.append(mutated)
                else: # No candidates to mutate from, break
                    break
            # Ensure population size
            if len(population) > self.population_size:
                population = random.sample(population, self.population_size)

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
            # selected.append(population[winner_idx]) # Original
            # Selected individuals should be cloned before being added to the next generation's pool
            winner_expr = population[winner_idx]
            selected.append(winner_expr.clone() if isinstance(winner_expr, Expression) else deepcopy(winner_expr))
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
        # Ensure parents are cloned before modification
        p1_copy = parent1.clone() if isinstance(parent1, Expression) else deepcopy(parent1)
        p2_copy = parent2.clone() if isinstance(parent2, Expression) else deepcopy(parent2)

        p1_nodes = self._get_all_subexpressions(p1_copy)
        p2_nodes = self._get_all_subexpressions(p2_copy)

        if not p1_nodes or not p2_nodes:
            return p1_copy, p2_copy # Return clones even if crossover doesn't happen

        crossover_point1 = random.choice(p1_nodes)
        # Make a clone of the subtree to be inserted
        replacement_node_for_p1 = (crossover_point2.clone() if isinstance(crossover_point2, Expression)
                                   else deepcopy(crossover_point2))

        crossover_point2_original_ref = random.choice(p2_nodes) # Find node in original p2_copy for replacement
        replacement_node_for_p2 = (crossover_point1.clone() if isinstance(crossover_point1, Expression)
                                   else deepcopy(crossover_point1))


        # The swap needs to happen at the parent level
        def find_and_replace(root_expr, target_node, replacement_node):
            if root_expr is target_node: # If root is the target, return replacement
                return replacement_node

            if not isinstance(root_expr, Expression): # Should not happen if target is in root_expr
                return root_expr

            new_operands = []
            modified = False
            for i, op_node in enumerate(root_expr.operands):
                if op_node is target_node:
                    new_operands.append(replacement_node)
                    modified = True
                elif isinstance(op_node, Expression):
                    # Recursively call on children, if one of them is the parent of target_node
                    # This part is tricky because we need to replace in place or reconstruct
                    # The current find_and_replace tries to modify in place, which is fine if op_node is mutable
                    # Let's ensure we are modifying a clone.
                    # The issue with in-place modification is that crossover_point1 and crossover_point2 are from clones.
                    # The find_and_replace should operate on p1_copy and p2_copy.

                    # Corrected logic: find_and_replace should modify the parent of the target node.
                    # This simplified version directly modifies the operands list of a CLONED parent.
                    # The initial call to find_and_replace will be on p1_copy and p2_copy which are already clones.
                    # The issue is if target_node is deep within the tree.
                    # The original logic for find_and_replace was better.
                    # Let's revert to a structure closer to the original find_and_replace, ensuring it operates on clones.
                    # The key is that p1_copy and p2_copy are already full clones.
                    # crossover_point1 and crossover_point2 are nodes *within* these clones.
                    # deepcopy(crossover_point2) and deepcopy(crossover_point1) for replacement is correct.

                    # The provided find_and_replace directly modifies root.operands.
                    # This is fine as `p1_copy` and `p2_copy` are already clones of originals.
                    find_and_replace(op_node, target_node, replacement_node) # This recursive call might not be right.
                                                                          # It should be about finding the parent.
                    new_operands.append(op_node) # Add the potentially modified op_node
                else:
                    new_operands.append(op_node)

            if modified: # If direct child was replaced
                 root_expr.operands = new_operands # This line is problematic if find_and_replace is meant to return root
                                               # Let's use the original find_and_replace structure.
                pass # Original find_and_replace modifies in place.

            return root_expr # Should return modified root or new root if root was target

        # Using a more standard way to replace nodes by finding parent and index
        def replace_node_in_tree(root_expr, target_node, replacement_node):
            if root_expr is target_node:
                return replacement_node # Root itself is replaced

            if not isinstance(root_expr, Expression):
                return root_expr

            for i, child_node in enumerate(root_expr.operands):
                if child_node is target_node:
                    root_expr.operands[i] = replacement_node
                    if hasattr(root_expr, '__post_init__'): root_expr.__post_init__() # Re-calculate complexity/symbolic
                    return root_expr # Node found and replaced
                elif isinstance(child_node, Expression):
                    if replace_node_in_tree(child_node, target_node, replacement_node) is not None : # Found in subtree
                        if hasattr(root_expr, '__post_init__'): root_expr.__post_init__()
                        return root_expr
            return None # Target not found in this branch

        replace_node_in_tree(p1_copy, crossover_point1, replacement_node_for_p1)
        replace_node_in_tree(p2_copy, crossover_point2_original_ref, replacement_node_for_p2)

        # Ensure __post_init__ is called on the new trees if not handled by replace_node_in_tree
        if hasattr(p1_copy, '__post_init__'): p1_copy.__post_init__()
        if hasattr(p2_copy, '__post_init__'): p2_copy.__post_init__()

        return p1_copy, p2_copy

    def _mutate(self,
               expression: 'Expression',
               variables: List['Variable']) -> 'Expression':
        # Ensure the expression is cloned before mutation
        expr_copy = expression.clone() if isinstance(expression, Expression) else deepcopy(expression)

        nodes = self._get_all_subexpressions(expr_copy)

        if not nodes: # Should not happen if expr_copy is a valid Expression
            return expr_copy

        node_to_mutate = random.choice(nodes)

        new_random_expr = self._random_expression(variables, max_complexity=3)
        if not new_random_expr: # Could not generate a valid random expression
            return expr_copy # Return original clone

        # Find parent of node_to_mutate in expr_copy and replace it
        # This is similar to the crossover replacement logic

        if expr_copy is node_to_mutate: # If the root is mutated
            expr_copy = new_random_expr
        else:
            parent_node = self._find_parent_and_replace(expr_copy, node_to_mutate, new_random_expr)
            # _find_parent_and_replace should handle the replacement. If it returns None, means node not found as child.

        if hasattr(expr_copy, '__post_init__'): expr_copy.__post_init__()
        return expr_copy

    # Helper for mutation: finds parent and replaces child.
    def _find_parent_and_replace(self, current_expr, target_node, replacement_node):
        if not isinstance(current_expr, Expression):
            return None

        for i, child in enumerate(current_expr.operands):
            if child is target_node:
                current_expr.operands[i] = replacement_node
                return current_expr # Parent found and child replaced

            found_parent = self._find_parent_and_replace(child, target_node, replacement_node)
            if found_parent:
                return found_parent # Replacement happened in a deeper recursive call
        return None


    def _random_expression(self,
                         variables: List['Variable'],
                         max_complexity: int) -> Optional['Expression']:
        """Generate a random expression up to a given complexity."""
        if max_complexity <= 0 : # Ensure max_complexity is positive
             # Potentially return a random variable if max_complexity is too low for an op
            if variables: return random.choice(variables)
            return None


        if max_complexity == 1: # Base case: return a variable or a newly created constant
            if variables and (not self.grammar.primitives['constants'] or random.random() < 0.7): # Favor variables if available
                 # Make sure variable is cloned if it's an Expression-like Variable, though current Variable is not.
                var_choice = random.choice(variables)
                return var_choice # Variables are fine as is, not Expressions themselves.
            elif self.grammar.primitives['constants']:
                const_name = random.choice(list(self.grammar.primitives['constants'].keys()))
                # This part seems to assume constants are expressions, which they are not directly.
                # Let's assume _random_expression should return an Expression object.
                # If we need a constant Expression:
                # return Expression(operator='const', operands=[self.grammar.primitives['constants'][const_name]])
                # However, the problem context usually has variables as base elements.
                # For now, let's stick to returning Variable instances if max_complexity is 1.
                # This means _mutate might replace a node with a Variable.
                # The type hints suggest Expression or Variable.
                # Let's assume we must return an Expression or None.
                # If it's a constant, it should be Expression('const', [value])
                # This requires grammar.create_expression to handle this.
                # The current Expression class expects string for var name, float for const.
                # Let's refine this part of _random_expression.
                # For simplicity, if max_complexity is 1, try to return a Variable as an Expression('var', [var.name])
                # or a constant as Expression('const', [val]).
                if variables and random.random() < 0.5:
                    selected_var = random.choice(variables)
                    # The progressive_grammar_system.Expression has 'var' take a string name.
                    return self.grammar.create_expression('var', [selected_var.name])

                # Create a constant expression
                # For simplicity, let's use a random float constant if no named constants or by chance.
                # This needs to align with how constants are defined in grammar.
                # Assuming constants are just float values for now in this random generation.
                # A better way would be to use grammar.primitives['constants'].
                random_const_val = round(random.uniform(-2,2),2) # Simple random float
                return self.grammar.create_expression('const', [random_const_val])

            elif variables: # Fallback to variable if no constants
                 selected_var = random.choice(variables)
                 return self.grammar.create_expression('var', [selected_var.name])
            else: # No variables or constants to choose from
                return None



        op_type = random.choice(['unary', 'binary'])

        # Ensure there are operators of the chosen type
        unary_ops = list(self.grammar.primitives['unary_ops'])
        binary_ops = list(self.grammar.primitives['binary_ops'])

        if op_type == 'unary' and not unary_ops:
            op_type = 'binary' if binary_ops else None
        if op_type == 'binary' and not binary_ops:
            op_type = 'unary' if unary_ops else None

        if not op_type: # No operators available
            # Fallback to creating a variable or constant if possible (max_complexity > 0)
            return self._random_expression(variables, 1)


        if op_type == 'unary':
            op_symbol = random.choice(unary_ops)
            # Operand complexity must be max_complexity - 1
            # Ensure operand_max_complexity is at least 1 if we want a var/const.
            operand_max_complexity = max(1, max_complexity - 1)
            operand = self._random_expression(variables, operand_max_complexity)
            return self.grammar.create_expression(op_symbol, [operand]) if operand else None
        else: # Binary
            op_symbol = random.choice(binary_ops)
            # Split complexity, ensuring each part is at least 1
            if max_complexity -1 < 2: # Not enough complexity for two operands of min complexity 1
                # Try to make a unary op or a terminal node instead
                if unary_ops:
                     op_symbol = random.choice(unary_ops)
                     operand_max_complexity = max(1, max_complexity - 1)
                     operand = self._random_expression(variables, operand_max_complexity)
                     return self.grammar.create_expression(op_symbol, [operand]) if operand else self._random_expression(variables,1)
                return self._random_expression(variables,1)


            c1 = random.randint(1, max_complexity - 2) # operand 1 complexity
            c2 = max_complexity - 1 - c1 # operand 2 complexity

            left = self._random_expression(variables, c1)
            right = self._random_expression(variables, c2)
            # If either operand failed to generate, this expression fails.
            return self.grammar.create_expression(op_symbol, [left, right]) if left and right else None


    def _find_parent(self, root, target):
        if not isinstance(root, Expression): return None
        for child in root.operands:
            if child is target: return root
            # Important: recurse on child, not on the result of recursion
            parent_found_in_subtree = self._find_parent(child, target)
            if parent_found_in_subtree: return parent_found_in_subtree
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
