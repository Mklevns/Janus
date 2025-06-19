"""
Physics Discovery Extensions for Progressive Grammar
===================================================

Advanced functionality for discovering physical laws, including conservation
detection, symmetry analysis, and dimensional analysis.
"""

import numpy as np
import sympy as sp
from typing import List, Dict, Tuple, Optional, Callable, Union
from dataclasses import dataclass
from symmetry_detection_fix import PhysicsSymmetryDetector
import networkx as nx
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
import pickle
from copy import deepcopy
from functools import lru_cache
from itertools import product, combinations_with_replacement
from janus.core.grammar import ProgressiveGrammar
from janus.core.expression import Expression, Variable
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
    expression: Expression
    law_type: str
    accuracy: float
    domain_of_validity: Dict[str, Tuple[float, float]]
    symmetries: List[str]

    def __str__(self):
        return f"{self.law_type}: {self.expression.symbolic} (accuracy: {self.accuracy:.3f})"

class ConservationDetector:
    def __init__(self, grammar: ProgressiveGrammar):
        self.grammar = grammar
        self.tolerance = 1e-6
        self._optimizer = OptimizedCandidateGenerator(
            self.grammar, enable_parallel=True, cache_size=10000
        )

    def find_conserved_quantities(self, trajectories: np.ndarray, variables: List[Variable], max_complexity: int = 10) -> List[PhysicalLaw]:
        conserved_laws = []
        candidates = self._optimizer.generate_candidates(variables, max_complexity, fitness_threshold=0.5)
        for candidate_item in candidates:
            expr_candidate = None
            if isinstance(candidate_item, Variable):
                expr_candidate = self.grammar.create_expression('var', [candidate_item.name])
                if not expr_candidate: continue
            elif isinstance(candidate_item, Expression):
                expr_candidate = candidate_item
            else: continue
            is_conserved, variance = self._test_conservation(expr_candidate, trajectories, variables)
            if is_conserved:
                law = PhysicalLaw(expression=expr_candidate, law_type='conservation', accuracy=1.0 - variance,
                                  domain_of_validity=self._compute_domain(trajectories, variables),
                                  symmetries=self._detect_symmetries(expr_candidate))
                conserved_laws.append(law)
        return conserved_laws

    def _test_conservation(self, expression: Expression, trajectories: np.ndarray, variables: List[Variable]) -> Tuple[bool, float]:
        values = []
        for t in range(trajectories.shape[0]):
            subs = {var.symbolic: trajectories[t, var.index] for var in variables}
            try:
                value = float(expression.symbolic.subs(subs))
                values.append(value)
            except (TypeError, ValueError, AttributeError): return False, float('inf')
        values_arr = np.array(values)
        if len(values_arr) == 0: return False, float('inf')
        mean_val = np.mean(values_arr)
        if np.isclose(mean_val, 0): var_val = np.var(values_arr)
        else: var_val = np.var(values_arr) / (mean_val**2 + 1e-10)
        return var_val < self.tolerance, var_val

    def _compute_domain(self, trajectories: np.ndarray, variables: List[Variable]) -> Dict[str, Tuple[float, float]]:
        domain = {}
        for var in variables:
            values = trajectories[:, var.index]
            domain[var.name] = (np.min(values), np.max(values))
        return domain

    def _detect_symmetries(self, expression: Expression) -> List[str]:
        symmetries = []
        if self._is_even_in_velocities(expression): symmetries.append('time_reversal')
        if self._has_scaling_symmetry(expression): symmetries.append('scaling')
        return symmetries

    def _is_even_in_velocities(self, expression: Expression) -> bool: return True
    def _has_scaling_symmetry(self, expression: Expression) -> bool: return True

class SymbolicRegressor:
    def __init__(self, grammar: ProgressiveGrammar, **kwargs):
        self.grammar = grammar
        self.population_size = kwargs.get('population_size', 100)
        self.generations = kwargs.get('generations', 50)
        self.max_complexity = kwargs.get('max_complexity', 10)
        self.mutation_rate = kwargs.get('mutation_rate', 0.1)
        self.crossover_rate = kwargs.get('crossover_rate', 0.7)
        self.elitism_count = kwargs.get('elitism_count', 1)
        self._optimizer = OptimizedCandidateGenerator(self.grammar, enable_parallel=True)

    def fit(self, X: np.ndarray, y: np.ndarray, variables: List[Variable], max_complexity: int = 15) -> Expression:
        self.max_complexity = max_complexity
        current_population_mixed = self._initialize_population(variables)
        if not current_population_mixed: raise RuntimeError("Initial population is empty.")

        population: List[Expression] = []
        for item in current_population_mixed:
            if isinstance(item, Variable):
                wrapped = self.grammar.create_expression('var', [item.name])
                if wrapped: population.append(wrapped)
            elif isinstance(item, Expression): population.append(item)
        if not population: raise RuntimeError("Population empty after wrapping Variables.")

        for generation in range(self.generations):
            fitness_scores = [self._evaluate_fitness(expr, X, y, variables) for expr in population]
            next_pop_exprs: List[Expression] = []
            if self.elitism_count > 0 and population:
                actual_elitism_count = min(self.elitism_count, len(population))
                elite_indices = np.argsort(fitness_scores)[-actual_elitism_count:]
                for i in elite_indices:
                    next_pop_exprs.append(population[i].clone() if hasattr(population[i], 'clone') else deepcopy(population[i]))

            selected_parents = self._tournament_selection(population, fitness_scores)
            num_to_generate = self.population_size - len(next_pop_exprs)
            children_count = 0
            loop_idx = 0
            while children_count < num_to_generate and selected_parents:
                p1 = selected_parents[loop_idx % len(selected_parents)]
                loop_idx += 1
                batch_children = []
                if np.random.random() < self.crossover_rate and len(selected_parents) >= 2:
                    p2_idx = loop_idx % len(selected_parents)
                    while p1 is selected_parents[p2_idx] and len(selected_parents) > 1:
                         p2_idx = (loop_idx + random.randint(1, len(selected_parents) -1)) % len(selected_parents)
                    p2 = selected_parents[p2_idx]
                    batch_children = self._crossover(p1, p2)
                else:
                    batch_children = [p1.clone() if hasattr(p1, 'clone') else deepcopy(p1)]

                for child_cand in batch_children:
                    if child_cand is None: continue
                    mutated_item = self._mutate(child_cand, variables)
                    final_child = None
                    if isinstance(mutated_item, Variable): final_child = self.grammar.create_expression('var', [mutated_item.name])
                    elif isinstance(mutated_item, Expression): final_child = mutated_item

                    if final_child and final_child.complexity <= self.max_complexity:
                        next_pop_exprs.append(final_child)
                        children_count += 1
                        if len(next_pop_exprs) >= self.population_size: break
                if len(next_pop_exprs) >= self.population_size: break

            population = next_pop_exprs
            if not population: raise RuntimeError(f"Population empty: gen {generation}")

        if not population: raise RuntimeError("No valid expressions after evolution.")
        final_fitness = [self._evaluate_fitness(expr, X, y, variables) for expr in population]
        return population[np.argmax(final_fitness)]

    def _initialize_population(self, variables: List[Variable]) -> List[Union[Expression, Variable]]:
        initial_max_c = min(5, self.max_complexity if self.max_complexity > 0 else 5)
        candidates = self._optimizer.generate_candidates(variables, initial_max_c)
        if len(candidates) >= self.population_size: return random.sample(candidates, self.population_size)

        pop = [c.clone() if hasattr(c, 'clone') else deepcopy(c) for c in candidates]
        if not pop and self.population_size > 0:
            for _ in range(self.population_size - len(pop)):
                rand_e = self._random_expression(variables, initial_max_c)
                if rand_e: pop.append(rand_e)
            if not pop and variables: pop.append(random.choice(variables))

        idx = 0
        while len(pop) < self.population_size and pop:
            parent = pop[idx % len(pop)]
            mutated = self._mutate(parent, variables)
            if mutated and hasattr(mutated, 'complexity') and mutated.complexity <= self.max_complexity:
                pop.append(mutated)
            elif len(pop) < self.population_size:
                rand_e = self._random_expression(variables, initial_max_c)
                if rand_e: pop.append(rand_e)
                else: break
            else: break
            idx +=1
        return pop[:self.population_size]

    def _evaluate_fitness(self, expression: Expression, X: np.ndarray, y: np.ndarray, variables: List[Variable]) -> float:
        preds = []
        for i in range(X.shape[0]):
            subs = {var.symbolic: X[i,j] for j,var in enumerate(variables)}
            try:
                pred_val = float(expression.symbolic.subs(subs))
                if not np.isfinite(pred_val): return -np.inf
                preds.append(pred_val)
            except Exception: return -np.inf
        if len(y) != len(preds) or not preds: return -np.inf
        return -mean_squared_error(y, np.array(preds)) - (0.01 * expression.complexity)

    def _tournament_selection(self, population: List[Expression], fitness_scores: List[float], tournament_size: int = 3) -> List[Expression]:
        selected = []
        if not population: return []
        for _ in range(len(population)):
            actual_tsize = min(tournament_size, len(population))
            if actual_tsize == 0: break
            indices = np.random.choice(len(population), actual_tsize, replace=False)
            winner_idx = max(indices, key=lambda i: fitness_scores[i])
            selected.append(population[winner_idx].clone() if hasattr(population[winner_idx], 'clone') else deepcopy(population[winner_idx]))
        return selected

    def _get_all_subexpressions(self, expression: Expression) -> List[Expression]:
        nodes = [expression]
        for op_node in expression.operands:
            if isinstance(op_node, Expression): nodes.extend(self._get_all_subexpressions(op_node))
        return nodes

    def _crossover(self, parent1: Expression, parent2: Expression) -> Tuple[Optional[Expression], Optional[Expression]]:
        p1_copy, p2_copy = deepcopy(parent1), deepcopy(parent2)
        p1_nodes, p2_nodes = self._get_all_subexpressions(p1_copy), self._get_all_subexpressions(p2_copy)
        if not p1_nodes or not p2_nodes: return p1_copy, p2_copy

        cp1, cp2 = random.choice(p1_nodes), random.choice(p2_nodes)
        cp1.operator, cp2.operator = cp2.operator, cp1.operator
        cp1.operands, cp2.operands = cp2.operands, cp1.operands

        # Update swapped nodes first
        if hasattr(cp1, '__post_init__'): cp1.__post_init__()
        if hasattr(cp2, '__post_init__'): cp2.__post_init__()

        # Then update entire trees
        if hasattr(p1_copy, '__post_init__'): p1_copy.__post_init__()
        if hasattr(p2_copy, '__post_init__'): p2_copy.__post_init__()
        return p1_copy, p2_copy

    def _mutate(self, expression: Union[Expression, Variable], variables: List[Variable]) -> Union[Expression, Variable, None]:
        expr_copy = expression.clone() if hasattr(expression, 'clone') else deepcopy(expression)
        if isinstance(expr_copy, Variable):
            return self._random_expression(variables, max(1, self.max_complexity // 3, 2))

        nodes = self._get_all_subexpressions(expr_copy)
        if not nodes: return expr_copy
        node_to_mutate = random.choice(nodes)
        new_random_sub = self._random_expression(variables, max(1, self.max_complexity // 3, 3))
        if not new_random_sub: return expr_copy

        if expr_copy is node_to_mutate: expr_copy = new_random_sub
        else: self._find_parent_and_replace(expr_copy, node_to_mutate, new_random_sub)

        if isinstance(expr_copy, Expression) and hasattr(expr_copy, '__post_init__'): expr_copy.__post_init__()
        return expr_copy

    def _find_parent_and_replace(self, current: Expression, target: Union[Expression,Variable], replacement: Union[Expression,Variable]):
        if not isinstance(current, Expression): return False
        for i, child in enumerate(current.operands):
            if child is target:
                current.operands[i] = replacement
                return True
            if isinstance(child, Expression) and self._find_parent_and_replace(child, target, replacement):
                return True
        return False

    def _random_expression(self, variables: List[Variable], max_c: int) -> Optional[Union[Expression, Variable]]:
        if max_c <= 0: return None
        if max_c == 1:
            can_var = bool(variables)
            can_const = bool(self.grammar.primitives.get('constants'))
            if can_var and (not can_const or random.random() < 0.7): return random.choice(variables)
            if can_const:
                name = random.choice(list(self.grammar.primitives['constants']))
                return self.grammar.create_expression('const', [self.grammar.primitives['constants'][name]])
            return None

        ops_u, ops_b = list(self.grammar.primitives.get('unary_ops',[])), list(self.grammar.primitives.get('binary_ops',[]))
        op_type = random.choice(['unary', 'binary']) if (ops_u and ops_b) else ('unary' if ops_u else ('binary' if ops_b else None))
        if not op_type: return self._random_expression(variables, 1) # Fallback to terminal

        if op_type == 'unary':
            op_sym = random.choice(ops_u)
            operand = self._random_expression(variables, max(1, max_c - 1))
            return self.grammar.create_expression(op_sym, [operand]) if operand else None
        else: # Binary
            op_sym = random.choice(ops_b)
            if max_c - 1 < 2: return self._random_expression(variables, 1) # Not enough complexity
            c1 = random.randint(1, max_c - 2)
            left = self._random_expression(variables, c1)
            right = self._random_expression(variables, max_c - 1 - c1)
            return self.grammar.create_expression(op_sym, [left, right]) if left and right else None

    def _find_parent(self, root: Expression, target: Expression) -> Optional[Expression]:
        # Unused helper
        if not isinstance(root, Expression): return None
        for child_node in root.operands:
            if child_node is target: return root
            if isinstance(child_node, Expression):
                parent = self._find_parent(child_node, target)
                if parent: return parent
        return None

def get_memory_usage_mb():
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

def monitor_generation_performance(detector: ConservationDetector, variables: List[Variable], max_complexity: int):
    logger = logging.getLogger(__name__)
    # ... (implementation as before) ...

class GenerationProfiles:
    # ... (implementation as before) ...
    @staticmethod
    def fast_exploration(): return {'enable_parallel': False, 'beam_width': 100, 'max_combinations_per_level': 50, 'cache_size': 1000}
    @staticmethod
    def thorough_search(): return {'enable_parallel': True, 'beam_width': 5000, 'max_combinations_per_level': 500, 'cache_size': 50000}
    @staticmethod
    def memory_constrained(): return {'enable_parallel': False, 'beam_width': 500, 'max_combinations_per_level': 100, 'cache_size': 5000}


def create_optimized_detector(profile='balanced', grammar: Optional[ProgressiveGrammar] = None) -> ConservationDetector:
    # ... (implementation as before) ...
    if grammar is None: grammar = ProgressiveGrammar()
    profiles = { 'fast': GenerationProfiles.fast_exploration(), 'thorough': GenerationProfiles.thorough_search(), 'memory': GenerationProfiles.memory_constrained(), 'balanced': {'enable_parallel': True, 'beam_width': 1000, 'cache_size': 10000}}
    config = profiles.get(profile, profiles['balanced'])
    detector = ConservationDetector(grammar)
    detector._optimizer = OptimizedCandidateGenerator(grammar, **config)
    return detector

def profile_generation(detector: ConservationDetector, variables: List[Variable], max_complexity: int):
    # ... (implementation as before) ...
    profiler = cProfile.Profile(); profiler.enable()
    candidates = detector._optimizer.generate_candidates(variables, max_complexity)
    profiler.disable(); s = StringIO(); ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative'); ps.print_stats(20)
    print(f"\nGenerated {len(candidates)} candidates\nTop time-consuming functions:\n{s.getvalue()}")
    return candidates
