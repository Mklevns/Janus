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
import networkx as nx
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
import pickle
from progressive_grammar_system import Expression, Variable, ProgressiveGrammar
import random
import copy

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

    def __init__(self, grammar: 'ProgressiveGrammar', **kwargs): # Added **kwargs
        self.grammar = grammar
        self.population_size = kwargs.get('population_size', 100)
        self.generations = kwargs.get('generations', 50)
        self.mutation_rate = kwargs.get('mutation_rate', 0.1)
        self.crossover_rate = kwargs.get('crossover_rate', 0.7)
        self.max_depth = kwargs.get('max_depth', 10)
        self.max_nodes = kwargs.get('max_nodes', 50)
        self.n_vars = kwargs.get('n_vars', 0)
        self.debug = kwargs.get('debug', False)

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
        self.n_vars = X.shape[1]
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
            # fitness_scores are from the original population before selection.

            next_population = []

            # 'variables' (List[Variable]) and 'max_complexity' (int) are from fit's arguments.

            idx = 0
            while len(next_population) < self.population_size:
                if not selected:
                    if self.debug: print("Warning: 'selected' list is empty. Stopping generation.")
                    break

                parent1 = selected[idx % len(selected)]
                idx += 1

                children_candidates = []
                if np.random.random() < self.crossover_rate and len(selected) >= 2:
                    parent2 = selected[idx % len(selected)]
                    idx += 1

                    child_expr1, child_expr2 = self._crossover(parent1, parent2)
                    children_candidates.extend([child_expr1, child_expr2])
                else:
                    children_candidates.append(self._deep_copy_expression(parent1))

                for child_candidate in children_candidates:
                    if len(next_population) >= self.population_size:
                        break

                    mutated_child = child_candidate

                    if np.random.random() < self.mutation_rate:
                        mutated_child = self._mutate(mutated_child, variables)

                    if hasattr(mutated_child, 'complexity') and mutated_child.complexity <= max_complexity:
                        next_population.append(mutated_child)
                    elif self.debug and not hasattr(mutated_child, 'complexity'):
                        m_str = str(mutated_child.symbolic) if hasattr(mutated_child, 'symbolic') else str(mutated_child)
                        print(f"Warning: Candidate {m_str} lacks complexity, not added.")

            if not next_population and selected:
                if self.debug: print("Warning: Next population was empty. Filling with copies from 'selected' list.")
                # This fallback is simple. A robust GP might use elitism or re-evaluate selected.
                for i in range(min(self.population_size, len(selected))):
                    if len(next_population) < self.population_size:
                        next_population.append(self._deep_copy_expression(selected[i]))
                    else:
                        break

            # If still empty, it's a problem, but the loop structure handles it.
            if not next_population:
                 if self.debug: print("Critical Warning: Population is empty after generation efforts.")
                 # Consider if we should break or return current best if population dies.
                 # For now, allowing population to become empty if all else fails.
                 # The existing code outside this loop will handle an empty population
                 # (e.g. by possibly erroring out or returning a poor 'best').

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

    def _deep_copy_expression(self, expr: 'Expression') -> 'Expression':
        new_expr = copy.deepcopy(expr)
        # Call __post_init__ if it exists and is responsible for resetting/rebuilding cache
        if hasattr(new_expr, '__post_init__'):
             new_expr.__post_init__()
        # Alternatively, if specific cache clearing methods exist:
        # if hasattr(new_expr, '_clear_cache'): new_expr._clear_cache()
        return new_expr

    def _get_crossover_points(self, expr: 'Expression') -> List[Tuple['Expression', Dict]]:
        points = []
        # from progressive_grammar_system import Expression # Ensure type hint is resolvable

        def traverse(node: 'Expression', parent: Optional['Expression'] = None,
                     parent_attr_name: Optional[str] = None, index_in_parent_attr: Optional[int] = None):
            if parent is not None:
                parent_info = {
                    'parent_node': parent,
                    'parent_attr_name': parent_attr_name,
                    'index_in_parent_attr': index_in_parent_attr
                }
                points.append((node, parent_info))

            if hasattr(node, 'operands') and isinstance(node.operands, list):
                for i, operand_child in enumerate(node.operands):
                    if isinstance(operand_child, Expression):
                        traverse(operand_child, parent=node, parent_attr_name='operands', index_in_parent_attr=i)

        if hasattr(expr, 'operands') and isinstance(expr.operands, list):
            for i, operand_child in enumerate(expr.operands):
                if isinstance(operand_child, Expression):
                     traverse(operand_child, parent=expr, parent_attr_name='operands', index_in_parent_attr=i)
        return points

    def _get_subtree_depth(self, node: Optional['Expression']) -> int:
        if not isinstance(node, Expression):
            return 1
        if not hasattr(node, 'operands') or not node.operands:
            return 1

        max_child_depth = 0
        for operand in node.operands:
            max_child_depth = max(max_child_depth, self._get_subtree_depth(operand))
        return 1 + max_child_depth

    def _get_node_depth(self, expr_root: 'Expression', target_node: 'Expression') -> int:
        q = [(expr_root, 1)]
        visited_ids = {id(expr_root)}
        while q:
            curr, depth = q.pop(0)
            if curr is target_node:
                return depth
            if isinstance(curr, Expression) and hasattr(curr, 'operands'):
                for op_node in curr.operands:
                    if isinstance(op_node, Expression) and id(op_node) not in visited_ids:
                        q.append((op_node, depth + 1))
                        visited_ids.add(id(op_node))
        return -1

    def _can_swap(self, node1_subtree_copy: 'Expression', parent_info1: Dict,
                  node2_subtree_copy: 'Expression', parent_info2: Dict,
                  child1_root_template: 'Expression', child2_root_template: 'Expression') -> bool:

        p1_actual_parent = parent_info1['parent_node']
        idx1 = parent_info1['index_in_parent_attr']
        original_subtree_in_p1 = p1_actual_parent.operands[idx1]

        p2_actual_parent = parent_info2['parent_node']
        idx2 = parent_info2['index_in_parent_attr']
        original_subtree_in_p2 = p2_actual_parent.operands[idx2]

        valid_swap = False
        try:
            p1_actual_parent.operands[idx1] = node2_subtree_copy
            if hasattr(p1_actual_parent, '__post_init__'): p1_actual_parent.__post_init__()
            if hasattr(child1_root_template, '__post_init__'): child1_root_template.__post_init__()

            p2_actual_parent.operands[idx2] = node1_subtree_copy
            if hasattr(p2_actual_parent, '__post_init__'): p2_actual_parent.__post_init__()
            if hasattr(child2_root_template, '__post_init__'): child2_root_template.__post_init__()

            depth1_ok = self._get_subtree_depth(child1_root_template) <= self.max_depth
            depth2_ok = self._get_subtree_depth(child2_root_template) <= self.max_depth
            valid_swap = depth1_ok and depth2_ok
        finally:
            p1_actual_parent.operands[idx1] = original_subtree_in_p1
            if hasattr(p1_actual_parent, '__post_init__'): p1_actual_parent.__post_init__()
            if hasattr(child1_root_template, '__post_init__'): child1_root_template.__post_init__()

            p2_actual_parent.operands[idx2] = original_subtree_in_p2
            if hasattr(p2_actual_parent, '__post_init__'): p2_actual_parent.__post_init__()
            if hasattr(child2_root_template, '__post_init__'): child2_root_template.__post_init__()

        return valid_swap

    def _swap_subtrees(self, parent_info1: Dict, node_to_place_in_parent1: 'Expression',
                           parent_info2: Dict, node_to_place_in_parent2: 'Expression'):
        parent1_node = parent_info1['parent_node']
        idx1 = parent_info1['index_in_parent_attr']
        parent1_node.operands[idx1] = node_to_place_in_parent1
        if hasattr(parent1_node, '__post_init__'): parent1_node.__post_init__()

        parent2_node = parent_info2['parent_node']
        idx2 = parent_info2['index_in_parent_attr']
        parent2_node.operands[idx2] = node_to_place_in_parent2
        if hasattr(parent2_node, '__post_init__'): parent2_node.__post_init__()

    def _count_nodes(self, node: Optional['Expression']) -> int:
        if not isinstance(node, Expression): return 0
        count = 1
        if hasattr(node, 'operands') and isinstance(node.operands, list):
            for operand in node.operands:
                count += self._count_nodes(operand)
        return count

    def _check_tree_integrity(self, node: Optional['Expression']) -> bool:
        if not isinstance(node, Expression): return True
        if not hasattr(node, 'operator') or not hasattr(node, 'operands'):
            if self.debug: print(f"Integrity Check Fail: Node missing operator/operands: {type(node)}")
            return False

        op = node.operator
        num_operands = len(node.operands)
        expected_arity = -1

        # Assuming ProgressiveGrammar class has 'primitives' dict and potentially 'is_variable_name' method
        # For Variable instances from progressive_grammar_system, op=name, operands=[]
        if op in self.grammar.primitives.get('unary_ops', {}): expected_arity = 1
        elif op in self.grammar.primitives.get('binary_ops', {}): expected_arity = 2
        elif op == 'const': expected_arity = 1
        elif num_operands == 0 : # Assumed variable node if no operands and op not in other categories
            # A better check for variable:
            # if hasattr(self.grammar, 'is_variable_operator') and self.grammar.is_variable_operator(op):
            #    expected_arity = 0
            # else: # Fallback if grammar doesn't have specific var check
            #    if op not in self.grammar.primitives.get('unary_ops', {}) and \
            #       op not in self.grammar.primitives.get('binary_ops', {}) and \
            #       op != 'const': # If unknown and no operands, assume var
            expected_arity = 0
            # This simplified logic: if it has 0 operands and isn't a known op, it must be a var.
            # This could be problematic if an unknown operator is encountered that *should* have operands.
            # A truly robust check needs full grammar introspection for all operator types.
        else:
            if self.debug: print(f"Integrity Check Fail: Unknown operator '{op}' with {num_operands} operands.")
            return False

        if num_operands != expected_arity:
            if self.debug: print(f"Integrity Check Fail: Arity mismatch for '{op}'. Expected {expected_arity}, got {num_operands}.")
            return False

        if op == 'const' and num_operands == 1 and isinstance(node.operands[0], Expression):
            if self.debug: print(f"Integrity Check Fail: Operand of 'const' is an Expression: {node.operands[0]}.")
            return False

        for operand_child in node.operands:
            if not self._check_tree_integrity(operand_child): return False
        return True

    def _validate_tree(self, expr: 'Expression') -> bool:
        if not isinstance(expr, Expression):
            if self.debug: print(f"Validate tree Fail: Not an Expression instance: {type(expr)}")
            return False
        try:
            if not self._check_tree_integrity(expr):
                # _check_tree_integrity will print its own debug message
                return False
            current_depth = self._get_subtree_depth(expr)
            if current_depth > self.max_depth:
                if self.debug: print(f"Validate tree Fail: Exceeds max_depth ({current_depth} > {self.max_depth}) for {expr.symbolic if hasattr(expr, 'symbolic') else 'N/A'}")
                return False
            current_nodes = self._count_nodes(expr)
            if current_nodes > self.max_nodes:
                if self.debug: print(f"Validate tree Fail: Exceeds max_nodes ({current_nodes} > {self.max_nodes}) for {expr.symbolic if hasattr(expr, 'symbolic') else 'N/A'}")
                return False

            if self.n_vars > 0 and hasattr(expr, 'symbolic') and expr.symbolic is not None:
                sympy_expr_to_eval = expr.symbolic
                if not hasattr(sympy_expr_to_eval, 'free_symbols'):
                    if isinstance(sympy_expr_to_eval, (int, float, complex)): return True
                    if self.debug: print(f"Validate tree Fail: expr.symbolic has no free_symbols and not num: {sympy_expr_to_eval}")
                    return False

                sympy_subs = {}
                for sym_obj in sympy_expr_to_eval.free_symbols:
                    if sym_obj.name in [f"x{i}" for i in range(self.n_vars)]:
                         sympy_subs[sym_obj] = 1.0

                if sympy_subs or not sympy_expr_to_eval.free_symbols:
                    val = sympy_expr_to_eval.subs(sympy_subs)
                    if hasattr(val, 'free_symbols') and val.free_symbols:
                        if self.debug: print(f"Validate tree Fail: Unbound symbols after subs: {val.free_symbols} in {val}")
                        return False
                    val.evalf()
            return True
        except Exception as e:
            if self.debug: print(f"Validate tree Fail: Exception for {expr.symbolic if hasattr(expr, 'symbolic') else 'N/A'}: {e}")
            return False

    def _get_all_subexpressions(self, expression: 'Expression') -> List['Expression']:
        """Recursively collect all subexpressions from an expression tree."""
        nodes = []
        if isinstance(expression, Expression):
            nodes.append(expression)
            for op in expression.operands:
                nodes.extend(self._get_all_subexpressions(op))
        return nodes

    def _crossover(self, parent1: 'Expression', parent2: 'Expression') -> Tuple['Expression', 'Expression']:
        if self.debug:
            p1s = parent1.symbolic if hasattr(parent1, 'symbolic') and parent1.symbolic is not None else str(parent1)
            p2s = parent2.symbolic if hasattr(parent2, 'symbolic') and parent2.symbolic is not None else str(parent2)
            print(f"Crossover attempt: P1='{p1s}' X P2='{p2s}'")

        child1_root = self._deep_copy_expression(parent1)
        child2_root = self._deep_copy_expression(parent2)

        points1_info = self._get_crossover_points(child1_root)
        points2_info = self._get_crossover_points(child2_root)

        if not points1_info or not points2_info:
            if self.debug: print("Crossover Bailout: No valid crossover points found.")
            return self._deep_copy_expression(parent1), self._deep_copy_expression(parent2)

        node1_to_swap_ref, parent_info1 = random.choice(points1_info)
        node2_to_swap_ref, parent_info2 = random.choice(points2_info)

        node1_subtree_copy_for_swap = self._deep_copy_expression(node1_to_swap_ref)
        node2_subtree_copy_for_swap = self._deep_copy_expression(node2_to_swap_ref)

        # _can_swap uses child1_root, child2_root (which are copies of parents)
        # and parent_infoX (which refers to nodes within child1_root, child2_root)
        # It will internally modify them, check, and revert.
        if not self._can_swap(node1_subtree_copy_for_swap, parent_info1,
                              node2_subtree_copy_for_swap, parent_info2,
                              child1_root, child2_root):
            if self.debug: print("Crossover Bailout: Swap deemed invalid by _can_swap (e.g. depth constraints).")
            return self._deep_copy_expression(parent1), self._deep_copy_expression(parent2)

        try:
            # Perform actual swap on child1_root, child2_root using the deep copied subtrees
            self._swap_subtrees(parent_info1, node2_subtree_copy_for_swap,
                                parent_info2, node1_subtree_copy_for_swap)

            if hasattr(child1_root, '__post_init__'): child1_root.__post_init__()
            if hasattr(child2_root, '__post_init__'): child2_root.__post_init__()

            valid_child1 = self._validate_tree(child1_root)
            valid_child2 = self._validate_tree(child2_root)

            if valid_child1 and valid_child2:
                if self.debug:
                    c1s = child1_root.symbolic if hasattr(child1_root, 'symbolic') and child1_root.symbolic is not None else str(child1_root)
                    c2s = child2_root.symbolic if hasattr(child2_root, 'symbolic') and child2_root.symbolic is not None else str(child2_root)
                    print(f"Crossover Success: C1='{c1s}', C2='{c2s}'")
                return child1_root, child2_root
            else:
                if self.debug: print(f"Crossover Bailout: Resulting children failed validation (C1_valid:{valid_child1}, C2_valid:{valid_child2}).")
                return self._deep_copy_expression(parent1), self._deep_copy_expression(parent2)

        except Exception as e:
            if self.debug:
                 p1s_fail = parent1.symbolic if hasattr(parent1, 'symbolic') and parent1.symbolic is not None else str(parent1)
                 p2s_fail = parent2.symbolic if hasattr(parent2, 'symbolic') and parent2.symbolic is not None else str(parent2)
                 print(f"Crossover Exception: {e}. Parents: P1='{p1s_fail}', P2='{p2s_fail}'")
            return self._deep_copy_expression(parent1), self._deep_copy_expression(parent2)

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
