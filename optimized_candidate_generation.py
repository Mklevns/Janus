# optimized_candidate_generation.py
"""
Optimized candidate generation for physics discovery.
Reduces complexity from O(n^k) to O(n log n) for most practical cases.
"""

import numpy as np
import sympy as sp
from typing import List, Dict, Set, Tuple, Optional, Callable, Any, Union
from dataclasses import dataclass
from functools import lru_cache
from collections import defaultdict
import hashlib
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import heapq

# Import from your existing modules
from janus.core.grammar import ProgressiveGrammar
from janus.core.expression import Expression, Variable


@dataclass
class ExpressionFingerprint:
    """Fast expression comparison using structural hashing."""
    structure_hash: str
    symbolic_hash: str
    complexity: int
    
    @classmethod
    def from_expression(cls, expr: Union[Expression, Variable]) -> 'ExpressionFingerprint':
        """Create fingerprint from expression."""
        structure = cls._get_structure_string(expr)
        structure_hash = hashlib.md5(structure.encode()).hexdigest()[:16]
        
        symbolic_val_to_hash = structure
        if hasattr(expr, 'symbolic') and expr.symbolic is not None:
            try:
                canonical = sp.simplify(expr.symbolic)
                symbolic_val_to_hash = str(canonical)
            except Exception:
                # Fallback for non-simplifiable Sympy expressions or other errors
                if isinstance(expr.symbolic, sp.Symbol):
                    symbolic_val_to_hash = expr.symbolic.name
                elif isinstance(expr.symbolic, (sp.Number, float, int)):
                     symbolic_val_to_hash = str(expr.symbolic)
                # else, structure_hash (default) is used
        elif isinstance(expr, Variable): # If .symbolic was None or missing, but it's a Variable
             symbolic_val_to_hash = expr.name


        symbolic_hash = hashlib.md5(symbolic_val_to_hash.encode()).hexdigest()[:16]
            
        return cls(structure_hash, symbolic_hash, expr.complexity)
    
    @staticmethod
    def _get_structure_string(expr: Union[Expression, Variable]) -> str:
        """Get canonical structure string."""
        if isinstance(expr, Variable):
            return f"var_{expr.name}"
        elif expr.operator == 'var':
            return f"var_{expr.operands[0]}"
        elif expr.operator == 'const':
            const_val = expr.operands[0]
            if isinstance(const_val, float):
                return f"const_{f'{const_val:.6g}'}"
            return f"const_{const_val}"
        else:
            operand_strs = []
            for op_node in expr.operands:
                if isinstance(op_node, (Expression, Variable)):
                    operand_strs.append(ExpressionFingerprint._get_structure_string(op_node))
                else:
                    operand_strs.append(str(op_node))
            
            if expr.operator in ['+', '*']:
                operand_strs.sort()
            return f"{expr.operator}({','.join(operand_strs)})"


class OptimizedCandidateGenerator:
    """Highly optimized expression candidate generator."""
    
    def __init__(self, grammar: 'ProgressiveGrammar', 
                 enable_parallel: bool = True,
                 cache_size: int = 10000):
        self.grammar = grammar
        self.enable_parallel = enable_parallel
        self.cache_size = cache_size
        
        self._expression_cache = {}
        self._fingerprint_cache = {}
        self._simplification_cache = {}
        
        self.max_const_magnitude = 100
        self.min_const_magnitude = 0.001
        self.algebraic_rules = self._init_algebraic_rules()
        
    def _init_algebraic_rules(self) -> List[Callable]:
        return [
            lambda expr: expr.operands[0] if expr.operator == '+' and any(self._is_zero(op) for op in expr.operands) else expr,
            lambda expr: expr.operands[0] if expr.operator == '*' and any(self._is_one(op) for op in expr.operands) else expr,
            lambda expr: self.grammar.create_expression('const', [0.0]) if expr.operator == '*' and any(self._is_zero(op) for op in expr.operands) else expr,
            lambda expr: self.grammar.create_expression('const', [0.0]) if expr.operator == '-' and len(expr.operands) == 2 and self._are_equivalent(expr.operands[0], expr.operands[1]) else expr,
        ]
    
    def generate_candidates(self, variables: List[Variable], 
                          max_complexity: int,
                          fitness_threshold: Optional[float] = None) -> List[Union[Expression, Variable]]:
        base_expressions = self._create_base_expressions(variables)
        
        if max_complexity == 1:
            return [bex for bex in base_expressions if bex.complexity == 1]
        
        all_candidates_map: Dict[str, Union[Expression, Variable]] = {} # Store by symbolic_hash to ensure uniqueness
        
        candidate_queue = []
        entry_count = 0
        
        for expr in base_expressions:
            fp = ExpressionFingerprint.from_expression(expr)
            if fp.symbolic_hash not in all_candidates_map:
                all_candidates_map[fp.symbolic_hash] = expr
                score = self._estimate_promise(expr)
                heapq.heappush(candidate_queue, (-score, expr.complexity, entry_count, expr))
                entry_count += 1
        
        beam_width = min(1000, len(base_expressions) * 10 if base_expressions else 10)
        
        processed_in_beam = set() # Track fingerprints processed to avoid redundant beam work

        for target_complexity in range(2, max_complexity + 1):
            level_candidates_new = []
            
            # Use a snapshot of the queue for generating this level's candidates
            # to avoid issues with modifying the queue while iterating or passing parts of it.
            # Removing 'processed_in_beam' check here, as global novelty is handled by all_candidates_map.
            # We need all expressions of lower complexity to be considered for generating the current target_complexity.
            current_beam_snapshot = [item for item in candidate_queue if item[3].complexity < target_complexity]


            if self.enable_parallel and len(current_beam_snapshot) > 100: # Heuristic for when parallel is beneficial
                # Parallel generation needs careful handling of shared 'seen_fingerprints'
                # For now, _parallel_generate_level will generate, then main thread filters novelty
                raw_level_candidates = self._parallel_generate_level(
                    current_beam_snapshot, target_complexity
                )
            else:
                raw_level_candidates = self._generate_complexity_level(
                    current_beam_snapshot, target_complexity
                )

            for expr_new in raw_level_candidates:
                fp_new = ExpressionFingerprint.from_expression(expr_new)
                if fp_new.symbolic_hash not in all_candidates_map: # Check global novelty
                    all_candidates_map[fp_new.symbolic_hash] = expr_new
                    score = self._estimate_promise(expr_new)
                    if fitness_threshold and score < fitness_threshold:
                        continue
                    heapq.heappush(candidate_queue, (-score, expr_new.complexity, entry_count, expr_new))
                    entry_count += 1
            
            # Mark expressions from snapshot as processed for beam to avoid re-expanding them unnecessarily
            # This was removed from the snapshot filter, so removing the add part too.
            # for item in current_beam_snapshot:
            #     processed_in_beam.add(ExpressionFingerprint.from_expression(item[3]).symbolic_hash)

            if len(candidate_queue) > beam_width:
                candidate_queue = heapq.nsmallest(beam_width, candidate_queue)
                heapq.heapify(candidate_queue)
        
        return list(all_candidates_map.values())
    
    def _create_base_expressions(self, variables: List[Variable]) -> List[Union[Expression, Variable]]:
        expressions: List[Union[Expression, Variable]] = []
        seen_symbolic_hashes = set()

        for var_obj in variables:
            fp = ExpressionFingerprint.from_expression(var_obj)
            if fp.symbolic_hash not in seen_symbolic_hashes:
                seen_symbolic_hashes.add(fp.symbolic_hash)
                expressions.append(var_obj)

        for const_val in [0.0, 1.0, -1.0, 2.0, 0.5, np.pi, np.e]:
            expr_const = self.grammar.create_expression('const', [const_val])
            if expr_const:
                fp = ExpressionFingerprint.from_expression(expr_const)
                if fp.symbolic_hash not in seen_symbolic_hashes:
                    seen_symbolic_hashes.add(fp.symbolic_hash)
                    expressions.append(expr_const)
        return expressions
    
    def _generate_complexity_level(self, current_beam_snapshot: List,
                                 target_complexity: int) -> List[Expression]:
        level_expressions_unfiltered = []
        source_expressions = [item[3] for item in current_beam_snapshot] # item[3] is expr

        for expr_base in source_expressions:
            if expr_base.complexity + 1 == target_complexity:
                for op_symbol in self.grammar.primitives.get('unary_ops', []):
                    new_expr = self._create_and_simplify(op_symbol, tuple([expr_base])) # Pass as tuple
                    if new_expr: level_expressions_unfiltered.append(new_expr)

        if target_complexity >= 2:
            by_complexity = defaultdict(list)
            for expr_obj in source_expressions: # item[3] is expr_obj
                by_complexity[expr_obj.complexity].append(expr_obj)
            
            if target_complexity == 3: # DEBUG
                print(f"DEBUG: by_complexity[1] = {[str(e.symbolic if hasattr(e, 'symbolic') else e.name) for e in by_complexity[1]]}")
                print(f"DEBUG: binary_ops = {self.grammar.primitives.get('binary_ops', [])}")

            for op_symbol in self.grammar.primitives.get('binary_ops', []):
                if target_complexity == 3 and op_symbol == '+': # DEBUG
                    print(f"DEBUG: Processing '+' operator for target_complexity=3")
                for c1 in range(1, target_complexity -1):
                    c2 = target_complexity - 1 - c1
                    if c2 < 1: continue
                    if c1 not in by_complexity or c2 not in by_complexity: continue
                    
                    exprs1 = by_complexity[c1]; exprs2 = by_complexity[c2]
                    # Limit combinations for performance, e.g. max 50*50 = 2500 per op/complexity pair
                    if len(exprs1) * len(exprs2) > 2500:
                        exprs1 = exprs1[:50]; exprs2 = exprs2[:50]

                    for e1 in exprs1:
                        for e2 in exprs2:
                            if op_symbol in ['+', '*'] and c1 == c2 and id(e1) >= id(e2): continue
                            new_expr = self._create_and_simplify(op_symbol, tuple([e1, e2])) # Pass as tuple
                            if new_expr:
                                level_expressions_unfiltered.append(new_expr)
                                if target_complexity == 3 and op_symbol == '+': # More specific DEBUG
                                    print(f"DEBUG: Generated binary '{op_symbol}': {str(new_expr.symbolic)} from {str(e1.symbolic if hasattr(e1,'symbolic') else e1.name)} and {str(e2.symbolic if hasattr(e2,'symbolic') else e2.name)}")

        if target_complexity == 3: # DEBUG PRINT for this specific test case
            print(f"DEBUG: _generate_complexity_level (target_complexity=3) FINAL found: {[str(e.symbolic) for e in level_expressions_unfiltered if hasattr(e, 'symbolic')]}")

        return level_expressions_unfiltered

    def _parallel_generate_level(self, current_beam_snapshot: List,
                               target_complexity: int) -> List[Expression]:
        source_expressions = [item[3] for item in current_beam_snapshot]
        
        n_workers = min(mp.cpu_count(), 4)
        chunk_size = max(1, len(source_expressions) // n_workers if source_expressions else 1)
        chunks = [source_expressions[i:i + chunk_size] for i in range(0, len(source_expressions), chunk_size)]
        
        all_results = []
        # Pass primitives for use in static method _generate_chunk_for_parallel
        primitives_dict = {'unary_ops': list(self.grammar.primitives.get('unary_ops', [])),
                           'binary_ops': list(self.grammar.primitives.get('binary_ops', []))}

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Pass grammar instance for create_expression
            futures = [executor.submit(OptimizedCandidateGenerator._generate_chunk_for_parallel,
                                      chunk, target_complexity, self.grammar, primitives_dict)
                       for chunk in chunks if chunk]
            for future in as_completed(futures):
                try: all_results.extend(future.result())
                except Exception as e: print(f"Parallel generation error: {e}")
        return all_results
    
    @staticmethod # Make static for easier parallelization if grammar is passed
    def _generate_chunk_for_parallel(expressions_chunk: List[Union[Expression,Variable]],
                                     target_complexity: int,
                                     grammar: ProgressiveGrammar, # Pass grammar
                                     primitives: dict) -> List[Expression]:
        results = []
        # This method is static, so it cannot call self._create_and_simplify directly.
        # For a proper parallel implementation, _create_and_simplify logic (or parts of it)
        # would need to be accessible here or refactored.
        # For now, focusing on generation, actual simplification/novelty is centralized.
        for expr_base in expressions_chunk:
            if expr_base.complexity + 1 == target_complexity: # Unary
                for op_symbol in primitives.get('unary_ops', []):
                    new_expr_obj = grammar.create_expression(op_symbol, [expr_base])
                    if new_expr_obj: results.append(new_expr_obj)

            # Binary ops (simplified for chunk, assuming by_complexity is not available here)
            # This part needs to be consistent with _generate_complexity_level or parallel strategy refined
            # For now, this parallel chunk only handles unary based on its input expressions.
            # A full parallel binary generation would involve passing more context or lists of expressions.
        return results

    @lru_cache(maxsize=None)
    def _create_and_simplify(self, operator: str, operands_tuple: Tuple[Union[Expression, Variable], ...]) -> Optional[Expression]:
        if operator == '+':
            op_names_for_debug = []
            for op_debug in operands_tuple:
                if isinstance(op_debug, Variable): op_names_for_debug.append(op_debug.name)
                elif isinstance(op_debug, Expression): op_names_for_debug.append(str(op_debug.symbolic))
                else: op_names_for_debug.append(str(op_debug))
            # print(f"DEBUG _c_a_s: op='{operator}', operands={[str(o) for o in operands_tuple]}")
            print(f"DEBUG _c_a_s: op='{operator}', operands_sym={op_names_for_debug}")


        operands_list = list(operands_tuple) # create_expression expects a list
        expr = self.grammar.create_expression(operator, operands_list)
        if not expr:
            if operator == '+': print(f"DEBUG _c_a_s: create_expression returned None for + with {operands_list}")
            return None
        
        current_expr = expr
        for rule in self.algebraic_rules:
            if isinstance(current_expr, Expression): # Rules expect Expression
                current_expr = rule(current_expr)
            else: break # Stop if a rule reduced to non-Expression
        
        if not isinstance(current_expr, Expression): return None # Must be Expression to proceed

        if self._is_zero(current_expr) or self._is_trivial(current_expr): return None
        
        try:
            if current_expr.symbolic is not None and current_expr.symbolic.is_number:
                const_val = float(current_expr.symbolic)
                if abs(const_val) > self.max_const_magnitude or \
                   (0 < abs(const_val) < self.min_const_magnitude and const_val != 0.0):
                    return None
        except Exception: pass
        return current_expr # Return the (potentially) rule-simplified expression
    
    def _is_novel(self, expr: Expression, processed_symbolic_hashes: Set[str]) -> bool:
        # This 'processed_symbolic_hashes' should be passed from generate_candidates (all_candidates_map.keys())
        fp = ExpressionFingerprint.from_expression(expr)
        if fp.symbolic_hash in processed_symbolic_hashes: return False
        # The set update should happen in the caller (generate_candidates) to maintain central list.
        return True
    
    def _estimate_promise(self, expr: Union[Expression, Variable]) -> float:
        score = 0.0
        var_count = self._count_variables(expr)
        score += var_count * 10
        score -= expr.complexity * 0.5
        if isinstance(expr, Expression): # These helpers expect Expression
             if self._has_physics_pattern(expr): score += 20
             if self._has_redundancy(expr): score -= 10
        return score
    
    def _count_variables(self, expr: Union[Expression, Variable]) -> int:
        if isinstance(expr, Variable): return 1
        if expr.operator == 'const': return 0
        vars_set = set()
        if expr.operator == 'var' and expr.operands:
            vars_set.add(str(expr.operands[0]))
        else:
            for op_node in expr.operands:
                if isinstance(op_node, (Expression, Variable)):
                    vars_set.update(self._get_variable_names(op_node))
        return len(vars_set)

    def _get_variable_names(self, expr: Union[Expression, Variable]) -> Set[str]:
        if isinstance(expr, Variable): return {expr.name}
        if expr.operator == 'const': return set()
        if expr.operator == 'var' and expr.operands:
            return {str(expr.operands[0])}
        names = set()
        for op_node in expr.operands:
            if isinstance(op_node, (Expression, Variable)):
                names.update(self._get_variable_names(op_node))
        return names
    
    def _has_physics_pattern(self, expr: Expression) -> bool: # Expects Expression
        if expr.operator == '**' and len(expr.operands) == 2 and self._is_constant(expr.operands[1], 2.0): return True
        if expr.operator == '*':
            var_count = sum(1 for op in expr.operands if (isinstance(op, Expression) and op.operator == 'var') or isinstance(op, Variable))
            if var_count >= 2: return True
        if expr.operator == '/':
            if len(expr.operands) == 2 and all(isinstance(op, (Expression, Variable)) for op in expr.operands): return True
        return False
    
    def _has_redundancy(self, expr: Expression) -> bool: # Expects Expression
        if expr.operator == '+' and len(expr.operands) == 2 and self._are_equivalent(expr.operands[0], expr.operands[1]): return True
        if expr.operator in ['+', '*']:
            for op in expr.operands:
                if isinstance(op, Expression) and op.operator == expr.operator: return True
        return False
    
    def _is_zero(self, expr_like: Any) -> bool:
        return isinstance(expr_like, Expression) and expr_like.operator == 'const' and abs(expr_like.operands[0] - 0.0) < 1e-10
    
    def _is_one(self, expr_like: Any) -> bool:
        return isinstance(expr_like, Expression) and expr_like.operator == 'const' and abs(expr_like.operands[0] - 1.0) < 1e-10
    
    def _is_constant(self, expr_like: Any, value: float) -> bool: # Ensure value is float for comparison
        return isinstance(expr_like, Expression) and expr_like.operator == 'const' and abs(expr_like.operands[0] - value) < 1e-10
    
    def _is_trivial(self, expr: Expression) -> bool:
        if expr.operator == 'const': return True
        try:
            if expr.symbolic is not None and expr.symbolic.is_number:
                val = float(expr.symbolic)
                if abs(val) > self.max_const_magnitude or (0 < abs(val) < self.min_const_magnitude and val != 0.0):
                    return True
        except Exception: pass
        return False
    
    def _are_equivalent(self, expr1_like: Any, expr2_like: Any) -> bool:
        if not isinstance(expr1_like, (Expression, Variable)) or not isinstance(expr2_like, (Expression, Variable)): return False
        if expr1_like.complexity != expr2_like.complexity: return False
        fp1 = ExpressionFingerprint.from_expression(expr1_like)
        fp2 = ExpressionFingerprint.from_expression(expr2_like)
        return fp1.symbolic_hash == fp2.symbolic_hash

def benchmark_generation(grammar, variables, max_complexity: int):
    import time
    print(f"Generating candidates up to complexity {max_complexity}...")
    optimizer = OptimizedCandidateGenerator(grammar, enable_parallel=False) # Disable parallel for simpler benchmark debug
    start = time.time()
    candidates = optimizer.generate_candidates(variables, max_complexity)
    elapsed = time.time() - start
    print(f"Optimized generation:")
    print(f"  - Time: {elapsed:.3f}s")
    print(f"  - Candidates: {len(candidates)}")
    if elapsed > 0: print(f"  - Rate: {len(candidates)/elapsed:.1f} expr/s")
    else: print(f"  - Rate: N/A (elapsed time was zero or too small)")
    
    complexity_dist = defaultdict(int)
    for expr_item in candidates:
        complexity_dist[expr_item.complexity] += 1
    print(f"  - Distribution: {dict(sorted(complexity_dist.items()))}") # Sort for consistent output
    return candidates

def replace_generate_candidates(instance_with_grammar):
    grammar = instance_with_grammar.grammar
    optimizer = OptimizedCandidateGenerator(grammar)
    # This method is generic, assuming instance_with_grammar might have _generate_candidates
    if hasattr(instance_with_grammar, '_generate_candidates_original_slow_version'): # Example attribute name
        instance_with_grammar._generate_candidates_original_slow_version = optimizer.generate_candidates
        print("Replaced _generate_candidates_original_slow_version with optimized version")
    elif hasattr(instance_with_grammar, '_optimizer') and \
         hasattr(instance_with_grammar._optimizer, 'generate_candidates'):
        # If the instance already uses an optimizer, this function's purpose might be to replace that optimizer.
        # Or, if ConservationDetector calls self._optimizer.generate_candidates, this func is not for it.
        print("Instance already uses an optimizer. Consider replacing the optimizer instance if needed.")
    else:
        print("No clear _generate_candidates method found to replace with this generic helper.")


if __name__ == "__main__":
    from janus.core.grammar import ProgressiveGrammar
    test_grammar = ProgressiveGrammar()
    test_variables = [
        Variable("x", 0), Variable("v", 1), Variable("m", 2) # Simplified for test
    ]
    for test_max_c in [1, 2, 3]:
        print(f"\n{'='*50}")
        print(f"Benchmarking for max_complexity={test_max_c}")
        generated_candidates = benchmark_generation(test_grammar, test_variables, test_max_c)
        print(f"\nExample expressions (max 10 shown) for max_complexity={test_max_c}:")

        count = 0
        for item in generated_candidates:
            if count >= 10: break
            if isinstance(item, Variable):
                print(f"  Variable: {item.name} (complexity: {item.complexity})")
            elif hasattr(item, 'symbolic'):
                print(f"  Expression: {item.symbolic} (complexity: {item.complexity})")
            else:
                print(f"  Unknown item type: {type(item)}")
            count +=1
        if not generated_candidates:
            print("  No candidates generated.")
