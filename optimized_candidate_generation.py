# optimized_candidate_generation.py
"""
Optimized candidate generation for physics discovery.
Reduces complexity from O(n^k) to O(n log n) for most practical cases.
"""

import numpy as np
import sympy as sp
from typing import List, Dict, Set, Tuple, Optional, Callable, Any
from dataclasses import dataclass
from functools import lru_cache
from collections import defaultdict
import hashlib
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import heapq

# Import from your existing modules
from progressive_grammar_system import Expression, Variable


@dataclass
class ExpressionFingerprint:
    """Fast expression comparison using structural hashing."""
    structure_hash: str
    symbolic_hash: str
    complexity: int
    
    @classmethod
    def from_expression(cls, expr: Expression) -> 'ExpressionFingerprint':
        """Create fingerprint from expression."""
        # Structure hash based on tree structure
        structure = cls._get_structure_string(expr)
        structure_hash = hashlib.md5(structure.encode()).hexdigest()[:16]
        
        # Symbolic hash for algebraic equivalence
        try:
            canonical = sp.simplify(expr.symbolic)
            symbolic_hash = hashlib.md5(str(canonical).encode()).hexdigest()[:16]
        except:
            symbolic_hash = structure_hash
            
        return cls(structure_hash, symbolic_hash, expr.complexity)
    
    @staticmethod
    def _get_structure_string(expr: Expression) -> str:
        """Get canonical structure string."""
        if expr.operator == 'var':
            return f"var_{expr.operands[0]}"
        elif expr.operator == 'const':
            return f"const_{hash(expr.operands[0]) % 1000}"
        else:
            operand_strs = []
            for op in expr.operands:
                if isinstance(op, Expression):
                    operand_strs.append(ExpressionFingerprint._get_structure_string(op))
                else:
                    operand_strs.append(str(op))
            
            # Sort for commutative operators
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
        
        # Caches for performance
        self._expression_cache = {}
        self._fingerprint_cache = {}
        self._simplification_cache = {}
        
        # Pruning thresholds
        self.max_const_magnitude = 100
        self.min_const_magnitude = 0.001
        self.algebraic_rules = self._init_algebraic_rules()
        
    def _init_algebraic_rules(self) -> List[Callable]:
        """Initialize algebraic simplification rules."""
        return [
            # x + 0 = x
            lambda expr: expr.operands[0] if (
                expr.operator == '+' and 
                any(self._is_zero(op) for op in expr.operands)
            ) else expr,
            
            # x * 1 = x
            lambda expr: expr.operands[0] if (
                expr.operator == '*' and 
                any(self._is_one(op) for op in expr.operands)
            ) else expr,
            
            # x * 0 = 0
            lambda expr: self.grammar.create_expression('const', [0]) if (
                expr.operator == '*' and 
                any(self._is_zero(op) for op in expr.operands)
            ) else expr,
            
            # x - x = 0
            lambda expr: self.grammar.create_expression('const', [0]) if (
                expr.operator == '-' and 
                len(expr.operands) == 2 and
                self._are_equivalent(expr.operands[0], expr.operands[1])
            ) else expr,
        ]
    
    def generate_candidates(self, variables: List[Variable], 
                          max_complexity: int,
                          fitness_threshold: Optional[float] = None) -> List[Expression]:
        """
        Generate candidate expressions with optimized algorithm.
        
        Time complexity: O(n * k * log(n)) where n is number of base expressions
                        and k is max_complexity
        Space complexity: O(n) with pruning
        """
        # Initialize with base expressions
        base_expressions = self._create_base_expressions(variables)
        
        if max_complexity == 1:
            return base_expressions
        
        # Use iterative deepening with pruning
        all_candidates = []
        seen_fingerprints = set()
        
        # Priority queue for beam search
        # (negative_promise_score, complexity, expression)
        candidate_queue = []
        
        # Add base expressions to queue
        for expr in base_expressions:
            score = self._estimate_promise(expr)
            heapq.heappush(candidate_queue, (-score, expr.complexity, expr))
            all_candidates.append(expr)
        
        # Beam search with pruning
        beam_width = min(1000, len(base_expressions) * 10)
        
        for target_complexity in range(2, max_complexity + 1):
            level_candidates = []
            
            if self.enable_parallel and len(candidate_queue) > 100:
                # Parallel generation for this complexity level
                level_candidates = self._parallel_generate_level(
                    candidate_queue, target_complexity, seen_fingerprints
                )
            else:
                # Serial generation
                level_candidates = self._generate_complexity_level(
                    candidate_queue, target_complexity, seen_fingerprints
                )
            
            # Add promising candidates to queue
            for expr in level_candidates:
                score = self._estimate_promise(expr)
                
                # Prune based on promise score if threshold provided
                if fitness_threshold and score < fitness_threshold:
                    continue
                    
                heapq.heappush(candidate_queue, (-score, expr.complexity, expr))
                all_candidates.append(expr)
            
            # Prune queue to beam width
            if len(candidate_queue) > beam_width:
                candidate_queue = heapq.nsmallest(beam_width, candidate_queue)
                heapq.heapify(candidate_queue)
        
        return all_candidates
    
    def _create_base_expressions(self, variables: List[Variable]) -> List[Expression]:
        """Create base expressions with deduplication."""
        expressions = []
        seen = set()
        
        # Add variables
        for var in variables:
            fp = ExpressionFingerprint.from_expression(var)
            if fp.symbolic_hash not in seen:
                seen.add(fp.symbolic_hash)
                expressions.append(var)
        
        # Add common constants
        for const in [0, 1, -1, 2, 0.5, np.pi, np.e]:
            expr = self.grammar.create_expression('const', [const])
            if expr:
                fp = ExpressionFingerprint.from_expression(expr)
                if fp.symbolic_hash not in seen:
                    seen.add(fp.symbolic_hash)
                    expressions.append(expr)
        
        return expressions
    
    def _generate_complexity_level(self, candidate_queue: List,
                                 target_complexity: int,
                                 seen_fingerprints: Set[str]) -> List[Expression]:
        """Generate expressions of specific complexity."""
        level_expressions = []
        
        # Extract expressions from priority queue
        current_expressions = [item[2] for item in candidate_queue 
                             if item[1] < target_complexity]
        
        # Unary operations
        for expr in current_expressions:
            if expr.complexity + 1 == target_complexity:
                for op in self.grammar.primitives.get('unary_ops', []):
                    new_expr = self._create_and_simplify(op, [expr])
                    if new_expr and self._is_novel(new_expr, seen_fingerprints):
                        level_expressions.append(new_expr)
        
        # Binary operations - optimized pairing
        if target_complexity >= 3:
            # Group expressions by complexity for efficient pairing
            by_complexity = defaultdict(list)
            for expr in current_expressions:
                by_complexity[expr.complexity].append(expr)
            
            for op in self.grammar.primitives.get('binary_ops', []):
                # Generate valid complexity pairs
                for c1 in range(1, target_complexity - 1):
                    c2 = target_complexity - 1 - c1
                    
                    if c1 not in by_complexity or c2 not in by_complexity:
                        continue
                    
                    # Limit combinations to prevent explosion
                    max_combinations = 100
                    exprs1 = by_complexity[c1][:20]
                    exprs2 = by_complexity[c2][:20]
                    
                    combinations = 0
                    for expr1 in exprs1:
                        for expr2 in exprs2:
                            if combinations >= max_combinations:
                                break
                                
                            # Skip symmetric duplicates for commutative ops
                            if op in ['+', '*'] and c1 == c2:
                                if id(expr1) >= id(expr2):
                                    continue
                            
                            new_expr = self._create_and_simplify(op, [expr1, expr2])
                            if new_expr and self._is_novel(new_expr, seen_fingerprints):
                                level_expressions.append(new_expr)
                                combinations += 1
        
        return level_expressions
    
    def _parallel_generate_level(self, candidate_queue: List,
                               target_complexity: int,
                               seen_fingerprints: Set[str]) -> List[Expression]:
        """Generate complexity level in parallel."""
        # Split work into chunks
        current_expressions = [item[2] for item in candidate_queue 
                             if item[1] < target_complexity]
        
        n_workers = min(mp.cpu_count(), 4)
        chunk_size = max(1, len(current_expressions) // n_workers)
        
        chunks = [current_expressions[i:i + chunk_size] 
                 for i in range(0, len(current_expressions), chunk_size)]
        
        all_results = []
        
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = []
            
            for chunk in chunks:
                future = executor.submit(
                    self._generate_chunk,
                    chunk, target_complexity, seen_fingerprints
                )
                futures.append(future)
            
            for future in as_completed(futures):
                try:
                    results = future.result()
                    all_results.extend(results)
                except Exception as e:
                    print(f"Parallel generation error: {e}")
        
        return all_results
    
    def _generate_chunk(self, expressions: List[Expression],
                       target_complexity: int,
                       seen_fingerprints: Set[str]) -> List[Expression]:
        """Generate candidates from a chunk of expressions."""
        # This method is called in parallel workers
        # Recreate necessary context
        results = []
        
        # Process unary operations
        for expr in expressions:
            if expr.complexity + 1 == target_complexity:
                for op in self.grammar.primitives.get('unary_ops', []):
                    new_expr = self._create_and_simplify(op, [expr])
                    if new_expr and self._is_novel(new_expr, seen_fingerprints):
                        results.append(new_expr)
        
        return results
    
    @lru_cache(maxsize=10000)
    def _create_and_simplify(self, operator: str, 
                           operands: Tuple) -> Optional[Expression]:
        """Create and algebraically simplify expression."""
        # Convert tuple back to list for creation
        operands_list = list(operands)
        
        # Create expression
        expr = self.grammar.create_expression(operator, operands_list)
        if not expr:
            return None
        
        # Apply algebraic rules
        for rule in self.algebraic_rules:
            expr = rule(expr)
            if self._is_zero(expr) or self._is_trivial(expr):
                return None
        
        # Simplify symbolically
        try:
            simplified = sp.simplify(expr.symbolic)
            
            # Check if simplification made it trivial
            if simplified.is_number:
                const_val = float(simplified)
                if abs(const_val) > self.max_const_magnitude:
                    return None
                if 0 < abs(const_val) < self.min_const_magnitude:
                    return None
        except:
            pass
        
        return expr
    
    def _is_novel(self, expr: Expression, seen_fingerprints: Set[str]) -> bool:
        """Check if expression is novel."""
        fp = ExpressionFingerprint.from_expression(expr)
        
        if fp.symbolic_hash in seen_fingerprints:
            return False
            
        seen_fingerprints.add(fp.symbolic_hash)
        return True
    
    def _estimate_promise(self, expr: Expression) -> float:
        """Estimate promise score for beam search."""
        score = 0.0
        
        # Prefer expressions with variables
        var_count = self._count_variables(expr)
        score += var_count * 10
        
        # Penalize complexity
        score -= expr.complexity * 0.5
        
        # Bonus for common physics patterns
        if self._has_physics_pattern(expr):
            score += 20
        
        # Penalize redundancy
        if self._has_redundancy(expr):
            score -= 10
        
        return score
    
    def _count_variables(self, expr: Expression) -> int:
        """Count unique variables in expression."""
        if expr.operator == 'var':
            return 1
        elif expr.operator == 'const':
            return 0
        else:
            vars_set = set()
            for op in expr.operands:
                if isinstance(op, Expression):
                    vars_set.update(self._get_variable_names(op))
            return len(vars_set)
    
    def _get_variable_names(self, expr: Expression) -> Set[str]:
        """Get all variable names in expression."""
        if expr.operator == 'var':
            return {str(expr.operands[0])}
        elif expr.operator == 'const':
            return set()
        else:
            names = set()
            for op in expr.operands:
                if isinstance(op, Expression):
                    names.update(self._get_variable_names(op))
            return names
    
    def _has_physics_pattern(self, expr: Expression) -> bool:
        """Check if expression matches common physics patterns."""
        # Check for energy-like terms (quadratic)
        if expr.operator == '**' and len(expr.operands) == 2:
            if self._is_constant(expr.operands[1], 2):
                return True
        
        # Check for product of variables (momentum, force)
        if expr.operator == '*':
            var_count = sum(1 for op in expr.operands 
                          if isinstance(op, Expression) and op.operator == 'var')
            if var_count >= 2:
                return True
        
        # Check for ratios
        if expr.operator == '/':
            if all(isinstance(op, Expression) for op in expr.operands):
                return True
        
        return False
    
    def _has_redundancy(self, expr: Expression) -> bool:
        """Check for redundant patterns."""
        # Check for x + x (should be 2*x)
        if expr.operator == '+' and len(expr.operands) == 2:
            if self._are_equivalent(expr.operands[0], expr.operands[1]):
                return True
        
        # Check for nested same operations
        if expr.operator in ['+', '*']:
            for op in expr.operands:
                if isinstance(op, Expression) and op.operator == expr.operator:
                    return True
        
        return False
    
    def _is_zero(self, expr: Any) -> bool:
        """Check if expression is zero."""
        if isinstance(expr, Expression):
            if expr.operator == 'const' and abs(expr.operands[0]) < 1e-10:
                return True
        return False
    
    def _is_one(self, expr: Any) -> bool:
        """Check if expression is one."""
        if isinstance(expr, Expression):
            if expr.operator == 'const' and abs(expr.operands[0] - 1) < 1e-10:
                return True
        return False
    
    def _is_constant(self, expr: Any, value: float) -> bool:
        """Check if expression equals specific constant."""
        if isinstance(expr, Expression):
            if expr.operator == 'const' and abs(expr.operands[0] - value) < 1e-10:
                return True
        return False
    
    def _is_trivial(self, expr: Expression) -> bool:
        """Check if expression is trivial."""
        # Single constant
        if expr.operator == 'const':
            return True
        
        # Very large or small constants
        try:
            if expr.symbolic.is_number:
                val = float(expr.symbolic)
                if abs(val) > self.max_const_magnitude:
                    return True
                if 0 < abs(val) < self.min_const_magnitude:
                    return True
        except:
            pass
        
        return False
    
    def _are_equivalent(self, expr1: Any, expr2: Any) -> bool:
        """Check if two expressions are equivalent."""
        if not isinstance(expr1, Expression) or not isinstance(expr2, Expression):
            return False
            
        # Quick complexity check
        if expr1.complexity != expr2.complexity:
            return False
        
        # Compare fingerprints
        fp1 = ExpressionFingerprint.from_expression(expr1)
        fp2 = ExpressionFingerprint.from_expression(expr2)
        
        return fp1.symbolic_hash == fp2.symbolic_hash


# Benchmarking utilities
def benchmark_generation(grammar, variables, max_complexity: int):
    """Benchmark the optimized generation."""
    import time
    
    # Original method (if available)
    print(f"Generating candidates up to complexity {max_complexity}...")
    
    # Optimized method
    optimizer = OptimizedCandidateGenerator(grammar, enable_parallel=True)
    
    start = time.time()
    candidates = optimizer.generate_candidates(variables, max_complexity)
    elapsed = time.time() - start
    
    print(f"Optimized generation:")
    print(f"  - Time: {elapsed:.3f}s")
    print(f"  - Candidates: {len(candidates)}")
    print(f"  - Rate: {len(candidates)/elapsed:.1f} expr/s")
    
    # Analyze complexity distribution
    complexity_dist = defaultdict(int)
    for expr in candidates:
        complexity_dist[expr.complexity] += 1
    
    print(f"  - Distribution: {dict(complexity_dist)}")
    
    return candidates


# Integration helper
def replace_generate_candidates(physics_extension_instance):
    """Replace the inefficient method with optimized version."""
    grammar = physics_extension_instance.grammar
    optimizer = OptimizedCandidateGenerator(grammar)
    
    # Monkey patch the method
    physics_extension_instance._generate_candidates = optimizer.generate_candidates
    
    print("Replaced _generate_candidates with optimized version")


# Example usage
if __name__ == "__main__":
    # Example testing the optimization
    from progressive_grammar_system import ProgressiveGrammar, Variable
    
    # Create test setup
    grammar = ProgressiveGrammar()
    variables = [
        Variable("x", 0, {"units": "m"}),
        Variable("v", 1, {"units": "m/s"}),
        Variable("m", 2, {"units": "kg"}),
        Variable("k", 3, {"units": "N/m"})
    ]
    
    # Benchmark different complexity levels
    for max_c in [5, 8, 10]:
        print(f"\n{'='*50}")
        candidates = benchmark_generation(grammar, variables, max_c)
        
        # Show some examples
        print(f"\nExample expressions:")
        for expr in candidates[-5:]:
            print(f"  {expr.symbolic} (complexity: {expr.complexity})")
