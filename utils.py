import sympy as sp
from typing import Dict, Optional

def _count_operations(expr: sp.Expr) -> Dict[str, int]:
    """Count operations in a SymPy expression."""
    ops = {}
    for arg in sp.preorder_traversal(expr):
        if arg.is_Function or arg.is_Add or arg.is_Mul or arg.is_Pow:
            op_name = type(arg).__name__
            ops[op_name] = ops.get(op_name, 0) + 1
    return ops

def calculate_symbolic_accuracy(
    discovered: Optional[str],
    ground_truth: Dict[str, sp.Expr]
) -> float:
    """Calculate similarity between discovered and ground truth expressions."""
    if not discovered:
        return 0.0

    try:
        discovered_expr = sp.sympify(discovered)

        # Check against each ground truth law
        max_similarity = 0.0
        for law_name, truth_expr in ground_truth.items():
            # Simplify difference
            diff = sp.simplify(discovered_expr - truth_expr)

            # If difference is zero, perfect match
            if diff.equals(0):
                return 1.0

            # Otherwise, calculate structural similarity
            # (simplified metric - could be more sophisticated)
            discovered_ops = _count_operations(discovered_expr)
            truth_ops = _count_operations(truth_expr)

            common_ops = sum(min(discovered_ops.get(op, 0), truth_ops.get(op, 0))
                           for op in set(discovered_ops) | set(truth_ops))
            total_ops = sum(discovered_ops.values()) + sum(truth_ops.values())

            similarity = 2 * common_ops / total_ops if total_ops > 0 else 0
            max_similarity = max(max_similarity, similarity)

        return max_similarity

    except: # Handles sympify errors for invalid discovered strings
        return 0.0
