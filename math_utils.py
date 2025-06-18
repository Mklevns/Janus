import sympy as sp
from typing import Dict, Optional

def _count_operations(expr: sp.Expr) -> Dict[str, int]:
    """Counts the occurrences of mathematical operations in a SymPy expression.

    Args:
        expr: The SymPy expression to analyze.

    Returns:
        A dictionary where keys are operation names (str) and values are
        their counts (int).
    """
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
    """Calculates the symbolic accuracy between a discovered expression string and ground truth.

    The accuracy is determined by comparing the discovered expression with a
    set of ground truth expressions. If the discovered expression exactly matches
    any ground truth expression (after simplification), accuracy is 1.0.
    Otherwise, it computes a structural similarity score based on common
    mathematical operations.

    Args:
        discovered: The string representation of the discovered symbolic expression.
                    If None or an empty string, accuracy is 0.0.
        ground_truth: A dictionary where keys are names of ground truth laws
                      (str) and values are the corresponding SymPy expressions
                      (sp.Expr).

    Returns:
        A float between 0.0 and 1.0 representing the symbolic accuracy.
        Returns 0.0 if the discovered expression is invalid or cannot be parsed.
    """
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
