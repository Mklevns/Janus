import sympy as sp
from typing import List, Any, Optional, Dict # Added Dict
from dataclasses import dataclass, field

# Forward declaration for type hinting within Expression and Variable
class Variable: # Note: Python doesn't strictly need forward declarations if classes are defined before use or within same module for type hints used as strings.
    pass

class Expression: # Same as above.
    pass

@dataclass(eq=True, frozen=False)
class Variable:
    """Discovered state variable with semantic properties."""
    name: str
    index: int  # Column in observation matrix
    properties: Dict[str, float] = field(default_factory=dict)
    symbolic: sp.Symbol = field(init=False)

    def __post_init__(self):
        self.symbolic = sp.Symbol(self.name)

    def __hash__(self):
        # Hash based on name and index for uniqueness in sets/dicts for caching
        return hash((self.name, self.index))

    @property
    def complexity(self) -> int:
        return 1

@dataclass
class Expression:
    """Represents a mathematical expression as a tree."""
    operator: str
    operands: List[Any]  # Can be Expression, Variable, or Constant
    complexity: int = field(init=False)
    symbolic: Optional[sp.Expr] = field(init=False)

    def __post_init__(self):
        self.complexity = self._compute_complexity()
        self.symbolic = self._to_sympy()

    def _compute_complexity(self) -> int:
        """MDL-based complexity: count nodes in expression tree."""
        if not self.operands: # Should not happen if operator implies operands (e.g. const(val))
            return 1 # Base case for a node (e.g. a lone Variable if it were an Expression subtype)

        # Complexity of the operator node itself is 1.
        # Then sum complexities of operands.
        # If an operand is an Expression, use its .complexity.
        # If an operand is a Variable, its complexity is 1 (from Variable.complexity).
        # If an operand is a raw constant (int/float), its complexity is 1.
        current_complexity = 1
        for op in self.operands:
            if isinstance(op, Expression):
                current_complexity += op.complexity
            elif isinstance(op, Variable): # Check if Variable is an operand
                current_complexity += op.complexity
            else: # Assumed to be a primitive constant (number, string)
                current_complexity += 1
        return current_complexity

    def _to_sympy(self) -> sp.Expr:
        if self.operator == 'var':
            # Operand should be a variable name string or a Variable object
            op_val = self.operands[0]
            if isinstance(op_val, sp.Symbol): # Already a sympy symbol
                return op_val
            elif isinstance(op_val, Variable): # If Variable object is passed
                return op_val.symbolic
            elif isinstance(op_val, str): # If name string is passed
                return sp.Symbol(op_val)
            else:
                raise ValueError(f"Invalid operand for 'var' operator: {op_val}")
        elif self.operator == 'const':
            return sp.Float(self.operands[0])
        elif self.operator in ['+', '-', '*', '/', '**']:
            args = []
            for op in self.operands:
                if isinstance(op, Expression):
                    args.append(op.symbolic)
                elif isinstance(op, Variable): # Variable objects have .symbolic
                    args.append(op.symbolic)
                else: # Raw constants
                    args.append(sp.Float(op) if isinstance(op, (float, int)) else op) # Ensure numbers are sympy numbers

            if self.operator == '+':
                return sp.Add(*args)
            elif self.operator == '-':
                if len(args) == 1: return -args[0] # Unary minus
                return sp.Add(args[0], sp.Mul(-1, *args[1:])) # a - b - c -> a + (-b) + (-c)
            elif self.operator == '*':
                return sp.Mul(*args)
            elif self.operator == '/':
                if len(args) != 2: raise ValueError("Division operator '/' expects 2 operands.")
                numerator = args[0]
                denominator = args[1]
                if hasattr(denominator, 'is_zero') and denominator.is_zero is True: # Check for sympy zero
                    return sp.nan
                if isinstance(denominator, (int,float,sp.Integer,sp.Float)) and denominator == 0: # Check for Python zero
                    return sp.nan
                return numerator / denominator
            elif self.operator == '**':
                if len(args) != 2: raise ValueError("Power operator '**' expects 2 operands.")
                return args[0] ** args[1]
        elif self.operator in ['neg', 'inv', 'sqrt', 'log', 'exp', 'sin', 'cos']:
            if len(self.operands) != 1: raise ValueError(f"Unary operator '{self.operator}' expects 1 operand.")
            operand_val = self.operands[0]
            if isinstance(operand_val, Expression):
                arg = operand_val.symbolic
            elif isinstance(operand_val, Variable):
                arg = operand_val.symbolic
            else:  # Constant
                arg = sp.Float(operand_val)

            if self.operator == 'neg': return -arg
            elif self.operator == 'inv':
                if hasattr(arg, 'is_zero') and arg.is_zero is True: return sp.nan
                if isinstance(arg, (int,float,sp.Integer,sp.Float)) and arg == 0: return sp.nan
                return 1 / arg
            elif self.operator == 'sqrt': return sp.sqrt(arg)
            elif self.operator == 'log': return sp.log(arg)
            elif self.operator == 'exp': return sp.exp(arg)
            elif self.operator == 'sin': return sp.sin(arg)
            elif self.operator == 'cos': return sp.cos(arg)
        elif self.operator == 'diff':
            if len(self.operands) != 2: raise ValueError("'diff' operator expects 2 operands.")
            expr_op, var_op = self.operands

            if isinstance(expr_op, Expression): sym_expr = expr_op.symbolic
            elif isinstance(expr_op, Variable): sym_expr = expr_op.symbolic
            else: sym_expr = sp.Float(expr_op)

            if not isinstance(var_op, Variable): raise ValueError("Second operand for 'diff' must be a Variable.")
            sym_var = var_op.symbolic
            return sp.diff(sym_expr, sym_var)
        elif self.operator == 'int':
            if len(self.operands) != 2: raise ValueError("'int' operator expects 2 operands.")
            expr_op, var_op = self.operands

            if isinstance(expr_op, Expression): sym_expr = expr_op.symbolic
            elif isinstance(expr_op, Variable): sym_expr = expr_op.symbolic
            else: sym_expr = sp.Float(expr_op)

            if not isinstance(var_op, Variable): raise ValueError("Second operand for 'int' must be a Variable.")
            sym_var = var_op.symbolic
            return sp.integrate(sym_expr, sym_var)

        # Fallback for unknown operators
        # Consider if grammar object should be passed to provide context for custom ops
        return sp.Function(self.operator.capitalize())(*[op.symbolic if hasattr(op, 'symbolic') else sp.Symbol(str(op)) for op in self.operands])


    def clone(self) -> 'Expression':
        cloned_operands = []
        for op in self.operands:
            if isinstance(op, Expression):
                cloned_operands.append(op.clone())
            elif isinstance(op, Variable): # Variables are shared by default
                cloned_operands.append(op)
            else: # Primitives like numbers or strings
                cloned_operands.append(op)
        # Create new Expression. __post_init__ will recalc complexity and symbolic.
        return Expression(operator=self.operator, operands=cloned_operands)

    def __hash__(self):
        # Make operands hashable for the tuple
        # Using str for non-hashable, non-Expression/Variable operands as a simple fallback
        # A more robust key generation for hashing might be needed if operand types are diverse
        operand_keys = []
        for op in self.operands:
            if isinstance(op, (Expression, Variable)):
                operand_keys.append(op) # Relies on their __hash__
            elif isinstance(op, (int, float, str, bool)): # Naturally hashable
                operand_keys.append(op)
            else: # Fallback for other types (e.g. list, dict if they sneak in)
                operand_keys.append(str(op))
        return hash((self.operator, tuple(operand_keys)))

    def __eq__(self, other):
        if not isinstance(other, Expression):
            return False
        # Compare operator and operands for equality.
        # This relies on operands also having a well-defined __eq__.
        # For Variables, dataclass provides __eq__. For primitive types, it's standard.
        return self.operator == other.operator and self.operands == other.operands

# Ensure type hints can resolve Expression and Variable within their own definitions
# This is generally handled by Python 3.7+ with `from __future__ import annotations`
# or by using string literals for type hints, e.g., 'Expression', 'Variable'.
# Given the forward class declarations, this should be okay.
# If issues arise, consider using string literals for hints inside the classes:
# e.g., operands: List[Union['Expression', 'Variable', Any]]
# Python 3.9+ handles this more gracefully with postponed evaluation of annotations by default.
# Python 3.7-3.8: from __future__ import annotations
# For this environment, explicit forward class stubs are used.
