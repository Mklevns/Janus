

# Assuming ProgressiveGrammar, Expression, Variable are in a place accessible by this path
# If they are part of the old root structure and not yet moved:
# from janus.core.grammar import ProgressiveGrammar
# from janus.core.expression import Expression, Variable
# If they are meant to be part of janus core or a shared module:
# from ....shared import ProgressiveGrammar, Expression, Variable
# For now, using placeholder relative imports if they are also being moved into janus structure
# from ...core.grammar import ProgressiveGrammar, Expression, Variable
# Based on file listing, `grammar.py` is now in `janus/core`.
# So, the import needs to be adjusted once its final location is decided.
# For this refactoring, we'll assume it will be findable from the new structure.
# A common pattern is to have a `core` or `common` module at `janus/` level.


import sympy as sp
from typing import List, Dict, Any, Optional
from progressive_grammar_system import ProgressiveGrammar, Expression, Variable

class AIGrammar(ProgressiveGrammar):
    """
    Extended grammar for AI interpretability with neural network primitives
    and proper symbolic conversion support.
    """

    def __init__(self):
        super().__init__()
        self._init_ai_primitives()
        self._ai_operator_arities = self._define_ai_operator_arities()

    def _init_ai_primitives(self):
        """Initialize AI-specific primitives and operators."""
        # Custom AI operators that aren't in standard ProgressiveGrammar
        ai_custom_ops = {
            'attention',         # Query-Key-Value attention mechanism
            'embedding_lookup',  # Token/index to embedding vector
            'if_then_else',     # Conditional logic
            'threshold',        # Step/threshold function  
            'weighted_sum',     # Weighted combination
            'max_pool',         # Max pooling operation
            'softmax',          # Softmax normalization
            'layer_norm',       # Layer normalization
        }
        
        # Add custom callable operators to primitives
        if 'custom_callable_ops' not in self.primitives:
            self.primitives['custom_callable_ops'] = set()
        self.primitives['custom_callable_ops'].update(ai_custom_ops)
        
        # AI-specific activation functions (can be used as unary operators)
        ai_activations = {'relu', 'sigmoid', 'tanh', 'gelu', 'swish'}
        if 'unary_ops' not in self.primitives:
            self.primitives['unary_ops'] = set()
        self.primitives['unary_ops'].update(ai_activations)
        
        # Add custom sets for organization
        if 'custom_sets' not in self.primitives:
            self.primitives['custom_sets'] = {}
        
        self.primitives['custom_sets']['activation_types'] = list(ai_activations)
        self.primitives['custom_sets']['ai_operations'] = list(ai_custom_ops)

    def _define_ai_operator_arities(self) -> Dict[str, int]:
        """Define the arity (number of arguments) for AI-specific operators."""
        return {
            'attention': 3,         # Query, Key, Value
            'embedding_lookup': 2,  # Indices, Embedding Matrix
            'if_then_else': 3,      # Condition, True_Branch, False_Branch
            'threshold': 2,         # Input, Threshold_Value
            'weighted_sum': 2,      # Weights, Values
            'max_pool': 1,          # Values (can be extended for kernel_size)
            'softmax': 1,           # Input values
            'layer_norm': 1,        # Input values
            # Activation functions (unary)
            'relu': 1,
            'sigmoid': 1,
            'tanh': 1,
            'gelu': 1,
            'swish': 1,
        }

    def get_arity(self, op_name: str) -> int:
        """
        Returns the arity of a given operator, including AI-specific operators.
        Overrides parent class to handle AI primitives.
        """
        # Check AI-specific operators first
        if op_name in self._ai_operator_arities:
            return self._ai_operator_arities[op_name]
        
        # Fall back to parent class for standard operators
        try:
            return super().get_arity(op_name)
        except ValueError:
            raise ValueError(f"Unknown operator: '{op_name}' in AIGrammar")

    def is_operator_known(self, op_name: str) -> bool:
        """Check if an operator is known to this grammar."""
        try:
            self.get_arity(op_name)
            return True
        except ValueError:
            return False

    def _validate_expression(self, operator: str, operands: List[Any]) -> bool:
        """
        Validate syntactic correctness of expression, extended for AI primitives.
        Overrides parent class validation.
        """
        # Handle AI custom callable operators
        if operator in self.primitives.get('custom_callable_ops', set()):
            expected_arity = self._ai_operator_arities.get(operator)
            if expected_arity is None:
                return False
            
            if len(operands) != expected_arity:
                return False
            
            # Additional semantic validation for specific operators
            if operator == 'if_then_else':
                # Condition should be boolean-like, branches can be any type
                return len(operands) == 3
            elif operator == 'threshold':
                # Input and threshold should be numeric
                return len(operands) == 2
            elif operator == 'attention':
                # Query, Key, Value tensors
                return len(operands) == 3
            elif operator == 'embedding_lookup':
                # Indices and embedding matrix
                return len(operands) == 2
            
            return True

        # Handle AI activation functions if they're used as operators
        if operator in self.primitives.get('unary_ops', set()) and \
           operator in self.primitives['custom_sets'].get('activation_types', []):
            return len(operands) == 1

        # Fall back to parent class validation for standard operators
        return super()._validate_expression(operator, operands)

    def convert_ai_expression_to_sympy(self, expr_node: Expression) -> sp.Expr:
        """
        Convert AI-specific expression nodes to SymPy expressions.
        
        This method provides the conversion policy for AI operators.
        It should be called by Expression._to_sympy when needed.
        """
        operator = expr_node.operator
        
        # Convert operands to SymPy first
        if hasattr(expr_node, 'operands') and expr_node.operands:
            sympy_operands = []
            for operand in expr_node.operands:
                if hasattr(operand, '_to_sympy'):
                    sympy_operands.append(operand._to_sympy())
                elif hasattr(operand, 'symbolic'):
                    sympy_operands.append(operand.symbolic)
                elif isinstance(operand, (int, float)):
                    sympy_operands.append(operand)
                else:
                    sympy_operands.append(sp.Symbol(str(operand)))
        else:
            sympy_operands = []

        # Handle AI-specific operators
        if operator == 'attention':
            # Simplified attention: score = Q*K, output = score*V
            if len(sympy_operands) >= 3:
                q, k, v = sympy_operands[0], sympy_operands[1], sympy_operands[2]
                # Simplified: attention(Q,K,V) ≈ (Q*K)*V
                score = q * k
                return score * v
            else:
                return sp.Function('attention')(*sympy_operands)

        elif operator == 'embedding_lookup':
            # embedding_lookup(indices, matrix) - represented as function
            return sp.Function('embedding_lookup')(*sympy_operands)

        elif operator == 'if_then_else':
            # Piecewise function: if_then_else(cond, true_val, false_val)
            if len(sympy_operands) >= 3:
                cond, true_val, false_val = sympy_operands[0], sympy_operands[1], sympy_operands[2]
                return sp.Piecewise((true_val, cond > 0), (false_val, True))
            else:
                return sp.Function('if_then_else')(*sympy_operands)

        elif operator == 'threshold':
            # threshold(x, t) -> Heaviside step function
            if len(sympy_operands) >= 2:
                x, t = sympy_operands[0], sympy_operands[1]
                return sp.Piecewise((1, x > t), (0, True))
            else:
                return sp.Function('threshold')(*sympy_operands)

        elif operator == 'weighted_sum':
            # weighted_sum(weights, values) - simplified as multiplication
            if len(sympy_operands) >= 2:
                weights, values = sympy_operands[0], sympy_operands[1]
                return weights * values
            else:
                return sp.Function('weighted_sum')(*sympy_operands)

        elif operator == 'max_pool':
            # max_pool(values) -> Max function
            if len(sympy_operands) >= 1:
                return sp.Max(*sympy_operands)
            else:
                return sp.Function('max_pool')(*sympy_operands)

        elif operator == 'softmax':
            # softmax(x) - represented as exp(x)/sum(exp(x)) for single input
            if len(sympy_operands) >= 1:
                x = sympy_operands[0]
                return sp.exp(x) / (1 + sp.exp(x))  # Simplified for symbolic
            else:
                return sp.Function('softmax')(*sympy_operands)

        elif operator == 'layer_norm':
            # layer_norm(x) - simplified normalization
            if len(sympy_operands) >= 1:
                x = sympy_operands[0]
                return x / sp.sqrt(1 + x**2)  # Simplified normalization
            else:
                return sp.Function('layer_norm')(*sympy_operands)

        # AI activation functions
        elif operator == 'relu':
            if len(sympy_operands) >= 1:
                x = sympy_operands[0]
                return sp.Max(0, x)
            else:
                return sp.Function('relu')(*sympy_operands)

        elif operator == 'sigmoid':
            if len(sympy_operands) >= 1:
                x = sympy_operands[0]
                return 1 / (1 + sp.exp(-x))
            else:
                return sp.Function('sigmoid')(*sympy_operands)

        elif operator == 'gelu':
            if len(sympy_operands) >= 1:
                x = sympy_operands[0]
                # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
                return 0.5 * x * (1 + sp.tanh(sp.sqrt(2/sp.pi) * (x + 0.044715 * x**3)))
            else:
                return sp.Function('gelu')(*sympy_operands)

        elif operator == 'swish':
            if len(sympy_operands) >= 1:
                x = sympy_operands[0]
                return x / (1 + sp.exp(-x))  # swish(x) = x * sigmoid(x)
            else:
                return sp.Function('swish')(*sympy_operands)

        # If operator is not recognized as AI-specific, raise error
        raise NotImplementedError(f"AI operator '{operator}' not implemented in convert_ai_expression_to_sympy")

    def create_ai_expression(self, operator: str, operands: List[Any]) -> Optional[Expression]:
        """
        Create an Expression with AI-specific operators.
        This is a convenience method for creating AI expressions.
        """
        if not self._validate_expression(operator, operands):
            return None
        
        # Use parent class create_expression but with AI validation
        return self.create_expression(operator, operands)

    def get_available_ai_operators(self) -> Dict[str, int]:
        """Return all available AI operators with their arities."""
        ai_ops = {}
        
        # Custom callable operations
        for op in self.primitives.get('custom_callable_ops', set()):
            if op in self._ai_operator_arities:
                ai_ops[op] = self._ai_operator_arities[op]
        
        # AI activation functions
        for op in self.primitives['custom_sets'].get('activation_types', []):
            if op in self._ai_operator_arities:
                ai_ops[op] = self._ai_operator_arities[op]
        
        return ai_ops

    def add_custom_ai_operator(self, operator: str, arity: int, 
                              sympy_converter: Optional[callable] = None):
        """
        Add a new custom AI operator to the grammar.
        
        Args:
            operator: Name of the operator
            arity: Number of arguments the operator takes
            sympy_converter: Optional function to convert to SymPy
        """
        # Add to custom callable ops
        if 'custom_callable_ops' not in self.primitives:
            self.primitives['custom_callable_ops'] = set()
        self.primitives['custom_callable_ops'].add(operator)
        
        # Set arity
        self._ai_operator_arities[operator] = arity
        
        # Store custom converter if provided
        if sympy_converter:
            if not hasattr(self, '_custom_converters'):
                self._custom_converters = {}
            self._custom_converters[operator] = sympy_converter


# Monkey patch to fix Expression._to_sympy integration with AIGrammar
def _enhanced_expression_to_sympy(self):
    """
    Enhanced _to_sympy method that works with AIGrammar.
    This should be monkey-patched onto the Expression class.
    """
    # Check if we have a grammar context that can handle AI operators
    if hasattr(self, '_grammar_context') and isinstance(self._grammar_context, AIGrammar):
        grammar = self._grammar_context
        if self.operator in grammar.primitives.get('custom_callable_ops', set()) or \
           self.operator in grammar.primitives.get('custom_sets', {}).get('activation_types', []):
            try:
                return grammar.convert_ai_expression_to_sympy(self)
            except NotImplementedError:
                pass  # Fall back to original method
    
    # Fall back to original _to_sympy implementation
    return self._original_to_sympy()

# Function to patch Expression class for AI compatibility
def patch_expression_for_ai_grammar():
    """
    Apply patches to Expression class to support AIGrammar operators.
    Call this once after importing to enable AI grammar support.
    """
    if not hasattr(Expression, '_original_to_sympy'):
        Expression._original_to_sympy = Expression._to_sympy
        Expression._to_sympy = _enhanced_expression_to_sympy
        
        # Add method to set grammar context
        def set_grammar_context(self, grammar):
            self._grammar_context = grammar
        Expression.set_grammar_context = set_grammar_context
