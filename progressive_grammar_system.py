"""
Progressive Grammar System for Autonomous Physics Discovery
==========================================================

A hierarchical grammar that discovers variables from observations and
progressively builds mathematical abstractions using information-theoretic principles.
"""

import numpy as np
import sympy as sp
from typing import Dict, List, Tuple, Optional, Set, Any, TypeVar, Generic, Union # Added TypeVar, Generic, Union
from dataclasses import dataclass, field
from collections import defaultdict
import torch
import torch.nn as nn
from scipy.stats import entropy
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


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
        if not self.operands:
            return 1
        return 1 + sum(
            op.complexity if isinstance(op, Expression) else 1
            for op in self.operands
        )

    def _to_sympy(self) -> sp.Expr:
        if self.operator == 'var':
            if isinstance(self.operands[0], sp.Symbol):
                return self.operands[0]  # Already a symbol
            return sp.Symbol(self.operands[0]) # Create from string name
        elif self.operator == 'const':
            return sp.Float(self.operands[0])
        elif self.operator in ['+', '-', '*', '/', '**']:
            # Build symbolic args
            args = []
            for op in self.operands:
                if isinstance(op, Expression):
                    args.append(op.symbolic)
                elif hasattr(op, 'symbolic'):
                    args.append(op.symbolic)
                else:
                    args.append(op)
            # Dispatch operators
            if self.operator == '+':
                return sum(args)
            elif self.operator == '-':
                return args[0] - args[1] if len(args) > 1 else -args[0]
            elif self.operator == '*':
                result = args[0]
                for arg in args[1:]:
                    result *= arg
                return result
            elif self.operator == '/':
                denominator = args[1]
                # SymPy returns zoo for division by zero rather than raising an
                # error. Explicitly check for zero so that such cases yield
                # ``nan`` instead of ``zoo`` in the resulting expression.
                if (
                    (isinstance(denominator, (int, float, sp.Integer, sp.Float))
                     and denominator == 0)
                    or (isinstance(denominator, sp.Expr)
                        and denominator.is_zero is True)
                ):
                    return sp.nan
                return args[0] / denominator
            elif self.operator == '**':
                return args[0] ** args[1]
        elif self.operator in ['neg', 'inv', 'sqrt', 'log', 'exp', 'sin', 'cos']:
            # Unary operators
            operand_val = self.operands[0]
            if isinstance(operand_val, Expression):
                arg = operand_val.symbolic
            elif hasattr(operand_val, 'symbolic'):  # Variable
                arg = operand_val.symbolic
            else:  # Constant
                arg = sp.Float(operand_val)

            if self.operator == 'neg':
                return -arg
            elif self.operator == 'inv':
                if ((isinstance(arg, (int, float, sp.Integer, sp.Float)) and arg == 0)
                        or (isinstance(arg, sp.Expr) and arg.is_zero is True)):
                    return sp.nan
                return 1 / arg
            elif self.operator == 'sqrt':
                return sp.sqrt(arg)
            elif self.operator == 'log':
                return sp.log(arg)
            elif self.operator == 'exp':
                return sp.exp(arg)
            elif self.operator == 'sin':
                return sp.sin(arg)
            elif self.operator == 'cos':
                return sp.cos(arg)
        elif self.operator == 'diff':
            expr_op, var_op = self.operands
            # First operand can be Expression, Variable, or constant
            if isinstance(expr_op, Expression):
                sym_expr = expr_op.symbolic
            elif hasattr(expr_op, 'symbolic'): # Variable
                sym_expr = expr_op.symbolic
            else: # Constant
                sym_expr = sp.Float(expr_op)
            # Second operand must be a Variable (ensured by _validate_expression)
            sym_var = var_op.symbolic
            return sp.diff(sym_expr, sym_var)
        elif self.operator == 'int':
            expr_op, var_op = self.operands
            # First operand can be Expression, Variable, or constant
            if isinstance(expr_op, Expression):
                sym_expr = expr_op.symbolic
            elif hasattr(expr_op, 'symbolic'): # Variable
                sym_expr = expr_op.symbolic
            else: # Constant
                sym_expr = sp.Float(expr_op)
            # Second operand must be a Variable (ensured by _validate_expression)
            sym_var = var_op.symbolic
            return sp.integrate(sym_expr, sym_var)
        # Fallback
        return sp.Symbol(f"Unknown({self.operator})")



@dataclass
class Variable:
    """Discovered state variable with semantic properties."""
    name: str
    index: int  # Column in observation matrix
    properties: Dict[str, float] = field(default_factory=dict)
    symbolic: sp.Symbol = field(init=False)

    def __post_init__(self):
        self.symbolic = sp.Symbol(self.name)

    @property
    def complexity(self) -> int:
        return 1


class NoisyObservationProcessor:
    """Handles noisy observations using denoising autoencoders."""

    def __init__(self, latent_dim: int = 32):
        self.latent_dim = latent_dim
        self.encoder = None
        self.decoder = None
        self.scaler = StandardScaler()

    def build_autoencoder(self, input_dim: int):
        """Build denoising autoencoder for preprocessing."""
        class DenoisingAutoencoder(nn.Module):
            def __init__(self, input_dim, latent_dim):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, 128),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, latent_dim)
                )
                self.decoder = nn.Sequential(
                    nn.Linear(latent_dim, 64),
                    nn.ReLU(),
                    nn.Linear(64, 128),
                    nn.ReLU(),
                    nn.Linear(128, input_dim)
                )

            def forward(self, x, noise_level=0.1):
                # Add noise for denoising training
                if self.training:
                    noisy_x = x + torch.randn_like(x) * noise_level
                else:
                    noisy_x = x
                latent = self.encoder(noisy_x)
                reconstructed = self.decoder(latent)
                return reconstructed, latent

        self.model = DenoisingAutoencoder(input_dim, self.latent_dim)
        return self.model

    def denoise(self, observations: np.ndarray, epochs: int = 50) -> np.ndarray:
        """Train denoising autoencoder and return cleaned observations."""
        if observations.shape[0] < 100:
            # Not enough data for autoencoder, use simple filtering
            return self._simple_denoise(observations)

        # Normalize
        observations_scaled = self.scaler.fit_transform(observations)

        # Convert to torch
        data = torch.FloatTensor(observations_scaled)

        # Build and train autoencoder
        self.build_autoencoder(data.shape[1])
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            reconstructed, _ = self.model(data)
            loss = criterion(reconstructed, data)
            loss.backward()
            optimizer.step()

        # Get denoised data
        self.model.eval()
        with torch.no_grad():
            denoised, _ = self.model(data, noise_level=0)

        # Inverse transform
        return self.scaler.inverse_transform(denoised.numpy())

    def _simple_denoise(self, observations: np.ndarray) -> np.ndarray:
        """Simple moving average denoising for small datasets."""
        window = min(5, observations.shape[0] // 10)
        if window < 2:
            return observations

        denoised = np.copy(observations)
        for i in range(observations.shape[1]):
            denoised[:, i] = np.convolve(
                observations[:, i],
                np.ones(window)/window,
                mode='same'
            )
        return denoised


class ProgressiveGrammar:
    """
    Core grammar system that discovers variables and builds mathematical abstractions.
    Handles noise, ensures syntactic validity, and implements MDL-based learning.
    """

    COMMUTATIVE_OPS = {'+', '*'}

    def __init__(self,
                 max_variables: int = 20,
                 noise_threshold: float = 0.1,
                 mdl_threshold: float = 10.0,
                 load_defaults: bool = True): # Add this new argument

        # Core grammar components
        self.primitives = {
            'constants': {},
            'binary_ops': set(),
            'unary_ops': set(),
            'calculus_ops': set()
        }

        if load_defaults:
            self.primitives['constants'] = {'0': 0, '1': 1, 'pi': np.pi, 'e': np.e}
            self.primitives['binary_ops'] = {'+', '-', '*', '/', '**'}
            self.primitives['unary_ops'] = {'neg', 'inv', 'sqrt', 'log', 'exp', 'sin', 'cos'}
            self.primitives['calculus_ops'] = {'diff', 'int'}

        # Discovered components
        self.variables: Dict[str, 'Variable'] = {}
        self.learned_functions: Dict[str, Expression] = {}
        self.proven_lemmas: Dict[str, Expression] = {}

        # Configuration
        self.max_variables = max_variables
        self.noise_threshold = noise_threshold
        self.mdl_threshold = mdl_threshold

        # Components
        self.denoiser = NoisyObservationProcessor()
        self._expression_cache = {}

    def add_operators(self, operators: List[str]):
        """
        Dynamically adds a list of operators to the grammar's primitives,
        placing them in the correct category (unary, binary, etc.).
        """
        # Define known operators and their types
        known_binary = {'+', '-', '*', '/', '**'}
        known_unary = {'neg', 'inv', 'sqrt', 'log', 'exp', 'sin', 'cos'}
        known_calculus = {'diff', 'int'}

        for op in operators:
            if op in known_binary:
                self.primitives['binary_ops'].add(op)
            elif op in known_unary:
                self.primitives['unary_ops'].add(op)
            elif op in known_calculus:
                self.primitives['calculus_ops'].add(op)
            # Handle special cases from your MAML script
            elif op == '**2' or op == '**3':
                self.primitives['binary_ops'].add('**')
            elif op == '1/':
                self.primitives['unary_ops'].add('inv')
            else:
                # This can be extended if you add more custom operators
                print(f"Warning: Operator '{op}' has unknown arity and was not added.")

    def discover_variables(self,
                          observations: np.ndarray,
                          time_stamps: Optional[np.ndarray] = None) -> List['Variable']:
        """
        Discover state variables from noisy observations using ICA and
        information-theoretic measures.
        """
        # Step 1: Denoise observations
        clean_obs = self.denoiser.denoise(observations)

        # Step 2: Independent Component Analysis
        ica = FastICA(n_components=min(self.max_variables, clean_obs.shape[1]))
        components = ica.fit_transform(clean_obs)

        # Step 3: Analyze each component
        discovered_vars = []

        for i in range(components.shape[1]):
            component = components[:, i]

            # Information-theoretic analysis
            properties = self._analyze_component(component, time_stamps)

            # Only keep informative components
            if properties['information_content'] > self.noise_threshold:
                var_name = self._generate_variable_name(properties)
                var = Variable(
                    name=var_name,
                    index=i,
                    properties=properties
                )
                discovered_vars.append(var)
                self.variables[var_name] = var

        return discovered_vars

    def _analyze_component(self,
                          component: np.ndarray,
                          time_stamps: Optional[np.ndarray]) -> Dict[str, float]:
        """Analyze properties of a discovered component."""
        properties = {}

        # Information content (normalized entropy)
        hist, _ = np.histogram(component, bins=50)
        hist = hist + 1e-10  # Avoid log(0)
        properties['information_content'] = entropy(hist) / np.log(len(hist))

        # Conservation analysis (variance over time)
        if len(component) > 10:
            windows = np.array_split(component, min(10, len(component)//10))
            variances = [np.var(w) for w in windows]
            properties['conservation_score'] = 1.0 / (1.0 + np.var(variances))
        else:
            properties['conservation_score'] = 0.0

        # Periodicity analysis (FFT-based)
        if len(component) > 20:
            fft = np.fft.fft(component - np.mean(component))
            power = np.abs(fft[:len(fft)//2])
            peak_power = np.max(power[1:])  # Exclude DC component
            avg_power = np.mean(power[1:])
            properties['periodicity_score'] = peak_power / (avg_power + 1e-10)
        else:
            properties['periodicity_score'] = 0.0

        # Smoothness (based on derivatives)
        if len(component) > 2:
            derivatives = np.diff(component)
            properties['smoothness'] = 1.0 / (1.0 + np.std(derivatives))
        else:
            properties['smoothness'] = 0.0

        return properties

    def _generate_variable_name(self, properties: Dict[str, float]) -> str:
        """Generate semantic variable names based on properties."""
        # Prioritize by property strength
        if properties['conservation_score'] > 0.8:
            prefix = 'E'  # Energy-like
        elif properties['periodicity_score'] > 5.0:
            prefix = 'theta'  # Instead of θ
        elif properties['smoothness'] > 0.7:
            prefix = 'x'  # Position-like
        else:
            prefix = 'q'  # Generic state

        # Add unique identifier
        existing = [v for v in self.variables if v.startswith(prefix)]
        suffix = len(existing) + 1

        return f"{prefix}_{suffix}"

    def create_expression(self,
                         operator: str,
                         operands: List[Any],
                         validate: bool = True) -> Optional[Expression]:
        """
        Create an expression with syntactic validation.
        Ensures type safety and operator arity.
        """
        if validate and not self._validate_expression(operator, operands):
            return None

        expr = Expression(operator, operands)

        # Cache for efficiency
        expr_key = self._expression_key(expr)
        self._expression_cache[expr_key] = expr

        return expr

    def _validate_expression(self, operator: str, operands: List[Any]) -> bool:
        """Validate syntactic correctness of expression."""
        # Check operator exists
        all_ops = (set(self.primitives['binary_ops']) |
                  set(self.primitives['unary_ops']) |
                  set(self.primitives['calculus_ops']) |
                  {'var', 'const'})

        if operator not in all_ops:
            return False

        # Arity and Type checking
        if operator in self.primitives['binary_ops']:
            if len(operands) != 2: return False
            for op_val in operands:
                if not isinstance(op_val, (Expression, Variable, int, float)): return False
        elif operator in self.primitives['unary_ops']:
            if len(operands) != 1: return False
            for op_val in operands: # All unary ops take these types for now
                if not isinstance(op_val, (Expression, Variable, int, float)): return False
        elif operator in self.primitives['calculus_ops']:
            if len(operands) != 2: return False
            # First operand (expression to be differentiated/integrated)
            if not isinstance(operands[0], (Expression, Variable, int, float)): return False
            # Second operand (variable of differentiation/integration)
            if not isinstance(operands[1], Variable): return False
        elif operator == 'var':
            if len(operands) != 1: return False
            if not isinstance(operands[0], str): return False
        elif operator == 'const':
            if len(operands) != 1: return False
            if not isinstance(operands[0], (int, float)): return False
        else:
            # Should not be reached if operator is in all_ops
            return False

        return True

    def add_learned_function(self,
                           name: str,
                           expression: Expression,
                           usage_data: List[Expression]) -> bool:
        """
        Add new function abstraction based on MDL principle.
        Returns True if abstraction was added.
        """
        # Calculate compression gain
        compression_gain = self._calculate_compression_gain(
            expression,
            usage_data
        )

        if compression_gain > self.mdl_threshold:
            self.learned_functions[name] = expression

            # Update grammar to include new function
            self.primitives['unary_ops'].add(name)

            return True

        return False

    def _calculate_compression_gain(self,
                                  candidate: Expression,
                                  corpus: List[Expression]) -> float:
        """Calculate MDL-based compression gain."""
        # Current description length
        current_length = sum(expr.complexity for expr in corpus)

        # Description length with new abstraction
        new_length = candidate.complexity  # Cost of defining the abstraction

        for expr in corpus:
            # Count occurrences of candidate pattern
            occurrences = self._count_subexpression(expr, candidate)
            if occurrences > 0:
                # Each occurrence can be replaced by a single symbol
                saved = occurrences * (candidate.complexity - 1)
                new_length += expr.complexity - saved
            else:
                new_length += expr.complexity

        return current_length - new_length

    def _count_subexpression(self,
                           expr: Expression,
                           pattern: Expression) -> int:
        """Count occurrences of pattern in expression tree."""
        if self._expression_key(expr) == self._expression_key(pattern):
            return 1

        count = 0
        if hasattr(expr, 'operands'):
            for op in expr.operands:
                if isinstance(op, Expression):
                    count += self._count_subexpression(op, pattern)

        return count

    def _expression_key(self, expr: Expression) -> str:
        """Generate unique key for expression (for caching)."""
        if isinstance(expr, Variable):
            return f"var:{expr.name}"
        elif isinstance(expr, (int, float)):
            # Ensure consistent string representation for floats
            return f"const:{float(expr):.6g}" # Using .6g for a general representation
        elif isinstance(expr, str): # Should not happen with Expression objects but as a safeguard
            return expr # If a string operand somehow gets here
        else:
            operand_keys = []
            for op in expr.operands:
                if isinstance(op, Expression):
                    operand_keys.append(self._expression_key(op))
                elif isinstance(op, Variable): # Explicitly handle Variable in operands
                    operand_keys.append(f"var:{op.name}")
                elif isinstance(op, (int, float)): # Explicitly handle const in operands
                    operand_keys.append(f"const:{float(op):.6g}")
                else:
                    operand_keys.append(str(op)) # Fallback for other types

            if expr.operator in self.COMMUTATIVE_OPS:
                operand_keys.sort() # Sort keys for commutative operators

            return f"{expr.operator}({','.join(operand_keys)})"

    def mine_abstractions(self,
                         hypothesis_library: List[Expression],
                         min_frequency: int = 3) -> Dict[str, Expression]:
        """
        Mine common patterns from hypothesis library for abstraction.
        """
        # Count all subexpressions
        pattern_counts = defaultdict(int)
        pattern_examples = defaultdict(list)

        for hypothesis in hypothesis_library:
            subexprs = self._extract_all_subexpressions(hypothesis)
            for subexpr in subexprs:
                if subexpr.complexity > 2:  # Only consider non-trivial patterns
                    key = self._expression_key(subexpr)
                    pattern_counts[key] += 1
                    pattern_examples[key].append(subexpr)

        # Find patterns worth abstracting
        abstractions = {}
        for pattern_key, count in pattern_counts.items():
            if count >= min_frequency:
                example = pattern_examples[pattern_key][0]
                name = f"f_{len(self.learned_functions)}"

                # Check if abstraction is worthwhile
                if self.add_learned_function(name, example, hypothesis_library):
                    abstractions[name] = example

        return abstractions

    def _extract_all_subexpressions(self,
                                   expr: Expression,
                                   collected: Optional[Set] = None) -> List[Expression]:
        """Extract all subexpressions from an expression tree."""
        if collected is None:
            collected = set()

        result = []
        key = self._expression_key(expr)

        if key not in collected:
            collected.add(key)
            result.append(expr)

            if hasattr(expr, 'operands'):
                for op in expr.operands:
                    if isinstance(op, Expression):
                        result.extend(
                            self._extract_all_subexpressions(op, collected)
                        )

        return result

    def export_grammar_state(self) -> Dict:
        """Export current grammar state for persistence."""
        return {
            'variables': {
                name: {
                    'index': var.index,
                    'properties': var.properties
                }
                for name, var in self.variables.items()
            },
            'learned_functions': {
                name: self._expression_to_dict(expr)
                for name, expr in self.learned_functions.items()
            },
            'proven_lemmas': {
                name: self._expression_to_dict(expr)
                for name, expr in self.proven_lemmas.items()
            }
        }

    def _expression_to_dict(self, expr: Expression) -> Dict:
        """Convert expression to dictionary for serialization."""
        return {
            'operator': expr.operator,
            'operands': [
                self._expression_to_dict(op) if isinstance(op, Expression)
                else {'type': 'var', 'name': op.name} if isinstance(op, Variable)
                else {'type': 'const', 'value': op}
                for op in expr.operands
            ],
            'complexity': expr.complexity
        }


# Example usage and testing
if __name__ == "__main__":
    # Create grammar
    grammar = ProgressiveGrammar()

    # Generate synthetic physics data (pendulum-like system)
    t = np.linspace(0, 10, 1000)
    theta = np.sin(2 * np.pi * 0.5 * t) + 0.1 * np.random.randn(len(t))
    omega = np.cos(2 * np.pi * 0.5 * t) * 2 * np.pi * 0.5 + 0.1 * np.random.randn(len(t))
    energy = 0.5 * omega**2 + 9.8 * (1 - np.cos(theta)) + 0.05 * np.random.randn(len(t))

    # Combine into observation matrix
    observations = np.column_stack([theta, omega, energy,
                                   np.random.randn(len(t)) * 0.5])  # Noise column

    # Discover variables
    print("Discovering variables from observations...")
    variables = grammar.discover_variables(observations, t)

    print(f"\nDiscovered {len(variables)} variables:")
    for var in variables:
        print(f"  {var.name}: {var.properties}")

    # Create some expressions
    print("\nCreating expressions...")
    if len(variables) >= 2:
        v1, v2 = variables[0], variables[1]

        # Kinetic energy-like expression
        expr1 = grammar.create_expression('*', [
            grammar.create_expression('const', [0.5]),
            grammar.create_expression('**', [v1, grammar.create_expression('const', [2])])
        ])

        print(f"Expression 1: {expr1.symbolic}")
        print(f"Complexity: {expr1.complexity}")

        # Test syntactic validation
        invalid = grammar.create_expression('+', [v1])  # Wrong arity
        print(f"\nInvalid expression (wrong arity): {invalid}")

        # Test calculus operations
        derivative = grammar.create_expression('diff', [expr1, v1])
        print(f"\nDerivative of expr1 w.r.t {v1.name}: {derivative.symbolic}")


class AIGrammar(ProgressiveGrammar):
    def __init__(self):
        super().__init__(load_defaults=False) # Assuming AI grammar might not need all default physics ops

        # Neural network primitives
        # For 'activation', these are values, not ops in the traditional sense.
        # Consider how they integrate. For now, adding as a list.
        self.add_primitive_set('activation_types', ['relu', 'sigmoid', 'tanh', 'gelu'])

        # These will be functional primitives.
        # Their evaluation will require specific logic, possibly outside direct sympy conversion.
        self.add_primitive('attention', self._attention_op)
        self.add_primitive('embedding_lookup', self._embedding_op)

        # Logical primitives
        self.add_primitive('if_then_else', lambda cond, true_val, false_val:
                          true_val if cond else false_val)
        self.add_primitive('threshold', lambda x, t: x > t) # x > t returns boolean

        # Aggregation primitives
        self.add_primitive('weighted_sum', lambda weights, values:
                          sum(w*v for w,v in zip(weights, values)))
        self.add_primitive('max_pool', lambda values: max(values) if values else None) # Handle empty list

    def add_primitive_set(self, name: str, values: List[str]):
        """Adds a named set of primitive values."""
        if 'custom_sets' not in self.primitives:
            self.primitives['custom_sets'] = {}
        self.primitives['custom_sets'][name] = values

    def add_primitive(self, name: str, func_or_values: Any, category: Optional[str] = None):
        """
        Adds a primitive. If func_or_values is callable, it's treated as an operator.
        Otherwise, it could be a list of values for a specific category.
        The 'category' argument helps classify operators for arity, validation etc.
        """
        if callable(func_or_values):
            # Determine category if not provided, or use a default like 'custom_callable'
            # This part needs refinement based on how these ops are used (unary, binary, etc.)
            # For now, let's assume they might be custom and need special handling in validation/evaluation.
            cat = category if category else 'custom_callable_ops'
            if cat not in self.primitives:
                self.primitives[cat] = {}
            self.primitives[cat][name] = func_or_values
        else:
            # This path is less clear for the new primitives.
            # 'activation' seems like a set of types, not a single primitive operator.
            # Handled by add_primitive_set for 'activation_types'.
            # For other cases, this might store lists of constants or specific values.
            # For now, focusing on callable primitives.
            # If 'name' is e.g. 'activation_function_types' and values are ['relu', 'sigmoid']
            if isinstance(func_or_values, list):
                if 'named_lists' not in self.primitives:
                    self.primitives['named_lists'] = {}
                self.primitives['named_lists'][name] = func_or_values
            else:
                # Fallback for single non-callable value, similar to constants
                if 'custom_values' not in self.primitives:
                    self.primitives['custom_values'] = {}
                self.primitives['custom_values'][name] = func_or_values


    def _attention_op(self, query: Any, key: Any, value: Any) -> Any:
        """Placeholder for attention mechanism operation."""
        # In a real scenario, this would involve complex tensor operations.
        # For symbolic representation, it might become a named function.
        # e.g., Attention(Q, K, V)
        print(f"Warning: _attention_op is a placeholder. Query: {query}, Key: {key}, Value: {value}")
        # raise NotImplementedError("Attention operation is not implemented symbolically yet.")
        return f"Attention({query}, {key}, {value})" # Placeholder symbolic string

    def _embedding_op(self, indices: Any, embedding_matrix: Any) -> Any:
        """Placeholder for embedding lookup operation."""
        # e.g., EmbeddingLookup(indices, matrix_name)
        print(f"Warning: _embedding_op is a placeholder. Indices: {indices}, Matrix: {embedding_matrix}")
        # raise NotImplementedError("Embedding lookup is not implemented symbolically yet.")
        return f"EmbeddingLookup({indices}, {embedding_matrix})" # Placeholder symbolic string

    def _to_sympy(self, expr_node: Expression) -> sp.Expr: # Overriding to handle new ops
        """
        Converts an Expression node to its Sympy representation.
        Extends base class method to handle AI-specific primitives.
        """
        # Custom callable ops (like if_then_else, threshold, etc.)
        # These might not have direct Sympy equivalents or might be represented as Functions.
        if expr_node.operator in self.primitives.get('custom_callable_ops', {}):
            # For now, represent them as named Sympy functions.
            # Their actual evaluation would happen outside Sympy, during numerical simulation.
            func_name = expr_node.operator

            # Recursively convert operands to Sympy expressions
            sympy_operands = []
            for op in expr_node.operands:
                if isinstance(op, Expression):
                    sympy_operands.append(self._to_sympy(op))
                elif isinstance(op, Variable):
                    sympy_operands.append(op.symbolic)
                elif isinstance(op, (int, float)):
                    sympy_operands.append(sp.Number(op))
                elif isinstance(op, str): # Could be a string literal or symbolic name
                    sympy_operands.append(sp.Symbol(op)) # Treat as symbol if string
                else: # Fallback for unknown operand types
                    sympy_operands.append(sp.Symbol(str(op)))

            # Handle specific lambda functions for more precise Sympy representation if possible
            if func_name == 'if_then_else' and len(sympy_operands) == 3:
                # Sympy's Piecewise can represent if-then-else
                # Piecewise((true_expr, cond_expr), (false_expr, True))
                # This assumes cond_expr evaluates to a boolean in Sympy context.
                # The lambda `cond` is a boolean, `true_val` and `false_val` are values.
                # This is tricky because the lambda is evaluated by Python, not Sympy directly.
                # For symbolic representation, we might need to create a Sympy Function.
                 return sp.Function(func_name)(*sympy_operands) # Default to Function for now

            elif func_name == 'threshold' and len(sympy_operands) == 2:
                # x > t  (returns boolean)
                # Sympy can represent this directly as a relational expression
                return sympy_operands[0] > sympy_operands[1]

            # For others like weighted_sum, max_pool, attention, embedding_lookup:
            # These are complex operations. Representing them as opaque Sympy Functions.
            return sp.Function(func_name)(*sympy_operands)

        # For activation types like 'relu', 'sigmoid' - these are not operators in Expression
        # They are typically applied to an expression. e.g. relu(expr).
        # If they are used as operators in an Expression e.g. Expression('relu', [operand_expr]),
        # then they should be in 'unary_ops' or similar.
        # The current AIGrammar setup adds 'activation_types' as a list of strings.
        # If an expression like `Expression('relu', [sub_expr])` is formed,
        # `_validate_expression` and `_to_sympy` need to know 'relu' is a valid unary op.
        # For now, assuming 'relu' etc. might be added to 'unary_ops' if used this way.

        # Fallback to parent's _to_sympy for standard operators
        # Need to ensure the Expression object is passed, not self.
        # The original _to_sympy in Expression class is `expr_node._to_sympy()`
        # The method signature in ProgressiveGrammar is `_to_sympy(self) -> sp.Expr` for an Expression instance.
        # This needs to be consistent. Let's assume we are working with an Expression instance `expr_node`.

        # If this AIGrammar._to_sympy is called on an Expression instance, it should be:
        # return super(AIGrammar, type(expr_node))._to_sympy(expr_node) # If AIGrammar is not an Expression subclass
        # However, ProgressiveGrammar.create_expression returns Expression instances.
        # Expression._to_sympy is the primary method.
        # This override implies AIGrammar itself might be an Expression or it's a utility method.
        # Given the context, it's likely that an Expression object `expr_node` is passed,
        # and this method is part of AIGrammar to customize its conversion.

        # The original `Expression._to_sympy` is what we'd call:
        # Let's assume this method is intended to be part of the Expression class logic
        # when the grammar is AIGrammar. This is architecturally complex.
        # A simpler way: ProgressiveGrammar.create_expression creates Expression objects.
        # Expression._to_sympy should be enhanced or AIGrammar should provide a dedicated
        # conversion utility that it uses.

        # For now, let's assume this method is called with an Expression `expr_node`
        # and if the operator is not custom, it defers to the standard Expression._to_sympy logic.
        # This requires `expr_node` to have access to its original `_to_sympy` if this one doesn't handle it.
        # This is problematic.

        # Alternative: AIGrammar's `create_expression` returns a specialized AIExpression(Expression)
        # which has its own `_to_sympy`.
        # Or, `Expression._to_sympy` itself becomes aware of the grammar context.

        # Safest assumption: if an operator is not custom AI, it's handled by base Expression logic.
        # This means this method should only handle *new* AI operators.
        # The `Expression` class's `_to_sympy` method would need to be able to call out to the
        # grammar system for custom types, or this logic needs to be merged/refactored.

        # Let's assume this is a helper, and the main call is still `Expression.symbolic` which calls `Expression._to_sympy`.
        # `Expression._to_sympy` would need to be modified to consult the grammar for these.

        # For the purpose of this step, we are modifying AIGrammar.
        # If Expression._to_sympy is called, and it encounters an op like 'if_then_else',
        # it needs to know how to handle it.
        # This suggests `Expression._to_sympy` should be modified, or it should delegate to the grammar.

        # Let's assume this method *replaces* the logic for AI specific ops if called.
        # The base ProgressiveGrammar does not have _to_sympy. Expression class does.
        # This method is incorrectly placed if it's meant to override Expression._to_sympy.

        # Re-evaluating: The `Expression` class has `_to_sympy`.
        # `AIGrammar` should not have its own `_to_sympy` with this signature unless `AIGrammar` instances are `Expression`s.
        # The change should ideally be within `Expression._to_sympy` to make it grammar-aware,
        # or `AIGrammar` provides lookup for these custom functions that `Expression._to_sympy` can use.

        # Sticking to the plan of modifying AIGrammar:
        # This method implies it's a utility FOR Expression, or it's a misinterpretation of where to put it.
        # If it's a utility:
        # def convert_ai_expression_to_sympy(self, expr_node: Expression) -> sp.Expr:
        # ... then Expression._to_sympy would call this.

        # For now, let's assume this is the intended override path, and Expression class will be modified
        # to call `grammar_instance.convert_expression_to_sympy(self)` if a grammar is associated.
        # This is a larger architectural change.

        # Simplification: The `Expression` class itself should be made extensible or grammar-aware.
        # Adding this method to `AIGrammar` means it's a helper.
        # Let's proceed with the definition here, and acknowledge that `Expression._to_sympy`
        # would need to be modified to use this.

        # If the operator is not one of the custom AI ones, it should defer to the base implementation.
        # However, ProgressiveGrammar doesn't have _to_sympy. Expression class does.
        # This method cannot `super()._to_sympy()` if AIGrammar is not an Expression.
        # This indicates a structural issue with the request vs. codebase.

        # Let's assume this method is intended to be ADDED to the Expression class,
        # or the Expression class's _to_sympy is MODIFIED.
        # Since the task is to modify AIGrammar, this method's role is likely a helper or a policy.

        # For now, if it's not a custom op, this specific method shouldn't handle it.
        # It should only define behavior for *new* ops.
        # The original Expression._to_sympy will handle the rest.
        # This means Expression._to_sympy needs to be modified to call:
        # `if self.operator in current_grammar.get_custom_ai_ops(): return current_grammar.custom_ai_op_to_sympy(self)`

        # Given the file is `progressive_grammar_system.py`, and this is `AIGrammar`,
        # this method defines *how* AIGrammar would want these ops converted.
        # The `Expression` class would be the one to *use* this definition.

        # So, this method is more of a policy/handler lookup for Expression.
        # It should not call super()._to_sympy(). It should raise error if op unknown to it.
        raise NotImplementedError(f"Operator '{expr_node.operator}' not handled by AIGrammar._to_sympy policy.")


    def _validate_expression(self, operator: str, operands: List[Any]) -> bool: # Overriding
        """
        Validate syntactic correctness of expression, extended for AI primitives.
        """
        # Handle AI custom callable operators
        if operator in self.primitives.get('custom_callable_ops', {}):
            # Basic validation: check if operator is known
            # Arity checks would be specific to each custom op.
            # Example: 'if_then_else' needs 3 operands, 'threshold' needs 2.
            # This needs to be made more robust, perhaps by storing arity with primitives.
            if operator == 'if_then_else':
                if len(operands) != 3: return False
                # Further type checks: cond (bool), true_val, false_val (any type)
            elif operator == 'threshold':
                if len(operands) != 2: return False
                # Further type checks: x (numeric), t (numeric)
            elif operator == 'weighted_sum':
                if len(operands) != 2: return False # weights_list, values_list
            elif operator == 'max_pool':
                if len(operands) != 1: return False # values_list
            elif operator == 'attention': # Q, K, V
                if len(operands) != 3: return False
            elif operator == 'embedding_lookup': # indices, matrix
                if len(operands) != 2: return False
            # Assume valid if basic arity (if checked) passes. More detailed type checking can be added.
            return True

        # Handle activation functions if they are treated as operators (e.g., unary)
        # If 'relu' is in `self.primitives['unary_ops']` (added by user or a setup method)
        # then super()._validate_expression should handle it.
        # The current AIGrammar adds 'activation_types' as a list of strings,
        # not as operators. If they become operators, they need to be added to unary_ops etc.
        # e.g., self.primitives['unary_ops'].update(self.primitives['custom_sets']['activation_types'])
        # For now, assuming they are not operators in this validation path.

        return super()._validate_expression(operator, operands)

    # Note: _expression_key might also need overriding if these new operators
    # have specific canonicalization needs (e.g., commutativity, though unlikely for these).
    # For now, the base class key generation using operator name and sorted operands (if commutative)
    # might suffice, assuming these new ops are not commutative.
    # If `if_then_else(cond, A, B)` is different from `if_then_else(cond, B, A)`, standard keying is fine.

    # The Expression class's _to_sympy method will need to be modified to correctly
    # use the AIGrammar's policies for converting these new primitives to Sympy forms.
    # This current implementation of AIGrammar._to_sympy is a *policy* that Expression._to_sympy could use.
    # It's not a direct override of Expression._to_sympy.

# Example usage for AIGrammar (illustrative)
if __name__ == "__main__":
    # Existing example code from ProgressiveGrammar...
    grammar = ProgressiveGrammar()
    # ... (rest of the ProgressiveGrammar example)

    print("\n--- AIGrammar Example ---")
    ai_grammar = AIGrammar()

    # Example: Using a variable (assuming it's discovered or defined)
    # For AIGrammar, variables might represent tensor shapes, features, etc.
    # Let's create dummy variables for illustration.
    q_var = Variable(name="query_tensor", index=0)
    k_var = Variable(name="key_tensor", index=1)
    v_var = Variable(name="value_tensor", index=2)
    feature_x = Variable(name="feature_x", index=3)
    threshold_val = Variable(name="threshold_const", index=4) # Or could be a const

    ai_grammar.variables = {
        "query_tensor": q_var, "key_tensor": k_var, "value_tensor": v_var,
        "feature_x": feature_x, "threshold_const": threshold_val
    }
    # Manually add 'relu' to unary ops for testing if it's used like `relu(x)`
    if 'unary_ops' not in ai_grammar.primitives: ai_grammar.primitives['unary_ops'] = set()
    ai_grammar.primitives['unary_ops'].add('relu') # Assume 'relu' is a known unary op


    # Test creating an expression with a new AI primitive
    # Note: The Expression class's _to_sympy would need to be aware of AIGrammar's policies.
    # This example shows creation; symbolic conversion needs Expression class changes.

    # 1. Threshold expression
    # threshold_expr_ai = ai_grammar.create_expression('threshold', [feature_x, threshold_val])
    # if threshold_expr_ai:
    #     print(f"AI Threshold Expr: {threshold_expr_ai.operator}({feature_x.name}, {threshold_val.name})")
    #     # To print symbolic: threshold_expr_ai.symbolic (requires Expression class to use AIGrammar's policy)
    #     # This will likely fail or give 'Unknown(threshold)' if Expression._to_sympy is not updated.
    #     # print(f"Symbolic (requires Expression update): {threshold_expr_ai.symbolic}")
    # else:
    #     print("Failed to create AI threshold expression.")

    # Illustrative: If Expression._to_sympy was modified to use AIGrammar's policy:
    # Assuming Expression class is modified like:
    # class Expression:
    #   ...
    #   def _to_sympy(self):
    #     if self.operator in grammar.primitives.get('custom_callable_ops',{}):
    #        return grammar._to_sympy(self) # Call grammar's policy for this expression node
    #     ... (original logic) ...

    # For now, the direct .symbolic call on Expression objects created with AIGrammar
    # for new AI ops will not work as intended without modifying Expression class.
    # The AIGrammar._to_sympy here defines *how* it should be done.

    # Let's test validation path
    print("\nTesting validation for AIGrammar:")
    valid_threshold = ai_grammar._validate_expression('threshold', [feature_x, 0.5])
    print(f"Validation for 'threshold' (2 operands): {valid_threshold}")
    invalid_threshold = ai_grammar._validate_expression('threshold', [feature_x])
    print(f"Validation for 'threshold' (1 operand): {invalid_threshold}")

    valid_attention = ai_grammar._validate_expression('attention', [q_var, k_var, v_var])
    print(f"Validation for 'attention' (3 operands): {valid_attention}")

    # Example of creating an expression that might use a traditional op via superclass validation
    # This assumes 'relu' was added to unary_ops for AIGrammar for this test.
    # relu_expr_ai = ai_grammar.create_expression('relu', [feature_x])
    # if relu_expr_ai:
    #    print(f"AI Relu Expr: {relu_expr_ai.operator}({feature_x.name})")
    #    # print(f"Symbolic (should work if relu is standard unary): {relu_expr_ai.symbolic}")
    # else:
    #    print("Failed to create AI relu expression.")

    # The main challenge remains: Expression class is independent of grammar object at instance level.
    # Making Expression operations (like .symbolic, .complexity, validation) grammar-aware
    # is a deeper refactoring. This AIGrammar class adds the *definitions* and *policies*
    # for AI primitives.

# --- AIGrammar get_arity override ---
def ai_grammar_get_arity(self, op_name: str) -> int:
    """
    Returns the arity of a given operator, including AI-specific custom operators.
    Raises ValueError if the operator is unknown.
    """
    # Arities for known custom AI operators
    # These should match the definitions in AIGrammar._validate_expression or where arities are defined.
    _ai_op_arities = {
        'attention': 3,         # Query, Key, Value
        'embedding_lookup': 2,  # Indices, Embedding Matrix
        'if_then_else': 3,      # Condition, True_Branch, False_Branch
        'threshold': 2,         # Input, Threshold_Value
        'weighted_sum': 2,      # Weights_List, Values_List
        'max_pool': 1           # Values_List
        # Any other custom callable ops defined in AIGrammar.__init__ should be added here
    }
    if op_name in _ai_op_arities:
        return _ai_op_arities[op_name]

    # Activation functions like 'relu', 'sigmoid', etc., if treated as unary operators
    # In AIGrammar, they are currently in 'activation_types' and might be added to 'unary_ops'
    # by a setup step or if create_expression uses them as such.
    # If they are in self.primitives['unary_ops'], super().get_arity will handle them.

    # Fallback to ProgressiveGrammar's get_arity for standard operators
    return super(AIGrammar, self).get_arity(op_name)

AIGrammar.get_arity = ai_grammar_get_arity
# --- End AIGrammar get_arity override ---

# Methods added to ProgressiveGrammar
ProgressiveGrammar.get_arity = lambda self, op_name: \
    2 if op_name in self.primitives.get('binary_ops', set()) else \
    1 if op_name in self.primitives.get('unary_ops', set()) else \
    2 if op_name in self.primitives.get('calculus_ops', set()) else \
    (_ for _ in ()).throw(ValueError(f"Unknown operator or function: '{op_name}' in ProgressiveGrammar"))

ProgressiveGrammar.is_operator_known = lambda self, op_name: \
    isinstance(self.get_arity(op_name), int) if hasattr(self, 'get_arity') and callable(getattr(self, 'get_arity')) else False \
    if True else True # The if True else True is a bit of a hack to make this a one-liner lambda that can catch the ValueError

# Temporary fix for is_operator_known to handle ValueError correctly in a lambda
def _is_operator_known_impl(grammar_instance, op_name):
    try:
        grammar_instance.get_arity(op_name)
        return True
    except ValueError:
        return False
ProgressiveGrammar.is_operator_known = lambda self, op_name: _is_operator_known_impl(self, op_name)
