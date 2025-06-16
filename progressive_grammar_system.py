"""
Progressive Grammar System for Autonomous Physics Discovery
==========================================================

A hierarchical grammar that discovers variables from observations and
progressively builds mathematical abstractions using information-theoretic principles.
"""

import numpy as np
import sympy as sp
from typing import Dict, List, Tuple, Optional, Set, Any
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
            return sp.Symbol(self.operands[0])
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
        elif self.operator == 'diff':
            expr, var = self.operands
            return sp.diff(expr.symbolic, var.symbolic)
        elif self.operator == 'int':
            expr, var = self.operands
            return sp.integrate(expr.symbolic, var.symbolic)
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
                 mdl_threshold: float = 10.0):

        # Core grammar components
        self.primitives = {
            'constants': {'0': 0, '1': 1, 'pi': np.pi, 'e': np.e},
            'binary_ops': {'+', '-', '*', '/', '**'},
            'unary_ops': {'neg', 'inv', 'sqrt', 'log', 'exp', 'sin', 'cos'},
            'calculus_ops': {'diff', 'int'}
        }

        # Discovered components
        self.variables: Dict[str, Variable] = {}
        self.learned_functions: Dict[str, Expression] = {}
        self.proven_lemmas: Dict[str, Expression] = {}

        # Configuration
        self.max_variables = max_variables
        self.noise_threshold = noise_threshold
        self.mdl_threshold = mdl_threshold

        # Components
        self.denoiser = NoisyObservationProcessor()
        self._expression_cache = {}

    def discover_variables(self,
                          observations: np.ndarray,
                          time_stamps: Optional[np.ndarray] = None) -> List[Variable]:
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

        # Check arity
        if operator in self.primitives['binary_ops']:
            if len(operands) != 2:
                return False
        elif operator in self.primitives['unary_ops']:
            if len(operands) != 1:
                return False
        elif operator in self.primitives['calculus_ops']:
            if len(operands) != 2:  # expression and variable
                return False
            # Second operand must be a variable for calculus ops
            if not isinstance(operands[1], Variable):
                return False

        # Type checking for operands
        for op in operands:
            if not isinstance(op, (Expression, Variable, int, float)):
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
