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
import random
import logging
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

    def clone(self) -> 'Expression':
        """Create a deep copy of this expression tree."""
        cloned_operands = []
        for op in self.operands:
            if isinstance(op, Expression):
                cloned_operands.append(op.clone())
            elif isinstance(op, Variable):
                # Variables are shared. If Variable instances could be mutated
                # in a way that cloning should prevent, then they would need
                # a .clone() method or deepcopy. For now, assume they are
                # effectively immutable or safe to share post-cloning.
                cloned_operands.append(op)
            else:
                # Basic types like int, float, str are immutable and can be copied directly.
                cloned_operands.append(op)

        # Create a new Expression instance.
        # The __post_init__ method will correctly set complexity and symbolic form.
        return Expression(operator=self.operator, operands=cloned_operands)

    def __hash__(self):
        # Hash based on operator and a tuple of operand hashes.
        # Operands can be Expression, Variable, or primitives.
        # Ensure operands are hashable or convert to a hashable representation (e.g., their own hash or a string key).

        # For primitive types in operands (like int, float, str), their natural hash is fine.
        # Variable class now has __hash__.
        # Expression class (this class) will have __hash__.

        # Create a tuple of hashes of operands.
        # This relies on operands themselves being hashable if they are custom objects.
        try:
            operand_hashes = []
            for op in self.operands:
                if isinstance(op, (Expression, Variable)): # If they have __hash__
                    operand_hashes.append(hash(op))
                elif isinstance(op, (int, float, str, bool, type(None))): # Standard hashable primitives
                    operand_hashes.append(hash(op))
                else: # Fallback for other types, could use str representation or raise error
                    operand_hashes.append(hash(str(op)))

            return hash((self.operator, tuple(operand_hashes)))
        except TypeError as e:
            # This might happen if an operand is an unhashable list/dict directly.
            # The Expression operands should ideally be Expression, Variable, or simple constants.
            # A more robust version might convert unhashable operands to a string representation.
            # For now, this relies on operands being one of the handled types.
            # print(f"Warning: TypeError during hashing Expression: {e}. Operands: {self.operands}")
            # Fallback hash if complex operands cause issues, this makes more expressions collide.
            return hash(self.operator)


@dataclass(eq=True, frozen=False) # Keep eq=True, frozen=False to allow properties to be mutable if needed
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

    # __eq__ is already provided by dataclass(eq=True) based on all fields.
    # If only name and index should define equality for caching purposes,
    # __eq__ would also need to be custom. For now, relying on default dataclass eq.

    @property
    def complexity(self) -> int:
        return 1


TargetType = TypeVar('TargetType')

@dataclass
class CFGRule(Generic[TargetType]):
    """Represents a single rule in a context-free grammar."""
    symbol: str
    expression: Union[str, List[Union[str, TargetType]]] # TargetType for direct objects/terminals
    weight: float = 1.0  # For weighted random choice

    def __post_init__(self):
        if self.weight <= 0:
            raise ValueError("Rule weight must be positive.")

class ContextFreeGrammar(Generic[TargetType]):
    """Represents a context-free grammar with support for weighted random generation."""
    # Rules are stored as a dictionary where keys are non-terminal symbols
    # and values are lists of CFGRule objects.
    rules: Dict[str, List[CFGRule[TargetType]]]

    def __init__(self, rules: Optional[List[CFGRule[TargetType]]] = None):
        self.rules = defaultdict(list)
        if rules:
            for rule in rules:
                self.add_rule(rule)

    def add_rule(self, rule: CFGRule[TargetType]):
        """Adds a rule to the grammar."""
        self.rules[rule.symbol].append(rule)

    def get_productions(self, symbol: str) -> List[CFGRule[TargetType]]:
        """Returns all production rules for a given symbol."""
        if symbol not in self.rules:
            raise ValueError(f"Symbol '{symbol}' not found in grammar rules.")
        return self.rules[symbol]

    def generate_random(self, start_symbol: str) -> List[Union[str, TargetType]]:
        """
        Generates a random sequence (string or list of items) from the grammar,
        respecting rule weights.
        Returns a list of terminal symbols or direct TargetType objects.
        """
        expansion_stack = [start_symbol]
        result_sequence = []

        max_depth = 100 # Protection against infinite recursion in cyclic grammars without proper terminal paths
        current_depth = 0

        while expansion_stack and current_depth < max_depth:
            current_symbol = expansion_stack.pop(0) # Process first-in (BFS-like for structure)

            if current_symbol not in self.rules:
                # If it's not a non-terminal, it's a terminal symbol or a direct object.
                result_sequence.append(current_symbol)
                continue

            productions = self.get_productions(current_symbol)
            if not productions:
                # This case should ideally not happen if grammar is well-formed
                # and all non-terminals have productions or lead to terminals.
                # Treating as a terminal if no productions found.
                result_sequence.append(current_symbol)
                continue

            # Weighted random choice of a rule for the current symbol
            total_weight = sum(rule.weight for rule in productions)
            chosen_weight = random.uniform(0, total_weight)
            cumulative_weight = 0
            chosen_rule = None
            for rule in productions:
                cumulative_weight += rule.weight
                if chosen_weight <= cumulative_weight:
                    chosen_rule = rule
                    break

            if chosen_rule is None: # Should not happen if productions list is not empty
                chosen_rule = productions[0]


            # The expression can be a list of symbols/objects or a single string (terminal)
            if isinstance(chosen_rule.expression, list):
                # Add symbols to the front of the stack to maintain order (if using pop(0))
                # or reverse and add to back (if using pop())
                expansion_stack = list(chosen_rule.expression) + expansion_stack
            else: # Single string or TargetType object
                expansion_stack.insert(0, chosen_rule.expression)

            current_depth +=1

        if current_depth >= max_depth:
            logging.warning(f"Max generation depth reached for start symbol '{start_symbol}'. Output may be incomplete.")

        return result_sequence


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
            for op_node in expr.operands: # Renamed op to op_node to avoid conflict
                if isinstance(op_node, Expression):
                    count += self._count_subexpression(op_node, pattern)

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
            for op_node in expr.operands: # Renamed op to op_node to avoid conflict
                if isinstance(op_node, Expression):
                    operand_keys.append(self._expression_key(op_node))
                elif isinstance(op_node, Variable): # Explicitly handle Variable in operands
                    operand_keys.append(f"var:{op_node.name}")
                elif isinstance(op_node, (int, float)): # Explicitly handle const in operands
                    operand_keys.append(f"const:{float(op_node):.6g}")
                else:
                    operand_keys.append(str(op_node)) # Fallback for other types

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

    def set_rules_from_cfg(self, rules: List[CFGRule[Union[str, Variable]]], start_symbol: str = "EXPR"):
        """
        Initializes or replaces the grammar's generation rules using a ContextFreeGrammar.
        The CFG can produce strings representing operators, variable names, or constant placeholders.
        It can also directly produce Variable objects.
        """
        self.cfg_grammar = ContextFreeGrammar[Union[str, Variable]](rules)
        self.cfg_start_symbol = start_symbol
        # Ensure that all variables and operators produced by the CFG are known
        # to the ProgressiveGrammar instance.
        # This might involve iterating through all possible productions, which can be complex.
        # A simpler approach is to validate symbols as they are used during generation.
        # For now, we assume operators are from existing primitives and variables are handled dynamically.
        logging.info(f"CFG rules set in ProgressiveGrammar with start symbol '{start_symbol}'.")


    def _generate_from_symbol_cfg(self, symbol: Union[str, Variable], max_depth: int, current_depth: int) -> Optional[Any]:
        """
        Recursive helper to generate part of an expression from a CFG symbol.
        Handles terminals (operators, variables, constants) and non-terminals.
        """
        if current_depth > max_depth:
            logging.warning(f"Max recursion depth {max_depth} exceeded in _generate_from_symbol_cfg for symbol '{symbol}'")
            # Attempt to return a terminal value if possible, like a variable or a default constant
            # This helps in gracefully degrading instead of outright failing.
            if isinstance(symbol, Variable):
                return symbol
            # Try to pick a random variable if symbol is a non-terminal that blew the stack
            if isinstance(symbol, str) and symbol.isupper(): # Convention for non-terminals
                if self.variables:
                    return random.choice(list(self.variables.values()))
            return None # Fallback, generation for this branch might fail

        if isinstance(symbol, Variable): # Terminal Variable object
            return symbol

        # If symbol is a string, it could be a non-terminal, an operator, a variable name, or a placeholder like 'CONST'
        if not isinstance(symbol, str):
            # This case should ideally not be reached if CFG rules are well-defined (string or Variable)
            logging.error(f"Unexpected symbol type in _generate_from_symbol_cfg: {type(symbol)}, value: {symbol}")
            return None

        # Check if the symbol is a non-terminal in the CFG
        if symbol in self.cfg_grammar.rules:
            # It's a non-terminal, expand it using the CFG.
            # The CFG's generate_random method returns a list of items (strings or Variables).
            # We need to process this list to form a structured Expression.
            # This is the tricky part: how do we interpret the flat list from CFG into a tree?
            # The current CFG generates a flat list. We need a CFG designed to produce Prefix/Polish notation.
            # Example: Rule 'EXPR' -> ['BINARY_OP', 'EXPR', 'EXPR']
            #          Rule 'BINARY_OP' -> ['+', '-', '*', '/'] (terminals for operators)
            #          Rule 'EXPR' -> ['VAR', 'CONST'] (terminals for operands)

            # Let's assume the CFG is structured to produce a sequence that can be parsed into an expression.
            # For instance, a prefix (Polish) notation string: [operator, operand1, operand2]
            generated_sequence = self.cfg_grammar.generate_random(symbol) # This returns a list

            if not generated_sequence:
                logging.warning(f"CFG generated an empty sequence for non-terminal '{symbol}'")
                return None

            # The first element of the sequence is treated as the operator
            op_name = generated_sequence[0]
            if not isinstance(op_name, str) or not self.is_operator_known(op_name):
                 # If op_name is a Variable or other non-string, it cannot be an operator.
                 # Or, if it's a string but not a recognized operator (e.g. a non-terminal like "VAR" or "CONST")
                 # This indicates the sequence is not an operator-led expression part.
                 # It might be a single terminal like a variable name, 'CONST', or a Variable object.
                 # We recursively call _generate_from_symbol_cfg on this single item.
                 # This handles cases like 'EXPR' -> 'VAR' or 'EXPR' -> Variable_object
                 if len(generated_sequence) == 1:
                    return self._generate_from_symbol_cfg(op_name, max_depth, current_depth + 1)
                 else:
                    # This case is problematic: sequence starts with non-operator but has multiple items.
                    # e.g. ['VAR', 'something_else'] from a rule for 'EXPR'.
                    # This implies the CFG structure is not directly mapping to Expression trees as expected.
                    logging.warning(f"CFG generated sequence for '{symbol}' starting with non-operator '{op_name}' and multiple items. Sequence: {generated_sequence}. Treating as error or trying first element.")
                    # We could try to salvage by processing the first element only, if it's a valid symbol.
                    return self._generate_from_symbol_cfg(op_name, max_depth, current_depth + 1)


            arity = self.get_arity(op_name)
            operands = []

            # The rest of the sequence are operands for this operator
            # We need to consume the correct number of elements from generated_sequence for these operands.
            # This part is complex because generated_sequence is flat.
            # The CFG needs to be designed such that its output list can be correctly parsed.
            # A simple flat list is not enough for nested structures if we just iterate.
            # E.g., if EXPR -> [OP, EXPR, EXPR], and then EXPR -> [VAR],
            # generate_random('EXPR') could give [OP, VAR, VAR] - flat, easy.
            # But if it gives [OP, [OP2, VAR, VAR], VAR], our current CFG.generate_random doesn't do that.
            # It gives: [OP, OP2, VAR, VAR, VAR] if rules are EXPR -> [OP, EXPR, EXPR]; EXPR -> [OP2, EXPR, EXPR]; EXPR -> [VAR]
            # This requires a parser for the generated sequence.

            # For simplicity, let's assume the CFG is designed such that for an operator,
            # the subsequent elements in the generated list are its direct arguments (non-terminals or terminals).
            # This means the CFG itself must ensure the correct structure for direct parsing.
            # E.g., 'EXPR' -> ['BINARY_OP', 'ARG', 'ARG'], 'UNARY_OP' -> ['UNARY_OP_TYPE', 'ARG']
            # 'ARG' -> ['VAR', 'CONST', 'EXPR'] (EXPR here means it will be another op-led sequence)

            # Let's adjust the interpretation: the `generated_sequence` from `cfg_grammar.generate_random(symbol)`
            # is for the current `symbol`. If `symbol` is 'EXPR', it might expand to `['*', 'VAR1', 'VAR2']`.
            # The `op_name` is '*' (generated_sequence[0]).
            # The operands are 'VAR1', 'VAR2' (generated_sequence[1:]).

            operand_symbols = generated_sequence[1:]

            if len(operand_symbols) != arity:
                logging.warning(f"Arity mismatch for operator '{op_name}'. Expected {arity}, got {len(operand_symbols)} symbols: {operand_symbols}. CFG rule for '{symbol}' might be ill-defined for this operator.")
                # Attempt to use available operands, or fill with defaults, or fail.
                # For now, let's try to proceed if too many operands, or fail if too few.
                if len(operand_symbols) < arity:
                    return None # Cannot satisfy arity
                operand_symbols = operand_symbols[:arity] # Truncate if too many

            for i in range(arity):
                operand_symbol = operand_symbols[i]
                # Recursively generate expression for each operand symbol
                operand_expr = self._generate_from_symbol_cfg(operand_symbol, max_depth, current_depth + 1)
                if operand_expr is None:
                    logging.warning(f"Failed to generate operand {i} for operator '{op_name}' from symbol '{operand_symbol}'.")
                    return None  # Failure to construct an operand
                operands.append(operand_expr)

            # Successfully built operands, now create the expression
            return self.create_expression(op_name, operands, validate=True)

        # Symbol is a terminal string from CFG (not a non-terminal)
        # It could be an operator name (e.g. '+'), a variable name (e.g. 'x_1'), or 'CONST'.
        elif symbol in self.primitives['binary_ops'] or \
             symbol in self.primitives['unary_ops'] or \
             symbol in self.primitives['calculus_ops']:
            # This case should be handled if CFG produces operator as part of a sequence,
            # as processed above. If a bare operator string is produced for a symbol that
            # was expected to be an operand, it's an issue.
            # However, if the CFG rule was like 'OP_TYPE' -> '+', then this is just returning the operator name.
            return symbol # Return the operator name as a string

        elif symbol in self.variables: # Terminal: existing variable name
            return self.variables[symbol]

        elif symbol == 'CONST': # Placeholder for a constant
            # Generate a random constant or use a default one.
            # For now, let's pick from a predefined set or generate a simple one.
            # This could be made more sophisticated (e.g., range, type).
            return random.choice([0, 1, -1, np.pi, np.e] + [round(random.uniform(-2,2),2)]) # Returns a number

        elif symbol.startswith("var_") or symbol in [v.name for v in self.variables.values()]: # Generic variable name from CFG
            # This allows CFG to specify variable names like "var_generic" or existing ones.
            # If it's an existing variable name, return the Variable object.
            if symbol in self.variables:
                return self.variables[symbol]
            else:
                # If CFG produced a new variable name not yet in self.variables,
                # this implies the CFG is suggesting a new variable.
                # For robust generation, we should probably ensure such variables are known
                # or handle their creation. For now, if not found, this is an issue.
                # Fallback: pick a random known variable if available.
                logging.warning(f"CFG produced an unknown variable name '{symbol}'. Using a random known variable if possible.")
                if self.variables:
                    return random.choice(list(self.variables.values()))
                else: # No variables known, cannot satisfy this.
                    logging.error(f"Cannot create expression from unknown variable '{symbol}' when no variables are discovered.")
                    return None

        else:
            # Symbol is a string but not a non-terminal, not an operator, not 'CONST', not a known variable.
            # It might be a newly proposed variable name by the CFG.
            # Or it could be an error / misconfiguration.
            # Example: CFG rule 'EXPR' -> 'my_special_undefined_symbol'
            logging.warning(f"Treating unknown terminal symbol '{symbol}' from CFG as a potential new variable name or error.")
            # Option 1: Try to treat as a new variable if it fits a naming convention.
            # Option 2: Pick a random existing variable.
            # Option 3: Fail.
            # For now, let's try Option 2 for robustness.
            if self.variables:
                # This might not be what the CFG intended, but prevents immediate failure.
                return random.choice(list(self.variables.values()))
            else:
                logging.error(f"Unknown terminal symbol '{symbol}' from CFG and no variables available to substitute.")
                return None


    def generate_random_expression_from_cfg(self, start_symbol: Optional[str] = None, max_depth: int = 10) -> Optional[Expression]:
        """
        Generates a random mathematical expression tree using the configured ContextFreeGrammar.
        The CFG should be designed to produce sequences that can be interpreted as expressions,
        typically in a prefix (Polish) notation.
        'start_symbol' defaults to self.cfg_start_symbol if not provided.
        """
        if not hasattr(self, 'cfg_grammar') or self.cfg_grammar is None:
            logging.error("ContextFreeGrammar (self.cfg_grammar) is not initialized. Call set_rules_from_cfg first.")
            return None

        current_start_symbol = start_symbol if start_symbol else self.cfg_start_symbol
        if not current_start_symbol:
            logging.error("No start symbol provided for CFG generation and no default is set.")
            return None

        # The _generate_from_symbol_cfg method is responsible for interpreting the CFG output.
        # It expects the CFG to guide the construction of an Expression tree.
        generated_expr_component = self._generate_from_symbol_cfg(current_start_symbol, max_depth, 0)

        if isinstance(generated_expr_component, Expression):
            return generated_expr_component
        elif generated_expr_component is None:
            logging.error(f"Failed to generate a complete expression from CFG start symbol '{current_start_symbol}'.")
            return None
        else:
            # If _generate_from_symbol_cfg returns something that is not an Expression
            # (e.g., a single Variable object or a constant value if the CFG directly produces them
            # from the start symbol without an operator), we need to wrap it appropriately
            # or decide if this is a valid outcome.
            # For example, if CFG is `S -> VAR` and `VAR -> x`, it might return Variable('x').
            # This is a valid expression component but not a complex Expression object.
            # We can wrap it in a 'var' or 'const' Expression.
            if isinstance(generated_expr_component, Variable):
                return self.create_expression('var', [generated_expr_component.name], validate=False) # Assume var name is valid
            elif isinstance(generated_expr_component, (int, float)):
                return self.create_expression('const', [generated_expr_component], validate=False)
            else:
                logging.error(f"CFG generation resulted in an unexpected type: {type(generated_expr_component)} for symbol '{current_start_symbol}'. Expected Expression, Variable, or constant.")
                return None


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
                for op_node in expr.operands: # Renamed op to op_node to avoid conflict
                    if isinstance(op_node, Expression):
                        result.extend(
                            self._extract_all_subexpressions(op_node, collected)
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
                self._expression_to_dict(op_node) if isinstance(op_node, Expression) # Renamed op to op_node
                else {'type': 'var', 'name': op_node.name} if isinstance(op_node, Variable) # Renamed op to op_node
                else {'type': 'const', 'value': op_node} # Renamed op to op_node
                for op_node in expr.operands # Renamed op to op_node
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
        self.add_primitive('embedding', self._embedding_op)

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

    # The __init__ method is preserved above this point.
    # Adding _attention_op as per the subtask description.
    # If a specific implementation from "the issue" was intended,
    # this placeholder version should be replaced with that.
    def _attention_op(self, query: Any, key: Any, value: Any) -> Any:
        """
        Implements attention mechanism operation for symbolic representation.

        For symbolic mode: Returns a SymPy function representation
        For numeric mode: Computes actual attention scores
        """
        # Check if we're in symbolic or numeric mode
        if isinstance(query, (sp.Symbol, sp.Expr)):
            # Symbolic mode - return a symbolic representation
            return sp.Function('Attention')(query, key, value)

        # Numeric mode - compute actual attention
        if isinstance(query, np.ndarray):
            # Convert to torch tensors if needed
            q = torch.tensor(query, dtype=torch.float32) if not isinstance(query, torch.Tensor) else query
            k = torch.tensor(key, dtype=torch.float32) if not isinstance(key, torch.Tensor) else key
            v = torch.tensor(value, dtype=torch.float32) if not isinstance(value, torch.Tensor) else value

            # Compute attention scores: softmax(Q @ K^T / sqrt(d_k)) @ V
            d_k = q.shape[-1]
            scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(d_k)
            attention_weights = torch.softmax(scores, dim=-1)
            output = torch.matmul(attention_weights, v)

            return output.numpy() if isinstance(query, np.ndarray) else output

        # Fallback for other types
        return f"Attention({query}, {key}, {value})"

    def _embedding_op(self, indices: Any, embedding_matrix: Any) -> Any:
        """
        Implements embedding lookup operation.

        For symbolic mode: Returns a SymPy function representation
        For numeric mode: Performs actual embedding lookup
        """
        # Symbolic mode
        if isinstance(indices, (sp.Symbol, sp.Expr)):
            return sp.Function('Embedding')(indices, embedding_matrix)

        # Numeric mode
        if isinstance(indices, (np.ndarray, list)):
            indices = np.array(indices, dtype=int)

            if isinstance(embedding_matrix, np.ndarray):
                # Perform embedding lookup
                return embedding_matrix[indices]
            elif isinstance(embedding_matrix, torch.Tensor):
                # Handle PyTorch tensors
                indices_tensor = torch.tensor(indices, dtype=torch.long)
                return embedding_matrix[indices_tensor].numpy()
            elif isinstance(embedding_matrix, str):
                # Symbolic reference to embedding matrix
                return f"Embedding({indices}, {embedding_matrix})"

        # Fallback
        return f"EmbeddingLookup({indices}, {embedding_matrix})"

    def _to_sympy(self, expr_node: Expression) -> sp.Expr:
        # ... (existing implementation for other custom callable ops)
        if expr_node.operator in self.primitives.get('custom_callable_ops', {}):
            func_name = expr_node.operator
            sympy_operands = []
            for op_node in expr_node.operands:
                if isinstance(op_node, Expression):
                    sympy_operands.append(self._to_sympy(op_node))
                elif isinstance(op_node, Variable):
                    sympy_operands.append(op_node.symbolic)
                elif isinstance(op_node, (int, float)):
                    sympy_operands.append(sp.Number(op_node))
                elif isinstance(op_node, str):
                    sympy_operands.append(sp.Symbol(op_node))
                else:
                    sympy_operands.append(sp.Symbol(str(op_node)))

            if func_name == 'if_then_else' and len(sympy_operands) == 3:
                return sp.Function(func_name)(*sympy_operands)
            elif func_name == 'threshold' and len(sympy_operands) == 2:
                return sympy_operands[0] > sympy_operands[1]

            # Add handling for attention and embedding
            if func_name == 'attention':
                return sp.Function('Attention')(*sympy_operands)
            # func_name will be 'embedding' as per the new requirement
            if func_name == 'embedding':
                return sp.Function('Embedding')(*sympy_operands)

            return sp.Function(func_name)(*sympy_operands)

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
    # print("\nTesting validation for AIGrammar:")
    # valid_threshold = ai_grammar._validate_expression('threshold', [feature_x, 0.5])
    # print(f"Validation for 'threshold' (2 operands): {valid_threshold}")
    # invalid_threshold = ai_grammar._validate_expression('threshold', [feature_x])
    # print(f"Validation for 'threshold' (1 operand): {invalid_threshold}")
    #
    # valid_attention = ai_grammar._validate_expression('attention', [q_var, k_var, v_var])
    # print(f"Validation for 'attention' (3 operands): {valid_attention}")
    #
    # # Example of creating an expression that might use a traditional op via superclass validation
    # # This assumes 'relu' was added to unary_ops for AIGrammar for this test.
    # # relu_expr_ai = ai_grammar.create_expression('relu', [feature_x])
    # # if relu_expr_ai:
    # #    print(f"AI Relu Expr: {relu_expr_ai.operator}({feature_x.name})")
    # #    # print(f"Symbolic (should work if relu is standard unary): {relu_expr_ai.symbolic}")
    # # else:
    # #    print("Failed to create AI relu expression.")

    # The main challenge remains: Expression class is independent of grammar object at instance level.
    # Making Expression operations (like .symbolic, .complexity, validation) grammar-aware
    # is a deeper refactoring. This AIGrammar class adds the *definitions* and *policies*
    # for AI primitives.

    print("\n--- ProgressiveGrammar CFG Generation Example ---")
    # Assumes 'grammar' is the ProgressiveGrammar instance from the earlier part of __main__
    # Ensure there are some variables in the grammar, e.g., from discover_variables or added manually
    if not grammar.variables:
        print("No variables in grammar for CFG example. Adding dummy variables.")
        grammar.variables['v1'] = Variable(name='v1', index=0)
        grammar.variables['v2'] = Variable(name='v2', index=1)
        grammar.variables['v3'] = Variable(name='v3', index=2)

    # Define CFG rules for expression generation
    # These rules should produce sequences that _generate_from_symbol_cfg can parse.
    # Specifically, an operator followed by symbols for its operands.
    # Terminal symbols for CFG: variable names (str), 'CONST' (str), operator names (str).
    # Non-terminal symbols for CFG: 'EXPR', 'BINARY_OP', 'UNARY_OP', 'VAR', 'CONST_SYMBOL' (or just use 'CONST')

    # Get available variable names and operator names from the grammar instance
    available_vars = list(grammar.variables.keys())
    if not available_vars:
        available_vars = ["default_var"] # Fallback if no variables discovered
        grammar.variables["default_var"] = Variable(name="default_var", index=0)


    binary_ops = list(grammar.primitives['binary_ops'])
    unary_ops = list(grammar.primitives['unary_ops'])

    cfg_rules = [
        # Core expression rule: can be a binary op, unary op, variable, or constant
        CFGRule('EXPR', ['BINARY_OP_EXPR'], weight=0.4),
        CFGRule('EXPR', ['UNARY_OP_EXPR'], weight=0.3),
        CFGRule('EXPR', ['VAR'], weight=0.2),
        CFGRule('EXPR', ['CONST'], weight=0.1),

        # Productions for binary operations
        # Each rule produces: [operator_str, operand_symbol_1, operand_symbol_2]
        # Operand symbols ('EXPR' here) will be recursively expanded.
    ]
    for op in binary_ops:
        cfg_rules.append(CFGRule('BINARY_OP_EXPR', [op, 'EXPR', 'EXPR']))

    # Productions for unary operations
    # Each rule produces: [operator_str, operand_symbol]
    for op in unary_ops:
        cfg_rules.append(CFGRule('UNARY_OP_EXPR', [op, 'EXPR']))

    # Productions for variables (VAR non-terminal leads to a specific variable name)
    for var_name in available_vars:
         # CFG can directly produce the variable name string
        cfg_rules.append(CFGRule('VAR', [var_name], weight=1.0/len(available_vars)))

    # Production for constants (CONST non-terminal leads to 'CONST' placeholder string)
    # The 'CONST' string is then handled by _generate_from_symbol_cfg to produce a number.
    cfg_rules.append(CFGRule('CONST', ['CONST']))

    # Set the CFG rules in the grammar
    grammar.set_rules_from_cfg(cfg_rules, start_symbol='EXPR')
    print("CFG rules set for ProgressiveGrammar.")

    # Generate a random expression using the CFG
    print("\nGenerating random expression using CFG...")
    # Configure logging to see warnings from generation process
    logging.basicConfig(level=logging.INFO) # Show info, warnings, errors

    generated_expression = grammar.generate_random_expression_from_cfg(max_depth=5)

    if generated_expression:
        print(f"\nSuccessfully generated expression from CFG:")
        print(f"  Symbolic: {generated_expression.symbolic}")
        print(f"  Complexity: {generated_expression.complexity}")
        print(f"  Operator: {generated_expression.operator}")
        print(f"  Operands: {[op.symbolic if isinstance(op, Expression) else op.name if isinstance(op, Variable) else op for op in generated_expression.operands]}")
    else:
        print("\nFailed to generate an expression using CFG. Check logs for details.")

    # Example of generating multiple expressions
    print("\nGenerating a few more examples:")
    for i in range(3):
        expr = grammar.generate_random_expression_from_cfg(max_depth=4)
        if expr:
            print(f"  Example {i+1}: {expr.symbolic} (Complexity: {expr.complexity})")
        else:
            print(f"  Example {i+1}: Failed to generate.")


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
