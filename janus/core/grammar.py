"""
Progressive Grammar System for Autonomous Physics Discovery
==========================================================

A hierarchical grammar that discovers variables from observations and
progressively builds mathematical abstractions using information-theoretic principles.
"""

import numpy as np
import sympy as sp
from typing import Dict, List, Tuple, Optional, Set, Any, TypeVar, Generic, Union
from dataclasses import dataclass # field removed as it was for Expression/Variable
from collections import defaultdict
import random
import logging
import torch
import torch.nn as nn
from scipy.stats import entropy
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler
import warnings

from .expression import Expression, Variable

warnings.filterwarnings('ignore')


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
                result_sequence.append(current_symbol)
                continue

            total_weight = sum(rule.weight for rule in productions)
            chosen_weight = random.uniform(0, total_weight)
            cumulative_weight = 0
            chosen_rule = None
            for rule in productions:
                cumulative_weight += rule.weight
                if chosen_weight <= cumulative_weight:
                    chosen_rule = rule
                    break

            if chosen_rule is None: chosen_rule = productions[0]


            if isinstance(chosen_rule.expression, list):
                expansion_stack = list(chosen_rule.expression) + expansion_stack
            else:
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
        self.model = None # Initialize model attribute

    def build_autoencoder(self, input_dim: int):
        """Build denoising autoencoder for preprocessing."""
        class DenoisingAutoencoder(nn.Module):
            def __init__(self, input_dim_ae, latent_dim_ae): # Renamed to avoid conflict
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim_ae, 128),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, latent_dim_ae)
                )
                self.decoder = nn.Sequential(
                    nn.Linear(latent_dim_ae, 64),
                    nn.ReLU(),
                    nn.Linear(64, 128),
                    nn.ReLU(),
                    nn.Linear(128, input_dim_ae)
                )

            def forward(self, x, noise_level=0.1):
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
            return self._simple_denoise(observations)

        observations_scaled = self.scaler.fit_transform(observations)
        data = torch.FloatTensor(observations_scaled)

        if self.model is None or self.model.encoder[0].in_features != data.shape[1]: # Check if model needs rebuild
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

        self.model.eval()
        with torch.no_grad():
            denoised, _ = self.model(data, noise_level=0)

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
    COMMUTATIVE_OPS = {'+', '*'}

    def __init__(self,
                 max_variables: int = 20,
                 noise_threshold: float = 0.1,
                 mdl_threshold: float = 10.0,
                 load_defaults: bool = True):

        self.primitives = {
            'constants': {}, 'binary_ops': set(),
            'unary_ops': set(), 'calculus_ops': set()
        }

        if load_defaults:
            self.primitives['constants'] = {'0': 0, '1': 1, 'pi': np.pi, 'e': np.e}
            self.primitives['binary_ops'] = {'+', '-', '*', '/', '**'}
            self.primitives['unary_ops'] = {'neg', 'inv', 'sqrt', 'log', 'exp', 'sin', 'cos'}
            self.primitives['calculus_ops'] = {'diff', 'int'}

        self.variables: Dict[str, Variable] = {}
        self.learned_functions: Dict[str, Expression] = {}
        self.proven_lemmas: Dict[str, Expression] = {}
        self.max_variables = max_variables
        self.noise_threshold = noise_threshold
        self.mdl_threshold = mdl_threshold
        self.denoiser = NoisyObservationProcessor()
        self._expression_cache = {}
        self.cfg_grammar: Optional[ContextFreeGrammar[Union[str, Variable]]] = None
        self.cfg_start_symbol: Optional[str] = None


    def add_operators(self, operators: List[str]):
        known_binary = {'+', '-', '*', '/', '**'}
        known_unary = {'neg', 'inv', 'sqrt', 'log', 'exp', 'sin', 'cos'}
        known_calculus = {'diff', 'int'}
        for op in operators:
            if op in known_binary: self.primitives['binary_ops'].add(op)
            elif op in known_unary: self.primitives['unary_ops'].add(op)
            elif op in known_calculus: self.primitives['calculus_ops'].add(op)
            elif op == '**2' or op == '**3': self.primitives['binary_ops'].add('**')
            elif op == '1/': self.primitives['unary_ops'].add('inv')
            else: print(f"Warning: Operator '{op}' has unknown arity and was not added.")

    def discover_variables(self,
                          observations: np.ndarray,
                          time_stamps: Optional[np.ndarray] = None) -> List[Variable]:
        clean_obs = self.denoiser.denoise(observations)
        num_components = min(self.max_variables, clean_obs.shape[1])
        if num_components <= 0: # Handle case with no features or no max_vars
             return []
        ica = FastICA(n_components=num_components, random_state=0) # Added random_state for reproducibility
        try:
            components = ica.fit_transform(clean_obs)
        except ValueError as e:
            # This can happen if all components are flat after denoising, or other ICA issues.
            logging.warning(f"ICA failed during variable discovery: {e}. Returning no variables.")
            return []


        discovered_vars = []
        for i in range(components.shape[1]):
            component = components[:, i]
            properties = self._analyze_component(component, time_stamps)
            if properties['information_content'] > self.noise_threshold:
                var_name = self._generate_variable_name(properties)
                var = Variable(name=var_name, index=i, properties=properties)
                discovered_vars.append(var)
                self.variables[var_name] = var
        return discovered_vars

    def _analyze_component(self,
                          component: np.ndarray,
                          time_stamps: Optional[np.ndarray]) -> Dict[str, float]:
        properties = {}
        hist, _ = np.histogram(component, bins=50)
        hist = hist + 1e-10
        properties['information_content'] = entropy(hist) / np.log(len(hist)) if len(hist) > 1 else 0.0

        if len(component) > 10:
            num_splits = min(10, len(component)//10)
            if num_splits > 0: # Ensure num_splits is at least 1 for array_split
                windows = np.array_split(component, num_splits)
                variances = [np.var(w) for w in windows if len(w) > 0] # Ensure window is not empty
                properties['conservation_score'] = 1.0 / (1.0 + np.var(variances)) if variances else 1.0
            else: # Not enough data for splitting
                 properties['conservation_score'] = 0.0
        else:
            properties['conservation_score'] = 0.0

        if len(component) > 20:
            fft_abs = np.abs(np.fft.fft(component - np.mean(component)))
            power = fft_abs[:len(fft_abs)//2]
            if len(power) > 1: # Need at least one non-DC component
                peak_power = np.max(power[1:])
                avg_power = np.mean(power[1:])
                properties['periodicity_score'] = peak_power / (avg_power + 1e-10)
            else: # Only DC component or less
                properties['periodicity_score'] = 0.0
        else:
            properties['periodicity_score'] = 0.0

        if len(component) > 2:
            properties['smoothness'] = 1.0 / (1.0 + np.std(np.diff(component)))
        else:
            properties['smoothness'] = 0.0
        return properties

    def _generate_variable_name(self, properties: Dict[str, float]) -> str:
        if properties.get('conservation_score', 0) > 0.8: prefix = 'E'
        elif properties.get('periodicity_score', 0) > 5.0: prefix = 'theta'
        elif properties.get('smoothness', 0) > 0.7: prefix = 'x'
        else: prefix = 'q'
        existing = [v for v in self.variables if v.startswith(prefix)]
        return f"{prefix}_{len(existing) + 1}"

    def create_expression(self,
                         operator: str,
                         operands: List[Any],
                         validate: bool = True) -> Optional[Expression]:
        if validate and not self._validate_expression(operator, operands):
            return None
        # Expression class is now imported
        expr = Expression(operator, operands)
        expr_key = self._expression_key(expr) # Uses self._expression_key
        self._expression_cache[expr_key] = expr
        return expr

    def _validate_expression(self, operator: str, operands: List[Any]) -> bool:
        all_ops = (self.primitives['binary_ops'] |
                  self.primitives['unary_ops'] |
                  self.primitives['calculus_ops'] |
                  set(self.learned_functions.keys()) | # Learned functions can be ops
                  {'var', 'const'})
        if operator not in all_ops: return False

        # Arity checks
        expected_arity = -1
        if operator in self.primitives['binary_ops']: expected_arity = 2
        elif operator in self.primitives['unary_ops']: expected_arity = 1
        elif operator in self.primitives['calculus_ops']: expected_arity = 2
        elif operator == 'var': expected_arity = 1
        elif operator == 'const': expected_arity = 1
        elif operator in self.learned_functions: # Assuming learned funcs are unary for now
            expected_arity = 1 # Or store arity with learned functions

        if expected_arity != -1 and len(operands) != expected_arity: return False

        # Type checks
        for op_val in operands:
            if not isinstance(op_val, (Expression, Variable, int, float, str)): # Allow str for var names
                return False
        if operator == 'var' and not isinstance(operands[0], str): return False
        if operator == 'const' and not isinstance(operands[0], (int, float)): return False
        if operator in self.primitives['calculus_ops'] and not isinstance(operands[1], Variable): return False

        return True

    def add_learned_function(self,
                           name: str,
                           expression: Expression,
                           usage_data: List[Expression]) -> bool:
        compression_gain = self._calculate_compression_gain(expression, usage_data)
        if compression_gain > self.mdl_threshold:
            self.learned_functions[name] = expression
            self.primitives['unary_ops'].add(name) # Assuming new functions are unary
            return True
        return False

    def _calculate_compression_gain(self,
                                  candidate: Expression,
                                  corpus: List[Expression]) -> float:
        current_length = sum(expr.complexity for expr in corpus)
        new_length = candidate.complexity
        for expr in corpus:
            occurrences = self._count_subexpression(expr, candidate)
            if occurrences > 0:
                saved = occurrences * (candidate.complexity - 1)
                new_length += expr.complexity - saved
            else:
                new_length += expr.complexity
        return current_length - new_length

    def _count_subexpression(self,
                           expr: Expression,
                           pattern: Expression) -> int:
        if self._expression_key(expr) == self._expression_key(pattern):
            return 1
        count = 0
        if hasattr(expr, 'operands'): # Check if it's an Expression-like object
            for op_node in expr.operands:
                if isinstance(op_node, Expression):
                    count += self._count_subexpression(op_node, pattern)
        return count

    def _expression_key(self, expr: Any) -> str: # Changed type hint for expr
        if isinstance(expr, Variable):
            return f"var:{expr.name}"
        elif isinstance(expr, (int, float)):
            return f"const:{float(expr):.6g}"
        elif isinstance(expr, str): # Should be a variable name string
            return f"var:{expr}"
        elif isinstance(expr, Expression): # Check if it's an Expression object
            operand_keys = []
            for op_node in expr.operands:
                operand_keys.append(self._expression_key(op_node)) # Recursive call
            if expr.operator in self.COMMUTATIVE_OPS:
                operand_keys.sort()
            return f"{expr.operator}({','.join(operand_keys)})"
        else: # Fallback for unknown types
            return str(expr)


    def mine_abstractions(self,
                         hypothesis_library: List[Expression],
                         min_frequency: int = 3) -> Dict[str, Expression]:
        pattern_counts = defaultdict(int)
        pattern_examples = defaultdict(list)
        for hypothesis in hypothesis_library:
            subexprs = self._extract_all_subexpressions(hypothesis)
            for subexpr in subexprs:
                if subexpr.complexity > 2:
                    key = self._expression_key(subexpr)
                    pattern_counts[key] += 1
                    pattern_examples[key].append(subexpr)
        abstractions = {}
        for pattern_key, count in pattern_counts.items():
            if count >= min_frequency:
                example = pattern_examples[pattern_key][0]
                name = f"f_{len(self.learned_functions)}"
                if self.add_learned_function(name, example, hypothesis_library):
                    abstractions[name] = example
        return abstractions

    def set_rules_from_cfg(self, rules: List[CFGRule[Union[str, Variable]]], start_symbol: str = "EXPR"):
        self.cfg_grammar = ContextFreeGrammar[Union[str, Variable]](rules)
        self.cfg_start_symbol = start_symbol
        logging.info(f"CFG rules set in ProgressiveGrammar with start symbol '{start_symbol}'.")

    def _generate_from_symbol_cfg(self, symbol: Union[str, Variable], max_depth: int, current_depth: int) -> Optional[Any]:
        if current_depth > max_depth:
            logging.warning(f"Max recursion depth {max_depth} exceeded for '{symbol}'")
            if isinstance(symbol, Variable): return symbol
            if isinstance(symbol, str) and symbol.isupper() and self.variables:
                return random.choice(list(self.variables.values()))
            return None

        if isinstance(symbol, Variable): return symbol
        if not isinstance(symbol, str):
            logging.error(f"Unexpected symbol type: {type(symbol)}, value: {symbol}")
            return None

        if self.cfg_grammar and symbol in self.cfg_grammar.rules: # Check cfg_grammar exists
            generated_sequence = self.cfg_grammar.generate_random(symbol)
            if not generated_sequence:
                logging.warning(f"CFG empty sequence for '{symbol}'")
                return None

            op_name_or_terminal = generated_sequence[0]
            if len(generated_sequence) == 1: # Terminal or non-terminal that expands to one item
                return self._generate_from_symbol_cfg(op_name_or_terminal, max_depth, current_depth +1) # Recursively resolve it

            # If sequence starts with a known operator
            if isinstance(op_name_or_terminal, str) and self.is_operator_known(op_name_or_terminal):
                op_name = op_name_or_terminal
                arity = self.get_arity(op_name)
                operand_symbols = generated_sequence[1:]
                if len(operand_symbols) != arity:
                    logging.warning(f"Arity mismatch for '{op_name}'. Exp {arity}, got {len(operand_symbols)}.")
                    if len(operand_symbols) < arity: return None
                    operand_symbols = operand_symbols[:arity] # Truncate

                operands = []
                for i in range(arity):
                    operand_expr = self._generate_from_symbol_cfg(operand_symbols[i], max_depth, current_depth + 1)
                    if operand_expr is None:
                        logging.warning(f"Failed to gen operand {i} for '{op_name}' from '{operand_symbols[i]}'.")
                        return None
                    operands.append(operand_expr)
                return self.create_expression(op_name, operands, validate=True) # Create expression
            else: # Sequence does not start with a known operator, treat first element as a terminal/symbol
                 return self._generate_from_symbol_cfg(op_name_or_terminal, max_depth, current_depth+1)


        elif symbol in self.variables: return self.variables[symbol]
        elif symbol == 'CONST': return random.choice([0, 1, -1, round(random.uniform(-2,2),2)])
        elif symbol.startswith("var_") or symbol in [v.name for v in self.variables.values()]: # Generic var from CFG
            if symbol in self.variables: return self.variables[symbol]
            logging.warning(f"CFG unknown var name '{symbol}'. Using random known var.")
            if self.variables: return random.choice(list(self.variables.values()))
            return None
        else: # Unknown terminal from CFG
            logging.warning(f"Unknown terminal symbol '{symbol}' from CFG. Using random known var.")
            if self.variables: return random.choice(list(self.variables.values()))
            return None


    def generate_random_expression_from_cfg(self, start_symbol: Optional[str] = None, max_depth: int = 10) -> Optional[Expression]:
        if not hasattr(self, 'cfg_grammar') or self.cfg_grammar is None:
            logging.error("CFG not initialized. Call set_rules_from_cfg.")
            return None
        current_start_symbol = start_symbol if start_symbol else self.cfg_start_symbol
        if not current_start_symbol:
            logging.error("No start symbol for CFG generation.")
            return None

        generated_component = self._generate_from_symbol_cfg(current_start_symbol, max_depth, 0)
        if isinstance(generated_component, Expression): return generated_component
        if generated_component is None: return None

        # Wrap non-Expression results (Variable, const) into an Expression
        if isinstance(generated_component, Variable):
            # Use create_expression to ensure it's handled like other 'var' expressions
            return self.create_expression('var', [generated_component.name], validate=False)
        elif isinstance(generated_component, (int, float)):
            return self.create_expression('const', [generated_component], validate=False)
        else:
            logging.error(f"CFG gen resulted in unexpected type: {type(generated_component)}")
            return None


    def _extract_all_subexpressions(self,
                                   expr: Expression,
                                   collected: Optional[Set[str]] = None) -> List[Expression]: # Changed collected type hint
        if collected is None: collected = set()
        result = []
        key = self._expression_key(expr)
        if key not in collected:
            collected.add(key)
            result.append(expr)
            if hasattr(expr, 'operands'):
                for op_node in expr.operands:
                    if isinstance(op_node, Expression):
                        result.extend(self._extract_all_subexpressions(op_node, collected))
        return result

    def export_grammar_state(self) -> Dict:
        return {
            'variables': { name: { 'index': var.index, 'properties': var.properties }
                           for name, var in self.variables.items() },
            'learned_functions': { name: self._expression_to_dict(expr)
                                  for name, expr in self.learned_functions.items() },
            'proven_lemmas': { name: self._expression_to_dict(expr)
                              for name, expr in self.proven_lemmas.items() }
        }

    def _expression_to_dict(self, expr: Expression) -> Dict:
        return {
            'operator': expr.operator,
            'operands': [ self._expression_to_dict(op) if isinstance(op, Expression)
                         else {'type': 'var', 'name': op.name} if isinstance(op, Variable)
                         else {'type': 'const', 'value': op}
                         for op in expr.operands ],
            'complexity': expr.complexity
        }


class AIGrammar(ProgressiveGrammar):
    def __init__(self):
        super().__init__(load_defaults=False)
        self.add_primitive_set('activation_types', ['relu', 'sigmoid', 'tanh', 'gelu'])
        self.add_primitive('attention', self._attention_op, category='custom_callable_ops')
        self.add_primitive('embedding_lookup', self._embedding_op, category='custom_callable_ops') # Renamed for clarity
        self.add_primitive('if_then_else', lambda cond, true_val, false_val: true_val if cond else false_val, category='custom_callable_ops')
        self.add_primitive('threshold', lambda x, t: x > t, category='custom_callable_ops')
        self.add_primitive('weighted_sum', lambda weights, values: sum(w*v for w,v in zip(weights, values)), category='custom_callable_ops')
        self.add_primitive('max_pool', lambda values: max(values) if values else None, category='custom_callable_ops')
        # Add common NN ops as placeholders, actual execution logic might be external
        self.primitives['unary_ops'].update(['relu', 'sigmoid', 'tanh', 'gelu', 'softmax', 'layer_norm'])
        self.primitives['binary_ops'].update(['residual']) # e.g. residual(x, y) = x + y


    def add_primitive_set(self, name: str, values: List[str]):
        if 'custom_sets' not in self.primitives: self.primitives['custom_sets'] = {}
        self.primitives['custom_sets'][name] = values

    def add_primitive(self, name: str, func_or_values: Any, category: Optional[str] = None):
        if callable(func_or_values):
            cat = category if category else 'custom_callable_ops'
            if cat not in self.primitives: self.primitives[cat] = {}
            self.primitives[cat][name] = func_or_values
        elif isinstance(func_or_values, list):
            if 'named_lists' not in self.primitives: self.primitives['named_lists'] = {}
            self.primitives['named_lists'][name] = func_or_values
        else:
            if 'custom_values' not in self.primitives: self.primitives['custom_values'] = {}
            self.primitives['custom_values'][name] = func_or_values

    def _attention_op(self, query: Any, key: Any, value: Any) -> Any:
        if isinstance(query, (sp.Symbol, sp.Expr)): # Symbolic mode
            return sp.Function('Attention')(query, key, value)
        # Numeric mode
        if isinstance(query, np.ndarray):
            q = torch.tensor(query, dtype=torch.float32) if not isinstance(query, torch.Tensor) else query
            k = torch.tensor(key, dtype=torch.float32) if not isinstance(key, torch.Tensor) else key
            v = torch.tensor(value, dtype=torch.float32) if not isinstance(value, torch.Tensor) else value
            d_k = q.shape[-1]
            scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(d_k)
            attention_weights = torch.softmax(scores, dim=-1)
            output = torch.matmul(attention_weights, v)
            return output.numpy() if isinstance(query, np.ndarray) else output
        return f"Attention({query}, {key}, {value})"

    def _embedding_op(self, indices: Any, embedding_matrix: Any) -> Any: # Renamed from _embedding_lookup
        if isinstance(indices, (sp.Symbol, sp.Expr)): # Symbolic mode
            return sp.Function('Embedding')(indices, embedding_matrix)
        # Numeric mode
        if isinstance(indices, (np.ndarray, list)):
            indices_arr = np.array(indices, dtype=int) # Ensure it's an array for indexing
            if isinstance(embedding_matrix, np.ndarray): return embedding_matrix[indices_arr]
            elif isinstance(embedding_matrix, torch.Tensor):
                indices_tensor = torch.tensor(indices_arr, dtype=torch.long)
                return embedding_matrix[indices_tensor].numpy()
            elif isinstance(embedding_matrix, str): # Symbolic reference to matrix name
                return f"Embedding({indices_arr}, {embedding_matrix})"
        return f"EmbeddingLookup({indices}, {embedding_matrix})"

    def _is_tensor_compatible(self, operand: Any) -> bool:
        return isinstance(operand, (Expression, Variable, np.ndarray, list, sp.Expr, int, float))

    def _to_sympy(self, expr_node: Expression) -> sp.Expr: # Policy for Expression class to use
        operator = expr_node.operator
        if operator in ['var', 'const']: return super()._to_sympy(expr_node) # This is Expression._to_sympy

        sympy_operands = []
        for op in expr_node.operands:
            if isinstance(op, Expression): sympy_operands.append(self._to_sympy(op)) # Recursive call to AIGrammar policy
            elif isinstance(op, Variable): sympy_operands.append(op.symbolic)
            elif isinstance(op, (int, float)): sympy_operands.append(sp.Number(op))
            elif isinstance(op, str): sympy_operands.append(sp.Symbol(op)) # For string variable names
            else: sympy_operands.append(sp.Symbol(str(op))) # Fallback

        # Standard ops that ProgressiveGrammar's Expression._to_sympy can handle
        if operator in ['+', '-', '*', '/', '**', 'sin', 'cos', 'exp', 'log', 'neg', 'inv', 'sqrt', 'diff', 'int']:
             # Need to call the base Expression's _to_sympy method, not AIGrammar's super()
             # This is tricky. The Expression class itself should handle these.
             # This _to_sympy in AIGrammar is a *policy provider*.
             # Let's assume Expression class does its job for these.
             # If AIGrammar needs to change how these are symbolized, it's more complex.
             # For now, assume these are handled by Expression's own _to_sympy if they reach it.
             # This method is primarily for AI-specific ops.
             # This part of logic is flawed if Expression calls this for standard ops.
             # Revisit: Expression should call grammar._to_sympy_policy(self)
             # And this policy method decides if it handles or defers to a base policy.
             # For now, let's assume standard ops are not routed here from Expression,
             # or Expression calls a different method for them.
             # If Expression *always* calls grammar's policy, then:
            temp_expr_for_std_op_symb = Expression(operator, sympy_operands) # Create temp Expression to use its _to_sympy
            return temp_expr_for_std_op_symb.symbolic # Get symbolic form via Expression's logic
            # This is a bit circular if Expression's _to_sympy calls back to grammar's policy.
            # A better way: Expression._to_sympy should handle standard ops itself.
            # This policy method is only for ops Expression doesn't know.

        # AI-specific operators
        if operator == 'if_then_else':
            return sp.Piecewise((sympy_operands[1], sympy_operands[0]), (sympy_operands[2], True))
        elif operator == 'threshold':
            return sympy_operands[0] > sympy_operands[1] # Results in a Sympy Boolean expression
        # For 'attention', 'embedding_lookup', 'relu', 'softmax', etc., return sp.Function
        # This makes them uninterpreted functions in Sympy, which is fine for symbolic representation.
        else: # Default for other AI ops (relu, gelu, attention, etc.)
            capitalized_op = operator.capitalize() if not operator.isupper() else operator
            return sp.Function(capitalized_op)(*sympy_operands)


    def _validate_expression(self, operator: str, operands: List[Any]) -> bool:
        # Check standard ops first using parent's validation (which is now ProgressiveGrammar's)
        # If it's a standard op, ProgressiveGrammar._validate_expression will check arity/types
        # If it's not known to ProgressiveGrammar, it will return False (unless it's var/const)
        # We need to allow AIGrammar to validate its own ops if parent fails.

        is_standard_valid = False
        try:
            # Temporarily remove AI ops from consideration for parent validation
            # This is hacky. A better way is a more structured primitive definition.
            ai_custom_callables = self.primitives.get('custom_callable_ops', {}).keys()
            temp_removed = {}
            for ai_op in ai_custom_callables:
                if ai_op in self.primitives['unary_ops']: # e.g. if 'relu' was added there
                    temp_removed[ai_op] = 'unary'
                    self.primitives['unary_ops'].remove(ai_op)
                elif ai_op in self.primitives['binary_ops']:
                    temp_removed[ai_op] = 'binary'
                    self.primitives['binary_ops'].remove(ai_op)

            is_standard_valid = super()._validate_expression(operator, operands)

        finally: # Restore any temporarily removed ops
            for op_name, op_type in temp_removed.items():
                if op_type == 'unary': self.primitives['unary_ops'].add(op_name)
                elif op_type == 'binary': self.primitives['binary_ops'].add(op_name)

        if is_standard_valid: return True # Parent class validated it

        # Now check AI-specific operators
        ai_operator_arity = {
            'attention': 3, 'embedding_lookup': 2, 'if_then_else': 3,
            'threshold': 2, 'weighted_sum': 2, 'max_pool': 1, # max_pool takes one list
             # Unary ops handled by ProgressiveGrammar if added to its unary_ops set
            'relu': 1, 'sigmoid': 1, 'tanh': 1, 'gelu': 1, 'softmax': 1, 'layer_norm': 1,
            'residual': 2 # binary op
        }
        # Check if operator is one of AI specific ops not covered by standard validation
        if operator not in ai_operator_arity and operator not in self.primitives.get('custom_callable_ops', {}):
            return False # Unknown to both parent and AI grammar specific list

        expected_arity = ai_operator_arity.get(operator)
        # If it's in custom_callable_ops, it might not be in ai_operator_arity if arity varies or not set there.
        # For now, assume custom_callable_ops are also in ai_operator_arity if they have fixed arity.
        if expected_arity is not None and len(operands) != expected_arity:
            logging.debug(f"AIGrammar validation: Arity mismatch for {operator}. Expected {expected_arity}, got {len(operands)}")
            return False

        # Type checks for AI operators (example for attention)
        if operator == 'attention':
            return all(self._is_tensor_compatible(op) for op in operands)
        # Add more specific type checks for other AI ops if needed.
        # For now, allow general operand types if arity matches for other AI ops.
        return True


def ai_grammar_get_arity(self, op_name: str) -> int:
    _ai_op_arities = {
        'attention': 3, 'embedding_lookup': 2, 'if_then_else': 3,
        'threshold': 2, 'weighted_sum': 2, 'max_pool': 1,
        'relu':1, 'sigmoid':1, 'tanh':1, 'gelu':1, 'softmax':1, 'layer_norm':1, 'residual':2
    }
    if op_name in _ai_op_arities: return _ai_op_arities[op_name]
    # Fallback to ProgressiveGrammar's get_arity for standard operators
    # Need to call it carefully if AIGrammar is a subclass of ProgressiveGrammar
    return super(AIGrammar, self).get_arity(op_name)

AIGrammar.get_arity = ai_grammar_get_arity


def _is_operator_known_impl(grammar_instance, op_name: str) -> bool:
    try:
        grammar_instance.get_arity(op_name) # This will now use the potentially overridden get_arity
        return True
    except ValueError:
        return False

ProgressiveGrammar.get_arity = lambda self, op_name: \
    2 if op_name in self.primitives.get('binary_ops', set()) else \
    1 if op_name in self.primitives.get('unary_ops', set()) else \
    2 if op_name in self.primitives.get('calculus_ops', set()) else \
    (_ for _ in ()).throw(ValueError(f"Unknown operator or function: '{op_name}' in ProgressiveGrammar"))

ProgressiveGrammar.is_operator_known = lambda self, op_name: _is_operator_known_impl(self, op_name)


# Example usage and testing (main block from original, kept for context, might need adjustment)
if __name__ == "__main__":
    grammar = ProgressiveGrammar()
    t_obs = np.linspace(0, 10, 1000)
    theta_obs = np.sin(2 * np.pi * 0.5 * t_obs) + 0.1 * np.random.randn(len(t_obs))
    omega_obs = np.cos(2 * np.pi * 0.5 * t_obs) * 2 * np.pi * 0.5 + 0.1 * np.random.randn(len(t_obs))
    energy_obs = 0.5 * omega_obs**2 + 9.8 * (1 - np.cos(theta_obs)) + 0.05 * np.random.randn(len(t_obs))
    observations_main = np.column_stack([theta_obs, omega_obs, energy_obs, np.random.randn(len(t_obs)) * 0.5])

    print("Discovering variables from observations...")
    variables_main = grammar.discover_variables(observations_main, t_obs)
    print(f"\nDiscovered {len(variables_main)} variables:")
    for var_main in variables_main: print(f"  {var_main.name}: {var_main.properties}")

    print("\nCreating expressions...")
    if len(variables_main) >= 2:
        v1_main, v2_main = variables_main[0], variables_main[1]
        expr1_main = grammar.create_expression('*', [grammar.create_expression('const', [0.5]),
                                             grammar.create_expression('**', [v1_main, grammar.create_expression('const', [2])])])
        if expr1_main: print(f"Expression 1: {expr1_main.symbolic}, Complexity: {expr1_main.complexity}")
        invalid_main = grammar.create_expression('+', [v1_main])
        print(f"\nInvalid expression (wrong arity): {invalid_main}")
        if expr1_main:
            derivative_main = grammar.create_expression('diff', [expr1_main, v1_main])
            if derivative_main: print(f"\nDerivative of expr1 w.r.t {v1_main.name}: {derivative_main.symbolic}")

    print("\n--- AIGrammar Example ---")
    ai_grammar_main = AIGrammar()
    # ... (AIGrammar example code from original, may need Variable/Expression from .expression)

    print("\n--- ProgressiveGrammar CFG Generation Example ---")
    if not grammar.variables:
        grammar.variables['v1'] = Variable(name='v1', index=0)
        grammar.variables['v2'] = Variable(name='v2', index=1)

    available_vars_main = list(grammar.variables.keys())
    if not available_vars_main: available_vars_main = ["default_var"]; grammar.variables["default_var"] = Variable(name="default_var", index=0)
    binary_ops_main = list(grammar.primitives['binary_ops'])
    unary_ops_main = list(grammar.primitives['unary_ops'])
    cfg_rules_main = [CFGRule('EXPR', ['BINARY_OP_EXPR'], weight=0.4), CFGRule('EXPR', ['UNARY_OP_EXPR'], weight=0.3),
                      CFGRule('EXPR', ['VAR'], weight=0.2), CFGRule('EXPR', ['CONST'], weight=0.1)]
    for op_main in binary_ops_main: cfg_rules_main.append(CFGRule('BINARY_OP_EXPR', [op_main, 'EXPR', 'EXPR']))
    for op_main in unary_ops_main: cfg_rules_main.append(CFGRule('UNARY_OP_EXPR', [op_main, 'EXPR']))
    for var_name_main in available_vars_main: cfg_rules_main.append(CFGRule('VAR', [var_name_main], weight=1.0/len(available_vars_main)))
    cfg_rules_main.append(CFGRule('CONST', ['CONST']))
    grammar.set_rules_from_cfg(cfg_rules_main, start_symbol='EXPR')
    logging.basicConfig(level=logging.INFO)
    generated_expression_main = grammar.generate_random_expression_from_cfg(max_depth=5)
    if generated_expression_main:
        print(f"\nSuccessfully generated expression from CFG:\n  Symbolic: {generated_expression_main.symbolic}")
    else:
        print("\nFailed to generate an expression using CFG.")
