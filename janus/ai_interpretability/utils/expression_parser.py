# expression_parser.py
"""
Improved expression parser for emergent_monitor.py
Uses Python AST for robust parsing instead of string matching.
"""

import ast
import numpy as np
from typing import List, Dict, Set, Optional


class ExpressionParser:
    """Parse mathematical expressions into operator sequences."""

    def __init__(self):
        self.operator_map = {
            ast.Add: 'add',
            ast.Sub: 'sub',
            ast.Mult: 'mul',
            ast.Div: 'div',
            ast.Pow: 'pow',
            ast.USub: 'neg',
            ast.UAdd: 'pos',
        }

    def parse_to_ops(self, expression: str) -> List[str]:
        """
        Parse expression string to operator sequence.

        Args:
            expression: Mathematical expression as string

        Returns:
            List of operator names in order of evaluation
        """
        try:
            # Parse expression to AST
            tree = ast.parse(expression, mode='eval')

            # Extract operators in evaluation order
            ops = []
            self._extract_ops(tree.body, ops)

            return ops

        except (SyntaxError, ValueError) as e:
            # Fallback for non-Python expressions (e.g., from SymPy)
            return self._fallback_parse(expression)

    def _extract_ops(self, node: ast.AST, ops: List[str]):
        """Recursively extract operators from AST."""
        if isinstance(node, ast.BinOp):
            # Binary operation
            self._extract_ops(node.left, ops)
            self._extract_ops(node.right, ops)
            op_type = type(node.op)
            if op_type in self.operator_map:
                ops.append(self.operator_map[op_type])

        elif isinstance(node, ast.UnaryOp):
            # Unary operation
            self._extract_ops(node.operand, ops)
            op_type = type(node.op)
            if op_type in self.operator_map:
                ops.append(self.operator_map[op_type])

        elif isinstance(node, ast.Call):
            # Function call
            if isinstance(node.func, ast.Name):
                ops.append(f'func_{node.func.id}')
            # Process arguments
            for arg in node.args:
                self._extract_ops(arg, ops)

        elif isinstance(node, ast.Compare):
            # Comparison operators
            for op_node in node.ops: # Corrected: iterate over node.ops, not node.op
                if isinstance(op_node, ast.Lt):
                    ops.append('lt')
                elif isinstance(op_node, ast.Gt):
                    ops.append('gt')
                elif isinstance(op_node, ast.Eq):
                    ops.append('eq')
            # Process operands
            self._extract_ops(node.left, ops)
            for comp in node.comparators:
                self._extract_ops(comp, ops)

    def _fallback_parse(self, expression: str) -> List[str]:
        """Fallback parser for non-standard expressions."""
        ops = []

        # Handle SymPy-style expressions
        operator_symbols = {
            '+': 'add',
            '-': 'sub',
            '*': 'mul',
            '/': 'div',
            '**': 'pow',
            '^': 'pow'
        }

        # Simple tokenization approach
        for symbol, op_name in operator_symbols.items():
            if symbol in expression:
                ops.append(op_name)

        # Check for functions
        import re
        functions = re.findall(r'\b(sin|cos|tan|exp|log|sqrt)\b', expression)
        for func in functions:
            ops.append(f'func_{func}')

        return ops

    def get_operator_embedding(self, ops: List[str],
                            embedding_dim: int = 64) -> np.ndarray:
        """
        Convert operator sequence to fixed-size embedding.

        Args:
            ops: List of operator names
            embedding_dim: Size of output embedding

        Returns:
            Fixed-size embedding vector
        """
        # Create vocabulary of all possible operators
        vocab = [
            'add', 'sub', 'mul', 'div', 'pow', 'neg', 'pos',
            'func_sin', 'func_cos', 'func_tan', 'func_exp',
            'func_log', 'func_sqrt', 'lt', 'gt', 'eq'
        ]

        # One-hot encode operators
        op_vectors = []
        for op in ops[:20]:  # Limit sequence length
            vec = np.zeros(len(vocab))
            if op in vocab:
                vec[vocab.index(op)] = 1
            op_vectors.append(vec)

        if not op_vectors:
            return np.zeros(embedding_dim)

        # Apply simple pooling
        op_matrix = np.array(op_vectors)

        # Mean and max pooling
        mean_pool = np.mean(op_matrix, axis=0)
        max_pool = np.max(op_matrix, axis=0)

        # Combine features
        features = np.concatenate([mean_pool, max_pool])

        # Project to desired dimension
        if len(features) > embedding_dim:
            return features[:embedding_dim]
        else:
            return np.pad(features, (0, embedding_dim - len(features)))


# Integration with emergent_monitor.py
# The following shows how ExpressionEmbedder in emergent_monitor.py
# should be updated to use this parser.
#
# class ExpressionEmbedder:
#     def __init__(self, embedding_dim: int = 128):
#         self.embedding_dim = embedding_dim
#         self.parser = ExpressionParser() # Initialize the new parser
#         # ... other initializations ...
#
#     def _parse_to_ops(self, expression: str) -> List[str]:
#         """Parse expression string to operator sequence using the new parser."""
#         return self.parser.parse_to_ops(expression)
#
#     def embed_expression(self, expression: str) -> np.ndarray:
#         ops = self._parse_to_ops(expression)
#         if not ops:
#             return np.zeros(self.embedding_dim)
#
#         # Use the get_operator_embedding method from the parser
#         embedding = self.parser.get_operator_embedding(ops, self.embedding_dim)
#         return embedding
#
#     # ... other methods like _get_expression_complexity etc. may also need updates
#     # if they relied on the old parsing logic.
