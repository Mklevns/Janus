import pytest
import pickle
from unittest.mock import patch

import physics_discovery_extensions # Add this import
from physics_discovery_extensions import SymbolicRegressor

class MockExpression:
    def __init__(self, operator, operands, complexity=0, symbolic=""):
        self.operator = operator
        self.operands = list(operands)
        self._post_init_called_count = 0
        # Call __post_init__ explicitly after all attributes are set
        # self.__post_init__() # This was called too early if complexity/symbolic were passed

        # Store initial complexity and symbolic if provided, otherwise they'll be computed
        self._initial_complexity = complexity
        self._initial_symbolic = symbolic
        self.__post_init__()


    def __post_init__(self):
        self._post_init_called_count += 1

        # If symbolic and complexity were provided to constructor, use them for the first call
        # For subsequent calls (e.g. after pickle), recalculate
        if self._post_init_called_count == 1 and self._initial_symbolic:
             self.symbolic = self._initial_symbolic
             self.complexity = self._initial_complexity
             return

        current_complexity = 1 # For the operator itself
        op_names = []
        for op in self.operands:
            if isinstance(op, MockExpression):
                current_complexity += op.complexity
                op_names.append(op.symbolic)
            else: # Assuming terminals are simple strings or numbers
                current_complexity += 0 # Terminals might not add to complexity here, or 1 if they do
                op_names.append(str(op))

        self.complexity = current_complexity

        if not self.operands:
            self.symbolic = str(self.operator)
        else:
            self.symbolic = f"{self.operator}({', '.join(op_names)})"

    def __getstate__(self):
        # Don't pickle _post_init_called_count if you want __post_init__ to run fresh on unpickle
        state = self.__dict__.copy()
        # Ensure _initial_complexity and _initial_symbolic are part of the state
        # if they were set.
        state['_initial_complexity'] = getattr(self, '_initial_complexity', 0)
        state['_initial_symbolic'] = getattr(self, '_initial_symbolic', "")

        # Reset post_init_called_count for the unpickled object so __post_init__ runs fully
        # This is important if unpickling should behave like a fresh construction + post_init
        state['_post_init_called_count'] = 0
        return state

    def __setstate__(self, d):
        self.__dict__.update(d)
        # Explicitly call __post_init__ after unpickling to ensure tree is correct
        # This is crucial because operand expressions might have been modified during crossover
        # before this expression was pickled (e.g. if it's a child of a crossover point).
        self.__post_init__()


    def __eq__(self, other):
        if not isinstance(other, MockExpression):
            return NotImplemented
        operands_equal = len(self.operands) == len(other.operands) and \
                         all(o1 == o2 for o1, o2 in zip(self.operands, other.operands))

        return (self.operator == other.operator and
                operands_equal and
                self.complexity == other.complexity and
                self.symbolic == other.symbolic)

    def __repr__(self):
        return (f"MockExpression(operator='{self.operator}', "
                f"operands={self.operands!r}, "
                f"complexity={self.complexity}, symbolic='{self.symbolic}')")

class MockGrammar:
    def __init__(self):
        self.primitives = {
            'binary_ops': {'+', '-', '*', '/'},
            'unary_ops': {'sin', 'cos', 'exp'}
        }

class TestSymbolicRegressorCrossover:

    @pytest.fixture
    def grammar(self):
        return MockGrammar()

    @pytest.fixture
    def regressor(self, grammar):
        # Temporarily set physics_discovery_extensions.Expression to MockExpression
        # so that `isinstance(..., Expression)` checks within SymbolicRegressor methods work.
        original_expression_type = getattr(physics_discovery_extensions, 'Expression', None)
        physics_discovery_extensions.Expression = MockExpression

        reg = SymbolicRegressor(grammar=grammar)

        # Teardown: Restore original Expression type if it existed, otherwise remove
        if original_expression_type is not None:
            physics_discovery_extensions.Expression = original_expression_type
        else:
            del physics_discovery_extensions.Expression

        return reg

    @pytest.fixture
    def sample_expressions(self):
        # P1 = +(A, *(B,C))  Complexity: A=1, B=1, C=1, *=1+1+1=3, +=1+1+3=5
        # P2 = -(D, E)      Complexity: D=1, E=1, -=1+1+1=3

        # Define with explicit complexity and symbolic form for clarity and to match __init__
        expr_b = MockExpression(operator='B', operands=[], complexity=1, symbolic="B")
        expr_c = MockExpression(operator='C', operands=[], complexity=1, symbolic="C")
        expr_mul_bc = MockExpression(operator='*', operands=[expr_b, expr_c], complexity=3, symbolic="*(B, C)")
        expr_a = MockExpression(operator='A', operands=[], complexity=1, symbolic="A")
        parent1 = MockExpression(operator='+', operands=[expr_a, expr_mul_bc], complexity=5, symbolic="+(A, *(B, C))")

        expr_d = MockExpression(operator='D', operands=[], complexity=1, symbolic="D")
        expr_e = MockExpression(operator='E', operands=[], complexity=1, symbolic="E")
        parent2 = MockExpression(operator='-', operands=[expr_d, expr_e], complexity=3, symbolic="-(D, E)")

        # Save representations of original parents for later comparison
        original_parent1_repr = repr(parent1)
        original_parent2_repr = repr(parent2)

        return parent1, parent2, original_parent1_repr, original_parent2_repr

    def test_crossover_swaps_subtrees_and_updates_children(self, regressor, sample_expressions):
        parent1, parent2, p1_orig_repr, p2_orig_repr = sample_expressions

        # Make deep copies for checking originals are untouched
        p1_before_crossover_copy = pickle.loads(pickle.dumps(parent1))
        p2_before_crossover_copy = pickle.loads(pickle.dumps(parent2))

        # Ensure copies are identical before operation
        assert parent1 == p1_before_crossover_copy
        assert parent2 == p2_before_crossover_copy


        # Mock _get_all_subexpressions
        # It should return a list of nodes (MockExpression instances)
        def mock_get_all_subexpressions_side_effect(expr_copy):
            # Simplified: just return the expression itself and its direct operands if they are MockExpressions
            # This is important: the actual _get_all_subexpressions in the code navigates the tree.
            # For this test, we need to provide the nodes that np.random.choice will pick from.
            nodes = [expr_copy]
            q = list(expr_copy.operands)
            visited_refs = {id(expr_copy)} # Keep track of visited object ids to avoid reprocessing due to shared instances or cycles

            idx = 0
            while idx < len(q):
                current_op = q[idx]
                idx += 1
                if isinstance(current_op, MockExpression) and id(current_op) not in visited_refs:
                    nodes.append(current_op)
                    visited_refs.add(id(current_op))
                    q.extend(current_op.operands) # Add its operands to continue traversal
            return nodes

        # Mock np.random.choice
        # This needs to select specific nodes to make the test deterministic
        # parent1: +(A, *(B,C)) -> we want to select 'A'
        # parent2: -(D, E)     -> we want to select 'E'
        # Expected child1: +(E, *(B,C))
        # Expected child2: -(D, A)

        # Store the nodes that will be "chosen"
        chosen_nodes_map = {}

        def custom_np_random_choice(nodes_from_expr_copy):
            # Identify which parent tree these nodes belong to (e.g., by root operator or a unique node)
            # For parent1 (root '+'), choose node 'A'.
            # For parent2 (root '-'), choose node 'E'.

            # This is a bit tricky as np.random.choice is called twice, once for p1_nodes, once for p2_nodes
            # We need to distinguish which call is which.
            # Let's use the symbolic form of the root node to distinguish.

            root_symbolic = nodes_from_expr_copy[0].symbolic

            if root_symbolic == "+(A, *(B, C))": # Parent 1
                for node in nodes_from_expr_copy:
                    if node.symbolic == "A":
                        chosen_nodes_map['p1_choice'] = node
                        return node
            elif root_symbolic == "-(D, E)": # Parent 2
                for node in nodes_from_expr_copy:
                    if node.symbolic == "E":
                        chosen_nodes_map['p2_choice'] = node
                        return node
            # Fallback if specific nodes not found (shouldn't happen in this test)
            return nodes_from_expr_copy[0]


        with patch.object(regressor, '_get_all_subexpressions', side_effect=mock_get_all_subexpressions_side_effect), \
             patch('physics_discovery_extensions.np.random.choice', side_effect=custom_np_random_choice):
            child1, child2 = regressor._crossover(parent1, parent2)

        # --- Assertions ---
        assert child1 is not None, "Child1 should not be None"
        assert child2 is not None, "Child2 should not be None"

        # 1. Check original parents are unmodified
        assert repr(parent1) == p1_orig_repr, "Original parent1 modified (repr check)"
        assert repr(parent2) == p2_orig_repr, "Original parent2 modified (repr check)"
        assert parent1 == p1_before_crossover_copy, "Original parent1 modified (equality check)"
        assert parent2 == p2_before_crossover_copy, "Original parent2 modified (equality check)"

        # 2. Check children are different objects from parents
        assert child1 is not parent1 and child1 is not parent2, "Child1 should be a new object"
        assert child2 is not parent1 and child2 is not parent2, "Child2 should be a new object"

        # 3. Check structure of child1: +(E, *(B,C))
        # Expected: parent1's 'A' is replaced by parent2's 'E'
        # Symbolic: +(E, *(B, C))
        # Complexity: E=1, B=1, C=1, *(B,C)=3, +(E, *(B,C))=1+1+3=5
        assert child1.symbolic == "+(E, *(B, C))", f"Child1 symbolic form is wrong: {child1.symbolic}"
        assert child1.complexity == 5, f"Child1 complexity is wrong: {child1.complexity}"
        assert child1.operator == '+', "Child1 operator is wrong"
        assert len(child1.operands) == 2, "Child1 should have 2 operands"
        assert child1.operands[0].symbolic == "E", "Child1's first operand is wrong"
        assert child1.operands[1].symbolic == "*(B, C)", "Child1's second operand is wrong"

        # 4. Check structure of child2: -(D, A)
        # Expected: parent2's 'E' is replaced by parent1's 'A'
        # Symbolic: -(D, A)
        # Complexity: D=1, A=1, -(D,A)=1+1+1=3
        assert child2.symbolic == "-(D, A)", f"Child2 symbolic form is wrong: {child2.symbolic}"
        assert child2.complexity == 3, f"Child2 complexity is wrong: {child2.complexity}"
        assert child2.operator == '-', "Child2 operator is wrong"
        assert len(child2.operands) == 2, "Child2 should have 2 operands"
        assert child2.operands[0].symbolic == "D", "Child2's first operand is wrong"
        assert child2.operands[1].symbolic == "A", "Child2's second operand is wrong"

        # 5. Check __post_init__ was called on children after modifications
        # The initial call happens in __init__. Pickle also calls it.
        # The crossover method calls it.
        # For child1 (copy of parent1), it's unpickled (1), then post_init in crossover (2)
        # For child2 (copy of parent2), it's unpickled (1), then post_init in crossover (2)
        assert child1._post_init_called_count >= 1 # Called by pickle and by _crossover
        assert child2._post_init_called_count >= 1 # Called by pickle and by _crossover

        # Check that the chosen nodes were actually swapped and are part of the new children.
        # chosen_nodes_map['p1_choice'] is crossover_point1 (object from p1_copy, was A, now E-like)
        # chosen_nodes_map['p2_choice'] is crossover_point2 (object from p2_copy, was E, now A-like)

        # The object instance chosen from p1_copy (which was A, now E-like) should be child1.operands[0]
        assert child1.operands[0] is chosen_nodes_map['p1_choice']
        assert child1.operands[0].symbolic == "E" # Double check its content

        # The object instance chosen from p2_copy (which was E, now A-like) should be child2.operands[1]
        assert child2.operands[1] is chosen_nodes_map['p2_choice']
        assert child2.operands[1].symbolic == "A" # Double check its content


    def test_crossover_when_no_crossover_points_returns_copies(self, regressor, sample_expressions):
        parent1, parent2, p1_orig_repr, p2_orig_repr = sample_expressions

        # Make deep copies for checking originals are untouched
        p1_before_crossover_copy = pickle.loads(pickle.dumps(parent1))
        p2_before_crossover_copy = pickle.loads(pickle.dumps(parent2))

        # Mock _get_all_subexpressions to return empty list, simulating no valid crossover points
        with patch.object(regressor, '_get_all_subexpressions', return_value=[]):
            child1, child2 = regressor._crossover(parent1, parent2)

        # --- Assertions ---
        assert child1 is not None, "Child1 should not be None in no-crossover case"
        assert child2 is not None, "Child2 should not be None in no-crossover case"

        # 1. Children should be copies of the parents
        assert child1 == parent1, "Child1 should be equal to parent1 if no crossover points"
        assert child1 is not parent1, "Child1 should be a copy, not the same object as parent1"
        assert child1.symbolic == parent1.symbolic, "Child1 symbolic form mismatch"
        assert child1.complexity == parent1.complexity, "Child1 complexity mismatch"

        assert child2 == parent2, "Child2 should be equal to parent2 if no crossover points"
        assert child2 is not parent2, "Child2 should be a copy, not the same object as parent2"
        assert child2.symbolic == parent2.symbolic, "Child2 symbolic form mismatch"
        assert child2.complexity == parent2.complexity, "Child2 complexity mismatch"

        # 2. Original parents should be unmodified
        assert repr(parent1) == p1_orig_repr, "Original parent1 modified (repr check)"
        assert repr(parent2) == p2_orig_repr, "Original parent2 modified (repr check)"
        assert parent1 == p1_before_crossover_copy, "Original parent1 modified (equality check)"
        assert parent2 == p2_before_crossover_copy, "Original parent2 modified (equality check)"

        # 3. Check __post_init__ calls (mainly due to unpickling)
        # In this case, _crossover returns early, so __post_init__ on children is only from pickle
        assert child1._post_init_called_count >= 1 # At least from unpickling
        assert child2._post_init_called_count >= 1 # At least from unpickling

        # Further check that they are indeed deep copies
        child1.operator = "MODIFIED"
        assert parent1.operator == "+", "Modifying child1 affected parent1"
        child2.operator = "MODIFIED_TOO"
        assert parent2.operator == "-", "Modifying child2 affected parent2"


def test_generate_candidates_grammar_guided():
    grammar = physics_discovery_extensions.ProgressiveGrammar()
    variables = [physics_discovery_extensions.Variable("x", 0, {}),
                 physics_discovery_extensions.Variable("y", 1, {})]
    detector = physics_discovery_extensions.ConservationDetector(grammar)

    candidates = detector._generate_candidates(variables, max_complexity=3)
    sym_strs = {str(c.symbolic) for c in candidates}

    # Basic variables included
    assert "x" in sym_strs and "y" in sym_strs

    # Should include at least one unary and one binary combination
    assert any("+" in s for s in sym_strs)
    assert any(token in s for s in sym_strs for token in ["sin", "cos", "log", "exp", "sqrt"])

    # At least one binary combination with complexity 3 should exist
    assert any(c.complexity == 3 for c in candidates)
