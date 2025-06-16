import os
import sys
import types
import pytest

# Provide a minimal torch stub to satisfy imports if torch is unavailable
if 'torch' not in sys.modules:
    torch_stub = types.SimpleNamespace(
        nn=types.SimpleNamespace(Module=object),
        optim=types.SimpleNamespace(Adam=object),
        randn_like=lambda x: x,
        FloatTensor=lambda *a, **k: None,
    )
    sys.modules['torch'] = torch_stub
    sys.modules['torch.nn'] = torch_stub.nn
    sys.modules['torch.optim'] = torch_stub.optim

if 'scipy' not in sys.modules:
    scipy_stats_stub = types.SimpleNamespace(entropy=lambda *a, **k: 0)
    scipy_stub = types.SimpleNamespace(stats=scipy_stats_stub)
    sys.modules['scipy'] = scipy_stub
    sys.modules['scipy.stats'] = scipy_stats_stub

if 'sklearn' not in sys.modules:
    sklearn_decomp_stub = types.SimpleNamespace(FastICA=object)
    sklearn_preproc_stub = types.SimpleNamespace(StandardScaler=object)
    sklearn_stub = types.SimpleNamespace(
        decomposition=sklearn_decomp_stub,
        preprocessing=sklearn_preproc_stub,
    )
    sys.modules['sklearn'] = sklearn_stub
    sys.modules['sklearn.decomposition'] = sklearn_decomp_stub
    sys.modules['sklearn.preprocessing'] = sklearn_preproc_stub

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from progressive_grammar_system import ProgressiveGrammar, Variable, Expression


def test_invalid_arity_returns_none():
    grammar = ProgressiveGrammar()
    var = Variable(name="x", index=0)
    result = grammar.create_expression('+', [var])
    assert result is None


def test_valid_expression_is_created():
    grammar = ProgressiveGrammar()
    var = Variable(name="x", index=0)
    expr = grammar.create_expression('+', [var, var])
    assert isinstance(expr, Expression)


# --- Tests for Commutative Keys ---

@pytest.fixture
def grammar_and_vars():
    grammar = ProgressiveGrammar()
    # Clear any predefined constants if they might interfere with names like '1'
    # grammar.primitives['constants'] = {}
    # Actually, the new key gen uses "const:1.00000" so it's fine.

    var_a = Variable(name="a", index=0)
    var_b = Variable(name="b", index=1)
    var_c = Variable(name="c", index=2)
    const_1_val = 1.0
    const_2_val = 2.0

    # How constants are handled in Expression creation for _expression_key:
    # The _expression_key method expects Expression, Variable, int, or float for operands.
    # When it gets int/float directly, it formats them.
    # When it gets an Expression of type 'const', it uses its operand.
    # For these tests, we'll pass Variables and direct float/int values as operands.
    return grammar, var_a, var_b, var_c, const_1_val, const_2_val

def test_commutative_addition_key(grammar_and_vars):
    grammar, var_a, var_b, _, _, _ = grammar_and_vars

    expr_ab = grammar.create_expression('+', [var_a, var_b])
    expr_ba = grammar.create_expression('+', [var_b, var_a])

    assert expr_ab is not None, "Expression a+b should be valid"
    assert expr_ba is not None, "Expression b+a should be valid"

    key_ab = grammar._expression_key(expr_ab)
    key_ba = grammar._expression_key(expr_ba)

    assert key_ab == key_ba, "Keys for a+b and b+a should be identical"

def test_commutative_multiplication_key(grammar_and_vars):
    grammar, var_a, var_b, _, _, _ = grammar_and_vars

    expr_ab = grammar.create_expression('*', [var_a, var_b])
    expr_ba = grammar.create_expression('*', [var_b, var_a])

    assert expr_ab is not None, "Expression a*b should be valid"
    assert expr_ba is not None, "Expression b*a should be valid"

    key_ab = grammar._expression_key(expr_ab)
    key_ba = grammar._expression_key(expr_ba)

    assert key_ab == key_ba, "Keys for a*b and b*a should be identical"

def test_non_commutative_subtraction_key(grammar_and_vars):
    grammar, var_a, var_b, _, _, _ = grammar_and_vars

    expr_ab = grammar.create_expression('-', [var_a, var_b])
    expr_ba = grammar.create_expression('-', [var_b, var_a])

    assert expr_ab is not None, "Expression a-b should be valid"
    assert expr_ba is not None, "Expression b-a should be valid"

    key_ab = grammar._expression_key(expr_ab)
    key_ba = grammar._expression_key(expr_ba)

    assert key_ab != key_ba, "Keys for a-b and b-a should be different"

def test_complex_commutative_keys(grammar_and_vars):
    grammar, var_a, var_b, var_c, _, _ = grammar_and_vars

    # (a+b)+c
    expr_ab = grammar.create_expression('+', [var_a, var_b])
    expr_ab_c = grammar.create_expression('+', [expr_ab, var_c])
    key_ab_c = grammar._expression_key(expr_ab_c)
    # Expected: +(+(var:a,var:b),var:c) -> sorted outer: +(var:c,+(var:a,var:b))
    # sorted inner for +: "var:a,var:b"
    # outer operands before sort: "+(var:a,var:b)", "var:c"
    # outer operands after sort: "var:c", "+(var:a,var:b)"
    # key: "+(var:c,+(var:a,var:b))"

    # c+(b+a)
    expr_ba = grammar.create_expression('+', [var_b, var_a]) # inner key is +(var:a,var:b)
    expr_c_ba = grammar.create_expression('+', [var_c, expr_ba])
    key_c_ba = grammar._expression_key(expr_c_ba)
    # Expected: +(var:c,+(var:a,var:b)) -> sorted outer: +(var:c,+(var:a,var:b))
    # inner expression b+a key: "+(var:a,var:b)"
    # outer operands before sort: "var:c", "+(var:a,var:b)"
    # outer operands after sort: "var:c", "+(var:a,var:b)"
    # key: "+(var:c,+(var:a,var:b))"

    assert key_ab_c == key_c_ba, "Keys for (a+b)+c and c+(b+a) should be identical"

    # Test a+(b+c) vs (a+b)+c - these should be different due to structure unless canonicalized
    # The current key logic only sorts direct operands of a commutative op.
    expr_bc = grammar.create_expression('+', [var_b, var_c]) # inner key +(var:b,var:c)
    expr_a_bc = grammar.create_expression('+', [var_a, expr_bc])
    key_a_bc = grammar._expression_key(expr_a_bc)
    # Expected: +(var:a,+(var:b,var:c)) -> sorted outer: +(var:a,+(var:b,var:c))
    # (no sort changes outer because "var:a" < "+(var:b,var:c)")

    assert key_ab_c != key_a_bc, "Keys for (a+b)+c and a+(b+c) should be different due to structure"


def test_keys_with_constants(grammar_and_vars):
    grammar, var_a, _, _, const_1, _ = grammar_and_vars

    # a + 1.0
    expr_a_const1 = grammar.create_expression('+', [var_a, const_1])
    # 1.0 + a
    expr_const1_a = grammar.create_expression('+', [const_1, var_a])

    assert expr_a_const1 is not None
    assert expr_const1_a is not None

    key_a_const1 = grammar._expression_key(expr_a_const1)
    key_const1_a = grammar._expression_key(expr_const1_a)
    # Expected: Operands "var:a", "const:1". Sorted: "const:1", "var:a".
    # Key: "+(const:1,var:a)" using .6g format for 1.0

    # The new key gen for consts: f"const:{float(expr):.6g}"
    # So 1.0 becomes "const:1"
    expected_key_part_const1 = f"const:{float(const_1):.6g}" # "const:1"
    expected_key_part_vara = f"var:{var_a.name}" # "var:a"

    # Sorted operands for key string: const_1_key_part, var_a_key_part
    # Because "const:1" < "var:a"

    expected_final_key = f"+({expected_key_part_const1},{expected_key_part_vara})" # "+(const:1,var:a)"

    assert key_a_const1 == expected_final_key, f"Key for a+1.0 was {key_a_const1}, expected {expected_final_key}"
    assert key_const1_a == expected_final_key, f"Key for 1.0+a was {key_const1_a}, expected {expected_final_key}"


def test_key_constant_normalization(grammar_and_vars):
    grammar, var_a, _, _, _, _ = grammar_and_vars # const_1 is 1.0

    # a + 1 (integer)
    expr_a_int1 = grammar.create_expression('+', [var_a, 1])
    # a + 1.0 (float)
    expr_a_float1 = grammar.create_expression('+', [var_a, 1.0])

    assert expr_a_int1 is not None
    assert expr_a_float1 is not None

    key_a_int1 = grammar._expression_key(expr_a_int1)
    key_a_float1 = grammar._expression_key(expr_a_float1)

    # Both 1 and 1.0 should be formatted to "const:1" by f"const:{float(expr):.6g}"
    # So keys should be identical.
    # Operands "var:a", "const:1". Sorted: "const:1", "var:a".
    # Key: "+(const:1,var:a)"
    expected_key = f"+(const:1,var:a)"

    assert key_a_int1 == expected_key, f"Key for a+1 (int) was {key_a_int1}, expected {expected_key}"
    assert key_a_float1 == expected_key, f"Key for a+1.0 (float) was {key_a_float1}, expected {expected_key}"

    # Test with a different float representation
    # a + 1.000000001 (should be different from a+1 due to .6g)
    expr_a_float_long = grammar.create_expression('+', [var_a, 1.000000001])
    key_a_float_long = grammar._expression_key(expr_a_float_long)
    # 1.000000001 formatted by .6g might be "1" or "1.00000" or similar.
    # float(1.000000001) -> 1.000000001
    # "%.6g" % 1.000000001 -> '1' (if it rounds significantly) OR '1.00000' (if it truncates/rounds to 6 sig-figs)
    # Let's check: "%.6g" % 1.000000001 is '1'. "%.6g" % 1.000001 is '1.00000'.
    # So "const:1" for 1.000000001
    assert key_a_float_long == expected_key, "Key for a + 1.000000001 should be same as a+1 due to .6g"

    # a + 1.00001 (should be different from a+1 due to .6g)
    val_float_precise = 1.00001
    expr_a_float_precise = grammar.create_expression('+', [var_a, val_float_precise])
    key_a_float_precise = grammar._expression_key(expr_a_float_precise)
    # "%.6g" % 1.00001 is '1.00001'
    expected_key_precise = f"+(const:{val_float_precise:.6g},var:a)" # Should be +(const:1.00001,var:a)
    assert key_a_float_precise == expected_key_precise
    assert key_a_float_precise != expected_key, "Key for a + 1.00001 should be different from a+1"
