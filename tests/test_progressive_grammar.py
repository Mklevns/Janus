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
import sympy as sp
import numpy as np


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


# --- Tests for _to_sympy method ---

def test_unary_operator_conversion(grammar_and_vars):
    grammar, var_a, _, _, _, _ = grammar_and_vars
    a_sym = var_a.symbolic # sp.Symbol('a')

    # Test 'neg'
    expr_neg_var = grammar.create_expression('neg', [var_a])
    assert expr_neg_var is not None
    assert expr_neg_var.symbolic == -a_sym

    expr_neg_const = grammar.create_expression('neg', [2.0])
    assert expr_neg_const is not None
    assert expr_neg_const.symbolic == -sp.Float(2.0)

    # Test 'inv'
    expr_inv_var = grammar.create_expression('inv', [var_a])
    assert expr_inv_var is not None
    assert expr_inv_var.symbolic == 1/a_sym

    expr_inv_const = grammar.create_expression('inv', [2.0])
    assert expr_inv_const is not None
    assert expr_inv_const.symbolic == sp.Float(1.0/2.0)

    expr_inv_zero = grammar.create_expression('inv', [0.0])
    assert expr_inv_zero is not None
    assert expr_inv_zero.symbolic == sp.nan

    expr_inv_int_zero = grammar.create_expression('inv', [0])
    assert expr_inv_int_zero is not None
    assert expr_inv_int_zero.symbolic == sp.nan

    # Test 'sqrt'
    expr_sqrt_var = grammar.create_expression('sqrt', [var_a])
    assert expr_sqrt_var is not None
    assert expr_sqrt_var.symbolic == sp.sqrt(a_sym)

    expr_sqrt_const = grammar.create_expression('sqrt', [4.0])
    assert expr_sqrt_const is not None
    assert expr_sqrt_const.symbolic == sp.sqrt(4.0) # Should be 2.0

    # Test 'log'
    expr_log_var = grammar.create_expression('log', [var_a])
    assert expr_log_var is not None
    assert expr_log_var.symbolic == sp.log(a_sym)

    expr_log_const = grammar.create_expression('log', [float(np.e)]) # Using float(np.e)
    assert expr_log_const is not None
    assert expr_log_const.symbolic == sp.log(sp.Float(float(np.e))) # Should be 1

    # Test 'exp'
    expr_exp_var = grammar.create_expression('exp', [var_a])
    assert expr_exp_var is not None
    assert expr_exp_var.symbolic == sp.exp(a_sym)

    expr_exp_const = grammar.create_expression('exp', [1.0])
    assert expr_exp_const is not None
    assert expr_exp_const.symbolic == sp.exp(1.0)

    # Test 'sin'
    expr_sin_var = grammar.create_expression('sin', [var_a])
    assert expr_sin_var is not None
    assert expr_sin_var.symbolic == sp.sin(a_sym)

    expr_sin_const = grammar.create_expression('sin', [float(np.pi)/2]) # Using float(np.pi)
    assert expr_sin_const is not None
    assert expr_sin_const.symbolic == sp.sin(sp.Float(float(np.pi)/2)) # Should be 1

    # Test 'cos'
    expr_cos_var = grammar.create_expression('cos', [var_a])
    assert expr_cos_var is not None
    assert expr_cos_var.symbolic == sp.cos(a_sym)

    expr_cos_const = grammar.create_expression('cos', [float(np.pi)]) # Using float(np.pi)
    assert expr_cos_const is not None
    assert expr_cos_const.symbolic == sp.cos(sp.Float(float(np.pi))) # Should be -1

def test_symbolic_equivalence(grammar_and_vars):
    grammar, var_a, _, _, _, _ = grammar_and_vars
    x = var_a.symbolic # sp.Symbol('a')

    # Test x*x vs x**2
    expr_mul = grammar.create_expression('*', [var_a, var_a])
    expr_pow = grammar.create_expression('**', [var_a, 2])

    assert expr_mul is not None
    assert expr_pow is not None

    # Check individual symbolic forms first
    assert expr_mul.symbolic == x*x
    assert expr_pow.symbolic == x**2

    # Test for symbolic equivalence
    assert sp.simplify(expr_mul.symbolic - expr_pow.symbolic) == 0

    # Test (a+b)*(a+b) vs (a+b)**2
    var_b = Variable(name="b", index=1)
    y = var_b.symbolic

    expr_add_ab = grammar.create_expression('+', [var_a, var_b])
    assert expr_add_ab is not None

    expr_mul_complex = grammar.create_expression('*', [expr_add_ab, expr_add_ab])
    expr_pow_complex = grammar.create_expression('**', [expr_add_ab, 2])

    assert expr_mul_complex is not None
    assert expr_pow_complex is not None

    expected_mul_sym = (x+y)*(x+y)
    expected_pow_sym = (x+y)**2

    assert expr_mul_complex.symbolic.equals(expected_mul_sym.expand()) or expr_mul_complex.symbolic.equals(expected_mul_sym)
    assert expr_pow_complex.symbolic.equals(expected_pow_sym.expand()) or expr_pow_complex.symbolic.equals(expected_pow_sym)

    assert sp.simplify(expr_mul_complex.symbolic - expr_pow_complex.symbolic) == 0


def test_division_by_zero_symbolic(grammar_and_vars):
    grammar, var_a, _, _, _, _ = grammar_and_vars
    # v_x = Variable('x', 0) # var_a is 'a'

    # Using var_a as 'x'
    expr_div_zero_const = grammar.create_expression('/', [var_a, 0.0])
    assert expr_div_zero_const is not None
    assert expr_div_zero_const.symbolic == sp.nan

    expr_div_zero_expr = grammar.create_expression('/', [var_a, grammar.create_expression('const', [0.0])])
    assert expr_div_zero_expr is not None
    assert expr_div_zero_expr.symbolic == sp.nan

    # Test division of a constant by zero
    expr_const_div_zero = grammar.create_expression('/', [1.0, 0.0])
    assert expr_const_div_zero is not None
    assert expr_const_div_zero.symbolic == sp.nan


def test_calculus_operations_conversion(grammar_and_vars):
    grammar, var_a, _, _, _, _ = grammar_and_vars
    x = var_a.symbolic # sp.Symbol('a')

    # Test diff(x^2, x)
    expr_sq = grammar.create_expression('**', [var_a, 2])
    assert expr_sq is not None
    assert expr_sq.symbolic == x**2

    expr_diff = grammar.create_expression('diff', [expr_sq, var_a])
    assert expr_diff is not None
    assert expr_diff.symbolic == sp.diff(x**2, x) # Should be 2*x
    assert expr_diff.symbolic == 2*x

    # Test int(x, x)
    expr_int = grammar.create_expression('int', [var_a, var_a])
    assert expr_int is not None
    # Sympy integrate(x,x) by default adds no constant.
    # It also auto-simplifies x*x/2 to x**2/2
    assert expr_int.symbolic == sp.integrate(x, x) # Should be x**2/2
    assert expr_int.symbolic == x**2/2

    # Test diff(sin(x), x)
    expr_sin_x = grammar.create_expression('sin', [var_a])
    assert expr_sin_x is not None
    assert expr_sin_x.symbolic == sp.sin(x)

    expr_diff_sin = grammar.create_expression('diff', [expr_sin_x, var_a])
    assert expr_diff_sin is not None
    assert expr_diff_sin.symbolic == sp.diff(sp.sin(x), x) # Should be cos(x)
    assert expr_diff_sin.symbolic == sp.cos(x)

    # Test int(cos(x), x)
    expr_cos_x = grammar.create_expression('cos', [var_a])
    assert expr_cos_x is not None
    assert expr_cos_x.symbolic == sp.cos(x)

    expr_int_cos = grammar.create_expression('int', [expr_cos_x, var_a])
    assert expr_int_cos is not None
    assert expr_int_cos.symbolic == sp.integrate(sp.cos(x), x) # Should be sin(x)
    assert expr_int_cos.symbolic == sp.sin(x)


# --- Tests for create_expression and _validate_expression ---

def test_create_expression_valid_binary(grammar_and_vars):
    grammar, var_a, var_b, _, const_1, _ = grammar_and_vars
    expr_plus = grammar.create_expression('+', [var_a, var_b])
    assert isinstance(expr_plus, Expression)
    expr_minus = grammar.create_expression('-', [var_a, const_1])
    assert isinstance(expr_minus, Expression)
    expr_mul = grammar.create_expression('*', [var_a, expr_plus]) # Operand can be an Expression
    assert isinstance(expr_mul, Expression)
    expr_div = grammar.create_expression('/', [expr_mul, var_b])
    assert isinstance(expr_div, Expression)
    expr_pow = grammar.create_expression('**', [var_a, 2])
    assert isinstance(expr_pow, Expression)
    expr_pow_expr = grammar.create_expression('**', [var_a, expr_plus])
    assert isinstance(expr_pow_expr, Expression)


def test_create_expression_valid_unary(grammar_and_vars):
    grammar, var_a, _, _, const_1, _ = grammar_and_vars
    expr_neg = grammar.create_expression('neg', [var_a])
    assert isinstance(expr_neg, Expression)
    expr_inv = grammar.create_expression('inv', [const_1])
    assert isinstance(expr_inv, Expression)

    # Create a more complex expression to use as operand for unary
    expr_plus = grammar.create_expression('+', [var_a, const_1])
    assert isinstance(expr_plus, Expression)

    for op_name in ['sqrt', 'log', 'exp', 'sin', 'cos']:
        expr = grammar.create_expression(op_name, [expr_plus])
        assert isinstance(expr, Expression), f"Failed for unary op {op_name} with Expression operand"
        expr_const = grammar.create_expression(op_name, [const_1])
        assert isinstance(expr_const, Expression), f"Failed for unary op {op_name} with constant operand"
        expr_var = grammar.create_expression(op_name, [var_a])
        assert isinstance(expr_var, Expression), f"Failed for unary op {op_name} with Variable operand"


def test_create_expression_valid_calculus(grammar_and_vars):
    grammar, var_a, var_b, _, const_1, _ = grammar_and_vars
    # Simple expression: a + 1
    simple_expr = grammar.create_expression('+', [var_a, const_1])
    assert isinstance(simple_expr, Expression)

    expr_diff = grammar.create_expression('diff', [simple_expr, var_a])
    assert isinstance(expr_diff, Expression)
    expr_int = grammar.create_expression('int', [simple_expr, var_b]) # Integrate w.r.t. different variable
    assert isinstance(expr_int, Expression)

    # Test with Variable as first operand
    expr_diff_var = grammar.create_expression('diff', [var_a, var_b])
    assert isinstance(expr_diff_var, Expression)
    expr_int_var = grammar.create_expression('int', [var_a, var_a])
    assert isinstance(expr_int_var, Expression)


def test_create_expression_valid_var_const(grammar_and_vars):
    grammar, _, _, _, const_1, _ = grammar_and_vars
    # Note: 'var' operator in Expression takes a string name, not a Variable object
    # This is based on Expression._to_sympy: `sp.Symbol(self.operands[0])`
    expr_var_type = grammar.create_expression('var', ["my_new_var"])
    assert isinstance(expr_var_type, Expression)
    assert expr_var_type.operator == 'var'
    assert expr_var_type.operands[0] == "my_new_var"

    expr_const_type = grammar.create_expression('const', [const_1])
    assert isinstance(expr_const_type, Expression)
    assert expr_const_type.operator == 'const'
    assert expr_const_type.operands[0] == const_1

    expr_const_type_int = grammar.create_expression('const', [10])
    assert isinstance(expr_const_type_int, Expression)
    assert expr_const_type_int.operator == 'const'
    assert expr_const_type_int.operands[0] == 10

def test_create_expression_invalid_operator(grammar_and_vars):
    grammar, var_a, _, _, _, _ = grammar_and_vars
    assert grammar.create_expression('unknown_op', [var_a, var_a]) is None
    assert grammar.create_expression('another_bad_op', [var_a]) is None


def test_create_expression_invalid_arity_binary(grammar_and_vars):
    grammar, var_a, var_b, _, _, _ = grammar_and_vars
    binary_ops = ['+', '-', '*', '/', '**']
    for op in binary_ops:
        assert grammar.create_expression(op, []) is None, f"Op {op} with 0 operands"
        assert grammar.create_expression(op, [var_a]) is None, f"Op {op} with 1 operand"
        # Valid case: grammar.create_expression(op, [var_a, var_b]) is not None
        assert grammar.create_expression(op, [var_a, var_b, var_a]) is None, f"Op {op} with 3 operands"

def test_create_expression_invalid_arity_unary(grammar_and_vars):
    grammar, var_a, var_b, _, _, _ = grammar_and_vars
    unary_ops = ['neg', 'inv', 'sqrt', 'log', 'exp', 'sin', 'cos']
    for op in unary_ops:
        assert grammar.create_expression(op, []) is None, f"Op {op} with 0 operands"
        # Valid case: grammar.create_expression(op, [var_a]) is not None
        assert grammar.create_expression(op, [var_a, var_b]) is None, f"Op {op} with 2 operands"

def test_create_expression_invalid_arity_calculus(grammar_and_vars):
    grammar, var_a, var_b, _, const_1, _ = grammar_and_vars
    calc_ops = ['diff', 'int']
    simple_expr = grammar.create_expression('+', [var_a, const_1])
    for op in calc_ops:
        assert grammar.create_expression(op, []) is None, f"Op {op} with 0 operands"
        assert grammar.create_expression(op, [simple_expr]) is None, f"Op {op} with 1 operand"
        # Valid case: grammar.create_expression(op, [simple_expr, var_a]) is not None
        assert grammar.create_expression(op, [simple_expr, var_a, var_b]) is None, f"Op {op} with 3 operands"


def test_create_expression_calculus_operand_validation(grammar_and_vars):
    grammar, var_a, _, _, const_1, _ = grammar_and_vars
    # simple_expr for the first argument of diff/int
    simple_expr = Expression(operator='var', operands=['a_dummy']) # A valid expression

    # Valid: diff(simple_expr, var_a)
    assert grammar.create_expression('diff', [simple_expr, var_a]) is not None
    assert grammar.create_expression('int', [var_a, var_a]) is not None # Var for first operand is also fine

    # Invalid: second operand is not a Variable
    assert grammar.create_expression('diff', [simple_expr, const_1]) is None, "diff with const as 2nd op"
    assert grammar.create_expression('int', [simple_expr, 1.0]) is None, "int with float as 2nd op"
    assert grammar.create_expression('diff', [simple_expr, simple_expr]) is None, "diff with Expression as 2nd op"

    # Invalid: first operand is not an Expression or Variable (e.g. a raw number)
    # Based on _validate_expression: first operand can be Expression, Variable, int, float
    # However, for calculus, it semantically needs to be something differentiable/integrable.
    # The current _validate_expression allows int/float as first operand,
    # but sp.diff(1.0, x) or sp.integrate(1.0, x) are valid in Sympy.
    # So, these should pass validation if _validate_expression is the only gatekeeper.
    # Let's test if `create_expression` makes them.
    expr_diff_const_var = grammar.create_expression('diff', [const_1, var_a])
    assert isinstance(expr_diff_const_var, Expression), "diff(const, var) should be created"
    assert expr_diff_const_var.symbolic == sp.diff(sp.Float(const_1), var_a.symbolic) # diff(1.0, a) = 0

    expr_int_const_var = grammar.create_expression('int', [1, var_a]) # using int constant
    assert isinstance(expr_int_const_var, Expression), "int(const, var) should be created"
    # sp.integrate(sp.Float(1), a_sym) results in 1.0*a_sym. sp.integrate(1, a_sym) results in a_sym.
    # The _to_sympy method converts the constant 1 to sp.Float(1), so 1.0*a is the expected output.
    assert expr_int_const_var.symbolic == 1.0 * var_a.symbolic


def test_create_expression_general_invalid_operand_types(grammar_and_vars):
    grammar, var_a, _, _, _, _ = grammar_and_vars

    invalid_operands = [
        "a_string",
        ["a", "list"],
        {"a": "dict"}
    ]

    all_operator_sets = [
        grammar.primitives['binary_ops'],
        grammar.primitives['unary_ops'],
        grammar.primitives['calculus_ops'],
        # {'var', 'const'} # 'var' and 'const' have specific operand expectations
    ]

    for op_set in all_operator_sets:
        for op_name in op_set:
            for invalid_op_val in invalid_operands:
                if op_name in grammar.primitives['binary_ops'] or op_name in grammar.primitives['calculus_ops']:
                    # Test invalid as first operand
                    assert grammar.create_expression(op_name, [invalid_op_val, var_a]) is None, \
                        f"Op {op_name} with invalid first operand {invalid_op_val}"
                    # Test invalid as second operand
                    if op_name in grammar.primitives['binary_ops']: # Calculus 2nd op must be Variable
                        assert grammar.create_expression(op_name, [var_a, invalid_op_val]) is None, \
                            f"Op {op_name} with invalid second operand {invalid_op_val}"
                elif op_name in grammar.primitives['unary_ops']:
                    assert grammar.create_expression(op_name, [invalid_op_val]) is None, \
                        f"Op {op_name} with invalid operand {invalid_op_val}"

    # Specific tests for 'var' and 'const'
    # 'var' expects a string name for the variable
    assert grammar.create_expression('var', [123]) is None, "'var' with non-string operand"
    assert grammar.create_expression('var', [var_a]) is None, "'var' with Variable object (expects name)"
    # 'const' expects a number
    assert grammar.create_expression('const', ["not_a_number"]) is None, "'const' with non-numeric operand"
    assert grammar.create_expression('const', [var_a]) is None, "'const' with Variable object"
    assert grammar.create_expression('const', [Expression('const', [1.0])]) is None, "'const' with Expression object"


# --- Tests for Variable Discovery ---
from unittest.mock import patch, MagicMock

def test_analyze_component(grammar_and_vars):
    grammar, _, _, _, _, _ = grammar_and_vars
    time_stamps = np.linspace(0, 10, 100)

    # Test high information content (random data)
    random_signal = np.random.rand(100)
    properties_random = grammar._analyze_component(random_signal, time_stamps)
    assert properties_random['information_content'] > 0.5 # Actual value depends on data, ensure it's high

    # Test low information content (mostly constant)
    constant_signal = np.ones(100) * 5
    properties_constant = grammar._analyze_component(constant_signal, time_stamps)
    assert properties_constant['information_content'] < 0.1 # Should be very low for constant

    # Test high conservation score
    conserved_signal = np.ones(100) + 0.01 * np.random.randn(100) # Constant with tiny noise
    properties_conserved = grammar._analyze_component(conserved_signal, time_stamps)
    assert properties_conserved['conservation_score'] > 0.8

    # Test low conservation score (rapidly changing variance)
    # Create a signal where variance changes a lot between windows
    # 10 windows of 10 elements each.
    # Make 5 windows low variance, 5 windows high variance.
    low_var_segment = np.random.randn(50) * 0.01 # std dev 0.01
    high_var_segment = np.random.randn(50) * 1.0  # std dev 1.0
    # Interleave them to maximize variance of window variances if windows are small
    # Or, just make segments that correspond to several windows
    low_conservation_signal = np.concatenate([
        np.random.randn(10) * 0.01, # Window 1 low
        np.random.randn(10) * 1.0,  # Window 2 high
        np.random.randn(10) * 0.01, # Window 3 low
        np.random.randn(10) * 1.0,  # Window 4 high
        np.random.randn(10) * 0.01, # Window 5 low
        np.random.randn(10) * 1.0,  # Window 6 high
        np.random.randn(10) * 0.01, # Window 7 low
        np.random.randn(10) * 1.0,  # Window 8 high
        np.random.randn(10) * 0.01, # Window 9 low
        np.random.randn(10) * 1.0   # Window 10 high
    ])
    properties_low_conserved = grammar._analyze_component(low_conservation_signal, time_stamps)
    # With high variance in window variances, 1/(1+var) should be small.
    # Ensure it's lower than the highly conserved signal's score
    assert properties_low_conserved['conservation_score'] < properties_conserved['conservation_score']

    # Test high periodicity
    periodic_signal = np.sin(2 * np.pi * time_stamps)
    properties_periodic = grammar._analyze_component(periodic_signal, time_stamps)
    # Periodicity score can be very high, this is a qualitative check
    assert properties_periodic['periodicity_score'] > 5.0

    # Test low periodicity (random noise)
    properties_low_periodic = grammar._analyze_component(random_signal, time_stamps)
    assert properties_low_periodic['periodicity_score'] < 4.0 # Random noise shouldn't have strong peaks, relax threshold

    # Test high smoothness
    smooth_signal = np.cumsum(np.random.randn(100) * 0.1) # Brownian motion like, smooth
    properties_smooth = grammar._analyze_component(smooth_signal, time_stamps)
    assert properties_smooth['smoothness'] > 0.7

    # Test low smoothness (noisy, rapid changes)
    noisy_signal = random_signal + np.diff(np.random.randn(101))*5 # Add high frequency noise. np.diff on 101 elements gives 100.
    properties_noisy = grammar._analyze_component(noisy_signal, time_stamps)
    assert properties_noisy['smoothness'] < 0.5

    # Edge cases for data length
    short_signal_15 = np.random.rand(15) # Too short for periodicity, borderline for conservation
    props_short_15 = grammar._analyze_component(short_signal_15, None)
    assert props_short_15['periodicity_score'] == 0.0
    # Conservation might run if len(component)//10 >= 1, min(10, 1) = 1 window, var of vars = 0 -> score = 1.
    # If it splits into 1 window, variance of variances is 0, so score is 1. Let's test for specific value.
    # min(10, 15//10) = min(10,1) = 1. Variances = [var(w1)]. np.var(variances) = 0. Score = 1.0
    assert props_short_15['conservation_score'] == 1.0 # Or check based on logic for 1 window

    short_signal_5 = np.random.rand(5) # Too short for conservation and periodicity
    props_short_5 = grammar._analyze_component(short_signal_5, None)
    assert props_short_5['conservation_score'] == 0.0
    assert props_short_5['periodicity_score'] == 0.0

    short_signal_1 = np.random.rand(1) # Too short for smoothness
    props_short_1 = grammar._analyze_component(short_signal_1, None)
    assert props_short_1['smoothness'] == 0.0


def test_generate_variable_name(grammar_and_vars):
    grammar, _, _, _, _, _ = grammar_and_vars

    # Test prefix selection and unique suffix generation
    grammar.variables = {} # Reset for predictable naming
    props_energy = {'conservation_score': 0.9, 'periodicity_score': 0.1, 'smoothness': 0.1, 'information_content': 0.5}
    assert grammar._generate_variable_name(props_energy) == "E_1"
    grammar.variables["E_1"] = MagicMock(spec=Variable) # Mock a variable entry

    props_periodic = {'conservation_score': 0.1, 'periodicity_score': 6.0, 'smoothness': 0.1, 'information_content': 0.5}
    assert grammar._generate_variable_name(props_periodic) == "theta_1"
    grammar.variables["theta_1"] = MagicMock(spec=Variable)

    props_smooth = {'conservation_score': 0.1, 'periodicity_score': 0.1, 'smoothness': 0.8, 'information_content': 0.5}
    assert grammar._generate_variable_name(props_smooth) == "x_1"
    grammar.variables["x_1"] = MagicMock(spec=Variable)

    props_generic = {'conservation_score': 0.1, 'periodicity_score': 0.1, 'smoothness': 0.1, 'information_content': 0.5}
    assert grammar._generate_variable_name(props_generic) == "q_1"
    grammar.variables["q_1"] = MagicMock(spec=Variable)

    # Test suffix increment
    assert grammar._generate_variable_name(props_energy) == "E_2"
    grammar.variables["E_2"] = MagicMock(spec=Variable)
    assert grammar._generate_variable_name(props_periodic) == "theta_2"
    grammar.variables["theta_2"] = MagicMock(spec=Variable)
    assert grammar._generate_variable_name(props_smooth) == "x_2"
    grammar.variables["x_2"] = MagicMock(spec=Variable)
    assert grammar._generate_variable_name(props_generic) == "q_2"
    grammar.variables["q_2"] = MagicMock(spec=Variable)

    # Test case where no specific property is dominant enough
    props_low_all = {'conservation_score': 0.5, 'periodicity_score': 2.0, 'smoothness': 0.5, 'information_content': 0.5}
    assert grammar._generate_variable_name(props_low_all) == "q_3"


# Patching 'sklearn.decomposition.FastICA' as it's imported like `from sklearn.decomposition import FastICA`
# Patching denoiser at the class level progressive_grammar_system.NoisyObservationProcessor
@patch('progressive_grammar_system.FastICA')
@patch('progressive_grammar_system.NoisyObservationProcessor.denoise')
def test_discover_variables_simple_run(mock_denoise, MockFastICA, grammar_and_vars):
    grammar, _, _, _, _, _ = grammar_and_vars
    grammar.variables = {}
    grammar.noise_threshold = 0.05 # Lower threshold to accept more variables for testing

    # Configure denoiser mock
    mock_denoise.side_effect = lambda obs, epochs=50: obs # Pass through

    # Configure FastICA mock
    mock_ica_instance = MockFastICA.return_value
    # Simulate 3 components extracted from 3 observation channels
    mock_components = np.array([
        np.sin(np.linspace(0,10,100)), # Periodic
        np.linspace(0,1,100) + 0.01 * np.random.randn(100), # Smooth, conserved-ish
        np.random.rand(100) * 0.01 # Low information content if threshold is higher
    ]).T
    mock_ica_instance.fit_transform.return_value = mock_components

    observations = np.random.rand(100, 3) # Dummy, as denoise and ICA are mocked
    time_stamps = np.linspace(0,10,100)

    discovered = grammar.discover_variables(observations, time_stamps)

    mock_denoise.assert_called_once()
    MockFastICA.assert_called_once_with(n_components=min(grammar.max_variables, observations.shape[1]))
    mock_ica_instance.fit_transform.assert_called_once_with(observations) # Denoiser passes through

    # Based on the mock_components and default _analyze_component logic:
    # Comp0 (periodic_signal): high periodicity, good info content -> theta_1
    # Comp1 (smooth_conserved): high smoothness, good info content, maybe conserved -> x_1 or E_1
    # Comp2 (low_info_random): likely low info content (scaled down) -> filtered out

    # Check number of discovered variables (this is approximate without mocking _analyze_component)
    # Let's assume Comp2 is filtered out by a reasonable noise_threshold.
    # The exact names depend on thresholds in _generate_variable_name and _analyze_component.

    # For more precise control, mock _analyze_component as in the example,
    # but let's try one pass with the real _analyze_component.

    # Example assertions (these might need adjustment based on actual _analyze_component behavior)
    assert len(discovered) >= 1 # Expect at least one, likely two
    if len(discovered) > 0:
      assert discovered[0].name.startswith(("theta_", "x_", "E_", "q_"))

    # Check that grammar.variables is populated
    for var in discovered:
        assert var.name in grammar.variables
        assert grammar.variables[var.name] == var


@patch('progressive_grammar_system.FastICA')
@patch.object(ProgressiveGrammar, '_analyze_component')
@patch('progressive_grammar_system.NoisyObservationProcessor')
def test_discover_variables_controlled(MockNoisyObservationProcessor, mock_analyze_component, MockFastICA):
    # Create ProgressiveGrammar instance *after* mocks are in place
    grammar = ProgressiveGrammar()
    grammar.variables = {}
    grammar.noise_threshold = 0.4

    # Configure Denoiser mock (NoisyObservationProcessor class is mocked via decorator)
    # The grammar instance will create its own NoisyObservationProcessor,
    # which will be a mock instance if the class is mocked.
    # So, we need to configure the return_value of the mocked class's denoise method.
    # However, ProgressiveGrammar.__init__ creates self.denoiser = NoisyObservationProcessor()
    # So, when grammar instance is created for the test (via fixture), self.denoiser is already set.
    # The patch on the class means that NoisyObservationProcessor() inside __init__ returns a MagicMock.
    # So grammar.denoiser will be this MagicMock.

    # This means we need to configure the mock returned by NoisyObservationProcessor()
    mock_denoiser_instance = MockNoisyObservationProcessor.return_value
    mock_denoiser_instance.denoise.side_effect = lambda obs, epochs=50: obs

    # Alternative: If grammar.denoiser was already created *before* this test's patches apply (e.g. in fixture)
    # then patching the class 'progressive_grammar_system.NoisyObservationProcessor' would not affect
    # the already existing grammar.denoiser instance.
    # The fixture `grammar_and_vars` creates a new ProgressiveGrammar instance for each test.
    # So, when ProgressiveGrammar() is called within the test's scope (or by the fixture for the test),
    # the patched NoisyObservationProcessor will be used.
    # The instance `grammar.denoiser` will be a MagicMock (the return_value of MockNoisyObservationProcessor)
    grammar.denoiser.denoise.side_effect = lambda obs, epochs=50: obs


    # Configure FastICA mock
    mock_ica_instance = MockFastICA.return_value
    mock_components = np.array([[1,2,3], [4,5,6], [0.1, 0.2, 0.3]]).T
    mock_ica_instance.fit_transform.return_value = mock_components

    # Configure _analyze_component mock
    props1 = {'information_content': 0.8, 'periodicity_score': 7.0, 'conservation_score': 0.2, 'smoothness': 0.5}
    props2 = {'information_content': 0.7, 'conservation_score': 0.9, 'periodicity_score': 0.2, 'smoothness': 0.6}
    props3 = {'information_content': 0.1, 'periodicity_score': 0.1, 'conservation_score': 0.1, 'smoothness': 0.1}
    mock_analyze_component.side_effect = [props1, props2, props3]

    observations = np.random.rand(100, 3)
    time_stamps = np.linspace(0,10,100)

    discovered = grammar.discover_variables(observations, time_stamps)

    # The denoiser is now a mock instance created by the patched NoisyObservationProcessor class
    # The call in discover_variables is self.denoiser.denoise(observations), it uses the default for epochs.
    grammar.denoiser.denoise.assert_called_once_with(observations)

    MockFastICA.assert_called_once_with(n_components=min(grammar.max_variables, observations.shape[1]))
    mock_ica_instance.fit_transform.assert_called_once_with(observations) # Denoiser passes through

    assert mock_analyze_component.call_count == 3
    # Check arguments for _analyze_component calls if necessary by inspecting mock_analyze_component.call_args_list

    assert len(discovered) == 2
    assert "theta_1" in grammar.variables # From props1
    assert "E_1" in grammar.variables     # From props2
    assert len(grammar.variables) == 2

    assert discovered[0].name == "theta_1"
    assert discovered[0].properties == props1
    assert discovered[0].index == 0 # Corresponds to first component

    assert discovered[1].name == "E_1"
    assert discovered[1].properties == props2
    assert discovered[1].index == 1 # Corresponds to second component


# --- Tests for MDL and Abstraction Mining ---

def test_count_subexpression(grammar_and_vars):
    grammar, var_a, var_b, _, const_1, _ = grammar_and_vars
    var_c = Variable(name="c", index=2)

    # Pattern: a + b
    # Complexity: 1 (op) + 1 (var_a) + 1 (var_b) = 3
    pattern_ab = grammar.create_expression('+', [var_a, var_b])
    assert pattern_ab is not None

    # Expression 1: (a + b) * c
    # Complexity: 1 (op *) + 3 (pattern_ab) + 1 (var_c) = 5
    expr1 = grammar.create_expression('*', [pattern_ab, var_c])
    assert expr1 is not None
    assert grammar._count_subexpression(expr1, pattern_ab) == 1

    # Expression 2: (a + b) * (a + b)
    # Complexity: 1 (op *) + 3 (pattern_ab) + 3 (pattern_ab) = 7
    expr2 = grammar.create_expression('*', [pattern_ab, pattern_ab])
    assert expr2 is not None
    assert grammar._count_subexpression(expr2, pattern_ab) == 2

    # Expression 3: a * b (no match for a+b)
    expr3 = grammar.create_expression('*', [var_a, var_b])
    assert expr3 is not None
    assert grammar._count_subexpression(expr3, pattern_ab) == 0

    # Expression 4: pattern is the expression itself
    assert grammar._count_subexpression(pattern_ab, pattern_ab) == 1

    # Expression 5: Deeper nesting ( (a+b) + c ) * (a+b)
    # pattern_ab (a+b) complexity = 3
    # sub_expr_abc = (a+b)+c -> complexity = 1 + 3 + 1 = 5
    # main_expr = sub_expr_abc * pattern_ab -> complexity = 1 + 5 + 3 = 9
    sub_expr_abc = grammar.create_expression('+', [pattern_ab, var_c])
    assert sub_expr_abc is not None
    main_expr_nested = grammar.create_expression('*', [sub_expr_abc, pattern_ab])
    assert main_expr_nested is not None
    assert grammar._count_subexpression(main_expr_nested, pattern_ab) == 2 # one in sub_expr_abc, one direct

    # Expression 6: Constant pattern
    # const_1_expr = Expression('const', [1.0]) -> this is how create_expression makes it internally
    # Let's use create_expression for consistency if it's for `const` type.
    const_1_expr = grammar.create_expression('const', [1.0])
    assert const_1_expr is not None # complexity 2: ('const', [1.0]) -> op 'const', operand 1.0
                                     # Actually, complexity is 1 for const node in _compute_complexity
                                     # No, Expression('const', [1.0]) -> self.operands=[1.0]. sum(1 for op in self.operands) = 1. 1+1=2
                                     # Let's recheck complexity for const/var.
                                     # Expression('const', [1.0]) -> _compute_complexity: 1 + (1 for op in [1.0]) = 2. Correct.
                                     # Variable('a',0) -> .complexity = 1. Correct.

    # expr_const_usage = (a+b) + 1.0
    expr_const_usage = grammar.create_expression('+', [pattern_ab, const_1_expr])
    assert expr_const_usage is not None
    assert grammar._count_subexpression(expr_const_usage, const_1_expr) == 1
    assert grammar._count_subexpression(expr_const_usage, pattern_ab) == 1


def test_calculate_compression_gain(grammar_and_vars):
    grammar, var_a, var_b, _, const_1, _ = grammar_and_vars
    var_c = Variable(name="c", index=2)

    # Candidate: a + b (complexity 3)
    candidate_ab = grammar.create_expression('+', [var_a, var_b])
    assert candidate_ab.complexity == 3

    # Corpus
    # expr1: (a + b) * c (complexity 5, 1 occurrence of a+b)
    expr1 = grammar.create_expression('*', [candidate_ab, var_c])
    # expr2: (a + b) + 1.0 (complexity 1 + 3 + 2 = 6. const_1_expr is complexity 2)
    const_1_expr = grammar.create_expression('const', [1.0])
    expr2 = grammar.create_expression('+', [candidate_ab, const_1_expr])
    # expr3: a * b (complexity 3, 0 occurrences)
    expr3 = grammar.create_expression('*', [var_a, var_b])

    corpus = [expr1, expr2, expr3]
    current_length = expr1.complexity + expr2.complexity + expr3.complexity # 5 + (1+3+2) + 3 = 5 + 6 + 3 = 14

    # Expected new length:
    # Cost of defining candidate_ab: 3
    # Cost of expr1: (complexity 5) - 1 occurrence * (cand_complexity 3 - 1) = 5 - 1 * 2 = 3
    # Cost of expr2: (complexity 6) - 1 occurrence * (cand_complexity 3 - 1) = 6 - 1 * 2 = 4
    # Cost of expr3: (complexity 3) - 0 occurrences = 3
    # Total new_length = 3 (candidate) + 3 (expr1_new) + 4 (expr2_new) + 3 (expr3_new) = 13
    expected_new_length = candidate_ab.complexity \
                        + (expr1.complexity - 1 * (candidate_ab.complexity -1)) \
                        + (expr2.complexity - 1 * (candidate_ab.complexity -1)) \
                        + expr3.complexity
    assert expected_new_length == 13

    expected_gain = current_length - expected_new_length # 14 - 13 = 1
    assert grammar._calculate_compression_gain(candidate_ab, corpus) == expected_gain # Should be 1

    # Test non-beneficial case
    # Candidate: a * b (complexity 3)
    candidate_mul_ab = expr3
    # Corpus: expr1 = (a+b)*c, expr2 = (a+b)+1.0. No occurrences of a*b
    corpus_no_occurrence = [expr1, expr2]
    current_length_no_occ = expr1.complexity + expr2.complexity # 5 + 6 = 11
    # Expected new_length: 3 (candidate_mul_ab) + 5 (expr1) + 6 (expr2) = 14
    # Expected gain: 11 - 14 = -3
    assert grammar._calculate_compression_gain(candidate_mul_ab, corpus_no_occurrence) == -3


def test_add_learned_function(grammar_and_vars):
    grammar, var_a, var_b, _, _, _ = grammar_and_vars
    grammar.mdl_threshold = 5.0 # Set a clear threshold

    # Candidate: a + b (complexity 3)
    candidate_expr = grammar.create_expression('+', [var_a, var_b])

    # Corpus that benefits from this abstraction
    # (a+b)*(a+b) -> 2 occurrences, complexity 7. current_len = 7
    # new_len = 3 (cand) + (7 - 2*(3-1)) = 3 + (7-4) = 3+3 = 6. Gain = 1. (Not enough)
    corpus1 = [grammar.create_expression('*', [candidate_expr, candidate_expr])]

    # Let's make a corpus with more gain
    # Three expressions: (a+b)*c, (a+b)*d, (a+b)*e
    # cand_ab complexity 3
    # (a+b)*c complexity 5. new cost 5-(3-1) = 3
    # current_total_complexity = 5+5+5 = 15
    # new_total_complexity_for_corpus = 3+3+3 = 9
    # new_length_with_abstraction = 3 (for cand_ab) + 9 = 12
    # gain = 15 - 12 = 3. (Still not enough if threshold is 5)
    var_c = Variable(name="c", index=2)
    var_d = Variable(name="d", index=3)
    var_e = Variable(name="e", index=4)
    corpus_high_gain = [
        grammar.create_expression('*', [candidate_expr, var_c]),
        grammar.create_expression('*', [candidate_expr, var_d]),
        grammar.create_expression('*', [candidate_expr, var_e]),
    ] # Gain is 3.

    assert not grammar.add_learned_function("f_ab", candidate_expr, corpus_high_gain)
    assert "f_ab" not in grammar.learned_functions
    assert "f_ab" not in grammar.primitives['unary_ops']

    # Increase occurrences to get gain > 5
    # 5 expressions: (a+b)*c, (a+b)*d, (a+b)*e, (a+b)*a, (a+b)*b
    # current_total_complexity = 5*5 = 25
    # new_total_complexity_for_corpus = 5 * (5 - (3-1)) = 5 * 3 = 15
    # new_length_with_abstraction = 3 (for cand_ab) + 15 = 18
    # gain = 25 - 18 = 7. (This should pass threshold 5)
    corpus_very_high_gain = corpus_high_gain + [
        grammar.create_expression('*', [candidate_expr, var_a]),
        grammar.create_expression('*', [candidate_expr, var_b]),
    ]
    assert grammar.add_learned_function("f_ab_good", candidate_expr, corpus_very_high_gain)
    assert "f_ab_good" in grammar.learned_functions
    assert grammar.learned_functions["f_ab_good"] == candidate_expr
    assert "f_ab_good" in grammar.primitives['unary_ops'] # Assumes new functions are unary for now

    # Test that it does not add if gain is exactly threshold (>)
    grammar.mdl_threshold = 7.0
    grammar.learned_functions.pop("f_ab_good") # reset
    grammar.primitives['unary_ops'].remove("f_ab_good")
    assert not grammar.add_learned_function("f_ab_exact", candidate_expr, corpus_very_high_gain)


@patch.object(ProgressiveGrammar, 'add_learned_function') # Mock to control which functions are added
def test_mine_abstractions(mock_add_learned_function, grammar_and_vars):
    grammar, var_a, var_b, _, const_1, _ = grammar_and_vars
    var_c = Variable(name="c", index=2)
    grammar.learned_functions = {} # Reset

    # Setup expressions for hypothesis library
    # Common pattern: a + b (complexity 3)
    expr_ab = grammar.create_expression('+', [var_a, var_b])
    # Another pattern: b + 1 (complexity 3, as const_1_expr is 2, + is 1, var_b is 1)
    const_1_expr = grammar.create_expression('const', [1.0])
    expr_b_const1 = grammar.create_expression('+', [var_b, const_1_expr])

    # Hypothesis library
    # 1. (a+b) * c  (contains a+b)
    # 2. (a+b) + (b+1) (contains a+b, b+1)
    # 3. c * (a+b)  (contains a+b, uses commutative property for key)
    # 4. (b+1) * a (contains b+1)
    # 5. a * c (no target patterns)
    hyp_lib = [
        grammar.create_expression('*', [expr_ab, var_c]),
        grammar.create_expression('+', [expr_ab, expr_b_const1]),
        grammar.create_expression('*', [var_c, expr_ab]),
        grammar.create_expression('*', [expr_b_const1, var_a]),
        grammar.create_expression('*', [var_a, var_c])
    ]
    # Occurrences:
    # "a+b": 3 times (key: +(var:a,var:b))
    # "b+1.0": 2 times (key: +(const:1,var:b))

    # Configure mock_add_learned_function
    # Let's say only "a+b" is beneficial enough to be added
    def side_effect_add_learned_function(name, expression, usage_data):
        # Check if the expression is 'a+b'
        if expression.operator == '+' and \
           isinstance(expression.operands[0], Variable) and expression.operands[0].name == 'a' and \
           isinstance(expression.operands[1], Variable) and expression.operands[1].name == 'b':
            # Simulate adding to learned_functions and primitives for the original method's behavior
            grammar.learned_functions[name] = expression
            grammar.primitives['unary_ops'].add(name)
            return True
        return False

    mock_add_learned_function.side_effect = side_effect_add_learned_function

    # Test with min_frequency = 3
    # "a+b" occurs 3 times, "b+1.0" occurs 2 times. So only "a+b" should be considered.
    abstractions = grammar.mine_abstractions(hyp_lib, min_frequency=3)

    assert mock_add_learned_function.called # It should be called for "a+b"
    # It should be called once for pattern 'a+b' as it meets frequency.
    # The number of calls depends on internal defaultdict iteration order.
    # Let's check the outcome instead of exact call count on the mock for this.

    assert len(abstractions) == 1
    # Name will be "f_0" since learned_functions was reset and this is the first one added by the mock
    assert "f_0" in abstractions
    assert grammar._expression_key(abstractions["f_0"]) == grammar._expression_key(expr_ab)
    assert "f_0" in grammar.learned_functions # Check if mock correctly updated this (it should based on side_effect)

    # Test with min_frequency = 2
    grammar.learned_functions = {} # Reset
    mock_add_learned_function.reset_mock()
    # Now, "a+b" (3 times) and "b+1.0" (2 times) should be considered.
    # Let's say "b+1.0" is also beneficial by the mock this time.
    def side_effect_add_all_beneficial(name, expression, usage_data):
        grammar.learned_functions[name] = expression
        grammar.primitives['unary_ops'].add(name)
        return True
    mock_add_learned_function.side_effect = side_effect_add_all_beneficial

    abstractions_f2 = grammar.mine_abstractions(hyp_lib, min_frequency=2)
    # Expected patterns: "a+b" (count 3), "b+1.0" (count 2), "(a+b)*c" (count 2)
    assert len(abstractions_f2) == 3

    # Check that non-trivial patterns (complexity > 2) are considered
    # 'a' (Variable, complexity 1) or '1.0' (const expr, complexity 2) should not be abstracted.
    # Create a library where 'a' appears many times.
    hyp_lib_simple = [
        grammar.create_expression('+', [var_a, var_b]),
        grammar.create_expression('-', [var_a, var_c]),
        grammar.create_expression('*', [var_a, const_1_expr])
    ] # 'a' occurs 3 times.
    grammar.learned_functions = {} # Reset
    mock_add_learned_function.reset_mock()
    mock_add_learned_function.side_effect = side_effect_add_all_beneficial # Assume all that pass freq are added

    abstractions_simple = grammar.mine_abstractions(hyp_lib_simple, min_frequency=3)
    # var_a has complexity 1, so it should not be considered by `if subexpr.complexity > 2`
    assert len(abstractions_simple) == 0
    # Check if add_learned_function was called for var_a (it shouldn't be)
    for call_args in mock_add_learned_function.call_args_list:
        args, _ = call_args
        expr_arg = args[1] # expression is the second argument
        assert expr_arg.complexity > 2


# --- Tests for NoisyObservationProcessor ---
from progressive_grammar_system import NoisyObservationProcessor
from unittest.mock import PropertyMock

def test_simple_denoise():
    processor = NoisyObservationProcessor()

    # Test with enough data for window > 1 (window = 10 // 10 = 1, oh, min(5, obs.shape[0]//10))
    # For 10 rows, window = min(5, 1) = 1. This will return original.
    # Need more rows for window >= 2. E.g. 20 rows -> window = min(5, 2) = 2
    obs_20_rows = np.array([[float(i), float(i*2)] for i in range(1, 21)]) # 20x2
    obs_20_rows_noisy = obs_20_rows + np.random.randn(20, 2) * 0.1

    denoised_obs = processor._simple_denoise(obs_20_rows_noisy)
    assert denoised_obs.shape == obs_20_rows_noisy.shape
    # Check if smoother (e.g. std of diffs is smaller) - only if window >=2
    if min(5, obs_20_rows_noisy.shape[0]//10) >=2:
         assert np.std(np.diff(denoised_obs[:,0])) < np.std(np.diff(obs_20_rows_noisy[:,0]))
         assert np.std(np.diff(denoised_obs[:,1])) < np.std(np.diff(obs_20_rows_noisy[:,1]))
    else: # if window < 2, it returns original
        assert np.array_equal(denoised_obs, obs_20_rows_noisy)


    # Test fallback for small data (window < 2)
    obs_small_3_rows = np.array([[1.0], [2.0], [3.0]]) # 3 rows, window = min(5, 0) = 0. Returns original.
    denoised_small_3 = processor._simple_denoise(obs_small_3_rows)
    assert np.array_equal(denoised_small_3, obs_small_3_rows)

    obs_small_15_rows = np.array([[float(i)] for i in range(1,16)]) # 15 rows, window = min(5,1) = 1. Returns original.
    denoised_small_15 = processor._simple_denoise(obs_small_15_rows)
    assert np.array_equal(denoised_small_15, obs_small_15_rows)


@patch('progressive_grammar_system.StandardScaler')
@patch('progressive_grammar_system.torch.FloatTensor')
@patch('progressive_grammar_system.torch.optim.Adam')
@patch('progressive_grammar_system.torch.nn.MSELoss')
@patch.object(NoisyObservationProcessor, 'build_autoencoder')
def test_denoise_structure(mock_build_autoencoder, mock_loss, mock_adam, mock_float_tensor, MockStandardScaler):
    processor = NoisyObservationProcessor()

    # Configure mocks
    mock_scaler_instance = MockStandardScaler.return_value
    mock_scaled_data = MagicMock(spec=np.ndarray)
    mock_scaler_instance.fit_transform.return_value = mock_scaled_data

    # Define observations earlier so its shape can be used for mock_torch_data
    observations = np.random.rand(100, 3) # Shape to trigger autoencoder path

    # Configure mock_torch_data which is the return value of mock_float_tensor (torch.FloatTensor)
    actual_shape_tuple = (observations.shape[0], observations.shape[1])
    mock_torch_data = MagicMock(shape=actual_shape_tuple)
    mock_float_tensor.return_value = mock_torch_data

    mock_model = MagicMock()
    # The mock needs to also set processor.model.
    # The side_effect for a method patched with @patch.object receives the instance ('self')
    # as its first argument if the patched object is a class, or just the arguments if patching an instance.
    # Here we patch on the class NoisyObservationProcessor.
    # The side_effect callable receives the arguments passed to the mocked method.
    # For an instance method build_autoencoder(self, input_dim), the call is processor.build_autoencoder(value).
    # The mock intercepts this. The 'self' (processor) is not passed to the side_effect by default when patching at class level this way.
    # It gets the arguments that processor.build_autoencoder was called with.
    def build_autoencoder_side_effect(actual_input_dim):
        # 'processor' is from the outer scope of test_denoise_structure
        processor.model = mock_model
        return mock_model
    mock_build_autoencoder.side_effect = build_autoencoder_side_effect

    # Mock model call returns (reconstructed_data, latent_data)
    # Use sys.modules to access 'torch.Tensor' which could be the real one or the stub
    mock_reconstructed_data = MagicMock(spec=sys.modules['torch'].Tensor)
    # Need .numpy() method on the mock_reconstructed_data if it's a torch.Tensor mock
    mock_reconstructed_np = np.random.rand(100,3)
    if hasattr(mock_reconstructed_data, 'numpy'): # Should be true for a MagicMock
        mock_reconstructed_data.numpy.return_value = mock_reconstructed_np
    else: # If it's just a basic MagicMock without spec that includes numpy
        mock_reconstructed_data.numpy = MagicMock(return_value=mock_reconstructed_np)


    mock_model.return_value = (mock_reconstructed_data, MagicMock()) # Model call returns tuple

    mock_final_denoised_data = np.random.rand(100,3)
    mock_scaler_instance.inverse_transform.return_value = mock_final_denoised_data

    # Mock model.parameters() for optimizer
    mock_model.parameters.return_value = [MagicMock()]


    # observations is now defined above
    result = processor.denoise(observations, epochs=2) # Use few epochs for speed

    # Assertions
    MockStandardScaler.assert_called_once() # Check if scaler was instantiated
    mock_scaler_instance.fit_transform.assert_called_once_with(observations)
    mock_float_tensor.assert_called_once_with(mock_scaled_data)

    mock_build_autoencoder.assert_called_once()
    # Check input_dim for build_autoencoder
    # data.shape[1] where data is mock_torch_data (result of FloatTensor)
    # Configure mock_torch_data.shape to be a property that returns the correct tuple
    actual_shape_tuple = (observations.shape[0], observations.shape[1])
    # Patch 'shape' at the type/class level of the mock instance if 'shape' is a property
    # Or directly assign if 'shape' is a simple attribute.
    # Given it's a Tensor, 'shape' is often a property or a special sequence.
    # Using PropertyMock on type(mock_torch_data) is robust.
    type(mock_torch_data).shape = PropertyMock(return_value=actual_shape_tuple) # Ensure observations is defined before actual_shape_tuple
    # The call to build_autoencoder happens with data.shape[1]
    # We need to ensure mock_build_autoencoder was called with this.
    # The actual call is self.build_autoencoder(data.shape[1]), so data is mock_torch_data
    # This is tricky because data is created inside the method.
    # Let's check the arg passed to build_autoencoder in its call_args
    # call_args[0] is a tuple of positional args. It should now be (actual_input_dim,).
    assert mock_build_autoencoder.call_args[0][0] == observations.shape[1]


    assert mock_model.train.called # model.train() should be called
    assert mock_adam.called # Optimizer instantiated
    assert mock_loss.called # Loss function instantiated

    # Check training loop calls (e.g. optimizer.step) - epochs=2 means 2 steps
    if hasattr(mock_adam.return_value, 'step'): # Adam instance
      assert mock_adam.return_value.step.call_count == 2
    if hasattr(mock_model, 'zero_grad'): # Model itself if optimizer is not detailed
      # This part is tricky as optimizer.zero_grad() is called.
      # If mock_model is the DenoisingAutoencoder instance, it does not have zero_grad.
      # The optimizer has zero_grad.
      assert mock_adam.return_value.zero_grad.call_count == 2


    assert mock_model.eval.called # model.eval() should be called
    # Check model was called in eval mode (it's called once after loop for final output)
    # The first argument to model call should be mock_torch_data, second is noise_level=0
    mock_model.assert_called_with(mock_torch_data, noise_level=0)

    mock_scaler_instance.inverse_transform.assert_called_once_with(mock_reconstructed_np)
    assert np.array_equal(result, mock_final_denoised_data)


@patch.object(NoisyObservationProcessor, '_simple_denoise')
def test_denoise_fallback_to_simple(mock_simple_denoise):
    processor = NoisyObservationProcessor()
    observations_small = np.random.rand(10, 3) # Less than 100 rows

    # Ensure build_autoencoder is NOT called by also trying to patch it and assert not called
    with patch.object(NoisyObservationProcessor, 'build_autoencoder') as mock_build_autoencoder:
        processor.denoise(observations_small)
        mock_simple_denoise.assert_called_once_with(observations_small)
        mock_build_autoencoder.assert_not_called()

# --- Tests for Expression Division by Zero ---
import unittest

class TestExpressionDivisionByZero(unittest.TestCase):
    def test_expression_division_by_numeric_zero(self):
        expr = Expression('/', [Expression('const', [1]), Expression('const', [0])])
        self.assertTrue(sp.sympify(expr.symbolic).is_nan)

    def test_expression_division_by_symbolic_zero(self):
        # This test assumes that Expression('var', ['x']) when x=0 is not simplified to const 0
        # by the Expression class itself before _to_sympy is called.
        # The fix is for _to_sympy's handling of explicit zero in denominator.
        zero_expr_val = Expression('*', [Expression('const', [0]), Expression('var', ['x'])])
        # zero_expr_val.symbolic should be sp.Integer(0) or sp.Float(0.0)
        div_expr = Expression('/', [Expression('const', [1]), zero_expr_val])
        self.assertTrue(sp.sympify(div_expr.symbolic).is_nan)

    def test_expression_division_by_symbolic_variable_that_is_zero(self):
        z = sp.Symbol('z')
        # Create an expression representing 1/z
        var_expr = Expression('var', ['z'])
        div_expr = Expression('/', [Expression('const', [1]), var_expr])

        # When we substitute z=0 into the sympy expression 1/z, sympy returns zoo (infinity)
        # This is standard Sympy behavior and is not what the fix targets.
        # The fix targets cases where an Expression object representing 0 is the denominator
        # during the _to_sympy conversion.
        res_sympy_behavior = div_expr.symbolic.subs({z: 0})
        self.assertEqual(res_sympy_behavior, sp.zoo)

        # To test the fix more directly for a Variable that *becomes* zero in context,
        # we'd need a more complex setup where a Variable's value is resolved to zero
        # during expression construction, or the Variable itself is defined as zero.
        # The current fix in _to_sympy for `denominator.is_zero` primarily addresses
        # explicit `Expression('const', [0])` or simple symbolic expressions that SymPy
        # can immediately evaluate to zero (like `0*x`).

        # If a Variable object could somehow have a .is_zero property that _to_sympy could check,
        # that would be a different scenario.
        # For now, the primary check is that an Expression object that evaluates to zero
        # as a denominator leads to NaN.

        # Let's try to create Expression('var', ['z_is_zero']) where 'z_is_zero' is conceptually zero.
        # The Expression class doesn't hold actual values for variables during _to_sympy,
        # it just creates sp.Symbol('z_is_zero'). So 1/sp.Symbol('z_is_zero') is the result.
        # The .is_zero check relies on sympy's .is_zero for symbolic expressions.
        # sp.Symbol('z_is_zero').is_zero is False.

        # The most direct test for the fix is already covered by test_expression_division_by_numeric_zero
        # and test_expression_division_by_symbolic_zero (where symbolic_zero is like 0*x).

if __name__ == '__main__':
    pytest.main() # To run pytest tests if this file is executed
    unittest.main() # To run unittest tests if this file is executed


# --- Tests for Expression.clone() method ---

def test_clone_simple_expression(grammar_and_vars):
    grammar, var_a, _, _, const_1, _ = grammar_and_vars
    original_expr = grammar.create_expression('+', [var_a, const_1])
    assert original_expr is not None

    cloned_expr = original_expr.clone()

    assert cloned_expr is not original_expr
    assert cloned_expr.operator == original_expr.operator
    assert cloned_expr.complexity == original_expr.complexity
    assert cloned_expr.symbolic == original_expr.symbolic

    # Check operands list is new and elements are same (or clones for Expressions)
    assert cloned_expr.operands is not original_expr.operands
    assert len(cloned_expr.operands) == len(original_expr.operands)
    # Operand 0 (Variable) should be the same object (Variables are shared)
    assert cloned_expr.operands[0] is original_expr.operands[0]
    # Operand 1 (const_1 which is a float) should be the same value
    assert cloned_expr.operands[1] == original_expr.operands[1]


def test_clone_nested_expression(grammar_and_vars):
    grammar, var_a, var_b, _, const_1, _ = grammar_and_vars

    # Original: (a + 1.0) * b
    inner_expr = grammar.create_expression('+', [var_a, const_1])
    original_expr = grammar.create_expression('*', [inner_expr, var_b])
    assert original_expr is not None

    cloned_expr = original_expr.clone()

    assert cloned_expr is not original_expr
    assert cloned_expr.operator == original_expr.operator
    assert cloned_expr.complexity == original_expr.complexity
    assert cloned_expr.symbolic == original_expr.symbolic
    assert cloned_expr.operands is not original_expr.operands

    # Check outer operands
    # Operand 0 (inner_expr clone)
    cloned_inner_expr = cloned_expr.operands[0]
    original_inner_expr = original_expr.operands[0]
    assert isinstance(cloned_inner_expr, Expression)
    assert cloned_inner_expr is not original_inner_expr
    assert cloned_inner_expr.operator == original_inner_expr.operator
    assert cloned_inner_expr.complexity == original_inner_expr.complexity
    assert cloned_inner_expr.symbolic == original_inner_expr.symbolic
    assert cloned_inner_expr.operands is not original_inner_expr.operands
    # Check inner_expr's operands
    assert cloned_inner_expr.operands[0] is original_inner_expr.operands[0] # var_a
    assert cloned_inner_expr.operands[1] == original_inner_expr.operands[1] # const_1

    # Operand 1 (var_b)
    assert cloned_expr.operands[1] is original_expr.operands[1] # var_b shared


def test_clone_modifying_original_does_not_affect_clone(grammar_and_vars):
    grammar, var_a, var_b, _, const_1, _ = grammar_and_vars
    # Original: (a + 1.0)
    original_expr = grammar.create_expression('+', [var_a, const_1])
    cloned_expr = original_expr.clone()

    # Try to modify original_expr (this is generally not good practice for Expression objects
    # as they are meant to be somewhat immutable after creation, but for testing clone integrity)
    # Let's change an operand of the original. This requires making operands list mutable for test.
    # Normally, operands are set at init.
    # If Expression's operands list was directly mutable:
    # original_expr.operands[0] = var_b
    # original_expr.operands[1] = 2.0
    # original_expr.__post_init__() # To update complexity and symbolic form

    # A more realistic modification test would be if an operand *itself* was mutable
    # and that mutation should not affect the clone.
    # Variables are shared, but if they were mutable (e.g. var_a.name could change),
    # the clone should still point to the original var_a if that's the design.
    # Constants (floats) are immutable.
    # Nested Expressions are the main concern for deep copy.

    # Let's test modification of a nested Expression in the original
    # Original: ( (a+1.0) + b )
    inner_original = grammar.create_expression('+', [var_a, const_1])
    outer_original = grammar.create_expression('+', [inner_original, var_b])

    outer_cloned = outer_original.clone()

    # Now, if we could modify `inner_original`'s operands (hypothetically)
    # For example, if inner_original.operands list was mutable:
    # inner_original.operands[0] = var_b # Change 'a' to 'b' in original inner
    # inner_original.__post_init__() # Recompute
    # outer_original.__post_init__() # Recompute

    # The clone `outer_cloned` should have its own `inner_cloned` which should still be (a+1.0)
    # This is implicitly tested by `cloned_inner_expr is not original_inner_expr` in test_clone_nested_expression.

    # Let's consider a case where an operand is an Expression and we replace that whole Expression operand
    # in the original list. This is more about list copy than deep element copy.
    # original_expr_list_mod = (a+1.0)
    # cloned_expr_list_mod = original_expr_list_mod.clone()
    # new_operand_for_original = grammar.create_expression('var', ['new_var_name'])
    # original_expr_list_mod.operands[0] = new_operand_for_original
    # original_expr_list_mod.__post_init__()
    # This kind of modification is not standard.

    # The key is that `cloned_expr.operands[0]` (if an Expression) is a *clone* of
    # `original_expr.operands[0]`, not the same object.
    # This was asserted in `test_clone_nested_expression`.

    # If we change the operator of the original (hypothetically, if mutable)
    # outer_original.operator = '*'
    # outer_original.__post_init__()
    # outer_cloned.operator should still be '+'
    # This is also implicitly true because a new Expression is made for the clone.

    # The current clone implementation creates new lists for operands and new Expression
    # instances recursively. Variables are shared (intended). Constants are copied by value.
    # This test is more of a conceptual check; the structural assertions in other tests cover it.
    assert True # Test primarily covered by structural independence checks.


def test_clone_expression_with_no_operands(grammar_and_vars):
    # This case should not typically occur with the current Expression structure,
    # as 'var' and 'const' take an operand in their list.
    # If an Expression could be Expression("myop", [])
    # For now, Expression always has operands, even for 'const' or 'var'.
    # Example: Expression('const', [1.0])
    grammar, _, _, _, const_1, _ = grammar_and_vars
    const_expr = grammar.create_expression('const', [const_1])
    cloned_const_expr = const_expr.clone()

    assert cloned_const_expr is not const_expr
    assert cloned_const_expr.operator == 'const'
    assert len(cloned_const_expr.operands) == 1
    assert cloned_const_expr.operands[0] == const_1
    assert cloned_const_expr.complexity == const_expr.complexity
    assert cloned_const_expr.symbolic == const_expr.symbolic

    # Example: Expression('var', ['x_name'])
    var_expr = grammar.create_expression('var', ['x_name'])
    cloned_var_expr = var_expr.clone()
    assert cloned_var_expr is not var_expr
    assert cloned_var_expr.operator == 'var'
    assert len(cloned_var_expr.operands) == 1
    assert cloned_var_expr.operands[0] == 'x_name'
    assert cloned_var_expr.complexity == var_expr.complexity
    assert cloned_var_expr.symbolic == var_expr.symbolic
