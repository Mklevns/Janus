import unittest
import sympy as sp
# Add project root to path to allow utils import
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import _count_operations, calculate_symbolic_accuracy

class TestUtils(unittest.TestCase):
    def test_count_operations(self):
        x, y, z = sp.symbols('x y z')

        # Simple expression: x*y + z
        expr1 = x*y + z
        ops1 = _count_operations(expr1)
        self.assertEqual(ops1.get('Add', 0), 1)
        self.assertEqual(ops1.get('Mul', 0), 1)
        self.assertEqual(sum(ops1.values()), 2)

        # Expression with functions: sin(x) + cos(y)
        expr2 = sp.sin(x) + sp.cos(y)
        ops2 = _count_operations(expr2)
        self.assertEqual(ops2.get('Add', 0), 1)
        self.assertEqual(ops2.get('sin', 0), 1)
        self.assertEqual(ops2.get('cos', 0), 1)
        self.assertEqual(sum(ops2.values()), 3)

        # More complex expression: x**2 + 3*x*y - log(z)
        # SymPy: Add(Pow(x, 2), Mul(3, x, y), Mul(-1, log(z)))
        expr3 = x**2 + 3*x*y - sp.log(z)
        ops3 = _count_operations(expr3)
        self.assertEqual(ops3.get('Pow', 0), 1)    # x**2
        self.assertEqual(ops3.get('Mul', 0), 2)    # 3*x*y and -1*log(z)
        self.assertEqual(ops3.get('Add', 0), 1)    # The main Add operation
        self.assertEqual(ops3.get('log', 0), 1)    # log(z)
        self.assertEqual(sum(ops3.values()), 5)    # Total ops = 1+2+1+1 = 5


        # Expression with no operations (single symbol)
        expr4 = x
        ops4 = _count_operations(expr4)
        self.assertEqual(sum(ops4.values()), 0)

        # Expression with no operations (a number)
        expr5 = sp.Integer(5)
        ops5 = _count_operations(expr5)
        self.assertEqual(sum(ops5.values()), 0)

    def test_calculate_symbolic_accuracy(self):
        x, y, a, b, c, d = sp.symbols('x y a b c d')

        # Ground truth laws for testing
        gt_laws1 = {'law1': a + b}
        gt_laws2 = {'law1': a * b + c, 'law2': a - b}

        # Perfect match
        self.assertEqual(calculate_symbolic_accuracy(str(a + b), gt_laws1), 1.0)

        # Discovered is None
        self.assertEqual(calculate_symbolic_accuracy(None, gt_laws1), 0.0)

        # Completely different expressions
        self.assertEqual(calculate_symbolic_accuracy(str(x * y), gt_laws1), 0.0) # x*y vs a+b

        # Partial match: a*b-d vs a*b+c. Ground truth: {'law':a*b+c}
        # Discovered (a*b-d): Ops: {'Mul':2, 'Add':1}. Sum: 3
        # Truth (a*b+c): Ops: {'Mul':1, 'Add':1}. Sum: 2
        # Common: Mul:1, Add:1. Sum common:2. Total ops = 3+2=5. Similarity = 2*2/5 = 0.8
        self.assertAlmostEqual(calculate_symbolic_accuracy(str(a*b-d), {'law':a*b+c}), 0.8)


        # Different variable names but same structure: x+y vs a+b.
        # Ops for x+y: Add:1. Total:1
        # Ops for a+b: Add:1. Total:1
        # Common Add:1. Total ops = 1+1=2. Similarity = 2*1/2 = 1.0
        self.assertEqual(calculate_symbolic_accuracy(str(x + y), gt_laws1), 1.0)


        # Test with a more complex ground truth and partial match
        # Discovered: a*b+d. Truth: {'law1': a*b+c, 'law2': a-b}
        # Ops for a*b+d: Add:1, Mul:1. Total:2
        # Ops for a*b+c (law1): Add:1, Mul:1. Total:2
        # Common with law1: Add:1, Mul:1. Sum common=2. Total=4. Sim = 2*2/4 = 1.0
        self.assertEqual(calculate_symbolic_accuracy(str(a*b+d), gt_laws2), 1.0)


        # Test with no common operations against multiple ground truths
        self.assertEqual(calculate_symbolic_accuracy(str(x**2), gt_laws2), 0.0)

        # Test with expression that sympifies to 0 for non-match (e.g. x-x)
        # Ops for 0: none. Ops for a+b: Add:1. Common:0. Total:1. Sim=0.
        self.assertEqual(calculate_symbolic_accuracy(str(x-x), gt_laws1), 0.0)

        # Test for perfect match with one of the laws in a larger dict
        gt_laws3 = {'eom': x*y**2, 'energy': a+b, 'momentum': c-d}
        self.assertEqual(calculate_symbolic_accuracy(str(a+b), gt_laws3), 1.0)

        # Test for partial match with one of the laws in a larger dict
        # Discovered: a*b. Truth: gt_laws2 = {'law1': a * b + c, 'law2': a - b}
        # Ops for a*b: Mul:1. Total:1
        # Ops for a*b+c: Mul:1, Add:1. Total:2
        # Common with law1: Mul:1. Sum common=1. Total ops = 1+2=3. Sim = 2*1/3 = 0.666...
        self.assertAlmostEqual(calculate_symbolic_accuracy(str(a*b), gt_laws2), 2/3)

# --- New tests for validate_inputs ---
from janus.ai_interpretability.utils.math_utils import validate_inputs
from typing import List, Dict, Tuple, Any # For type hints in test functions

# Define a dummy InvalidConfigError as it was mentioned in requirements,
# though not directly used by validate_inputs itself.
class InvalidConfigError(Exception):
    pass

@validate_inputs
def example_func_for_test(a: int, b: str, c: float = 3.14, *args: tuple, **kwargs: dict) -> str:
    """An example function."""
    return f"a={a}-b={b}-c={c}-args={args}-kwargs={kwargs}"

@validate_inputs
def func_with_list_dict(d: List[int], e: Dict[str, float]):
    """Function with list and dict annotations."""
    return f"list={d}-dict={e}"

@validate_inputs
def func_missing_annotations(a: int, b, c=100): # b and c are not annotated
    """Function with some missing annotations."""
    return f"a={a}-b={b}-c={c}"

@validate_inputs
def func_only_args_kwargs(*args: tuple, **kwargs: dict):
    """Function with only *args and **kwargs."""
    return f"args={args}-kwargs={kwargs}"

@validate_inputs
def func_no_params() -> str:
    """Function with no parameters."""
    return "no_params_called"

# This function is for testing how the decorator handles *args annotation.
# The current decorator validates the type of the 'numbers' tuple itself against the annotation.
@validate_inputs
def func_args_annotated_tuple(*numbers: tuple):
    return f"numbers={numbers}"

@validate_inputs
def func_args_annotated_int_incorrectly(*numbers: int): # This annotation is unusual for *args
    return f"numbers={numbers}"


class TestValidateInputsDecorator(unittest.TestCase):

    def test_correct_types(self):
        result = example_func_for_test(10, "hello", 2.5, "extra_arg1", "extra_arg2", kwarg1="val1", kwarg2="val2")
        self.assertEqual(result, "a=10-b=hello-c=2.5-args=('extra_arg1', 'extra_arg2')-kwargs={'kwarg1': 'val1', 'kwarg2': 'val2'}")

        # Test with default value
        result_default = example_func_for_test(1, "world")
        self.assertEqual(result_default, "a=1-b=world-c=3.14-args=()-kwargs={}")

    def test_incorrect_positional_type(self):
        with self.assertRaisesRegex(TypeError, "Argument 'a' expected type int, but got str"):
            example_func_for_test("not_an_int", "hello")

    def test_incorrect_keyword_type(self):
        with self.assertRaisesRegex(TypeError, "Argument 'b' expected type str, but got int"):
            example_func_for_test(a=10, b=123)

    def test_incorrect_default_override_type(self):
        with self.assertRaisesRegex(TypeError, "Argument 'c' expected type float, but got str"):
            example_func_for_test(10, "hello", c="not_a_float")

    def test_complex_types_correct(self):
        # Tests basic list and dict type checking (not element types)
        result = func_with_list_dict([1, 2], {"key": 3.0})
        self.assertEqual(result, "list=[1, 2]-dict={'key': 3.0}")

    def test_complex_types_incorrect_list(self):
        with self.assertRaisesRegex(TypeError, "Argument 'd' expected type list, but got dict"):
            func_with_list_dict({"not_a_list": 1}, {"key": 3.0})

    def test_complex_types_incorrect_dict(self):
        with self.assertRaisesRegex(TypeError, "Argument 'e' expected type dict, but got list"):
            func_with_list_dict([1, 2], ["not_a_dict"])

    def test_missing_annotations(self):
        result = func_missing_annotations(a=1, b="any_type_for_b", c=[1,2,3])
        self.assertEqual(result, "a=1-b=any_type_for_b-c=[1, 2, 3]")

        with self.assertRaisesRegex(TypeError, "Argument 'a' expected type int, but got str"):
            func_missing_annotations(a="wrong_type", b="any_type")

    def test_args_correct_type_when_annotated_as_tuple(self):
        # example_func_for_test has *args: tuple
        result = example_func_for_test(1, "test", 3.0, "arg1", 2)
        self.assertEqual(result, "a=1-b=test-c=3.0-args=('arg1', 2)-kwargs={}")

        # Test with func_args_annotated_tuple specifically for *args: tuple
        result_args_only = func_args_annotated_tuple("x", "y")
        self.assertEqual(result_args_only, "numbers=('x', 'y')")

    def test_args_incorrect_type_when_annotated_as_int(self):
        # func_args_annotated_int_incorrectly has *args: int
        # The decorator expects the collected tuple itself to be an int, which will fail.
        with self.assertRaisesRegex(TypeError, "Argument 'numbers' (for \\*args) expected to be a int, but got tuple"):
            func_args_annotated_int_incorrectly(1, 2, 3)

    def test_kwargs_correct_type_when_annotated_as_dict(self):
        # example_func_for_test has **kwargs: dict
        result = example_func_for_test(1, "test", 3.0, kw1="val1", kw2=100)
        self.assertEqual(result, "a=1-b=test-c=3.0-args=()-kwargs={'kw1': 'val1', 'kw2': 100}")

    def test_return_value_preservation(self):
        self.assertEqual(example_func_for_test(1, "s", 1.0), "a=1-b=s-c=1.0-args=()-kwargs={}")

    def test_metadata_preservation(self):
        self.assertEqual(example_func_for_test.__name__, "example_func_for_test")
        self.assertEqual(example_func_for_test.__doc__, "An example function.")

    def test_no_params_func(self):
        self.assertEqual(func_no_params(), "no_params_called")

    def test_only_args_kwargs_correct(self):
        result = func_only_args_kwargs(1, "two", key="val") # args will be (1, "two"), kwargs {"key":"val"}
        self.assertEqual(result, "args=(1, 'two')-kwargs={'key': 'val'}")

        # Test with no args or kwargs passed
        result_empty = func_only_args_kwargs()
        self.assertEqual(result_empty, "args=()-kwargs={}")

if __name__ == '__main__':
    unittest.main()

# --- New tests for safe_import ---
from janus.ai_interpretability.utils.math_utils import safe_import # safe_import is already in math_utils
from unittest.mock import patch
import io
import math # For testing existing module import

class TestSafeImport(unittest.TestCase):

    def test_safe_import_existing_module(self):
        """Test importing an existing module (e.g., math)."""
        module = safe_import("math")
        self.assertIsNotNone(module)
        self.assertEqual(module.pi, math.pi)

    def test_safe_import_non_existing_module_with_pip_name(self):
        """Test importing a non-existent module with a pip install name."""
        module_name = "this_module_clearly_does_not_exist_abc"
        pip_name = "install_this_package_for_it"

        with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
            module = safe_import(module_name, pip_install_name=pip_name)

        self.assertIsNone(module)
        output = mock_stdout.getvalue()
        self.assertIn(f"Warning: Optional dependency '{module_name}' not found.", output)
        self.assertIn(f"You can install it with `pip install {pip_name}`.", output)

    def test_safe_import_non_existing_module_no_pip_name(self):
        """Test importing a non-existent module without a pip install name."""
        module_name = "another_non_existent_module_xyz"

        with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
            module = safe_import(module_name) # pip_install_name is None by default

        self.assertIsNone(module)
        output = mock_stdout.getvalue()
        self.assertIn(f"Warning: Optional dependency '{module_name}' not found.", output)
        self.assertIn("Please install it if you need this functionality.", output)

    def test_safe_import_non_existing_module_empty_pip_name(self):
        """Test importing a non-existent module with an empty pip install name."""
        module_name = "yet_another_fake_module_123"

        with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
            module = safe_import(module_name, pip_install_name="")

        self.assertIsNone(module)
        output = mock_stdout.getvalue()
        self.assertIn(f"Warning: Optional dependency '{module_name}' not found.", output)
        self.assertIn("Please install it if you need this functionality.", output)

    def test_safe_import_with_alias_parameter_no_effect(self):
        """Test that the alias parameter doesn't break anything (though not used yet)."""
        module = safe_import("math", alias="mymath")
        self.assertIsNotNone(module)
        self.assertEqual(module.pi, math.pi)

        module_name = "fake_module_for_alias_test"
        with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
            module_none = safe_import(module_name, alias="my_fake_alias")
        self.assertIsNone(module_none)
        output = mock_stdout.getvalue()
        self.assertIn(f"Warning: Optional dependency '{module_name}' not found.", output)

# To ensure these tests run if the file is executed directly,
# especially after appending to a file that might already have unittest.main()
# It's generally better to rely on a test runner.
# However, if __main__ block from previous edit is kept, it should be fine.
# If there are multiple unittest.main() calls, the behavior might be unexpected.
# Standard practice is one call to unittest.main().
# The previous edit correctly appended, so the existing __main__ block will run all tests.
