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

if __name__ == '__main__':
    unittest.main()
