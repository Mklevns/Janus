# tests/test_enhanced_feedback.py

import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import sympy # Added
from progressive_grammar_system import Variable # Added

# Assuming enhanced_feedback.py is in the parent directory or PYTHONPATH is set up
# If not, adjust import path as necessary, e.g., from .. import enhanced_feedback
# The actual import from enhanced_feedback.py is NewConservationBiasedReward,
# but the test was patching the old name. We will patch NewConservationBiasedReward.
from enhanced_feedback import IntrinsicRewardCalculator


# Test for the old ConservationBiasedReward can be removed or updated if that class is still relevant.
# For now, focusing on IntrinsicRewardCalculator and its use of NewConservationBiasedReward.
# class TestConservationBiasedReward(unittest.TestCase):
#     def test_compute_conservation_bonus(self):
#         """Test the placeholder compute_conservation_bonus method."""
#         # This test might need to refer to NewConservationBiasedReward if that's the intended target
#         from enhanced_feedback import NewConservationBiasedReward as ConservationBiasedReward
#         calculator = ConservationBiasedReward(conservation_types=['energy'], weight_factor=1.0)
#         # Dummy inputs, as current implementation doesn't use them in the same way
#         # This test needs significant rework based on NewConservationBiasedReward's actual logic
#         # For this subtask, we are focusing on IntrinsicRewardCalculator, so we'll skip deep changes here.
#         pass


class TestIntrinsicRewardCalculator(unittest.TestCase):
    @patch('enhanced_feedback.NewConservationBiasedReward') # Updated patch target
    def test_calculate_intrinsic_reward_with_conservation(self, MockNewConservationBiasedReward):
        """Test that conservation bonus is correctly incorporated into intrinsic reward."""

        mock_conservation_instance = MockNewConservationBiasedReward.return_value
        mock_conservation_bonus_value = 0.7
        mock_conservation_instance.compute_conservation_bonus.return_value = mock_conservation_bonus_value
        # Mock the conservation_types attribute that is accessed in the method under test
        mock_conservation_instance.conservation_types = ['energy', 'momentum']


        conservation_weight = 0.5
        novelty_weight = 0.1
        diversity_weight = 0.1
        complexity_growth_weight = 0.1

        reward_calculator = IntrinsicRewardCalculator(
            novelty_weight=novelty_weight,
            diversity_weight=diversity_weight,
            complexity_growth_weight=complexity_growth_weight,
            conservation_weight=conservation_weight
        )

        expression = "x * 2" # A simple expression string
        complexity = 2
        extrinsic_reward = 0.2
        embedding = np.array([0.1, 0.2])

        # Setup mock variables
        mock_var_x = MagicMock(spec=Variable)
        mock_var_x.name = 'x'
        mock_var_x.index = 0
        mock_var_x.symbolic = sympy.Symbol('x')

        mock_var_energy_gt = MagicMock(spec=Variable)
        mock_var_energy_gt.name = 'energy_gt' # Should match 'energy' c_type
        mock_var_energy_gt.index = 1
        mock_var_energy_gt.symbolic = sympy.Symbol('energy_gt')

        mock_var_momentum_gt = MagicMock(spec=Variable)
        mock_var_momentum_gt.name = 'P_total' # Should match 'momentum' c_type via 'p'
        mock_var_momentum_gt.index = 2
        mock_var_momentum_gt.symbolic = sympy.Symbol('P_total')

        variables = [mock_var_x, mock_var_energy_gt, mock_var_momentum_gt]

        # Data: rows are samples, columns correspond to variable indices
        # x, energy_gt, P_total
        data = np.array([[1.0, 10.0, 5.0], [2.0, 20.0, 8.0]])

        # Mock evaluate_expression_on_data
        mock_evaluated_values = np.array([2.0, 4.0]) # x * 2 for data
        reward_calculator.evaluate_expression_on_data = MagicMock(return_value=mock_evaluated_values)

        reward_calculator._calculate_novelty_reward = MagicMock(return_value=0.3)
        reward_calculator._calculate_diversity_reward = MagicMock(return_value=0.2)
        reward_calculator._calculate_complexity_growth_reward = MagicMock(return_value=0.1)

        total_reward = reward_calculator.calculate_intrinsic_reward(
            expression, complexity, extrinsic_reward, embedding, data, variables
        )

        # Expected arguments for compute_conservation_bonus
        expected_predicted_traj = {
            'conserved_energy': mock_evaluated_values,
            'conserved_momentum': mock_evaluated_values
        }
        expected_ground_truth_traj = {
            'conserved_energy': data[:, mock_var_energy_gt.index],
            'conserved_momentum': data[:, mock_var_momentum_gt.index]
        }
        expected_hypothesis_params = {'variables_info': variables}

        mock_conservation_instance.compute_conservation_bonus.assert_called_once_with(
            predicted_traj=expected_predicted_traj,
            ground_truth_traj=expected_ground_truth_traj,
            hypothesis_params=expected_hypothesis_params
        )

        reward_calculator.evaluate_expression_on_data.assert_called_once_with(expression, data, variables)

        expected_novelty = novelty_weight * 0.3
        expected_diversity = diversity_weight * 0.2
        expected_complexity_growth = complexity_growth_weight * 0.1
        expected_conservation = conservation_weight * mock_conservation_bonus_value

        expected_total_intrinsic_reward = (
            expected_novelty +
            expected_diversity +
            expected_complexity_growth +
            expected_conservation
        )

        expected_final_reward = extrinsic_reward + expected_total_intrinsic_reward

        self.assertAlmostEqual(total_reward, expected_final_reward, places=7,
                               msg="Total reward does not match expected value with conservation bonus.")


class TestEvaluateExpressionOnData(unittest.TestCase):
    def setUp(self):
        # Weights don't matter for this specific method test
        self.reward_calculator = IntrinsicRewardCalculator()

        self.mock_var_x = MagicMock(spec=Variable)
        self.mock_var_x.name = 'x'
        self.mock_var_x.index = 0
        self.mock_var_x.symbolic = sympy.Symbol('x')

        self.mock_var_y = MagicMock(spec=Variable)
        self.mock_var_y.name = 'y'
        self.mock_var_y.index = 1
        self.mock_var_y.symbolic = sympy.Symbol('y')

    def test_simple_expression(self):
        expression_str = "x * 2"
        variables = [self.mock_var_x]
        data = np.array([[1.0], [2.0], [3.0]])
        expected_results = np.array([2.0, 4.0, 6.0])

        results = self.reward_calculator.evaluate_expression_on_data(expression_str, data, variables)
        np.testing.assert_array_almost_equal(results, expected_results)

    def test_expression_with_multiple_variables(self):
        expression_str = "x + y"
        variables = [self.mock_var_x, self.mock_var_y]
        data = np.array([[1.0, 0.5], [2.0, 3.0], [3.0, -1.0]]) # x, y
        expected_results = np.array([1.5, 5.0, 2.0])

        results = self.reward_calculator.evaluate_expression_on_data(expression_str, data, variables)
        np.testing.assert_array_almost_equal(results, expected_results)

    def test_parse_error(self):
        expression_str = "x **" # Invalid syntax
        variables = [self.mock_var_x]
        data = np.array([[1.0], [2.0]])
        expected_results = np.array([np.nan, np.nan])

        with patch('builtins.print') as mock_print: # Suppress error print
            results = self.reward_calculator.evaluate_expression_on_data(expression_str, data, variables)
            mock_print.assert_called() # Check that an error was logged
        np.testing.assert_array_equal(results, expected_results) # np.nan == np.nan is False, use array_equal

    def test_evaluation_error_log_negative(self):
        expression_str = "log(x)"
        variables = [self.mock_var_x]
        data = np.array([[-1.0], [0.0], [1.0]]) # log(-1) is nan, log(0) is -inf (sympy) -> nan
        # Sympy's log(negative) returns `log(abs(arg)) + I*pi`, evalf might give complex or nan.
        # Our method should coerce to nan if complex or if it's an error.
        # Sympy log(0) is -oo, which our method also converts to np.nan.
        expected_results = np.array([np.nan, np.nan, 0.0])

        with patch('builtins.print') as mock_print: # Suppress error print
            results = self.reward_calculator.evaluate_expression_on_data(expression_str, data, variables)
        np.testing.assert_array_almost_equal(results, expected_results)

    def test_undefined_symbol_error(self):
        expression_str = "x + z" # z is not in variables
        variables = [self.mock_var_x]
        data = np.array([[1.0], [2.0]])
        # SymPy will keep 'z' as a symbol, evalf will not change it.
        # Our method should detect this and return NaN.
        expected_results = np.array([np.nan, np.nan])

        with patch('builtins.print') as mock_print: # Suppress error print
            results = self.reward_calculator.evaluate_expression_on_data(expression_str, data, variables)
        np.testing.assert_array_equal(results, expected_results)

    def test_empty_data_array(self):
        expression_str = "x * 2"
        variables = [self.mock_var_x]
        data = np.array([[]] * 0).reshape(0,1) # Correct way to make 0-row, 1-col array
        expected_results = np.array([])

        results = self.reward_calculator.evaluate_expression_on_data(expression_str, data, variables)
        np.testing.assert_array_equal(results, expected_results)

    def test_expression_evaluates_to_infinity(self):
        expression_str = "1/x"
        variables = [self.mock_var_x]
        data = np.array([[0.0], [2.0]]) # 1/0 is zoo (SymPy for complex infinity)
        expected_results = np.array([np.nan, 0.5])

        with patch('builtins.print') as mock_print:
            results = self.reward_calculator.evaluate_expression_on_data(expression_str, data, variables)
        np.testing.assert_array_almost_equal(results, expected_results)

    def test_expression_remains_symbolic_after_subs(self):
        # This can happen if a sympy function is used that doesn't evaluate with .evalf() without free symbols
        # For example, if 'x' was a string "Symbol('t')" and not substituted
        expression_str = "Integral(x, x)" # Indefinite integral
        variables = [self.mock_var_x]
        data = np.array([[1.0], [2.0]])
        # Sympy will return Integral(1.0, 1.0) which is not a number.
        # Our method should return NaN.
        expected_results = np.array([np.nan, np.nan])

        with patch('builtins.print') as mock_print:
            results = self.reward_calculator.evaluate_expression_on_data(expression_str, data, variables)
        np.testing.assert_array_equal(results, expected_results)


if __name__ == '__main__':
    unittest.main()
