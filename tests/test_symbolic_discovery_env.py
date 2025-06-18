import unittest
import numpy as np

# Assuming paths are set up correctly for these imports
# Add try-except for imports if running script directly vs through test runner might be an issue
from symbolic_discovery_env import SymbolicDiscoveryEnv, ExpressionNode, NodeType
from progressive_grammar_system import ProgressiveGrammar, Variable, Expression

class MockGrammar(ProgressiveGrammar):
    def __init__(self):
        super().__init__()
        # Minimal grammar for testing
        self.primitives['constants'] = {'1.0': 1.0}
        self.primitives['binary_ops'] = {'+'}

    def create_expression(self, operator: str, operands: list) -> Expression:
        # Simplified for testing, no actual sympy or complex validation needed here
        # if operator == 'const':
        #     return Expression(operator='const', operands=[operands[0]])
        # return Expression(operator=operator, operands=operands)
        # The Expression class itself will handle this structure.
        return Expression(operator, operands)


class TestSymbolicDiscoveryEnv(unittest.TestCase):

    def setUp(self):
        self.grammar = MockGrammar()
        self.variables = [Variable(name="v0", index=0, properties={})]
        # Target data: v0, v1, target_value (for _evaluate_expression)
        self.target_data = np.array([
            [1.0, 2.0, 3.0],  # v0=1, target=3 (if v0+const_val)
            [2.0, 3.0, 4.0],  # v0=2, target=4
            [3.0, 4.0, 5.0],  # v0=3, target=5
            [4.0, 5.0, 6.0],  # v0=4, target=6
        ])

        self.env = SymbolicDiscoveryEnv(
            grammar=self.grammar,
            target_data=self.target_data,
            variables=self.variables,
            max_depth=3,
            max_complexity=10,
            reward_config={'mse_weight': -1.0, 'complexity_penalty': -0.01} # Simple reward
        )

    def test_deterministic_rewards(self):
        """
        Test that _evaluate_expression returns deterministic rewards
        when the expression and target data are fixed.
        """
        # Construct a simple complete expression: v0 + 1.0
        # Root: OPERATOR '+'
        # Child 1: VARIABLE 'v0'
        # Child 2: CONSTANT '1.0'

        # Constants in our mock grammar are just floats for Expression operands
        const_node_val = 1.0

        # Actual Expression objects for children, not ExpressionNode
        var_expr = self.grammar.create_expression('var', [self.variables[0].symbolic]) # This uses sympy symbol
        const_expr = self.grammar.create_expression('const', [const_node_val])

        # The ExpressionNode needs to represent the structure that leads to an Expression
        # Let's build the ExpressionNode tree that would generate `v0 + 1.0`
        # The `to_expression` method will convert this to an `Expression`

        root_node = ExpressionNode(node_type=NodeType.OPERATOR, value='+')

        var_node = ExpressionNode(node_type=NodeType.VARIABLE, value=self.variables[0], parent=root_node)
        var_node.children = [] # Variables are terminals in ExpressionNode tree for to_expression

        # For constants, the ExpressionNode holds the direct value if it's simple,
        # or it could be an empty node that gets filled.
        # Here, we assume it's already a terminal node representing a constant.
        const_node = ExpressionNode(node_type=NodeType.CONSTANT, value=const_node_val, parent=root_node)
        const_node.children = []

        root_node.children = [var_node, const_node]

        self.env.current_state.root = root_node
        self.assertTrue(self.env.current_state.is_complete(), "Constructed expression tree is not complete")

        # Evaluate multiple times
        reward1 = self.env._evaluate_expression()
        reward2 = self.env._evaluate_expression()
        reward3 = self.env._evaluate_expression()

        self.assertEqual(reward1, reward2, "Reward is not deterministic (run 1 vs 2)")
        self.assertEqual(reward2, reward3, "Reward is not deterministic (run 2 vs 3)")

    def test_explicit_target_evaluation(self):
        """
        Test that _evaluate_expression uses the target_variable_index correctly.
        """
        # Target data: feature0, feature1 (target for discovery), feature2
        target_data_for_test = np.array([
            [1.0, 10.0, 5.0], # f0=1, target=10 (if expr is f0+9)
            [2.0, 20.0, 6.0], # f0=2, target=20
            [3.0, 30.0, 7.0], # f0=3, target=30
        ])

        # Variable 'x' is feature0 (index 0)
        # Variable 'y' is feature2 (index 2) - this is NOT the target
        variables_for_test = [
            Variable(name="x", index=0, properties={}),
            Variable(name="y", index=2, properties={})
        ]

        # Initialize env to target feature1 (index 1)
        env_explicit_target = SymbolicDiscoveryEnv(
            grammar=self.grammar,
            target_data=target_data_for_test,
            variables=variables_for_test,
            max_depth=3,
            max_complexity=10,
            reward_config={'mse_weight': -1.0},
            target_variable_index=1 # Target is column 1 (10.0, 20.0, 30.0)
        )
        self.assertEqual(env_explicit_target.target_variable_index, 1)

        # Construct a simple expression: x + 9.0
        # (Expression: x + 9.0, so prediction for row 0 is 1.0 + 9.0 = 10.0. Target is 10.0. MSE should be 0)
        const_val = 9.0

        root_node = ExpressionNode(node_type=NodeType.OPERATOR, value='+')
        var_node_x = ExpressionNode(node_type=NodeType.VARIABLE, value=variables_for_test[0], parent=root_node)
        const_node_9 = ExpressionNode(node_type=NodeType.CONSTANT, value=const_val, parent=root_node)
        root_node.children = [var_node_x, const_node_9]

        env_explicit_target.current_state.root = root_node
        self.assertTrue(env_explicit_target.current_state.is_complete(), "Expression not complete for target eval test")

        # Call _evaluate_expression. We expect it to use column 1 of target_data_for_test as target.
        # If it correctly uses column 1 (values 10, 20, 30) and expression is x+9
        # Predictions: 1+9=10, 2+9=11, 3+9=12
        # Targets:     10,     20,     30
        # MSE for (10-10)^2=0, (11-20)^2=81, (12-30)^2=324. Mean = (0+81+324)/3 = 405/3 = 135
        # reward = completion_bonus (0.1) + mse_weight (-1.0) * log(norm_mse + 1e-10)
        # Let's check the MSE part by temporarily modifying _evaluate_expression or by checking tars

        # To check `tars` directly, we'd need to modify the original code or use a more complex mock.
        # Instead, let's verify the reward. If it's wrong, it might be due to wrong target selection.
        # A more direct test of `tars` would involve asserting its content after the loop in _evaluate_expression.

        # For this test, we'll mainly rely on the fact that `target_variable_index` is set.
        # A full reward calculation check is complex. The main point is if `target_variable_index` is used.
        # The original code is: tars.append(self.target_data[i, self.target_variable_index])

        # Let's check the expected reward if the target is column 1 (the correct one)
        # Predictions: x+9 -> [1+9, 2+9, 3+9] = [10, 11, 12]
        # Actual targets (col 1): [10, 20, 30]
        # MSE = ((10-10)^2 + (11-20)^2 + (12-30)^2) / 3 = (0 + 81 + 324) / 3 = 135.0
        # Var_tars = np.var([10,20,30]) = np.mean((tars_mean - tars)^2) = np.mean((-10,0,10)^2) = (100+0+100)/3 = 200/3 = 66.66...
        # norm_mse = 135.0 / (200/3 + 1e-10) = 135.0 / 66.666... = 2.025
        # log(norm_mse + 1e-10) = log(2.025) = 0.705...
        # reward = 0.1 (completion) -1.0 * 0.705... = -0.605... (approx)

        # If target_variable_index was -1 (i.e., column 2: [5,6,7])
        # MSE = ((10-5)^2 + (11-6)^2 + (12-7)^2) / 3 = (25 + 25 + 25) / 3 = 25.0
        # Var_tars = np.var([5,6,7]) = np.mean((tars_mean - tars)^2) = np.mean((-1,0,1)^2) = (1+0+1)/3 = 2/3
        # norm_mse = 25.0 / (2/3 + 1e-10) = 37.5
        # log(norm_mse + 1e-10) = log(37.5) = 3.624...
        # reward = 0.1 - 3.624 = -3.524... (approx)

        # This reward calculation is sensitive and good for testing.
        reward_from_col1_target = env_explicit_target._evaluate_expression()

        # Now, create an env that defaults to the last column (index 2)
        env_default_target = SymbolicDiscoveryEnv(
            grammar=self.grammar,
            target_data=target_data_for_test, # same data
            variables=variables_for_test,     # same variables
            max_depth=3,
            max_complexity=10,
            reward_config={'mse_weight': -1.0},
            target_variable_index=None # Default to last column
        )
        self.assertEqual(env_default_target.target_variable_index, 2) # Default is last col index
        env_default_target.current_state.root = root_node # same expression
        reward_from_col2_target = env_default_target._evaluate_expression()

        self.assertNotAlmostEqual(reward_from_col1_target, reward_from_col2_target,
                                  msg="Rewards should differ based on target_variable_index", places=5)

        # Expected norm_mse for col 1 target:
        preds_c1 = np.array([10.0, 11.0, 12.0])
        tars_c1 = target_data_for_test[:, 1] # 10, 20, 30
        mse_c1 = np.mean((preds_c1 - tars_c1)**2) # 135.0
        norm_mse_c1 = mse_c1 / (np.var(tars_c1) + 1e-10) # 135.0 / (200/3) = 2.025

        # The reward calculation in _evaluate_expression now ALWAYS uses np.exp and mse_scale_factor.
        # For env_explicit_target, reward_config is {'mse_weight': -1.0}.
        # mse_scale_factor will be fetched with its default from get(), which is 1.0.
        # mse_weight will be -1.0 from the config.
        # completion_bonus defaults to 0.1
        # complexity_penalty defaults to -0.01
        # depth_penalty defaults to -0.001

        expected_mse_weight = env_explicit_target.reward_config.get('mse_weight', 1.0) # Should be -1.0
        expected_mse_scale_factor = env_explicit_target.reward_config.get('mse_scale_factor', 1.0) # Should be 1.0 (default)

        expected_reward_c1 = (env_explicit_target.reward_config.get('completion_bonus', 0.1) +
                              expected_mse_weight * np.exp(-expected_mse_scale_factor * norm_mse_c1) +
                              env_explicit_target.reward_config.get('complexity_penalty', -0.01) * root_node.to_expression(self.grammar).complexity +
                              env_explicit_target.reward_config.get('depth_penalty', -0.001) * env_explicit_target.max_depth)

        self.assertAlmostEqual(reward_from_col1_target, expected_reward_c1, places=5,
                               msg="Reward for target_index=1 does not match expected calculation.")

    def test_reward_function_mse_component(self):
        """
        Tests the MSE component of the reward function with new parameters.
        """
        reward_config_test = {
            'completion_bonus': 0.1,
            'mse_weight': 1.0,  # Positive weight
            'mse_scale_factor': 0.5,
            'complexity_penalty': -0.01,
            'depth_penalty': -0.001, # Assuming max_depth might be used
        }

        # Environment for this specific test
        test_env = SymbolicDiscoveryEnv(
            grammar=self.grammar,
            target_data=self.target_data, # Uses self.target_data [[1,2,3], [2,3,4], [3,4,5], [4,5,6]]
            variables=self.variables,     # Uses self.variables [v0 (index 0)]
            max_depth=3,                  # Corresponds to depth_penalty calculation if used
            max_complexity=10,
            reward_config=reward_config_test,
            target_variable_index=2 # Target is column 2: [3,4,5,6]
        )
        self.assertEqual(test_env.target_variable_index, 2)

        # --- Scenario 1: Low MSE ---
        # Expression: v0 + 2.0. For v0=[1,2,3,4], preds=[3,4,5,6]. Targets=[3,4,5,6]. MSE = 0.
        const_val_low_mse = 2.0
        root_low_mse = ExpressionNode(node_type=NodeType.OPERATOR, value='+')
        var_node_v0_low = ExpressionNode(node_type=NodeType.VARIABLE, value=self.variables[0], parent=root_low_mse)
        const_node_low = ExpressionNode(node_type=NodeType.CONSTANT, value=const_val_low_mse, parent=root_low_mse)
        root_low_mse.children = [var_node_v0_low, const_node_low]
        test_env.current_state.root = root_low_mse
        self.assertTrue(test_env.current_state.is_complete(), "Low MSE tree not complete")

        expr_low_mse = root_low_mse.to_expression(self.grammar)
        expr_complexity_low = expr_low_mse.complexity # Assume complexity = 3 (var, const, op)

        preds_low = np.array([self.variables[0].symbolic.subs({self.variables[0].symbolic: r[0]}) + const_val_low_mse for r in self.target_data])
        tars_low = self.target_data[:, test_env.target_variable_index]
        mse_low = np.mean((preds_low - tars_low)**2)
        self.assertAlmostEqual(mse_low, 0.0, places=5, msg="MSE for low scenario should be near zero.")

        norm_low = float(mse_low / (np.var(tars_low) + 1e-10)) # Ensure float
        self.assertAlmostEqual(norm_low, 0.0, places=5, msg="Norm_MSE for low scenario should be near zero.")

        expected_reward_low_mse = (
            reward_config_test['completion_bonus'] +
            reward_config_test['mse_weight'] * np.exp(-reward_config_test['mse_scale_factor'] * norm_low) +
            reward_config_test['complexity_penalty'] * float(expr_complexity_low) + # Ensure float
            reward_config_test['depth_penalty'] * float(test_env.max_depth) # Ensure float
        )
        actual_reward_low_mse = test_env._evaluate_expression()
        self.assertAlmostEqual(actual_reward_low_mse, expected_reward_low_mse, places=5,
                               msg="Reward for low MSE scenario does not match expected.")

        # --- Scenario 2: High MSE ---
        # Expression: v0 + 10.0. For v0=[1,2,3,4], preds=[11,12,13,14]. Targets=[3,4,5,6].
        const_val_high_mse = 10.0
        root_high_mse = ExpressionNode(node_type=NodeType.OPERATOR, value='+')
        var_node_v0_high = ExpressionNode(node_type=NodeType.VARIABLE, value=self.variables[0], parent=root_high_mse)
        const_node_high = ExpressionNode(node_type=NodeType.CONSTANT, value=const_val_high_mse, parent=root_high_mse)
        root_high_mse.children = [var_node_v0_high, const_node_high]
        test_env.current_state.root = root_high_mse # Update env to use this new expression
        self.assertTrue(test_env.current_state.is_complete(), "High MSE tree not complete")

        expr_high_mse = root_high_mse.to_expression(self.grammar)
        expr_complexity_high = expr_high_mse.complexity # Assume complexity = 3

        preds_high = np.array([self.variables[0].symbolic.subs({self.variables[0].symbolic: r[0]}) + const_val_high_mse for r in self.target_data])
        tars_high = self.target_data[:, test_env.target_variable_index] # Same targets
        mse_high = np.mean((preds_high - tars_high)**2) # (11-3)^2=64, (12-4)^2=64, (13-5)^2=64, (14-6)^2=64. MSE = 64.
        self.assertAlmostEqual(mse_high, 64.0, places=5, msg="MSE for high scenario calculation error.")

        norm_high = float(mse_high / (np.var(tars_high) + 1e-10)) # Ensure float; var([3,4,5,6]) = 1.25. norm_high = 64 / 1.25 = 51.2
        self.assertAlmostEqual(norm_high, 51.2, places=5, msg="Norm_MSE for high scenario calculation error.")

        expected_reward_high_mse = (
            reward_config_test['completion_bonus'] +
            reward_config_test['mse_weight'] * np.exp(-reward_config_test['mse_scale_factor'] * norm_high) +
            reward_config_test['complexity_penalty'] * float(expr_complexity_high) + # Ensure float
            reward_config_test['depth_penalty'] * float(test_env.max_depth) # Ensure float
        )
        actual_reward_high_mse = test_env._evaluate_expression()
        self.assertAlmostEqual(actual_reward_high_mse, expected_reward_high_mse, places=5,
                               msg="Reward for high MSE scenario does not match expected.")

        self.assertTrue(actual_reward_low_mse > actual_reward_high_mse,
                        f"Low MSE reward ({actual_reward_low_mse}) should be greater than high MSE reward ({actual_reward_high_mse}).")

        # --- Scenario 3: Impact of mse_scale_factor ---
        # Using the high MSE expression (v0 + 10.0), vary mse_scale_factor
        reward_config_scale_varied = {
            'completion_bonus': 0.1,
            'mse_weight': 1.0,
            'mse_scale_factor': 0.1, # Smaller scale factor
            'complexity_penalty': -0.01,
            'depth_penalty': -0.001,
        }
        test_env_scale_varied = SymbolicDiscoveryEnv(
            grammar=self.grammar, target_data=self.target_data, variables=self.variables,
            max_depth=3, max_complexity=10, reward_config=reward_config_scale_varied,
            target_variable_index=2
        )
        test_env_scale_varied.current_state.root = root_high_mse # Same high MSE expression

        expected_reward_high_mse_small_scale = (
            reward_config_scale_varied['completion_bonus'] +
            reward_config_scale_varied['mse_weight'] * np.exp(-reward_config_scale_varied['mse_scale_factor'] * norm_high) + # norm_high is float from previous calc
            reward_config_scale_varied['complexity_penalty'] * float(expr_complexity_high) + # Ensure float
            reward_config_scale_varied['depth_penalty'] * float(test_env_scale_varied.max_depth) # Ensure float
        )
        actual_reward_high_mse_small_scale = test_env_scale_varied._evaluate_expression()
        self.assertAlmostEqual(actual_reward_high_mse_small_scale, expected_reward_high_mse_small_scale, places=5,
                               msg="Reward for high MSE with smaller scale_factor does not match.")

        # With a smaller mse_scale_factor, the penalty for MSE is less severe, so reward should be higher
        # than the reward with a larger mse_scale_factor for the same high MSE.
        # actual_reward_high_mse was with mse_scale_factor = 0.5
        # actual_reward_high_mse_small_scale is with mse_scale_factor = 0.1
        self.assertTrue(actual_reward_high_mse_small_scale > actual_reward_high_mse,
                        f"Reward with smaller mse_scale_factor ({actual_reward_high_mse_small_scale}) "
                        f"should be greater than with larger mse_scale_factor ({actual_reward_high_mse}) for the same high MSE.")


if __name__ == '__main__':
    unittest.main()
