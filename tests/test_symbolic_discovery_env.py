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

    def test_reward_complexity_penalty(self):
        """Tests that higher complexity expressions get a more negative complexity penalty."""
        reward_config_complexity = {
            'completion_bonus': 0.1,
            'mse_weight': -1.0, # Assuming perfect MSE for this test (norm_mse=0)
            'mse_scale_factor': 1.0,
            'complexity_penalty': -0.1, # Non-zero complexity penalty
            'depth_penalty': 0.0 # No depth penalty for this test
        }
        self.env.reward_config = reward_config_complexity

        # Expression 1: v0 + 1.0 (Complexity: var, const, op -> 3)
        root_node1 = ExpressionNode(node_type=NodeType.OPERATOR, value='+')
        var_node1 = ExpressionNode(node_type=NodeType.VARIABLE, value=self.variables[0], parent=root_node1)
        const_node1 = ExpressionNode(node_type=NodeType.CONSTANT, value=1.0, parent=root_node1)
        root_node1.children = [var_node1, const_node1]
        self.env.current_state.root = root_node1
        expr1_obj = root_node1.to_expression(self.grammar)
        complexity1 = expr1_obj.complexity
        # Simulate perfect prediction for expr1 (MSE=0)
        # To do this, we need to make target_data match predictions of v0 + 1.0
        original_target_data = self.env.target_data.copy()
        self.env.target_data = np.column_stack((self.target_data[:,0], self.target_data[:,1], self.target_data[:,0] + 1.0))
        reward1 = self.env._evaluate_expression()
        self.env.target_data = original_target_data # Restore

        # Expression 2: (v0 + 1.0) + 0.0 (Complexity: 3 for (v0+1), const 0.0, op + -> 5)
        # This expression is equivalent in value to expr1 if evaluated perfectly.
        # (v0 + 1.0)
        sub_expr_node = ExpressionNode(node_type=NodeType.OPERATOR, value='+')
        sub_expr_node.children = [
            ExpressionNode(node_type=NodeType.VARIABLE, value=self.variables[0], parent=sub_expr_node),
            ExpressionNode(node_type=NodeType.CONSTANT, value=1.0, parent=sub_expr_node)
        ]
        # ((v0 + 1.0) + 0.0)
        root_node2 = ExpressionNode(node_type=NodeType.OPERATOR, value='+')
        const_node_zero = ExpressionNode(node_type=NodeType.CONSTANT, value=0.0, parent=root_node2)
        root_node2.children = [sub_expr_node, const_node_zero]
        sub_expr_node.parent = root_node2 # Set parent for sub_expr_node

        self.env.current_state.root = root_node2
        expr2_obj = root_node2.to_expression(self.grammar)
        complexity2 = expr2_obj.complexity
        # Simulate perfect prediction for expr2 (MSE=0)
        self.env.target_data = np.column_stack((self.target_data[:,0], self.target_data[:,1], self.target_data[:,0] + 1.0 + 0.0))
        reward2 = self.env._evaluate_expression()
        self.env.target_data = original_target_data # Restore

        self.assertGreater(complexity1, 0, "Complexity1 should be > 0")
        self.assertGreater(complexity2, complexity1, "Complexity2 should be greater than Complexity1")

        # Expected rewards (assuming perfect MSE, so norm_mse=0, exp(-scale*norm_mse)=1)
        # reward = completion + mse_weight * 1 + complexity_penalty * complexity + depth_penalty * max_depth
        expected_reward1 = (reward_config_complexity['completion_bonus'] +
                            reward_config_complexity['mse_weight'] * 1.0 +
                            reward_config_complexity['complexity_penalty'] * complexity1 +
                            reward_config_complexity['depth_penalty'] * self.env.max_depth)

        expected_reward2 = (reward_config_complexity['completion_bonus'] +
                            reward_config_complexity['mse_weight'] * 1.0 +
                            reward_config_complexity['complexity_penalty'] * complexity2 +
                            reward_config_complexity['depth_penalty'] * self.env.max_depth)

        self.assertAlmostEqual(reward1, expected_reward1, places=5)
        self.assertAlmostEqual(reward2, expected_reward2, places=5)
        self.assertLess(reward2, reward1, "Reward for more complex expression should be less due to penalty.")

    def test_reward_depth_penalty(self):
        """Tests the depth penalty component of the reward function."""
        # The current implementation of depth_penalty in _evaluate_expression is:
        # reward_config.get('depth_penalty', -0.001) * self.max_depth
        # This means it's a constant penalty based on the *environment's* max_depth,
        # not the actual depth of the generated expression. This test will verify this behavior.

        reward_config_depth = {
            'completion_bonus': 0.1,
            'mse_weight': -1.0, # Assuming perfect MSE (norm_mse=0)
            'mse_scale_factor': 1.0,
            'complexity_penalty': 0.0, # No complexity penalty for this test
            'depth_penalty': -0.05 # Non-zero depth penalty
        }
        self.env.reward_config = reward_config_depth
        self.env.max_depth = 5 # Set a specific max_depth for the environment

        # Expression 1: v0 + 1.0 (Depth 2, Complexity 3)
        root_node1 = ExpressionNode(node_type=NodeType.OPERATOR, value='+', depth=0)
        var_node1 = ExpressionNode(node_type=NodeType.VARIABLE, value=self.variables[0], parent=root_node1, depth=1)
        const_node1 = ExpressionNode(node_type=NodeType.CONSTANT, value=1.0, parent=root_node1, depth=1)
        root_node1.children = [var_node1, const_node1]
        self.env.current_state.root = root_node1
        expr1_obj = root_node1.to_expression(self.grammar)
        complexity1 = expr1_obj.complexity

        original_target_data = self.env.target_data.copy()
        self.env.target_data = np.column_stack((self.target_data[:,0], self.target_data[:,1], self.target_data[:,0] + 1.0))
        reward1 = self.env._evaluate_expression()
        self.env.target_data = original_target_data

        expected_penalty_contribution = reward_config_depth['depth_penalty'] * self.env.max_depth
        expected_reward1 = (reward_config_depth['completion_bonus'] +
                            reward_config_depth['mse_weight'] * 1.0 + # Perfect MSE
                            reward_config_depth['complexity_penalty'] * complexity1 +
                            expected_penalty_contribution)

        self.assertAlmostEqual(reward1, expected_reward1, places=5,
                               msg=f"Reward for depth test does not match. Expected penalty part: {expected_penalty_contribution}")

        # If we change self.env.max_depth, the reward should change if depth_penalty is non-zero.
        self.env.max_depth = 10 # Increase max_depth
        reward_config_depth_higher_max = reward_config_depth.copy()
        self.env.reward_config = reward_config_depth_higher_max # ensure it's using the same penalty rates

        self.env.current_state.root = root_node1 # Same expression
        self.env.target_data = np.column_stack((self.target_data[:,0], self.target_data[:,1], self.target_data[:,0] + 1.0))
        reward_higher_max_depth = self.env._evaluate_expression()
        self.env.target_data = original_target_data

        expected_penalty_contribution_higher = reward_config_depth_higher_max['depth_penalty'] * self.env.max_depth
        expected_reward_higher = (reward_config_depth_higher_max['completion_bonus'] +
                                 reward_config_depth_higher_max['mse_weight'] * 1.0 +
                                 reward_config_depth_higher_max['complexity_penalty'] * complexity1 +
                                 expected_penalty_contribution_higher)

        self.assertAlmostEqual(reward_higher_max_depth, expected_reward_higher, places=5)
        self.assertLess(reward_higher_max_depth, reward1,
                        "Reward should be lower (more penalized) when env.max_depth is higher, given negative depth_penalty.")


    def test_reward_incomplete_expression_penalty(self):
        """Tests the penalty applied for incomplete expressions."""
        # Current code in _evaluate_expression returns timeout_penalty if expr is None
        # (i.e. current_state.root.to_expression(self.grammar) returns None for incomplete expressions)
        timeout_penalty_val = -0.75
        self.env.reward_config['timeout_penalty'] = timeout_penalty_val

        # Create an incomplete expression (e.g., operator '+' with only one child)
        root_node = ExpressionNode(node_type=NodeType.OPERATOR, value='+')
        var_node = ExpressionNode(node_type=NodeType.VARIABLE, value=self.variables[0], parent=root_node)
        root_node.children = [var_node] # Missing second child

        self.env.current_state.root = root_node
        self.assertFalse(self.env.current_state.is_complete(), "Expression should be incomplete.")

        expr_obj = root_node.to_expression(self.grammar) # This should be None
        self.assertIsNone(expr_obj, "Incomplete ExpressionNode tree should yield None from to_expression.")

        reward = self.env._evaluate_expression()
        self.assertEqual(reward, timeout_penalty_val,
                         "Reward for incomplete expression should match timeout_penalty.")

    def test_action_add_operator(self):
        """Tests the effect of ACTION_ADD_OPERATOR."""
        self.env.reset()
        self.assertIsNotNone(self.env.current_state.root, "Initial root should not be None")
        self.assertEqual(self.env.current_state.root.node_type, NodeType.EMPTY, "Initial root should be EMPTY")

        # Find the action ID for adding '+' operator
        op_to_add = '+'
        action_id = -1
        for i, (atype, aval) in enumerate(self.env.action_to_element):
            if atype == 'operator' and aval == op_to_add:
                action_id = i
                break
        self.assertNotEqual(action_id, -1, f"Action for operator '{op_to_add}' not found.")

        # Take the action
        obs, reward, terminated, truncated, info = self.env.step(action_id)

        # Verify root node
        root = self.env.current_state.root
        self.assertEqual(root.node_type, NodeType.OPERATOR, "Root node type should be OPERATOR after adding operator.")
        self.assertEqual(root.value, op_to_add, f"Root node value should be '{op_to_add}'.")
        self.assertEqual(len(root.children), 2, "Operator '+' should have 2 children.")
        self.assertEqual(root.children[0].node_type, NodeType.EMPTY, "First child should be EMPTY.")
        self.assertEqual(root.children[1].node_type, NodeType.EMPTY, "Second child should be EMPTY.")
        self.assertEqual(root.children[0].depth, 1, "First child depth incorrect.")
        self.assertEqual(root.children[1].depth, 1, "Second child depth incorrect.")

        # Verify current_node (next empty node)
        next_empty = self.env.current_state.get_next_empty_node()
        self.assertIsNotNone(next_empty, "There should be a next empty node.")
        self.assertEqual(next_empty, root.children[0], "Current node should point to the first child.")

    def test_expression_completeness(self):
        """Tests the is_complete() method of ExpressionNode and TreeState."""
        # Case 1: Single constant
        const_node = ExpressionNode(node_type=NodeType.CONSTANT, value=1.0)
        self.assertTrue(const_node.is_complete(), "Single constant node should be complete.")
        self.env.current_state.root = const_node
        self.assertTrue(self.env.current_state.is_complete(), "TreeState with single constant root should be complete.")

        # Case 2: Single variable
        var_node = ExpressionNode(node_type=NodeType.VARIABLE, value=self.variables[0])
        self.assertTrue(var_node.is_complete(), "Single variable node should be complete.")
        self.env.current_state.root = var_node
        self.assertTrue(self.env.current_state.is_complete(), "TreeState with single variable root should be complete.")

        # Case 3: Operator with all operands filled
        op_node_full = ExpressionNode(node_type=NodeType.OPERATOR, value='+')
        child1_full = ExpressionNode(node_type=NodeType.CONSTANT, value=1.0, parent=op_node_full)
        child2_full = ExpressionNode(node_type=NodeType.VARIABLE, value=self.variables[0], parent=op_node_full)
        op_node_full.children = [child1_full, child2_full]
        self.assertTrue(op_node_full.is_complete(), "Operator with all children filled should be complete.")
        self.env.current_state.root = op_node_full
        self.assertTrue(self.env.current_state.is_complete(), "TreeState with full operator root should be complete.")

        # Case 4: Operator with one EMPTY child (binary op)
        op_node_missing_one = ExpressionNode(node_type=NodeType.OPERATOR, value='+')
        child_const = ExpressionNode(node_type=NodeType.CONSTANT, value=1.0, parent=op_node_missing_one)
        child_empty = ExpressionNode(node_type=NodeType.EMPTY, value=None, parent=op_node_missing_one)
        op_node_missing_one.children = [child_const, child_empty]
        self.assertFalse(op_node_missing_one.is_complete(), "Operator with one empty child should be incomplete.")
        self.env.current_state.root = op_node_missing_one
        self.assertFalse(self.env.current_state.is_complete(), "TreeState with one empty child should be incomplete.")

        # Case 5: Operator with one variable and one EMPTY (already covered by case 4 essentially)

        # Case 6: An EMPTY root node
        empty_root = ExpressionNode(node_type=NodeType.EMPTY, value=None)
        self.assertFalse(empty_root.is_complete(), "Empty root node should be incomplete.")
        self.env.current_state.root = empty_root
        self.assertFalse(self.env.current_state.is_complete(), "TreeState with empty root should be incomplete.")

        # Case 7: Operator with no children yet (should be incomplete)
        op_node_no_children = ExpressionNode(node_type=NodeType.OPERATOR, value='+')
        self.assertFalse(op_node_no_children.is_complete(), "Operator with no children should be incomplete.")
        self.env.current_state.root = op_node_no_children
        self.assertFalse(self.env.current_state.is_complete(), "TreeState with op no children root should be incomplete.")


if __name__ == '__main__':
    unittest.main()
