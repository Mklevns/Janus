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

import pytest
from symbolic_discovery_env import TreeState # Already imported SymbolicDiscoveryEnv, ExpressionNode, NodeType
from progressive_grammar_system import ProgressiveGrammar, Variable # Already imported ProgressiveGrammar, Variable, Expression

# It's good practice to define a fixture for the environment setup
# if it's used across multiple tests.
@pytest.fixture
def env_setup():
    grammar = ProgressiveGrammar()
    # Populate grammar primitives if necessary for the test expressions
    # This ensures that operators like 'sqrt', '/', '-', '*' are recognized
    grammar.primitives = {
        'binary_ops': {'+', '-', '*', '/'},
        'unary_ops': {'sqrt', 'neg', 'exp', 'log', 'sin', 'cos'}, # Added common ops
        'calculus_ops': set(), # Added empty set for calculus_ops
        'constants': {'1.0': 1.0, '0.0':0.0, '2.0':2.0} # Ensure some constants are there if grammar needs them explicitly
    }
    # Ensure grammar can handle symbolic variables correctly, e.g. by registering them
    # or ensuring its create_expression can handle unknown symbols if necessary.
    # The Variable class itself should provide the symbolic representation.

    x_var = Variable("x", 0, properties={"description": "input variable x"}) # Ensure Variable can be created like this
    variables = [x_var]

    # Target data: y = 2x + 1, with some noise for variance
    # Using a simpler, deterministic dataset for easier debugging of test logic
    x_data = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    y_data = 2 * x_data + 1
    target_data = np.hstack([x_data, y_data])

    env = SymbolicDiscoveryEnv(
        grammar=grammar,
        target_data=target_data,
        variables=variables,
        max_depth=5,
        max_complexity=30, # Increased complexity for some test expressions
        reward_config={
            'completion_bonus': 0.1,
            'mse_weight': 1.0,
            'mse_scale_factor': 1.0,
            'complexity_penalty': -0.01,
            'depth_penalty': -0.001,
            'timeout_penalty': -1.0
        }
    )
    # Ensure target_variable_index is set correctly by the environment, or set it explicitly
    # SymbolicDiscoveryEnv defaults to the last column if target_variable_index is None
    # For target_data = np.hstack([x_data, y_data]), y_data is the last column.
    # So env.target_variable_index should be 1.
    assert env.target_variable_index == 1

    return env, grammar, x_var

def test_evaluate_expression_handles_invalid_predictions_and_exceptions(env_setup):
    env, grammar, x_var = env_setup

    # --- Test Case 1: Prediction is NaN ---
    # Expression: sqrt(x)
    env.current_state = TreeState(max_depth=env.max_depth)
    root_node = ExpressionNode(NodeType.OPERATOR, 'sqrt', depth=0)
    # The Variable object itself is used as the value for VARIABLE nodes
    var_node = ExpressionNode(NodeType.VARIABLE, x_var, parent=root_node, depth=1)
    root_node.children.append(var_node)
    env.current_state.root = root_node

    # Original target data's y-variance is needed for penalty calculation
    # The penalty is calculated based on the variance of the *current* env.target_data's target column

    # Data point that causes NaN (sqrt(-1))
    problematic_x = -1.0
    # y-value for problematic_x doesn't matter for error if it's a penalty, but good to have a "correct" one
    problematic_y = 2 * problematic_x + 1

    test_target_data_nan = np.array([
        [1.0, 2 * 1.0 + 1],      # x=1, y=3. pred=sqrt(1)=1. error=(1-3)^2=4
        [problematic_x, problematic_y], # x=-1, y=-1. pred=sqrt(-1)=NaN. error=penalty
        [4.0, 2 * 4.0 + 1]       # x=4, y=9. pred=sqrt(4)=2. error=(2-9)^2=49
    ])
    env.target_data = test_target_data_nan # Switch env to this data

    # Recalculate penalty based on the new test_target_data_nan's y-column
    current_target_y_values_nan = env.target_data[:, env.target_variable_index]
    target_variance_nan = np.var(current_target_y_values_nan)
    penalty_on_fail_nan = target_variance_nan if target_variance_nan > 1e-9 else 1.0

    reward_nan = env._evaluate_expression()
    mse_nan = env._evaluation_cache['mse']

    expected_error_0_nan = (np.sqrt(test_target_data_nan[0,0]) - test_target_data_nan[0,1])**2
    expected_error_2_nan = (np.sqrt(test_target_data_nan[2,0]) - test_target_data_nan[2,1])**2
    expected_mse_nan = np.mean([expected_error_0_nan, penalty_on_fail_nan, expected_error_2_nan])

    assert np.isclose(mse_nan, expected_mse_nan), f"MSE for NaN case incorrect. Expected {expected_mse_nan}, got {mse_nan}"

    # --- Test Case 2: Prediction is very large ---
    # Expression: 1/x
    env.current_state = TreeState(max_depth=env.max_depth)
    op_node = ExpressionNode(NodeType.OPERATOR, '/', depth=0)
    const_node_one = ExpressionNode(NodeType.CONSTANT, 1.0, parent=op_node, depth=1, position=0)
    var_node_div = ExpressionNode(NodeType.VARIABLE, x_var, parent=op_node, depth=1, position=1)
    op_node.children.extend([const_node_one, var_node_div])
    env.current_state.root = op_node

    very_small_x = 1e-15
    test_target_data_large = np.array([
        [1.0, 1.0/1.0],      # pred = 1, target = 1, error = 0
        [very_small_x, 0.0], # pred will be 1/1e-15 = 1e15 (large). Target doesn't matter for penalty. error=penalty
        [2.0, 1.0/2.0]       # pred = 0.5, target = 0.5, error = 0
    ])
    env.target_data = test_target_data_large

    current_target_y_values_large = env.target_data[:, env.target_variable_index]
    target_variance_large = np.var(current_target_y_values_large)
    penalty_on_fail_large = target_variance_large if target_variance_large > 1e-9 else 1.0

    reward_large = env._evaluate_expression()
    mse_large = env._evaluation_cache['mse']

    expected_error_large_0 = ( (1.0/test_target_data_large[0,0]) - test_target_data_large[0,1] )**2
    expected_error_large_2 = ( (1.0/test_target_data_large[2,0]) - test_target_data_large[2,1] )**2
    expected_mse_large = np.mean([expected_error_large_0, penalty_on_fail_large, expected_error_large_2])
    assert np.isclose(mse_large, expected_mse_large), f"MSE for large number case incorrect. Expected {expected_mse_large}, got {mse_large}"


    # --- Test Case 3: Exception during evaluation (e.g., division by zero from x-x) ---
    # Expression: 1 / (x - x) . (x-x) will be zero.
    env.current_state = TreeState(max_depth=env.max_depth)
    op_node_exc_div = ExpressionNode(NodeType.OPERATOR, '/', depth=0) # Division
    const_node_exc_one = ExpressionNode(NodeType.CONSTANT, 1.0, parent=op_node_exc_div, depth=1, position=0) # Numerator: 1.0

    sub_op_node = ExpressionNode(NodeType.OPERATOR, '-', parent=op_node_exc_div, depth=1, position=1) # Denominator: (x-x)
    var_node_exc_lhs = ExpressionNode(NodeType.VARIABLE, x_var, parent=sub_op_node, depth=2, position=0) # x (LHS of subtraction)
    var_node_exc_rhs = ExpressionNode(NodeType.VARIABLE, x_var, parent=sub_op_node, depth=2, position=1) # x (RHS of subtraction)
    sub_op_node.children.extend([var_node_exc_lhs, var_node_exc_rhs])
    op_node_exc_div.children.extend([const_node_exc_one, sub_op_node])
    env.current_state.root = op_node_exc_div

    # For x-x, any x value will make the denominator zero.
    test_target_data_exc = np.array([
        [2.0, 0.0],      # pred = 1/(2-2) -> exception. error=penalty. Target doesn't matter.
        [1.0, 0.0],      # pred = 1/(1-1) -> exception. error=penalty. Target doesn't matter.
        [3.0, 0.0]       # pred = 1/(3-3) -> exception. error=penalty. Target doesn't matter.
    ])
    env.target_data = test_target_data_exc

    current_target_y_values_exc = env.target_data[:, env.target_variable_index]
    target_variance_exc = np.var(current_target_y_values_exc) # Will be 0 for [0,0,0]
    penalty_on_fail_exc = target_variance_exc if target_variance_exc > 1e-9 else 1.0 # Should be 1.0

    reward_exc = env._evaluate_expression()
    mse_exc = env._evaluation_cache['mse']

    # All points should result in penalty
    expected_mse_exc = np.mean([penalty_on_fail_exc, penalty_on_fail_exc, penalty_on_fail_exc])
    assert np.isclose(mse_exc, expected_mse_exc), f"MSE for exception case incorrect. Expected {expected_mse_exc}, got {mse_exc}"
    assert np.isclose(penalty_on_fail_exc, 1.0), "Penalty for zero variance target in exception case should be 1.0"


    # --- Test Case 4: Zero target variance and penalty is 1.0 ---
    # Use target data where y is constant. Expression: sqrt(x)
    constant_y_value = 5.0
    target_data_zero_var = np.array([
        [1.0, constant_y_value], # x=1, y=5. pred=sqrt(1)=1. error=(1-5)^2=16
        [-1.0, constant_y_value],# x=-1, y=5. pred=sqrt(-1)=NaN. error=penalty (should be 1.0)
        [4.0, constant_y_value]  # x=4, y=5. pred=sqrt(4)=2. error=(2-5)^2=9
    ])
    env.target_data = target_data_zero_var

    # Rebuild sqrt(x) expression from Test Case 1
    env.current_state = TreeState(max_depth=env.max_depth)
    root_node_zv = ExpressionNode(NodeType.OPERATOR, 'sqrt', depth=0)
    var_node_zv = ExpressionNode(NodeType.VARIABLE, x_var, parent=root_node_zv, depth=1)
    root_node_zv.children.append(var_node_zv)
    env.current_state.root = root_node_zv

    current_target_y_values_zv = env.target_data[:, env.target_variable_index]
    target_variance_zv = np.var(current_target_y_values_zv) # Should be 0 for [5,5,5]
    penalty_on_fail_zv = target_variance_zv if target_variance_zv > 1e-9 else 1.0 # Should be 1.0

    assert np.isclose(target_variance_zv, 0.0), "Target variance for constant y should be 0."
    assert np.isclose(penalty_on_fail_zv, 1.0), "Penalty for zero variance case should be 1.0."

    reward_zero_var = env._evaluate_expression()
    mse_zero_var = env._evaluation_cache['mse']

    expected_error_zv_0 = (np.sqrt(target_data_zero_var[0,0]) - target_data_zero_var[0,1])**2
    expected_error_zv_2 = (np.sqrt(target_data_zero_var[2,0]) - target_data_zero_var[2,1])**2
    expected_mse_zero_var = np.mean([expected_error_zv_0, penalty_on_fail_zv, expected_error_zv_2])

    assert np.isclose(mse_zero_var, expected_mse_zero_var), f"MSE for zero variance case incorrect. Expected {expected_mse_zero_var}, got {mse_zero_var}"

    # --- Test Case 5: All predictions are valid ---
    # Expression: x * 2.0
    env.current_state = TreeState(max_depth=env.max_depth)
    op_node_valid = ExpressionNode(NodeType.OPERATOR, '*', depth=0)
    var_node_valid_mult = ExpressionNode(NodeType.VARIABLE, x_var, parent=op_node_valid, depth=1, position=0)
    const_node_valid_two = ExpressionNode(NodeType.CONSTANT, 2.0, parent=op_node_valid, depth=1, position=1)
    op_node_valid.children.extend([var_node_valid_mult, const_node_valid_two])
    env.current_state.root = op_node_valid

    # Use a simple, all-valid target data
    test_target_data_valid = np.array([
        [1.0, 2.0], # pred = 1*2=2, target=2, error=0
        [2.0, 4.0], # pred = 2*2=4, target=4, error=0
        [3.0, 6.0]  # pred = 3*2=6, target=6, error=0
    ])
    env.target_data = test_target_data_valid

    # Verify penalty calculation for this valid case (it won't be used if all valid, but good to check)
    current_target_y_values_valid = env.target_data[:, env.target_variable_index]
    target_variance_valid = np.var(current_target_y_values_valid) # Var of [2,4,6]
    # penalty_on_fail_valid = target_variance_valid if target_variance_valid > 1e-9 else 1.0

    reward_valid = env._evaluate_expression()
    mse_valid = env._evaluation_cache['mse']
    complexity_valid = env._evaluation_cache['complexity'] # Check complexity is reported

    expected_mse_valid = np.mean([
        (test_target_data_valid[0,0]*2.0 - test_target_data_valid[0,1])**2,
        (test_target_data_valid[1,0]*2.0 - test_target_data_valid[1,1])**2,
        (test_target_data_valid[2,0]*2.0 - test_target_data_valid[2,1])**2,
    ])
    assert np.isclose(mse_valid, expected_mse_valid), f"MSE for all valid case incorrect. Expected {expected_mse_valid}, got {mse_valid}"
    assert isinstance(reward_valid, float), "Reward for valid case should be a float."
    assert 'complexity' in env._evaluation_cache, "Complexity should be in evaluation cache."

    # Check reward calculation components for the valid case
    # norm = mse / (target_variance + 1e-10)
    # reward = (
    #     self.reward_config.get('completion_bonus', 0.1) +
    #     self.reward_config.get('mse_weight', 1.0) * np.exp(-self.reward_config.get('mse_scale_factor', 1.0) * norm) +
    #     self.reward_config.get('complexity_penalty', -0.01) * expr.complexity +
    #     self.reward_config.get('depth_penalty', -0.001) * self.max_depth
    # )

    norm_valid = mse_valid / (target_variance_valid + 1e-10)
    expected_reward_calc = (
        env.reward_config.get('completion_bonus') +
        env.reward_config.get('mse_weight') * np.exp(-env.reward_config.get('mse_scale_factor') * norm_valid) +
        env.reward_config.get('complexity_penalty') * complexity_valid +
        env.reward_config.get('depth_penalty') * env.max_depth
    )
    assert np.isclose(reward_valid, expected_reward_calc), f"Full reward calculation for valid case is incorrect. Expected {expected_reward_calc}, got {reward_valid}"

    # --- Test Case 6: Expression complexity exceeds max_complexity ---
    env.max_complexity = 2 # Set a very low max_complexity for testing this
    # Use the x*2.0 expression (complexity should be 3: x, 2.0, *)
    env.current_state.root = op_node_valid # Same as Test Case 5
    env.target_data = test_target_data_valid # Use valid data, outcome shouldn't depend on MSE here

    reward_exceed_complexity = env._evaluate_expression()
    # Expected reward: complexity_penalty * expr.complexity
    # The expression is x*2.0. Assume its complexity is 3.
    # The grammar and to_expression must correctly calculate complexity.
    # We need to ensure the expression object from to_expression has 'complexity'.
    # The Expression class from progressive_grammar_system.py should provide this.

    # Need to get the actual complexity value that ProgressiveGrammar would assign.
    # For testing, let's assume a fixed complexity for "x*2.0" if ProgressiveGrammar is complex.
    # However, ProgressiveGrammar's Expression class calculates it as 1 (op) + sum of child complexities.
    # Variable is 1. A constant node like '2.0' becomes Expression('const', [2.0]), which has complexity 1 (for 'const') + 1 (for 2.0) = 2.
    # So x*2.0 is 1 (for '*') + 1 (for x_var) + 2 (for Expression('const', [2.0])) = 4.
    expected_complexity_for_mult = 4

    expected_reward_exceed = env.reward_config.get('complexity_penalty', -0.01) * expected_complexity_for_mult
    assert np.isclose(reward_exceed_complexity, expected_reward_exceed), \
        f"Reward for exceeding max_complexity is incorrect. Expected {expected_reward_exceed}, got {reward_exceed_complexity}"

    # Reset max_complexity for other tests if env is reused (pytest fixtures usually recreate)
    env.max_complexity = 30


    # --- Test Case 7: Incomplete expression (root is EMPTY) ---
    env.current_state = TreeState(max_depth=env.max_depth) # Fresh empty state
    assert env.current_state.root.node_type == NodeType.EMPTY

    reward_incomplete = env._evaluate_expression()
    expected_reward_incomplete = env.reward_config.get('timeout_penalty', -1.0)
    assert np.isclose(reward_incomplete, expected_reward_incomplete), \
        f"Reward for incomplete (empty root) expression is incorrect. Expected {expected_reward_incomplete}, got {reward_incomplete}"

    # --- Test Case 8: Incomplete expression (operator with missing child) ---
    env.current_state = TreeState(max_depth=env.max_depth) # Ensure it's a fresh state for this sub-test
    root_incomplete_op = ExpressionNode(NodeType.OPERATOR, '+', depth=0)
    var_child_incomplete = ExpressionNode(NodeType.VARIABLE, x_var, parent=root_incomplete_op, depth=1)
    root_incomplete_op.children.append(var_child_incomplete) # Only one child for binary op '+'
    env.current_state.root = root_incomplete_op

    # to_expression should return None for this incomplete tree
    assert env.current_state.root.to_expression(grammar) is None, "Incomplete op node should not produce an expression."

    reward_incomplete_op_eval = env._evaluate_expression()
    assert np.isclose(reward_incomplete_op_eval, expected_reward_incomplete), \
        f"Reward for incomplete (operator missing child) expression is incorrect. Expected {expected_reward_incomplete}, got {reward_incomplete_op_eval}"


# Ensure logging is available for caplog fixture
import logging
from unittest.mock import patch, MagicMock

@pytest.fixture
def env_for_reset_test():
    grammar = MockGrammar() # Use the simple MockGrammar from above
    variables = [Variable(name="v0", index=0, properties={})]
    target_data = np.array([[1.0, 2.0],[2.0, 3.0]])
    env = SymbolicDiscoveryEnv(
        grammar=grammar,
        target_data=target_data,
        variables=variables,
        max_nodes=5 # Smaller max_nodes for easier testing of obs shape
    )
    return env

class TestSymbolicDiscoveryEnvReset:

    def test_successful_reset(self, env_for_reset_test):
        env = env_for_reset_test
        obs, info = env.reset()

        assert isinstance(obs, np.ndarray), "Observation should be a numpy array"
        assert obs.shape == env.observation_space.shape, "Observation shape mismatch"
        assert isinstance(info, dict), "Info should be a dictionary"
        assert not info.get("reset_failed", False), "Reset should not indicate failure"

    def test_reset_with_uninitialized_grammar(self, env_for_reset_test, caplog):
        env = env_for_reset_test
        env.grammar = MagicMock(spec=ProgressiveGrammar) # Mock it
        # Simulate uninitialized primitives
        # Option 1: grammar is None (though __init__ type hint is Any, it expects ProgressiveGrammar compatible)
        # env.grammar = None # This would likely cause AttributeError earlier if not checked.
        # Option 2: grammar.primitives is missing
        del env.grammar.primitives
        # Option 3: grammar.primitives is empty
        # env.grammar.primitives = {}

        with caplog.at_level(logging.ERROR):
            obs, info = env.reset()

        assert isinstance(obs, np.ndarray), "Default observation should be a numpy array"
        assert obs.shape == env.observation_space.shape, "Default observation shape mismatch"
        assert np.all(obs == 0), "Default observation should be zeros"
        assert isinstance(info, dict), "Default info should be a dictionary"
        assert info.get("reset_failed", False) is True, "Reset should indicate failure"
        assert "Grammar not properly initialized" in caplog.text

    def test_reset_with_get_observation_returns_none(self, env_for_reset_test, caplog):
        env = env_for_reset_test
        with patch.object(env, '_get_observation', return_value=None):
            with caplog.at_level(logging.ERROR):
                obs, info = env.reset()

        assert isinstance(obs, np.ndarray)
        assert obs.shape == env.observation_space.shape
        assert np.all(obs == 0)
        assert info.get("reset_failed", True)
        assert "Generated initial observation is None" in caplog.text

    def test_reset_with_get_observation_wrong_shape_no_reshape(self, env_for_reset_test, caplog):
        env = env_for_reset_test
        wrong_shape_obs = np.array([1, 2, 3], dtype=np.float32) # Clearly wrong shape and size

        with patch.object(env, '_get_observation', return_value=wrong_shape_obs):
            with caplog.at_level(logging.ERROR):
                obs, info = env.reset()

        assert isinstance(obs, np.ndarray)
        assert obs.shape == env.observation_space.shape
        assert np.all(obs == 0)
        assert info.get("reset_failed", True)
        assert "Initial observation shape/type mismatch" in caplog.text
        assert "incompatible with expected" in caplog.text # From the more specific error message

    def test_reset_with_get_observation_wrong_shape_can_reshape(self, env_for_reset_test, caplog):
        env = env_for_reset_test
        # Correct total size, but wrong shape (e.g., flattened version)
        correct_total_size = np.prod(env.observation_space.shape)
        obs_flattened = np.arange(correct_total_size, dtype=np.float32) # Shape (N,)

        with patch.object(env, '_get_observation', return_value=obs_flattened):
            with caplog.at_level(logging.WARNING): # Reshape attempt logs a warning
                obs, info = env.reset()

        assert isinstance(obs, np.ndarray)
        assert obs.shape == env.observation_space.shape # Should be reshaped
        assert np.array_equal(obs, obs_flattened.reshape(env.observation_space.shape))
        assert not info.get("reset_failed", False)
        assert "Successfully reshaped observation" in caplog.text

    def test_reset_with_get_observation_wrong_shape_cannot_reshape(self, env_for_reset_test, caplog):
        env = env_for_reset_test
        # Correct total size, but wrong shape that cannot be broadcast by reshape (e.g. add extra dim)
        correct_total_size = np.prod(env.observation_space.shape)
        obs_wrong_dim = np.arange(correct_total_size, dtype=np.float32).reshape(1, correct_total_size)
        # This specific reshape error check might be tricky if the target shape is 1D already.
        # Let's make a shape that is guaranteed to fail reshaping if the target is not (1, N)
        # Default obs space shape is (max_nodes * 128,). Let's make obs_wrong_dim (max_nodes, 128)
        # if max_nodes * 128 is the total size.

        if len(env.observation_space.shape) == 1: # e.g. (640,)
            if env.max_nodes * 128 == correct_total_size : # Check if it's the default structure
                 obs_bad_reshape = np.arange(correct_total_size, dtype=np.float32).reshape(env.max_nodes, 128)
            else: # Cannot make a meaningful bad reshape for this test case, skip or adapt
                pytest.skip("Cannot construct a meaningful bad-reshapable obs for this obs space dynamically yet.")
                return
        else: # If obs space is already multi-dimensional
            obs_bad_reshape = np.arange(correct_total_size, dtype=np.float32).reshape(1, correct_total_size)


        with patch.object(env, '_get_observation', return_value=obs_bad_reshape):
            with caplog.at_level(logging.ERROR):
                obs, info = env.reset()

        assert isinstance(obs, np.ndarray)
        assert obs.shape == env.observation_space.shape # Should be default zeros
        assert np.all(obs == 0)
        assert info.get("reset_failed", True)
        assert "Initial observation shape/type mismatch" in caplog.text
        assert "incompatible (cannot reshape)" in caplog.text # Check for the reshape failure message
