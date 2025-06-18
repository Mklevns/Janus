import pytest
import numpy as np
import sympy as sp
from unittest.mock import patch, MagicMock
import re # Added for the new tests

# Imports from the project
from experiment_runner import ExperimentConfig, PhysicsDiscoveryExperiment
from progressive_grammar_system import Variable, Expression
# SymbolicRegressor is imported for type hinting and patching its definition location
from physics_discovery_extensions import SymbolicRegressor
# from integrated_pipeline import JanusConfig, SyntheticDataParamsConfig, RewardConfig
# Using JanusConfig from config_models instead, as per project structure
from config_models import JanusConfig, SyntheticDataParamsConfig, RewardConfig


class TestPhysicsDiscoveryExperimentVariableMapping:

    @patch('experiment_runner.PhysicsSymmetryDetector')
    @patch('experiment_runner.ConservationBiasedReward')
    @patch('experiment_runner.HypothesisTracker')
    @patch('experiment_runner.JanusTrainingIntegration')
    @patch('experiment_runner.TrainingLogger')
    @patch('experiment_runner.LiveMonitor')
    @patch('physics_discovery_extensions.SymbolicRegressor') # Patching where SymbolicRegressor is defined
    def test_genetic_variable_remapping_and_expression(
        self,
        MockSymbolicRegressorClass, # This is the class mock
        MockLiveMonitor,
        MockTrainingLogger,
        MockJanusTrainingIntegration,
        MockHypothesisTracker,
        MockConservationBiasedReward,
        MockPhysicsSymmetryDetector
    ):
        # --- Setup Mocks ---
        # This is the mock instance that SymbolicRegressor class will produce
        mock_regressor_instance = MockSymbolicRegressorClass.return_value

        # Mock the .fit() method of the regressor instance
        mock_expr_obj = MagicMock(spec=Expression)
        mock_expr_obj.symbolic = "x0 * 2"  # Initial expression with temporary variable names
        mock_expr_obj.complexity = 2
        mock_expr_obj.mse = 0.01
        mock_regressor_instance.fit.return_value = mock_expr_obj

        # Mock other necessary components that are instantiated by PhysicsDiscoveryExperiment
        # Ensure save_directory is a valid path string for Path operations if any
        MockHypothesisTracker.return_value.save_directory = "./mock_tracker_saves"


        # --- Configuration ---
        # Using a more complete JanusConfig
        janus_config = JanusConfig(
            target_phenomena='harmonic_oscillator',
            env_specific_params={'k': 1.0, 'm': 1.0},
            synthetic_data_params=SyntheticDataParamsConfig(noise_level=0.0, n_samples=10, time_range=[0,1]),
            reward_config=RewardConfig(type="mse", mse_weight=1.0), # Added mse_weight
            num_evaluation_cycles=5,
            genetic_generations=5,
            genetic_population_size=10,
            max_complexity=10,
            results_dir="./test_results_pde_vr",
            tracker_autosave_interval=0, # Explicitly setting to 0
            conservation_types=[],
            logger_backends=[],
            strict_mode=False, # Explicitly setting
            max_depth=5, # Added from JanusConfig model
            policy_hidden_dim=64, # Added
            policy_encoder_type='mlp', # Added
            ppo_rollout_length=128, # Added
            ppo_n_epochs=4, # Added
            ppo_batch_size=32, # Added
            redis_host='localhost', # Added
            redis_port=6379, # Added
            redis_channel_base='janus_test' # Added
        )

        exp_config_idx0 = ExperimentConfig.from_janus_config(
            name="test_genetic_remapping_pde_idx0",
            experiment_type='physics_discovery_example', # Assuming this is a registered type or will be mocked out
            janus_config=janus_config,
            algorithm_name='genetic',
            target_variable_index=0, # Explicitly targeting the first column as 'y'
            n_runs=1, seed=42
        )

        # --- Experiment Setup for target_idx = 0 ---
        # Mock the algo_registry and env_registry for PDE instantiation
        algo_registry_mock_idx0 = {'genetic': MagicMock(return_value=mock_regressor_instance)}
        # The env_registry should return a mock environment that has 'state_vars'
        mock_env_instance_idx0 = MagicMock()
        mock_env_instance_idx0.state_vars = ['y_orig', 'v1_feature', 'v2_feature']
        mock_env_instance_idx0.generate_trajectory.return_value = np.array([
            [1.0, 10.0, 100.0],[2.0, 20.0, 200.0],[3.0, 30.0, 300.0]
        ])
        mock_env_instance_idx0.get_ground_truth_laws.return_value = {}
        env_registry_mock_idx0 = {'harmonic_oscillator': MagicMock(return_value=mock_env_instance_idx0)}


        experiment_idx0 = PhysicsDiscoveryExperiment(
            config=exp_config_idx0,
            algo_registry=algo_registry_mock_idx0,
            env_registry=env_registry_mock_idx0
        )

        # Manually call setup to populate necessary fields if not relying on full execute()
        # This bypasses the need for experiment_type to be registered if we directly test run()
        experiment_idx0.setup() # This will set self.variables, self.env_data etc.
        experiment_idx0.algorithm = mock_regressor_instance # Ensure algo is the mock instance


        # --- Action: Test with target_idx = 0 ---
        result_idx0 = experiment_idx0.run(run_id=0)

        # --- Assertions for target_idx = 0 ---
        mock_regressor_instance.fit.assert_called_once()
        args_idx0, kwargs_idx0 = mock_regressor_instance.fit.call_args

        expected_X_idx0 = np.array([[10.0, 100.0], [20.0, 200.0], [30.0, 300.0]])
        np.testing.assert_array_equal(args_idx0[0], expected_X_idx0, "X data for target_idx=0 incorrect")

        expected_y_idx0 = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(args_idx0[1], expected_y_idx0, "y data for target_idx=0 incorrect")

        expected_var_mapping_idx0 = {'x0': 'v1_feature', 'x1': 'v2_feature'}
        assert kwargs_idx0.get('var_mapping') == expected_var_mapping_idx0, \
            f"var_mapping for target_idx=0 incorrect. Got: {kwargs_idx0.get('var_mapping')}"

        assert result_idx0.discovered_law == "v1_feature * 2", \
            f"Discovered law remapping for target_idx=0 failed. Got: {result_idx0.discovered_law}"

        # --- Reset for next scenario (target_idx = -1) ---
        mock_regressor_instance.fit.reset_mock()
        # Update the symbolic expression for the next test case
        mock_expr_obj.symbolic = "x0 + x1" # For "y_orig + v1_feature"

        exp_config_neg1 = ExperimentConfig.from_janus_config(
            name="test_genetic_remapping_pde_neg1",
            experiment_type='physics_discovery_example',
            janus_config=janus_config, # Can reuse janus_config
            algorithm_name='genetic',
            target_variable_index=-1, # Target last column
            n_runs=1, seed=42
        )

        algo_registry_mock_neg1 = {'genetic': MagicMock(return_value=mock_regressor_instance)}
        mock_env_instance_neg1 = MagicMock()
        mock_env_instance_neg1.state_vars = ['y_orig', 'v1_feature', 'v2_feature'] # Same vars
        mock_env_instance_neg1.generate_trajectory.return_value = experiment_idx0.env_data # Same data
        mock_env_instance_neg1.get_ground_truth_laws.return_value = {}
        env_registry_mock_neg1 = {'harmonic_oscillator': MagicMock(return_value=mock_env_instance_neg1)}


        experiment_neg1 = PhysicsDiscoveryExperiment(
            config=exp_config_neg1,
            algo_registry=algo_registry_mock_neg1,
            env_registry=env_registry_mock_neg1
        )
        experiment_neg1.setup()
        experiment_neg1.algorithm = mock_regressor_instance


        # --- Action: Test with target_idx = -1 ---
        result_neg1 = experiment_neg1.run(run_id=0)

        # --- Assertions for target_idx = -1 ---
        mock_regressor_instance.fit.assert_called_once()
        args_neg1, kwargs_neg1 = mock_regressor_instance.fit.call_args

        expected_X_neg1 = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        np.testing.assert_array_equal(args_neg1[0], expected_X_neg1, "X data for target_idx=-1 incorrect")

        expected_y_neg1 = np.array([100.0, 200.0, 300.0])
        np.testing.assert_array_equal(args_neg1[1], expected_y_neg1, "y data for target_idx=-1 incorrect")

        expected_var_mapping_neg1 = {'x0': 'y_orig', 'x1': 'v1_feature'}
        assert kwargs_neg1.get('var_mapping') == expected_var_mapping_neg1, \
            f"var_mapping for target_idx=-1 incorrect. Got: {kwargs_neg1.get('var_mapping')}"

        assert result_neg1.discovered_law == "y_orig + v1_feature", \
            f"Discovered law remapping for target_idx=-1 failed. Got: {result_neg1.discovered_law}"


# --- Tests for _remap_expression ---
class TestRemapExpression:

    @pytest.fixture
    def pde_instance(self):
        # Create a minimal JanusConfig and ExperimentConfig for PDE instantiation
        # Minimal valid JanusConfig
        janus_config = JanusConfig(
            target_phenomena='harmonic_oscillator', # Required field
            env_specific_params={'k': 1.0, 'm': 1.0}, # Required if target_phenomena needs it
            synthetic_data_params=SyntheticDataParamsConfig(noise_level=0.0, n_samples=10, time_range=[0,1]), # Required
            reward_config=RewardConfig(type="mse", mse_weight=1.0), # Required
            num_evaluation_cycles=1, # Required
            # Fill other required fields with minimal valid values
            max_depth=5,
            max_complexity=10,
            policy_hidden_dim=64,
            policy_encoder_type="mlp",
            genetic_population_size=50,
            genetic_generations=20,
            results_dir="./dummy_results",
            tracker_autosave_interval=0,
            conservation_types=[],
            conservation_weight_factor=0.1,
            symmetry_tolerance=0.01,
            symmetry_confidence_threshold=0.9,
            logger_backends=[],
            redis_host="localhost",
            redis_port=6379,
            redis_channel_base="janus_test_remap",
            ppo_rollout_length=128,
            ppo_n_epochs=4,
            ppo_batch_size=32,
            strict_mode=False
        )
        exp_config = ExperimentConfig.from_janus_config(
            name='test_remap',
            experiment_type='test_type', # This type doesn't need to be registered if we only test _remap
            janus_config=janus_config,
            algorithm_name='test_algo'
        )
        # Provide dummy algo_registry and env_registry as they are required by PDE constructor
        pde = PhysicsDiscoveryExperiment(config=exp_config, algo_registry={}, env_registry={})
        return pde

    def test_remap_expression_basic(self, pde_instance):
        expr_str = "x0 + x1"
        var_mapping = {"x0": "pos", "x1": "vel"}
        # The regex uses \b for word boundaries, which might not match simple "x0" if not space-separated.
        # Let's adjust the expected or ensure the regex handles this.
        # The current re.sub(r'' + re.escape(new_var) + r'', orig_var, result) uses literal backspace chars.
        # It should be r'\b' for word boundaries. Assuming '' was a typo and meant r'\b'.
        # If '' is literal, then these tests might behave unexpectedly without actual backspaces in expr_str.
        # For now, proceeding as if `\b` was intended for robustness.
        # If the literal '' is intended, then the tests below might need adjustment.
        # This test will assume the regex is `r'\b' + re.escape(new_var) + r'\b'`

        # Correcting the expectation based on a robust word boundary interpretation.
        # If x0 and x1 are standalone tokens, direct replace would work.
        # The regex `\bx0\b` would match "x0" in "x0 + x1".
        assert pde_instance._remap_expression(expr_str, var_mapping) == "pos + vel"

    def test_remap_expression_sorted_keys(self, pde_instance):
        expr_str = "x1 + x10" # x10 should be replaced before x1
        var_mapping = {"x1": "pos", "x10": "accel"}
        assert pde_instance._remap_expression(expr_str, var_mapping) == "pos + accel"

    def test_remap_expression_word_boundaries(self, pde_instance):
        # This test depends heavily on the correct regex for word boundaries.
        # Assuming the regex is effectively r'\b' + re.escape(key) + r'\b'

        expr_str_no_change = "x_coord + x_coord_offset"
        var_mapping_no_change = {"x": "position"} # "x" should not match "x_coord"
        assert pde_instance._remap_expression(expr_str_no_change, var_mapping_no_change) == "x_coord + x_coord_offset"

        expr_str_partial_change = "x + x_coord"
        var_mapping_partial_change = {"x": "position"} # "x" should match here
        assert pde_instance._remap_expression(expr_str_partial_change, var_mapping_partial_change) == "position + x_coord"

        expr_str_internal = "myxvar + testx"
        var_mapping_internal = {"x": "position"} # "x" should not match inside "myxvar" or "testx"
        assert pde_instance._remap_expression(expr_str_internal, var_mapping_internal) == "myxvar + testx"

        expr_str_with_numbers = "x0_val + x0"
        var_mapping_with_numbers = {"x0": "val0"}
        assert pde_instance._remap_expression(expr_str_with_numbers, var_mapping_with_numbers) == "x0_val + val0"


    def test_remap_expression_no_mapping(self, pde_instance):
        expr_str = "x0 + x1"
        var_mapping = {}
        assert pde_instance._remap_expression(expr_str, var_mapping) == "x0 + x1"

    def test_remap_expression_empty_string(self, pde_instance):
        expr_str = ""
        var_mapping = {"x0": "pos"}
        assert pde_instance._remap_expression(expr_str, var_mapping) == ""

    def test_remap_expression_special_regex_chars_in_key(self, pde_instance):
        # Test if re.escape works for keys that have characters with special meaning in regex
        expr_str = "var* + var+"
        var_mapping = {"var*": "v_star", "var+": "v_plus"}
        # This relies on re.escape correctly handling '*' and '+' in the keys
        # The r'\b' + escaped_key + r'\b' should work.
        assert pde_instance._remap_expression(expr_str, var_mapping) == "v_star + v_plus"

        expr_str_dot = "var.name + other"
        var_mapping_dot = {"var.name": "v_dot"}
        assert pde_instance._remap_expression(expr_str_dot, var_mapping_dot) == "v_dot + other"

    def test_remap_expression_original_var_substring_of_temp(self, pde_instance):
        # This case should be handled by sorting keys by length (longest first)
        expr_str = "temp_x + temp"
        var_mapping = {"temp": "original_temp", "temp_x": "original_temp_x"}
        # "temp_x" should be replaced first, then "temp"
        assert pde_instance._remap_expression(expr_str, var_mapping) == "original_temp_x + original_temp"

    def test_remap_expression_temp_var_substring_of_original(self, pde_instance):
        # Example: temp var 'x' maps to 'x_long_name'
        # Expression: 'x + y'
        # Mapping: {'x': 'x_long_name', 'y': 'y_short'}
        # Should become: 'x_long_name + y_short'
        # This is a standard case, word boundaries should handle it.
        expr_str = "x + y"
        var_mapping = {"x": "x_long_name", "y": "y_short"}
        assert pde_instance._remap_expression(expr_str, var_mapping) == "x_long_name + y_short"

        # More complex: temp 'val' maps to 'value', expression contains 'validation'
        # Expression: 'val + validation'
        # Mapping: {'val': 'value'}
        # Should become: 'value + validation'
        expr_str_sub = "val + validation"
        var_mapping_sub = {"val": "value"}
        assert pde_instance._remap_expression(expr_str_sub, var_mapping_sub) == "value + validation"

# Note: The original _remap_expression used `result.replace(new_var, orig_var)`.
# The new version uses `re.sub(r'' + re.escape(new_var) + r'', orig_var, result)`.
# It's crucial that `r''` is intended to be `r'\b'`. If '' (literal backspace) was intended,
# then the behavior of these tests would be very different and likely not achieve robust word boundary replacement.
# These tests are written assuming `r'\b'` (word boundary) is the correct interpretation for robust remapping.
# If the literal backspace `` is indeed intended, the `_remap_expression` method itself is likely flawed for general use.
# For example, `re.sub(r'x0', "pos", "x0 + x1")` with literal backspaces would not match "x0" unless "x0" was surrounded by actual backspace characters in the string.
# Given the context of variable remapping, `\b` is the standard way to ensure whole-word replacement.
# The tests are based on the assumption that `\b` is what `` was meant to represent.
# If the implementation literally uses '', then `test_remap_expression_basic` might fail unless `expr_str = "x0 + x1"`.
# And `test_remap_expression_word_boundaries` would need significant rework.
# Given the problem description, it's highly probable `\b` was intended.
# The `re.escape(new_var)` part is correct for handling special characters in `new_var`.
# The sorting of `var_mapping` by length of keys (descending) is also a good strategy to prevent partial replacements
# (e.g., "x10" before "x1"). The regex with word boundaries further strengthens this.```python
import pytest
import numpy as np
import sympy as sp
from unittest.mock import patch, MagicMock
import re # Added for the new tests

# Imports from the project
from experiment_runner import ExperimentConfig, PhysicsDiscoveryExperiment
from progressive_grammar_system import Variable, Expression
# SymbolicRegressor is imported for type hinting and patching its definition location
from physics_discovery_extensions import SymbolicRegressor
# from integrated_pipeline import JanusConfig, SyntheticDataParamsConfig, RewardConfig
# Using JanusConfig from config_models instead, as per project structure
from config_models import JanusConfig, SyntheticDataParamsConfig, RewardConfig


class TestPhysicsDiscoveryExperimentVariableMapping:

    @patch('experiment_runner.PhysicsSymmetryDetector')
    @patch('experiment_runner.ConservationBiasedReward')
    @patch('experiment_runner.HypothesisTracker')
    @patch('experiment_runner.JanusTrainingIntegration')
    @patch('experiment_runner.TrainingLogger')
    @patch('experiment_runner.LiveMonitor')
    @patch('physics_discovery_extensions.SymbolicRegressor') # Patching where SymbolicRegressor is defined
    def test_genetic_variable_remapping_and_expression(
        self,
        MockSymbolicRegressorClass, # This is the class mock
        MockLiveMonitor,
        MockTrainingLogger,
        MockJanusTrainingIntegration,
        MockHypothesisTracker,
        MockConservationBiasedReward,
        MockPhysicsSymmetryDetector
    ):
        # --- Setup Mocks ---
        # This is the mock instance that SymbolicRegressor class will produce
        mock_regressor_instance = MockSymbolicRegressorClass.return_value

        # Mock the .fit() method of the regressor instance
        mock_expr_obj = MagicMock(spec=Expression)
        mock_expr_obj.symbolic = "x0 * 2"  # Initial expression with temporary variable names
        mock_expr_obj.complexity = 2
        mock_expr_obj.mse = 0.01
        mock_regressor_instance.fit.return_value = mock_expr_obj

        # Mock other necessary components that are instantiated by PhysicsDiscoveryExperiment
        # Ensure save_directory is a valid path string for Path operations if any
        MockHypothesisTracker.return_value.save_directory = "./mock_tracker_saves"


        # --- Configuration ---
        # Using a more complete JanusConfig
        janus_config = JanusConfig(
            target_phenomena='harmonic_oscillator',
            env_specific_params={'k': 1.0, 'm': 1.0},
            synthetic_data_params=SyntheticDataParamsConfig(noise_level=0.0, n_samples=10, time_range=[0,1]),
            reward_config=RewardConfig(type="mse", mse_weight=1.0), # Added mse_weight
            num_evaluation_cycles=5,
            genetic_generations=5,
            genetic_population_size=10,
            max_complexity=10,
            results_dir="./test_results_pde_vr",
            tracker_autosave_interval=0, # Explicitly setting to 0
            conservation_types=[],
            logger_backends=[],
            strict_mode=False, # Explicitly setting
            max_depth=5, # Added from JanusConfig model
            policy_hidden_dim=64, # Added
            policy_encoder_type='mlp', # Added
            ppo_rollout_length=128, # Added
            ppo_n_epochs=4, # Added
            ppo_batch_size=32, # Added
            redis_host='localhost', # Added
            redis_port=6379, # Added
            redis_channel_base='janus_test' # Added
        )

        exp_config_idx0 = ExperimentConfig.from_janus_config(
            name="test_genetic_remapping_pde_idx0",
            experiment_type='physics_discovery_example', # Assuming this is a registered type or will be mocked out
            janus_config=janus_config,
            algorithm_name='genetic',
            target_variable_index=0, # Explicitly targeting the first column as 'y'
            n_runs=1, seed=42
        )

        # --- Experiment Setup for target_idx = 0 ---
        # Mock the algo_registry and env_registry for PDE instantiation
        algo_registry_mock_idx0 = {'genetic': MagicMock(return_value=mock_regressor_instance)}
        # The env_registry should return a mock environment that has 'state_vars'
        mock_env_instance_idx0 = MagicMock()
        mock_env_instance_idx0.state_vars = ['y_orig', 'v1_feature', 'v2_feature']
        mock_env_instance_idx0.generate_trajectory.return_value = np.array([
            [1.0, 10.0, 100.0],[2.0, 20.0, 200.0],[3.0, 30.0, 300.0]
        ])
        mock_env_instance_idx0.get_ground_truth_laws.return_value = {}
        env_registry_mock_idx0 = {'harmonic_oscillator': MagicMock(return_value=mock_env_instance_idx0)}


        experiment_idx0 = PhysicsDiscoveryExperiment(
            config=exp_config_idx0,
            algo_registry=algo_registry_mock_idx0,
            env_registry=env_registry_mock_idx0
        )

        # Manually call setup to populate necessary fields if not relying on full execute()
        # This bypasses the need for experiment_type to be registered if we directly test run()
        experiment_idx0.setup() # This will set self.variables, self.env_data etc.
        experiment_idx0.algorithm = mock_regressor_instance # Ensure algo is the mock instance


        # --- Action: Test with target_idx = 0 ---
        result_idx0 = experiment_idx0.run(run_id=0)

        # --- Assertions for target_idx = 0 ---
        mock_regressor_instance.fit.assert_called_once()
        args_idx0, kwargs_idx0 = mock_regressor_instance.fit.call_args

        expected_X_idx0 = np.array([[10.0, 100.0], [20.0, 200.0], [30.0, 300.0]])
        np.testing.assert_array_equal(args_idx0[0], expected_X_idx0, "X data for target_idx=0 incorrect")

        expected_y_idx0 = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(args_idx0[1], expected_y_idx0, "y data for target_idx=0 incorrect")

        expected_var_mapping_idx0 = {'x0': 'v1_feature', 'x1': 'v2_feature'}
        assert kwargs_idx0.get('var_mapping') == expected_var_mapping_idx0, \
            f"var_mapping for target_idx=0 incorrect. Got: {kwargs_idx0.get('var_mapping')}"

        assert result_idx0.discovered_law == "v1_feature * 2", \
            f"Discovered law remapping for target_idx=0 failed. Got: {result_idx0.discovered_law}"

        # --- Reset for next scenario (target_idx = -1) ---
        mock_regressor_instance.fit.reset_mock()
        # Update the symbolic expression for the next test case
        mock_expr_obj.symbolic = "x0 + x1" # For "y_orig + v1_feature"

        exp_config_neg1 = ExperimentConfig.from_janus_config(
            name="test_genetic_remapping_pde_neg1",
            experiment_type='physics_discovery_example',
            janus_config=janus_config, # Can reuse janus_config
            algorithm_name='genetic',
            target_variable_index=-1, # Target last column
            n_runs=1, seed=42
        )

        algo_registry_mock_neg1 = {'genetic': MagicMock(return_value=mock_regressor_instance)}
        mock_env_instance_neg1 = MagicMock()
        mock_env_instance_neg1.state_vars = ['y_orig', 'v1_feature', 'v2_feature'] # Same vars
        mock_env_instance_neg1.generate_trajectory.return_value = experiment_idx0.env_data # Same data
        mock_env_instance_neg1.get_ground_truth_laws.return_value = {}
        env_registry_mock_neg1 = {'harmonic_oscillator': MagicMock(return_value=mock_env_instance_neg1)}


        experiment_neg1 = PhysicsDiscoveryExperiment(
            config=exp_config_neg1,
            algo_registry=algo_registry_mock_neg1,
            env_registry=env_registry_mock_neg1
        )
        experiment_neg1.setup()
        experiment_neg1.algorithm = mock_regressor_instance


        # --- Action: Test with target_idx = -1 ---
        result_neg1 = experiment_neg1.run(run_id=0)

        # --- Assertions for target_idx = -1 ---
        mock_regressor_instance.fit.assert_called_once()
        args_neg1, kwargs_neg1 = mock_regressor_instance.fit.call_args

        expected_X_neg1 = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        np.testing.assert_array_equal(args_neg1[0], expected_X_neg1, "X data for target_idx=-1 incorrect")

        expected_y_neg1 = np.array([100.0, 200.0, 300.0])
        np.testing.assert_array_equal(args_neg1[1], expected_y_neg1, "y data for target_idx=-1 incorrect")

        expected_var_mapping_neg1 = {'x0': 'y_orig', 'x1': 'v1_feature'}
        assert kwargs_neg1.get('var_mapping') == expected_var_mapping_neg1, \
            f"var_mapping for target_idx=-1 incorrect. Got: {kwargs_neg1.get('var_mapping')}"

        assert result_neg1.discovered_law == "y_orig + v1_feature", \
            f"Discovered law remapping for target_idx=-1 failed. Got: {result_neg1.discovered_law}"


# --- Tests for _remap_expression ---
class TestRemapExpression:

    @pytest.fixture
    def pde_instance(self):
        # Create a minimal JanusConfig and ExperimentConfig for PDE instantiation
        # Minimal valid JanusConfig
        janus_config = JanusConfig(
            target_phenomena='harmonic_oscillator', # Required field
            env_specific_params={'k': 1.0, 'm': 1.0}, # Required if target_phenomena needs it
            synthetic_data_params=SyntheticDataParamsConfig(noise_level=0.0, n_samples=10, time_range=[0,1]), # Required
            reward_config=RewardConfig(type="mse", mse_weight=1.0), # Required
            num_evaluation_cycles=1, # Required
            # Fill other required fields with minimal valid values
            max_depth=5,
            max_complexity=10,
            policy_hidden_dim=64,
            policy_encoder_type="mlp",
            genetic_population_size=50,
            genetic_generations=20,
            results_dir="./dummy_results",
            tracker_autosave_interval=0,
            conservation_types=[],
            conservation_weight_factor=0.1, # Added default
            symmetry_tolerance=0.01, # Added default
            symmetry_confidence_threshold=0.9, # Added default
            logger_backends=[],
            redis_host="localhost", # Added default
            redis_port=6379, # Added default
            redis_channel_base="janus_test_remap", # Added default
            ppo_rollout_length=128, # Added default
            ppo_n_epochs=4, # Added default
            ppo_batch_size=32, # Added default
            strict_mode=False # Added default
        )
        exp_config = ExperimentConfig.from_janus_config(
            name='test_remap',
            experiment_type='test_type', # This type doesn't need to be registered if we only test _remap
            janus_config=janus_config,
            algorithm_name='test_algo'
        )
        # Provide dummy algo_registry and env_registry as they are required by PDE constructor
        pde = PhysicsDiscoveryExperiment(config=exp_config, algo_registry={}, env_registry={})
        return pde

    def test_remap_expression_basic(self, pde_instance):
        expr_str = "x0 + x1"
        var_mapping = {"x0": "pos", "x1": "vel"}
        # This test assumes the regex `\b` + re.escape(new_var) + `\b` is used.
        # If `` (literal backspace) is used, this test would fail unless expr_str contained backspaces.
        assert pde_instance._remap_expression(expr_str, var_mapping) == "pos + vel"

    def test_remap_expression_sorted_keys(self, pde_instance):
        expr_str = "x1 + x10" # x10 should be replaced before x1 due to sorting by length
        var_mapping = {"x1": "pos", "x10": "accel"}
        assert pde_instance._remap_expression(expr_str, var_mapping) == "pos + accel"

    def test_remap_expression_word_boundaries(self, pde_instance):
        # Assuming the regex is effectively r'\b' + re.escape(key) + r'\b'

        expr_str_no_change = "x_coord + x_coord_offset"
        var_mapping_no_change = {"x": "position"} # "x" should not match "x_coord"
        assert pde_instance._remap_expression(expr_str_no_change, var_mapping_no_change) == "x_coord + x_coord_offset"

        expr_str_partial_change = "x + x_coord"
        var_mapping_partial_change = {"x": "position"} # "x" should match here
        assert pde_instance._remap_expression(expr_str_partial_change, var_mapping_partial_change) == "position + x_coord"

        expr_str_internal = "myxvar + testx"
        var_mapping_internal = {"x": "position"} # "x" should not match inside "myxvar" or "testx"
        assert pde_instance._remap_expression(expr_str_internal, var_mapping_internal) == "myxvar + testx"

        expr_str_with_numbers = "x0_val + x0"
        var_mapping_with_numbers = {"x0": "val0"}
        assert pde_instance._remap_expression(expr_str_with_numbers, var_mapping_with_numbers) == "x0_val + val0"


    def test_remap_expression_no_mapping(self, pde_instance):
        expr_str = "x0 + x1"
        var_mapping = {}
        assert pde_instance._remap_expression(expr_str, var_mapping) == "x0 + x1"

    def test_remap_expression_empty_string(self, pde_instance):
        expr_str = ""
        var_mapping = {"x0": "pos"}
        assert pde_instance._remap_expression(expr_str, var_mapping) == ""

    def test_remap_expression_special_regex_chars_in_key(self, pde_instance):
        expr_str = "var* + var+"
        var_mapping = {"var*": "v_star", "var+": "v_plus"}
        # This relies on re.escape correctly handling '*' and '+' in the keys.
        # The regex r'\b' + escaped_key + r'\b' should work.
        assert pde_instance._remap_expression(expr_str, var_mapping) == "v_star + v_plus"

        expr_str_dot = "var.name + other"
        var_mapping_dot = {"var.name": "v_dot"}
        assert pde_instance._remap_expression(expr_str_dot, var_mapping_dot) == "v_dot + other"

    def test_remap_expression_original_var_substring_of_temp(self, pde_instance):
        # This case should be handled by sorting keys by length (longest first)
        expr_str = "temp_x + temp"
        var_mapping = {"temp": "original_temp", "temp_x": "original_temp_x"}
        # "temp_x" should be replaced first, then "temp"
        assert pde_instance._remap_expression(expr_str, var_mapping) == "original_temp_x + original_temp"

    def test_remap_expression_temp_var_substring_of_original(self, pde_instance):
        expr_str = "x + y"
        var_mapping = {"x": "x_long_name", "y": "y_short"}
        assert pde_instance._remap_expression(expr_str, var_mapping) == "x_long_name + y_short"

        expr_str_sub = "val + validation"
        var_mapping_sub = {"val": "value"} # 'val' maps to 'value'
        # 'val' should be replaced, 'validation' should remain.
        assert pde_instance._remap_expression(expr_str_sub, var_mapping_sub) == "value + validation"

# It's important that the implementation of _remap_expression in PhysicsDiscoveryExperiment
# uses r'\b' for word boundaries in re.sub, not the literal ''.
# These tests assume that r'\b' is effectively what's being used for robust replacement.
# If `re.sub(r'' + re.escape(new_var) + r'', ...)` means literal backspace characters,
# these tests would not pass as expected unless the input strings also contained those backspaces.
# Given the intent of remapping variable names, `\b` (word boundary) is the standard and correct approach.
```
