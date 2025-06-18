import pytest
import numpy as np
import sympy as sp
from unittest.mock import patch, MagicMock

# Imports from the project
from experiment_runner import ExperimentConfig, PhysicsDiscoveryExperiment
from progressive_grammar_system import Variable, Expression
# SymbolicRegressor is imported for type hinting and patching its definition location
from physics_discovery_extensions import SymbolicRegressor
from integrated_pipeline import JanusConfig, SyntheticDataParamsConfig, RewardConfig


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
            reward_config=RewardConfig(type="mse", mse_weight=1.0),
            num_evaluation_cycles=5,
            genetic_generations=5,
            genetic_population_size=10,
            max_complexity=10,
            results_dir="./test_results_pde_vr",
            tracker_autosave_interval=0,
            conservation_types=[],
            logger_backends=[],
            strict_mode=False,
            max_depth=5,
            policy_hidden_dim=64,
            policy_encoder_type='mlp',
            ppo_rollout_length=128,
            ppo_n_epochs=4,
            ppo_batch_size=32,
            redis_host='localhost',
            redis_port=6379,
            redis_channel_base='janus_test'
        )

        exp_config_idx0 = ExperimentConfig.from_janus_config(
            name="test_genetic_remapping_pde_idx0",
            experiment_type='physics_discovery_example',
            janus_config=janus_config,
            algorithm_name='genetic',
            target_variable_index=0,
            n_runs=1, seed=42
        )

        # --- Experiment Setup for target_idx = 0 ---
        algo_registry_mock_idx0 = {'genetic': MagicMock(return_value=mock_regressor_instance)}
        env_registry_mock_idx0 = {'harmonic_oscillator': MagicMock()}

        experiment_idx0 = PhysicsDiscoveryExperiment(
            config=exp_config_idx0,
            algo_registry=algo_registry_mock_idx0,
            env_registry=env_registry_mock_idx0
        )

        original_variables_list = [
            Variable(name="y_orig", index=0, properties={}),
            Variable(name="v1_feature", index=1, properties={}),
            Variable(name="v2_feature", index=2, properties={}),
        ]
        experiment_idx0.variables = original_variables_list
        experiment_idx0.env_data = np.array([
            [1.0, 10.0, 100.0],
            [2.0, 20.0, 200.0],
            [3.0, 30.0, 300.0],
        ])
        experiment_idx0.algorithm = mock_regressor_instance
        experiment_idx0.janus_config = janus_config
        experiment_idx0.sympy_vars = [sp.symbols(v.name) for v in original_variables_list]


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
        mock_expr_obj.symbolic = "x0 + x1"

        exp_config_neg1 = ExperimentConfig.from_janus_config(
            name="test_genetic_remapping_pde_neg1",
            experiment_type='physics_discovery_example',
            janus_config=janus_config,
            algorithm_name='genetic',
            target_variable_index=-1,
            n_runs=1, seed=42
        )

        algo_registry_mock_neg1 = {'genetic': MagicMock(return_value=mock_regressor_instance)}
        env_registry_mock_neg1 = {'harmonic_oscillator': MagicMock()}

        experiment_neg1 = PhysicsDiscoveryExperiment(
            config=exp_config_neg1,
            algo_registry=algo_registry_mock_neg1,
            env_registry=env_registry_mock_neg1
        )
        experiment_neg1.variables = original_variables_list
        experiment_neg1.env_data = experiment_idx0.env_data
        experiment_neg1.algorithm = mock_regressor_instance
        experiment_neg1.janus_config = janus_config
        experiment_neg1.sympy_vars = experiment_idx0.sympy_vars


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
