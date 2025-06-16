import pytest
import numpy as np # Still needed for ExperimentConfig if it uses it

# Global mocks and sys.path manipulations are now in conftest.py.

from unittest.mock import patch, MagicMock

# These imports should now work due to conftest.py handling path and global mocks
from experiment_runner import ExperimentRunner, ExperimentConfig, ExperimentResult
from progressive_grammar_system import Variable # Added for type hinting or simple var creation
from physics_discovery_extensions import SymbolicRegressor # For mocking


class TestExperimentRunner:
    @patch('experiment_runner.ExperimentRunner._run_janus_experiment')
    @patch('experiment_runner.ExperimentRunner.setup_experiment') # Also mock setup_experiment
    def test_janus_hypothesis_extraction_mocked(self, mock_setup_experiment, mock_internal_run_janus_experiment):
        """
        Tests that run_single_experiment correctly utilizes the result from
        _run_janus_experiment, specifically that discovered_law is propagated.
        Global mocks (from conftest.py) handle import issues;
        patches here handle runtime behavior of specific methods.
        """

        mock_config_instance = ExperimentConfig(
            name="test_janus_extraction_mocked",
            environment_type='harmonic_oscillator',
            algorithm='janus_full',
            env_params={'k': 1.0, 'm': 1.0},
            noise_level=0.0,
            max_experiments=1,
            trajectory_length=10,
            n_trajectories=1,
            sampling_rate=0.1,
            n_runs=1,
            seed=42
        )

        # Configure mock for setup_experiment
        mock_setup_data = {
            'physics_env': MagicMock(),
            'env_data': np.array([[1.0, 2.0]]),
            'variables': [MagicMock()],
            'algorithm': MagicMock(),
            'ground_truth': {}
        }
        mock_setup_experiment.return_value = mock_setup_data

        # Configure mock for _run_janus_experiment
        def side_effect_func_janus(setup_dict, config_obj, result_obj_passed_in):
            result_obj_passed_in.discovered_law = "mocked_hypothesis_law_string"
            result_obj_passed_in.predictive_mse = 0.123
            result_obj_passed_in.sample_efficiency_curve = [(10, 0.123)]
            result_obj_passed_in.n_experiments_to_convergence = 10
            return result_obj_passed_in

        mock_internal_run_janus_experiment.side_effect = side_effect_func_janus

        runner = ExperimentRunner(use_wandb=False)
        final_result = runner.run_single_experiment(mock_config_instance, run_id=0)

        mock_setup_experiment.assert_called_once_with(mock_config_instance)
        mock_internal_run_janus_experiment.assert_called_once()

        args, kwargs = mock_internal_run_janus_experiment.call_args
        assert args[0] == mock_setup_data
        assert args[1] == mock_config_instance
        assert isinstance(args[2], ExperimentResult)

        assert final_result.discovered_law == "mocked_hypothesis_law_string"
        assert isinstance(final_result.discovered_law, str)
        assert final_result.predictive_mse == 0.123

    def test_genetic_explicit_target(self):
        """
        Tests that _run_genetic_experiment uses the target_variable_index
        from config to correctly select X and y for the regressor.
        """
        target_col_idx = 0 # Target is the first column
        config = ExperimentConfig(
            name="test_genetic_target_idx",
            environment_type='harmonic_oscillator', # Doesn't matter much for this test
            algorithm='genetic',
            target_variable_index=target_col_idx,
            # Other params don't matter much for this specific test
            n_runs=1, seed=42
        )

        env_data = np.array([
            [1.0, 10.0, 100.0],
            [2.0, 20.0, 200.0],
            [3.0, 30.0, 300.0],
            [4.0, 40.0, 400.0]
        ])

        # Mock variables; their internal details don't matter much, only their count perhaps
        # For X, we'd have 2 variables if target is one column out of 3
        mock_vars = [
            Variable(name="x0", index=0, properties={}), # Corresponds to original col 1 after target removal
            Variable(name="x1", index=1, properties={})  # Corresponds to original col 2 after target removal
        ]


        mock_regressor = MagicMock(spec=SymbolicRegressor)
        # Mock the fit method to return a dummy Expression object or similar
        dummy_expr_result = MagicMock()
        dummy_expr_result.symbolic = "mocked_expr"
        dummy_expr_result.complexity = 1
        mock_regressor.fit.return_value = dummy_expr_result

        setup_dict = {
            'env_data': env_data,
            'variables': mock_vars, # These variables are for the features X, not all original cols
            'algorithm': mock_regressor,
            'ground_truth': {} # Not used by _run_genetic_experiment
        }

        # Initial result object
        result_obj = ExperimentResult(config=config, run_id=0)

        runner = ExperimentRunner(use_wandb=False)
        runner._run_genetic_experiment(setup_dict, config, result_obj)

        # Assert that regressor.fit was called
        mock_regressor.fit.assert_called_once()

        # Get the arguments passed to fit
        args, kwargs_fit = mock_regressor.fit.call_args

        # X_fit should be env_data without the target_col_idx
        expected_X_fit = np.delete(env_data, target_col_idx, axis=1)
        np.testing.assert_array_equal(args[0], expected_X_fit,
                                      "X data passed to regressor.fit is incorrect.")

        # y_fit should be the target_col_idx from env_data
        expected_y_fit = env_data[:, target_col_idx]
        np.testing.assert_array_equal(args[1], expected_y_fit,
                                      "y data passed to regressor.fit is incorrect.")

        # Check variables passed to fit - this is tricky because X's columns are re-indexed.
        # The current _run_genetic_experiment implementation does not re-index variables passed to fit.
        # It passes setup_dict['variables']. This might be a bug in _run_genetic_experiment or
        # an assumption that SymbolicRegressor handles it.
        # For now, let's assert that the original variables (meant for X) were passed.
        # If target_col_idx = 0, then variables for X (cols 1, 2) should be passed.
        # The `mock_vars` I created are already for the reduced X.
        # The number of variables should match number of columns in X_fit.
        self.assertEqual(len(args[2]), expected_X_fit.shape[1],
                         "Number of variables passed to fit does not match X columns.")


    @patch('experiment_runner.SymbolicDiscoveryEnv') # Mock at the location it's imported/used
    def test_rl_env_explicit_target_instantiation(self, MockSymbolicDiscoveryEnv):
        """
        Tests that SymbolicDiscoveryEnv is instantiated with the correct
        target_variable_index when specified in ExperimentConfig, for RL algorithms.
        """
        target_col_idx = 1 # Target is the second column
        config = ExperimentConfig(
            name="test_rl_target_idx_instantiation",
            environment_type='pendulum', # Example, doesn't deeply matter
            algorithm='janus_full', # This uses SymbolicDiscoveryEnv
            target_variable_index=target_col_idx,
            algo_params={'env_params': {}}, # Ensure env_params exists
            n_runs=1, seed=42
        )

        # Dummy data for setup_experiment to run far enough
        env_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        variables = [Variable("v1", 0, {}), Variable("v2", 1, {}), Variable("v3", 2, {})]

        # Mock the return of SymbolicDiscoveryEnv constructor if needed for later calls
        mock_env_instance = MagicMock()
        MockSymbolicDiscoveryEnv.return_value = mock_env_instance

        runner = ExperimentRunner(use_wandb=False)

        # The crucial call is within algo_registry['janus_full'](...)
        # This is called by setup_experiment.
        # We can call setup_experiment or call the registry function more directly
        # For simplicity, let's assume setup_experiment calls it.
        # We need to provide enough mocks for setup_experiment to run up to that point.

        # Let create_janus_full be called by the algo_registry
        # No, this is simpler: the config.target_variable_index is passed to create_janus_full
        # which then passes it to SymbolicDiscoveryEnv.
        # So we need to check the args of SymbolicDiscoveryEnv.

        # `create_janus_full` is a local function in experiment_runner.
        # It's easier to just call it via the registry after setting it up.
        # The algorithm is created within `setup_experiment`.
        # Let's allow `setup_experiment` to run, but it will use the patched SymbolicDiscoveryEnv.

        # To let setup_experiment run but control algorithm creation slightly,
        # we can also patch 'experiment_runner.ProgressiveGrammar' and 'experiment_runner.HypothesisNet',
        # 'experiment_runner.PPOTrainer' if their instantiation is complex.
        # For now, let's assume they can be instantiated simply or their mocks are enough.
        with patch('experiment_runner.ProgressiveGrammar', MagicMock()), \
             patch('experiment_runner.HypothesisNet', MagicMock()), \
             patch('experiment_runner.PPOTrainer', MagicMock()):
            # setup_experiment populates the algo_registry and then calls it.
            # The actual instantiation of SymbolicDiscoveryEnv happens inside `create_janus_full`
            # which is called by `self.algo_registry[config.algorithm](env_data, variables, config)`
            # within `setup_experiment`.
            runner.setup_experiment(config)


        # Assert that SymbolicDiscoveryEnv was called with target_variable_index
        MockSymbolicDiscoveryEnv.assert_called_once()
        args, kwargs = MockSymbolicDiscoveryEnv.call_args

        # target_variable_index is passed via env_params in create_janus_full
        # which are then **expanded in SymbolicDiscoveryEnv constructor
        # The modified create_janus_full in experiment_runner.py does:
        #   env_creation_params = config.algo_params.get('env_params', {})
        #   env_creation_params['target_variable_index'] = config.target_variable_index
        #   discovery_env = SymbolicDiscoveryEnv(..., **env_creation_params)
        # So, target_variable_index should be in kwargs of SymbolicDiscoveryEnv constructor.
        self.assertIn('target_variable_index', kwargs, "target_variable_index not in SDE kwargs")
        self.assertEqual(kwargs['target_variable_index'], target_col_idx,
                         "SymbolicDiscoveryEnv instantiated with incorrect target_variable_index.")


if __name__ == "__main__":
    pytest.main(['-v', '-s', __file__])
