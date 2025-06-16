import pytest
import numpy as np # Still needed for ExperimentConfig if it uses it

# Global mocks and sys.path manipulations are now in conftest.py.

from unittest.mock import patch, MagicMock

# These imports should now work due to conftest.py handling path and global mocks
from experiment_runner import ExperimentRunner, ExperimentConfig, ExperimentResult


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

if __name__ == "__main__":
    pytest.main(['-v', '-s', __file__])
