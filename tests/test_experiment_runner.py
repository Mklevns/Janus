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

# For testing validate_imports
from experiment_runner import validate_imports, REQUIRED_MODULES
from custom_exceptions import MissingDependencyError
import importlib # To get the original import_module for some tests


# Store the original import_module to be able to use it in tests if needed
# And to ensure we can restore it if patching affects other tests (though pytest handles this)
original_import_module = importlib.import_module


class TestValidateImports:
    def test_successful_import(self):
        mock_globals = {}
        test_modules = {
            "numpy": "np",
            "typing": ["Dict", "List"],
            "json": None # Direct import
        }

        # Mock import_module behavior
        def mock_import(name):
            mocked_module = MagicMock()
            mocked_module.Dict = dict # Simulate Dict object
            mocked_module.List = list # Simulate List object
            if name == "json":
                 return importlib.import_module("json") # Use real json for simplicity
            return mocked_module

        with patch('importlib.import_module', side_effect=mock_import) as mock_importer:
            validate_imports(test_modules, mock_globals, scope="global")

            assert "np" in mock_globals
            assert "Dict" in mock_globals
            assert "List" in mock_globals
            assert "json" in mock_globals
            assert mock_globals["Dict"] == dict
            mock_importer.assert_any_call("numpy")
            mock_importer.assert_any_call("typing")
            mock_importer.assert_any_call("json")

    def test_missing_required_module(self):
        mock_globals = {}
        test_modules = {"non_existent_module": None}

        with patch('importlib.import_module', side_effect=ImportError("Module not found!")):
            with pytest.raises(MissingDependencyError, match="Module 'non_existent_module' could not be imported"):
                validate_imports(test_modules, mock_globals, scope="global")

    def test_missing_specific_import(self):
        mock_globals = {}
        test_modules = {"typing": ["NonExistentObject"]}

        mock_typing_module = MagicMock()
        # Make getattr raise AttributeError when NonExistentObject is accessed
        def mock_getattr(module, name):
            if name == "NonExistentObject":
                raise AttributeError(f"Mock module has no attribute {name}")
            return getattr(module, name) # Default behavior for other attributes if any

        mock_typing_module.configure_mock(**{'__getattr__': mock_getattr})


        with patch('importlib.import_module', return_value=mock_typing_module) as mock_importer:
            with pytest.raises(MissingDependencyError, match="Object 'NonExistentObject' not found in module 'typing'"):
                validate_imports(test_modules, mock_globals, scope="global")
            mock_importer.assert_called_with("typing")


    def test_optional_module_missing(self, caplog): # Use caplog to check warnings
        mock_globals = {}
        # Use a module from the actual REQUIRED_MODULES that is optional
        # Example: emergent_monitor
        test_modules = {
            "emergent_monitor": REQUIRED_MODULES["emergent_monitor"] # Get details from main config
        }

        original_logging_level = logging.getLogger().getEffectiveLevel()
        logging.getLogger().setLevel(logging.WARNING) # Ensure warnings are captured

        with patch('importlib.import_module', side_effect=ImportError("No module named emergent_monitor")):
            validate_imports(test_modules, mock_globals, scope="global")
            # Check that no error was raised
            assert "EmergentBehaviorTracker" in mock_globals # Dummy should be created
            assert "_EMERGENT_MONITOR_AVAILABLE" in mock_globals
            assert not mock_globals["_EMERGENT_MONITOR_AVAILABLE"]

            # Check for specific warning log
            assert "Optional module EmergentBehaviorTracker (emergent_monitor) not found." in caplog.text
            assert "Defining dummy." in caplog.text

        logging.getLogger().setLevel(original_logging_level) # Restore original logging level


    def test_local_scope_module_availability_check(self, caplog):
        mock_globals = {}
        # Example of a local module from REQUIRED_MODULES
        test_modules = {
             "symbolic_discovery_env": REQUIRED_MODULES["symbolic_discovery_env"]
        }

        # Mock import_module to succeed for this "local" module
        mock_sde_module = MagicMock()
        with patch('importlib.import_module', return_value=mock_sde_module) as mock_importer:
            validate_imports(test_modules, mock_globals, scope="global") # Global scope validation

            # Module should NOT be in globals
            assert "SymbolicDiscoveryEnv" not in mock_globals
            assert "symbolic_discovery_env" not in mock_globals
            # But importlib.import_module should have been called to check availability
            mock_importer.assert_called_with(REQUIRED_MODULES["symbolic_discovery_env"]["module_name"])
            assert "Checked availability of local module" in caplog.text


class TestExperimentRunnerParallel:
    @pytest.fixture
    def mock_experiment_config(self):
        # Mock JanusConfig and other dependencies if ExperimentConfig requires them
        mock_janus_cfg = MagicMock(spec=JanusConfig)
        mock_janus_cfg.num_evaluation_cycles = 1 # Keep it simple
        mock_janus_cfg.synthetic_data_params = None
        mock_janus_cfg.target_phenomena = "test_phenomena"

        return ExperimentConfig(
            name="test_exp",
            experiment_type="test_type",
            janus_config=mock_janus_cfg,
            environment_type="test_env",
            algorithm="test_algo",
            noise_level=0.0,
            n_runs=2 # Test with 2 runs
        )

    @pytest.fixture
    def runner(self):
        # Minimal ExperimentRunner for testing run_experiment_suite logic
        # Patching _discover_experiments and _register_algorithms to avoid actual plugin/algo loading
        with patch('experiment_runner.ExperimentRunner._discover_experiments', return_value=None):
            with patch('experiment_runner.ExperimentRunner._register_algorithms', return_value=None):
                runner = experiment_runner.ExperimentRunner(use_wandb=False, strict_mode=False)
                # Mock essential components if run_single_experiment is not fully mocked
                runner.experiment_plugins['test_type'] = MagicMock()
                return runner

    @patch('experiment_runner.mp.Pool')
    @patch('experiment_runner.Manager') # Patch where Manager is used (experiment_runner.Manager)
    @patch('experiment_runner.ExperimentRunner._save_result') # Mock saving results
    def test_run_experiment_suite_parallel_execution(self, mock_save_result, MockManager, MockPool, runner, mock_experiment_config, caplog):
        # Setup mocks for Pool and Queue
        mock_pool_instance = MockPool.return_value.__enter__.return_value # Allows 'with Pool(...) as pool:'
        mock_queue_instance = MockManager.return_value.Queue.return_value

        # Mock run_single_experiment to simulate work and put result on queue via callback
        mock_result = MagicMock(spec=experiment_runner.ExperimentResult)
        mock_result.config = mock_experiment_config # Attach config for _save_result
        mock_result.run_id = 0

        # This simulates the callback part of apply_async
        def side_effect_apply_async(target, args, callback, error_callback):
            # Simulate successful execution by calling the callback
            # In real scenario, this happens in worker process after target returns
            callback(mock_result) # For first call
            callback(mock_result) # For second call (n_runs=2)
            async_res = MagicMock()
            async_res.get.return_value = mock_result # Not strictly needed if callback handles all
            return async_res

        mock_pool_instance.apply_async.side_effect = side_effect_apply_async

        # Mock queue.get to return results
        # Total 2 runs for mock_experiment_config
        mock_queue_instance.get.side_effect = [mock_result, mock_result, mp.TimeoutError("Simulated timeout after all results")]


        runner.run_experiment_suite([mock_experiment_config], parallel=True, num_parallel_workers=2)

        MockPool.assert_called_once_with(processes=2)
        MockManager.assert_called_once()
        MockManager.return_value.Queue.assert_called_once()

        assert mock_pool_instance.apply_async.call_count == mock_experiment_config.n_runs
        # Check if run_single_experiment was the target
        first_call_args = mock_pool_instance.apply_async.call_args_list[0][1] # args tuple of first call
        assert first_call_args['target'] == runner.run_single_experiment

        # Results should be collected from the queue
        assert len(runner._results_to_dataframe([])) == 0 # Placeholder, check based on all_results
        # The test needs to check all_results populated inside run_experiment_suite
        # This part is tricky as all_results is local. We check by side effects like _save_result calls.
        assert mock_save_result.call_count == mock_experiment_config.n_runs

        # Check logging for parallel execution start
        assert "Running experiment suite in parallel" in caplog.text


    @patch('experiment_runner.ExperimentRunner.run_single_experiment')
    @patch('experiment_runner.ExperimentRunner._save_result')
    def test_run_experiment_suite_sequential_execution(self, mock_save_result, mock_run_single, runner, mock_experiment_config, caplog):
        mock_result = MagicMock(spec=experiment_runner.ExperimentResult)
        mock_run_single.return_value = mock_result

        runner.run_experiment_suite([mock_experiment_config], parallel=False)

        assert mock_run_single.call_count == mock_experiment_config.n_runs
        assert mock_save_result.call_count == mock_experiment_config.n_runs
        assert "Running experiment suite sequentially" in caplog.text


    @patch('experiment_runner.mp.Pool')
    @patch('experiment_runner.Manager')
    @patch('experiment_runner._mp_error_callback') # Patch the actual error callback function
    def test_run_experiment_suite_parallel_error_callback(self, mock_error_callback, MockManager, MockPool, runner, mock_experiment_config, caplog):
        mock_pool_instance = MockPool.return_value.__enter__.return_value
        MockManager.return_value.Queue.return_value # Just need the queue mock

        simulated_exception = ValueError("Simulated error in worker")

        def side_effect_apply_async_error(target, args, callback, error_callback):
            # Simulate error by calling the error_callback
            error_callback(simulated_exception) # For all calls
            async_res = MagicMock()
            # async_res.get.side_effect = simulated_exception # If get() was used to fetch results
            return async_res

        mock_pool_instance.apply_async.side_effect = side_effect_apply_async_error

        # Mock queue.get to simulate no results due to errors or timeout to finish loop
        mock_queue_instance = MockManager.return_value.Queue.return_value
        mock_queue_instance.get.side_effect = mp.TimeoutError("Simulated timeout")


        runner.run_experiment_suite([mock_experiment_config], parallel=True, num_parallel_workers=1)

        # Error callback should be triggered for each task
        assert mock_error_callback.call_count == mock_experiment_config.n_runs
        mock_error_callback.assert_any_call(simulated_exception)

        # Check logs for warnings about result mismatch if that's implemented
        # Or check that the final dataframe is empty or reflects missing results
        # For now, just ensuring error_callback is hit.
        assert "Error in multiprocessing worker" not in caplog.text # Because we mocked _mp_error_callback itself
                                                                   # If we wanted to check its logging, we'd let original run.


# Need to import mp for the TimeoutError in the test
import multiprocessing as mp
import experiment_runner # for ExperimentRunner class and ExperimentResult

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
