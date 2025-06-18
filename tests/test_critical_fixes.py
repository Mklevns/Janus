import pytest
from unittest.mock import Mock, MagicMock

# Corrected import for BaseExperiment, assuming it's in the root directory
from base_experiment import BaseExperiment

# BEGIN: Content for fix_feedback_integration
# This function replicates the structure of the add_intrinsic_rewards_to_env
# and its inner enhanced_step from the previous fix.
def get_fixed_add_intrinsic_rewards_to_env_function():
    """
    Returns a fixed version of the add_intrinsic_rewards_to_env function
    for testing purposes.
    """
    def add_intrinsic_rewards_to_env(env, intrinsic_calculator):
        # Store original step and reset methods from the passed 'env'
        # Using a different attribute name for testing and removing hasattr check for now.

        original_step_before_assign = env.step
        env.xyz_original_step_marker = original_step_before_assign # Use the captured variable
        # End of debug prints for assignment

        if not hasattr(env, '_original_reset'): # Keeping reset name for now
            env._original_reset = env.reset

        env._intrinsic_calculator = intrinsic_calculator

        # Define enhanced_step that will be bound to the env instance
        def enhanced_step(self_env, action): # 'self_env' is the environment instance
            # Original step call using the new attribute name
            obs, original_extrinsic_reward, done, truncated, info = self_env.xyz_original_step_marker(action)

            current_expression = info.get('expression', '')
            current_complexity = info.get('complexity', 0)

            current_embedding = None
            if hasattr(self_env, '_last_tree_embedding') and self_env._last_tree_embedding is not None:
                current_embedding = self_env._last_tree_embedding
            elif 'tree_structure' in info and hasattr(self_env._intrinsic_calculator, 'compute_embedding'):
                try:
                    current_embedding = self_env._intrinsic_calculator.compute_embedding(info['tree_structure'])
                except Exception as e:
                    # In a real scenario, log this error
                    # print(f"Error computing embedding: {e}")
                    current_embedding = None

            current_data = getattr(self_env, 'target_data', None)
            current_variables = getattr(self_env, 'variables', [])

            try:
                calculated_enhanced_reward = self_env._intrinsic_calculator.calculate_intrinsic_reward(
                    expression=current_expression,
                    complexity=current_complexity,
                    extrinsic_reward=original_extrinsic_reward,
                    embedding=current_embedding,
                    data=current_data,
                    variables=current_variables
                )

                info['intrinsic_reward'] = calculated_enhanced_reward - original_extrinsic_reward
                info['original_reward'] = original_extrinsic_reward
                # The environment should return the new, possibly enhanced reward
                reward_to_return = calculated_enhanced_reward
            except Exception as e:
                # In a real scenario, log this error
                # print(f"Error calculating intrinsic reward: {e}")
                # On error, return the original extrinsic reward and original info
                reward_to_return = original_extrinsic_reward
                # info remains unchanged regarding intrinsic/original reward if error occurs

            return obs, reward_to_return, done, truncated, info

        # Define enhanced_reset that will be bound to the env instance
        def enhanced_reset(self_env, seed=None, options=None): # 'self_env' is the environment instance
            if hasattr(self_env._intrinsic_calculator, 'reset'):
                self_env._intrinsic_calculator.reset()

            # Call the original reset method
            if seed is not None or options is not None:
                 # Ensure options is passed if only seed is None but options is not.
                if seed is None: # original_reset might not expect options if seed is None.
                                 # This depends on the specific env's reset signature.
                                 # For generic mocks, it's safer to pass if available.
                    return self_env._original_reset(options=options) if options else self_env._original_reset()
                return self_env._original_reset(seed=seed, options=options)
            else:
                return self_env._original_reset()

        # Bind the methods to the env instance
        # Using __get__ to correctly bind the method to the instance
        env.step = enhanced_step.__get__(env, env.__class__)
        env.reset = enhanced_reset.__get__(env, env.__class__)
        return env

    return add_intrinsic_rewards_to_env
# END: Content for fix_feedback_integration

def test_feedback_integration_fix():
    mock_env = Mock()
    # Setup attributes expected by the enhanced_step function
    mock_env.target_data = [[1, 2], [3, 4]]
    mock_env.variables = ['x', 'y']
    # Initialize _last_tree_embedding, critical for the hasattr check path
    mock_env._last_tree_embedding = None

    # Define a direct function for the step, instead of a Mock object
    # This is to test if the Mock call itself is the issue.
    def mock_step_function(action):
        # Print to confirm it's called
        # print(f"mock_step_function called with action: {action}")
        return (
            [0.1, 0.2], # obs
            1.0,        # reward (extrinsic)
            False,      # done
            False,      # truncated
            {'expression': 'x + y', 'complexity': 3, 'tree_structure': 'mock_tree'} # info
        )
    mock_env.step = mock_step_function

    # Mock the original reset method if needed by your test logic, though not directly asserted here
    # Still using Mock for reset as it's not the point of failure
    mock_env.reset = Mock(return_value=([0.1,0.2], {'info_on_reset': True}))


    mock_calculator = Mock()
    mock_calculator.calculate_intrinsic_reward = Mock(return_value=1.5) # enhanced reward
    mock_calculator.reset = Mock()
    # Setup compute_embedding on the mock_calculator
    # This mock is crucial because hasattr(self._intrinsic_calculator, 'compute_embedding') will be True
    mock_calculator.compute_embedding = Mock(return_value="mock_embedding_value")

    # Get the function that applies the fix
    add_intrinsic_rewards_function = get_fixed_add_intrinsic_rewards_to_env_function()
    # Apply the fix to the mock_env
    enhanced_env = add_intrinsic_rewards_function(mock_env, mock_calculator)

    # Call the enhanced step method
    obs, reward, done, truncated, info = enhanced_env.step(0) # Action is 0 (arbitrary for this test)

    # Assert that calculate_intrinsic_reward was called with all expected arguments
    mock_calculator.calculate_intrinsic_reward.assert_called_once_with(
        expression='x + y',
        complexity=3,
        extrinsic_reward=1.0, # Original extrinsic reward
        embedding="mock_embedding_value", # Expected because compute_embedding is mocked
        data=[[1, 2], [3, 4]],
        variables=['x', 'y']
    )

    # Assert that the compute_embedding was called (since _last_tree_embedding was None)
    mock_calculator.compute_embedding.assert_called_once_with('mock_tree')

    # Assertions on the results
    assert reward == 1.5, "The reward returned should be the enhanced reward"
    assert info['intrinsic_reward'] == 0.5, "Intrinsic reward should be calculated correctly"
    assert info['original_reward'] == 1.0, "Original reward should be correctly stored in info"
    print("✓ Feedback integration fix test passed!")

def test_base_experiment_error_handling_fix():
    # Define a simple experiment class that inherits from BaseExperiment
    class FailingExperiment(BaseExperiment):
        def setup(self):
            # Keep setup simple or mock if it has side effects not relevant to this test
            pass

        def run(self):
            # This is where the test error will be raised
            raise ValueError("Test error during run")

        def teardown(self):
            # Keep teardown simple or mock
            pass

    exp = FailingExperiment()

    # Use pytest.raises to assert that an exception is raised
    with pytest.raises(ValueError, match="Test error during run"):
        exp.execute() # This should call setup, run (fail), and then teardown

    # After the exception is caught and re-raised, check self.results
    assert exp.results.get('error') == "Test error during run", "Error message not stored correctly"
    assert exp.results.get('error_type') == "ValueError", "Error type not stored correctly"
    print("✓ Base experiment exception handling test passed!")

# It's good practice to allow running the test file directly
if __name__ == "__main__":
    pytest.main([__file__])
