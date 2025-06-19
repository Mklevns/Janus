import sympy as sp
from typing import Dict
import time
import inspect
import functools

def calculate_symbolic_accuracy(expr_str: str, ground_truth_map: Dict[str, sp.Expr]) -> float:
    """
    Checks if a given string expression is symbolically equivalent to any of the
    ground truth expressions.

    Args:
        expr_str: The string representation of the discovered expression.
        ground_truth_map: A dictionary where keys are names and values are the
                          ground truth SymPy expression objects.

    Returns:
        1.0 if the expression is equivalent to a ground truth expression, 0.0 otherwise.
    """
    if not expr_str:
        return 0.0

    try:
        # Attempt to parse the discovered expression string into a SymPy object
        discovered_expr = sp.sympify(expr_str)

        # Iterate through all ground truth expressions and check for equality
        for true_expr in ground_truth_map.values():
            # .equals() checks for symbolic equivalence, which is more robust than ==
            if discovered_expr.equals(true_expr):
                return 1.0  # Found a match

    except (sp.SympifyError, SyntaxError, TypeError):
        # If parsing fails, the expression is invalid and thus not accurate
        return 0.0

    # If no match was found after checking all ground truth expressions
    return 0.0

def validate_inputs(func):
    """
    A decorator that validates the types of arguments passed to a function
    based on its type annotations.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults() # Apply default values for arguments not provided

        for name, value in bound_args.arguments.items():
            param = sig.parameters[name]
            # Check type annotation
            if param.annotation is not inspect.Parameter.empty:
                expected_type = param.annotation
                # For generic types like List[int], List, Dict, Tuple, etc.,
                # inspect.annotation gives the origin type (e.g., list, dict).
                # For basic types like int, str, it's the type itself.
                # This basic check works for main types as requested.
                actual_type = type(value)

                # Handle *args (param.kind == inspect.Parameter.VAR_POSITIONAL)
                # The annotation for *args is typically for the tuple itself, e.g., *args: tuple
                # Or for its elements e.g. *args: int (less common for direct annotation like this)
                # For now, if it's VAR_POSITIONAL, expected_type will be for the tuple.
                if param.kind == inspect.Parameter.VAR_POSITIONAL:
                    if not isinstance(value, expected_type): # `value` is the tuple of args
                         raise TypeError(
                            f"Argument '{name}' (for *args) expected to be a {expected_type.__name__}, "
                            f"but got {actual_type.__name__}."
                        )
                    # Optionally, could iterate through `value` (the tuple items) if a more specific
                    # element type annotation was available and desired, but problem statement says main type is fine.

                # Handle **kwargs (param.kind == inspect.Parameter.VAR_KEYWORD)
                # Annotation is for the dict itself, e.g., **kwargs: dict
                elif param.kind == inspect.Parameter.VAR_KEYWORD:
                    if not isinstance(value, expected_type): # `value` is the dict of kwargs
                        raise TypeError(
                            f"Argument '{name}' (for **kwargs) expected to be a {expected_type.__name__}, "
                            f"but got {actual_type.__name__}."
                        )
                    # Similar to *args, could iterate through `value.values()` if needed.

                # Handle regular parameters
                elif not isinstance(value, expected_type):
                    raise TypeError(
                        f"Argument '{name}' expected type {expected_type.__name__}, "
                        f"but got {actual_type.__name__}."
                    )
        return func(*args, **kwargs)
    return wrapper

import importlib
from typing import Optional, Any
import torch
import numpy as np

def safe_import(module_name: str, pip_install_name: Optional[str] = None, alias: Optional[str] = None) -> Optional[Any]:
    """
    Safely imports a module, providing user-friendly error messages if the import fails.

    Args:
        module_name: The name of the module to import (e.g., 'numpy').
        pip_install_name: The name of the package to install via pip (e.g., 'numpy').
                          If None or empty, a generic install message is shown.
        alias: For future use, if the module should be assigned to a global alias.
               Currently not implemented beyond being a parameter.

    Returns:
        The imported module if successful, otherwise None.
    """
    try:
        module = importlib.import_module(module_name)
        return module
    except ImportError:
        message = f"Warning: Optional dependency '{module_name}' not found."
        if pip_install_name and pip_install_name.strip():
            message += f" You can install it with `pip install {pip_install_name}`."
        else:
            message += " Please install it if you need this functionality."
        print(message)
        return None

def safe_env_reset(env, max_retries=3, retry_delay_seconds=0.5):
    """
    Safely resets the environment with retries.

    Args:
        env: The environment object to reset.
        max_retries: Maximum number of reset attempts.
        retry_delay_seconds: Delay between retries in seconds.

    Returns:
        A tuple containing the observation and info from the environment reset.

    Raises:
        RuntimeError: If the environment reset fails after multiple attempts.
    """
    last_exception = None
    for attempt in range(max_retries):
        try:
            reset_output = env.reset()
            # Unpack based on typical gym env.reset() output,
            # but also handle cases where only observation is returned.
            if isinstance(reset_output, tuple) and len(reset_output) == 2:
                observation, info = reset_output
            else:
                # If not a 2-tuple, assume it's the observation directly
                # and info is empty. This matches the subtask requirement.
                observation = reset_output
                info = {}

            if observation is not None:
                return observation, info
            else:
                # This case handles when reset_output was e.g. (None, {}) or just None
                print(f"Attempt {attempt + 1} of {max_retries}: env.reset() returned None observation.")
                # Store a generic exception if observation is None to ensure retry logic continues
                # and eventually raises RuntimeError if all attempts fail.
                last_exception = ValueError("env.reset() returned None observation")
        except Exception as e:
            last_exception = e
            print(f"Attempt {attempt + 1} of {max_retries}: env.reset() failed with exception: {e}")

        if attempt < max_retries - 1:
            print(f"Retrying in {retry_delay_seconds}s...")
            time.sleep(retry_delay_seconds)

    # If loop finishes, all retries have been exhausted
    error_message = f"Environment reset failed after {max_retries} attempts."
    if last_exception:
        error_message += f" Last issue: {last_exception}"
    raise RuntimeError(error_message)

if __name__ == '__main__':
    # Example Usage (requires a mock environment)
    class MockEnv:
        def __init__(self, succeed_after_attempts=1, return_type="tuple_obs_info"):
            # return_type can be "tuple_obs_info", "obs_only", "none_obs_tuple", "none_direct", "exception"
            self.reset_attempts = 0
            self.succeed_after_attempts = succeed_after_attempts
            self.return_type = return_type

        def reset(self):
            self.reset_attempts += 1
            print(f"MockEnv.reset() called, attempt {self.reset_attempts}")

            if self.reset_attempts < self.succeed_after_attempts:
                if self.return_type == "exception_early": # Fail early for retries
                    raise ValueError(f"Simulated reset error on attempt {self.reset_attempts}")
                print(f"Simulating recoverable issue on attempt {self.reset_attempts}")
                # Forcing a path that would lead to retry
                if self.return_type == "none_obs_tuple":
                    return None, {"info": "attempting_recovery"}
                elif self.return_type == "none_direct":
                    return None
                else: # Other recoverable issues might lead to throwing an error that gets caught
                    raise InterruptedError(f"Simulated recoverable error on attempt {self.reset_attempts}")


            # Success condition
            if self.return_type == "tuple_obs_info":
                return {"obs": "valid_observation"}, {"info": "some_info"}
            elif self.return_type == "obs_only":
                return {"obs": "valid_observation_only"}
            elif self.return_type == "none_obs_tuple": # Should have been caught by < succeed_after_attempts
                return None, {"info": "finally_valid_info_but_obs_is_none"}
            elif self.return_type == "none_direct": # Should have been caught by < succeed_after_attempts
                return None
            elif self.return_type == "exception": # Simulate final failure with exception
                 raise ValueError("Simulated final reset error")
            else:
                return {"obs": "default_success_obs"}, {"info": "default_success_info"}

    # Test case 1: Successful reset on the first attempt (obs and info tuple)
    print("\n--- Test Case 1: Success on first attempt (obs and info tuple) ---")
    mock_env_tc1 = MockEnv(succeed_after_attempts=1, return_type="tuple_obs_info")
    try:
        obs, info = safe_env_reset(mock_env_tc1)
        print(f"Reset successful: obs={obs}, info={info}")
        assert obs == {"obs": "valid_observation"}
        assert info == {"info": "some_info"}
        assert mock_env_tc1.reset_attempts == 1
    except RuntimeError as e:
        print(f"Error TC1: {e}")

    # Test case 2: Successful reset on the first attempt (obs only)
    print("\n--- Test Case 2: Success on first attempt (obs only) ---")
    mock_env_tc2 = MockEnv(succeed_after_attempts=1, return_type="obs_only")
    try:
        obs, info = safe_env_reset(mock_env_tc2)
        print(f"Reset successful: obs={obs}, info={info}")
        assert obs == {"obs": "valid_observation_only"}
        assert info == {} # Info should be an empty dict
        assert mock_env_tc2.reset_attempts == 1
    except RuntimeError as e:
        print(f"Error TC2: {e}")

    # Test case 3: Successful reset after a few retries (simulating recoverable errors)
    print("\n--- Test Case 3: Success after 2 retries (recoverable errors) ---")
    mock_env_tc3 = MockEnv(succeed_after_attempts=3, return_type="tuple_obs_info")
    # To make it retry, it will use the "InterruptedError" path for attempts < succeed_after_attempts
    try:
        obs, info = safe_env_reset(mock_env_tc3, max_retries=3, retry_delay_seconds=0.01)
        print(f"Reset successful: obs={obs}, info={info}")
        assert obs == {"obs": "default_success_obs"} # Falls to default as return_type is tuple_obs_info
        assert info == {"info": "default_success_info"}
        assert mock_env_tc3.reset_attempts == 3
    except RuntimeError as e:
        print(f"Error TC3: {e}")

    # Test case 4: Failure after all retries (returns None observation consistently)
    print("\n--- Test Case 4: Failure - all retries return None observation (as part of tuple) ---")
    mock_env_tc4 = MockEnv(succeed_after_attempts=5, return_type="none_obs_tuple") # Will always return (None, info)
    try:
        safe_env_reset(mock_env_tc4, max_retries=3, retry_delay_seconds=0.01)
        assert False, "TC4 should have raised RuntimeError"
    except RuntimeError as e:
        print(f"Caught expected error TC4: {e}")
        assert "Environment reset failed after 3 attempts" in str(e)
        assert "env.reset() returned None observation" in str(e) # Check for last_exception message
        assert mock_env_tc4.reset_attempts == 3
    except Exception as e:
        print(f"Unexpected error TC4: {type(e)} {e}")


    # Test case 5: Failure after all retries (env.reset() returns None directly consistently)
    print("\n--- Test Case 5: Failure - all retries return None directly ---")
    mock_env_tc5 = MockEnv(succeed_after_attempts=5, return_type="none_direct") # Will always return None
    try:
        safe_env_reset(mock_env_tc5, max_retries=3, retry_delay_seconds=0.01)
        assert False, "TC5 should have raised RuntimeError"
    except RuntimeError as e:
        print(f"Caught expected error TC5: {e}")
        assert "Environment reset failed after 3 attempts" in str(e)
        assert "env.reset() returned None observation" in str(e)
        assert mock_env_tc5.reset_attempts == 3
    except Exception as e:
        print(f"Unexpected error TC5: {type(e)} {e}")

    # Test case 6: Failure due to persistent exception from env.reset()
    print("\n--- Test Case 6: Failure - persistent exception from env.reset() ---")
    mock_env_tc6 = MockEnv(succeed_after_attempts=5, return_type="exception_early") # Always raises ValueError
    try:
        safe_env_reset(mock_env_tc6, max_retries=3, retry_delay_seconds=0.01)
        assert False, "TC6 should have raised RuntimeError"
    except RuntimeError as e:
        print(f"Caught expected error TC6: {e}")
        assert "Environment reset failed after 3 attempts" in str(e)
        assert "Simulated reset error on attempt 3" in str(e) # Check for last_exception message
        assert mock_env_tc6.reset_attempts == 3
    except Exception as e:
        print(f"Unexpected error TC6: {type(e)} {e}")

    # Test case 7: Successful reset where early attempts return None directly
    print("\n--- Test Case 7: Success after 2 retries (early attempts return None directly) ---")
    mock_env_tc7 = MockEnv(succeed_after_attempts=3, return_type="none_direct")
    # This setup means: attempt 1 returns None, attempt 2 returns None, attempt 3 succeeds.
    # For succeed_after_attempts=3, and return_type="none_direct":
    # Attempt 1: reset_attempts=1. 1 < 3 is true. Returns None.
    # Attempt 2: reset_attempts=2. 2 < 3 is true. Returns None.
    # Attempt 3: reset_attempts=3. 3 < 3 is false. Switches to success.
    # BUT, the success part for "none_direct" in MockEnv is also "return None".
    # This test case needs adjustment in MockEnv or the test logic.
    # Let's redefine MockEnv for this specific case or add a new return_type.
    # For simplicity, let's make succeed_after_attempts=3 and return_type="obs_only" for success.
    # The early failures will be "none_direct".

    class MockEnvCustom(MockEnv):
        def reset(self):
            self.reset_attempts += 1
            print(f"MockEnvCustom.reset() called, attempt {self.reset_attempts}")

            if self.reset_attempts < self.succeed_after_attempts:
                # Fail with None for early attempts
                print(f"Simulating None return on attempt {self.reset_attempts}")
                return None
            else:
                # Succeed on the target attempt
                print(f"Simulating success on attempt {self.reset_attempts}")
                if self.return_type == "obs_only":
                    return {"obs": "successful_observation_after_none"}
                else: # Default to tuple
                    return {"obs": "successful_observation_after_none"}, {"info": "successful_info"}

    mock_env_tc7_custom = MockEnvCustom(succeed_after_attempts=3, return_type="obs_only")
    try:
        obs, info = safe_env_reset(mock_env_tc7_custom, max_retries=3, retry_delay_seconds=0.01)
        print(f"Reset successful TC7: obs={obs}, info={info}")
        assert obs == {"obs": "successful_observation_after_none"}
        assert info == {}
        assert mock_env_tc7_custom.reset_attempts == 3
    except RuntimeError as e:
        print(f"Error TC7: {e}")


    print("\nAll local tests for math_utils.py complete (pending execution).")


def self_attention_primitive(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Implements self-attention mechanism for transformer models.
    """
    hidden_dim = query.shape[-1]
    scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(hidden_dim)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    attention_weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, value)
    return output

def position_encoding_primitive(seq_len: int, hidden_dim: int, max_len: int = 5000) -> torch.Tensor:
    """
    Generate sinusoidal position encodings.
    """
    position = torch.arange(seq_len).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * -(np.log(10000.0) / hidden_dim))
    pos_encoding = torch.zeros(seq_len, hidden_dim)
    pos_encoding[:, 0::2] = torch.sin(position * div_term)
    pos_encoding[:, 1::2] = torch.cos(position * div_term)
    return pos_encoding
