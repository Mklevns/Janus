import pytest
import torch
from unittest.mock import MagicMock

# Assuming maml-training-framework.py is in the parent directory or accessible in PYTHONPATH
# For testing purposes, we might need to adjust the import path based on the project structure.
# If maml-training-framework.py is in the root, and tests is a subdir, this should work if run from root.
import sys
from pathlib import Path
# The PYTHONPATH environment variable should handle finding maml_training_framework
# from the /app directory when tests are run from there.

from maml_training_framework import MAMLTrainer, MetaLearningConfig, MetaLearningPolicy

# Define a fixed observation dimension for testing
TEST_OBS_DIM = 10

@pytest.fixture
def mock_config():
    """Provides a mock MetaLearningConfig."""
    config = MetaLearningConfig()
    config.device = 'cpu' # Use CPU for tests
    # Add any other necessary config attributes
    return config

@pytest.fixture
def mock_policy():
    """Provides a mock MetaLearningPolicy."""
    policy = MagicMock(spec=MetaLearningPolicy)
    policy.observation_dim = TEST_OBS_DIM
    # Mock other methods/attributes of policy if _compute_task_embedding interacts with them
    # For this specific method, only observation_dim is directly used from the policy object.
    return policy

@pytest.fixture
def maml_trainer(mock_config, mock_policy):
    """Provides a MAMLTrainer instance with mock dependencies."""
    # MAMLTrainer constructor requires task_distribution, which is not used by _compute_task_embedding
    # We can pass a MagicMock for it.
    mock_task_distribution = MagicMock()
    trainer = MAMLTrainer(config=mock_config, policy=mock_policy, task_distribution=mock_task_distribution)
    return trainer

def test_compute_task_embedding_with_observations(maml_trainer):
    """Tests _compute_task_embedding with trajectories containing observations."""
    trajectories = [
        {'observations': [[1.0, 2.0]* (TEST_OBS_DIM // 2) , [3.0, 4.0]* (TEST_OBS_DIM // 2)]},
        {'observations': [[5.0, 6.0]* (TEST_OBS_DIM // 2)]}
    ]
    # Expected observations: concatenate all observations from trajectories
    expected_obs_list = [[1.0, 2.0]* (TEST_OBS_DIM // 2), [3.0, 4.0]* (TEST_OBS_DIM // 2), [5.0, 6.0]* (TEST_OBS_DIM // 2)]
    expected_tensor = torch.FloatTensor(expected_obs_list).to(maml_trainer.config.device)

    result_tensor = maml_trainer._compute_task_embedding(trajectories)

    assert result_tensor.shape == (len(expected_obs_list), TEST_OBS_DIM), "Tensor shape mismatch"
    assert torch.equal(result_tensor, expected_tensor), "Tensor content mismatch"

def test_compute_task_embedding_without_observations(maml_trainer):
    """Tests _compute_task_embedding with trajectories lacking observations."""
    trajectories_empty_obs = [
        {'observations': []},
        {'observations': []}
    ]
    trajectories_missing_obs_key = [
        {}, # Missing 'observations' key
        {'observations': []}
    ]
    trajectories_mixed = [
        {'observations': [[1.0, 2.0]* (TEST_OBS_DIM // 2)]}, # Valid
        {'observations': []}, # Empty
        {} # Missing
    ]
    expected_obs_for_mixed = [[1.0, 2.0]* (TEST_OBS_DIM // 2)]


    # Case 1: All trajectories have empty observations lists
    result_tensor_empty = maml_trainer._compute_task_embedding(trajectories_empty_obs)
    expected_zero_tensor = torch.zeros((1, TEST_OBS_DIM), device=maml_trainer.config.device)
    assert result_tensor_empty.shape == (1, TEST_OBS_DIM), "Shape mismatch for empty observations"
    assert torch.equal(result_tensor_empty, expected_zero_tensor), "Tensor not zero for empty observations"

    # Case 2: Trajectories might be missing the 'observations' key or have empty lists
    result_tensor_missing_key = maml_trainer._compute_task_embedding(trajectories_missing_obs_key)
    assert result_tensor_missing_key.shape == (1, TEST_OBS_DIM), "Shape mismatch for missing obs key"
    assert torch.equal(result_tensor_missing_key, expected_zero_tensor), "Tensor not zero for missing obs key"

    # Case 3: Mixed trajectories (some valid, some empty/missing)
    # The method should only collect valid observations
    expected_tensor_mixed = torch.FloatTensor(expected_obs_for_mixed).to(maml_trainer.config.device)
    result_tensor_mixed = maml_trainer._compute_task_embedding(trajectories_mixed)
    assert result_tensor_mixed.shape == (len(expected_obs_for_mixed), TEST_OBS_DIM), "Tensor shape mismatch for mixed trajectories"
    assert torch.equal(result_tensor_mixed, expected_tensor_mixed), "Tensor content mismatch for mixed trajectories"


def test_compute_task_embedding_fully_empty_or_invalid(maml_trainer):
    """Tests _compute_task_embedding when all trajectories are effectively empty."""
    trajectories = [
        {'observations': []}, # Empty list
        {},                  # Missing 'observations' key
        {'observations': None} # 'observations' is None (though type hint is List[Dict], good to be robust)
    ]
    expected_zero_tensor = torch.zeros((1, TEST_OBS_DIM), device=maml_trainer.config.device)

    result_tensor = maml_trainer._compute_task_embedding(trajectories)

    assert result_tensor.shape == (1, TEST_OBS_DIM), "Shape mismatch for fully empty/invalid observations"
    assert torch.equal(result_tensor, expected_zero_tensor), "Tensor not zero for fully empty/invalid observations"

# To run these tests, navigate to the root directory of the project and run:
# python -m pytest
# Ensure maml_training_framework.py is in the root or adjust sys.path accordingly.
# Also, ensure pytest and torch are installed.
# Example: pip install pytest torch unittest.mock
# (unittest.mock is standard library for Python 3.3+)
