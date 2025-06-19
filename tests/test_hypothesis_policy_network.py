import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock
import math # For math.ceil

# Assuming these can be imported from the project structure
from hypothesis_policy_network import PPOTrainer, HypothesisNet
from janus.ai_interpretability.environments import SymbolicDiscoveryEnv
from janus.core.grammar import ProgressiveGrammar
from janus.core.expression import Variable

@pytest.fixture(scope="module")
def dummy_grammar_and_vars():
    grammar = ProgressiveGrammar()
    variables = [Variable("x", 0, {}), Variable("y", 1, {})]
    return grammar, variables

@pytest.fixture
def mock_env(dummy_grammar_and_vars):
    grammar, variables = dummy_grammar_and_vars
    env = MagicMock(spec=SymbolicDiscoveryEnv)
    env.observation_space = MagicMock()
    env.observation_space.shape = (128,)
    env.action_space = MagicMock()
    env.action_space.n = 10
    return env

@pytest.fixture
def mock_policy(mock_env):
    policy = MagicMock(spec=HypothesisNet)
    policy.parameters.return_value = [torch.nn.Parameter(torch.randn(1))]
    policy.return_value = {'value': torch.randn(1,1)}
    return policy

@patch('hypothesis_policy_network.PPOTrainer.train_step')
@patch('hypothesis_policy_network.PPOTrainer.collect_rollouts')
def test_ppo_train_minibatching(mock_collect_rollouts, mock_train_step, mock_policy, mock_env):
    trainer = PPOTrainer(policy=mock_policy, env=mock_env)

    num_samples_in_rollout = 90
    batch_size = 32
    rollout_length = num_samples_in_rollout
    total_timesteps = rollout_length * 2
    n_epochs = 3

    mock_obs = torch.randn(num_samples_in_rollout, mock_env.observation_space.shape[0])
    for i in range(num_samples_in_rollout):
        mock_obs[i,0] = float(i)

    mock_actions = torch.randint(0, mock_env.action_space.n, (num_samples_in_rollout,))
    mock_log_probs = torch.randn(num_samples_in_rollout)
    mock_advantages = torch.randn(num_samples_in_rollout)
    mock_returns = torch.randn(num_samples_in_rollout)
    mock_action_masks = torch.ones(num_samples_in_rollout, mock_env.action_space.n).bool()
    mock_tree_structures = [None] * num_samples_in_rollout

    mock_rollout_data = {
        'observations': mock_obs, 'actions': mock_actions, 'log_probs': mock_log_probs,
        'advantages': mock_advantages, 'returns': mock_returns,
        'action_masks': mock_action_masks, 'tree_structures': mock_tree_structures
    }
    mock_collect_rollouts.return_value = mock_rollout_data

    trainer.train(total_timesteps=total_timesteps, rollout_length=rollout_length,
                  n_epochs=n_epochs, batch_size=batch_size, log_interval=1)

    num_updates = total_timesteps // rollout_length
    expected_train_step_calls_per_epoch = math.ceil(num_samples_in_rollout / batch_size)
    expected_total_train_step_calls = num_updates * n_epochs * expected_train_step_calls_per_epoch

    assert mock_train_step.call_count == expected_total_train_step_calls,         f"Expected {expected_total_train_step_calls} calls to train_step, got {mock_train_step.call_count}"

    all_batches_passed = [call[0][0] for call in mock_train_step.call_args_list]

    for update_idx in range(num_updates):
        update_batches = all_batches_passed[
            update_idx * n_epochs * expected_train_step_calls_per_epoch :
            (update_idx + 1) * n_epochs * expected_train_step_calls_per_epoch
        ]

        ordered_indices_per_epoch = [[] for _ in range(n_epochs)]

        for epoch_idx in range(n_epochs):
            epoch_batches = update_batches[
                epoch_idx * expected_train_step_calls_per_epoch :
                (epoch_idx + 1) * expected_train_step_calls_per_epoch
            ]

            batch_sizes_this_epoch = []
            current_epoch_processed_indices = set()

            for batch_idx, batch_dict in enumerate(epoch_batches):
                obs_in_batch = batch_dict['observations']
                batch_sizes_this_epoch.append(len(obs_in_batch))

                assert 'tree_structures' in batch_dict
                assert isinstance(batch_dict['tree_structures'], list)
                assert len(batch_dict['tree_structures']) == len(obs_in_batch)

                for obs_tensor in obs_in_batch:
                    original_idx = int(obs_tensor[0].item())
                    current_epoch_processed_indices.add(original_idx)
                    ordered_indices_per_epoch[epoch_idx].append(original_idx)

            for i, size in enumerate(batch_sizes_this_epoch):
                is_last_batch_in_epoch = (i == expected_train_step_calls_per_epoch - 1)
                if not is_last_batch_in_epoch:
                    assert size == batch_size, f"Update {update_idx} Epoch {epoch_idx} Batch {i}"
                else:
                    assert size <= batch_size, f"Update {update_idx} Epoch {epoch_idx} Batch {i} (last)"
                    expected_last_batch_size = num_samples_in_rollout % batch_size
                    if expected_last_batch_size == 0: expected_last_batch_size = batch_size
                    assert size == expected_last_batch_size, f"Update {update_idx} Epoch {epoch_idx} Batch {i} (last)"

            assert len(current_epoch_processed_indices) == num_samples_in_rollout,                 f"Update {update_idx} Epoch {epoch_idx}: Not all samples processed. Got {len(current_epoch_processed_indices)}, expected {num_samples_in_rollout}"

        if n_epochs > 1 and num_samples_in_rollout > 1:
            assert set(ordered_indices_per_epoch[0]) == set(ordered_indices_per_epoch[1]),                 f"Update {update_idx}: Index sets differ between epochs 0 and 1, data processing error."
            if num_samples_in_rollout > batch_size :
                 assert ordered_indices_per_epoch[0] != ordered_indices_per_epoch[1],                     f"Update {update_idx}: Order of data processing is identical across epochs 0 and 1, indicates shuffling issue."
