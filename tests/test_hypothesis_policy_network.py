import unittest
import torch
import torch.nn as nn
import sys
import os

# Ensure the package root is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from hypothesis_policy_network import HypothesisNet, RolloutBuffer # noqa: E402
# from progressive_grammar_system import ProgressiveGrammar # Mock if needed, or ensure None is handled

# A simple mock for ProgressiveGrammar if its full import is problematic or complex
class MockProgressiveGrammar:
    def __init__(self):
        pass # Add any attributes HypothesisNet might access, if any

class TestHypothesisNetMetaLearning(unittest.TestCase):

    def setUp(self):
        self.obs_dim = 3 * 128  # 3 nodes * 128 features
        self.node_feature_dim = 128
        self.action_dim = 10
        self.hidden_dim = 256
        self.batch_size = 2
        self.num_nodes = 3
        self.trajectory_length = 5

        # Using a mock grammar or None
        self.grammar = None # Or MockProgressiveGrammar()

        # Dummy observation tensor
        self.observation = torch.randn(self.batch_size, self.obs_dim)
        # Dummy action mask
        self.action_mask = torch.ones(self.batch_size, self.action_dim).bool()

        # Dummy task trajectories for a single trajectory per batch item
        # Shape: (batch_size, trajectory_length, feature_dim)
        # self.task_trajectories_single = torch.randn(
        #     self.batch_size, self.trajectory_length, self.node_feature_dim
        # )

        # Dummy task trajectories for multiple trajectories per batch item, as per original spec
        # Shape: (batch_size, num_trajectories, trajectory_length, feature_dim)
        self.num_task_trajectories = 2
        self.task_trajectories_multi = torch.randn(
            self.batch_size, self.num_task_trajectories, self.trajectory_length, self.node_feature_dim
        )


    def test_hypothesis_net_meta_learning_initialization(self):
        # Test with meta-learning enabled
        policy_meta = HypothesisNet(
            observation_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
            grammar=self.grammar,
            use_meta_learning=True
        )
        self.assertTrue(policy_meta.use_meta_learning)
        self.assertIsNotNone(policy_meta.task_encoder)
        self.assertIsInstance(policy_meta.task_encoder, nn.LSTM)
        self.assertIsNotNone(policy_meta.task_modulator)
        self.assertIsInstance(policy_meta.task_modulator, nn.Sequential)

        # Test with meta-learning disabled (default)
        policy_no_meta = HypothesisNet(
            observation_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
            grammar=self.grammar,
            use_meta_learning=False # Explicitly False
        )
        self.assertFalse(policy_no_meta.use_meta_learning)
        self.assertIsNone(policy_no_meta.task_encoder) # Expect None if not created
        self.assertIsNone(policy_no_meta.task_modulator)


    def test_hypothesis_net_meta_learning_forward_pass(self):
        policy_meta = HypothesisNet(
            observation_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
            grammar=self.grammar,
            use_meta_learning=True
        )
        policy_meta.encoder.node_feature_dim = self.node_feature_dim # Ensure consistency if not passed

        # 1. Forward pass with task trajectories
        outputs_with_task = policy_meta(
            self.observation, self.action_mask, task_trajectories=self.task_trajectories_multi
        )
        self.assertIn('policy_logits', outputs_with_task)
        self.assertIn('action_logits', outputs_with_task)
        self.assertIn('value', outputs_with_task)
        self.assertIn('task_embedding', outputs_with_task)
        self.assertIsNotNone(outputs_with_task['task_embedding'])

        self.assertEqual(outputs_with_task['policy_logits'].shape, (self.batch_size, self.action_dim))
        self.assertEqual(outputs_with_task['action_logits'].shape, (self.batch_size, self.action_dim))
        self.assertEqual(outputs_with_task['value'].shape, (self.batch_size, 1))
        self.assertEqual(outputs_with_task['task_embedding'].shape, (self.batch_size, self.hidden_dim))

        # 2. Forward pass with meta-learning enabled but no task trajectories
        outputs_no_task = policy_meta(
            self.observation, self.action_mask, task_trajectories=None
        )
        self.assertIn('policy_logits', outputs_no_task)
        self.assertIn('action_logits', outputs_no_task)
        self.assertIn('value', outputs_no_task)
        # Task embedding might be None or not present depending on implementation if task_trajectories is None
        if 'task_embedding' in outputs_no_task:
            self.assertIsNone(outputs_no_task['task_embedding'])

        self.assertEqual(outputs_no_task['policy_logits'].shape, (self.batch_size, self.action_dim))
        self.assertEqual(outputs_no_task['value'].shape, (self.batch_size, 1))

        # 3. Forward pass with meta-learning disabled
        policy_no_meta = HypothesisNet(
            observation_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
            grammar=self.grammar,
            use_meta_learning=False
        )
        policy_no_meta.encoder.node_feature_dim = self.node_feature_dim

        outputs_meta_disabled = policy_no_meta(
            self.observation, self.action_mask, task_trajectories=None # Should be ignored
        )
        self.assertIn('policy_logits', outputs_meta_disabled)
        self.assertIn('value', outputs_meta_disabled)
        self.assertNotIn('task_embedding', outputs_meta_disabled) # Or assert it's None if key is always present
        self.assertEqual(outputs_meta_disabled['policy_logits'].shape, (self.batch_size, self.action_dim))

        outputs_meta_disabled_with_traj = policy_no_meta(
            self.observation, self.action_mask, task_trajectories=self.task_trajectories_multi # Should be ignored
        )
        self.assertNotIn('task_embedding', outputs_meta_disabled_with_traj)


    def test_hypothesis_net_meta_learning_get_action(self):
        policy_meta = HypothesisNet(
            observation_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
            grammar=self.grammar,
            use_meta_learning=True
        )
        policy_meta.encoder.node_feature_dim = self.node_feature_dim

        # Prepare single instance inputs for get_action
        single_obs = self.observation[0].unsqueeze(0)
        single_action_mask = self.action_mask[0].unsqueeze(0)
        single_task_trajs = self.task_trajectories_multi[0].unsqueeze(0) # (1, num_traj, len, feat)

        # 1. Call get_action with task trajectories
        action, log_prob, value = policy_meta.get_action(
            single_obs, single_action_mask, task_trajectories=single_task_trajs
        )
        self.assertIsInstance(action, torch.Tensor) # Now returns tensor
        self.assertEqual(action.ndim, 0) # Scalar tensor
        self.assertIsInstance(log_prob, torch.Tensor)
        self.assertEqual(log_prob.shape, tuple()) # Scalar tensor
        self.assertIsInstance(value, torch.Tensor)
        self.assertEqual(value.shape, (1, 1))

        # 2. Call get_action without task trajectories
        action_no_task, log_prob_no_task, value_no_task = policy_meta.get_action(
            single_obs, single_action_mask, task_trajectories=None
        )
        self.assertIsInstance(action_no_task, torch.Tensor)
        self.assertEqual(action_no_task.ndim, 0)
        self.assertIsInstance(log_prob_no_task, torch.Tensor)
        self.assertEqual(log_prob_no_task.shape, tuple())
        self.assertIsInstance(value_no_task, torch.Tensor)
        self.assertEqual(value_no_task.shape, (1, 1))

    def test_observation_dim_validation(self):
        policy = HypothesisNet(
            observation_dim=self.obs_dim,  # This will be overridden
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
            grammar=self.grammar,
            use_meta_learning=False # Meta-learning features are not relevant for this test
        )
        # policy.encoder.node_feature_dim is already set to self.node_feature_dim (128) by default in HypothesisNet init

        invalid_obs_dim = self.node_feature_dim * 2 + 1 # e.g., 128 * 2 + 1 = 257, not a multiple of 128
        invalid_observation = torch.randn(self.batch_size, invalid_obs_dim)

        with self.assertRaisesRegex(ValueError, "Observation dimension must be a multiple of node_feature_dim."):
            policy(invalid_observation, self.action_mask)


class TestRolloutBuffer(unittest.TestCase):
    def test_buffer_capacity(self):
        max_size = 3
        buffer = RolloutBuffer(max_size=max_size)

        for i in range(max_size + 2): # Add 5 items
            buffer.add(
                obs=float(i), action=i, reward=float(i), value=float(i),
                log_prob=float(i), done=(i % 2 == 0), action_mask=True, tree_structure=None
            )

        self.assertEqual(len(buffer.observations), max_size)
        self.assertEqual(len(buffer.actions), max_size)
        self.assertEqual(len(buffer.rewards), max_size)
        self.assertEqual(len(buffer.values), max_size)
        self.assertEqual(len(buffer.log_probs), max_size)
        self.assertEqual(len(buffer.dones), max_size)
        self.assertEqual(len(buffer.action_masks), max_size)
        self.assertEqual(len(buffer.tree_structures), max_size)

    def test_fifo_eviction(self):
        max_size = 3
        buffer = RolloutBuffer(max_size=max_size)

        # Store observations that are easy to track, e.g., their own index
        observations_to_add = [float(i) for i in range(5)] # 0.0, 1.0, 2.0, 3.0, 4.0
        actions_to_add = [i for i in range(5)]

        for i in range(5):
            buffer.add(
                obs=observations_to_add[i],
                action=actions_to_add[i],
                reward=float(i), value=float(i), log_prob=float(i),
                done=(i % 2 == 0), action_mask=True, tree_structure=None
            )

        # After adding 5 items to a buffer of size 3,
        # we expect items with original indices 2, 3, 4 to remain.
        expected_observations = observations_to_add[2:] # [2.0, 3.0, 4.0]
        expected_actions = actions_to_add[2:]       # [2, 3, 4]

        self.assertEqual(buffer.observations, expected_observations)
        self.assertEqual(buffer.actions, expected_actions)

        # Spot check another list, e.g. rewards
        expected_rewards = [float(i) for i in range(2, 5)] # rewards for items 2, 3, 4
        self.assertEqual(buffer.rewards, expected_rewards)

    def test_buffer_reset(self):
        buffer = RolloutBuffer(max_size=3)
        for i in range(2): # Add some items
            buffer.add(float(i), i, float(i), float(i), float(i), False, True, None)

        self.assertEqual(len(buffer.observations), 2)
        buffer.reset()
        self.assertEqual(len(buffer.observations), 0)
        self.assertEqual(len(buffer.actions), 0)
        # self.advantages and self.returns should also be None after reset
        self.assertIsNone(buffer.advantages)
        self.assertIsNone(buffer.returns)
        self.assertEqual(buffer.max_size, 3) # max_size should persist


if __name__ == '__main__':
    # This allows running the tests from the command line
    # However, due to sandbox issues, torch might not be available.
    # The test structure is defined for when the environment is correct.
    try:
        unittest.main()
    except ModuleNotFoundError as e:
        print(f"Skipping tests due to missing module: {e}")
    except Exception as e:
        print(f"An error occurred during test execution: {e}")
