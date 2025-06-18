import torch
from torch.distributions import Categorical
import pytest # Using pytest for better test organization and fixtures if needed later

# Attempt to import HypothesisNet from the corrected path
try:
    from hypothesis_policy_network import HypothesisNet
except ImportError:
    # Fallback for different project structures if the above fails
    # This might be needed if 'tests' is not in the python path correctly during execution
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from hypothesis_policy_network import HypothesisNet


class TestHypothesisNetMasking:
    def test_action_masking_in_forward_pass(self):
        # 1. Define parameters
        batch_size = 4
        action_dim = 10
        hidden_dim = 64  # Smaller for faster tests

        # HypothesisNet internally uses self.node_feature_dim = 128
        # We need to ensure observation_dim is compatible with this.
        # Let's assume max_nodes for our test observation.
        max_nodes = 3
        internal_node_feature_dim = 128 # This is hardcoded in HypothesisNet
        observation_dim = max_nodes * internal_node_feature_dim

        # 2. Instantiate HypothesisNet
        # Setting grammar=None as it's optional and simplifies testing
        policy = HypothesisNet(
            observation_dim=observation_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            grammar=None,
            encoder_type='transformer' # or 'tree_lstm' - transformer might be easier if no tree_structure is needed
        )
        policy.eval() # Set to evaluation mode

        # 3. Create a sample observation tensor
        obs = torch.randn(batch_size, observation_dim)

        # Store entropies for comparison
        entropies = {}

        # --- Scenario 1: No mask ---
        action_mask_none = None
        outputs_none = policy.forward(obs, action_mask_none)
        action_logits_none = outputs_none['action_logits']

        # Assert all original logits are preserved (none should be -1e9 unless policy inherently produces them)
        # This checks against 'policy_logits' which are pre-masking
        assert torch.allclose(action_logits_none, outputs_none['policy_logits']), \
            "Action logits should be same as policy logits when no mask is applied."

        dist_none = Categorical(logits=action_logits_none)
        entropies['none'] = dist_none.entropy().mean().item() # Using mean for scalar value

        # --- Scenario 2: Some actions masked ---
        action_mask_some = torch.ones(batch_size, action_dim, dtype=torch.bool)
        action_mask_some[:, ::2] = False # Mask every other action

        outputs_some = policy.forward(obs, action_mask_some)
        action_logits_some = outputs_some['action_logits']

        # Assert masked logits are very low
        assert torch.all(action_logits_some[~action_mask_some] < -1e8), \
            "Masked action logits should be very small."
        # Assert unmasked logits are not -1e9 (compare with original policy_logits for those positions)
        original_policy_logits = outputs_none['policy_logits'] # Using 'none' scenario's policy_logits as baseline
        assert torch.allclose(action_logits_some[action_mask_some], original_policy_logits[action_mask_some]), \
             "Unmasked action logits should match original policy_logits for those actions."


        dist_some = Categorical(logits=action_logits_some)
        # Assert probabilities for masked actions are close to zero
        assert torch.all(dist_some.probs[~action_mask_some] < 1e-6), \
            "Probabilities of masked actions should be near zero."
        entropies['some'] = dist_some.entropy().mean().item()

        # --- Scenario 3: All but one action masked ---
        action_mask_one = torch.zeros(batch_size, action_dim, dtype=torch.bool)
        # For each item in batch, unmask a different action to avoid all items having same unmasked action
        for i in range(batch_size):
            action_mask_one[i, i % action_dim] = True

        outputs_one = policy.forward(obs, action_mask_one)
        action_logits_one = outputs_one['action_logits']

        assert torch.all(action_logits_one[~action_mask_one] < -1e8), \
            "Masked action logits (all but one) should be very small."
        # Assert unmasked logits are not -1e9
        assert torch.allclose(action_logits_one[action_mask_one], original_policy_logits[action_mask_one]), \
            "The single unmasked action logit should match original policy_logits for that action."


        dist_one = Categorical(logits=action_logits_one)
        assert torch.all(dist_one.probs[~action_mask_one] < 1e-6), \
            "Probabilities of (all but one) masked actions should be near zero."
        # Prob of unmasked action should be close to 1
        # Need to select the unmasked probs carefully
        unmasked_probs = torch.zeros(batch_size)
        for i in range(batch_size):
            unmasked_probs[i] = dist_one.probs[i, i % action_dim]
        assert torch.all(unmasked_probs > 0.99), \
            "Probability of the single unmasked action should be near one."
        entropies['one'] = dist_one.entropy().mean().item()

        # --- Scenario 4: All actions masked (edge case) ---
        # This tests if the policy handles this gracefully (e.g., uniform distribution over very small numbers, or specific error)
        # PPO typically expects at least one valid action. If not, Categorical might fail or produce NaNs.
        # Let's see how the current implementation handles it.
        action_mask_all_false = torch.zeros(batch_size, action_dim, dtype=torch.bool)

        outputs_all_false = policy.forward(obs, action_mask_all_false)
        action_logits_all_false = outputs_all_false['action_logits']

        assert torch.all(action_logits_all_false < -1e8), \
            "All action logits should be very small when all actions are masked."

        dist_all_false = Categorical(logits=action_logits_all_false)
        # Probabilities should be uniform and small
        assert torch.allclose(dist_all_false.probs, torch.ones_like(dist_all_false.probs) / action_dim, atol=1e-5), \
            "Probabilities should be uniform when all actions are masked."
        entropies['all_false'] = dist_all_false.entropy().mean().item()

        # 6. Compare entropies
        # Entropy should generally decrease as more actions are masked (less uncertainty)
        # H(all_false) might be high because it's uniform. H(one) should be lowest.
        assert entropies['one'] <= entropies['some'] + 1e-6, \
            f"Entropy with one unmasked ({entropies['one']}) should be <= entropy with some unmasked ({entropies['some']})"
        assert entropies['some'] <= entropies['none'] + 1e-6, \
            f"Entropy with some unmasked ({entropies['some']}) should be <= entropy with no mask ({entropies['none']})"

        # Entropy when all actions are masked (uniform distribution) should be log(action_dim)
        # This should be higher than when only one action is available (entropy near 0)
        # and potentially higher than 'some' if 'some' still leaves a few choices.
        # It should be comparable to 'none' if 'none' also results in a somewhat uniform distribution.
        expected_uniform_entropy = torch.log(torch.tensor(action_dim, dtype=torch.float)).item()
        assert abs(entropies['all_false'] - expected_uniform_entropy) < 1e-5, \
            f"Entropy with all actions masked ({entropies['all_false']}) should be close to log(action_dim) ({expected_uniform_entropy})."

        # Specific check: H(one) should be very close to 0
        assert entropies['one'] < 1e-5, f"Entropy for 'all but one' masked should be close to 0, got {entropies['one']}"

        print("TestHypothesisNetMasking.test_action_masking_in_forward_pass completed successfully.")
        print(f"Entropies: {entropies}")

# Example of how to run this test using pytest from the command line:
# Ensure you are in the root directory of the project.
# `pytest tests/test_hypothesis_policy_network.py`
#
# Or, to run it as a script if pytest is not set up (less ideal for real projects):
if __name__ == "__main__":
    # This is a simplified run, pytest is preferred
    test_runner = TestHypothesisNetMasking()
    test_runner.test_action_masking_in_forward_pass()
    print("Test executed directly. Consider using pytest for better test management.")
