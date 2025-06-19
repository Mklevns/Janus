import unittest
import numpy as np
import sympy as sp
from conservation_reward_fix import ConservationBiasedReward # Assuming this is the correct path

class TestConservationBiasedReward(unittest.TestCase):

    def test_calculate_violation_scalar(self):
        cbr = ConservationBiasedReward(conservation_types=[], weight_factor=1.0) # Dummy instance
        self.assertEqual(cbr._calculate_violation(10.0, 10.0, 'scalar', 0.01, 0.01), 0.0)
        self.assertGreater(cbr._calculate_violation(10.0, 5.0, 'scalar', 0.01, 0.01), 0.0)
        self.assertEqual(cbr._calculate_violation(10.0, 0.0, 'scalar', 0.01, 0.01), 1.0) # Relative to zero
        self.assertEqual(cbr._calculate_violation(0.0, 10.0, 'scalar', 0.01, 0.01), 1.0) # Predicted zero, GT non-zero
        self.assertEqual(cbr._calculate_violation(0.0, 0.0, 'scalar', 0.01, 0.01), 0.0)

        self.assertEqual(cbr._calculate_violation(None, 5.0, 'scalar', 0.01, 0.01), 1.0)
        self.assertEqual(cbr._calculate_violation(5.0, None, 'scalar', 0.01, 0.01), 1.0)
        self.assertEqual(cbr._calculate_violation(None, None, 'scalar', 0.01, 0.01), 1.0)

        # Test tolerance
        self.assertEqual(cbr._calculate_violation(10.0, 10.05, 'scalar', abs_tol=0.1, rel_tol=0.001), 0.0) # Within abs_tol
        self.assertEqual(cbr._calculate_violation(10.0, 10.005, 'scalar', abs_tol=0.001, rel_tol=0.1), 0.0) # Within rel_tol
        self.assertGreater(cbr._calculate_violation(10.0, 10.1, 'scalar', abs_tol=0.01, rel_tol=0.001), 0.0) # Exceeds both


    def test_calculate_violation_vector(self):
        cbr = ConservationBiasedReward(conservation_types=[], weight_factor=1.0)
        vec1 = np.array([1.0, 2.0, 3.0])
        vec2 = np.array([1.0, 2.0, 3.0])
        vec3 = np.array([1.0, 2.5, 3.0])
        vec_zero = np.array([0.0, 0.0, 0.0])

        self.assertEqual(cbr._calculate_violation(vec1, vec2, 'vector', 0.01, 0.01), 0.0)
        self.assertGreater(cbr._calculate_violation(vec1, vec3, 'vector', 0.01, 0.01), 0.0)
        self.assertEqual(cbr._calculate_violation(vec1, vec_zero, 'vector', 0.01, 0.01), 1.0)
        self.assertEqual(cbr._calculate_violation(vec_zero, vec1, 'vector', 0.01, 0.01), 1.0)
        self.assertEqual(cbr._calculate_violation(vec_zero, vec_zero.copy(), 'vector', 0.01, 0.01), 0.0)

        # Shape mismatch
        vec4 = np.array([1.0, 2.0])
        self.assertEqual(cbr._calculate_violation(vec1, vec4, 'vector', 0.01, 0.01), 1.0)
        self.assertEqual(cbr._calculate_violation(vec4, vec1, 'vector', 0.01, 0.01), 1.0)

        # With None
        self.assertEqual(cbr._calculate_violation(None, vec1, 'vector', 0.01, 0.01), 1.0)
        self.assertEqual(cbr._calculate_violation(vec1, None, 'vector', 0.01, 0.01), 1.0)
        self.assertEqual(cbr._calculate_violation(None, None, 'vector', 0.01, 0.01), 1.0)

        # Test tolerance
        vec_tol1 = np.array([1.0, 2.0, 3.0])
        vec_tol2 = np.array([1.05, 2.0, 3.0]) # 1.05 is 5% diff from 1.0
        self.assertEqual(cbr._calculate_violation(vec_tol1, vec_tol2, 'vector', abs_tol=0.1, rel_tol=0.01), 0.0) # Within abs_tol for first element
        self.assertEqual(cbr._calculate_violation(vec_tol1, vec_tol2, 'vector', abs_tol=0.01, rel_tol=0.1), 0.0) # Within rel_tol for first element

        vec_tol3 = np.array([1.1, 2.0, 3.0]) # 1.1 is 10% diff from 1.0
        self.assertGreater(cbr._calculate_violation(vec_tol1, vec_tol3, 'vector', abs_tol=0.05, rel_tol=0.05), 0.0)


    def test_calculate_violation_tensor(self):
        cbr = ConservationBiasedReward(conservation_types=[], weight_factor=1.0)
        tensor1 = np.array([[1.0, 2.0], [3.0, 4.0]])
        tensor2 = np.array([[1.0, 2.0], [3.0, 4.0]])
        tensor3 = np.array([[1.0, 2.5], [3.0, 4.0]])
        tensor_zero = np.zeros_like(tensor1)

        self.assertEqual(cbr._calculate_violation(tensor1, tensor2, 'tensor', 0.01, 0.01), 0.0)
        self.assertGreater(cbr._calculate_violation(tensor1, tensor3, 'tensor', 0.01, 0.01), 0.0)
        self.assertEqual(cbr._calculate_violation(tensor1, tensor_zero, 'tensor', 0.01, 0.01), 1.0)
        self.assertEqual(cbr._calculate_violation(tensor_zero, tensor1, 'tensor', 0.01, 0.01), 1.0)
        self.assertEqual(cbr._calculate_violation(tensor_zero, tensor_zero.copy(), 'tensor', 0.01, 0.01), 0.0)

        # Shape mismatch
        tensor4 = np.array([[1.0, 2.0]])
        self.assertEqual(cbr._calculate_violation(tensor1, tensor4, 'tensor', 0.01, 0.01), 1.0)
        self.assertEqual(cbr._calculate_violation(tensor4, tensor1, 'tensor', 0.01, 0.01), 1.0)

        # With None
        self.assertEqual(cbr._calculate_violation(None, tensor1, 'tensor', 0.01, 0.01), 1.0)
        self.assertEqual(cbr._calculate_violation(tensor1, None, 'tensor', 0.01, 0.01), 1.0)
        self.assertEqual(cbr._calculate_violation(None, None, 'tensor', 0.01, 0.01), 1.0)

    def test_compute_conservation_bonus(self):
        cbr = ConservationBiasedReward(
            conservation_types=['energy', 'momentum'],
            weight_factor=0.5,
            abs_tolerance=0.01,
            rel_tolerance=0.01
        )

        # Scenario 1: Perfect conservation
        predicted_traj1 = {
            'conserved_energy': np.array([10.0, 10.0, 10.0]),
            'conserved_momentum': np.array([5.0, 5.0, 5.0])
        }
        ground_truth_traj1 = {
            'conserved_energy': np.array([10.0, 10.0, 10.0]),
            'conserved_momentum': np.array([5.0, 5.0, 5.0])
        }
        bonus1 = cbr.compute_conservation_bonus(predicted_traj1, ground_truth_traj1, {})
        self.assertEqual(bonus1, 0.5 * 1.0) # weight_factor * max_bonus_per_type * num_types / num_types

        # Scenario 2: Some violations
        predicted_traj2 = {
            'conserved_energy': np.array([10.0, 10.5, 10.0]), # Slight violation
            'conserved_momentum': np.array([5.0, 7.0, 5.0])   # Larger violation
        }
        ground_truth_traj2 = {
            'conserved_energy': np.array([10.0, 10.0, 10.0]),
            'conserved_momentum': np.array([5.0, 5.0, 5.0])
        }
        # energy violation: (0 + 0.05 + 0) / 3 = 0.0166 -> score = 1 - 0.0166/0.5 approx (using default deviation_scale_factor)
        # momentum violation: (0 + 0.4 + 0) / 3 = 0.1333 -> score = 1 - 0.1333/0.5 approx
        # Actual calculation within _calculate_violation is max relative diff
        # For energy: max(|10.5-10|/10) = 0.05. Score = 1 - (0.05 / (10*0.01 + 0.01)) if not for scale factor.
        # Simplified: just check it's less than perfect and greater than zero if not total violation
        bonus2 = cbr.compute_conservation_bonus(predicted_traj2, ground_truth_traj2, {})
        self.assertLess(bonus2, 0.5 * 1.0)
        self.assertGreater(bonus2, 0.0)

        # Expected violation for energy: (10.5-10)/10 = 0.05. Score_energy = 1 - min(1, 0.05 / cbr.violation_scale_factor)
        # Expected violation for momentum: (7-5)/5 = 0.4. Score_momentum = 1 - min(1, 0.4 / cbr.violation_scale_factor)
        # bonus = 0.5 * (score_energy + score_momentum) / 2
        score_energy = 1.0 - min(1.0, abs(10.5 - 10.0) / (cbr.rel_tolerance * 10.0 + cbr.abs_tolerance) / cbr.violation_scale_factor if (cbr.rel_tolerance * 10.0 + cbr.abs_tolerance) > 1e-9 else (abs(10.5-10.0) / cbr.violation_scale_factor if cbr.violation_scale_factor > 1e-9 else 1.0 if abs(10.5-10.0)>1e-9 else 0.0) )
        score_momentum = 1.0 - min(1.0, abs(7.0 - 5.0) / (cbr.rel_tolerance * 5.0 + cbr.abs_tolerance) / cbr.violation_scale_factor if (cbr.rel_tolerance * 5.0 + cbr.abs_tolerance) > 1e-9 else (abs(7.0-5.0) / cbr.violation_scale_factor if cbr.violation_scale_factor > 1e-9 else 1.0 if abs(7.0-5.0)>1e-9 else 0.0) )
        expected_bonus2 = cbr.weight_factor * (score_energy + score_momentum) / 2
        self.assertAlmostEqual(bonus2, expected_bonus2)


        # Scenario 3: Missing data for one conservation type
        predicted_traj3 = {
            'conserved_energy': np.array([10.0, 10.0, 10.0]),
            # 'conserved_momentum': missing
        }
        ground_truth_traj3 = {
            'conserved_energy': np.array([10.0, 10.0, 10.0]),
            'conserved_momentum': np.array([5.0, 5.0, 5.0]) # GT has it, but predicted doesn't
        }
        bonus3 = cbr.compute_conservation_bonus(predicted_traj3, ground_truth_traj3, {})
        # Should only consider energy. Energy is perfect. Momentum violation is 1.0 (score 0).
        # So, (1.0 + 0.0) / 2 * weight_factor
        self.assertEqual(bonus3, 0.5 * (1.0 + 0.0) / 2)


        predicted_traj3_alt = {
            'conserved_energy': np.array([10.0, 10.0, 10.0]),
            'conserved_momentum': None # Predicted is None
        }
        bonus3_alt = cbr.compute_conservation_bonus(predicted_traj3_alt, ground_truth_traj1, {}) # ground_truth_traj1 is complete
        self.assertEqual(bonus3_alt, 0.5 * (1.0 + 0.0) / 2) # Energy perfect, momentum violation


        # Scenario 4: All data missing
        predicted_traj4 = {}
        ground_truth_traj4 = {
            'conserved_energy': np.array([10.0, 10.0, 10.0]),
            'conserved_momentum': np.array([5.0, 5.0, 5.0])
        }
        bonus4 = cbr.compute_conservation_bonus(predicted_traj4, ground_truth_traj4, {})
        self.assertEqual(bonus4, 0.0)

        predicted_traj4_alt = {
            'conserved_energy': None,
            'conserved_momentum': None
        }
        bonus4_alt = cbr.compute_conservation_bonus(predicted_traj4_alt, ground_truth_traj1, {})
        self.assertEqual(bonus4_alt, 0.0)

        # Test with hypothesis_params (should not affect current basic implementation)
        bonus_with_params = cbr.compute_conservation_bonus(predicted_traj1, ground_truth_traj1, {'some_param': 1})
        self.assertEqual(bonus_with_params, 0.5 * 1.0)

if __name__ == '__main__':
    unittest.main()
