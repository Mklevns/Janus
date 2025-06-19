# symmetry_detection_fix.py
import numpy as np
import sympy
from typing import Callable, List, Dict, Any, Tuple, Optional

class PhysicsSymmetryDetector:
    def __init__(self, tolerance: float = 1e-5, confidence_threshold: float = 0.9):
        self.tolerance = tolerance
        self.confidence_threshold = confidence_threshold
        self.supported_symmetries = {
            'velocity_parity': self._check_velocity_parity,
            'time_reversal': self._check_time_reversal, # CPT inspired
            'scaling': self._check_scaling_symmetry,
            'translation': self._check_translational_symmetry,
            'rotation': self._check_rotational_symmetry,
        }

    def _evaluate_law_at_points(self, law_function: Callable, points: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        # Helper to evaluate the law_function at multiple points.
        results = []
        for point in points:
            try:
                res = law_function(point, params)
                results.append(res)
            except Exception as e:
                results.append(np.nan)
        return np.array(results)

    def _generate_test_points(self, trajectory: np.ndarray, num_samples: int = 100) -> np.ndarray:
        # Generate diverse test points from a trajectory or its feature space.
        if trajectory.shape[0] < num_samples:
            return trajectory

        indices = np.random.choice(trajectory.shape[0], size=num_samples, replace=False)
        return trajectory[indices]

    def _check_velocity_parity(self, law_function: Callable, trajectory: np.ndarray, params: Dict[str, Any]) -> float:
        # Checks f(x, v) = f(x, -v). Assumes state vector [pos, vel, ...].
        test_points = self._generate_test_points(trajectory)
        if test_points.size == 0: return 0.0
        num_dims = test_points.shape[1] // 2

        points_v = np.copy(test_points)
        points_neg_v = np.copy(test_points)
        points_neg_v[:, num_dims:] *= -1

        eval_v = self._evaluate_law_at_points(law_function, points_v, params)
        eval_neg_v = self._evaluate_law_at_points(law_function, points_neg_v, params)

        if np.isnan(eval_v).any() or np.isnan(eval_neg_v).any(): return 0.0

        diff = np.abs(eval_v - eval_neg_v)
        satisfied_points = np.sum(diff < self.tolerance)
        return satisfied_points / len(test_points)

    def _check_time_reversal(self, law_function: Callable, trajectory: np.ndarray, params: Dict[str, Any]) -> float:
        # Simplified CPT-like symmetry check.
        return self._check_velocity_parity(law_function, trajectory, params)

    def _check_scaling_symmetry(self, law_function: Callable, trajectory: np.ndarray, params: Dict[str, Any], scale_factor: float = 2.0) -> float:
        # Simplified check for scaling consistency.
        test_points = self._generate_test_points(trajectory)
        if test_points.size == 0: return 0.0
        points_scaled = np.copy(test_points) * scale_factor
        eval_original = self._evaluate_law_at_points(law_function, test_points, params)
        eval_scaled = self._evaluate_law_at_points(law_function, points_scaled, params)

        if np.isnan(eval_original).any() or np.isnan(eval_scaled).any(): return 0.0

        non_zero_mask = np.abs(eval_original) > self.tolerance
        if not np.any(non_zero_mask): return 0.0

        ratios = eval_scaled[non_zero_mask] / eval_original[non_zero_mask]
        if ratios.size == 0: return 0.0

        consistency_score = 1.0 - np.std(ratios) / (np.abs(np.mean(ratios)) + 1e-9)
        return max(0, consistency_score) if consistency_score > self.confidence_threshold * 0.5 else 0.0

    def _check_translational_symmetry(self, law_function: Callable, trajectory: np.ndarray, params: Dict[str, Any], shift_amount: Optional[np.ndarray] = None) -> float:
        # Checks f(x + dx, v) = f(x, v). Assumes positions are first components.
        test_points = self._generate_test_points(trajectory)
        if test_points.size == 0: return 0.0

        num_pos_dims = test_points.shape[1] // 2
        if shift_amount is None:
            shift_amount = np.random.rand(num_pos_dims) * 0.1

        points_original = np.copy(test_points)
        points_shifted = np.copy(test_points)
        points_shifted[:, :num_pos_dims] += shift_amount

        eval_original = self._evaluate_law_at_points(law_function, points_original, params)
        eval_shifted = self._evaluate_law_at_points(law_function, points_shifted, params)

        if np.isnan(eval_original).any() or np.isnan(eval_shifted).any(): return 0.0

        diff = np.abs(eval_original - eval_shifted)
        satisfied_points = np.sum(diff < self.tolerance)
        return satisfied_points / len(test_points)

    def _check_rotational_symmetry(self, law_function: Callable, trajectory: np.ndarray, params: Dict[str, Any]) -> float:
        # Checks f(R*x, R*v) = f(x, v) for a rotation matrix R.
        test_points = self._generate_test_points(trajectory)
        if test_points.size == 0: return 0.0
        if test_points.shape[1] < 4: return 0.0

        is_2d = test_points.shape[1] == 4 or (test_points.shape[1] > 4 and test_points.shape[1] % 2 == 0 and test_points.shape[1] < 6)
        is_3d = test_points.shape[1] == 6 or (test_points.shape[1] > 6 and test_points.shape[1] % 3 == 0)
        angle = np.random.uniform(0, np.pi / 2)

        points_original = np.copy(test_points)
        points_rotated = np.copy(test_points)

        if is_2d:
            rot_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                   [np.sin(angle), np.cos(angle)]])
            points_rotated[:, 0:2] = points_rotated[:, 0:2] @ rot_matrix.T
            points_rotated[:, 2:4] = points_rotated[:, 2:4] @ rot_matrix.T
        elif is_3d:
            rz_matrix = np.array([[np.cos(angle), -np.sin(angle), 0],
                                 [np.sin(angle), np.cos(angle), 0],
                                 [0,0,1]])
            points_rotated[:, 0:3] = points_rotated[:, 0:3] @ rz_matrix.T
            points_rotated[:, 3:6] = points_rotated[:, 3:6] @ rz_matrix.T
        else:
            return 0.0

        eval_original = self._evaluate_law_at_points(law_function, points_original, params)
        eval_rotated = self._evaluate_law_at_points(law_function, points_rotated, params)

        if np.isnan(eval_original).any() or np.isnan(eval_rotated).any(): return 0.0

        diff = np.abs(eval_original - eval_rotated)
        satisfied_points = np.sum(diff < self.tolerance)
        return satisfied_points / len(test_points)

    def detect_all_symmetries(self, law_function: Callable, trajectory: np.ndarray, params: Dict[str, Any]) -> Dict[str, bool]:
        results = {}
        for name, checker_func in self.supported_symmetries.items():
            try:
                score = checker_func(law_function, trajectory, params)
                results[name] = score >= self.confidence_threshold
            except Exception as e:
                results[name] = False
        return results

    def symmetry_guided_score(self, law_function: Callable, trajectory: np.ndarray, params: Dict[str, Any], expected_symmetries: Optional[List[str]] = None) -> float:
        if expected_symmetries is None:
            expected_symmetries = ['velocity_parity', 'time_reversal']

        total_score = 0.0
        num_expected = 0
        detected_symmetries = self.detect_all_symmetries(law_function, trajectory, params)

        for sym_name in expected_symmetries:
            if sym_name in detected_symmetries:
                total_score += 1.0 if detected_symmetries[sym_name] else 0.0
                num_expected +=1

        if num_expected == 0: return 0.0
        return total_score / num_expected

if __name__ == '__main__':
    def kinetic_energy_law(state_vector: np.ndarray, params: Dict[str, Any]) -> float:
        if len(state_vector) < 4: return np.nan
        vx, vy = state_vector[2], state_vector[3]
        m = params.get('m', 1.0)
        return 0.5 * m * (vx**2 + vy**2)

    def potential_energy_law(state_vector: np.ndarray, params: Dict[str, Any]) -> float:
        if len(state_vector) < 2: return np.nan
        y = state_vector[1]
        m = params.get('m', 1.0)
        g = params.get('g', 9.8)
        return m * g * y

    def simple_force_law(state_vector: np.ndarray, params: Dict[str, Any]) -> float:
        x = state_vector[0]
        k = params.get('k', 1.0)
        return -k * x

    detector = PhysicsSymmetryDetector(tolerance=1e-4, confidence_threshold=0.9)
    mock_trajectory = np.random.rand(200, 4) * 2 - 1
    mock_trajectory[:, 2:] *= 5

    params_ke = {'m': 2.0}
    params_pe = {'m': 2.0, 'g': 9.8}
    params_force = {'k': 1.5}

    print("--- Testing Kinetic Energy Law (0.5*m*v^2) ---")
    symmetries_ke = detector.detect_all_symmetries(kinetic_energy_law, mock_trajectory, params_ke)
    print(f"Detected Symmetries for KE: {symmetries_ke}")
    score_ke = detector.symmetry_guided_score(kinetic_energy_law, mock_trajectory, params_ke, expected_symmetries=['velocity_parity', 'time_reversal', 'rotational'])
    print(f"Symmetry Guided Score for KE: {score_ke}")

    print("\n--- Testing Potential Energy Law (m*g*y) ---")
    symmetries_pe = detector.detect_all_symmetries(potential_energy_law, mock_trajectory, params_pe)
    print(f"Detected Symmetries for PE: {symmetries_pe}")
    score_pe = detector.symmetry_guided_score(potential_energy_law, mock_trajectory, params_pe, expected_symmetries=['velocity_parity', 'time_reversal', 'translation'])
    print(f"Symmetry Guided Score for PE: {score_pe}")

    print("\n--- Testing Simple Force Law (-k*x) ---")
    symmetries_force = detector.detect_all_symmetries(simple_force_law, mock_trajectory, params_force)
    print(f"Detected Symmetries for Force Law: {symmetries_force}")
    score_force = detector.symmetry_guided_score(simple_force_law, mock_trajectory, params_force, expected_symmetries=['velocity_parity'])
    print(f"Symmetry Guided Score for Force Law: {score_force}")

    print("\n--- Specific Symmetry Checks for KE ---")
    print(f"Velocity Parity (KE): {detector._check_velocity_parity(kinetic_energy_law, mock_trajectory, params_ke)}")
    print(f"Translational Symmetry (KE): {detector._check_translational_symmetry(kinetic_energy_law, mock_trajectory, params_ke)}")
    print(f"Rotational Symmetry (KE, 2D): {detector._check_rotational_symmetry(kinetic_energy_law, mock_trajectory, params_ke)}")

    print("\n--- Specific Symmetry Checks for PE (mgy) ---")
    print(f"Velocity Parity (PE): {detector._check_velocity_parity(potential_energy_law, mock_trajectory, params_pe)}")
    print(f"Translational Symmetry (PE): {detector._check_translational_symmetry(potential_energy_law, mock_trajectory, params_pe)}")
    print(f"Rotational Symmetry (PE, 2D): {detector._check_rotational_symmetry(potential_energy_law, mock_trajectory, params_pe)}")
    print(f"Scaling Symmetry (KE): {detector._check_scaling_symmetry(kinetic_energy_law, mock_trajectory, params_ke, scale_factor=2.0)}")
