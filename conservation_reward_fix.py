# conservation_reward_fix.py
import numpy as np
from typing import List, Dict, Any

class ConservationBiasedReward:
    def __init__(self, conservation_types: List[str], weight_factor: float):
        self.conservation_types = conservation_types
        self.weight_factor = weight_factor
        self.tolerances = {'energy': 1e-3, 'momentum': 1e-4, 'mass': 1e-5, 'angular_momentum': 1e-4}
        self.history = {} # For advanced diagnostics or adaptive weighting

    def _calculate_violation(self, predicted_val: np.ndarray, ground_truth_val: np.ndarray, c_type: str) -> float:
        if predicted_val is None or ground_truth_val is None:
            return 1.0 # Max violation if data is missing

        predicted_val = np.asarray(predicted_val)
        ground_truth_val = np.asarray(ground_truth_val)

        if predicted_val.shape != ground_truth_val.shape:
            return 1.0

        if predicted_val.ndim == 0:
             violation = np.abs(predicted_val - ground_truth_val)
        elif predicted_val.ndim == 1:
             violation = np.linalg.norm(predicted_val - ground_truth_val) / (np.linalg.norm(ground_truth_val) + 1e-9)
        elif predicted_val.ndim > 1:
            diff_norms = np.linalg.norm(predicted_val - ground_truth_val, axis=tuple(range(1, predicted_val.ndim)))
            gt_norms = np.linalg.norm(ground_truth_val, axis=tuple(range(1, ground_truth_val.ndim)))
            relative_diff = diff_norms / (gt_norms + 1e-9)
            violation = np.mean(relative_diff)
        else:
            return 1.0

        return float(violation)

    def compute_conservation_bonus(self, predicted_traj: Dict[str, Any], ground_truth_traj: Dict[str, Any], hypothesis_params: Dict[str, Any]) -> float:
        total_bonus = 0.0
        num_laws_evaluated = 0

        for c_type in self.conservation_types:
            pred_val = predicted_traj.get(f'conserved_{c_type}')
            gt_val = ground_truth_traj.get(f'conserved_{c_type}')

            if pred_val is None or gt_val is None:
                continue

            violation = self._calculate_violation(pred_val, gt_val, c_type)
            tolerance = self.tolerances.get(c_type, 1e-3)

            bonus = np.exp(-violation / tolerance)
            total_bonus += bonus
            num_laws_evaluated += 1

            if c_type not in self.history: self.history[c_type] = []
            self.history[c_type].append({'violation': violation, 'bonus': bonus})

        if num_laws_evaluated == 0:
            return 0.0

        average_bonus = total_bonus / num_laws_evaluated
        return self.weight_factor * average_bonus

    def diagnose_conservation_violations(self) -> Dict[str, Dict[str, float]]:
        diagnostics = {}
        for c_type, records in self.history.items():
            if records:
                avg_violation = np.mean([r['violation'] for r in records])
                avg_bonus = np.mean([r['bonus'] for r in records])
                diagnostics[c_type] = {'average_violation': avg_violation, 'average_bonus': avg_bonus, 'num_evals': len(records)}
        return diagnostics

    def get_entropy_production_penalty(self, predicted_forward_traj, predicted_backward_traj, hypothesis_params) -> float:
        if predicted_forward_traj is None or predicted_backward_traj is None:
            return 0.0

        energy_fwd = predicted_forward_traj.get('conserved_energy')
        energy_bwd = predicted_backward_traj.get('conserved_energy_reversed_path')

        if energy_fwd is not None and energy_bwd is not None:
            fwd_change = np.abs(energy_fwd[-1] - energy_fwd[0]) if len(energy_fwd) > 1 else 0
            bwd_change = np.abs(energy_bwd[-1] - energy_bwd[0]) if len(energy_bwd) > 1 else 0

            dissipation_metric = np.abs(fwd_change - bwd_change) / ( (fwd_change + bwd_change)/2 + 1e-9)

            penalty = -np.log(1 + dissipation_metric)
            return penalty * self.weight_factor * 0.1
        return 0.0

if __name__ == '__main__':
    reward_system = ConservationBiasedReward(
        conservation_types=['energy', 'momentum'],
        weight_factor=0.5
    )
    predicted_trajectory = {
        'conserved_energy': np.array([10.0, 10.0, 10.0]),
        'conserved_momentum': np.array([5.0, 5.1, 4.9]),
    }
    ground_truth_trajectory = {
        'conserved_energy': np.array([10.0, 10.0, 10.0]),
        'conserved_momentum': np.array([5.0, 5.0, 5.0]),
    }
    hypothesis_parameters = {'mass': 1.0, 'k_spring': 2.0}
    bonus = reward_system.compute_conservation_bonus(
        predicted_trajectory, ground_truth_trajectory, hypothesis_parameters
    )
    print(f"Conservation Bonus: {bonus}")
    diagnostics = reward_system.diagnose_conservation_violations()
    print(f"Diagnostics: {diagnostics}")
    predicted_scalar = { 'conserved_energy': 10.0, 'conserved_momentum': 5.05 }
    ground_truth_scalar = { 'conserved_energy': 10.0, 'conserved_momentum': 5.0 }
    scalar_bonus = reward_system.compute_conservation_bonus(
        predicted_scalar, ground_truth_scalar, hypothesis_parameters
    )
    print(f"Scalar Conservation Bonus: {scalar_bonus}")
    predicted_multi_momentum = {
        'conserved_momentum': np.array([[[1,0.5],[0.2,0.1],[0,0]], [[1,0.55],[0.21,0.1],[0,0]]]),
    }
    gt_multi_momentum = {
        'conserved_momentum': np.array([[[1,0.5],[0.2,0.1],[0,0]], [[1,0.5],[0.2,0.1],[0,0]]]),
    }
    multi_momentum_bonus = reward_system.compute_conservation_bonus(
        {'conserved_momentum': predicted_multi_momentum['conserved_momentum']},
        {'conserved_momentum': gt_multi_momentum['conserved_momentum']},
        hypothesis_parameters
    )
    print(f"Multi-dim Momentum Conservation Bonus: {multi_momentum_bonus}")
    pred_fwd = {'conserved_energy': np.array([10.0, 9.9, 9.8])}
    pred_bwd_reversed = {'conserved_energy_reversed_path': np.array([9.8, 9.9, 10.0])}
    entropy_penalty = reward_system.get_entropy_production_penalty(pred_fwd, pred_bwd_reversed, {})
    print(f"Entropy Production Penalty (ideal reversal): {entropy_penalty}")
    pred_bwd_imperfect = {'conserved_energy_reversed_path': np.array([9.7, 9.8, 9.9])}
    entropy_penalty_imperfect = reward_system.get_entropy_production_penalty(pred_fwd, pred_bwd_imperfect, {})
    print(f"Entropy Production Penalty (imperfect reversal): {entropy_penalty_imperfect}")
