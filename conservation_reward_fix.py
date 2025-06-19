# conservation_reward_fix.py
import numpy as np
from typing import List, Dict, Any

import numpy as np
from typing import List, Dict, Any, Optional, Union

class ConservationBiasedReward:
    """
    Calculates a reward bonus based on the adherence to specified conservation laws.

    This class is designed to be used in a reinforcement learning environment
    where an agent's actions (or discovered hypotheses) are evaluated based on
    how well they conserve physical quantities like energy, momentum, etc.
    It also includes a mechanism to penalize entropy production, approximated
    by the irreversibility of energy changes in forward and backward trajectories.

    Attributes:
        conservation_types (List[str]): A list of conservation laws to check (e.g., ['energy', 'momentum']).
        weight_factor (float): A factor to scale the computed conservation bonus.
        tolerances (Dict[str, float]): A dictionary mapping conservation types to their
                                       allowed violation tolerances. Violations within
                                       tolerance receive a higher bonus.
        history (Dict[str, List[Dict[str, float]]]): Stores a history of violations and bonuses
                                                     for each conservation type, useful for diagnostics.
    """
    def __init__(self, conservation_types: List[str], weight_factor: float) -> None:
        """
        Initializes the ConservationBiasedReward instance.

        Args:
            conservation_types: A list of strings identifying the conservation laws
                                to be evaluated (e.g., "energy", "momentum").
            weight_factor: A float that scales the final conservation bonus.
        """
        self.conservation_types: List[str] = conservation_types
        self.weight_factor: float = weight_factor
        self.tolerances: Dict[str, float] = {
            'energy': 1e-3,
            'momentum': 1e-4,
            'mass': 1e-5,
            'angular_momentum': 1e-4
        }
        self.history: Dict[str, List[Dict[str, float]]] = {} # For advanced diagnostics or adaptive weighting

    def _calculate_violation(self,
                             predicted_val: Optional[Union[np.ndarray, float]],
                             ground_truth_val: Optional[Union[np.ndarray, float]],
                             c_type: str) -> float:
        """
        Calculates the normalized violation between predicted and ground truth conserved quantities.

        The violation is normalized by the magnitude of the ground truth value to make it
        relative. Different normalization strategies are used for scalar, vector, and
        higher-dimensional tensor quantities.

        Args:
            predicted_val: The predicted value of the conserved quantity. Can be None.
            ground_truth_val: The ground truth value of the conserved quantity. Can be None.
            c_type: The type of conservation law (e.g., 'energy'), used for context (not directly in calculation here).

        Returns:
            A float representing the calculated violation. Returns 1.0 (max violation)
            if input data is missing, shapes mismatch, or for unhandled dimensions.

        Raises:
            ValueError: If inputs cannot be converted to np.asarray (though np.asarray handles many types).
        """
        if predicted_val is None or ground_truth_val is None:
            return 1.0  # Max violation if data is missing

        try:
            predicted_val_np = np.asarray(predicted_val)
            ground_truth_val_np = np.asarray(ground_truth_val)
        except Exception as e:
            # print(f"Error converting values to numpy arrays for {c_type}: {e}") # For debugging
            return 1.0 # Max violation if conversion fails

        if predicted_val_np.shape != ground_truth_val_np.shape:
            # print(f"Shape mismatch for {c_type}: {predicted_val_np.shape} vs {ground_truth_val_np.shape}") # For debugging
            return 1.0 # Max violation for shape mismatch

        violation: float
        if predicted_val_np.ndim == 0:  # Scalar value
            diff = np.abs(predicted_val_np - ground_truth_val_np)
            # Normalize by ground truth magnitude, or use absolute diff if GT is near zero
            # This part was missing a proper scalar normalization in the original code
            gt_mag = np.abs(ground_truth_val_np)
            if gt_mag < 1e-9: # Avoid division by zero or very small numbers
                 violation = diff
            else:
                 violation = diff / gt_mag
        elif predicted_val_np.ndim == 1:  # Vector value
            norm_gt = np.linalg.norm(ground_truth_val_np)
            if norm_gt < 1e-9: # If ground truth vector is zero vector
                violation = np.linalg.norm(predicted_val_np - ground_truth_val_np)
            else:
                violation = np.linalg.norm(predicted_val_np - ground_truth_val_np) / norm_gt
        elif predicted_val_np.ndim > 1:  # Tensor value
            # Calculate norm along all axes except the first (batch axis, if any)
            # This assumes the first dimension might be a batch or time series dimension.
            # If it's a single tensor, these axes will be (0, 1, ...)
            axes_to_norm = tuple(range(predicted_val_np.ndim)) # Norm over all elements for higher-dim tensors

            diff_norm = np.linalg.norm((predicted_val_np - ground_truth_val_np).flatten())
            gt_norm = np.linalg.norm(ground_truth_val_np.flatten())

            if gt_norm < 1e-9:
                violation = diff_norm
            else:
                violation = diff_norm / gt_norm
        else: # Should not happen given ndim checks
            return 1.0

        return float(np.clip(violation, 0.0, 1.0)) # Clip violation to [0,1] as it's used in exp

    def compute_conservation_bonus(self,
                                   predicted_traj: Dict[str, Any],
                                   ground_truth_traj: Dict[str, Any],
                                   hypothesis_params: Dict[str, Any]) -> float:
        """
        Computes a bonus based on how well predicted trajectories adhere to conservation laws.

        The bonus is calculated for each specified conservation type. For each type,
        the violation between predicted and ground truth values is computed. This
        violation is then transformed into a bonus using an exponential decay function,
        scaled by a tolerance factor. The final bonus is the average of individual
        bonuses, multiplied by the overall weight_factor.

        Args:
            predicted_traj: A dictionary containing predicted conserved quantities.
                            Expected keys are like 'conserved_energy', 'conserved_momentum'.
            ground_truth_traj: A dictionary containing ground truth conserved quantities.
                               Expected keys match those in predicted_traj.
            hypothesis_params: Parameters of the hypothesis that generated the prediction.
                               Currently unused in this method but kept for API consistency.

        Returns:
            A float representing the total conservation bonus. Returns 0.0 if no
            conservation laws are evaluated (e.g., due to missing data).
        """
        total_bonus: float = 0.0
        num_laws_evaluated: int = 0

        for c_type in self.conservation_types:
            pred_val: Optional[Union[np.ndarray, float]] = predicted_traj.get(f'conserved_{c_type}')
            gt_val: Optional[Union[np.ndarray, float]] = ground_truth_traj.get(f'conserved_{c_type}')

            if pred_val is None or gt_val is None:
                # print(f"Warning: Missing data for conservation type '{c_type}'. Skipping.") # For debugging
                continue

            violation: float = self._calculate_violation(pred_val, gt_val, c_type)
            tolerance: float = self.tolerances.get(c_type, 1e-3) # Default tolerance if not specified

            # Bonus is higher for lower violation, scaled by tolerance
            bonus: float = np.exp(-violation / tolerance)
            total_bonus += bonus
            num_laws_evaluated += 1

            # Store history for diagnostics
            if c_type not in self.history:
                self.history[c_type] = []
            self.history[c_type].append({'violation': violation, 'bonus': bonus, 'timestamp': np.datetime64('now')}) # type: ignore

        if num_laws_evaluated == 0:
            return 0.0

        average_bonus: float = total_bonus / num_laws_evaluated
        return self.weight_factor * average_bonus

    def diagnose_conservation_violations(self) -> Dict[str, Dict[str, Union[float, int]]]:
        """
        Provides diagnostic information about conservation law violations.

        This method aggregates the history of violations and bonuses for each
        conservation type, calculating average violations, average bonuses, and
        the number of evaluations.

        Returns:
            A dictionary where keys are conservation types (e.g., 'energy').
            Each value is another dictionary containing:
                'average_violation': The mean violation recorded for that type.
                'average_bonus': The mean bonus awarded for that type.
                'num_evals': The number of times this conservation type was evaluated.
        """
        diagnostics: Dict[str, Dict[str, Union[float, int]]] = {}
        for c_type, records in self.history.items():
            if records:
                avg_violation: float = float(np.mean([r['violation'] for r in records]))
                avg_bonus: float = float(np.mean([r['bonus'] for r in records]))
                diagnostics[c_type] = {
                    'average_violation': avg_violation,
                    'average_bonus': avg_bonus,
                    'num_evals': len(records)
                }
        return diagnostics

    def get_entropy_production_penalty(self,
                                       predicted_forward_traj: Optional[Dict[str, Any]],
                                       predicted_backward_traj: Optional[Dict[str, Any]],
                                       hypothesis_params: Dict[str, Any]) -> float:
        """
        Calculates a penalty based on an approximation of entropy production.

        This method compares the change in a conserved quantity (typically 'energy')
        over a forward trajectory with its change over a time-reversed backward trajectory.
        A larger difference (dissipation metric) implies higher entropy production and
        results in a more negative penalty (i.e., a larger penalty value).
        The penalty is scaled by `self.weight_factor` and an additional factor of 0.1.

        Args:
            predicted_forward_traj: Dictionary of conserved quantities from the forward pass.
                                    Expected to contain 'conserved_energy' as a list/array of values.
            predicted_backward_traj: Dictionary of conserved quantities from the backward pass
                                     (with initial and final states swapped relative to forward).
                                     Expected to contain 'conserved_energy_reversed_path'.
            hypothesis_params: Parameters of the hypothesis. Currently unused.

        Returns:
            A float representing the entropy production penalty. This value is typically
            negative or zero. Returns 0.0 if required data is missing.
        """
        if predicted_forward_traj is None or predicted_backward_traj is None:
            return 0.0

        energy_fwd: Optional[Union[List[float], np.ndarray]] = predicted_forward_traj.get('conserved_energy')
        energy_bwd: Optional[Union[List[float], np.ndarray]] = predicted_backward_traj.get('conserved_energy_reversed_path')

        if energy_fwd is not None and energy_bwd is not None:
            # Ensure they are numpy arrays for consistent processing
            energy_fwd_np = np.asarray(energy_fwd)
            energy_bwd_np = np.asarray(energy_bwd)

            if energy_fwd_np.size < 2 or energy_bwd_np.size < 2 : # Need at least two points to see a change
                return 0.0

            fwd_change: float = float(np.abs(energy_fwd_np[-1] - energy_fwd_np[0]))
            bwd_change: float = float(np.abs(energy_bwd_np[-1] - energy_bwd_np[0])) # Assumes bwd path energy is ordered from new start to new end

            # Dissipation metric: relative difference in energy changes
            # The sum in denominator acts as normalization
            denominator = (fwd_change + bwd_change) / 2.0 + 1e-9 # Avoid division by zero
            dissipation_metric: float = np.abs(fwd_change - bwd_change) / denominator

            # Penalty is logarithmic with the dissipation; higher dissipation = larger negative penalty
            # Using -log(1+x) ensures penalty is <= 0.
            # For x=0 (no dissipation), penalty=0. For x>0, penalty < 0.
            penalty: float = -np.log(1 + dissipation_metric)

            # Apply weighting factors
            return penalty * self.weight_factor * 0.1 # Additional scaling factor for this specific penalty
        return 0.0

if __name__ == '__main__':
    # Example Usage:
    print("--- Example 1: Vector Conservation ---")
    reward_system_vector = ConservationBiasedReward(
        conservation_types=['energy', 'momentum'], # Specify which laws to check
        weight_factor=0.5 # Overall scaling for the bonus
    )

    # Simulated predicted and ground truth data for a trajectory
    predicted_trajectory_vec = {
        'conserved_energy': np.array([10.0, 10.01, 9.99]), # Small energy fluctuations
        'conserved_momentum': np.array([5.0, 5.05, 4.95]), # Small momentum fluctuations
        # 'conserved_angular_momentum': np.array([1.0, 1.0, 0.9]) # Example if we add more
    }
    ground_truth_trajectory_vec = {
        'conserved_energy': np.array([10.0, 10.0, 10.0]), # Ideal energy conservation
        'conserved_momentum': np.array([5.0, 5.0, 5.0]),  # Ideal momentum conservation
        # 'conserved_angular_momentum': np.array([1.0, 1.0, 1.0])
    }
    # Hypothesis parameters (not used by current bonus computation but part of API)
    hypothesis_parameters_vec = {'mass': 1.0, 'spring_constant': 2.0}

    bonus_vec = reward_system_vector.compute_conservation_bonus(
        predicted_trajectory_vec, ground_truth_trajectory_vec, hypothesis_parameters_vec
    )
    print(f"Conservation Bonus (Vector): {bonus_vec:.4f}")

    diagnostics_vec = reward_system_vector.diagnose_conservation_violations()
    print(f"Diagnostics (Vector): {diagnostics_vec}")
    print("\n")

    print("--- Example 2: Scalar Conservation ---")
    reward_system_scalar = ConservationBiasedReward(conservation_types=['energy'], weight_factor=1.0)
    predicted_scalar_data = {'conserved_energy': 10.05} # Single predicted scalar value
    ground_truth_scalar_data = {'conserved_energy': 10.0}   # Single ground truth scalar value

    scalar_bonus = reward_system_scalar.compute_conservation_bonus(
        predicted_scalar_data, ground_truth_scalar_data, {}
    )
    print(f"Conservation Bonus (Scalar): {scalar_bonus:.4f}")
    diagnostics_scalar = reward_system_scalar.diagnose_conservation_violations()
    print(f"Diagnostics (Scalar): {diagnostics_scalar}")
    print("\n")

    print("--- Example 3: Multi-dimensional Tensor Conservation (e.g., stress tensor) ---")
    # For this example, let's assume 'stress_tensor' is a conserved quantity
    reward_system_tensor = ConservationBiasedReward(conservation_types=['stress_tensor'], weight_factor=0.8)
    reward_system_tensor.tolerances['stress_tensor'] = 1e-2 # Set a specific tolerance

    predicted_tensor_data = {
        'stress_tensor': np.array([[[1.0, 0.05], [0.05, 2.0]]]) # 1x2x2 tensor
    }
    ground_truth_tensor_data = {
        'stress_tensor': np.array([[[1.0, 0.0], [0.0, 2.0]]]) # Ideal tensor
    }
    tensor_bonus = reward_system_tensor.compute_conservation_bonus(
        predicted_tensor_data, ground_truth_tensor_data, {}
    )
    print(f"Conservation Bonus (Tensor): {tensor_bonus:.4f}")
    diagnostics_tensor = reward_system_tensor.diagnose_conservation_violations()
    print(f"Diagnostics (Tensor): {diagnostics_tensor}")
    print("\n")

    print("--- Example 4: Entropy Production Penalty ---")
    # Forward trajectory (e.g., energy values over time)
    pred_fwd_traj = {'conserved_energy': np.array([10.0, 9.9, 9.8, 9.75])}
    # Backward trajectory (energy values for the time-reversed process)
    # Ideal reversal would bring energy back to initial state: e.g. [9.75, 9.8, 9.9, 10.0]
    pred_bwd_traj_ideal_reversed = {'conserved_energy_reversed_path': np.array([9.75, 9.8, 9.9, 10.0])}
    pred_bwd_traj_imperfect_reversed = {'conserved_energy_reversed_path': np.array([9.7, 9.75, 9.8, 9.85])} # Dissipation

    entropy_penalty_ideal = reward_system_vector.get_entropy_production_penalty(
        pred_fwd_traj, pred_bwd_traj_ideal_reversed, {}
    )
    print(f"Entropy Production Penalty (Ideal Reversal): {entropy_penalty_ideal:.4f}")

    entropy_penalty_imperfect = reward_system_vector.get_entropy_production_penalty(
        pred_fwd_traj, pred_bwd_traj_imperfect_reversed, {}
    )
    print(f"Entropy Production Penalty (Imperfect Reversal): {entropy_penalty_imperfect:.4f}")

    # Example with missing data for entropy penalty
    entropy_penalty_missing = reward_system_vector.get_entropy_production_penalty(
        pred_fwd_traj, None, {}
    )
    print(f"Entropy Production Penalty (Missing Backward Data): {entropy_penalty_missing:.4f}")
    print("\n")

    print("--- Example 5: Missing data in compute_conservation_bonus ---")
    predicted_missing = {'conserved_energy': np.array([10.0, 10.0])} # Momentum missing
    bonus_missing = reward_system_vector.compute_conservation_bonus(
        predicted_missing, ground_truth_trajectory_vec, {}
    )
    print(f"Bonus with missing predicted data: {bonus_missing:.4f}")
    # After this call, 'momentum' would not be in reward_system_vector.history for this computation
    # print(f"Diagnostics after missing data: {reward_system_vector.diagnose_conservation_violations()}")
    # To show individual effects, let's get fresh diagnostics
    reward_system_temp = ConservationBiasedReward(conservation_types=['energy', 'momentum'], weight_factor=0.5)
    reward_system_temp.compute_conservation_bonus(predicted_missing, ground_truth_trajectory_vec, {})
    print(f"Diagnostics for missing data scenario: {reward_system_temp.diagnose_conservation_violations()}")
