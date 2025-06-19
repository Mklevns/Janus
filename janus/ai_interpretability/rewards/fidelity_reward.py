import numpy as np
from typing import Any, Callable

# Assuming Expression and AIBehaviorData are accessible.
# from ...core.progressive_grammar_system import Expression
# from ..environments.neural_net_env import AIBehaviorData (or a common types module)

# TEMPORARY: Using direct/potentially adjusted imports or forward references.
# These will be fixed in the "Adjust Imports" step.
from progressive_grammar_system import Expression
from ..environments.neural_net_env import AIBehaviorData # Assuming it's here for now


class FidelityRewardCalculator:
    """
    Calculates fidelity-related rewards for symbolic expressions explaining AI model behavior.
    """

    def __init__(self, behavior_data: AIBehaviorData,
                 evaluate_expression_func: Callable[[Expression, AIBehaviorData], np.ndarray]):
        """
        Args:
            behavior_data: The AI model's input-output data.
            evaluate_expression_func: A function that takes an Expression and AIBehaviorData,
                                      and returns the expression's predictions as a numpy array.
                                      Example: `env._evaluate_expression_on_data`
        """
        self.behavior_data = behavior_data
        self.evaluate_expression_func = evaluate_expression_func

    def calculate_fidelity_score(self, expression: Expression) -> float:
        """
        Measures how faithfully the expression reproduces AI behavior using correlation.
        This is extracted from AIInterpretabilityEnv._calculate_fidelity.
        """
        try:
            # Evaluate expression on input data using the provided function
            predicted = self.evaluate_expression_func(expression, self.behavior_data)
            actual = self.behavior_data.outputs

            # Ensure predicted and actual are compatible for correlation
            valid_mask = ~np.isnan(predicted)
            if not np.any(valid_mask):
                return 0.0

            predicted_valid = predicted[valid_mask]

            if actual.ndim == 2 and actual.shape[1] > 1:
                # Multi-output: For now, use correlation with the first output column.
                # A more sophisticated approach might average correlations or use other metrics.
                actual_flat = actual[valid_mask, 0] if actual.ndim == 2 else actual[valid_mask]
            else:
                actual_flat = actual.flatten()[valid_mask]

            if len(predicted_valid) < 2 or len(actual_flat) < 2: # Correlation requires at least 2 points
                return 0.0

            # Calculate correlation
            correlation = np.corrcoef(predicted_valid, actual_flat)[0, 1]

            # Return correlation, ensuring it's non-negative and handles NaN (e.g., if std dev is zero)
            return max(0, correlation) if not np.isnan(correlation) else 0.0

        except Exception as e:
            # print(f"Error during fidelity calculation: {e}") # Optional: for debugging
            return 0.0

    def calculate_mse_score(self, expression: Expression, normalized: bool = True) -> float:
        """
        Calculates Mean Squared Error between expression predictions and actual outputs.
        Returns negative MSE (so higher is better) or R-squared if normalized.
        """
        try:
            predicted = self.evaluate_expression_func(expression, self.behavior_data)
            actual = self.behavior_data.outputs.flatten() # Assuming single output for simplicity

            valid_mask = ~np.isnan(predicted)
            if not np.any(valid_mask):
                return -np.inf # Large penalty

            predicted_valid = predicted[valid_mask]
            actual_valid = actual[valid_mask]

            if len(predicted_valid) == 0:
                return -np.inf

            mse = np.mean((predicted_valid - actual_valid)**2)

            if normalized:
                variance_actual = np.var(actual_valid)
                if variance_actual < 1e-9: # Actual is constant
                    return -mse # Return unnormalized negative MSE
                r_squared = 1 - (mse / variance_actual)
                return r_squared
            else:
                return -mse # Negative MSE (higher is better)

        except Exception:
            return -np.inf


# Example of how this might be used within an environment or reward calculation pipeline:
#
# class YourEnvironment:
#     def __init__(self, ..., behavior_data: AIBehaviorData):
#         self.behavior_data = behavior_data
#         # _evaluate_expression_on_data is a method of this environment
#         self.fidelity_calculator = FidelityRewardCalculator(
#             behavior_data,
#             self._evaluate_expression_on_data
#         )
#
#     def _evaluate_expression_on_data(self, expression: Expression, current_behavior_data: AIBehaviorData) -> np.ndarray:
#         # ... implementation to evaluate expression ...
#         # This function is passed to FidelityRewardCalculator
#         pass
#
#     def _calculate_reward_component(self, expression: Expression):
#         fidelity_c = self.fidelity_calculator.calculate_fidelity_score(expression)
#         # ... use fidelity_c in total reward ...
#         return fidelity_c

```
