import numpy as np
from typing import Any, Callable, Dict, Union, Optional
import torch
import sympy as sp

# Assuming Expression and AIBehaviorData are accessible.
# from ...core.grammar import Expression
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

    def _get_model_output(self, features: np.ndarray) -> np.ndarray:
        """
        Get model outputs for given features.
        Handles different model types and ensures consistent output format.
        """
        self.ai_model.eval()
        with torch.no_grad():
            inputs = torch.tensor(features, dtype=torch.float32)
            if inputs.dim() == 1:
                inputs = inputs.unsqueeze(0)

            output_tensor = self.ai_model(inputs)

            if hasattr(output_tensor, 'logits'):
                output_tensor = output_tensor.logits
            elif isinstance(output_tensor, tuple):
                output_tensor = output_tensor[0]

            output_array = output_tensor.cpu().numpy()

            if output_array.ndim > 2:
                output_array = output_array.reshape(output_array.shape[0], -1)
            return output_array

    def _evaluate_expression_on_data(self, expression: Any, data: Union[np.ndarray, Dict[str, np.ndarray]]) -> Optional[np.ndarray]:
        """
        Evaluate a symbolic expression on input data.
        """
        if expression is None or not hasattr(expression, 'symbolic'):
            return None
        try:
            var_mapping = {}
            if isinstance(data, dict):
                for var in self.variables:
                    if var.name in data:
                        var_mapping[var.symbolic] = data[var.name]
            else:
                for var in self.variables:
                    if var.index < data.shape[1]:
                        var_mapping[var.symbolic] = data[:, var.index]

            # Use lambdify for fast evaluation
            func = sp.lambdify(
                list(var_mapping.keys()),
                expression.symbolic,
                modules=['numpy', {'Attention': self._compute_attention_value}]
            )
            result = func(*var_mapping.values())
            return np.array(result).flatten()
        except Exception as e:
            print(f"Expression evaluation failed: {e}")
            return None

    def _compute_attention_value(self, query, key, value):
        """Computes simplified scalar attention."""
        score = query * key
        attention_weight = np.exp(score) / (np.exp(score) + 1)
        return attention_weight * value

    def _compute_embedding_value(self, index, embedding_ref):
        """Computes placeholder embedding lookup."""
        index = int(index)
        return float(hash(f"embed_{index}_{embedding_ref}") % 100) / 100.0

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
