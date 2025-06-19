import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
import sympy as sp

# Assuming SymbolicDiscoveryEnv, NeuralGrammar, Expression, Variable are accessible.
# These imports will need to be relative to their new locations.

# from ..grammars.neural_grammar import NeuralGrammar # If NeuralGrammar is in grammars
# from .base_symbolic_env import SymbolicDiscoveryEnv # If SymbolicDiscoveryEnv is moved
# from ...core.progressive_grammar_system import Expression, Variable # If Variable is from a core module

# TEMPORARY: Using direct/potentially adjusted imports.
# These will be fixed in the "Adjust Imports" step.
from progressive_grammar_system import Expression, Variable
from .base_symbolic_env import SymbolicDiscoveryEnv # Adjusted import
from ..grammars.neural_grammar import NeuralGrammar


@dataclass
class AIBehaviorData:
    """Represents input-output data from an AI model."""
    inputs: np.ndarray  # Shape: (n_samples, input_dim)
    outputs: np.ndarray  # Shape: (n_samples, output_dim)
    intermediate_activations: Optional[Dict[str, np.ndarray]] = None
    attention_weights: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = None


class AIInterpretabilityEnv(SymbolicDiscoveryEnv):
    """Environment for discovering AI behavior laws."""

    def __init__(self,
                 ai_model: nn.Module,
                 grammar: NeuralGrammar, # Type hint updated
                 behavior_data: AIBehaviorData,
                 interpretation_mode: str = 'global',
                 **kwargs):
        """
        Args:
            ai_model: The AI model to interpret
            grammar: Neural grammar for expression generation
            behavior_data: Input-output data from the model
            interpretation_mode: 'global', 'local', or 'modular'
        """
        self.ai_model = ai_model
        self.interpretation_mode = interpretation_mode
        self.behavior_data = behavior_data

        # Extract variables from AI model structure
        variables = self._extract_variables_from_model()

        # Determine target_data for SymbolicDiscoveryEnv
        # The parent class SymbolicDiscoveryEnv expects target_data where the last column is the target.
        # Here, behavior_data.outputs is the direct target.
        # We need to ensure that the input data used by the expression evaluation in SymbolicDiscoveryEnv
        # corresponds to behavior_data.inputs.
        # The `variables` list created by `_extract_variables_from_model` will be based on `behavior_data.inputs`
        # and potentially `behavior_data.intermediate_activations`.
        # The parent's `_evaluate_expression` will use `self.target_data` which is `X_data` (inputs)
        # and `y_true` (target output).

        # If behavior_data.outputs is a 1D array, reshape it to 2D for consistency if SymbolicDiscoveryEnv expects that.
        # Based on SymbolicDiscoveryEnv, target_data is used to split into X_data (inputs for expression)
        # and y_true (target for expression).
        # However, AIInterpretabilityEnv directly uses behavior_data.inputs for expression evaluation context
        # and behavior_data.outputs as the target.
        # Let's pass behavior_data.inputs as `X_data_for_env` and behavior_data.outputs as `y_data_for_env`
        # to SymbolicDiscoveryEnv if it can accept them separately, or construct target_data appropriately.

        # Looking at SymbolicDiscoveryEnv:
        # self.target_data is assigned, then X_data = np.delete(self.target_data, self.target_variable_index, axis=1)
        # and y_true = self.target_data[:, self.target_variable_index].
        # This means self.target_data should be a combined array.

        # Let's construct `env_target_data` for the parent:
        # The `variables` passed to parent are derived from `behavior_data.inputs`.
        # So, `X_data` in parent should correspond to `behavior_data.inputs`.
        # `y_true` in parent should correspond to `behavior_data.outputs`.

        # Ensure outputs is 2D
        y_outputs = self.behavior_data.outputs
        if y_outputs.ndim == 1:
            y_outputs = y_outputs.reshape(-1, 1)

        # The `variables` list defines the inputs to the symbolic expressions.
        # These variables are derived from `self.behavior_data.inputs` and potentially activations.
        # The `SymbolicDiscoveryEnv` will try to predict `env_target_data[:, target_variable_index]`.
        # We need to ensure that the `X_data` used by `SymbolicDiscoveryEnv` corresponds to the values
        # that these `variables` refer to.

        # For now, the structure of SymbolicDiscoveryEnv implies that `variables` argument corresponds
        # to the columns of `X_data` (which is `target_data` excluding the target column).
        # If `_extract_variables_from_model` creates variables that are not simple columns of `behavior_data.inputs`,
        # this could be tricky.
        # `_extract_variables_from_model` uses `behavior_data.inputs` for "input" type variables.
        # And `behavior_data.intermediate_activations` for "activation" type variables.
        # This means the data source for evaluation in `SymbolicDiscoveryEnv` must contain all these values.

        # This part is complex. `SymbolicDiscoveryEnv`'s `_evaluate_expression` uses `self.target_data`.
        # `AIInterpretabilityEnv` overrides `_calculate_reward` and has its own `_evaluate_expression_on_data`.
        # Let's assume the `target_data` for parent is primarily for its own potential reward calculations if not fully overridden,
        # or for structural compatibility. The actual evaluation relevant for this class happens in its own methods.

        # For simplicity in this step, we'll pass `behavior_data.outputs` as the primary target data
        # and the `variables` as extracted. The evaluation logic in this class will handle using these correctly.
        # The parent's `target_data` might need to be a concatenation if its evaluation is used.
        # Given `AIInterpretabilityEnv` has its own `_calculate_reward`, `_calculate_fidelity`, etc.,
        # the `target_data` passed to parent might be less critical if parent's evaluation is fully bypassed.

        # Let's assume `SymbolicDiscoveryEnv` needs some form of `target_data` for its setup.
        # We can pass `behavior_data.outputs` and `variables` as is.
        # The `_evaluate_expression_on_data` method within this class will be responsible for
        # correctly using `self.behavior_data.inputs` and the `variables` to evaluate expressions.

        super().__init__(
            grammar=grammar,
            # This is the y-values the symbolic expressions will try to match.
            target_data=y_outputs,
            variables=variables, # These are the x-values (inputs) for the expressions
            # target_variable_index might need care if y_outputs has multiple columns.
            # If y_outputs has one column, target_variable_index=0.
            # If SymbolicDiscoveryEnv expects target_data to be X and y combined, this needs adjustment.
            # For now, assuming target_data is just 'y' and variables define 'X'.
            # SymbolicDiscoveryEnv's default _evaluate_expression might not work correctly if this assumption is wrong.
            # However, AIInterpretabilityEnv has its own _calculate_reward, _calculate_fidelity.
            **kwargs
        )

        # Add AI-specific reward components
        self.fidelity_weight = kwargs.get('fidelity_weight', 0.5)
        self.simplicity_weight = kwargs.get('simplicity_weight', 0.3)
        self.coverage_weight = kwargs.get('coverage_weight', 0.2)
        # Ensure base_reward_weight is handled if other weights don't sum to 1
        self.base_reward_weight = max(0, 1 - self.fidelity_weight - self.simplicity_weight - self.coverage_weight)


    def _extract_variables_from_model(self) -> List[Variable]:
        """Extract relevant variables from AI model structure."""
        variables = []

        # Input features
        input_dim = self.behavior_data.inputs.shape[1]
        for i in range(input_dim):
            var = Variable(
                name=f"x_{i}",
                index=i, # This index refers to columns in self.behavior_data.inputs
                properties={
                    "type": "input",
                    "statistics": self._compute_input_stats(i)
                }
            )
            variables.append(var)

        # Intermediate activations if available
        if self.behavior_data.intermediate_activations:
            for layer_name, activations in self.behavior_data.intermediate_activations.items():
                # Sample a few important neurons
                important_neurons = self._identify_important_neurons(activations)
                for neuron_idx in important_neurons[:5]:  # Limit for tractability
                    # These variables would need their data source during evaluation.
                    # The SymbolicDiscoveryEnv needs to know how to get data for these.
                    # This implies that the `_evaluate_expression_on_data` method below,
                    # or the parent's evaluation, must be able to map these variables to data.
                    var = Variable(
                        name=f"{layer_name}_n{neuron_idx}",
                        # Indexing for these variables needs careful handling.
                        # For now, assign a unique index.
                        index=len(variables),
                        properties={
                            "type": "activation",
                            "layer": layer_name,
                            "neuron": neuron_idx,
                            # We'd need to store/pass the actual activation data if used by parent's eval
                        }
                    )
                    variables.append(var)
        return variables

    def _compute_input_stats(self, feature_idx: int) -> Dict:
        """Compute statistics for input features."""
        feature_data = self.behavior_data.inputs[:, feature_idx]
        return {
            "mean": float(np.mean(feature_data)),
            "std": float(np.std(feature_data)),
            "min": float(np.min(feature_data)),
            "max": float(np.max(feature_data)),
            "unique_values": len(np.unique(feature_data))
        }

    def _identify_important_neurons(self, activations: np.ndarray) -> List[int]:
        """Identify neurons with high variance or correlation with output."""
        # Use variance as a simple importance measure
        if activations.shape[1] == 0: return []
        neuron_variance = np.var(activations, axis=0)
        # Sort and pick top, ensuring we don't exceed number of available neurons
        num_to_pick = min(10, activations.shape[1])
        important_indices = np.argsort(neuron_variance)[-num_to_pick:]
        return important_indices.tolist()

    def _evaluate_expression_on_data(self, expression: Expression) -> np.ndarray:
        """
        Evaluate the given symbolic expression on the input data stored in self.behavior_data.inputs.
        This method is crucial for calculating fidelity and coverage.
        It needs to correctly substitute variables in the expression with their corresponding data.
        """
        num_samples = self.behavior_data.inputs.shape[0]
        # Assuming expression.symbolic is a Sympy expression
        # The number of outputs from expression should match self.behavior_data.outputs.shape[1]
        # This is a simplified placeholder; actual evaluation depends on expression structure.
        # If expression is scalar, result is (num_samples,). If vector, (num_samples, output_dim).

        # This is a simplified version. A full implementation would need to handle
        # different types of variables (inputs, activations) and correctly map them.
        # For now, assume variables in the expression primarily refer to input features (x_i).

        # Create a list of sympy symbols for substitution based on self.variables from parent
        sympy_vars = [v.symbolic for v in self.variables if v.properties.get("type") == "input"]

        predictions = []

        for i in range(num_samples):
            subs_dict = {}
            input_sample = self.behavior_data.inputs[i, :]

            for var_obj in self.variables:
                var_name = var_obj.name
                var_idx = var_obj.index # This index should map to the correct data column
                var_type = var_obj.properties.get("type")

                if var_type == "input":
                    subs_dict[var_obj.symbolic] = input_sample[var_idx]
                elif var_type == "activation":
                    # This part is more complex: where does the data for activation variables come from?
                    # It would be from self.behavior_data.intermediate_activations[layer_name][:, neuron_idx_in_layer]
                    # This requires `_extract_variables_from_model` to store enough info or
                    # `_evaluate_expression_on_data` to look it up.
                    # For now, this is a gap if activation variables are used by expressions.
                    # Let's assume for this step that expressions primarily use input variables.
                    pass # Placeholder for activation variable substitution

            try:
                # Ensure expression.symbolic is not None
                if expression.symbolic is None:
                    # This can happen if the expression from grammar is invalid before symbolic conversion
                    pred_val = np.nan
                else:
                    pred_val = expression.symbolic.evalf(subs=subs_dict)
                predictions.append(float(pred_val))
            except Exception:
                predictions.append(np.nan) # Or some other error indicator

        return np.array(predictions)


    def _calculate_reward(self, expression: Expression) -> float:
        """Calculate reward for AI interpretation task."""
        # Base MSE reward from parent - this might not be directly usable if
        # parent's target_data setup is different.
        # Let's assume for now we don't use parent's _calculate_reward directly,
        # or that it's adapted to work with just y_outputs.
        # base_reward = super()._calculate_reward(expression)
        # For now, let's calculate a base reward (e.g. R-squared or inverse MSE) here directly.

        predicted_values = self._evaluate_expression_on_data(expression)
        actual_values = self.behavior_data.outputs.flatten() # Assuming single output for simplicity here

        # Filter out NaNs from predictions if any
        valid_indices = ~np.isnan(predicted_values)
        if not np.any(valid_indices):
            base_reward_metric = -np.inf # Or a large penalty
        else:
            predicted_valid = predicted_values[valid_indices]
            actual_valid = actual_values[valid_indices]
            if len(predicted_valid) < 2: # Need at least 2 points for correlation/SSE
                 base_reward_metric = -1.0 # Penalty for insufficient valid points
            else:
                # Using negative MSE as a base reward component (higher is better)
                mse = np.mean((predicted_valid - actual_valid)**2)
                # Normalize MSE, e.g., by variance of actual_values, or use R^2
                variance_actual = np.var(actual_valid)
                if variance_actual < 1e-9: # Avoid division by zero if actual is constant
                    base_reward_metric = -mse # Unnormalized negative MSE
                else:
                    r_squared = 1 - (mse / variance_actual)
                    base_reward_metric = r_squared # R-squared: 1 is perfect, 0 is mean, <0 is worse than mean


        # Fidelity: How well does the expression match AI behavior?
        fidelity = self._calculate_fidelity(expression) # Uses _evaluate_expression_on_data

        # Simplicity: Prefer simpler explanations (Occam's Razor)
        simplicity = 1.0 / (1.0 + expression.complexity) if expression.complexity is not None else 0.0

        # Coverage: What fraction of behaviors does this explain?
        coverage = self._calculate_coverage(expression) # Uses _evaluate_expression_on_data

        total_reward = (
            self.fidelity_weight * fidelity +
            self.simplicity_weight * simplicity +
            self.coverage_weight * coverage +
            self.base_reward_weight * base_reward_metric # Use the locally computed base metric
        )
        return total_reward

    def _calculate_fidelity(self, expression: Expression) -> float:
        """Measure how faithfully the expression reproduces AI behavior."""
        try:
            # Evaluate expression on input data
            predicted = self._evaluate_expression_on_data(expression)
            actual = self.behavior_data.outputs

            # Ensure predicted and actual are compatible for correlation
            # Handle NaNs from prediction
            valid_mask = ~np.isnan(predicted)
            if not np.any(valid_mask): return 0.0

            predicted_valid = predicted[valid_mask]

            # Actual might be 2D (n_samples, output_dim) or 1D (n_samples,)
            # Predicted is currently 1D (n_samples,) from _evaluate_expression_on_data
            # This needs to align. For now, assume actual is (n_samples,) or (n_samples, 1)

            if actual.ndim == 2 and actual.shape[1] > 1:
                # Multi-output case: calculate correlation per output and average
                # This requires _evaluate_expression_on_data to return (n_samples, output_dim)
                # For now, this part is simplified / assumes single output from expression
                # Or that fidelity is calculated for the primary output if multiple exist.
                # Let's assume actual refers to the first output column if multiple exist.
                actual_flat = actual[valid_mask, 0] if actual.ndim == 2 else actual[valid_mask]
            else:
                actual_flat = actual.flatten()[valid_mask]

            if len(predicted_valid) < 2 or len(actual_flat) < 2: # Need at least 2 points for correlation
                return 0.0

            correlation = np.corrcoef(predicted_valid, actual_flat)[0, 1]
            return max(0, correlation) if not np.isnan(correlation) else 0.0

        except Exception:
            return 0.0

    def _calculate_coverage(self, expression: Expression) -> float:
        """Measure what fraction of the input space this expression covers (is well-defined)."""
        try:
            predicted = self._evaluate_expression_on_data(expression)
            valid_predictions = ~np.isnan(predicted)
            return np.mean(valid_predictions)
        except Exception:
            return 0.0


class LocalInterpretabilityEnv(AIInterpretabilityEnv):
    """Specialized environment for local interpretability around specific inputs."""

    def __init__(self,
                 ai_model: nn.Module,
                 grammar: NeuralGrammar, # Type hint updated
                 # behavior_data is optional here, will be generated if None
                 behavior_data: Optional[AIBehaviorData],
                 anchor_input: np.ndarray,
                 neighborhood_size: float = 0.1,
                 **kwargs):
        """
        Args:
            anchor_input: The specific input to explain
            neighborhood_size: Size of local neighborhood to consider
        """
        self.anchor_input = anchor_input
        self.neighborhood_size = neighborhood_size

        # Generate local perturbations if behavior_data is not provided
        if behavior_data is None:
            # Need to pass the model to _generate_local_data if it's not yet part of self
            # This is tricky due to super().__init__ call order.
            # Simplest is to generate data before calling super().__init__.
            # We need `ai_model` for this.
            generated_local_data = self._generate_local_data(ai_model, anchor_input, neighborhood_size)
        else:
            generated_local_data = behavior_data


        super().__init__(
            ai_model=ai_model,
            grammar=grammar,
            behavior_data=generated_local_data,
            interpretation_mode='local',
            **kwargs
        )

    def _generate_local_data(self, ai_model_ref: nn.Module, # Pass model explicitly
                               anchor: np.ndarray,
                               neighborhood_size: float) -> AIBehaviorData:
        """Generate perturbed inputs around anchor point."""
        n_samples = kwargs.get('local_n_samples', 1000) # Allow customization via kwargs
        dim = anchor.shape[0]

        # Generate perturbations
        perturbations = np.random.normal(0, neighborhood_size, (n_samples, dim))
        inputs = anchor + perturbations # anchor might need to be (1, dim) for broadcasting if inputs is (N, dim)
        if anchor.ndim == 1:
            inputs = anchor.reshape(1, -1) + perturbations
        else:
            inputs = anchor + perturbations


        # Get model outputs
        ai_model_ref.eval() # Use the passed model reference
        with torch.no_grad():
            inputs_tensor = torch.FloatTensor(inputs)
            outputs_tensor = ai_model_ref(inputs_tensor)
            outputs = outputs_tensor.numpy()
            if outputs.ndim == 1:
                outputs = outputs.reshape(-1,1)


        return AIBehaviorData(inputs=inputs, outputs=outputs)
