"""
AI Interpretability Rewards
===========================

Defines reward components tailored for evaluating the interpretability
of symbolic expressions that aim to explain AI model behavior.
"""

import numpy as np
import torch
from sklearn.metrics import mean_squared_error
import sympy as sp
from typing import Any, Dict, Optional, List # Add List if not present

# Assuming Expression class will be available, e.g., from janus.core.expression
# from janus.core.expression import Expression
# For now, using Any for Expression type hint if not directly importable yet
ExpressionType = Any
AIModelType = Any # Placeholder for AI Model type
TestDataTyp = np.ndarray # Placeholder for Test Data type, likely np.ndarray or torch.Tensor

class InterpretabilityReward:
    """
    Calculates rewards for symbolic expressions based on how well they
    explain an AI model's behavior, focusing on fidelity, simplicity,
    consistency, and insightfulness.
    """

    def __init__(self,
                 reward_weights: Optional[Dict[str, float]] = None,
                 complexity_penalty_factor: float = 0.01, # Example factor
                 max_complexity_for_penalty: Optional[int] = None # No penalty if below this
                ):
        """
        Initializes the InterpretabilityReward calculator.

        Args:
            reward_weights: A dictionary specifying the weights for different
                            reward components (e.g., fidelity, simplicity).
                            Defaults to equal weights if not provided.
            complexity_penalty_factor: Factor to scale complexity penalty.
            max_complexity_for_penalty: If expression complexity exceeds this, penalty applies.
                                       If None, penalty always applies based on complexity.
        """
        if reward_weights is None:
            self.reward_weights: Dict[str, float] = {
                'fidelity': 0.25,
                'simplicity': 0.25,
                'consistency': 0.25,
                'insight': 0.25
            }
        else:
            self.reward_weights = reward_weights

        self.complexity_penalty_factor = complexity_penalty_factor
        self.max_complexity_for_penalty = max_complexity_for_penalty


    def calculate_reward(self,
                         expression: ExpressionType,
                         ai_model: AIModelType,
                         test_data: TestDataTyp,
                         additional_context: Optional[Dict[str, Any]] = None # For future use
                        ) -> float:
        """
        Calculates the overall interpretability reward for a given symbolic expression.

        Args:
            expression: The symbolic expression (e.g., an Expression object) to evaluate.
            ai_model: The AI model being interpreted.
            test_data: Data used to test fidelity and generalization.
            additional_context: Optional dictionary for any other relevant information.

        Returns:
            A scalar reward value.
        """
        if expression is None: # Handle cases where no valid expression is formed
            return -1.0 # Or some other suitable penalty

        # 1. Fidelity: How well does the expression match the AI model's behavior?
        fidelity_score = self._calculate_fidelity(expression, ai_model, test_data)

        # 2. Simplicity: Prefer simpler explanations (e.g., based on expression complexity).
        # The user request had: 1.0 / (1.0 + expression.complexity)
        # This is a good starting point. We can also add penalties for excessive complexity.
        simplicity_score = self._calculate_simplicity(expression)

        # 3. Consistency/Generalization: Does the explanation hold across different inputs or data subsets?
        # This might be partly covered by fidelity if test_data includes unseen samples.
        # Or, it could be a separate test.
        consistency_score = self._test_consistency(expression, ai_model, test_data) # test_data might be split

        # 4. Insightfulness: Does the expression reveal something non-obvious or useful?
        # This is the most abstract and hardest to quantify. Placeholder for now.
        insight_score = self._calculate_insight_score(expression, ai_model, additional_context)

        # Combine rewards using the defined weights
        total_reward = self.combine_rewards(
            fidelity=fidelity_score,
            simplicity=simplicity_score,
            consistency=consistency_score,
            insight=insight_score
        )

        return total_reward

    def _calculate_fidelity(self, expression: ExpressionType, ai_model: AIModelType, test_data: np.ndarray) -> float:
        """
        Calculate how well the symbolic expression matches the AI model's behavior.
        Returns a score between 0 and 1, where 1 is perfect fidelity.
        """
        try:
            # Get model predictions as the ground truth
            ai_predictions = self._get_interpretation_target(ai_model, test_data)

            # Evaluate expression on the same inputs
            expr_predictions = self._evaluate_expression_on_data(expression, test_data)

            if expr_predictions is None or len(expr_predictions) != len(ai_predictions):
                return 0.0

            # Filter out NaN values
            valid_mask = ~(np.isnan(expr_predictions) | np.isnan(ai_predictions))
            if not np.any(valid_mask):
                return 0.0

            expr_valid = expr_predictions[valid_mask]
            ai_valid = ai_predictions[valid_mask]

            # Calculate correlation (primary metric)
            if len(expr_valid) >= 2:
                correlation = np.corrcoef(expr_valid, ai_valid)[0, 1]
                correlation = max(0, correlation) if not np.isnan(correlation) else 0.0
            else:
                correlation = 0.0

            # Calculate normalized MSE (secondary metric)
            mse = mean_squared_error(ai_valid, expr_valid)
            ai_variance = np.var(ai_valid)
            normalized_mse = 1.0 / (1.0 + mse / (ai_variance + 1e-10))

            # Combine metrics (weighted average)
            fidelity = 0.7 * correlation + 0.3 * normalized_mse

            return float(fidelity)

        except Exception as e:
            print(f"Error in fidelity calculation: {e}")
            return 0.0

    def _calculate_simplicity(self, expression: ExpressionType) -> float:
        """
        Calculates the simplicity score for an expression.
        Uses the formula 1.0 / (1.0 + expression.complexity) and applies
        an optional penalty for exceeding a complexity threshold.
        """
        complexity = getattr(expression, 'complexity', 100) # Default to high complexity if attribute missing

        base_simplicity_score = 1.0 / (1.0 + complexity)

        penalty = 0.0
        if self.max_complexity_for_penalty is not None:
            if complexity > self.max_complexity_for_penalty:
                penalty = self.complexity_penalty_factor * (complexity - self.max_complexity_for_penalty)
        else:
            # If no max_complexity_for_penalty, penalty might always apply based on factor
            # Or, interpret as no penalty if max_complexity_for_penalty is None.
            # Let's assume the original formula is the primary score, and penalty is conditional.
            pass # No penalty if max_complexity_for_penalty is not set for thresholding

        simplicity_score = base_simplicity_score - penalty
        # Ensure score is not excessively negative, e.g., clip at 0 or a small negative value.
        return max(0, simplicity_score) # Ensure simplicity isn't < 0


    def _test_consistency(self, expression: ExpressionType, ai_model: AIModelType, test_data: np.ndarray) -> float:
        """
        Test how consistently the expression performs across different data subsets.
        Uses k-fold validation approach to measure consistency.
        """
        try:
            features = test_data.inputs if hasattr(test_data, 'inputs') else test_data
            n_samples = features.shape[0]
            n_folds = min(5, n_samples // 10) # Ensure reasonable fold size

            if n_folds < 2: # If not enough data for k-fold, fall back to overall fidelity
                return self._calculate_fidelity(expression, ai_model, test_data)

            fold_scores = []
            fold_size = n_samples // n_folds

            for i in range(n_folds):
                start_idx, end_idx = i * fold_size, (i + 1) * fold_size
                # Create a new TestDataTyp or np.ndarray for the fold
                # Assuming test_data is a simple np.ndarray of features here based on `features = test_data`
                fold_data_features = features[start_idx:end_idx]
                # If test_data was a more complex object, ensure fold_data retains that structure
                # For simplicity, passing features directly as per new _calculate_fidelity expects np.ndarray
                fold_score = self._calculate_fidelity(expression, ai_model, fold_data_features)
                fold_scores.append(fold_score)

            avg_score = np.mean(fold_scores)
            score_variance = np.var(fold_scores)
            # Consistency rewards high average score and penalizes high variance
            consistency = avg_score * (1.0 - min(score_variance, 0.5)) # Cap variance penalty

            return float(consistency)

        except Exception as e:
            print(f"Error in consistency calculation: {e}")
            return 0.0

    def _calculate_insight_score(self,
                                 expression: ExpressionType,
                                 ai_model: AIModelType,
                                 additional_context: Optional[Dict[str, Any]] = None
                                ) -> float:
        """
        Calculate the insightfulness of the discovered expression.
        """
        if expression is None or not hasattr(expression, 'symbolic'):
            return 0.0

        insight_score = 0.0
        # Ensure expression.symbolic is a string for pattern matching
        expr_str = str(expression.symbolic)

        # Define known patterns and their scores
        known_patterns = {
            'exp(-': 0.3, 'log(': 0.2, '**2': 0.1,
            'sin(': 0.2, 'cos(': 0.2, 'Attention(': 0.4 # Example for a domain-specific pattern
        }
        for pattern, score in known_patterns.items():
            if pattern in expr_str:
                insight_score += score

        # Score based on complexity (less complex is often more insightful)
        complexity = getattr(expression, 'complexity', len(expr_str) / 10) # Default if no attr
        if complexity < 5: insight_score += 0.2
        elif complexity < 10: insight_score += 0.1

        # Score based on number of free symbols (variables)
        if hasattr(expression, 'symbolic') and hasattr(expression.symbolic, 'free_symbols'):
            free_symbols = expression.symbolic.free_symbols
            if len(free_symbols) >= 2: insight_score += 0.1
            # Interaction term bonus
            if '*' in expr_str and len(free_symbols) >= 2: insight_score += 0.1 # Simple check for interaction

        # Use additional context if provided (e.g., important features from domain knowledge)
        if additional_context:
            important_features = additional_context.get('important_features', [])
            for feature in important_features:
                if str(feature) in expr_str: # Ensure feature is string for search
                    insight_score += 0.1

        return min(1.0, insight_score) # Cap score at 1.0

    def _get_interpretation_target(self, ai_model: AIModelType, features: np.ndarray) -> np.ndarray:
        """
        Extract specific interpretation target from the model.
        This might involve hooks if targeting internal states (e.g., attention).
        """
        ai_model.eval() # Ensure model is in evaluation mode
        extracted_values: List[np.ndarray] = [] # Explicitly list of numpy arrays

        # Example for attention, assuming self.interpretation_target is set in __init__ or elsewhere
        # This target_str logic needs to be robust.
        target_str = getattr(self, 'interpretation_target', 'output') # Default to 'output'

        if target_str.startswith('attention'):
            # This part is highly model-specific. The example assumes a Transformer-like model.
            # It needs access to ai_model's internal structure (e.g., layers, attention heads).
            try:
                parts = target_str.split('_') # e.g., "attention_layer0_head2"
                layer_idx = int(parts[1].replace('layer', ''))
                head_idx = int(parts[2].replace('head', ''))

                # Define the hook function
                def attention_hook(module, input, output):
                    # Output structure depends on the model.
                    # For many HuggingFace Transformers, output[1] is attention_weights.
                    # (batch_size, num_heads, seq_len, seq_len)
                    attn_weights = output[1]
                    # Select specific head, detach, move to CPU, convert to numpy
                    extracted_values.append(attn_weights[:, head_idx, :, :].detach().cpu().numpy())

                # Register the hook: This path to the attention mechanism is an EXAMPLE.
                # It must be adapted to the actual model architecture.
                # e.g., model.transformer.h[layer_idx].attn.register_forward_hook(attention_hook)
                # Or for BERT: model.bert.encoder.layer[layer_idx].attention.self.register_forward_hook(attention_hook)
                # This requires knowing the model structure. For a generic solution, this is hard.
                # We'll assume a path like `ai_model.transformer.h[layer_idx].attn` for now.
                target_module = ai_model.transformer.h[layer_idx].attn
                hook = target_module.register_forward_hook(attention_hook)

                with torch.no_grad():
                    # Input to the model should be a tensor.
                    # Assuming features are token IDs if it's a language model.
                    _ = ai_model(torch.tensor(features, dtype=torch.long))

                hook.remove() # Important to remove hooks after use

                if extracted_values:
                    # Concatenate if multiple batches/segments were processed by the hook
                    result = np.concatenate(extracted_values, axis=0)
                    # Post-process attention: e.g., max attention value per input sample
                    # This aggregation (max over seq_len, seq_len) is an example.
                    return np.max(result, axis=(-1, -2))
                return np.zeros(features.shape[0]) # Return zeros if no values extracted
            except AttributeError as e:
                print(f"AttributeError in attention hook setup: {e}. Model structure might not match expected path.")
                # Fallback to model's final output if hook fails
                with torch.no_grad():
                    # Assuming features are appropriate for direct model input
                    output = ai_model(torch.tensor(features, dtype=torch.long))
                    # Adjust based on actual model output structure
                    return output.detach().cpu().numpy() if hasattr(output, 'detach') else np.array(output)

            except Exception as e:
                print(f"Error during attention extraction: {e}")
                return np.zeros(features.shape[0]) # Fallback
        else:
            # Default to model's final output if not 'attention'
            with torch.no_grad():
                # Assuming features are appropriate for direct model input
                # Input needs to be a tensor
                model_input = torch.tensor(features, dtype=torch.long if features.ndim == 1 or features.dtype == np.int_ else torch.float32)
                output = ai_model(model_input)
                # Ensure output is a numpy array
                if hasattr(output, 'logits'): # Common for HuggingFace classification models
                    output = output.logits
                return output.detach().cpu().numpy()

    def _evaluate_expression_on_data(self, expression: ExpressionType, features: np.ndarray) -> Optional[np.ndarray]:
        """
        Evaluates the symbolic expression on the given data.
        'expression' is assumed to have a 'symbolic' attribute (sympy expression)
        and 'free_symbols'.
        'features' is a NumPy array where columns correspond to variables.
        """
        try:
            if not hasattr(expression, 'symbolic') or not hasattr(expression.symbolic, 'free_symbols'):
                print("Expression does not have 'symbolic' attribute or 'free_symbols'.")
                return None

            # Get the free symbols (variables) from the sympy expression
            symbols = list(expression.symbolic.free_symbols)

            # Create a callable function from the sympy expression
            # Using 'numpy' module for numerical evaluation
            func = sp.lambdify(symbols, expression.symbolic, 'numpy')

            # The 'features' np.ndarray needs to be mapped to these symbols.
            # This assumes that the order of columns in 'features' matches the
            # order of symbols in 'symbols' if expression.input_variables is not available.
            # A more robust way is to have metadata about feature columns.
            # For now, let's assume 'features' has columns in an order that can be passed to func.
            # If `expression` has an `input_variables` attribute (list of symbol names in order), use it.

            # Example: if symbols are [x, y] and features is N x M (M >= 2)
            # We need to select columns corresponding to x and y.
            # This part is tricky without knowing how feature names map to sympy symbols.
            # Simplest assumption: features.shape[1] == len(symbols) and order matches.

            if features.shape[1] < len(symbols):
                print(f"Not enough feature columns ({features.shape[1]}) for symbols ({len(symbols)}).")
                return None

            # Pass each column of features as a separate argument to the lambdified function
            # This means if func expects (x, y), we call func(features[:,0], features[:,1])
            feature_columns = [features[:, i] for i in range(len(symbols))]
            predictions = func(*feature_columns)

            # Ensure predictions are a numpy array
            if not isinstance(predictions, np.ndarray):
                predictions = np.array(predictions)

            # If predictions is scalar (e.g. due to single feature column and single data point), make it 1D array
            if predictions.ndim == 0:
                predictions = predictions.reshape(1)
            # If the function evaluated to a constant for all inputs, it might return a scalar
            # We need to broadcast it to the shape of the input data's first dimension (number of samples)
            if predictions.size == 1 and features.shape[0] > 1:
                 predictions = np.full(features.shape[0], predictions.item())

            return predictions

        except Exception as e:
            print(f"Error evaluating expression on data: {e}")
            return None # Return None or raise error

    def combine_rewards(self, **kwargs: float) -> float:
        """
        Combines individual reward components into a single scalar reward value
        using the weights defined in `self.reward_weights`.
        """
        total_reward = 0.0
        sum_of_weights_used = 0.0

        for component, score in kwargs.items():
            if component in self.reward_weights:
                total_reward += self.reward_weights[component] * score
                sum_of_weights_used += self.reward_weights[component]
            else:
                print(f"Warning: Reward component '{component}' has no defined weight. It will be ignored.")

        # Optional: Normalize by sum of weights used, if weights don't sum to 1 or vary.
        # if sum_of_weights_used > 1e-6: # Avoid division by zero
        #     total_reward /= sum_of_weights_used
        # For now, assume weights are meant as direct multipliers.

        return total_reward

# Example Usage (Illustrative)
if __name__ == "__main__":
    # Dummy Expression class for testing
    class DummyExpression:
        def __init__(self, name: str, complexity: int):
            self.name = name
            self.complexity = complexity
            self.symbolic = f"Symbolic({name})" # Dummy symbolic representation

        def __str__(self) -> str:
            return f"Expression({self.name}, Complexity: {self.complexity})"

    # Dummy AI Model
    class DummyAIModel:
        pass

    # Dummy Test Data
    dummy_data = object() # Placeholder

    # Initialize InterpretabilityReward
    # Using custom weights for this example
    custom_weights = {
        'fidelity': 0.5,
        'simplicity': 0.2,
        'consistency': 0.2,
        'insight': 0.1
    }
    reward_calculator = InterpretabilityReward(reward_weights=custom_weights)

    print(f"Reward Calculator initialized with weights: {reward_calculator.reward_weights}")

    # Create a few dummy expressions
    expr1 = DummyExpression("SimpleExpr", complexity=5)
    expr2 = DummyExpression("ComplexExpr", complexity=50)

    ai_model_instance = DummyAIModel()

    # Calculate rewards
    print(f"\nCalculating reward for {expr1}:")
    reward1 = reward_calculator.calculate_reward(expr1, ai_model_instance, dummy_data)
    print(f"Total reward for {expr1}: {reward1:.4f}")

    print(f"\nCalculating reward for {expr2}:")
    reward2 = reward_calculator.calculate_reward(expr2, ai_model_instance, dummy_data)
    print(f"Total reward for {expr2}: {reward2:.4f}")

    # Test with no expression
    print(f"\nCalculating reward for None expression:")
    reward_none = reward_calculator.calculate_reward(None, ai_model_instance, dummy_data)
    print(f"Total reward for None expression: {reward_none:.4f}")

    # Test simplicity calculation with penalty
    reward_calculator_with_penalty = InterpretabilityReward(
        reward_weights=custom_weights,
        complexity_penalty_factor=0.01,
        max_complexity_for_penalty=20
    )
    print(f"\nReward Calculator with complexity penalty (threshold 20, factor 0.01)")
    simplicity1_pen = reward_calculator_with_penalty._calculate_simplicity(expr1) # Complexity 5 (below threshold)
    simplicity2_pen = reward_calculator_with_penalty._calculate_simplicity(expr2) # Complexity 50 (above threshold)
    print(f"Simplicity score for {expr1} (complexity 5): {simplicity1_pen:.4f}")
    print(f"Simplicity score for {expr2} (complexity 50): {simplicity2_pen:.4f}")

    print("\nInterpretabilityReward example usage finished.")
"""
