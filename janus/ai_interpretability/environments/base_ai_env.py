"""
AI Discovery Environment
========================

An RL environment specifically for discovering interpretable symbolic
representations of AI model components or behaviors.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
from collections import defaultdict, deque
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import sympy as sp

from .base_symbolic_env import SymbolicDiscoveryEnv, ObsType, InfoType
from janus.core.expression import Variable, Expression

# Type definitions
AIModelType = Union[nn.Module, Any]  # PyTorch model or generic


class AIDiscoveryEnv(SymbolicDiscoveryEnv):
    def __init__(self,
                 ai_model: AIModelType,
                 data_type: str = 'tabular',
                 test_size: float = 0.2,
                 interpretation_target: str = 'output',  # 'output', 'layer_X', 'attention', etc.
                 hook_layers: Optional[List[str]] = None,
                 **kwargs: Any):
        """
        Initializes the AI Discovery Environment.

        Args:
            ai_model: The AI model to be interpreted.
            data_type: The type of data the AI model processes ('tabular', 'image', 'text', 'sequence').
            test_size: Fraction of data to use for generalization testing.
            interpretation_target: What aspect of the model to interpret.
            hook_layers: Specific layers to hook into for internal state capture.
            **kwargs: Additional arguments for the parent SymbolicDiscoveryEnv.
        """
        # Split data for generalization testing if target_data provided
        if 'target_data' in kwargs and kwargs['target_data'] is not None:
            # Store full data
            self.full_target_data = kwargs['target_data'].copy()
            
            # Split into train/test
            n_samples = len(self.full_target_data)
            indices = np.arange(n_samples)
            train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=42)
            
            self.train_indices = train_idx
            self.test_indices = test_idx
            
            # Use training data for discovery
            kwargs['target_data'] = self.full_target_data[train_idx]
        
        super().__init__(**kwargs)
        
        self.ai_model = ai_model
        self.data_type = data_type
        self.interpretation_target = interpretation_target
        self.hook_layers = hook_layers or []
        
        # Storage for captured internal states
        self.captured_activations = defaultdict(list)
        self.hook_handles = []
        
        # Register hooks into the AI model
        self._register_hooks()
        
        # AI-specific reward components
        self.ai_reward_components = {
            'fidelity': 0.4,      # How well the symbolic expression matches AI behavior
            'simplicity': 0.3,    # Occam's razor: prefer simpler expressions
            'generalization': 0.2, # How well the expression works on unseen data
            'interpretability': 0.1 # Human-understandability score
        }
        
        # Cache for model predictions to avoid recomputation
        self._prediction_cache = {}
        
        # Known interpretable patterns for bonus rewards
        self.interpretable_patterns = self._init_interpretable_patterns()

    def _init_interpretable_patterns(self) -> Dict[str, Callable]:
        """Initialize patterns that are known to be interpretable."""
        patterns = {
            'linear': lambda expr: 'Add' in str(expr.func) and all('Mul' in str(arg.func) or arg.is_number for arg in expr.args if hasattr(arg, 'func')),
            'polynomial': lambda expr: 'Pow' in str(expr.func) or any('Pow' in str(arg) for arg in expr.args if hasattr(expr, 'args')),
            'threshold': lambda expr: 'Piecewise' in str(expr.func) or 'Max' in str(expr.func) or 'Min' in str(expr.func),
            'periodic': lambda expr: any(func in str(expr) for func in ['sin', 'cos', 'tan']),
            'exponential': lambda expr: 'exp' in str(expr) or 'log' in str(expr),
        }
        return patterns

    def _register_hooks(self) -> None:
        """
        Registers hooks into the AI model to capture internal activations.
        Supports PyTorch models primarily, with fallback for other frameworks.
        """
        if hasattr(self.ai_model, 'named_modules') and callable(self.ai_model.named_modules):
            # PyTorch model
            self._register_pytorch_hooks()
        elif hasattr(self.ai_model, 'layers'):
            # Keras/TensorFlow model
            self._register_tf_hooks()
        else:
            # Generic model - try to extract what we can
            warnings.warn(f"Model type {type(self.ai_model)} not fully supported for hooking. "
                         f"Will use output-only interpretation.")

    def _register_pytorch_hooks(self) -> None:
        """Register hooks for PyTorch models."""
        def get_activation_hook(name: str):
            def hook(module, input_val, output_val):
                # Store activations
                if isinstance(output_val, torch.Tensor):
                    self.captured_activations[name].append(output_val.detach().cpu().numpy())
                elif isinstance(output_val, tuple):
                    # Handle modules that return tuples
                    for i, out in enumerate(output_val):
                        if isinstance(out, torch.Tensor):
                            self.captured_activations[f"{name}_{i}"].append(out.detach().cpu().numpy())
            return hook
        
        # Register hooks on specified layers or all layers
        for name, module in self.ai_model.named_modules():
            if not self.hook_layers or name in self.hook_layers:
                # Skip container modules
                if not list(module.children()):
                    handle = module.register_forward_hook(get_activation_hook(name))
                    self.hook_handles.append(handle)

    def _register_tf_hooks(self) -> None:
        """Register hooks for TensorFlow/Keras models."""
        # TensorFlow doesn't have hooks like PyTorch, but we can create intermediate models
        try:
            import tensorflow as tf
            
            # Create models that output intermediate layers
            for layer in self.ai_model.layers:
                if not self.hook_layers or layer.name in self.hook_layers:
                    intermediate_model = tf.keras.Model(
                        inputs=self.ai_model.input,
                        outputs=layer.output
                    )
                    # Store the intermediate model
                    self.captured_activations[layer.name] = intermediate_model
        except ImportError:
            warnings.warn("TensorFlow not available for hook registration")

    def _get_model_output(self, inputs: np.ndarray) -> np.ndarray:
        """Get model output with caching."""
        input_hash = hash(inputs.tobytes())
        
        if input_hash in self._prediction_cache:
            return self._prediction_cache[input_hash]
        
        # Clear previous captures
        for key in self.captured_activations:
            if isinstance(self.captured_activations[key], list):
                self.captured_activations[key].clear()
        
        # Get model predictions
        if hasattr(self.ai_model, 'predict'):
            # Sklearn-style or Keras model
            output = self.ai_model.predict(inputs)
        elif hasattr(self.ai_model, 'forward'):
            # PyTorch model
            self.ai_model.eval()
            with torch.no_grad():
                input_tensor = torch.FloatTensor(inputs)
                output = self.ai_model(input_tensor).numpy()
        else:
            # Generic callable
            output = self.ai_model(inputs)
        
        self._prediction_cache[input_hash] = output
        return output

    def _evaluate_expression(self) -> float:
        """
        Evaluates the constructed symbolic expression against AI model behavior
        and calculates a reward based on AI-specific components.
        """
        # Get the expression
        final_expr_obj = self.current_state.root.to_expression(self.grammar)
        if final_expr_obj is None:
            return self.reward_config.get('timeout_penalty', -1.0)
        
        # Calculate individual reward components
        fidelity_score = self._calculate_fidelity_reward(final_expr_obj)
        simplicity_score = self._calculate_simplicity_reward(final_expr_obj)
        generalization_score = self._calculate_generalization_reward(final_expr_obj)
        interpretability_score = self._calculate_interpretability_reward(final_expr_obj)
        
        # Combine rewards
        reward = (
            self.ai_reward_components['fidelity'] * fidelity_score +
            self.ai_reward_components['simplicity'] * simplicity_score +
            self.ai_reward_components['generalization'] * generalization_score +
            self.ai_reward_components['interpretability'] * interpretability_score
        )
        
        # Update evaluation cache
        self._evaluation_cache.clear()
        self._evaluation_cache.update({
            'expression_str': str(final_expr_obj.symbolic) if final_expr_obj and final_expr_obj.symbolic else "N/A",
            'expression_obj': final_expr_obj,
            'ai_fidelity': fidelity_score,
            'ai_simplicity': simplicity_score,
            'ai_generalization': generalization_score,
            'ai_interpretability': interpretability_score,
            'reward': reward,
            'complexity': final_expr_obj.complexity if final_expr_obj else 0
        })
        
        return reward

    def _calculate_fidelity_reward(self, expr: Expression) -> float:
        """Calculate how well the expression matches the AI model's behavior."""
        try:
            # Get training data
            if hasattr(self, 'train_indices'):
                train_data = self.full_target_data[self.train_indices]
            else:
                train_data = self.target_data
            
            # Extract features and target
            features = train_data[:, :-1]
            
            # Get model predictions or internal states based on interpretation target
            if self.interpretation_target == 'output':
                ai_predictions = self._get_model_output(features)
                if ai_predictions.ndim > 1:
                    ai_predictions = ai_predictions.flatten()
            else:
                # Interpret specific layer or component
                ai_predictions = self._get_interpretation_target(features)
            
            # Evaluate expression on the same inputs
            expr_predictions = self._evaluate_expression_on_data(expr, features)
            
            # Calculate fidelity metrics
            if expr_predictions is not None and len(expr_predictions) == len(ai_predictions):
                # Normalize both to same scale
                ai_std = np.std(ai_predictions)
                if ai_std > 0:
                    ai_norm = (ai_predictions - np.mean(ai_predictions)) / ai_std
                    expr_norm = (expr_predictions - np.mean(expr_predictions)) / np.std(expr_predictions + 1e-8)
                    
                    # Calculate correlation
                    correlation = np.corrcoef(ai_norm, expr_norm)[0, 1]
                    
                    # Calculate normalized MSE
                    mse = mean_squared_error(ai_norm, expr_norm)
                    nmse = 1.0 / (1.0 + mse)  # Normalized to [0, 1]
                    
                    # Combine metrics
                    fidelity = 0.6 * max(0, correlation) + 0.4 * nmse
                else:
                    fidelity = 0.0
            else:
                fidelity = 0.0
                
        except Exception as e:
            warnings.warn(f"Fidelity calculation failed: {e}")
            fidelity = 0.0
        
        return fidelity

    def _calculate_simplicity_reward(self, expr: Expression) -> float:
        """Calculate simplicity score based on expression complexity."""
        # Use expression complexity with diminishing returns
        complexity = expr.complexity if expr else float('inf')
        
        # Different scoring based on data type
        if self.data_type == 'tabular':
            # Tabular data can handle moderate complexity
            simplicity = 1.0 / (1.0 + 0.1 * complexity)
        elif self.data_type in ['image', 'sequence']:
            # Complex data might need more complex expressions
            simplicity = 1.0 / (1.0 + 0.05 * complexity)
        else:
            simplicity = 1.0 / (1.0 + 0.1 * complexity)
        
        # Bonus for very simple expressions
        if complexity <= 3:
            simplicity *= 1.2
        
        return min(1.0, simplicity)

    def _calculate_generalization_reward(self, expr: Expression) -> float:
        """Calculate how well the expression generalizes to unseen data."""
        if not hasattr(self, 'test_indices') or len(self.test_indices) == 0:
            # No test set available
            return 0.5  # Neutral score
        
        try:
            # Get test data
            test_data = self.full_target_data[self.test_indices]
            test_features = test_data[:, :-1]
            
            # Get model predictions on test set
            if self.interpretation_target == 'output':
                test_ai_predictions = self._get_model_output(test_features)
                if test_ai_predictions.ndim > 1:
                    test_ai_predictions = test_ai_predictions.flatten()
            else:
                test_ai_predictions = self._get_interpretation_target(test_features)
            
            # Evaluate expression on test set
            test_expr_predictions = self._evaluate_expression_on_data(expr, test_features)
            
            if test_expr_predictions is not None and len(test_expr_predictions) == len(test_ai_predictions):
                # Calculate test set fidelity
                test_correlation = np.corrcoef(test_ai_predictions, test_expr_predictions)[0, 1]
                test_r2 = r2_score(test_ai_predictions, test_expr_predictions)
                
                # Generalization score
                generalization = 0.6 * max(0, test_correlation) + 0.4 * max(0, test_r2)
            else:
                generalization = 0.0
                
        except Exception as e:
            warnings.warn(f"Generalization calculation failed: {e}")
            generalization = 0.0
        
        return generalization

    def _calculate_interpretability_reward(self, expr: Expression) -> float:
        """Calculate interpretability score based on known patterns."""
        if not expr or not expr.symbolic:
            return 0.0
        
        interpretability = 0.0
        expr_str = str(expr.symbolic)
        
        # Check for known interpretable patterns
        pattern_scores = {
            'linear': 0.9,      # Very interpretable
            'polynomial': 0.7,   # Quite interpretable
            'threshold': 0.8,    # Interpretable (decision boundaries)
            'periodic': 0.6,     # Moderately interpretable
            'exponential': 0.5,  # Less interpretable
        }
        
        for pattern_name, check_func in self.interpretable_patterns.items():
            try:
                if check_func(expr.symbolic):
                    interpretability = max(interpretability, pattern_scores.get(pattern_name, 0.5))
            except:
                pass
        
        # Penalty for very long expressions
        if len(expr_str) > 100:
            interpretability *= 0.8
        
        # Bonus for expressions with clear variable names
        if hasattr(self, 'variables'):
            var_names = [v.name for v in self.variables]
            if any(name in expr_str for name in var_names if len(name) > 3):
                interpretability *= 1.1
        
        return min(1.0, interpretability)

    def _evaluate_expression_on_data(self, expr: Expression, data: np.ndarray) -> Optional[np.ndarray]:
        """Evaluate symbolic expression on data."""
        if not expr or not expr.symbolic:
            return None
        
        try:
            predictions = []
            
            # Create variable mapping
            var_mapping = {}
            for i, var in enumerate(self.variables):
                var_mapping[var.symbolic] = data[:, i]
            
            # Vectorized evaluation if possible
            if all(isinstance(v, np.ndarray) for v in var_mapping.values()):
                # Try to lambdify for faster evaluation
                try:
                    func = sp.lambdify(list(var_mapping.keys()), expr.symbolic, 'numpy')
                    result = func(*var_mapping.values())
                    return np.array(result).flatten()
                except:
                    pass
            
            # Fall back to row-by-row evaluation
            for row in data:
                var_values = {self.variables[i].symbolic: row[i] for i in range(len(self.variables))}
                try:
                    pred = float(expr.symbolic.subs(var_values))
                    predictions.append(pred)
                except:
                    predictions.append(0.0)
            
            return np.array(predictions)
            
        except Exception as e:
            warnings.warn(f"Expression evaluation failed: {e}")
            return None

    def _get_interpretation_target(self, features: np.ndarray) -> np.ndarray:
        """Get the specific interpretation target (layer activation, attention, etc.)."""
        # This would extract the specific component we're trying to interpret
        # For now, default to model output
        return self._get_model_output(features).flatten()

    def close(self) -> None:
        """Clean up resources."""
        # Remove PyTorch hooks
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()
        
        # Clear caches
        self._prediction_cache.clear()
        self.captured_activations.clear()
        
        # Call parent cleanup
        super().close()


# Enhanced Example Usage
if __name__ == "__main__":
    from janus.core.grammar import AIGrammar
    from janus.core.expression import Variable
    import torch.nn.functional as F
    
    # 1. Create a more realistic AI Model (PyTorch)
    class InterpretableNN(nn.Module):
        def __init__(self, input_dim=3, hidden_dim=10, output_dim=1):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, output_dim)
            
        def forward(self, x):
            # Interpretable: ReLU(W1*x + b1) followed by linear
            hidden = F.relu(self.fc1(x))
            output = self.fc2(hidden)
            return output
    
    # Initialize model with known weights for testing
    model = InterpretableNN()
    with torch.no_grad():
        # Set weights to create a known function: approximately x1^2 + 0.5*x2 - 0.3*x3
        model.fc1.weight.data = torch.tensor([
            [1.0, 0.0, 0.0],   # Hidden unit 1: x1
            [0.0, 0.5, 0.0],   # Hidden unit 2: 0.5*x2
            [0.0, 0.0, -0.3],  # Hidden unit 3: -0.3*x3
            [1.0, 0.0, 0.0],   # Hidden unit 4: x1 (for squaring)
        ] + [[0.0, 0.0, 0.0]] * 6)  # Rest zeros
        model.fc1.bias.data = torch.zeros(10)
        
        # Output weights to approximate squaring and linear combination
        model.fc2.weight.data = torch.tensor([[1.0, 1.0, 1.0, 1.0] + [0.0] * 6])
        model.fc2.bias.data = torch.zeros(1)
    
    # 2. Initialize Grammar
    ai_grammar = AIGrammar()
    
    # 3. Create variables
    variables = [
        Variable(name="x1", index=0, properties={"range": (-2, 2)}),
        Variable(name="x2", index=1, properties={"range": (-2, 2)}),
        Variable(name="x3", index=2, properties={"range": (-2, 2)}),
    ]
    
    # 4. Generate synthetic data
    np.random.seed(42)
    n_samples = 200
    X = np.random.uniform(-2, 2, (n_samples, 3))
    
    # Get model outputs
    model.eval()
    with torch.no_grad():
        y_model = model(torch.FloatTensor(X)).numpy().flatten()
    
    # Create target data (features + target)
    target_data = np.column_stack([X, y_model])
    
    # 5. Initialize AIDiscoveryEnv
    env = AIDiscoveryEnv(
        ai_model=model,
        data_type='tabular',
        grammar=ai_grammar,
        target_data=target_data,
        variables=variables,
        max_depth=5,
        max_complexity=15,
        test_size=0.3,  # 30% for generalization testing
        interpretation_target='output',
        hook_layers=['fc1', 'fc2']  # Hook into both layers
    )
    
    print("=" * 70)
    print("AI DISCOVERY ENVIRONMENT - COMPLETE IMPLEMENTATION")
    print("=" * 70)
    print(f"Model: {type(model).__name__}")
    print(f"Data shape: {X.shape}")
    print(f"Action space: {env.action_space.n} actions")
    print(f"Observation space: {env.observation_space.shape}")
    print(f"AI Reward weights: {env.ai_reward_components}")
    
    # 6. Run a discovery episode
    print("\n" + "=" * 50)
    print("RUNNING DISCOVERY EPISODE")
    print("=" * 50)
    
    obs, info = env.reset()
    print(f"Initial state: {info}")
    
    total_reward = 0
    step_count = 0
    
    while step_count < 20:  # Limit steps
        # Get valid actions
        action_mask = env.get_action_mask()
        valid_actions = np.where(action_mask)[0]
        
        if len(valid_actions) == 0:
            print("No valid actions available.")
            break
        
        # Take action (random for demo, would be policy in practice)
        action = np.random.choice(valid_actions)
        action_element = env.action_to_element[action]
        
        # Step
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1
        
        print(f"\nStep {step_count}: Action = {action_element}")
        print(f"  Reward = {reward:.4f}, Total = {total_reward:.4f}")
        
        if terminated or truncated:
            print("\n" + "=" * 50)
            print("EPISODE COMPLETE")
            print("=" * 50)
            
            if 'expression_str' in info:
                print(f"Discovered Expression: {info['expression_str']}")
                print(f"Expression Complexity: {info.get('complexity', 'N/A')}")
                print(f"\nReward Components:")
                print(f"  Fidelity: {info.get('ai_fidelity', 0):.3f}")
                print(f"  Simplicity: {info.get('ai_simplicity', 0):.3f}")
                print(f"  Generalization: {info.get('ai_generalization', 0):.3f}")
                print(f"  Interpretability: {info.get('ai_interpretability', 0):.3f}")
                print(f"  Final Reward: {info.get('reward', 0):.3f}")
            
            break
    
    # 7. Analyze captured activations
    if env.captured_activations:
        print("\n" + "=" * 50)
        print("CAPTURED INTERNAL STATES")
        print("=" * 50)
        for layer_name, activations in env.captured_activations.items():
            if isinstance(activations, list) and activations:
                print(f"{layer_name}: {len(activations)} captures")
                if isinstance(activations[0], np.ndarray):
                    print(f"  Shape: {activations[0].shape}")
    
    # Clean up
    env.close()
    print("\nEnvironment closed successfully.")
