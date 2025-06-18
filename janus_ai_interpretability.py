# ai_interpretability_extension.py
"""
Janus Extension for AI Interpretability
======================================

Discovers symbolic laws governing how AI models map inputs to outputs.
Can be used to understand neural networks, language models, and other AI systems.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
import sympy as sp
from abc import ABC, abstractmethod

from progressive_grammar_system import ProgressiveGrammar, Expression, Variable
from symbolic_discovery_env import SymbolicDiscoveryEnv
from optimized_candidate_generation import OptimizedCandidateGenerator


@dataclass
class AIBehaviorData:
    """Represents input-output data from an AI model."""
    inputs: np.ndarray  # Shape: (n_samples, input_dim)
    outputs: np.ndarray  # Shape: (n_samples, output_dim)
    intermediate_activations: Optional[Dict[str, np.ndarray]] = None
    attention_weights: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = None


class NeuralGrammar(ProgressiveGrammar):
    """Extended grammar for discovering neural network behaviors."""
    
    def __init__(self):
        super().__init__()
        self._init_neural_primitives()
    
    def _init_neural_primitives(self):
        """Initialize neural network-specific primitives."""
        # Add activation functions
        self.add_primitive('relu', lambda x: sp.Max(0, x))
        self.add_primitive('sigmoid', lambda x: 1 / (1 + sp.exp(-x)))
        self.add_primitive('tanh', sp.tanh)
        self.add_primitive('softplus', lambda x: sp.log(1 + sp.exp(x)))
        
        # Add aggregation operations
        self.add_primitive('mean', lambda *args: sum(args) / len(args))
        self.add_primitive('max_pool', lambda *args: sp.Max(*args))
        self.add_primitive('attention', self._attention_primitive)
        
        # Add threshold operations
        self.add_primitive('threshold', lambda x, t: sp.Piecewise((1, x > t), (0, True)))
        self.add_primitive('step', lambda x: sp.Piecewise((1, x > 0), (0, True)))
        
        # Add modular arithmetic for token operations
        self.add_primitive('mod', lambda x, n: x % n)
        
    def _attention_primitive(self, query, key, value):
        """Simplified attention mechanism."""
        # This is a symbolic representation of attention
        score = query * key
        weight = sp.exp(score) / sp.exp(score)  # Simplified softmax
        return weight * value
    
    def add_embedding_primitives(self, vocab_size: int, embed_dim: int):
        """Add primitives for token embeddings."""
        # Create symbolic embedding lookup
        for i in range(min(vocab_size, 100)):  # Limit for tractability
            self.add_primitive(f'embed_{i}', lambda idx, i=i: sp.Piecewise(
                (sp.Symbol(f'e_{i}'), idx == i), (0, True)
            ))


class AIInterpretabilityEnv(SymbolicDiscoveryEnv):
    """Environment for discovering AI behavior laws."""
    
    def __init__(self, 
                 ai_model: nn.Module,
                 grammar: NeuralGrammar,
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
        
        # Initialize parent environment
        super().__init__(
            grammar=grammar,
            target_data=behavior_data.outputs,
            variables=variables,
            **kwargs
        )
        
        # Add AI-specific reward components
        self.fidelity_weight = kwargs.get('fidelity_weight', 0.5)
        self.simplicity_weight = kwargs.get('simplicity_weight', 0.3)
        self.coverage_weight = kwargs.get('coverage_weight', 0.2)
    
    def _extract_variables_from_model(self) -> List[Variable]:
        """Extract relevant variables from AI model structure."""
        variables = []
        
        # Input features
        input_dim = self.behavior_data.inputs.shape[1]
        for i in range(input_dim):
            var = Variable(
                name=f"x_{i}",
                index=i,
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
                    var = Variable(
                        name=f"{layer_name}_n{neuron_idx}",
                        index=len(variables),
                        properties={
                            "type": "activation",
                            "layer": layer_name,
                            "neuron": neuron_idx
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
        neuron_variance = np.var(activations, axis=0)
        important_indices = np.argsort(neuron_variance)[-10:]  # Top 10
        return important_indices.tolist()
    
    def _calculate_reward(self, expression: Expression) -> float:
        """Calculate reward for AI interpretation task."""
        # Base MSE reward
        base_reward = super()._calculate_reward(expression)
        
        # Fidelity: How well does the expression match AI behavior?
        fidelity = self._calculate_fidelity(expression)
        
        # Simplicity: Prefer simpler explanations (Occam's Razor)
        simplicity = 1.0 / (1.0 + expression.complexity)
        
        # Coverage: What fraction of behaviors does this explain?
        coverage = self._calculate_coverage(expression)
        
        total_reward = (
            self.fidelity_weight * fidelity +
            self.simplicity_weight * simplicity +
            self.coverage_weight * coverage +
            (1 - self.fidelity_weight - self.simplicity_weight - self.coverage_weight) * base_reward
        )
        
        return total_reward
    
    def _calculate_fidelity(self, expression: Expression) -> float:
        """Measure how faithfully the expression reproduces AI behavior."""
        try:
            # Evaluate expression on input data
            predicted = self._evaluate_expression_on_data(expression)
            actual = self.behavior_data.outputs
            
            # Calculate correlation or agreement
            if actual.ndim == 1:
                correlation = np.corrcoef(predicted.flatten(), actual.flatten())[0, 1]
                return max(0, correlation)
            else:
                # For multi-output, use mean correlation
                correlations = []
                for i in range(actual.shape[1]):
                    corr = np.corrcoef(predicted[:, i], actual[:, i])[0, 1]
                    correlations.append(max(0, corr))
                return np.mean(correlations)
                
        except Exception:
            return 0.0
    
    def _calculate_coverage(self, expression: Expression) -> float:
        """Measure what fraction of the input space this expression covers."""
        # Simple coverage: fraction of samples where expression is well-defined
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
                 grammar: NeuralGrammar,
                 behavior_data: AIBehaviorData,
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
        
        # Generate local perturbations
        local_data = self._generate_local_data(anchor_input, neighborhood_size)
        
        super().__init__(
            ai_model=ai_model,
            grammar=grammar,
            behavior_data=local_data,
            interpretation_mode='local',
            **kwargs
        )
    
    def _generate_local_data(self, anchor: np.ndarray, 
                           neighborhood_size: float) -> AIBehaviorData:
        """Generate perturbed inputs around anchor point."""
        n_samples = 1000
        dim = anchor.shape[0]
        
        # Generate perturbations
        perturbations = np.random.normal(0, neighborhood_size, (n_samples, dim))
        inputs = anchor + perturbations
        
        # Get model outputs
        self.ai_model.eval()
        with torch.no_grad():
            inputs_tensor = torch.FloatTensor(inputs)
            outputs_tensor = self.ai_model(inputs_tensor)
            outputs = outputs_tensor.numpy()
        
        return AIBehaviorData(inputs=inputs, outputs=outputs)


class TransformerInterpretabilityEnv(AIInterpretabilityEnv):
    """Specialized environment for interpreting transformer models."""
    
    def __init__(self,
                 transformer_model: nn.Module,
                 tokenizer: Any,
                 grammar: NeuralGrammar,
                 text_samples: List[str],
                 **kwargs):
        """Initialize environment for transformer interpretation."""
        self.tokenizer = tokenizer
        self.text_samples = text_samples
        
        # Process text samples to create behavior data
        behavior_data = self._process_text_samples(transformer_model, text_samples)
        
        # Add attention-specific primitives
        grammar.add_primitive('self_attention', self._self_attention_primitive)
        grammar.add_primitive('position_encoding', self._position_encoding_primitive)
        
        super().__init__(
            ai_model=transformer_model,
            grammar=grammar,
            behavior_data=behavior_data,
            **kwargs
        )
    
    def _process_text_samples(self, model: nn.Module, 
                            texts: List[str]) -> AIBehaviorData:
        """Process text samples through transformer."""
        all_inputs = []
        all_outputs = []
        all_attentions = []
        
        for text in texts:
            # Tokenize
            tokens = self.tokenizer(text, return_tensors='pt')
            
            # Get model outputs with attention
            outputs = model(**tokens, output_attentions=True)
            
            # Extract relevant data
            all_inputs.append(tokens['input_ids'].numpy())
            all_outputs.append(outputs.logits.numpy())
            
            # Average attention across heads and layers
            attention = torch.stack(outputs.attentions).mean(dim=(0, 1))
            all_attentions.append(attention.numpy())
        
        return AIBehaviorData(
            inputs=np.concatenate(all_inputs),
            outputs=np.concatenate(all_outputs),
            attention_weights=np.concatenate(all_attentions)
        )
    
    def _self_attention_primitive(self, tokens, positions):
        """Symbolic representation of self-attention."""
        # Simplified symbolic attention
        return sp.Symbol('attn') * tokens * sp.exp(-abs(positions))
    
    def _position_encoding_primitive(self, position):
        """Symbolic position encoding."""
        return sp.sin(position / 10000)


class AILawDiscovery:
    """Main interface for discovering laws in AI systems."""
    
    def __init__(self, ai_model: nn.Module, model_type: str = 'neural_network'):
        self.ai_model = ai_model
        self.model_type = model_type
        self.grammar = NeuralGrammar()
        self.discovered_laws = []
    
    def discover_global_laws(self, 
                           input_data: np.ndarray,
                           max_complexity: int = 10,
                           n_epochs: int = 100) -> List[Expression]:
        """Discover global laws governing AI behavior."""
        # Get model outputs
        behavior_data = self._collect_behavior_data(input_data)
        
        # Create environment
        env = AIInterpretabilityEnv(
            ai_model=self.ai_model,
            grammar=self.grammar,
            behavior_data=behavior_data,
            interpretation_mode='global',
            max_complexity=max_complexity
        )
        
        # Train policy to discover laws
        from hypothesis_policy_network import HypothesisNet, PPOTrainer
        
        policy = HypothesisNet(
            obs_dim=env.observation_space.shape[0],
            act_dim=env.action_space.n,
            grammar=self.grammar
        )
        
        trainer = PPOTrainer(policy, env)
        trainer.train(total_timesteps=n_epochs * 1000)
        
        # Extract discovered laws
        laws = self._extract_laws_from_env(env)
        self.discovered_laws.extend(laws)
        
        return laws
    
    def discover_neuron_roles(self, 
                            layer_name: str,
                            input_data: np.ndarray) -> Dict[int, Expression]:
        """Discover symbolic roles of individual neurons."""
        # Hook to extract activations
        activations = {}
        
        def hook_fn(module, input, output):
            activations[layer_name] = output.detach().numpy()
        
        # Register hook
        for name, module in self.ai_model.named_modules():
            if name == layer_name:
                handle = module.register_forward_hook(hook_fn)
                break
        
        # Collect activations
        self.ai_model.eval()
        with torch.no_grad():
            _ = self.ai_model(torch.FloatTensor(input_data))
        
        handle.remove()
        
        # Discover role for each neuron
        neuron_roles = {}
        layer_activations = activations[layer_name]
        
        for neuron_idx in range(layer_activations.shape[1]):
            # Create specialized environment for this neuron
            neuron_data = AIBehaviorData(
                inputs=input_data,
                outputs=layer_activations[:, neuron_idx:neuron_idx+1]
            )
            
            env = AIInterpretabilityEnv(
                ai_model=self.ai_model,
                grammar=self.grammar,
                behavior_data=neuron_data,
                max_complexity=5  # Keep neuron explanations simple
            )
            
            # Quick discovery
            laws = self._quick_discovery(env, n_steps=1000)
            if laws:
                neuron_roles[neuron_idx] = laws[0]
        
        return neuron_roles
    
    def explain_decision(self, 
                        input_sample: np.ndarray,
                        neighborhood_size: float = 0.1) -> Expression:
        """Explain AI's decision for a specific input."""
        env = LocalInterpretabilityEnv(
            ai_model=self.ai_model,
            grammar=self.grammar,
            behavior_data=None,  # Will be generated
            anchor_input=input_sample,
            neighborhood_size=neighborhood_size
        )
        
        # Quick local discovery
        laws = self._quick_discovery(env, n_steps=5000)
        
        return laws[0] if laws else None
    
    def _collect_behavior_data(self, input_data: np.ndarray) -> AIBehaviorData:
        """Collect input-output behavior from AI model."""
        self.ai_model.eval()
        
        with torch.no_grad():
            inputs_tensor = torch.FloatTensor(input_data)
            outputs_tensor = self.ai_model(inputs_tensor)
            
            # Handle different output types
            if isinstance(outputs_tensor, tuple):
                outputs = outputs_tensor[0].numpy()
            else:
                outputs = outputs_tensor.numpy()
        
        return AIBehaviorData(inputs=input_data, outputs=outputs)
    
    def _quick_discovery(self, env: AIInterpretabilityEnv, 
                        n_steps: int = 1000) -> List[Expression]:
        """Quick discovery using random search for simple cases."""
        best_expressions = []
        best_rewards = []
        
        for _ in range(n_steps):
            env.reset()
            done = False
            
            while not done:
                action = env.action_space.sample()
                _, reward, done, _, info = env.step(action)
            
            if 'expression_obj' in info:
                expr = info['expression_obj']
                
                # Keep top expressions
                if len(best_expressions) < 10:
                    best_expressions.append(expr)
                    best_rewards.append(reward)
                elif reward > min(best_rewards):
                    min_idx = best_rewards.index(min(best_rewards))
                    best_expressions[min_idx] = expr
                    best_rewards[min_idx] = reward
        
        # Sort by reward
        sorted_exprs = [expr for _, expr in sorted(
            zip(best_rewards, best_expressions), reverse=True
        )]
        
        return sorted_exprs
    
    def _extract_laws_from_env(self, env: AIInterpretabilityEnv) -> List[Expression]:
        """Extract discovered laws from environment history."""
        # This would extract from training history
        # For now, return top expressions from final state
        return self._quick_discovery(env, n_steps=100)


# Example usage
if __name__ == "__main__":
    # Example 1: Interpret a simple neural network
    class SimpleNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(2, 10)
            self.fc2 = nn.Linear(10, 1)
            
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            return self.fc2(x)
    
    # Create and train a simple model
    model = SimpleNN()
    
    # Generate some data where output = x1^2 + x2
    X = np.random.randn(1000, 2)
    y = X[:, 0]**2 + X[:, 1]
    
    # Discover the law
    discoverer = AILawDiscovery(model)
    laws = discoverer.discover_global_laws(X, max_complexity=5)
    
    print("Discovered laws:")
    for law in laws[:3]:
        print(f"  {law.symbolic}")
    
    # Example 2: Explain a specific decision
    test_input = np.array([1.0, 2.0])
    explanation = discoverer.explain_decision(test_input)
    print(f"\nLocal explanation at {test_input}: {explanation.symbolic}")
    
    # Example 3: Discover neuron roles
    neuron_roles = discoverer.discover_neuron_roles('fc1', X)
    print("\nNeuron roles in fc1:")
    for neuron_id, role in list(neuron_roles.items())[:3]:
        print(f"  Neuron {neuron_id}: {role.symbolic}")
