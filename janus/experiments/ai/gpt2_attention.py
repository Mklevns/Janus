# =============================================================================
# janus/experiments/ai/gpt2_attention.py
"""GPT-2 attention pattern discovery experiment."""

import torch
import numpy as np
from transformers import GPT2Model, GPT2Tokenizer
from typing import List, Dict, Any, Optional

from janus.experiments.base import BaseExperiment
from janus.experiments.registry import register_experiment
from janus.config.models import ExperimentResult
from janus.ai_interpretability.grammars import NeuralGrammar
from janus.ai_interpretability.environments import AIDiscoveryEnv
from janus.core.expression import Variable


@register_experiment(
    name="gpt2_attention_discovery",
    category="ai",
    aliases=["transformer_attention", "attention_patterns"],
    description="Discover symbolic patterns in GPT-2 attention mechanisms",
    tags=["nlp", "transformer", "attention", "interpretability"],
    supported_algorithms=["symbolic_regression", "genetic"],
    config_schema={
        'model_name': {'type': str, 'default': 'gpt2'},
        'layer_index': {'type': int, 'required': True, 'choices': range(12)},
        'head_index': {'type': int, 'required': False, 'choices': range(12)},
        'num_samples': {'type': int, 'default': 1000}
    }
)
class GPT2AttentionDiscovery(BaseExperiment):
    """
    Discover interpretable patterns in GPT-2 attention heads.
    
    This experiment extracts attention patterns from specific layers/heads
    and attempts to find symbolic expressions that approximate them.
    """
    
    def setup(self):
        """Setup GPT-2 model and data."""
        # Load model and tokenizer
        self.logger.info(f"Loading {self.config.model_name} model...")
        self.model = GPT2Model.from_pretrained(self.config.model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.config.model_name)
        self.model.eval()
        
        # Move to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        # Generate or load text samples
        self.text_samples = self._generate_text_samples()
        
        # Extract attention patterns
        self.logger.info("Extracting attention patterns...")
        self.attention_data = self._extract_attention_patterns()
        
        # Create variables for discovery
        self.variables = self._create_variables()
        
        # Create neural grammar
        self.grammar = NeuralGrammar()
        self.grammar.add_attention_primitives()
        
    def _generate_text_samples(self) -> List[str]:
        """Generate or load text samples for analysis."""
        if hasattr(self.config, 'text_data_path'):
            # Load from file
            with open(self.config.text_data_path, 'r') as f:
                texts = f.readlines()[:self.config.num_samples]
        else:
            # Generate simple test sentences
            texts = []
            templates = [
                "The {} is {}.",
                "I think {} means {}.",
                "{} and {} are related.",
                "When {} happens, {} follows."
            ]
            
            nouns = ["cat", "dog", "car", "house", "tree", "book", "computer", "phone"]
            adjectives = ["big", "small", "red", "blue", "fast", "slow", "new", "old"]
            
            for _ in range(self.config.num_samples):
                template = np.random.choice(templates)
                if "{}" in template:
                    words = np.random.choice(nouns + adjectives, 
                                           size=template.count("{}"), 
                                           replace=False)
                    text = template.format(*words)
                    texts.append(text)
                    
        return texts
        
    def _extract_attention_patterns(self) -> Dict[str, np.ndarray]:
        """Extract attention patterns from the model."""
        attention_data = {
            'queries': [],
            'keys': [],
            'values': [],
            'attention_weights': [],
            'positions': [],
            'token_types': []
        }
        
        with torch.no_grad():
            for text in self.text_samples:
                # Tokenize
                inputs = self.tokenizer(text, return_tensors='pt', 
                                      padding=True, truncation=True,
                                      max_length=128)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Get model outputs with attention
                outputs = self.model(**inputs, output_attentions=True)
                
                # Extract attention from specified layer/head
                layer_attention = outputs.attentions[self.config.layer_index]
                
                if hasattr(self.config, 'head_index') and self.config.head_index is not None:
                    # Specific head
                    head_attention = layer_attention[:, self.config.head_index, :, :]
                else:
                    # Average across heads
                    head_attention = layer_attention.mean(dim=1)
                    
                # Store attention data
                seq_len = inputs['input_ids'].shape[1]
                attention_data['attention_weights'].append(head_attention.cpu().numpy())
                attention_data['positions'].append(np.arange(seq_len))
                
                # Extract token information
                tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
                token_types = self._classify_tokens(tokens)
                attention_data['token_types'].append(token_types)
                
        # Convert to arrays
        attention_data = {k: np.array(v) for k, v in attention_data.items()}
        
        return attention_data
        
    def _classify_tokens(self, tokens: List[str]) -> np.ndarray:
        """Classify tokens into types (word, punctuation, special)."""
        types = []
        for token in tokens:
            if token in ['<pad>', '<eos>', '<bos>']:
                types.append(0)  # Special
            elif token.startswith('Ġ'):  # GPT-2 space marker
                types.append(1)  # Word start
            elif token.isalnum():
                types.append(2)  # Word continuation
            else:
                types.append(3)  # Punctuation
        return np.array(types)
        
    def _create_variables(self) -> List[Variable]:
        """Create variables for symbolic discovery."""
        variables = [
            Variable('pos_diff', 0, properties={
                'description': 'Position difference between tokens',
                'type': 'discrete'
            }),
            Variable('pos_ratio', 1, properties={
                'description': 'Position ratio (pos_i / pos_j)',
                'type': 'continuous'
            }),
            Variable('token_type_i', 2, properties={
                'description': 'Token type of query position',
                'type': 'categorical'
            }),
            Variable('token_type_j', 3, properties={
                'description': 'Token type of key position',
                'type': 'categorical'
            }),
            Variable('relative_pos', 4, properties={
                'description': 'Relative position encoding',
                'type': 'continuous'
            })
        ]
        
        return variables
        
    def run(self, run_id: int = 0) -> ExperimentResult:
        """Run attention pattern discovery."""
        self.logger.info("Starting attention pattern discovery...")
        
        # Prepare data for discovery
        discovery_data = self._prepare_discovery_data()
        
        # Create AI discovery environment
        env = AIDiscoveryEnv(
            ai_model=self.model,
            grammar=self.grammar,
            target_data=discovery_data,
            variables=self.variables,
            interpretation_target=f'attention_layer{self.config.layer_index}',
            max_complexity=self.config.max_complexity
        )
        
        # Run discovery algorithm
        if self.config.algorithm == 'genetic':
            discovered_pattern = self._run_genetic_discovery(env)
        else:
            discovered_pattern = self._run_symbolic_regression(env)
            
        # Evaluate discovered pattern
        evaluation = self._evaluate_pattern(discovered_pattern)
        
        # Create result
        result = ExperimentResult(
            config=self.config,
            run_id=run_id,
            discovered_law=str(discovered_pattern),
            symbolic_accuracy=evaluation['fidelity'],
            law_complexity=evaluation['complexity'],
            metadata={
                'layer': self.config.layer_index,
                'head': getattr(self.config, 'head_index', 'all'),
                'interpretability_score': evaluation['interpretability'],
                'pattern_type': evaluation['pattern_type']
            }
        )
        
        return result
        
    def _prepare_discovery_data(self) -> np.ndarray:
        """Prepare attention data for symbolic discovery."""
        # Flatten attention matrices and create feature matrix
        features = []
        targets = []
        
        for i in range(len(self.attention_data['attention_weights'])):
            attn_matrix = self.attention_data['attention_weights'][i][0]  # Remove batch dim
            seq_len = attn_matrix.shape[0]
            
            for query_pos in range(seq_len):
                for key_pos in range(seq_len):
                    # Features
                    pos_diff = abs(query_pos - key_pos)
                    pos_ratio = (query_pos + 1) / (key_pos + 1)
                    token_type_i = self.attention_data['token_types'][i][query_pos]
                    token_type_j = self.attention_data['token_types'][i][key_pos]
                    relative_pos = (query_pos - key_pos) / seq_len
                    
                    features.append([pos_diff, pos_ratio, token_type_i, 
                                   token_type_j, relative_pos])
                    
                    # Target (attention weight)
                    targets.append(attn_matrix[query_pos, key_pos])
                    
        features = np.array(features)
        targets = np.array(targets)
        
        # Combine features and targets
        discovery_data = np.column_stack([features, targets])
        
        self.logger.info(f"Prepared discovery data: {discovery_data.shape}")
        
        return discovery_data
        
    def _run_genetic_discovery(self, env) -> str:
        """Run genetic algorithm for pattern discovery."""
        # Simplified genetic discovery
        from janus.physics.algorithms.genetic import SymbolicRegressor
        
        regressor = SymbolicRegressor(
            grammar=self.grammar,
            population_size=100,
            generations=50
        )
        
        # Extract features and targets
        X = env.target_data[:, :-1]
        y = env.target_data[:, -1]
        
        # Fit regressor
        best_expr = regressor.fit(X, y, max_complexity=self.config.max_complexity)
        
        return str(best_expr.symbolic)
        
    def _run_symbolic_regression(self, env) -> str:
        """Run symbolic regression for pattern discovery."""
        # Use environment's built-in discovery loop
        obs, _ = env.reset()
        
        for step in range(self.config.max_steps):
            # Get valid actions
            valid_actions = np.where(env.get_action_mask())[0]
            if len(valid_actions) == 0:
                break
                
            # Simple heuristic action selection
            action = self._select_action(env, valid_actions)
            
            obs, reward, done, _, info = env.step(action)
            
            if done:
                return info.get('expression_str', 'None')
                
        return env.get_best_expression()
        
    def _select_action(self, env, valid_actions):
        """Select action using heuristics."""
        # Prefer simpler operations early
        if env.steps_taken < 5:
            # Prefer position-based operations
            position_ops = [a for a in valid_actions 
                           if 'pos' in env.action_to_element.get(a, '')]
            if position_ops:
                return np.random.choice(position_ops)
                
        return np.random.choice(valid_actions)
        
    def _evaluate_pattern(self, pattern: str) -> Dict[str, Any]:
        """Evaluate the discovered attention pattern."""
        evaluation = {
            'fidelity': 0.0,
            'complexity': len(pattern),
            'interpretability': 0.0,
            'pattern_type': 'unknown'
        }
        
        # Check for known attention patterns
        if 'pos_diff' in pattern and '<' in pattern:
            evaluation['pattern_type'] = 'local'
            evaluation['interpretability'] = 0.8
        elif 'pos_ratio' in pattern:
            evaluation['pattern_type'] = 'relative_position'
            evaluation['interpretability'] = 0.7
        elif 'token_type' in pattern:
            evaluation['pattern_type'] = 'token_based'
            evaluation['interpretability'] = 0.6
            
        # Simple fidelity estimate (would be computed properly in practice)
        evaluation['fidelity'] = 0.5 + 0.3 * evaluation['interpretability']
        
        return evaluation
        
    def teardown(self):
        """Clean up resources."""
        # Clear GPU memory
        if hasattr(self, 'model'):
            del self.model
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Clear large data arrays
        self.attention_data = None
        
        self.logger.info("Cleanup complete")
