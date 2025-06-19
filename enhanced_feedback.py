"""
Enhanced Feedback Loop System for Janus
=======================================

Implements tighter integration between components for more effective discovery.
"""

import numpy as np
import sympy
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Deque
from collections import deque
from dataclasses import dataclass
import time

from janus.ai_interpretability.environments import SymbolicDiscoveryEnv, ExpressionNode
from hypothesis_policy_network import HypothesisNet, PPOTrainer
from progressive_grammar_system import Expression, Variable # Added Variable
from conservation_reward_fix import ConservationBiasedReward as NewConservationBiasedReward


class IntrinsicRewardCalculator:
    """Calculate intrinsic rewards based on novelty and discovery value."""
    
    def __init__(self, 
                 novelty_weight: float = 0.3,
                 diversity_weight: float = 0.2,
                 complexity_growth_weight: float = 0.1,
                 conservation_weight: float = 0.4): # Add new weight
        
        self.novelty_weight = novelty_weight
        self.diversity_weight = diversity_weight
        self.complexity_growth_weight = complexity_growth_weight
        self.conservation_weight = conservation_weight # Initialize new weight
        self.conservation_calculator = NewConservationBiasedReward(conservation_types=['energy', 'momentum', 'mass'], weight_factor=self.conservation_weight) # Instantiate new module
        
        # History tracking
        self.expression_history: Deque[str] = deque(maxlen=1000)
        self.complexity_history: Deque[int] = deque(maxlen=100)
        self.discovery_embeddings: Deque[np.ndarray] = deque(maxlen=500)
        
        # Novelty detector (simplified version)
        self.expression_cache = {}
        
    def calculate_intrinsic_reward(self,
                                 expression: str,
                                 complexity: int,
                                 extrinsic_reward: float,
                                 embedding: Optional[np.ndarray], # Keep this, but ensure it's Optional if it can be None
                                 data: np.ndarray, # Add data
                                 variables: List[Any] # Add variables
                                 ) -> float:
        """Calculate combined intrinsic and extrinsic reward."""
        
        # 1. Novelty reward
        novelty_reward = self._calculate_novelty_reward(expression)
        
        # 2. Diversity reward
        diversity_reward = self._calculate_diversity_reward(expression, embedding)
        
        # 3. Complexity growth reward
        complexity_reward = self._calculate_complexity_growth_reward(complexity)

        # 4. Conservation Bonus
        evaluated_values = self.evaluate_expression_on_data(expression, data, variables)

        predicted_traj = {}
        if evaluated_values is not None and not np.all(np.isnan(evaluated_values)): # Check if evaluation was successful and not all NaN
            for c_type in self.conservation_calculator.conservation_types:
                predicted_traj[f'conserved_{c_type}'] = evaluated_values
        else: # if evaluate_expression_on_data returned None or all NaNs (e.g. parse error or universal eval error)
            # Populate with None to ensure compute_conservation_bonus handles it gracefully
            for c_type in self.conservation_calculator.conservation_types:
                predicted_traj[f'conserved_{c_type}'] = None

        ground_truth_traj = {}
        for c_type in self.conservation_calculator.conservation_types:
            gt_column_index = None
            # Attempt 1: Exact match or common variations
            potential_names = [
                c_type.lower(),
                f"gt_{c_type.lower()}",
                f"{c_type.lower()}_gt",
                f"true_{c_type.lower()}",
                c_type.upper(),
                c_type # Original case
            ]
            if c_type.lower() == 'energy':
                potential_names.extend(['e', 'e_total', 'hamiltonian'])
            elif c_type.lower() == 'momentum':
                potential_names.extend(['p', 'mom'])
            elif c_type.lower() == 'mass':
                 potential_names.extend(['m'])

            found_match = False
            for var_obj in variables: # variables is List[Variable]
                var_name_lower = var_obj.name.lower()
                for p_name in potential_names:
                    if p_name == var_name_lower: # Exact match (case insensitive for p_name vs var_name_lower)
                        gt_column_index = var_obj.index
                        found_match = True
                        break
                if found_match:
                    break

            # Attempt 2: Substring match if no specific match found
            if not found_match:
                for var_obj in variables:
                    if c_type.lower() in var_obj.name.lower():
                        gt_column_index = var_obj.index
                        found_match = True
                        break

            if found_match and gt_column_index is not None and gt_column_index < data.shape[1]:
                ground_truth_traj[f'conserved_{c_type}'] = data[:, gt_column_index]
            else:
                ground_truth_traj[f'conserved_{c_type}'] = None
                # Optionally print a warning:
                # print(f"Warning: No ground truth column found for {c_type} or index {gt_column_index} out of bounds for data shape {data.shape}")

        hypothesis_params = {'variables_info': variables}

        conservation_bonus = self.conservation_calculator.compute_conservation_bonus(
            predicted_traj=predicted_traj,
            ground_truth_traj=ground_truth_traj,
            hypothesis_params=hypothesis_params
        )
        
        # Combine rewards
        intrinsic_reward = (
            self.novelty_weight * novelty_reward +
            self.diversity_weight * diversity_reward +
            self.complexity_growth_weight * complexity_reward +
            self.conservation_weight * conservation_bonus # Add weighted conservation bonus
        )
        
        # Update history
        self.expression_history.append(expression)
        self.complexity_history.append(complexity)
        if embedding is not None:
            self.discovery_embeddings.append(embedding)
        
        # Return combined reward
        return extrinsic_reward + intrinsic_reward

    def evaluate_expression_on_data(self, expression_str: str, data: np.ndarray, variables: List[Variable]) -> np.ndarray:
        """
        Evaluates a symbolic expression string on given data.

        Args:
            expression_str: The string representation of the mathematical expression.
            data: A NumPy array where rows are data points and columns correspond to variables.
            variables: A list of Variable objects, defining the symbols and their data indices.

        Returns:
            A 1D NumPy array with the evaluated results. NaN for errors.
        """
        try:
            sympy_expr = sympy.parse_expr(expression_str)
        except Exception as e:
            print(f"Error parsing expression '{expression_str}': {e}")
            return np.full(data.shape[0], np.nan)

        # Create a list of SymPy symbols from the Variable objects
        # These are the symbols that SymPy will recognize in the expression
        sympy_symbols = [var.symbolic for var in variables]

        evaluated_results = []
        for i in range(data.shape[0]):
            row_data = data[i, :]
            substitution_dict = {}
            for var_obj in variables:
                # var_obj.index gives the column in data for this variable
                # var_obj.symbolic is the SymPy symbol for this variable
                substitution_dict[var_obj.symbolic] = row_data[var_obj.index]

            try:
                # Substitute values and evaluate
                evaluated_value = sympy_expr.subs(substitution_dict).evalf()

                # Check if the result is complex or symbolic
                if (isinstance(evaluated_value, sympy.Expr) and evaluated_value.has(sympy.I)) or not evaluated_value.is_number:
                     print(f"Warning: Expression '{expression_str}' evaluated to non-numeric type '{type(evaluated_value)}' for data row {i}: {evaluated_value}")
                     evaluated_results.append(np.nan)
                elif isinstance(evaluated_value, (sympy.Number, float, int)):
                    evaluated_results.append(float(evaluated_value))
                elif evaluated_value == sympy.zoo or evaluated_value == sympy.oo or evaluated_value == -sympy.oo: # Check for infinity
                    print(f"Warning: Expression '{expression_str}' evaluated to infinity for data row {i}.")
                    evaluated_results.append(np.nan)
                else: # Catch other non-numeric results
                    print(f"Warning: Expression '{expression_str}' evaluated to non-numeric type '{type(evaluated_value)}' for data row {i}: {evaluated_value}")
                    evaluated_results.append(np.nan)
            except (TypeError, AttributeError, sympy.SympifyError, ValueError, ZeroDivisionError) as e:
                # TypeError can occur for undefined functions (e.g., log(-1))
                # AttributeError can occur for various reasons with SymPy objects
                # SympifyError for issues during internal conversion/evaluation
                # ValueError for math domain errors not caught by SymPy directly
                # ZeroDivisionError for explicit division by zero if not handled by SymPy's 1/0 -> zoo
                print(f"Error evaluating expression '{expression_str}' with data row {i} (substitutions: {substitution_dict}): {e}")
                evaluated_results.append(np.nan)
            except Exception as e: # Catch any other unexpected errors
                print(f"Unexpected error evaluating expression '{expression_str}' with data row {i}: {e}")
                evaluated_results.append(np.nan)

        return np.array(evaluated_results, dtype=float)

    def _calculate_novelty_reward(self, expression: str) -> float:
        """Reward for discovering new expressions."""
        
        # Simple novelty: have we seen this exact expression?
        if expression not in self.expression_cache:
            self.expression_cache[expression] = 1
            base_novelty = 1.0
        else:
            # Decay reward for repeated discoveries
            self.expression_cache[expression] += 1
            base_novelty = 1.0 / self.expression_cache[expression]
        
        # Structural novelty: new operators or combinations
        operators = ['+', '-', '*', '/', '**', 'sin', 'cos', 'log', 'exp']
        used_ops = [op for op in operators if op in expression]
        
        # Bonus for using rare operators
        op_novelty = 0.0
        rare_ops = ['sin', 'cos', 'log', 'exp', '**']
        for op in rare_ops:
            if op in used_ops:
                op_novelty += 0.2
        
        return base_novelty + op_novelty
    
    def _calculate_diversity_reward(self, 
                                  expression: str,
                                  embedding: Optional[np.ndarray]) -> float:
        """Reward for increasing diversity of discoveries."""
        
        if embedding is None or len(self.discovery_embeddings) < 2:
            return 0.0
        
        # Calculate distance to nearest neighbors
        distances = []
        for past_embedding in list(self.discovery_embeddings)[-20:]:
            dist = np.linalg.norm(embedding - past_embedding)
            distances.append(dist)
        
        if distances:
            # Reward expressions that are far from recent discoveries
            min_dist = np.min(distances)
            mean_dist = np.mean(distances)
            
            # Normalize to [0, 1]
            diversity_score = np.tanh(mean_dist / 2.0)
            return diversity_score
        
        return 0.0
    
    def _calculate_complexity_growth_reward(self, complexity: int) -> float:
        """Reward for appropriate complexity growth."""
        
        if len(self.complexity_history) < 2:
            return 0.0
        
        # Calculate complexity trend
        recent_complexities = list(self.complexity_history)[-10:]
        mean_complexity = np.mean(recent_complexities)
        
        # Reward moderate complexity growth
        if complexity > mean_complexity:
            # Growing complexity (good if not too fast)
            growth_rate = (complexity - mean_complexity) / (mean_complexity + 1)
            if growth_rate < 0.5:  # Not growing too fast
                return 0.5 * growth_rate
            else:
                return 0.0  # Penalize huge jumps
        else:
            # Reward finding simpler equivalent expressions
            if complexity < mean_complexity * 0.8:
                return 0.3
        
        return 0.0
    
    def get_exploration_bonus(self) -> float:
        """Get current exploration bonus based on discovery stagnation."""
        
        if len(self.expression_history) < 50:
            return 0.0
        
        # Check for stagnation (repeated discoveries)
        recent = list(self.expression_history)[-50:]
        unique_recent = len(set(recent))
        stagnation_ratio = unique_recent / len(recent)
        
        # Higher bonus when stagnating
        if stagnation_ratio < 0.3:
            return 0.5  # High stagnation
        elif stagnation_ratio < 0.5:
            return 0.2  # Moderate stagnation
        else:
            return 0.0  # Good diversity


class DynamicSelfPlayController:
    """Dynamically adjusts self-play strategy based on performance."""
    
    def __init__(self):
        self.performance_history: Deque[float] = deque(maxlen=100)
        self.mode_history: Deque[str] = deque(maxlen=50)
        self.opponent_performance: Dict[str, Deque[float]] = {}
        
        # Adaptive thresholds
        self.exploration_threshold = 0.3
        self.exploitation_threshold = 0.7
        
    def select_training_mode(self, 
                           current_performance: float,
                           discovery_rate: float) -> str:
        """Select training mode based on current state."""
        
        self.performance_history.append(current_performance)
        
        # Calculate performance trend
        if len(self.performance_history) < 10:
            return "standard"  # Not enough data
        
        recent_perf = list(self.performance_history)[-10:]
        perf_trend = np.polyfit(range(10), recent_perf, 1)[0]
        
        # Mode selection logic
        if discovery_rate < self.exploration_threshold:
            # Low discovery rate - need adversarial challenge
            mode = "adversarial"
        elif current_performance > self.exploitation_threshold and perf_trend > 0:
            # Doing well and improving - cooperative refinement
            mode = "cooperative"
        elif perf_trend < -0.01:
            # Performance declining - back to basics
            mode = "standard"
        else:
            # Mixed strategy
            modes = ["standard", "adversarial", "cooperative"]
            weights = [0.4, 0.4, 0.2]
            mode = np.random.choice(modes, p=weights)
        
        self.mode_history.append(mode)
        return mode
    
    def select_opponent(self,
                       league_agents: List[Any],
                       current_skill: float) -> Any:
        """Select opponent based on skill matching and diversity."""
        
        if not league_agents:
            return None
        
        # Categorize opponents by skill level
        skill_gaps = []
        for agent in league_agents:
            agent_skill = getattr(agent, 'elo_rating', 1500) / 1500
            skill_gap = abs(agent_skill - current_skill)
            skill_gaps.append(skill_gap)
        
        # Prefer opponents at similar skill level
        skill_weights = np.exp(-np.array(skill_gaps) * 2)
        
        # Add diversity bonus
        diversity_weights = []
        for i, agent in enumerate(league_agents):
            agent_id = f"agent_{i}"
            if agent_id not in self.opponent_performance:
                # Haven't played this opponent recently
                diversity_weights.append(2.0)
            else:
                # Decay based on how recently we played
                recency = len(self.opponent_performance[agent_id])
                diversity_weights.append(1.0 / (recency + 1))
        
        diversity_weights = np.array(diversity_weights)
        
        # Combine weights
        combined_weights = skill_weights * diversity_weights
        combined_weights /= combined_weights.sum()
        
        # Select opponent
        selected_idx = np.random.choice(len(league_agents), p=combined_weights)
        selected_opponent = league_agents[selected_idx]
        
        # Track selection
        agent_id = f"agent_{selected_idx}"
        if agent_id not in self.opponent_performance:
            self.opponent_performance[agent_id] = deque(maxlen=10)
        
        return selected_opponent
    
    def update_strategy(self, 
                       episode_results: Dict[str, float],
                       mode: str):
        """Update strategy based on episode results."""
        
        # Analyze effectiveness of different modes
        mode_performance = {}
        for i, m in enumerate(self.mode_history):
            if i < len(self.performance_history):
                if m not in mode_performance:
                    mode_performance[m] = []
                mode_performance[m].append(self.performance_history[i])
        
        # Adjust thresholds based on mode effectiveness
        if len(mode_performance.get("adversarial", [])) > 5:
            adv_mean = np.mean(mode_performance["adversarial"])
            if adv_mean < 0.4:
                # Adversarial mode not helping
                self.exploration_threshold *= 0.9
        
        if len(mode_performance.get("cooperative", [])) > 5:
            coop_mean = np.mean(mode_performance["cooperative"])
            if coop_mean > 0.6:
                # Cooperative mode working well
                self.exploitation_threshold *= 0.95


class EnhancedObservationEncoder:
    """Richer observation encoding for the RL agent."""
    
    def __init__(self, 
                 base_dim: int = 128,
                 history_length: int = 10):
        
        self.base_dim = base_dim
        self.history_length = history_length
        
        # History tracking
        self.action_history: Deque[int] = deque(maxlen=history_length)
        self.reward_history: Deque[float] = deque(maxlen=history_length)
        self.expression_history: Deque[str] = deque(maxlen=history_length)
        
        # Context encoder
        self.context_encoder = nn.Sequential(
            nn.Linear(history_length * 3, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
    
    def enhance_observation(self,
                          base_obs: np.ndarray,
                          current_state: 'TreeState',
                          grammar: Any) -> np.ndarray:
        """Add rich contextual information to observation."""
        
        enhanced_features = []
        
        # 1. Base observation
        enhanced_features.append(base_obs)
        
        # 2. Tree structure features
        tree_features = self._extract_tree_features(current_state)
        enhanced_features.append(tree_features)
        
        # 3. Historical context
        history_features = self._encode_history()
        enhanced_features.append(history_features)
        
        # 4. Grammar state features
        grammar_features = self._extract_grammar_features(grammar)
        enhanced_features.append(grammar_features)
        
        # 5. Current complexity budget
        complexity_features = self._extract_complexity_features(current_state)
        enhanced_features.append(complexity_features)
        
        # Concatenate all features
        enhanced_obs = np.concatenate([
            f.flatten() if isinstance(f, np.ndarray) else np.array([f])
            for f in enhanced_features
        ])
        
        return enhanced_obs
    
    def _extract_tree_features(self, state: 'TreeState') -> np.ndarray:
        """Extract structural features from current tree."""
        
        features = []
        
        # Depth statistics
        depths = self._get_node_depths(state.root)
        features.extend([
            np.mean(depths) if depths else 0,
            np.max(depths) if depths else 0,
            np.std(depths) if depths else 0
        ])
        
        # Node type distribution
        node_types = self._count_node_types(state.root)
        features.extend([
            node_types.get('operator', 0) / (sum(node_types.values()) + 1),
            node_types.get('variable', 0) / (sum(node_types.values()) + 1),
            node_types.get('constant', 0) / (sum(node_types.values()) + 1)
        ])
        
        # Completion status
        features.append(1.0 if state.is_complete() else 0.0)
        
        # Balance (left vs right subtree sizes)
        balance = self._calculate_tree_balance(state.root)
        features.append(balance)
        
        return np.array(features, dtype=np.float32)
    
    def _get_node_depths(self, node: 'ExpressionNode', depth: int = 0) -> List[int]:
        """Get depths of all nodes in tree."""
        if node.node_type.value == "empty":
            return []
        
        depths = [depth]
        for child in node.children:
            depths.extend(self._get_node_depths(child, depth + 1))
        
        return depths
    
    def _count_node_types(self, node: 'ExpressionNode') -> Dict[str, int]:
        """Count node types in tree."""
        if node.node_type.value == "empty":
            return {}
        
        counts = {node.node_type.value: 1}
        for child in node.children:
            child_counts = self._count_node_types(child)
            for node_type, count in child_counts.items():
                counts[node_type] = counts.get(node_type, 0) + count
        
        return counts
    
    def _calculate_tree_balance(self, node: 'ExpressionNode') -> float:
        """Calculate tree balance metric."""
        if not node.children:
            return 0.0
        
        subtree_sizes = [self._get_subtree_size(child) for child in node.children]
        if len(subtree_sizes) >= 2:
            return abs(subtree_sizes[0] - subtree_sizes[1]) / (sum(subtree_sizes) + 1)
        return 0.0
    
    def _get_subtree_size(self, node: 'ExpressionNode') -> int:
        """Get size of subtree."""
        if node.node_type.value == "empty":
            return 0
        return 1 + sum(self._get_subtree_size(child) for child in node.children)
    
    def _encode_history(self) -> np.ndarray:
        """Encode action and reward history."""
        
        # Pad histories if needed
        action_vec = list(self.action_history) + [0] * (self.history_length - len(self.action_history))
        reward_vec = list(self.reward_history) + [0] * (self.history_length - len(self.reward_history))
        
        # Simple statistics
        features = []
        
        # Action diversity
        if action_vec:
            unique_actions = len(set(action_vec))
            features.append(unique_actions / (len(action_vec) + 1))
        else:
            features.append(0.0)
        
        # Reward trend
        if len(reward_vec) > 2:
            reward_trend = np.polyfit(range(len(reward_vec)), reward_vec, 1)[0]
            features.append(np.tanh(reward_trend))
        else:
            features.append(0.0)
        
        # Recent performance
        recent_rewards = reward_vec[-5:] if reward_vec else [0]
        features.append(np.mean(recent_rewards))
        features.append(np.std(recent_rewards))
        
        return np.array(features, dtype=np.float32)
    
    def _extract_grammar_features(self, grammar: Any) -> np.ndarray:
        """Extract features about grammar state."""
        
        features = []
        
        # Number of learned functions
        n_learned = len(getattr(grammar, 'learned_functions', {}))
        features.append(n_learned / 10.0)  # Normalize
        
        # Number of discovered variables
        n_vars = len(getattr(grammar, 'variables', {}))
        features.append(n_vars / 10.0)  # Normalize
        
        # Grammar complexity (size of primitive set)
        n_primitives = sum(len(ops) for ops in grammar.primitives.values())
        features.append(n_primitives / 20.0)  # Normalize
        
        return np.array(features, dtype=np.float32)
    
    def _extract_complexity_features(self, state: 'TreeState') -> np.ndarray:
        """Extract features about complexity budget."""
        
        features = []
        
        current_complexity = state.count_nodes()
        max_complexity = getattr(state, 'max_complexity', 30)
        
        # Complexity usage ratio
        features.append(current_complexity / max_complexity)
        
        # Remaining complexity budget
        remaining = max_complexity - current_complexity
        features.append(remaining / max_complexity)
        
        # Complexity per depth
        max_depth = getattr(state, 'max_depth', 10)
        current_depth = self._get_max_depth(state.root)
        if current_depth > 0:
            features.append(current_complexity / current_depth)
        else:
            features.append(0.0)
        
        return np.array(features, dtype=np.float32)
    
    def _get_max_depth(self, node: 'ExpressionNode', depth: int = 0) -> int:
        """Get maximum depth of tree."""
        if node.node_type.value == "empty" or not node.children:
            return depth
        
        return max(self._get_max_depth(child, depth + 1) for child in node.children)
    
    def update_history(self, action: int, reward: float, expression: str):
        """Update history after each step."""
        self.action_history.append(action)
        self.reward_history.append(reward)
        self.expression_history.append(expression)


class AdaptiveTrainingController:
    """Uses emergent behavior monitoring to adapt training parameters."""
    
    def __init__(self):
        self.phase_history: List[str] = []
        self.metric_history: Dict[str, Deque[float]] = {
            'learning_rate': deque(maxlen=100),
            'exploration_rate': deque(maxlen=100),
            'complexity_penalty': deque(maxlen=100)
        }
        
        # Base parameters
        self.base_lr = 3e-4
        self.base_exploration = 0.1
        self.base_complexity_penalty = 0.01
        
    def adapt_parameters(self,
                        phase_type: str,
                        current_metrics: Dict[str, float]) -> Dict[str, float]:
        """Adapt training parameters based on detected phase."""
        
        self.phase_history.append(phase_type)
        
        # Different strategies for different phases
        if phase_type == "breakthrough":
            # Exploit the breakthrough
            lr = self.base_lr * 0.5  # Slower learning to stabilize
            exploration = self.base_exploration * 0.5  # Less exploration
            complexity_penalty = self.base_complexity_penalty * 0.5  # Allow complexity
            
        elif phase_type == "simplification":
            # Encourage finding simpler forms
            lr = self.base_lr * 1.2
            exploration = self.base_exploration * 0.8
            complexity_penalty = self.base_complexity_penalty * 2.0  # Stronger penalty
            
        elif phase_type == "exploration":
            # Boost exploration
            lr = self.base_lr * 1.5
            exploration = self.base_exploration * 2.0
            complexity_penalty = self.base_complexity_penalty * 0.8
            
        elif phase_type == "refinement":
            # Fine-tune existing discoveries
            lr = self.base_lr * 0.3
            exploration = self.base_exploration * 0.3
            complexity_penalty = self.base_complexity_penalty * 1.0
            
        else:  # "stagnation" or unknown
            # Try to break out of stagnation
            lr = self.base_lr * 2.0
            exploration = self.base_exploration * 3.0
            complexity_penalty = self.base_complexity_penalty * 0.5
        
        # Smooth transitions
        if len(self.metric_history['learning_rate']) > 0:
            prev_lr = self.metric_history['learning_rate'][-1]
            lr = 0.7 * prev_lr + 0.3 * lr  # Smooth change
        
        # Update history
        self.metric_history['learning_rate'].append(lr)
        self.metric_history['exploration_rate'].append(exploration)
        self.metric_history['complexity_penalty'].append(complexity_penalty)
        
        return {
            'learning_rate': lr,
            'entropy_coeff': exploration,
            'complexity_penalty': complexity_penalty
        }
    
    def detect_stagnation(self, 
                         discovery_rate: float,
                         performance_trend: float) -> bool:
        """Detect if training is stagnating."""
        
        # Check recent phase history
        if len(self.phase_history) > 20:
            recent_phases = self.phase_history[-20:]
            exploration_ratio = recent_phases.count("exploration") / len(recent_phases)
            
            # High exploration ratio + low discovery = stagnation
            if exploration_ratio > 0.7 and discovery_rate < 0.1:
                return True
        
        # Check performance trend
        if performance_trend < -0.01 and discovery_rate < 0.2:
            return True
        
        return False
    
    def suggest_intervention(self,
                           current_state: Dict[str, float]) -> Optional[str]:
        """Suggest intervention based on current state."""
        
        if self.detect_stagnation(
            current_state.get('discovery_rate', 0),
            current_state.get('performance_trend', 0)
        ):
            # Suggest interventions
            recent_complexities = current_state.get('recent_complexities', [])
            if recent_complexities and np.mean(recent_complexities) > 20:
                return "reduce_max_complexity"
            else:
                return "increase_exploration_bonus"
        
        # Check if we should increase difficulty
        if current_state.get('success_rate', 0) > 0.8:
            return "increase_difficulty"
        
        return None


class EnhancedSymbolicDiscoveryEnv(SymbolicDiscoveryEnv):
    """Enhanced environment with integrated feedback systems."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Enhanced components
        self.intrinsic_calculator = IntrinsicRewardCalculator()
        self.observation_encoder = EnhancedObservationEncoder()
        self.training_controller = AdaptiveTrainingController()
        
        # Metrics for adaptation
        self.episode_discoveries: List[str] = []
        self.episode_complexities: List[int] = []
        
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Enhanced step with intrinsic rewards and rich observations."""
        
        # Standard step
        obs, reward, terminated, truncated, info = super().step(action)
        
        # Calculate intrinsic reward if episode complete
        if terminated and 'expression' in info:
            expr = info['expression']
            complexity = info.get('complexity', 0)
            
            # Get expression embedding (simplified)
            embedding = self._get_expression_embedding(expr)
            
            # Calculate enhanced reward
            enhanced_reward = self.intrinsic_calculator.calculate_intrinsic_reward(
                expression=expr,
                complexity=complexity,
                extrinsic_reward=reward,
                embedding=embedding,
                data=self.target_data, # Pass self.target_data
                variables=self.variables # Pass self.variables
            )
            
            # Track for metrics
            self.episode_discoveries.append(expr)
            self.episode_complexities.append(complexity)
            
            # Update info
            info['intrinsic_reward'] = enhanced_reward - reward
            info['total_reward'] = enhanced_reward
            
            reward = enhanced_reward
        
        # Enhance observation
        enhanced_obs = self.observation_encoder.enhance_observation(
            obs, 
            self.current_state,
            self.grammar
        )
        
        # Update observation encoder history
        self.observation_encoder.update_history(
            action, 
            reward,
            info.get('expression', '')
        )
        
        return enhanced_obs, reward, terminated, truncated, info
    
    def _get_expression_embedding(self, expr: str) -> np.ndarray:
        """Get embedding for expression (simplified)."""
        
        # Simple feature extraction
        features = []
        
        # Operator counts
        ops = ['+', '-', '*', '/', '**', 'sin', 'cos', 'log', 'exp']
        for op in ops:
            features.append(expr.count(op))
        
        # Variable usage
        for var in self.variables:
            features.append(expr.count(var.name))
        
        # Length and depth approximation
        features.append(len(expr))
        features.append(expr.count('('))  # Parentheses as depth proxy
        
        return np.array(features, dtype=np.float32)
    
    def get_adaptation_metrics(self) -> Dict[str, float]:
        """Get metrics for training adaptation."""
        
        discovery_rate = len(set(self.episode_discoveries)) / (len(self.episode_discoveries) + 1)
        
        if len(self.episode_complexities) > 10:
            complexity_trend = np.polyfit(
                range(len(self.episode_complexities)),
                self.episode_complexities,
                1
            )[0]
        else:
            complexity_trend = 0.0
        
        return {
            'discovery_rate': discovery_rate,
            'complexity_trend': complexity_trend,
            'unique_discoveries': len(set(self.episode_discoveries)),
            'mean_complexity': np.mean(self.episode_complexities) if self.episode_complexities else 0
        }
    
    def reset_episode_metrics(self):
        """Reset episode tracking metrics."""
        self.episode_discoveries = []
        self.episode_complexities = []


class IntegratedTrainingLoop:
    """Main training loop with all feedback systems integrated."""
    
    def __init__(self,
                 env: EnhancedSymbolicDiscoveryEnv,
                 policy: HypothesisNet,
                 config: Dict[str, Any]):
        
        self.env = env
        self.policy = policy
        self.config = config
        
        # Controllers
        self.selfplay_controller = DynamicSelfPlayController()
        
        # Base trainer
        from hypothesis_policy_network import PPOTrainer
        self.trainer = PPOTrainer(policy, env)
        
    def train_with_feedback(self, total_timesteps: int):
        """Training loop with integrated feedback systems."""
        
        updates_per_cycle = 10
        n_cycles = total_timesteps // (updates_per_cycle * 2048)
        
        for cycle in range(n_cycles):
            # Get current metrics
            env_metrics = self.env.get_adaptation_metrics()
            
            # Determine training mode
            discovery_rate = env_metrics['discovery_rate']
            current_performance = np.mean(self.trainer.episode_rewards) if self.trainer.episode_rewards else 0.0
            
            mode = self.selfplay_controller.select_training_mode(
                current_performance,
                discovery_rate
            )
            
            print(f"\nCycle {cycle}: Mode = {mode}, Discovery Rate = {discovery_rate:.3f}")
            
            # Adapt training parameters
            if hasattr(self.env, 'training_controller'):
                # Detect phase (simplified)
                if discovery_rate > 0.7:
                    phase = "breakthrough"
                elif discovery_rate < 0.2:
                    phase = "stagnation"
                else:
                    phase = "exploration"
                
                adapted_params = self.env.training_controller.adapt_parameters(
                    phase,
                    env_metrics
                )
                
                # Apply adapted parameters
                self.trainer.optimizer.param_groups[0]['lr'] = adapted_params['learning_rate']
                self.trainer.entropy_coef = adapted_params['entropy_coeff']
                
                print(f"  Adapted LR: {adapted_params['learning_rate']:.5f}, "
                      f"Entropy: {adapted_params['entropy_coeff']:.3f}")
            
            # Run training
            self.trainer.train(
                total_timesteps=updates_per_cycle * 2048,
                rollout_length=2048,
                n_epochs=10,
                log_interval=5
            )
            
            # Reset episode metrics for next cycle
            self.env.reset_episode_metrics()
            
            # Check for intervention
            intervention = self.env.training_controller.suggest_intervention({
                'discovery_rate': discovery_rate,
                'performance_trend': 0.0,  # Could calculate this
                'success_rate': current_performance,
                'recent_complexities': env_metrics.get('mean_complexity', 0)
            })
            
            if intervention:
                print(f"  Intervention suggested: {intervention}")
                self._apply_intervention(intervention)
    
    def _apply_intervention(self, intervention: str):
        """Apply suggested intervention."""
        
        if intervention == "reduce_max_complexity":
            self.env.max_complexity = int(self.env.max_complexity * 0.8)
            print(f"    Reduced max complexity to {self.env.max_complexity}")
            
        elif intervention == "increase_exploration_bonus":
            bonus = self.env.intrinsic_calculator.get_exploration_bonus()
            self.env.intrinsic_calculator.novelty_weight *= (1 + bonus)
            print(f"    Increased novelty weight to {self.env.intrinsic_calculator.novelty_weight:.3f}")
            
        elif intervention == "increase_difficulty":
            if hasattr(self.env, 'curriculum_manager'):
                # Update curriculum
                print("    Advanced curriculum stage")


# Example usage
if __name__ == "__main__":
    from progressive_grammar_system import ProgressiveGrammar, Variable
    
    # Setup
    grammar = ProgressiveGrammar()
    variables = [
        Variable("x", 0, {"smoothness": 0.9}),
        Variable("v", 1, {"conservation_score": 0.8})
    ]
    
    # Create data
    n_samples = 1000
    x_data = np.random.randn(n_samples)
    v_data = np.random.randn(n_samples) * 2
    energy = 0.5 * v_data**2 + 0.5 * x_data**2
    data = np.column_stack([x_data, v_data, energy])
    
    # Create enhanced environment
    env = EnhancedSymbolicDiscoveryEnv(
        grammar=grammar,
        target_data=data,
        variables=variables,
        max_depth=7,
        max_complexity=15
    )
    
    # Create policy with proper observation space
    enhanced_obs_dim = env.observation_encoder.enhance_observation(
        np.zeros(env.observation_space.shape[0]),
        env.current_state,
        grammar
    ).shape[0]
    
    policy = HypothesisNet(
        observation_dim=enhanced_obs_dim,
        action_dim=env.action_space.n,
        hidden_dim=256
    )
    
    # Create integrated trainer
    trainer = IntegratedTrainingLoop(env, policy, {})
    
    print("Starting training with enhanced feedback loops...")
    trainer.train_with_feedback(total_timesteps=100000)
