import logging # Added for safe error logging

"""
Symbolic Discovery Environment
==============================

A reinforcement learning environment for intelligent hypothesis generation.
Transforms the combinatorial search problem into a sequential decision process.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import gymnasium as gym
from gymnasium import spaces


def float_cast(val):
    """Ensure native Python float for torch assignment and handle non-finite values."""
    try:
        f_val = float(val)
        if not np.isfinite(f_val):
            return 0.0
        return f_val
    except (ValueError, TypeError):
        if hasattr(val, 'item'):
            try:
                f_val = float(val.item())
                if not np.isfinite(f_val):
                    return 0.0
                return f_val
            except (ValueError, TypeError):
                return 0.0
        return 0.0

class NodeType(Enum):
    """Types of nodes in expression tree."""
    EMPTY = "empty"
    OPERATOR = "operator"
    VARIABLE = "variable"
    CONSTANT = "constant"
    COMPLETE = "complete"

@dataclass
class ExpressionNode:
    """Node in the expression tree being constructed."""
    node_type: NodeType
    value: Any
    children: List['ExpressionNode'] = field(default_factory=list)
    parent: Optional['ExpressionNode'] = None
    depth: int = 0
    position: int = 0

    def is_complete(self) -> bool:
        if self.node_type in (NodeType.VARIABLE, NodeType.CONSTANT):
            return True
        if self.node_type == NodeType.EMPTY:
            return False
        expected = self._expected_children()
        return len(self.children) == expected and all(c.is_complete() for c in self.children)

    def _expected_children(self) -> int:
        if self.node_type != NodeType.OPERATOR:
            return 0
        binary = {'+', '-', '*', '/', '**'}
        unary = {'neg', 'inv', 'sqrt', 'log', 'exp', 'sin', 'cos'}
        calculus = {'diff', 'int'}
        if self.value in binary:
            return 2
        if self.value in unary:
            return 1
        if self.value in calculus:
            return 2
        return 0

    def to_expression(self, grammar) -> Optional[Any]:
        if not self.is_complete():
            return None
        if self.node_type == NodeType.VARIABLE:
            return self.value
        if self.node_type == NodeType.CONSTANT:
            return grammar.create_expression('const', [self.value])
        if self.node_type == NodeType.OPERATOR:
            args = [c.to_expression(grammar) for c in self.children]
            if all(args):
                return grammar.create_expression(self.value, args)
        return None

class TreeState:
    """Represents the current state of expression tree construction."""
    def __init__(self, root: Optional[ExpressionNode] = None, max_depth: int = 10):
        self.root = root or ExpressionNode(NodeType.EMPTY, None)
        self.max_depth = max_depth
        self.construction_history: List[Dict[str, Any]] = []

    def get_next_empty_node(self) -> Optional[ExpressionNode]:
        return self._find_empty(self.root)

    def _find_empty(self, node: ExpressionNode) -> Optional[ExpressionNode]:
        if node.node_type == NodeType.EMPTY:
            return node
        for c in node.children:
            res = self._find_empty(c)
            if res:
                return res
        return None

    def is_complete(self) -> bool:
        return self.root.is_complete()

    def count_nodes(self) -> int:
        return self._count(self.root)

    def _count(self, node: ExpressionNode) -> int:
        if node.node_type == NodeType.EMPTY:
            return 0
        return 1 + sum(self._count(c) for c in node.children)

    def to_tensor_representation(self, grammar, max_nodes: int = 50) -> torch.Tensor:
        feature_dim = 128
        tensor = torch.zeros((max_nodes, feature_dim), dtype=torch.float32)
        nodes = []
        queue = deque([self.root])
        while queue and len(nodes) < max_nodes:
            n = queue.popleft()
            nodes.append(n)
            queue.extend(n.children)
        for i, node in enumerate(nodes):
            if node.node_type == NodeType.EMPTY:
                tensor[i, 0] = 1.0
            elif node.node_type == NodeType.OPERATOR:
                tensor[i, 1] = 1.0
                all_ops = (
                    sorted(grammar.primitives.get('binary_ops', set())) +
                    sorted(grammar.primitives.get('unary_ops', set())) +
                    sorted(grammar.primitives.get('calculus_ops', set()))
                )
                if node.value in all_ops:
                    idx = all_ops.index(node.value)
                    tensor[i, 10 + idx] = 1.0
            elif node.node_type == NodeType.VARIABLE:
                tensor[i, 2] = 1.0
                if hasattr(node.value, 'properties'):
                    for j, (_, v) in enumerate(node.value.properties.items()):
                        if j < 10:
                            tensor[i, 50 + j] = float_cast(v)
            elif node.node_type == NodeType.CONSTANT:
                tensor[i, 3] = 1.0
                tensor[i, 60] = float_cast(node.value)
            tensor[i, 70] = float_cast(node.depth)
            tensor[i, 71] = float_cast(node.position)
            tensor[i, 72] = float_cast(len(node.children))
        return tensor

def _build_action_space(grammar, variables):
    actions: List[Tuple[str, Any]] = []
    for op_type in ['binary_ops', 'unary_ops', 'calculus_ops']:
        ops = sorted(grammar.primitives.get(op_type, []))
        actions.extend(('operator', op) for op in ops)
    actions.extend(('variable', var) for var in variables)
    consts = [
        ('constant', 0.0),
        ('constant', 1.0),
        ('constant', -1.0),
        ('constant', 'random_small'),
        ('constant', 'random_large'),
    ]
    actions.extend(consts)
    actions.extend(('function', fn) for fn in sorted(grammar.learned_functions))
    return actions

class SymbolicDiscoveryEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(
        self,
        grammar: Any,
        target_data: np.ndarray,
        variables: List[Any],
        max_depth: int = 10,
        max_complexity: int = 30,
        reward_config: Optional[Dict[str, Any]] = None,
        max_nodes: int = 50,
        target_variable_index: Optional[int] = None,
        action_space_size: Optional[int] = None, # Added parameter
        provide_tree_structure: bool = False # Added parameter
    ):
        super().__init__()
        self.grammar = grammar
        self.target_data = target_data
        self.variables = variables
        self.max_depth = max_depth
        self.max_complexity = max_complexity
        self.max_nodes = max_nodes
        self.provide_tree_structure = provide_tree_structure
        self.target_variable_index = target_variable_index if target_variable_index is not None else -1
        # Ensure target_variable_index is a valid integer index for array slicing
        if self.target_variable_index == -1:
            self.target_variable_index = self.target_data.shape[1] - 1
        elif not isinstance(self.target_variable_index, int) or \
             not (0 <= self.target_variable_index < self.target_data.shape[1]):
            raise ValueError(f"Invalid target_variable_index: {self.target_variable_index}. "
                             f"Must be an int within data bounds [0, {self.target_data.shape[1] -1}] or None/-1 for last column.")

        default_reward_config = {
            'completion_bonus':    0.1,
            'validity_bonus':      0.05,
            'mse_weight':          1.0,
            'mse_scale_factor':    1.0,
            'complexity_penalty': -0.01,
            'depth_penalty':      -0.001,
            'timeout_penalty':    -1.0,
        }
        self.reward_config = {**default_reward_config, **(reward_config or {})}
        self.current_state = TreeState(max_depth=max_depth)
        self.steps_taken = 0
        self.max_steps = 100
        self._evaluation_cache: Dict[str, Any] = {}
        self.action_to_element = _build_action_space(grammar, variables)
        n_actions = action_space_size if action_space_size is not None else len(self.action_to_element)
        self.action_space = spaces.Discrete(n_actions)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.max_nodes * 128,),
            dtype=np.float32
        )

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        try:
            super().reset(seed=seed, options=options) # Pass options to parent
            if seed is not None:
                np.random.seed(seed)
                # Conditionally seed torch if it's determined to be used for randomness
                # that affects the environment's state generation directly.
                # torch.manual_seed(seed) # Uncomment if torch randomness is critical for reset sequence

            # Validate grammar initialization
            if not hasattr(self, 'grammar') or self.grammar is None or \
               not hasattr(self.grammar, 'primitives') or not self.grammar.primitives:
                grammar_detail = "None"
                if hasattr(self, 'grammar') and self.grammar is not None:
                    grammar_detail = f"Primitives: {getattr(self.grammar, 'primitives', 'Not Found')}"
                logging.error(f"Grammar not properly initialized. Details: {grammar_detail}")
                raise ValueError("Grammar not properly initialized or primitives are empty.")

            self.current_state = TreeState(max_depth=self.max_depth)
            self.steps_taken = 0
            self._evaluation_cache.clear()

            obs = self._get_observation()

            # Validate observation
            if obs is None:
                logging.error("Initial observation generated by _get_observation() is None.")
                raise ValueError("Generated initial observation is None.")

            if not hasattr(self, 'observation_space') or self.observation_space is None:
                logging.error("Observation space is not initialized prior to reset completion.")
                raise ValueError("Observation space is not initialized.")

            expected_shape = self.observation_space.shape
            if not isinstance(obs, np.ndarray) or obs.shape != expected_shape:
                obs_shape_str = str(obs.shape) if hasattr(obs, 'shape') else type(obs).__name__
                logging.error(f"Initial observation shape/type mismatch. Expected {expected_shape} (np.ndarray), got {obs_shape_str}.")

                if isinstance(obs, np.ndarray) and obs.size == np.prod(expected_shape): # Check if total number of elements matches
                    try:
                        obs = obs.reshape(expected_shape)
                        logging.warning(f"Successfully reshaped observation from {obs_shape_str} to {expected_shape}.")
                    except ValueError as reshape_e: # Log specific reshape error
                        logging.error(f"Failed to reshape observation to {expected_shape}: {reshape_e}", exc_info=True)
                        raise ValueError(f"Generated initial observation {obs_shape_str} is incompatible (cannot reshape) with expected {expected_shape}.") from reshape_e
                else: # Not a numpy array or wrong number of elements
                    raise ValueError(f"Generated initial observation {obs_shape_str} (size {obs.size if hasattr(obs, 'size') else 'N/A'}) is incompatible with expected {expected_shape} (size {np.prod(expected_shape)}).")

            info = self._get_info()
            return obs, info

        except Exception as e:
            logging.error(f"Critical error during SymbolicDiscoveryEnv reset: {e}", exc_info=True)

            default_obs_shape = None
            # Try to get shape from observation_space if it exists and is valid
            if hasattr(self, 'observation_space') and self.observation_space is not None and \
               hasattr(self.observation_space, 'shape') and self.observation_space.shape is not None and \
               self.observation_space.shape and \
               all(isinstance(dim, int) and dim > 0 for dim in self.observation_space.shape):
                default_obs_shape = self.observation_space.shape

            if default_obs_shape is None:
                # Fallback shape if observation_space is not available or invalid
                fallback_nodes = getattr(self, 'max_nodes', 10)
                feature_dim = 128
                default_obs_shape = (fallback_nodes * feature_dim,)
                logging.warning(f"Observation space shape unavailable or invalid. Using fallback shape: {default_obs_shape}")

            default_obs = np.zeros(default_obs_shape, dtype=np.float32)
            default_info = {"error": str(e), "reset_failed": True, "message": "Default state returned due to error."}
            return default_obs, default_info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        self.steps_taken += 1
        node = self.current_state.get_next_empty_node()
        if not node:
            return self._get_observation(), 0.0, True, False, self._get_info()
        a_type, a_val = self.action_to_element[action]
        if not self._is_valid_action(node, a_type, a_val):
            return self._get_observation(), -0.05, False, False, self._get_info()
        self._apply_action(node, a_type, a_val)
        if self.current_state.is_complete():
            reward = self._evaluate_expression()
            terminated = True
        else:
            reward = self.reward_config.get('validity_bonus', 0.0)
            terminated = False
        truncated = self.steps_taken >= self.max_steps
        if truncated and not terminated:
            reward += self.reward_config.get('timeout_penalty', 0.0)
        obs = self._get_observation()
        info = self._get_info()
        return obs, reward, terminated, truncated, info
    def _is_valid_action(self, node, a_type, a_val) -> bool:
        depth = node.depth
        if depth >= self.max_depth and a_type not in ('variable', 'constant'):
            return False
        return True
    def _apply_action(self, node, a_type, a_val):
        if a_type in ('operator', 'function'):
            node.node_type = NodeType.OPERATOR
            node.value = a_val
            for i in range(node._expected_children()):
                child = ExpressionNode(NodeType.EMPTY, None, parent=node, depth=node.depth+1, position=i)
                node.children.append(child)
        elif a_type == 'variable':
            node.node_type = NodeType.VARIABLE
            node.value = a_val
        elif a_type == 'constant':
            node.node_type = NodeType.CONSTANT
            if a_val == 'random_small':
                node.value = np.random.randn()
            elif a_val == 'random_large':
                node.value = np.random.randn() * 10
            else:
                node.value = a_val
        self.current_state.construction_history.append({
            'step': self.steps_taken, 'action_type': a_type, 'action_value': str(a_val)
        })
    def _evaluate_expression(self) -> float:
        expr = self.current_state.root.to_expression(self.grammar)
        if not expr:
            return self.reward_config.get('timeout_penalty', -1.0)

        if expr.complexity > self.max_complexity:
            return self.reward_config.get('complexity_penalty', -0.01) * expr.complexity

        errors = []
        # Pre-calculate the variance of the target data to use as a penalty for invalid predictions.
        # This makes the penalty scaled to the problem's difficulty.
        target_variance = np.var(self.target_data[:, self.target_variable_index])
        # If variance is zero (constant target), use a default penalty of 1.0.
        penalty_on_fail = target_variance if target_variance > 1e-9 else 1.0

        for i in range(len(self.target_data)):
            subs = {v.symbolic: self.target_data[i, v.index] for v in self.variables}
            target_val = self.target_data[i, self.target_variable_index]

            try:
                pred = float(expr.symbolic.subs(subs))
                if not np.isfinite(pred) or abs(pred) > 1e12: # Check for NaN, inf, or huge numbers
                    errors.append(penalty_on_fail) # Add high penalty
                else:
                    errors.append((pred - target_val)**2) # Add squared error
            except Exception:
                errors.append(penalty_on_fail) # Add high penalty on any evaluation error

        # The final MSE is the mean of all errors, including penalties.
        mse = np.mean(errors)

        # The rest of the reward calculation remains the same.
        norm = mse / (target_variance + 1e-10)
        reward = (
            self.reward_config.get('completion_bonus', 0.1) +
            self.reward_config.get('mse_weight', 1.0) * np.exp(-self.reward_config.get('mse_scale_factor', 1.0) * norm) +
            self.reward_config.get('complexity_penalty', -0.01) * expr.complexity +
            self.reward_config.get('depth_penalty', -0.001) * self.max_depth
        )
        self._evaluation_cache.update({'expression': str(expr.symbolic), 'mse': mse,
                                       'complexity': expr.complexity, 'reward': reward})
        return float(reward)
    def _get_observation(self) -> np.ndarray:
        tensor = self.current_state.to_tensor_representation(self.grammar, max_nodes=self.max_nodes)
        return tensor.flatten().numpy()
    def _get_info(self) -> Dict[str, Any]:
        info = {'steps': self.steps_taken, 'nodes': self.current_state.count_nodes(),
                'complete': self.current_state.is_complete()}
        info.update(self._evaluation_cache)
        return info
    def render(self, mode='human'):
        print(f"Step {self.steps_taken}: Nodes {self.current_state.count_nodes()}")
        if self.current_state.is_complete():
            expr = self.current_state.root.to_expression(self.grammar)
            if expr:
                print(f"Expr: {expr.symbolic} Comp:{expr.complexity}")
    def get_action_mask(self) -> np.ndarray:
        empty_node = self.current_state.get_next_empty_node()
        if not empty_node:
            return np.zeros(self.action_space.n, dtype=bool)
        mask = np.zeros(self.action_space.n, dtype=bool)
        for i, (atype, aval) in enumerate(self.action_to_element):
            if self._is_valid_action(empty_node, atype, aval): mask[i] = True
        return mask

class CurriculumManager:
    """Manages curriculum learning for expression discovery."""
    def __init__(self, base_env: SymbolicDiscoveryEnv):
        self.base_env = base_env
        self.difficulty_level = 0
        self.success_rate_history = deque(maxlen=100)
        self.curriculum = [
            {'max_depth':3, 'max_complexity':5},
            {'max_depth':5, 'max_complexity':10},
            {'max_depth':7, 'max_complexity':15},
            {'max_depth':10, 'max_complexity':30},
        ]
    def get_current_env(self) -> SymbolicDiscoveryEnv:
        cfg = self.curriculum[self.difficulty_level]
        self.base_env.max_depth = cfg['max_depth']
        self.base_env.max_complexity = cfg['max_complexity']
        return self.base_env
    def update_curriculum(self, episode_success: bool):
        self.success_rate_history.append(float(episode_success))
        if len(self.success_rate_history) >= 50:
            rate = np.mean(self.success_rate_history)
            if rate > 0.7 and self.difficulty_level < len(self.curriculum)-1:
                self.difficulty_level += 1; self.success_rate_history.clear()
            elif rate < 0.3 and self.difficulty_level > 0:
                self.difficulty_level -= 1; self.success_rate_history.clear()

if __name__ == "__main__":
    from progressive_grammar_system import ProgressiveGrammar, Variable
    grammar = ProgressiveGrammar()
    var_x = Variable("x", 0, {"smoothness":0.9})
    variables = [var_x]
    n = 1000; x = np.random.randn(n); y = 2*x + 1; data = np.column_stack([x, y])
    env = SymbolicDiscoveryEnv(grammar, data, variables, max_nodes=50)
    obs, info = env.reset()
    for _ in range(5):
        mask = env.get_action_mask()
        a = np.random.choice(np.where(mask)[0])
        obs, rew, done, truncated, info = env.step(a)
        print(rew)

__all__ = ["SymbolicDiscoveryEnv", "CurriculumManager"]