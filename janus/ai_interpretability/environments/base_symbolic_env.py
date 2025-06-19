"""
Symbolic Discovery Environment
==============================

A reinforcement learning environment for intelligent hypothesis generation.
Transforms the combinatorial search problem into a sequential decision process.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any, Union, TYPE_CHECKING, cast, Type, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import gymnasium as gym  # type: ignore
from gymnasium import spaces
import logging

if TYPE_CHECKING:
    from janus.core.grammar import ProgressiveGrammar
    from janus.core.expression import Variable, Expression as GrammarExpression
    from strict_mode_validator import StrictModeValidator


def float_cast(val: Any) -> float:
    """Ensure native Python float for torch assignment and handle non-finite values."""
    try:
        f_val = float(val)
        if not np.isfinite(f_val):
            return 0.0  # Default for non-finite
        return f_val
    except (ValueError, TypeError):
        if hasattr(val, 'item'):  # For tensor elements
            try:
                f_val = float(val.item())
                if not np.isfinite(f_val):
                    return 0.0
                return f_val
            except (ValueError, TypeError):
                return 0.0  # Default if item() conversion fails
        return 0.0  # Default for other unconvertible types


class NodeType(Enum):
    """Types of nodes in expression tree."""
    EMPTY = "empty"
    OPERATOR = "operator"
    VARIABLE = "variable"
    CONSTANT = "constant"


@dataclass
class ExpressionNode:
    """Node in the expression tree being constructed."""
    node_type: NodeType
    value: Any
    children: List['ExpressionNode'] = field(default_factory=list)
    parent: Optional['ExpressionNode'] = None
    depth: int = 0
    position: int = 0

    def is_complete(self, grammar: 'ProgressiveGrammar') -> bool:
        """Checks if the node and its subtree are complete."""
        if self.node_type == NodeType.VARIABLE or self.node_type == NodeType.CONSTANT:
            return True
        if self.node_type == NodeType.EMPTY:
            return False
        
        expected_children_count: int = self._expected_children(grammar)
        if len(self.children) != expected_children_count:
            return False
        
        return all(child.is_complete(grammar) for child in self.children)

    def _expected_children(self, grammar: 'ProgressiveGrammar') -> int:
        """Returns the number of children expected by querying the grammar."""
        if self.node_type != NodeType.OPERATOR:
            return 0
        
        try:
            # The grammar is now the single source of truth
            return grammar.get_arity(str(self.value))
        except ValueError:
            # Operator not known to the grammar, should not form children
            logging.warning(f"ExpressionNode: Operator '{self.value}' not found in grammar. Assuming 0 arity for safety.")
            return 0 # Or raise an error, depending on desired strictness

    def to_expression(self, grammar: 'ProgressiveGrammar') -> Optional['GrammarExpression']:
        """Converts this node (and its subtree) to a grammar.Expression object."""
        if not self.is_complete(grammar):
            return None
        
        ProgVarType = type(next(iter(grammar.variables.values()))) if grammar.variables else Type[Any]

        if self.node_type == NodeType.VARIABLE:
            if isinstance(self.value, ProgVarType):
                return grammar.create_expression(operator='var', operands=[self.value.name])  # type: ignore
            elif isinstance(self.value, str):
                return grammar.create_expression(operator='var', operands=[self.value])
            logging.warning(f"ExpressionNode.to_expression: Invalid variable node value type: {type(self.value)}")
            return None

        if self.node_type == NodeType.CONSTANT:
            return grammar.create_expression(operator='const', operands=[self.value])

        if self.node_type == NodeType.OPERATOR:
            child_expressions: List[Optional['GrammarExpression']] = [
                child.to_expression(grammar) for child in self.children
            ]
            if not all(expr is not None for expr in child_expressions):
                return None
            
            valid_child_expressions = cast(List['GrammarExpression'], child_expressions)
            return grammar.create_expression(operator=str(self.value), operands=valid_child_expressions)
        
        logging.error(f"ExpressionNode.to_expression: Unhandled node type: {self.node_type}")
        return None


@dataclass
class TreeState:
    """Represents the current state of expression tree construction."""
    max_depth: int = 10

    def __init__(self, root: Optional[ExpressionNode] = None, max_depth: int = 10) -> None:
        self.root: ExpressionNode = root if root is not None else ExpressionNode(NodeType.EMPTY, None)
        self.max_depth = max_depth
        self.construction_history: List[Dict[str, Any]] = []

    def get_next_empty_node(self) -> Optional[ExpressionNode]:
        return self._find_empty_recursive(self.root)

    def _find_empty_recursive(self, current_node: ExpressionNode) -> Optional[ExpressionNode]:
        if current_node.node_type == NodeType.EMPTY:
            return current_node
        for child in current_node.children:
            empty_node_in_child = self._find_empty_recursive(child)
            if empty_node_in_child:
                return empty_node_in_child
        return None

    def is_complete(self, grammar: 'ProgressiveGrammar') -> bool:
        return self.root.is_complete(grammar)

    def count_nodes(self) -> int:
        return self._count_recursive(self.root)

    def _count_recursive(self, current_node: ExpressionNode) -> int:
        if current_node.node_type == NodeType.EMPTY:
            return 0
        count = 1
        for child in current_node.children:
            count += self._count_recursive(child)
        return count

    def to_tensor_representation(self, grammar: 'ProgressiveGrammar', max_nodes: int = 50) -> torch.Tensor:
        feature_dim = 128
        tensor_repr = torch.zeros((max_nodes, feature_dim), dtype=torch.float32)
        nodes_to_process: deque[Tuple[ExpressionNode, int]] = deque()
        if self.root:  # Should always be true after __init__
            nodes_to_process.append((self.root, 0))
        
        next_tensor_idx = 1  # Next available row in tensor for children
        processed_node_ids: Set[int] = set()

        while nodes_to_process:
            node_obj, current_node_idx_in_tensor = nodes_to_process.popleft()

            if current_node_idx_in_tensor >= max_nodes: continue  # Tensor is full
            if id(node_obj) in processed_node_ids: continue
            processed_node_ids.add(id(node_obj))

            # Populate tensor_repr[current_node_idx_in_tensor]
            if node_obj.node_type == NodeType.EMPTY:
                tensor_repr[current_node_idx_in_tensor, 0] = 1.0
            elif node_obj.node_type == NodeType.OPERATOR:
                tensor_repr[current_node_idx_in_tensor, 1] = 1.0
                all_ops_list = sorted(list(
                    cast(Set[str], grammar.primitives.get('binary_ops', set())) |
                    cast(Set[str], grammar.primitives.get('unary_ops', set())) |
                    cast(Set[str], grammar.primitives.get('calculus_ops', set()))
                ))
                if node_obj.value in all_ops_list:
                    try:
                        op_idx = all_ops_list.index(str(node_obj.value))
                        if 10 + op_idx < feature_dim:
                            tensor_repr[current_node_idx_in_tensor, 10 + op_idx] = 1.0
                    except ValueError:
                        logging.debug(f"Operator {node_obj.value} not in grammar's known list for tensorization.")
            elif node_obj.node_type == NodeType.VARIABLE:
                tensor_repr[current_node_idx_in_tensor, 2] = 1.0
                if hasattr(node_obj.value, 'properties') and isinstance(node_obj.value.properties, dict):
                    var_props = cast(Dict[str, float], node_obj.value.properties)
                    prop_vals = [float_cast(v) for v in var_props.values()]
                    for i, p_val in enumerate(prop_vals[:10]):
                        if 50 + i < feature_dim:
                            tensor_repr[current_node_idx_in_tensor, 50 + i] = p_val
            elif node_obj.node_type == NodeType.CONSTANT:
                tensor_repr[current_node_idx_in_tensor, 3] = 1.0
                if 60 < feature_dim:
                    tensor_repr[current_node_idx_in_tensor, 60] = float_cast(node_obj.value)
            # Positional/structural features
            if 70 < feature_dim: tensor_repr[current_node_idx_in_tensor, 70] = float_cast(node_obj.depth)
            if 71 < feature_dim: tensor_repr[current_node_idx_in_tensor, 71] = float_cast(node_obj.position)
            if 72 < feature_dim: tensor_repr[current_node_idx_in_tensor, 72] = float_cast(len(node_obj.children))

            for child in node_obj.children:
                if next_tensor_idx < max_nodes:
                    nodes_to_process.append((child, next_tensor_idx))
                    next_tensor_idx += 1
                else:
                    logging.debug(f"Max_nodes ({max_nodes}) reached. Not all tree nodes included in tensor.")
                    break  # Stop adding children if tensor is full
            if next_tensor_idx >= max_nodes and nodes_to_process: break  # Stop BFS if tensor full
        return tensor_repr


# Type aliases for clarity
ActionElement = Tuple[str, Any]
ObsType = np.ndarray
InfoType = Dict[str, Any]


class SymbolicDiscoveryEnv(gym.Env[ObsType, int]):
    metadata: Dict[str, Any] = {'render.modes': ['human']}
    
    def __init__(
        self,
        grammar: 'ProgressiveGrammar',
        target_data: np.ndarray,
        variables: List['Variable'],
        max_depth: int = 10,
        max_complexity: int = 30,
        reward_config: Optional[Dict[str, float]] = None,
        max_nodes: int = 50,
        target_variable_index: Optional[int] = None,
        action_space_size: Optional[int] = None,
        provide_tree_structure: bool = False
    ):
        super().__init__()
        self.grammar: 'ProgressiveGrammar' = grammar
        self.target_data: np.ndarray = target_data
        self.variables: List['Variable'] = variables
        self.max_depth: int = max_depth
        self.max_complexity: int = max_complexity
        self.max_nodes: int = max_nodes
        self.provide_tree_structure: bool = provide_tree_structure

        if target_data.ndim != 2:
            raise ValueError(f"target_data must be a 2D array, got shape {target_data.shape}")
        
        num_total_cols = target_data.shape[1]
        if target_variable_index is None:
            self.target_variable_index: int = num_total_cols - 1
        else:
            self.target_variable_index = target_variable_index
        
        if not (0 <= self.target_variable_index < num_total_cols):
            raise ValueError(
                f"Invalid target_variable_index: {self.target_variable_index}. "
                f"Must be an int within data bounds [0, {num_total_cols - 1}]."
            )

        default_rw_config: Dict[str, float] = {
            'completion_bonus': 0.1, 'validity_bonus': 0.05, 'mse_weight': 1.0,
            'mse_scale_factor': 1.0, 'complexity_penalty': -0.01,
            'depth_penalty': -0.001, 'timeout_penalty': -1.0,
        }
        self.reward_config: Dict[str, float] = {**default_rw_config, **(reward_config or {})}
        
        self.current_state: TreeState = TreeState(max_depth=self.max_depth)
        self.steps_taken: int = 0
        self.max_steps_per_episode: int = 100
        
        self._evaluation_cache: Dict[str, Any] = {}
        
        self.action_to_element: List[ActionElement] = _build_action_space(self.grammar, self.variables)
        num_actions = action_space_size if action_space_size is not None else len(self.action_to_element)
        self.action_space = spaces.Discrete(num_actions)  # type: ignore
        
        obs_shape = (self.max_nodes * 128,)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32)  # type: ignore
        
        self.strict_mode_validator: Optional['StrictModeValidator'] = None
        self.strict_mode_active: bool = False

    def set_strict_mode_validator(self, validator: Optional['StrictModeValidator'], active: bool) -> None:
        self.strict_mode_validator = validator
        self.strict_mode_active = active

    def validate_target_data_if_strict(self) -> None:
        if self.strict_mode_active and self.strict_mode_validator and self.target_data is not None:
            try:
                self.strict_mode_validator.validate_data(
                    self.target_data,
                    data_name="SymbolicDiscoveryEnv.target_data",
                    expected_dims=2
                )
            except Exception as e_svd:
                raise ValueError(f"Strict mode validation failed for SymbolicDiscoveryEnv.target_data: {e_svd}") from e_svd
        elif self.strict_mode_active and self.strict_mode_validator and self.target_data is None:
            if hasattr(self.strict_mode_validator, '_handle_error'):
                cast('StrictModeValidator', self.strict_mode_validator)._handle_error(  # type: ignore
                    "SymbolicDiscoveryEnv.validate_target_data_if_strict called but target_data is None."
                )
            else:
                logging.warning("SymbolicDiscoveryEnv.validate_target_data_if_strict called but target_data is None, and validator has no _handle_error.")

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[ObsType, InfoType]:
        super().reset(seed=seed)
        if seed is not None:
            torch.manual_seed(seed)
        
        self.current_state = TreeState(max_depth=self.max_depth)
        self.steps_taken = 0
        self._evaluation_cache.clear()
        
        obs: ObsType = self._get_observation()
        info: InfoType = self._get_info()
        return obs, info

    def step(self, action: int) -> Tuple[ObsType, float, bool, bool, InfoType]:
        self.steps_taken += 1
        empty_node: Optional[ExpressionNode] = self.current_state.get_next_empty_node()
        
        reward: float = 0.0
        terminated: bool = False
        truncated: bool = False

        if not empty_node:
            terminated = True
            reward = self.reward_config.get('timeout_penalty', -1.0)
            obs = self._get_observation()
            info = self._get_info()
            info['error'] = "No empty node found to apply action."
            return obs, reward, terminated, truncated, info

        action_type_str: str
        action_val: Any
        try:
            action_type_str, action_val = self.action_to_element[action]
        except IndexError:
            terminated = True
            reward = self.reward_config.get('timeout_penalty', -2.0)
            obs = self._get_observation()
            info = self._get_info()
            info['error'] = f"Invalid action index: {action}."
            return obs, reward, terminated, truncated, info

        if not self._is_valid_action(empty_node, action_type_str, action_val):
            reward = -0.05
        else:
            self._apply_action(empty_node, action_type_str, action_val)
            if self.current_state.is_complete(self.grammar):
                reward = self._evaluate_expression()
                terminated = True
            else:
                reward = self.reward_config.get('validity_bonus', 0.0)
        
        if self.steps_taken >= self.max_steps_per_episode:
            truncated = True
            if not terminated:
                reward += self.reward_config.get('timeout_penalty', 0.0)
        
        obs = self._get_observation()
        info = self._get_info()
        return obs, reward, terminated, truncated, info

    def _is_valid_action(self, node: ExpressionNode, action_type: str, action_value: Any) -> bool:
        if node.depth >= self.max_depth:
            if action_type == 'operator' or action_type == 'function':
                return False
        
        # Additional validation: check if operator is known to grammar
        if action_type == 'operator' or action_type == 'function':
            if hasattr(self.grammar, 'is_operator_known'):
                if not self.grammar.is_operator_known(str(action_value)):
                    return False
        
        return True

    def _apply_action(self, node_to_fill: ExpressionNode, action_type: str, action_value: Any) -> None:
        node_to_fill.value = action_value

        if action_type == 'operator' or action_type == 'function':
            node_to_fill.node_type = NodeType.OPERATOR
            num_expected = self.grammar.get_arity(str(action_value))
            node_to_fill.children = []
            for i in range(num_expected):
                child = ExpressionNode(NodeType.EMPTY, None, parent=node_to_fill, depth=node_to_fill.depth + 1, position=i)
                node_to_fill.children.append(child)
        
        elif action_type == 'variable':
            node_to_fill.node_type = NodeType.VARIABLE
        elif action_type == 'constant':
            node_to_fill.node_type = NodeType.CONSTANT
            if action_value == 'random_small':
                node_to_fill.value = float(self.np_random.standard_normal())
            elif action_value == 'random_large':
                node_to_fill.value = float(self.np_random.standard_normal() * 10)
            else:
                node_to_fill.value = float(action_value)
        
        self.current_state.construction_history.append({
            'step': self.steps_taken,
            'action_type': action_type,
            'action_value': str(node_to_fill.value)
        })

    def _evaluate_expression(self) -> float:
        final_expr_obj: Optional['GrammarExpression'] = self.current_state.root.to_expression(self.grammar)
        self._evaluation_cache.clear()

        if final_expr_obj is None:
            self._evaluation_cache['error'] = "Failed to convert tree to grammar Expression"
            return self.reward_config.get('timeout_penalty', -1.0)

        if final_expr_obj.symbolic is None:
            self._evaluation_cache['error'] = "Failed to create symbolic form of the expression"
            self._evaluation_cache['expression_str'] = final_expr_obj.operator
            self._evaluation_cache['complexity'] = final_expr_obj.complexity
            return self.reward_config.get('timeout_penalty', -1.0)

        if final_expr_obj.complexity > self.max_complexity:
            self._evaluation_cache['expression_str'] = str(final_expr_obj.symbolic)
            self._evaluation_cache['complexity'] = final_expr_obj.complexity
            self._evaluation_cache['mse'] = float('inf')
            penalty = self.reward_config.get('complexity_penalty', -0.01) * (final_expr_obj.complexity - self.max_complexity)
            return penalty

        errors: List[float] = []
        X_data = np.delete(self.target_data, self.target_variable_index, axis=1)
        y_true = self.target_data[:, self.target_variable_index]

        if X_data.shape[1] != len(self.variables):
            err_msg = (f"Mismatch between X_data columns ({X_data.shape[1]}) and "
                      f"number of environment variables ({len(self.variables)}) for substitution.")
            self._evaluation_cache['error'] = err_msg
            logging.error(err_msg)
            return self.reward_config.get('timeout_penalty', -2.0)

        target_variance: float = float(np.var(y_true))
        penalty_on_fail: float = target_variance if target_variance > 1e-9 else 1.0

        for i in range(X_data.shape[0]):
            subs_dict: Dict[Any, float] = {
                var.symbolic: X_data[i, j] for j, var in enumerate(self.variables)
            }
            try:
                pred_sympy_val = final_expr_obj.symbolic.evalf(subs=subs_dict)
                pred = float(pred_sympy_val)

                if not np.isfinite(pred) or abs(pred) > 1e12:
                    errors.append(penalty_on_fail**2)
                else:
                    errors.append((pred - y_true[i])**2)
            except Exception as e:
                logging.debug(f"Sympy eval error for {final_expr_obj.symbolic} with {subs_dict}: {e}")
                errors.append(penalty_on_fail**2)

        mse: float = float(np.mean(errors)) if errors else penalty_on_fail**2
        norm_mse = mse / (target_variance + 1e-10)
        
        current_reward: float = (
            self.reward_config.get('completion_bonus', 0.1) +
            self.reward_config.get('mse_weight', 1.0) * np.exp(-self.reward_config.get('mse_scale_factor', 1.0) * norm_mse) +
            self.reward_config.get('complexity_penalty', -0.01) * final_expr_obj.complexity +
            self.reward_config.get('depth_penalty', -0.001) * self.current_state.root.depth
        )
        
        self._evaluation_cache.update({
            'expression_str': str(final_expr_obj.symbolic),
            'mse': mse,
            'complexity': final_expr_obj.complexity,
            'reward': current_reward
        })
        return current_reward

    def _get_observation(self) -> ObsType:
        tensor_repr: torch.Tensor = self.current_state.to_tensor_representation(self.grammar, max_nodes=self.max_nodes)
        return tensor_repr.flatten().numpy()

    def _get_info(self) -> InfoType:
        info: InfoType = {
            'steps_taken': self.steps_taken,
            'current_nodes': self.current_state.count_nodes(),
            'is_complete': self.current_state.is_complete(self.grammar),
        }
        info.update(self._evaluation_cache)
        # if self.provide_tree_structure:
        #     tree_structure_data = ...  # Generate and add tree structure data
        #     info['tree_structure'] = tree_structure_data
        return info

    def render(self, mode: str = 'human') -> None:
        print(f"Step: {self.steps_taken}, Nodes: {self.current_state.count_nodes()}")
        if self.current_state.is_complete(self.grammar):
            expr_obj = self.current_state.root.to_expression(self.grammar)
            if expr_obj and expr_obj.symbolic is not None:
                print(f"  Completed Expression: {expr_obj.symbolic} (Complexity: {expr_obj.complexity})")
            elif expr_obj:
                print(f"  Completed Expression (no symbolic form): Operator {expr_obj.operator}, Comp: {expr_obj.complexity}")
            else:
                print("  Current tree is complete but failed to form grammar.Expression.")
        else:
            print("  Expression under construction.")

    def get_action_mask(self) -> np.ndarray:
        empty_node: Optional[ExpressionNode] = self.current_state.get_next_empty_node()
        if not empty_node:
            return np.zeros(self.action_space.n, dtype=bool)
        
        mask = np.zeros(self.action_space.n, dtype=bool)
        for i, (action_type, action_val) in enumerate(self.action_to_element):
            if self._is_valid_action(empty_node, action_type, action_val):
                mask[i] = True
        return mask


class CurriculumManager:
    def __init__(self, base_env: SymbolicDiscoveryEnv) -> None:
        self.base_env: SymbolicDiscoveryEnv = base_env
        self.difficulty_level: int = 0
        self.success_rate_history: deque[float] = deque(maxlen=100)
        self.curriculum: List[Dict[str, int]] = [
            {'max_depth': 3, 'max_complexity': 5},
            {'max_depth': 5, 'max_complexity': 10},
            {'max_depth': 7, 'max_complexity': 15},
            {'max_depth': 10, 'max_complexity': 30},
        ]
        self.apply_curriculum_to_env()

    def get_current_env_config(self) -> Dict[str, int]:
        return self.curriculum[self.difficulty_level]

    def apply_curriculum_to_env(self) -> None:
        current_config = self.get_current_env_config()
        self.base_env.max_depth = current_config['max_depth']
        self.base_env.max_complexity = current_config['max_complexity']
        logging.info(f"CurriculumManager: Applied config for level {self.difficulty_level}: {current_config}")

    def update_curriculum(self, episode_reward: float, success_threshold: float = 0.5) -> None:
        is_success: bool = episode_reward > success_threshold
        self.success_rate_history.append(1.0 if is_success else 0.0)
        
        if len(self.success_rate_history) == self.success_rate_history.maxlen:
            current_success_rate = float(np.mean(list(self.success_rate_history)))
            
            if current_success_rate > 0.7 and self.difficulty_level < len(self.curriculum) - 1:
                self.difficulty_level += 1
                self.success_rate_history.clear()
                self.apply_curriculum_to_env()
                logging.info(f"CurriculumManager: Increased difficulty to level {self.difficulty_level}")
            elif current_success_rate < 0.3 and self.difficulty_level > 0:
                self.difficulty_level -= 1
                self.success_rate_history.clear()
                self.apply_curriculum_to_env()
                logging.info(f"CurriculumManager: Decreased difficulty to level {self.difficulty_level}")


def _build_action_space(grammar: 'ProgressiveGrammar', variables: List['Variable']) -> List[ActionElement]:
    """Build the action space from grammar and variables."""
    actions: List[ActionElement] = []
    
    # Add operators
    for op in grammar.primitives.get('binary_ops', set()):
        actions.append(('operator', op))
    for op in grammar.primitives.get('unary_ops', set()):
        actions.append(('function', op))
    for op in grammar.primitives.get('calculus_ops', set()):
        actions.append(('function', op))
    
    # Add variables
    for var in variables:
        actions.append(('variable', var))
    
    # Add constants
    actions.extend([
        ('constant', 0.0),
        ('constant', 1.0),
        ('constant', -1.0),
        ('constant', 2.0),
        ('constant', 0.5),
        ('constant', 'random_small'),
        ('constant', 'random_large'),
    ])
    
    return actions


if __name__ == "__main__":
    from janus.core.grammar import ProgressiveGrammar as MainGrammar
    from janus.core.expression import Variable as MainVariable

    logging.basicConfig(level=logging.INFO)

    example_grammar_instance: MainGrammar = MainGrammar()
    var_x_instance: MainVariable = MainVariable("x", 0, {"smoothness": 0.9})
    example_vars_list: List[MainVariable] = [var_x_instance]
    
    n_s = 100
    x_d = np.random.randn(n_s)
    y_d = 2 * x_d + 1 + 0.1 * np.random.randn(n_s)
    
    full_data_arr = np.column_stack([x_d, y_d])
    
    env_instance = SymbolicDiscoveryEnv(
        grammar=example_grammar_instance,
        target_data=full_data_arr,
        variables=example_vars_list,
        target_variable_index=1,
        max_nodes=50
    )
    
    print(f"Environment Action Space Size: {env_instance.action_space.n}")
    print(f"Environment Observation Space Shape: {env_instance.observation_space.shape}")

    obs_arr, info_d = env_instance.reset(seed=42)
    print(f"Initial Observation Shape: {obs_arr.shape}")
    print(f"Initial Info: {info_d}")

    for i_step in range(5):
        action_mask_ndarr = env_instance.get_action_mask()
        valid_actions_arr = np.where(action_mask_ndarr)[0]
        if len(valid_actions_arr) == 0:
            print(f"Step {i_step+1}: No valid actions. Tree might be complete or stuck.")
            break
        
        chosen_act: int
        if hasattr(env_instance, 'np_random') and env_instance.np_random is not None:
            chosen_act = env_instance.np_random.choice(valid_actions_arr)
        else:
            chosen_act = np.random.choice(valid_actions_arr)

        print(f"\nStep {i_step+1}: Taking action {chosen_act} ({env_instance.action_to_element[chosen_act]})")
        
        obs_arr, reward_f, terminated_b, truncated_b, info_d = env_instance.step(chosen_act)
        
        print(f"  Observation Shape: {obs_arr.shape}")
        print(f"  Reward: {reward_f:.4f}")
        print(f"  Terminated: {terminated_b}, Truncated: {truncated_b}")
        print(f"  Info: {info_d}")
        env_instance.render()

        if terminated_b or truncated_b:
            print(f"Episode finished at step {i_step+1}.")
            break


__all__ = ["SymbolicDiscoveryEnv", "CurriculumManager", "ExpressionNode", "TreeState", "NodeType"]