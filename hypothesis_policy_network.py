"""
HypothesisNet Policy Network
============================

Neural policy for intelligent hypothesis generation in the SymbolicDiscoveryEnv.
Implements both TreeLSTM and Transformer variants with action masking.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from typing import Dict, List, Tuple, Optional, Deque, Any, Union
from math_utils import safe_env_reset, safe_import
from symbolic_discovery_env import SymbolicDiscoveryEnv # Ensure this is imported if needed, or adjust if not.
from progressive_grammar_system import ProgressiveGrammar, Variable # Import from correct file
from dataclasses import dataclass
import torch.optim as optim
from collections import deque

# Use safe_import for wandb
wandb = safe_import("wandb", "wandb")
# HAS_WANDB = wandb is not None # Not strictly needed here as wandb is not directly used with guards


class TreeLSTMCell(nn.Module):
    """Tree-structured LSTM cell for processing expression trees."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.W_i = nn.Linear(input_dim, hidden_dim)
        self.U_i = nn.Linear(hidden_dim, hidden_dim)
        self.W_f = nn.Linear(input_dim, hidden_dim)
        self.U_f = nn.Linear(hidden_dim, hidden_dim)
        self.W_o = nn.Linear(input_dim, hidden_dim)
        self.U_o = nn.Linear(hidden_dim, hidden_dim)
        self.W_c = nn.Linear(input_dim, hidden_dim)
        self.U_c = nn.Linear(hidden_dim, hidden_dim)

    def forward(self,
                x: torch.Tensor,
                children_h: List[torch.Tensor],
                children_c: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        if not children_h:
            h_sum = torch.zeros(x.size(0), self.hidden_dim, device=x.device)
        else:
            h_sum = sum(children_h)

        i = torch.sigmoid(self.W_i(x) + self.U_i(h_sum))
        o = torch.sigmoid(self.W_o(x) + self.U_o(h_sum))
        c_tilde = torch.tanh(self.W_c(x) + self.U_c(h_sum))

        c = i * c_tilde
        if children_c:
            for child_c, child_h in zip(children_c, children_h):
                f = torch.sigmoid(self.W_f(x) + self.U_f(child_h))
                c = c + f * child_c
        h = o * torch.tanh(c)
        return h, c

class TreeEncoder(nn.Module):
    def __init__(self, node_feature_dim: int, hidden_dim: int, n_layers: int = 2):
        super().__init__()
        self.node_embedding = nn.Linear(node_feature_dim, hidden_dim)
        self.tree_cells = nn.ModuleList([
            TreeLSTMCell(hidden_dim, hidden_dim) for _ in range(n_layers)
        ])
        self.hidden_dim = hidden_dim

    def forward(self, tree_features: torch.Tensor, tree_structure: Optional[Dict[int, List[int]]] = None) -> torch.Tensor:
        batch_size, max_nodes, _ = tree_features.shape
        node_embeds = self.node_embedding(tree_features)

        if tree_structure is None:
            h_prev_layer = node_embeds
            for cell in self.tree_cells:
                h_current_layer_nodes = []
                # This simplified sequential processing might not be what's intended for TreeLSTM.
                # Typically, TreeLSTM processes based on tree structure, not just sequence.
                # However, adhering to the existing fallback logic if tree_structure is None.
                for i in range(max_nodes):
                    # For sequential, pass empty children list
                    h_node, _ = cell(h_prev_layer[:, i, :], [], [])
                    h_current_layer_nodes.append(h_node)
                h_prev_layer = torch.stack(h_current_layer_nodes, dim=1)
            # The output should be a single vector per tree in the batch, typically the root or an aggregate.
            return torch.mean(h_prev_layer, dim=1)


        # Structured processing
        memo: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]] = {} # (layer_idx, node_idx) -> (h, c)

        def get_hc(layer_idx: int, node_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
            if (layer_idx, node_idx) in memo:
                return memo[(layer_idx, node_idx)]

            node_feature_vector = node_embeds[:, node_idx, :]

            if layer_idx == 0: # First layer uses original node embeddings as input to cell
                current_cell_input = node_feature_vector
                children_h_prev_layer, children_c_prev_layer = [], []
            else: # Subsequent layers use hidden states from previous layer
                current_cell_input, _ = get_hc(layer_idx - 1, node_idx) # Use h from previous layer
                children_h_prev_layer, children_c_prev_layer = [], []


            children_indices = tree_structure.get(node_idx, [])
            current_layer_children_h: List[torch.Tensor] = []
            current_layer_children_c: List[torch.Tensor] = []

            for child_idx in children_indices:
                # Children's h and c for the current cell must come from the *same* layer_idx
                h_child, c_child = get_hc(layer_idx, child_idx)
                current_layer_children_h.append(h_child)
                current_layer_children_c.append(c_child)

            # For the cell input (x), if layer_idx > 0, it should be h from previous layer.
            # If layer_idx == 0, it's the original node_embeds.
            # The U matrices operate on sum of children's h from the *same* layer.

            # The input to the cell's W matrices is h_node from the previous layer (or embedding if layer 0)
            # The input to the cell's U matrices is sum of h_children from the current layer

            cell_input_x = current_cell_input

            h, c = self.tree_cells[layer_idx](cell_input_x, current_layer_children_h, current_layer_children_c)
            memo[(layer_idx, node_idx)] = (h, c)
            return h, c

        # Assume root node is index 0 for all trees in batch, or it's the last node in topological sort
        # For simplicity, let's assume node 0 is always present and is a root-like node to return.
        # A more robust way would be to identify actual roots if they can vary.
        num_layers = len(self.tree_cells)

        # Populate memo table layer by layer, node by node
        for layer_idx in range(num_layers):
            for node_idx in range(max_nodes): # Assumes post-order traversal if structure is used
                # Nodes need to be processed in an order that their children are already processed for that layer
                # This requires a defined traversal (e.g. post-order) for each layer
                # The current simple loop might not respect this if tree_structure is complex.
                # However, typical TreeLSTM implementations process nodes recursively (which implies post-order).

                # Let's refine the traversal for structured processing:
                # We need to compute for all nodes, layer by layer.
                # Within a layer, for a node, its children for *that same layer* must be computed first.
                pass # The recursive get_hc should handle the correct order.

        # Get the hidden state of the root node (assumed to be node 0) from the last layer
        root_h, _ = get_hc(num_layers - 1, 0) # Root node index 0
        return root_h


class TransformerEncoder(nn.Module):
    def __init__(self, node_feature_dim: int, hidden_dim: int, n_heads: int = 8, n_layers: int = 6):
        super().__init__()
        self.node_embedding = nn.Linear(node_feature_dim, hidden_dim)
        self.positional_encoding = nn.Parameter(torch.randn(1, 100, hidden_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 4,
            dropout=0.1, activation='gelu', batch_first=True, norm_first=True # Added norm_first for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.hidden_dim = hidden_dim

    def forward(self, tree_features: torch.Tensor) -> torch.Tensor:
        batch_size, max_nodes, _ = tree_features.shape
        node_embeds = self.node_embedding(tree_features)
        node_embeds = node_embeds + self.positional_encoding[:, :max_nodes, :]
        padding_mask = (tree_features.sum(dim=-1) == 0) # True for padding

        encoded = self.transformer(node_embeds, src_key_padding_mask=padding_mask)

        # Masked mean pooling: sum only non-padded embeddings
        mask_expanded = (~padding_mask).unsqueeze(-1).float() # (B, N, 1), 1 for non-pad, 0 for pad
        sum_embeddings = (encoded * mask_expanded).sum(dim=1) # Sums features of non-padded tokens
        num_non_padded_tokens = mask_expanded.sum(dim=1).clamp(min=1e-9) # Counts non-padded tokens per item in batch

        tree_repr = sum_embeddings / num_non_padded_tokens
        return tree_repr

class HypothesisNet(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 256,
                 encoder_type: str = 'transformer', grammar: Optional['ProgressiveGrammar'] = None,
                 debug: bool = False, use_meta_learning: bool = False):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.grammar = grammar
        self.debug = debug
        self.use_meta_learning = use_meta_learning
        self.node_feature_dim = 128

        if encoder_type == 'transformer':
            self.encoder = TransformerEncoder(self.node_feature_dim, hidden_dim)
        else:
            # Pass tree_structure to TreeEncoder if it needs it for initialization, though typically it's a forward arg.
            self.encoder = TreeEncoder(self.node_feature_dim, hidden_dim, n_layers=2) # Adjusted n_layers to match TreeLSTMCell example

        self.task_encoder = None
        self.task_modulator = None
        if self.use_meta_learning:
            self.task_encoder = nn.LSTM(
                input_size=self.node_feature_dim, hidden_size=self.hidden_dim // 2,
                batch_first=True, bidirectional=True, num_layers=1 # Simplified LSTM
            )
            self.task_modulator = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim), # From concatenated LSTM hidden states
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim * 2)
            )

        self.policy_net = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        ])
        self.value_net = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        ])
        self.action_embeddings = nn.Embedding(action_dim, 64)
        # self.grammar_context = nn.LSTM(input_size=64, hidden_size=128, num_layers=2, batch_first=True) # Currently unused

    def forward(self, observation: torch.Tensor,
                action_mask: Optional[torch.Tensor] = None,
                task_trajectories: Optional[torch.Tensor] = None,
                tree_structure: Optional[Dict[int, List[int]]] = None # Added for TreeEncoder
                ) -> Dict[str, Union[torch.Tensor, Optional[torch.Tensor]]]:
        batch_size, obs_dim = observation.shape
        if obs_dim % self.node_feature_dim != 0:
            raise ValueError("Observation dimension must be a multiple of node_feature_dim.")
        num_nodes = obs_dim // self.node_feature_dim
        tree_features = observation.view(batch_size, num_nodes, self.node_feature_dim)

        if isinstance(self.encoder, TreeEncoder):
            tree_repr = self.encoder(tree_features, tree_structure)
        else: # TransformerEncoder
            tree_repr = self.encoder(tree_features)

        task_embedding_vector = None
        gains_policy = torch.ones(batch_size, self.hidden_dim // 2, device=tree_repr.device)
        biases_policy = torch.zeros(batch_size, self.hidden_dim // 2, device=tree_repr.device)
        gains_value = torch.ones(batch_size, self.hidden_dim // 2, device=tree_repr.device)
        biases_value = torch.zeros(batch_size, self.hidden_dim // 2, device=tree_repr.device)

        if self.use_meta_learning and self.task_encoder is not None and self.task_modulator is not None and task_trajectories is not None:
            bt_size, num_traj, traj_len, feat_dim = task_trajectories.shape
            if bt_size == 0: # Handle empty task_trajectories
                 pass # Keep default gains/biases
            else:
                task_trajectories_reshaped = task_trajectories.view(bt_size * num_traj, traj_len, feat_dim)
                # Ensure LSTM input is not empty if bt_size > 0
                if task_trajectories_reshaped.shape[0] > 0:
                    _, (hidden, _) = self.task_encoder(task_trajectories_reshaped)
                    # hidden for bidir LSTM: (num_layers*2, B*NumTraj, H_lstm/2)
                    # H_lstm = self.hidden_dim // 2. So hidden is (2, B*NumTraj, H_lstm/2) for num_layers=1
                    # Concatenate forward (hidden[0]) and backward (hidden[1])
                    hidden_concat = torch.cat((hidden[0,:,:], hidden[1,:,:]), dim=-1) # (B*NumTraj, H_lstm) which is (B*NumTraj, H/2)
                                                                                       # Wait, task_encoder hidden_size is H/2. BiLSTM makes it H.
                                                                                       # So hidden is (2, B*NumTraj, H/2). hidden.transpose(0,1).reshape(B*NumTraj, H)
                    hidden = hidden.transpose(0, 1).reshape(bt_size * num_traj, self.hidden_dim) # (B*NumTraj, H)
                    task_embedding_vector = hidden.view(bt_size, num_traj, self.hidden_dim).mean(dim=1) # (B, H)

                    modulation_params = self.task_modulator(task_embedding_vector) # (B, H*2)
                    all_gains_raw, all_biases_raw = modulation_params.chunk(2, dim=-1) # Each (B, H)

                    gains_policy_raw, gains_value_raw = all_gains_raw.chunk(2, dim=-1) # Each (B, H/2)
                    biases_policy_raw, biases_value_raw = all_biases_raw.chunk(2, dim=-1)

                    gains_policy = 1 + 0.1 * torch.tanh(gains_policy_raw)
                    biases_policy = 0.1 * torch.tanh(biases_policy_raw)
                    gains_value = 1 + 0.1 * torch.tanh(gains_value_raw)
                    biases_value = 0.1 * torch.tanh(biases_value_raw)

        x_policy = tree_repr
        modulated_policy = False
        for i, layer in enumerate(self.policy_net):
            x_policy = layer(x_policy)
            if isinstance(layer, nn.ReLU) and self.use_meta_learning and task_trajectories is not None and not modulated_policy:
                if x_policy.shape[-1] == self.hidden_dim // 2:
                    x_policy = x_policy * gains_policy + biases_policy
                    modulated_policy = True
        policy_logits = x_policy

        x_value = tree_repr
        modulated_value = False
        for i, layer in enumerate(self.value_net):
            x_value = layer(x_value)
            if isinstance(layer, nn.ReLU) and self.use_meta_learning and task_trajectories is not None and not modulated_value:
                if x_value.shape[-1] == self.hidden_dim // 2:
                    x_value = x_value * gains_value + biases_value
                    modulated_value = True
        value = x_value

        masked_logits = policy_logits
        if action_mask is not None:
            # Ensure action_mask is same batch_size as policy_logits
            if action_mask.shape[0] == policy_logits.shape[0]:
                 masked_logits = policy_logits.clone() # Ensure clone if modifying
                 masked_logits[~action_mask] = -1e9 # Apply mask where True indicates valid
            # else: if debug: print("Action mask shape mismatch") # Optional: for debugging

        action_logits = masked_logits.nan_to_num(nan=-1e9, posinf=1e9, neginf=-1e9)

        return_dict: Dict[str, Union[torch.Tensor, Optional[torch.Tensor]]] = {
            'policy_logits': policy_logits, 'action_logits': action_logits,
            'value': value, 'tree_representation': tree_repr
        }
        if self.use_meta_learning and task_embedding_vector is not None:
            return_dict['task_embedding'] = task_embedding_vector
        return return_dict

    def get_action(self, observation: torch.Tensor,
                   action_mask: Optional[torch.Tensor] = None,
                   task_trajectories: Optional[torch.Tensor] = None,
                   tree_structure: Optional[Dict[int, List[int]]] = None, # Added
                   deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: # Return action as tensor
        outputs = self.forward(observation, action_mask, task_trajectories, tree_structure) # Pass tree_structure
        dist = Categorical(logits=outputs['action_logits'])
        action = torch.argmax(outputs['action_logits'], dim=-1) if deterministic else dist.sample()
        log_prob = dist.log_prob(action) # action is shape (B,)

        if observation.shape[0] == 1: # If batch size is 1
            action = action.squeeze(0)    # (1,) -> ()
            log_prob = log_prob.squeeze(0)  # (1,) -> ()
            # value from forward is (B, 1). If B=1, it's (1,1), which matches the test. So no change to value.

        return action, log_prob, outputs['value'] # Return action tensor

class PPOTrainer:
    def __init__(self, policy: HypothesisNet, env: 'SymbolicDiscoveryEnv', lr: float = 3e-4,
                 gamma: float = 0.99, gae_lambda: float = 0.95, clip_epsilon: float = 0.2,
                 value_coef: float = 0.5, entropy_coef: float = 0.01, max_grad_norm: float = 0.5):
        self.policy = policy
        self.env = env # Make sure env has get_action_mask and other required methods.
        self.optimizer = optim.Adam(policy.parameters(), lr=lr)
        self.gamma, self.gae_lambda, self.clip_epsilon = gamma, gae_lambda, clip_epsilon
        self.value_coef, self.entropy_coef, self.max_grad_norm = value_coef, entropy_coef, max_grad_norm
        self.rollout_buffer = RolloutBuffer()
        self.episode_rewards: Deque[float] = deque(maxlen=100)
        # Removed episode_complexities and episode_mse as they were not consistently updated/used.

    def collect_rollouts(self, n_steps: int, task_trajectories: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        self.rollout_buffer.reset()
        obs, info = safe_env_reset(self.env) # Use safe_env_reset
        # tree_structure may come from info if env provides it
        tree_structure = info.get('tree_structure') if isinstance(info, dict) else None


        for _ in range(n_steps):
            obs_tensor = torch.FloatTensor(np.array(obs)).unsqueeze(0) # Ensure obs is array for FloatTensor
            action_mask_np = self.env.get_action_mask() # (action_dim,)
            action_mask_tensor = torch.BoolTensor(action_mask_np).unsqueeze(0) # (1, action_dim)

            with torch.no_grad():
                action_tensor, log_prob, value = self.policy.get_action(
                    obs_tensor, action_mask_tensor,
                    task_trajectories=task_trajectories, # Pass along
                    tree_structure=tree_structure
                )
            action_item = action_tensor.item() # For env.step
            next_obs, reward, terminated, truncated, info = self.env.step(action_item)
            next_tree_structure = info.get('tree_structure') if isinstance(info, dict) else None

            self.rollout_buffer.add(obs, action_item, reward, value.item(), log_prob.item(), terminated or truncated, action_mask_np, tree_structure)
            obs, tree_structure = next_obs, next_tree_structure

            if terminated or truncated:
                ep_rew = info.get('episode_reward', reward) # Fallback to current reward if not in info
                self.episode_rewards.append(ep_rew)
                obs, info = safe_env_reset(self.env) # Use safe_env_reset
                tree_structure = info.get('tree_structure') if isinstance(info, dict) else None


        self.rollout_buffer.compute_returns_and_advantages(self.policy, self.gamma, self.gae_lambda, task_trajectories=task_trajectories)
        return self.rollout_buffer.get()

    def train_step(self, batch: Dict[str, torch.Tensor], task_trajectories_batch: Optional[torch.Tensor] = None) -> Dict[str, float]:
        obs, actions, old_log_probs = batch['observations'], batch['actions'], batch['log_probs']
        advantages, returns, action_masks = batch['advantages'], batch['returns'], batch['action_masks']
        tree_structures_batch = batch.get('tree_structures') # Might be None if not stored or all None

        # This part is tricky: if tree_structures is a list of dicts, it can't be a tensor.
        # For batch processing, policy needs to handle either a single structure if shared, or None if varying / not used by encoder.
        # For now, let's assume TransformerEncoder (no tree_structure needed) or TreeEncoder handles tree_structures=None.
        # A more robust solution would involve padding/batching tree structures if they vary.

        outputs = self.policy(obs, action_masks, task_trajectories=task_trajectories_batch, tree_structure=None) # Pass None for tree_structure for now

        dist = Categorical(logits=outputs['action_logits'])
        log_probs = dist.log_prob(actions)
        ratio = torch.exp(log_probs - old_log_probs)
        surr1, surr2 = ratio * advantages, torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        value_pred = outputs['value'].squeeze(-1) # Ensure value_pred is 1D
        value_loss = F.mse_loss(value_pred, returns)
        entropy = dist.entropy().mean()
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

        self.optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm); self.optimizer.step()
        return {'loss': loss.item(), 'policy_loss': policy_loss.item(), 'value_loss': value_loss.item(), 'entropy': entropy.item()}

    def train(self, total_timesteps: int, rollout_length: int = 2048, n_epochs: int = 10,
              batch_size: int = 64, log_interval: int = 10,
              # task_trajectories_for_rollout: Optional[torch.Tensor] = None, # For entire training
              # task_trajectories_for_batch: Optional[torch.Tensor] = None # If fixed for all batches
             ):
        n_updates = total_timesteps // rollout_length
        for update in range(1, n_updates + 1):
            # In a real meta-PPO, task_trajectories might change per update or be sampled with rollouts
            rollout_data = self.collect_rollouts(rollout_length, task_trajectories=None) # Pass None for now

            num_samples_in_rollout = len(rollout_data['observations'])
            if num_samples_in_rollout == 0:
                if update % log_interval == 0:
                    avg_reward = np.mean(list(self.episode_rewards)) if self.episode_rewards else float('nan')
                    print(f"Update {update}/{n_updates}, Avg Reward: {avg_reward:.3f}. No samples in rollout. Skipping training for this update.")
                return # Return from train method if no samples

            data_indices = np.arange(num_samples_in_rollout)

            for epoch_num in range(n_epochs):  # Loop for n_epochs, added epoch_num for logging
                np.random.shuffle(data_indices)  # Shuffle indices at the beginning of each epoch

                for start_idx in range(0, num_samples_in_rollout, batch_size):
                    mini_batch_indices_np = data_indices[start_idx : start_idx + batch_size]

                    if len(mini_batch_indices_np) == 0:
                        continue

                    mini_batch_indices_list = mini_batch_indices_np.tolist()

                    batch = {}
                    for key, value_from_rollout in rollout_data.items():
                        if key == 'tree_structures':
                            batch[key] = [value_from_rollout[i] for i in mini_batch_indices_list]
                        elif isinstance(value_from_rollout, torch.Tensor):
                            batch[key] = value_from_rollout[mini_batch_indices_np]
                        else:
                            try:
                                # Attempt to index with numpy array first (e.g. if it's a list of numpy arrays)
                                batch[key] = value_from_rollout[mini_batch_indices_np]
                            except (TypeError, IndexError): # More specific exceptions
                                try:
                                    # Fallback to list indexing if numpy array indexing failed
                                    batch[key] = [value_from_rollout[i] for i in mini_batch_indices_list]
                                except Exception as e_inner: # Catch any other exception during list indexing
                                    print(f"Warning: Could not create batch for key '{key}' (type: {type(value_from_rollout)}). Error: {e_inner}. Skipping this key for the batch.")

                    if batch and 'observations' in batch and len(batch['observations']) > 0: # Ensure essential keys are present and batch is not empty
                        # If RolloutBuffer stored task_trajs, extract batch['task_trajectories_batch'] here
                        self.train_step(batch, task_trajectories_batch=None) # Pass None for now
                    elif 'observations' not in batch or len(batch['observations']) == 0 :
                        print(f"Skipping train_step due to empty or invalid batch for update {update}, epoch {epoch_num}, start_idx {start_idx}")

            if update % log_interval == 0:
                avg_reward = np.mean(list(self.episode_rewards)) if self.episode_rewards else float('nan')
                print(f"Update {update}/{n_updates}, Avg Reward: {avg_reward:.3f}")

class RolloutBuffer:
    def __init__(self): self.reset()
    def reset(self):
        self.observations, self.actions, self.rewards, self.values = [], [], [], []
        self.log_probs, self.dones, self.action_masks = [], [], []
        self.tree_structures: List[Optional[Dict[int,List[int]]]] = [] # Added to store tree_structures
        self.advantages, self.returns = None, None

    def add(self, obs, action, reward, value, log_prob, done, action_mask, tree_structure):
        self.observations.append(obs); self.actions.append(action); self.rewards.append(reward)
        self.values.append(value); self.log_probs.append(log_prob); self.dones.append(done)
        self.action_masks.append(action_mask)
        self.tree_structures.append(tree_structure) # Store it

    def compute_returns_and_advantages(self, policy: HypothesisNet, gamma: float, gae_lambda: float,
                                       task_trajectories: Optional[torch.Tensor] = None):
        rewards = np.array(self.rewards, dtype=np.float32)
        values = np.array(self.values, dtype=np.float32)
        dones = np.array(self.dones, dtype=np.float32)

        last_obs_np = np.array(self.observations[-1])
        last_obs = torch.FloatTensor(last_obs_np).unsqueeze(0)
        last_action_mask_np = np.array(self.action_masks[-1])
        last_action_mask = torch.BoolTensor(last_action_mask_np).unsqueeze(0)
        last_tree_structure = self.tree_structures[-1]

        with torch.no_grad():
            last_value_tensor = policy(last_obs, last_action_mask, task_trajectories=task_trajectories, tree_structure=last_tree_structure)['value']
            last_value = last_value_tensor.squeeze().item() # Squeeze if (1,1)

        advantages = np.zeros_like(rewards)
        last_gae_lam = 0
        for t in reversed(range(len(rewards))):
            next_non_terminal = 1.0 - dones[t]
            # next_value = values[t+1] if t + 1 < len(values) else last_value # Incorrect: if done, next_value is 0
            # Correct GAE:
            if t == len(rewards) - 1: # Last step in buffer
                next_value = last_value # Value of state after this buffer ends
            else:
                next_value = values[t+1] # Value of S_t+1 from buffer

            delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
            advantages[t] = last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam

        self.returns = advantages + values
        self.advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)


    def get(self) -> Dict[str, Any]: # Value can be list of dicts for tree_structures
        obs_tensor = torch.FloatTensor(np.array(self.observations))
        action_masks_tensor = torch.BoolTensor(np.array(self.action_masks))

        return {'observations': obs_tensor, 'actions': torch.LongTensor(self.actions),
                'rewards': torch.FloatTensor(self.rewards), 'values': torch.FloatTensor(self.values),
                'log_probs': torch.FloatTensor(self.log_probs), 'advantages': torch.FloatTensor(self.advantages),
                'returns': torch.FloatTensor(self.returns), 'action_masks': action_masks_tensor,
                'tree_structures': self.tree_structures # Keep as list of dicts/None
                }

if __name__ == "__main__":
    from symbolic_discovery_env import SymbolicDiscoveryEnv # Keep this
    from progressive_grammar_system import ProgressiveGrammar, Variable # Import from correct file
    grammar = ProgressiveGrammar(); variables = [Variable("x",0,{}), Variable("v",1,{})]
    data = np.column_stack([np.random.randn(100), np.random.randn(100)*2, np.random.randn(100)])
    # Ensure SymbolicDiscoveryEnv can provide 'tree_structure' in info dict from reset/step if TreeEncoder is used.
    env = SymbolicDiscoveryEnv(grammar, data, variables, max_depth=5, max_complexity=10, provide_tree_structure=True) # Added flag

    obs_dim, action_dim = env.observation_space.shape[0], env.action_space.n

    # Test Transformer
    policy_transformer = HypothesisNet(obs_dim, action_dim, grammar=grammar, use_meta_learning=True, encoder_type='transformer')
    print(f"Policy (Transformer, Meta) params: {sum(p.numel() for p in policy_transformer.parameters())}")
    # obs_info_tuple = env.reset() # Env might return (obs, info)
    # obs, info = obs_info_tuple if isinstance(obs_info_tuple, tuple) else (obs_info_tuple, {})
    obs, info = safe_env_reset(env) # Use safe_env_reset here

    obs_tensor = torch.FloatTensor(np.array(obs)).unsqueeze(0)
    action_mask_np = env.get_action_mask()
    action_mask_tensor = torch.BoolTensor(action_mask_np).unsqueeze(0)
    dummy_trajs = torch.randn(1, 3, 10, policy_transformer.node_feature_dim)

    outputs_transformer_meta = policy_transformer(obs_tensor, action_mask_tensor, task_trajectories=dummy_trajs)
    print("Outputs (Transformer, Meta, with trajectories):")
    for k, v_ in outputs_transformer_meta.items(): print(f"  {k}: {v_.shape if isinstance(v_, torch.Tensor) else v_}")

    # Test TreeLSTM
    # policy_tree = HypothesisNet(obs_dim, action_dim, grammar=grammar, use_meta_learning=True, encoder_type='treelstm')
    # print(f"\nPolicy (TreeLSTM, Meta) params: {sum(p.numel() for p in policy_tree.parameters())}")
    # tree_structure_example = info.get('tree_structure') # Get from env
    # outputs_tree_meta = policy_tree(obs_tensor, action_mask_tensor, task_trajectories=dummy_trajs, tree_structure=tree_structure_example)
    # print("Outputs (TreeLSTM, Meta, with trajectories):")
    # for k, v_ in outputs_tree_meta.items(): print(f"  {k}: {v_.shape if isinstance(v_, torch.Tensor) else v_}")

    print("\nTesting PPO with Transformer meta-policy...")
    trainer_transformer_meta = PPOTrainer(policy_transformer, env)
    # Pass task_trajectories=None to train method, which will propagate to collect_rollouts
    trainer_transformer_meta.train(total_timesteps=256, rollout_length=64, n_epochs=1, batch_size=32, log_interval=1)
    print("PPO with Transformer meta-policy finished.")
