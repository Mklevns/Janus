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
from typing import Dict, List, Tuple, Optional, Deque, Any
from dataclasses import dataclass
import torch.optim as optim
from collections import deque
import wandb


class TreeLSTMCell(nn.Module):
    """Tree-structured LSTM cell for processing expression trees."""
    
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Input gate
        self.W_i = nn.Linear(input_dim, hidden_dim)
        self.U_i = nn.Linear(hidden_dim, hidden_dim)
        
        # Forget gates (one per child)
        self.W_f = nn.Linear(input_dim, hidden_dim)
        self.U_f = nn.Linear(hidden_dim, hidden_dim)
        
        # Output gate
        self.W_o = nn.Linear(input_dim, hidden_dim)
        self.U_o = nn.Linear(hidden_dim, hidden_dim)
        
        # Cell gate
        self.W_c = nn.Linear(input_dim, hidden_dim)
        self.U_c = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, 
                x: torch.Tensor, 
                children_h: List[torch.Tensor],
                children_c: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for tree LSTM.
        x: Node features (batch_size, input_dim)
        children_h: Hidden states from children
        children_c: Cell states from children
        """
        if not children_h:
            # Leaf node
            h_sum = torch.zeros(x.size(0), self.hidden_dim, device=x.device)
        else:
            h_sum = sum(children_h)
        
        # Gates
        i = torch.sigmoid(self.W_i(x) + self.U_i(h_sum))
        o = torch.sigmoid(self.W_o(x) + self.U_o(h_sum))
        c_tilde = torch.tanh(self.W_c(x) + self.U_c(h_sum))
        
        # Cell state
        c = i * c_tilde
        if children_c:
            for child_c, child_h in zip(children_c, children_h):
                f = torch.sigmoid(self.W_f(x) + self.U_f(child_h))
                c = c + f * child_c
        
        # Hidden state
        h = o * torch.tanh(c)
        
        return h, c


class TreeEncoder(nn.Module):
    """Encodes partial expression trees using TreeLSTM."""

    def __init__(self,
                 node_feature_dim: int,
                 hidden_dim: int,
                 n_layers: int = 2):
        super().__init__()

        self.node_embedding = nn.Linear(node_feature_dim, hidden_dim)
        self.tree_cells = nn.ModuleList([
            TreeLSTMCell(hidden_dim, hidden_dim) for _ in range(n_layers)
        ])
        self.hidden_dim = hidden_dim

    def forward(
        self,
        tree_features: torch.Tensor,
        tree_structure: Optional[Dict[int, List[int]]] = None,
    ) -> torch.Tensor:
        """Encode a tree represented as a tensor with optional structure.

        Parameters
        ----------
        tree_features : torch.Tensor
            Tensor of shape ``(batch_size, max_nodes, feature_dim)`` containing
            node features in topological order (children before parents).
        tree_structure : dict[int, list[int]] | None, optional
            Mapping from node index to indices of its children. If ``None`` the
            nodes are processed sequentially without structural information.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(batch_size, hidden_dim)`` representing the root
            node.
        """

        batch_size, max_nodes, _ = tree_features.shape

        # Embed nodes once
        node_embeds = self.node_embedding(tree_features)  # (B, N, H)

        # Fallback: sequential processing when no structure is provided
        if tree_structure is None:
            h = node_embeds
            for cell in self.tree_cells:
                h_states = []
                for i in range(max_nodes):
                    h_i, _ = cell(h[:, i, :], [], [])
                    h_states.append(h_i)
                h = torch.stack(h_states, dim=1)
            return torch.mean(h, dim=1)

        # Structured processing using post-order traversal
        def run_layer(layer_input, cell):
            node_hidden_states: Dict[int, torch.Tensor] = {}
            node_cell_states: Dict[int, torch.Tensor] = {}

            def traverse(node_idx: int):
                if node_idx in node_hidden_states:
                    return node_hidden_states[node_idx], node_cell_states[node_idx]

                children_indices = tree_structure.get(node_idx, [])
                children_h = []
                children_c = []
                for child_idx in children_indices:
                    h_child, c_child = traverse(child_idx)
                    children_h.append(h_child)
                    children_c.append(c_child)

                node_input = layer_input[:, node_idx, :]
                h, c = cell(node_input, children_h, children_c)

                node_hidden_states[node_idx] = h
                node_cell_states[node_idx] = c
                return h, c

            root_h, root_c = traverse(0)

            # Collect hidden states for all nodes to feed to next layer
            ordered_h = [node_hidden_states.get(i, layer_input.new_zeros(batch_size, self.hidden_dim))
                         for i in range(max_nodes)]
            stacked_h = torch.stack(ordered_h, dim=1)
            return root_h, stacked_h

        h_layer = node_embeds
        root_h = None
        for cell in self.tree_cells:
            root_h, h_layer = run_layer(h_layer, cell)

        return root_h


class TransformerEncoder(nn.Module):
    """Alternative: Encode trees using Transformer architecture."""
    
    def __init__(self,
                 node_feature_dim: int,
                 hidden_dim: int,
                 n_heads: int = 8,
                 n_layers: int = 6):
        super().__init__()
        
        self.node_embedding = nn.Linear(node_feature_dim, hidden_dim)
        self.positional_encoding = nn.Parameter(
            torch.randn(1, 100, hidden_dim) * 0.02
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )
        
        self.hidden_dim = hidden_dim
        
    def forward(self, tree_features: torch.Tensor) -> torch.Tensor:
        """
        Encode tree using self-attention.
        tree_features: (batch_size, max_nodes, feature_dim)
        """
        batch_size, max_nodes, _ = tree_features.shape
        
        # Embed nodes
        node_embeds = self.node_embedding(tree_features)
        
        # Add positional encoding
        node_embeds = node_embeds + self.positional_encoding[:, :max_nodes, :]
        
        # Create attention mask for padding
        # Assuming nodes with all-zero features are padding
        padding_mask = (tree_features.sum(dim=-1) == 0)
        
        # Apply transformer
        encoded = self.transformer(
            node_embeds,
            src_key_padding_mask=padding_mask
        )
        
        # Pool to get tree representation
        # Masked mean pooling
        mask_expanded = (~padding_mask).unsqueeze(-1).float()
        sum_embeddings = (encoded * mask_expanded).sum(dim=1)
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
        tree_repr = sum_embeddings / sum_mask
        
        return tree_repr


class HypothesisNet(nn.Module):
    """
    Policy network for hypothesis generation.
    Maps tree states to action distributions over grammar elements.
    """
    
    def __init__(self,
                 observation_dim: int,
                 action_dim: int,
                 hidden_dim: int = 256,
                 encoder_type: str = 'transformer',
                 grammar: Optional['ProgressiveGrammar'] = None,
                 debug: bool = False):
        super().__init__()
        
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.grammar = grammar
        self.debug = debug
        
        # Unflatten observation to tree representation
        self.node_feature_dim = 128
        
        # Choose encoder
        if encoder_type == 'transformer':
            self.encoder = TransformerEncoder(
                self.node_feature_dim,
                hidden_dim,
                n_heads=8,
                n_layers=4
            )
        else:
            self.encoder = TreeEncoder(
                self.node_feature_dim,
                hidden_dim,
                n_layers=3
            )
        
        # Policy head
        self.policy_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        # Value head (for actor-critic)
        self.value_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Action embeddings for better representation
        self.action_embeddings = nn.Embedding(action_dim, 64)
        
        # Grammar-aware components
        self.grammar_context = nn.LSTM(
            input_size=64,
            hidden_size=128,
            num_layers=2,
            batch_first=True
        )
        
    def forward(self,
                observation: torch.Tensor,
                action_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through policy network.

        observation: Flattened tree state (batch_size, obs_dim)
        action_mask: Valid actions (batch_size, action_dim)

        Returns dict with 'policy_logits', 'value', 'action_logits'
        """
        if self.debug:
            # DEBUG: print out dims so we know what's actually happening
            print(
                f"[DEBUG] node_feature_dim={self.node_feature_dim}, "
                f"obs_size={observation.numel()}"
            )

        batch_size, obs_dim = observation.shape
        # Dynamically infer how many nodes are present
        num_nodes = obs_dim // self.node_feature_dim
        if self.debug:
            print(f"[DEBUG] computed num_nodes={num_nodes}, node_feature_dim={self.node_feature_dim}, obs_size={obs_dim}")
        # Now reshape correctly (e.g. (1,3,128) for obs_size=384)
        tree_features = observation.view(
            batch_size,
            num_nodes,
            self.node_feature_dim
        )
        
        # Encode tree state
        tree_repr = self.encoder(tree_features)
        
        # Get policy logits
        policy_logits = self.policy_net(tree_repr)
        
        # Apply action mask if provided
        if action_mask is not None:
            # Set invalid actions to very negative value
            masked_logits = policy_logits.clone()
            masked_logits[~action_mask] = -1e9
        else:
            masked_logits = policy_logits
        
        # Clean up any stray NaNs just in case
        action_logits = masked_logits.nan_to_num(nan=-1e9, posinf=1e9, neginf=-1e9)
        
        # Get value estimate
        value = self.value_net(tree_repr)
        
        return {
            'policy_logits': policy_logits,
            'action_logits': action_logits,       # use these for Categorical
            'value': value,
            'tree_representation': tree_repr
        }
    
    def get_action(self, 
                   observation: torch.Tensor,
                   action_mask: Optional[torch.Tensor] = None,
                   deterministic: bool = False) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.
        
        Returns: (action, log_prob, value)
        """
        outputs = self.forward(observation, action_mask)

        # Build distribution directly from logits (no need to softmax)
        dist = Categorical(logits=outputs['action_logits'])

        if deterministic:
            action = torch.argmax(outputs['action_logits'], dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob, outputs['value']


class PPOTrainer:
    """
    Proximal Policy Optimization trainer for HypothesisNet.
    Implements PPO with advantage estimation and action masking.
    """
    
    def __init__(self,
                 policy: HypothesisNet,
                 env: 'SymbolicDiscoveryEnv',
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_epsilon: float = 0.2,
                 value_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 max_grad_norm: float = 0.5):
        
        self.policy = policy
        self.env = env
        self.optimizer = optim.Adam(policy.parameters(), lr=lr)
        
        # PPO hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        # Buffers
        self.rollout_buffer = RolloutBuffer()
        
        # Logging
        self.episode_rewards: Deque[float] = deque(maxlen=100)
        self.episode_complexities: Deque[int] = deque(maxlen=100)
        self.episode_mse: Deque[float] = deque(maxlen=100)
        
    def collect_rollouts(self, n_steps: int) -> Dict[str, List[Any]]:
        """
        Collect experience by interacting with environment.

        Returns a dict like:
          {
            "rollouts": List[Dict[str, Any]],
            "episode_stats": List[Dict[str, float]]
          }
        """
        self.rollout_buffer.reset()

        obs, _ = self.env.reset()
        episode_reward: float = 0.0
        episode_length: int = 0
        
        for step in range(n_steps):
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            action_mask = self.env.get_action_mask()
            mask_tensor = torch.BoolTensor(action_mask).unsqueeze(0)
            
            # Get action from policy
            with torch.no_grad():
                action, log_prob, value = self.policy.get_action(
                    obs_tensor, 
                    mask_tensor
                )
            
            # Take environment step
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            
            # Store transition
            self.rollout_buffer.add(
                obs=obs,
                action=action,
                reward=reward,  # reward is float
                value=value.item(),
                log_prob=log_prob.item(),
                done=terminated or truncated,
                action_mask=action_mask
            )

            episode_reward += reward  # now both sides are float
            episode_length += 1
            
            if terminated or truncated:
                # Log episode statistics
                self.episode_rewards.append(episode_reward)
                if 'complexity' in info:
                    self.episode_complexities.append(info['complexity'])
                if 'mse' in info:
                    self.episode_mse.append(info['mse'])
                
                # Reset environment
                obs, _ = self.env.reset()
                episode_reward = 0
                episode_length = 0
            else:
                obs = next_obs
        
        # Compute returns and advantages
        self.rollout_buffer.compute_returns_and_advantages(
            self.policy,
            self.gamma,
            self.gae_lambda
        )
        
        return self.rollout_buffer.get()
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step on batch of data."""
        obs = batch['observations']
        actions = batch['actions']
        old_log_probs = batch['log_probs']
        advantages = batch['advantages']
        returns = batch['returns']
        action_masks = batch['action_masks']

        assert isinstance(action_masks, torch.Tensor), "action_masks should be a torch.Tensor"
        assert action_masks.dtype == torch.bool, f"action_masks dtype should be torch.bool, got {action_masks.dtype}"
        batch_size = obs.shape[0]
        expected_shape = (batch_size, self.policy.action_dim)
        assert action_masks.shape == expected_shape, f"action_masks shape should be {expected_shape}, got {action_masks.shape}"
        
        # Get current policy outputs
        outputs = self.policy(obs, action_masks)

        # Get log probs for taken actions
        dist = Categorical(logits=outputs['action_logits'])
        log_probs = dist.log_prob(actions)
        
        # Policy loss (PPO objective)
        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        value_pred = outputs['value'].squeeze()
        value_loss = F.mse_loss(value_pred, returns)
        
        # Entropy bonus
        entropy = dist.entropy().mean()
        
        # Total loss
        loss = (
            policy_loss + 
            self.value_coef * value_loss - 
            self.entropy_coef * entropy
        )
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        return {
            'loss': loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item()
        }
    
    def train(self, 
              total_timesteps: int,
              rollout_length: int = 2048,
              n_epochs: int = 10,
              batch_size: int = 64,
              log_interval: int = 10):
        """Main training loop."""
        
        n_updates = max(1, total_timesteps // rollout_length)
        
        for update in range(n_updates):
            # Collect rollouts
            rollout_data = self.collect_rollouts(rollout_length)
            
            # Train on collected data
            losses = []
            for epoch in range(n_epochs):
                # Create mini-batches
                indices = np.random.permutation(rollout_length)
                
                for start_idx in range(0, rollout_length, batch_size):
                    batch_indices = indices[start_idx:start_idx + batch_size]
                    batch = {
                        key: value[batch_indices] 
                        for key, value in rollout_data.items()
                    }
                    
                    loss_info = self.train_step(batch)
                    losses.append(loss_info)
            
            # Logging
            if update % log_interval == 0:
                avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0
                avg_complexity = np.mean(self.episode_complexities) if self.episode_complexities else 0
                avg_mse = np.mean(self.episode_mse) if self.episode_mse else float('inf')
                
                print(f"Update {update}/{n_updates}")
                print(f"  Avg Reward: {avg_reward:.3f}")
                print(f"  Avg Complexity: {avg_complexity:.1f}")
                print(f"  Avg MSE: {avg_mse:.3e}")
                print(f"  Avg Loss: {np.mean([l['loss'] for l in losses]):.3f}")
                
                # Log to wandb if available
                try:
                    wandb.log({
                        'reward': avg_reward,
                        'complexity': avg_complexity,
                        'mse': avg_mse,
                        'policy_loss': np.mean([l['policy_loss'] for l in losses]),
                        'value_loss': np.mean([l['value_loss'] for l in losses]),
                        'entropy': np.mean([l['entropy'] for l in losses])
                    })
                except:
                    pass


class RolloutBuffer:
    """Buffer for storing rollout data."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.action_masks = []
        self.advantages = None
        self.returns = None
    
    def add(self, obs, action, reward, value, log_prob, done, action_mask):
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        self.action_masks.append(action_mask)
    
    def compute_returns_and_advantages(self, policy, gamma, gae_lambda):
        """Compute returns and GAE advantages."""
        rewards = np.array(self.rewards)
        values = np.array(self.values)
        dones = np.array(self.dones)
        
        # Get value of last state
        with torch.no_grad():
            last_obs = torch.FloatTensor(self.observations[-1]).unsqueeze(0)
            last_mask = torch.BoolTensor(self.action_masks[-1]).unsqueeze(0)
            last_value = policy(last_obs, last_mask)['value'].item()
        
        # Compute returns and advantages
        advantages = np.zeros_like(rewards)
        last_advantage = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = last_value
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            last_advantage = delta + gamma * gae_lambda * (1 - dones[t]) * last_advantage
            advantages[t] = last_advantage
        
        returns = advantages + values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        self.advantages = advantages
        self.returns = returns
    
    def get(self) -> Dict[str, torch.Tensor]:
        """Get all data as tensors."""
        return {
            'observations': torch.FloatTensor(np.array(self.observations)),
            'actions': torch.LongTensor(self.actions),
            'rewards': torch.FloatTensor(self.rewards),
            'values': torch.FloatTensor(self.values),
            'log_probs': torch.FloatTensor(self.log_probs),
            'advantages': torch.FloatTensor(self.advantages),
            'returns': torch.FloatTensor(self.returns),
            'action_masks': torch.BoolTensor(np.array(self.action_masks))
        }


# Integration and testing
if __name__ == "__main__":
    from symbolic_discovery_env import SymbolicDiscoveryEnv
    from progressive_grammar_system import ProgressiveGrammar, Variable
    
    # Setup environment
    grammar = ProgressiveGrammar()
    variables = [
        Variable("x", 0, {"smoothness": 0.9}),
        Variable("v", 1, {"conservation_score": 0.3})
    ]
    
    # Create data
    n_samples = 1000
    x_data = np.random.randn(n_samples)
    v_data = np.random.randn(n_samples) * 2
    target = 0.5 * v_data**2
    data = np.column_stack([x_data, v_data, target])
    
    # Create environment
    env = SymbolicDiscoveryEnv(
        grammar=grammar,
        target_data=data,
        variables=variables,
        max_depth=5,
        max_complexity=10
    )
    
    # Create policy network
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    policy = HypothesisNet(
        observation_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=256,
        encoder_type='transformer',
        grammar=grammar
    )
    
    print(f"Policy network parameters: {sum(p.numel() for p in policy.parameters())}")
    
    # Test forward pass
    obs, _ = env.reset()
    obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
    action_mask = torch.BoolTensor(env.get_action_mask()).unsqueeze(0)
    
    outputs = policy(obs_tensor, action_mask)
    print(f"Policy output shapes:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
    
    # Create trainer
    trainer = PPOTrainer(policy, env)
    
    # Small training run
    print("\nStarting training...")
    trainer.train(
        total_timesteps=10000,
        rollout_length=512,
        n_epochs=3,
        batch_size=32,
        log_interval=1
    )
