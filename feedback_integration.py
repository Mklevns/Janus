"""
Practical Integration Guide: Adding Enhanced Feedback to Existing Janus
======================================================================

This module shows how to integrate the enhanced feedback systems into
your currently running training without major disruption.
"""

import numpy as np
from typing import Dict, Any, Optional
from collections import deque

from symbolic_discovery_env import SymbolicDiscoveryEnv
from hypothesis_policy_network import PPOTrainer


class EnhancedPPOTrainer(PPOTrainer):
    """Drop-in replacement for PPOTrainer with enhanced feedback."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Add intrinsic reward calculator
        from enhanced_feedback import IntrinsicRewardCalculator
        self.intrinsic_calculator = IntrinsicRewardCalculator(
            novelty_weight=0.2,  # Start conservative
            diversity_weight=0.1,
            complexity_growth_weight=0.05
        )
        
        # Track discoveries for online adaptation
        self.discovery_tracker = {
            'unique_expressions': set(),
            'discovery_counts': {},
            'complexity_history': deque(maxlen=100),
            'reward_history': deque(maxlen=100)
        }
        
        # Dynamic learning rate adjustment
        self.lr_scheduler = AdaptiveLRScheduler(
            base_lr=self.optimizer.param_groups[0]['lr']
        )
        
    def collect_rollouts(self, n_steps: int) -> Dict[str, Any]:
        """Enhanced rollout collection with intrinsic rewards."""
        
        # Store original reward function
        original_step = self.env.step
        
        # Wrap environment step to add intrinsic rewards
        def enhanced_step(action):
            obs, reward, terminated, truncated, info = original_step(action)
            
            if terminated and 'expression' in info:
                # Calculate intrinsic reward
                expr = info['expression']
                complexity = info.get('complexity', 0)
                
                enhanced_reward = self.intrinsic_calculator.calculate_intrinsic_reward(
                    expression=expr,
                    complexity=complexity,
                    extrinsic_reward=reward
                )
                
                # Track discovery
                self.discovery_tracker['unique_expressions'].add(expr)
                self.discovery_tracker['discovery_counts'][expr] = \
                    self.discovery_tracker['discovery_counts'].get(expr, 0) + 1
                self.discovery_tracker['complexity_history'].append(complexity)
                self.discovery_tracker['reward_history'].append(enhanced_reward)
                
                # Update info
                info['intrinsic_reward'] = enhanced_reward - reward
                info['enhanced_reward'] = enhanced_reward
                
                # Use enhanced reward
                reward = enhanced_reward
            
            return obs, reward, terminated, truncated, info
        
        # Temporarily replace step function
        self.env.step = enhanced_step
        
        # Collect rollouts with enhanced rewards
        rollouts = super().collect_rollouts(n_steps)
        
        # Restore original step
        self.env.step = original_step
        
        return rollouts
    
    def train_step(self, batch: Dict) -> Dict[str, float]:
        """Enhanced training step with adaptive learning rate."""
        
        # Adapt learning rate based on discovery rate
        discovery_rate = self._calculate_discovery_rate()
        new_lr = self.lr_scheduler.update(discovery_rate)
        self.optimizer.param_groups[0]['lr'] = new_lr
        
        # Standard training step
        losses = super().train_step(batch)
        
        # Add discovery metrics to losses
        losses['discovery_rate'] = discovery_rate
        losses['unique_discoveries'] = len(self.discovery_tracker['unique_expressions'])
        
        return losses
    
    def _calculate_discovery_rate(self) -> float:
        """Calculate rate of new discoveries."""
        
        recent_window = 100
        if len(self.episode_rewards) < recent_window:
            return 1.0  # High rate at start
        
        # Count unique discoveries in recent episodes
        recent_discoveries = list(self.discovery_tracker['unique_expressions'])[-recent_window:]
        if len(recent_discoveries) > 10:
            # Rate of new vs repeated
            unique_recent = len(set(recent_discoveries))
            return unique_recent / len(recent_discoveries)
        
        return 0.5  # Default medium rate


class AdaptiveLRScheduler:
    """Simple adaptive learning rate scheduler based on discovery progress."""
    
    def __init__(self, base_lr: float = 3e-4):
        self.base_lr = base_lr
        self.current_lr = base_lr
        self.discovery_history = deque(maxlen=50)
        
    def update(self, discovery_rate: float) -> float:
        """Update learning rate based on discovery rate."""
        
        self.discovery_history.append(discovery_rate)
        
        if len(self.discovery_history) < 10:
            return self.current_lr
        
        # Calculate trend
        recent_rate = np.mean(list(self.discovery_history)[-10:])
        
        # Adjust learning rate
        if recent_rate < 0.1:
            # Low discovery - increase exploration via higher LR
            self.current_lr = min(self.base_lr * 2, self.current_lr * 1.1)
        elif recent_rate > 0.5:
            # High discovery - stabilize with lower LR
            self.current_lr = max(self.base_lr * 0.5, self.current_lr * 0.95)
        else:
            # Gradually return to baseline
            self.current_lr = 0.99 * self.current_lr + 0.01 * self.base_lr
        
        return self.current_lr


def quick_enhance_training(trainer: PPOTrainer) -> EnhancedPPOTrainer:
    """Quick function to enhance existing trainer."""
    
    # Create enhanced trainer with same configuration
    enhanced = EnhancedPPOTrainer(trainer.policy, trainer.env)
    
    # Copy optimizer state
    enhanced.optimizer.load_state_dict(trainer.optimizer.state_dict())
    
    # Copy training history
    enhanced.episode_rewards = trainer.episode_rewards
    enhanced.episode_complexities = trainer.episode_complexities
    enhanced.episode_mse = trainer.episode_mse
    
    return enhanced


# Monkey-patch enhancement for minimal disruption
def add_intrinsic_rewards_to_env(env: SymbolicDiscoveryEnv, 
                                weight: float = 0.2) -> None:
    """Add intrinsic rewards to existing environment (monkey-patch)."""
    
    from enhanced_feedback import IntrinsicRewardCalculator
    
    # Add calculator
    env._intrinsic_calculator = IntrinsicRewardCalculator(
        novelty_weight=weight,
        diversity_weight=weight * 0.5,
        complexity_growth_weight=weight * 0.25
    )
    
    # Store original step
    env._original_step = env.step
    
    # Define enhanced step
    def enhanced_step(self, action):
        obs, reward, terminated, truncated, info = self._original_step(action)
        
        if terminated and 'expression' in info:
            # Calculate intrinsic reward
            enhanced_reward = self._intrinsic_calculator.calculate_intrinsic_reward(
                expression=info['expression'],
                complexity=info.get('complexity', 0),
                extrinsic_reward=reward,
                # FIX: Add the missing arguments
                embedding=None,  # Pass None as a placeholder for the embedding
                data=self.target_data,
                variables=self.variables
            )
            
            info['intrinsic_bonus'] = enhanced_reward - reward
            reward = enhanced_reward
        
        return obs, reward, terminated, truncated, info
    
    # Bind enhanced step
    import types
    env.step = types.MethodType(enhanced_step, env)


# Configuration adjustments for better discovery
def optimize_reward_config(current_config: Dict[str, Any]) -> Dict[str, Any]:
    """Optimize reward configuration for better discovery."""
    
    optimized = current_config.copy()
    
    # Reduce MSE weight slightly to allow more exploration
    if 'reward_config' in optimized:
        optimized['reward_config']['mse_weight'] = -0.8  # From -1.0
        optimized['reward_config']['complexity_penalty'] = -0.005  # From -0.01
        
        # Add novelty bonus
        optimized['reward_config']['novelty_bonus'] = 0.2
    
    return optimized


# Live monitoring and adjustment
class LiveTrainingMonitor:
    """Monitor and adjust training in real-time."""
    
    def __init__(self, trainer: PPOTrainer):
        self.trainer = trainer
        self.checkpoint_discoveries = set()
        self.stagnation_counter = 0
        
    def check_and_adjust(self, update_num: int):
        """Check training progress and make adjustments."""
        
        # Get current discoveries
        if hasattr(self.trainer, 'discovery_tracker'):
            current_discoveries = self.trainer.discovery_tracker['unique_expressions']
            
            # Check for stagnation
            new_discoveries = current_discoveries - self.checkpoint_discoveries
            
            if len(new_discoveries) < 5 and update_num > 10:
                self.stagnation_counter += 1
                
                if self.stagnation_counter > 3:
                    print(f"\n⚠️  Stagnation detected at update {update_num}")
                    self._apply_stagnation_fix()
                    self.stagnation_counter = 0
            else:
                self.stagnation_counter = 0
            
            # Update checkpoint
            self.checkpoint_discoveries = current_discoveries.copy()
            
            # Report progress
            if update_num % 10 == 0:
                print(f"\n📊 Discovery Report:")
                print(f"   Unique expressions: {len(current_discoveries)}")
                print(f"   New this period: {len(new_discoveries)}")
                if new_discoveries:
                    print(f"   Latest: {list(new_discoveries)[:3]}")
    
    def _apply_stagnation_fix(self):
        """Apply fixes when training stagnates."""
        
        # Increase exploration
        if hasattr(self.trainer, 'entropy_coef'):
            self.trainer.entropy_coef *= 1.5
            print(f"   → Increased entropy coefficient to {self.trainer.entropy_coef:.3f}")
        
        # Boost novelty rewards
        if hasattr(self.trainer, 'intrinsic_calculator'):
            self.trainer.intrinsic_calculator.novelty_weight *= 1.3
            print(f"   → Increased novelty weight to {self.trainer.intrinsic_calculator.novelty_weight:.3f}")
        
        # Temporarily increase learning rate
        current_lr = self.trainer.optimizer.param_groups[0]['lr']
        self.trainer.optimizer.param_groups[0]['lr'] = current_lr * 1.5
        print(f"   → Boosted learning rate to {current_lr * 1.5:.5f}")


# Example integration script
if __name__ == "__main__":
    print("Enhanced Feedback Integration Guide")
    print("=" * 50)
    print()
    print("Option 1: Quick Enhancement (minimal changes)")
    print("-" * 30)
    print("""
# In your training script, after creating trainer:
from feedback_integration import quick_enhance_training

# Replace your trainer
enhanced_trainer = quick_enhance_training(trainer)

# Continue training with enhanced version
enhanced_trainer.train(total_timesteps=remaining_steps)
""")
    
    print("\nOption 2: Monkey-patch Existing Environment")
    print("-" * 30)
    print("""
# Add intrinsic rewards to running environment
from feedback_integration import add_intrinsic_rewards_to_env

# Enhance environment (no restart needed)
add_intrinsic_rewards_to_env(trainer.env, weight=0.2)

# Training will now use intrinsic rewards automatically
""")
    
    print("\nOption 3: Live Monitoring")
    print("-" * 30)
    print("""
# Add live monitoring to detect stagnation
from feedback_integration import LiveTrainingMonitor

monitor = LiveTrainingMonitor(trainer)

# In your training loop:
for update in range(n_updates):
    trainer.train(...)
    monitor.check_and_adjust(update)
""")
    
    print("\nRecommended Approach:")
    print("-" * 20)
    print("1. Let current training run for ~20-30k steps")
    print("2. If MSE not improving, apply Option 2 (intrinsic rewards)")
    print("3. Monitor with Option 3 for automatic adjustments")
    print("4. For next run, use Option 1 from the start")
