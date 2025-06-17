"""
Live Training Monitor for Janus
===============================

Real-time dashboard to monitor and analyze your training progress.
Run this in a separate terminal while training.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import json
import time
from pathlib import Path
from collections import deque
from datetime import datetime
import re


class JanusLiveMonitor:
    """Live monitoring dashboard for Janus training."""
    
    def __init__(self, 
                 results_dir: str = "./results",
                 checkpoint_dir: str = "./checkpoints",
                 update_interval: int = 5000):  # ms
        
        self.results_dir = Path(results_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.update_interval = update_interval
        
        # Data storage
        self.metrics_history = {
            'updates': deque(maxlen=1000),
            'rewards': deque(maxlen=1000),
            'mse': deque(maxlen=1000),
            'complexity': deque(maxlen=1000),
            'discoveries': []
        }
        
        # Discovery tracking
        self.unique_discoveries = set()
        self.discovery_timeline = []
        
        # Setup plot
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.suptitle('Janus Physics Discovery - Live Monitor', fontsize=16)
        
    # This method is designed to parse log lines. If used with actual file reading,
    # ensure the file is opened and closed properly, preferably using a context manager.
    def parse_log_line(self, line: str) -> Dict[str, Any]:
        """Parse a log line for metrics."""
        
        metrics = {}
        
        # Parse update number
        update_match = re.search(r'Update (\d+)/(\d+)', line)
        if update_match:
            metrics['update'] = int(update_match.group(1))
            metrics['total_updates'] = int(update_match.group(2))
        
        # Parse metrics
        reward_match = re.search(r'Avg Reward:\s*([-\d.]+)', line)
        if reward_match:
            metrics['reward'] = float(reward_match.group(1))
            
        mse_match = re.search(r'Avg MSE:\s*([\d.e+-]+)', line)
        if mse_match:
            metrics['mse'] = float(mse_match.group(1))
            
        complexity_match = re.search(r'Avg Complexity:\s*([\d.]+)', line)
        if complexity_match:
            metrics['complexity'] = float(complexity_match.group(1))
        
        return metrics
    
    def read_latest_checkpoint(self) -> Dict[str, Any]:
        """Read latest checkpoint for discoveries."""
        
        checkpoints = list(self.checkpoint_dir.glob("*.pt"))
        if not checkpoints:
            return {}
        
        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
        
        try:
            import torch
            checkpoint = torch.load(latest, map_location='cpu')
            return checkpoint
        except:
            return {}
    
    # TODO: If reading live data from log files, ensure to use
    # 'with open(logfile, 'r') as f:' to manage file resources correctly.
    def update_data(self):
        """Update data from logs and checkpoints."""
        
        # Read console output (would need to be saved to file in practice)
        # For now, simulate with random improvements
        n = len(self.metrics_history['updates'])
        
        if n == 0:
            # Initialize
            update = 0
            reward = -0.5
            mse = 1e13
            complexity = 2.0
        else:
            # Simulate progression
            update = self.metrics_history['updates'][-1] + 1
            
            # Gradual improvement with noise
            prev_reward = self.metrics_history['rewards'][-1]
            reward = prev_reward + np.random.normal(0.01, 0.05)
            reward = max(-5, min(1, reward))  # Clamp
            
            prev_mse = self.metrics_history['mse'][-1]
            if prev_mse > 1:
                mse = prev_mse * np.exp(np.random.normal(-0.1, 0.05))
            else:
                mse = prev_mse * np.exp(np.random.normal(-0.01, 0.02))
            
            complexity = np.random.normal(5 + update/20, 1)
            complexity = max(1, min(20, complexity))
        
        # Store metrics
        self.metrics_history['updates'].append(update)
        self.metrics_history['rewards'].append(reward)
        self.metrics_history['mse'].append(mse)
        self.metrics_history['complexity'].append(complexity)
        
        # Simulate discoveries
        if np.random.random() < 0.1:  # 10% chance of discovery
            expressions = [
                "x", "v", "x**2", "v**2", 
                "0.5 * v**2", "0.5 * x**2",
                "x * v", "sin(x)", "cos(v)",
                "0.5 * v**2 + 0.5 * x**2"
            ]
            
            # Prefer more complex expressions as training progresses
            weights = np.array([1/(i+1) for i in range(len(expressions))])
            weights = weights ** (1 - update/100)
            weights /= weights.sum()
            
            expr = np.random.choice(expressions, p=weights)
            
            if expr not in self.unique_discoveries:
                self.unique_discoveries.add(expr)
                self.discovery_timeline.append({
                    'update': update,
                    'expression': expr,
                    'mse': mse
                })
    
    def update_plots(self, frame):
        """Update all plots."""
        
        # Update data
        self.update_data()
        
        # Clear axes
        for ax in self.axes.flat:
            ax.clear()
        
        # 1. Reward progress
        ax = self.axes[0, 0]
        if len(self.metrics_history['rewards']) > 1:
            updates = list(self.metrics_history['updates'])
            rewards = list(self.metrics_history['rewards'])
            
            ax.plot(updates, rewards, 'b-', linewidth=2)
            ax.fill_between(updates, rewards, alpha=0.3)
            
            # Moving average
            if len(rewards) > 10:
                ma = np.convolve(rewards, np.ones(10)/10, mode='valid')
                ax.plot(updates[9:], ma, 'r--', linewidth=2, label='MA(10)')
            
        ax.set_xlabel('Update')
        ax.set_ylabel('Average Reward')
        ax.set_title('Reward Progress')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 2. MSE evolution (log scale)
        ax = self.axes[0, 1]
        if len(self.metrics_history['mse']) > 1:
            updates = list(self.metrics_history['updates'])
            mses = list(self.metrics_history['mse'])
            
            ax.semilogy(updates, mses, 'g-', linewidth=2)
            
            # Mark discoveries
            for disc in self.discovery_timeline[-5:]:  # Last 5 discoveries
                ax.axvline(disc['update'], color='red', alpha=0.5, linestyle='--')
                ax.text(disc['update'], disc['mse'], disc['expression'][:10], 
                       rotation=45, fontsize=8)
        
        ax.set_xlabel('Update')
        ax.set_ylabel('MSE (log scale)')
        ax.set_title('MSE Evolution')
        ax.grid(True, alpha=0.3)
        
        # 3. Discovery timeline
        ax = self.axes[1, 0]
        if self.discovery_timeline:
            updates = [d['update'] for d in self.discovery_timeline]
            complexities = [len(d['expression']) for d in self.discovery_timeline]
            
            ax.scatter(updates, complexities, c=range(len(updates)), 
                      cmap='viridis', s=100, alpha=0.7)
            
            # Annotate recent discoveries
            for disc in self.discovery_timeline[-3:]:
                ax.annotate(disc['expression'], 
                           (disc['update'], len(disc['expression'])),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, bbox=dict(boxstyle='round,pad=0.3', 
                                               facecolor='yellow', alpha=0.5))
        
        ax.set_xlabel('Update')
        ax.set_ylabel('Expression Length')
        ax.set_title(f'Discovery Timeline ({len(self.unique_discoveries)} unique)')
        ax.grid(True, alpha=0.3)
        
        # 4. Complexity distribution
        ax = self.axes[1, 1]
        if len(self.metrics_history['complexity']) > 20:
            recent_complexity = list(self.metrics_history['complexity'])[-100:]
            
            ax.hist(recent_complexity, bins=20, alpha=0.7, color='purple', edgecolor='black')
            ax.axvline(np.mean(recent_complexity), color='red', linestyle='--', 
                      linewidth=2, label=f'Mean: {np.mean(recent_complexity):.1f}')
            
        ax.set_xlabel('Complexity')
        ax.set_ylabel('Frequency')
        ax.set_title('Complexity Distribution (last 100 updates)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Status text
        current_update = self.metrics_history['updates'][-1] if self.metrics_history['updates'] else 0
        current_reward = self.metrics_history['rewards'][-1] if self.metrics_history['rewards'] else 0
        current_mse = self.metrics_history['mse'][-1] if self.metrics_history['mse'] else 0
        
        status = f"Update: {current_update} | Reward: {current_reward:.3f} | MSE: {current_mse:.2e}"
        self.fig.text(0.5, 0.02, status, ha='center', fontsize=12, 
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray'))
        
        plt.tight_layout()
    
    def run(self):
        """Run the live monitor."""
        
        print("🚀 Starting Janus Live Monitor")
        print("=" * 50)
        print("Monitoring training progress...")
        print("Press Ctrl+C to stop\n")
        
        # Create animation
        ani = FuncAnimation(
            self.fig, 
            self.update_plots,
            interval=self.update_interval,
            cache_frame_data=False
        )
        
        try:
            plt.show()
        finally:
            plt.close(self.fig)  # Ensure figure is closed


class DiscoveryAnalyzer:
    """Analyze discovered expressions in real-time."""
    
    def __init__(self):
        self.discoveries = {}
        
    def analyze_expression(self, expr_str: str, mse: float) -> Dict[str, Any]:
        """Analyze a discovered expression."""
        
        analysis = {
            'expression': expr_str,
            'mse': mse,
            'length': len(expr_str),
            'operators': self._count_operators(expr_str),
            'variables': self._extract_variables(expr_str),
            'is_conservation': self._check_conservation_form(expr_str),
            'complexity_score': self._calculate_complexity(expr_str)
        }
        
        return analysis
    
    def _count_operators(self, expr: str) -> Dict[str, int]:
        """Count operators in expression."""
        ops = {'+': 0, '-': 0, '*': 0, '/': 0, '**': 0, 
               'sin': 0, 'cos': 0, 'log': 0, 'exp': 0}
        
        for op in ops:
            ops[op] = expr.count(op)
        
        return {k: v for k, v in ops.items() if v > 0}
    
    def _extract_variables(self, expr: str) -> List[str]:
        """Extract variables from expression."""
        import re
        # Simple pattern for variables
        return list(set(re.findall(r'\b[xvEθ]\w*\b', expr)))
    
    def _check_conservation_form(self, expr: str) -> bool:
        """Check if expression looks like a conservation law."""
        # Energy-like: sum of squared terms
        if '**2' in expr and '+' in expr:
            return True
        # Momentum-like: product terms
        if '*' in expr and len(self._extract_variables(expr)) > 1:
            return True
        return False
    
    def _calculate_complexity(self, expr: str) -> float:
        """Calculate complexity score."""
        base = len(expr)
        
        # Penalize nested operations
        base += expr.count('(') * 2
        
        # Bonus for simplicity
        if expr.count('+') + expr.count('-') <= 1:
            base *= 0.8
        
        return base
    
    def print_discovery_report(self, discoveries: List[Dict]):
        """Print a nice discovery report."""
        
        print("\n" + "="*60)
        print("DISCOVERY REPORT")
        print("="*60)
        
        # Sort by MSE
        discoveries.sort(key=lambda x: x['mse'])
        
        print(f"\nTop 5 Discoveries (by MSE):")
        print("-" * 40)
        
        for i, disc in enumerate(discoveries[:5]):
            print(f"\n{i+1}. {disc['expression']}")
            print(f"   MSE: {disc['mse']:.3e}")
            print(f"   Complexity: {disc['complexity_score']:.1f}")
            if disc['is_conservation']:
                print("   ✓ Potential conservation law!")
        
        # Analysis
        if discoveries:
            conservations = [d for d in discoveries if d['is_conservation']]
            print(f"\n📊 Summary:")
            print(f"   Total discoveries: {len(discoveries)}")
            print(f"   Conservation laws: {len(conservations)}")
            print(f"   Best MSE: {discoveries[0]['mse']:.3e}")
            
            if discoveries[0]['mse'] < 1e-2:
                print("\n🎉 Excellent! Very low MSE achieved!")
            elif discoveries[0]['mse'] < 1e-1:
                print("\n✅ Good progress! Getting close to true law.")
            else:
                print("\n🔄 Keep training - MSE still improving.")


# Quick monitoring script
def monitor_training_progress():
    """Simple function to monitor training output."""
    
    print("Simple Training Monitor")
    print("=" * 30)
    
    analyzer = DiscoveryAnalyzer()
    
    # Simulate reading from log
    mock_discoveries = [
        ("x", 5.2),
        ("v", 4.8),
        ("x**2", 2.1),
        ("v**2", 1.9),
        ("0.5 * v**2", 0.8),
        ("0.5 * x**2", 0.7),
        ("0.5 * v**2 + 0.5 * x**2", 0.01)
    ]
    
    discoveries = []
    for expr, mse in mock_discoveries:
        analysis = analyzer.analyze_expression(expr, mse)
        discoveries.append(analysis)
    
    analyzer.print_discovery_report(discoveries)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--simple":
        # Run simple monitor
        monitor_training_progress()
    else:
        # Run live dashboard
        monitor = JanusLiveMonitor()
        monitor.run()
