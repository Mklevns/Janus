"""
Emergent Discovery Monitoring System for Janus
=============================================

Tracks and analyzes emergent behaviors, novel discoveries, and
phase transitions in the physics discovery process.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
import networkx as nx
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy
import sympy as sp
import time
import json
from pathlib import Path


@dataclass
class DiscoveryEvent:
    """Represents a discovery event with full context."""
    timestamp: float
    expression: str
    symbolic_form: Optional[sp.Expr]
    complexity: int
    accuracy: float
    mse: float
    discovery_path: List[Dict[str, Any]]  # Action sequence
    environmental_state: Dict[str, Any]
    agent_id: Optional[str] = None
    parent_discoveries: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
            'expression': self.expression,
            'complexity': self.complexity,
            'accuracy': self.accuracy,
            'mse': self.mse,
            'path_length': len(self.discovery_path),
            'agent_id': self.agent_id
        }


class NoveltyDetector:
    """Detects novel patterns and behaviors in discovery process."""
    
    def __init__(self, 
                 embedding_dim: int = 64,
                 novelty_threshold: float = 0.7):
        
        self.embedding_dim = embedding_dim
        self.novelty_threshold = novelty_threshold
        
        # Expression embedding network
        self.embedder = ExpressionEmbedder(embedding_dim)
        
        # History of embeddings
        self.embedding_history = deque(maxlen=1000)
        self.expression_cache = {}
        
    def compute_novelty(self, expression: str) -> Tuple[float, Dict[str, Any]]:
        """Compute novelty score for an expression."""
        
        # Get embedding
        if expression in self.expression_cache:
            embedding = self.expression_cache[expression]
        else:
            embedding = self.embedder.embed(expression)
            self.expression_cache[expression] = embedding
        
        if len(self.embedding_history) < 10:
            # Not enough history
            novelty_score = 1.0
            metrics = {'nearest_distance': float('inf')}
        else:
            # Compute distances to historical embeddings
            history_array = np.array(list(self.embedding_history))
            distances = np.linalg.norm(
                history_array - embedding.reshape(1, -1), 
                axis=1
            )
            
            # Novelty metrics
            min_distance = np.min(distances)
            mean_distance = np.mean(distances)
            
            # k-nearest neighbors novelty
            k = min(15, len(self.embedding_history))
            k_nearest = np.sort(distances)[:k]
            knn_novelty = np.mean(k_nearest)
            
            # Combined novelty score
            novelty_score = 1.0 - np.exp(-knn_novelty)
            
            metrics = {
                'nearest_distance': min_distance,
                'mean_distance': mean_distance,
                'knn_novelty': knn_novelty,
                'k': k
            }
        
        # Update history
        self.embedding_history.append(embedding)
        
        return novelty_score, metrics
    
    def find_discovery_clusters(self) -> Dict[int, List[str]]:
        """Cluster discoveries to find families of solutions."""
        
        if len(self.expression_cache) < 5:
            return {}
        
        # Get all embeddings
        expressions = list(self.expression_cache.keys())
        embeddings = np.array([self.expression_cache[expr] for expr in expressions])
        
        # Cluster using DBSCAN
        clustering = DBSCAN(eps=0.3, min_samples=2).fit(embeddings)
        
        # Group by cluster
        clusters = defaultdict(list)
        for expr, label in zip(expressions, clustering.labels_):
            if label != -1:  # Not noise
                clusters[label].append(expr)
        
        return dict(clusters)


class ExpressionEmbedder(nn.Module):
    """Neural network for embedding mathematical expressions."""
    
    def __init__(self, embedding_dim: int = 64):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        
        # Operation embeddings
        self.op_embedding = nn.Embedding(50, 32)
        
        # Tree aggregation
        self.tree_lstm = nn.LSTM(
            input_size=32,
            hidden_size=embedding_dim,
            num_layers=2,
            batch_first=True
        )
        
        # Final projection
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
    def embed(self, expression: str) -> np.ndarray:
        """Convert expression string to embedding."""
        
        # Parse expression to get operation sequence
        op_sequence = self._parse_to_ops(expression)
        
        # Convert to tensor
        op_indices = torch.LongTensor([self._op_to_index(op) for op in op_sequence])
        
        # Embed operations
        op_embeds = self.op_embedding(op_indices).unsqueeze(0)
        
        # Process through LSTM
        _, (h_n, _) = self.tree_lstm(op_embeds)
        
        # Project to final embedding
        embedding = self.projection(h_n[-1])
        
        return embedding.detach().numpy().squeeze()
    
    def _parse_to_ops(self, expression: str) -> List[str]:
        """Simple parsing to extract operations."""
        
        ops = []
        
        # Common operations
        for op in ['+', '-', '*', '/', '**', 'sin', 'cos', 'log', 'exp', 'sqrt']:
            if op in expression:
                ops.append(op)
        
        # Variables (simplified)
        for var in ['x', 'v', 'E', 'theta']:
            if var in expression:
                ops.append(f'var_{var}')
        
        # Constants
        if any(char.isdigit() for char in expression):
            ops.append('const')
        
        return ops[:20]  # Limit sequence length
    
    def _op_to_index(self, op: str) -> int:
        """Map operation to index."""
        
        op_map = {
            '+': 1, '-': 2, '*': 3, '/': 4, '**': 5,
            'sin': 6, 'cos': 7, 'log': 8, 'exp': 9, 'sqrt': 10,
            'var_x': 11, 'var_v': 12, 'var_E': 13, 'var_theta': 14,
            'const': 15
        }
        
        return op_map.get(op, 0)


class PhaseTransitionDetector:
    """Detects phase transitions in discovery dynamics."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metric_history = defaultdict(lambda: deque(maxlen=window_size))
        self.phase_transitions = []
        
    def update(self, metrics: Dict[str, float]):
        """Update with new metrics."""
        
        for key, value in metrics.items():
            self.metric_history[key].append(value)
        
        # Check for phase transitions
        if self._detect_transition():
            self.phase_transitions.append({
                'timestamp': time.time(),
                'metrics': dict(metrics),
                'type': self._classify_transition()
            })
    
    def _detect_transition(self) -> bool:
        """Detect if a phase transition occurred."""
        
        if len(self.metric_history['complexity']) < self.window_size:
            return False
        
        # Check for sudden changes in key metrics
        for metric_name, history in self.metric_history.items():
            if len(history) < self.window_size:
                continue
            
            # Split window
            first_half = list(history)[:self.window_size//2]
            second_half = list(history)[self.window_size//2:]
            
            # Statistical test for distribution change
            mean_change = abs(np.mean(second_half) - np.mean(first_half))
            pooled_std = np.sqrt((np.var(first_half) + np.var(second_half)) / 2)
            
            if pooled_std > 0:
                z_score = mean_change / (pooled_std / np.sqrt(self.window_size//2))
                
                if z_score > 3:  # Significant change
                    return True
        
        return False
    
    def _classify_transition(self) -> str:
        """Classify the type of phase transition."""
        
        # Analyze recent trends
        complexity_trend = np.polyfit(
            range(len(self.metric_history['complexity'])),
            list(self.metric_history['complexity']),
            1
        )[0]
        
        accuracy_trend = np.polyfit(
            range(len(self.metric_history['accuracy'])),
            list(self.metric_history['accuracy']),
            1
        )[0]
        
        if complexity_trend > 0.1 and accuracy_trend > 0.01:
            return "breakthrough"  # More complex but more accurate
        elif complexity_trend < -0.1:
            return "simplification"  # Finding simpler forms
        elif accuracy_trend > 0.02:
            return "refinement"  # Improving accuracy
        else:
            return "exploration"  # General exploration


class DiscoveryGraph:
    """Tracks relationships between discoveries."""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.discovery_events = {}
        
    def add_discovery(self, event: DiscoveryEvent):
        """Add a discovery to the graph."""
        
        expr = event.expression
        self.graph.add_node(expr, **event.to_dict())
        self.discovery_events[expr] = event
        
        # Add edges based on relationships
        for parent in event.parent_discoveries:
            if parent in self.graph:
                self.graph.add_edge(parent, expr)
        
        # Find structural relationships
        self._add_structural_edges(expr)
    
    def _add_structural_edges(self, expr: str):
        """Add edges based on structural similarity."""
        
        for other_expr in self.graph.nodes():
            if other_expr == expr:
                continue
            
            similarity = self._structural_similarity(expr, other_expr)
            
            if similarity > 0.7:
                self.graph.add_edge(
                    other_expr, 
                    expr, 
                    weight=similarity,
                    type='structural'
                )
    
    def _structural_similarity(self, expr1: str, expr2: str) -> float:
        """Compute structural similarity between expressions."""
        
        # Simple token-based similarity
        tokens1 = set(expr1.split())
        tokens2 = set(expr2.split())
        
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)
        
        return intersection / union
    
    def find_discovery_paths(self) -> List[List[str]]:
        """Find significant discovery paths."""
        
        paths = []
        
        # Find all simple paths between low and high complexity nodes
        simple_nodes = [n for n in self.graph.nodes() 
                       if self.graph.nodes[n]['complexity'] < 5]
        complex_nodes = [n for n in self.graph.nodes() 
                        if self.graph.nodes[n]['complexity'] > 10]
        
        for start in simple_nodes:
            for end in complex_nodes:
                try:
                    path = nx.shortest_path(self.graph, start, end)
                    if len(path) > 2:
                        paths.append(path)
                except nx.NetworkXNoPath:
                    continue
        
        return paths
    
    def compute_influence_scores(self) -> Dict[str, float]:
        """Compute influence scores for discoveries."""
        
        # PageRank-based influence
        try:
            scores = nx.pagerank(self.graph)
        except:
            scores = {n: 1.0 for n in self.graph.nodes()}
        
        return scores


class EmergentBehaviorTracker:
    """Main class for tracking emergent behaviors."""
    
    def __init__(self, 
                 save_dir: str = "./emergent_analysis"):
        
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Components
        self.novelty_detector = NoveltyDetector()
        self.phase_detector = PhaseTransitionDetector()
        self.discovery_graph = DiscoveryGraph()
        
        # Metrics
        self.diversity_history = deque(maxlen=1000)
        self.complexity_history = deque(maxlen=1000)
        self.accuracy_history = deque(maxlen=1000)
        
        # Real-time tracking
        self.current_epoch = 0
        self.discoveries_this_epoch = []
        
    def log_discovery(self, 
                     expression: str,
                     info: Dict[str, Any],
                     discovery_path: List[Dict[str, Any]],
                     agent_id: Optional[str] = None):
        """Log a new discovery event."""
        
        # Create discovery event
        event = DiscoveryEvent(
            timestamp=time.time(),
            expression=expression,
            symbolic_form=self._try_parse_symbolic(expression),
            complexity=info.get('complexity', 0),
            accuracy=info.get('accuracy', 0),
            mse=info.get('mse', float('inf')),
            discovery_path=discovery_path,
            environmental_state=info,
            agent_id=agent_id
        )
        
        # Compute novelty
        novelty_score, novelty_metrics = self.novelty_detector.compute_novelty(expression)
        
        # Update phase detector
        self.phase_detector.update({
            'complexity': event.complexity,
            'accuracy': event.accuracy,
            'novelty': novelty_score
        })
        
        # Add to graph
        self.discovery_graph.add_discovery(event)
        
        # Update histories
        self.discoveries_this_epoch.append(event)
        self.complexity_history.append(event.complexity)
        self.accuracy_history.append(event.accuracy)
        
        # Compute diversity
        if len(self.discoveries_this_epoch) > 1:
            diversity = self._compute_diversity()
            self.diversity_history.append(diversity)
        
        # Log significant discoveries
        if novelty_score > self.novelty_detector.novelty_threshold:
            self._log_significant_discovery(event, novelty_score, novelty_metrics)
    
    def _try_parse_symbolic(self, expression: str) -> Optional[sp.Expr]:
        """Try to parse expression to SymPy."""
        try:
            return sp.sympify(expression)
        except:
            return None
    
    def _compute_diversity(self) -> float:
        """Compute diversity of recent discoveries."""
        
        if len(self.discoveries_this_epoch) < 2:
            return 0.0
        
        # Get embeddings for recent discoveries
        embeddings = []
        for event in self.discoveries_this_epoch[-20:]:
            if event.expression in self.novelty_detector.expression_cache:
                embeddings.append(
                    self.novelty_detector.expression_cache[event.expression]
                )
        
        if len(embeddings) < 2:
            return 0.0
        
        # Compute pairwise distances
        embeddings = np.array(embeddings)
        distances = pdist(embeddings)
        
        # Diversity as mean pairwise distance
        return np.mean(distances)
    
    def _log_significant_discovery(self, 
                                 event: DiscoveryEvent,
                                 novelty_score: float,
                                 metrics: Dict):
        """Log significant discoveries to file."""
        
        log_entry = {
            'timestamp': event.timestamp,
            'expression': event.expression,
            'novelty_score': novelty_score,
            'novelty_metrics': metrics,
            'complexity': event.complexity,
            'accuracy': event.accuracy,
            'mse': event.mse,
            'agent_id': event.agent_id
        }
        
        log_file = self.save_dir / "significant_discoveries.jsonl"
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def end_epoch(self, epoch: int):
        """End of epoch analysis."""
        
        self.current_epoch = epoch
        
        # Find discovery clusters
        clusters = self.novelty_detector.find_discovery_clusters()
        
        # Compute influence scores
        influence_scores = self.discovery_graph.compute_influence_scores()
        
        # Generate visualizations
        self._generate_epoch_plots(epoch, clusters, influence_scores)
        
        # Save epoch summary
        summary = {
            'epoch': epoch,
            'num_discoveries': len(self.discoveries_this_epoch),
            'num_clusters': len(clusters),
            'phase_transitions': len(self.phase_detector.phase_transitions),
            'mean_novelty': np.mean([
                self.novelty_detector.compute_novelty(e.expression)[0]
                for e in self.discoveries_this_epoch
            ]) if self.discoveries_this_epoch else 0,
            'diversity': np.mean(list(self.diversity_history)) if self.diversity_history else 0
        }
        
        summary_file = self.save_dir / f"epoch_{epoch}_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Reset for next epoch
        self.discoveries_this_epoch = []
        
        return summary
    
    def _generate_epoch_plots(self, 
                            epoch: int,
                            clusters: Dict[int, List[str]],
                            influence_scores: Dict[str, float]):
        """Generate analysis plots for the epoch."""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Discovery timeline
        ax = axes[0, 0]
        if self.complexity_history and self.accuracy_history:
            x = range(len(self.complexity_history))
            ax.scatter(x, self.complexity_history, c=self.accuracy_history, 
                      cmap='viridis', alpha=0.6)
            ax.set_xlabel('Discovery Index')
            ax.set_ylabel('Complexity')
            ax.set_title('Discovery Timeline (colored by accuracy)')
        
        # 2. Phase transitions
        ax = axes[0, 1]
        if self.phase_detector.phase_transitions:
            transition_times = [t['timestamp'] for t in self.phase_detector.phase_transitions]
            transition_types = [t['type'] for t in self.phase_detector.phase_transitions]
            
            colors = {'breakthrough': 'red', 'simplification': 'blue', 
                     'refinement': 'green', 'exploration': 'gray'}
            
            for time, t_type in zip(transition_times, transition_types):
                ax.axvline(time, color=colors.get(t_type, 'gray'), 
                          alpha=0.7, linestyle='--', label=t_type)
            
            ax.set_xlabel('Time')
            ax.set_title('Phase Transitions')
            ax.legend()
        
        # 3. Discovery clusters (if enough data)
        ax = axes[1, 0]
        if len(self.novelty_detector.expression_cache) > 10:
            # t-SNE visualization
            expressions = list(self.novelty_detector.expression_cache.keys())
            embeddings = np.array([
                self.novelty_detector.expression_cache[expr] 
                for expr in expressions
            ])
            
            if embeddings.shape[0] > 3:
                tsne = TSNE(n_components=2, random_state=42)
                coords = tsne.fit_transform(embeddings)
                
                # Color by cluster
                colors = ['red', 'blue', 'green', 'orange', 'purple']
                for cluster_id, cluster_exprs in clusters.items():
                    cluster_indices = [
                        i for i, expr in enumerate(expressions)
                        if expr in cluster_exprs
                    ]
                    if cluster_indices:
                        ax.scatter(
                            coords[cluster_indices, 0],
                            coords[cluster_indices, 1],
                            c=colors[cluster_id % len(colors)],
                            label=f'Cluster {cluster_id}',
                            alpha=0.7
                        )
                
                ax.set_xlabel('t-SNE 1')
                ax.set_ylabel('t-SNE 2')
                ax.set_title('Discovery Clusters')
                ax.legend()
        
        # 4. Influence network
        ax = axes[1, 1]
        if len(self.discovery_graph.graph) > 3:
            # Top influential discoveries
            top_discoveries = sorted(
                influence_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            
            subgraph = self.discovery_graph.graph.subgraph(
                [d[0] for d in top_discoveries]
            )
            
            if len(subgraph) > 0:
                pos = nx.spring_layout(subgraph)
                nx.draw(
                    subgraph, pos, ax=ax,
                    node_size=[influence_scores.get(n, 1) * 1000 for n in subgraph],
                    node_color='lightblue',
                    edge_color='gray',
                    with_labels=False,
                    alpha=0.7
                )
                ax.set_title('Influence Network (top discoveries)')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / f'epoch_{epoch}_analysis.png', dpi=150)
        plt.close()
    
    def generate_final_report(self):
        """Generate comprehensive final report."""
        
        report = {
            'total_discoveries': len(self.discovery_graph.graph),
            'discovery_clusters': len(self.novelty_detector.find_discovery_clusters()),
            'phase_transitions': len(self.phase_detector.phase_transitions),
            'discovery_paths': len(self.discovery_graph.find_discovery_paths()),
            'mean_complexity': np.mean(list(self.complexity_history)) if self.complexity_history else 0,
            'mean_accuracy': np.mean(list(self.accuracy_history)) if self.accuracy_history else 0,
            'final_diversity': self.diversity_history[-1] if self.diversity_history else 0
        }
        
        # Top discoveries by influence
        influence_scores = self.discovery_graph.compute_influence_scores()
        top_discoveries = sorted(
            influence_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:20]
        
        report['top_discoveries'] = [
            {
                'expression': expr,
                'influence': score,
                'complexity': self.discovery_graph.graph.nodes[expr]['complexity'],
                'accuracy': self.discovery_graph.graph.nodes[expr]['accuracy']
            }
            for expr, score in top_discoveries
        ]
        
        # Save report
        report_file = self.save_dir / "final_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report


# Integration with training loop
def integrate_emergent_tracking(trainer, tracker: EmergentBehaviorTracker):
    """Integrate emergent tracking with training."""
    
    # Wrap environment step
    original_step = trainer.env.step
    
    def tracked_step(action):
        result = original_step(action)
        obs, reward, terminated, truncated, info = result
        
        if terminated and 'expression' in info:
            # Get discovery path from environment
            discovery_path = trainer.env.current_state.construction_history
            
            # Log to tracker
            tracker.log_discovery(
                expression=info['expression'],
                info=info,
                discovery_path=discovery_path,
                agent_id=getattr(trainer, 'agent_id', 'main')
            )
        
        return result
    
    trainer.env.step = tracked_step
    
    # Add epoch callback
    original_train = trainer.train
    
    def tracked_train(*args, **kwargs):
        result = original_train(*args, **kwargs)
        
        # End of epoch analysis
        epoch = getattr(trainer, 'training_iteration', 0)
        summary = tracker.end_epoch(epoch)
        
        print(f"\nEmergent Behavior Summary - Epoch {epoch}:")
        print(f"  Discoveries: {summary['num_discoveries']}")
        print(f"  Clusters: {summary['num_clusters']}")
        print(f"  Phase Transitions: {summary['phase_transitions']}")
        print(f"  Diversity: {summary['diversity']:.3f}")
        
        return result
    
    trainer.train = tracked_train


# Example usage
if __name__ == "__main__":
    # Create tracker
    tracker = EmergentBehaviorTracker(save_dir="./emergent_analysis")
    
    # Simulate some discoveries
    expressions = [
        "0.5 * v**2",
        "0.5 * x**2",
        "0.5 * v**2 + 0.5 * x**2",
        "sin(x)",
        "cos(x)",
        "exp(-x**2)",
        "log(1 + x**2)",
        "x * v",
        "x**2 + v**2",
        "sqrt(x**2 + v**2)"
    ]
    
    for i, expr in enumerate(expressions):
        info = {
            'complexity': len(expr) // 3,
            'accuracy': 0.9 - i * 0.05,
            'mse': 0.01 + i * 0.001
        }
        
        path = [
            {'action': 'operator', 'value': '+'},
            {'action': 'variable', 'value': 'x'},
            {'action': 'variable', 'value': 'v'}
        ]
        
        tracker.log_discovery(expr, info, path)
    
    # End epoch
    summary = tracker.end_epoch(1)
    
    # Generate final report
    report = tracker.generate_final_report()
    
    print("\nFinal Report:")
    print(f"Total Discoveries: {report['total_discoveries']}")
    print(f"Discovery Clusters: {report['discovery_clusters']}")
    print(f"Top Discovery: {report['top_discoveries'][0] if report['top_discoveries'] else 'None'}")
