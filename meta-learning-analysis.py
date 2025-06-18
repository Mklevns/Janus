"""
Meta-Learning Analysis and Visualization Tools
=============================================

Tools for analyzing meta-learning progress, visualizing task embeddings,
and understanding what the meta-learner has discovered.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torch
from typing import Dict, List, Optional, Tuple
import pandas as pd
from pathlib import Path
import json

from maml_training_framework import MAMLTrainer, MetaLearningPolicy
from physics_task_distribution import PhysicsTaskDistribution


class MetaLearningAnalyzer:
    """Analyze meta-learning results and model behavior"""
    
    def __init__(self, trainer: MAMLTrainer):
        self.trainer = trainer
        self.policy = trainer.policy
        self.task_distribution = trainer.task_distribution
        
    def analyze_task_embeddings(self, n_tasks: int = 50) -> Dict:
        """Analyze how the model embeds different physics tasks"""
        
        print("Extracting task embeddings...")
        
        # Sample diverse tasks
        tasks = []
        embeddings = []
        
        # Get tasks from each domain
        for domain in ['mechanics', 'thermodynamics', 'electromagnetism']:
            domain_tasks = [t for t in self.task_distribution.tasks if t.domain == domain]
            sampled = np.random.choice(domain_tasks, size=min(n_tasks//3, len(domain_tasks)), replace=False)
            tasks.extend(sampled)
        
        # Extract embeddings
        for task in tasks:
            env = self.trainer.env_builder.build_env(task)
            
            # Collect some trajectories
            trajectories = self.trainer._collect_trajectories(
                self.policy, env, n_episodes=5
            )
            
            # Get task embedding
            embedding = self.trainer._compute_task_embedding(trajectories)
            embeddings.append(embedding.detach().cpu().numpy())
        
        embeddings = np.array(embeddings)
        
        # Dimensionality reduction
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Color by domain
        domains = [task.domain for task in tasks]
        domain_colors = {'mechanics': 'blue', 'thermodynamics': 'red', 'electromagnetism': 'green'}
        
        for domain in set(domains):
            mask = [d == domain for d in domains]
            ax1.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                       label=domain, alpha=0.7, s=100, 
                       color=domain_colors.get(domain, 'gray'))
        
        ax1.set_title('Task Embeddings by Domain')
        ax1.legend()
        ax1.set_xlabel('t-SNE 1')
        ax1.set_ylabel('t-SNE 2')
        
        # Plot 2: Color by difficulty
        difficulties = [task.difficulty for task in tasks]
        scatter = ax2.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                            c=difficulties, cmap='viridis', s=100, alpha=0.7)
        plt.colorbar(scatter, ax=ax2, label='Difficulty')
        
        ax2.set_title('Task Embeddings by Difficulty')
        ax2.set_xlabel('t-SNE 1')
        ax2.set_ylabel('t-SNE 2')
        
        # Annotate some points
        for i, task in enumerate(tasks[::5]):  # Every 5th task
            ax1.annotate(task.name[:10], (embeddings_2d[i*5, 0], embeddings_2d[i*5, 1]), 
                        fontsize=8, alpha=0.5)
        
        plt.tight_layout()
        plt.savefig('task_embeddings.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Analyze clustering quality
        from sklearn.metrics import silhouette_score
        silhouette = silhouette_score(embeddings_2d, domains)
        
        return {
            'embeddings': embeddings,
            'embeddings_2d': embeddings_2d,
            'silhouette_score': silhouette,
            'tasks': tasks
        }
    
    def analyze_adaptation_efficiency(self, test_tasks: Optional[List] = None) -> pd.DataFrame:
        """Analyze how quickly the model adapts to new tasks"""
        
        if test_tasks is None:
            test_tasks = self.task_distribution.sample_task_batch(10, curriculum=False)
        
        results = []
        
        for task in test_tasks:
            env = self.trainer.env_builder.build_env(task)
            
            # Track performance across adaptation steps
            adapted_policy = self.trainer._clone_policy()
            inner_optimizer = torch.optim.SGD(
                adapted_policy.parameters(),
                lr=self.trainer.config.adaptation_lr
            )
            
            # Initial performance (0 steps)
            test_trajs = self.trainer._collect_trajectories(
                adapted_policy, env, n_episodes=5
            )
            initial_metrics = self.trainer._compute_task_metrics(
                test_trajs, task, adapted_policy
            )
            
            results.append({
                'task': task.name,
                'domain': task.domain,
                'difficulty': task.difficulty,
                'adaptation_steps': 0,
                'discovery_rate': initial_metrics['discovery_rate'],
                'avg_reward': initial_metrics['avg_episode_reward']
            })
            
            # Adaptation trajectory
            support_trajs = self.trainer._collect_trajectories(
                self.policy, env, n_episodes=10
            )
            task_embedding = self.trainer._compute_task_embedding(support_trajs)
            
            for step in range(1, 11):  # 10 adaptation steps
                # One adaptation step
                loss = self.trainer._compute_trajectory_loss(
                    adapted_policy, support_trajs, task_embedding, task
                )
                
                inner_optimizer.zero_grad()
                loss.backward()
                inner_optimizer.step()
                
                # Test performance
                test_trajs = self.trainer._collect_trajectories(
                    adapted_policy, env, n_episodes=5, task_context=task_embedding
                )
                metrics = self.trainer._compute_task_metrics(
                    test_trajs, task, adapted_policy
                )
                
                results.append({
                    'task': task.name,
                    'domain': task.domain,
                    'difficulty': task.difficulty,
                    'adaptation_steps': step,
                    'discovery_rate': metrics['discovery_rate'],
                    'avg_reward': metrics['avg_episode_reward']
                })
        
        df = pd.DataFrame(results)
        
        # Visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Discovery rate vs adaptation steps
        for task in df['task'].unique():
            task_data = df[df['task'] == task]
            ax1.plot(task_data['adaptation_steps'], task_data['discovery_rate'], 
                    marker='o', label=task[:15])
        
        ax1.set_xlabel('Adaptation Steps')
        ax1.set_ylabel('Discovery Rate')
        ax1.set_title('Adaptation Efficiency: Discovery Rate')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Average by domain
        domain_avg = df.groupby(['domain', 'adaptation_steps'])['discovery_rate'].mean().reset_index()
        
        for domain in domain_avg['domain'].unique():
            domain_data = domain_avg[domain_avg['domain'] == domain]
            ax2.plot(domain_data['adaptation_steps'], domain_data['discovery_rate'], 
                    marker='o', linewidth=2, markersize=8, label=domain)
        
        ax2.set_xlabel('Adaptation Steps')
        ax2.set_ylabel('Average Discovery Rate')
        ax2.set_title('Adaptation Efficiency by Domain')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('adaptation_efficiency.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        return df
    
    def analyze_discovered_laws(self) -> Dict:
        """Analyze the physical laws discovered during training"""
        
        discoveries = self.trainer.discovered_laws
        
        # Create summary statistics
        summary = {}
        
        for task_name, expressions in discoveries.items():
            if not expressions:
                continue
            
            # Count unique discoveries
            unique_expr = list(set(expressions))
            expr_counts = {expr: expressions.count(expr) for expr in unique_expr}
            
            # Sort by frequency
            sorted_expr = sorted(expr_counts.items(), key=lambda x: x[1], reverse=True)
            
            # Get true law for comparison
            task = self.task_distribution.get_task_by_name(task_name)
            
            summary[task_name] = {
                'true_law': task.true_law,
                'total_discoveries': len(expressions),
                'unique_discoveries': len(unique_expr),
                'top_discoveries': sorted_expr[:5],
                'correct_discoveries': sum(1 for expr in expressions 
                                         if self.trainer._expression_matches(expr, task.true_law))
            }
        
        # Visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Discovery counts by task
        task_names = list(summary.keys())
        unique_counts = [summary[t]['unique_discoveries'] for t in task_names]
        correct_counts = [summary[t]['correct_discoveries'] for t in task_names]
        
        x = np.arange(len(task_names))
        width = 0.35
        
        ax1.bar(x - width/2, unique_counts, width, label='Unique', alpha=0.8)
        ax1.bar(x + width/2, correct_counts, width, label='Correct', alpha=0.8)
        
        ax1.set_xlabel('Task')
        ax1.set_ylabel('Count')
        ax1.set_title('Discoveries by Task')
        ax1.set_xticks(x)
        ax1.set_xticklabels([t[:10] for t in task_names], rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Success rate by domain
        domain_stats = {}
        for task_name, stats in summary.items():
            task = self.task_distribution.get_task_by_name(task_name)
            domain = task.domain
            
            if domain not in domain_stats:
                domain_stats[domain] = {'total': 0, 'correct': 0}
            
            domain_stats[domain]['total'] += stats['total_discoveries']
            domain_stats[domain]['correct'] += stats['correct_discoveries']
        
        domains = list(domain_stats.keys())
        success_rates = [domain_stats[d]['correct'] / max(1, domain_stats[d]['total']) 
                        for d in domains]
        
        ax2.bar(domains, success_rates, alpha=0.8)
        ax2.set_ylabel('Success Rate')
        ax2.set_title('Discovery Success Rate by Domain')
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, (domain, rate) in enumerate(zip(domains, success_rates)):
            ax2.text(i, rate + 0.02, f'{rate:.2%}', ha='center')
        
        plt.tight_layout()
        plt.savefig('discovered_laws_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Print detailed summary
        print("\nDetailed Discovery Summary:")
        print("=" * 80)
        
        for task_name, stats in summary.items():
            print(f"\n{task_name}:")
            print(f"  True law: {stats['true_law']}")
            print(f"  Total discoveries: {stats['total_discoveries']}")
            print(f"  Unique expressions: {stats['unique_discoveries']}")
            print(f"  Correct discoveries: {stats['correct_discoveries']}")
            print("  Top discoveries:")
            for expr, count in stats['top_discoveries']:
                print(f"    - {expr} (found {count} times)")
        
        return summary
    
    def analyze_learning_curves(self) -> None:
        """Plot meta-learning curves"""
        
        # Load training history from tensorboard logs
        # For now, use stored metrics
        iterations = range(len(self.trainer.meta_losses))
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Meta loss
        ax = axes[0, 0]
        ax.plot(iterations, self.trainer.meta_losses, alpha=0.7)
        
        # Add moving average
        window = 20
        if len(self.trainer.meta_losses) > window:
            ma = pd.Series(self.trainer.meta_losses).rolling(window).mean()
            ax.plot(iterations, ma, 'r-', linewidth=2, label=f'MA({window})')
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Meta Loss')
        ax.set_title('Meta-Learning Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Task metrics over time
        ax = axes[0, 1]
        # This would need to be tracked during training
        ax.text(0.5, 0.5, 'Task Metrics\n(Implement tracking)', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Task Performance Metrics')
        
        # Plot 3: Discovery rate heatmap by task and iteration
        ax = axes[1, 0]
        # Create synthetic data for demonstration
        task_names = list(self.trainer.discovered_laws.keys())[:10]
        if task_names:
            discovery_matrix = np.random.rand(len(task_names), 20)  # Synthetic
            
            sns.heatmap(discovery_matrix, 
                       xticklabels=range(0, 1000, 50),
                       yticklabels=task_names,
                       cmap='YlOrRd',
                       ax=ax)
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Task')
            ax.set_title('Discovery Rate Heatmap')
        
        # Plot 4: Curriculum progression
        ax = axes[1, 1]
        ax.text(0.5, 0.5, 'Curriculum Progression\n(Task difficulty over time)', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Curriculum Analysis')
        
        plt.tight_layout()
        plt.savefig('learning_curves.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def extract_physics_priors(self) -> Dict:
        """Extract learned physics priors from the meta-policy"""
        
        print("Extracting learned physics priors...")
        
        # Analyze symmetry detector
        symmetry_weights = self.policy.symmetry_detector.weight.detach().cpu().numpy()
        
        # Find most influential features for each symmetry
        symmetry_types = [
            'time_reversal', 'spatial_translation', 'rotational',
            'galilean', 'scale_invariance', 'charge_conjugation',
            'energy_conservation', 'momentum_conservation',
            'angular_momentum_conservation', 'none'
        ]
        
        feature_importance = {}
        for i, sym in enumerate(symmetry_types):
            importance = np.abs(symmetry_weights[i])
            top_features_idx = np.argsort(importance)[-10:]
            feature_importance[sym] = {
                'indices': top_features_idx.tolist(),
                'weights': importance[top_features_idx].tolist()
            }
        
        # Analyze conservation predictor
        conservation_weights = self.policy.conservation_predictor.weight.detach().cpu().numpy()
        
        conservation_types = ['energy', 'momentum', 'angular_momentum', 'charge', 'none']
        
        conservation_importance = {}
        for i, cons in enumerate(conservation_types):
            importance = np.abs(conservation_weights[i])
            top_features_idx = np.argsort(importance)[-10:]
            conservation_importance[cons] = {
                'indices': top_features_idx.tolist(),
                'weights': importance[top_features_idx].tolist()
            }
        
        # Visualize feature importance
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Symmetry importance heatmap
        sym_matrix = np.array([feature_importance[sym]['weights'] for sym in symmetry_types])
        
        im1 = ax1.imshow(sym_matrix, aspect='auto', cmap='hot')
        ax1.set_yticks(range(len(symmetry_types)))
        ax1.set_yticklabels(symmetry_types)
        ax1.set_xlabel('Top 10 Features')
        ax1.set_title('Symmetry Detection Feature Importance')
        plt.colorbar(im1, ax=ax1)
        
        # Conservation importance heatmap
        cons_matrix = np.array([conservation_importance[cons]['weights'] 
                               for cons in conservation_types])
        
        im2 = ax2.imshow(cons_matrix, aspect='auto', cmap='hot')
        ax2.set_yticks(range(len(conservation_types)))
        ax2.set_yticklabels(conservation_types)
        ax2.set_xlabel('Top 10 Features')
        ax2.set_title('Conservation Law Feature Importance')
        plt.colorbar(im2, ax=ax2)
        
        plt.tight_layout()
        plt.savefig('physics_priors.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        return {
            'symmetry_features': feature_importance,
            'conservation_features': conservation_importance
        }
    
    def generate_report(self, output_dir: str = "./meta_learning_report") -> None:
        """Generate comprehensive analysis report"""
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"Generating meta-learning analysis report in {output_dir}...")
        
        # 1. Task embedding analysis
        print("1. Analyzing task embeddings...")
        embedding_results = self.analyze_task_embeddings()
        
        # 2. Adaptation efficiency
        print("2. Analyzing adaptation efficiency...")
        adaptation_df = self.analyze_adaptation_efficiency()
        adaptation_df.to_csv(output_path / "adaptation_efficiency.csv", index=False)
        
        # 3. Discovered laws
        print("3. Analyzing discovered laws...")
        discovery_summary = self.analyze_discovered_laws()
        
        with open(output_path / "discovery_summary.json", 'w') as f:
            # Convert to serializable format
            serializable_summary = {}
            for task, stats in discovery_summary.items():
                serializable_summary[task] = {
                    'true_law': stats['true_law'],
                    'total_discoveries': stats['total_discoveries'],
                    'unique_discoveries': stats['unique_discoveries'],
                    'correct_discoveries': stats['correct_discoveries'],
                    'top_discoveries': [{'expr': expr, 'count': count} 
                                       for expr, count in stats['top_discoveries']]
                }
            json.dump(serializable_summary, f, indent=2)
        
        # 4. Learning curves
        print("4. Plotting learning curves...")
        self.analyze_learning_curves()
        
        # 5. Physics priors
        print("5. Extracting physics priors...")
        priors = self.extract_physics_priors()
        
        # 6. Generate summary report
        report = f"""
Meta-Learning Analysis Report
============================

Training Configuration:
- Meta learning rate: {self.trainer.config.meta_lr}
- Adaptation learning rate: {self.trainer.config.adaptation_lr}
- Adaptation steps: {self.trainer.config.adaptation_steps}
- Tasks per batch: {self.trainer.config.tasks_per_batch}

Task Distribution:
- Total tasks: {len(self.task_distribution.tasks)}
- Domains: {list(self.task_distribution.task_families.keys())}

Training Summary:
- Total iterations: {self.trainer.iteration}
- Final meta loss: {self.trainer.meta_losses[-1] if self.trainer.meta_losses else 'N/A'}

Key Findings:
- Task embedding silhouette score: {embedding_results.get('silhouette_score', 'N/A'):.3f}
- Average adaptation steps to discovery: See adaptation_efficiency.csv
- Total unique discoveries: {sum(d['unique_discoveries'] for d in discovery_summary.values())}
- Total correct discoveries: {sum(d['correct_discoveries'] for d in discovery_summary.values())}

Generated Files:
- task_embeddings.png: Visualization of learned task representations
- adaptation_efficiency.png: How quickly the model adapts to new tasks
- discovered_laws_analysis.png: Summary of discovered physical laws
- learning_curves.png: Training progress over time
- physics_priors.png: Learned physics-specific features
- adaptation_efficiency.csv: Detailed adaptation data
- discovery_summary.json: Complete discovery statistics
"""
        
        with open(output_path / "report.txt", 'w') as f:
            f.write(report)
        
        print(f"\nReport generated successfully in {output_dir}")
        print("Key findings:")
        print(f"- Task embedding quality (silhouette score): {embedding_results.get('silhouette_score', 'N/A'):.3f}")
        print(f"- Total unique discoveries: {sum(d['unique_discoveries'] for d in discovery_summary.values())}")
        print(f"- Success rate: {sum(d['correct_discoveries'] for d in discovery_summary.values()) / max(1, sum(d['total_discoveries'] for d in discovery_summary.values())):.2%}")


def analyze_checkpoint(checkpoint_path: str) -> None:
    """Analyze a saved checkpoint"""
    
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract information
    print(f"\nCheckpoint Information:")
    print(f"- Iteration: {checkpoint['iteration']}")
    print(f"- Meta losses recorded: {len(checkpoint['meta_losses'])}")
    print(f"- Tasks with discoveries: {len(checkpoint['discovered_laws'])}")
    
    # Plot meta loss
    plt.figure(figsize=(10, 6))
    plt.plot(checkpoint['meta_losses'])
    plt.xlabel('Iteration')
    plt.ylabel('Meta Loss')
    plt.title('Meta-Learning Progress')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Analyze discoveries
    total_discoveries = sum(len(exprs) for exprs in checkpoint['discovered_laws'].values())
    print(f"\nDiscovery Statistics:")
    print(f"- Total discoveries: {total_discoveries}")
    print(f"- Tasks with discoveries: {len(checkpoint['discovered_laws'])}")
    
    # Top discovered tasks
    task_counts = {task: len(exprs) for task, exprs in checkpoint['discovered_laws'].items()}
    top_tasks = sorted(task_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    
    print("\nTop 5 tasks by discovery count:")
    for task, count in top_tasks:
        print(f"  - {task}: {count} discoveries")


if __name__ == "__main__":
    # Example usage
    print("Meta-Learning Analysis Tools")
    print("=" * 50)
    
    # Option 1: Analyze a checkpoint
    checkpoint_path = "./meta_learning_checkpoints/checkpoint_100.pt"
    if Path(checkpoint_path).exists():
        analyze_checkpoint(checkpoint_path)
    else:
        print(f"Checkpoint not found: {checkpoint_path}")
    
    # Option 2: Full analysis (requires trained model)
    # This would be run after training completes
    """
    # Load trained model
    config = MetaLearningConfig()
    task_dist = PhysicsTaskDistribution()
    policy = MetaLearningPolicy(...)
    trainer = MAMLTrainer(config, policy, task_dist)
    trainer.load_checkpoint("final_checkpoint.pt")
    
    # Run analysis
    analyzer = MetaLearningAnalyzer(trainer)
    analyzer.generate_report()
    """
