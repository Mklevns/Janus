"""
Experiment Visualizer: Advanced Analysis and Plotting for Janus Validation
=========================================================================

Sophisticated visualization tools for interpreting experimental results across
all four phases of the validation protocol.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
from scipy import stats
from sklearn.manifold import TSNE
from pathlib import Path
import sympy as sp
import warnings
warnings.filterwarnings('ignore')


class ExperimentVisualizer:
    """Advanced visualization and analysis for Janus experiments."""
    
    def __init__(self, results_dir: str = "./experiments"):
        self.results_dir = Path(results_dir)
        self.plot_dir = self.results_dir / "plots"
        self.plot_dir.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
        
    def plot_sample_efficiency_curves(self, 
                                    results: List['ExperimentResult'],
                                    save_path: Optional[str] = None):
        """Plot learning curves showing sample efficiency."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'MSE vs Experiments', 
                'Discovery Time Distribution',
                'Complexity vs Accuracy Trade-off',
                'Convergence Rate by Algorithm'
            )
        )
        
        # Group by algorithm
        algo_groups = {}
        for result in results:
            algo = result.config.algorithm
            if algo not in algo_groups:
                algo_groups[algo] = []
            algo_groups[algo].append(result)
        
        colors = px.colors.qualitative.Plotly
        
        # 1. MSE vs Experiments
        for i, (algo, algo_results) in enumerate(algo_groups.items()):
            # Average curves across runs
            all_curves = [r.sample_efficiency_curve for r in algo_results 
                         if r.sample_efficiency_curve]
            
            if all_curves:
                # Interpolate to common x-axis
                max_exp = max(max(curve[-1][0] for curve in all_curves), 1000)
                x_common = np.linspace(0, max_exp, 100)
                y_interpolated = []
                
                for curve in all_curves:
                    if len(curve) > 1:
                        x_curve, y_curve = zip(*curve)
                        y_interp = np.interp(x_common, x_curve, y_curve)
                        y_interpolated.append(y_interp)
                
                if y_interpolated:
                    y_mean = np.mean(y_interpolated, axis=0)
                    y_std = np.std(y_interpolated, axis=0)
                    
                    # Add mean line
                    fig.add_trace(
                        go.Scatter(
                            x=x_common,
                            y=y_mean,
                            mode='lines',
                            name=algo,
                            line=dict(color=colors[i % len(colors)], width=2),
                            showlegend=True
                        ),
                        row=1, col=1
                    )
                    
                    # Add confidence interval
                    fig.add_trace(
                        go.Scatter(
                            x=np.concatenate([x_common, x_common[::-1]]),
                            y=np.concatenate([y_mean + y_std, (y_mean - y_std)[::-1]]),
                            fill='toself',
                            fillcolor=colors[i % len(colors)],
                            opacity=0.2,
                            line=dict(color='rgba(255,255,255,0)'),
                            showlegend=False
                        ),
                        row=1, col=1
                    )
        
        # 2. Discovery time distribution
        for i, (algo, algo_results) in enumerate(algo_groups.items()):
            times = [r.n_experiments_to_convergence for r in algo_results]
            
            fig.add_trace(
                go.Box(
                    y=times,
                    name=algo,
                    boxpoints='all',
                    jitter=0.3,
                    pointpos=-1.8,
                    marker=dict(color=colors[i % len(colors)])
                ),
                row=1, col=2
            )
        
        # 3. Complexity vs Accuracy
        for i, (algo, algo_results) in enumerate(algo_groups.items()):
            complexities = [r.law_complexity for r in algo_results if r.law_complexity > 0]
            accuracies = [r.symbolic_accuracy for r in algo_results if r.law_complexity > 0]
            
            if complexities and accuracies:
                fig.add_trace(
                    go.Scatter(
                        x=complexities,
                        y=accuracies,
                        mode='markers',
                        name=algo,
                        marker=dict(
                            size=10,
                            color=colors[i % len(colors)],
                            symbol=i
                        )
                    ),
                    row=2, col=1
                )
        
        # 4. Convergence rate
        for i, (algo, algo_results) in enumerate(algo_groups.items()):
            # Calculate convergence rate (experiments needed to reach 90% accuracy)
            convergence_exps = []
            for result in algo_results:
                if result.symbolic_accuracy > 0.9:
                    convergence_exps.append(result.n_experiments_to_convergence)
            
            if convergence_exps:
                fig.add_trace(
                    go.Histogram(
                        x=convergence_exps,
                        name=algo,
                        opacity=0.7,
                        marker=dict(color=colors[i % len(colors)])
                    ),
                    row=2, col=2
                )
        
        # Update layout
        fig.update_xaxes(title_text="Number of Experiments", row=1, col=1)
        fig.update_yaxes(title_text="MSE", type="log", row=1, col=1)
        
        fig.update_yaxes(title_text="Experiments to Convergence", row=1, col=2)
        
        fig.update_xaxes(title_text="Expression Complexity", row=2, col=1)
        fig.update_yaxes(title_text="Symbolic Accuracy", row=2, col=1)
        
        fig.update_xaxes(title_text="Experiments to 90% Accuracy", row=2, col=2)
        fig.update_yaxes(title_text="Count", row=2, col=2)
        
        fig.update_layout(
            height=800,
            title="Sample Efficiency Analysis",
            showlegend=True
        )
        
        if save_path:
            fig.write_html(save_path)
        else:
            fig.show()
        
        return fig
    
    def plot_noise_resilience(self,
                            df: pd.DataFrame,
                            metric: str = 'symbolic_accuracy'):
        """Plot performance vs noise level."""
        plt.figure(figsize=(10, 6))
        
        # Group by algorithm and noise level
        grouped = df.groupby(['algorithm', 'noise_level'])[metric].agg(['mean', 'std'])
        
        algorithms = df['algorithm'].unique()
        markers = ['o', 's', '^', 'D', 'v']
        
        for i, algo in enumerate(algorithms):
            algo_data = grouped.loc[algo]
            
            plt.errorbar(
                algo_data.index,
                algo_data['mean'],
                yerr=algo_data['std'],
                marker=markers[i % len(markers)],
                markersize=8,
                capsize=5,
                capthick=2,
                label=algo,
                linewidth=2
            )
        
        plt.xlabel('Noise Level (σ/signal)', fontsize=12)
        plt.ylabel(f'{metric.replace("_", " ").title()}', fontsize=12)
        plt.title('Robustness to Observational Noise', fontsize=14)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        
        # Add performance regions
        plt.axhspan(0.9, 1.0, alpha=0.1, color='green', label='Excellent')
        plt.axhspan(0.7, 0.9, alpha=0.1, color='yellow', label='Good')
        plt.axhspan(0.0, 0.7, alpha=0.1, color='red', label='Poor')
        
        plt.tight_layout()
        plt.savefig(self.plot_dir / 'noise_resilience.png', dpi=300)
        plt.show()
    
    def plot_ablation_study(self, ablation_results: Dict[str, pd.DataFrame]):
        """Visualize ablation study results."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        components = list(ablation_results.keys())
        metrics = ['symbolic_accuracy', 'predictive_mse', 'n_experiments', 'wall_time']
        metric_names = ['Symbolic Accuracy', 'Predictive MSE', 'Experiments Needed', 'Wall Time (s)']
        
        for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            ax = axes[i]
            
            # Calculate relative performance
            baseline = ablation_results.get('janus_full', pd.DataFrame())
            if baseline.empty or metric not in baseline.columns:
                continue
                
            baseline_value = baseline[metric].mean()
            
            relative_perf = []
            errors = []
            
            for component in components:
                df = ablation_results[component]
                if metric in df.columns:
                    mean_val = df[metric].mean()
                    std_val = df[metric].std()
                    
                    if metric in ['symbolic_accuracy']:
                        # Higher is better
                        rel_perf = mean_val / baseline_value if baseline_value > 0 else 0
                    else:
                        # Lower is better
                        rel_perf = baseline_value / mean_val if mean_val > 0 else 0
                    
                    relative_perf.append(rel_perf)
                    errors.append(std_val / baseline_value if baseline_value > 0 else 0)
                else:
                    relative_perf.append(0)
                    errors.append(0)
            
            # Create bar plot
            bars = ax.bar(components, relative_perf, yerr=errors, capsize=5)
            
            # Color bars based on performance
            for j, bar in enumerate(bars):
                if relative_perf[j] >= 0.9:
                    bar.set_color('green')
                elif relative_perf[j] >= 0.7:
                    bar.set_color('orange')
                else:
                    bar.set_color('red')
            
            ax.set_title(metric_name, fontsize=12)
            ax.set_ylabel('Relative Performance', fontsize=10)
            ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
            ax.set_ylim(0, 1.2)
            
            # Rotate labels
            ax.set_xticklabels(components, rotation=45, ha='right')
        
        plt.suptitle('Ablation Study: Component Contributions', fontsize=14)
        plt.tight_layout()
        plt.savefig(self.plot_dir / 'ablation_study.png', dpi=300)
        plt.show()
    
    def plot_discovered_laws_gallery(self, 
                                   results: List['ExperimentResult'],
                                   n_examples: int = 10):
        """Create a gallery of discovered laws with their properties."""
        # Filter for successful discoveries
        successful = [r for r in results if r.discovered_law and r.symbolic_accuracy > 0.5]
        
        if not successful:
            print("No successful discoveries to plot")
            return
        
        # Sort by accuracy
        successful.sort(key=lambda x: x.symbolic_accuracy, reverse=True)
        examples = successful[:n_examples]
        
        fig, axes = plt.subplots(
            nrows=(n_examples + 2) // 3, 
            ncols=3, 
            figsize=(15, 4 * ((n_examples + 2) // 3))
        )
        axes = axes.flatten() if n_examples > 3 else [axes]
        
        for i, (result, ax) in enumerate(zip(examples, axes)):
            # Parse the discovered law
            try:
                expr = sp.sympify(result.discovered_law)
                latex_expr = sp.latex(expr)
            except:
                latex_expr = result.discovered_law
            
            # Create info text
            info_text = (
                f"Algorithm: {result.config.algorithm}\n"
                f"Environment: {result.config.environment_type}\n"
                f"Accuracy: {result.symbolic_accuracy:.2%}\n"
                f"MSE: {result.predictive_mse:.2e}\n"
                f"Complexity: {result.law_complexity}\n"
                f"Time: {result.wall_time_seconds:.1f}s"
            )
            
            # Display the law
            ax.text(0.5, 0.7, f"${latex_expr}$", 
                   horizontalalignment='center',
                   verticalalignment='center',
                   transform=ax.transAxes,
                   fontsize=14,
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue"))
            
            ax.text(0.5, 0.3, info_text,
                   horizontalalignment='center',
                   verticalalignment='center',
                   transform=ax.transAxes,
                   fontsize=10,
                   family='monospace')
            
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            ax.set_title(f"Discovery {i+1}", fontsize=12)
        
        # Hide unused subplots
        for j in range(len(examples), len(axes)):
            axes[j].axis('off')
        
        plt.suptitle('Gallery of Discovered Physical Laws', fontsize=16)
        plt.tight_layout()
        plt.savefig(self.plot_dir / 'discovered_laws_gallery.png', dpi=300)
        plt.show()
    
    def plot_hypothesis_evolution(self, 
                                experiment_logs: List[Dict],
                                max_steps: int = 100):
        """Visualize how hypotheses evolve during training."""
        fig = go.Figure()
        
        # Extract hypothesis complexity and accuracy over time
        steps = []
        complexities = []
        accuracies = []
        
        for i, log_entry in enumerate(experiment_logs[:max_steps]):
            if 'hypothesis' in log_entry and 'accuracy' in log_entry:
                steps.append(i)
                complexities.append(log_entry.get('complexity', 0))
                accuracies.append(log_entry.get('accuracy', 0))
        
        # Create 3D scatter plot
        fig.add_trace(go.Scatter3d(
            x=steps,
            y=complexities,
            z=accuracies,
            mode='markers+lines',
            marker=dict(
                size=5,
                color=accuracies,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Accuracy")
            ),
            line=dict(color='darkblue', width=2),
            text=[f"Step {s}<br>Complexity: {c}<br>Accuracy: {a:.3f}" 
                  for s, c, a in zip(steps, complexities, accuracies)],
            hoverinfo='text'
        ))
        
        fig.update_layout(
            title='Hypothesis Evolution During Discovery',
            scene=dict(
                xaxis_title='Training Step',
                yaxis_title='Expression Complexity',
                zaxis_title='Accuracy',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            height=700
        )
        
        fig.write_html(self.plot_dir / 'hypothesis_evolution.html')
        return fig
    
    def plot_generalization_matrix(self,
                                 transfer_results: Dict[Tuple[str, str], float]):
        """Plot transfer learning performance as a heatmap."""
        # Extract unique environments
        envs = sorted(set([k[0] for k in transfer_results.keys()] + 
                         [k[1] for k in transfer_results.keys()]))
        
        # Create matrix
        n_envs = len(envs)
        matrix = np.zeros((n_envs, n_envs))
        
        for (from_env, to_env), score in transfer_results.items():
            i = envs.index(from_env)
            j = envs.index(to_env)
            matrix[i, j] = score
        
        # Plot heatmap
        plt.figure(figsize=(10, 8))
        
        sns.heatmap(
            matrix,
            xticklabels=envs,
            yticklabels=envs,
            annot=True,
            fmt='.2f',
            cmap='RdYlGn',
            vmin=0,
            vmax=1,
            cbar_kws={'label': 'Transfer Success Score'},
            linewidths=0.5
        )
        
        plt.title('Knowledge Transfer Between Physics Domains', fontsize=14)
        plt.xlabel('Target Domain', fontsize=12)
        plt.ylabel('Source Domain', fontsize=12)
        
        # Add diagonal line
        plt.plot([0, n_envs], [0, n_envs], 'k--', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plot_dir / 'generalization_matrix.png', dpi=300)
        plt.show()
    
    def create_summary_report(self, 
                            all_results: pd.DataFrame,
                            output_path: str = 'experiment_report.html'):
        """Generate comprehensive HTML report of all experiments."""
        html_content = f"""
        <html>
        <head>
            <title>Janus Validation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #2c3e50; }}
                h2 {{ color: #34495e; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #3498db; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                .metric {{ font-weight: bold; color: #e74c3c; }}
                .success {{ color: #27ae60; }}
                .warning {{ color: #f39c12; }}
                .failure {{ color: #e74c3c; }}
            </style>
        </head>
        <body>
            <h1>Janus Physics Discovery System - Validation Report</h1>
            <p>Generated: {pd.Timestamp.now()}</p>
            
            <h2>Executive Summary</h2>
            <ul>
                <li>Total Experiments Run: <span class="metric">{len(all_results)}</span></li>
                <li>Unique Configurations: <span class="metric">{all_results['experiment'].nunique()}</span></li>
                <li>Average Symbolic Accuracy: <span class="metric">{all_results['symbolic_accuracy'].mean():.2%}</span></li>
                <li>Best Performing Algorithm: <span class="metric">{all_results.groupby('algorithm')['symbolic_accuracy'].mean().idxmax()}</span></li>
            </ul>
            
            <h2>Phase 1: Known Law Rediscovery</h2>
            {self._generate_phase1_summary(all_results)}
            
            <h2>Phase 2: Robustness Analysis</h2>
            {self._generate_phase2_summary(all_results)}
            
            <h2>Phase 3: Ablation Studies</h2>
            {self._generate_phase3_summary(all_results)}
            
            <h2>Phase 4: Novel Discovery</h2>
            {self._generate_phase4_summary(all_results)}
            
            <h2>Detailed Results Table</h2>
            {all_results.to_html(index=False, classes='results-table')}
        </body>
        </html>
        """
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        print(f"Report saved to {output_path}")
    
    def _generate_phase1_summary(self, df: pd.DataFrame) -> str:
        """Generate Phase 1 summary for report."""
        phase1_df = df[df['noise_level'] == 0.0]
        
        if phase1_df.empty:
            return "<p>No Phase 1 experiments found.</p>"
        
        summary = "<table><tr><th>Environment</th><th>Algorithm</th><th>Success Rate</th><th>Avg Time (s)</th></tr>"
        
        for env in phase1_df['environment'].unique():
            for algo in phase1_df['algorithm'].unique():
                subset = phase1_df[(phase1_df['environment'] == env) & 
                                 (phase1_df['algorithm'] == algo)]
                if not subset.empty:
                    success_rate = (subset['symbolic_accuracy'] > 0.9).mean()
                    avg_time = subset['wall_time'].mean()
                    
                    status_class = 'success' if success_rate > 0.8 else 'warning' if success_rate > 0.5 else 'failure'
                    
                    summary += f"""
                    <tr>
                        <td>{env}</td>
                        <td>{algo}</td>
                        <td class="{status_class}">{success_rate:.0%}</td>
                        <td>{avg_time:.1f}</td>
                    </tr>
                    """
        
        summary += "</table>"
        return summary
    
    def _generate_phase2_summary(self, df: pd.DataFrame) -> str:
        """Generate Phase 2 summary for report."""
        phase2_df = df[df['noise_level'] > 0.0]
        
        if phase2_df.empty:
            return "<p>No Phase 2 experiments found.</p>"
        
        # Calculate noise resilience scores
        summary = "<h3>Noise Resilience Scores</h3><ul>"
        
        for algo in phase2_df['algorithm'].unique():
            algo_df = phase2_df[phase2_df['algorithm'] == algo]
            
            # Linear regression of accuracy vs noise
            if len(algo_df) > 1:
                slope, intercept, r_value, _, _ = stats.linregress(
                    algo_df['noise_level'], 
                    algo_df['symbolic_accuracy']
                )
                resilience = -slope  # Higher is better
                
                summary += f"<li>{algo}: {resilience:.2f} (R²={r_value**2:.2f})</li>"
        
        summary += "</ul>"
        return summary
    
    def _generate_phase3_summary(self, df: pd.DataFrame) -> str:
        """Generate Phase 3 summary for report."""
        # Filter for ablation experiments
        ablation_algos = [col for col in df['algorithm'].unique() if 'janus' in col]
        
        if len(ablation_algos) < 2:
            return "<p>Insufficient ablation experiments found.</p>"
        
        summary = "<h3>Component Contribution Analysis</h3><ul>"
        
        baseline = df[df['algorithm'] == 'janus_full']['symbolic_accuracy'].mean()
        
        for algo in ablation_algos:
            if algo != 'janus_full':
                performance = df[df['algorithm'] == algo]['symbolic_accuracy'].mean()
                contribution = (baseline - performance) / baseline * 100
                
                component = algo.replace('janus_no_', '').replace('_', ' ').title()
                summary += f"<li>{component}: {contribution:.1f}% contribution</li>"
        
        summary += "</ul>"
        return summary
    
    def _generate_phase4_summary(self, df: pd.DataFrame) -> str:
        """Generate Phase 4 summary for report."""
        # This would be populated with actual novel discovery results
        return """
        <p>Phase 4 experiments pending. Will include:</p>
        <ul>
            <li>Transfer learning results</li>
            <li>Novel law discoveries</li>
            <li>Generalization performance</li>
        </ul>
        """


# Statistical analysis functions
def perform_statistical_tests(results_df: pd.DataFrame) -> Dict[str, Any]:
    """Perform statistical significance tests on results."""
    tests = {}
    
    # 1. ANOVA for algorithm comparison
    algorithms = results_df['algorithm'].unique()
    if len(algorithms) > 2:
        groups = [results_df[results_df['algorithm'] == algo]['symbolic_accuracy'].values 
                 for algo in algorithms]
        f_stat, p_value = stats.f_oneway(*groups)
        tests['anova'] = {
            'f_statistic': f_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    # 2. Paired t-tests for specific comparisons
    if 'janus_full' in algorithms and 'genetic' in algorithms:
        janus_scores = results_df[results_df['algorithm'] == 'janus_full']['symbolic_accuracy']
        genetic_scores = results_df[results_df['algorithm'] == 'genetic']['symbolic_accuracy']
        
        if len(janus_scores) == len(genetic_scores):
            t_stat, p_value = stats.ttest_rel(janus_scores, genetic_scores)
        else:
            t_stat, p_value = stats.ttest_ind(janus_scores, genetic_scores)
        
        tests['janus_vs_genetic'] = {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'effect_size': (janus_scores.mean() - genetic_scores.mean()) / genetic_scores.std()
        }
    
    # 3. Correlation analysis
    numeric_cols = ['noise_level', 'symbolic_accuracy', 'predictive_mse', 
                   'law_complexity', 'n_experiments']
    
    correlation_matrix = results_df[numeric_cols].corr()
    tests['correlations'] = correlation_matrix.to_dict()
    
    return tests


if __name__ == "__main__":
    # Example usage
    visualizer = ExperimentVisualizer()
    
    # Load some example data
    example_df = pd.DataFrame({
        'experiment': ['exp1'] * 10 + ['exp2'] * 10,
        'algorithm': ['janus_full'] * 5 + ['genetic'] * 5 + ['janus_full'] * 5 + ['genetic'] * 5,
        'environment': ['harmonic_oscillator'] * 20,
        'noise_level': [0.0, 0.05, 0.1, 0.15, 0.2] * 4,
        'symbolic_accuracy': np.random.beta(8, 2, 20),
        'predictive_mse': np.random.exponential(0.01, 20),
        'law_complexity': np.random.randint(5, 20, 20),
        'n_experiments': np.random.randint(100, 1000, 20),
        'wall_time': np.random.exponential(100, 20)
    })
    
    # Test visualizations
    print("Testing noise resilience plot...")
    visualizer.plot_noise_resilience(example_df)
    
    print("\nPerforming statistical tests...")
    stats_results = perform_statistical_tests(example_df)
    for test_name, results in stats_results.items():
        print(f"\n{test_name}:")
        if isinstance(results, dict):
            for key, value in results.items():
                print(f"  {key}: {value}")
