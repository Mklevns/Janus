# janus/cli/commands/benchmark.py
"""Benchmark command implementation."""

import click
import logging
from typing import List, Dict, Any
import pandas as pd

from janus.experiments.benchmarks import BenchmarkSuite
from janus.utils.visualization import plot_benchmark_results

logger = logging.getLogger(__name__)

@click.command()
@click.option('--suite', '-s', 
              type=click.Choice(['physics-basic', 'physics-advanced', 
                               'ai-attention', 'ai-full', 'all']),
              default='physics-basic',
              help='Benchmark suite to run')
@click.option('--algorithms', '-a', multiple=True,
              help='Specific algorithms to benchmark')
@click.option('--output', '-o', type=click.Path(),
              help='Output directory for results')
@click.option('--plot/--no-plot', default=True,
              help='Generate result plots')
@click.option('--parallel/--no-parallel', default=False,
              help='Run benchmarks in parallel')
def benchmark(suite, algorithms, output, plot, parallel):
    """
    Run standardized benchmarks.
    
    Examples:
    
        # Run basic physics benchmarks
        janus benchmark --suite physics-basic
        
        # Compare specific algorithms
        janus benchmark --suite ai-attention --algorithms genetic reinforcement
        
        # Full benchmark suite with plots
        janus benchmark --suite all --output results/ --plot
    """
    logger.info(f"Running benchmark suite: {suite}")
    
    # Get benchmark suite
    if suite == 'all':
        suites = ['physics-basic', 'physics-advanced', 'ai-attention', 'ai-full']
    else:
        suites = [suite]
    
    all_results = []
    
    for suite_name in suites:
        click.echo(f"\nRunning {suite_name} benchmarks...")
        
        # Create benchmark suite
        benchmark = BenchmarkSuite(suite_name)
        
        # Filter algorithms if specified
        if algorithms:
            benchmark.filter_algorithms(algorithms)
        
        # Run benchmarks
        results = benchmark.run(parallel=parallel)
        all_results.extend(results)
        
        # Display results
        click.echo(f"\n{suite_name} Results:")
        click.echo("-" * 60)
        
        for result in results:
            click.echo(f"{result.name}:")
            click.echo(f"  Accuracy: {result.accuracy:.4f}")
            click.echo(f"  Time: {result.time_seconds:.2f}s")
            click.echo(f"  Success: {result.success}")
    
    # Save results
    if output:
        output_path = Path(output)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save as CSV
        df = pd.DataFrame([r.to_dict() for r in all_results])
        df.to_csv(output_path / "benchmark_results.csv", index=False)
        
        # Generate plots
        if plot:
            fig = plot_benchmark_results(all_results)
            fig.savefig(output_path / "benchmark_plots.png")
            click.echo(f"\nResults saved to {output_path}")
    
    # Summary statistics
    click.echo("\n" + "="*60)
    click.echo("BENCHMARK SUMMARY")
    click.echo("="*60)
    
    df = pd.DataFrame([r.to_dict() for r in all_results])
    summary = df.groupby('algorithm').agg({
        'accuracy': ['mean', 'std'],
        'time_seconds': ['mean', 'std'],
        'success': 'mean'
    })
    
    click.echo(summary.to_string())
