

# janus/cli/commands/discover.py
"""Discovery command implementations."""

import click
import logging
from pathlib import Path
from typing import Optional, Dict, Any

from janus.experiments.registry import ExperimentRegistry
from janus.config.models import DiscoveryConfig
from janus.utils.io import save_results

logger = logging.getLogger(__name__)

@click.group()
def discover():
    """Run discovery experiments."""
    pass

@discover.command()
@click.option('--env', '-e', required=True, 
              type=click.Choice(['harmonic', 'pendulum', 'kepler', 'custom']),
              help='Physics environment to use')
@click.option('--algorithm', '-a', 
              type=click.Choice(['genetic', 'reinforcement', 'hybrid', 'random']),
              default='genetic', help='Discovery algorithm')
@click.option('--config', '-c', type=click.Path(exists=True), 
              help='Configuration file')
@click.option('--max-complexity', type=int, default=10,
              help='Maximum expression complexity')
@click.option('--population-size', type=int, default=100,
              help='Population size for genetic algorithm')
@click.option('--generations', type=int, default=50,
              help='Number of generations')
@click.option('--output', '-o', type=click.Path(), 
              help='Output directory for results')
@click.option('--seed', type=int, help='Random seed')
@click.option('--noise', type=float, default=0.0,
              help='Noise level for observations')
def physics(env, algorithm, config, max_complexity, population_size, 
           generations, output, seed, noise):
    """
    Discover physics laws from observations.
    
    Examples:
    
        # Basic harmonic oscillator discovery
        janus discover physics --env harmonic --algorithm genetic
        
        # With custom configuration
        janus discover physics --env pendulum --config configs/physics/pendulum.yaml
        
        # High-noise scenario
        janus discover physics --env kepler --noise 0.1 --algorithm reinforcement
    """
    logger.info(f"Starting physics discovery: env={env}, algorithm={algorithm}")
    
    # Load configuration
    if config:
        cfg = load_config(config)
    else:
        cfg = DiscoveryConfig(
            mode='physics',
            environment=env,
            algorithm=algorithm,
            max_complexity=max_complexity,
            population_size=population_size,
            generations=generations,
            noise_level=noise,
            seed=seed
        )
    
    # Get experiment class
    experiment_name = f"physics_{env}_{algorithm}"
    experiment_class = ExperimentRegistry.get(experiment_name)
    
    if not experiment_class:
        # Fallback to generic physics discovery
        experiment_class = ExperimentRegistry.get("physics_discovery")
    
    # Run experiment
    experiment = experiment_class(config=cfg)
    results = experiment.execute()
    
    # Save results
    if output:
        output_path = Path(output)
        output_path.mkdir(parents=True, exist_ok=True)
        save_results(results, output_path / "results.json")
        logger.info(f"Results saved to {output_path}")
    
    # Display results
    click.echo("\n" + "="*60)
    click.echo("DISCOVERY RESULTS")
    click.echo("="*60)
    click.echo(f"Discovered Law: {results.discovered_law}")
    click.echo(f"Accuracy: {results.accuracy:.4f}")
    click.echo(f"Complexity: {results.complexity}")
    click.echo(f"Discovery Time: {results.time_seconds:.2f}s")
    
    return results

@discover.command()
@click.option('--model', '-m', required=True,
              type=click.Choice(['gpt2', 'bert', 'custom', 'neural-net']),
              help='AI model to interpret')
@click.option('--target', '-t', required=True,
              type=click.Choice(['attention', 'embeddings', 'layer', 'output']),
              help='Interpretation target')
@click.option('--layer', type=int, help='Specific layer index')
@click.option('--head', type=int, help='Specific attention head')
@click.option('--config', '-c', type=click.Path(exists=True),
              help='Configuration file')
@click.option('--data', '-d', type=click.Path(exists=True),
              help='Path to data for interpretation')
@click.option('--max-complexity', type=int, default=15,
              help='Maximum expression complexity')
@click.option('--output', '-o', type=click.Path(),
              help='Output directory for results')
@click.option('--checkpoint', type=click.Path(exists=True),
              help='Model checkpoint path')
def ai(model, target, layer, head, config, data, max_complexity, output, checkpoint):
    """
    Discover interpretable patterns in AI models.
    
    Examples:
    
        # Interpret GPT-2 attention patterns
        janus discover ai --model gpt2 --target attention --layer 0 --head 0
        
        # Interpret neural network layer
        janus discover ai --model neural-net --target layer --layer 2
        
        # With custom model
        janus discover ai --model custom --checkpoint model.pt --target output
    """
    logger.info(f"Starting AI interpretability: model={model}, target={target}")
    
    # Load configuration
    if config:
        cfg = load_config(config)
    else:
        cfg = DiscoveryConfig(
            mode='ai',
            model_type=model,
            interpretation_target=target,
            layer_index=layer,
            head_index=head,
            max_complexity=max_complexity,
            model_checkpoint=checkpoint,
            data_path=data
        )
    
    # Get experiment class
    experiment_name = f"ai_{model}_{target}"
    experiment_class = ExperimentRegistry.get(experiment_name)
    
    if not experiment_class:
        # Fallback to generic AI interpretability
        experiment_class = ExperimentRegistry.get("ai_interpretability")
    
    # Run experiment
    experiment = experiment_class(config=cfg)
    results = experiment.execute()
    
    # Save results
    if output:
        output_path = Path(output)
        output_path.mkdir(parents=True, exist_ok=True)
        save_results(results, output_path / "results.json")
        
        # Save visualizations if available
        if hasattr(results, 'visualizations'):
            for name, fig in results.visualizations.items():
                fig.savefig(output_path / f"{name}.png")
        
        logger.info(f"Results saved to {output_path}")
    
    # Display results
    click.echo("\n" + "="*60)
    click.echo("INTERPRETABILITY RESULTS")
    click.echo("="*60)
    click.echo(f"Target: {target}")
    if layer is not None:
        click.echo(f"Layer: {layer}")
    if head is not None:
        click.echo(f"Head: {head}")
    click.echo(f"Discovered Pattern: {results.discovered_pattern}")
    click.echo(f"Fidelity: {results.fidelity:.4f}")
    click.echo(f"Interpretability Score: {results.interpretability_score:.4f}")
    click.echo(f"Discovery Time: {results.time_seconds:.2f}s")
    
    return results
