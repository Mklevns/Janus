import click
from janus.cli.commands.train import train
from janus.cli.commands.evaluate import evaluate
from janus.cli.commands.discover import discover
from janus.cli.commands.visualize import visualize

# janus/cli/main.py
"""Main CLI interface for Janus framework."""

import click
import logging
from pathlib import Path
from typing import Optional

from janus.cli.commands import discover, train, evaluate, visualize, benchmark
from janus.config.loader import load_config
from janus.utils.logging import setup_logging

logger = logging.getLogger(__name__)

@click.group()
@click.version_option(version="0.2.0", prog_name="Janus")
@click.option('--debug', is_flag=True, help='Enable debug logging')
@click.option('--quiet', is_flag=True, help='Minimize output')
@click.pass_context
def cli(ctx, debug, quiet):
    """
    Janus: Unified framework for physics discovery and AI interpretability.
    
    Examples:
    
        # Discover physics laws
        janus discover physics --env harmonic --algorithm genetic
        
        # Interpret AI models  
        janus discover ai --model gpt2 --target attention
        
        # Train a model
        janus train --config configs/default.yaml
        
        # Run benchmarks
        janus benchmark --suite physics-basic
    """
    # Setup logging based on flags
    log_level = logging.WARNING if quiet else (logging.DEBUG if debug else logging.INFO)
    setup_logging(level=log_level)
    
    # Store context for subcommands
    ctx.ensure_object(dict)
    ctx.obj['debug'] = debug
    ctx.obj['quiet'] = quiet

# Add commands
cli.add_command(discover.discover)
cli.add_command(train.train)
cli.add_command(evaluate.evaluate)
cli.add_command(visualize.visualize)
cli.add_command(benchmark.benchmark)

# Backward compatibility aliases
cli.add_command(train.train, name="train-advanced")
cli.add_command(discover.discover, name="run-experiment")
