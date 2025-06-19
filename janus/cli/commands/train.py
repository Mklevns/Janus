# =============================================================================
# janus/cli/commands/train.py
"""Training command implementation."""

import click
import logging
from pathlib import Path

from janus.training.trainer import UnifiedTrainer
from janus.config.loader import load_config

logger = logging.getLogger(__name__)

@click.command()
@click.option('--config', '-c', required=True, type=click.Path(exists=True),
              help='Training configuration file')
@click.option('--mode', type=click.Choice(['physics', 'ai', 'hybrid']),
              help='Training mode (overrides config)')
@click.option('--checkpoint', type=click.Path(),
              help='Resume from checkpoint')
@click.option('--output', '-o', type=click.Path(),
              default='./outputs', help='Output directory')
@click.option('--wandb/--no-wandb', default=True,
              help='Enable Weights & Biases logging')
@click.option('--distributed/--no-distributed', default=False,
              help='Enable distributed training')
@click.option('--gpus', type=int, help='Number of GPUs to use')
@click.option('--seed', type=int, help='Random seed')
def train(config, mode, checkpoint, output, wandb, distributed, gpus, seed):
    """
    Train models for discovery tasks.
    
    Examples:
    
        # Train with configuration file
        janus train --config configs/training/physics_rl.yaml
        
        # Resume training
        janus train --config configs/training/ai_maml.yaml --checkpoint model.pt
        
        # Distributed training
        janus train --config configs/training/hybrid.yaml --distributed --gpus 4
    """
    logger.info(f"Starting training with config: {config}")
    
    # Load configuration
    cfg = load_config(config)
    
    # Override with command line arguments
    if mode:
        cfg.mode = mode
    if seed is not None:
        cfg.seed = seed
    if gpus is not None:
        cfg.num_gpus = gpus
    
    cfg.use_wandb = wandb
    cfg.distributed = distributed
    cfg.output_dir = Path(output)
    
    # Create trainer
    trainer = UnifiedTrainer(config=cfg)
    
    # Resume from checkpoint if provided
    if checkpoint:
        trainer.load_checkpoint(checkpoint)
        logger.info(f"Resumed from checkpoint: {checkpoint}")
    
    # Run training
    trainer.train()
    
    # Save final model
    final_checkpoint = cfg.output_dir / "final_model.pt"
    trainer.save_checkpoint(final_checkpoint)
    
    click.echo(f"\nTraining complete! Model saved to {final_checkpoint}")
