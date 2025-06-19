import click
from janus.config.loader import ConfigLoader
from integrated_pipeline import AdvancedJanusTrainer # Assuming this is the correct import
from janus.config.models import JanusConfig # Explicitly import for type hinting if needed

@click.command()
@click.option(
    '--config-path',
    default='config/default.yaml',
    help='Path to the Janus configuration file.',
    type=click.Path(exists=True, dir_okay=False, resolve_path=True)
)
@click.option(
    '--strict',
    is_flag=True,
    help='Enable strict mode for configuration and execution.'
)
# Add other relevant options as needed, e.g., resume, specific overrides.
def train(config_path: str, strict: bool):
    """Load configuration and run the Janus training pipeline."""
    click.echo(f"Starting Janus training pipeline with config: {config_path}")

    try:
        loader = ConfigLoader(primary_config_path=config_path)
        # load_resolved_config already applies env variables.
        # The strict flag from CLI will override what's in the file or env vars.
        janus_config: JanusConfig = loader.load_resolved_config()

        # Apply CLI strict mode override if provided
        # This assumes JanusConfig.experiment.strict_mode exists.
        if hasattr(janus_config, 'experiment') and janus_config.experiment is not None:
            # Only override if the flag is actually set, to respect config/env if flag is false
            if strict:
                janus_config.experiment.strict_mode = True
            click.echo(f"Strict mode set to: {janus_config.experiment.strict_mode} "
                       f"(CLI override: {'active' if strict else 'not active'})")
        elif strict: # CLI flag was set but no place in config
            click.echo(f"Warning: --strict flag was set, but no 'experiment.strict_mode' found in JanusConfig to update.")

        click.echo("Configuration loaded successfully.")

        # Setup environment (mimicking parts of launch_advanced_training.py)
        # Note: The original launch_advanced_training.py has a setup_environment function
        # which also handles Ray initialization. This CLI command currently doesn't call it.
        # For a production CLI, refactoring setup_environment to be reusable would be ideal.
        click.echo("Skipping full environment setup (e.g., Ray init) for this basic CLI train command.")
        click.echo(f"Ensure necessary directories exist (e.g., {janus_config.experiment.results_dir}, {janus_config.experiment.checkpoint_dir}).")

        trainer = AdvancedJanusTrainer(config=janus_config)

        # Prepare data
        click.echo("Preparing data...")
        # Assuming synthetic data for CLI default, this can be made configurable
        # Also assuming data_path from config is None for synthetic generation
        data_path_from_config = janus_config.experiment.data_dir # This is a dir, not a file path for loading data.
                                                                # prepare_data needs to handle this logic.
                                                                # For now, assuming generate_synthetic=True bypasses loading.
        data = trainer.prepare_data(data_path=None, generate_synthetic=True)
        click.echo(f"Data prepared: shape {data.shape if data is not None else 'N/A'}")

        # Create environment
        click.echo("Creating environment...")
        trainer.env = trainer.create_environment(data) # data should be np.ndarray
        click.echo(f"Environment created: {type(trainer.env).__name__}")

        # Create RL trainer (PPO/etc.)
        click.echo("Initializing PPO/RL trainer...")
        trainer.trainer = trainer.create_trainer() # This is the PPO/RLlib trainer
        click.echo(f"PPO/RL trainer initialized: {type(trainer.trainer).__name__}")

        click.echo("Starting training process...")
        trainer.train() # This calls the main training loop of AdvancedJanusTrainer

        # Optionally run validation suite if specified in config
        if hasattr(janus_config, 'experiment') and janus_config.experiment is not None and janus_config.experiment.run_validation_suite:
            click.echo("Running validation suite as per config...")
            trainer.run_experiment_suite()

        click.secho("Training completed successfully!", fg="green")

    except FileNotFoundError as e:
        click.secho(f"Error: Configuration file not found. Details: {e}", fg="red", err=True)
    except ValueError as e:
        click.secho(f"Error: Invalid configuration or value. Details: {e}", fg="red", err=True)
    except Exception as e:
        click.secho(f"An unexpected error occurred during training: {e}", fg="red", err=True)
        # For debugging, you might want to re-raise or log the full traceback
        # import traceback
        # click.secho(traceback.format_exc(), fg="red", err=True)
        # raise # Re-raise if you want Click to show its full error handling for unhandled exceptions
    pass # Ensure function ends well for Click if no exception.
