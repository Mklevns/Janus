import click
import numpy as np
from typing import List, Tuple # Added for type hinting
from janus.config.loader import ConfigLoader
from janus.config.models import JanusConfig
# For direct algorithm use, we might need to import algorithm classes and environment setup
# from experiment_runner import ExperimentConfig # Not strictly needed if not creating ExperimentConfig
from janus.core.grammar import ProgressiveGrammar
from janus.core.expression import Variable

# Attempt to import SymbolicRegressor directly or use a factory if available
# This part is speculative based on ExperimentRunner's algo_registry
try:
    from physics_discovery_extensions import SymbolicRegressor
    HAS_SYMBOLIC_REGRESSOR = True
except ImportError:
    HAS_SYMBOLIC_REGRESSOR = False
    SymbolicRegressor = None # Placeholder

# Placeholder for data generation (simplified from AdvancedJanusTrainer)
def generate_simple_data(config: JanusConfig) -> Tuple[np.ndarray, List[Variable], int]:
    # Ensure config.experiment exists before accessing target_phenomena
    target_phenomena = "default"
    if config.experiment:
        target_phenomena = config.experiment.target_phenomena

    if target_phenomena == "harmonic_oscillator":
        t = np.linspace(0, 10, 500) # Shorter, simpler data for quick discovery
        x = np.sin(t) + 0.01 * np.random.randn(500)
        v = np.cos(t) + 0.01 * np.random.randn(500)
        data = np.column_stack([x, v])
        variables = [Variable('x', 0, {}), Variable('v', 1, {})]
        return data, variables, 1 # Target index for dv/dt or similar (e.g. predict v from x or vice versa)
    else:
        click.secho(f"Simplified data generation for '{target_phenomena}' not implemented for discover command. Using random data.", fg="yellow")
        data = np.random.rand(100, 3)
        variables = [Variable(f'var{i}', i, {}) for i in range(data.shape[1])] # Corrected variable count
        return data, variables, data.shape[1] - 1 # Assume last column is target

@click.command()
@click.option(
    '--config-path',
    default='config/default.yaml',
    help='Path to the Janus configuration file.',
    type=click.Path(exists=True, dir_okay=False, resolve_path=True)
)
@click.option(
    '--algorithm',
    default='genetic',
    type=click.Choice(['genetic', 'placeholder_rl']), # Add more as they become suitable for direct run
    help='Discovery algorithm to use.'
)
@click.option(
    '--target-idx',
    type=int,
    default=None,
    help='Index of the target variable in the data for regression-based algorithms. Overrides default from generate_simple_data.'
)
def discover(config_path: str, algorithm: str, target_idx: int):
    """Run a specific discovery algorithm directly on generated or simple data."""
    click.echo(f"Starting Janus discovery with algorithm: {algorithm}, config: {config_path}")

    try:
        loader = ConfigLoader(primary_config_path=config_path)
        janus_config = loader.load_resolved_config()
        click.echo("Configuration loaded successfully.")

        data, variables, default_data_target_idx = generate_simple_data(janus_config)

        # Use provided target_idx if available, else default from data gen
        current_target_idx = target_idx if target_idx is not None else default_data_target_idx

        # Validate target_idx against the actual data shape
        if not (0 <= current_target_idx < data.shape[1]):
             # Allow negative indexing if that's intended, e.g. -1 for last column
            if current_target_idx < 0 and -data.shape[1] <= current_target_idx:
                current_target_idx = data.shape[1] + current_target_idx # Convert negative to positive index
            else:
                click.secho(f"Error: target_idx {target_idx if target_idx is not None else '(default)'} resolved to {current_target_idx} is out of bounds for data with {data.shape[1]} columns.", fg="red")
                return

        click.echo(f"Using data shape: {data.shape}, Target variable: '{variables[current_target_idx].name}' (index {current_target_idx})")

        if algorithm == 'genetic':
            if not HAS_SYMBOLIC_REGRESSOR or SymbolicRegressor is None:
                click.secho("Error: SymbolicRegressor (genetic algorithm) not available. Please ensure 'physics_discovery_extensions' is importable.", fg="red")
                return

            grammar = ProgressiveGrammar() # Use a fresh grammar
            # Populate grammar with variables used in the data
            for var_obj in variables:
                grammar.variables[var_obj.name] = var_obj

            pop_size = janus_config.algorithm.hyperparameters.get('genetic_population_size', 50)
            gens = janus_config.algorithm.hyperparameters.get('genetic_generations', 20) # Increased default for CLI
            max_comp = janus_config.environment.max_complexity

            reg_params = {
                'population_size': pop_size,
                'generations': gens,
                'max_complexity': max_comp,
                'tournament_size': janus_config.algorithm.hyperparameters.get('genetic_tournament_size', 5),
                'mutation_rate': janus_config.algorithm.hyperparameters.get('genetic_mutation_rate', 0.1),
                'crossover_rate': janus_config.algorithm.hyperparameters.get('genetic_crossover_rate', 0.7),
                'patience': janus_config.algorithm.hyperparameters.get('genetic_patience', 10), # Example patience
            }
            click.echo(f"SymbolicRegressor params: {reg_params}")

            regressor = SymbolicRegressor(grammar=grammar, **reg_params)

            y = data[:, current_target_idx]
            X_indices = [i for i in range(data.shape[1]) if i != current_target_idx]
            X = data[:, X_indices]

            # Create var_mapping for SymbolicRegressor: maps "x0", "x1"... to original variable names in X
            var_mapping = {f"x{new_idx}": variables[original_idx].name for new_idx, original_idx in enumerate(X_indices)}

            click.echo(f"Fitting SymbolicRegressor (genetic algorithm) to predict '{variables[current_target_idx].name}' from {list(var_mapping.values())}...")
            best_expr_obj = regressor.fit(X, y, var_mapping=var_mapping)

            if best_expr_obj and hasattr(best_expr_obj, 'symbolic'):
                raw_expr_str = str(best_expr_obj.symbolic)
                # SymbolicRegressor's fit method should ideally use var_mapping to return expression with original names
                discovered_law = raw_expr_str
                complexity = getattr(best_expr_obj, 'complexity', len(raw_expr_str))
                mse_attr = getattr(best_expr_obj, 'mse', 'N/A')
                if isinstance(mse_attr, (float, int)):
                    mse_str = f"{mse_attr:.4e}"
                else:
                    mse_str = str(mse_attr)

                click.secho(f"\nDiscovered Law: {discovered_law}", fg="green")
                click.secho(f"  Complexity: {complexity}", fg="green")
                click.secho(f"  MSE: {mse_str}", fg="green")
            else:
                click.secho("Genetic algorithm did not return a valid expression.", fg="yellow")

        elif algorithm == 'placeholder_rl':
            click.secho("Placeholder for a direct RL discovery run (not fully implemented).", fg="yellow")
            # This would involve setting up SymbolicDiscoveryEnv and a PPO trainer for a short run.
        else:
            click.secho(f"Algorithm '{algorithm}' not yet supported for direct discovery.", fg="red")

    except FileNotFoundError as e:
        click.secho(f"Error: Configuration file not found. {e}", fg="red", err=True)
    except ValueError as e:
        click.secho(f"Error: Invalid configuration or value. {e}", fg="red", err=True)
    except ImportError as e:
        click.secho(f"Error: A required module could not be imported. {e}", fg="red", err=True)
    except Exception as e:
        click.secho(f"An unexpected error occurred: {e}", fg="red", err=True)
        # import traceback
        # click.secho(traceback.format_exc(), fg="red", err=True) # For debugging
        # raise # Uncomment for full traceback during development
