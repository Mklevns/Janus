import click
import pandas as pd
from pathlib import Path
import json # For loading summary files if needed to build up a DataFrame
import logging # Added for more consistent logging

# Attempt to import ExperimentVisualizer
try:
    from janus.ai_interpretability.utils.visualization import ExperimentVisualizer
    HAS_VISUALIZER = True
except ImportError:
    HAS_VISUALIZER = False
    ExperimentVisualizer = None # Placeholder
    # Using logging for warnings is more consistent than click.echo at import time
    logging.warning("ExperimentVisualizer not found in janus.ai_interpretability.utils.visualization. Visualization capabilities will be limited.")

# Helper function similar to one in 'evaluate.py' to load results
def load_aggregated_results(results_path: Path) -> pd.DataFrame:
    """Loads an aggregated CSV results file or tries to build one from JSON summaries."""
    # A potential conventional name, but ExperimentRunner saves with timestamp
    # aggregated_csv_file = results_path / "all_results_aggregated.csv"
    # Instead, look for any top-level CSV in results_path that seems like an aggregation

    # Prioritize loading a single aggregated CSV if one exists directly in results_path
    # (e.g., created by ExperimentRunner.run_experiment_suite)
    potential_agg_csvs = list(results_path.glob('all_results_*.csv')) # ExperimentRunner saves with timestamp
    if not potential_agg_csvs: # Check for specific name if timestamped one not found
        potential_agg_csvs = list(results_path.glob('aggregated_results.csv'))

    if potential_agg_csvs:
        # Sort by modification time if multiple, take the latest. Or just take first found.
        aggregated_csv_file = sorted(potential_agg_csvs, key=os.path.getmtime, reverse=True)[0] if potential_agg_csvs else None # Added os import for getmtime
        if aggregated_csv_file and aggregated_csv_file.exists():
            click.echo(f"Found aggregated results CSV: {aggregated_csv_file}")
            try:
                return pd.read_csv(aggregated_csv_file)
            except Exception as e:
                click.secho(f"Error reading aggregated CSV {aggregated_csv_file}: {e}. Trying JSON summaries.", fg="yellow", err=True)

    click.echo(f"No single aggregated CSV found directly in {results_path}. Trying to load from individual JSON summaries...")
    all_data = []
    # Search for JSON summaries in subdirectories (results_dir / config_hash / summary_run_*.json)
    json_summaries = list(results_path.rglob('summary_run_*.json'))
    if not json_summaries:
        click.secho(f"No JSON summary files (summary_run_*.json) found in {results_path} or its subdirectories.", fg="yellow")
        return pd.DataFrame()

    click.echo(f"Found {len(json_summaries)} JSON summary files to aggregate.")
    for summary_file in json_summaries:
        try:
            with open(summary_file, 'r') as f:
                data = json.load(f)
                all_data.append(data)
        except Exception as e:
            click.echo(f"Warning: Error loading {summary_file.name}: {e}. Skipping.")

    if not all_data:
        return pd.DataFrame()
    return pd.DataFrame(all_data)

@click.command()
@click.option(
    '--results-dir',
    required=True,
    help='Path to the root directory containing experiment results (e.g., ./experiments_phase1). This directory should contain subdirectories for each experiment run, or an aggregated CSV.',
    type=click.Path(exists=True, file_okay=False, resolve_path=True)
)
@click.option(
    '--output-dir',
    default=None,
    help='Directory to save generated visualizations. Defaults to a "visualizations" subdirectory within results-dir.',
    type=click.Path(file_okay=False, writable=True, resolve_path=True) # Writable checks if parent exists for new dir
)
def visualize(results_dir: str, output_dir: str):
    """Load experiment results and generate visualizations."""
    click.echo(f"Visualizing results from: {results_dir}")

    results_path = Path(results_dir)

    if output_dir:
        viz_output_dir = Path(output_dir)
    else:
        viz_output_dir = results_path / "visualizations_cli" # Changed default to avoid conflict with ExperimentRunner's own output

    try:
        viz_output_dir.mkdir(parents=True, exist_ok=True)
        click.echo(f"Saving visualizations to: {viz_output_dir}")
    except Exception as e:
        click.secho(f"Error creating output directory {viz_output_dir}: {e}", fg="red", err=True)
        return

    results_df = load_aggregated_results(results_path)

    if results_df.empty:
        click.secho("No results data found or loaded to visualize. Exiting.", fg="yellow")
        return

    if HAS_VISUALIZER and ExperimentVisualizer is not None:
        try:
            # ExperimentVisualizer might expect results_dir to be the parent of individual run folders
            # or the folder containing an aggregated CSV.
            # The load_aggregated_results already provides a DataFrame.
            # We pass the original results_path for context and viz_output_dir for saving.
            visualizer = ExperimentVisualizer(results_dir=str(results_path), output_dir=str(viz_output_dir))

            report_path = viz_output_dir / "experiment_summary_report_cli.html"
            if hasattr(visualizer, 'create_summary_report'):
                try:
                    visualizer.create_summary_report(results_df, output_path=report_path)
                    click.secho(f"Summary report generated: {report_path}", fg="green")
                except Exception as e_report:
                     click.secho(f"Failed to generate summary report: {e_report}", fg="yellow", err=True)
            else:
                click.echo("ExperimentVisualizer does not have 'create_summary_report'. Attempting other plots.")

            if hasattr(visualizer, 'plot_accuracy_vs_complexity'):
                if all(col in results_df.columns for col in ['algorithm', 'symbolic_accuracy', 'law_complexity']):
                    try:
                        # Assuming plot_accuracy_vs_complexity saves its own file or returns a figure
                        fig = visualizer.plot_accuracy_vs_complexity(results_df)
                        if fig: # If it returns a figure object
                             fig.savefig(viz_output_dir / "accuracy_vs_complexity.png")
                             click.secho("Accuracy vs. Complexity plot saved.", fg="green")
                        else: # Assume it saved itself
                             click.secho("Accuracy vs. Complexity plot generated (if visualizer saved it).", fg="green")
                    except Exception as e_plot_acc:
                        click.secho(f"Failed to generate accuracy vs complexity plot: {e_plot_acc}", fg="yellow", err=True)

                else:
                    click.echo("Missing columns for accuracy vs. complexity plot (need 'algorithm', 'symbolic_accuracy', 'law_complexity'). Skipping.")

            # Add more specific plot calls if ExperimentVisualizer supports them with a DataFrame
            # Example:
            # if hasattr(visualizer, 'plot_some_other_metric_distribution'):
            #     visualizer.plot_some_other_metric_distribution(results_df, 'my_metric_column')

            click.secho("Visualizations attempted using ExperimentVisualizer.", fg="green")

        except Exception as e:
            click.secho(f"Error using ExperimentVisualizer: {e}", fg="red", err=True)
            click.echo("Falling back to basic plotting if possible (not implemented in this CLI command yet).")
    else:
        click.secho("ExperimentVisualizer not available. No advanced visualizations will be generated by this command.", fg="yellow")
        # You could add very basic pandas plotting here as a fallback:
        if not results_df.empty:
            try:
                if 'symbolic_accuracy' in results_df.columns:
                    results_df['symbolic_accuracy'].plot(kind='hist', title='Basic Accuracy Histogram')
                    import matplotlib.pyplot as plt # Lazy import for basic fallback
                    plt.savefig(viz_output_dir / "basic_accuracy_histogram.png")
                    plt.close() # Close plot
                    click.echo(f"Generated a basic accuracy histogram at {viz_output_dir / 'basic_accuracy_histogram.png'}")
            except ImportError:
                click.echo("Matplotlib not available for basic fallback plots.")
            except Exception as e_basic_plot:
                click.secho(f"Error during basic plotting: {e_basic_plot}", fg="yellow", err=True)

# Required for potential_agg_csvs sorting by mtime
import os
