import click
import pandas as pd
from pathlib import Path
import pickle # For loading ExperimentResult .pkl files
import json # For loading summary .json files
from typing import List, Dict, Any

# Assuming ExperimentRunner and ExperimentResult might be needed for context or direct use
# For now, we'll focus on loading data ExperimentRunner would have saved.
# from experiment_runner import ExperimentRunner, ExperimentResult

def load_all_results_from_dir(results_dir: Path) -> List[Dict[str, Any]]:
    """Loads all experiment result summaries (JSON) or pickles from a directory."""
    all_results_data = []
    if not results_dir.is_dir():
        click.secho(f"Error: Results directory '{results_dir}' not found.", fg="red", err=True)
        return all_results_data

    # Prefer loading from summary JSON files if they exist, fallback to pickles
    json_summaries = list(results_dir.rglob('summary_run_*.json'))
    # pickle_files = list(results_dir.rglob('run_*.pkl')) # Pickle loading is complex without class defs

    if json_summaries:
        click.echo(f"Found {len(json_summaries)} JSON summary files in {results_dir}.")
        for summary_file in json_summaries:
            try:
                with open(summary_file, 'r') as f:
                    data = json.load(f)
                    # Basic check for expected keys from ExperimentRunner._save_result summary
                    # These keys were used in ExperimentRunner's summary_data
                    expected_keys = ['config_name', 'run_id', 'discovered_law', 'symbolic_accuracy',
                                     'predictive_mse', 'wall_time_seconds', 'config_hash']
                    if all(k in data for k in expected_keys):
                        all_results_data.append(data)
                    else:
                        # Find missing keys for a more informative warning
                        missing_keys = [k for k in expected_keys if k not in data]
                        click.echo(f"Warning: JSON file {summary_file.name} missing expected keys: {missing_keys}. Skipping.")
            except json.JSONDecodeError:
                click.echo(f"Warning: Could not decode JSON from {summary_file.name}. Skipping.")
            except Exception as e:
                click.echo(f"Warning: Error loading {summary_file.name}: {e}. Skipping.")
    # Pickle loading logic commented out as it requires ExperimentResult class definition,
    # which can cause import complexities for a simple CLI script if that class is complex.
    # elif pickle_files:
    #     click.echo(f"Found {len(pickle_files)} PKL result files in {results_dir} (JSON summaries preferred but not found).")
    #     click.echo("Pickle loading for full ExperimentResult objects is complex here; focusing on JSON or pre-aggregated CSV for now.")
    else:
        click.echo(f"No JSON summary result files (summary_run_*.json) found in {results_dir} or its subdirectories.")

    return all_results_data

def basic_analysis(df: pd.DataFrame):
    """Performs and prints basic analysis similar to ExperimentRunner.analyze_results."""
    if df.empty:
        click.echo("No data to analyze.")
        return

    click.echo("\n=== Summary Statistics (Mean Performance) ===")
    # Define relevant columns for grouping and aggregation
    # Ensure these columns exist in your DataFrame (loaded from JSON/Pickle)
    # Based on the JSON summary structure:
    group_by_cols = ['config_name'] # Could add 'algorithm', 'environment_type' if they are in summary

    # Filter out non-existing columns for groupby
    valid_group_by_cols = [col for col in group_by_cols if col in df.columns]

    agg_dict = {}
    if 'symbolic_accuracy' in df.columns:
        agg_dict['mean_accuracy'] = ('symbolic_accuracy', 'mean')
        agg_dict['std_accuracy'] = ('symbolic_accuracy', 'std')
    if 'run_id' in df.columns: # Assuming run_id indicates number of runs if unique per group
        agg_dict['count_runs'] = ('run_id', 'nunique') # Or 'count' if run_id is just an index per file
    if 'predictive_mse' in df.columns:
        agg_dict['mean_mse'] = ('predictive_mse', 'mean')
    if 'wall_time_seconds' in df.columns:
        agg_dict['mean_time'] = ('wall_time_seconds', 'mean')

    if not valid_group_by_cols or not agg_dict:
        click.secho("Could not perform grouped analysis due to missing grouping or aggregation columns in the data.", fg="yellow")
        click.echo("Available columns: " + ", ".join(df.columns))
        # Print overall means if possible
        if 'symbolic_accuracy' in df.columns: click.echo(f"Overall mean symbolic_accuracy: {df['symbolic_accuracy'].mean():.3f}")
        if 'predictive_mse' in df.columns: click.echo(f"Overall mean predictive_mse: {df['predictive_mse'].mean():.3f}")
        return

    summary_stats = df.groupby(valid_group_by_cols).agg(**agg_dict).round(3)
    click.echo(summary_stats.to_string()) # .to_string() for better CLI display

    # Example: Print best performing experiments by accuracy
    if 'mean_accuracy' in summary_stats.columns: # Check if mean_accuracy was calculated
        click.echo("\n--- Top 5 Configurations by Mean Symbolic Accuracy ---")
        top_configs = summary_stats.sort_values(by='mean_accuracy', ascending=False).head(5)
        click.echo(top_configs.to_string())


@click.command()
@click.option(
    '--results-dir',
    required=True,
    help='Path to the root directory containing experiment results (e.g., ./experiments_phase1).',
    type=click.Path(exists=True, file_okay=False, resolve_path=True)
)
@click.option(
    '--output-csv',
    default=None,
    help='Optional path to save the aggregated results DataFrame as a CSV file.',
    type=click.Path(dir_okay=False, writable=True, resolve_path=True) # writable checks if parent dir exists for new file
)
def evaluate(results_dir: str, output_csv: str):
    """Load, aggregate, and analyze experiment results from a directory."""
    click.echo(f"Loading results from: {results_dir}")

    results_path = Path(results_dir)
    all_data_list = load_all_results_from_dir(results_path)

    if not all_data_list:
        click.secho("No results loaded. Exiting.", fg="yellow")
        return

    try:
        results_df = pd.DataFrame(all_data_list)
        click.echo(f"Successfully loaded {len(results_df)} result entries into a DataFrame.")
    except Exception as e:
        click.secho(f"Error creating DataFrame from loaded data: {e}", fg="red", err=True)
        return

    if output_csv:
        try:
            output_path_obj = Path(output_csv)
            output_path_obj.parent.mkdir(parents=True, exist_ok=True) # Ensure parent directory exists
            results_df.to_csv(output_path_obj, index=False)
            click.secho(f"Aggregated results saved to: {output_path_obj}", fg="green")
        except Exception as e:
            click.secho(f"Error saving CSV to {output_csv}: {e}", fg="red", err=True)

    basic_analysis(results_df)
