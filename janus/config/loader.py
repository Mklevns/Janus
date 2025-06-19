from typing import Any, Dict
import yaml
import os
from pathlib import Path

from .models import JanusConfig # Assuming JanusConfig might be a common return type or used internally

# Define a more specific type for configuration dictionaries if possible
ConfigDict = Dict[str, Any]

def load_config_file(config_path: str) -> ConfigDict:
    """
    Loads a single YAML configuration file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        A dictionary containing the configuration.

    Raises:
        FileNotFoundError: If the config_path does not exist.
        yaml.YAMLError: If there is an error parsing the YAML file.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(path, 'r') as f:
        try:
            config_data = yaml.safe_load(f)
            if not isinstance(config_data, dict):
                raise yaml.YAMLError(f"Configuration file {config_path} did not load as a dictionary.")
            return config_data
        except yaml.YAMLError as e:
            print(f"Error parsing YAML file {config_path}: {e}")
            raise

class ConfigLoader:
    """
    Manages loading of the primary Janus configuration,
    applying environment variable overrides.
    """
    def __init__(self, primary_config_path: str = "config/default.yaml"):
        """
        Initializes the ConfigLoader.

        Args:
            primary_config_path: Path to the primary YAML configuration file.
        """
        self.primary_config_path = Path(primary_config_path)
        self.loaded_config: ConfigDict = {}

        if self.primary_config_path.exists():
            self.loaded_config = load_config_file(str(self.primary_config_path))
            print(f"Loaded primary configuration from: {self.primary_config_path}")
        else:
            print(f"Warning: Primary configuration file not found at {self.primary_config_path}. Using empty base config.")

    def load_resolved_config(self,
                             apply_env_overrides: bool = True,
                             env_prefix: str = "JANUS_") -> JanusConfig:
        """
        Loads the primary configuration and applies environment variable overrides.

        Args:
            apply_env_overrides: Whether to apply environment variable overrides.
            env_prefix: Prefix for environment variables that should override config values.

        Returns:
            A JanusConfig object representing the fully resolved configuration.
        """
        # Start with a copy of the loaded primary configuration
        final_config_data = self.loaded_config.copy()

        # Apply environment variable overrides if enabled
        if apply_env_overrides:
            self._apply_env_overrides(final_config_data, prefix=env_prefix)
            if final_config_data != self.loaded_config: # Check if any overrides were actually applied
                 print("Applied environment variable overrides.")
            else:
                 print("No relevant environment variable overrides found or applied.")

        # Validate and parse the final configuration using Pydantic model
        try:
            final_config = JanusConfig(**final_config_data)
        except Exception as e:
            print(f"Error creating JanusConfig from resolved data: {e}")
            raise ValueError(f"Configuration validation failed. Base path: '{self.primary_config_path}'.") from e

        return final_config

    def _deep_merge_dicts(self, base_dict: Dict, override_dict: Dict) -> None: # Kept for potential future use, but not used by load_resolved_config directly anymore
        """
        Recursively merges override_dict into base_dict.
        Lists are typically overridden.
        """
        for key, value in override_dict.items():
            if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
                self._deep_merge_dicts(base_dict[key], value)
            else:
                base_dict[key] = value

    def _apply_env_overrides(self, config_dict: Dict, prefix: str) -> None:
        """
        Overrides values in config_dict with environment variables.
        Supports nested keys via double underscore (e.g., JANUS_TRAINING__LEARNING_RATE).
        """
        for env_var, value in os.environ.items():
            if env_var.startswith(prefix):
                key_path_str = env_var[len(prefix):].lower()
                keys = key_path_str.split('__')

                current_level = config_dict
                applied_override = False
                for i, key_segment in enumerate(keys):
                    if i == len(keys) - 1:
                        original_value = current_level.get(key_segment)
                        if value.lower() in ['true', 'false']:
                            typed_value = value.lower() == 'true'
                        elif value.isdigit():
                            typed_value = int(value)
                        else:
                            try:
                                typed_value = float(value)
                            except ValueError:
                                typed_value = value

                        if original_value != typed_value:
                            current_level[key_segment] = typed_value
                            print(f"  Overridden '{'.'.join(keys)}' with value '{typed_value}' from env var '{env_var}' (was '{original_value}')")
                            applied_override = True
                        # else:
                        #     print(f"  Env var '{env_var}' value '{typed_value}' matches existing config. No change for '{'.'.join(keys)}'.")

                    elif key_segment not in current_level or not isinstance(current_level[key_segment], dict):
                        # print(f"  Path segment '{key_segment}' not found or not a dict for env var '{env_var}'. Cannot apply override deeper.")
                        applied_override = False # Path broken
                        break
                    current_level = current_level[key_segment]
                # if not applied_override and env_var.startswith(prefix): # Log if an env var was targeted but not applied
                     # This might be too noisy if many JANUS_ env vars are set for other purposes.
                     # print(f"  Targeted env var '{env_var}' did not result in an override.")


# Example usage
if __name__ == "__main__":
    # Create a dummy default config file in config/default.yaml
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)
    default_config_content = {
        "mode": "physics",
        "experiment": {
            "name": "default_physics_exp",
            "target_phenomena": "pendulum",
            "results_dir": "results/default_physics",
            "strict_mode": False
        },
        "training": {
            "training_mode": "basic",
            "total_timesteps": 10000,
            "learning_rate": 0.003,
            "batch_size": 128
        },
        "environment": {
            "env_type": "simple_physics",
            "max_depth": 5
        },
        "algorithm": {
            "algo_name": "PPO",
            "ppo_gamma": 0.98
        },
        "reward_config": {"type": "classic_mse", "scale": 1.5} # Example, assuming RewardConfig has 'type'
    }
    default_config_file = config_dir / "default.yaml"
    with open(default_config_file, "w") as f:
        yaml.dump(default_config_content, f)

    # Set an environment variable for override example
    os.environ["JANUS_TRAINING__LEARNING_RATE"] = "0.00077"
    os.environ["JANUS_EXPERIMENT__NAME"] = "env_override_exp"
    os.environ["JANUS_ALGORITHM__PPO_GAMMA"] = "0.995" # Test float override
    os.environ["JANUS_EXPERIMENT__STRICT_MODE"] = "true" # Test bool override

    # Instantiate loader (now uses config/default.yaml by default)
    loader = ConfigLoader()

    try:
        print("\n--- Loading resolved configuration ---")
        # No experiment_name needed, it loads the primary config and applies env overrides
        resolved_janus_config = loader.load_resolved_config()

        print("\nFinal JanusConfig object:")
        print(f"  Mode: {resolved_janus_config.mode}")
        print(f"  Experiment Name: {resolved_janus_config.experiment.name}")
        print(f"  Experiment Strict Mode: {resolved_janus_config.experiment.strict_mode}")
        print(f"  Training Learning Rate: {resolved_janus_config.training.learning_rate}")
        print(f"  Training Batch Size: {resolved_janus_config.training.batch_size}")
        print(f"  Algorithm PPO Gamma: {resolved_janus_config.algorithm.ppo_gamma}")
        if resolved_janus_config.reward_config: # RewardConfig is not optional by default in JanusConfig
             print(f"  Reward Type: {resolved_janus_config.reward_config.type if hasattr(resolved_janus_config.reward_config, 'type') else 'N/A'}")


    except FileNotFoundError as e:
        print(f"Error: {e}")
    except ValueError as e:
        print(f"Configuration Error: {e}")
    finally:
        # Clean up dummy file and env vars
        if default_config_file.exists():
            os.remove(default_config_file)
        if config_dir.exists():
            try: os.rmdir(config_dir) # Only removes if empty
            except OSError: pass
        del os.environ["JANUS_TRAINING__LEARNING_RATE"]
        del os.environ["JANUS_EXPERIMENT__NAME"]
        del os.environ["JANUS_ALGORITHM__PPO_GAMMA"]
        del os.environ["JANUS_EXPERIMENT__STRICT_MODE"]

print("janus.config.loader updated.")
