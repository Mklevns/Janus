# janus/config/models.py

from pydantic import BaseModel, Field, model_validator, field_validator
from typing import Dict, List, Optional, Any, Union # Added Union
from pathlib import Path

# --- Component Models (Existing and New) ---

class ExperimentConfig(BaseModel):
    name: str = "default_experiment"
    description: Optional[str] = None
    target_phenomena: str = "harmonic_oscillator" # Moved from old JanusConfig
    results_dir: str = "./results"               # Moved from old JanusConfig
    data_dir: str = "./data"                     # Moved from old JanusConfig
    wandb_project: Optional[str] = "janus-physics-discovery" # Moved from old JanusConfig, made optional
    wandb_entity: Optional[str] = None           # Moved from old JanusConfig
    strict_mode: bool = False                    # For experiment validation consistency

class RewardConfig(BaseModel): # Existing model, kept as is
    completion_bonus: float = 0.1
    mse_weight: float = -1.0
    complexity_penalty: float = -0.01
    depth_penalty: float = -0.001
    novelty_bonus: float = 0.2        # Existing field
    conservation_bonus: float = 0.5   # Existing field

class CurriculumStageConfig(BaseModel): # Existing model, kept as is
    name: str
    max_depth: int
    max_complexity: int
    success_threshold: float
    episodes_required: int = 1000
    ppo_rollout_length: Optional[int] = None
    ppo_learning_rate: Optional[float] = None
    exploration_bonus: Optional[float] = None

class SyntheticDataParamsConfig(BaseModel): # Existing model, kept as is
    n_samples: int = 2000
    noise_level: float = 0.05
    time_range: List[int] = Field(default_factory=lambda: [0, 20])

class TrainingConfig(BaseModel):
    training_mode: str = "advanced"         # Moved from old JanusConfig
    total_timesteps: int = 1_000_000        # Moved from old JanusConfig
    use_curriculum: bool = True             # Moved from old JanusConfig
    curriculum_stages: List[CurriculumStageConfig] = Field( # Moved from old JanusConfig
        default_factory=lambda: [
            CurriculumStageConfig(name="basic_patterns", max_depth=3, max_complexity=5, success_threshold=0.8, episodes_required=1000, ppo_rollout_length=32, ppo_learning_rate=3e-4, exploration_bonus=0.1),
            CurriculumStageConfig(name="simple_laws", max_depth=5, max_complexity=10, success_threshold=0.7, episodes_required=2000, ppo_rollout_length=64, ppo_learning_rate=1e-4, exploration_bonus=0.05),
            CurriculumStageConfig(name="complex_laws", max_depth=7, max_complexity=15, success_threshold=0.6, episodes_required=5000, ppo_rollout_length=128, ppo_learning_rate=5e-5, exploration_bonus=0.01),
            CurriculumStageConfig(name="full_complexity", max_depth=10, max_complexity=30, success_threshold=0.5, episodes_required=10000, ppo_rollout_length=256, ppo_learning_rate=1e-5, exploration_bonus=0.0)
        ]
    )
    synthetic_data_params: Optional[SyntheticDataParamsConfig] = Field(default_factory=SyntheticDataParamsConfig)
    checkpoint_freq: int = 10000            # Moved from old JanusConfig
    log_interval: int = 100                 # Moved from old JanusConfig
    checkpoint_dir: str = "./checkpoints"       # Moved from old JanusConfig
    # Placeholders for common training params, can be expanded
    learning_rate: float = 1e-4 # Global learning rate if not per stage
    batch_size: int = 64
    epochs: int = 100 # Generic epochs, might be superseded by total_timesteps

    @model_validator(mode='after')
    def validate_curriculum_stages_training(self): # Renamed validator
        if self.use_curriculum and self.curriculum_stages:
            prev_complexity = 0
            for stage in self.curriculum_stages:
                if stage.max_complexity <= prev_complexity:
                    raise ValueError(
                        f"Curriculum stage '{stage.name}' has complexity "
                        f"{stage.max_complexity} which is not greater than "
                        f"previous stage complexity {prev_complexity}"
                    )
                prev_complexity = stage.max_complexity
        return self

class EnvironmentConfig(BaseModel):
    env_type: str = "physics_discovery"     # Moved from old JanusConfig
    env_name: str = "SymbolicDiscoveryEnv"  # Placeholder
    max_depth: int = 10                     # Moved from old JanusConfig
    max_complexity: int = 30                # Moved from old JanusConfig
    params: Dict[str, Any] = Field(default_factory=dict) # For other env params

class RayConfig(BaseModel): # Existing model, kept as is
    num_cpus: Optional[int] = None # Default to None, let Ray decide or use system max
    num_gpus: Optional[int] = None
    object_store_memory: Optional[int] = None
    # placement_group_strategy: Optional[str] = None # Less common, can be added if needed
    include_dashboard: Optional[bool] = False
    dashboard_host: Optional[str] = "127.0.0.1"
    temp_dir: Optional[str] = Field(None, alias="_temp_dir") # Keep alias for env var loading
    local_mode: Optional[bool] = False

    class Config:
        populate_by_name = True


class AlgorithmConfig(BaseModel):
    algo_name: str = "PPO" # Placeholder
    hyperparameters: Dict[str, Any] = Field(default_factory=dict) # For general algo params
    # PPO specific, can be a sub-model if PPO is the main focus
    ppo_rollout_length: int = 2048
    ppo_n_epochs: int = 10
    ppo_gamma: float = 0.99
    ppo_gae_lambda: float = 0.95

    # Self-play related fields from old JanusConfig
    n_agents: int = 4                       # Moved from old JanusConfig
    league_size: int = 50                   # Moved from old JanusConfig
    opponent_sampling: str = "prioritized_quality_diversity" # Moved from old JanusConfig
    snapshot_interval: int = 10000          # Moved from old JanusConfig

    # Distributed related fields from old JanusConfig
    num_workers: int = 1 # Default to 1 for non-distributed
    num_gpus: int = 0    # Default to 0
    use_pbt: bool = False                   # Moved from old JanusConfig, default False
    ray_config: RayConfig = Field(default_factory=RayConfig)

    # Emergence tracking related fields
    track_emergence: bool = True            # Moved from old JanusConfig
    emergence_analysis_dir: Optional[str] = "./results/emergence" # Moved from old JanusConfig

# --- Main Janus Configuration Model ---

class JanusConfig(BaseModel): # Changed from BaseSettings to BaseModel as per prompt
    mode: str  # 'physics', 'ai', 'hybrid'
    run_validation_suite: bool = Field(False, description="Whether to run the validation suite after training.")
    validation_phases: List[str] = Field(default_factory=lambda: ["phase1", "phase2", "phase3"], description="Which validation phases to run.")

    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    algorithm: AlgorithmConfig = Field(default_factory=AlgorithmConfig)
    reward_config: RewardConfig = Field(default_factory=RewardConfig) # Kept from old JanusConfig

    # Retain model_config = SettingsConfigDict(env_prefix='JANUS_') if this still needs to load from env vars
    # However, BaseSettings is usually for top-level. If JanusConfig is BaseModel,
    # sub-models (if they were BaseSettings) could load from env vars.
    # For now, adhering to BaseModel for JanusConfig. Environment variable loading
    # for this top-level model would need to be handled by the loader if not using BaseSettings.

    @field_validator('mode') # Pydantic v2 validator
    @classmethod
    def mode_must_be_valid(cls, value: str) -> str:
        if value not in ['physics', 'ai', 'hybrid']:
            raise ValueError("Mode must be 'physics', 'ai', or 'hybrid'")
        return value

    # Example of a root validator if cross-field validation is needed later
    # @model_validator(mode='after')
    # def check_ai_mode_settings(self) -> 'JanusConfig':
    #     if self.mode == 'ai' and self.algorithm.algo_name == 'PPO':
    #         if not self.training.curriculum_stages: # Example check
    #             raise ValueError("AI mode with PPO requires curriculum stages to be defined.")
    #     return self

# It's good practice to add a .model_dump() example or usage note if this is the primary config object.
# Example:
# if __name__ == "__main__":
#     exp_config = ExperimentConfig(name="test_exp", target_phenomena="pendulum")
#     train_config = TrainingConfig(total_timesteps=5000)
#     env_config = EnvironmentConfig(env_name="PendulumEnv")
#     algo_config = AlgorithmConfig(algo_name="CustomPPO")
#
#     janus_conf = JanusConfig(
#         mode='physics',
#         experiment=exp_config,
#         training=train_config,
#         environment=env_config,
#         algorithm=algo_config
#     )
#     print(janus_conf.model_dump_json(indent=2))

# Note: The original JanusConfig was BaseSettings, which implies loading from environment variables.
# The new JanusConfig is BaseModel. The ConfigLoader will handle loading YAML and then environment overrides
# before parsing into this JanusConfig model.
# Sub-models like RayConfig still use `populate_by_name = True` for alias handling if they were ever BaseSettings
# or if their fields are meant to be populated from dicts using aliases.
# For BaseModel, aliases are primarily for serialization/deserialization, not direct env var loading by Pydantic.

print("janus.config.models updated with new JanusConfig structure.")
