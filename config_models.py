# config_models.py

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Dict, List, Optional, Any
from pathlib import Path

class CurriculumStageConfig(BaseModel):
    name: str
    max_depth: int
    max_complexity: int
    success_threshold: float
    episodes_required: int = 1000

    # Additional fields for distributed training
    ppo_rollout_length: Optional[int] = None
    ppo_learning_rate: Optional[float] = None
    exploration_bonus: Optional[float] = None


class SyntheticDataParamsConfig(BaseModel):
    n_samples: int
    noise_level: float
    time_range: List[int]


class RayConfig(BaseModel):
    num_cpus: Optional[int] = 8
    num_gpus: Optional[int] = None
    object_store_memory: Optional[int] = None
    placement_group_strategy: Optional[str] = None
    include_dashboard: Optional[bool] = False
    dashboard_host: Optional[str] = "127.0.0.1"
    temp_dir: Optional[str] = Field(None, alias="_temp_dir")
    local_mode: Optional[bool] = False

    class Config:
        populate_by_name = True


class RewardConfig(BaseModel):
    completion_bonus: float = 0.1
    mse_weight: float = -1.0
    complexity_penalty: float = -0.01
    depth_penalty: float = -0.001
    novelty_bonus: float = 0.2
    conservation_bonus: float = 0.5


class JanusConfig(BaseSettings):
    """Master configuration for Janus training."""

    # Environment
    env_type: str = "physics_discovery"
    max_depth: int = 10
    max_complexity: int = 30
    target_phenomena: str = "harmonic_oscillator"

    # Training
    training_mode: str = "advanced"
    total_timesteps: int = 1_000_000
    n_agents: int = 4
    use_curriculum: bool = True

    # Self-play
    league_size: int = 50
    opponent_sampling: str = "prioritized_quality_diversity"
    snapshot_interval: int = 10000

    # Distributed
    num_workers: int = 8
    num_gpus: int = 4
    use_pbt: bool = True

    # Monitoring
    track_emergence: bool = True
    wandb_project: str = "janus-physics-discovery"
    wandb_entity: Optional[str] = None
    checkpoint_freq: int = 10000
    log_interval: int = 100

    # Paths
    data_dir: str = "./data"
    checkpoint_dir: str = "./checkpoints"
    results_dir: str = "./results"
    emergence_analysis_dir: Optional[str] = None

    # Reward configuration
    reward_config: RewardConfig = Field(default_factory=RewardConfig)

    # Curriculum stages - moved from distributed_training.py
    curriculum_stages: List[CurriculumStageConfig] = Field(
        default_factory=lambda: [
            CurriculumStageConfig(
                name="basic_patterns",
                max_depth=3,
                max_complexity=5,
                success_threshold=0.8,
                episodes_required=1000,
                ppo_rollout_length=32,
                ppo_learning_rate=3e-4,
                exploration_bonus=0.1
            ),
            CurriculumStageConfig(
                name="simple_laws",
                max_depth=5,
                max_complexity=10,
                success_threshold=0.7,
                episodes_required=2000,
                ppo_rollout_length=64,
                ppo_learning_rate=1e-4,
                exploration_bonus=0.05
            ),
            CurriculumStageConfig(
                name="complex_laws",
                max_depth=7,
                max_complexity=15,
                success_threshold=0.6,
                episodes_required=5000,
                ppo_rollout_length=128,
                ppo_learning_rate=5e-5,
                exploration_bonus=0.01
            ),
            CurriculumStageConfig(
                name="full_complexity",
                max_depth=10,
                max_complexity=30,
                success_threshold=0.5,
                episodes_required=10000,
                ppo_rollout_length=256,
                ppo_learning_rate=1e-5,
                exploration_bonus=0.0
            )
        ]
    )

    # Synthetic data parameters
    synthetic_data_params: Optional[SyntheticDataParamsConfig] = None

    # ... rest of the config remains the same ...

    @model_validator(mode='after')
    def validate_curriculum_stages(self):
        """Ensure curriculum stages are properly ordered."""
        if self.curriculum_stages:
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
