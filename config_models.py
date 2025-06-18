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

    # Curriculum and Synthetic Data
    curriculum_stages: Optional[List[CurriculumStageConfig]] = None
    synthetic_data_params: Optional[SyntheticDataParamsConfig] = None

    # Distributed Training (Ray)
    ray_config: Optional[RayConfig] = Field(default_factory=RayConfig)

    # Hyperparameter Search
    hyperparam_search: Optional[Dict[str, Any]] = None
    validation_phases: Optional[List[str]] = None
    run_validation_suite: bool = False

    # Parameters often found in algo_params
    policy_hidden_dim: int = 256
    policy_encoder_type: str = 'transformer'

    timesteps_per_eval_cycle: int = 1000
    num_evaluation_cycles: int = 50
    ppo_rollout_length: int = 512
    ppo_n_epochs: int = 3
    ppo_batch_size: int = 64
    ppo_learning_rate: float = 3e-4
    ppo_gamma: float = 0.99
    ppo_gae_lambda: float = 0.95

    genetic_population_size: int = 100
    genetic_generations: int = 50

    tracker_autosave_interval: int = 100

    conservation_types: List[str] = Field(default_factory=lambda: ['energy', 'momentum'])
    conservation_weight_factor: float = 0.3

    symmetry_tolerance: float = 1e-4
    symmetry_confidence_threshold: float = 0.7
    expected_symmetries: List[str] = Field(default_factory=lambda: ['velocity_parity', 'time_reversal'])

    logger_backends: List[str] = Field(default_factory=lambda: ["file", "memory"])
    redis_host: str = 'localhost'
    redis_port: int = 6379
    redis_channel_base: str = 'janus_experiment_metrics'

    env_specific_params: Dict[str, Any] = Field(default_factory=dict)

    strict_mode: bool = False

    enable_conservation_detection: bool = False
    enable_symmetry_analysis: bool = False
    enable_dimensional_analysis: bool = False
    mine_abstractions_every: int = 5000
    abstraction_min_frequency: int = 3


    @model_validator(mode='after')
    def set_default_emergence_analysis_dir(self) -> 'JanusConfig':
        if self.emergence_analysis_dir is None and self.results_dir is not None:
            base_path = Path(self.results_dir) if isinstance(self.results_dir, str) else self.results_dir
            self.emergence_analysis_dir = str(base_path / "emergence")
        return self

    @model_validator(mode='after')
    def validate_config(self) -> 'JanusConfig':
        # 1. num_evaluation_cycles must be positive
        if self.num_evaluation_cycles <= 0:
            raise ValueError(f"num_evaluation_cycles must be positive, got {self.num_evaluation_cycles}")

        # 2. If curriculum_stages are defined, complexity must be non-decreasing
        if self.use_curriculum and self.curriculum_stages:
            if len(self.curriculum_stages) > 1:
                previous_complexity = -1 # Assuming complexity is always non-negative
                for i, stage in enumerate(self.curriculum_stages):
                    current_complexity = stage.max_complexity
                    if current_complexity < previous_complexity:
                        raise ValueError(
                            f"Complexity in curriculum_stages must be non-decreasing. "
                            f"Stage {i} ('{stage.name}') has max_complexity {current_complexity} "
                            f"which is less than previous stage's max_complexity {previous_complexity}."
                        )
                    previous_complexity = current_complexity

        # 3. If synthetic_data_params are defined and have a time_range, it must be valid
        if self.synthetic_data_params and self.synthetic_data_params.time_range:
            if not (isinstance(self.synthetic_data_params.time_range, list) and len(self.synthetic_data_params.time_range) == 2):
                raise ValueError(
                    f"synthetic_data_params.time_range must be a list of two integers, got {self.synthetic_data_params.time_range}"
                )
            if self.synthetic_data_params.time_range[0] >= self.synthetic_data_params.time_range[1]:
                raise ValueError(
                    f"synthetic_data_params.time_range must have time_range[0] < time_range[1], "
                    f"got {self.synthetic_data_params.time_range}"
                )

        return self

    model_config = SettingsConfigDict(
        env_prefix='JANUS_',
        extra='ignore',
    )

    @classmethod
    def from_yaml(cls, file_path: str) -> 'JanusConfig':
        import yaml
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)
