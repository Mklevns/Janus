"""
Strict Mode Validator for Janus
================================

Provides comprehensive validation for configurations and data
when strict mode is enabled.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import logging
from dataclasses import fields

from janus.config.models import JanusConfig


class StrictModeValidator:
    """Validates configurations and data when strict mode is active."""

    def __init__(self, strict_mode: bool = True):
        self.strict_mode = strict_mode
        self.logger = logging.getLogger(__name__)

    def validate_config(self, config: JanusConfig) -> None:
        """Validate all aspects of JanusConfig."""
        if not self.strict_mode:
            return

        # Validate required fields are not None
        required_fields = ['target_phenomena', 'results_dir']
        for field_name in required_fields:
            if getattr(config, field_name, None) is None:
                raise ValueError(f"Required field '{field_name}' is None")

        # Validate numeric ranges
        self._validate_numeric_ranges(config)

        # Validate file paths
        self._validate_paths(config)

        # Validate list fields
        self._validate_lists(config)

    def _validate_numeric_ranges(self, config: JanusConfig) -> None:
        """Validate numeric fields are within reasonable ranges."""
        validations = {
            'max_depth': (1, 50),
            'max_complexity': (1, 100),
            'policy_hidden_dim': (16, 2048),
            'ppo_learning_rate': (1e-6, 1e-1),
            'ppo_gamma': (0.0, 1.0),
            'ppo_gae_lambda': (0.0, 1.0),
            'conservation_weight_factor': (0.0, 10.0),
            'symmetry_tolerance': (1e-10, 1.0),
            'symmetry_confidence_threshold': (0.0, 1.0),
        }

        for field_name, (min_val, max_val) in validations.items():
            value = getattr(config, field_name, None)
            if value is not None and not (min_val <= value <= max_val):
                raise ValueError(
                    f"{field_name} = {value} is outside valid range [{min_val}, {max_val}]"
                )

    def _validate_paths(self, config: JanusConfig) -> None:
        """Validate that required directories exist or can be created."""
        path_fields = ['results_dir', 'emergence_analysis_dir']

        for field_name in path_fields:
            path_str = getattr(config, field_name, None)
            if path_str:
                path = Path(path_str)
                try:
                    path.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    raise ValueError(f"Cannot create directory for {field_name}: {e}")

    def _validate_lists(self, config: JanusConfig) -> None:
        """Validate list fields contain valid values."""
        if config.conservation_types:
            valid_types = {'energy', 'momentum', 'angular_momentum', 'charge', 'mass'}
            for c_type in config.conservation_types:
                if c_type not in valid_types:
                    self.logger.warning(
                        f"Unknown conservation type '{c_type}'. "
                        f"Valid types: {valid_types}"
                    )

        if config.logger_backends:
            valid_backends = {'file', 'memory', 'redis', 'wandb'}
            for backend in config.logger_backends:
                if backend not in valid_backends:
                    raise ValueError(
                        f"Invalid logger backend '{backend}'. "
                        f"Valid backends: {valid_backends}"
                    )

    def validate_data(
        self,
        data: np.ndarray,
        data_name: str = "data",
        expected_dims: Optional[int] = None
    ) -> None:
        """Validate data array for common issues."""
        if not self.strict_mode:
            return

        if data is None:
            raise ValueError(f"{data_name} is None")

        # Check for NaN/Inf
        if np.any(~np.isfinite(data)):
            nan_count = np.sum(np.isnan(data))
            inf_count = np.sum(np.isinf(data))
            raise ValueError(
                f"{data_name} contains {nan_count} NaN and {inf_count} Inf values"
            )

        # Verify shape
        if expected_dims is not None and data.ndim != expected_dims:
            raise ValueError(
                f"{data_name} has {data.ndim} dimensions, expected {expected_dims}"
            )

        # Check for empty data
        if data.size == 0:
            raise ValueError(f"{data_name} is empty")

        # Check value ranges (warn for extreme values)
        max_abs_val = np.max(np.abs(data))
        if max_abs_val > 1e10:
            self.logger.warning(
                f"{data_name} contains very large values (max abs: {max_abs_val:.2e})"
            )

        # Check for constant columns (might indicate issues)
        if data.ndim == 2:
            for i in range(data.shape[1]):
                if np.all(data[:, i] == data[0, i]):
                    self.logger.warning(
                        f"{data_name} column {i} is constant"
                    )

    def _handle_error(self, message: str) -> None:
        """Handle errors based on strict mode setting."""
        if self.strict_mode:
            raise RuntimeError(f"Strict mode error: {message}")
        else:
            self.logger.error(message)
