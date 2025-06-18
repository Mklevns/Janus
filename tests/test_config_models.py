import pytest
from pydantic import ValidationError
from config_models import JanusConfig, CurriculumStageConfig, SyntheticDataParamsConfig, RewardConfig

class TestJanusConfigValidation:

    def test_valid_default_config(self):
        """Test that a default or minimal valid config passes validation."""
        try:
            _ = JanusConfig(target_phenomena="test") # target_phenomena is required if not set by env var
        except ValueError: # Or ValidationError if Pydantic catches something first
            pytest.fail("Default JanusConfig instantiation failed validation unexpectedly.")

    def test_num_evaluation_cycles_validation(self):
        """Test validation for num_evaluation_cycles."""
        with pytest.raises(ValueError, match="num_evaluation_cycles must be positive"):
            JanusConfig(target_phenomena="test", num_evaluation_cycles=0)

        with pytest.raises(ValueError, match="num_evaluation_cycles must be positive"):
            JanusConfig(target_phenomena="test", num_evaluation_cycles=-10)

        try:
            _ = JanusConfig(target_phenomena="test", num_evaluation_cycles=1) # Valid
        except ValueError:
            pytest.fail("num_evaluation_cycles=1 should be valid.")

    def test_curriculum_stages_complexity_validation(self):
        """Test validation for non-decreasing complexity in curriculum_stages."""
        valid_stages = [
            CurriculumStageConfig(name="s1", max_depth=5, max_complexity=10, success_threshold=0.8),
            CurriculumStageConfig(name="s2", max_depth=5, max_complexity=10, success_threshold=0.8),
            CurriculumStageConfig(name="s3", max_depth=5, max_complexity=15, success_threshold=0.8),
        ]
        invalid_stages = [
            CurriculumStageConfig(name="s1", max_depth=5, max_complexity=10, success_threshold=0.8),
            CurriculumStageConfig(name="s2", max_depth=5, max_complexity=8, success_threshold=0.8), # Decreasing complexity
        ]

        # Valid stages
        try:
            _ = JanusConfig(target_phenomena="test", use_curriculum=True, curriculum_stages=valid_stages)
        except ValueError:
            pytest.fail("Valid curriculum_stages raised ValueError unexpectedly.")

        # Invalid stages
        with pytest.raises(ValueError, match="Complexity in curriculum_stages must be non-decreasing"):
            JanusConfig(target_phenomena="test", use_curriculum=True, curriculum_stages=invalid_stages)

        # Single stage (should be valid)
        try:
            _ = JanusConfig(target_phenomena="test", use_curriculum=True, curriculum_stages=[valid_stages[0]])
        except ValueError:
            pytest.fail("Single curriculum_stage raised ValueError unexpectedly.")

        # use_curriculum = False, so invalid stages should not raise error from this validator
        try:
            _ = JanusConfig(target_phenomena="test", use_curriculum=False, curriculum_stages=invalid_stages)
        except ValueError as e:
            if "Complexity in curriculum_stages must be non-decreasing" in str(e):
                 pytest.fail("Curriculum validation ran even when use_curriculum=False.")
            # Other Pydantic errors for stages might still occur if fields are invalid, but not our custom one.

    def test_synthetic_data_params_time_range_validation(self):
        """Test validation for synthetic_data_params.time_range."""

        # Valid time_range
        try:
            _ = JanusConfig(
                target_phenomena="test",
                synthetic_data_params=SyntheticDataParamsConfig(n_samples=10, noise_level=0.1, time_range=[0, 10])
            )
        except ValueError:
            pytest.fail("Valid synthetic_data_params.time_range raised ValueError unexpectedly.")

        # Invalid time_range (end <= start)
        with pytest.raises(ValueError, match="synthetic_data_params.time_range must have time_range\\[0\\] < time_range\\[1\\]"):
            JanusConfig(
                target_phenomena="test",
                synthetic_data_params=SyntheticDataParamsConfig(n_samples=10, noise_level=0.1, time_range=[10, 0])
            )
        with pytest.raises(ValueError, match="synthetic_data_params.time_range must have time_range\\[0\\] < time_range\\[1\\]"):
            JanusConfig(
                target_phenomena="test",
                synthetic_data_params=SyntheticDataParamsConfig(n_samples=10, noise_level=0.1, time_range=[5, 5])
            )

        # Invalid format (Pydantic's ValidationError should catch this typically before our custom validator)
        # Our custom validator also checks for list and length 2.
        with pytest.raises(ValueError, match="synthetic_data_params.time_range must be a list of two integers"):
            JanusConfig(
                target_phenomena="test",
                synthetic_data_params=SyntheticDataParamsConfig(n_samples=10, noise_level=0.1, time_range=[0, 1, 2]) # type: ignore
            )
        with pytest.raises(ValueError, match="synthetic_data_params.time_range must be a list of two integers"):
             JanusConfig(
                target_phenomena="test",
                synthetic_data_params=SyntheticDataParamsConfig(n_samples=10, noise_level=0.1, time_range="not_a_list") # type: ignore
            )

        # synthetic_data_params is None (should be valid)
        try:
            _ = JanusConfig(target_phenomena="test", synthetic_data_params=None)
        except ValueError:
            pytest.fail("synthetic_data_params=None raised ValueError unexpectedly.")

    def test_combined_invalid_configs(self):
        """Test that multiple validation errors can be caught (Pydantic usually raises on first error)."""
        # This test primarily ensures that if Pydantic's own validation passes,
        # our custom validator is still run. Our custom validator raises on first error it finds.
        with pytest.raises(ValueError, match="num_evaluation_cycles must be positive"): # This is likely the first check
            JanusConfig(
                target_phenomena="test",
                num_evaluation_cycles=0,
                use_curriculum=True,
                curriculum_stages=[
                    CurriculumStageConfig(name="s1", max_depth=5, max_complexity=10, success_threshold=0.8),
                    CurriculumStageConfig(name="s2", max_depth=5, max_complexity=8, success_threshold=0.8),
                ],
                synthetic_data_params=SyntheticDataParamsConfig(n_samples=10, noise_level=0.1, time_range=[10, 0])
            )
