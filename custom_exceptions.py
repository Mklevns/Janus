# custom_exceptions.py

class MissingDependencyError(ImportError):
    """Custom exception for missing optional or core dependencies."""
    pass

class PluginNotFoundError(LookupError):
    """Custom exception for when a specified plugin, algorithm, or environment cannot be found."""
    pass

class InvalidConfigError(ValueError):
    """Custom exception for errors in experiment configurations."""
    pass

class DataGenerationError(RuntimeError):
    """Custom exception for failures during data generation in an environment."""
    pass
