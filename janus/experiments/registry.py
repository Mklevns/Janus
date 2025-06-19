# janus/experiments/registry.py
"""
Experiment Registry System for Janus Framework

Provides automatic discovery, registration, and management of experiments
for both physics discovery and AI interpretability.
"""

import inspect
import importlib
import pkgutil
import logging
from typing import Type, Dict, List, Optional, Callable, Any, Union
from pathlib import Path
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import yaml

from janus.experiments.base import BaseExperiment
from janus.config.models import ExperimentConfig
from janus.utils.exceptions import ExperimentNotFoundError, InvalidExperimentError

logger = logging.getLogger(__name__)


@dataclass
class ExperimentMetadata:
    """Metadata for registered experiments."""
    name: str
    display_name: str
    description: str
    category: str  # 'physics', 'ai', 'hybrid'
    tags: List[str] = field(default_factory=list)
    supported_algorithms: List[str] = field(default_factory=list)
    supported_environments: List[str] = field(default_factory=list)
    config_schema: Optional[Dict[str, Any]] = None
    citation: Optional[str] = None
    

class ExperimentRegistry:
    """
    Central registry for all Janus experiments.
    
    Supports:
    - Manual registration via decorator
    - Automatic discovery from modules
    - Categorization and tagging
    - Configuration validation
    - Plugin system support
    """
    
    _experiments: Dict[str, Type[BaseExperiment]] = {}
    _metadata: Dict[str, ExperimentMetadata] = {}
    _aliases: Dict[str, str] = {}
    _categories: Dict[str, List[str]] = {
        'physics': [],
        'ai': [],
        'hybrid': []
    }
    
    @classmethod
    def register(cls, 
                 name: Optional[str] = None,
                 category: str = 'physics',
                 aliases: Optional[List[str]] = None,
                 **metadata_kwargs) -> Callable:
        """
        Decorator for registering experiments.
        
        Args:
            name: Unique experiment name (defaults to class name)
            category: Experiment category ('physics', 'ai', 'hybrid')
            aliases: Alternative names for the experiment
            **metadata_kwargs: Additional metadata fields
            
        Example:
            @ExperimentRegistry.register(
                name="harmonic_oscillator",
                category="physics",
                aliases=["ho", "harmonic"],
                description="Discover harmonic oscillator laws",
                tags=["classical", "oscillator"]
            )
            class HarmonicOscillatorExperiment(BaseExperiment):
                ...
        """
        def decorator(experiment_class: Type[BaseExperiment]) -> Type[BaseExperiment]:
            # Determine experiment name
            exp_name = name or experiment_class.__name__.lower()
            
            # Validate experiment class
            if not issubclass(experiment_class, BaseExperiment):
                raise InvalidExperimentError(
                    f"{experiment_class} must inherit from BaseExperiment"
                )
            
            # Register the experiment
            cls._experiments[exp_name] = experiment_class
            
            # Create metadata
            metadata = ExperimentMetadata(
                name=exp_name,
                display_name=experiment_class.__name__,
                description=metadata_kwargs.get('description', experiment_class.__doc__ or ''),
                category=category,
                **metadata_kwargs
            )
            cls._metadata[exp_name] = metadata
            
            # Register category
            if category in cls._categories:
                cls._categories[category].append(exp_name)
            
            # Register aliases
            if aliases:
                for alias in aliases:
                    cls._aliases[alias] = exp_name
                    
            logger.info(f"Registered experiment: {exp_name} ({category})")
            
            # Add metadata to class for introspection
            experiment_class._registry_metadata = metadata
            
            return experiment_class
            
        return decorator
    
    @classmethod
    def get(cls, name: str) -> Type[BaseExperiment]:
        """Get experiment class by name or alias."""
        # Check aliases first
        if name in cls._aliases:
            name = cls._aliases[name]
            
        if name not in cls._experiments:
            # Try case-insensitive search
            for exp_name in cls._experiments:
                if exp_name.lower() == name.lower():
                    name = exp_name
                    break
            else:
                raise ExperimentNotFoundError(
                    f"Experiment '{name}' not found. "
                    f"Available: {', '.join(cls._experiments.keys())}"
                )
                
        return cls._experiments[name]
    
    @classmethod
    def get_metadata(cls, name: str) -> ExperimentMetadata:
        """Get experiment metadata."""
        if name in cls._aliases:
            name = cls._aliases[name]
        return cls._metadata.get(name)
    
    @classmethod
    def list_experiments(cls, 
                        category: Optional[str] = None,
                        tags: Optional[List[str]] = None) -> List[str]:
        """
        List registered experiments with optional filtering.
        
        Args:
            category: Filter by category ('physics', 'ai', 'hybrid')
            tags: Filter by tags (experiments must have all specified tags)
            
        Returns:
            List of experiment names
        """
        experiments = list(cls._experiments.keys())
        
        # Filter by category
        if category:
            experiments = [
                name for name in experiments
                if cls._metadata[name].category == category
            ]
            
        # Filter by tags
        if tags:
            experiments = [
                name for name in experiments
                if all(tag in cls._metadata[name].tags for tag in tags)
            ]
            
        return sorted(experiments)
    
    @classmethod
    def get_experiment_info(cls, name: str) -> Dict[str, Any]:
        """Get detailed information about an experiment."""
        metadata = cls.get_metadata(name)
        experiment_class = cls.get(name)
        
        return {
            'name': metadata.name,
            'display_name': metadata.display_name,
            'description': metadata.description,
            'category': metadata.category,
            'tags': metadata.tags,
            'aliases': [alias for alias, target in cls._aliases.items() if target == name],
            'supported_algorithms': metadata.supported_algorithms,
            'supported_environments': metadata.supported_environments,
            'class': experiment_class.__name__,
            'module': experiment_class.__module__,
            'citation': metadata.citation
        }
    
    @classmethod
    def create_experiment(cls, 
                         name: str,
                         config: Union[ExperimentConfig, Dict[str, Any]],
                         **kwargs) -> BaseExperiment:
        """
        Create an experiment instance with configuration.
        
        Args:
            name: Experiment name or alias
            config: Experiment configuration
            **kwargs: Additional arguments for the experiment
            
        Returns:
            Instantiated experiment
        """
        experiment_class = cls.get(name)
        
        # Convert dict to config if needed
        if isinstance(config, dict):
            config = ExperimentConfig(**config)
            
        # Validate configuration against schema if available
        metadata = cls.get_metadata(name)
        if metadata.config_schema:
            cls._validate_config(config, metadata.config_schema)
            
        # Create experiment instance
        return experiment_class(config=config, **kwargs)
    
    @classmethod
    def auto_discover(cls, 
                     package_paths: Optional[List[str]] = None,
                     pattern: str = '*_experiment.py'):
        """
        Auto-discover and register experiments from modules.
        
        Args:
            package_paths: List of package paths to search
            pattern: File pattern to match experiment modules
        """
        if package_paths is None:
            package_paths = [
                'janus.experiments.physics',
                'janus.experiments.ai',
                'janus.experiments.hybrid'
            ]
            
        for package_path in package_paths:
            try:
                cls._discover_in_package(package_path, pattern)
            except ImportError as e:
                logger.warning(f"Could not import {package_path}: {e}")
                
    @classmethod
    def _discover_in_package(cls, package_path: str, pattern: str):
        """Discover experiments in a specific package."""
        try:
            package = importlib.import_module(package_path)
        except ImportError:
            logger.warning(f"Package {package_path} not found")
            return
            
        # Get package directory
        if hasattr(package, '__path__'):
            package_dir = package.__path__[0]
        else:
            return
            
        # Iterate through modules
        for finder, module_name, is_pkg in pkgutil.iter_modules([package_dir]):
            if is_pkg:
                # Recursively discover in subpackages
                sub_package = f"{package_path}.{module_name}"
                cls._discover_in_package(sub_package, pattern)
            else:
                # Check if module matches pattern
                if pattern.replace('*', '') in module_name:
                    full_module_name = f"{package_path}.{module_name}"
                    try:
                        module = importlib.import_module(full_module_name)
                        cls._register_from_module(module)
                    except Exception as e:
                        logger.error(f"Error loading {full_module_name}: {e}")
                        
    @classmethod
    def _register_from_module(cls, module):
        """Register all experiments found in a module."""
        for name, obj in inspect.getmembers(module):
            if (inspect.isclass(obj) and 
                issubclass(obj, BaseExperiment) and 
                obj is not BaseExperiment):
                
                # Check if already registered (via decorator)
                if hasattr(obj, '_registry_metadata'):
                    continue
                    
                # Auto-register with default metadata
                exp_name = obj.__name__.lower()
                if exp_name not in cls._experiments:
                    # Infer category from module path
                    category = 'physics'  # default
                    if 'ai' in obj.__module__:
                        category = 'ai'
                    elif 'hybrid' in obj.__module__:
                        category = 'hybrid'
                        
                    cls._experiments[exp_name] = obj
                    cls._metadata[exp_name] = ExperimentMetadata(
                        name=exp_name,
                        display_name=obj.__name__,
                        description=obj.__doc__ or '',
                        category=category
                    )
                    cls._categories[category].append(exp_name)
                    
                    logger.info(f"Auto-discovered experiment: {exp_name}")
                    
    @classmethod
    def _validate_config(cls, config: ExperimentConfig, schema: Dict[str, Any]):
        """Validate configuration against schema."""
        # Simple validation - can be extended with jsonschema or pydantic
        for key, constraints in schema.items():
            value = getattr(config, key, None)
            
            if 'required' in constraints and constraints['required'] and value is None:
                raise ValueError(f"Configuration missing required field: {key}")
                
            if value is not None:
                if 'type' in constraints:
                    expected_type = constraints['type']
                    if not isinstance(value, expected_type):
                        raise TypeError(
                            f"Configuration field {key} must be {expected_type}, "
                            f"got {type(value)}"
                        )
                        
                if 'choices' in constraints and value not in constraints['choices']:
                    raise ValueError(
                        f"Configuration field {key} must be one of {constraints['choices']}, "
                        f"got {value}"
                    )
                    
    @classmethod
    def load_from_yaml(cls, yaml_path: Union[str, Path]):
        """Load experiment definitions from a YAML file."""
        with open(yaml_path, 'r') as f:
            definitions = yaml.safe_load(f)
            
        for exp_def in definitions.get('experiments', []):
            # Dynamic class creation for simple experiments
            if 'class_path' in exp_def:
                # Load existing class
                module_path, class_name = exp_def['class_path'].rsplit('.', 1)
                module = importlib.import_module(module_path)
                experiment_class = getattr(module, class_name)
            else:
                # Create simple wrapper class
                experiment_class = cls._create_wrapper_class(exp_def)
                
            # Register with metadata
            cls.register(
                name=exp_def['name'],
                category=exp_def.get('category', 'physics'),
                aliases=exp_def.get('aliases', []),
                **exp_def.get('metadata', {})
            )(experiment_class)
            
    @classmethod
    def _create_wrapper_class(cls, definition: Dict[str, Any]) -> Type[BaseExperiment]:
        """Create a wrapper experiment class from definition."""
        class DynamicExperiment(BaseExperiment):
            def __init__(self, config: ExperimentConfig, **kwargs):
                super().__init__(config, **kwargs)
                self.definition = definition
                
            def setup(self):
                # Implement based on definition
                pass
                
            def run(self, run_id: int = 0):
                # Implement based on definition
                pass
                
            def teardown(self):
                # Implement based on definition
                pass
                
        # Set class name
        DynamicExperiment.__name__ = definition['name'].replace('_', '').title() + 'Experiment'
        DynamicExperiment.__qualname__ = DynamicExperiment.__name__
        
        return DynamicExperiment
    
    @classmethod
    def export_catalog(cls, output_path: Union[str, Path]):
        """Export experiment catalog to file."""
        catalog = {
            'experiments': [
                cls.get_experiment_info(name)
                for name in cls.list_experiments()
            ],
            'categories': cls._categories,
            'total_count': len(cls._experiments)
        }
        
        output_path = Path(output_path)
        if output_path.suffix == '.yaml':
            with open(output_path, 'w') as f:
                yaml.dump(catalog, f, default_flow_style=False)
        elif output_path.suffix == '.json':
            import json
            with open(output_path, 'w') as f:
                json.dump(catalog, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {output_path.suffix}")
            
        logger.info(f"Exported experiment catalog to {output_path}")


# Convenience function for registration decorator
register_experiment = ExperimentRegistry.register


# =============================================================================
# Example Usage and Pre-registered Experiments
# =============================================================================

# Physics Experiments
@register_experiment(
    name="harmonic_oscillator",
    category="physics",
    aliases=["ho", "harmonic", "spring"],
    description="Discover laws governing harmonic oscillator motion",
    tags=["classical", "mechanics", "oscillator"],
    supported_algorithms=["genetic", "reinforcement", "hybrid"],
    supported_environments=["1d", "2d", "damped"],
    citation="Hooke, R. (1678). De Potentia Restitutiva"
)
class HarmonicOscillatorExperiment(BaseExperiment):
    """Discover F = -kx for spring systems."""
    pass


@register_experiment(
    name="pendulum",
    category="physics", 
    aliases=["simple_pendulum"],
    description="Discover pendulum motion equations",
    tags=["classical", "mechanics", "nonlinear"],
    supported_algorithms=["genetic", "reinforcement"],
    supported_environments=["simple", "damped", "driven"]
)
class PendulumExperiment(BaseExperiment):
    """Discover pendulum dynamics."""
    pass


@register_experiment(
    name="kepler_orbits",
    category="physics",
    aliases=["kepler", "planetary", "orbits"],
    description="Discover Kepler's laws of planetary motion",
    tags=["classical", "astronomy", "conservation"],
    supported_algorithms=["genetic", "reinforcement"],
    citation="Kepler, J. (1619). Harmonices Mundi"
)
class KeplerExperiment(BaseExperiment):
    """Discover laws of planetary motion."""
    pass


# AI Interpretability Experiments
@register_experiment(
    name="gpt2_attention",
    category="ai",
    aliases=["transformer_attention", "gpt2"],
    description="Interpret attention patterns in GPT-2",
    tags=["nlp", "transformer", "attention"],
    supported_algorithms=["symbolic_regression", "genetic"],
    config_schema={
        'layer_index': {'type': int, 'required': True},
        'head_index': {'type': int, 'required': False}
    }
)
class GPT2AttentionExperiment(BaseExperiment):
    """Discover symbolic patterns in GPT-2 attention."""
    pass


@register_experiment(
    name="cnn_features",
    category="ai",
    aliases=["convnet", "vision"],
    description="Interpret CNN feature detectors",
    tags=["vision", "cnn", "features"],
    supported_algorithms=["symbolic_regression", "genetic"]
)
class CNNFeatureExperiment(BaseExperiment):
    """Discover what CNN layers detect."""
    pass


# Hybrid Experiments
@register_experiment(
    name="physics_informed_nn",
    category="hybrid",
    aliases=["pinn"],
    description="Combine physics discovery with neural network interpretation",
    tags=["physics", "neural", "hybrid"],
    supported_algorithms=["meta_learning", "transfer"]
)
class PhysicsInformedNNExperiment(BaseExperiment):
    """Discover physics laws embedded in neural networks."""
    pass


# =============================================================================
# Registry Initialization and Plugin Support
# =============================================================================

class ExperimentPlugin(ABC):
    """Base class for experiment plugins."""
    
    @abstractmethod
    def register_experiments(self, registry: ExperimentRegistry):
        """Register experiments with the registry."""
        pass
        
    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Get plugin metadata."""
        pass


def load_plugins(plugin_dir: Union[str, Path]):
    """Load experiment plugins from a directory."""
    plugin_dir = Path(plugin_dir)
    
    for plugin_file in plugin_dir.glob("*.py"):
        if plugin_file.name.startswith("_"):
            continue
            
        # Import plugin module
        spec = importlib.util.spec_from_file_location(
            plugin_file.stem, 
            plugin_file
        )
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find and instantiate plugin classes
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, ExperimentPlugin) and
                    obj is not ExperimentPlugin):
                    
                    plugin = obj()
                    plugin.register_experiments(ExperimentRegistry)
                    logger.info(f"Loaded plugin: {name}")


# Auto-discover experiments on import
def initialize_registry():
    """Initialize the experiment registry."""
    # Auto-discover built-in experiments
    ExperimentRegistry.auto_discover()
    
    # Load plugins if plugin directory exists
    plugin_dir = Path(__file__).parent / "plugins"
    if plugin_dir.exists():
        load_plugins(plugin_dir)
        
    # Load user-defined experiments from config
    user_experiments = Path.home() / ".janus" / "experiments.yaml"
    if user_experiments.exists():
        ExperimentRegistry.load_from_yaml(user_experiments)
        
    logger.info(
        f"Experiment registry initialized with "
        f"{len(ExperimentRegistry._experiments)} experiments"
    )


# Initialize on import
initialize_registry()


# =============================================================================
# Integration with CLI
# =============================================================================

def get_experiment_cli_choices() -> Dict[str, List[str]]:
    """Get experiment choices for CLI commands."""
    return {
        'physics': ExperimentRegistry.list_experiments(category='physics'),
        'ai': ExperimentRegistry.list_experiments(category='ai'),
        'hybrid': ExperimentRegistry.list_experiments(category='hybrid'),
        'all': ExperimentRegistry.list_experiments()
    }


def create_experiment_from_cli(args) -> BaseExperiment:
    """Create experiment instance from CLI arguments."""
    # Determine experiment name
    if hasattr(args, 'experiment'):
        exp_name = args.experiment
    else:
        # Infer from environment and algorithm
        exp_name = f"{args.env}_{args.algorithm}"
        
    # Create configuration
    config = ExperimentConfig(
        name=exp_name,
        mode=args.mode,
        algorithm=args.algorithm,
        environment=getattr(args, 'env', None),
        **vars(args)
    )
    
    # Create and return experiment
    return ExperimentRegistry.create_experiment(exp_name, config)
