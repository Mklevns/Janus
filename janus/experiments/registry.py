# =============================================================================
# janus/experiments/registry.py
"""Experiment registry for dynamic experiment discovery."""

import logging
from typing import Type, Dict, Optional
from importlib import import_module

from janus.experiments.base import BaseExperiment

logger = logging.getLogger(__name__)

class ExperimentRegistry:
    """Registry for experiment classes."""
    
    _experiments: Dict[str, Type[BaseExperiment]] = {}
    _aliases: Dict[str, str] = {}
    
    @classmethod
    def register(cls, 
                 name: str, 
                 experiment_class: Type[BaseExperiment],
                 aliases: Optional[List[str]] = None):
        """Register an experiment class."""
        cls._experiments[name] = experiment_class
        logger.debug(f"Registered experiment: {name}")
        
        # Register aliases
        if aliases:
            for alias in aliases:
                cls._aliases[alias] = name
    
    @classmethod
    def get(cls, name: str) -> Optional[Type[BaseExperiment]]:
        """Get an experiment class by name or alias."""
        # Check if it's an alias
        if name in cls._aliases:
            name = cls._aliases[name]
        
        return cls._experiments.get(name)
    
    @classmethod
    def list_experiments(cls) -> List[str]:
        """List all registered experiments."""
        return list(cls._experiments.keys())
    
    @classmethod
    def auto_discover(cls):
        """Auto-discover and register experiments."""
        # Physics experiments
        try:
            from janus.experiments.physics import (
                HarmonicOscillatorDiscovery,
                PendulumDiscovery,
                KeplerDiscovery,
                GenericPhysicsDiscovery
            )
            
            cls.register("physics_harmonic_genetic", HarmonicOscillatorDiscovery,
                        aliases=["harmonic", "ho"])
            cls.register("physics_pendulum_genetic", PendulumDiscovery,
                        aliases=["pendulum"])
            cls.register("physics_kepler_genetic", KeplerDiscovery,
                        aliases=["kepler"])
            cls.register("physics_discovery", GenericPhysicsDiscovery)
            
        except ImportError as e:
            logger.warning(f"Could not import physics experiments: {e}")
        
        # AI experiments
        try:
            from janus.experiments.ai import (
                GPT2AttentionDiscovery,
                TransformerInterpretability,
                GenericAIInterpretability
            )
            
            cls.register("ai_gpt2_attention", GPT2AttentionDiscovery,
                        aliases=["gpt2-attention"])
            cls.register("ai_transformer_interpretability", TransformerInterpretability,
                        aliases=["transformer"])
            cls.register("ai_interpretability", GenericAIInterpretability)
            
        except ImportError as e:
            logger.warning(f"Could not import AI experiments: {e}")

# Auto-discover on import
ExperimentRegistry.auto_discover()
