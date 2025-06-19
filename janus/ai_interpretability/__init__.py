# Init file for janus.ai_interpretability module

"""
Note: This package may depend on root-level modules not yet fully integrated
into the 'janus' package structure, such as:
- janus.core.grammar (formerly progressive_grammar_system.py)
- hypothesis_policy_network
- config_models
- custom_exceptions
- etc.

Ensure the project root directory is in PYTHONPATH when using this package
if these dependencies are not co-located or installed as part of 'janus'.
"""

# Import key components from submodules to make them available at this level, e.g.:
# from .environments import AIInterpretabilityEnv, SymbolicDiscoveryEnv
# from .grammars import NeuralGrammar
# from .interpreters import AILawDiscovery
# from .rewards import InterpretabilityReward, FidelityRewardCalculator
# from .utils import ExperimentVisualizer, ModelHookManager

# Or, users can import directly from submodules:
# import janus.ai_interpretability.environments as environments
# etc.

# For now, keep this minimal. Users can import from submodules.
# If specific classes are very commonly used, they can be exposed here.

VERSION = "0.1.0" # Example version

# Example of selectively exposing key classes:
from .grammars import NeuralGrammar
from .environments import (
    AIBehaviorData,
    AIInterpretabilityEnv,
    LocalInterpretabilityEnv,
    TransformerInterpretabilityEnv,
    SymbolicDiscoveryEnv,
    AIDiscoveryEnv
)
from .rewards import InterpretabilityReward, FidelityRewardCalculator
from .interpreters import AILawDiscovery
from .utils import (
    ExperimentVisualizer,
    ExpressionParser,
    ModelHookManager,
    register_hooks_for_layers,
    math_utils # Expose the math_utils module itself
)

__all__ = [
    "NeuralGrammar",
    "AIBehaviorData",
    "AIInterpretabilityEnv",
    "LocalInterpretabilityEnv",
    "TransformerInterpretabilityEnv",
    "SymbolicDiscoveryEnv",
    "AIDiscoveryEnv",
    "InterpretabilityReward",
    "FidelityRewardCalculator",
    "AILawDiscovery",
    "ExperimentVisualizer",
    "ExpressionParser",
    "ModelHookManager",
    "register_hooks_for_layers",
    "math_utils",
    "VERSION",
]
