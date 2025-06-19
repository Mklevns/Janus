# test_imports.py
import sys
import os

# Add project root to PYTHONPATH to allow imports from 'janus'
# and also to find root-level modules like progressive_grammar_system
# This assumes test_imports.py is run from the project root directory.
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

print("Attempting to import key modules from janus.ai_interpretability...")

try:
    from janus.ai_interpretability.environments import AIDiscoveryEnv, SymbolicDiscoveryEnv, AIInterpretabilityEnv
    print("✓ Successfully imported environments: AIDiscoveryEnv, SymbolicDiscoveryEnv, AIInterpretabilityEnv")

    # math_utils itself is exposed, and also its functions. Let's test importing a specific function.
    # The user suggested `symbolic_derivative`, but that's not in the `math_utils.py` I have.
    # I'll use `validate_inputs` which is present.
    from janus.ai_interpretability.utils.math_utils import validate_inputs
    print("✓ Successfully imported from utils.math_utils: validate_inputs")

    from janus.ai_interpretability.utils.visualization import ExperimentVisualizer
    print("✓ Successfully imported from utils.visualization: ExperimentVisualizer")

    from janus.ai_interpretability.rewards.interpretability_reward import InterpretabilityReward
    print("✓ Successfully imported from rewards.interpretability_reward: InterpretabilityReward")

    from janus.ai_interpretability.rewards.fidelity_reward import FidelityRewardCalculator
    print("✓ Successfully imported from rewards.fidelity_reward: FidelityRewardCalculator")

    from janus.ai_interpretability.grammars.neural_grammar import NeuralGrammar
    print("✓ Successfully imported from grammars.neural_grammar: NeuralGrammar")

    from janus.ai_interpretability.interpreters.base_interpreter import AILawDiscovery
    print("✓ Successfully imported from interpreters.base_interpreter: AILawDiscovery")

    from janus.ai_interpretability.utils.model_hooks import ModelHookManager
    print("✓ Successfully imported from utils.model_hooks: ModelHookManager")

    from janus.ai_interpretability.utils.expression_parser import ExpressionParser
    print("✓ Successfully imported from utils.expression_parser: ExpressionParser")

    # Test importing the main package to see if __init__ works
    import janus.ai_interpretability
    print(f"✓ Successfully imported janus.ai_interpretability package, version: {janus.ai_interpretability.VERSION}")


    # Test if a root-level dependency can still be imported by a newly structured file
    # For example, NeuralGrammar imports ProgressiveGrammar
    # This implicitly tests if NeuralGrammar could be initialized.
    ng = NeuralGrammar()
    print(f"✓ Successfully initialized NeuralGrammar (tests import of root's progressive_grammar_system)")


    print("\n✓ All key imports successful!")

except ImportError as e:
    print(f"✗ Import failed: {e}")
    print("\nEnsure that the project root directory is in your PYTHONPATH.")
    print("Also ensure all refactored files and __init__.py files are correctly in place.")

except Exception as e_other:
    print(f"✗ An unexpected error occurred during import testing: {e_other}")

print("\nRunning py_compile for potential circular dependency check (superficial)...")
# This is a very basic check. `python -m compileall .` would be more thorough if allowed.
# For now, just compiling the main package init.
try:
    import py_compile
    py_compile.compile("janus/ai_interpretability/__init__.py", doraise=True)
    print("✓ janus.ai_interpretability.__init__.py compiled successfully.")
    # To be more thorough, would compile more files or the whole directory.
    # py_compile.compile("janus/ai_interpretability/", doraise=True) # This would compile the dir
except py_compile.PyCompileError as e_compile:
    print(f"✗ Compilation Error (potential issue or circular dependency): {e_compile}")
except Exception as e_compile_other:
    print(f"✗ Error during py_compile: {e_compile_other}")

```
