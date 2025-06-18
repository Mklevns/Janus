import sys
from pathlib import Path
from unittest.mock import MagicMock

# --- Path Setup ---
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# --- Diagnostic Import for sklearn ---
try:
    import sklearn.metrics
    print("Successfully imported sklearn.metrics in conftest.py")
except ImportError as e:
    print(f"ERROR: Failed to import sklearn.metrics in conftest.py: {e}")
    # Optionally, re-raise or sys.exit to halt tests if this core dep is missing
    # For now, just printing, the error should show up in test collection anyway if it's an issue.

# --- Global Mocks for Heavy/Problematic Imports ---

# Create a dedicated mock for torch.nn
torch_nn_mock = MagicMock(name="TorchNNMock")
torch_nn_mock.Module = MagicMock(name="TorchNNModuleMock")

# Create the main torch mock
torch_mock = MagicMock(name="GlobalTorchMock")
torch_mock.manual_seed = MagicMock(name="TorchManualSeedMock")
torch_mock.Tensor = MagicMock(name="TorchTensorMock")
torch_mock.optim = MagicMock(name="TorchOptimMock")
torch_mock.nn = torch_nn_mock

MOCK_MODULES = {
    'torch': torch_mock,
    'torch.nn': torch_nn_mock,
    'wandb': MagicMock(name="GlobalWandBMock"),
    'hypothesis_policy_network': MagicMock(name="GlobalHypothesisPolicyNetworkMock"),
    # 'symbolic_discovery_env': MagicMock(name="GlobalSymbolicDiscoveryEnvMock"), # Commented out to allow real module usage
    # sklearn is not mocked, attempting to install and use it.
}

# Apply the global mocks
for mod_name, mock_obj in MOCK_MODULES.items():
    sys.modules[mod_name] = mock_obj
