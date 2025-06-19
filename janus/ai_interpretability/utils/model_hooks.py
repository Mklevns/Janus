import torch
import torch.nn as nn
from typing import Dict, List, Any, Callable, Tuple

# Define a type for the hook handle for better type hinting
HookHandle = Any # torch.utils.hooks.RemovableHandle

class ModelHookManager:
    """
    Manages PyTorch forward hooks for capturing model activations or other data.
    """
    def __init__(self, model: nn.Module):
        self.model = model
        self.active_hooks: List[HookHandle] = []
        self.captured_data: Dict[str, Any] = {}

    def _create_hook_fn(self, layer_name: str, data_key: str,
                        data_processor: Optional[Callable[[Any], Any]] = None) -> Callable:
        """
        Creates a generic forward hook function.

        Args:
            layer_name: Name of the layer (as known to this manager, can be arbitrary).
            data_key: Key under which data from this hook will be stored in self.captured_data.
            data_processor: Optional function to process the output tensor (e.g., detach, move to CPU).
        """
        def hook_fn(module: nn.Module, inputs: Tuple[torch.Tensor, ...], outputs: Any) -> None:
            processed_output = outputs
            if data_processor:
                processed_output = data_processor(outputs)

            # If multiple hooks target the same data_key (e.g. activations from different layers
            # but stored under a general 'activations' key), this will overwrite.
            # Usually, data_key should be unique per hook or designed to aggregate if not.
            # For simple activation capture per layer, data_key can be layer_name itself.
            self.captured_data[data_key] = processed_output
        return hook_fn

    def add_hook(self, layer_module_name: str,
                 data_key: str,
                 data_processor: Optional[Callable[[Any], Any]] = None) -> bool:
        """
        Adds a forward hook to a specific module within the model.

        Args:
            layer_module_name: The string name of the module (e.g., 'fc1', 'conv_layers.0').
            data_key: Key to store captured data under.
            data_processor: Function to process the output (e.g., lambda x: x.detach().cpu().numpy()).
                            Defaults to storing raw tensor output.

        Returns:
            True if hook was successfully registered, False otherwise.
        """
        module_found = False
        for name, module_obj in self.model.named_modules():
            if name == layer_module_name:
                hook_function = self._create_hook_fn(layer_module_name, data_key, data_processor)
                handle = module_obj.register_forward_hook(hook_function)
                self.active_hooks.append(handle)
                module_found = True
                break
        if not module_found:
            print(f"Warning: Module '{layer_module_name}' not found in model. Hook not registered.")
            return False
        return True

    def run_with_hooks(self, input_data: Any) -> Tuple[Any, Dict[str, Any]]:
        """
        Performs a forward pass with the registered hooks active, then removes them.
        Clears previously captured data before the run.

        Args:
            input_data: Data to pass to the model's forward method.

        Returns:
            A tuple containing:
                - The model's output.
                - A dictionary of all data captured by the hooks during this run.
        """
        self.clear_captured_data() # Clear data from previous runs
        self.model.eval() # Set model to evaluation mode

        # Assuming input_data is already a tensor or compatible with the model
        # If input_data needs conversion (e.g., numpy to tensor), it should be done here or before.
        # For generality, this example assumes input_data is ready.
        # Example conversion if needed:
        # if isinstance(input_data, np.ndarray):
        #     input_data = torch.FloatTensor(input_data)

        with torch.no_grad():
            model_output = self.model(input_data)

        # Hooks have populated self.captured_data during the forward pass.
        # Create a copy before clearing hooks, as some data_processors might involve tensors
        # that could be affected if hooks are removed while they are still being processed elsewhere.
        captured_data_copy = dict(self.captured_data)

        # Hooks are typically removed after use if they are temporary for one pass
        # If hooks are meant to be persistent across multiple calls, removal logic might differ.
        # For this manager, let's assume hooks added are for a specific run_with_hooks call.
        # self.remove_all_hooks() # Or manage hook lifetime differently.
        # For this specific use case (like AILawDiscovery._collect_behavior_data),
        # hooks are added, a pass is made, then hooks are removed.

        return model_output, captured_data_copy

    def clear_captured_data(self) -> None:
        """Clears any data captured from previous forward passes."""
        self.captured_data.clear()

    def remove_all_hooks(self) -> None:
        """Removes all active hooks from the model."""
        for handle in self.active_hooks:
            handle.remove()
        self.active_hooks = []

    # --- Static utility methods that might be useful ---

    @staticmethod
    def get_module_by_name(model: nn.Module, module_name: str) -> Optional[nn.Module]:
        """Utility to find a module by its string name."""
        for name, mod in model.named_modules():
            if name == module_name:
                return mod
        return None

    @staticmethod
    def default_data_processor(output_tensor: Any) -> Any:
        """Default processor: detach tensor, move to CPU, convert to numpy."""
        if isinstance(output_tensor, torch.Tensor):
            return output_tensor.detach().cpu().numpy()
        elif isinstance(output_tensor, tuple): # Handle tuple outputs from layers
            return tuple(ModelHookManager.default_data_processor(t) for t in output_tensor)
        return output_tensor # Or raise error for unexpected types


# --- Functions adapted from existing codebase ---

def register_hooks_for_layers(model: nn.Module, layer_names: List[str],
                              captured_data_dict: Dict[str, Any]) -> List[HookHandle]:
    """
    Registers forward hooks for specified layers to capture their outputs.
    This is based on the logic in AILawDiscovery._collect_behavior_data's hook registration.

    Args:
        model: The PyTorch model.
        layer_names: A list of string names of the layers to hook.
        captured_data_dict: A dictionary that will be populated by the hooks.
                            Keys will be layer names, values will be their outputs.

    Returns:
        A list of hook handles that can be used to remove the hooks.
    """
    handles = []
    for layer_to_capture in layer_names:
        module = ModelHookManager.get_module_by_name(model, layer_to_capture)
        if module:
            def hook_fn(mod, inp, outp, lname=layer_to_capture):
                # Use default_data_processor for consistent data format
                captured_data_dict[lname] = ModelHookManager.default_data_processor(outp)

            handle = module.register_forward_hook(hook_fn)
            handles.append(handle)
        else:
            print(f"Warning (register_hooks_for_layers): Layer '{layer_to_capture}' not found in model.")
    return handles


# The hook logic from AIDiscoveryEnv was more of a placeholder.
# The ModelHookManager above provides a more robust way to handle hooks.
# If AIDiscoveryEnv needs to use hooks, it could instantiate ModelHookManager.

# Example:
# class AIDiscoveryEnv:
#     def __init__(self, ai_model, ...):
#         self.ai_model = ai_model
#         self.hook_manager = ModelHookManager(ai_model)
#         self.layers_to_monitor = [...] # List of layer names
#         self._setup_env_hooks()

#     def _setup_env_hooks(self):
#         for layer_name in self.layers_to_monitor:
#             self.hook_manager.add_hook(
#                 layer_name,
#                 data_key=f"activation_{layer_name}", # Store under a specific key
#                 data_processor=ModelHookManager.default_data_processor
#             )

#     def step(self, action):
#         # ...
#         # If a model pass is needed that uses hooks:
#         # model_output, captured_hook_data = self.hook_manager.run_with_hooks(current_input_tensor)
#         # self.env_specific_captured_data.update(captured_hook_data)
#         # ...
#         pass

#     def reset(self, ...):
#         # ...
#         self.hook_manager.clear_captured_data()
#         # Potentially re-register hooks if they were removed or if model changes
#         # Or if hooks are persistent, just clear data.
#         # ...
#         pass

#     # Ensure hooks are removed if the environment is closed or deleted
#     def close(self):
#         self.hook_manager.remove_all_hooks()
#         super().close()


if __name__ == '__main__':
    # Example Usage of ModelHookManager
    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 5)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(5, 2)

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x

    model = SimpleNet()
    hook_manager = ModelHookManager(model)

    # Register a hook for 'fc1' output
    hook_manager.add_hook(
        layer_module_name='fc1',
        data_key='fc1_output',
        data_processor=ModelHookManager.default_data_processor
    )
    # Register a hook for 'relu' output
    hook_manager.add_hook(
        layer_module_name='relu',
        data_key='relu_output',
        data_processor=ModelHookManager.default_data_processor
    )

    dummy_input = torch.randn(3, 10) # Batch of 3, 10 features
    model_output, captured_data = hook_manager.run_with_hooks(dummy_input)

    print("Model Output Shape:", model_output.shape)
    print("Captured Data Keys:", captured_data.keys())
    if 'fc1_output' in captured_data:
        print("FC1 Output Shape:", captured_data['fc1_output'].shape)
    if 'relu_output' in captured_data:
        print("ReLU Output Shape:", captured_data['relu_output'].shape)

    # Hooks are still active until explicitly removed or manager is re-scoped
    hook_manager.remove_all_hooks()
    print("All hooks removed.")

    # Example using register_hooks_for_layers utility
    print("\nExample with register_hooks_for_layers:")
    captured_dict = {}
    handles = register_hooks_for_layers(model, ['fc1', 'fc2', 'non_existent_layer'], captured_dict)
    print(f"Number of handles from register_hooks_for_layers: {len(handles)}")

    _ = model(dummy_input) # Forward pass to trigger hooks

    print("Captured data via utility function:")
    for name, data in captured_dict.items():
        print(f"  {name}: shape {data.shape if hasattr(data, 'shape') else type(data)}")

    for h in handles: # Clean up
        h.remove()
    print("Utility function hooks removed.")

```
