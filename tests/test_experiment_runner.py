import pytest
import re # Import re at the module level

# Dummy class to hold the method for testing, as PhysicsDiscoveryExperiment is complex to instantiate
class Remapper:
    def _remap_expression(self, expr_str: str, var_mapping: dict) -> str:
        # This method's code is copied from experiment_runner.PhysicsDiscoveryExperiment._remap_expression
        # Ensure it's kept in sync if the original changes.
        # For this test, we use the version with re.sub.
        result = expr_str
        # Sort variables by length of the key (e.g., 'x10' before 'x1') to prevent partial replacements.
        sorted_vars = sorted(var_mapping.items(), key=lambda item: len(item[0]), reverse=True)
        for new_var, orig_var in sorted_vars:
            result = re.sub(r'' + re.escape(new_var) + r'', orig_var, result)
        return result

def test_remap_expression_logic():
    remapper = Remapper()

    assert remapper._remap_expression("x + 1", {"x": "position"}) == "position + 1"
    assert remapper._remap_expression("x_old + x", {"x": "position"}) == "x_old + position"
    assert remapper._remap_expression("x + x1", {"x": "long", "x1": "short"}) == "long + short"
    assert remapper._remap_expression("v + vel", {"v": "velocity", "vel": "nope"}) == "velocity + nope"
    assert remapper._remap_expression("vel + v", {"vel": "nope", "v": "velocity"}) == "nope + velocity"
    assert remapper._remap_expression("x* + y", {"x*": "x_mult"}) == "x_mult + y"
    assert remapper._remap_expression("data[0] + data[1]", {"data[0]": "d0", "data[1]": "d1"}) == "d0 + d1"
    assert remapper._remap_expression("temp is high, temporary_value", {"temp": "temperature"}) == "temperature is high, temporary_value"
    assert remapper._remap_expression("x is value", {"x": "position"}) == "position is value"
    assert remapper._remap_expression("value is x", {"x": "position"}) == "value is position"
    assert remapper._remap_expression("x + y * x", {"x": "var_x", "y": "var_y"}) == "var_x + var_y * var_x"
    assert remapper._remap_expression("x1 + x10", {'x1': 'pos', 'x10': 'vel'}) == "pos + vel"
    assert remapper._remap_expression("x10 + x1", {'x1': 'pos', 'x10': 'vel'}) == "vel + pos"
    # Test case where new_var is a substring of another word, should not replace due to 
    assert remapper._remap_expression("vexing problem", {"vex": "vexation"}) == "vexing problem"
    # Test case with numbers and variable names like x1, x2
    assert remapper._remap_expression("x1+x2", {"x1": "a", "x2": "b"}) == "a+b"
    # Test with empty string
    assert remapper._remap_expression("", {"x": "y"}) == ""
    # Test with no matching variables
    assert remapper._remap_expression("a + b", {"x": "y"}) == "a + b"
    # Test with mapping to empty string
    assert remapper._remap_expression("remove x here", {"x": ""}) == "remove  here"
    assert remapper._remap_expression("x remove x here x", {"x": ""}) == " remove  here "
