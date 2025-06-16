import os
import sys
import types
import pytest

# Provide a minimal torch stub to satisfy imports if torch is unavailable
if 'torch' not in sys.modules:
    torch_stub = types.SimpleNamespace(
        nn=types.SimpleNamespace(Module=object),
        optim=types.SimpleNamespace(Adam=object),
        randn_like=lambda x: x,
        FloatTensor=lambda *a, **k: None,
    )
    sys.modules['torch'] = torch_stub
    sys.modules['torch.nn'] = torch_stub.nn
    sys.modules['torch.optim'] = torch_stub.optim

if 'scipy' not in sys.modules:
    scipy_stats_stub = types.SimpleNamespace(entropy=lambda *a, **k: 0)
    scipy_stub = types.SimpleNamespace(stats=scipy_stats_stub)
    sys.modules['scipy'] = scipy_stub
    sys.modules['scipy.stats'] = scipy_stats_stub

if 'sklearn' not in sys.modules:
    sklearn_decomp_stub = types.SimpleNamespace(FastICA=object)
    sklearn_preproc_stub = types.SimpleNamespace(StandardScaler=object)
    sklearn_stub = types.SimpleNamespace(
        decomposition=sklearn_decomp_stub,
        preprocessing=sklearn_preproc_stub,
    )
    sys.modules['sklearn'] = sklearn_stub
    sys.modules['sklearn.decomposition'] = sklearn_decomp_stub
    sys.modules['sklearn.preprocessing'] = sklearn_preproc_stub

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from progressive_grammar_system import ProgressiveGrammar, Variable, Expression


def test_invalid_arity_returns_none():
    grammar = ProgressiveGrammar()
    var = Variable(name="x", index=0)
    result = grammar.create_expression('+', [var])
    assert result is None


def test_valid_expression_is_created():
    grammar = ProgressiveGrammar()
    var = Variable(name="x", index=0)
    expr = grammar.create_expression('+', [var, var])
    assert isinstance(expr, Expression)
