"""Importing dfm_python attaches nothing but a NullHandler (checked in a fresh interpreter)."""

import json
import os
import subprocess
import sys
import textwrap

import pytest

_PROBE = textwrap.dedent(
    """
    import json, logging, warnings
    root_before = list(logging.getLogger().handlers)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        import dfm_python
    pkg = logging.getLogger("dfm_python")
    print(json.dumps({
        "handlers": [type(h).__name__ for h in pkg.handlers],
        "marked": [bool(getattr(h, "_dfm_python_verbosity_handler", False)) for h in pkg.handlers],
        "streams": [getattr(getattr(h, "stream", None), "name", None) for h in pkg.handlers],
        "level": pkg.level,
        "propagate": pkg.propagate,
        "root_new": [type(h).__name__ for h in logging.getLogger().handlers if h not in root_before],
        "root_level": logging.getLogger().level,
        "verbosity": dfm_python.get_verbosity(),
        "warnings": [str(w.message) for w in caught if "DFM_PYTHON_VERBOSITY" in str(w.message)],
    }))
    """
)


def _import_state(env_value=None):
    env = {k: v for k, v in os.environ.items() if k != "DFM_PYTHON_VERBOSITY"}
    if env_value is not None:
        env["DFM_PYTHON_VERBOSITY"] = env_value
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE], env=env, capture_output=True, text=True, timeout=300
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1]), proc


def test_import_adds_only_a_null_handler():
    state, proc = _import_state()
    assert state["handlers"] == ["NullHandler"]
    assert state["propagate"] is True
    assert state["level"] == 0  # NOTSET: the application decides
    assert state["root_new"] == []
    assert state["warnings"] == []
    assert proc.stdout.strip().count("\n") == 0  # nothing printed besides the probe


@pytest.mark.parametrize("value, expected", [("info", 20), ("40", 40)])
def test_env_var_sets_verbosity(value, expected):
    state, _ = _import_state(value)
    assert state["handlers"] == ["NullHandler", "StreamHandler"]
    assert state["marked"] == [False, True]
    assert state["streams"][1] == "<stdout>"
    assert state["level"] == expected
    assert state["verbosity"] == expected
    assert state["root_new"] == []


def test_invalid_env_var_warns_and_stays_silent():
    state, _ = _import_state("loud")
    assert state["handlers"] == ["NullHandler"]
    assert state["level"] == 0
    assert len(state["warnings"]) == 1
    assert "loud" in state["warnings"][0]


def test_empty_env_var_is_ignored():
    state, _ = _import_state("")
    assert state["handlers"] == ["NullHandler"]
    assert state["warnings"] == []
