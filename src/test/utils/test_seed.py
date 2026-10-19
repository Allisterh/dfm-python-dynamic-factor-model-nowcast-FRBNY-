"""Tests for dfm_python.utils.seed and the ban on global seeding elsewhere."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest
import torch

import dfm_python
from dfm_python.utils.seed import apply_seed


@pytest.fixture(autouse=True)
def restore_global_rng():
    """Leave the torch and numpy global generators as the test found them."""
    torch_state = torch.get_rng_state()
    np_state = np.random.get_state()
    yield
    torch.set_rng_state(torch_state)
    np.random.set_state(np_state)


def _np_states_equal(a, b) -> bool:
    return a[0] == b[0] and np.array_equal(a[1], b[1]) and tuple(a[2:]) == tuple(b[2:])


def test_none_leaves_rng_states_untouched():
    torch.manual_seed(123)
    np.random.seed(123)
    torch_before = torch.get_rng_state()
    np_before = np.random.get_state()

    apply_seed(None)
    apply_seed(None, torch_global=True, numpy_global=True)

    assert torch.equal(torch.get_rng_state(), torch_before)
    assert _np_states_equal(np.random.get_state(), np_before)


def test_none_makes_no_calls(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("seeding function called for seed=None")

    monkeypatch.setattr(torch, "manual_seed", boom)
    monkeypatch.setattr(np.random, "seed", boom)
    apply_seed(None, torch_global=True, numpy_global=True)


def test_matches_torch_manual_seed():
    torch.manual_seed(7)
    expected_state = torch.get_rng_state()
    expected_draw = torch.randn(5)

    torch.manual_seed(999)
    apply_seed(7)
    assert torch.equal(torch.get_rng_state(), expected_state)
    assert torch.equal(torch.randn(5), expected_draw)


def test_default_does_not_touch_numpy():
    np.random.seed(11)
    np_before = np.random.get_state()
    apply_seed(7)
    assert _np_states_equal(np.random.get_state(), np_before)


def test_numpy_global_seeds_numpy():
    np.random.seed(7)
    expected = np.random.rand(5)

    np.random.seed(999)
    apply_seed(7, numpy_global=True)
    np.testing.assert_array_equal(np.random.rand(5), expected)


def test_numpy_only_leaves_torch_untouched():
    torch.manual_seed(5)
    torch_before = torch.get_rng_state()
    apply_seed(7, torch_global=False, numpy_global=True)
    assert torch.equal(torch.get_rng_state(), torch_before)


def test_torch_then_numpy_order(monkeypatch):
    calls = []
    monkeypatch.setattr(torch, "manual_seed", lambda s: calls.append(("torch", s)))
    monkeypatch.setattr(np.random, "seed", lambda s: calls.append(("numpy", s)))
    apply_seed(3, torch_global=True, numpy_global=True)
    assert calls == [("torch", 3), ("numpy", 3)]


def test_seed_passed_without_cast(monkeypatch):
    seen = []
    monkeypatch.setattr(torch, "manual_seed", lambda s: seen.append(s))
    monkeypatch.setattr(np.random, "seed", lambda s: seen.append(s))
    seed = np.int64(4)
    apply_seed(seed, numpy_global=True)
    assert all(s is seed for s in seen) and len(seen) == 2


# --- No global seeding outside utils/seed.py --------------------------------

PACKAGE_DIR = Path(dfm_python.__file__).resolve().parent
SEED_MODULE = PACKAGE_DIR / "utils" / "seed.py"
FORBIDDEN = [
    re.compile(r"torch\.manual_seed\("),
    re.compile(r"torch\.cuda\.manual_seed"),
    re.compile(r"np\.random\.seed\("),
    re.compile(r"random\.seed\("),
]


def _offending_lines(text: str) -> list[tuple[int, str]]:
    hits = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if any(p.search(line) for p in FORBIDDEN):
            hits.append((lineno, line.strip()))
    return hits


@pytest.mark.parametrize(
    "line",
    [
        "torch.manual_seed(0)",
        "torch.cuda.manual_seed_all(0)",
        "np.random.seed(0)",
        "random.seed(0)",
    ],
)
def test_forbidden_patterns_match(line):
    assert _offending_lines(line)


def test_no_global_seeding_outside_helper():
    files = [p for p in PACKAGE_DIR.rglob("*.py") if p.resolve() != SEED_MODULE]
    assert len(files) > 10, f"scanned too few files under {PACKAGE_DIR}"
    assert SEED_MODULE.is_file()

    offenders = []
    for path in files:
        for lineno, line in _offending_lines(path.read_text(encoding="utf-8")):
            offenders.append(f"{path.relative_to(PACKAGE_DIR)}:{lineno}: {line}")
    assert not offenders, (
        "Global seeding outside dfm_python/utils/seed.py; call apply_seed instead:\n"
        + "\n".join(offenders)
    )
