"""Helpers for the golden (parity) suite.

Golden fixtures pin the numbers each engine produces for a given dataset and a
given set of hyperparameters. They are written by ``make_golden.py`` and read by
the ``test_golden_*`` modules.

Comparison rules (design spec, section 11):

* Every run happens under :func:`pinned`: one BLAS/OpenMP thread, one torch
  thread and deterministic torch algorithms.
* When the current platform fingerprint equals the one recorded in a fixture's
  manifest, arrays must be bitwise equal (``assert_array_equal``).
* On any other platform the comparison falls back to ``assert_allclose`` with a
  tolerance chosen by dtype (float64 tighter than float32), and a
  :class:`GoldenToleranceWarning` names the fallback in the test output.
"""

from __future__ import annotations

import contextlib
import json
import platform
import warnings
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import scipy.linalg  # noqa: F401  (load scipy's BLAS so the fingerprint is stable)
import threadpoolctl
import torch

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"

__all__ = [
    "FIXTURES_DIR",
    "GoldenToleranceWarning",
    "assert_golden",
    "fingerprint",
    "load_golden",
    "pinned",
]


class GoldenToleranceWarning(UserWarning):
    """The golden comparison fell back from bitwise equality to a tolerance."""


@contextlib.contextmanager
def pinned() -> Iterator[None]:
    """Pin threads and determinism for a golden run, restoring all three on exit.

    Sets ``threadpoolctl.threadpool_limits(1)`` (BLAS and OpenMP),
    ``torch.set_num_threads(1)`` and ``torch.use_deterministic_algorithms(True)``.
    """
    prev_threads = torch.get_num_threads()
    prev_det = torch.are_deterministic_algorithms_enabled()
    prev_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    with threadpoolctl.threadpool_limits(limits=1):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        try:
            yield
        finally:
            torch.use_deterministic_algorithms(prev_det, warn_only=prev_warn_only)
            torch.set_num_threads(prev_threads)


def _blas_info() -> list[dict[str, Any]]:
    keys = ("internal_api", "prefix", "version", "architecture", "threading_layer")
    seen = set()
    out = []
    for lib in threadpoolctl.threadpool_info():
        if lib.get("user_api") != "blas":
            continue
        entry = {k: lib.get(k) for k in keys}
        key = tuple(entry.values())
        if key in seen:
            continue
        seen.add(key)
        out.append(entry)
    return sorted(out, key=lambda e: tuple(str(v) for v in e.values()))


def fingerprint() -> dict[str, Any]:
    """Platform fingerprint that decides between bitwise and tolerance comparison."""
    return {
        "machine": platform.machine(),
        "processor": platform.processor(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "blas": _blas_info(),
    }


def load_golden(name: str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Return ``(manifest, arrays)`` for fixture ``name``."""
    manifest = json.loads((FIXTURES_DIR / f"{name}.json").read_text())
    with np.load(FIXTURES_DIR / f"{name}.npz", allow_pickle=False) as npz:
        arrays = {k: npz[k] for k in npz.files}
    return manifest, arrays


def _normalize_fingerprint(fp: Mapping[str, Any]) -> Any:
    # JSON turns tuples into lists; compare through the same encoding.
    return json.loads(json.dumps(fp, sort_keys=True))


def assert_golden(actual: Mapping[str, Any], name: str, *, rtol_f64: float = 1e-10,
                  rtol_f32: float = 1e-5, atol: float = 1e-8, exclude: Iterable[str] = ()) -> None:
    """Compare ``actual`` arrays with the golden fixture ``name``.

    Bitwise on the fingerprinted platform; otherwise ``assert_allclose`` with
    ``rtol_f64`` for float64 arrays, ``rtol_f32`` for other floating arrays and
    exact equality for integer and boolean arrays, plus a
    :class:`GoldenToleranceWarning`.

    ``exclude`` names fixture outputs that this comparison leaves out; each
    must exist in the fixture and be absent from ``actual``. Every other
    fixture output must be in ``actual``.
    """
    manifest, expected = load_golden(name)
    actual = {k: np.asarray(v) for k, v in actual.items()}
    exclude = set(exclude)
    unknown = sorted(exclude - set(expected))
    assert not unknown, f"golden '{name}': excluded keys {unknown} are not fixture outputs"
    overlap = sorted(exclude & set(actual))
    assert not overlap, f"golden '{name}': keys {overlap} are both excluded and given"
    expected = {k: v for k, v in expected.items() if k not in exclude}

    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    assert not missing and not extra, (
        f"golden '{name}': output keys differ (missing={missing}, unexpected={extra})")

    bitwise = _normalize_fingerprint(fingerprint()) == _normalize_fingerprint(manifest["fingerprint"])
    if not bitwise:
        warnings.warn(
            f"golden '{name}': platform fingerprint differs from the capture platform; "
            f"falling back to assert_allclose (rtol_f64={rtol_f64}, rtol_f32={rtol_f32}, "
            f"atol={atol}) instead of bitwise assert_array_equal",
            GoldenToleranceWarning, stacklevel=2)

    for key in sorted(expected):
        exp, act = expected[key], actual[key]
        assert act.dtype == exp.dtype, (
            f"golden '{name}' [{key}]: dtype {act.dtype} != expected {exp.dtype}")
        assert act.shape == exp.shape, (
            f"golden '{name}' [{key}]: shape {act.shape} != expected {exp.shape}")
        msg = f"golden '{name}' [{key}]"
        if bitwise or not np.issubdtype(exp.dtype, np.floating):
            np.testing.assert_array_equal(act, exp, err_msg=msg, strict=True)
        else:
            rtol = rtol_f64 if exp.dtype == np.float64 else rtol_f32
            np.testing.assert_allclose(act, exp, rtol=rtol, atol=atol, err_msg=msg)
