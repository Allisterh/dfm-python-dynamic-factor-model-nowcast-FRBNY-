"""Registry of the public estimators built on :class:`dfm_python.base.BaseFactorModel`.

The list is explicit (no import-time scanning). A model is added here when it
is migrated to the estimator API; the contract tests in ``src/test/contract``
run :func:`dfm_python.testing.check_estimator` on every entry.
"""

from __future__ import annotations

from importlib import import_module

__all__ = ["all_estimators"]

# (public name, module, attribute), imported lazily so that importing the
# registry does not import every model.
_ESTIMATORS: tuple[tuple[str, str, str], ...] = (
    ("AFM", "dfm_python.models.afm.estimator", "AFM"),
    ("iVDFM", "dfm_python.models.ivdfm.estimator", "iVDFM"),
)


def all_estimators() -> list[tuple[str, type]]:
    """Return ``(name, class)`` for every migrated estimator, sorted by name."""
    out = []
    for name, module, attr in _ESTIMATORS:
        out.append((name, getattr(import_module(module), attr)))
    return sorted(out, key=lambda item: item[0])
