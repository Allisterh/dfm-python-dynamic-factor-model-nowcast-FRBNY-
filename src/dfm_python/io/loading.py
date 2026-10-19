"""``dfm_python.load``: load any saved run, dispatching on its ``model_type``."""

from __future__ import annotations

from typing import Any

from ..utils.errors import ConfigurationError
from .layout import PathLike, read_config

__all__ = ["load"]


def load(path: PathLike, *, device: Any = None, allow_pickle: bool = False) -> Any:
    """Load a run saved with ``save(path)``, whatever its model.

    Reads ``model_type`` from ``config.json``, finds that class in
    :func:`dfm_python.registry.all_estimators` and returns
    ``Model.load(path, device=device, allow_pickle=allow_pickle)``.

    Parameters
    ----------
    path : str or os.PathLike
        The saved-run directory.
    device : str or torch.device, optional
        Device for torch models (see ``Model.load``); ignored by numpy models.
    allow_pickle : bool, default False
        Allow reading ``scaler.pkl`` and importing estimator classes outside
        ``sklearn.preprocessing`` named in ``config.json``. Both can run
        arbitrary code; enable only for directories you trust.

    Raises
    ------
    ConfigurationError
        If ``path`` is a single-file checkpoint from an earlier release, is
        not a saved run, or names a ``model_type`` that is not a registered
        estimator.
    """
    from ..registry import all_estimators

    config = read_config(path)
    model_type = config["model_type"]
    registered = dict(all_estimators())
    cls = registered.get(model_type)
    if cls is None:
        names = ", ".join(sorted(registered)) or "none yet"
        raise ConfigurationError(
            f"{str(path)!r} holds a {model_type!r} run, which is not a registered estimator (registered: {names}). "
            f"Load it with {model_type}.load(path) from the class that saved it."
        )
    return cls.load(path, device=device, allow_pickle=allow_pickle)
