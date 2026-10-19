"""The save directory layout: file names, ``config.json`` reading and the index spec.

A saved run is a directory::

    config.json      format_version, model_type, library_version, params, tags, runtime
    data_spec.json   feature, exog and series-exog names; the training index
    state.npz        non-parameter arrays (namespaced keys; scaler/<attr> for the scaler)
    state.json       non-array state
    result.npz       arrays of result_
    result.json      scalars of result_
    weights.pt       torch models only: network_.state_dict()
    scaler.json      the scaler, for the five scalers written without pickle (and None)
    scaler.pkl       any other scaler; loading it needs allow_pickle=True

Only directories in this layout are read. A single ``.pkl`` or ``.pt``
checkpoint written by an earlier release raises ``ConfigurationError``.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Union

import numpy as np
import pandas as pd

from ..utils.errors import ConfigurationError
from .codecs import read_json, to_jsonable

__all__ = [
    "FORMAT_VERSION",
    "CONFIG_FILE",
    "DATA_SPEC_FILE",
    "STATE_NPZ",
    "STATE_JSON",
    "RESULT_NPZ",
    "RESULT_JSON",
    "WEIGHTS_FILE",
    "REQUIRED_FILES",
    "read_config",
    "index_to_spec",
    "index_from_spec",
]

PathLike = Union[str, "os.PathLike[str]"]

#: File-format counter of the save directory, independent of library releases.
FORMAT_VERSION = 1

CONFIG_FILE = "config.json"
DATA_SPEC_FILE = "data_spec.json"
STATE_NPZ = "state.npz"
STATE_JSON = "state.json"
RESULT_NPZ = "result.npz"
RESULT_JSON = "result.json"
WEIGHTS_FILE = "weights.pt"

#: Files every saved run has (``weights.pt`` and the scaler files depend on the model).
REQUIRED_FILES = (CONFIG_FILE, DATA_SPEC_FILE, STATE_NPZ, STATE_JSON, RESULT_NPZ, RESULT_JSON)

_CONFIG_KEYS = ("format_version", "model_type", "params")


def _old_checkpoint_error(path: Path, reason: str) -> ConfigurationError:
    return ConfigurationError(
        f"{str(path)!r} {reason}. dfm-python saves a run as a directory with a config.json; single-file "
        "checkpoints (.pkl, .pt) written by earlier releases cannot be read by this release. Load them with the "
        "release of dfm-python that wrote them."
    )


def read_config(path: PathLike, *, allow_import: bool = False) -> dict:
    """Read and check ``config.json`` of a saved run.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ConfigurationError
        If ``path`` is a file (a single-file checkpoint from an earlier
        release), a directory without ``config.json``, or ``config.json`` lacks
        a required key or has an unsupported ``format_version``.
    """
    path = Path(path)
    if path.is_file():
        raise _old_checkpoint_error(path, "is a file, not a saved-run directory")
    if not path.exists():
        raise FileNotFoundError(f"No saved run at {str(path)!r}.")
    if not (path / CONFIG_FILE).is_file():
        raise _old_checkpoint_error(path, f"has no {CONFIG_FILE}, so it is not a saved run")
    config = read_json(path / CONFIG_FILE, allow_import=allow_import)
    if not isinstance(config, Mapping):
        raise ConfigurationError(f"{str(path / CONFIG_FILE)!r} does not hold a JSON object.")
    missing = [key for key in _CONFIG_KEYS if key not in config]
    if missing:
        raise ConfigurationError(f"{str(path / CONFIG_FILE)!r} lacks {', '.join(map(repr, missing))}.")
    if config["format_version"] != FORMAT_VERSION:
        raise ConfigurationError(
            f"{str(path)!r} has format_version {config['format_version']!r}; this release reads format_version "
            f"{FORMAT_VERSION}. Load it with the release of dfm-python that wrote it."
        )
    if not isinstance(config["params"], Mapping):
        raise ConfigurationError(f"{str(path / CONFIG_FILE)!r}: 'params' is not a JSON object.")
    return dict(config)


# ---------------------------------------------------------------------------
# Index spec (data_spec.json)
# ---------------------------------------------------------------------------


def index_to_spec(index: pd.Index) -> dict:
    """Describe a training index for ``data_spec.json`` so that it can be rebuilt exactly.

    Every spec has ``kind`` (``"range"``, ``"datetime"`` or ``"period"``),
    ``name``, ``dtype``, ``freq``, ``first``, ``last`` and ``length``. A
    ``RangeIndex`` adds ``start``, ``stop`` and ``step``; a datetime or period
    index adds ``values`` (int64 nanoseconds-since-epoch in the index's unit,
    UTC for time-zone-aware indexes, or period ordinals), plus ``unit`` and
    ``tz`` for datetimes.
    """
    name = to_jsonable(index.name, where="index name")
    spec: dict = {"name": name, "dtype": str(index.dtype), "length": len(index)}
    if isinstance(index, pd.RangeIndex):
        spec.update(kind="range", freq=None, start=index.start, stop=index.stop, step=index.step)
        spec.update(first=int(index[0]) if len(index) else None, last=int(index[-1]) if len(index) else None)
        return spec
    if isinstance(index, pd.DatetimeIndex):
        spec.update(
            kind="datetime",
            freq=index.freqstr,
            unit=index.unit,
            tz=None if index.tz is None else str(index.tz),
            values=index.asi8.tolist(),
        )
    elif isinstance(index, pd.PeriodIndex):
        spec.update(kind="period", freq=index.freqstr, values=index.asi8.tolist())
    else:
        raise TypeError(f"Cannot describe a {type(index).__name__}; the training index is a Range, Datetime or Period index.")
    spec.update(first=str(index[0]) if len(index) else None, last=str(index[-1]) if len(index) else None)
    return spec


def index_from_spec(spec: Mapping[str, Any]) -> pd.Index:
    """Rebuild the index described by :func:`index_to_spec`."""
    kind = spec.get("kind")
    name = spec.get("name")
    if kind == "range":
        return pd.RangeIndex(spec["start"], spec["stop"], spec["step"], name=name)
    if kind == "datetime":
        values = np.asarray(spec["values"], dtype=np.int64).view(f"datetime64[{spec['unit']}]")
        index = pd.DatetimeIndex(values, name=name)
        if spec.get("tz") is not None:
            index = index.tz_localize("UTC").tz_convert(spec["tz"])
        if spec.get("freq") is not None:
            index = pd.DatetimeIndex(index, freq=spec["freq"])
        return index
    if kind == "period":
        return pd.PeriodIndex.from_ordinals(np.asarray(spec["values"], dtype=np.int64), freq=spec["freq"], name=name)
    raise ConfigurationError(f"data_spec.json: unknown index kind {kind!r}.")
