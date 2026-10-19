"""Pickle-free persistence of fitted scikit-learn scalers.

Five scalers are written without pickle: ``StandardScaler``,
``RobustScaler``, ``MinMaxScaler``, ``MaxAbsScaler`` and
``QuantileTransformer`` (exact classes, not subclasses). Their constructor
parameters and non-array fitted attributes go to ``scaler.json``; their
fitted array attributes (and numpy scalars such as ``n_samples_seen_``) go to
``state.npz`` under the ``scaler/`` prefix. ``None`` is written as
``{"class": null}``.

Any other scaler is pickled to ``scaler.pkl``. Reading it needs
``allow_pickle=True``, because unpickling can run arbitrary code.
"""

from __future__ import annotations

import os
import pickle
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Union

import numpy as np

from ..utils.errors import ConfigurationError
from .codecs import _tuple_params, read_json, to_jsonable, write_json

__all__ = [
    "SCALER_PREFIX",
    "SCALER_JSON",
    "SCALER_PKL",
    "KNOWN_SCALERS",
    "is_known_scaler",
    "encode_scaler",
    "decode_scaler",
    "write_scaler",
    "read_scaler",
    "has_scaler",
]

PathLike = Union[str, "os.PathLike[str]"]

SCALER_PREFIX = "scaler/"
SCALER_JSON = "scaler.json"
SCALER_PKL = "scaler.pkl"


def _known_classes() -> dict:
    from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, QuantileTransformer, RobustScaler, StandardScaler

    classes = (StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler, QuantileTransformer)
    return {f"sklearn.preprocessing.{cls.__name__}": cls for cls in classes}


#: Public names of the scalers written without pickle.
KNOWN_SCALERS: tuple[str, ...] = (
    "sklearn.preprocessing.StandardScaler",
    "sklearn.preprocessing.RobustScaler",
    "sklearn.preprocessing.MinMaxScaler",
    "sklearn.preprocessing.MaxAbsScaler",
    "sklearn.preprocessing.QuantileTransformer",
)


def _known_name(scaler: Any) -> str | None:
    for name, cls in _known_classes().items():
        if type(scaler) is cls:
            return name
    return None


def is_known_scaler(scaler: Any) -> bool:
    """True for ``None`` and the five scalers that are written without pickle."""
    return scaler is None or _known_name(scaler) is not None


def encode_scaler(scaler: Any) -> tuple[dict, dict]:
    """Split a known scaler into ``(json_dict, arrays)``.

    ``json_dict`` holds ``class`` (the public class path), ``params``
    (``get_params(deep=False)``) and ``attributes``: for every fitted
    attribute (public, ending in ``_``) either its JSON value or a marker
    saying where it is stored. ``arrays`` maps ``"scaler/<attr>"`` to the numeric
    arrays and numpy scalars. ``None`` gives ``({"class": None}, {})``.

    Raises
    ------
    TypeError
        If ``scaler`` is not ``None`` or one of the five known scalers (those
        go to ``scaler.pkl``; see :func:`write_scaler`).
    """
    if scaler is None:
        return {"class": None}, {}
    name = _known_name(scaler)
    if name is None:
        raise TypeError(
            f"{type(scaler).__name__} is not one of the scalers written without pickle ({', '.join(KNOWN_SCALERS)})."
        )
    attributes: dict = {}
    arrays: dict = {}
    for attr, value in vars(scaler).items():
        if not attr.endswith("_") or attr.startswith("_"):
            continue
        where = f"{type(scaler).__name__}.{attr}"
        if isinstance(value, np.ndarray) and value.dtype.hasobject:
            attributes[attr] = {"kind": "object_array", "value": to_jsonable(value, where=where)}
        elif isinstance(value, (np.ndarray, np.generic)):
            attributes[attr] = {"kind": "array"}
            arrays[SCALER_PREFIX + attr] = np.asarray(value)
        elif isinstance(value, tuple):
            attributes[attr] = {"kind": "tuple", "value": to_jsonable(value, where=where)}
        else:
            attributes[attr] = {"kind": "json", "value": to_jsonable(value, where=where)}
    json_dict = {
        "class": name,
        "params": to_jsonable(scaler.get_params(deep=False), where=f"{type(scaler).__name__}.params"),
        "attributes": attributes,
    }
    return json_dict, arrays


def decode_scaler(json_dict: Mapping, arrays: Mapping[str, np.ndarray]) -> Any:
    """Rebuild a scaler from :func:`encode_scaler` output (``arrays`` may hold other keys too)."""
    name = json_dict.get("class")
    if name is None:
        return None
    classes = _known_classes()
    if name not in classes:
        raise ConfigurationError(
            f"scaler.json names {name!r}, which is not one of the scalers written without pickle "
            f"({', '.join(KNOWN_SCALERS)})."
        )
    cls = classes[name]
    scaler = cls(**_tuple_params(cls, dict(json_dict.get("params", {}))))
    for attr, entry in json_dict.get("attributes", {}).items():
        kind = entry["kind"]
        if kind == "array":
            key = SCALER_PREFIX + attr
            if key not in arrays:
                raise ConfigurationError(f"state.npz has no {key!r}, which scaler.json lists.")
            array = arrays[key]
            value = array[()] if array.ndim == 0 else array
        elif kind == "object_array":
            value = np.asarray(entry["value"], dtype=object)
        elif kind == "tuple":
            value = tuple(entry["value"])
        elif kind == "json":
            value = entry["value"]
        else:
            raise ConfigurationError(f"scaler.json: unknown attribute kind {kind!r} for {attr!r}.")
        setattr(scaler, attr, value)
    return scaler


def write_scaler(directory: PathLike, scaler: Any) -> dict:
    """Write ``scaler.json`` (known scalers and None) or ``scaler.pkl`` (any other scaler).

    A known scaler whose parameters JSON cannot hold (for example a
    ``QuantileTransformer`` with a ``RandomState``) is pickled too.

    Returns the arrays to add to ``state.npz`` (keys start with ``"scaler/"``;
    empty for a pickled scaler).
    """
    directory = Path(directory)
    if is_known_scaler(scaler):
        try:
            json_dict, arrays = encode_scaler(scaler)
        except TypeError:  # a parameter JSON cannot hold, e.g. a RandomState: fall back to pickle
            json_dict = None
        if json_dict is not None:
            write_json(directory / SCALER_JSON, json_dict)
            return arrays
    with open(directory / SCALER_PKL, "wb") as fh:
        pickle.dump(scaler, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return {}


def has_scaler(directory: PathLike) -> bool:
    """True if the directory holds ``scaler.json`` or ``scaler.pkl``."""
    directory = Path(directory)
    return (directory / SCALER_JSON).is_file() or (directory / SCALER_PKL).is_file()


def read_scaler(directory: PathLike, arrays: Mapping[str, np.ndarray], *, allow_pickle: bool = False) -> Any:
    """Read the scaler written by :func:`write_scaler`.

    ``arrays`` are the contents of ``state.npz``. ``scaler.pkl`` is read only
    with ``allow_pickle=True``.

    Raises
    ------
    ConfigurationError
        If the scaler is pickled and ``allow_pickle`` is False, or the files
        are inconsistent.
    FileNotFoundError
        If the directory holds no scaler file.
    """
    directory = Path(directory)
    if (directory / SCALER_JSON).is_file():
        return decode_scaler(read_json(directory / SCALER_JSON), arrays)
    pkl = directory / SCALER_PKL
    if pkl.is_file():
        if not allow_pickle:
            raise ConfigurationError(
                f"{str(directory)!r} stores its fitted scaler as {SCALER_PKL}, because its class is not one of "
                f"{', '.join(KNOWN_SCALERS)}. Unpickling can run arbitrary code, so it is off by default: pass "
                "allow_pickle=True to load(...) only if you trust the directory."
            )
        with open(pkl, "rb") as fh:
            return pickle.load(fh)  # noqa: S301 - gated by allow_pickle
    raise FileNotFoundError(f"{str(directory)!r} holds neither {SCALER_JSON} nor {SCALER_PKL}.")
