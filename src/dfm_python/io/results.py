"""Split a dataclass result into ``result.npz`` arrays and ``result.json`` scalars, and back.

:func:`result_to_files` sorts every field of a result dataclass:

- numpy arrays and torch tensors go to the arrays (tensors as numpy arrays,
  restored as CPU tensors);
- numpy scalars and non-finite Python floats go to the arrays as 0-d arrays,
  so their dtype and bit pattern survive;
- ``None``, bools, ints, finite floats, strings, and lists, tuples and
  string-keyed dicts of these go to JSON (tuples are restored as tuples).

Fields named in ``exclude`` are skipped; the caller rebuilds them on loading
(for example a ``model_state_dict`` that ``weights.pt`` already holds) and
passes them to :func:`result_from_files` as keyword arguments.
"""

from __future__ import annotations

import dataclasses
import math
import sys
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from ..utils.errors import ConfigurationError
from .codecs import class_path, to_jsonable

__all__ = ["result_to_files", "result_from_files"]

_KIND_NDARRAY = "ndarray"
_KIND_TENSOR = "tensor"
_KIND_NUMPY_SCALAR = "numpy_scalar"
_KIND_FLOAT = "float"


def _is_tensor(value: Any) -> bool:
    torch = sys.modules.get("torch")  # a tensor can only exist if torch is imported
    return torch is not None and isinstance(value, torch.Tensor)


def result_to_files(result: Any, *, exclude: Iterable[str] = ()) -> tuple[dict, dict]:
    """Return ``(arrays, meta)``: the arrays for ``result.npz`` and the JSON content of ``result.json``.

    ``meta`` holds ``class`` (the result's ``module.QualName``), ``fields``
    (the JSON values), ``arrays`` (field name to how it is stored),
    ``tuples`` (JSON fields to restore as tuples) and ``excluded``.

    Raises
    ------
    TypeError
        If ``result`` is not a dataclass instance, or a field holds a value
        that neither file can store without pickle (pass its name in
        ``exclude`` and rebuild it on loading).
    """
    if not dataclasses.is_dataclass(result) or isinstance(result, type):
        raise TypeError(f"result_ must be a dataclass instance to be saved, got {type(result).__name__}.")
    exclude = set(exclude)
    cls_name = type(result).__name__
    arrays: dict = {}
    fields: dict = {}
    kinds: dict = {}
    tuples: list = []
    for f in dataclasses.fields(result):
        name = f.name
        if name in exclude:
            continue
        value = getattr(result, name)
        if _is_tensor(value):
            arrays[name] = value.detach().cpu().numpy().copy()
            kinds[name] = _KIND_TENSOR
        elif isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                raise TypeError(
                    f"{cls_name}.{name} is an object array, which cannot be saved without pickle; "
                    "exclude it and rebuild it on loading."
                )
            arrays[name] = value
            kinds[name] = _KIND_NDARRAY
        elif isinstance(value, np.generic):
            arrays[name] = np.asarray(value)
            kinds[name] = _KIND_NUMPY_SCALAR
        elif isinstance(value, float) and not math.isfinite(value):
            arrays[name] = np.asarray(value, dtype=np.float64)
            kinds[name] = _KIND_FLOAT
        else:
            try:
                fields[name] = to_jsonable(value, where=f"{cls_name}.{name}")
            except TypeError as err:
                raise TypeError(f"{err} Exclude the field and rebuild it on loading.") from None
            if isinstance(value, tuple):
                tuples.append(name)
    meta = {
        "class": class_path(type(result)),
        "fields": fields,
        "arrays": kinds,
        "tuples": tuples,
        "excluded": sorted(exclude & {f.name for f in dataclasses.fields(result)}),
    }
    return arrays, meta


def result_from_files(result_cls: type, arrays: Mapping[str, np.ndarray], meta: Mapping, **extra: Any) -> Any:
    """Rebuild a result written by :func:`result_to_files`; ``extra`` supplies the excluded fields.

    Raises
    ------
    ConfigurationError
        If ``meta`` was written for another result class, names a field the
        class does not have, or lists an array that ``arrays`` lacks.
    """
    saved = str(meta.get("class", ""))
    if saved != class_path(result_cls) and not saved.endswith("." + result_cls.__qualname__):
        raise ConfigurationError(f"result.json holds a {saved!r}, expected {class_path(result_cls)!r}.")
    values: dict = dict(meta.get("fields", {}))
    for name in meta.get("tuples", []):
        values[name] = tuple(values[name])
    for name, kind in meta.get("arrays", {}).items():
        if name not in arrays:
            raise ConfigurationError(f"result.npz has no array {name!r}, which result.json lists.")
        array = arrays[name]
        if kind == _KIND_NDARRAY:
            values[name] = array
        elif kind == _KIND_TENSOR:
            import torch

            values[name] = torch.from_numpy(np.array(array, copy=True))
        elif kind == _KIND_NUMPY_SCALAR:
            values[name] = array[()]
        elif kind == _KIND_FLOAT:
            values[name] = float(array[()])
        else:
            raise ConfigurationError(f"result.json: unknown storage kind {kind!r} for {name!r}.")
    values.update(extra)
    all_fields = {f.name: f for f in dataclasses.fields(result_cls)}
    unknown = sorted(set(values) - set(all_fields))
    if unknown:
        raise ConfigurationError(f"{result_cls.__name__} has no field(s) {unknown} named in result.json.")
    result = result_cls(**{k: v for k, v in values.items() if all_fields[k].init})
    for name, value in values.items():
        if not all_fields[name].init:
            object.__setattr__(result, name, value)
    return result
