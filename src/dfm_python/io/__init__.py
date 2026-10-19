"""Pickle-free save directories (spec section 7).

- :mod:`.directory`: :func:`atomic_save_dir` writes into a temporary sibling,
  verifies and renames into place.
- :mod:`.codecs`: JSON (with scikit-learn estimators as ``{"class", "params"}``),
  ``.npz`` without pickle, and torch ``weights.pt``.
- :mod:`.scaler`: the five known scalers without pickle; any other scaler
  through a gated ``scaler.pkl``.
- :mod:`.results`: a result dataclass split into ``result.npz`` and ``result.json``.
- :mod:`.layout`: file names, ``config.json`` checks and the index spec.
- :func:`load`: ``dfm_python.load(path)``, dispatching on ``model_type``.

The estimator side (``save`` / ``load`` and the state hooks) lives on
:class:`dfm_python.base.BaseFactorModel`.
"""

from .codecs import read_json, read_npz, read_weights, write_json, write_npz, write_weights
from .directory import atomic_save_dir
from .layout import FORMAT_VERSION, read_config
from .loading import load
from .results import result_from_files, result_to_files
from .scaler import decode_scaler, encode_scaler, read_scaler, write_scaler

__all__ = [
    "atomic_save_dir",
    "write_json",
    "read_json",
    "write_npz",
    "read_npz",
    "write_weights",
    "read_weights",
    "encode_scaler",
    "decode_scaler",
    "write_scaler",
    "read_scaler",
    "result_to_files",
    "result_from_files",
    "read_config",
    "FORMAT_VERSION",
    "load",
]
