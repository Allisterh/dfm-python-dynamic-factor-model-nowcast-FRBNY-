"""Result dataclasses split into result.npz arrays and result.json scalars, and back."""

import dataclasses
import json
from typing import Any, Optional

import numpy as np
import pytest
import torch

from dfm_python.config import AFMResult
from dfm_python.io.codecs import read_json, read_npz, write_json, write_npz
from dfm_python.io.results import result_from_files, result_to_files
from dfm_python.utils.errors import ConfigurationError


@dataclasses.dataclass
class _Result:
    weights: np.ndarray
    loss: torch.Tensor
    sharpe: float
    loglik: float
    count: np.int64
    shape: tuple
    names: list
    info: dict
    label: Optional[str] = None
    converged: bool = False
    scaler: Any = None
    derived: float = dataclasses.field(init=False, default=0.0)


def _make():
    result = _Result(
        weights=np.arange(6, dtype=np.float32).reshape(2, 3) / 3,
        loss=torch.tensor([0.25, 0.5], dtype=torch.float64),
        sharpe=1.2345678901234567,
        loglik=float("-inf"),
        count=np.int64(7),
        shape=(2, 3),
        names=["a", "b"],
        info={"k": [1, 2.5, None]},
        converged=True,
    )
    result.derived = 0.5
    return result


def _write(tmp_path, arrays, meta):
    write_npz(tmp_path / "result.npz", arrays)
    write_json(tmp_path / "result.json", meta)
    return read_npz(tmp_path / "result.npz"), read_json(tmp_path / "result.json")


def test_round_trip_keeps_types_and_bits(tmp_path):
    result = _make()
    arrays, meta = result_to_files(result)
    assert sorted(arrays) == ["count", "loglik", "loss", "weights"]
    assert meta["arrays"] == {"weights": "ndarray", "loss": "tensor", "count": "numpy_scalar", "loglik": "float"}
    assert meta["class"].endswith("test_results._Result")
    assert meta["fields"]["sharpe"] == result.sharpe and meta["tuples"] == ["shape"]
    json.dumps(meta)
    back = result_from_files(_Result, *_write(tmp_path, arrays, meta))
    assert back.weights.dtype == np.float32 and back.weights.tobytes() == result.weights.tobytes()
    assert isinstance(back.loss, torch.Tensor) and torch.equal(back.loss, result.loss)
    assert back.sharpe.hex() == result.sharpe.hex() and back.loglik == float("-inf")
    assert type(back.count) is np.int64 and back.count == 7
    assert back.shape == (2, 3) and back.names == ["a", "b"] and back.info == {"k": [1, 2.5, None]}
    assert back.label is None and back.converged is True and back.derived == 0.5


def test_nan_bit_pattern_survives(tmp_path):
    result = _make()
    result.sharpe = float(np.float64(-np.nan))
    arrays, meta = result_to_files(result)
    back = result_from_files(_Result, *_write(tmp_path, arrays, meta))
    assert np.asarray(back.sharpe).tobytes() == np.asarray(result.sharpe).tobytes()


def test_excluded_fields_come_from_extra(tmp_path):
    result = _make()
    result.scaler = object()
    with pytest.raises(TypeError, match="_Result.scaler.*Exclude"):
        result_to_files(result)
    arrays, meta = result_to_files(result, exclude=("scaler", "not_a_field"))
    assert meta["excluded"] == ["scaler"]
    back = result_from_files(_Result, *_write(tmp_path, arrays, meta), scaler="rebuilt")
    assert back.scaler == "rebuilt"


def test_refuses_object_arrays_and_non_dataclasses():
    result = _make()
    result.weights = np.array([None], dtype=object)
    with pytest.raises(TypeError, match="object array"):
        result_to_files(result)
    with pytest.raises(TypeError, match="dataclass instance"):
        result_to_files({"a": 1})
    with pytest.raises(TypeError, match="dataclass instance"):
        result_to_files(_Result)


def test_class_mismatch_and_missing_arrays(tmp_path):
    arrays, meta = result_to_files(_make())
    with pytest.raises(ConfigurationError, match="expected"):
        result_from_files(AFMResult, arrays, meta)
    with pytest.raises(ConfigurationError, match="result.npz has no array 'weights'"):
        result_from_files(_Result, {k: v for k, v in arrays.items() if k != "weights"}, meta)
    with pytest.raises(ConfigurationError, match="no field"):
        result_from_files(_Result, arrays, {**meta, "fields": {**meta["fields"], "bogus": 1}})


def test_library_result_class(tmp_path):
    result = AFMResult(
        x_sm=np.ones((4, 2)),
        Z=np.zeros((4, 2)),
        residual_portfolios=np.full((1, 4, 2), 0.5),
        book_returns=np.linspace(0, 1, 3)[None],
        sharpe=0.75,
        num_iter=3,
        converged=True,
        factor_model="pca",
        trading="ou",
        num_factors=2,
    )
    arrays, meta = result_to_files(result, exclude=("target_scaler",))
    back = result_from_files(AFMResult, *_write(tmp_path, arrays, meta))
    for f in dataclasses.fields(AFMResult):
        a, b = getattr(result, f.name), getattr(back, f.name)
        if isinstance(a, np.ndarray):
            assert a.tobytes() == b.tobytes() and a.shape == b.shape
        else:
            assert a == b, f.name
