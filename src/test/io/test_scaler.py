"""Scaler codec: the five known scalers without pickle; any other through a gated scaler.pkl."""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, QuantileTransformer, RobustScaler, StandardScaler

from dfm_python.io.codecs import read_npz, write_npz
from dfm_python.io.scaler import (
    KNOWN_SCALERS,
    SCALER_JSON,
    SCALER_PKL,
    decode_scaler,
    encode_scaler,
    has_scaler,
    is_known_scaler,
    read_scaler,
    write_scaler,
)
from dfm_python.utils.errors import ConfigurationError

from ._estimators import DoublingScaler

_SCALERS = [
    StandardScaler(),
    StandardScaler(with_mean=False),
    RobustScaler(quantile_range=(10.0, 90.0)),
    MinMaxScaler(feature_range=(-1.0, 2.0)),
    MaxAbsScaler(),
    QuantileTransformer(n_quantiles=15, output_distribution="normal"),
]


def _data(nan: bool = False):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((40, 3)) * [1.0, 5.0, 0.1] + [0.0, 2.0, -1.0]
    if nan:
        x[[2, 7], 1] = np.nan
    return x


def _round_trip(scaler, tmp_path):
    arrays = write_scaler(tmp_path, scaler)
    write_npz(tmp_path / "state.npz", {"other": np.zeros(1), **arrays})
    return read_scaler(tmp_path, read_npz(tmp_path / "state.npz"))


def _assert_same_scaler(a, b):
    assert type(a) is type(b)
    assert a.get_params() == b.get_params()
    assert sorted(vars(a)) == sorted(vars(b))
    for key, value in vars(a).items():
        other = vars(b)[key]
        assert type(other) is type(value), key
        if isinstance(value, np.ndarray):
            assert value.dtype == other.dtype and value.shape == other.shape, key
            if value.dtype.hasobject:
                assert value.tolist() == other.tolist(), key
            else:
                assert value.tobytes() == other.tobytes(), key
        elif isinstance(value, np.generic):
            assert value.dtype == other.dtype and value == other, key
        else:
            assert value == other, key


@pytest.mark.parametrize("scaler", _SCALERS, ids=lambda s: repr(s))
@pytest.mark.parametrize("frame", [False, True], ids=["array", "dataframe"])
def test_known_scalers_round_trip_without_pickle(tmp_path, scaler, frame):
    x = _data()
    fit_on = pd.DataFrame(x, columns=["a", "b", "c"]) if frame else x
    scaler.fit(fit_on)
    loaded = _round_trip(scaler, tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == [SCALER_JSON, "state.npz"]
    _assert_same_scaler(scaler, loaded)
    new = _data() * 1.5 + 0.3
    new_in = pd.DataFrame(new, columns=["a", "b", "c"]) if frame else new
    assert scaler.transform(new_in).tobytes() == loaded.transform(new_in).tobytes()
    z = scaler.transform(new_in)
    z = pd.DataFrame(z, columns=["a", "b", "c"]) if frame else z
    assert scaler.inverse_transform(z).tobytes() == loaded.inverse_transform(z).tobytes()


def test_standard_scaler_with_nan_keeps_per_feature_counts(tmp_path):
    scaler = StandardScaler().fit(_data(nan=True))
    assert isinstance(scaler.n_samples_seen_, np.ndarray)
    loaded = _round_trip(scaler, tmp_path)
    _assert_same_scaler(scaler, loaded)


def test_encode_layout():
    scaler = StandardScaler().fit(pd.DataFrame(_data(), columns=["a", "b", "c"]))
    json_dict, arrays = encode_scaler(scaler)
    assert json_dict["class"] == "sklearn.preprocessing.StandardScaler"
    assert json_dict["params"] == scaler.get_params()
    assert sorted(arrays) == ["scaler/mean_", "scaler/n_samples_seen_", "scaler/scale_", "scaler/var_"]
    assert json_dict["attributes"]["n_features_in_"] == {"kind": "json", "value": 3}
    assert json_dict["attributes"]["feature_names_in_"] == {"kind": "object_array", "value": ["a", "b", "c"]}
    json.dumps(json_dict)  # plain JSON
    assert decode_scaler(json_dict, arrays).mean_.tobytes() == scaler.mean_.tobytes()


def test_unfitted_scaler_round_trips(tmp_path):
    loaded = _round_trip(RobustScaler(with_scaling=False), tmp_path)
    assert type(loaded) is RobustScaler and loaded.get_params() == RobustScaler(with_scaling=False).get_params()
    assert not [k for k in vars(loaded) if k.endswith("_")]


def test_none_round_trips(tmp_path):
    assert encode_scaler(None) == ({"class": None}, {})
    assert decode_scaler({"class": None}, {}) is None
    assert write_scaler(tmp_path, None) == {}
    assert json.loads((tmp_path / SCALER_JSON).read_text()) == {"class": None}
    assert has_scaler(tmp_path)
    assert read_scaler(tmp_path, {}) is None


def test_known_names():
    assert KNOWN_SCALERS == tuple(
        f"sklearn.preprocessing.{c.__name__}"
        for c in (StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler, QuantileTransformer)
    )
    assert is_known_scaler(None) and is_known_scaler(MaxAbsScaler())
    assert not is_known_scaler(DoublingScaler())


class _MyStandardScaler(StandardScaler):
    pass


@pytest.mark.parametrize(
    "scaler",
    [DoublingScaler(factor=3.0), _MyStandardScaler(), QuantileTransformer(n_quantiles=10, random_state=np.random.RandomState(0))],
    ids=["custom", "subclass", "unencodable-params"],
)
def test_other_scalers_are_pickled_and_gated(tmp_path, scaler):
    scaler.fit(_data())
    assert write_scaler(tmp_path, scaler) == {}
    assert sorted(p.name for p in tmp_path.iterdir()) == [SCALER_PKL]
    assert has_scaler(tmp_path)
    with pytest.raises(ConfigurationError, match="allow_pickle=True"):
        read_scaler(tmp_path, {})
    loaded = read_scaler(tmp_path, {}, allow_pickle=True)
    assert type(loaded) is type(scaler)
    assert loaded.transform(_data()).tobytes() == scaler.transform(_data()).tobytes()


def test_encode_refuses_other_scalers():
    with pytest.raises(TypeError, match="not one of the scalers"):
        encode_scaler(DoublingScaler())


def test_decode_refuses_unknown_classes_and_missing_arrays():
    with pytest.raises(ConfigurationError, match="not one of the scalers"):
        decode_scaler({"class": "sklearn.preprocessing.Normalizer", "params": {}}, {})
    json_dict, _ = encode_scaler(StandardScaler().fit(_data()))
    with pytest.raises(ConfigurationError, match="state.npz has no 'scaler/"):
        decode_scaler(json_dict, {})


def test_read_scaler_without_files(tmp_path):
    assert not has_scaler(tmp_path)
    with pytest.raises(FileNotFoundError):
        read_scaler(tmp_path, {})
