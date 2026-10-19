"""Golden tests for AFM.

``test_golden_afm_engine`` runs the engine (the pre-redesign class, now
``AFMEngine``) exactly as the fixtures were captured. ``test_golden_afm_estimator``
runs the public ``AFM`` estimator on the same data with the recorded
hyperparameters (``seed`` -> ``random_state``, ``device="cpu"``), built from the
constructor and from a YAML file, and must reproduce the same numbers.
"""

import importlib.util

import numpy as np
import pytest

from dfm_python import AFM, PanelData
from dfm_python.config import AFMConfig

from ._cases import AFM_ATTRS, _attrs, _reset_global_rngs, afm_inputs, cases_for, check_case, normalized
from ._golden import assert_golden, load_golden

_YAML = importlib.util.find_spec("omegaconf") is not None or importlib.util.find_spec("yaml") is not None


@pytest.mark.parametrize("case", cases_for("afm"), ids=repr)
def test_golden_afm_engine(case, pinned):
    check_case(case.name, case.run())


def _estimator_params(hyper: dict) -> dict:
    """Constructor parameters for the recorded engine hyperparameters."""
    params = {k: v for k, v in hyper.items() if k not in ("seed", "n_char", "device")}
    params["random_state"] = hyper["seed"]
    params["device"] = "cpu"
    return params


def _build(build: str, params: dict, tmp_path) -> AFM:
    if build == "constructor":
        return AFM(**params)
    if not _YAML:
        pytest.skip("YAML needs omegaconf or PyYAML")
    path = tmp_path / "afm.yaml"
    AFMConfig(**params).to_yaml(path)
    return AFM.from_config(AFMConfig.from_yaml(path))


@pytest.mark.parametrize("build", ["constructor", "yaml"])
@pytest.mark.parametrize("case", cases_for("afm"), ids=repr)
def test_golden_afm_estimator(case, build, pinned, tmp_path):
    manifest, _ = load_golden(case.name)
    hyper = manifest["hyperparameters"]
    params = _estimator_params(hyper)
    returns, characteristics, recipe = afm_inputs(hyper["factor_model"])
    assert normalized(recipe) == manifest["data"]

    model = _build(build, params, tmp_path)
    assert model.get_params() == {**AFM().get_params(), **params}
    data = PanelData.from_arrays(returns, series_exog=characteristics)

    _reset_global_rngs()
    assert model.fit(data) is model
    weights = model.allocate()

    # Every hyperparameter reached the engine unchanged.
    assert normalized(_attrs(model.network_, AFM_ATTRS)) == hyper
    assert model.n_char_ == hyper["n_char"]
    assert all(dtype == np.float64 for dtype in weights.dtypes)

    outputs = {
        "residuals": model.result_.residual_portfolios,
        "book_returns": model.result_.book_returns,
        "sharpe": np.float32(model.result_.sharpe),
        "explained_variance": np.float32(model.result_.explained_variance),
        "num_iter": np.int64(model.n_iter_),
        # float32 -> float64 -> float32 is exact, so this stays a bitwise comparison.
        "predict_weights": weights.to_numpy().astype(np.float32)[None],
    }
    for key, value in model.network_.state_dict().items():
        outputs[f"sd/{key}"] = value.detach().cpu().numpy()
    assert_golden(outputs, case.name)

    # The derived attributes are the same numbers in float64.
    np.testing.assert_array_equal(model.residuals_.to_numpy(), model.result_.residual_portfolios[0].astype(np.float64))
    np.testing.assert_array_equal(model.book_returns_.to_numpy(), model.result_.book_returns[0].astype(np.float64))
