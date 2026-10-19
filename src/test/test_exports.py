"""The top-level ``dfm_python`` namespace."""

import pytest

import dfm_python
import dfm_python.config
from dfm_python.base import BaseFactorModel
from dfm_python.dataset.panel import PanelData
from dfm_python.io.loading import load
from dfm_python.logger import get_verbosity, set_verbosity
from dfm_python.models.afm.estimator import AFM
from dfm_python.models.ivdfm.estimator import iVDFM


def test_new_api_names_are_exported():
    assert dfm_python.PanelData is PanelData
    assert dfm_python.load is load
    assert dfm_python.set_verbosity is set_verbosity and dfm_python.get_verbosity is get_verbosity
    assert dfm_python.BaseFactorModel is BaseFactorModel
    assert dfm_python.AFM is AFM
    assert dfm_python.iVDFM is iVDFM
    for name in ("PanelData", "load", "set_verbosity", "get_verbosity", "BaseFactorModel", "AFM", "iVDFM"):
        assert name in dfm_python.__all__, name


def test_every_name_in_all_exists():
    missing = [name for name in dfm_python.__all__ if not hasattr(dfm_python, name)]
    assert not missing


def test_afm_dataset_is_not_public():
    # The AFM dataset is the estimator's private adapter (dfm_python.dataset.afm_dataset).
    assert "AFMDataset" not in dfm_python.__all__
    assert not hasattr(dfm_python, "AFMDataset")


def test_afm_model_state_is_not_public():
    # AFMModelState is used only by the engine and AFM's save/load.
    assert "AFMModelState" not in dfm_python.config.__all__
    assert not hasattr(dfm_python.config, "AFMModelState")


def test_ivdfm_is_the_estimator():
    from dfm_python.models.ivdfm._engine import iVDFMEngine

    assert dfm_python.iVDFM is iVDFM and "iVDFM" in dfm_python.__all__
    assert issubclass(dfm_python.iVDFM, BaseFactorModel) and dfm_python.iVDFM is not iVDFMEngine


def test_ivdfm_dataset_is_not_public():
    # The iVDFM dataset is the estimator's private adapter base (dfm_python.dataset.ivdfm_dataset).
    import dfm_python.dataset

    assert "iVDFMDataset" not in dfm_python.__all__
    assert not hasattr(dfm_python, "iVDFMDataset")
    assert "iVDFMDataset" not in dfm_python.dataset.__all__
    assert not hasattr(dfm_python.dataset, "iVDFMDataset")


def test_ivdfm_model_state_is_not_public():
    # iVDFMModelState is used only by the engine and iVDFM's save/load.
    import dfm_python.config.schema as schema

    assert "iVDFMModelState" not in schema.__all__ and not hasattr(schema, "iVDFMModelState")
    assert "iVDFMModelState" not in dfm_python.config.__all__
    assert not hasattr(dfm_python.config, "iVDFMModelState")
    assert "iVDFMModelState" not in dfm_python.__all__ and not hasattr(dfm_python, "iVDFMModelState")


def test_ivdfm_engine_and_engine_config_are_not_public():
    import dfm_python.config.schema as schema
    import dfm_python.models
    import dfm_python.models.ivdfm as ivdfm_pkg

    for namespace in (dfm_python, dfm_python.config, schema, dfm_python.models, ivdfm_pkg):
        for name in ("iVDFMEngine", "_iVDFMEngineConfig"):
            assert name not in getattr(namespace, "__all__", ()), (namespace.__name__, name)
        assert not hasattr(namespace, "iVDFMEngine"), namespace.__name__
    assert not hasattr(dfm_python, "_iVDFMEngineConfig") and not hasattr(dfm_python.config, "_iVDFMEngineConfig")
    assert ivdfm_pkg.iVDFM is iVDFM and "iVDFM" in ivdfm_pkg.__all__


def test_ivdfm_optuna_module_is_gone():
    import importlib

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("dfm_python.models.ivdfm.optuna")


def test_ivdfm_config_is_the_estimator_mirror():
    from dfm_python.config import EstimatorConfig

    assert dfm_python.iVDFMConfig is dfm_python.config.iVDFMConfig is iVDFM._config_class
    assert issubclass(dfm_python.iVDFMConfig, EstimatorConfig)
