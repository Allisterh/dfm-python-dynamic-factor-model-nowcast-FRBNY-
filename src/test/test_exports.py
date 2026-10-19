"""The top-level ``dfm_python`` namespace."""

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


def test_ivdfm_model_state_is_not_public():
    # iVDFMModelState is used only by the engine and iVDFM's save/load.
    import dfm_python.config.schema as schema

    assert "iVDFMModelState" not in schema.__all__ and not hasattr(schema, "iVDFMModelState")
    assert "iVDFMModelState" not in dfm_python.config.__all__
    assert not hasattr(dfm_python.config, "iVDFMModelState")


def test_ivdfm_config_is_the_estimator_mirror():
    from dfm_python.config import EstimatorConfig

    assert dfm_python.iVDFMConfig is dfm_python.config.iVDFMConfig is iVDFM._config_class
    assert issubclass(dfm_python.iVDFMConfig, EstimatorConfig)
