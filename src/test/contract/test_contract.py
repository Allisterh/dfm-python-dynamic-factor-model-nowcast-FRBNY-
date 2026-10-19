"""The API contract (spec section 11) for every estimator in ``registry.all_estimators()``."""

import pytest

from dfm_python.base import BaseFactorModel
from dfm_python.registry import all_estimators
from dfm_python.testing import check_estimator

_ESTIMATORS = all_estimators()


def test_registry_lists_estimator_classes():
    names = [name for name, _ in _ESTIMATORS]
    assert names == sorted(names) and len(set(names)) == len(names)
    for name, cls in _ESTIMATORS:
        assert isinstance(cls, type) and issubclass(cls, BaseFactorModel)
        assert cls.__name__ == name


@pytest.mark.parametrize("name, cls", _ESTIMATORS, ids=[name for name, _ in _ESTIMATORS])
def test_contract(name, cls):
    ran = check_estimator(cls)
    # Every check that applies to every estimator ran, save/load included.
    for check in ("pure_construction", "config_serialization", "unfitted", "fit", "purity", "silence", "save_load"):
        assert check in ran, check


def test_registry_lists_exactly_the_four_models():
    assert [name for name, _ in all_estimators()] == ["AFM", "DDFM", "DFM", "iVDFM"]
    import dfm_python

    for name, cls in all_estimators():
        assert getattr(dfm_python, name) is cls, name


@pytest.mark.parametrize("name, cls", _ESTIMATORS, ids=[name for name, _ in _ESTIMATORS])
def test_package_load_dispatches_to_each_model(name, cls, tmp_path):
    import warnings

    import dfm_python
    from dfm_python.testing import make_panel

    model = cls(**cls.get_test_params()[0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(make_panel(cls))
    model.save(tmp_path / "run")
    loaded = dfm_python.load(tmp_path / "run")
    assert type(loaded) is cls
    assert loaded.get_params() == model.get_params()
