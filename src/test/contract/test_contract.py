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
    check_estimator(cls)
