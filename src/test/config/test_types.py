"""Tests for config schema module."""

import pytest
from dfm_python.config import (
    DFMConfig,
    DDFMConfig,
)
from dfm_python.config.constants import DEFAULT_EM_THRESHOLD, DEFAULT_MAX_ITER


class TestDFMConfig:
    """Test suite for DFMConfig."""
    
    def test_dfm_config_initialization(self):
        """Test DFMConfig can be initialized with default values."""
        # DFMConfig requires at least one block
        config = DFMConfig(blocks={'block1': {'num_factors': 2, 'series': []}})
        assert config is not None
    
    def test_dfm_config_parameters(self):
        """Test DFMConfig parameter setting."""
        config = DFMConfig(
            blocks={'block1': {'num_factors': 2, 'series': []}},
            max_iter=DEFAULT_MAX_ITER,
            threshold=DEFAULT_EM_THRESHOLD
        )
        assert config.max_iter == DEFAULT_MAX_ITER
        assert config.threshold == DEFAULT_EM_THRESHOLD


class TestDDFMConfig:
    """Test suite for DDFMConfig."""
    
    def test_ddfm_config_initialization(self):
        """DDFMConfig() holds the DDFM constructor defaults; it has no clock (DDFM is single frequency)."""
        from dfm_python import DDFM
        config = DDFMConfig()
        assert config is not None
        assert config.to_dict() == DDFM().get_params()
        assert not hasattr(config, "clock")

