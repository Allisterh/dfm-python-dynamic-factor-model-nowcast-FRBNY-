"""Tests for tent weights parameterization in _DFMEngineConfig."""

import importlib.util

import pytest
import numpy as np
import yaml
from pathlib import Path
from dfm_python import DFMConfig
from dfm_python.config.schema.model import _DFMEngineConfig
from dfm_python.dataset.dfm_dataset import DFMDataset
from dfm_python.models.dfm.tent import get_agg_structure


class TestTentWeightsConfig:
    """Test suite for tent weights configuration."""
    
    def test_tent_weights_in_config_dict(self):
        """Test that tent_weights parameter can be set in config dict."""
        config_dict = {
            'clock': 'w',
            'frequency': {'m': ['series1'], 'w': ['series2']},
            'blocks': {'Block_Global': {'num_factors': 1, 'series': ['series1', 'series2']}},
            'tent_weights': {'m:w': [1, 2, 1]},  # Custom tent weights
        }
        
        config = _DFMEngineConfig.from_dict(config_dict)
        assert config.tent_weights is not None
        assert 'm:w' in config.tent_weights
        assert config.tent_weights['m:w'] == [1, 2, 1]
    
    def test_tent_weights_override_lookup(self):
        """Test that config tent_weights are used correctly (no lookup table anymore)."""
        # Custom tent weights different from lookup
        custom_weights = [1, 2, 3, 2, 1]  # 5 periods
        config_dict = {
            'clock': 'w',
            'frequency': {'m': ['series1'], 'w': ['series2']},
            'blocks': {'Block_Global': {'num_factors': 1, 'series': ['series1', 'series2']}},
            'tent_weights': {'m:w': custom_weights},
        }
        
        config = _DFMEngineConfig.from_dict(config_dict)
        
        # Verify config has custom weights
        assert config.tent_weights['m:w'] == custom_weights
        
        # Verify get_agg_structure uses config weights
        agg_structure = get_agg_structure(config, clock='w')
        assert 'm' in agg_structure['tent_weights']
        assert np.array_equal(agg_structure['tent_weights']['m'], np.array(custom_weights))
    
    def test_tent_weights_required_for_mixed_freq(self):
        """Test that tent_weights are required for mixed-frequency data (no fallback)."""
        from dfm_python.utils.errors import DataValidationError
        
        config_dict = {
            'clock': 'w',
            'frequency': {'m': ['series1'], 'w': ['series2']},
            'blocks': {'Block_Global': {'num_factors': 1, 'series': ['series1', 'series2']}},
            # No tent_weights specified - should raise error for mixed-frequency data
        }
        
        config = _DFMEngineConfig.from_dict(config_dict)
        assert config.tent_weights is None
        
        # Should raise error when trying to get aggregation structure for mixed-frequency data
        with pytest.raises(DataValidationError, match="tent_weights"):
            get_agg_structure(config, clock='w')
    
    def test_tent_weights_frequency_shorthand(self):
        """Test that tent_weights can use frequency-only key (shorthand)."""
        config_dict = {
            'clock': 'w',
            'frequency': {'m': ['series1'], 'w': ['series2']},
            'blocks': {'Block_Global': {'num_factors': 1, 'series': ['series1', 'series2']}},
            'tent_weights': {'m': [1, 2, 1]},  # Shorthand: just frequency, uses clock
        }
        
        config = _DFMEngineConfig.from_dict(config_dict)
        assert config.tent_weights is not None
        assert 'm' in config.tent_weights
        assert config.tent_weights['m'] == [1, 2, 1]
    
    def test_tent_weights_multiple_pairs(self):
        """Test tent_weights with multiple frequency pairs."""
        config_dict = {
            'clock': 'w',
            'frequency': {'m': ['series1'], 'q': ['series2'], 'w': ['series3']},
            'blocks': {'Block_Global': {'num_factors': 1, 'series': ['series1', 'series2', 'series3']}},
            'tent_weights': {
                'm:w': [1, 2, 1],  # Monthly -> Weekly
                'q:w': [1, 2, 3, 4, 5, 4, 3, 2, 1],  # Quarterly -> Weekly
            },
        }
        
        config = _DFMEngineConfig.from_dict(config_dict)
        assert config.tent_weights is not None
        assert 'm:w' in config.tent_weights
        assert 'q:w' in config.tent_weights
        assert config.tent_weights['m:w'] == [1, 2, 1]
        assert len(config.tent_weights['q:w']) == 9


@pytest.mark.skipif(
    importlib.util.find_spec("omegaconf") is None and importlib.util.find_spec("yaml") is None,
    reason="YAML needs omegaconf or PyYAML",
)
class TestInvestmentProductionConfigs:
    """Tent kernels read from YAML files.

    These tests read ``config/model/dfm/{investment,production}.yaml`` through
    the removed ``YamlSource`` and skipped, because the repository has no such
    files. They now write the two kernels (5-period for investment, 3-period
    for production) in the ``model:``/``data:`` layout and read them through
    ``DFMConfig.from_yaml``.
    """

    @staticmethod
    def _write(path, kernel):
        path.write_text(
            "model:\n"
            "  clock: w\n"
            f"  tent_weights: {{'m:w': {kernel}}}\n"
            "data:\n  freq: {m: [series1], w: [series2]}\n",
            encoding="utf-8",
        )
        return path

    def test_investment_config_tent_weights(self, tmp_path):
        config = DFMConfig.from_yaml(self._write(tmp_path / "investment.yaml", [1, 2, 3, 2, 1]))
        assert config.tent_weights == {'m:w': [1, 2, 3, 2, 1]}

    def test_production_config_tent_weights(self, tmp_path):
        config = DFMConfig.from_yaml(self._write(tmp_path / "production.yaml", [1, 2, 1]))
        assert config.tent_weights == {'m:w': [1, 2, 1]}

    def test_config_tent_weights_different(self, tmp_path):
        investment = DFMConfig.from_yaml(self._write(tmp_path / "investment.yaml", [1, 2, 3, 2, 1]))
        production = DFMConfig.from_yaml(self._write(tmp_path / "production.yaml", [1, 2, 1]))
        assert investment.tent_weights['m:w'] != production.tent_weights['m:w']
        # The engine config the estimator builds receives each kernel unchanged.
        for config in (investment, production):
            engine_config = _DFMEngineConfig.from_dict(
                {'clock': 'w', 'frequency': {'series1': 'm', 'series2': 'w'},
                 'blocks': {'Block_Global': {'num_factors': 1, 'series': ['series1', 'series2']}},
                 'tent_weights': config.tent_weights}
            )
            assert engine_config.tent_weights == config.tent_weights


class TestDatasetTentWeightsIntegration:
    """Test suite for dataset integration with tent weights."""
    
    def test_dataset_uses_config_tent_weights(self):
        """Test that DFMDataset uses tent_weights from config."""
        import pandas as pd
        
        # Create minimal data
        dates = pd.date_range('2020-01-01', periods=20, freq='W')
        data = pd.DataFrame({
            'date': dates,
            'weekly_series': np.random.randn(20),
            'monthly_series': [np.nan] * 19 + [1.0],  # Only last value (monthly observation)
        })
        
        # Config with custom tent weights
        config_dict = {
            'clock': 'w',
            'frequency': {'w': ['weekly_series'], 'm': ['monthly_series']},
            'blocks': {'Block_Global': {'num_factors': 1, 'series': ['weekly_series', 'monthly_series']}},
            'tent_weights': {'m:w': [1, 2, 1]},  # 3-period tent kernel
        }
        
        config = _DFMEngineConfig.from_dict(config_dict)
        
        # Create dataset
        dataset = DFMDataset(config=config, data=data, time_index='date')
        
        # Mixed-frequency parameters are resolved from the dataset's config
        # and columns (formerly DFMDataset.get_initialization_params()).
        from dfm_python.models.dfm.mixed_freq import setup_mixed_frequency_params
        columns = list(dataset.variables.columns)
        init_params = setup_mixed_frequency_params(
            dataset.config, dataset.config.clock, columns, len(columns)
        )
        assert 'tent_weights_dict' in init_params
        assert init_params['tent_weights_dict'] is not None
        
        # Verify tent weights match config
        tent_weights_dict = init_params['tent_weights_dict']
        assert 'm' in tent_weights_dict
        assert np.array_equal(tent_weights_dict['m'], np.array([1, 2, 1], dtype=np.float32))
