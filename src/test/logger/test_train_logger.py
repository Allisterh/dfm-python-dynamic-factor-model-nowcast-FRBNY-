"""Tests for logger.train_logger module."""

import pytest


class TestTrainLogger:
    """Test suite for BaseTrainLogger."""
    
    def test_train_logger_initialization(self):
        """Test BaseTrainLogger can be initialized."""
        from dfm_python.logger.train_logger import BaseTrainLogger
        
        logger = BaseTrainLogger(model_name="TestModel", verbose=True)
        assert logger.model_name == "TestModel"
        assert logger.verbose is True
        assert hasattr(logger, 'iterations')
        assert logger.iterations == 0  # Initialized to 0



def test_log_training_start_for_ddfm_uses_the_base_logger():
    """The DDFM-specific logger module is gone; DDFM falls back to the base logger with its name."""
    import importlib

    from dfm_python.logger.train_logger import BaseTrainLogger, log_training_start

    logger = log_training_start("DDFM")
    assert type(logger) is BaseTrainLogger
    assert logger.model_name == "DDFM" and logger.verbose is True
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("dfm_python.logger.ddfm_logger")
