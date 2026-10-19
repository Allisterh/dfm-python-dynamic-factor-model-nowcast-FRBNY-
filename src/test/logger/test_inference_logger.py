"""Tests for logger.inference_logger module."""

import pytest


class TestInferenceLogger:
    """Test suite for BaseInferenceLogger."""
    
    def test_inference_logger_initialization(self):
        """Test BaseInferenceLogger can be initialized."""
        from dfm_python.logger.inference_logger import BaseInferenceLogger
        
        logger = BaseInferenceLogger(model_name="TestModel", verbose=True)
        assert logger.model_name == "TestModel"
        assert logger.verbose is True
        assert logger.num_predictions == 0



def test_log_inference_start_for_ddfm_uses_the_base_logger():
    """The DDFM-specific logger module is gone; DDFM falls back to the base logger with its name."""
    from dfm_python.logger.inference_logger import BaseInferenceLogger, log_inference_start

    logger = log_inference_start("DDFM")
    assert type(logger) is BaseInferenceLogger
    assert logger.model_name == "DDFM" and logger.verbose is True
