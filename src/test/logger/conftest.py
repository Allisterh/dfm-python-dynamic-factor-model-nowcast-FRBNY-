"""Fixtures for logger tests."""

import logging

import pytest


@pytest.fixture(autouse=True)
def restore_package_logger():
    """Restore the 'dfm_python' logger (handlers, level, propagation) after each test.

    Several tests here install handlers or change levels on the package logger;
    without this they would leak into tests that run later in the session.
    """
    package_logger = logging.getLogger("dfm_python")
    handlers = list(package_logger.handlers)
    level = package_logger.level
    propagate = package_logger.propagate
    yield
    for handler in list(package_logger.handlers):
        if handler not in handlers:
            package_logger.removeHandler(handler)
            handler.close()
    for handler in handlers:
        if handler not in package_logger.handlers:
            package_logger.addHandler(handler)
    package_logger.setLevel(level)
    package_logger.propagate = propagate
