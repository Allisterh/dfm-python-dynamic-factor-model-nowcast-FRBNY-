"""Fixtures for the golden suite."""

import pytest

from ._golden import pinned as _pinned


@pytest.fixture
def pinned():
    """Run the test with one thread and deterministic torch algorithms (restored afterwards)."""
    with _pinned():
        yield
