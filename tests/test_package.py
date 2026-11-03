"""Smoke tests for the package."""

import pmc


def test_version() -> None:
    assert pmc.__version__ == "0.1.0"
