"""Tests for PyPI distribution metadata (name and version)."""

import importlib.metadata

import why


def test_distribution_name() -> None:
    """The installed distribution must be named 'git-why'."""
    meta = importlib.metadata.metadata("git-why")
    assert meta["Name"] == "git-why"


def test_distribution_version_matches_module() -> None:
    """Distribution metadata version must match why.__version__."""
    assert importlib.metadata.version("git-why") == why.__version__
