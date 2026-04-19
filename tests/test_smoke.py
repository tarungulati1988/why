"""Smoke tests for the why package bootstrap scaffolding."""

from click.testing import CliRunner

import why
from why.cli import main


def test_version_attribute() -> None:
    """Test A: the package exposes a non-empty __version__ string."""
    assert isinstance(why.__version__, str)
    assert why.__version__


def test_cli_version_flag() -> None:
    """Test B: the CLI --version flag exits 0 and outputs the version string."""
    result = CliRunner().invoke(main, ["--version"])
    assert result.exit_code == 0
    assert why.__version__ in result.output
