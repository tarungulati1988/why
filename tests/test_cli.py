"""Tests for the why CLI entry point."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from why.cli import main
from why.llm import LLMError

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def existing_file(tmp_path: Path) -> Path:
    """Create a real file on disk that parse_target can resolve."""
    f = tmp_path / "example.py"
    f.write_text("# example\n")
    return f


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_help_flag() -> None:
    """--help exits 0 and documents TARGET, SYMBOL, --model, and argument descriptions."""
    result = CliRunner().invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "TARGET" in result.output
    assert "SYMBOL" in result.output
    assert "--model" in result.output
    assert "ARGUMENTS:" in result.output
    assert "whole-file analysis" in result.output


def test_happy_path(existing_file: Path) -> None:
    """Valid target file → synthesize_why is called → its output is printed, exit 0."""
    # Patch Path.cwd so parse_target treats tmp_path as the repo root, avoiding
    # the "path escapes repository root" error when the file is in /tmp.
    with (
        patch("why.cli.Path") as mock_path_cls,
        patch("why.cli.LLMClient") as mock_llm_cls,
        patch("why.cli.synthesize_why", return_value="explanation") as mock_synth,
    ):
        mock_llm_instance = MagicMock()
        mock_llm_cls.return_value = mock_llm_instance
        # Make Path.cwd() return the parent of existing_file so the path resolves.
        mock_path_cls.cwd.return_value = existing_file.parent

        runner = CliRunner()
        result = runner.invoke(main, [str(existing_file)])

    assert result.exit_code == 0, result.output
    assert "explanation" in result.output
    mock_synth.assert_called_once()


def test_missing_file_exits_1() -> None:
    """Non-existent path → exit code 1, output contains 'Error:'."""
    # CliRunner in Click 8.3 mixes stderr into output by default.
    runner = CliRunner()
    result = runner.invoke(main, ["/nonexistent/path/to/file.py"])
    assert result.exit_code == 1
    assert "Error:" in result.output


def test_model_flag_forwarded(existing_file: Path) -> None:
    """--model mymodel → LLMClient is constructed with 'mymodel'."""
    # Patch Path.cwd so parse_target treats tmp_path as the repo root.
    with (
        patch("why.cli.Path") as mock_path_cls,
        patch("why.cli.LLMClient") as mock_llm_cls,
        patch("why.cli.synthesize_why", return_value="ok"),
    ):
        mock_llm_cls.return_value = MagicMock()
        mock_path_cls.cwd.return_value = existing_file.parent

        runner = CliRunner()
        result = runner.invoke(main, ["--model", "mymodel", str(existing_file)])

    assert result.exit_code == 0, result.output
    # Confirm LLMClient was constructed with the custom model name
    mock_llm_cls.assert_called_once_with("mymodel")


def test_llm_error_exits_1(existing_file: Path) -> None:
    """synthesize_why raising LLMError → exit code 1, output contains 'Error:'."""
    with (
        patch("why.cli.Path") as mock_path_cls,
        patch("why.cli.LLMClient") as mock_llm_cls,
        patch("why.cli.synthesize_why", side_effect=LLMError("api failed")),
    ):
        mock_llm_cls.return_value = MagicMock()
        mock_path_cls.cwd.return_value = existing_file.parent

        runner = CliRunner()
        result = runner.invoke(main, [str(existing_file)])

    assert result.exit_code == 1
    assert "Error:" in result.output
