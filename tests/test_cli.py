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


def test_short_help_flag() -> None:
    """-h is an alias for --help: exits 0 and shows the same content."""
    result = CliRunner().invoke(main, ["-h"])
    assert result.exit_code == 0
    assert "TARGET" in result.output
    assert "--model" in result.output


def test_no_args_shows_help() -> None:
    """Invoking why with no arguments prints help.

    Click's no_args_is_help=True raises NoArgsIsHelpError (a UsageError subclass)
    which exits with code 2, not 0. The help text is still shown to the user.
    """
    result = CliRunner().invoke(main, [])
    # Click's no_args_is_help always exits 2 (UsageError), not 0
    assert result.exit_code == 2
    assert "TARGET" in result.output


# ---------------------------------------------------------------------------
# Real-git integration tests
# ---------------------------------------------------------------------------


@pytest.fixture
def line_history_cli_repo(tmp_path: Path) -> Path:
    """Build a real git repo where 3 commits each touch line 1 of foo.py.

    This mirrors the `line_history_repo` fixture in test_history.py but lives
    here so CLI end-to-end tests have a self-contained isolated repo.

    History (oldest to newest):
      A — write 10 lines to foo.py (line 1 = "line1")
      B — change line 1 to "line1 updated"
      C — change line 1 to "line1 final"

    Returns the repo root Path.
    """
    from conftest import _tmp_path_is_clean, make_git_runner

    if not _tmp_path_is_clean(tmp_path):
        pytest.skip("tmp_path is inside an existing git repo — skipping integration test")

    repo = tmp_path / "repo"
    repo.mkdir()

    git = make_git_runner(repo)
    import subprocess as _sp
    _git_ver = _sp.run(["git", "--version"], capture_output=True, text=True, check=True).stdout
    # "git version 2.39.2" → (2, 39)
    _major, _minor = (int(x) for x in _git_ver.split()[2].split(".")[:2])
    if (_major, _minor) >= (2, 28):
        git("init", "-b", "main")
    else:
        git("init")
        git("symbolic-ref", "HEAD", "refs/heads/main")

    # Commit A: create foo.py with 10 lines
    lines = "\n".join(f"line{i}" for i in range(1, 11)) + "\n"
    (repo / "foo.py").write_text(lines)
    git("add", "foo.py")
    git("commit", "-m", "A: create foo.py")

    # Commit B: update line 1
    lines_b = "\n".join(
        ["line1 updated"] + [f"line{i}" for i in range(2, 11)]
    ) + "\n"
    (repo / "foo.py").write_text(lines_b)
    git("add", "foo.py")
    git("commit", "-m", "B: update line 1")

    # Commit C: update line 1 again
    lines_c = "\n".join(
        ["line1 final"] + [f"line{i}" for i in range(2, 11)]
    ) + "\n"
    (repo / "foo.py").write_text(lines_c)
    git("add", "foo.py")
    git("commit", "-m", "C: finalize line 1")

    return repo


class TestLineTargetEndToEnd:
    """CLI end-to-end: `why <file>:1` runs the full stack against a real git repo."""

    def test_file_line_target_runs_end_to_end(
        self, line_history_cli_repo: Path
    ) -> None:
        """Invoking `why foo.py:1` on a real git repo:

        - Passes through parse_target, get_line_history, and synthesize_why
          for real (none of those are mocked).
        - LLMClient.complete() is mocked to return "line explanation" so the
          test is hermetic without a live API key.
        - Path.cwd() is redirected to the test repo root so parse_target can
          resolve the absolute path without raising "path escapes repository root".
        - Exit code must be 0, output must contain the mocked explanation, and
          LLMClient must be constructed exactly once.
        """
        repo = line_history_cli_repo
        # Resolve symlinks so parse_target's relative_to guard passes on macOS
        # (tmp_path may be /var/folders/… which resolves to /private/var/folders/…).
        target_spec = str((repo / "foo.py").resolve()) + ":1"

        with (
            patch("why.cli.Path") as mock_path_cls,
            patch("why.cli.LLMClient") as mock_llm_cls,
        ):
            # Forward all Path construction to the real Path so synthesize_why
            # and parse_target keep working — only override cwd() to point at
            # the test repo so the "path escapes repo root" guard passes.
            mock_path_cls.side_effect = Path
            mock_path_cls.cwd.return_value = repo.resolve()

            # Wire the mock instance so .complete() returns our sentinel string
            mock_llm_instance = MagicMock()
            mock_llm_instance.complete.return_value = "line explanation"
            mock_llm_cls.return_value = mock_llm_instance

            result = CliRunner().invoke(main, [target_spec])

        assert result.exit_code == 0, result.output
        assert result.output.strip() == "line explanation"
        # LLMClient must be constructed once (with the default model)
        mock_llm_cls.assert_called_once()
        # complete() must be called — guards against future short-circuit paths
        mock_llm_instance.complete.assert_called_once()
        # Verify the LLM received a prompt that references the target file,
        # catching regressions where the pipeline passes an empty or wrong prompt.
        call_args = mock_llm_instance.complete.call_args
        messages = call_args.args[1]
        assert any("foo.py" in str(m) for m in messages), "LLM messages should reference the target file"
