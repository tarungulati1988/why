"""Tests for the why.git subprocess wrapper."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from why.git import GitError, GitNotFoundError, GitTimeoutError, NotAGitRepoError, run_git


def _tmp_path_is_clean(tmp_path: Path) -> bool:
    """Return True if tmp_path is NOT inside an existing git repo."""
    result = subprocess.run(
        ["git", "rev-parse", "--git-dir"],
        cwd=tmp_path, capture_output=True, text=True, check=False,
    )
    return result.returncode != 0

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REPO_NOT_FOUND_MSG = (
    "fatal: not a git repository (or any of the parent directories): .git"
)


def _completed(
    returncode: int, stdout: str = "", stderr: str = ""
) -> subprocess.CompletedProcess[str]:
    """Build a CompletedProcess stub with the given fields."""
    return subprocess.CompletedProcess(
        args=["git"],
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


# ---------------------------------------------------------------------------
# Unit tests — subprocess.run is mocked
# ---------------------------------------------------------------------------


def test_run_git_returns_stdout_verbatim(tmp_path: Path) -> None:
    """run_git returns stdout exactly as received, no stripping."""
    with patch("why.git.subprocess.run", return_value=_completed(0, stdout="abc\n")) as mock_run:
        result = run_git(["log", "--oneline"], cwd=tmp_path)
    assert result == "abc\n"
    mock_run.assert_called_once()
    assert mock_run.call_args.args[0] == ["git", "log", "--oneline"]


def test_not_a_git_repo_error(tmp_path: Path) -> None:
    """returncode=128 + 'not a git repository' in stderr raises NotAGitRepoError."""
    fake = _completed(128, stderr=_REPO_NOT_FOUND_MSG)
    with (
        patch("why.git.subprocess.run", return_value=fake),
        pytest.raises(NotAGitRepoError, match=str(tmp_path)),
    ):
        run_git(["status"], cwd=tmp_path)


def test_generic_git_error(tmp_path: Path) -> None:
    """Non-zero returncode without repo-missing message raises GitError (not NotAGitRepoError)."""
    fake = _completed(1, stderr="some other failure")
    with (
        patch("why.git.subprocess.run", return_value=fake),
        pytest.raises(GitError) as exc_info,
    ):
        run_git(["status"], cwd=tmp_path)
    # Must be base GitError, NOT the NotAGitRepoError subclass
    assert type(exc_info.value) is GitError
    assert "some other failure" in str(exc_info.value)


def test_git_not_found(tmp_path: Path) -> None:
    """FileNotFoundError from subprocess.run is translated to GitNotFoundError."""
    with (
        patch("why.git.subprocess.run", side_effect=FileNotFoundError),
        pytest.raises(GitNotFoundError),
    ):
        run_git(["status"], cwd=tmp_path)


def test_git_timeout(tmp_path: Path) -> None:
    """TimeoutExpired from subprocess.run is translated to GitTimeoutError with timeout value."""
    expired = subprocess.TimeoutExpired(cmd=["git"], timeout=30.0)
    with (
        patch("why.git.subprocess.run", side_effect=expired),
        pytest.raises(GitTimeoutError, match="30"),
    ):
        run_git(["status"], cwd=tmp_path, timeout=30.0)


def test_hardening_env_vars_applied(tmp_path: Path) -> None:
    """subprocess.run is called with hardening env vars layered on top of os.environ."""
    with patch("why.git.subprocess.run", return_value=_completed(0, stdout="")) as mock_run:
        run_git(["status"], cwd=tmp_path)

    env = mock_run.call_args.kwargs["env"]

    # Hardening keys must be present with the correct values
    assert env["GIT_TERMINAL_PROMPT"] == "0"
    assert env["GIT_PAGER"] == "cat"
    assert env["GIT_EDITOR"] == "true"
    assert env["LC_ALL"] == "C"

    # Must be layered on top of os.environ, not a replacement — at least one real env key present
    assert any(k in env for k in os.environ)


def test_cwd_passed_through(tmp_path: Path) -> None:
    """subprocess.run receives the cwd argument that was passed to run_git."""
    with patch("why.git.subprocess.run", return_value=_completed(0, stdout="")) as mock_run:
        run_git(["status"], cwd=tmp_path)

    assert mock_run.call_args.kwargs["cwd"] == tmp_path


# ---------------------------------------------------------------------------
# Integration tests — real git process
# ---------------------------------------------------------------------------


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """Create a minimal real git repo in tmp_path and return the path."""
    if not _tmp_path_is_clean(tmp_path):
        pytest.skip("tmp_path is inside an existing git repo; isolation required")
    # fmt: off
    subprocess.run(
        ["git", "init", "-q"], cwd=tmp_path, check=True, capture_output=True
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=tmp_path, check=True, capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=tmp_path, check=True, capture_output=True,
    )
    # fmt: on
    # Write a file and make an initial commit so HEAD exists
    (tmp_path / "README.md").write_text("hello\n")
    subprocess.run(
        ["git", "add", "."], cwd=tmp_path, check=True, capture_output=True
    )
    subprocess.run(
        ["git", "commit", "--quiet", "-m", "initial"],
        cwd=tmp_path, check=True, capture_output=True,
    )
    return tmp_path


def test_integration_rev_parse_head(git_repo: Path) -> None:
    """run_git on a real repo returns a 40-char hex SHA followed by a newline."""
    result = run_git(["rev-parse", "HEAD"], cwd=git_repo)
    # Strip trailing newline to validate the SHA itself, but verify original is unstripped
    sha = result.rstrip("\n")
    assert len(sha) == 40
    assert all(c in "0123456789abcdef" for c in sha)
    assert result.endswith("\n")


def test_integration_not_a_git_repo(tmp_path: Path) -> None:
    """run_git on a plain directory raises NotAGitRepoError."""
    if not _tmp_path_is_clean(tmp_path):
        pytest.skip("tmp_path is inside an existing git repo; isolation required")
    with pytest.raises(NotAGitRepoError):
        run_git(["status"], cwd=tmp_path)


def test_cwd_not_a_directory_raises_git_error(tmp_path: Path) -> None:
    """If cwd is a file (not a directory), run_git wraps the OSError as GitError."""
    not_a_dir = tmp_path / "file.txt"
    not_a_dir.write_text("x")
    with pytest.raises(GitError) as exc_info:
        run_git(["status"], cwd=not_a_dir)
    # Must be plain GitError, not a more specific subclass
    assert type(exc_info.value) is GitError
    assert "could not launch git" in str(exc_info.value)
