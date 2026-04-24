"""Tests for why.diff.get_commit_diff."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import call, patch

import pytest
from conftest import _tmp_path_is_clean, make_git_runner

from why.diff import get_commit_diff
from why.git import GitError, GitTimeoutError

# ---------------------------------------------------------------------------
# Unit tests — why.diff.run_git is mocked
# ---------------------------------------------------------------------------


def test_whole_file_mode_passes_correct_args(tmp_path: Path) -> None:
    """Whole-file mode (line_range=None) calls git show with the expected args."""
    sha = "abc1234"
    file = Path("path/to/file.py")
    expected_args = ["show", "--format=", sha, "--", str(file)]

    with patch("why.diff.run_git", return_value="diff output\n") as mock_run_git:
        get_commit_diff(sha=sha, file=file, line_range=None, repo=tmp_path)

    mock_run_git.assert_called_once_with(expected_args, cwd=tmp_path)


def test_line_range_mode_passes_correct_args(tmp_path: Path) -> None:
    """Line-range mode calls git log -L with the correct range and sha^! args."""
    sha = "abc1234"
    file = Path("path/to/file.py")
    line_range = (5, 10)
    expected_args = ["log", "-L", "5,10:path/to/file.py", "abc1234^!"]

    with patch("why.diff.run_git", return_value="diff output\n") as mock_run_git:
        get_commit_diff(sha=sha, file=file, line_range=line_range, repo=tmp_path)

    mock_run_git.assert_called_once_with(expected_args, cwd=tmp_path)


def test_output_shorter_than_max_chars_returned_verbatim(tmp_path: Path) -> None:
    """Output shorter than max_chars is returned unchanged."""
    output = "short output\n"
    with patch("why.diff.run_git", return_value=output):
        result = get_commit_diff(
            sha="abc1234", file=Path("f.py"), line_range=None, repo=tmp_path, max_chars=2000
        )
    assert result == output


def test_output_truncated_when_over_max_chars(tmp_path: Path) -> None:
    """Output of max_chars+1 chars is truncated to max_chars with a truncation suffix."""
    max_chars = 50
    # Build a string that is exactly max_chars+1 characters long
    output = "x" * (max_chars + 1)
    with patch("why.diff.run_git", return_value=output):
        result = get_commit_diff(
            sha="abc1234", file=Path("f.py"), line_range=None, repo=tmp_path, max_chars=max_chars
        )
    assert result == "x" * max_chars + "\n... [truncated]"


def test_root_commit_fallback_on_git_error(tmp_path: Path) -> None:
    """When git log -L raises GitError (root commit), falls back to git show whole-file."""
    sha = "abc1234"
    file = Path("path/to/file.py")
    fallback_output = "fallback diff output\n"

    # First call (line-range) raises GitError; second call (whole-file) succeeds.
    with patch(
        "why.diff.run_git",
        side_effect=[GitError("fatal: ambiguous argument"), fallback_output],
    ) as mock_run_git:
        result = get_commit_diff(
            sha=sha, file=file, line_range=(1, 2), repo=tmp_path
        )

    # Should have made exactly two calls
    assert mock_run_git.call_count == 2
    # Second call must be the whole-file fallback
    second_call_args = mock_run_git.call_args_list[1]
    assert second_call_args == call(
        ["show", "--format=", sha, "--", str(file)], cwd=tmp_path
    )
    assert result == fallback_output


def test_git_error_subclass_propagates_from_line_range_mode(tmp_path: Path) -> None:
    """GitError subclasses (e.g. GitTimeoutError) on the -L path must not be swallowed."""
    with patch("why.diff.run_git", side_effect=GitTimeoutError("timed out")):
        with pytest.raises(GitTimeoutError):
            get_commit_diff("abc1234", Path("f.py"), line_range=(1, 5), repo=tmp_path)


# ---------------------------------------------------------------------------
# Integration tests — real git repo
# ---------------------------------------------------------------------------


@pytest.fixture
def diff_repo(tmp_path: Path) -> tuple[Path, str, str]:
    """Return (repo_path, root_sha, second_sha) for a repo with two commits.

    Commit 1 (root): hello.py with 5 lines.
    Commit 2: lines 2-3 of hello.py modified.
    """
    if not _tmp_path_is_clean(tmp_path):
        pytest.skip("tmp_path is inside an existing git repo; isolation required")

    git = make_git_runner(tmp_path)

    git("init", "-q")

    # Root commit: write hello.py with 5 lines
    hello = tmp_path / "hello.py"
    hello.write_text(
        "line1\n"
        "line2\n"
        "line3\n"
        "line4\n"
        "line5\n"
    )
    git("add", "hello.py")
    git("commit", "-m", "initial commit", "--quiet", date="2024-01-01T00:00:00")

    # Capture root SHA
    root_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=tmp_path, capture_output=True, text=True, check=True,
    ).stdout.strip()

    # Second commit: modify lines 2-3
    hello.write_text(
        "line1\n"
        "modified line2\n"
        "modified line3\n"
        "line4\n"
        "line5\n"
    )
    git("add", "hello.py")
    git("commit", "-m", "modify lines 2-3", "--quiet", date="2024-01-02T00:00:00")

    second_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=tmp_path, capture_output=True, text=True, check=True,
    ).stdout.strip()

    return tmp_path, root_sha, second_sha


def test_integration_whole_file_diff_contains_plus_lines(
    diff_repo: tuple[Path, str, str],
) -> None:
    """Whole-file diff on the second commit contains lines beginning with '+'."""
    repo, _root_sha, second_sha = diff_repo
    result = get_commit_diff(
        sha=second_sha, file=Path("hello.py"), line_range=None, repo=repo
    )
    assert "+" in result


def test_integration_line_range_diff_returns_nonempty(
    diff_repo: tuple[Path, str, str],
) -> None:
    """Line-range diff on the second commit for lines 2-3 returns a non-empty string."""
    repo, _root_sha, second_sha = diff_repo
    result = get_commit_diff(
        sha=second_sha, file=Path("hello.py"), line_range=(2, 3), repo=repo
    )
    assert len(result) > 0


def test_integration_root_commit_fallback_does_not_raise(
    diff_repo: tuple[Path, str, str],
) -> None:
    """Calling with line_range on the root commit triggers fallback and returns non-empty output."""
    repo, root_sha, _second_sha = diff_repo
    # Root commits have no parent; git log -L <sha>^! will fail — expect fallback
    result = get_commit_diff(
        sha=root_sha, file=Path("hello.py"), line_range=(1, 2), repo=repo
    )
    assert len(result) > 0
