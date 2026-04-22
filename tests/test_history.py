"""Tests for why.history.get_file_history."""

from __future__ import annotations

import os
import subprocess
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from conftest import _tmp_path_is_clean

from why.history import get_file_history

# ---------------------------------------------------------------------------
# Unit tests — run_git and parse_porcelain are mocked
# ---------------------------------------------------------------------------


class TestSinceFilter:
    """Verify that the --since flag is added iff `since` is provided."""

    def test_since_is_included_when_provided(self) -> None:
        """--since=<iso> must appear in the args forwarded to run_git."""
        since_dt = datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC)
        mock_commits = [MagicMock()]
        fake_repo = Path("/fake/repo")

        with (
            patch("why.history.run_git", return_value="raw") as mock_run_git,
            patch("why.history.parse_porcelain", return_value=mock_commits) as mock_parse,
        ):
            result = get_file_history(fake_repo / "foo.py", fake_repo, since=since_dt)

        mock_run_git.assert_called_once()
        args_passed = mock_run_git.call_args.args[0]

        assert "--follow" in args_passed
        assert "--" in args_passed
        assert f"--since={since_dt.isoformat()}" in args_passed
        mock_parse.assert_called_once_with("raw")
        assert result == mock_commits

    def test_since_is_absent_when_not_provided(self) -> None:
        """No --since flag must appear in args when since=None."""
        fake_repo = Path("/fake/repo")
        with (
            patch("why.history.run_git", return_value="raw") as mock_run_git,
            patch("why.history.parse_porcelain", return_value=[]),
        ):
            get_file_history(fake_repo / "foo.py", fake_repo)

        args_passed = mock_run_git.call_args.args[0]

        assert "--follow" in args_passed
        assert "--" in args_passed
        # No element should start with --since
        since_args = [a for a in args_passed if a.startswith("--since")]
        assert since_args == []

    def test_file_path_appears_after_double_dash(self) -> None:
        """The file path must come after '--' in the args list."""
        fake_repo = Path("/fake/repo")
        target_file = fake_repo / "bar.py"

        with (
            patch("why.history.run_git", return_value="") as mock_run_git,
            patch("why.history.parse_porcelain", return_value=[]),
        ):
            get_file_history(target_file, fake_repo)

        args_passed = mock_run_git.call_args.args[0]
        sep_index = args_passed.index("--")
        # File path string must appear after the '--' separator
        assert str(target_file) in args_passed[sep_index + 1 :]


# ---------------------------------------------------------------------------
# Integration tests — real git subprocess
# ---------------------------------------------------------------------------


@pytest.fixture
def renamed_repo(tmp_path: Path) -> Path:
    """Build a real git repo with a rename in its history.

    History (oldest to newest), with explicit timestamps spaced 1 hour apart
    so the since-filter test has unambiguous temporal boundaries:
      A — 2024-01-01T10:00:00+00:00 — create old_name.py and other.py
      B — 2024-01-01T11:00:00+00:00 — rename old_name.py -> new_name.py; edit other.py
      C — 2024-01-01T12:00:00+00:00 — edit new_name.py after rename

    Returns the repo root Path.
    """
    if not _tmp_path_is_clean(tmp_path):
        pytest.skip("tmp_path is inside an existing git repo — skipping integration test")

    repo = tmp_path / "repo"
    repo.mkdir()

    base_env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "Test",
        "GIT_AUTHOR_EMAIL": "test@example.com",
        "GIT_COMMITTER_NAME": "Test",
        "GIT_COMMITTER_EMAIL": "test@example.com",
        "GIT_TERMINAL_PROMPT": "0",
    }

    def git(*args: str, date: str | None = None) -> None:
        env = {**base_env}
        if date is not None:
            # Override both author and committer timestamps so the commit is
            # recorded at a deterministic time regardless of when the test runs.
            env["GIT_AUTHOR_DATE"] = date
            env["GIT_COMMITTER_DATE"] = date
        subprocess.run(
            ["git", *args],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )

    git("init")

    # Commit A: create old_name.py and other.py
    (repo / "old_name.py").write_text("version 1\n")
    (repo / "other.py").write_text("sibling\n")
    git("add", "old_name.py", "other.py")
    # Use a non-UTC offset (-05:00) — git 2.47+ emits "Z" for UTC timestamps,
    # which Python 3.9's datetime.fromisoformat cannot parse.  A non-UTC
    # offset is preserved verbatim in git's %aI output and roundtrips safely.
    git("commit", "-m", "A: create old_name.py and other.py", date="2024-01-01T10:00:00 -0500")

    # Commit B: rename old_name.py -> new_name.py; also edit other.py
    git("mv", "old_name.py", "new_name.py")
    (repo / "other.py").write_text("sibling edited\n")
    git("add", "other.py")
    git("commit", "-m", "B: rename to new_name.py, edit other.py", date="2024-01-01T11:00:00 -0500")

    # Commit C: edit new_name.py after rename
    (repo / "new_name.py").write_text("version 2\n")
    git("add", "new_name.py")
    git("commit", "-m", "C: edit new_name.py post-rename", date="2024-01-01T12:00:00 -0500")

    return repo


class TestRenameFollowed:
    """Integration: --follow causes rename boundary to be traversed."""

    def test_returns_three_commits_across_rename(self, renamed_repo: Path) -> None:
        """get_file_history must return all 3 commits touching new_name.py (via rename)."""
        commits = get_file_history(renamed_repo / "new_name.py", renamed_repo)
        assert len(commits) == 3

    def test_commits_are_newest_first(self, renamed_repo: Path) -> None:
        """git log returns newest commit first by default."""
        commits = get_file_history(renamed_repo / "new_name.py", renamed_repo)
        assert commits[0].subject == "C: edit new_name.py post-rename"
        assert commits[2].subject == "A: create old_name.py and other.py"

    def test_since_filter_excludes_earliest_commit(self, renamed_repo: Path) -> None:
        """Passing since between commit A and B should return only B and C.

        The fixture pins A at 10:00-05:00, B at 11:00-05:00, C at 12:00-05:00.
        Using 10:30-05:00 as the boundary reliably excludes A only.
        """
        tz_minus5 = timezone(timedelta(hours=-5))
        since = datetime(2024, 1, 1, 10, 30, 0, tzinfo=tz_minus5)
        filtered = get_file_history(
            renamed_repo / "new_name.py", renamed_repo, since=since
        )
        assert len(filtered) == 2
        subjects = {c.subject for c in filtered}
        assert "B: rename to new_name.py, edit other.py" in subjects
        assert "C: edit new_name.py post-rename" in subjects


class TestNoHistory:
    """get_file_history returns [] for a file with no commits."""

    def test_nonexistent_file_returns_empty(self, renamed_repo: Path) -> None:
        """A path that was never committed should produce an empty list."""
        ghost = renamed_repo / "never_existed.py"
        result = get_file_history(ghost, renamed_repo)
        assert result == []
