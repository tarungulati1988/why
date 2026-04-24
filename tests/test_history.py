"""Tests for why.history: get_file_history, get_line_history, and find_introduction."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from conftest import _tmp_path_is_clean, make_git_runner

from why.git import GitError
from why.history import find_introduction, get_file_history, get_line_history

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

    # Delegate to the shared factory — eliminates duplicated base_env boilerplate.
    git = make_git_runner(repo)

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


# ---------------------------------------------------------------------------
# get_line_history — unit tests
# ---------------------------------------------------------------------------

class TestLineHistoryArgs:
    """Verify get_line_history builds the correct git args."""

    def test_args_contain_L_flag(self) -> None:
        """-L<line>,<line>:<rel> must appear in the args passed to run_git."""
        fake_repo = Path("/fake/repo")
        target_file = fake_repo / "foo.py"
        line = 3

        with (
            patch("why.history.run_git", return_value="") as mock_run_git,
            patch("why.history.parse_porcelain", return_value=[]),
        ):
            get_line_history(target_file, fake_repo, line=line)

        args_passed = mock_run_git.call_args.args[0]
        rel = target_file.relative_to(fake_repo)
        assert f"-L{line},{line}:{rel}" in args_passed

    def test_args_contain_no_patch(self) -> None:
        """--no-patch must appear in args (shortstat is skipped for line history)."""
        fake_repo = Path("/fake/repo")
        target_file = fake_repo / "foo.py"

        with (
            patch("why.history.run_git", return_value="") as mock_run_git,
            patch("why.history.parse_porcelain", return_value=[]),
        ):
            get_line_history(target_file, fake_repo, line=1)

        args_passed = mock_run_git.call_args.args[0]
        assert "--no-patch" in args_passed

    def test_args_do_not_contain_follow(self) -> None:
        """--follow must NOT appear in args — line history doesn't traverse renames."""
        fake_repo = Path("/fake/repo")
        target_file = fake_repo / "foo.py"

        with (
            patch("why.history.run_git", return_value="") as mock_run_git,
            patch("why.history.parse_porcelain", return_value=[]),
        ):
            get_line_history(target_file, fake_repo, line=1)

        args_passed = mock_run_git.call_args.args[0]
        assert "--follow" not in args_passed


class TestLineHistoryOutOfBounds:
    """Verify out-of-bounds line number returns [] instead of raising."""

    def test_out_of_bounds_returns_empty(self) -> None:
        """GitError with 'has only' in the message should return []."""
        fake_repo = Path("/fake/repo")
        target_file = fake_repo / "foo.py"

        with patch(
            "why.history.run_git",
            side_effect=GitError("file foo.py has only 5 lines"),
        ):
            result = get_line_history(target_file, fake_repo, line=99)

        assert result == []

    def test_other_git_error_re_raises(self) -> None:
        """GitError without 'has only' must propagate unchanged."""
        fake_repo = Path("/fake/repo")
        target_file = fake_repo / "foo.py"

        with (
            patch(
                "why.history.run_git",
                side_effect=GitError("some other error"),
            ),
            pytest.raises(GitError, match="some other error"),
        ):
            get_line_history(target_file, fake_repo, line=1)


# ---------------------------------------------------------------------------
# get_line_history — integration tests
# ---------------------------------------------------------------------------


@pytest.fixture
def line_history_repo(tmp_path: Path) -> Path:
    """Build a real git repo where 3 commits each touch line 1 of foo.py.

    History (oldest to newest):
      A — write "line1\\nline2\\nline3\\n" to foo.py
      B — change line 1 to "line1 updated"
      C — change line 1 to "line1 final"

    Returns the repo root Path.
    """
    if not _tmp_path_is_clean(tmp_path):
        pytest.skip("tmp_path is inside an existing git repo — skipping integration test")

    repo = tmp_path / "repo"
    repo.mkdir()

    # Delegate to the shared factory — eliminates duplicated base_env boilerplate.
    git = make_git_runner(repo)

    git("init")

    # Commit A: create foo.py with 3 lines
    (repo / "foo.py").write_text("line1\nline2\nline3\n")
    git("add", "foo.py")
    git("commit", "-m", "A: create foo.py")

    # Commit B: update line 1
    (repo / "foo.py").write_text("line1 updated\nline2\nline3\n")
    git("add", "foo.py")
    git("commit", "-m", "B: update line 1")

    # Commit C: update line 1 again
    (repo / "foo.py").write_text("line1 final\nline2\nline3\n")
    git("add", "foo.py")
    git("commit", "-m", "C: finalize line 1")

    return repo


class TestLineHistoryGuards:
    """Verify get_line_history input validation."""

    def test_line_zero_raises_value_error(self) -> None:
        """line=0 must raise ValueError before any git call."""
        fake_repo = Path("/fake/repo")
        target_file = fake_repo / "foo.py"
        with pytest.raises(ValueError, match="line must be >= 1"):
            get_line_history(target_file, fake_repo, line=0)

    def test_negative_line_raises_value_error(self) -> None:
        """Negative line must raise ValueError before any git call."""
        fake_repo = Path("/fake/repo")
        target_file = fake_repo / "foo.py"
        with pytest.raises(ValueError, match="line must be >= 1"):
            get_line_history(target_file, fake_repo, line=-5)

    def test_file_outside_repo_raises_value_error(self, tmp_path: Path) -> None:
        """A file outside the repo root must raise ValueError."""
        repo = tmp_path / "repo"
        repo.mkdir()
        outside_file = tmp_path / "outside.py"
        outside_file.write_text("x\n")
        with pytest.raises(ValueError, match="escapes repository root"):
            get_line_history(outside_file, repo, line=1)


class TestLineHistoryIntegration:
    """Integration: get_line_history runs real git and returns correct commits."""

    def test_returns_all_commits_touching_line_1(self, line_history_repo: Path) -> None:
        """All 3 commits modify line 1 — all 3 must be returned."""
        commits = get_line_history(
            line_history_repo / "foo.py", line_history_repo, line=1
        )
        assert len(commits) == 3

    def test_line_out_of_bounds_returns_empty(self, line_history_repo: Path) -> None:
        """Requesting a line beyond the file's length must return []."""
        result = get_line_history(
            line_history_repo / "foo.py", line_history_repo, line=999
        )
        assert result == []

    def test_commits_are_newest_first(self, line_history_repo: Path) -> None:
        """get_line_history returns commits newest-first (matching git log default)."""
        commits = get_line_history(
            line_history_repo / "foo.py", line_history_repo, line=1
        )
        assert commits[0].subject == "C: finalize line 1"
        assert commits[2].subject == "A: create foo.py"


# ---------------------------------------------------------------------------
# find_introduction — unit tests
# ---------------------------------------------------------------------------


class TestFindIntroductionArgs:
    """Verify find_introduction builds the correct git args."""

    def test_args_contain_diff_filter_A(self) -> None:
        """--diff-filter=A must appear in args forwarded to run_git."""
        fake_repo = Path("/fake/repo")
        target_file = fake_repo / "foo.py"

        with (
            patch("why.history.run_git", return_value="raw") as mock_run_git,
            patch("why.history.parse_porcelain", return_value=[MagicMock()]),
        ):
            find_introduction(target_file, fake_repo)

        args_passed = mock_run_git.call_args.args[0]
        assert "--diff-filter=A" in args_passed

    def test_args_contain_follow(self) -> None:
        """--follow must appear in args."""
        fake_repo = Path("/fake/repo")
        target_file = fake_repo / "foo.py"

        with (
            patch("why.history.run_git", return_value="raw") as mock_run_git,
            patch("why.history.parse_porcelain", return_value=[MagicMock()]),
        ):
            find_introduction(target_file, fake_repo)

        args_passed = mock_run_git.call_args.args[0]
        assert "--follow" in args_passed

    def test_empty_output_returns_none(self) -> None:
        """Empty git output must return None without calling parse_porcelain."""
        fake_repo = Path("/fake/repo")
        target_file = fake_repo / "foo.py"

        with (
            patch("why.history.run_git", return_value="  "),
            patch("why.history.parse_porcelain") as mock_parse,
        ):
            result = find_introduction(target_file, fake_repo)

        assert result is None
        mock_parse.assert_not_called()

    def test_returns_oldest_commit_when_multiple_returned(self) -> None:
        """When parse_porcelain returns multiple commits, the oldest (last) is returned."""
        fake_repo = Path("/fake/repo")
        target_file = fake_repo / "foo.py"
        older = MagicMock(name="older_commit")
        newer = MagicMock(name="newer_commit")
        # parse_porcelain returns newest-first, so newer is index 0, older is index -1
        mock_commits = [newer, older]

        with (
            patch("why.history.run_git", return_value="raw"),
            patch("why.history.parse_porcelain", return_value=mock_commits),
        ):
            result = find_introduction(target_file, fake_repo)

        assert result is older


# ---------------------------------------------------------------------------
# find_introduction — integration tests
# ---------------------------------------------------------------------------


class TestFindIntroductionIntegration:
    """Integration: find_introduction runs real git and identifies the correct commit."""

    def test_introduction_of_renamed_file_is_commit_A(self, renamed_repo: Path) -> None:
        """new_name.py was introduced as old_name.py in commit A — must return commit A."""
        # renamed_repo history: A creates old_name.py, B renames it to new_name.py, C edits it
        commit = find_introduction(renamed_repo / "new_name.py", renamed_repo)
        assert commit is not None
        assert commit.subject == "A: create old_name.py and other.py"

    def test_untracked_file_returns_none(self, renamed_repo: Path) -> None:
        """A file with no git history must return None."""
        ghost = renamed_repo / "never_committed.py"
        result = find_introduction(ghost, renamed_repo)
        assert result is None
