"""Tests for the why.commit dataclass and porcelain parser."""

from __future__ import annotations

import subprocess
from datetime import datetime
from pathlib import Path

import pytest
from conftest import _tmp_path_is_clean

from why.commit import PORCELAIN_FORMAT, SHORTSTAT_FLAG, parse_porcelain

# ---------------------------------------------------------------------------
# Helpers — build raw porcelain output strings
# ---------------------------------------------------------------------------


def _make_chunk(
    sha: str = "a" * 40,
    author_name: str = "Alice",
    author_email: str = "alice@example.com",
    date: str = "2024-01-15T10:30:00+00:00",
    subject: str = "fix: resolve the thing",
    body: str = "Detailed explanation.\n\nMore info.",
    parents: str = "b" * 40,
    shortstat: str = "",
) -> str:
    """Build a single porcelain chunk (fields joined by NUL, terminated by RS).

    ``shortstat`` is appended after the RS separator, mirroring real git output
    where --shortstat emits a blank line + stats line *after* the format block.
    """
    fields = "\x00".join([sha, author_name, author_email, date, subject, body, parents])
    # The \x1e (RS) is the commit separator; shortstat appears after it on
    # the next chunk's leading whitespace (we model the *next* chunk's prefix).
    chunk = fields + "\x1e"
    if shortstat:
        chunk += shortstat
    return chunk


# ---------------------------------------------------------------------------
# Unit tests — fixture strings with \x00 and \x1e built explicitly
# ---------------------------------------------------------------------------


def test_parses_normal_commit() -> None:
    """One commit with multi-line body, single parent, shortstat with additions + deletions."""
    sha = "a" * 40
    body = "First paragraph.\n\nSecond paragraph."
    parent = "b" * 40
    shortstat = "\n 3 files changed, 10 insertions(+), 4 deletions(-)\n"

    # Build: <fields>\x1e\n<shortstat> — the shortstat trails after \x1e, and the
    # next chunk would lstrip the leading \n.  We end with a trailing \x1e so the
    # split produces a trailing empty chunk that parse_porcelain must drop.
    output = (
        "\x00".join([sha, "Alice", "alice@example.com", "2024-01-15T10:30:00+00:00",
                     "fix: resolve the thing", body, parent])
        + "\x1e"
        + shortstat
        + "\x1e"  # trailing separator → empty chunk
    )

    commits = parse_porcelain(output)
    assert len(commits) == 1
    c = commits[0]

    assert c.sha == sha
    assert c.short_sha == sha[:7]
    assert c.author_name == "Alice"
    assert c.author_email == "alice@example.com"
    assert isinstance(c.date, datetime)
    assert c.subject == "fix: resolve the thing"
    assert c.body == body
    assert c.parents == (parent,)
    assert c.is_merge is False
    assert c.additions == 10
    assert c.deletions == 4


def test_parses_merge_commit() -> None:
    """Merge commit has two parents, no shortstat line. additions and deletions stay 0."""
    sha = "c" * 40
    parent1 = "d" * 40
    parent2 = "e" * 40

    output = (
        "\x00".join([sha, "Bob", "bob@example.com", "2024-02-01T09:00:00+00:00",
                     "Merge branch 'feat/x'", "", f"{parent1} {parent2}"])
        + "\x1e"
    )

    commits = parse_porcelain(output)
    assert len(commits) == 1
    c = commits[0]

    assert c.sha == sha
    assert c.is_merge is True
    assert c.parents == (parent1, parent2)
    assert c.additions == 0
    assert c.deletions == 0


def test_parses_empty_body() -> None:
    """Body field is an empty string between \x00 separators — must round-trip as ''."""
    sha = "f" * 40
    parent = "0" * 40

    output = (
        "\x00".join([sha, "Carol", "carol@example.com", "2024-03-10T12:00:00-05:00",
                     "chore: bump version", "", parent])
        + "\x1e"
    )

    commits = parse_porcelain(output)
    assert len(commits) == 1
    assert commits[0].body == ""


def test_parses_special_chars_in_subject_and_body() -> None:
    """Tabs, quotes, pipes, and a newline in body must survive parsing verbatim."""
    sha = "1" * 40
    parent = "2" * 40
    subject = 'fix: handle "special" chars | tabs\there'
    body = "line one\ttabbed\nline two with 'quotes' and | pipe"

    output = (
        "\x00".join([sha, "Dave", "dave@example.com", "2024-04-20T08:15:00+02:00",
                     subject, body, parent])
        + "\x1e"
    )

    commits = parse_porcelain(output)
    assert len(commits) == 1
    assert commits[0].subject == subject
    assert commits[0].body == body


def test_parses_only_insertions() -> None:
    """Shortstat with only insertions — deletions must be 0."""
    sha = "3" * 40
    parent = "4" * 40
    shortstat = "\n 1 file changed, 5 insertions(+)\n"

    output = (
        "\x00".join([sha, "Eve", "eve@example.com", "2024-05-01T00:00:00+00:00",
                     "add: new feature", "", parent])
        + "\x1e"
        + shortstat
        + "\x1e"
    )

    commits = parse_porcelain(output)
    assert len(commits) == 1
    assert commits[0].additions == 5
    assert commits[0].deletions == 0


def test_parses_only_deletions() -> None:
    """Shortstat with only deletions — additions must be 0."""
    sha = "5" * 40
    parent = "6" * 40
    shortstat = "\n 1 file changed, 3 deletions(-)\n"

    output = (
        "\x00".join([sha, "Frank", "frank@example.com", "2024-06-15T14:30:00+00:00",
                     "remove: dead code", "", parent])
        + "\x1e"
        + shortstat
        + "\x1e"
    )

    commits = parse_porcelain(output)
    assert len(commits) == 1
    assert commits[0].additions == 0
    assert commits[0].deletions == 3


def test_malformed_chunk_raises_value_error() -> None:
    """A chunk with 5 fields instead of 7 must raise ValueError."""
    sha = "7" * 40
    # Only 5 fields (missing parents and body)
    bad_output = "\x00".join([sha, "Grace", "grace@example.com", "2024-07-01T10:00:00+00:00",
                               "bad commit"]) + "\x1e"

    with pytest.raises(ValueError, match="malformed commit chunk"):
        parse_porcelain(bad_output)


def test_parses_multiple_commits() -> None:
    """Two commits in one output string are both parsed correctly."""
    sha1 = "a" * 40
    sha2 = "b" * 40
    parent1 = "c" * 40
    parent2 = sha1  # second commit's parent is the first

    output = (
        "\x00".join([sha1, "Alice", "a@example.com", "2024-01-01T00:00:00+00:00",
                     "first commit", "body one", parent1])
        + "\x1e"
        + "\x00".join([sha2, "Bob", "b@example.com", "2024-01-02T00:00:00+00:00",
                       "second commit", "body two", parent2])
        + "\x1e"
    )

    commits = parse_porcelain(output)
    assert len(commits) == 2
    assert commits[0].sha == sha1
    assert commits[1].sha == sha2
    assert commits[0].subject == "first commit"
    assert commits[1].subject == "second commit"


def test_empty_output_returns_empty_list() -> None:
    """parse_porcelain('') must return an empty list, not raise."""
    result = parse_porcelain("")
    assert result == []


def test_body_line_matching_shortstat_regex_is_not_misparsed() -> None:
    """A body line that looks like a shortstat must NOT be treated as one.

    The body contains '3 files changed today had this problem' — the regex
    _SHORTSTAT_LINE_RE matches this line when searching the whole chunk, but it
    should only search the region before the first NUL (the shortstat prefix area).
    """
    sha = "a" * 40
    parent = "b" * 40
    # Body contains a line that superficially matches the shortstat regex.
    tricky_body = "Fixed the issue\n 3 files changed today had this problem"

    output = (
        "\x00".join([
            sha, "Alice", "alice@example.com", "2024-01-15T10:30:00+00:00",
            "fix: resolve the thing", tricky_body, parent,
        ])
        + "\x1e"
    )

    commits = parse_porcelain(output)
    assert len(commits) == 1
    c = commits[0]
    # Body must round-trip verbatim — no false shortstat extraction.
    assert c.body == tricky_body
    # No stats should have been detected.
    assert c.additions == 0
    assert c.deletions == 0


def test_parses_root_commit() -> None:
    """A root commit has an empty parents field — parents must be () and is_merge False."""
    sha = "9" * 40

    output = (
        "\x00".join([
            sha, "Alice", "alice@example.com", "2024-01-01T00:00:00+00:00",
            "initial commit", "", "",  # empty parents field
        ])
        + "\x1e"
    )

    commits = parse_porcelain(output)
    assert len(commits) == 1
    c = commits[0]
    assert c.parents == ()
    assert c.is_merge is False


# ---------------------------------------------------------------------------
# Integration tests — real git process
# ---------------------------------------------------------------------------


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """Create a minimal real git repo in tmp_path and return the path.

    Layout:
      - initial commit (adds file.txt with 5 lines)
      - edit commit on main (modifies file.txt — adds + removes lines)
      - feature branch created from initial, then merged into main (merge commit)
    """
    if not _tmp_path_is_clean(tmp_path):
        pytest.skip("tmp_path is inside an existing git repo; isolation required")

    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.email", "test@example.com")
    git("config", "user.name", "Test User")

    # Initial commit: add a file with content
    (tmp_path / "file.txt").write_text("line1\nline2\nline3\nline4\nline5\n")
    git("add", ".")
    git("commit", "--quiet", "-m", "initial: add file.txt")

    # Detect the default branch name (git >= 2.28 uses "main" when configured,
    # older versions default to "master"). We capture stdout to read it back.
    default_branch = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=tmp_path, check=True, capture_output=True, text=True,
    ).stdout.strip()

    # Feature branch: branch off initial, add a new file
    git("checkout", "-q", "-b", "feat/branch")
    (tmp_path / "feature.txt").write_text("feature content\n")
    git("add", ".")
    git("commit", "--quiet", "-m", "feat: add feature.txt")

    # Back to the default branch, make an edit commit that adds and removes lines
    git("checkout", "-q", default_branch)
    (tmp_path / "file.txt").write_text("line1\nline2\nline3\nLINE4_EDITED\nline5\nextra\n")
    git("add", ".")
    git("commit", "--quiet", "-m", "edit: modify file.txt")

    # Merge the feature branch — creates a merge commit
    git("merge", "--no-ff", "--quiet", "-m", "Merge branch 'feat/branch'", "feat/branch")

    return tmp_path


def test_integration_round_trip(git_repo: Path) -> None:
    """Full round-trip: real git log → parse_porcelain → structured Commit objects.

    Asserts:
    - Correct commit count (4: merge + edit + feat + initial)
    - Exactly one merge commit
    - SHAs match git rev-list HEAD
    - The edit commit has nonzero additions and deletions
    """
    # Run git log with the porcelain format + shortstat
    result = subprocess.run(
        ["git", "log", f"--format={PORCELAIN_FORMAT}", SHORTSTAT_FLAG],
        cwd=git_repo,
        capture_output=True,
        text=True,
        check=True,
    )
    raw = result.stdout

    commits = parse_porcelain(raw)

    # Verify count: merge, edit, feat, initial = 4 commits
    assert len(commits) == 4

    # Get the canonical SHA order from git rev-list
    rev_list = subprocess.run(
        ["git", "rev-list", "HEAD"],
        cwd=git_repo,
        capture_output=True,
        text=True,
        check=True,
    )
    expected_shas = rev_list.stdout.strip().splitlines()
    parsed_shas = [c.sha for c in commits]
    assert parsed_shas == expected_shas

    # Exactly one merge commit
    merge_commits = [c for c in commits if c.is_merge]
    assert len(merge_commits) == 1

    # Find the edit commit by subject
    edit_commit = next(c for c in commits if "edit" in c.subject)
    assert edit_commit.additions > 0
    assert edit_commit.deletions > 0

    # short_sha is correct for all commits
    for c in commits:
        assert c.short_sha == c.sha[:7]
        assert isinstance(c.date, datetime)
