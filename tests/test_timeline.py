"""Tests for why.timeline — validate_and_repair_timeline and _render_deterministic_timeline.

Written BEFORE implementation (TDD). Covers: empty list, single commit without PR,
single commit with PR, missing section appended, valid SHA passes through,
hallucinated SHA replaced, no SHAs in block passes through.
"""

from __future__ import annotations

from datetime import UTC, datetime

from why.commit import Commit
from why.prompts import CommitWithPR
from why.timeline import (
    render_deterministic_timeline as _render_deterministic_timeline,
)
from why.timeline import (
    validate_and_repair_timeline,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

FIXED_DATE = datetime(2026, 1, 15, tzinfo=UTC)

FIXED_COMMIT = Commit(
    sha="abc1234def5678901234567890",
    author_name="Jane Smith",
    author_email="jane@example.com",
    date=FIXED_DATE,
    subject="fix: handle null token",
    body="",
    parents=(),
    additions=3,
    deletions=1,
)

FIXED_CWPR = CommitWithPR(commit=FIXED_COMMIT)


# ---------------------------------------------------------------------------
# _render_deterministic_timeline tests
# ---------------------------------------------------------------------------


def test_deterministic_empty() -> None:
    """Empty key_commits list produces 'No commit history available' message."""
    result = _render_deterministic_timeline([])
    assert "No commit history available" in result
    # Should not contain a fenced code block
    assert "```" not in result


def test_deterministic_single_no_pr() -> None:
    """Single commit with no PR body produces row with SHA and date."""
    result = _render_deterministic_timeline([FIXED_CWPR])
    assert "abc1234" in result
    assert "2026-01-15" in result
    # No PR annotation expected
    assert "[PR #" not in result
    # Should be wrapped in a fenced block
    assert "```text" in result


def test_deterministic_single_with_pr() -> None:
    """Single commit whose pr_body contains 'PR #42' gets [PR #42] appended to row."""
    cwpr_with_pr = CommitWithPR(commit=FIXED_COMMIT, pr_body="PR #42 description")
    result = _render_deterministic_timeline([cwpr_with_pr])
    assert "[PR #42]" in result
    assert "abc1234" in result


def test_deterministic_subject_newlines_stripped() -> None:
    """Newlines in commit subject are replaced with spaces in the rendered row."""
    commit_with_newline = Commit(
        sha="abc1234def5678901234567890",
        author_name="Jane Smith",
        author_email="jane@example.com",
        date=FIXED_DATE,
        subject="fix: handle\nnull token",
        body="",
        parents=(),
        additions=3,
        deletions=1,
    )
    cwpr = CommitWithPR(commit=commit_with_newline)
    result = _render_deterministic_timeline([cwpr])
    # Newline in subject should be replaced with a space
    assert "fix: handle null token" in result


# ---------------------------------------------------------------------------
# validate_and_repair_timeline tests
# ---------------------------------------------------------------------------


def test_validate_missing_section_appends() -> None:
    """Response with no timeline section gets timeline appended to the end."""
    response = "## 🏗️ How it started\n\nSome narrative here."
    result = validate_and_repair_timeline(response, [FIXED_CWPR])
    expected = "## 📊 Timeline\n\n```text\n2026-01-15  abc1234  fix: handle null token\n```"
    assert result.endswith(expected)


def test_validate_valid_sha_passes_through() -> None:
    """Response whose timeline block contains only known SHAs is returned unchanged."""
    response = (
        "## 🏗️ Some section\n\nNarrative.\n\n"
        "## 📊 Timeline\n\n```text\n2026-01-15  abc1234  fix\n```"
    )
    result = validate_and_repair_timeline(response, [FIXED_CWPR])
    assert result == response


def test_validate_hallucinated_sha_replaced() -> None:
    """Timeline block containing an unknown SHA is replaced with deterministic output."""
    response = (
        "## 🏗️ Some section\n\nNarrative.\n\n"
        "## 📊 Timeline\n\n```text\n2026-01-15  deadbee  hallucinated commit\n```"
    )
    result = validate_and_repair_timeline(response, [FIXED_CWPR])
    # The hallucinated SHA should be gone
    assert "deadbee" not in result
    # The real SHA should be present
    assert "abc1234" in result
    # The leading narrative section should be preserved
    assert "## 🏗️ Some section" in result


def test_validate_no_shas_in_block_passes_through() -> None:
    """Timeline block that contains no hex tokens passes through unchanged."""
    response = (
        "## 🏗️ Some section\n\nNarrative.\n\n"
        "## 📊 Timeline\n\nNo commit history available."
    )
    result = validate_and_repair_timeline(response, [FIXED_CWPR])
    assert result == response


def test_validate_empty_key_commits_missing_section() -> None:
    """When key_commits is empty and section missing, appends 'No commit history' message."""
    response = "## 🏗️ Some section\n\nNarrative."
    result = validate_and_repair_timeline(response, [])
    assert "No commit history available" in result
    assert "## 📊 Timeline" in result
