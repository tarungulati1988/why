"""Tests for why.citations.validate_citations — written BEFORE implementation (TDD).

Covers: clean output, unknown SHA, unknown PR, multiple issues, prefix matching,
strict=True raises, strict=False returns list, empty output, SHA too short.
"""

from __future__ import annotations

import pytest

from why.citations import CitationError, validate_citations

# ---------------------------------------------------------------------------
# Shared golden fixtures
# ---------------------------------------------------------------------------

# A realistic 40-char SHA that we treat as "known"
KNOWN_SHA_FULL = "a3f8c1d2e4b6f7a8b9c0d1e2f3a4b5c6d7e8f9a0"
# A second known SHA for multi-issue tests
KNOWN_SHA_2 = "b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0"

# 7-char prefix of KNOWN_SHA_FULL — valid short reference
KNOWN_SHA_PREFIX = KNOWN_SHA_FULL[:7]  # "a3f8c1d"

# A SHA that does NOT appear in any known set
UNKNOWN_SHA = "deadbeef1234abc"  # 15 hex chars, not a prefix of known

# Known PR numbers
KNOWN_PRS: set[int] = {42, 101}

# An unknown PR number
UNKNOWN_PR = 999


# ---------------------------------------------------------------------------
# 1. Clean output — known SHAs and PRs only → empty list
# ---------------------------------------------------------------------------

def test_clean_output_returns_no_issues() -> None:
    """Output referencing only known SHAs and PRs produces no issues."""
    output = (
        f"Commit {KNOWN_SHA_PREFIX} fixed the bug introduced in PR #42.\n"
        f"See also #101 and {KNOWN_SHA_2[:8]}."
    )
    issues = validate_citations(output, {KNOWN_SHA_FULL, KNOWN_SHA_2}, KNOWN_PRS)
    assert issues == []


# ---------------------------------------------------------------------------
# 2. Unknown SHA → one ValidationIssue(kind='unknown_sha')
# ---------------------------------------------------------------------------

def test_unknown_sha_flagged() -> None:
    """A SHA not matching any known SHA is returned as an unknown_sha issue."""
    line = f"See commit {UNKNOWN_SHA} for context."
    issues = validate_citations(line, {KNOWN_SHA_FULL}, set())
    assert len(issues) == 1
    issue = issues[0]
    assert issue.kind == "unknown_sha"
    assert issue.value == UNKNOWN_SHA
    assert line in issue.output_line


# ---------------------------------------------------------------------------
# 3. Unknown PR → one ValidationIssue(kind='unknown_pr')
# ---------------------------------------------------------------------------

def test_unknown_pr_flagged() -> None:
    """A PR number not in known_prs is returned as an unknown_pr issue."""
    line = f"Fixed in #{UNKNOWN_PR}."
    issues = validate_citations(line, set(), KNOWN_PRS)
    assert len(issues) == 1
    issue = issues[0]
    assert issue.kind == "unknown_pr"
    assert issue.value == str(UNKNOWN_PR)
    assert line in issue.output_line


# ---------------------------------------------------------------------------
# 4. Multiple issues — unknown SHA + unknown PR in same output
# ---------------------------------------------------------------------------

def test_multiple_issues_returned() -> None:
    """Both unknown SHA and unknown PR in the output produce two issues."""
    output = f"Commit {UNKNOWN_SHA} was reverted via #{UNKNOWN_PR}."
    issues = validate_citations(output, {KNOWN_SHA_FULL}, KNOWN_PRS)
    assert len(issues) == 2
    kinds = {i.kind for i in issues}
    assert kinds == {"unknown_sha", "unknown_pr"}


# ---------------------------------------------------------------------------
# 5. Prefix matching — short SHA that is a prefix of a known SHA → not flagged
# ---------------------------------------------------------------------------

def test_prefix_sha_not_flagged() -> None:
    """A 7-char prefix of a known 40-char SHA is valid and should not be flagged."""
    output = f"See {KNOWN_SHA_PREFIX} for the fix."
    issues = validate_citations(output, {KNOWN_SHA_FULL}, set())
    assert issues == []


def test_full_sha_prefix_of_short_known_sha_not_flagged() -> None:
    """Bidirectional: known set has short SHA, output has longer prefix — not flagged."""
    short_known = KNOWN_SHA_FULL[:8]  # e.g. "a3f8c1d2"
    output = f"Commit {KNOWN_SHA_FULL} landed here."
    # known_shas holds only the short version; full SHA startswith short → valid
    issues = validate_citations(output, {short_known}, set())
    assert issues == []


# ---------------------------------------------------------------------------
# 6. strict=True raises ValueError when issues exist
# ---------------------------------------------------------------------------

def test_strict_true_raises_on_issues() -> None:
    """strict=True raises ValueError when there are citation issues."""
    output = f"Commit {UNKNOWN_SHA} is suspicious."
    with pytest.raises(ValueError, match="citation validation failed"):
        validate_citations(output, {KNOWN_SHA_FULL}, set(), strict=True)


# ---------------------------------------------------------------------------
# 7. strict=False returns list (does not raise)
# ---------------------------------------------------------------------------

def test_strict_false_returns_list() -> None:
    """strict=False (the default) returns a list of issues without raising."""
    output = f"Commit {UNKNOWN_SHA} is suspicious."
    # Should not raise, should return a non-empty list
    issues = validate_citations(output, {KNOWN_SHA_FULL}, set(), strict=False)
    assert isinstance(issues, list)
    assert len(issues) == 1


# ---------------------------------------------------------------------------
# 8. Empty output → empty list
# ---------------------------------------------------------------------------

def test_empty_output_returns_no_issues() -> None:
    """Empty string produces no issues."""
    issues = validate_citations("", {KNOWN_SHA_FULL}, KNOWN_PRS)
    assert issues == []


# ---------------------------------------------------------------------------
# 9. SHA too short — 6-char hex string is NOT extracted (below 7-char threshold)
# ---------------------------------------------------------------------------

def test_sha_too_short_not_extracted() -> None:
    """A 6-char hex string falls below the 7-char minimum and is never flagged."""
    short_hex = "a1b2c3"  # exactly 6 chars
    output = f"Token {short_hex} is not a SHA."
    # Even though it's not in known_shas, it should not produce an issue
    issues = validate_citations(output, set(), set())
    assert issues == []


# ---------------------------------------------------------------------------
# 10. PR check skipped when known_prs is empty
# ---------------------------------------------------------------------------

def test_pr_check_skipped_when_known_prs_empty() -> None:
    """When known_prs is empty, PR references in output are NOT flagged.

    This guards against false positives when the caller hasn't supplied PR data.
    """
    output = "See #999 for the original discussion."
    # Pass empty known_prs — no PR validation should occur
    issues = validate_citations(output, set(), known_prs=set())
    # No unknown_pr issues should be produced
    assert all(i.kind != "unknown_pr" for i in issues), (
        "Expected no PR issues when known_prs is empty, got: " + repr(issues)
    )


# ---------------------------------------------------------------------------
# 11. SHA check skipped when known_shas is empty
# ---------------------------------------------------------------------------

def test_sha_check_skipped_when_known_shas_empty() -> None:
    """When known_shas is empty, SHA references in output produce no issues."""
    output = f"Changed in {UNKNOWN_SHA} for reasons."
    issues = validate_citations(output, known_shas=set(), known_prs=set())
    assert issues == []


# ---------------------------------------------------------------------------
# 12. strict=True raises CitationError (typed exception, not bare ValueError)
# ---------------------------------------------------------------------------

def test_strict_raises_citation_error() -> None:
    """strict=True raises CitationError, not a bare ValueError."""
    output = f"Commit {UNKNOWN_SHA} is suspicious."
    with pytest.raises(CitationError):
        validate_citations(output, {KNOWN_SHA_FULL}, set(), strict=True)


# ---------------------------------------------------------------------------
# 13. CitationError carries the issues list
# ---------------------------------------------------------------------------

def test_citation_error_carries_issues() -> None:
    """Caught CitationError exposes .issues with the ValidationIssue objects."""
    output = f"Commit {UNKNOWN_SHA} is suspicious."
    with pytest.raises(CitationError) as exc_info:
        validate_citations(output, {KNOWN_SHA_FULL}, set(), strict=True)
    err = exc_info.value
    assert len(err.issues) == 1
    assert err.issues[0].kind == "unknown_sha"
    assert err.issues[0].value == UNKNOWN_SHA


# ---------------------------------------------------------------------------
# 14. CitationError is a ValueError subclass (backwards compat)
# ---------------------------------------------------------------------------

def test_citation_error_is_value_error() -> None:
    """CitationError is a subclass of ValueError for backwards compatibility."""
    output = f"Commit {UNKNOWN_SHA} is suspicious."
    with pytest.raises(CitationError) as exc_info:
        validate_citations(output, {KNOWN_SHA_FULL}, set(), strict=True)
    assert isinstance(exc_info.value, ValueError)
