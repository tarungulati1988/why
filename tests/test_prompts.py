"""Tests for why.prompts — written BEFORE the implementation (TDD).

Fixed, deterministic test data; no datetime.now() calls.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from why.commit import Commit
from why.llm import Message
from why.prompts import (
    GROUNDING_SYSTEM_PROMPT,
    WHY_SYSTEM_PROMPT,
    CommitWithPR,
    PRMetadata,
    _render_commit,
    build_grounding_prompt,
    build_system_prompt,
    build_why_prompt,
)
from why.target import Target

# ---------------------------------------------------------------------------
# Deterministic test fixtures
# ---------------------------------------------------------------------------

FIXED_DATE = datetime(2026, 1, 15, 12, 0, 0, tzinfo=UTC)
FIXED_COMMIT = Commit(
    sha="abc1234def5678901234567890",
    author_name="Jane Smith",
    author_email="jane@example.com",
    date=FIXED_DATE,
    subject="fix: handle null token in auth check",
    body="- removed None guard\n+ added explicit null check",
    parents=("aaabbbccc",),
    additions=3,
    deletions=1,
)
FIXED_TARGET = Target(file=Path("src/why/scoring.py"), line=42)
FIXED_CURRENT_CODE = (
    "def score_commit(c: Commit, now: date, has_pr: bool) -> float:\n    return 0.0"
)


# ---------------------------------------------------------------------------
# Helper: build a prompt with the fixed commit and target
# ---------------------------------------------------------------------------


def _default_result() -> list[Message]:
    """Return build_why_prompt with the FIXED_COMMIT (no PR body)."""
    commits = [CommitWithPR(commit=FIXED_COMMIT)]
    return build_why_prompt(FIXED_TARGET, FIXED_CURRENT_CODE, commits)


# ---------------------------------------------------------------------------
# Test 1: result structure — single user Message
# ---------------------------------------------------------------------------


def test_returns_single_user_message() -> None:
    """build_why_prompt must return a list with exactly one Message(role='user')."""
    result = _default_result()
    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], Message)
    assert result[0].role == "user"


# ---------------------------------------------------------------------------
# Test 2-4: target rendering variants
# ---------------------------------------------------------------------------


def test_target_file_only() -> None:
    """A target with no line or symbol must render just the file path."""
    target = Target(file=Path("src/why/scoring.py"))
    result = build_why_prompt(target, FIXED_CURRENT_CODE, [])
    content = result[0].content
    # Should show the file but NOT 'Line:' or 'Symbol:'
    assert "File: `src/why/scoring.py`" in content
    assert "Line:" not in content
    assert "Symbol:" not in content


def test_target_with_line() -> None:
    """A target with line=42 must render 'File: ..., Line: 42'."""
    target = Target(file=Path("src/why/scoring.py"), line=42)
    result = build_why_prompt(target, FIXED_CURRENT_CODE, [])
    content = result[0].content
    assert "File: `src/why/scoring.py`, Line: 42" in content


def test_target_with_symbol() -> None:
    """A target with symbol must append ', Symbol: `<symbol>`'."""
    target = Target(file=Path("src/why/scoring.py"), symbol="score_commit")
    result = build_why_prompt(target, FIXED_CURRENT_CODE, [])
    content = result[0].content
    assert "Symbol: `score_commit`" in content


# ---------------------------------------------------------------------------
# Test 5-6: commit rendering - SHA and date
# ---------------------------------------------------------------------------


def test_commit_sha_is_short() -> None:
    """Commit section header must use the 7-char short SHA (abc1234)."""
    content = _default_result()[0].content
    assert "abc1234" in content
    # Full SHA must NOT appear as the header SHA
    assert "abc1234def5678901234567890" not in content.split("###")[1]


def test_commit_date_format() -> None:
    """Commit date must appear as YYYY-MM-DD only (no time component)."""
    content = _default_result()[0].content
    assert "2026-01-15" in content
    # The full ISO timestamp must not appear in the commit header line
    assert "12:00:00" not in content


# ---------------------------------------------------------------------------
# Test 7-8: PR body presence / absence
# ---------------------------------------------------------------------------


def test_pr_body_present() -> None:
    """When pr_number and pr_body are set, the content string must appear verbatim."""
    commits = [CommitWithPR(commit=FIXED_COMMIT, pr_number=42, pr_title="Fix null check", pr_body="Fixes #99: adds null check")]
    result = build_why_prompt(FIXED_TARGET, FIXED_CURRENT_CODE, commits)
    assert "Fixes #99: adds null check" in result[0].content


def test_pr_body_absent() -> None:
    """When pr_number is None, the PR section is omitted entirely (no 'N/A')."""
    commits = [CommitWithPR(commit=FIXED_COMMIT, pr_body=None)]
    result = build_why_prompt(FIXED_TARGET, FIXED_CURRENT_CODE, commits)
    assert "N/A" not in result[0].content
    assert "PR Body" not in result[0].content


# ---------------------------------------------------------------------------
# Test 9: empty commit list
# ---------------------------------------------------------------------------


def test_no_commits() -> None:
    """An empty key_commits list must return a valid single user Message."""
    result = build_why_prompt(FIXED_TARGET, FIXED_CURRENT_CODE, [])
    assert len(result) == 1
    assert result[0].role == "user"
    # Target section must still be present
    assert "## Target" in result[0].content
    assert "## Current Code" in result[0].content
    # Commits section must be present with just the header — no trailing empty body.
    # When key_commits is empty, "## Commits" has no body: it is immediately followed
    # by the section separator "---" (not by blank commit content).
    assert "## Commits" in result[0].content
    assert "## Commits\n\n---" in result[0].content


# ---------------------------------------------------------------------------
# Test 10: WHY_SYSTEM_PROMPT is a non-empty string
# ---------------------------------------------------------------------------


def test_system_prompt_not_empty() -> None:
    """WHY_SYSTEM_PROMPT must be a non-empty string."""
    assert isinstance(WHY_SYSTEM_PROMPT, str)
    assert len(WHY_SYSTEM_PROMPT) > 0


# ---------------------------------------------------------------------------
# Test 11: golden file comparison
# ---------------------------------------------------------------------------


def test_golden_file(update_goldens: bool) -> None:
    """Content must match the golden fixture exactly."""
    commits = [CommitWithPR(commit=FIXED_COMMIT, pr_number=42, pr_title="Fix null check", pr_body="Fixes #99: adds null check")]
    result = build_why_prompt(FIXED_TARGET, FIXED_CURRENT_CODE, commits)
    content = result[0].content
    golden = Path(__file__).parent / "fixtures" / "prompts" / "why_prompt_golden.txt"
    if update_goldens:
        golden.parent.mkdir(parents=True, exist_ok=True)
        golden.write_text(content)
        return
    assert golden.exists(), "Golden file missing — run with --update-goldens to create"
    assert content == golden.read_text()


# ---------------------------------------------------------------------------
# B1: diff field on CommitWithPR
# ---------------------------------------------------------------------------


def test_diff_field_used_in_render() -> None:
    """CommitWithPR.diff is rendered in the diff fence, not commit.body."""
    cwpr = CommitWithPR(
        commit=FIXED_COMMIT,
        diff="+ new line added\n- old line removed",
    )
    result = build_why_prompt(FIXED_TARGET, FIXED_CURRENT_CODE, [cwpr])
    content = result[0].content
    # The explicit diff field should appear
    assert "+ new line added" in content
    assert "- old line removed" in content
    # commit.body should NOT be used as the diff
    assert "removed None guard" not in content


def test_diff_empty_shows_placeholder() -> None:
    """When diff is empty (not yet fetched), the placeholder text is shown."""
    cwpr = CommitWithPR(commit=FIXED_COMMIT, diff="")
    result = build_why_prompt(FIXED_TARGET, FIXED_CURRENT_CODE, [cwpr])
    assert "(no diff available)" in result[0].content


# ---------------------------------------------------------------------------
# B2: Prompt injection mitigations
# ---------------------------------------------------------------------------


def test_newline_in_subject_is_stripped() -> None:
    """Newlines in commit.subject must be collapsed to spaces in the header."""
    injected_commit = Commit(
        sha="abc1234def5678901234567890",
        author_name="Jane Smith",
        author_email="jane@example.com",
        date=FIXED_DATE,
        subject="fix: legit\n## Injected Section",
        body="",
        parents=("aaabbbccc",),
    )
    cwpr = CommitWithPR(commit=injected_commit, diff="some diff")
    result = build_why_prompt(FIXED_TARGET, FIXED_CURRENT_CODE, [cwpr])
    content = result[0].content
    # The injected newline must be replaced — no raw newline in the ### header line
    header_line = next(ln for ln in content.splitlines() if ln.startswith("### `abc1234`"))
    assert "\n" not in header_line
    # "## Injected Section" must NOT appear as a standalone Markdown heading line —
    # it should be inlined into the header text after the newline was stripped to a space
    assert "\n## Injected Section" not in content


def test_backtick_fence_in_diff_is_escaped() -> None:
    """Triple backticks inside the diff must be escaped to prevent fence breakout."""
    cwpr = CommitWithPR(
        commit=FIXED_COMMIT,
        diff="some code\n```\nmore code",
    )
    result = build_why_prompt(FIXED_TARGET, FIXED_CURRENT_CODE, [cwpr])
    content = result[0].content
    # The raw triple-backtick must not appear unescaped inside the diff block
    # Find the diff fence and check the content within it
    assert "\\`\\`\\`" in content


def test_pr_body_is_fenced() -> None:
    """PR body must be wrapped in a ```text fence to prevent Markdown injection."""
    cwpr = CommitWithPR(
        commit=FIXED_COMMIT,
        pr_number=1,
        pr_title="Injection test",
        pr_body="## Injected Heading\nsome content",
    )
    result = build_why_prompt(FIXED_TARGET, FIXED_CURRENT_CODE, [cwpr])
    content = result[0].content
    assert "```text" in content
    assert "## Injected Heading" in content  # content preserved but fenced


# ---------------------------------------------------------------------------
# N2: pr_body="" should render as "N/A"
# ---------------------------------------------------------------------------


def test_pr_body_empty_string() -> None:
    """When pr_number is present but pr_body is empty, the PR section still renders (with empty fence)."""
    cwpr = CommitWithPR(commit=FIXED_COMMIT, pr_number=1, pr_title="Empty body PR", pr_body="")
    result = build_why_prompt(FIXED_TARGET, FIXED_CURRENT_CODE, [cwpr])
    # PR section should be present since pr_number is set
    assert "PR #1:" in result[0].content
    assert "N/A" not in result[0].content


# ---------------------------------------------------------------------------
# N3: target with both line and symbol
# ---------------------------------------------------------------------------


def test_target_with_line_and_symbol() -> None:
    """A target with both line and symbol must render both in the header."""
    target = Target(file=Path("src/why/scoring.py"), line=42, symbol="score_commit")
    result = build_why_prompt(target, FIXED_CURRENT_CODE, [])
    assert "File: `src/why/scoring.py`, Line: 42, Symbol: `score_commit`" in result[0].content


# ---------------------------------------------------------------------------
# Sparse-history injection tests
# ---------------------------------------------------------------------------

def test_sparse_history_notice_injected_when_few_commits() -> None:
    """Sparse-history notice must appear when len(key_commits) < SPARSE_COMMIT_THRESHOLD."""
    # Use 2 distinct commits (below threshold of 3) to verify dynamic count
    second_commit = Commit(
        sha="def5678abc1234901234567890",
        author_name="Jane Smith",
        author_email="jane@example.com",
        date=FIXED_DATE,
        subject="chore: second commit",
        body="",
        parents=("aaabbbccc",),
    )
    commits = [CommitWithPR(commit=FIXED_COMMIT), CommitWithPR(commit=second_commit)]
    result = build_why_prompt(FIXED_TARGET, FIXED_CURRENT_CODE, commits)
    content = result[0].content
    assert "## Sparse History Notice" in content
    assert "structural" in content.lower()  # instruction must mention structural analysis
    assert "2 commit(s)" in content  # count must be dynamic, not hardcoded
    # Sparse notice must appear before the commits section
    assert content.index("## Sparse History Notice") < content.index("## Commits")


def test_sparse_history_notice_absent_when_enough_commits() -> None:
    """When len(key_commits) >= SPARSE_COMMIT_THRESHOLD, NO sparse-history notice must appear."""
    commits = [CommitWithPR(commit=FIXED_COMMIT) for _ in range(3)]
    result = build_why_prompt(FIXED_TARGET, FIXED_CURRENT_CODE, commits)
    content = result[0].content
    assert "## Sparse History Notice" not in content


def test_sparse_history_notice_zero_commits_wording() -> None:
    """Zero commits must render distinct wording, not '0 commit(s)'."""
    result = build_why_prompt(FIXED_TARGET, FIXED_CURRENT_CODE, [])
    content = result[0].content
    assert "## Sparse History Notice" in content
    assert "0 commit(s)" not in content
    assert "No git history is available" in content


# ---------------------------------------------------------------------------
# System prompt structural-analysis tests
# ---------------------------------------------------------------------------

def test_system_prompt_structural_contract() -> None:
    """WHY_SYSTEM_PROMPT must instruct structural analysis in prose without tag syntax."""
    # The prompt must still instruct structural analysis (uses "code structure" not "structural")
    assert "code structure" in WHY_SYSTEM_PROMPT.lower()
    # The new prompt uses prose instructions, not tag syntax
    assert "[STRUCTURAL]" not in WHY_SYSTEM_PROMPT
    # The new prompt references the sparse-history trigger by name
    assert "Sparse History Notice" in WHY_SYSTEM_PROMPT
    # The new prompt instructs code-structure analysis via naming conventions
    assert "naming conventions" in WHY_SYSTEM_PROMPT
    # New voice guidance: prompt must mention emoji formatting
    assert "emoji" in WHY_SYSTEM_PROMPT.lower()
    # Decision-test: structural insights must require causal language (because / so that)
    assert "because" in WHY_SYSTEM_PROMPT.lower()  # decision-test requires causal language
    # Anti-drift directive: explicitly tell the LLM not to narrate what the code does
    assert "do not describe what the code does" in WHY_SYSTEM_PROMPT.lower()


# ---------------------------------------------------------------------------
# build_system_prompt — parameterized repo URL tests
# ---------------------------------------------------------------------------


def test_build_system_prompt_with_repo_url() -> None:
    """When repo_url is provided, commit and PR links use the real URL."""
    prompt = build_system_prompt("https://github.com/acme/myrepo")
    assert "https://github.com/acme/myrepo/commit/" in prompt
    assert "https://github.com/acme/myrepo/pull/" in prompt
    assert "<repo-url>" not in prompt


def test_build_system_prompt_without_repo_url() -> None:
    """When repo_url is None, links use a generic placeholder."""
    prompt = build_system_prompt(None)
    assert "<repo-url>" in prompt
    assert "tarungulati1988" not in prompt  # no hardcoded owner


def test_build_system_prompt_with_non_github_url() -> None:
    """Non-GitHub URLs fall back to the generic placeholder."""
    prompt = build_system_prompt("https://gitlab.com/acme/myrepo")
    assert "<repo-url>" in prompt
    assert "gitlab.com" not in prompt


# ---------------------------------------------------------------------------
# Timeline Data section tests
# ---------------------------------------------------------------------------


def test_timeline_data_empty_commits() -> None:
    """Empty key_commits list must render the header and a no-commits notice."""
    from why.prompts import _render_timeline_data

    result = _render_timeline_data([])
    assert "## Timeline Data" in result
    assert "(no commits available)" in result


def test_timeline_data_single_commit_no_pr() -> None:
    """A single commit with no PR body must render a row without [PR #N]."""
    from why.prompts import _render_timeline_data

    cwpr = CommitWithPR(commit=FIXED_COMMIT)
    result = _render_timeline_data([cwpr])
    # Row must contain date, short SHA, and subject
    assert "2026-01-15" in result
    assert "abc1234" in result
    assert "fix: handle null token in auth check" in result
    # No PR label when pr_body is None
    assert "[PR #" not in result


def test_timeline_data_single_commit_with_pr() -> None:
    """A commit with pr_number=42 must render [PR #42] in the row."""
    from why.prompts import _render_timeline_data

    cwpr = CommitWithPR(commit=FIXED_COMMIT, pr_number=42)
    result = _render_timeline_data([cwpr])
    assert "[PR #42]" in result


def test_timeline_data_pr_number_used_not_pr_body_regex() -> None:
    """pr_number=None means NO [PR #N] label even if pr_body contains 'PR #42'."""
    from why.prompts import _render_timeline_data

    cwpr = CommitWithPR(commit=FIXED_COMMIT, pr_number=None, pr_body="Merged PR #42: fix")
    result = _render_timeline_data([cwpr])
    assert "[PR #" not in result


def test_timeline_data_pr_number_overrides_body() -> None:
    """When pr_number=7, [PR #7] appears regardless of pr_body content."""
    from why.prompts import _render_timeline_data

    cwpr = CommitWithPR(commit=FIXED_COMMIT, pr_number=7, pr_body="unrelated body text")
    result = _render_timeline_data([cwpr])
    assert "[PR #7]" in result


def test_timeline_data_section_in_user_message() -> None:
    """build_why_prompt result content must contain '## Timeline Data'."""
    commits = [CommitWithPR(commit=FIXED_COMMIT)]
    result = build_why_prompt(FIXED_TARGET, FIXED_CURRENT_CODE, commits)
    assert "## Timeline Data" in result[0].content


def test_timeline_data_section_is_last() -> None:
    """'## Timeline Data' must appear after '## Commits' in the content."""
    commits = [CommitWithPR(commit=FIXED_COMMIT)]
    result = build_why_prompt(FIXED_TARGET, FIXED_CURRENT_CODE, commits)
    content = result[0].content
    assert content.index("## Timeline Data") > content.index("## Commits")


# ---------------------------------------------------------------------------
# System prompt timeline instruction tests (Stride 2 of Issue #71)
# ---------------------------------------------------------------------------


def test_system_prompt_contains_timeline_instruction() -> None:
    """build_system_prompt() result must contain '## Appending the Timeline'."""
    prompt = build_system_prompt()
    assert "## Appending the Timeline" in prompt


def test_system_prompt_timeline_references_data_section() -> None:
    """build_system_prompt() result must reference '## Timeline Data' as source of truth."""
    prompt = build_system_prompt()
    assert "## Timeline Data" in prompt


def test_system_prompt_timeline_text_fence_instruction() -> None:
    """build_system_prompt() result must instruct wrapping the block in a ```text fence."""
    prompt = build_system_prompt()
    assert "```text" in prompt


# ---------------------------------------------------------------------------
# Grounding pass — GROUNDING_SYSTEM_PROMPT and build_grounding_prompt()
# ---------------------------------------------------------------------------


def test_grounding_system_prompt_exists() -> None:
    """GROUNDING_SYSTEM_PROMPT must be a non-empty string."""
    assert isinstance(GROUNDING_SYSTEM_PROMPT, str)
    assert len(GROUNDING_SYSTEM_PROMPT) > 0


def test_grounding_system_prompt_contains_key_instructions() -> None:
    """GROUNDING_SYSTEM_PROMPT must mention all four table column concepts."""
    assert "Grounding Check" in GROUNDING_SYSTEM_PROMPT
    assert "Claim" in GROUNDING_SYSTEM_PROMPT
    assert "Grounded" in GROUNDING_SYSTEM_PROMPT
    assert "Evidence" in GROUNDING_SYSTEM_PROMPT


def test_build_grounding_prompt_returns_single_user_message() -> None:
    """build_grounding_prompt must return a list with exactly one Message(role='user')."""
    commits = [CommitWithPR(commit=FIXED_COMMIT)]
    result = build_grounding_prompt("some explanation", commits)
    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], Message)
    assert result[0].role == "user"


def test_build_grounding_prompt_includes_first_pass_text() -> None:
    """The first-pass explanation must appear verbatim in the message content."""
    first_pass = "The team chose X for performance reasons."
    commits = [CommitWithPR(commit=FIXED_COMMIT)]
    result = build_grounding_prompt(first_pass, commits)
    assert first_pass in result[0].content


def test_build_grounding_prompt_includes_commit_context() -> None:
    """The commit short SHA must appear in the message content."""
    commits = [CommitWithPR(commit=FIXED_COMMIT)]
    result = build_grounding_prompt("some explanation", commits)
    assert "abc1234" in result[0].content


def test_build_grounding_prompt_with_empty_commits() -> None:
    """build_grounding_prompt must not crash when commits=[]."""
    result = build_grounding_prompt("some explanation", [])
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0].role == "user"


# ---------------------------------------------------------------------------
# --brief flag tests (Stride 1 of Issue #29)
# ---------------------------------------------------------------------------


def test_build_why_prompt_brief_appends_tail() -> None:
    """When brief=True, message content contains the brief-mode summary instruction."""
    commits = [CommitWithPR(commit=FIXED_COMMIT)]
    result = build_why_prompt(FIXED_TARGET, FIXED_CURRENT_CODE, commits, brief=True)
    assert "Output ONLY a 3-sentence summary" in result[0].content


def test_build_why_prompt_brief_tail_exact_text() -> None:
    """When brief=True, the exact tail string must appear verbatim in message content."""
    commits = [CommitWithPR(commit=FIXED_COMMIT)]
    result = build_why_prompt(FIXED_TARGET, FIXED_CURRENT_CODE, commits, brief=True)
    assert "Output ONLY a 3-sentence summary." in result[0].content
    assert "No sections, no citations block (inline only)." in result[0].content


def test_build_why_prompt_no_brief_omits_tail() -> None:
    """When brief=False (default), content must NOT contain the brief-mode instruction."""
    commits = [CommitWithPR(commit=FIXED_COMMIT)]
    result = build_why_prompt(FIXED_TARGET, FIXED_CURRENT_CODE, commits)
    assert "Output ONLY a 3-sentence summary" not in result[0].content


def test_build_why_prompt_brief_section_header() -> None:
    """When brief=True, content must contain the '## Output Format' section header."""
    commits = [CommitWithPR(commit=FIXED_COMMIT)]
    result = build_why_prompt(FIXED_TARGET, FIXED_CURRENT_CODE, commits, brief=True)
    assert "## Output Format" in result[0].content


# ---------------------------------------------------------------------------
# PRMetadata NamedTuple tests (Stride 1 of Issue #76)
# ---------------------------------------------------------------------------


def test_prmetadata_construction() -> None:
    """PRMetadata can be constructed with number, title, and body fields."""
    meta = PRMetadata(number=42, title="Fix null check", body="Fixes #99: adds null check")
    assert meta.number == 42
    assert meta.title == "Fix null check"
    assert meta.body == "Fixes #99: adds null check"


def test_prmetadata_fields_accessible_by_name() -> None:
    """PRMetadata fields are accessible by attribute name."""
    meta = PRMetadata(number=7, title="Some feature", body="some body text")
    assert meta.number == 7
    assert meta.title == "Some feature"
    assert meta.body == "some body text"


def test_prmetadata_is_namedtuple() -> None:
    """PRMetadata is a NamedTuple and supports tuple unpacking."""
    meta = PRMetadata(number=1, title="A title", body="test")
    number, title, body = meta
    assert number == 1
    assert title == "A title"
    assert body == "test"


def test_render_commit_with_pr_shows_title_and_body() -> None:
    """When PR is present, rendered output shows PR number and title as bold header."""
    cwpr = CommitWithPR(
        commit=FIXED_COMMIT,
        pr_number=42,
        pr_title="Fix the title",
        pr_body="This PR fixes things.",
    )
    result = _render_commit(cwpr)
    assert "PR #42:" in result
    assert "Fix the title" in result


def test_render_commit_with_pr_truncates_body_at_1000() -> None:
    """Body longer than 1000 chars is truncated with ' …' suffix."""
    long_body = "x" * 1100
    cwpr = CommitWithPR(
        commit=FIXED_COMMIT,
        pr_number=5,
        pr_title="Long PR",
        pr_body=long_body,
    )
    result = _render_commit(cwpr)
    assert " …" in result
    # The truncated body should be 1000 chars + " …"
    assert "x" * 1000 + " …" in result
    # The full long body must not appear
    assert "x" * 1100 not in result


def test_render_commit_without_pr_omits_pr_section() -> None:
    """When no PR is associated, the PR section is entirely omitted (no 'PR Body' or 'N/A')."""
    cwpr = CommitWithPR(commit=FIXED_COMMIT)
    result = _render_commit(cwpr)
    assert "PR Body" not in result
    assert "N/A" not in result


# ---------------------------------------------------------------------------
# CommitWithPR.pr_number field tests (Stride 1 of Issue #76)
# ---------------------------------------------------------------------------


def test_commitwithpr_pr_number_defaults_to_none() -> None:
    """CommitWithPR.pr_number defaults to None when not supplied."""
    cwpr = CommitWithPR(commit=FIXED_COMMIT)
    assert cwpr.pr_number is None


def test_commitwithpr_pr_number_can_be_set() -> None:
    """CommitWithPR can be constructed with an explicit pr_number."""
    cwpr = CommitWithPR(commit=FIXED_COMMIT, pr_number=42)
    assert cwpr.pr_number == 42


def test_commitwithpr_pr_number_with_pr_body() -> None:
    """CommitWithPR accepts both pr_number and pr_body simultaneously."""
    cwpr = CommitWithPR(commit=FIXED_COMMIT, pr_number=99, pr_body="Fixes #99")
    assert cwpr.pr_number == 99
    assert cwpr.pr_body == "Fixes #99"


def test_commitwithpr_is_still_frozen() -> None:
    """CommitWithPR remains a frozen dataclass — assignment must raise FrozenInstanceError."""
    import dataclasses

    cwpr = CommitWithPR(commit=FIXED_COMMIT, pr_number=1)
    with pytest.raises(dataclasses.FrozenInstanceError):
        cwpr.pr_number = 2  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Markdown injection — PR title bold-marker sanitization
# ---------------------------------------------------------------------------


def test_render_commit_pr_title_with_bold_markers() -> None:
    """A PR title containing '**' must be wrapped in backticks, not injected as raw Markdown."""
    cwpr = CommitWithPR(
        commit=FIXED_COMMIT,
        pr_number=1,
        pr_title="foo** injected **bar",
        pr_body="body",
    )
    result = _render_commit(cwpr)
    # The title must appear wrapped in backtick delimiters
    assert "`foo** injected **bar`" in result
    # The title must NOT appear as a raw unquoted bold-marker sequence that would
    # break the surrounding **PR #N:** bold span.
    assert "**PR #1: foo**" not in result
