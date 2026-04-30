"""Tests for _estimate_tokens and _shrink_for_budget pure helpers in why.synth.

Written BEFORE implementation (TDD red-green-refactor).
"""

from __future__ import annotations

import logging

import pytest

from tests._helpers import make_cwpr as _make_cwpr
from why.llm import _resolve_max_ctx
from why.synth import _estimate_tokens, _shrink_for_budget

# ---------------------------------------------------------------------------
# _estimate_tokens
# ---------------------------------------------------------------------------


def test_estimate_tokens_basic() -> None:
    assert _estimate_tokens("x" * 100) == 25


# ---------------------------------------------------------------------------
# _shrink_for_budget
# ---------------------------------------------------------------------------


def test_shrink_no_op_when_under_budget() -> None:
    """3 small commits with a generous budget → no drops, no truncations."""
    commits = [
        _make_cwpr("aaa", diff="short diff"),
        _make_cwpr("bbb", diff="another short diff"),
        _make_cwpr("ccc", diff="yet another short diff"),
    ]
    system_prompt = "tiny system"
    current_code = "tiny code"
    # Very large budget — nothing should be dropped or truncated
    target_tokens = 100_000

    shrunk, dropped, truncated = _shrink_for_budget(
        commits, current_code, system_prompt, target_tokens
    )

    assert len(shrunk) == 3
    assert dropped == 0
    assert truncated == 0


def test_shrink_drops_oldest_first() -> None:
    """5 commits sorted oldest→newest; tight budget that fits ~3 → first 2 dropped."""
    # Each diff is 400 chars ≈ 100 tokens; message ≈ 5 tokens.
    # 5 commits * ~105 tokens each = ~525 tokens from commits alone.
    # We want a budget that holds ~3 commits. Set system+code to near-zero
    # and target_tokens so that headroom ≈ 320 tokens (fits 3 * ~105 but not 5).
    diff = "a" * 400  # 100 tokens
    commits = [_make_cwpr(f"commit{i}", diff=diff, subject="msg") for i in range(5)]

    # headroom = int(380 * 0.85) ≈ 323 tokens
    # fixed ≈ 0 (empty system + code)
    # budget ≈ 323; each commit costs ~100 tokens → fits 3, not 5
    target_tokens = 380

    shrunk, dropped, truncated = _shrink_for_budget(
        commits, "", "", target_tokens
    )

    assert dropped == 2
    assert truncated == 0
    assert len(shrunk) == 3
    # Oldest two should be gone; remaining are commits[2], commits[3], commits[4]
    assert [c.commit.sha for c in shrunk] == ["commit2", "commit3", "commit4"]


def test_shrink_truncates_long_diffs() -> None:
    """A commit with 200-line diff → truncated to 80 lines + sentinel."""
    diff_lines = [f"line {i}" for i in range(200)]
    diff = "\n".join(diff_lines)
    commits = [_make_cwpr("sha1", diff=diff)]

    # Very large budget so no dropping happens
    shrunk, dropped, truncated = _shrink_for_budget(
        commits, "", "", target_tokens=100_000
    )

    assert dropped == 0
    assert truncated == 1
    assert len(shrunk) == 1

    result_diff = shrunk[0].diff
    result_lines = result_diff.splitlines()
    # 80 content lines + 1 sentinel line
    assert len(result_lines) == 81
    assert result_lines[80] == "... [truncated 120 lines]"
    assert result_lines[0] == diff_lines[0]
    assert result_lines[79] == diff_lines[79]


def test_shrink_truncates_then_drops() -> None:
    """Commits with long diffs AND total over budget → both truncation and dropping occur."""
    # 4 commits, each with a 200-line diff.  After truncation each diff = first 80 lines + sentinel.
    _TRUNCATED_LINES = 80
    _SENTINEL_DROPPED = 120  # 200 - 80
    _TOTAL_INPUT_LINES = 200

    diff_lines = [f"line {i}" for i in range(_TOTAL_INPUT_LINES)]
    diff = "\n".join(diff_lines)
    commits = [_make_cwpr(f"sha{i}", diff=diff) for i in range(4)]

    # Compute per-commit cost exactly to choose a deterministic target_tokens.
    # _estimate_tokens(text) = len(text) // 4
    sentinel = f"\n... [truncated {_SENTINEL_DROPPED} lines]"
    truncated_diff = "\n".join(diff_lines[:_TRUNCATED_LINES]) + sentinel
    expected_diff_tokens = len(truncated_diff) // 4  # 163
    # Default subject in _make_cwpr is "some commit message" (19 chars → 4 tokens)
    _DEFAULT_SUBJECT = "some commit message"
    expected_subject_tokens = len(_DEFAULT_SUBJECT) // 4  # 4
    per_commit_tokens = expected_diff_tokens + expected_subject_tokens  # 167

    # headroom = int(target * 0.85); we want headroom to fit exactly 2 commits (not 3).
    # 2 * 167 = 334; headroom must be > 334 to keep 2 but < 3*167=501.
    # target = ceil((2 * per_commit_tokens + 1) / 0.85) = 395 → headroom = 335.
    import math
    target_tokens = math.ceil((2 * per_commit_tokens + 1) / 0.85)  # 395

    shrunk, dropped, truncated = _shrink_for_budget(
        commits, "", "", target_tokens
    )

    # Every commit had a >80-line diff, so all 4 were truncated.
    assert truncated == 4
    # Budget fits 2 commits → 2 oldest are dropped.
    assert dropped == 2
    assert len(shrunk) == 2
    # Newest 2 commits are kept (sha2 and sha3; sha0 and sha1 were dropped).
    assert [c.commit.sha for c in shrunk] == ["sha2", "sha3"]


def test_shrink_empty_commits() -> None:
    """Empty input → returns empty list with (0, 0) counts."""
    shrunk, dropped, truncated = _shrink_for_budget([], "", "", target_tokens=1000)

    assert shrunk == []
    assert dropped == 0
    assert truncated == 0


def test_shrink_handles_negative_budget() -> None:
    """When system+code alone exceed target * 0.85, all commits are dropped."""
    # Each "word" is 4 chars → 1 token; 10000 chars = 2500 tokens
    large_system = "x" * 10_000   # 2500 tokens
    large_code = "y" * 10_000     # 2500 tokens
    # total fixed = 5000 tokens; headroom = int(1000 * 0.85) = 850; budget = 850 - 5000 = -4150

    commits = [_make_cwpr(f"sha{i}", diff="small diff") for i in range(3)]

    shrunk, dropped, _truncated = _shrink_for_budget(
        commits, large_code, large_system, target_tokens=1000
    )

    assert shrunk == []
    assert dropped == 3


# ---------------------------------------------------------------------------
# _resolve_max_ctx
# ---------------------------------------------------------------------------


def test_resolve_explicit_positive(monkeypatch: pytest.MonkeyPatch) -> None:
    """WHY_LLM_MAX_CTX=2048 → returns 2048."""
    monkeypatch.setenv("WHY_LLM_MAX_CTX", "2048")
    assert _resolve_max_ctx("groq") == 2048


def test_resolve_explicit_zero_disables(monkeypatch: pytest.MonkeyPatch) -> None:
    """WHY_LLM_MAX_CTX=0 → returns None (explicit disable)."""
    monkeypatch.setenv("WHY_LLM_MAX_CTX", "0")
    assert _resolve_max_ctx("groq") is None


def test_resolve_unset_openai_compatible_defaults_4096(monkeypatch: pytest.MonkeyPatch) -> None:
    """WHY_LLM_MAX_CTX unset, provider=openai-compatible → returns 4096."""
    monkeypatch.delenv("WHY_LLM_MAX_CTX", raising=False)
    assert _resolve_max_ctx("openai-compatible") == 4096


def test_resolve_unset_groq_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """WHY_LLM_MAX_CTX unset, provider=groq → returns None."""
    monkeypatch.delenv("WHY_LLM_MAX_CTX", raising=False)
    assert _resolve_max_ctx("groq") is None


def test_resolve_unset_provider_unset_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """WHY_LLM_MAX_CTX unset, provider=groq → returns None."""
    monkeypatch.delenv("WHY_LLM_MAX_CTX", raising=False)
    assert _resolve_max_ctx("groq") is None


def test_resolve_explicit_overrides_provider_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """WHY_LLM_MAX_CTX=8192, provider=openai-compatible → returns 8192."""
    monkeypatch.setenv("WHY_LLM_MAX_CTX", "8192")
    assert _resolve_max_ctx("openai-compatible") == 8192


def test_resolve_invalid_value_returns_none_and_warns(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """WHY_LLM_MAX_CTX=abc → returns None, emits a warning."""
    monkeypatch.setenv("WHY_LLM_MAX_CTX", "abc")
    with caplog.at_level(logging.WARNING, logger="why.synth"):
        result = _resolve_max_ctx("groq")
    assert result is None
    assert len(caplog.records) >= 1
    assert caplog.records[0].levelno == logging.WARNING


def test_resolve_negative_value_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """WHY_LLM_MAX_CTX=-100 → returns None."""
    monkeypatch.setenv("WHY_LLM_MAX_CTX", "-100")
    assert _resolve_max_ctx("groq") is None


def test_resolve_empty_string_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """WHY_LLM_MAX_CTX="" → returns None (empty string treated as not-a-valid-integer)."""
    monkeypatch.setenv("WHY_LLM_MAX_CTX", "")
    assert _resolve_max_ctx("openai-compatible") is None
