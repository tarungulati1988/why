"""Tests for why.scoring.score_commit — written BEFORE the implementation (TDD)."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from why.commit import Commit
from why.scoring import score_commit, _JUNK_PENALTY

# ---------------------------------------------------------------------------
# Helper factory — builds a Commit with sensible defaults, all fields
# overridable through keyword arguments.
# ---------------------------------------------------------------------------

_BASE_DATE = datetime(2024, 6, 1, tzinfo=timezone.utc)
NOW = date(2024, 6, 1)


def make_commit(
    subject: str = "fix: thing",
    body: str = "",
    parents: tuple[str, ...] = ("a" * 40,),
    additions: int = 0,
    deletions: int = 0,
    days_ago: int = 0,
) -> Commit:
    """Return a frozen Commit whose date is ``days_ago`` before NOW."""
    dt = _BASE_DATE - timedelta(days=days_ago)
    return Commit(
        sha="a" * 40,
        author_name="Alice",
        author_email="a@b.com",
        date=dt,
        subject=subject,
        body=body,
        parents=parents,
        additions=additions,
        deletions=deletions,
    )


# ---------------------------------------------------------------------------
# Individual behaviour tests
# ---------------------------------------------------------------------------


def test_large_diff_scores_higher() -> None:
    """A commit with 500 additions must score higher than one with 5 additions."""
    big = make_commit(additions=500)
    small = make_commit(additions=5)
    assert score_commit(big, NOW, has_pr=False) > score_commit(small, NOW, has_pr=False)


def test_keyword_adds_points() -> None:
    """A subject containing a keyword ('refactor') must outscore an identical commit without."""
    with_kw = make_commit(subject="refactor: clean up auth module")
    without_kw = make_commit(subject="clean up auth module")
    assert score_commit(with_kw, NOW, has_pr=False) > score_commit(without_kw, NOW, has_pr=False)


def test_pr_adds_3_points() -> None:
    """has_pr=True must contribute exactly 3.0 extra points over has_pr=False."""
    c = make_commit()
    diff = score_commit(c, NOW, has_pr=True) - score_commit(c, NOW, has_pr=False)
    assert diff == pytest.approx(3.0)


def test_recency_bonus() -> None:
    """A commit from today must score higher than the same commit from 5 years ago."""
    recent = make_commit(days_ago=0)
    old = make_commit(days_ago=5 * 365)
    assert score_commit(recent, NOW, has_pr=False) > score_commit(old, NOW, has_pr=False)


def test_merge_commit_penalty() -> None:
    """A merge commit (2 parents) must score 5.0 points lower than a single-parent commit."""
    single = make_commit(parents=("a" * 40,))
    merge = make_commit(parents=("a" * 40, "b" * 40))
    diff = score_commit(single, NOW, has_pr=False) - score_commit(merge, NOW, has_pr=False)
    assert diff == pytest.approx(5.0)


def test_junk_pattern_penalty() -> None:
    """A junk-prefixed subject scores significantly lower (by ~10 points)."""
    junk = make_commit(subject="chore: minor formatting")
    clean = make_commit(subject="minor formatting tweak")
    diff = score_commit(clean, NOW, has_pr=False) - score_commit(junk, NOW, has_pr=False)
    # The -10 penalty dominates; small log-length differences are < 0.1
    assert diff > 9.5


def test_future_commit_does_not_crash() -> None:
    """A commit dated in the future must not raise ValueError."""
    future = make_commit(days_ago=-60)  # 60 days in the future
    score = score_commit(future, NOW, has_pr=False)
    # Score should be finite and not NaN
    assert isinstance(score, float)
    assert score == score  # not NaN


# ---------------------------------------------------------------------------
# Canonical rank-order test — four representative commits must rank in order.
# ---------------------------------------------------------------------------

# Each entry: (label, commit-kwargs, has_pr)
_CANONICAL = [
    (
        "security+pr",
        dict(subject="security: fix auth bypass", additions=500, days_ago=10),
        True,
    ),
    (
        "fix+no_pr",
        dict(subject="fix: null pointer in parser", additions=50, days_ago=90),
        False,
    ),
    (
        "merge+old",
        dict(
            subject="Merge branch 'main'",
            additions=0,
            days_ago=365,
            parents=("a" * 40, "b" * 40),  # merge commit
        ),
        False,
    ),
    (
        "junk_chore",
        dict(subject="chore: format whitespace", additions=5, days_ago=0),
        False,
    ),
]


@pytest.mark.parametrize(
    "higher_label,lower_label",
    [
        ("security+pr", "fix+no_pr"),
        ("fix+no_pr", "junk_chore"),
        ("junk_chore", "merge+old"),
    ],
    ids=[
        "security+pr > fix+no_pr",
        "fix+no_pr > junk_chore",
        "junk_chore > merge+old",
    ],
)
def test_canonical_rank_order(higher_label: str, lower_label: str) -> None:
    """Canonical commits rank in order driven by the scoring formula.

    Actual order (high to low):
      security+pr (~25.4) > fix+no_pr (~14.8) > junk_chore (~-0.2) > merge+old (~-1.6)

    Note: junk_chore edges above merge+old because its recency bonus (+3 for
    today) and non-zero diff outweigh the -10 junk penalty vs the -5 merge
    penalty on a year-old, zero-diff commit.
    """
    # Build a lookup of label -> (commit, has_pr)
    entries = {
        label: (make_commit(**kwargs), has_pr)  # type: ignore[arg-type]
        for label, kwargs, has_pr in _CANONICAL
    }
    higher_commit, higher_pr = entries[higher_label]
    lower_commit, lower_pr = entries[lower_label]

    higher_score = score_commit(higher_commit, NOW, has_pr=higher_pr)
    lower_score = score_commit(lower_commit, NOW, has_pr=lower_pr)

    assert higher_score > lower_score, (
        f"Expected {higher_label} ({higher_score:.2f}) > {lower_label} ({lower_score:.2f})"
    )
