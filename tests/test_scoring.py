"""Tests for why.scoring.score_commit — written BEFORE the implementation (TDD)."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from why.commit import Commit
from why.scoring import score_commit, select_key_commits

# ---------------------------------------------------------------------------
# Helper factory — builds a Commit with sensible defaults, all fields
# overridable through keyword arguments.
# ---------------------------------------------------------------------------

_BASE_DATE = datetime(2024, 6, 1, tzinfo=UTC)
NOW = date(2024, 6, 1)


def make_commit(
    subject: str = "fix: thing",
    body: str = "",
    parents: tuple[str, ...] = ("a" * 40,),
    additions: int = 0,
    deletions: int = 0,
    days_ago: int = 0,
    sha: str = "a" * 40,
) -> Commit:
    """Return a frozen Commit whose date is ``days_ago`` before NOW."""
    dt = _BASE_DATE - timedelta(days=days_ago)
    return Commit(
        sha=sha,
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
    ("higher_label", "lower_label"),
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


# ---------------------------------------------------------------------------
# select_key_commits tests
# ---------------------------------------------------------------------------


def test_select_empty_returns_empty() -> None:
    """Empty commit list must return an empty list."""
    assert select_key_commits([], {}, n=5, now=NOW) == []


def test_select_single_commit() -> None:
    """A single commit must be returned as-is."""
    c = make_commit(sha="1" * 40)
    result = select_key_commits([c], {}, n=5, now=NOW)
    assert result == [c]


def test_select_always_includes_oldest() -> None:
    """The oldest commit must always appear in the result regardless of its score."""
    # Build 10 commits — oldest is a junk commit so it would normally score low.
    commits = [
        make_commit(
            subject="chore: boring housekeeping",  # junk penalty → low score
            sha=f"{i:040d}",
            days_ago=100 - i,  # commit 0 is oldest (100 days ago), commit 9 is newest
        )
        for i in range(10)
    ]
    # Give commits 1-9 high scores so the top-n would not naturally include commit 0.
    high_scorers = [
        make_commit(
            subject="fix: important thing",
            additions=200,
            sha=f"{10 + i:040d}",
            days_ago=90 - i,
        )
        for i in range(10)
    ]
    all_commits = commits[:1] + high_scorers + commits[1:]
    oldest = min(all_commits, key=lambda c: c.date)
    result = select_key_commits(all_commits, {}, n=5, now=NOW)
    result_shas = {c.sha for c in result}
    assert oldest.sha in result_shas, "Oldest commit must always be included"


def test_select_includes_recent_substantive() -> None:
    """The most-recent commit with score > 3.0 must always be included."""
    # Commit at days_ago=0 with a good subject — will score well above 3.0.
    newest_substantive = make_commit(
        subject="fix: critical security patch",
        additions=100,
        sha="f" * 40,
        days_ago=0,
    )
    older_commits = [
        make_commit(sha=f"{i:040d}", days_ago=10 + i, subject="chore: bump dep")
        for i in range(8)
    ]
    all_commits = [*older_commits, newest_substantive]
    result = select_key_commits(all_commits, {}, n=3, now=NOW)
    result_shas = {c.sha for c in result}
    assert newest_substantive.sha in result_shas, (
        "Most-recent substantive commit must always be included"
    )


def test_select_fallback_when_no_substantive() -> None:
    """When no commit scores above 3.0 the most-recent commit is still included."""
    # All junk/merge commits score very low — well below _SUBSTANTIVE_THRESHOLD.
    commits = [
        make_commit(
            subject="chore: format",  # junk pattern → -10 penalty
            sha=f"{i:040d}",
            days_ago=10 - i,  # commit 0 is oldest, commit 9 is newest (days_ago=1)
        )
        for i in range(10)
    ]
    most_recent = max(commits, key=lambda c: c.date)
    result = select_key_commits(commits, {}, n=3, now=NOW)
    result_shas = {c.sha for c in result}
    assert most_recent.sha in result_shas, (
        "Most-recent commit must be included as fallback when no substantive commit exists"
    )


def test_select_no_duplicates() -> None:
    """Result must contain no repeated SHAs."""
    commits = [
        make_commit(sha=f"{i:040d}", days_ago=i, additions=50)
        for i in range(6)
    ]
    result = select_key_commits(commits, {}, n=5, now=NOW)
    shas = [c.sha for c in result]
    assert len(shas) == len(set(shas)), "Result must have no duplicate SHAs"


def test_select_sorted_oldest_first() -> None:
    """Result must be sorted in ascending date order (oldest first)."""
    commits = [
        make_commit(sha=f"{i:040d}", days_ago=i * 5, additions=20)
        for i in range(6)
    ]
    result = select_key_commits(commits, {}, n=5, now=NOW)
    dates = [c.date for c in result]
    assert dates == sorted(dates), "Result must be sorted oldest-first"


def test_select_n_cap_with_must_includes_win() -> None:
    """When n=1 but oldest and most-recent substantive are distinct, result has 2 commits."""
    oldest = make_commit(sha="0" * 40, days_ago=100, subject="chore: old junk")
    # A good commit that will be the most-recent substantive.
    recent_good = make_commit(
        sha="1" * 40,
        days_ago=0,
        subject="fix: critical bug",
        additions=100,
    )
    # The two must-includes are distinct; n=1 cannot suppress either.
    result = select_key_commits([oldest, recent_good], {}, n=1, now=NOW)
    assert len(result) == 2, (
        "Both must-include anchors must appear even when n=1"
    )
    result_shas = {c.sha for c in result}
    assert oldest.sha in result_shas
    assert recent_good.sha in result_shas


def test_select_with_prs() -> None:
    """A commit whose SHA is in the prs dict gets a score boost and is selected."""
    # This commit has a low base score but the PR bonus should lift it into selection.
    pr_commit = make_commit(
        sha="pr" + "a" * 38,
        days_ago=50,
        subject="minor tweak",  # no keyword, no diff — low base score
        additions=0,
    )
    # Many other decent commits fill the slots without the PR one.
    fillers = [
        make_commit(sha=f"{i:040d}", days_ago=i, additions=30, subject="fix: thing")
        for i in range(10)
    ]
    prs = {pr_commit.sha: object()}  # membership is all that matters
    result = select_key_commits([pr_commit, *fillers], prs, n=3, now=NOW)
    result_shas = {c.sha for c in result}
    # PR bonus is +3.0, which should lift this commit above low-scoring fillers.
    assert pr_commit.sha in result_shas, "PR-backed commit must be boosted into selection"


def test_select_order_agnostic() -> None:
    """Result is identical whether input is newest-first or oldest-first."""
    commits = [
        make_commit(sha=f"{i:040d}", days_ago=i * 3, additions=10 * i)
        for i in range(8)
    ]
    result_asc = select_key_commits(commits, {}, n=4, now=NOW)
    result_desc = select_key_commits(list(reversed(commits)), {}, n=4, now=NOW)
    assert result_asc == result_desc, (
        "select_key_commits must return the same result regardless of input order"
    )
