"""Key-commit scoring function for why.

Assigns a float score to a commit based on:
  - diff size (log-scaled additions + deletions)
  - message length (log-scaled subject + body)
  - presence of important keywords in the commit message
  - whether the commit has an associated PR
  - recency (bonus decays logarithmically as the commit ages)
  - penalties for merge commits and junk-pattern subjects
"""

from __future__ import annotations

import re
from datetime import date
from math import log

from .commit import Commit

# Keywords that signal a meaningful commit — each occurrence adds _KEYWORD_BONUS points.
KEYWORDS = [
    "refactor",
    "rewrite",
    "redesign",
    "migrate",
    "fix",
    "security",
    "incident",
    "breaking",
    "add support",
    "remove",
    "deprecated",
]

# Regex patterns matched against the *lowercased* subject.
# A match subtracts _JUNK_PENALTY points (routine/housekeeping commits).
# Note: r"^bump version" was removed — it was unreachable because r"^bump "
# is a broader prefix that would match first via any().
JUNK_PATTERNS = [
    r"^typo",
    r"^fix typo",
    r"^format",
    r"^style:",
    r"^lint:",
    r"^chore:",
    r"^bump ",
]

# Pre-compiled versions of JUNK_PATTERNS for performance.
_JUNK_RES = [re.compile(p) for p in JUNK_PATTERNS]

# Named constants for all magic numbers — makes tuning explicit and testable.
_DIFF_WEIGHT = 2.0
_PR_BONUS = 3.0
_RECENCY_MAX = 3.0
_MERGE_PENALTY = 5.0
_JUNK_PENALTY = 10.0
_KEYWORD_BONUS = 2.0


def score_commit(c: Commit, now: date, has_pr: bool) -> float:
    """Return a float score representing how 'key' a commit is.

    Higher scores indicate commits more likely to explain why a line or
    symbol was introduced or changed.

    Parameters
    ----------
    c:       the commit to score
    now:     reference date used to compute age (typically today)
    has_pr:  whether a pull-request is associated with this commit
    """
    s = 0.0

    # Diff-size contribution — logarithmic to avoid huge diffs dominating.
    s += log(1 + c.additions + c.deletions) * _DIFF_WEIGHT

    # Message-length contribution — longer messages tend to be more descriptive.
    s += log(1 + len(c.body) + len(c.subject))

    # Keyword bonus — each keyword found adds _KEYWORD_BONUS points.
    # Single-word keywords use \b word-boundary matching to avoid false positives
    # (e.g. "refactoring" should NOT match "refactor"). Multi-word keywords like
    # "add support" use simple substring matching since \b doesn't span spaces.
    msg = (c.subject + " " + c.body).lower()
    s += sum(
        _KEYWORD_BONUS for kw in KEYWORDS
        if (re.search(r"\b" + re.escape(kw) + r"\b", msg) if " " not in kw else kw in msg)
    )

    # PR bonus — a PR usually means more review/context.
    if has_pr:
        s += _PR_BONUS

    # Recency bonus — decays from ~_RECENCY_MAX at day 0 toward 0 as the commit ages.
    # Clamp days_ago to 0 to guard against future-dated commits causing log(negative).
    days_ago = max(0, (now - c.date.date()).days)
    s += max(0.0, _RECENCY_MAX - log(1 + days_ago / 30))

    # Merge-commit penalty — merge commits rarely explain intent.
    if c.is_merge:
        s -= _MERGE_PENALTY

    # Junk-pattern penalty — housekeeping commits are unlikely to be key.
    # Uses pre-compiled regexes (_JUNK_RES) for performance.
    if any(rx.match(c.subject.lower()) for rx in _JUNK_RES):
        s -= _JUNK_PENALTY

    return s
