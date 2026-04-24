"""Key-commit scoring function for why.

## Scoring model

The goal is to rank commits by how likely they are to *explain* why a line or
symbol looks the way it does today.  The score is a single float; higher is
more interesting.  It is not normalised — only the relative order matters.

### Signal design

Each signal answers a different question about the commit:

  1. **Diff size** — did this commit actually change a lot of code?
  2. **Message length** — did the author bother to explain what they did?
  3. **Keywords** — does the message use vocabulary associated with meaningful
     change (refactor, security, breaking, …)?
  4. **PR presence** — was this change reviewed?  PRs carry discussion context
     that rarely fits in a commit message.
  5. **Recency** — recent commits are more likely to explain current behaviour;
     old commits may describe code that has since been replaced.
  6. **Merge penalty** — merge commits are bookkeeping, not explanations.
  7. **Junk penalty** — housekeeping commits (typos, formatting, bumps) almost
     never explain intent.

### Why log-scaling for continuous signals

Raw diff size and message length have unbounded range: a refactor might touch
5 lines or 5 000.  Using the raw count would let a single huge diff swamp every
other signal.  ``log(1 + x)`` compresses the range so that going from 0→10
lines and 100→1 000 lines contribute roughly equally on the score axis, while
still preserving the direction (more is better).  The ``+1`` keeps the domain
non-negative: ``log(1 + 0) = 0``, so a zero-line commit contributes nothing.

### Why log-decay for recency

A commit from today is more relevant than one from five years ago, but the
relevance curve is not linear — a commit from last week is only marginally
more relevant than one from last month.  ``_RECENCY_MAX - log(1 + days/30)``
gives a bonus that starts at ~3 (day 0), falls to ~2.1 at 30 days, ~1.4 at
90 days, and reaches 0 at roughly 1 800 days (~5 years).  The outer
``max(0.0, …)`` clamps to zero so ancient commits get no bonus and no penalty
from this term.  ``days_ago`` is also clamped to 0 to prevent future-dated
commits (clock skew, timezone artefacts) from producing a negative argument
to ``log``.

### Why additive and not multiplicative

An additive model is easier to reason about and tune: each term has a clear
unit (points), penalties subtract from the same pool as bonuses, and the
constants table at the top of this file is the complete specification of the
model.  A multiplicative model would amplify the effect of every term and make
a zero in any signal collapse the whole score, which is not what we want (a
very recent, zero-line commit should still score above zero).
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

# Scoring weights and penalties — change these to tune the model.
#
#   _DIFF_WEIGHT   multiplier on log(1 + additions + deletions); doubling
#                  gives diff size twice the pull of message length
#   _PR_BONUS      flat bonus for any commit backed by a pull request
#   _RECENCY_MAX   ceiling of the recency bonus (reached at days_ago == 0)
#   _MERGE_PENALTY subtracted when is_merge is True
#   _JUNK_PENALTY  subtracted when subject matches a JUNK_PATTERN; chosen to
#                  be large enough to push junk commits below most real commits
#                  even with a strong recency bonus
#   _KEYWORD_BONUS added once per keyword found in subject + body
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

    # log(1 + lines) * _DIFF_WEIGHT
    # Large diffs indicate substantial work; log-scaled so a 1 000-line change
    # doesn't dwarf a 50-line change by 20x.
    s += log(1 + c.additions + c.deletions) * _DIFF_WEIGHT

    # log(1 + chars)
    # Authors who wrote a long message spent time explaining their reasoning —
    # a proxy for commit quality.  Same log-scaling rationale as diff size.
    s += log(1 + len(c.body) + len(c.subject))

    # +_KEYWORD_BONUS per keyword hit
    # Vocabulary in the subject/body is a strong signal: "security fix" or
    # "breaking change" almost always means the commit is important context.
    # Single-word keywords use \b word-boundary matching to avoid false
    # positives (e.g. "prefix" should not match "fix").  Multi-word keywords
    # like "add support" use plain substring matching because \b does not span
    # spaces.
    msg = (c.subject + " " + c.body).lower()
    s += sum(
        _KEYWORD_BONUS for kw in KEYWORDS
        if (re.search(r"\b" + re.escape(kw) + r"\b", msg) if " " not in kw else kw in msg)
    )

    # +_PR_BONUS (flat)
    # A PR means the change was reviewed and usually has a description, linked
    # issues, and discussion — all richer context than the commit message alone.
    if has_pr:
        s += _PR_BONUS

    # max(0, _RECENCY_MAX - log(1 + days_ago / 30))
    # Bonus decays from _RECENCY_MAX at day 0 to 0 at ~1 800 days.
    # The /30 stretches the decay across months rather than days so that a
    # commit from last week is not penalised much relative to one from yesterday.
    # days_ago is clamped to 0 so future-dated commits (clock skew, timezone
    # artefacts) don't pass a negative argument to log.
    days_ago = max(0, (now - c.date.date()).days)
    s += max(0.0, _RECENCY_MAX - log(1 + days_ago / 30))

    # -_MERGE_PENALTY
    # Merge commits record that a branch was integrated; they don't explain
    # why code was written.  The penalty pushes them below real commits even
    # when they have a non-trivial diff size (fast-forward squash merges).
    if c.is_merge:
        s -= _MERGE_PENALTY

    # -_JUNK_PENALTY
    # Typo fixes, formatting passes, version bumps, and chore commits contain
    # no design intent.  The penalty is intentionally large (_JUNK_PENALTY >
    # _RECENCY_MAX) so that even a today's formatting commit ranks below a
    # year-old real fix.
    if any(rx.match(c.subject.lower()) for rx in _JUNK_RES):
        s -= _JUNK_PENALTY

    return s
