"""Key-commit scoring and selection for why.

Stage: scoring — runs after history retrieval and before diff fetching; filters
       the full commit list down to the most explanatory subset.

Inputs:
    commits — list[Commit] in any order (callers need not pre-sort).
    prs     — Mapping[sha, PRMetadata]; only SHA membership is tested (O(1)).
    n       — target result size (default 5).
    now     — reference date for recency scoring (default: today).

Outputs:
    list[Commit] — at most n commits sorted oldest-first. Two anchors are always
                   forced in: the oldest commit and the most-recent substantive
                   commit (score > _SUBSTANTIVE_THRESHOLD). This may push the
                   result above n when both anchors are distinct and n=1.

Scoring model (additive float, higher = more explanatory):
    + log(1 + additions + deletions) * 2.0   — diff size (log-scaled)
    + log(1 + len(subject) + len(body))       — message length (log-scaled)
    + 2.0 per keyword hit                     — refactor, fix, security, etc.
    + 3.0 if has_pr                           — PR presence bonus
    + max(0, 3.0 - log(1 + days_ago/30))      — recency bonus (log-decay)
    - 5.0 if merge commit                     — merge penalty
    - 10.0 if subject matches JUNK_PATTERNS   — housekeeping penalty

Notes:
    Log-scaling is used for diff size and message length to prevent a single
    massive commit from dominating all other signals. The model is additive
    (not multiplicative) so a zero in any one term doesn't collapse the score.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import date
from math import log

from .commit import Commit
from .prompts import PRMetadata

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
#   _KEYWORD_BONUS          added once per keyword found in subject + body
#   _SUBSTANTIVE_THRESHOLD  score floor for the "most recent substantive" anchor
#                           in select_key_commits; commits at or below this value
#                           are treated as bookkeeping/noise (exclusive boundary)
_DIFF_WEIGHT = 2.0
_PR_BONUS = 3.0
_RECENCY_MAX = 3.0
_MERGE_PENALTY = 5.0
_JUNK_PENALTY = 10.0
_KEYWORD_BONUS = 2.0
_SUBSTANTIVE_THRESHOLD = 3.0


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


def select_key_commits(
    commits: list[Commit],
    prs: Mapping[str, PRMetadata],  # SHA -> PRMetadata; only membership (sha in prs) is checked
    n: int = 5,
    now: date | None = None,
) -> list[Commit]:
    """Select the most explanatory commits from a history, targeting n results.

    Two anchors are always forced into the result regardless of score:
      1. The oldest commit — it establishes the original intent of the symbol.
      2. The most-recent substantive commit — explains current behaviour.

    Remaining slots are filled by top-scoring commits up to n total.
    The result is always sorted oldest-first, with no duplicates.

    Note: n is a target, not a hard cap. When both anchors are distinct and
    n=1, the function returns 2 commits so neither anchor is silently dropped.

    Parameters
    ----------
    commits:  candidate commits in any order (caller need not sort)
    prs:      mapping of SHA -> PR; only membership (sha in prs) is tested
    n:        target number of commits to return; must-includes may push result above n
    now:      reference date for recency scoring; defaults to today
    """
    if not commits:
        return []

    if now is None:
        now = date.today()

    # Cap n to the number of available commits so remaining_slots arithmetic
    # never over-allocates candidates beyond what exists.
    n = min(n, len(commits))

    # Sort by date ascending (oldest first) regardless of what git log returns.
    # git log defaults to newest-first, but callers shouldn't need to know that —
    # we normalise here so the rest of the function can assume chronological order.
    sorted_commits = sorted(commits, key=lambda c: c.date)

    # Edge case: single commit — no anchor logic needed.
    if len(sorted_commits) == 1:
        return [sorted_commits[0]]

    # Score every commit; PR lookup is O(1) per commit.
    scored: list[tuple[float, Commit]] = [
        (score_commit(c, now, c.sha in prs), c) for c in sorted_commits
    ]

    # Build a lookup from SHA to score for O(1) access later.
    score_by_sha: dict[str, float] = {c.sha: s for s, c in scored}

    # --- Must-include set ---

    # Anchor 1: oldest commit always explains where the symbol came from.
    oldest = sorted_commits[0]

    # Anchor 2: most-recent substantive commit — iterate newest→oldest and pick
    # the first one whose score exceeds _SUBSTANTIVE_THRESHOLD.
    # Fallback: if no commit clears the threshold (all junk/merges), we still
    # want the newest commit so the result doesn't look like it stopped in the past.
    most_recent_substantive: Commit | None = None
    for c in reversed(sorted_commits):
        if score_by_sha[c.sha] > _SUBSTANTIVE_THRESHOLD:
            most_recent_substantive = c
            break
    if most_recent_substantive is None:
        # Fallback: no substantive commit — use the most recent one anyway.
        most_recent_substantive = sorted_commits[-1]

    must_includes: set[str] = {oldest.sha, most_recent_substantive.sha}

    # --- Fill picked ---

    # Seed picked with must-includes (order doesn't matter here — final return
    # sorts by date, so no need for an intermediate score-order sort).
    picked: list[Commit] = [c for c in sorted_commits if c.sha in must_includes]
    picked_shas: set[str] = {c.sha for c in picked}

    # Fill remaining slots from top-scoring commits not already picked.
    remaining_slots = n - len(picked)
    if remaining_slots > 0:
        # Sort remaining candidates by score descending; take up to remaining_slots.
        candidates = sorted(
            [c for c in sorted_commits if c.sha not in picked_shas],
            key=lambda c: score_by_sha[c.sha],
            reverse=True,
        )
        picked.extend(candidates[:remaining_slots])

    # Return in chronological order (oldest first), with no duplicates.
    return sorted(picked, key=lambda c: c.date)
