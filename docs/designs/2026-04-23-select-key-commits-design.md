# Design: Key-commit selector (`select_key_commits`)

**Date:** 2026-04-23
**Issue:** #10 — Key-commit selector
**Milestone:** M1 — Core (git-only why)
**Status:** Implemented

---

## Problem

Given a scored list of commits, we need to surface the N most useful ones for explaining why a symbol looks the way it does today. Raw top-N by score is insufficient: the oldest commit (origin of the symbol) and the most recent substantive commit (current behaviour) are always critical context regardless of their score rank.

---

## Approaches considered

### A — Pure top-N by score (no anchors)

Select the `n` highest-scoring commits and return them sorted oldest-first. Simple, predictable.

**Pro:** Clean contract; `n` is always honoured exactly.
**Con:** The oldest commit is often a low-scoring "initial commit" or placeholder and would be dropped. The most recent commit explaining today's behaviour could also fall outside top-N if the repo has a burst of recent high-scoring activity. Result would feel incomplete to the user.

### B — Anchors as pre-allocated slots

Reserve 2 of the `n` slots for anchors (oldest + most-recent-substantive), fill the rest with top scorers from the remainder.

**Pro:** `n` is always honoured exactly.
**Con:** When `n=2` you get zero filler commits. When `n=1` you must drop one anchor — forces an arbitrary tie-break. Anchor selection and filler selection become coupled, making the logic harder to follow.

### C — Anchors always win, then fill to n (chosen)

Score all commits. Force two anchors into the result regardless of score. Fill remaining capacity (`max(0, n - len(anchors))`) with top scorers. Result may exceed `n` only when both anchors are distinct and `n=1` — documented behaviour, not a bug.

**Pro:** Anchors are always present (correct product behaviour). Logic is easy to reason about: anchors first, then fillers. No tie-breaking needed.
**Con:** `n` is a soft cap, not a hard maximum. Callers relying on a strict upper bound must be warned. Docstring must state this clearly.

**Decision: C.** The guarantee that both anchors appear is more valuable than strict `n` enforcement. `n=1` with two distinct anchors returning 2 commits is the correct trade-off for this use case.

---

## Design decisions

### D1 — Substantive threshold: `_SUBSTANTIVE_THRESHOLD = 3.0`

A commit is "substantive" if `score > 3.0`. The recency ceiling is `_RECENCY_MAX = 3.0`, so a today-commit with zero diff and no keywords scores exactly 3.0 — just under the threshold. This intentionally excludes empty/trivial recent commits from the anchor.

Boundary is **exclusive** (`>`). A commit scoring exactly 3.0 is not substantive. This is intentional and mirrors the existing scoring model's design where 3.0 is the no-signal baseline.

**Fallback:** If no commit clears the threshold (e.g. repo of only junk commits), fall back to `commits[-1]` (most recent after sort). A "most recent" anchor is always more useful than no anchor.

### D2 — Must-includes win over `n`

Anchors are added before filler. `remaining_slots = n - len(picked)` goes negative when both anchors are distinct and `n=1`; the filler loop is skipped (`candidates[:negative] == []`). Result has 2 commits, not 1. Documented in the function docstring.

### D3 — Defensive internal sort

Input is sorted by `c.date` ascending (oldest first) before any selection. This makes `select_key_commits` order-agnostic — callers coming from `git log` (newest-first) and callers coming from a DB (any order) both get consistent results. The input list is not mutated.

---

## Implementation

**File:** `src/why/scoring.py`

```
_SUBSTANTIVE_THRESHOLD = 3.0   ← should live in constants block (known debt)

select_key_commits(commits, prs, n=5, now=None) -> list[Commit]
  1. Defensive sort by date ascending
  2. Score all commits; build score_by_sha dict (O(1) lookup)
  3. Anchor 1: sorted_commits[0] (oldest) if len > 1
  4. Anchor 2: newest commit with score > threshold; fallback to sorted_commits[-1]
  5. must_includes = {anchor1.sha, anchor2.sha}
  6. picked = must_includes + top-(n-2) fillers (no duplicates)
  7. return sorted(picked, key=date)
```

**Tests:** 10 new tests in `tests/test_scoring.py` covering empty input, single commit, always-oldest anchor, substantive anchor, no-substantive fallback, no duplicates, sorted output, n-cap-must-include-win, PR boost, and order-agnostic input.

---

## Known issues / follow-up

- `_SUBSTANTIVE_THRESHOLD` should be moved to the constants block alongside `_DIFF_WEIGHT`, `_PR_BONUS`, etc. (practices finding from review)
- `test_select_with_prs` passes via anchor selection rather than PR-bonus lift; should be tightened if filler selection semantics change
- `n <= 0` is undocumented; returns anchors instead of `[]`
