# Design Doc — `## 📊 Timeline` ASCII Block (Issue #71)

**Date:** 2026-04-25
**Issue:** https://github.com/tarungulati1988/why/issues/71
**Status:** Approved

---

## Problem

The `why` LLM output is prose-heavy. Visual thinkers have no quick scannable view of how code evolved chronologically. We need a `## 📊 Timeline` section appended to every response — a compact ASCII diagram that renders in both terminal and GitHub markdown.

**Hard constraint:** Timeline nodes must only reference commits/PRs present in the provided context. No hallucination.

---

## Non-Goals

- Mermaid diagrams (doesn't render in terminal)
- Clickable hyperlinks in the ASCII block (not universally supported)
- Phase grouping when there are fewer than 3 commits (not enough to group)

---

## Approaches Considered

### A — Prompt-only (LLM generates timeline freely)

Add instructions to `build_system_prompt()` telling the LLM to append the timeline. The LLM reads commit data from the user message and formats it.

**Pros:** Simple — one place to change. LLM can do smart phase grouping.
**Cons:** Non-deterministic; LLM may drift, abbreviate SHAs incorrectly, or invent dates under token pressure. Fails the no-hallucination constraint.

### B — Deterministic post-processing (Python generates timeline)

Generate the ASCII block directly in Python from `key_commits`, append to the LLM response as a post-processing step.

**Pros:** Zero hallucination risk. Deterministic, easily unit-tested.
**Cons:** No smart phase grouping. LLM narrative stays separate from timeline.

### C — Hybrid (LLM groups, Python grounds + validates) ✅ CHOSEN

Provide a pre-formatted `## Timeline Data` section in the user message with exact SHAs/dates/subjects. System prompt instructs the LLM to copy those entries verbatim, only adding phase grouping labels. A post-processing validator checks that every SHA in the timeline exists in `key_commits`; if not, replaces the block deterministically.

**Pros:** Smart phase grouping (LLM contribution) with zero hallucination risk (Python validation + fallback).
**Cons:** Slightly more code — need both the prompt change and the validator.

---

## Chosen Design — Approach C

### ASCII format

Wrapped in a ` ```text ` fence so it renders identically in terminal and GitHub markdown:

```
## 📊 Timeline

\`\`\`text
2026-01-10  44be1b8  Initial scaffold
2026-02-01  a1b2c3d  Citation validator added          [PR #63]
             --- Phase: Resilience work ---
2026-03-15  ef1f022  Sparse history fallback           [PR #69]
\`\`\`
```

Each row: `<YYYY-MM-DD>  <short_sha>  <description>  [PR #N]`
- PR column is optional, appended when a PR number is known
- LLM may insert `--- Phase: <label> ---` separator rows between major phases
- Short SHA is always 7 characters (matches `commit.short_sha`)

### Moving Parts

#### 1. `build_why_prompt` — `## Timeline Data` section

A new helper `_render_timeline_data(key_commits, repo_url)` produces:

```
## Timeline Data

(Copy SHAs and dates verbatim into the timeline — do not alter them)

2026-01-10  44be1b8  Initial scaffold
2026-02-01  a1b2c3d  Citation validator added  [PR #63]
2026-03-15  ef1f022  Sparse history fallback  [PR #69]
```

This section is appended after `## Commits` in the user message. It is the primary hallucination guard — the LLM copies from it rather than recalling from memory.

#### 2. `build_system_prompt` — timeline instruction

A new block appended after the hard constraints section:

> **Appending the Timeline**
>
> After the narrative, always append a `## 📊 Timeline` section. Use the `## Timeline Data` entries from the user message as the source of truth — copy dates and short SHAs verbatim; do not invent or alter them. You may rewrite the description to be more concise. You may insert `--- Phase: <label> ---` separator rows between major phases if the history warrants it (3+ commits with a clear inflection point). Wrap the block in a ` ```text ` fence.
>
> If there are no commits (sparse history with 0 commits), emit:
> `## 📊 Timeline`
> `No commit history available.`

#### 3. Post-processing validator — `src/why/timeline.py` (new file)

```python
def validate_and_repair_timeline(
    response: str,
    key_commits: list[CommitWithPR],
    repo_url: str | None = None,
) -> str:
    """Validate timeline SHAs; replace block with deterministic fallback if invalid."""
```

Logic:
1. Extract the ` ```text ` block under `## 📊 Timeline` via regex
2. Parse every token that looks like a 7-char hex SHA
3. Check each against `{c.commit.short_sha for c in key_commits}`
4. If any SHA is unrecognized → call `_render_deterministic_timeline(key_commits)` and replace the block
5. If no timeline section present → append the deterministic fallback
6. Return (possibly repaired) response

`_render_deterministic_timeline` generates the ASCII table directly from `key_commits` — no LLM, no phase grouping, always correct.

### Files Touched

| File | Change |
|------|--------|
| `src/why/prompts.py` | Add `_render_timeline_data()` helper; append section in `build_why_prompt`; add timeline instruction to `build_system_prompt` |
| `src/why/timeline.py` | New file — `validate_and_repair_timeline`, `_render_deterministic_timeline` |
| `src/why/synth.py` | Call `validate_and_repair_timeline` on LLM response before returning |
| `tests/test_prompts.py` | Tests for `_render_timeline_data`: empty commits, PR numbers, no PR |
| `tests/test_timeline.py` | Tests for validator: valid pass-through, hallucinated SHA triggers replacement, missing section triggers append |

### Responsibility Split

| Responsibility | Owner |
|---|---|
| Phase grouping / phase label names | LLM |
| Description rewrite (concise) | LLM |
| SHAs, dates, PR numbers | Python-provided (copied verbatim) |
| Fallback when validation fails or section missing | Python (deterministic) |

---

## Acceptance Criteria

- [ ] Output always ends with `## 📊 Timeline` block
- [ ] Timeline nodes only reference commits present in `key_commits`
- [ ] Renders correctly in GitHub markdown preview (fenced `text` block)
- [ ] Renders correctly in terminal (plain text in code fence)
- [ ] Validator replaces hallucinated SHAs with deterministic fallback
- [ ] Existing tests pass
