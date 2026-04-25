# Why Prompt Template — Design Doc

**Date:** 2026-04-25  
**Issue:** #13  
**Status:** Draft

---

## Problem

The `why` CLI needs a prompt layer to turn raw git + diff data into a structured LLM call. Without it, the M1 synthesis pipeline has no input contract: no system prompt, no user message shape, and no citation enforcement.

---

## Goals

- Define `WHY_SYSTEM_PROMPT` (from issue #13 comment — verbatim)
- Define `CommitWithPR` — joins a `Commit` with its optional PR body
- Implement `build_why_prompt(target, current_code, key_commits)` → `list[Message]`
- Golden-file test covering a fixed, deterministic input

## Non-Goals

- PR fetching (GitHub API) — that's a future milestone
- Streaming output or multi-turn conversation
- Prompt versioning / A/B testing

---

## Design

### `CommitWithPR`

A thin frozen dataclass that joins a `Commit` with its resolved PR body. Lives in `prompts.py` because it's prompt-layer data, not core commit data.

```python
@dataclass(frozen=True)
class CommitWithPR:
    commit: Commit
    pr_body: str | None = None
```

Callers (the future CLI layer) call `select_key_commits()`, then resolve PR bodies from the GitHub API (or pass `None`), and produce `list[CommitWithPR]` before calling `build_why_prompt`.

---

### `WHY_SYSTEM_PROMPT`

Stored verbatim from the comment on issue #13. It instructs the model to:
- Act as a "code archaeology assistant"
- Ground every claim in a commit hash or PR ID
- Tag every reason `[STATED]` or `[INFERRED]`
- Follow an exact output format (Code Region → Key Evolutions → Why → Gaps → Confidence → Citations)
- Return a fixed failure string when history is insufficient

---

### `build_why_prompt` — Serialization Format

Returns `list[Message]` with a single `Message(role="user", ...)`. The content is **Markdown** — consistent with the system prompt's output format and readable by the LLM as natural prose structure.

```
## Target

File: `src/why/scoring.py`, Line: 42

---

## Current Code

```python
<current_code snippet>
```

---

## Commits

### `abc1234` — "fix: null check on token" · 2026-04-01 · Jane Smith

**Diff:**

```diff
<diff>
```

**PR Body:**

<pr body or "N/A">

---
```

Each commit section is a level-3 Markdown header (`###`) with the short SHA, subject, date, and author on a single line. Diff and PR body are fenced code blocks / prose under that header. Sections are separated by `---` horizontal rules to create clear parse boundaries.

**Why Markdown over XML tags:**
- The system prompt already uses Markdown headings for its OUTPUT FORMAT — feeding Markdown in is consistent
- Diffs contain `+`/`-` prefixes that would need escaping inside XML attributes but are natural in fenced blocks
- Markdown is more readable in tests and golden files

---

### Golden-file test

Stored at `tests/fixtures/prompts/why_prompt_golden.txt`. Contains the expected `content` string of the single returned `Message`.

Test pattern:
```python
def test_build_why_prompt_golden(update_goldens):
    result = build_why_prompt(target, current_code, key_commits)
    assert len(result) == 1
    assert result[0].role == "user"
    golden_path = Path("tests/fixtures/prompts/why_prompt_golden.txt")
    if update_goldens:
        golden_path.write_text(result[0].content)
    else:
        assert result[0].content == golden_path.read_text()
```

`--update-goldens` is a `conftest.py` fixture flag (addoption + fixture). No pytest plugin needed.

---

## File changes

| File | Change |
|------|--------|
| `src/why/prompts.py` | New — `WHY_SYSTEM_PROMPT`, `CommitWithPR`, `build_why_prompt` |
| `tests/test_prompts.py` | New — unit tests + golden-file test |
| `tests/fixtures/prompts/why_prompt_golden.txt` | New — golden output |
| `tests/conftest.py` | Add `--update-goldens` addoption + fixture |

---

## Acceptance criteria (from issue)

- [ ] `WHY_SYSTEM_PROMPT` defined in `src/why/prompts.py`
- [ ] `build_why_prompt(target, current_code, key_commits_with_prs)` returns `list[Message]`
- [ ] Prompt includes: target, current code snippet, per-commit (SHA, date, author, message, diff), PR bodies if present
- [ ] Prompt enforces citation format and `stated vs inferred` labeling (via system prompt)
- [ ] Golden-file test on prompt output for a fixed input
