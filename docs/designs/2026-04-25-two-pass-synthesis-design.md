# Two-Pass Synthesis — Self-Grounding Check

**Issue:** #72  
**Date:** 2026-04-25  
**Status:** Approved

---

## Problem

The current `synthesize_why()` pipeline makes a single LLM call. The post-processing
`validate_citations` step catches hallucinated SHAs, but has no mechanism to detect
**ungrounded intent claims** — statements about team decisions or motivations that
aren't actually supported by any commit message or PR body in context.

Examples of claims the citation validator cannot catch:
- "the team chose this for performance reasons" (no commit says that)
- "this was added to handle a race condition" (inferred, not stated)
- "the original design decision was to avoid X" (pure hallucination)

---

## Desired Behaviour

An optional second LLM call evaluates each intent claim in the first-pass explanation
against the original commit/PR context. The result is appended as a
`## 🔍 Grounding Check` section — the main narrative is never modified.

Single-pass mode (the default) is unchanged in behaviour and latency.

---

## Approach: Appended Grounding Report (Approach B)

The grounding pass follows the same pattern as `validate_and_repair_timeline`:
it is **additive** — it appends a structured section rather than rewriting the
first-pass explanation. This keeps the main narrative intact and makes the
grounding output easy to test (section is present or absent; content is parseable).

### Pipeline with `two_pass=True`

```
synthesize_why(..., two_pass=True)
  │
  ├─ [pass 1 — unchanged]
  │    build_why_prompt → llm.complete → first_pass
  │    validate_citations(first_pass, known_shas)
  │    validate_and_repair_timeline(first_pass, commits_with_prs, repo_url)
  │
  └─ [pass 2 — new]
       build_grounding_prompt(first_pass, commits_with_prs)
       llm.complete(GROUNDING_SYSTEM_PROMPT, grounding_messages)
       → grounding_section ("## 🔍 Grounding Check\n\n...")
       → appended to result
```

### Grounding Section Format

```markdown
## 🔍 Grounding Check

| Claim | Grounded? | Evidence |
|-------|-----------|----------|
| "the team chose X for performance" | ⚠️ Not supported | No commit or PR mentions performance |
| "added to handle a race condition" | ✅ Supported | abc1234 — "fix: prevent concurrent write" |
```

A claim is **supported** when it can be traced to specific text in a commit subject,
commit body, or PR body. A claim is **not supported** when the LLM inferred it from
code structure alone without acknowledging that inference.

---

## Implementation Plan

### 1. `src/why/prompts.py`

Add:
- `GROUNDING_SYSTEM_PROMPT: str` — instructs the LLM to act as a fact-checker,
  evaluate each intent claim, and return a Markdown table.
- `build_grounding_prompt(first_pass: str, commits: list[CommitWithPR]) -> list[Message]`
  — builds the user message: the first-pass explanation + the same commit context.

### 2. `src/why/synth.py`

- Add `two_pass: bool = False` parameter to `synthesize_why()`.
- When `two_pass=True`, after the existing post-processing, call
  `build_grounding_prompt` + `llm.complete` and append the result to the output.

### 3. `src/why/cli.py`

Add `--verify` flag:

```python
@click.option(
    "--verify",
    is_flag=True,
    default=False,
    help=(
        "Enable two-pass grounding check. A second LLM call evaluates each intent "
        "claim for evidence support and appends a Grounding Check section. "
        "Adds ~1 LLM call of latency and cost."
    ),
)
```

Wire to `synthesize_why(..., two_pass=verify)`.

---

## Acceptance Criteria

- [ ] `synthesize_why(target, repo, llm, two_pass=True)` makes a second LLM call
- [ ] Output with `two_pass=True` contains a `## 🔍 Grounding Check` section
- [ ] Ungrounded claims are flagged with ⚠️; grounded claims marked ✅
- [ ] Single-pass mode (default) produces identical output to today
- [ ] `why <target> --verify` invokes two-pass mode via the CLI
- [ ] Help text documents the latency/cost trade-off

---

## Out of Scope

- Modifying or rewriting the first-pass narrative (Approach A) — deferred
- Structured JSON extraction of claim metadata (Approach C) — deferred
- Grounding check for the Timeline section (SHA validation already covers this)
- Strict mode that fails when ungrounded claims are found
