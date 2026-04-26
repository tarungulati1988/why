# --brief Flag (Concise Mode)

**Issue:** #29  
**Date:** 2026-04-25  
**Status:** Approved

---

## Problem

The default `why` output is a full sectioned narrative. Users who want a quick orientation — not a deep archaeology — have no way to ask for a shorter answer.

---

## Desired Behaviour

`why <target> --brief` emits a 3-sentence summary instead of the full narrative. All other pipeline behaviour is unchanged:
- Timeline section still appended by `validate_and_repair_timeline()`
- `--verify` can be combined with `--brief` (user opts into the cost)
- Citation validation still runs

---

## Approach: Append tail in `build_why_prompt()` (Approach A)

Add `brief: bool = False` to `build_why_prompt()`. When `True`, append a brief-mode instruction as a final section in the user message before returning. The instruction overrides the format rules from the system prompt for this call.

The tail text (verbatim from issue spec):

```
## Output Format

Output ONLY a 3-sentence summary. No sections, no citations block (inline only).
```

---

## Implementation Plan

### 1. `src/why/prompts.py`

Add `brief: bool = False` to `build_why_prompt()`. When set, append the tail section to `content` before returning.

### 2. `src/why/synth.py`

Add `brief: bool = False` to `synthesize_why()`. Pass through to `build_why_prompt()`.

### 3. `src/why/cli.py`

Add `--brief` flag:

```python
@click.option(
    "--brief",
    is_flag=True,
    default=False,
    help="Emit a 3-sentence summary instead of the full narrative.",
)
```

Wire to `synthesize_why(..., brief=brief)`.

---

## Decisions

- **Timeline**: not suppressed — `validate_and_repair_timeline()` runs as normal
- **`--brief` + `--verify`**: both allowed; user opts into the extra LLM call
- **Prompt placement**: tail appended to user message (not system prompt) so it acts as a per-call format override

---

## Acceptance Criteria

- [ ] `--brief` flag on CLI
- [ ] Prompt appended with: `Output ONLY a 3-sentence summary. No sections, no citations block (inline only).`
- [ ] `synthesize_why(..., brief=True)` passes the flag to `build_why_prompt()`
- [ ] Single-pass mode without `--brief` is unchanged
- [ ] Tests: assert brief tail appears in `messages[0].content` when `brief=True`; assert it is absent when `brief=False`
