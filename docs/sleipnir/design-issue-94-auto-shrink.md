# Design — Issue #94: Auto-shrink commits and diffs for context-constrained providers

**Issue:** [#94](https://github.com/tarungulati1988/why/issues/94)
**Branch:** `sleipnir/issue-94-auto-shrink-context`
**Depends on:** #93 (merged) — openai-compatible provider
**Scope:** small (~1-3h)

## Problem

Local 3B-class models behind `openai-compatible` (Ollama default `num_ctx=2048`, qwen2.5:3b) degrade sharply once the prompt fills their window. `why`'s prompt routinely runs 8K–20K tokens. Without shrinking, users see truncated mid-sentence answers or the model regressing into hallucination.

The user-facing fix is auto-shrink with a visible warning, not silent failure.

## Decisions

### 1. Hook location: `src/why/synth.py`, after `commits_with_prs` is built

Insert between commit assembly (synth.py:250) and prompt build (synth.py:252). `system_prompt` is currently built *after* `messages` — reorder so it's available for the budget computation.

### 2. Env var with auto-default for openai-compatible

`WHY_LLM_MAX_CTX` semantics:

| State | Behavior |
|---|---|
| Set to positive int | Shrink to that target |
| Set to `0` | Disable (escape hatch) |
| Unset, provider=`openai-compatible` | **Default to 4096** |
| Unset, provider=`groq` | Disable (no shrinking) |

Provider is read from the same source as `LLMClient`: `WHY_LLM_PROVIDER` env, default `groq`. We do **not** thread the `LLMClient` instance into `synthesize_why` for this — the env-var read is local and deterministic.

### 3. Naive char/4 token estimate

`len(text) // 4`. No `tiktoken` dependency. The issue's out-of-scope list explicitly rejects real tokenization and the 15% headroom (`target * 0.85`) absorbs estimate error.

### 4. Diff truncation: 80 lines + sentinel

Per issue spec. `"\n".join(lines[:80]) + f"\n... [truncated {N-80} lines]"`. No hunk-aware logic in v1.

### 5. Drop oldest first

Per issue spec. Trade-off acknowledged: oldest commit is often the most informative ("why does this exist?") commit. Future issue can add relevance ranking if user testing shows it matters.

### 6. Combined warning, not two lines

When auto-default kicked in **and** something was dropped/truncated, single stderr line includes the disable hint:

```
⚠ Auto-shrunk for local model: dropped N commit(s), truncated M diff(s). Set WHY_LLM_MAX_CTX=0 to disable.
```

When the user explicitly set `WHY_LLM_MAX_CTX`, the disable hint is omitted (they already know).

### 7. `--max-commits` ordering

User cap (`max_commits`) is applied during history selection (synth.py:191) — already happens before shrinking. No code change needed; just verify in tests.

## File layout

```
src/why/synth.py        Add _estimate_tokens, _shrink_for_budget, _resolve_max_ctx;
                        wire into synthesize_why between commit assembly and prompt build
tests/test_synth.py     Extend with shrink tests
                        (or new tests/test_shrink.py if test_synth.py is crowded)
README.md               Document WHY_LLM_MAX_CTX in the provider section
```

## Acceptance

- `WHY_LLM_MAX_CTX=2048` with prompt that needs shrinking → drops oldest commits, truncates diffs >80 lines, emits warning.
- `WHY_LLM_MAX_CTX=2048` with prompt that already fits → no warning, no mutation.
- Provider=`openai-compatible`, `WHY_LLM_MAX_CTX` unset → defaults to 4096.
- Provider=`openai-compatible`, `WHY_LLM_MAX_CTX=0` → no shrinking (escape hatch).
- Provider=`groq`, `WHY_LLM_MAX_CTX` unset → no shrinking (current behavior).
- `--max-commits 3` + shrink budget that fits 5 → 3 commits passed to shrink, no further drops.
- Diff with 200 lines → truncated to 80 + sentinel.

## Strides

```
Stride 1: _estimate_tokens + _shrink_for_budget pure helpers
  test: tests/test_shrink.py (NEW)
  impl: src/why/synth.py (add helpers)
  depends: []

Stride 2: _resolve_max_ctx env-var resolver with provider-aware default
  test: tests/test_shrink.py (extend)
  impl: src/why/synth.py (add helper)
  depends: []

Stride 3: Wire into synthesize_why + warning emission
  test: tests/test_synth.py (extend) or tests/test_shrink.py integration block
  impl: src/why/synth.py (call site, system_prompt reorder)
  depends: [1, 2]

Stride 4: README documentation
  test: (skip — docs only; covered by Stride 3 code paths)
  impl: README.md
  depends: [3]
```

Wave 1: Strides 1 + 2 (independent helpers in same file — dispatch sequentially to avoid edit conflicts). Wave 2: Stride 3. Wave 3: Stride 4.

## Out of scope

- Map-reduce summarization (issue rejects).
- Real tokenization via `tiktoken` (issue rejects).
- Relevance-ranked drop order (Approach B; deferred).
- Hunk-aware diff truncation (deferred).
- `num_ctx` plumbing into the openai-compatible request (separate issue #95).
- Citation guardrails for weaker models (separate issue #96).
