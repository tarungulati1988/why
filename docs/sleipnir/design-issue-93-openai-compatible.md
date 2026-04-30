# Design — Issue #93: OpenAI-compatible LLM provider

**Issue:** [#93](https://github.com/tarungulati1988/why/issues/93)
**Branch:** `sleipnir/issue-93-openai-compatible`
**Depends on:** #92 (merged) — Backend Protocol + retry loop refactor
**Scope:** small (~1-3h)

## Problem

`why` only works with Groq today. Users want local synthesis (Ollama, llama.cpp `server`, LM Studio, vLLM, HF TGI) for privacy / cost / offline. All five expose the same OpenAI-compatible `/v1/chat/completions` shape, so one connector covers them all.

The provider-agnostic retry loop (#92) made this trivial: add a class implementing `Backend`, wire one new branch in `LLMClient.__init__`.

## Decisions

### 1. Location: `src/why/_backends/openai_compatible.py`

The `_backends/` package was created in #92 specifically to host new providers. `GroqBackend` was kept inline only because existing tests patch `why.llm.groq_sdk.Groq` — no such constraint for `openai`. Use the package as intended.

`llm.py` imports the new backend lazily inside the `elif` branch to keep `openai` an importable-but-not-imported dependency for users who only use Groq.

### 2. Connection errors retry: `openai.APIConnectionError → LLMServerError`

Local servers restart and crash far more than Groq. Mapping connection errors to the retryable bucket gives `ollama serve` a 7-second window (1s + 2s + 4s) to come back. Bounded — won't hang forever, won't fail on a 200ms blip.

`openai.RateLimitError`, `APITimeoutError`, retryable `APIStatusError` (429/500/503) follow the same pattern as `GroqBackend`. Non-retryable status codes raise plain `LLMError`.

### 3. `ChatResult` populates all three token fields

`GroqBackend` sets `prompt_tokens`, `completion_tokens`, **and** `total_tokens`. Mirror that. The verbose-logging path in `LLMClient.complete()` prefers `result.total_tokens` over the computed sum; skipping it forces the fallback unnecessarily.

### 4. Defaults

- `WHY_LLM_API_KEY` defaults to literal `"not-needed"` — Ollama ignores it, but the OpenAI SDK rejects empty.
- `WHY_LLM_BASE_URL` is **required** when `WHY_LLM_PROVIDER=openai-compatible`. Missing → `LLMMissingKeyError("WHY_LLM_BASE_URL not set")` at construction.
- No normalization of `base_url` (don't auto-append `/v1`). Trust user input; Ollama users will pass `http://localhost:11434/v1`, llama.cpp users may pass a different path.

### 5. Dependency

Add `openai>=1.40` to `[project.dependencies]` in `pyproject.toml`. Required (not optional/extras) — keeps install instructions simple and the package is small (~1MB wheel).

## File layout

```
src/why/_backends/
  base.py                    (unchanged from #92)
  openai_compatible.py       NEW — OpenAICompatibleBackend
  __init__.py                add re-export
src/why/llm.py               replace NotImplementedError("openai")
                             branch with openai-compatible wiring
tests/test_openai_compatible_backend.py  NEW
tests/test_llm.py            extend: provider resolution + missing base_url
pyproject.toml               add openai>=1.40
```

## Acceptance

- `LLMClient(provider="openai-compatible")` with `WHY_LLM_BASE_URL` set → constructs.
- Same without `WHY_LLM_BASE_URL` → `LLMMissingKeyError`.
- Mocked `openai.OpenAI`: `complete()` returns content; `RateLimitError` / `APIConnectionError` retried per loop; non-retryable status raised immediately.
- Default provider unchanged (Groq).
- `--model` default unchanged (`llama-3.3-70b-versatile`).

## Strides

```
Stride 1: OpenAICompatibleBackend exists and translates errors
  test: tests/test_openai_compatible_backend.py
  impl: src/why/_backends/openai_compatible.py, src/why/_backends/__init__.py
  depends: []

Stride 2: LLMClient wires openai-compatible provider
  test: tests/test_llm.py (extend)
  impl: src/why/llm.py
  depends: [1]

Stride 3: Add openai dependency
  test: (skip — config only; covered by Stride 1's import path)
  impl: pyproject.toml
  depends: []
```

Wave 1: Strides 1 + 3 (independent files). Wave 2: Stride 2.

## Out of scope

Documentation, `num_ctx`/context-window plumbing, citation guardrails, smoke-test against a live Ollama. Per issue, those are tracked separately (#94-#97).
