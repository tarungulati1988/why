# Issue #92 — Backend protocol refactor

## Problem

`LLMClient.complete()` in `src/why/llm.py:131-183` catches `groq_sdk.RateLimitError`, `groq_sdk.APIStatusError`, and `groq_sdk.APITimeoutError` directly inside its retry loop. Adding the OpenAI-compatible provider (#93) and Ollama `num_ctx` plumbing (#95) would either duplicate the retry logic per provider or leak Groq exceptions into non-Groq paths. Pure refactor — no behavior change.

## Decisions

1. **Package layout** — extract backends to `src/why/_backends/`. Underscore signals internal. Files: `__init__.py`, `base.py` (Protocol + `ChatResult`), `groq.py` (`GroqBackend`). Sets up cleanly for #93's `OpenAICompatibleBackend`.
2. **Backend signature** — forward-compatible: `chat(self, model: str, payload: list[dict], **extra: Any) -> ChatResult`. `**extra` lets #95 pass `extra_body={"options": {"num_ctx": N}}` without reopening this protocol. `GroqBackend` ignores unknown kwargs for now.
3. **Exception translation at the boundary** — backends translate provider exceptions into the existing `LLMError` hierarchy. The retry loop catches only `LLMRateLimitError | LLMServerError | LLMTimeoutError`. Non-retryable provider errors become a plain `LLMError` and propagate immediately.
4. **Constants stay module-level in `llm.py`** — `_RETRYABLE_STATUS`, `_MAX_RETRIES`, `_BASE_DELAY` keep their current location. The retry loop lives in `LLMClient.complete()`.
5. **Verbose token logging stays in `LLMClient.complete()`** — `ChatResult` carries `prompt_tokens` / `completion_tokens` so the client logs them after a successful call. Behavior at `llm.py:139-147` is preserved.

## Shape

```python
# src/why/_backends/base.py
from dataclasses import dataclass
from typing import Any, Protocol

@dataclass
class ChatResult:
    content: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None

class Backend(Protocol):
    def chat(self, model: str, payload: list[dict], **extra: Any) -> ChatResult: ...
```

```python
# src/why/_backends/groq.py
import groq as groq_sdk
from why.llm import LLMError, LLMRateLimitError, LLMServerError, LLMTimeoutError, _RETRYABLE_STATUS
from ._backends.base import Backend, ChatResult  # actual import will be relative

class GroqBackend:
    def __init__(self, api_key: str) -> None:
        self._client = groq_sdk.Groq(api_key=api_key)

    def chat(self, model: str, payload: list[dict], **_extra: Any) -> ChatResult:
        try:
            r = self._client.chat.completions.create(model=model, messages=payload)
        except groq_sdk.RateLimitError as e:
            raise LLMRateLimitError(str(e)) from e
        except groq_sdk.APITimeoutError as e:
            raise LLMTimeoutError(str(e)) from e
        except groq_sdk.APIStatusError as e:
            if e.status_code in _RETRYABLE_STATUS:
                raise LLMServerError(f"status {e.status_code}") from e
            raise LLMError(f"API error {e.status_code}") from e
        content = r.choices[0].message.content
        if content is None:
            raise LLMError("model returned no text content")
        u = r.usage
        return ChatResult(
            content=content,
            prompt_tokens=u.prompt_tokens if u else None,
            completion_tokens=u.completion_tokens if u else None,
        )
```

The circular-import risk between `llm.py` and `_backends/groq.py` (errors live in `llm.py`, backend imports them) is acceptable because `_backends/groq.py` is only imported lazily inside `LLMClient.__init__` after the module has fully loaded. Alternative: move the exception hierarchy to `_backends/errors.py` — rejected as scope creep for a protocol-only refactor.

## Test strategy

- Existing `tests/test_llm.py` (12 cases) **must pass unmodified** — they patch `why.llm.groq_sdk.Groq`, which still works because `GroqBackend` instantiates `groq_sdk.Groq(api_key=...)` exactly as before. Verify by re-importing path: `why.llm.groq_sdk` symbol must remain.
  - **Subtle:** when `GroqBackend` moves to `_backends/groq.py`, the patch target `why.llm.groq_sdk.Groq` no longer covers it. Two paths:
    - **Option A:** keep `import groq as groq_sdk` at top of `why.llm` for backwards-compat, even if unused there.
    - **Option B:** rename patches in tests.
  - AC says "tests pass without modification" → **Option A**. Re-export `groq_sdk` from `why.llm` and import the same alias inside `_backends/groq.py`. Patching `why.llm.groq_sdk.Groq` still mutates the same module-level binding that `_backends/groq.py` references via `from why.llm import groq_sdk` (or equivalent). Need to verify this works in Python's import machinery.
  - **Safer Option C:** keep `GroqBackend` defined in `why.llm` (still re-exported via `_backends/__init__.py`) so the existing patch path is untouched. Hybrid: only `Backend` protocol + `ChatResult` move to `_backends/base.py`; backends stay in `llm.py` for now. Trades the (b) win for test stability.

  **Recommendation:** start with Option C. We get the protocol/dataclass abstraction (the actual blocker for #93) without disturbing existing tests. When #93 lands, `OpenAICompatibleBackend` goes into `_backends/openai_compatible.py`. Eventually we may move `GroqBackend` too, but that move happens with an explicit test-rewrite ticket, not silently here.

- New tests added in `tests/test_llm.py`:
  1. **Backend retry success after transient failures** — inject a fake `Backend` that raises `LLMRateLimitError` 3× then returns a `ChatResult`. Assert: `time.sleep` was called with `[1.0, 2.0, 4.0]` (in order), final result content matches.
  2. **Non-retryable LLMError propagates** — fake `Backend` raises `LLMError("API error 400")` once. Assert: raised immediately, `time.sleep` never called, backend `chat` called exactly once.

## Strides

| # | Behavior | Test paths | Impl paths | Depends |
|---|----------|-----------|------------|---------|
| 1 | `ChatResult` dataclass + `Backend` protocol exist; importable from `why._backends.base` | `tests/test_backends_base.py` (new — protocol smoke test, ChatResult equality) | `src/why/_backends/__init__.py`, `src/why/_backends/base.py` | — |
| 2 | `LLMClient` uses internal backend object; existing 12 tests still pass; verbose logging still works | existing `tests/test_llm.py` (unchanged) | `src/why/llm.py` (extract Groq translation into inline `GroqBackend` class, retry loop now catches only `LLMError` subclasses, calls `self._backend.chat(...)`) | 1 |
| 3 | Retry loop is provider-agnostic — verified against fake backend | `tests/test_llm.py` (add 2 cases: fake backend retries succeed; fake backend non-retryable propagates) | none (test-only) | 2 |

Stride 2 is the load-bearing change. Strides 1 and 3 bookend it: 1 establishes the abstraction surface; 3 proves the retry loop no longer depends on Groq.

## Out of scope

- New providers (#93+).
- `extra_body` plumbing in `LLMClient.complete()` (just the protocol accepts `**extra`; client doesn't pass anything yet).
- Moving `GroqBackend` to its own file (deferred — see Option C above).
- Changing public `LLMClient.complete()` signature.

## Risks

- **Test breakage from import path moves.** Mitigated by Option C: `GroqBackend` stays in `llm.py`, only protocol/dataclass move to `_backends/base.py`.
- **Verbose logging regression.** `ChatResult` doesn't include `total_tokens`. Current log line uses `usage.total_tokens`. Fix: log `prompt_tokens + completion_tokens` (still informative; total field was redundant).
