# Design: LLM Provider Abstraction (Issue #12)

**Date:** 2026-04-24
**Milestone:** M1 - Core (git-only why)
**Depends on:** #1 (git wrapper — already shipped)

---

## Problem

The LLM synthesis step needs a stable call surface that is independent of which provider/model is in use. For the first iteration the provider is Groq; Anthropic and OpenAI will follow.

---

## Interface

```python
# src/why/llm.py

@dataclass
class Message:
    role: str      # 'user' | 'assistant'
    content: str

class LLMError(Exception): ...
class LLMMissingKeyError(LLMError): ...   # required env var absent
class LLMRateLimitError(LLMError): ...    # retries exhausted on 429
class LLMTimeoutError(LLMError): ...      # retries exhausted on timeout

class LLMClient:
    def __init__(
        self,
        model: str = "llama-3.3-70b-versatile",
        provider: str | None = None,   # defaults to WHY_LLM_PROVIDER env var, then "groq"
    ): ...

    def complete(self, system: str, messages: list[Message]) -> str: ...
```

---

## Provider routing

`provider` is resolved at construction time:

```
provider param → WHY_LLM_PROVIDER env var → "groq" (default)
```

| provider value | Backend | Required env var |
|---|---|---|
| `"groq"` | Groq SDK | `GROQ_API_KEY` |
| `"anthropic"` | *(NotImplementedError — M2)* | `ANTHROPIC_API_KEY` |
| `"openai"` | *(NotImplementedError — M2)* | `OPENAI_API_KEY` |

Unknown provider → `LLMError("unknown provider: ...")`.

---

## Groq backend

Uses the `groq` Python SDK (`groq` package — adds one runtime dep):

```python
import groq as groq_sdk

client = groq_sdk.Groq(api_key=os.environ["GROQ_API_KEY"])
response = client.chat.completions.create(
    model=self.model,
    messages=[{"role": "system", "content": system}, *converted_messages],
)
return response.choices[0].message.content
```

`LLMMissingKeyError` is raised at construction time if `GROQ_API_KEY` is absent (fail-fast, not at call time).

---

## Retry logic (hand-rolled)

3 retries, exponential backoff (1 s, 2 s, 4 s), on:
- `groq.RateLimitError` (HTTP 429)
- `groq.APIStatusError` with status 500 or 503
- `groq.APITimeoutError`

Non-retryable `APIStatusError` (e.g. 400 bad request, 401 auth) re-raises immediately as `LLMError`.

After retries exhausted: raises `LLMRateLimitError` or `LLMTimeoutError` wrapping the original.

```python
_RETRYABLE_STATUS = frozenset({429, 500, 503})
_MAX_RETRIES = 3
_BASE_DELAY = 1.0  # seconds; doubles each attempt: 1s, 2s, 4s
```

---

## Token logging

When `verbose=True` is passed to `complete()`, log to stderr:

```
[llm] model=llama-3.3-70b-versatile  prompt_tokens=312  completion_tokens=87  total=399
```

Uses `logging.getLogger("why.llm")` at DEBUG level (caller controls handler/level).

---

## Module structure

```
src/why/llm.py          # new module
tests/test_llm.py       # new test file
```

`pyproject.toml` gains one new runtime dep: `groq>=0.9`.
No new dev deps — tests mock via `unittest.mock`.

---

## Tests (`tests/test_llm.py`)

**Unit — mock `groq.Groq` client:**

1. `complete()` passes correct `model` + formatted messages to SDK
2. Returns the `choices[0].message.content` string
3. `GROQ_API_KEY` absent → `LLMMissingKeyError` at construction
4. Unknown provider → `LLMError`
5. `RateLimitError` retries 3× then raises `LLMRateLimitError`
6. Non-retryable `APIStatusError` (400) re-raises immediately (no retry)
7. `APITimeoutError` retries 3× then raises `LLMTimeoutError`
8. Successful response after 1 retry (transient failure) returns content
9. `verbose=True` logs token counts to stderr

**Sleep is mocked** (`time.sleep` patched to a no-op) in all retry tests so the suite stays fast.

---

## What this is NOT

- No streaming support — `complete()` returns full string only.
- No conversation history management — caller owns the `messages` list.
- No cost calculation — token counts are logged, not priced.
- Anthropic/OpenAI backends are stubs only (`NotImplementedError`).
