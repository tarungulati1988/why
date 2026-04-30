# Design — Issue #95: Pipe num_ctx to Ollama via OpenAI extra_body

**Issue:** [#95](https://github.com/tarungulati1988/why/issues/95)
**Branch:** `sleipnir/issue-95-num-ctx-ollama`
**Depends on:** #93 (merged) — openai-compatible provider; #94 (merged) — auto-shrink
**Scope:** small (~1-2h)

## Problem

Ollama's default `num_ctx=2048` silently truncates prompts above 2048 tokens. `why`'s prompts (post-shrink) target 4096 by default for `openai-compatible`. Without raising `num_ctx`, the auto-shrink work from #94 still hits a wall: the prompt fits the *budget* but exceeds the *server's* allocated window.

Other OpenAI-compatible servers (vLLM, TGI, llama.cpp, LM Studio) ignore unknown options inside `extra_body`, so the change is harmless there.

## Decisions

### 1. New env var `WHY_LLM_NUM_CTX`

| State | Behavior |
|---|---|
| Set to positive int | Use as `num_ctx` in `extra_body` |
| Set to `0`, negative, or non-int | `LLMError` at `LLMClient` construction |
| Set to empty string | Treated as unset |
| Unset, provider=`openai-compatible`, `WHY_LLM_MAX_CTX` resolves to a positive int | **Auto-couple**: use that value as `num_ctx` |
| Unset, provider=`openai-compatible`, `WHY_LLM_MAX_CTX` resolves to `None` (disabled) | No `extra_body` |
| Unset, provider=`groq` | No effect |

The auto-couple closes the loop opened by #94: the auto-default `WHY_LLM_MAX_CTX=4096` exists *because* of small-context models; sending a 4096-token prompt to a server still pinned at `num_ctx=2048` defeats the point.

### 2. Resolution lives on `LLMClient`, not the backend

`OpenAICompatibleBackend.__init__` takes a plain `num_ctx: int | None`. `LLMClient` does the env reading, validation, and auto-couple computation, then passes the resolved int (or None) to the backend. Mirrors the pattern from #93/#94: backend stays a dumb transport; policy lives in `LLMClient`.

The auto-couple needs `_resolve_max_ctx` from #94 — `LLMClient` imports it from `why.synth`. Acceptable inward dependency: `synth` is already imported by callers of `LLMClient`, and `_resolve_max_ctx` is pure.

### 3. Validation: positive int only

```python
raw = os.getenv("WHY_LLM_NUM_CTX")
if raw is not None and raw != "":
    try:
        value = int(raw)
        if value < 1:
            raise ValueError
    except ValueError:
        raise LLMError("WHY_LLM_NUM_CTX must be a positive integer")
    num_ctx = value
```

Strict (not soft like MAX_CTX) because num_ctx has no semantic disable — `0` is always a user mistake. Empty string treated as unset (matches `_resolve_max_ctx`).

### 4. Backend wires `extra_body` only when set

```python
def chat(self, model, payload, **_extra) -> ChatResult:
    kwargs: dict[str, Any] = {"model": model, "messages": payload}
    if self._num_ctx is not None:
        kwargs["extra_body"] = {"options": {"num_ctx": self._num_ctx}}
    r = self._client.chat.completions.create(**kwargs)
    ...
```

Unconditionally including `extra_body={}` would be a no-op for Ollama but adds a request body field for vLLM/TGI; gating on `is not None` keeps the wire format identical to today when unset.

### 5. Verbose log line

When `verbose=True` and `num_ctx` is set, log once per `chat()` at DEBUG via the existing `why.llm` logger:
`num_ctx=N (Ollama)`

Per issue AC.

## File layout

```
src/why/_backends/openai_compatible.py   Add num_ctx param; conditional extra_body
src/why/llm.py                           Resolve+validate WHY_LLM_NUM_CTX; auto-couple
                                         to WHY_LLM_MAX_CTX; pass to backend
tests/test_llm.py                        Extend with num_ctx tests (or new file
                                         if test_llm.py is crowded)
README.md                                Document WHY_LLM_NUM_CTX + auto-couple
```

## Acceptance

- `WHY_LLM_NUM_CTX=8192` → `extra_body={"options":{"num_ctx":8192}}` on every chat call.
- `WHY_LLM_NUM_CTX` unset, provider=`openai-compatible`, `WHY_LLM_MAX_CTX` unset → auto-default 4096 applies → num_ctx=4096.
- `WHY_LLM_NUM_CTX` unset, `WHY_LLM_MAX_CTX=8192`, provider=`openai-compatible` → num_ctx=8192.
- `WHY_LLM_NUM_CTX` unset, `WHY_LLM_MAX_CTX=0`, provider=`openai-compatible` → no extra_body.
- `WHY_LLM_NUM_CTX=abc` / `=0` / `=-1` → `LLMError` at construction.
- `WHY_LLM_NUM_CTX=""` → treated as unset.
- Provider=`groq` with `WHY_LLM_NUM_CTX=8192` set → ignored, no effect.

## Strides

```
Stride 1: OpenAICompatibleBackend num_ctx plumbing
  test: tests/test_llm.py (extend)
  impl: src/why/_backends/openai_compatible.py
  depends: []

Stride 2: LLMClient env resolution + validation + auto-couple
  test: tests/test_llm.py (extend)
  impl: src/why/llm.py
  depends: [1]

Stride 3: README documentation
  test: (skip — docs only; behavior covered by Strides 1+2)
  impl: README.md
  depends: [2]
```

Wave 1: Stride 1. Wave 2: Stride 2. Wave 3: Stride 3.

## Out of scope

- Auto-detect `num_ctx` from system RAM.
- Per-model `num_ctx` overrides.
- Pushing `num_ctx` into Modelfile (Ollama-side concern).
- Other Ollama options (`top_k`, `repeat_penalty`, etc.) — Approach B from brainstorm; deferred until a real user need surfaces.
