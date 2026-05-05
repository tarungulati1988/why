"""Backend Protocol and ChatResult dataclass — shared contract for all LLM adapters.

Invariants:
    - Every backend must implement chat(model, payload, **extra) -> ChatResult.
    - Retryable failures MUST be raised as LLMRateLimitError, LLMServerError, or
      LLMTimeoutError so LLMClient.complete() can apply back-off correctly.
    - Plain LLMError is treated as non-retryable and propagates immediately.
    - The **extra kwargs slot is reserved for provider-specific options (e.g.
      Ollama's num_ctx); backends that don't recognise a kwarg should ignore it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class ChatResult:
    content: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


class Backend(Protocol):
    """Protocol that every LLM provider backend must satisfy.

    The `**extra` kwargs slot is reserved for provider-specific options
    (e.g. Ollama's `extra_body={"options": {"num_ctx": N}}`). Backends
    that don't recognise a kwarg should silently ignore it.
    """

    def chat(self, model: str, payload: list[Any], **extra: Any) -> ChatResult:
        """Send a chat completion and return a ChatResult.

        Retryable failures (rate limit, transient server error, timeout) MUST be
        raised as the typed subclasses LLMRateLimitError, LLMServerError, or
        LLMTimeoutError. Plain LLMError is treated as non-retryable and propagates
        to the caller without backoff.
        """
        ...
