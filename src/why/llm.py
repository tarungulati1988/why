"""LLM client abstraction for why.

Supports Groq as the default backend with hand-rolled retry logic.
Provider is resolved via: constructor param → WHY_LLM_PROVIDER env var → "groq".
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Literal

import groq as groq_sdk  # KEEP — tests patch why.llm.groq_sdk.Groq

from why._backends.base import Backend, ChatResult
from why._errors import (
    _RETRYABLE_STATUS,
    LLMError,
    LLMMissingKeyError,
    LLMRateLimitError,
    LLMServerError,
    LLMTimeoutError,
)

logger = logging.getLogger("why.llm")

# ---------------------------------------------------------------------------
# Retry constants
# ---------------------------------------------------------------------------

# Maximum number of retry attempts after the initial attempt.
_MAX_RETRIES = 3

# Base delay in seconds; doubles each attempt: 1s → 2s → 4s.
_BASE_DELAY = 1.0


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

# Re-export so `from why.llm import LLMError` etc. continue to work.
__all__ = [
    "LLMError",
    "LLMMissingKeyError",
    "LLMRateLimitError",
    "LLMServerError",
    "LLMTimeoutError",
]


@dataclass
class Message:
    """A single chat message with a role and content."""

    role: Literal["user", "assistant"]
    content: str

    def __post_init__(self) -> None:
        if self.role not in ("user", "assistant"):
            raise ValueError(f"Message.role must be 'user' or 'assistant', got {self.role!r}")


# ---------------------------------------------------------------------------
# GroqBackend
# ---------------------------------------------------------------------------


class GroqBackend:
    """Groq-specific Backend. Translates groq_sdk exceptions to LLMError types."""

    def __init__(self, api_key: str) -> None:
        self._client = groq_sdk.Groq(api_key=api_key)

    def chat(self, model: str, payload: list[Any], **_extra: Any) -> ChatResult:
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
        if not content:
            raise LLMError("model returned no text content")
        u = r.usage
        return ChatResult(
            content=content,
            prompt_tokens=u.prompt_tokens if u else None,
            completion_tokens=u.completion_tokens if u else None,
            total_tokens=u.total_tokens if u else None,
        )


# ---------------------------------------------------------------------------
# LLMClient
# ---------------------------------------------------------------------------


class LLMClient:
    """Thin LLM client that routes to a provider backend.

    Provider resolution order:
      1. ``provider`` constructor argument
      2. ``WHY_LLM_PROVIDER`` environment variable
      3. Default: ``"groq"``

    Supported providers:

    - ``"groq"`` — Groq cloud API. Requires ``GROQ_API_KEY``.
    - ``"openai-compatible"`` — Any OpenAI-compatible server (Ollama, llama.cpp,
      LM Studio, vLLM, TGI, …). Requires ``WHY_LLM_BASE_URL`` (e.g.
      ``http://localhost:11434/v1``). ``WHY_LLM_API_KEY`` is optional; defaults
      to ``"not-needed"`` for servers that do not require authentication.
    """

    def __init__(
        self,
        model: str = "llama-3.3-70b-versatile",
        provider: str | None = None,
    ) -> None:
        self.model = model

        # Resolve provider: param → env var → default "groq"
        resolved_provider = provider or os.getenv("WHY_LLM_PROVIDER") or "groq"

        if resolved_provider == "groq":
            # Read the API key from the environment at construction time so
            # callers get a clear error immediately rather than at call time.
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise LLMMissingKeyError("GROQ_API_KEY not set")
            self._backend: Backend = GroqBackend(api_key)
        elif resolved_provider == "openai-compatible":
            base_url = os.getenv("WHY_LLM_BASE_URL")
            if not base_url:
                raise LLMMissingKeyError("WHY_LLM_BASE_URL not set")
            api_key = os.getenv("WHY_LLM_API_KEY") or "not-needed"
            # Warn when sending to a non-local host without an explicit API key,
            # since prompt data will be transmitted without credentials.
            _local_prefixes = ("http://localhost", "http://127.", "http://[::1]")
            if not os.getenv("WHY_LLM_API_KEY") and not base_url.startswith(_local_prefixes):
                logger.warning(
                    "WHY_LLM_BASE_URL points to a non-local host (%s) but WHY_LLM_API_KEY is "
                    "not set; prompt data will be sent without credentials.",
                    base_url,
                )
            # Lazy import — keeps `openai` an importable-but-not-imported dependency
            # for users who only use Groq.
            from why._backends.openai_compatible import OpenAICompatibleBackend
            self._backend = OpenAICompatibleBackend(base_url=base_url, api_key=api_key)
        elif resolved_provider == "anthropic":
            raise NotImplementedError("anthropic backend not yet implemented")
        elif resolved_provider == "openai":
            raise NotImplementedError(
                "Direct OpenAI API support is not yet implemented. "
                "For OpenAI-compatible local servers (Ollama, llama.cpp, LM Studio, vLLM, TGI), "
                "use provider='openai-compatible' and set WHY_LLM_BASE_URL."
            )
        else:
            raise LLMError(f"unknown provider: {resolved_provider!r}")

    def complete(self, system: str, messages: list[Message], verbose: bool = False) -> str:
        """Send a chat completion request and return the response content.

        Retries up to _MAX_RETRIES times on transient failures (rate limit,
        server error, timeout) with exponential back-off. Non-retryable errors
        (e.g. 400, 401) are raised immediately as LLMError.

        Parameters
        ----------
        system:   System prompt prepended to the messages list.
        messages: Conversation history as a list of Message objects.
        verbose:  When True, log token usage at DEBUG level via "why.llm".
        """
        # Build the messages payload once; reused for every attempt.
        payload: list[Any] = [
            {"role": "system", "content": system},
            *[{"role": m.role, "content": m.content} for m in messages],
        ]

        last_exc: Exception | None = None

        for attempt in range(_MAX_RETRIES + 1):  # attempt 0 … _MAX_RETRIES
            try:
                result = self._backend.chat(self.model, payload)
            except LLMRateLimitError as exc:
                last_exc = exc
            except LLMServerError as exc:
                last_exc = exc
            except LLMTimeoutError as exc:
                last_exc = exc
            else:
                if (
                    verbose
                    and result.prompt_tokens is not None
                    and result.completion_tokens is not None
                ):
                    total = result.total_tokens if result.total_tokens is not None else (
                        result.prompt_tokens + result.completion_tokens
                    )
                    logger.debug(
                        "model=%s  prompt_tokens=%d  completion_tokens=%d  total=%d",
                        self.model, result.prompt_tokens, result.completion_tokens, total,
                    )
                return result.content

            # Sleep before the next attempt using exponential back-off.
            # On the last attempt we skip sleeping because we're about to raise.
            if attempt < _MAX_RETRIES:
                time.sleep(_BASE_DELAY * (2**attempt))

        # Re-raise the last exception directly. Wrapping it as a fresh same-type
        # instance with `from last_exc` produces a confusing chained traceback
        # ("During handling of ... another exception of the same type occurred").
        # Append the "max retries exceeded" context by adding a note via add_note
        # when available (Python 3.11+).
        assert last_exc is not None
        last_exc.add_note("max retries exceeded")
        raise last_exc
