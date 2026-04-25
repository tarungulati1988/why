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

import groq as groq_sdk

logger = logging.getLogger("why.llm")

# ---------------------------------------------------------------------------
# Retry constants
# ---------------------------------------------------------------------------

# HTTP status codes that warrant a retry (server-side transient errors).
_RETRYABLE_STATUS = frozenset({429, 500, 503})

# Maximum number of retry attempts after the initial attempt.
_MAX_RETRIES = 3

# Base delay in seconds; doubles each attempt: 1s → 2s → 4s.
_BASE_DELAY = 1.0


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass
class Message:
    """A single chat message with a role and content."""

    role: Literal["user", "assistant"]
    content: str

    def __post_init__(self) -> None:
        if self.role not in ("user", "assistant"):
            raise ValueError(f"Message.role must be 'user' or 'assistant', got {self.role!r}")


class LLMError(Exception):
    """Base exception for all LLM client errors."""


class LLMMissingKeyError(LLMError):
    """Required API key environment variable is absent at construction time."""


class LLMRateLimitError(LLMError):
    """Rate limit retries exhausted (HTTP 429)."""


class LLMServerError(LLMError):
    """Server error retries exhausted (HTTP 500/503)."""


class LLMTimeoutError(LLMError):
    """Timeout retries exhausted."""


# ---------------------------------------------------------------------------
# LLMClient
# ---------------------------------------------------------------------------


class LLMClient:
    """Thin LLM client that routes to a provider backend.

    Provider resolution order:
      1. ``provider`` constructor argument
      2. ``WHY_LLM_PROVIDER`` environment variable
      3. Default: ``"groq"``
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
            # Instantiate the Groq SDK client; stored for reuse across calls.
            self._client = groq_sdk.Groq(api_key=api_key)
        elif resolved_provider == "anthropic":
            raise NotImplementedError("anthropic backend not yet implemented")
        elif resolved_provider == "openai":
            raise NotImplementedError("openai backend not yet implemented")
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
        # Typed as list[Any] to satisfy the Groq SDK's union message param type.
        payload: list[Any] = [
            {"role": "system", "content": system},
            *[{"role": m.role, "content": m.content} for m in messages],
        ]

        last_exc: Exception | None = None

        for attempt in range(_MAX_RETRIES + 1):  # attempt 0 … _MAX_RETRIES
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=payload,
                )

                # Log token usage when verbose mode is requested.
                if verbose and response.usage is not None:
                    usage = response.usage
                    logger.debug(
                        "model=%s  prompt_tokens=%d  completion_tokens=%d  total=%d",
                        self.model,
                        usage.prompt_tokens,
                        usage.completion_tokens,
                        usage.total_tokens,
                    )

                content = response.choices[0].message.content
                if content is None:
                    raise LLMError("model returned no text content")
                return content

            except groq_sdk.RateLimitError as exc:
                # 429 — server is throttling; retryable.
                last_exc = exc

            except groq_sdk.APIStatusError as exc:
                if exc.status_code in _RETRYABLE_STATUS:
                    # Transient server-side error (500, 503); retryable.
                    last_exc = exc
                else:
                    # Non-retryable (e.g. 400 bad request, 401 unauthorized).
                    # Expose only the status code — raw error body may contain
                    # provider internals not appropriate to surface to callers.
                    raise LLMError(f"API error {exc.status_code}") from exc

            except groq_sdk.APITimeoutError as exc:
                # Network timeout; retryable.
                last_exc = exc

            # Sleep before the next attempt using exponential back-off.
            # On the last attempt we skip sleeping because we're about to raise.
            if attempt < _MAX_RETRIES:
                time.sleep(_BASE_DELAY * (2**attempt))

        # Discriminate the final raise by inspecting the last exception type so
        # callers can distinguish rate-limit, server error, and timeout exhaustion.
        if isinstance(last_exc, groq_sdk.APITimeoutError):
            raise LLMTimeoutError("max retries exceeded") from last_exc
        if isinstance(last_exc, groq_sdk.RateLimitError):
            raise LLMRateLimitError("max retries exceeded") from last_exc
        raise LLMServerError("max retries exceeded") from last_exc
