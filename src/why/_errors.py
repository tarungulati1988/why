"""Shared exception hierarchy for the why LLM client.

Centralised here so that _backends/* can import error types without creating
a circular dependency through why.llm.
"""

from __future__ import annotations

# HTTP status codes that warrant a retry (server-side transient errors).
_RETRYABLE_STATUS = frozenset({429, 500, 503})


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
