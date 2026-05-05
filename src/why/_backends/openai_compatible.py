"""OpenAI-compatible backend — adapter for any /v1/chat/completions server.

Stage: llm backend — instantiated by LLMClient when provider="openai-compatible".

Inputs:
    base_url — server base URL, e.g. "http://localhost:11434/v1" for Ollama.
    api_key  — authentication token; defaults to "not-needed" for local servers.
    num_ctx  — optional Ollama KV-cache size passed via extra_body; ignored by
               other servers (vLLM in strict-schema mode may reject it — leave
               unset for non-Ollama backends).

Outputs:
    ChatResult — content string plus optional token usage counts.

Notes:
    openai.APIConnectionError (local server crash/restart) is mapped to
    LLMServerError so LLMClient's retry logic treats it as transient.
"""
from __future__ import annotations

import logging
from typing import Any

import openai

from why._backends.base import ChatResult
from why._errors import (
    _RETRYABLE_STATUS,
    LLMError,
    LLMRateLimitError,
    LLMServerError,
    LLMTimeoutError,
)

logger = logging.getLogger("why.llm")


class OpenAICompatibleBackend:
    """Backend for any OpenAI-compatible /v1/chat/completions server."""

    def __init__(self, base_url: str, api_key: str, num_ctx: int | None = None) -> None:
        self._client = openai.OpenAI(base_url=base_url, api_key=api_key)
        self._num_ctx = num_ctx

    def chat(self, model: str, payload: list[Any], **_extra: Any) -> ChatResult:
        kwargs: dict[str, Any] = {"model": model, "messages": payload}
        if self._num_ctx is not None:
            kwargs["extra_body"] = {"options": {"num_ctx": self._num_ctx}}
            logger.debug("num_ctx=%d (Ollama)", self._num_ctx)
        try:
            r = self._client.chat.completions.create(**kwargs)
        except openai.RateLimitError as e:
            raise LLMRateLimitError(str(e)) from e
        except openai.APITimeoutError as e:
            raise LLMTimeoutError(str(e)) from e
        except openai.APIConnectionError as e:
            # Local servers (Ollama, llama.cpp) can restart/crash — make this
            # retryable so the client gets a window to reconnect.
            raise LLMServerError(f"connection error: {e}") from e
        except openai.APIStatusError as e:
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
