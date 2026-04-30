"""OpenAI-compatible backend — works with Ollama, llama.cpp, LM Studio, vLLM, TGI."""
from __future__ import annotations

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


class OpenAICompatibleBackend:
    """Backend for any OpenAI-compatible /v1/chat/completions server."""

    def __init__(self, base_url: str, api_key: str) -> None:
        self._client = openai.OpenAI(base_url=base_url, api_key=api_key)

    def chat(self, model: str, payload: list[Any], **_extra: Any) -> ChatResult:
        try:
            r = self._client.chat.completions.create(model=model, messages=payload)
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
