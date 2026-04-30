"""Tests for OpenAICompatibleBackend — written FIRST (TDD).

All external openai API calls are patched via unittest.mock; no network access.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import openai
import pytest

from why._backends.base import ChatResult
from why._backends.openai_compatible import OpenAICompatibleBackend
from why.llm import LLMError, LLMRateLimitError, LLMServerError, LLMTimeoutError

# ---------------------------------------------------------------------------
# Helpers — build mock openai SDK exception instances
# ---------------------------------------------------------------------------

def _make_mock_openai_client(content: str = "hello") -> MagicMock:
    """Return a mock OpenAI client whose create() returns a response with the given content."""
    mock_response = MagicMock()
    mock_response.choices[0].message.content = content
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 5
    mock_response.usage.total_tokens = 15

    mock_client = MagicMock()
    mock_client.chat.completions.create = MagicMock(return_value=mock_response)
    return mock_client


def _make_api_status_error(status_code: int, message: str = "error") -> openai.APIStatusError:
    """Build an openai.APIStatusError with the given status code.

    Uses plain APIStatusError (not RateLimitError subclass) so the APIStatusError
    handler in chat() is exercised, not the dedicated RateLimitError handler.
    For 500, InternalServerError subclass is also a plain APIStatusError without
    overriding status_code, so we set it on the mock response directly.
    """
    mock_response = MagicMock()
    mock_response.status_code = status_code
    mock_request = MagicMock()
    mock_response.request = mock_request
    # Always use the base APIStatusError so status_code routing in chat() is tested.
    # Subclasses like RateLimitError are tested separately in test 4.
    return openai.APIStatusError(message, response=mock_response, body=None)


def _make_rate_limit_error() -> openai.RateLimitError:
    """Build an openai.RateLimitError."""
    mock_response = MagicMock()
    mock_response.status_code = 429
    mock_response.request = MagicMock()
    return openai.RateLimitError("rate limited", response=mock_response, body=None)


def _make_timeout_error() -> openai.APITimeoutError:
    """Build an openai.APITimeoutError."""
    mock_request = MagicMock()
    return openai.APITimeoutError(request=mock_request)


def _make_connection_error() -> openai.APIConnectionError:
    """Build an openai.APIConnectionError."""
    mock_request = MagicMock()
    return openai.APIConnectionError(request=mock_request)


# ---------------------------------------------------------------------------
# Test 1: Successful chat — ChatResult populated
# ---------------------------------------------------------------------------

def test_successful_chat_returns_populated_chat_result() -> None:
    """chat() must return a ChatResult with content and token counts populated."""
    mock_client = _make_mock_openai_client(content="test response")

    with patch("openai.OpenAI", return_value=mock_client):
        backend = OpenAICompatibleBackend(base_url="http://localhost:11434/v1", api_key="ollama")
        result = backend.chat("llama3", [{"role": "user", "content": "hello"}])

    assert isinstance(result, ChatResult)
    assert result.content == "test response"
    assert result.prompt_tokens == 10
    assert result.completion_tokens == 5
    assert result.total_tokens == 15


# ---------------------------------------------------------------------------
# Test 2: None or empty content raises LLMError
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("empty_content", [None, ""])
def test_empty_content_raises_llm_error(empty_content: str | None) -> None:
    """When choices[0].message.content is None or '', chat() must raise LLMError."""
    mock_client = _make_mock_openai_client(content=empty_content)

    with patch("openai.OpenAI", return_value=mock_client):
        backend = OpenAICompatibleBackend(base_url="http://localhost:11434/v1", api_key="ollama")
        with pytest.raises(LLMError, match="model returned no text content"):
            backend.chat("llama3", [{"role": "user", "content": "hello"}])


# ---------------------------------------------------------------------------
# Test 3: No usage object — tokens are all None, content returned
# ---------------------------------------------------------------------------

def test_no_usage_object_tokens_are_none() -> None:
    """When r.usage is None, all token fields must be None but content returned."""
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "response text"
    mock_response.usage = None

    mock_client = MagicMock()
    mock_client.chat.completions.create = MagicMock(return_value=mock_response)

    with patch("openai.OpenAI", return_value=mock_client):
        backend = OpenAICompatibleBackend(base_url="http://localhost:11434/v1", api_key="ollama")
        result = backend.chat("llama3", [{"role": "user", "content": "hello"}])

    assert result.content == "response text"
    assert result.prompt_tokens is None
    assert result.completion_tokens is None
    assert result.total_tokens is None


# ---------------------------------------------------------------------------
# Test 4: RateLimitError → LLMRateLimitError
# ---------------------------------------------------------------------------

def test_rate_limit_error_raises_llm_rate_limit_error() -> None:
    """openai.RateLimitError must be translated to LLMRateLimitError."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = _make_rate_limit_error()

    with patch("openai.OpenAI", return_value=mock_client):
        backend = OpenAICompatibleBackend(base_url="http://localhost:11434/v1", api_key="ollama")
        with pytest.raises(LLMRateLimitError):
            backend.chat("llama3", [{"role": "user", "content": "hello"}])


# ---------------------------------------------------------------------------
# Test 5: APITimeoutError → LLMTimeoutError
# ---------------------------------------------------------------------------

def test_api_timeout_error_raises_llm_timeout_error() -> None:
    """openai.APITimeoutError must be translated to LLMTimeoutError."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = _make_timeout_error()

    with patch("openai.OpenAI", return_value=mock_client):
        backend = OpenAICompatibleBackend(base_url="http://localhost:11434/v1", api_key="ollama")
        with pytest.raises(LLMTimeoutError):
            backend.chat("llama3", [{"role": "user", "content": "hello"}])


# ---------------------------------------------------------------------------
# Test 6: APIConnectionError → LLMServerError (retryable for local LLMs)
# ---------------------------------------------------------------------------

def test_api_connection_error_raises_llm_server_error() -> None:
    """openai.APIConnectionError must be translated to LLMServerError (retryable)."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = _make_connection_error()

    with patch("openai.OpenAI", return_value=mock_client):
        backend = OpenAICompatibleBackend(base_url="http://localhost:11434/v1", api_key="ollama")
        with pytest.raises(LLMServerError, match="connection error"):
            backend.chat("llama3", [{"role": "user", "content": "hello"}])


# ---------------------------------------------------------------------------
# Test 7: APIStatusError 429 → LLMServerError (retryable)
# ---------------------------------------------------------------------------

def test_api_status_error_429_raises_llm_server_error() -> None:
    """openai.APIStatusError with status 429 must be translated to LLMServerError."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = _make_api_status_error(429)

    with patch("openai.OpenAI", return_value=mock_client):
        backend = OpenAICompatibleBackend(base_url="http://localhost:11434/v1", api_key="ollama")
        with pytest.raises(LLMServerError, match="status 429"):
            backend.chat("llama3", [{"role": "user", "content": "hello"}])


# ---------------------------------------------------------------------------
# Test 8: APIStatusError 500 → LLMServerError (retryable)
# ---------------------------------------------------------------------------

def test_api_status_error_500_raises_llm_server_error() -> None:
    """openai.APIStatusError with status 500 must be translated to LLMServerError."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = _make_api_status_error(500)

    with patch("openai.OpenAI", return_value=mock_client):
        backend = OpenAICompatibleBackend(base_url="http://localhost:11434/v1", api_key="ollama")
        with pytest.raises(LLMServerError, match="status 500"):
            backend.chat("llama3", [{"role": "user", "content": "hello"}])


# ---------------------------------------------------------------------------
# Test 9: APIStatusError 400 → LLMError (non-retryable)
# ---------------------------------------------------------------------------

def test_api_status_error_400_raises_llm_error_non_retryable() -> None:
    """openai.APIStatusError with status 400 must be translated to plain LLMError
    (non-retryable)."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = _make_api_status_error(400)

    with patch("openai.OpenAI", return_value=mock_client):
        backend = OpenAICompatibleBackend(base_url="http://localhost:11434/v1", api_key="ollama")
        with pytest.raises(LLMError) as exc_info:
            backend.chat("llama3", [{"role": "user", "content": "hello"}])

    # Must be plain LLMError, not LLMServerError (which is retryable)
    assert type(exc_info.value) is LLMError, (
        f"expected exact LLMError, got {type(exc_info.value).__name__}"
    )
    assert "API error 400" in str(exc_info.value)
