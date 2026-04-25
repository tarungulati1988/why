"""Tests for why.llm — written BEFORE the implementation (TDD).

All external Groq API calls are patched via unittest.mock; no network access.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import groq
import pytest

from why.llm import (
    LLMClient,
    LLMError,
    LLMMissingKeyError,
    LLMRateLimitError,
    LLMServerError,
    LLMTimeoutError,
    Message,
)

# ---------------------------------------------------------------------------
# Helpers — build mock groq SDK exception instances
# ---------------------------------------------------------------------------

def _make_api_status_error(status_code: int, message: str = "error") -> groq.APIStatusError:
    """Build a groq.APIStatusError with the given status code."""
    mock_response = MagicMock()
    mock_response.status_code = status_code
    return groq.APIStatusError(message, response=mock_response, body=None)


def _make_rate_limit_error() -> groq.RateLimitError:
    """Build a groq.RateLimitError."""
    mock_response = MagicMock()
    mock_response.status_code = 429
    return groq.RateLimitError("rate limited", response=mock_response, body=None)


def _make_timeout_error() -> groq.APITimeoutError:
    """Build a groq.APITimeoutError."""
    mock_request = MagicMock()
    return groq.APITimeoutError(request=mock_request)


def _make_mock_groq_client(content: str = "hello") -> MagicMock:
    """Return a mock Groq client whose create() returns a response with the given content."""
    mock_response = MagicMock()
    mock_response.choices[0].message.content = content
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 5
    mock_response.usage.total_tokens = 15

    mock_create = MagicMock(return_value=mock_response)
    mock_client = MagicMock()
    mock_client.chat.completions.create = mock_create
    return mock_client


# ---------------------------------------------------------------------------
# Test 1: complete() passes correct model and messages to Groq
# ---------------------------------------------------------------------------

def test_complete_passes_correct_model_and_messages(monkeypatch: pytest.MonkeyPatch) -> None:
    """complete() must forward model name and system + user messages to Groq create()."""
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    mock_client = _make_mock_groq_client()

    with patch("why.llm.groq_sdk.Groq", return_value=mock_client):
        llm = LLMClient(model="llama-3.3-70b-versatile")
        llm.complete("sys", [Message("user", "hi")])

    mock_client.chat.completions.create.assert_called_once_with(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
        ],
    )


# ---------------------------------------------------------------------------
# Test 2: complete() returns the content string from Groq response
# ---------------------------------------------------------------------------

def test_complete_returns_content_string(monkeypatch: pytest.MonkeyPatch) -> None:
    """complete() must return choices[0].message.content as a plain string."""
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    mock_client = _make_mock_groq_client(content="hello")

    with patch("why.llm.groq_sdk.Groq", return_value=mock_client):
        llm = LLMClient()
        result = llm.complete("system prompt", [Message("user", "question")])

    assert result == "hello"


# ---------------------------------------------------------------------------
# Test 3: missing GROQ_API_KEY raises LLMMissingKeyError at construction
# ---------------------------------------------------------------------------

def test_missing_api_key_raises_at_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    """LLMClient() must raise LLMMissingKeyError immediately when GROQ_API_KEY is absent."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    with pytest.raises(LLMMissingKeyError, match="GROQ_API_KEY not set"):
        LLMClient()


# ---------------------------------------------------------------------------
# Test 4: unknown provider raises LLMError
# ---------------------------------------------------------------------------

def test_unknown_provider_raises_llm_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """LLMClient(provider='unknown') must raise LLMError."""
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    with pytest.raises(LLMError, match="unknown provider"):
        LLMClient(provider="unknown")


# ---------------------------------------------------------------------------
# Test 5: rate limit retries 3 times then raises LLMRateLimitError
# ---------------------------------------------------------------------------

def test_rate_limit_retries_3_times_then_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """On persistent RateLimitError, create() must be called 4 times (1 + 3 retries)."""
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = _make_rate_limit_error()

    with (
        patch("why.llm.groq_sdk.Groq", return_value=mock_client),
        patch("why.llm.time.sleep"),  # no-op sleep to keep test fast
    ):
        llm = LLMClient()
        with pytest.raises(LLMRateLimitError, match="max retries exceeded"):
            llm.complete("sys", [Message("user", "hi")])

    assert mock_client.chat.completions.create.call_count == 4


# ---------------------------------------------------------------------------
# Test 6: non-retryable APIStatusError (400) raises LLMError immediately
# ---------------------------------------------------------------------------

def test_non_retryable_status_raises_immediately(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 400 APIStatusError must not be retried — create() called exactly once."""
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = _make_api_status_error(400, "bad request")

    with (
        patch("why.llm.groq_sdk.Groq", return_value=mock_client),
        patch("why.llm.time.sleep"),
    ):
        llm = LLMClient()
        with pytest.raises(LLMError):
            llm.complete("sys", [Message("user", "hi")])

    assert mock_client.chat.completions.create.call_count == 1


# ---------------------------------------------------------------------------
# Test 7: timeout retries 3 times then raises LLMTimeoutError
# ---------------------------------------------------------------------------

def test_timeout_retries_3_times_then_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """On persistent APITimeoutError, create() must be called 4 times then raise LLMTimeoutError."""
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = _make_timeout_error()

    with (
        patch("why.llm.groq_sdk.Groq", return_value=mock_client),
        patch("why.llm.time.sleep"),
    ):
        llm = LLMClient()
        with pytest.raises(LLMTimeoutError, match="max retries exceeded"):
            llm.complete("sys", [Message("user", "hi")])

    assert mock_client.chat.completions.create.call_count == 4


# ---------------------------------------------------------------------------
# Test 8: succeeds after one transient failure
# ---------------------------------------------------------------------------

def test_succeeds_after_one_transient_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """create() failing once then succeeding must return the success content."""
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    mock_client = MagicMock()

    # Build a success response
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "success"
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 5
    mock_response.usage.total_tokens = 15

    # First call raises, second succeeds
    mock_client.chat.completions.create.side_effect = [
        _make_rate_limit_error(),
        mock_response,
    ]

    with (
        patch("why.llm.groq_sdk.Groq", return_value=mock_client),
        patch("why.llm.time.sleep"),
    ):
        llm = LLMClient()
        result = llm.complete("sys", [Message("user", "hi")])

    assert result == "success"
    assert mock_client.chat.completions.create.call_count == 2


# ---------------------------------------------------------------------------
# Test 9: verbose=True logs token counts via logger.debug
# ---------------------------------------------------------------------------

def test_verbose_logs_token_counts(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """complete(..., verbose=True) must emit a debug log with token counts."""
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    mock_client = _make_mock_groq_client(content="result")

    with patch("why.llm.groq_sdk.Groq", return_value=mock_client):
        llm = LLMClient()
        with caplog.at_level(logging.DEBUG, logger="why.llm"):
            llm.complete("sys", [Message("user", "hi")], verbose=True)

    # Verify that the debug logger emitted at least one record for token counts
    assert any("prompt_tokens" in record.message for record in caplog.records), (
        "Expected a debug log message containing 'prompt_tokens'"
    )


# ---------------------------------------------------------------------------
# Test 10: WHY_LLM_PROVIDER env-var used when no provider param given
# ---------------------------------------------------------------------------

def test_why_llm_provider_env_var_is_respected(monkeypatch: pytest.MonkeyPatch) -> None:
    """WHY_LLM_PROVIDER=groq must route to Groq when no provider param is passed."""
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("WHY_LLM_PROVIDER", "groq")
    mock_client = _make_mock_groq_client(content="env-var-routed")

    with patch("why.llm.groq_sdk.Groq", return_value=mock_client) as mock_groq:
        llm = LLMClient()  # no provider= arg
        result = llm.complete("sys", [Message("user", "hi")])

    mock_groq.assert_called_once_with(api_key="test-key")
    assert result == "env-var-routed"


# ---------------------------------------------------------------------------
# Test 11: server error (500) exhaustion raises LLMServerError, not LLMRateLimitError
# ---------------------------------------------------------------------------

def test_server_error_retries_then_raises_llm_server_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Persistent 500 APIStatusError must exhaust retries and raise LLMServerError."""
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = _make_api_status_error(500, "internal error")

    with (
        patch("why.llm.groq_sdk.Groq", return_value=mock_client),
        patch("why.llm.time.sleep"),
    ):
        llm = LLMClient()
        with pytest.raises(LLMServerError, match="max retries exceeded"):
            llm.complete("sys", [Message("user", "hi")])

    assert mock_client.chat.completions.create.call_count == 4


# ---------------------------------------------------------------------------
# Test 12: invalid Message.role raises ValueError
# ---------------------------------------------------------------------------

def test_message_invalid_role_raises_value_error() -> None:
    """Message with an unrecognised role must raise ValueError at construction."""
    with pytest.raises(ValueError, match="must be 'user' or 'assistant'"):
        Message(role="system", content="sneaky system prompt")
