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
# Test 2b: None or empty content from Groq raises LLMError
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("empty_content", [None, ""])
def test_groq_empty_content_raises_llm_error(
    monkeypatch: pytest.MonkeyPatch,
    empty_content: str | None,
) -> None:
    """When GroqBackend receives None or '' content, complete() must raise LLMError."""
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    mock_client = _make_mock_groq_client(content=empty_content)

    with patch("why.llm.groq_sdk.Groq", return_value=mock_client):
        llm = LLMClient()
        with pytest.raises(LLMError, match="model returned no text content"):
            llm.complete("sys", [Message("user", "hi")])


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


# ---------------------------------------------------------------------------
# Test 13: retry loop is provider-agnostic — succeeds after transient
#          LLMRateLimitError raised by a fake backend (no groq involvement).
# ---------------------------------------------------------------------------

def test_retry_loop_provider_agnostic_succeeds_after_transient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inject a fake Backend that raises LLMRateLimitError 3x then returns a ChatResult.

    Confirms LLMClient.complete() retries on our typed exception (not groq's), uses
    exponential back-off, and returns the eventual content.
    """
    from why._backends.base import ChatResult
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    call_count = {"n": 0}

    class FakeBackend:
        def chat(self, model, payload, **extra):
            call_count["n"] += 1
            if call_count["n"] <= 3:
                raise LLMRateLimitError("synthetic")
            return ChatResult(content="ok", prompt_tokens=1, completion_tokens=1)

    sleep_args: list[float] = []
    def fake_sleep(s: float) -> None:
        sleep_args.append(s)

    with patch("why.llm.time.sleep", side_effect=fake_sleep):
        # Construct a normal LLMClient (which builds a real GroqBackend), then swap it.
        with patch("why.llm.groq_sdk.Groq", return_value=MagicMock()):
            llm = LLMClient()
        llm._backend = FakeBackend()  # type: ignore[assignment]
        result = llm.complete("sys", [Message("user", "hi")])

    assert result == "ok"
    assert call_count["n"] == 4
    assert sleep_args == [1.0, 2.0, 4.0]


# ---------------------------------------------------------------------------
# Test 14: non-retryable LLMError raised by a fake backend propagates immediately
#          and time.sleep is never called.
# ---------------------------------------------------------------------------

def test_non_retryable_llm_error_propagates_without_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A plain LLMError raised by the backend must propagate on the first call."""
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    call_count = {"n": 0}

    class FakeBackend:
        def chat(self, model, payload, **extra):
            call_count["n"] += 1
            raise LLMError("API error 400")

    with patch("why.llm.time.sleep") as mock_sleep:
        with patch("why.llm.groq_sdk.Groq", return_value=MagicMock()):
            llm = LLMClient()
        llm._backend = FakeBackend()  # type: ignore[assignment]
        with pytest.raises(LLMError) as exc_info:
            llm.complete("sys", [Message("user", "hi")])

    assert type(exc_info.value) is LLMError, (
        f"expected exact LLMError, got {type(exc_info.value).__name__}"
    )
    assert "API error 400" in str(exc_info.value)
    assert call_count["n"] == 1
    mock_sleep.assert_not_called()


# ---------------------------------------------------------------------------
# Test 15: openai-compatible provider constructs an OpenAICompatibleBackend
# ---------------------------------------------------------------------------

def test_provider_openai_compatible_constructs_with_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LLMClient with openai-compatible provider must build an OpenAICompatibleBackend."""
    from unittest.mock import patch as _patch

    from why._backends.openai_compatible import OpenAICompatibleBackend

    monkeypatch.setenv("WHY_LLM_PROVIDER", "openai-compatible")
    monkeypatch.setenv("WHY_LLM_BASE_URL", "http://localhost:11434/v1")

    with _patch("openai.OpenAI"):
        client = LLMClient()

    assert isinstance(client._backend, OpenAICompatibleBackend)


# ---------------------------------------------------------------------------
# Test 16: openai-compatible without WHY_LLM_BASE_URL raises LLMMissingKeyError
# ---------------------------------------------------------------------------

def test_provider_openai_compatible_missing_base_url_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LLMClient() with openai-compatible but no WHY_LLM_BASE_URL must raise LLMMissingKeyError."""
    monkeypatch.setenv("WHY_LLM_PROVIDER", "openai-compatible")
    monkeypatch.delenv("WHY_LLM_BASE_URL", raising=False)

    with pytest.raises(LLMMissingKeyError, match="WHY_LLM_BASE_URL"):
        LLMClient()


# ---------------------------------------------------------------------------
# Test 17: openai-compatible with no WHY_LLM_API_KEY defaults to "not-needed"
# ---------------------------------------------------------------------------

def test_provider_openai_compatible_default_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When WHY_LLM_API_KEY is absent, api_key='not-needed' is passed to openai.OpenAI."""
    from unittest.mock import patch as _patch

    monkeypatch.setenv("WHY_LLM_PROVIDER", "openai-compatible")
    monkeypatch.setenv("WHY_LLM_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.delenv("WHY_LLM_API_KEY", raising=False)

    with _patch("openai.OpenAI") as mock_openai_cls:
        LLMClient()

    _, kwargs = mock_openai_cls.call_args
    assert kwargs.get("api_key") == "not-needed"


# ---------------------------------------------------------------------------
# Test 18: openai-compatible uses WHY_LLM_API_KEY when set
# ---------------------------------------------------------------------------

def test_provider_openai_compatible_custom_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When WHY_LLM_API_KEY=sk-test-123, that exact value is passed to openai.OpenAI."""
    from unittest.mock import patch as _patch

    monkeypatch.setenv("WHY_LLM_PROVIDER", "openai-compatible")
    monkeypatch.setenv("WHY_LLM_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("WHY_LLM_API_KEY", "sk-test-123")

    with _patch("openai.OpenAI") as mock_openai_cls:
        LLMClient()

    _, kwargs = mock_openai_cls.call_args
    assert kwargs.get("api_key") == "sk-test-123"


# ---------------------------------------------------------------------------
# Test 19: constructor provider= param overrides WHY_LLM_PROVIDER env var
# ---------------------------------------------------------------------------

def test_provider_openai_compatible_constructor_param_overrides_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """provider='openai-compatible' kwarg must win over WHY_LLM_PROVIDER=groq env var."""
    from unittest.mock import patch as _patch

    from why._backends.openai_compatible import OpenAICompatibleBackend

    monkeypatch.setenv("WHY_LLM_PROVIDER", "groq")  # env says groq ...
    monkeypatch.setenv("WHY_LLM_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.delenv("WHY_LLM_API_KEY", raising=False)

    with _patch("openai.OpenAI"):
        # ... but constructor kwarg says openai-compatible → must win
        client = LLMClient(provider="openai-compatible")

    assert isinstance(client._backend, OpenAICompatibleBackend)


# ---------------------------------------------------------------------------
# Test 20: non-local base_url without API key emits a warning
# ---------------------------------------------------------------------------

def test_non_local_base_url_without_api_key_warns(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """LLMClient should warn when WHY_LLM_BASE_URL is non-local and WHY_LLM_API_KEY is absent."""
    from unittest.mock import patch as _patch

    monkeypatch.setenv("WHY_LLM_PROVIDER", "openai-compatible")
    monkeypatch.setenv("WHY_LLM_BASE_URL", "https://remote.example.com/v1")
    monkeypatch.delenv("WHY_LLM_API_KEY", raising=False)

    with _patch("openai.OpenAI"), caplog.at_level(logging.WARNING, logger="why.llm"):
        LLMClient()

    assert any(
        "non-local host" in record.message and "WHY_LLM_API_KEY" in record.message
        for record in caplog.records
    ), "Expected a warning about non-local host without credentials"


# ---------------------------------------------------------------------------
# Test 21: local base_url without API key does NOT warn
# ---------------------------------------------------------------------------

def test_local_base_url_without_api_key_does_not_warn(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """LLMClient must NOT warn when WHY_LLM_BASE_URL is localhost, even without API key."""
    from unittest.mock import patch as _patch

    monkeypatch.setenv("WHY_LLM_PROVIDER", "openai-compatible")
    monkeypatch.setenv("WHY_LLM_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.delenv("WHY_LLM_API_KEY", raising=False)

    with _patch("openai.OpenAI"), caplog.at_level(logging.WARNING, logger="why.llm"):
        LLMClient()

    assert not any(
        "non-local host" in record.message for record in caplog.records
    ), "Unexpected warning logged for a localhost base_url"


# ---------------------------------------------------------------------------
# Test 22: OpenAICompatibleBackend passes num_ctx in extra_body when set
# ---------------------------------------------------------------------------

def test_openai_compatible_backend_passes_num_ctx_in_extra_body() -> None:
    """When num_ctx is set, chat() must pass extra_body={"options": {"num_ctx": N}}."""
    from unittest.mock import patch as _patch

    from why._backends.openai_compatible import OpenAICompatibleBackend

    with _patch("openai.OpenAI") as mock_openai_cls:
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "ok"
        mock_response.usage.prompt_tokens = 5
        mock_response.usage.completion_tokens = 3
        mock_response.usage.total_tokens = 8

        mock_create = MagicMock(return_value=mock_response)
        mock_openai_cls.return_value.chat.completions.create = mock_create

        backend = OpenAICompatibleBackend(
            base_url="http://localhost:11434/v1",
            api_key="not-needed",
            num_ctx=8192,
        )
        backend.chat("llama3", [{"role": "user", "content": "hi"}])

    _, kwargs = mock_create.call_args
    assert kwargs.get("extra_body") == {"options": {"num_ctx": 8192}}


# ---------------------------------------------------------------------------
# Test 23: OpenAICompatibleBackend omits extra_body when num_ctx=None
# ---------------------------------------------------------------------------

def test_openai_compatible_backend_omits_extra_body_when_num_ctx_none() -> None:
    """When num_ctx=None, chat() must NOT include extra_body in the call kwargs."""
    from unittest.mock import patch as _patch

    from why._backends.openai_compatible import OpenAICompatibleBackend

    with _patch("openai.OpenAI") as mock_openai_cls:
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "ok"
        mock_response.usage.prompt_tokens = 5
        mock_response.usage.completion_tokens = 3
        mock_response.usage.total_tokens = 8

        mock_create = MagicMock(return_value=mock_response)
        mock_openai_cls.return_value.chat.completions.create = mock_create

        backend = OpenAICompatibleBackend(
            base_url="http://localhost:11434/v1",
            api_key="not-needed",
            num_ctx=None,
        )
        backend.chat("llama3", [{"role": "user", "content": "hi"}])

    _, kwargs = mock_create.call_args
    assert "extra_body" not in kwargs


# ---------------------------------------------------------------------------
# Test 24: OpenAICompatibleBackend default num_ctx is None (no extra_body)
# ---------------------------------------------------------------------------

def test_openai_compatible_backend_default_num_ctx_is_none() -> None:
    """When num_ctx is not passed, chat() must NOT include extra_body in the call kwargs."""
    from unittest.mock import patch as _patch

    from why._backends.openai_compatible import OpenAICompatibleBackend

    with _patch("openai.OpenAI") as mock_openai_cls:
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "ok"
        mock_response.usage.prompt_tokens = 5
        mock_response.usage.completion_tokens = 3
        mock_response.usage.total_tokens = 8

        mock_create = MagicMock(return_value=mock_response)
        mock_openai_cls.return_value.chat.completions.create = mock_create

        backend = OpenAICompatibleBackend(
            base_url="http://localhost:11434/v1",
            api_key="not-needed",
        )
        backend.chat("llama3", [{"role": "user", "content": "hi"}])

    _, kwargs = mock_create.call_args
    assert "extra_body" not in kwargs


# ---------------------------------------------------------------------------
# Helpers for num_ctx / auto-couple tests
# ---------------------------------------------------------------------------

def _make_openai_compatible_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set the minimal env vars needed for openai-compatible provider."""
    monkeypatch.setenv("WHY_LLM_PROVIDER", "openai-compatible")
    monkeypatch.setenv("WHY_LLM_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.delenv("WHY_LLM_API_KEY", raising=False)


# ---------------------------------------------------------------------------
# Test 25: WHY_LLM_NUM_CTX explicit positive value is passed to backend
# ---------------------------------------------------------------------------

def test_num_ctx_explicit_positive_passed_to_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """When WHY_LLM_NUM_CTX=8192, backend must be constructed with num_ctx=8192."""
    _make_openai_compatible_env(monkeypatch)
    monkeypatch.setenv("WHY_LLM_NUM_CTX", "8192")

    with patch("why._backends.openai_compatible.OpenAICompatibleBackend") as mock_backend_cls, \
         patch("openai.OpenAI"):
        mock_backend_cls.return_value = MagicMock()
        LLMClient()

    _, kwargs = mock_backend_cls.call_args
    assert kwargs.get("num_ctx") == 8192


# ---------------------------------------------------------------------------
# Test 26: WHY_LLM_NUM_CTX=0 raises LLMError
# ---------------------------------------------------------------------------

def test_num_ctx_zero_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """WHY_LLM_NUM_CTX=0 must raise LLMError with appropriate message."""
    _make_openai_compatible_env(monkeypatch)
    monkeypatch.setenv("WHY_LLM_NUM_CTX", "0")

    with pytest.raises(LLMError, match="WHY_LLM_NUM_CTX must be a positive integer"):
        LLMClient()


# ---------------------------------------------------------------------------
# Test 27: WHY_LLM_NUM_CTX=-5 raises LLMError
# ---------------------------------------------------------------------------

def test_num_ctx_negative_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """WHY_LLM_NUM_CTX=-5 must raise LLMError with appropriate message."""
    _make_openai_compatible_env(monkeypatch)
    monkeypatch.setenv("WHY_LLM_NUM_CTX", "-5")

    with pytest.raises(LLMError, match="WHY_LLM_NUM_CTX must be a positive integer"):
        LLMClient()


# ---------------------------------------------------------------------------
# Test 28: WHY_LLM_NUM_CTX=abc raises LLMError
# ---------------------------------------------------------------------------

def test_num_ctx_non_integer_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """WHY_LLM_NUM_CTX=abc must raise LLMError with appropriate message."""
    _make_openai_compatible_env(monkeypatch)
    monkeypatch.setenv("WHY_LLM_NUM_CTX", "abc")

    with pytest.raises(LLMError, match="WHY_LLM_NUM_CTX must be a positive integer"):
        LLMClient()


# ---------------------------------------------------------------------------
# Test 29: WHY_LLM_NUM_CTX="" treated as unset, auto-couples to default 4096
# ---------------------------------------------------------------------------

def test_num_ctx_empty_string_treated_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """WHY_LLM_NUM_CTX='' is treated as unset; backend gets num_ctx=4096."""
    _make_openai_compatible_env(monkeypatch)
    monkeypatch.setenv("WHY_LLM_NUM_CTX", "")
    monkeypatch.delenv("WHY_LLM_MAX_CTX", raising=False)

    with patch("why._backends.openai_compatible.OpenAICompatibleBackend") as mock_backend_cls, \
         patch("openai.OpenAI"):
        mock_backend_cls.return_value = MagicMock()
        LLMClient()

    _, kwargs = mock_backend_cls.call_args
    assert kwargs.get("num_ctx") == 4096


# ---------------------------------------------------------------------------
# Test 30: WHY_LLM_NUM_CTX unset, auto-couples to openai-compatible default 4096
# ---------------------------------------------------------------------------

def test_num_ctx_unset_auto_couples_to_max_ctx_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """When NUM_CTX and MAX_CTX are unset with openai-compatible, backend gets num_ctx=4096."""
    _make_openai_compatible_env(monkeypatch)
    monkeypatch.delenv("WHY_LLM_NUM_CTX", raising=False)
    monkeypatch.delenv("WHY_LLM_MAX_CTX", raising=False)

    with patch("why._backends.openai_compatible.OpenAICompatibleBackend") as mock_backend_cls, \
         patch("openai.OpenAI"):
        mock_backend_cls.return_value = MagicMock()
        LLMClient()

    _, kwargs = mock_backend_cls.call_args
    assert kwargs.get("num_ctx") == 4096


# ---------------------------------------------------------------------------
# Test 31: WHY_LLM_NUM_CTX unset, WHY_LLM_MAX_CTX=2048 → backend gets num_ctx=2048
# ---------------------------------------------------------------------------

def test_num_ctx_unset_auto_couples_to_explicit_max_ctx(monkeypatch: pytest.MonkeyPatch) -> None:
    """When WHY_LLM_NUM_CTX is unset but WHY_LLM_MAX_CTX=2048, backend gets num_ctx=2048."""
    _make_openai_compatible_env(monkeypatch)
    monkeypatch.delenv("WHY_LLM_NUM_CTX", raising=False)
    monkeypatch.setenv("WHY_LLM_MAX_CTX", "2048")

    with patch("why._backends.openai_compatible.OpenAICompatibleBackend") as mock_backend_cls, \
         patch("openai.OpenAI"):
        mock_backend_cls.return_value = MagicMock()
        LLMClient()

    _, kwargs = mock_backend_cls.call_args
    assert kwargs.get("num_ctx") == 2048


# ---------------------------------------------------------------------------
# Test 32: WHY_LLM_NUM_CTX unset, WHY_LLM_MAX_CTX=0 → backend gets num_ctx=None
# ---------------------------------------------------------------------------

def test_num_ctx_unset_max_ctx_zero_disables(monkeypatch: pytest.MonkeyPatch) -> None:
    """When WHY_LLM_NUM_CTX is unset and WHY_LLM_MAX_CTX=0, backend gets num_ctx=None."""
    _make_openai_compatible_env(monkeypatch)
    monkeypatch.delenv("WHY_LLM_NUM_CTX", raising=False)
    monkeypatch.setenv("WHY_LLM_MAX_CTX", "0")

    with patch("why._backends.openai_compatible.OpenAICompatibleBackend") as mock_backend_cls, \
         patch("openai.OpenAI"):
        mock_backend_cls.return_value = MagicMock()
        LLMClient()

    _, kwargs = mock_backend_cls.call_args
    assert kwargs.get("num_ctx") is None


# ---------------------------------------------------------------------------
# Test 33: WHY_LLM_NUM_CTX=16384 overrides WHY_LLM_MAX_CTX=4096
# ---------------------------------------------------------------------------

def test_num_ctx_explicit_overrides_max_ctx_couple(monkeypatch: pytest.MonkeyPatch) -> None:
    """When WHY_LLM_NUM_CTX=16384 is set, it wins over WHY_LLM_MAX_CTX=4096."""
    _make_openai_compatible_env(monkeypatch)
    monkeypatch.setenv("WHY_LLM_NUM_CTX", "16384")
    monkeypatch.setenv("WHY_LLM_MAX_CTX", "4096")

    with patch("why._backends.openai_compatible.OpenAICompatibleBackend") as mock_backend_cls, \
         patch("openai.OpenAI"):
        mock_backend_cls.return_value = MagicMock()
        LLMClient()

    _, kwargs = mock_backend_cls.call_args
    assert kwargs.get("num_ctx") == 16384


# ---------------------------------------------------------------------------
# Test 34: groq provider silently ignores WHY_LLM_NUM_CTX
# ---------------------------------------------------------------------------

def test_num_ctx_groq_provider_no_effect(monkeypatch: pytest.MonkeyPatch) -> None:
    """WHY_LLM_NUM_CTX is silently ignored on groq; no error, GroqBackend is constructed."""
    monkeypatch.setenv("WHY_LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "k")
    monkeypatch.setenv("WHY_LLM_NUM_CTX", "8192")

    with patch("why.llm.groq_sdk.Groq") as mock_groq:
        mock_groq.return_value = MagicMock()
        client = LLMClient()

    assert client.provider == "groq"
    mock_groq.assert_called_once_with(api_key="k")
    from why.llm import GroqBackend
    assert isinstance(client._backend, GroqBackend)
    assert not hasattr(client._backend, "_num_ctx")


# ---------------------------------------------------------------------------
# Test 35: num_ctx DEBUG log emitted when num_ctx is set
# ---------------------------------------------------------------------------

def test_openai_compatible_backend_logs_num_ctx_debug(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When num_ctx is set, chat() must emit a DEBUG log line 'num_ctx=N (Ollama)'."""
    import logging as _logging
    from unittest.mock import patch as _patch

    from why._backends.openai_compatible import OpenAICompatibleBackend

    with _patch("openai.OpenAI") as mock_openai_cls:
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "ok"
        mock_response.usage.prompt_tokens = 1
        mock_response.usage.completion_tokens = 1
        mock_response.usage.total_tokens = 2
        mock_openai_cls.return_value.chat.completions.create.return_value = mock_response

        backend = OpenAICompatibleBackend(
            base_url="http://localhost:11434/v1",
            api_key="not-needed",
            num_ctx=4096,
        )
        with caplog.at_level(_logging.DEBUG, logger="why.llm"):
            backend.chat("llama3", [{"role": "user", "content": "hi"}])

    assert any("num_ctx=4096 (Ollama)" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Test 36: invalid WHY_LLM_NUM_CTX echoes the bad value in the error message
# ---------------------------------------------------------------------------

def test_num_ctx_invalid_error_echoes_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """The LLMError must include the bad value the user provided."""
    _make_openai_compatible_env(monkeypatch)
    monkeypatch.setenv("WHY_LLM_NUM_CTX", "8k")
    with pytest.raises(LLMError, match="got '8k'"):
        LLMClient()
