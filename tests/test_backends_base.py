"""Tests for why._backends.base — Backend Protocol and ChatResult dataclass.

Written BEFORE the implementation (TDD). All tests target the public surface
of the abstraction; no Groq SDK or network access is required.
"""

from __future__ import annotations

from typing import Any

import pytest

from why._backends.base import Backend, ChatResult


# ---------------------------------------------------------------------------
# ChatResult
# ---------------------------------------------------------------------------


def test_chatresult_defaults() -> None:
    """ChatResult(content=...) sets token counts to None by default."""
    result = ChatResult(content="x")
    assert result.content == "x"
    assert result.prompt_tokens is None
    assert result.completion_tokens is None


def test_chatresult_round_trip() -> None:
    """ChatResult stores all fields when explicitly provided."""
    result = ChatResult(content="x", prompt_tokens=3, completion_tokens=4, total_tokens=7)
    assert result.content == "x"
    assert result.prompt_tokens == 3
    assert result.completion_tokens == 4
    assert result.total_tokens == 7


# ---------------------------------------------------------------------------
# Backend Protocol — structural compatibility
# ---------------------------------------------------------------------------


class _ConcreteBackend:
    """Minimal concrete implementation used to verify the Protocol shape."""

    def chat(self, model: str, payload: list[dict], **extra: Any) -> ChatResult:
        return ChatResult(content=f"echo:{model}", prompt_tokens=1, completion_tokens=2)


def test_backend_protocol_structural_compatibility() -> None:
    """A class with the right chat() signature satisfies the Backend Protocol."""
    backend = _ConcreteBackend()
    result = backend.chat("m", [{"role": "user", "content": "hi"}], foo=1)
    assert isinstance(result, ChatResult)
    assert result.content == "echo:m"
    assert result.prompt_tokens == 1
    assert result.completion_tokens == 2
