"""Internal backends package — protocol and concrete LLM provider adapters."""
from why._backends.base import Backend, ChatResult
from why._backends.openai_compatible import OpenAICompatibleBackend

__all__ = ["Backend", "ChatResult", "OpenAICompatibleBackend"]
