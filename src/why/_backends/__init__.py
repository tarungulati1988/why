"""Internal backends package — protocol and concrete LLM provider adapters."""
from why._backends.base import Backend, ChatResult

__all__ = ["Backend", "ChatResult"]
