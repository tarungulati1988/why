"""Internal backends package — re-exports Backend protocol and concrete adapters.

Consumers should import from why._backends rather than from submodules directly.
"""
from why._backends.base import Backend, ChatResult
from why._backends.openai_compatible import OpenAICompatibleBackend

__all__ = ["Backend", "ChatResult", "OpenAICompatibleBackend"]
