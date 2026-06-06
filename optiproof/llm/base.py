"""Pluggable candidate generators. The orchestrator only ever sees Candidates."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from ..models import Candidate, Target


class LLMProvider(ABC):
    @abstractmethod
    def propose(self, target: Target, feedback: list[str], n: int) -> list[Candidate]:
        """Return up to ``n`` candidate optimizations for ``target``.

        ``feedback`` accumulates reasons prior candidates were rejected (build
        errors, differential counterexamples, "not significant") so the next
        round is informed rather than blind.
        """

    @staticmethod
    def create(provider: str = "auto", model: Optional[str] = None, **kwargs) -> "LLMProvider":
        if provider == "auto":
            import os
            import shutil

            if os.environ.get("ANTHROPIC_API_KEY"):
                provider = "anthropic"
            elif shutil.which("claude"):
                provider = "claude-code"   # use the logged-in Claude Code subscription — no API key
            else:
                raise RuntimeError(
                    "no LLM provider available: set ANTHROPIC_API_KEY, install/log in to the "
                    "`claude` CLI, or pass --provider null"
                )
        if provider == "null":
            from .null_provider import NullProvider

            return NullProvider(**kwargs)
        if provider == "anthropic":
            from .anthropic_provider import AnthropicProvider

            return AnthropicProvider(model=model)
        if provider in ("claude-code", "claude", "subscription"):
            from .claude_code_provider import ClaudeCodeProvider

            return ClaudeCodeProvider(model=model)
        raise ValueError(f"unknown LLM provider: {provider!r}")
