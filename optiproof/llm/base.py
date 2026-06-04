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
    def create(provider: str = "anthropic", model: Optional[str] = None, **kwargs) -> "LLMProvider":
        if provider == "null":
            from .null_provider import NullProvider

            return NullProvider(**kwargs)
        if provider == "anthropic":
            from .anthropic_provider import AnthropicProvider

            return AnthropicProvider(model=model)
        raise ValueError(f"unknown LLM provider: {provider!r}")
