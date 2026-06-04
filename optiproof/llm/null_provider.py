"""Deterministic provider for tests — returns canned candidates, no network.

Lets the whole agent loop be exercised in CI with zero API calls and zero timing
flakiness: feed it a known-correct-fast candidate, a correct-but-slow one, and a
fast-but-wrong one, and assert the loop accepts/ rejects exactly the right ones.
"""

from __future__ import annotations

from typing import Optional

from ..models import Candidate, Target
from .base import LLMProvider


class NullProvider(LLMProvider):
    def __init__(
        self,
        candidates: Optional[list[Candidate]] = None,
        by_symbol: Optional[dict[str, list[Candidate]]] = None,
    ):
        self._candidates = candidates or []
        self._by_symbol = by_symbol or {}
        self._calls = 0

    def propose(self, target: Target, feedback: list[str], n: int) -> list[Candidate]:
        self._calls += 1
        if self._calls > 1:  # canned candidates are offered once; then the loop drains
            return []
        pool = self._by_symbol.get(target.symbol, self._candidates)
        return list(pool)
