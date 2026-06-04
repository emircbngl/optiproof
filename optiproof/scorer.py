"""Rank surviving candidates: speedup first, then prefer the simpler kind / shorter source."""

from __future__ import annotations

from typing import Optional

from .models import Candidate, OptimizeKind

_KIND_ORDER = {OptimizeKind.REWRITE: 0, OptimizeKind.NATIVE: 1, OptimizeKind.PORT: 2}


def better_of(current: Optional[Candidate], cand: Optional[Candidate]) -> Optional[Candidate]:
    if current is None:
        return cand
    if cand is None:
        return current
    cur_ratio = current.verdict.ratio if current.verdict else 0.0
    new_ratio = cand.verdict.ratio if cand.verdict else 0.0
    if new_ratio > cur_ratio:
        return cand
    if new_ratio < cur_ratio:
        return current
    # tie-break: simpler kind, then shorter source (less risk / more readable)
    if _KIND_ORDER.get(cand.kind, 9) < _KIND_ORDER.get(current.kind, 9):
        return cand
    if len(cand.new_source) < len(current.new_source):
        return cand
    return current
