"""Default provider — Anthropic Claude via the Messages API.

Uses forced structured output (a tool call) so candidates come back as validated
JSON, and prompt caching on the static prefix (system + target source) so repeated
rounds only pay for the small, varying feedback.
"""

from __future__ import annotations

from typing import Optional

from ..models import Candidate, OptimizeKind, Target
from .base import LLMProvider
from .prompts import SYSTEM, TOOL, context_block, feedback_block

DEFAULT_MODEL = "claude-sonnet-4-6"


class AnthropicProvider(LLMProvider):
    def __init__(self, model: Optional[str] = None, max_tokens: int = 4096):
        self.model = model or DEFAULT_MODEL
        self.max_tokens = max_tokens
        self._round = 0
        try:
            import anthropic
        except ImportError as e:  # pragma: no cover - optional dep
            raise RuntimeError(
                "the Anthropic provider needs the 'anthropic' package "
                "(`pip install anthropic`), or use a different --provider"
            ) from e
        self._client = anthropic.Anthropic()

    def propose(self, target: Target, feedback: list[str], n: int) -> list[Candidate]:
        self._round += 1
        system = [
            {"type": "text", "text": SYSTEM},
            {"type": "text", "text": context_block(target), "cache_control": {"type": "ephemeral"}},
        ]
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            tools=[TOOL],
            tool_choice={"type": "tool", "name": "propose_optimizations"},
            messages=[{"role": "user", "content": feedback_block(target.symbol, feedback, n)}],
        )
        return self._parse(resp, target)

    def _parse(self, resp, target: Target) -> list[Candidate]:
        raw: list[dict] = []
        for block in resp.content:
            if getattr(block, "type", None) == "tool_use" and block.name == "propose_optimizations":
                raw = (getattr(block, "input", None) or {}).get("candidates", []) or []
                break

        candidates: list[Candidate] = []
        for i, c in enumerate(raw):
            if not isinstance(c, dict):
                continue
            source = (c.get("new_source") or "").strip()
            if not source:
                continue
            try:
                kind = OptimizeKind(c.get("kind") or "rewrite")
            except ValueError:
                kind = OptimizeKind.REWRITE
            prelude = (c.get("module_prelude") or "").strip() or None
            candidates.append(
                Candidate(
                    id=f"{target.symbol}-r{self._round}-{i}",
                    kind=kind,
                    title=(c.get("title") or "")[:120],   # tolerate JSON null
                    rationale=c.get("rationale") or "",
                    new_source=source,
                    module_prelude=prelude,
                )
            )
        return candidates
