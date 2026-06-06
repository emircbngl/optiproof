"""Candidate generation via the local ``claude`` CLI (Claude Code) in headless mode.

This uses the user's logged-in Claude Code session (e.g. a Pro/Max subscription) —
**no ANTHROPIC_API_KEY required.** The raw Anthropic SDK can't use subscription
auth, but the ``claude`` CLI can, so we shell out to ``claude -p`` and parse the
candidates it returns.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from typing import Optional

from ..models import Candidate, OptimizeKind, Target
from .base import LLMProvider
from .prompts import SYSTEM, context_block, feedback_block

_OUTPUT = (
    "\n\nRespond with ONLY a JSON array (no prose, no markdown code fences) of up to {n} objects. "
    "Each object has: \"kind\" (\"rewrite\" or \"native\"), \"title\" (short string), "
    "\"rationale\" (short string), \"new_source\" (the COMPLETE replacement function as a string), "
    "and optional \"module_prelude\" (top-of-file code such as imports, or null). "
    "Do NOT use any tools or read any files — the function source is included above. "
    "Output the JSON array and nothing else."
)


def _extract_result(stdout: str) -> str:
    """``claude -p --output-format json`` wraps the answer in a result envelope."""
    stdout = stdout.strip()
    try:
        obj = json.loads(stdout)
    except Exception:
        return stdout
    if isinstance(obj, dict):
        if obj.get("is_error"):
            raise RuntimeError(f"claude CLI returned an error: {str(obj.get('result') or obj)[:300]}")
        return obj.get("result") or obj.get("text") or ""
    return stdout


def _extract_json_array(text: str) -> list:
    """Pull the first balanced JSON array out of the model's text (tolerant of fences/prose)."""
    text = text.strip()
    if text.startswith("```"):
        body = text.split("```", 2)
        if len(body) >= 2:
            chunk = body[1]
            text = chunk[4:] if chunk.startswith("json") else chunk
            text = text.strip()
    start = text.find("[")
    if start == -1:
        return []
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "[":
            depth += 1
        elif text[i] == "]":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except Exception:
                    return []
    return []


class ClaudeCodeProvider(LLMProvider):
    def __init__(
        self,
        model: Optional[str] = None,
        binary: str = "claude",
        max_budget_usd: Optional[float] = None,
        timeout: float = 300.0,
    ):
        if shutil.which(binary) is None:
            raise RuntimeError(
                "`claude` CLI not found on PATH; install Claude Code (and log in) "
                "or use --provider anthropic with an API key"
            )
        self.binary = binary
        self.model = model
        self.max_budget_usd = max_budget_usd
        self.timeout = timeout
        self._round = 0

    def propose(self, target: Target, feedback: list[str], n: int) -> list[Candidate]:
        self._round += 1
        prompt = (
            SYSTEM
            + "\n\n"
            + context_block(target)
            + "\n\n"
            + feedback_block(target.symbol, feedback, n)
            + _OUTPUT.format(n=n)
        )
        cmd = [self.binary, "-p", "--output-format", "json", "--no-session-persistence"]
        if self.model:
            cmd += ["--model", self.model]
        if self.max_budget_usd is not None:
            cmd += ["--max-budget-usd", str(self.max_budget_usd)]

        proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=self.timeout)

        # claude -p prints a JSON envelope to stdout even on failure; surface its error clearly.
        envelope = None
        try:
            envelope = json.loads(proc.stdout.strip())
        except Exception:
            envelope = None
        if proc.returncode != 0 or (isinstance(envelope, dict) and envelope.get("is_error")):
            msg = ""
            if isinstance(envelope, dict):
                msg = str(envelope.get("result") or "")
                if envelope.get("api_error_status"):
                    msg = f"{msg} (status {envelope['api_error_status']})"
            msg = msg.strip() or proc.stderr.strip() or f"exit code {proc.returncode}"
            if "auth" in msg.lower() or "401" in msg:
                msg += (" — log in to Claude Code in this terminal (`claude`), or set ANTHROPIC_API_KEY. "
                        "Note: subscription auth is not available to sandboxed/agent shells.")
            raise RuntimeError(f"claude CLI: {msg[:500]}")

        raw = _extract_json_array(_extract_result(proc.stdout))
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
            candidates.append(
                Candidate(
                    id=f"{target.symbol}-cc{self._round}-{i}",
                    kind=kind,
                    title=(c.get("title") or "")[:120],
                    rationale=c.get("rationale") or "",
                    new_source=source,
                    module_prelude=(c.get("module_prelude") or "").strip() or None,
                )
            )
        return candidates
