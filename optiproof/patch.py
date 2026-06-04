"""Apply a candidate to a file and render diffs.

Candidates are represented as the *replacement source for one function* (plus an
optional module-level prelude for new imports) rather than as an LLM-authored
unified diff. Splicing a known line span is far more reliable than parsing a
model-generated patch, and we render the unified diff ourselves with ``difflib``
purely for the human report.
"""

from __future__ import annotations

import difflib

from .models import Candidate, Target


def _inject_prelude(text: str, prelude: str) -> str:
    """Insert module-level code, keeping any shebang and ``__future__`` imports first."""
    lines = text.splitlines(keepends=True)
    insert_at = 0
    if lines and lines[0].startswith("#!"):
        insert_at = 1
    for i in range(insert_at, len(lines)):
        if lines[i].lstrip().startswith("from __future__"):
            insert_at = i + 1
    if not prelude.endswith("\n"):
        prelude = prelude + "\n"
    return "".join(lines[:insert_at] + [prelude] + lines[insert_at:])


def apply_candidate(file_text: str, target: Target, candidate: Candidate) -> str:
    """Return ``file_text`` with the target function replaced by the candidate."""
    lines = file_text.splitlines(keepends=True)
    start = target.start_line - 1          # 1-based inclusive -> 0-based
    end = target.end_line                  # slice end is exclusive == end_line
    if start < 0 or end > len(lines) or start >= end:
        raise ValueError(
            f"target span {target.start_line}-{target.end_line} out of range for {target.file}"
        )
    block = candidate.new_source
    if not block.endswith("\n"):
        block = block + "\n"
    new_text = "".join(lines[:start] + [block] + lines[end:])
    if candidate.module_prelude:
        new_text = _inject_prelude(new_text, candidate.module_prelude)
    return new_text


def unified_diff(old_text: str, new_text: str, path: str) -> str:
    """Render a unified diff for the report."""
    return "".join(
        difflib.unified_diff(
            old_text.splitlines(keepends=True),
            new_text.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
        )
    )
