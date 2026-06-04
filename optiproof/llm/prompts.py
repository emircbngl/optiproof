"""Prompts + the structured-output tool schema for candidate generation.

The static parts (system instructions + the target source) are sent as cacheable
blocks; only the per-round feedback varies, so repeated rounds hit the prompt
cache.
"""

from __future__ import annotations

from ..models import Target

SYSTEM = """You are an expert performance engineer. You are given ONE function and must \
propose alternative implementations that are FASTER but BEHAVIOURALLY IDENTICAL.

Hard rules — a violation gets the candidate automatically rejected by a measured harness:
- Keep the function name and signature EXACTLY the same.
- Preserve semantics exactly: same return values, same raised exception types, same stdout, \
for every input. The harness differential-tests you against the original on hundreds of inputs.
- Return each candidate's `new_source` as the COMPLETE replacement function (the full `def ...:` \
block), not a diff and not a fragment.
- If you need imports, put them in `module_prelude` (e.g. "import numpy as np"); they are added \
at the top of the file.

Optimization levels to consider, in rough priority:
1. Algorithmic / data-structure (O(n^2) -> O(n), list membership -> set, add memoization, \
   hoist invariant work out of loops, avoid quadratic string concatenation).
2. `kind="native"`: delegate the hot work to a fast native library (NumPy, etc.) when that \
   preserves behaviour. This is often the biggest win for numeric/array code.

Only the harness decides if you succeeded — it measures real speedup and proves correctness. \
Prefer a few high-confidence, genuinely different approaches over many trivial variants."""


def context_block(target: Target) -> str:
    params = ", ".join(f"{k}: {v or 'unannotated'}" for k, v in target.param_types.items()) or "(none)"
    return (
        f"Language: {target.language}\n"
        f"Function to optimize: {target.symbol}\n"
        f"Parameters: {params}\n\n"
        f"Current implementation:\n```{target.language}\n{target.source}\n```"
    )


def feedback_block(symbol: str, feedback: list[str], n: int) -> str:
    msg = f"Propose {n} optimized implementations of `{symbol}` and return them as structured output."
    if feedback:
        joined = "\n".join(f"- {f}" for f in feedback[-12:])
        msg = (
            "Previous attempts were rejected by the harness for these reasons:\n"
            f"{joined}\n\nLearn from them and " + msg
        )
    return msg


TOOL = {
    "name": "propose_optimizations",
    "description": "Return optimized, behaviour-preserving rewrites of the target function.",
    "input_schema": {
        "type": "object",
        "properties": {
            "candidates": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "kind": {
                            "type": "string",
                            "enum": ["rewrite", "native"],
                            "description": "rewrite = same-language; native = delegate to a native lib",
                        },
                        "title": {"type": "string", "description": "short label, e.g. 'O(n) set lookup'"},
                        "rationale": {"type": "string", "description": "why this is faster"},
                        "new_source": {"type": "string", "description": "the COMPLETE replacement def block"},
                        "module_prelude": {
                            "type": "string",
                            "description": "optional top-of-file code such as imports; omit if none",
                        },
                    },
                    "required": ["kind", "title", "new_source"],
                },
            }
        },
        "required": ["candidates"],
    },
}
