"""Probe-based input-shape inference.

Name/annotation heuristics mislabel parameters they don't recognize — e.g. a scalar
``rho`` in ``f(n, m, rho)`` gets the ambiguous-numeric *list* default, so the original
crashes on every generated input and the function looks UNBENCHMARKABLE.

This module refines such parameters by *running the original*: for each ambiguous param it
tries a few candidate shapes and keeps the one the original actually accepts (fewest
exceptions). Crucially the scalar candidate is ``num`` (int+float), so a discovered scalar
is still exercised across both domains — soundness is preserved, not traded for reach.

Only ambiguous (unannotated + unrecognized-name) parameters are probed; everything else
keeps its prior, so annotated code (and Rust/Go, which is always typed) costs no probes.
"""

from __future__ import annotations

import random

from .input_gen import gen_value, is_ambiguous, prior_tags

# Candidate shapes, in tie-break priority order. `num` (scalar int+float) first so a param
# that accepts both a scalar and a list is treated as the simpler scalar.
_PROBE_CANDIDATES = ["num", "list_num", "str", "list_str", "bool"]


def _probe_inputs(tags: list[str], seed: int, k: int = 5, size: int = 6) -> list[tuple]:
    r = random.Random(seed)
    rows = []
    for i in range(k):
        row = []
        for t in tags:
            if t == "num":
                row.append(gen_value("float" if i % 2 else "int", r, size))
            elif t == "list_num":
                row.append(gen_value("list_float" if i % 2 else "list_int", r, size))
            else:
                row.append(gen_value(t, r, size))
        rows.append(tuple(row))
    return rows


def probe_tags(target, adapter, base_ws, sandbox, seed: int = 1234, timeout: float = 15.0) -> list[str]:
    """Return a per-parameter tag list, refining ambiguous params by probing the original."""
    tags = prior_tags(target)
    params = list(target.param_types.items())
    ambiguous = [i for i, (name, ann) in enumerate(params) if is_ambiguous(name, ann)]
    if not ambiguous:
        return tags

    for i in ambiguous:
        best_tag, best_score = tags[i], -1
        for cand in _PROBE_CANDIDATES:
            trial = list(tags)
            trial[i] = cand
            res = adapter.observe(base_ws, target, _probe_inputs(trial, seed + i), sandbox, timeout=timeout)
            score = sum(1 for o in res.observations if o.ok) if res.ok else -1
            if score > best_score:
                best_score, best_tag = score, cand
        tags[i] = best_tag
    return tags
