"""Differential testing — the primary correctness oracle.

We compare the *candidate* against the *original* on identical inputs, rather
than against generated golden values: an LLM that writes a wrong optimization can
also write a wrong assertion, but it's far harder to make a wrong implementation
match the original on hundreds of shared inputs.

Determinism is checked up front by running the original twice. If it isn't
self-consistent (RNG/time/threads), the target is quarantined and differential
testing is disabled (the correctness layer then falls back to existing tests, or
refuses to claim correctness — it never silently passes).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..models import DifferentialResult
from .compare import compare_observations


@dataclass
class BaselineBehavior:
    observations: list = field(default_factory=list)
    deterministic: bool = True
    warning: str = ""


def _short(value, limit: int = 160) -> str:
    r = repr(value)
    return r if len(r) <= limit else r[:limit] + "…"


def record_baseline(base_ws, target, inputs, adapter, sandbox, timeout: float = 30.0):
    """Run the original twice; return (BaselineBehavior, error). error != '' means abort."""
    first = adapter.observe(base_ws, target, inputs, sandbox, timeout)
    if not first.ok:
        return None, first.error
    second = adapter.observe(base_ws, target, inputs, sandbox, timeout)
    if not second.ok:
        return None, second.error

    # Two runs of the original must agree in COUNT and in every value, else it is
    # non-deterministic (e.g. output count varies) and gets quarantined.
    deterministic = len(first.observations) == len(second.observations)
    if deterministic:
        for a, b in zip(first.observations, second.observations):
            equal, _ = compare_observations(a, b)
            if not equal:
                deterministic = False
                break

    warning = "" if deterministic else "original is non-deterministic; differential testing disabled"
    return BaselineBehavior(first.observations, deterministic, warning), ""


def compare_candidate(
    baseline: BaselineBehavior,
    cand_ws,
    target,
    inputs,
    adapter,
    sandbox,
    rel_tol: float,
    abs_tol: float,
    timeout: float = 30.0,
) -> DifferentialResult:
    if not baseline.deterministic:
        return DifferentialResult(
            equivalent=True, checked=0, quarantined=True, warning=baseline.warning
        )

    res = adapter.observe(cand_ws, target, inputs, sandbox, timeout)
    if not res.ok:
        return DifferentialResult(equivalent=False, detail=f"candidate failed to run: {res.error}")

    # Fail closed on a count mismatch: never declare equivalence on a truncated prefix
    # (e.g. an output stream cut short). The candidate must answer every input.
    if len(res.observations) != len(baseline.observations):
        return DifferentialResult(
            equivalent=False,
            checked=min(len(res.observations), len(baseline.observations)),
            detail=(
                f"candidate produced {len(res.observations)} outputs vs "
                f"baseline {len(baseline.observations)}"
            ),
        )
    for i in range(len(baseline.observations)):
        equal, reason = compare_observations(
            baseline.observations[i], res.observations[i], rel_tol, abs_tol
        )
        if not equal:
            inp = inputs[i] if i < len(inputs) else None
            return DifferentialResult(
                equivalent=False, checked=i + 1, counterexample=_short(inp), detail=reason
            )
    return DifferentialResult(equivalent=True, checked=len(baseline.observations))
