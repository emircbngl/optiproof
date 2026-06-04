"""Correctness gate: existing tests + differential testing, cheap-fail ordered.

Any behavior change => reject. A non-deterministic target with no tests is
*unverifiable*, so we refuse it rather than pretend it's correct.
"""

from __future__ import annotations

from ..models import CorrectnessResult
from .compare import ABS_TOL, REL_TOL
from .differential import BaselineBehavior, compare_candidate


def check_candidate(
    cand_ws,
    target,
    inputs,
    adapter,
    sandbox,
    baseline: BaselineBehavior,
    rel_tol: float = REL_TOL,
    abs_tol: float = ABS_TOL,
    observe_timeout: float = 30.0,
) -> CorrectnessResult:
    # 1) existing tests (highest signal, cheapest)
    tests = adapter.run_tests(cand_ws, sandbox)
    if tests.has_tests and not tests.ok:
        return CorrectnessResult(
            ok=False, tests=tests, reason=f"existing tests failed ({tests.failed} failing)"
        )

    # 2) differential testing
    diff = compare_candidate(
        baseline, cand_ws, target, inputs, adapter, sandbox, rel_tol, abs_tol, observe_timeout
    )

    if diff.quarantined:
        if tests.has_tests:
            return CorrectnessResult(
                ok=True, tests=tests, differential=diff,
                reason="verified by existing tests only (non-deterministic target)",
            )
        return CorrectnessResult(
            ok=False, tests=tests, differential=diff,
            reason="cannot verify: non-deterministic target and no tests",
        )

    if not diff.equivalent:
        where = f" on input {diff.counterexample}" if diff.counterexample else ""
        return CorrectnessResult(
            ok=False, tests=tests, differential=diff,
            reason=f"behavior changed: {diff.detail}{where}",
        )

    return CorrectnessResult(ok=True, tests=tests, differential=diff)
