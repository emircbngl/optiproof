"""The agent loop: measure → generate → prove → keep the best, with anti-luck re-measure.

This is the brain. It owns control flow only; everything language-, provider-, or
statistics-specific lives behind an interface. The cheap-fail gate order
(build → existing tests → differential → benchmark) means correctness is
established cheaply and only genuinely-correct candidates are ever benchmarked.
"""

from __future__ import annotations

from typing import Optional

from .adapters.base import AdapterRegistry, LanguageAdapter
from .bench import runner
from .bench.environment import describe_host, noise_note
from .bench.stats import speedup_verdict
from .llm.base import LLMProvider
from .models import (
    Candidate,
    OptimizationResult,
    OptimizeRequest,
    RejectedCandidate,
    Target,
)
from .patch import apply_candidate, unified_diff
from .sandbox.base import Sandbox
from .sandbox.workspace import Workspace
from .scorer import better_of
from .verify.correctness import check_candidate
from .verify.differential import BaselineBehavior, record_baseline
from .verify.input_gen import describe_inputs, generate_inputs, make_workload


def _first_line(text: str) -> str:
    for line in (text or "").splitlines():
        if line.strip():
            return line.strip()[:200]
    return ""


def evaluate_candidate(
    cand: Candidate,
    base_ws: Workspace,
    target: Target,
    inputs: list,
    workload,
    adapter: LanguageAdapter,
    sandbox: Sandbox,
    baseline: BaselineBehavior,
    base_meas,
    request: OptimizeRequest,
    observe_timeout: float = 30.0,
    bench_timeout: float = 120.0,
) -> Candidate:
    """Run one candidate through build → correctness → benchmark, annotating it."""
    cw = base_ws.fork()
    try:
        try:
            cw.apply(target, cand)
            cand.applied = True
        except Exception as e:  # malformed candidate source
            cand.rejected_reason = f"unapplyable: {e}"
            return cand

        # GATE A: build / syntax
        build = adapter.build(cw, sandbox)
        cand.build_ok = build.ok
        cand.build_error = build.error
        if not build.ok:
            cand.rejected_reason = f"build failed: {_first_line(build.error)}"
            return cand

        # GATE B: correctness (existing tests + differential)
        corr = check_candidate(
            cw, target, inputs, adapter, sandbox, baseline, observe_timeout=observe_timeout
        )
        cand.correctness = corr
        if not corr.ok:
            cand.rejected_reason = corr.reason
            return cand

        # GATE C: speed (only correct candidates get here)
        meas, err = runner.measure(
            cw, target, workload, adapter, sandbox,
            warmup=request.warmup, min_rounds=request.min_runs,
            max_rounds=request.max_runs, timeout=bench_timeout,
        )
        if meas is None:
            cand.rejected_reason = f"benchmark failed: {err}"
            return cand
        cand.measurement = meas

        verdict = speedup_verdict(
            base_meas.samples, meas.samples,
            threshold=request.threshold, alpha=request.significance_alpha, seed=request.seed,
        )
        cand.verdict = verdict
        if not verdict.is_faster:
            cand.rejected_reason = verdict.reason
        return cand
    finally:
        cw.cleanup()


def _is_survivor(cand: Candidate) -> bool:
    return cand.rejected_reason is None and cand.verdict is not None and cand.verdict.is_faster


def optimize(request: OptimizeRequest, provider: Optional[LLMProvider] = None) -> OptimizationResult:
    adapter = AdapterRegistry.detect(request.path, request.language)
    target = adapter.locate_target(request.path, request.selector)
    sandbox = Sandbox.create(request.sandbox, toolchain_image=request.toolchain_image)
    provider = provider or LLMProvider.create(request.provider, request.model)

    result = OptimizationResult(target=target, language=target.language, runtime=adapter.runtime_label())
    result.inputs_tested = describe_inputs(target)
    notes: list[str] = []
    base_ws = Workspace.fork_from_file(target.file)
    try:
        # --- establish a trustworthy baseline (measure first) ---
        build = adapter.build(base_ws, sandbox)
        if not build.ok:
            result.notes = [f"original does not build: {_first_line(build.error)}"]
            return result
        base_tests = adapter.run_tests(base_ws, sandbox)
        if base_tests.has_tests and not base_tests.ok:
            result.notes = ["original's own tests fail — refusing to optimize broken code"]
            return result

        inputs = generate_inputs(target, seed=request.seed, n=request.num_diff_inputs)
        workload = make_workload(target, seed=request.seed, big=request.workload_size)

        baseline, err = record_baseline(base_ws, target, inputs, adapter, sandbox)
        if baseline is None:
            result.unbenchmarkable = True
            result.notes = [
                "UNBENCHMARKABLE — couldn't run the original on generated inputs; the generator likely "
                "produced the wrong shape/type for a parameter (annotate the params). detail: "
                + _first_line(err)
            ]
            return result
        if not baseline.deterministic:
            tail = " — relying on existing tests" if base_tests.has_tests else " — correctness cannot be proven"
            notes.append("⚠ " + baseline.warning + tail)

        raised = sum(1 for o in baseline.observations if not o.ok)
        if baseline.observations and raised == len(baseline.observations):
            result.unbenchmarkable = True
            result.notes = notes + [
                f"UNBENCHMARKABLE — the original raised on all {len(baseline.observations)} generated "
                "inputs; the generator couldn't produce inputs it accepts (likely a scalar or "
                "specially-typed parameter). Annotate the parameters and retry."
            ]
            return result

        base_meas, err = runner.measure(
            base_ws, target, workload, adapter, sandbox,
            warmup=request.warmup, min_rounds=request.min_runs, max_rounds=request.max_runs,
        )
        if base_meas is None:
            result.unbenchmarkable = True
            result.notes = notes + [
                "UNBENCHMARKABLE — couldn't benchmark the original (input generation likely produced "
                "inputs it rejects). detail: " + _first_line(err)
            ]
            return result
        result.baseline = base_meas

        # --- generate → prove loop ---
        best: Optional[Candidate] = None
        feedback: list[str] = []
        rounds = 0
        evaluated = 0
        while rounds < request.max_rounds:
            rounds += 1
            try:
                candidates = provider.propose(target, feedback, request.candidates_per_round)
            except Exception as e:
                notes.append(f"provider error: {e}")
                break
            if not candidates:
                break

            built = 0
            for cand in candidates:
                evaluated += 1
                evaluate_candidate(
                    cand, base_ws, target, inputs, workload, adapter, sandbox, baseline, base_meas, request
                )
                if cand.build_ok:
                    built += 1
                if _is_survivor(cand):
                    best = better_of(best, cand)
                else:
                    result.rejected.append(
                        RejectedCandidate(
                            id=cand.id, kind=cand.kind, title=cand.title or cand.id,
                            reason=cand.rejected_reason or "rejected",
                            ratio=cand.verdict.ratio if cand.verdict else None,
                        )
                    )
                    feedback.append(f"{cand.title or cand.id}: {cand.rejected_reason}")

            if best is not None and best.verdict.ratio >= request.satisfied_at:
                break
            if built == 0:
                notes.append("no candidate compiled this round — stopping")
                break

        # --- anti-luck: re-measure the winner from scratch with extra rounds ---
        if best is not None:
            cw = base_ws.fork()
            try:
                cw.apply(target, best)
                confirm, err = runner.measure(
                    cw, target, workload, adapter, sandbox,
                    warmup=request.warmup, min_rounds=request.min_runs * 2, max_rounds=request.max_runs * 2,
                )
            finally:
                cw.cleanup()
            if confirm is None:
                # Fail closed: if we can't re-confirm the win, don't ship it.
                result.rejected.append(
                    RejectedCandidate(
                        id=best.id, kind=best.kind, title=best.title or best.id,
                        reason=f"discarded — confirmation re-measure failed: {err}",
                        ratio=best.verdict.ratio if best.verdict else None,
                    )
                )
                notes.append("winner discarded — confirmation re-measure failed (fail-closed)")
                best = None
            else:
                v2 = speedup_verdict(
                    base_meas.samples, confirm.samples,
                    threshold=request.threshold, alpha=request.significance_alpha, seed=request.seed + 1,
                )
                if not v2.is_faster:
                    result.rejected.append(
                        RejectedCandidate(
                            id=best.id, kind=best.kind, title=best.title or best.id,
                            reason=f"did not survive re-measure: {v2.reason}", ratio=v2.ratio,
                        )
                    )
                    notes.append("winner demoted after from-scratch re-measure (regression to the mean)")
                    best = None
                else:
                    best.measurement = confirm
                    best.verdict = v2

        result.rounds = rounds
        result.candidates_evaluated = evaluated

        if best is not None:
            original_text = target.file.read_text()
            new_text = apply_candidate(original_text, target, best)
            result.improved = True
            result.best = best
            result.speedup = best.verdict.ratio
            result.ci = (best.verdict.ci_low, best.verdict.ci_high)
            result.diff = unified_diff(original_text, new_text, target.file.name)

        nn = noise_note(base_meas)
        if nn:
            notes.append(nn)
        notes.append(f"host: {describe_host()}")
        result.notes = notes
        return result
    finally:
        base_ws.cleanup()
        sandbox.cleanup()
