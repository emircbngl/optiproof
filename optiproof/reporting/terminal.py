"""Human-readable report — diff + measured speedup + correctness evidence, or an honest no-win."""

from __future__ import annotations

from ..models import OptimizationResult


def fmt_time(s: float) -> str:
    if s <= 0:
        return "0"
    if s < 1e-6:
        return f"{s * 1e9:.1f} ns"
    if s < 1e-3:
        return f"{s * 1e6:.2f} us"
    if s < 1:
        return f"{s * 1e3:.2f} ms"
    return f"{s:.3f} s"


def render(result: OptimizationResult) -> str:
    t = result.target
    out: list[str] = [f"OptiProof — {t.file.name}::{t.symbol}   [{result.language} · {result.runtime}]"]

    if result.improved and result.best and result.best.verdict and result.best.measurement:
        v = result.best.verdict
        b = result.baseline
        c = result.best.measurement
        out.append("Status:    VERIFIED FASTER")
        out.append(
            f"Speedup:   {v.ratio:.2f}x faster  "
            f"({fmt_time(b.median)} -> {fmt_time(c.median)} median; 95% CI {v.ci_low:.2f}-{v.ci_high:.2f}x; p={v.p_value:.1e})"
        )
        out.append(f"Approach:  [{result.best.kind.value}] {result.best.title}")
        if result.best.rationale:
            out.append(f"           {result.best.rationale}")
        corr = result.best.correctness
        if corr:
            ev = []
            if corr.tests and corr.tests.has_tests:
                ev.append(f"existing tests {corr.tests.passed} passed")
            if corr.differential and not corr.differential.quarantined:
                ev.append(f"differential {corr.differential.checked} inputs identical")
            if ev:
                out.append("Correct:   " + "; ".join(ev))
    else:
        out.append("Status:    NO VERIFIED IMPROVEMENT")
        if result.baseline:
            out.append(f"Baseline:  {fmt_time(result.baseline.median)} median (n={result.baseline.n})")

    if result.rejected:
        out.append(f"Rejected ({len(result.rejected)}):")
        for rc in result.rejected[:8]:
            out.append(f"  - [{rc.kind.value}] {rc.title}: {rc.reason}")

    for note in result.notes:
        out.append(f"Note:      {note}")

    if result.diff:
        out.append("")
        out.append("--- proposed diff ---")
        out.append(result.diff.rstrip("\n"))

    return "\n".join(out)
