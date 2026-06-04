"""The central statistics gate — the single place 'is it really faster?' is decided.

Acceptance is deliberately strict, because timing noise is the #1 way a measured
loop lies to itself:

    accept  iff  (Mann–Whitney U rejects equal-speed at alpha)
            AND  (the *entire* 95% bootstrap CI of the median speedup ratio lies
                  at or above the threshold, e.g. 1.10x)

Gating on the CI lower bound — not the point estimate — kills "statistically
significant but trivially small / possibly-noise" wins.
"""

from __future__ import annotations

import statistics
from typing import Sequence

import numpy as np
from scipy import stats as sps

from ..models import Measurement, SpeedupVerdict


def summarize(samples: Sequence[float], inner_loops: int = 1) -> Measurement:
    s = list(samples)
    n = len(s)
    return Measurement(
        samples=s,
        median=statistics.median(s) if s else 0.0,
        mean=statistics.fmean(s) if s else 0.0,
        stdev=statistics.pstdev(s) if n >= 2 else 0.0,
        n=n,
        inner_loops=inner_loops,
    )


def _bootstrap_ratio_ci(
    baseline: Sequence[float], candidate: Sequence[float], seed: int = 1234, B: int = 2000, conf: float = 0.95
) -> tuple[float, float]:
    b = np.asarray(baseline, dtype=float)
    c = np.asarray(candidate, dtype=float)
    if b.size == 0 or c.size == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    b_med = np.median(b[rng.integers(0, b.size, size=(B, b.size))], axis=1)
    c_med = np.median(c[rng.integers(0, c.size, size=(B, c.size))], axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.where(c_med > 0, b_med / c_med, np.inf)
    lo = float(np.nanpercentile(ratios, (1 - conf) / 2 * 100))
    hi = float(np.nanpercentile(ratios, (1 + conf) / 2 * 100))
    return lo, hi


def speedup_verdict(
    baseline: Sequence[float],
    candidate: Sequence[float],
    threshold: float = 1.10,
    alpha: float = 0.05,
    seed: int = 1234,
) -> SpeedupVerdict:
    b_med = statistics.median(baseline) if baseline else 0.0
    c_med = statistics.median(candidate) if candidate else float("inf")
    ratio = (b_med / c_med) if c_med > 0 else float("inf")

    # Significance: are baseline times stochastically GREATER (i.e. candidate faster)?
    try:
        if len(set(baseline)) <= 1 and len(set(candidate)) <= 1:
            p = 0.0 if b_med > c_med else 1.0
        else:
            p = float(sps.mannwhitneyu(baseline, candidate, alternative="greater").pvalue)
    except Exception:
        p = 1.0

    lo, hi = _bootstrap_ratio_ci(baseline, candidate, seed=seed)
    is_faster = (p < alpha) and (lo >= threshold)

    if is_faster:
        reason = f"{ratio:.2f}x faster (95% CI {lo:.2f}-{hi:.2f}x, p={p:.1e})"
    elif ratio < 1.0:
        reason = f"slower ({ratio:.2f}x)"
    elif lo < threshold:
        reason = f"{ratio:.2f}x but CI {lo:.2f}-{hi:.2f}x not entirely above {threshold:.2f}x (n.s.)"
    else:
        reason = f"{ratio:.2f}x, not statistically significant (p={p:.1e})"

    return SpeedupVerdict(is_faster=is_faster, ratio=ratio, ci_low=lo, ci_high=hi, p_value=p, reason=reason)
