"""Statistics gate self-check — independent of any LLM or real timing noise.

Feeds synthetic distributions with known ground truth so the accept/reject logic
is tested in isolation. The false-positive check guards the most dangerous
failure: fabricating a speedup that isn't there.
"""

from __future__ import annotations

import numpy as np

from optiproof.bench.stats import speedup_verdict


def test_identical_distributions_not_significant():
    rng = np.random.default_rng(0)
    base = list(rng.normal(100, 5, 40))
    cand = list(rng.normal(100, 5, 40))
    assert not speedup_verdict(base, cand, threshold=1.10).is_faster


def test_clear_2x_speedup_accepted():
    rng = np.random.default_rng(1)
    base = list(rng.normal(100, 5, 40))
    cand = list(rng.normal(50, 3, 40))
    v = speedup_verdict(base, cand, threshold=1.10)
    assert v.is_faster and 1.8 < v.ratio < 2.2 and v.ci_low >= 1.10


def test_small_within_threshold_rejected():
    rng = np.random.default_rng(2)
    base = list(rng.normal(100, 5, 60))
    cand = list(rng.normal(95, 5, 60))  # ~5% faster, below the 10% bar
    assert not speedup_verdict(base, cand, threshold=1.10).is_faster


def test_slower_candidate_rejected():
    rng = np.random.default_rng(3)
    base = list(rng.normal(100, 5, 40))
    cand = list(rng.normal(130, 5, 40))
    assert not speedup_verdict(base, cand, threshold=1.10).is_faster


def test_false_positive_rate_is_low():
    # Over many null pairs (no real difference), the strict CI gate should almost
    # never declare a winner.
    false_positives = 0
    trials = 60
    for s in range(trials):
        rng = np.random.default_rng(1000 + s)
        base = list(rng.normal(100, 6, 30))
        cand = list(rng.normal(100, 6, 30))
        if speedup_verdict(base, cand, threshold=1.10, seed=s).is_faster:
            false_positives += 1
    assert false_positives <= trials * 0.05
