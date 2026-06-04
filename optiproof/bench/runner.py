"""Benchmark runner — asks the adapter to time a workload, summarizes the samples.

The adapter does warmup + adaptive sampling inside its driver and returns RAW
per-call times; the runner just summarizes. Baseline and candidate are always
measured on the *same* workload (the orchestrator generates it once).
"""

from __future__ import annotations

from typing import Optional

from ..models import Measurement
from .stats import summarize


def measure(
    ws,
    target,
    workload,
    adapter,
    sandbox,
    *,
    warmup: int = 3,
    min_rounds: int = 12,
    max_rounds: int = 60,
    target_rse: float = 0.02,
    timeout: float = 120.0,
) -> tuple[Optional[Measurement], str]:
    rb = adapter.benchmark(
        ws, target, workload, sandbox,
        warmup=warmup, min_rounds=min_rounds, max_rounds=max_rounds,
        target_rse=target_rse, timeout=timeout,
    )
    if not rb.ok:
        return None, rb.error
    return summarize(rb.samples, rb.inner_loops), ""
