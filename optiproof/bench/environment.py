"""Benchmark environment description + noise hints.

The strongest anti-noise mechanism in the MVP is the orchestrator's from-scratch
re-measure of the winner (regression-to-the-mean insurance). True A/B-interleaved
rounds and CPU pinning are a Phase-3 upgrade (they need same-process control that
the cross-process driver design defers).
"""

from __future__ import annotations

import os
import platform

from ..models import Measurement


def describe_host() -> str:
    return (
        f"{platform.system()} {platform.machine()}, {os.cpu_count()} CPUs, "
        f"{platform.python_implementation()} {platform.python_version()}"
    )


def noise_note(m: Measurement | None) -> str:
    if m and m.mean > 0 and m.n:
        rse = (m.stdev / (m.n ** 0.5)) / m.mean
        if rse > 0.05:
            return f"high benchmark noise (RSE {rse * 100:.1f}%)"
    return ""
