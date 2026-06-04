"""Contract tests every language adapter must pass (here: the Python adapter).

Adding a new language = making an analogous suite green.
"""

from __future__ import annotations

import statistics
from pathlib import Path

import pytest

from optiproof.adapters.base import AdapterRegistry
from optiproof.sandbox.base import Sandbox
from optiproof.sandbox.workspace import Workspace
from optiproof.models import SandboxBackend


@pytest.fixture
def sandbox():
    return Sandbox.create(SandboxBackend.LOCAL)


def _write(tmp_path: Path, body: str) -> Path:
    f = tmp_path / "mod.py"
    f.write_text(body)
    return f


def test_detect_and_locate(tmp_path):
    f = _write(tmp_path, "def has_dup(xs):\n    return len(xs) != len(set(xs))\n")
    ad = AdapterRegistry.detect(f)
    assert ad.name == "python"
    t = ad.locate_target(f, "has_dup")
    assert t.symbol == "has_dup" and t.start_line == 1 and t.end_line == 2


def test_build_good_and_bad(tmp_path, sandbox):
    f = _write(tmp_path, "def f(x):\n    return x\n")
    ad = AdapterRegistry.detect(f)
    ws = Workspace.fork_from_file(f)
    try:
        assert ad.build(ws, sandbox).ok
        ws.write_target("def f(:\n    pass\n")  # syntax error
        assert not ad.build(ws, sandbox).ok
    finally:
        ws.cleanup()


def test_observe_values_and_exceptions(tmp_path, sandbox):
    f = _write(tmp_path, "def inv(x):\n    return 10 // x\n")
    ad = AdapterRegistry.detect(f)
    t = ad.locate_target(f, "inv")
    ws = Workspace.fork_from_file(f)
    try:
        res = ad.observe(ws, t, [(2,), (0,)], sandbox)
        assert res.ok
        assert res.observations[0].ok and res.observations[0].value == 5
        assert not res.observations[1].ok and res.observations[1].exception == "ZeroDivisionError"
    finally:
        ws.cleanup()


def test_benchmark_scales_with_work(tmp_path, sandbox):
    f = _write(
        tmp_path,
        "import time\n"
        "def slow():\n    time.sleep(0.003)\n"
        "def fast():\n    time.sleep(0.001)\n",
    )
    ad = AdapterRegistry.detect(f)
    ws = Workspace.fork_from_file(f)
    try:
        slow_t = ad.locate_target(f, "slow")
        fast_t = ad.locate_target(f, "fast")
        slow_b = ad.benchmark(ws, slow_t, (), sandbox, warmup=1, min_rounds=5, max_rounds=10)
        fast_b = ad.benchmark(ws, fast_t, (), sandbox, warmup=1, min_rounds=5, max_rounds=10)
        assert slow_b.ok and fast_b.ok
        assert statistics.median(slow_b.samples) > statistics.median(fast_b.samples) * 1.8
    finally:
        ws.cleanup()
