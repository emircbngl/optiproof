"""Rust adapter contract tests.

The pure-parsing checks (locate + shape) always run. The end-to-end optimize is
heavy (Docker + rustc compiles, ~minutes) so it's opt-in: set ``OPTIPROOF_RUST_E2E=1``
and have ``docker`` + the ``rust:1-slim`` image present. This keeps the default
suite fast while still proving the polyglot seam when asked.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from optiproof.adapters.base import AdapterRegistry

_ORIGINAL = (
    "fn count_distinct(xs: &[i64]) -> i64 {\n"
    "    let mut c = 0i64;\n"
    "    for i in 0..xs.len() {\n"
    "        let mut seen = false;\n"
    "        for j in 0..i {\n"
    "            if xs[j] == xs[i] { seen = true; break; }\n"
    "        }\n"
    "        if !seen { c += 1; }\n"
    "    }\n"
    "    c\n"
    "}\n"
)
_FAST = (
    "fn count_distinct(xs: &[i64]) -> i64 {\n"
    "    let mut s: HashSet<i64> = HashSet::new();\n"
    "    for &x in xs { s.insert(x); }\n"
    "    s.len() as i64\n"
    "}\n"
)
_TRAP = "fn count_distinct(xs: &[i64]) -> i64 {\n    xs.len() as i64\n}\n"


def test_locate_and_shape(tmp_path):
    f = tmp_path / "cd.rs"
    f.write_text(_ORIGINAL)
    ad = AdapterRegistry.get("rust")
    assert ad.detect(f) == 1.0
    t = ad.locate_target(f, "count_distinct")
    assert t.language == "rust" and t.symbol == "count_distinct"
    assert t.param_types == {"xs": "&[i64]"}
    kinds, ret = ad._shape(t)
    assert ret == "int" and ad._supported(kinds, ret)


def _rust_e2e_ready() -> bool:
    if os.environ.get("OPTIPROOF_RUST_E2E") != "1":
        return False
    if not shutil.which("docker"):
        return False
    return (
        subprocess.run(
            ["docker", "image", "inspect", "rust:1-slim"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )


@pytest.mark.skipif(not _rust_e2e_ready(), reason="set OPTIPROOF_RUST_E2E=1 with docker + rust:1-slim")
def test_rust_optimize_end_to_end():
    from optiproof.llm.null_provider import NullProvider
    from optiproof.models import Candidate, OptimizeKind, OptimizeRequest, SandboxBackend
    from optiproof.orchestrator import optimize

    d = Path(tempfile.mkdtemp())
    f = d / "count_distinct.rs"
    f.write_text(_ORIGINAL)
    req = OptimizeRequest(
        path=f, selector=f"{f}::count_distinct",
        sandbox=SandboxBackend.DOCKER, toolchain_image="rust:1-slim",
        workload_size=3000, num_diff_inputs=40, min_runs=6, max_runs=12, max_rounds=1,
    )
    cands = [
        Candidate(id="trap", kind=OptimizeKind.REWRITE, title="xs.len()", new_source=_TRAP),
        Candidate(id="fast", kind=OptimizeKind.REWRITE, title="HashSet", new_source=_FAST,
                  module_prelude="use std::collections::HashSet;"),
    ]
    res = optimize(req, provider=NullProvider(candidates=cands))
    assert res.improved and res.best and res.best.id == "fast"
    assert res.language == "rust" and res.speedup and res.speedup >= 1.10
    rejected = {r.id: r.reason for r in res.rejected}
    assert "trap" in rejected and "behavior changed" in rejected["trap"].lower()
