"""End-to-end DoD: the agent loop on the corpus, driven by a deterministic provider.

No API calls, no flaky timing assertions beyond "the fast one really is faster".
For every corpus case the loop must:
  * ACCEPT the correct-fast candidate (true positive),
  * REJECT the trap on correctness (differential testing catches the behavior change),
  * REJECT the no-win on the speed gate (no fabricated win).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from benchmarks_corpus.cases import CASES
from optiproof.llm.null_provider import NullProvider
from optiproof.models import Candidate, OptimizeKind, OptimizeRequest, SandboxBackend
from optiproof.orchestrator import optimize


def _run(case, specs):
    d = Path(tempfile.mkdtemp())
    f = d / "mod.py"
    f.write_text(case.original)
    cands = [
        Candidate(id=name, kind=OptimizeKind.REWRITE, title=name, new_source=src)
        for name, src in specs
    ]
    req = OptimizeRequest(
        path=f,
        selector=f"{f}::{case.symbol}",
        sandbox=SandboxBackend.LOCAL,
        num_diff_inputs=80,
        min_runs=6,
        max_runs=14,
        candidates_per_round=5,
        max_rounds=1,
    )
    return optimize(req, provider=NullProvider(candidates=cands))


@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_accepts_fast_rejects_trap_and_nowin(case):
    res = _run(case, [("trap", case.trap), ("nowin", case.nowin), ("fast", case.fast)])
    diagnostics = f"notes={res.notes} rejected={[(r.id, r.reason) for r in res.rejected]}"

    assert res.improved and res.best and res.best.id == "fast", diagnostics
    assert res.speedup and res.speedup >= 1.10, diagnostics

    rejected = {r.id: r.reason for r in res.rejected}
    assert "trap" in rejected and "behavior changed" in rejected["trap"].lower(), diagnostics
    assert "nowin" in rejected, diagnostics  # rejected on the speed gate


@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_trap_alone_yields_no_win(case):
    res = _run(case, [("trap", case.trap)])
    assert not res.improved
    assert any(r.id == "trap" for r in res.rejected)


@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_nowin_alone_yields_no_win(case):
    res = _run(case, [("nowin", case.nowin)])
    assert not res.improved
