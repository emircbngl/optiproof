"""Regression tests for the code-review fixes."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from optiproof.adapters.base import AdapterRegistry, BehaviorObservation, ObserveResult
from optiproof.models import (
    Candidate,
    OptimizeKind,
    OptimizeRequest,
    SandboxBackend,
    Target,
)


# ---- #1 compare: bool is not int -------------------------------------------------
def test_compare_bool_not_int():
    from optiproof.verify.compare import compare_observations, values_equal

    assert not values_equal(True, 1)
    assert not values_equal([True], [1])
    assert values_equal(True, True)

    def obs(v):
        return BehaviorObservation(ok=True, value=v, value_repr=repr(v))

    assert not compare_observations(obs(True), obs(1))[0]


# ---- #2 differential: count mismatch fails closed --------------------------------
def _obs_list(vals):
    return [BehaviorObservation(ok=True, value=v, value_repr=repr(v)) for v in vals]


class _ObsAdapter:
    def __init__(self, observations):
        self._observations = observations

    def observe(self, *args, **kwargs):
        return ObserveResult(ok=True, observations=self._observations)


def test_differential_rejects_short_candidate():
    from optiproof.verify.differential import BaselineBehavior, compare_candidate

    baseline = BaselineBehavior(observations=_obs_list([1, 2, 3]), deterministic=True)
    cand = _ObsAdapter(_obs_list([1, 2]))  # produced fewer outputs
    res = compare_candidate(baseline, None, None, [(0,), (1,), (2,)], cand, None, 1e-9, 1e-12)
    assert not res.equivalent


def test_differential_accepts_full_match():
    from optiproof.verify.differential import BaselineBehavior, compare_candidate

    baseline = BaselineBehavior(observations=_obs_list([1, 2, 3]), deterministic=True)
    cand = _ObsAdapter(_obs_list([1, 2, 3]))
    res = compare_candidate(baseline, None, None, [(0,), (1,), (2,)], cand, None, 1e-9, 1e-12)
    assert res.equivalent and res.checked == 3


# ---- #5b record_baseline: count mismatch -> non-deterministic --------------------
class _SeqAdapter:
    def __init__(self, results):
        self._results = results
        self._i = 0

    def observe(self, *args, **kwargs):
        r = self._results[self._i]
        self._i += 1
        return r


def test_record_baseline_count_mismatch_quarantines():
    from optiproof.verify.differential import record_baseline

    a = ObserveResult(ok=True, observations=_obs_list([1, 2, 3]))
    b = ObserveResult(ok=True, observations=_obs_list([1, 2]))  # different count
    baseline, err = record_baseline(None, None, [(0,)], _SeqAdapter([a, b]), None)
    assert err == "" and baseline is not None and not baseline.deterministic


# ---- #8 run_tests: fail closed / detect missing pytest ---------------------------
class _FakeWorkspace:
    root = Path("/tmp")
    target_rel = Path("mod.py")


class _FakeSandbox:
    python_executable = None

    def __init__(self, rc, out):
        self._rc = rc
        self._out = out

    def run(self, cmd, cwd, timeout, env=None, stdin=None):
        from optiproof.sandbox.base import ExecResult

        return ExecResult(returncode=self._rc, stdout=self._out.encode(), stderr=b"")


def test_run_tests_failclosed_on_internal_error():
    from optiproof.adapters.python_adapter import PythonAdapter

    tr = PythonAdapter().run_tests(_FakeWorkspace(), _FakeSandbox(3, "INTERNALERROR> boom"))
    assert tr.has_tests and not tr.ok  # unparseable non-zero exit -> treated as failing suite


def test_run_tests_no_pytest_runner():
    from optiproof.adapters.python_adapter import PythonAdapter

    tr = PythonAdapter().run_tests(_FakeWorkspace(), _FakeSandbox(1, "No module named pytest"))
    assert not tr.has_tests and tr.ok  # missing runner -> 'no tests', not a failure


# ---- #9 locate_target picks the LAST top-level def -------------------------------
def test_locate_picks_last_def(tmp_path):
    f = tmp_path / "mod.py"
    f.write_text("def f(x):\n    return x\n\ndef f(x, y=0):\n    return x + y\n")
    t = AdapterRegistry.detect(f).locate_target(f, "f")
    assert "y" in t.param_types and t.start_line == 4  # the second def


# ---- #14 prompt names the function ----------------------------------------------
def test_feedback_block_names_symbol():
    from optiproof.llm.prompts import feedback_block

    assert "`solve`" in feedback_block("solve", [], 3)


# ---- #13 anthropic _parse tolerates null fields ---------------------------------
def test_anthropic_parse_tolerates_nulls():
    from optiproof.llm.anthropic_provider import AnthropicProvider

    prov = object.__new__(AnthropicProvider)  # bypass __init__ (no client / network)
    prov._round = 1

    class _Block:
        type = "tool_use"
        name = "propose_optimizations"
        input = {"candidates": [{"new_source": "def f():\n    return 1", "title": None,
                                 "rationale": None, "kind": None}]}

    class _Resp:
        content = [_Block()]

    t = Target(file="x.py", symbol="f", language="python", start_line=1, end_line=1, source="")
    cands = prov._parse(_Resp(), t)
    assert len(cands) == 1 and cands[0].title == "" and cands[0].rationale == ""
    assert cands[0].kind == OptimizeKind.REWRITE


# ---- #11 Rust adapter honest supported-shape set --------------------------------
def _rust_target(params: dict, ret: str) -> Target:
    sig = "fn f(" + ", ".join(f"{k}: {v}" for k, v in params.items()) + f") -> {ret}"
    return Target(file="x.rs", symbol="f", language="rust", start_line=1, end_line=1,
                  source="", signature=sig, param_types=params)


def test_rust_supported_shapes():
    ad = AdapterRegistry.get("rust")

    def ok(params, ret):
        return ad._supported(*ad._shape(_rust_target(params, ret)))

    assert ok({"xs": "&[i64]"}, "i64")
    assert ok({"xs": "&Vec<i64>"}, "bool")
    assert ok({"xs": "&[i64]", "k": "i64"}, "i64")
    assert not ok({"xs": "&[u64]"}, "i64")       # unsigned element
    assert not ok({"xs": "&mut [i64]"}, "i64")   # mutable
    assert not ok({"xs": "Vec<i64>"}, "i64")     # by-value (harness passes &__arr)
    assert not ok({"xs": "&[i64]"}, "String")    # unsupported return


# ---- #10 Rust locate is string/comment aware ------------------------------------
def test_rust_locate_brace_in_string(tmp_path):
    ad = AdapterRegistry.get("rust")
    f = tmp_path / "m.rs"
    f.write_text(
        'fn g(xs: &[i64]) -> i64 {\n'
        '    let s = "}";  // a brace } in a string and comment {\n'
        '    let _ = s;\n'
        '    xs.len() as i64\n'
        '}\n'
    )
    t = ad.locate_target(f, "g")
    assert t.end_line == 5 and "xs.len() as i64" in t.source


# ---- #5 / #4 integration on the local sandbox -----------------------------------
@pytest.fixture
def local_sandbox():
    from optiproof.sandbox.base import Sandbox

    return Sandbox.create(SandboxBackend.LOCAL)


def test_benchmark_handles_mutating_target(local_sandbox):
    from optiproof.sandbox.workspace import Workspace

    d = Path(tempfile.mkdtemp())
    f = d / "mod.py"
    f.write_text("def srt(xs):\n    xs.sort()\n    return xs[0] if xs else 0\n")
    ad = AdapterRegistry.detect(f)
    t = ad.locate_target(f, "srt")
    ws = Workspace.fork_from_file(f)
    try:
        b = ad.benchmark(ws, t, ([5, 3, 1, 4, 2],), local_sandbox, warmup=1, min_rounds=4, max_rounds=8)
        assert b.ok and len(b.samples) >= 4  # bounded-copy path runs for a mutating target
    finally:
        ws.cleanup()


def test_anti_luck_discards_when_reconfirm_fails(monkeypatch):
    import optiproof.orchestrator as orch
    from optiproof.llm.null_provider import NullProvider

    real_measure = orch.runner.measure

    def fake_measure(*args, **kwargs):
        # The confirmation re-measure doubles min_rounds (8 -> 16); fail only that call.
        if kwargs.get("min_rounds", 0) >= 16:
            return None, "forced failure"
        return real_measure(*args, **kwargs)

    monkeypatch.setattr(orch.runner, "measure", fake_measure)

    d = Path(tempfile.mkdtemp())
    f = d / "mod.py"
    f.write_text(
        "def count_unique(xs):\n"
        "    seen = []\n"
        "    for x in xs:\n"
        "        if x not in seen:\n"
        "            seen.append(x)\n"
        "    return len(seen)\n"
    )
    req = OptimizeRequest(
        path=f, selector=f"{f}::count_unique", sandbox=SandboxBackend.LOCAL,
        num_diff_inputs=40, min_runs=8, max_runs=12, max_rounds=1,
    )
    fast = Candidate(id="fast", kind=OptimizeKind.REWRITE, title="set()",
                     new_source="def count_unique(xs):\n    return len(set(xs))")
    res = orch.optimize(req, provider=NullProvider(candidates=[fast]))
    assert not res.improved
    assert any("re-measure" in n for n in res.notes)


# ---- workspace gather is safe (no blind copytree of an arbitrary parent) ----------
def test_workspace_gather_is_safe_and_targeted(tmp_path):
    import os
    from optiproof.sandbox.workspace import Workspace

    (tmp_path / "mod.py").write_text("def f(x):\n    return x\n")
    (tmp_path / "helper.py").write_text("X = 1\n")        # same-language sibling -> copied
    (tmp_path / "big.bin").write_bytes(b"0" * 4096)       # non-source -> NOT copied
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "deep.py").write_text("Y = 1\n")  # subdir -> NOT recursed into
    ws = Workspace.fork_from_file(tmp_path / "mod.py")
    try:
        names = set(os.listdir(ws.root))
        assert {"mod.py", "helper.py"} <= names
        assert "big.bin" not in names and "sub" not in names
    finally:
        ws.cleanup()


# ---- #2/#3 soundness: ambiguous numeric lists are differential-tested with int AND float ----
def _run_opt(original_src, symbol, specs):
    from optiproof.llm.null_provider import NullProvider
    from optiproof.orchestrator import optimize

    d = Path(tempfile.mkdtemp())
    f = d / "m.py"
    f.write_text(original_src)
    cands = [Candidate(id=i, kind=OptimizeKind.REWRITE, title=i, new_source=s) for i, s in specs]
    req = OptimizeRequest(
        path=f, selector=f"{f}::{symbol}", sandbox=SandboxBackend.LOCAL,
        num_diff_inputs=60, min_runs=6, max_runs=12, max_rounds=1,
    )
    return optimize(req, provider=NullProvider(candidates=cands))


def test_int_rounding_candidate_rejected_on_float_inputs():
    # #3a: `coeffs` is unannotated -> ambiguous numeric -> differential spans int AND float, so a
    # candidate that is correct on ints but truncates floats MUST be rejected (the real-world near-miss).
    original = "def ssum(coeffs):\n    return sum(c * c for c in coeffs)\n"
    trap = "def ssum(coeffs):\n    return sum(int(c) * int(c) for c in coeffs)"
    res = _run_opt(original, "ssum", [("introunding", trap)])
    assert not res.improved
    reasons = {r.id: r.reason for r in res.rejected}
    assert "introunding" in reasons and "behavior changed" in reasons["introunding"].lower()
    assert "int+float" in res.inputs_tested  # #1: tested domain is reported


def test_valid_optimization_accepted_with_mixed_inputs():
    # #3b: adding float inputs (#2) must NOT falsely reject a genuinely-equivalent win.
    original = (
        "def dedup_count(items):\n"
        "    seen = []\n"
        "    for x in items:\n"
        "        if x not in seen:\n"
        "            seen.append(x)\n"
        "    return len(seen)\n"
    )
    fast = "def dedup_count(items):\n    return len(set(items))"
    res = _run_opt(original, "dedup_count", [("fast", fast)])
    assert res.improved and res.best and res.best.id == "fast"


def test_probe_resolves_scalar_and_stays_sound():
    # #4: `rho` is an un-inferrable scalar; probing the original discovers it's a scalar (so the
    # function is MEASURABLE, not UNBENCHMARKABLE) and still tests it with int+float, so an
    # int-truncating candidate is rejected.
    original = "def quant(rho):\n    return rho * rho + rho\n"
    trap = "def quant(rho):\n    return int(rho) * int(rho) + int(rho)"  # correct on ints, wrong on floats
    res = _run_opt(original, "quant", [("trap", trap)])
    assert not res.unbenchmarkable
    assert "int+float" in res.inputs_tested  # probe -> num -> int+float reported
    reasons = {r.id: r.reason for r in res.rejected}
    assert "trap" in reasons and "behavior changed" in reasons["trap"].lower()


def test_uninferrable_param_is_unbenchmarkable():
    # `cfg` must be a dict with key "k" — no probe candidate (scalar/list/str/bool) satisfies it,
    # so the original raises on all inputs -> clear UNBENCHMARKABLE, not a confusing traceback.
    res = _run_opt(
        'def lookup(cfg):\n    return cfg["k"] * 2\n',
        "lookup",
        [("c", 'def lookup(cfg):\n    return cfg["k"] * 2\n')],
    )
    assert res.unbenchmarkable
    assert any("UNBENCHMARKABLE" in n for n in res.notes)
