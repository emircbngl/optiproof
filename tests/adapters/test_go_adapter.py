"""Go adapter contract tests. Pure-parsing checks always run; the end-to-end
optimize runs whenever `go` is on PATH (local sandbox, no Docker needed)."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest

from optiproof.adapters.base import AdapterRegistry
from optiproof.models import Target

_ORIG = (
    "package opt\n\n"
    "func CountDistinct(xs []int64) int64 {\n"
    "\tvar c int64 = 0\n"
    "\tfor i := 0; i < len(xs); i++ {\n"
    "\t\tseen := false\n"
    "\t\tfor j := 0; j < i; j++ {\n"
    "\t\t\tif xs[j] == xs[i] {\n"
    "\t\t\t\tseen = true\n"
    "\t\t\t\tbreak\n"
    "\t\t\t}\n"
    "\t\t}\n"
    "\t\tif !seen {\n"
    "\t\t\tc++\n"
    "\t\t}\n"
    "\t}\n"
    "\treturn c\n}\n"
)
_FAST = (
    "func CountDistinct(xs []int64) int64 {\n"
    "\tm := make(map[int64]struct{})\n"
    "\tfor _, x := range xs {\n"
    "\t\tm[x] = struct{}{}\n"
    "\t}\n"
    "\treturn int64(len(m))\n}\n"
)
_TRAP = "func CountDistinct(xs []int64) int64 {\n\treturn int64(len(xs))\n}\n"


def test_locate_and_shape(tmp_path):
    f = tmp_path / "cd.go"
    f.write_text(_ORIG)
    ad = AdapterRegistry.get("go")
    assert ad.detect(f) == 1.0
    t = ad.locate_target(f, "CountDistinct")
    assert t.language == "go" and t.param_types == {"xs": "[]int64"}
    kinds, ret = ad._shape(t)
    assert ret == "int" and ad._supported(kinds, ret)


def test_go_shape_rejects_unsupported():
    ad = AdapterRegistry.get("go")

    def shp(params, ret):
        sig = "func f(" + ", ".join(f"{k} {v}" for k, v in params.items()) + f") {ret}"
        t = Target(file="x.go", symbol="f", language="go", start_line=1, end_line=1,
                   source="", signature=sig, param_types=params)
        return ad._supported(*ad._shape(t))

    assert shp({"xs": "[]int64"}, "int64")
    assert shp({"xs": "[]int64", "k": "int64"}, "bool")
    assert not shp({"xs": "[]string"}, "int64")   # unsupported element type
    assert not shp({"xs": "[]int64"}, "string")   # unsupported return


def test_go_locate_brace_in_string(tmp_path):
    ad = AdapterRegistry.get("go")
    f = tmp_path / "m.go"
    f.write_text(
        'package opt\n\n'
        'func g(xs []int64) int64 {\n'
        '\ts := "}" // a brace } in a string and comment {\n'
        '\t_ = s\n'
        '\treturn int64(len(xs))\n'
        '}\n'
    )
    t = ad.locate_target(f, "g")
    assert t.end_line == 7 and "return int64(len(xs))" in t.source


@pytest.mark.skipif(shutil.which("go") is None, reason="needs `go` on PATH")
def test_go_optimize_end_to_end():
    from optiproof.llm.null_provider import NullProvider
    from optiproof.models import Candidate, OptimizeKind, OptimizeRequest, SandboxBackend
    from optiproof.orchestrator import optimize

    d = Path(tempfile.mkdtemp())
    f = d / "cd.go"
    f.write_text(_ORIG)
    req = OptimizeRequest(
        path=f, selector=f"{f}::CountDistinct", sandbox=SandboxBackend.LOCAL,
        workload_size=2000, num_diff_inputs=40, min_runs=6, max_runs=12, max_rounds=1,
    )
    cands = [
        Candidate(id="trap", kind=OptimizeKind.REWRITE, title="len", new_source=_TRAP),
        Candidate(id="fast", kind=OptimizeKind.REWRITE, title="map", new_source=_FAST),
    ]
    res = optimize(req, provider=NullProvider(candidates=cands))
    assert res.improved and res.best and res.best.id == "fast" and res.language == "go"
    assert res.speedup and res.speedup >= 1.10
    rejected = {r.id: r.reason for r in res.rejected}
    assert "trap" in rejected and "behavior changed" in rejected["trap"].lower()
