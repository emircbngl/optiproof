"""Go language adapter — the third language, proving the seam generalizes.

Same engine, a different compiled runtime (AOT + GC). std-only, compiled with
``go build`` in GOPATH mode (``GO111MODULE=off``) so a single self-contained file
builds with no go.mod and no network — it runs on the local sandbox (host ``go``)
or in a ``golang`` container.

MVP-supported signature subset (documented):
    func NAME(xs []int64 [, k int64]) (int64 | bool | []int64)
covering the optimization-corpus spirit (count / sum / dedup over int slices).
The target function must be self-contained (builtins only — no extra imports), so
it can be lifted into a generated ``package main`` harness. Results go to a FILE
(not stdout); panics are recovered WITH their message.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from ..models import BuildResult, Target, TestResult
from ..sandbox.base import Sandbox
from ..sandbox.workspace import Workspace
from .base import BehaviorObservation, LanguageAdapter, ObserveResult, RawBenchmark

# Self-contained, offline, writable cache locations that work on host + container.
GO_ENV = {
    "GO111MODULE": "off",
    "GOFLAGS": "-mod=mod",
    "GOCACHE": "/tmp/optiproof-gocache",
    "GOPATH": "/tmp/optiproof-gopath",
    "HOME": "/tmp",
    "CGO_ENABLED": "0",
}
_INT_GO = {"int64", "int", "int32", "int16", "int8", "uint64", "uint32", "uint", "rune"}
_ARRAY_PARAMS = {"[]int64"}

_PARSE_FN = r"""
func __optiproofParseArr(s string) []int64 {
	s = strings.TrimSpace(s)
	if s == "" {
		return []int64{}
	}
	parts := strings.Split(s, ",")
	out := make([]int64, 0, len(parts))
	for _, p := range parts {
		v, _ := strconv.ParseInt(strings.TrimSpace(p), 10, 64)
		out = append(out, v)
	}
	return out
}

func __optiproofFmtList(v []int64) string {
	parts := make([]string, len(v))
	for i, x := range v {
		parts[i] = strconv.FormatInt(x, 10)
	}
	return strings.Join(parts, ",")
}
"""

_OBS_TEMPLATE = r"""package main

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"
)

__FUNC_SRC__
""" + _PARSE_FN + r"""
func __optiproofCall(arr []int64__SCALAR_SIG__) (res __RET_TYPE__, panicked bool, msg string) {
	defer func() {
		if r := recover(); r != nil {
			panicked = true
			msg = fmt.Sprint(r)
		}
	}()
	res = __CALL__
	return
}

func main() {
	data, _ := os.ReadFile(os.Args[1])
	var sb strings.Builder
	sc := bufio.NewScanner(strings.NewReader(string(data)))
	sc.Buffer(make([]byte, 1024*1024), 256*1024*1024)
	for sc.Scan() {
		line := sc.Text()
		semi := strings.SplitN(line, ";", 2)
		arr := __optiproofParseArr(semi[0])
		__SCALAR_PARSE__
		res, panicked, msg := __optiproofCall(arr__SCALAR_ARG__)
		if panicked {
			sb.WriteString("PANIC " + strings.ReplaceAll(msg, "\n", " ") + "\n")
		} else {
			sb.WriteString("OK " + __FORMAT__ + "\n")
		}
	}
	os.WriteFile(os.Args[2], []byte(sb.String()), 0644)
}
"""

_BENCH_TEMPLATE = r"""package main

import (
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"
	"time"
)

__FUNC_SRC__
""" + _PARSE_FN + r"""
var __optiproofSink int64

func main() {
	data, _ := os.ReadFile(os.Args[1])
	warmup, _ := strconv.Atoi(os.Args[2])
	minRounds, _ := strconv.Atoi(os.Args[3])
	maxRounds, _ := strconv.Atoi(os.Args[4])
	targetRse, _ := strconv.ParseFloat(os.Args[5], 64)
	outPath := os.Args[6]
	first := strings.SplitN(strings.TrimRight(string(data), "\n"), "\n", 2)[0]
	semi := strings.SplitN(first, ";", 2)
	arr := __optiproofParseArr(semi[0])
	__SCALAR_PARSE__
	runChunk := func(l int) float64 {
		start := time.Now()
		for i := 0; i < l; i++ {
			__SINK_OP__
		}
		return time.Since(start).Seconds()
	}
	l := 1
	for {
		dt := runChunk(l)
		if dt >= 0.02 || l >= 1000000 {
			break
		}
		if dt <= 0 {
			l *= 10
		} else {
			nl := int(float64(l) * (0.02 / dt) * 1.3)
			if nl < l+1 {
				nl = l + 1
			}
			l = nl
		}
	}
	for w := 0; w < warmup; w++ {
		runChunk(l)
	}
	samples := []float64{}
	for r := 0; r < maxRounds; r++ {
		dt := runChunk(l)
		samples = append(samples, dt/float64(l))
		if len(samples) >= minRounds && len(samples) >= 3 {
			n := float64(len(samples))
			mean := 0.0
			for _, s := range samples {
				mean += s
			}
			mean /= n
			vv := 0.0
			for _, s := range samples {
				vv += (s - mean) * (s - mean)
			}
			vv /= n
			rse := 0.0
			if mean > 0 {
				rse = (math.Sqrt(vv) / math.Sqrt(n)) / mean
			}
			if rse <= targetRse {
				break
			}
		}
	}
	parts := make([]string, len(samples))
	for i, s := range samples {
		parts[i] = strconv.FormatFloat(s, 'e', 12, 64)
	}
	j := fmt.Sprintf("{\"samples\":[%s],\"inner_loops\":%d}", strings.Join(parts, ","), l)
	os.WriteFile(outPath, []byte(j), 0644)
}
"""


def _blank_noncode(src: str) -> str:
    """Blank Go string/raw-string/char literals and comments (length-preserving)."""
    out = list(src)
    i, n = 0, len(src)
    while i < n:
        c = src[i]
        if c == '"':
            out[i] = " "
            i += 1
            while i < n:
                if src[i] == "\\":
                    out[i] = " "
                    if i + 1 < n:
                        out[i + 1] = " "
                    i += 2
                    continue
                if src[i] == '"':
                    out[i] = " "
                    i += 1
                    break
                if src[i] != "\n":
                    out[i] = " "
                i += 1
        elif c == "`":  # raw string (can span lines, no escapes)
            out[i] = " "
            i += 1
            while i < n and src[i] != "`":
                if src[i] != "\n":
                    out[i] = " "
                i += 1
            if i < n:
                out[i] = " "
                i += 1
        elif c == "/" and i + 1 < n and src[i + 1] == "/":
            while i < n and src[i] != "\n":
                out[i] = " "
                i += 1
        elif c == "/" and i + 1 < n and src[i + 1] == "*":  # Go block comments do not nest
            out[i] = out[i + 1] = " "
            i += 2
            while i < n and not (src[i] == "*" and i + 1 < n and src[i + 1] == "/"):
                if src[i] != "\n":
                    out[i] = " "
                i += 1
            if i + 1 < n:
                out[i] = out[i + 1] = " "
                i += 2
        elif c == "'":
            j = i + 1
            if j < n and src[j] == "\\":
                j += 2
            else:
                j += 1
            if j < n and src[j] == "'":
                for k in range(i, j + 1):
                    out[k] = " "
                i = j + 1
            else:
                i += 1
        else:
            i += 1
    return "".join(out)


class GoAdapter(LanguageAdapter):
    name = "go"

    def detect(self, path: Path) -> float:
        return 1.0 if str(path).endswith(".go") else 0.0

    def _resolve(self, path: Path, selector: str) -> tuple[Path, str]:
        if "::" in selector:
            file_part, symbol = selector.rsplit("::", 1)
            file = Path(file_part)
            if not file.exists():
                file = Path(path)
        else:
            symbol, file = selector, Path(path)
        return file.resolve(), symbol

    def locate_target(self, path: Path, selector: str) -> Target:
        file, symbol = self._resolve(path, selector)
        src = file.read_text()
        scan = _blank_noncode(src)

        m = re.search(r"\bfunc\s+" + re.escape(symbol) + r"\s*\(", scan)
        if not m:
            raise ValueError(f"function {symbol!r} not found in {file}")

        paren = scan.index("(", m.start())
        depth, i = 0, paren
        while i < len(scan):
            if scan[i] == "(":
                depth += 1
            elif scan[i] == ")":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        params_str = src[paren + 1:i]

        brace = scan.index("{", i)
        ret_str = src[i + 1:brace].strip()
        depth, j = 0, brace
        while j < len(scan):
            if scan[j] == "{":
                depth += 1
            elif scan[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1

        start_line = src.count("\n", 0, m.start()) + 1
        end_line = src.count("\n", 0, j) + 1
        source = "\n".join(src.splitlines()[start_line - 1:end_line])

        param_types: dict[str, str] = {}
        for p in params_str.split(","):
            p = p.strip()
            if not p:
                continue
            toks = p.split(None, 1)  # Go: "name type"
            if len(toks) == 2:
                param_types[toks[0].strip()] = toks[1].strip()

        signature = f"func {symbol}({params_str.strip()}) {ret_str}".strip()
        return Target(
            file=file, symbol=symbol, language="go",
            start_line=start_line, end_line=end_line, source=source,
            signature=signature, param_types=param_types,
        )

    # ---- shape ----
    def _ret_kind(self, ret_str: str) -> str:
        r = ret_str.strip()
        if r.startswith("(") and r.endswith(")"):
            inner = r[1:-1].strip()
            r = inner.split()[-1] if inner else ""
        r = r.replace(" ", "")
        if r in _INT_GO:
            return "int"
        if r == "bool":
            return "bool"
        if r.startswith("[]int"):
            return "list"
        return "unit" if r == "" else "unknown"

    def _shape(self, target: Target) -> tuple[list[tuple[str, str]], str]:
        kinds: list[tuple[str, str]] = []
        for ty in target.param_types.values():
            t = ty.replace(" ", "")
            if t in _ARRAY_PARAMS:
                kinds.append(("array", t))
            elif t == "int64":
                kinds.append(("scalar", t))
            else:
                kinds.append(("unknown", t))
        msig = re.search(r"\)\s*([^{]*)$", target.signature or "", re.DOTALL)
        ret = self._ret_kind(msig.group(1)) if msig else "unit"
        return kinds, ret

    def _supported(self, kinds, ret) -> bool:
        arrays = [k for k, _ in kinds if k == "array"]
        scalars = [k for k, _ in kinds if k == "scalar"]
        unknown = [k for k, _ in kinds if k == "unknown"]
        return (not unknown) and len(arrays) == 1 and len(scalars) <= 1 and ret in ("int", "bool", "list")

    # ---- codegen ----
    def _ret_type(self, ret: str) -> str:
        return {"int": "int64", "bool": "bool", "list": "[]int64"}[ret]

    def _call(self, symbol, kinds) -> str:
        args = []
        for kind, _ in kinds:
            if kind == "array":
                args.append("arr")
            elif kind == "scalar":
                args.append("k")
        return f"{symbol}({', '.join(args)})"

    def _has_scalar(self, kinds) -> bool:
        return any(k == "scalar" for k, _ in kinds)

    def _scalar_parse(self, kinds) -> str:
        if self._has_scalar(kinds):
            return ("var k int64\n\t\tif len(semi) > 1 {\n\t\t\t"
                    "k, _ = strconv.ParseInt(strings.TrimSpace(semi[1]), 10, 64)\n\t\t}")
        return ""

    def _format(self, ret: str) -> str:
        return {
            "int": "strconv.FormatInt(res, 10)",
            "bool": "strconv.FormatBool(res)",
            "list": "__optiproofFmtList(res)",
        }[ret]

    def _sink_op(self, symbol, kinds, ret) -> str:
        call = self._call(symbol, kinds)
        if ret == "int":
            return f"__optiproofSink += {call}"
        if ret == "bool":
            return f"if {call} {{\n\t\t\t\t__optiproofSink++\n\t\t\t}}"
        return f"__optiproofSink += int64(len({call}))"

    def _func_src(self, ws: Workspace, target: Target) -> str:
        # Re-locate on the (possibly candidate-patched) file to get the current function body.
        return self.locate_target(ws.target_path, target.symbol).source

    def _obs_harness(self, func_src, symbol, kinds, ret) -> str:
        return (
            _OBS_TEMPLATE
            .replace("__FUNC_SRC__", func_src)
            .replace("__SCALAR_SIG__", ", k int64" if self._has_scalar(kinds) else "")
            .replace("__SCALAR_ARG__", ", k" if self._has_scalar(kinds) else "")
            .replace("__SCALAR_PARSE__", self._scalar_parse(kinds))
            .replace("__RET_TYPE__", self._ret_type(ret))
            .replace("__CALL__", self._call(symbol, kinds))
            .replace("__FORMAT__", self._format(ret))
        )

    def _bench_harness(self, func_src, symbol, kinds, ret) -> str:
        return (
            _BENCH_TEMPLATE
            .replace("__FUNC_SRC__", func_src)
            .replace("__SCALAR_PARSE__", self._scalar_parse(kinds))
            .replace("__SINK_OP__", self._sink_op(symbol, kinds, ret))
        )

    def _serialize(self, inputs, kinds) -> str:
        lines = []
        for tup in inputs:
            arr, scalar = [], None
            for (kind, _), val in zip(kinds, tup):
                if kind == "array":
                    arr = val or []
                elif kind == "scalar":
                    scalar = val
            arr_s = ",".join(str(int(x)) for x in arr)
            lines.append(f"{arr_s};{int(scalar)}" if scalar is not None else arr_s)
        return "\n".join(lines) + "\n"

    def _parse_value(self, payload: str, ret: str):
        if ret == "bool":
            return payload.strip() == "true"
        if ret == "list":
            payload = payload.strip()
            return [int(x) for x in payload.split(",") if x.strip() != ""]
        return int(payload.strip())

    # ---- build / correctness ----
    def build(self, ws: Workspace, sandbox: Sandbox) -> BuildResult:
        # Compile the candidate file as-is (a library package) to catch syntax/type errors.
        res = sandbox.run(
            ["go", "build", str(ws.target_rel)],
            cwd=ws.root, timeout=120, env=GO_ENV,
        )
        if res.ok:
            return BuildResult(ok=True)
        return BuildResult(ok=False, error=res.stderr_text()[:4000])

    def run_tests(self, ws: Workspace, sandbox: Sandbox) -> TestResult:
        return TestResult(has_tests=False, ok=True)  # MVP: differential only

    # ---- execution ----
    def observe(self, ws, target, inputs, sandbox, timeout: float = 60.0) -> ObserveResult:
        kinds, ret = self._shape(target)
        if not self._supported(kinds, ret):
            return ObserveResult(ok=False, error=f"unsupported signature for MVP Go adapter: {target.signature}")

        (ws.root / "optiproofgen_obs.go").write_text(
            self._obs_harness(self._func_src(ws, target), target.symbol, kinds, ret)
        )
        (ws.root / "optiproofgen_inputs.txt").write_text(self._serialize(inputs, kinds))

        c = sandbox.run(["go", "build", "-o", "optiproofgen_obs_bin", "optiproofgen_obs.go"],
                        cwd=ws.root, timeout=180, env=GO_ENV)
        if not c.ok:
            return ObserveResult(ok=False, error="harness compile failed: " + c.stderr_text()[:3000])

        r = sandbox.run(["./optiproofgen_obs_bin", "optiproofgen_inputs.txt", "optiproofgen_obs_out.txt"],
                        cwd=ws.root, timeout=timeout, env=GO_ENV)
        if r.timed_out:
            return ObserveResult(ok=False, error="observe timed out")
        if r.returncode != 0:
            return ObserveResult(ok=False, error=r.stderr_text()[:3000])
        try:
            text = (ws.root / "optiproofgen_obs_out.txt").read_text()
        except OSError as e:
            return ObserveResult(ok=False, error=f"could not read observations: {e}")

        obs = []
        for line in text.splitlines():
            if line.startswith("PANIC"):
                msg = line[5:].strip() or "panic"
                obs.append(BehaviorObservation(ok=False, value_repr=msg, exception=msg))
            elif line.startswith("OK"):
                val = self._parse_value(line[2:].strip(), ret)
                obs.append(BehaviorObservation(ok=True, value=val, value_repr=repr(val)))
            elif line.strip():
                obs.append(BehaviorObservation(ok=False, value_repr=line, exception="parse_error"))
        return ObserveResult(ok=True, observations=obs)

    def benchmark(self, ws, target, workload, sandbox, warmup=3, min_rounds=12, max_rounds=60,
                  target_rse=0.02, timeout=180.0) -> RawBenchmark:
        kinds, ret = self._shape(target)
        if not self._supported(kinds, ret):
            return RawBenchmark(ok=False, error="unsupported signature for MVP Go adapter")

        (ws.root / "optiproofgen_bench.go").write_text(
            self._bench_harness(self._func_src(ws, target), target.symbol, kinds, ret)
        )
        (ws.root / "optiproofgen_workload.txt").write_text(self._serialize([workload], kinds))

        c = sandbox.run(["go", "build", "-o", "optiproofgen_bench_bin", "optiproofgen_bench.go"],
                        cwd=ws.root, timeout=180, env=GO_ENV)
        if not c.ok:
            return RawBenchmark(ok=False, error="bench harness compile failed: " + c.stderr_text()[:3000])

        r = sandbox.run(
            ["./optiproofgen_bench_bin", "optiproofgen_workload.txt",
             str(warmup), str(min_rounds), str(max_rounds), str(target_rse), "optiproofgen_bench_out.json"],
            cwd=ws.root, timeout=timeout, env=GO_ENV,
        )
        if r.timed_out:
            return RawBenchmark(ok=False, error="benchmark timed out")
        if r.returncode != 0:
            return RawBenchmark(ok=False, error=r.stderr_text()[:3000])
        try:
            data = json.loads((ws.root / "optiproofgen_bench_out.json").read_text())
        except Exception as e:
            return RawBenchmark(ok=False, error=f"could not parse benchmark output: {e}")
        return RawBenchmark(ok=True, samples=data["samples"], inner_loops=data["inner_loops"])

    def runtime_label(self) -> str:
        return "Go (go build, GC)"
