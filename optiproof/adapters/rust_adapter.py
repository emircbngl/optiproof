"""Rust language adapter — proves the polyglot seam.

Same engine (differential testing + the central statistics gate + the sandbox),
a different compiled language behind the same ABC. To stay offline-friendly (so it
runs in a ``--network none`` container) it uses **std only** and compiles with
``rustc -O`` — no Cargo, no crates.io.

MVP-supported signature subset (documented, like the Python adapter's scope):
    fn NAME(xs: &[i64] | &Vec<i64>  [, k: i64]) -> i64 | bool | Vec<i64>
This covers the optimization-corpus spirit (count / sum / dedup / pair-search over
integer arrays). Only the reference array forms are accepted (the harness binds an
immutable ``Vec<i64>`` and passes ``&__arr``); other element types / by-value / &mut
are reported as unsupported rather than silently failing to compile.

Results are transported via FILES (not stdout), so the sandbox's stdout cap can't
truncate the protocol. Panics are captured WITH their message, so two different
panics are not conflated as equivalent.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from ..models import BuildResult, Target, TestResult
from ..sandbox.base import Sandbox
from ..sandbox.workspace import Workspace
from .base import BehaviorObservation, LanguageAdapter, ObserveResult, RawBenchmark

# Return types we can format+parse. Params are restricted further (see _shape).
_INT_RUST = {"i64", "i32", "u64", "u32", "usize", "isize", "i16", "u16", "i8", "u8"}
_ARRAY_PARAMS = {"&[i64]", "&Vec<i64>"}


# ---------------------------------------------------------------- harness templates
_PARSE_FN = r"""
fn __optiproof_parse_arr(s: &str) -> Vec<i64> {
    let t = s.trim();
    if t.is_empty() { return Vec::new(); }
    t.split(',').map(|x| x.trim().parse::<i64>().unwrap()).collect()
}
"""

_OBS_TEMPLATE = (
    "__TARGET_SRC__\n"
    + _PARSE_FN
    + r"""
fn main() {
    std::panic::set_hook(Box::new(|_| {}));
    let args: Vec<String> = std::env::args().collect();
    let input = std::fs::read_to_string(&args[1]).unwrap();
    let mut out = String::new();
    for line in input.lines() {
        let parts: Vec<&str> = line.splitn(2, ';').collect();
        let __arr = __optiproof_parse_arr(parts[0]);
        __SCALAR_PARSE__
        let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| __CALL__));
        match res {
            Ok(v) => { out.push_str("OK "); __FORMAT__; out.push('\n'); }
            Err(e) => {
                let msg = e.downcast_ref::<&str>().map(|s| s.to_string())
                    .or_else(|| e.downcast_ref::<String>().cloned())
                    .unwrap_or_else(|| "panic".to_string());
                out.push_str("PANIC ");
                out.push_str(&msg.replace('\n', " "));
                out.push('\n');
            }
        }
    }
    std::fs::write(&args[2], out).unwrap();
}
"""
)

_BENCH_TEMPLATE = (
    "__TARGET_SRC__\n"
    + _PARSE_FN
    + r"""
fn main() {
    std::panic::set_hook(Box::new(|_| {}));
    let args: Vec<String> = std::env::args().collect();
    let input = std::fs::read_to_string(&args[1]).unwrap();
    let warmup: i64 = args[2].parse().unwrap();
    let min_rounds: usize = args[3].parse().unwrap();
    let max_rounds: usize = args[4].parse().unwrap();
    let target_rse: f64 = args[5].parse().unwrap();
    let out_path = &args[6];
    let line = input.lines().next().unwrap_or("");
    let parts: Vec<&str> = line.splitn(2, ';').collect();
    let __arr = __optiproof_parse_arr(parts[0]);
    __SCALAR_PARSE__
    let run_chunk = |l: usize| -> f64 {
        let start = std::time::Instant::now();
        for _ in 0..l {
            let r = __CALL_BB__;
            std::hint::black_box(r);
        }
        start.elapsed().as_secs_f64()
    };
    let mut l: usize = 1;
    loop {
        let dt = run_chunk(l);
        if dt >= 0.02 || l >= 1_000_000 { break; }
        l = if dt <= 0.0 { l * 10 } else { std::cmp::max(l + 1, (l as f64 * (0.02 / dt) * 1.3) as usize) };
    }
    for _ in 0..warmup { run_chunk(l); }
    let mut samples: Vec<f64> = Vec::new();
    for _ in 0..max_rounds {
        let dt = run_chunk(l);
        samples.push(dt / l as f64);
        if samples.len() >= std::cmp::max(min_rounds, 3) {
            let n = samples.len() as f64;
            let mean = samples.iter().sum::<f64>() / n;
            let var = samples.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n;
            let rse = if mean > 0.0 { (var.sqrt() / n.sqrt()) / mean } else { 0.0 };
            if rse <= target_rse { break; }
        }
    }
    let s: Vec<String> = samples.iter().map(|x| format!("{:.12e}", x)).collect();
    let json = format!("{{\"samples\":[{}],\"inner_loops\":{}}}", s.join(","), l);
    std::fs::write(out_path, json).unwrap();
}
"""
)


def _blank_noncode(src: str) -> str:
    """Return src with string/char literals and comments blanked to spaces (length-preserving),
    so structural brace/paren scanning isn't fooled by braces inside them."""
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
        elif c == "/" and i + 1 < n and src[i + 1] == "/":
            while i < n and src[i] != "\n":
                out[i] = " "
                i += 1
        elif c == "/" and i + 1 < n and src[i + 1] == "*":
            depth = 1
            out[i] = out[i + 1] = " "
            i += 2
            while i < n and depth > 0:
                if src[i] == "/" and i + 1 < n and src[i + 1] == "*":
                    depth += 1
                    out[i] = out[i + 1] = " "
                    i += 2
                elif src[i] == "*" and i + 1 < n and src[i + 1] == "/":
                    depth -= 1
                    out[i] = out[i + 1] = " "
                    i += 2
                else:
                    if src[i] != "\n":
                        out[i] = " "
                    i += 1
        elif c == "'":
            # char literal ('x' or '\n') vs lifetime ('a): only blank a real char literal
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


def _split_params(s: str) -> list[str]:
    parts, depth, cur = [], 0, ""
    for ch in s:
        if ch in "<[(":
            depth += 1
            cur += ch
        elif ch in ">])":
            depth -= 1
            cur += ch
        elif ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    if cur.strip():
        parts.append(cur)
    return parts


class RustAdapter(LanguageAdapter):
    name = "rust"

    # ---- discovery ----
    def detect(self, path: Path) -> float:
        return 1.0 if str(path).endswith(".rs") else 0.0

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
        scan = _blank_noncode(src)  # structural scan ignores strings/chars/comments

        m = re.search(r"\bfn\s+" + re.escape(symbol) + r"\s*\(", scan)
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
        ret_str = src[i + 1:brace].strip()  # e.g. "-> i64"
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
        for p in _split_params(params_str):
            if ":" in p:
                nm, ty = p.split(":", 1)
                param_types[nm.strip()] = ty.strip()

        signature = f"fn {symbol}({params_str.strip()}) {ret_str}".strip()
        return Target(
            file=file, symbol=symbol, language="rust",
            start_line=start_line, end_line=end_line, source=source,
            signature=signature, param_types=param_types,
        )

    # ---- shape / codegen ----
    def _shape(self, target: Target) -> tuple[list[tuple[str, str]], str]:
        kinds: list[tuple[str, str]] = []
        for ty in target.param_types.values():
            t = ty.replace(" ", "")
            if t in _ARRAY_PARAMS:               # only the forms `&__arr` actually satisfies
                kinds.append(("array", t))
            elif t == "i64":                     # harness parses the scalar as i64
                kinds.append(("scalar", t))
            else:
                kinds.append(("unknown", t))
        ret = "unit"
        # Use only the return type (the part of the signature after '->' and before any where/brace).
        msig = re.search(r"->\s*([A-Za-z0-9_:<>\[\] ]+)", target.signature or "")
        if msig:
            r = msig.group(1).split(" where")[0].strip().replace(" ", "")
            if r in _INT_RUST:
                ret = "int"
            elif r == "bool":
                ret = "bool"
            elif r.startswith("Vec<i") or r.startswith("Vec<u"):
                ret = "list"
            else:
                ret = "unknown"
        return kinds, ret

    def _supported(self, kinds, ret) -> bool:
        arrays = [k for k, _ in kinds if k == "array"]
        scalars = [k for k, _ in kinds if k == "scalar"]
        unknown = [k for k, _ in kinds if k == "unknown"]
        return (not unknown) and len(arrays) == 1 and len(scalars) <= 1 and ret in ("int", "bool", "list")

    def _call(self, symbol, kinds, blackbox: bool) -> str:
        args = []
        for kind, _ in kinds:
            if kind == "array":
                args.append("std::hint::black_box(&__arr)" if blackbox else "&__arr")
            elif kind == "scalar":
                args.append("std::hint::black_box(__k)" if blackbox else "__k")
        return f"{symbol}({', '.join(args)})"

    def _scalar_parse(self, kinds) -> str:
        if any(k == "scalar" for k, _ in kinds):
            return "let __k: i64 = parts.get(1).map(|s| s.trim().parse::<i64>().unwrap()).unwrap_or(0);"
        return ""

    def _format(self, ret: str) -> str:
        if ret == "int":
            return "out.push_str(&v.to_string())"
        if ret == "bool":
            return 'out.push_str(if v { "true" } else { "false" })'
        return "out.push_str(&v.iter().map(|z| z.to_string()).collect::<Vec<String>>().join(\",\"))"

    def _obs_harness(self, target_src, symbol, kinds, ret) -> str:
        return (
            _OBS_TEMPLATE
            .replace("__TARGET_SRC__", target_src)
            .replace("__SCALAR_PARSE__", self._scalar_parse(kinds))
            .replace("__CALL__", self._call(symbol, kinds, blackbox=False))
            .replace("__FORMAT__", self._format(ret))
        )

    def _bench_harness(self, target_src, symbol, kinds, ret) -> str:
        return (
            _BENCH_TEMPLATE
            .replace("__TARGET_SRC__", target_src)
            .replace("__SCALAR_PARSE__", self._scalar_parse(kinds))
            .replace("__CALL_BB__", self._call(symbol, kinds, blackbox=True))
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
        res = sandbox.run(
            ["rustc", "--crate-type", "lib", "--edition", "2021", "-O",
             "--emit=metadata", str(ws.target_rel), "-o", "__optiproof_meta.rmeta"],
            cwd=ws.root, timeout=120,
        )
        if res.ok:
            return BuildResult(ok=True)
        return BuildResult(ok=False, error=res.stderr_text()[:4000])

    def run_tests(self, ws: Workspace, sandbox: Sandbox) -> TestResult:
        # MVP: no cargo test runner; correctness rests on differential testing.
        return TestResult(has_tests=False, ok=True)

    # ---- execution ----
    def observe(self, ws, target, inputs, sandbox, timeout: float = 60.0) -> ObserveResult:
        kinds, ret = self._shape(target)
        if not self._supported(kinds, ret):
            return ObserveResult(ok=False, error=f"unsupported signature for MVP Rust adapter: {target.signature}")

        (ws.root / "__optiproof_obs.rs").write_text(self._obs_harness(ws.read_target(), target.symbol, kinds, ret))
        (ws.root / "__optiproof_inputs.txt").write_text(self._serialize(inputs, kinds))

        c = sandbox.run(
            ["rustc", "--edition", "2021", "-O", "__optiproof_obs.rs", "-o", "__optiproof_obs_bin"],
            cwd=ws.root, timeout=180,
        )
        if not c.ok:
            return ObserveResult(ok=False, error="harness compile failed: " + c.stderr_text()[:3000])

        r = sandbox.run(
            ["./__optiproof_obs_bin", "__optiproof_inputs.txt", "__optiproof_obs_out.txt"],
            cwd=ws.root, timeout=timeout,
        )
        if r.timed_out:
            return ObserveResult(ok=False, error="observe timed out")
        if r.returncode != 0:
            return ObserveResult(ok=False, error=r.stderr_text()[:3000])
        try:
            text = (ws.root / "__optiproof_obs_out.txt").read_text()
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
            return RawBenchmark(ok=False, error="unsupported signature for MVP Rust adapter")

        (ws.root / "__optiproof_bench.rs").write_text(self._bench_harness(ws.read_target(), target.symbol, kinds, ret))
        (ws.root / "__optiproof_workload.txt").write_text(self._serialize([workload], kinds))

        c = sandbox.run(
            ["rustc", "--edition", "2021", "-O", "__optiproof_bench.rs", "-o", "__optiproof_bench_bin"],
            cwd=ws.root, timeout=180,
        )
        if not c.ok:
            return RawBenchmark(ok=False, error="bench harness compile failed: " + c.stderr_text()[:3000])

        r = sandbox.run(
            ["./__optiproof_bench_bin", "__optiproof_workload.txt",
             str(warmup), str(min_rounds), str(max_rounds), str(target_rse), "__optiproof_bench_out.json"],
            cwd=ws.root, timeout=timeout,
        )
        if r.timed_out:
            return RawBenchmark(ok=False, error="benchmark timed out")
        if r.returncode != 0:
            return RawBenchmark(ok=False, error=r.stderr_text()[:3000])
        try:
            data = json.loads((ws.root / "__optiproof_bench_out.json").read_text())
        except Exception as e:
            return RawBenchmark(ok=False, error=f"could not parse benchmark output: {e}")
        return RawBenchmark(ok=True, samples=data["samples"], inner_loops=data["inner_loops"])

    def runtime_label(self) -> str:
        return "Rust (rustc -O, edition 2021)"
