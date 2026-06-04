"""Python language adapter — the MVP worked example and the template for the rest.

It locates a function with the stdlib ``ast`` module, syntax-checks with
``py_compile``, runs the project's ``pytest`` suite, and executes the target via
two small driver scripts written into the ephemeral workspace:

* the **observe** driver runs the function on each input and captures
  ``(return value, stdout, exception type)`` — the differential unit;
* the **benchmark** driver autoranges inner-loop count, warms up, then returns
  raw per-call timing samples (deep-copying the workload outside the timed region
  so mutation and copy cost never bias the measurement).
"""

from __future__ import annotations

import ast
import json
import pickle
import re
import sys
from pathlib import Path

from ..models import BuildResult, Target, TestResult
from ..sandbox.base import Sandbox
from ..sandbox.workspace import Workspace
from .base import BehaviorObservation, LanguageAdapter, ObserveResult, RawBenchmark


_OBSERVE_DRIVER = r'''
import sys, pickle, io, importlib.util

def _safe_repr(v):
    try:
        return repr(v)[:2000]
    except Exception:
        return "<unreprable>"

def _load(target_file, symbol):
    spec = importlib.util.spec_from_file_location("optiproof_target", target_file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["optiproof_target"] = mod
    spec.loader.exec_module(mod)
    return getattr(mod, symbol)

def main():
    target_file, symbol, inputs_path, out_path = sys.argv[1:5]
    func = _load(target_file, symbol)
    with open(inputs_path, "rb") as f:
        inputs = pickle.load(f)
    results = []
    for args in inputs:
        if not isinstance(args, tuple):
            args = (args,)
        buf = io.StringIO()
        saved = sys.stdout
        sys.stdout = buf
        entry = {}
        try:
            val = func(*args)
            entry["ok"] = True
            entry["exception"] = None
            try:
                pickle.dumps(val)
                entry["value"] = val
                entry["pickled"] = True
            except Exception:
                entry["value"] = None
                entry["pickled"] = False
            entry["value_repr"] = _safe_repr(val)
        except Exception as e:
            entry["ok"] = False
            entry["exception"] = type(e).__name__
            entry["value"] = None
            entry["pickled"] = True
            entry["value_repr"] = _safe_repr(e)
        finally:
            sys.stdout = saved
        entry["stdout"] = buf.getvalue()[:10000]
        results.append(entry)
    with open(out_path, "wb") as f:
        pickle.dump(results, f)

main()
'''


_BENCH_DRIVER = r'''
import sys, pickle, json, importlib.util, copy, time, statistics

MIN_CHUNK = 0.02  # seconds; grow inner-loop count until a timed chunk reaches this

def _load(target_file, symbol):
    spec = importlib.util.spec_from_file_location("optiproof_target", target_file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["optiproof_target"] = mod
    spec.loader.exec_module(mod)
    return getattr(mod, symbol)

def main():
    target_file, symbol, workload_path, out_path = sys.argv[1:5]
    warmup = int(sys.argv[5]); min_rounds = int(sys.argv[6])
    max_rounds = int(sys.argv[7]); target_rse = float(sys.argv[8])
    func = _load(target_file, symbol)
    with open(workload_path, "rb") as f:
        workload = pickle.load(f)
    if not isinstance(workload, tuple):
        workload = (workload,)

    def _mutates(wl):
        # Decide once whether the target mutates its args, so we only pay for copies when needed.
        try:
            a = copy.deepcopy(wl)
            b = copy.deepcopy(wl)
            func(*a)
            return a != b
        except Exception:
            return True  # can't tell -> be safe and copy

    MUTATES = _mutates(workload)
    COPY_BATCH = 64  # cap of deep-copies held in memory at once (bounds memory for huge L)

    def run_chunk(L):
        if not MUTATES:
            # Immutable args: reuse the same workload, no per-call copy -> O(1) extra memory.
            t0 = time.perf_counter()
            for _ in range(L):
                func(*workload)
            return time.perf_counter() - t0
        # Mutating target: fresh copy per call, but in bounded sub-batches so L can't OOM.
        elapsed = 0.0
        done = 0
        while done < L:
            b = min(COPY_BATCH, L - done)
            batch = [copy.deepcopy(workload) for _ in range(b)]  # bounded, outside the timer
            t0 = time.perf_counter()
            for a in batch:
                func(*a)
            elapsed += time.perf_counter() - t0
            done += b
        return elapsed

    L = 1
    while True:
        dt = run_chunk(L)
        if dt >= MIN_CHUNK or L >= 1_000_000:
            break
        L = L * 10 if dt <= 0 else max(L + 1, int(L * (MIN_CHUNK / dt) * 1.3))

    for _ in range(max(0, warmup)):
        run_chunk(L)

    samples = []
    for _ in range(max_rounds):
        dt = run_chunk(L)
        samples.append(dt / L)
        if len(samples) >= max(min_rounds, 3):
            mean = statistics.fmean(samples)
            sd = statistics.pstdev(samples)
            rse = (sd / (len(samples) ** 0.5)) / mean if mean > 0 else 0.0
            if rse <= target_rse:
                break

    with open(out_path, "w") as f:
        json.dump({"samples": samples, "inner_loops": L}, f)

main()
'''


def _count(text: str, pattern: str) -> int:
    m = re.search(pattern, text)
    return int(m.group(1)) if m else 0


class PythonAdapter(LanguageAdapter):
    name = "python"

    def __init__(self, python: str | None = None):
        self.python = python or sys.executable

    def _interp(self, sandbox) -> str:
        """Interpreter for this backend: host venv (local) or the container's python3."""
        return getattr(sandbox, "python_executable", None) or self.python

    # ---- discovery ----
    def detect(self, path: Path) -> float:
        return 1.0 if str(path).endswith(".py") else 0.0

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
        tree = ast.parse(src)

        # Pick the LAST top-level def with this name: that is the binding `getattr(mod, symbol)`
        # resolves to at runtime, so analysis and execution agree on a redefined/shadowed function.
        node = None
        for n in tree.body:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == symbol:
                node = n
        if node is None:  # fall back to nested functions / methods (last match)
            for n in ast.walk(tree):
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == symbol:
                    node = n
        if node is None:
            raise ValueError(f"function {symbol!r} not found in {file}")

        start = min([node.lineno] + [d.lineno for d in node.decorator_list])
        end = node.end_lineno or node.lineno
        source = "\n".join(src.splitlines()[start - 1:end])

        param_types: dict[str, str] = {}
        args = node.args
        for a in [*args.posonlyargs, *args.args, *args.kwonlyargs]:
            param_types[a.arg] = ast.unparse(a.annotation) if a.annotation else ""
        try:
            signature = f"def {symbol}({ast.unparse(args)})"
        except Exception:
            signature = f"def {symbol}(...)"

        return Target(
            file=file,
            symbol=symbol,
            language="python",
            start_line=start,
            end_line=end,
            source=source,
            signature=signature,
            param_types=param_types,
        )

    # ---- build / correctness ----
    def build(self, ws: Workspace, sandbox: Sandbox) -> BuildResult:
        res = sandbox.run(
            [self._interp(sandbox), "-m", "py_compile", str(ws.target_rel)], cwd=ws.root, timeout=30
        )
        if res.ok:
            return BuildResult(ok=True)
        return BuildResult(ok=False, error=(res.stderr_text() or res.stdout_text())[:4000])

    def run_tests(self, ws: Workspace, sandbox: Sandbox) -> TestResult:
        res = sandbox.run(
            [self._interp(sandbox), "-m", "pytest", "-q", "-p", "no:cacheprovider", "."],
            cwd=ws.root,
            timeout=120,
        )
        out = (res.stdout_text() + res.stderr_text())[:6000]
        low = out.lower()
        no_runner = "no module named" in low and "pytest" in low
        if no_runner or res.returncode == 5:  # no pytest available, or no tests collected
            return TestResult(has_tests=False, ok=True, output=out)
        passed = _count(out, r"(\d+) passed")
        failed = _count(out, r"(\d+) failed") + _count(out, r"(\d+) error")
        has_tests = (passed + failed) > 0
        ok = res.returncode == 0
        # Fail closed: pytest exited non-zero (e.g. 2 interrupted / 3 internal / 4 usage) but we
        # parsed no summary -> treat as a failing suite, not "no tests", so a candidate that breaks
        # the project's own tests cannot slip through the correctness gate.
        if res.returncode not in (0, 5) and not has_tests:
            has_tests, ok = True, False
        return TestResult(has_tests=has_tests, ok=ok, passed=passed, failed=failed, output=out)

    # ---- execution ----
    def observe(
        self, ws: Workspace, target: Target, inputs: list, sandbox: Sandbox, timeout: float = 30.0
    ) -> ObserveResult:
        inputs_path = ws.root / "__optiproof_inputs.pkl"
        out_path = ws.root / "__optiproof_obs_out.pkl"
        driver_path = ws.root / "__optiproof_obs_driver.py"
        with open(inputs_path, "wb") as f:
            pickle.dump(inputs, f)
        driver_path.write_text(_OBSERVE_DRIVER)

        res = sandbox.run(
            [self._interp(sandbox), driver_path.name, str(ws.target_rel), target.symbol,
             inputs_path.name, out_path.name],
            cwd=ws.root,
            timeout=timeout,
        )
        if res.timed_out:
            return ObserveResult(ok=False, error="observe timed out")
        if res.returncode != 0:
            return ObserveResult(ok=False, error=res.stderr_text()[:4000])
        try:
            with open(out_path, "rb") as f:
                raw = pickle.load(f)
        except Exception as e:
            return ObserveResult(
                ok=False, error=f"could not read observations: {e}; stderr={res.stderr_text()[:2000]}"
            )
        obs = [
            BehaviorObservation(
                ok=e["ok"], value=e["value"], value_repr=e["value_repr"],
                pickled=e["pickled"], stdout=e["stdout"], exception=e["exception"],
            )
            for e in raw
        ]
        return ObserveResult(ok=True, observations=obs)

    def benchmark(
        self,
        ws: Workspace,
        target: Target,
        workload,
        sandbox: Sandbox,
        warmup: int = 3,
        min_rounds: int = 12,
        max_rounds: int = 60,
        target_rse: float = 0.02,
        timeout: float = 120.0,
    ) -> RawBenchmark:
        wl_path = ws.root / "__optiproof_workload.pkl"
        out_path = ws.root / "__optiproof_bench_out.json"
        driver_path = ws.root / "__optiproof_bench_driver.py"
        with open(wl_path, "wb") as f:
            pickle.dump(workload, f)
        driver_path.write_text(_BENCH_DRIVER)

        res = sandbox.run(
            [self._interp(sandbox), driver_path.name, str(ws.target_rel), target.symbol, wl_path.name,
             out_path.name, str(warmup), str(min_rounds), str(max_rounds), str(target_rse)],
            cwd=ws.root,
            timeout=timeout,
        )
        if res.timed_out:
            return RawBenchmark(ok=False, error="benchmark timed out")
        if res.returncode != 0:
            return RawBenchmark(ok=False, error=res.stderr_text()[:4000])
        try:
            data = json.loads(out_path.read_text())
        except Exception as e:
            return RawBenchmark(
                ok=False, error=f"could not read benchmark: {e}; stderr={res.stderr_text()[:2000]}"
            )
        return RawBenchmark(ok=True, samples=data["samples"], inner_loops=data["inner_loops"])

    def runtime_label(self) -> str:
        import platform

        return f"CPython {platform.python_version()}"
