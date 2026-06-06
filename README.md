# OptiProof

**Like code review, but it optimizes code — and proves the speedup.**

You point OptiProof at a function. It proposes faster implementations (via an LLM),
then a measured harness **proves** each one is (a) behaviourally identical to the
original and (b) faster by a statistically significant margin. Only candidates that
pass *both* gates are ever reported. The model proposes; the harness decides.

It's an open-source, polyglot take on the idea behind [Codeflash](https://www.codeflash.ai/):
a Python engine + agent loop that optimizes code in any language through a
`LanguageAdapter` seam (Python, Rust, and Go ship today).

## Why it doesn't mislead you

Every measurement is **in-language, same runtime, same machine, same inputs** — the
original vs. the candidate. OptiProof never compares across languages, so a result is
always correctly scoped: *"in CPython 3.13, on these inputs, candidate B is 4.2× faster,
proven."* Candidate generation works at three levels, all passing the same proof gate:

- `rewrite` — same-language algorithmic / data-structure / source-level changes;
- `native` — delegate the hot work to a native library (NumPy, etc.) — often the real
  win for numeric/array code (the right answer to "image processing is faster in C");
- `port` *(advisory, Phase 3)* — when a hotspot hits the language ceiling, prototype it
  as a Python-callable native extension and measure the real speedup *including* FFI cost.

## Install

```bash
python -m venv .venv && ./.venv/bin/pip install -e .
# optional, for the default LLM provider:
./.venv/bin/pip install anthropic   # and set ANTHROPIC_API_KEY
```

## Usage

```bash
# Find + prove a faster version of a function (needs an LLM provider):
optiproof optimize mymod.py::solve --candidates 6 --apply

# Trust the harness without an LLM:
optiproof profile mymod.py::solve                 # locate the target
optiproof verify  mymod.py::solve faster_solve.py # run only the correctness + speed gates
```

`verify` and `profile` need no API key — they let you trust the measurement machinery on
its own. A run reports the measured speedup, the 95% CI, the correctness evidence
(existing tests + N differential inputs identical), and the diff — or an honest
"no verified improvement" with the best *rejected* candidate and why.

## Use it from Claude Code (no API key)

You don't need an Anthropic API key — two subscription-friendly paths:

- **`prove` (agent-driven):** Claude Code writes the candidate implementations; the harness
  proves them. `optiproof prove FILE::FUNC cand1.py cand2.py [--sandbox docker] [--apply]`.
  A bundled **`optiproof` skill** (`.claude/skills/optiproof/`) wires this up, so you can just
  say *"optimize this function"* in Claude Code and it runs the loop.
- **`--provider claude-code`:** `optimize` shells out to the logged-in `claude` CLI instead of
  the API, using your Claude Code subscription. (`--provider` defaults to `auto`: API key if set,
  else the `claude` CLI, else `null` for tests.)

## How it works

The agent loop (`optiproof/orchestrator.py`), measure → generate → prove:

1. **Baseline** — build the original, run its tests, generate inputs, record behaviour,
   and measure baseline timing. (Refuses to optimize code that's already broken.)
2. **Generate → prove**, cheap-fail ordered so only correct candidates get benchmarked:
   - **build** (syntax/compile)
   - **correctness**: existing tests, then **differential testing** — original vs. candidate
     on hundreds of shared inputs; any difference in return value, stdout, or exception
     type ⇒ reject (with a minimal counterexample fed back to the model).
   - **speed**: warmup + adaptive sampling, then the **central significance gate** —
     Mann–Whitney U *and* the entire bootstrap CI of the speedup ratio above the threshold.
3. **Anti-luck** — the winner is re-measured from scratch and demoted if it no longer clears
   the bar (regression-to-the-mean insurance).

## Architecture

| Module | Responsibility |
|---|---|
| `orchestrator.py` | the agent loop; ties everything together |
| `adapters/base.py` | `LanguageAdapter` ABC — the polyglot seam |
| `adapters/python_adapter.py` | Python: `ast` locate, `py_compile`, `pytest`, driver-based observe/benchmark |
| `adapters/rust_adapter.py` | Rust: regex locate, std-only `rustc -O` observe/benchmark harnesses (polyglot proof) |
| `adapters/go_adapter.py` | Go: regex locate, std-only `go build` harnesses, GOPATH-mode offline (third language) |
| `verify/differential.py` + `compare.py` | the primary correctness oracle (original vs. candidate) |
| `bench/stats.py` | the single place "really faster?" is decided (Mann–Whitney + bootstrap CI) |
| `sandbox/` | ephemeral per-candidate workspaces + isolated execution |
| `llm/` | pluggable candidate generators (Anthropic default; `null` for tests) |

## Testing

```bash
./.venv/bin/python -m pytest -q
```

The suite validates the tool against its own philosophy via `benchmarks_corpus/` —
known (slow, fast, **trap**, no-win) quadruples. It must *prove* the real speedups,
*reject* the behaviour-changing traps (caught by differential testing), and *not
fabricate* wins where none exist. The whole loop is exercised deterministically with a
canned provider — no API calls, no flaky timing.

## Scope and what's next

- **Done:** **three language adapters** — Python (full measured loop), **Rust** (`&[i64] →
  i64 | bool | Vec<i64>`, std-only `rustc -O`) and **Go** (`[]int64 → int64 | bool | []int64`,
  std-only `go build`, offline GOPATH mode) — each proven end-to-end (Python 247×, Rust 19.8×,
  Go 16.4× on the same O(n²)→hash optimization); isolated **Docker sandbox** (`--sandbox docker`)
  + rich toolchain image (`--toolchain-image`, runs `pytest` + NumPy `native` candidates
  in-container); configurable benchmark workload size.
- **Phase 2:** `hyperfine` universal benchmarking, tree-sitter hotspot ranking + file scope,
  broaden the Rust/Go type subsets.
- **Phase 3:** PR/diff mode + GitHub Action, hardened nsjail sandbox, measured
  cross-language `port` advisories, more languages (C/C++/JS).

> ⚠ The `local` sandbox runs candidate code with rlimits + a timeout but is **not**
> isolated from the host. Use **`--sandbox docker`** for isolation: each execution runs in
> a throwaway container with `--network none`, CPU/memory/PID caps, and the workspace
> bind-mounted (host FS not exposed). The default `python:3.13-slim` image runs pure-Python
> targets; targets with a `pytest` suite or `native` (NumPy) candidates need a richer image
> (build one and pass `--toolchain-image`, a small follow-up to thread through the CLI).

---

## Built with Claude Code

OptiProof was designed and built end-to-end with [Claude Code](https://claude.com/claude-code),
Anthropic's agentic CLI: the competitive/landscape research, the architecture, all three
language adapters (Python, Rust, Go), the isolated Docker sandbox, and a full adversarial
self-review of the codebase. If you're exploring what an agentic coding workflow can produce —
or want a worked example of *measure-don't-guess* optimization as a code-review-style tool —
this repo is a reference. Issues and discussion welcome.

