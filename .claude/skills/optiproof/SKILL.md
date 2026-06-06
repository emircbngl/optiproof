---
name: optiproof
description: >
  Optimize a function and PROVE the speedup — like code review, but for performance.
  YOU (Claude Code) write candidate implementations; the `optiproof` harness proves each
  is behaviourally identical (differential testing) and statistically faster (Mann–Whitney
  + bootstrap CI), then keeps the verified-fastest. No API key needed — the agent is the
  candidate generator, optiproof is the prover.
  Triggers on: "optimize this function", "make this faster", "speed up <func>", "/optiproof",
  "optimize <file>::<func> and prove it".
allowed-tools: Read Write Edit Bash
---

# optiproof: optimize code and prove the speedup

The model proposes; the harness decides. You generate candidate implementations; `optiproof prove`
runs the correctness + speed gates and reports only verified wins. This works on your Claude Code
subscription — no API key, no `claude -p` subprocess.

## Prerequisite

`optiproof` must be installed (`which optiproof`). If missing, install it:
`uv tool install --editable <path-to-optiproof-repo> --with anthropic`.

## Workflow

1. **Resolve the target** as `FILE::FUNCTION`. If unsure which function, run
   `optiproof profile FILE.py` to list functions, or ask the user.

2. **Read the function** and understand its exact contract: parameters/types, return value,
   raised exceptions, stdout, and ordering guarantees. Behaviour must be preserved EXACTLY.

3. **Write 2–5 candidate implementations**, each the COMPLETE replacement function (just the
   `def`/`fn`/`func` block), to its own temp file. Aim for genuinely different, high-confidence
   approaches:
   - algorithmic / data-structure (O(n²)→O(n), list-membership→set, add memoization, hoist
     invariant work, avoid quadratic string building);
   - for Python numeric/array code, a NumPy version is often the biggest win (put
     `import numpy as np` inside the function so it splices cleanly);
   - keep the signature and semantics identical — same returns, exceptions, stdout, order.

4. **Prove them** (no LLM, no key):
   ```
   optiproof prove FILE::FUNC cand1.py cand2.py [cand3.py ...] [options]
   ```
   - `--sandbox local` (default) for code you trust; `--sandbox docker` to isolate untrusted code.
   - Rust targets: `--sandbox docker --toolchain-image rust:1-slim`. Python & Go: local works.
     (Rich Python image with pytest+NumPy: `--toolchain-image optiproof-python:latest`.)
   - `--workload-size N` — raise it (e.g. 5000+) if the function is small/fast so the benchmark
     is meaningful; lower it if a call is expensive.
   - `--apply` to write the winning diff back to the file.

5. **Report honestly**:
   - `VERIFIED FASTER` → give the speedup, the 95% CI, the correctness evidence (N differential
     inputs identical), and the diff. Offer to `--apply`.
   - `NO VERIFIED IMPROVEMENT` → say so plainly. Show why candidates were rejected (behaviour
     change with the counterexample, or "not statistically significant"). Iterate with new ideas
     if promising, or conclude the function is already near-optimal.

6. **Never claim a speedup the harness didn't verify.** The harness is the oracle, not you.
   A rejected "optimization" is a good outcome — it means the tool caught a bug or a non-win.

## How the proof works (so you can reason about results)

- **Correctness:** differential testing — original vs candidate on hundreds of shared generated
  inputs; ANY difference in return value, stdout, or exception type rejects the candidate.
- **Speed:** warmup + adaptive sampling, then a Mann–Whitney U test AND the whole bootstrap CI of
  the speedup ratio must clear the threshold (default ≥1.10×); the winner is re-measured from
  scratch (anti-luck).
- **Languages:** Python (`.py`), Rust (`.rs`), Go (`.go`) behind one adapter seam.
