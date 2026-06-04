"""Single source of truth for OptiProof's data model.

Every component speaks in these types, so the orchestrator never has to know
anything language- or provider-specific. Dependency order matters: types
referenced by ``Candidate`` / ``OptimizationResult`` are defined above them so
no forward references are needed.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class OptimizeKind(str, Enum):
    """How a candidate changes the code (see plan §"Optimizasyon kapsamı")."""

    REWRITE = "rewrite"   # same language, same deps — algorithmic/data-structure/source-level
    NATIVE = "native"     # same language, delegate to a native lib (NumPy/OpenCV/Cython)
    PORT = "port"         # advisory: re-implement the hotspot in another language as an extension


class SandboxBackend(str, Enum):
    LOCAL = "local"       # subprocess + rlimits (dev; not isolated from host FS/net)
    DOCKER = "docker"     # default isolated backend


class OptimizeRequest(BaseModel):
    """Everything the orchestrator needs for one optimization run."""

    path: Path
    selector: str                       # function name, or "file.py::func" (path may carry the file)
    language: Optional[str] = None       # auto-detect when None

    # search budget
    candidates_per_round: int = 5
    max_rounds: int = 3
    satisfied_at: float = 3.0            # early-stop once a survivor reaches this speedup

    # acceptance gates
    threshold: float = 1.10             # min median speedup ratio to accept
    significance_alpha: float = 0.05

    # correctness
    num_diff_inputs: int = 200
    seed: int = 1234

    # benchmarking
    warmup: int = 3
    min_runs: int = 12
    max_runs: int = 60
    workload_size: int = 1200           # size of the generated benchmark workload

    # plumbing
    sandbox: SandboxBackend = SandboxBackend.LOCAL
    toolchain_image: Optional[str] = None  # Docker image for the docker backend
    provider: str = "anthropic"         # or "null" (deterministic, for tests)
    model: Optional[str] = None
    budget_usd: Optional[float] = None


class Target(BaseModel):
    """A located optimization target (one function, for the MVP)."""

    file: Path
    symbol: str
    language: str
    start_line: int                      # 1-based, inclusive (includes decorators)
    end_line: int                        # 1-based, inclusive
    source: str                          # exact source text of the target (incl. decorators)
    signature: Optional[str] = None
    param_types: dict[str, str] = Field(default_factory=dict)  # param name -> annotation text


class BuildResult(BaseModel):
    ok: bool
    error: str = ""


class TestResult(BaseModel):
    has_tests: bool
    ok: bool = True
    passed: int = 0
    failed: int = 0
    output: str = ""


class DifferentialResult(BaseModel):
    """Outcome of comparing original vs candidate on identical inputs."""

    equivalent: bool
    checked: int = 0
    counterexample: Optional[str] = None  # minimal human-readable mismatch
    detail: str = ""
    quarantined: bool = False             # original was non-deterministic -> cannot differential-test
    warning: str = ""


class CorrectnessResult(BaseModel):
    ok: bool
    tests: Optional[TestResult] = None
    differential: Optional[DifferentialResult] = None
    reason: str = ""


class Measurement(BaseModel):
    """Raw benchmark samples + summary stats (seconds per call)."""

    samples: list[float] = Field(default_factory=list)
    median: float
    mean: float
    stdev: float
    n: int
    inner_loops: int = 1


class SpeedupVerdict(BaseModel):
    is_faster: bool
    ratio: float                          # baseline_median / candidate_median
    ci_low: float                         # 95% CI of the speedup ratio
    ci_high: float
    p_value: float
    reason: str = ""


class Candidate(BaseModel):
    """An LLM-proposed optimization, annotated as it passes through the gates."""

    id: str
    kind: OptimizeKind = OptimizeKind.REWRITE
    title: str = ""
    rationale: str = ""
    new_source: str                       # replacement source for the target function
    module_prelude: Optional[str] = None  # extra top-of-file code (e.g. imports)

    # filled in during evaluation
    applied: bool = False
    build_ok: Optional[bool] = None
    build_error: Optional[str] = None
    correctness: Optional[CorrectnessResult] = None
    measurement: Optional[Measurement] = None
    verdict: Optional[SpeedupVerdict] = None
    rejected_reason: Optional[str] = None

    @property
    def speedup(self) -> Optional[float]:
        return self.verdict.ratio if self.verdict else None


class RejectedCandidate(BaseModel):
    """Compact record of a candidate that did not win (for an honest report)."""

    id: str
    kind: OptimizeKind
    title: str
    reason: str
    ratio: Optional[float] = None


class OptimizationResult(BaseModel):
    """The deliverable: a verified-faster diff + proof, or an honest 'no win'."""

    target: Target
    language: str
    runtime: str = ""                     # e.g. "CPython 3.13.5"
    improved: bool = False
    baseline: Optional[Measurement] = None

    # winner (when improved)
    best: Optional[Candidate] = None
    speedup: Optional[float] = None
    ci: Optional[tuple[float, float]] = None
    diff: Optional[str] = None

    # transparency
    rejected: list[RejectedCandidate] = Field(default_factory=list)
    rounds: int = 0
    candidates_evaluated: int = 0
    notes: list[str] = Field(default_factory=list)
