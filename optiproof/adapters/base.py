"""The polyglot seam: one ABC every language plugs into, plus a registry.

Design rule that keeps the engine honest across languages: an adapter *executes*
and returns **raw** observations / timing samples — it never returns a verdict.
Correctness comparison lives in ``verify/`` and significance testing lives in
``bench/stats.py``, so every language gets identical rigor.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ..models import BuildResult, Target, TestResult
from ..sandbox.base import Sandbox
from ..sandbox.workspace import Workspace


@dataclass
class BehaviorObservation:
    """What a single call to the target produced (the differential unit)."""

    ok: bool                       # True if the call returned, False if it raised
    value: Any = None              # the real return value (when picklable)
    value_repr: str = ""           # repr fallback / display
    pickled: bool = True           # False => `value` is not authoritative, use repr
    stdout: str = ""
    exception: Optional[str] = None  # exception type name when ok is False


@dataclass
class ObserveResult:
    ok: bool
    observations: list[BehaviorObservation] = field(default_factory=list)
    error: str = ""


@dataclass
class RawBenchmark:
    ok: bool
    samples: list[float] = field(default_factory=list)  # seconds per call
    inner_loops: int = 1
    error: str = ""


class LanguageAdapter(ABC):
    """Everything language-specific lives behind this interface."""

    name: str = "base"

    # ---- discovery ----
    @abstractmethod
    def detect(self, path: Path) -> float:
        """Confidence in [0, 1] that this adapter handles ``path``."""

    @abstractmethod
    def locate_target(self, path: Path, selector: str) -> Target:
        """Resolve a selector into a concrete Target (file, symbol, line span, types)."""

    # ---- build / correctness ----
    @abstractmethod
    def build(self, ws: Workspace, sandbox: Sandbox) -> BuildResult:
        """Compile / syntax-check the workspace (no-op-able for interpreted langs)."""

    @abstractmethod
    def run_tests(self, ws: Workspace, sandbox: Sandbox) -> TestResult:
        """Run the project's own test suite. ``has_tests=False`` when none found."""

    @abstractmethod
    def observe(
        self, ws: Workspace, target: Target, inputs: list, sandbox: Sandbox, timeout: float = 30.0
    ) -> ObserveResult:
        """Run the target on each input; return raw per-input behavior."""

    # ---- benchmarking ----
    @abstractmethod
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
        """Time the target on a fixed workload; return RAW per-call samples."""

    # ---- optional ----
    def profile(self, ws: Workspace, target: Target, workload, sandbox: Sandbox) -> dict:
        """Dynamic profile for hotspot selection. MVP takes an explicit target -> no-op."""
        return {}

    def runtime_label(self) -> str:
        return self.name


class AdapterRegistry:
    """Holds the known adapters and picks the best one for a path."""

    _adapters: list[LanguageAdapter] = []
    _loaded = False

    @classmethod
    def _ensure_builtins(cls) -> None:
        if cls._loaded:
            return
        cls._loaded = True
        from .go_adapter import GoAdapter
        from .python_adapter import PythonAdapter
        from .rust_adapter import RustAdapter

        cls.register(PythonAdapter())
        cls.register(RustAdapter())
        cls.register(GoAdapter())

    @classmethod
    def register(cls, adapter: LanguageAdapter) -> None:
        cls._adapters.append(adapter)

    @classmethod
    def detect(cls, path: Path, language: Optional[str] = None) -> LanguageAdapter:
        cls._ensure_builtins()
        if language:
            return cls.get(language)
        ranked = sorted(cls._adapters, key=lambda a: a.detect(Path(path)), reverse=True)
        if not ranked or ranked[0].detect(Path(path)) <= 0:
            raise ValueError(f"no language adapter can handle {path!r}")
        return ranked[0]

    @classmethod
    def get(cls, name: str) -> LanguageAdapter:
        cls._ensure_builtins()
        for a in cls._adapters:
            if a.name == name:
                return a
        raise ValueError(f"no adapter named {name!r}")
