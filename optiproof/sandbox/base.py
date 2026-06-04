"""Sandbox abstraction — every execution of (possibly LLM-written) code goes through it."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from ..models import SandboxBackend

# Captured output is truncated to keep a runaway `print` from blowing up memory.
OUTPUT_CAP = 1_000_000


@dataclass
class ExecResult:
    returncode: int
    stdout: bytes
    stderr: bytes
    timed_out: bool = False
    duration: float = 0.0

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out

    def stderr_text(self) -> str:
        return self.stderr.decode("utf-8", "replace")

    def stdout_text(self) -> str:
        return self.stdout.decode("utf-8", "replace")


class Sandbox(ABC):
    """Run a command in an isolated way and report how it went."""

    # The interpreter an adapter should use for this backend (host venv vs the
    # container's python). Adapters read this instead of hardcoding sys.executable.
    python_executable: Optional[str] = None

    @abstractmethod
    def run(
        self,
        cmd: Sequence[str],
        cwd: Path,
        timeout: float,
        env: Optional[dict] = None,
        stdin: Optional[bytes] = None,
    ) -> ExecResult: ...

    def cleanup(self) -> None:  # pragma: no cover - backends override if needed
        pass

    @staticmethod
    def create(
        backend: SandboxBackend = SandboxBackend.LOCAL,
        toolchain_image: Optional[str] = None,
        **kwargs,
    ) -> "Sandbox":
        if backend == SandboxBackend.LOCAL:
            from .local_sandbox import LocalSandbox

            return LocalSandbox(**kwargs)
        if backend == SandboxBackend.DOCKER:
            from .docker_sandbox import DockerSandbox

            return DockerSandbox(image=toolchain_image, **kwargs)
        raise ValueError(f"unknown sandbox backend: {backend!r}")
