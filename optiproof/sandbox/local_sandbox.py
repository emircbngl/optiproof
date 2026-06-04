"""Local dev sandbox: subprocess + POSIX rlimits + a hard wall-clock timeout.

This backend is NOT isolated from the host filesystem or network — it exists so
the engine runs on a bare laptop. The default, isolated backend is Docker; this
one requires the explicit ``--unsafe-local`` flag at the CLI.

We deliberately do *not* set ``RLIMIT_AS`` by default: on macOS the interpreter
(and especially NumPy) reserve large virtual address spaces, so an address-space
cap produces spurious crashes. The real guards here are CPU time + wall timeout;
memory capping is left to the Docker backend.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

from .base import OUTPUT_CAP, ExecResult, Sandbox

try:
    import resource
except ImportError:  # pragma: no cover - non-POSIX
    resource = None  # type: ignore


class LocalSandbox(Sandbox):
    def __init__(
        self,
        cpu_seconds: int = 300,
        nproc: Optional[int] = 1024,
        fsize_bytes: Optional[int] = 256 * 1024 * 1024,
        mem_bytes: Optional[int] = None,
    ):
        self.cpu_seconds = cpu_seconds
        self.nproc = nproc
        self.fsize_bytes = fsize_bytes
        self.mem_bytes = mem_bytes
        self.python_executable = sys.executable

    def _preexec(self):
        if resource is None:
            return None

        def set_limits():
            for res_name, value in (
                ("RLIMIT_CPU", self.cpu_seconds),
                ("RLIMIT_NPROC", self.nproc),
                ("RLIMIT_FSIZE", self.fsize_bytes),
                ("RLIMIT_AS", self.mem_bytes),
            ):
                if value is None:
                    continue
                res = getattr(resource, res_name, None)
                if res is None:
                    continue
                try:
                    soft_cap = value + 1 if res_name == "RLIMIT_CPU" else value
                    resource.setrlimit(res, (value, soft_cap))
                except (ValueError, OSError):
                    pass

        return set_limits

    def run(
        self,
        cmd: Sequence[str],
        cwd: Path,
        timeout: float,
        env: Optional[dict] = None,
        stdin: Optional[bytes] = None,
    ) -> ExecResult:
        full_env = dict(os.environ)
        if env:
            full_env.update(env)
        start = time.perf_counter()
        try:
            proc = subprocess.run(
                [str(c) for c in cmd],
                cwd=str(cwd),
                env=full_env,
                input=stdin,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
                preexec_fn=self._preexec(),
            )
            return ExecResult(
                returncode=proc.returncode,
                stdout=proc.stdout[:OUTPUT_CAP],
                stderr=proc.stderr[:OUTPUT_CAP],
                timed_out=False,
                duration=time.perf_counter() - start,
            )
        except subprocess.TimeoutExpired as exc:
            return ExecResult(
                returncode=-9,
                stdout=(exc.stdout or b"")[:OUTPUT_CAP],
                stderr=(exc.stderr or b"")[:OUTPUT_CAP],
                timed_out=True,
                duration=time.perf_counter() - start,
            )
