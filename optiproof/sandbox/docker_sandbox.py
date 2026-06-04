"""Default isolated backend: run each command in a throwaway Docker container.

Isolation per execution:
- ``--network none``    : no network (no exfiltration / no surprise downloads)
- ``--cpus`` / ``--memory`` / ``--pids-limit`` : resource caps
- ``--user uid:gid``    : files written into the mounted workspace stay owned by the
                          host user, so teardown (rmtree) doesn't hit permission errors
- ``--rm`` + a hard wall-clock timeout (the container is killed on timeout)

The workspace directory is bind-mounted at ``/work`` and the command runs there, so
the adapter passes workspace-relative paths and ``python3`` (the image's interpreter)
via ``python_executable``.

Image note: the default ``python:3.13-slim`` runs the (stdlib-only) drivers for
pure-Python targets. Candidates that delegate to native libs (NumPy, etc.) or targets
with a pytest suite need a richer image — pass ``--toolchain-image`` / build one.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Optional, Sequence

from .base import OUTPUT_CAP, ExecResult, Sandbox

DEFAULT_IMAGE = "python:3.13-slim"


class DockerSandbox(Sandbox):
    python_executable = "python3"

    def __init__(
        self,
        image: Optional[str] = None,
        cpus: str = "2",
        memory: str = "2g",
        pids: int = 512,
        pull: bool = True,
    ):
        self.image = image or DEFAULT_IMAGE
        self.cpus = cpus
        self.memory = memory
        self.pids = pids
        self._counter = 0
        self._ensure_available(pull)

    def _docker(self, args: Sequence[str], timeout: float = 600) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["docker", *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout
        )

    def _ensure_available(self, pull: bool) -> None:
        try:
            info = self._docker(["info"], timeout=30)
        except FileNotFoundError as e:
            raise RuntimeError("docker CLI not found — install Docker or use --sandbox local") from e
        except subprocess.TimeoutExpired as e:
            raise RuntimeError("docker daemon not responding — is Docker running?") from e
        if info.returncode != 0:
            raise RuntimeError(
                "docker daemon not available (is Docker running?): "
                + info.stderr.decode("utf-8", "replace")[:300]
            )
        present = self._docker(["image", "inspect", self.image], timeout=30).returncode == 0
        if not present:
            if not pull:
                raise RuntimeError(f"image {self.image!r} not present and pull disabled")
            res = self._docker(["pull", self.image], timeout=600)
            if res.returncode != 0:
                raise RuntimeError(
                    f"failed to pull image {self.image!r}: "
                    + res.stderr.decode("utf-8", "replace")[:300]
                )

    def run(
        self,
        cmd: Sequence[str],
        cwd: Path,
        timeout: float,
        env: Optional[dict] = None,
        stdin: Optional[bytes] = None,
    ) -> ExecResult:
        self._counter += 1
        name = f"optiproof-{os.getpid()}-{self._counter}"
        docker_cmd = [
            "docker", "run", "--rm", "--name", name,
            "--network", "none",
            "--cpus", str(self.cpus),
            "--memory", str(self.memory),
            "--pids-limit", str(self.pids),
            "--user", f"{os.getuid()}:{os.getgid()}",
            "-v", f"{cwd}:/work", "-w", "/work",
        ]
        for k, v in (env or {}).items():
            docker_cmd += ["-e", f"{k}={v}"]
        docker_cmd += [self.image, *[str(c) for c in cmd]]

        start = time.perf_counter()
        try:
            proc = subprocess.run(
                docker_cmd, input=stdin, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout
            )
            return ExecResult(
                returncode=proc.returncode,
                stdout=proc.stdout[:OUTPUT_CAP],
                stderr=proc.stderr[:OUTPUT_CAP],
                timed_out=False,
                duration=time.perf_counter() - start,
            )
        except subprocess.TimeoutExpired as exc:
            try:
                self._docker(["kill", name], timeout=20)  # stop the lingering container
            except Exception:
                pass
            return ExecResult(
                returncode=-9,
                stdout=(exc.stdout or b"")[:OUTPUT_CAP],
                stderr=(exc.stderr or b"")[:OUTPUT_CAP],
                timed_out=True,
                duration=time.perf_counter() - start,
            )
