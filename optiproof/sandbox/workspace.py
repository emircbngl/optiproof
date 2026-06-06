"""Ephemeral per-candidate workspaces.

Every candidate is evaluated in a throwaway copy of the project so it can never
mutate the user's real files. The baseline workspace is forked once from the
target file's directory; each candidate gets its own fork of that baseline.

MVP assumption: the "project" is the target file's directory and is small and
self-contained (true for the validation corpus). Pointing at a file inside a
large repo is a Phase-3 concern (git worktrees + scoped copy).
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from ..models import Candidate, Target
from ..patch import apply_candidate

# Don't drag virtualenvs / caches into the workspace copy.
_IGNORE = shutil.ignore_patterns(
    ".venv", "venv", "__pycache__", "*.pyc", ".git", ".mypy_cache", ".pytest_cache", "node_modules"
)
_COPY_MAX_BYTES = 25 * 1024 * 1024  # per-file and total cap when gathering the workspace
_COPY_MAX_FILES = 300


class Workspace:
    def __init__(self, root: Path, target_rel: Path, tmp: Path):
        self.root = root
        self.target_rel = target_rel
        self._tmp = tmp

    @classmethod
    def fork_from_file(cls, file: Path) -> "Workspace":
        file = Path(file).resolve()
        tmp = Path(tempfile.mkdtemp(prefix="optiproof-base-"))
        dst = tmp / "work"
        dst.mkdir(parents=True)
        # Gather a minimal, SAFE workspace: the target + same-language sibling sources
        # (+ conftest.py for Python) from the SAME directory only. Non-recursive, regular
        # files only (skips sockets/FIFOs/dirs), size/count capped. We never blind-copytree
        # an arbitrary parent — pointing at a file in /tmp or $HOME must not drag the whole
        # directory (or its sockets/huge files) in. Multi-file/nested projects: Phase-2 git worktree.
        suffix = file.suffix
        copied = total = 0
        try:
            entries = sorted(file.parent.iterdir())
        except OSError:
            entries = []
        for entry in entries:
            if copied >= _COPY_MAX_FILES or total >= _COPY_MAX_BYTES:
                break
            try:
                if not entry.is_file():           # skips dirs, sockets, FIFOs, devices
                    continue
                size = entry.stat().st_size
            except OSError:
                continue
            is_target = entry == file
            eligible = (
                is_target
                or entry.suffix == suffix
                or (suffix == ".py" and entry.name == "conftest.py")
            )
            if not eligible or (size > _COPY_MAX_BYTES and not is_target):
                continue
            try:
                shutil.copy2(entry, dst / entry.name)
                copied += 1
                total += size
            except OSError:
                continue
        if not (dst / file.name).exists():        # always ensure the target is present
            shutil.copy2(file, dst / file.name)
        return cls(root=dst, target_rel=Path(file.name), tmp=tmp)

    def fork(self) -> "Workspace":
        tmp = Path(tempfile.mkdtemp(prefix="optiproof-cand-"))
        dst = tmp / "work"
        shutil.copytree(self.root, dst, ignore=_IGNORE)
        return Workspace(root=dst, target_rel=self.target_rel, tmp=tmp)

    @property
    def target_path(self) -> Path:
        return self.root / self.target_rel

    def read_target(self) -> str:
        return self.target_path.read_text()

    def write_target(self, text: str) -> None:
        self.target_path.write_text(text)

    def apply(self, target: Target, candidate: Candidate) -> None:
        """Splice the candidate's source into a fresh copy of the original target."""
        new_text = apply_candidate(self.read_target(), target, candidate)
        self.write_target(new_text)

    def cleanup(self) -> None:
        shutil.rmtree(self._tmp, ignore_errors=True)

    def __enter__(self) -> "Workspace":
        return self

    def __exit__(self, *exc) -> None:
        self.cleanup()
