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
        shutil.copytree(file.parent, dst, ignore=_IGNORE)
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
