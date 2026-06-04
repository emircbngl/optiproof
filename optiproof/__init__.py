"""OptiProof — proof-carrying, polyglot code optimizer.

The LLM *proposes* an optimization; a measured harness *proves* it. Only diffs
that are both (a) behaviourally identical on existing + differential tests and
(b) faster by a statistically significant margin are ever accepted.

Public API is imported lazily so individual submodules stay independently
importable (and testable) without pulling in the whole engine.
"""

__version__ = "0.1.0"


def __getattr__(name: str):  # pragma: no cover - thin lazy re-export
    if name in {"optimize", "OptimizeRequest"}:
        from . import orchestrator, models

        return {"optimize": orchestrator.optimize, "OptimizeRequest": models.OptimizeRequest}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
