"""Machine-readable report. Heavy sample arrays are trimmed for readability."""

from __future__ import annotations

import json

from ..models import OptimizationResult


def to_dict(result: OptimizationResult) -> dict:
    d = result.model_dump(mode="json")
    if d.get("baseline"):
        d["baseline"].pop("samples", None)
    best = d.get("best")
    if best and best.get("measurement"):
        best["measurement"].pop("samples", None)
    if best and best.get("correctness") and best["correctness"].get("tests"):
        best["correctness"]["tests"].pop("output", None)
    return d


def to_json(result: OptimizationResult) -> str:
    return json.dumps(to_dict(result), indent=2, default=str)
