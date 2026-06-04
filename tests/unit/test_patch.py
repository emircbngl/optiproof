"""Patch/splice tests — function replacement + prelude injection."""

from __future__ import annotations

from optiproof.models import Candidate, Target
from optiproof.patch import apply_candidate


def _target(src, symbol, start, end):
    return Target(file="m.py", symbol=symbol, language="python", start_line=start, end_line=end, source=src)


def test_replaces_only_the_function():
    text = "A = 1\n\ndef f(x):\n    return x + 1\n\nB = 2\n"
    t = _target("def f(x):\n    return x + 1", "f", 3, 4)
    cand = Candidate(id="c", new_source="def f(x):\n    return x + 2")
    out = apply_candidate(text, t, cand)
    assert "return x + 2" in out
    assert out.startswith("A = 1\n") and out.rstrip().endswith("B = 2")


def test_prelude_goes_after_future_import():
    text = "from __future__ import annotations\n\ndef f(x):\n    return x\n"
    t = _target("def f(x):\n    return x", "f", 3, 4)
    cand = Candidate(id="c", new_source="def f(x):\n    return x", module_prelude="import numpy as np")
    out = apply_candidate(text, t, cand)
    lines = out.splitlines()
    assert lines[0] == "from __future__ import annotations"
    assert lines[1] == "import numpy as np"
