"""The equality oracle.

Compares two behavior observations (original vs candidate). Any difference in
return value, stdout, or raised-exception type is a behavior change. Floats are
compared with tolerance (optimizations legitimately reassociate FP ops); sets and
dicts compare order-insensitively by nature; lists/tuples are order-sensitive
because order is behavior.
"""

from __future__ import annotations

import math

from ..adapters.base import BehaviorObservation

REL_TOL = 1e-9
ABS_TOL = 1e-12


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def values_equal(x, y, rel_tol: float = REL_TOL, abs_tol: float = ABS_TOL) -> bool:
    # bool is behaviourally distinct from int — True must NOT equal 1 (they serialize
    # and type-check differently). If either side is a bool, require an exact type+value match.
    if isinstance(x, bool) or isinstance(y, bool):
        return type(x) is type(y) and x == y
    if _is_number(x) and _is_number(y):
        if isinstance(x, float) and isinstance(y, float) and math.isnan(x) and math.isnan(y):
            return True
        return math.isclose(x, y, rel_tol=rel_tol, abs_tol=abs_tol)
    if isinstance(x, (list, tuple)) and isinstance(y, (list, tuple)):
        return (
            type(x) is type(y)
            and len(x) == len(y)
            and all(values_equal(a, b, rel_tol, abs_tol) for a, b in zip(x, y))
        )
    if isinstance(x, dict) and isinstance(y, dict):
        if set(x.keys()) != set(y.keys()):
            return False
        return all(values_equal(x[k], y[k], rel_tol, abs_tol) for k in x)
    if isinstance(x, (set, frozenset)) and isinstance(y, (set, frozenset)):
        return x == y
    try:
        return bool(x == y)
    except Exception:
        return repr(x) == repr(y)


def compare_observations(
    a: BehaviorObservation, b: BehaviorObservation, rel_tol: float = REL_TOL, abs_tol: float = ABS_TOL
) -> tuple[bool, str]:
    """Return (equal, reason). ``a`` is the original, ``b`` the candidate."""
    if a.ok != b.ok:
        orig = a.exception or a.value_repr
        cand = b.exception or b.value_repr
        return False, f"one raised, the other returned (orig={orig!r}, cand={cand!r})"
    if not a.ok:
        if a.exception == b.exception:
            return True, ""
        return False, f"different exception type (orig {a.exception}, cand {b.exception})"
    if a.stdout != b.stdout:
        return False, f"different stdout (orig {a.stdout!r}, cand {b.stdout!r})"
    if a.pickled and b.pickled:
        if values_equal(a.value, b.value, rel_tol, abs_tol):
            return True, ""
        return False, f"different return value (orig {a.value_repr}, cand {b.value_repr})"
    if a.value_repr == b.value_repr:
        return True, ""
    return False, f"different return value (orig {a.value_repr}, cand {b.value_repr})"
