"""Deterministic, type-driven input generation.

Inputs are derived from the target's parameter annotations, falling back to
name heuristics. Everything is driven by a single seeded ``random.Random`` so a
candidate that fails differential testing fails reproducibly, and the *same*
inputs are reused across all candidates of a target.

Two products:
* ``generate_inputs`` — many small inputs (incl. edge cases) for differential testing.
  Integer ranges are kept small so a still-unoptimized (e.g. exponential) original
  can actually run on every input — you can't differential-test against an original
  that never returns.
* ``make_workload`` — one larger, fixed input that makes the benchmark cost measurable.

MVP scope: annotation- + name-driven. Probe-based shape discovery and Hypothesis
strategies are a Phase-2 robustness upgrade behind this same interface.
"""

from __future__ import annotations

import random
import string

from ..models import Target

_INT_NAMES = {"n", "k", "i", "j", "x", "y", "count", "num", "size", "length",
              "len", "idx", "index", "amount", "r", "c", "m", "steps", "depth",
              "target", "goal", "limit", "threshold", "bound", "cap", "total",
              "start", "stop", "step", "base", "offset", "width", "height"}
_STR_NAMES = {"s", "text", "string", "word", "name", "line", "msg", "message", "char"}
_BOOL_NAMES = {"flag", "cond", "enabled", "verbose", "reverse", "strict"}
_LIST_NAMES = {"xs", "arr", "nums", "items", "lst", "data", "seq", "values", "vals",
               "array", "numbers", "elements", "l", "points", "rows", "samples"}


def infer_tag(annotation: str, name: str) -> str:
    a = (annotation or "").lower().replace(" ", "")
    if a:
        # Rust-ish types (so the Rust adapter gets correct inputs by annotation, not just name)
        if "vec<i" in a or "vec<u" in a or a.startswith(("&[i", "&[u", "[i", "[u", "&mut[i", "&vec<i", "&vec<u")):
            return "list_int"
        if "vec<f" in a or a.startswith(("&[f", "[f", "&vec<f")):
            return "list_float"
        if a in ("i64", "i32", "u64", "u32", "usize", "isize", "i16", "u16", "i8", "u8"):
            return "int"
        if a in ("f64", "f32"):
            return "float"
        # Go-ish types
        if a.startswith(("[]int", "[]uint", "[]rune")):
            return "list_int"
        if a.startswith("[]float"):
            return "list_float"
        if a in ("int64", "int32", "int16", "int8", "uint64", "uint32", "uint", "rune"):
            return "int"
        if a in ("float64", "float32"):
            return "float"
        if a.startswith(("list", "sequence", "iterable", "tuple")) or "[" in a and a.split("[")[0] in ("list", "sequence", "iterable", "tuple"):
            if "str" in a:
                return "list_str"
            if "float" in a:
                return "list_float"
            return "list_int"
        if a == "int":
            return "int"
        if a == "float":
            return "float"
        if a == "str":
            return "str"
        if a == "bool":
            return "bool"
        if a.startswith(("dict", "mapping")):
            return "dict_str_int"
        if a.startswith(("set", "frozenset")):
            return "set_int"
    n = name.lower()
    if n in _INT_NAMES:
        return "int"
    if n in _STR_NAMES:
        return "str"
    if n in _BOOL_NAMES:
        return "bool"
    if n in _LIST_NAMES:
        return "list_int"
    return "list_int"  # safest default for an optimization corpus (sequence processing)


def _rand_str(r: random.Random, length: int) -> str:
    return "".join(r.choice(string.ascii_lowercase) for _ in range(length))


def gen_value(tag: str, r: random.Random, size: int):
    if tag == "int":
        return r.randint(0, 18)            # small + non-negative so slow originals still terminate
    if tag == "float":
        return round(r.uniform(-50, 50), 6)
    if tag == "bool":
        return r.choice([True, False])
    if tag == "str":
        return _rand_str(r, r.randint(0, max(1, size)))
    if tag == "list_int":
        return [r.randint(-20, 20) for _ in range(r.randint(0, size))]
    if tag == "list_float":
        return [round(r.uniform(-20, 20), 4) for _ in range(r.randint(0, size))]
    if tag == "list_str":
        return [_rand_str(r, r.randint(0, 6)) for _ in range(r.randint(0, size))]
    if tag == "set_int":
        return {r.randint(-20, 20) for _ in range(r.randint(0, size))}
    if tag == "dict_str_int":
        return {_rand_str(r, 3): r.randint(0, 100) for _ in range(r.randint(0, size))}
    return [r.randint(-20, 20) for _ in range(r.randint(0, size))]


def _tags(target: Target) -> list[str]:
    return [infer_tag(ann, name) for name, ann in target.param_types.items()]


def generate_inputs(target: Target, seed: int = 1234, n: int = 200, size: int = 40) -> list[tuple]:
    tags = _tags(target)
    if not tags:
        return [()]  # no-arg function: a single empty call
    r = random.Random(seed)
    inputs: list[tuple] = []
    for edge in (0, 1, 2, 5, 10):                       # deterministic edge sizes first
        inputs.append(tuple(gen_value(t, r, edge) for t in tags))
    while len(inputs) < n:
        inputs.append(tuple(gen_value(t, r, size) for t in tags))
    return inputs[:n]


def make_workload(target: Target, seed: int = 1234, big: int = 1200) -> tuple:
    tags = _tags(target)
    if not tags:
        return ()
    r = random.Random(seed + 999)
    vals = []
    for t in tags:
        if t == "int":
            vals.append(26)                            # moderate: cheap for exp, autoranged for poly
        elif t == "float":
            vals.append(round(r.uniform(-1000, 1000), 6))
        elif t == "bool":
            vals.append(True)
        elif t == "str":
            vals.append(_rand_str(r, big))
        elif t == "list_int":
            vals.append([r.randint(-1000, 1000) for _ in range(big)])
        elif t == "list_float":
            vals.append([round(r.uniform(-1000, 1000), 4) for _ in range(big)])
        elif t == "list_str":
            vals.append([_rand_str(r, r.randint(1, 8)) for _ in range(big)])
        elif t == "set_int":
            vals.append({r.randint(-big, big) for _ in range(big)})
        elif t == "dict_str_int":
            vals.append({_rand_str(r, 5): r.randint(0, 1000) for _ in range(big)})
        else:
            vals.append([r.randint(-1000, 1000) for _ in range(big)])
    return tuple(vals)
