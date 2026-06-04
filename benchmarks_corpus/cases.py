"""The corpus quadruples.

Each ``Case`` carries:
- ``original``: the full source of a file containing a slow function;
- ``fast``: a correct, faster replacement the tool MUST accept (true positive);
- ``trap``: a plausible-but-wrong "optimization" the tool MUST reject (caught by
  differential testing even though it might fool a naive generated test);
- ``nowin``: a correct rewrite that is NOT meaningfully faster — the tool MUST
  report no significant improvement (guards the statistics).

Functions are chosen so a generic large workload reliably exercises the slow path
(no early-return), so the measured speedup is stable.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Case:
    name: str
    symbol: str
    original: str
    fast: str
    trap: str
    nowin: str


CASES: list[Case] = [
    Case(
        name="count_unique_set",
        symbol="count_unique",
        original=(
            "def count_unique(xs):\n"
            "    seen = []\n"
            "    for x in xs:\n"
            "        if x not in seen:\n"
            "            seen.append(x)\n"
            "    return len(seen)\n"
        ),
        fast="def count_unique(xs):\n    return len(set(xs))\n",
        trap="def count_unique(xs):\n    return len(xs)\n",  # counts all, not unique
        nowin=(
            "def count_unique(xs):\n"
            "    seen = []\n"
            "    for x in xs:\n"
            "        if x not in seen:\n"
            "            seen.append(x)\n"
            "    return len(seen)\n"
        ),
    ),
    Case(
        name="join_strings",
        symbol="join_strings",
        original=(
            "def join_strings(parts: list[str]):\n"
            "    s = ''\n"
            "    for p in parts:\n"
            "        s += p\n"
            "    return s\n"
        ),
        fast="def join_strings(parts: list[str]):\n    return ''.join(parts)\n",
        trap="def join_strings(parts: list[str]):\n    return ' '.join(parts)\n",  # adds spaces
        nowin=(
            "def join_strings(parts: list[str]):\n"
            "    s = ''\n"
            "    for p in parts:\n"
            "        s += p\n"
            "    return s\n"
        ),
    ),
    Case(
        name="count_pairs_counter",
        symbol="count_pairs",
        original=(
            "def count_pairs(xs: list[int], target: int):\n"
            "    c = 0\n"
            "    for i in range(len(xs)):\n"
            "        for j in range(i + 1, len(xs)):\n"
            "            if xs[i] + xs[j] == target:\n"
            "                c += 1\n"
            "    return c\n"
        ),
        fast=(
            "def count_pairs(xs: list[int], target: int):\n"
            "    from collections import Counter\n"
            "    cnt = Counter()\n"
            "    c = 0\n"
            "    for x in xs:\n"
            "        c += cnt[target - x]\n"
            "        cnt[x] += 1\n"
            "    return c\n"
        ),
        trap=(
            "def count_pairs(xs: list[int], target: int):\n"
            "    c = 0\n"
            "    for i in range(len(xs)):\n"
            "        for j in range(len(xs)):\n"
            "            if i != j and xs[i] + xs[j] == target:\n"
            "                c += 1\n"
            "    return c\n"  # counts every pair twice
        ),
        nowin=(
            "def count_pairs(xs: list[int], target: int):\n"
            "    c = 0\n"
            "    for i in range(len(xs)):\n"
            "        for j in range(i + 1, len(xs)):\n"
            "            if xs[i] + xs[j] == target:\n"
            "                c += 1\n"
            "    return c\n"
        ),
    ),
    Case(
        name="fib_memoize",
        symbol="fib",
        original=(
            "def fib(n):\n"
            "    if n < 2:\n"
            "        return n\n"
            "    return fib(n - 1) + fib(n - 2)\n"
        ),
        fast=(
            "def fib(n):\n"
            "    from functools import lru_cache\n"
            "    @lru_cache(maxsize=None)\n"
            "    def _f(k):\n"
            "        return k if k < 2 else _f(k - 1) + _f(k - 2)\n"
            "    return _f(n)\n"
        ),
        trap=(
            "def fib(n):\n"
            "    a, b = 0, 1\n"
            "    for _ in range(n):\n"
            "        a, b = b, a + b\n"
            "    return b\n"  # off-by-one: returns fib(n+1)
        ),
        nowin=(
            "def fib(n):\n"
            "    if n < 2:\n"
            "        return n\n"
            "    return fib(n - 1) + fib(n - 2)\n"
        ),
    ),
]
