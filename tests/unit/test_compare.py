"""Equality oracle tests — float tolerance, exceptions-as-behavior, ordering."""

from __future__ import annotations

from optiproof.adapters.base import BehaviorObservation
from optiproof.verify.compare import compare_observations, values_equal


def obs(value=None, ok=True, exc=None, stdout=""):
    return BehaviorObservation(
        ok=ok, value=value, value_repr=repr(value), pickled=True, stdout=stdout, exception=exc
    )


def test_equal_numbers_and_float_tolerance():
    assert compare_observations(obs(3), obs(3))[0]
    assert values_equal(0.1 + 0.2, 0.3)            # reassociated FP within tolerance
    assert not values_equal(1.0, 1.1)


def test_list_order_is_behavior():
    assert compare_observations(obs([1, 2, 3]), obs([1, 2, 3]))[0]
    assert not compare_observations(obs([1, 2, 3]), obs([3, 2, 1]))[0]


def test_sets_and_dicts_order_insensitive():
    assert values_equal({1, 2, 3}, {3, 2, 1})
    assert values_equal({"a": 1, "b": 2}, {"b": 2, "a": 1})
    assert not values_equal({"a": 1}, {"a": 2})


def test_exception_is_behavior():
    assert compare_observations(obs(ok=False, exc="ValueError"), obs(ok=False, exc="ValueError"))[0]
    assert not compare_observations(obs(ok=False, exc="ValueError"), obs(ok=False, exc="KeyError"))[0]
    # one raised, the other returned -> not equal
    assert not compare_observations(obs(ok=False, exc="ValueError"), obs(5))[0]


def test_stdout_is_behavior():
    assert not compare_observations(obs(1, stdout="hi\n"), obs(1, stdout=""))[0]
