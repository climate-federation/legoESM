"""Central deterministic RNG (Stage A1).

split_keys must be deterministic, order-independent, and per-name distinct so a
run is reproducible from its single master seed regardless of how many consumers
draw keys or in what order.
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm.runtime.rng import split_key, split_keys


def _eq(a, b) -> bool:
    return bool(jnp.array_equal(a, b))


def test_deterministic_for_same_seed_and_name() -> None:
    assert _eq(split_keys(0, ["a"])["a"], split_keys(0, ["a"])["a"])
    assert _eq(split_key(123, "x"), split_key(123, "x"))


def test_order_independent() -> None:
    """Adding/reordering consumers must not perturb an existing consumer's key."""
    a_alone = split_keys(7, ["a"])["a"]
    a_with_b = split_keys(7, ["a", "b"])["a"]
    a_reordered = split_keys(7, ["b", "a"])["a"]
    assert _eq(a_alone, a_with_b)
    assert _eq(a_alone, a_reordered)


def test_distinct_names_give_distinct_keys() -> None:
    keys = split_keys(0, ["a", "b", "c"])
    assert not _eq(keys["a"], keys["b"])
    assert not _eq(keys["a"], keys["c"])
    assert not _eq(keys["b"], keys["c"])


def test_distinct_seeds_give_distinct_keys() -> None:
    assert not _eq(split_key(0, "a"), split_key(1, "a"))


def test_returns_all_requested_names() -> None:
    names = ["ensemble_ic", "sppt", "ml_init"]
    keys = split_keys(42, names)
    assert set(keys) == set(names)
