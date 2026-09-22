"""A NumPy leaf must take the LOCAL placement path, not the asserting one.

``jnp.ndarray`` IS ``jax.Array``, so a guard naming both types names one, and
the NumPy arrays the mesh builders produce fell through to a placement that
asserts the value is bit-identical on every process.  That assert gathers the
whole field onto every process, so per-process memory grows with the process
count.  It ended a ten-million-cell ladder at 192 devices on a 31.6 GiB
allocation, which is how it was found.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from legoesm.parallel import mesh as mesh_mod


def test_the_two_named_types_are_actually_one():
    """The premise of the defect, pinned so it cannot quietly stop being true."""
    assert jnp.ndarray is jax.Array
    assert not isinstance(np.zeros(3), (jax.Array, jnp.ndarray))


def test_a_numpy_leaf_does_not_reach_the_asserting_placement(monkeypatch):
    calls = []

    def boom(leaf, sharding):
        calls.append(type(leaf).__name__)
        raise AssertionError("a NumPy leaf reached the asserting placement: "
                             "under multiple processes this gathers the whole "
                             "field onto every one of them")

    made = []

    def fake_make(shape, sharding, cb):
        made.append(shape)
        return "placed-locally"

    monkeypatch.setattr(mesh_mod.jax, "device_put", boom)
    monkeypatch.setattr(mesh_mod.jax, "make_array_from_callback", fake_make)
    monkeypatch.setattr(mesh_mod.jax, "process_count", lambda: 4)

    out = mesh_mod.multiprocess_safe_device_put(np.zeros((8, 4)), object())
    assert out == "placed-locally"
    assert made == [(8, 4)]
    assert calls == []


def test_a_non_array_leaf_still_delegates(monkeypatch):
    """Scalars and the like have no local path and must keep their behaviour."""
    seen = []
    monkeypatch.setattr(mesh_mod.jax, "device_put",
                        lambda leaf, sharding: seen.append(leaf) or "delegated")
    monkeypatch.setattr(mesh_mod.jax, "process_count", lambda: 4)
    assert mesh_mod.multiprocess_safe_device_put(3.5, object()) == "delegated"
    assert seen == [3.5]
