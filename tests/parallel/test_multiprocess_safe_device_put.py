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


def test_a_masked_array_is_refused_not_silently_unmasked(monkeypatch):
    """Placing a masked array raises, and JAX says so itself on one process.
    The local path must not turn that rejection into silently wrong numbers
    by stripping the mask and placing the fill values as if they were data."""
    import pytest

    masked = np.ma.array([1.0, 99.0], mask=[False, True])

    # One process: JAX's own refusal, unchanged.
    with pytest.raises(ValueError, match="masked"):
        mesh_mod.multiprocess_safe_device_put(masked, object())

    # Several processes: the local path must refuse it too, rather than
    # converting first and placing 99.0 as if it were data.
    monkeypatch.setattr(mesh_mod.jax, "process_count", lambda: 4)
    monkeypatch.setattr(mesh_mod.jax, "make_array_from_callback",
                        lambda *a, **k: pytest.fail(
                            "a masked array reached the placement: its mask "
                            "was stripped and the fill value placed as data"))
    with pytest.raises(ValueError, match="masked"):
        mesh_mod.multiprocess_safe_device_put(masked, object())


def test_the_callback_returns_the_right_data_for_each_shard():
    """The routing test mocks the placement away, so it would pass even if the
    callback handed back the wrong values.  This one runs the real placement
    on real devices and checks the placed array equals the source."""
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    devices = jax.devices()
    n = len(devices)
    src = np.arange(8 * n, dtype=np.float32).reshape(8, n)
    m = Mesh(np.asarray(devices).reshape(n), ("d",))
    for spec in (P(None, "d"), P()):
        out = mesh_mod.multiprocess_safe_device_put(src, NamedSharding(m, spec))
        np.testing.assert_array_equal(np.asarray(out), src)
