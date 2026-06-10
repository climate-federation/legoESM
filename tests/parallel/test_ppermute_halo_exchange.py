"""End-to-end SPMD cubed-sphere halo exchange (ppermute + all_gather) vs serial.

The ppermute halo backend (``cubesphere_exchange``, 4 rounds of
``jax.lax.ppermute`` inside ``shard_map``) is the JAX-0.10-native, mpi4jax-FREE
distributed halo — it runs on the main venv across multiple devices, unlike the
mpi4jax path (which needs JAX <= 0.9). This test verifies on a real 6-device mesh
that the ppermute exchange BIT-MATCHES the serial/local halo (and so does
all_gather), and that the requested kernel is actually routed (HLO check), for
scalar 3D, 4D and vector fields. Hardened per codex adversarial review.

Run with 6 host devices:
    XLA_FLAGS="--xla_force_host_platform_device_count=6" \\
        JAX_ENABLE_X64=1 .venv/bin/python -m pytest tests/parallel/test_ppermute_halo_exchange.py
(skips otherwise — a shared session that already pinned 1 device.)
"""
from __future__ import annotations

import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=6")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

jax.config.update("jax_enable_x64", True)

if jax.device_count() < 6:
    pytest.skip(
        "needs 6 devices (XLA_FLAGS=--xla_force_host_platform_device_count=6)",
        allow_module_level=True,
    )

from jax.sharding import Mesh  # noqa: E402

from legoesm.grids import halo as halo_mod  # noqa: E402
from legoesm.grids.halo import (  # noqa: E402
    pad_halo, pad_halo_4d, pad_halo_vector_4d, set_halo_backend,
)
from legoesm.parallel import cubesphere_exchange as cx  # noqa: E402


def _mesh():
    return Mesh(np.array(jax.devices()[:6]).reshape(6), ("face",))


@pytest.fixture(autouse=True)
def _restore_globals():
    """Snapshot+restore ALL process-global halo state after each test (codex:
    globals are test-fragile — restore _use_ppermute, the backend string, both
    _spmd_mesh refs and the mpi topology, and clear the kernel cache)."""
    snap = (cx._use_ppermute, cx._spmd_mesh, halo_mod._halo_backend,
            halo_mod._spmd_mesh, halo_mod._mpi_topology,
            halo_mod.get_corner_fill_mode())
    yield
    (cx._use_ppermute, cx._spmd_mesh, halo_mod._halo_backend,
     halo_mod._spmd_mesh, halo_mod._mpi_topology) = snap[:5]
    halo_mod.set_corner_fill_mode(snap[5])
    cx._cache.clear()


def _serial(fn, *args, **kw):
    set_halo_backend("local")
    return fn(*args, halo=1, **kw)


@pytest.mark.parametrize("use_ppermute", [True, False])
def test_spmd_scalar_halo_matches_serial(use_ppermute):
    n = 8
    data = jax.random.normal(jax.random.PRNGKey(0), (6, n, n))
    ref = _serial(pad_halo, data)
    cx.set_ppermute_default(use_ppermute)
    got = cx.explicit_pad_halo(data, _mesh(), halo=1)
    assert got.shape == ref.shape == (6, n + 2, n + 2)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(ref))  # bit-exact


@pytest.mark.parametrize("use_ppermute", [True, False])
def test_spmd_4d_halo_matches_serial(use_ppermute):
    n, nlev = 6, 4
    data = jax.random.normal(jax.random.PRNGKey(1), (6, n, n, nlev))
    ref = _serial(pad_halo_4d, data)
    cx.set_ppermute_default(use_ppermute)
    got = cx.explicit_pad_halo_4d(data, _mesh(), halo=1)
    assert got.shape == ref.shape == (6, n + 2, n + 2, nlev)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(ref))


def test_routing_emits_the_requested_collective():
    """The HLO must actually contain collective-permute for ppermute and
    all-gather for all_gather (codex: the value test alone could pass even if it
    silently fell back)."""
    n = 8
    data = jax.random.normal(jax.random.PRNGKey(5), (6, n, n))
    mesh = _mesh()
    cx.set_ppermute_default(True)
    hlo_pp = jax.jit(lambda d: cx.explicit_pad_halo(d, mesh, halo=1)).lower(
        data).compile().as_text()          # COMPILED HLO surfaces the collective
    assert "collective-permute" in hlo_pp and "all-gather" not in hlo_pp
    cx._cache.clear()
    cx.set_ppermute_default(False)
    hlo_ag = jax.jit(lambda d: cx.explicit_pad_halo(d, mesh, halo=1)).lower(
        data).compile().as_text()
    assert "all-gather" in hlo_ag


def test_spmd_vector_halo_matches_serial():
    """Vector ppermute path (explicit_pad_halo_vector_4d) — identity rotation
    (cos=1, sin=0) so u/v exchange like scalars across edges."""
    n, nlev = 6, 3
    u = jax.random.normal(jax.random.PRNGKey(2), (6, n, n, nlev))
    v = jax.random.normal(jax.random.PRNGKey(3), (6, n, n, nlev))
    cos = jnp.ones((6, n, n)); sin = jnp.zeros((6, n, n))
    cos_p = jnp.ones((6, n + 2, n + 2)); sin_p = jnp.zeros((6, n + 2, n + 2))
    set_halo_backend("local")
    ru, rv = pad_halo_vector_4d(u, v, cos, sin, cos_p, sin_p, halo=1)
    cx.set_ppermute_default(True)
    gu, gv = cx.explicit_pad_halo_vector_4d(
        u, v, cos, sin, cos_p, sin_p, _mesh(), halo=1)
    np.testing.assert_array_equal(np.asarray(gu), np.asarray(ru))
    np.testing.assert_array_equal(np.asarray(gv), np.asarray(rv))


def test_halo2_falls_back_to_allgather_and_matches():
    """halo=2 is all_gather-only; with the ppermute default ON it must still
    fall back and bit-match serial (codex: lock the fallback)."""
    n = 8
    data = jax.random.normal(jax.random.PRNGKey(4), (6, n, n))
    set_halo_backend("local")
    ref = pad_halo(data, halo=2)
    cx.set_ppermute_default(True)
    got = cx.explicit_pad_halo(data, _mesh(), halo=2)
    assert got.shape == ref.shape == (6, n + 4, n + 4)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(ref))


def test_ppermute_schedule_covers_24_adjacencies_once():
    """Invariant the ppermute correctness relies on: the 4 rounds cover all 24
    directed face adjacencies exactly once, and each round is a permutation
    (every face appears once as src and once as dst)."""
    perms = cx._PPERMUTE_PERMS
    assert len(perms) == 4
    pairs = [(s, d) for rnd in perms for (s, d) in rnd]
    assert len(pairs) == 24 and len(set(pairs)) == 24      # all distinct
    assert all(s != d for (s, d) in pairs)                 # no self-moves
    for rnd in perms:                                      # each round = perm
        assert sorted(s for s, _ in rnd) == list(range(6))
        assert sorted(d for _, d in rnd) == list(range(6))


def test_activate_rejects_non_avg_corner_fill():
    """SPMD hard-codes avg corners; activating it under a non-avg corner mode
    must raise rather than silently mismatch the serial corners (codex)."""
    mesh_before, backend_before = cx.get_spmd_mesh(), halo_mod.get_halo_backend()
    halo_mod.set_corner_fill_mode("fv3_agrid_xdir")
    try:
        with pytest.raises(NotImplementedError, match="corner_fill_mode"):
            cx.activate_spmd_halo_backend(_mesh(), n=8, nlev=1)
        # No-leak: the guard runs FIRST, so a raise leaves globals untouched.
        assert cx.get_spmd_mesh() is mesh_before
        assert halo_mod.get_halo_backend() == backend_before != "spmd"
    finally:
        halo_mod.set_corner_fill_mode("avg")


def teardown_module(_):
    cx.set_ppermute_default(False)
    cx._cache.clear()
    set_halo_backend("local")
