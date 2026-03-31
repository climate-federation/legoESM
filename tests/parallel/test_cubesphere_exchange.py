"""Tests for the explicit cubed-sphere SPMD halo exchange.

Validates that both the all_gather and ppermute backends produce
identical results to the local pad_halo, and that vector and packed
exchanges are correct.
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np


def _make_6way_mesh():
    """Create a 6-device face-sharded mesh on CPU."""
    devices = jax.devices("cpu")
    if len(devices) < 6:
        pytest.skip("Need at least 6 CPU devices "
                     "(set XLA_FLAGS=--xla_force_host_platform_device_count=6)")
    mesh = jax.sharding.Mesh(
        np.array(devices[:6]).reshape(6),
        axis_names=("face",),
    )
    return mesh


def _shard_on_face(data, mesh):
    """Shard a (6, ...) array on the face axis."""
    P = jax.sharding.PartitionSpec
    spec = P("face", *((None,) * (data.ndim - 1)))
    sharding = jax.sharding.NamedSharding(mesh, spec)
    return jax.device_put(data, sharding)


@pytest.fixture
def mesh_6():
    return _make_6way_mesh()


# =======================================================================
# all_gather backend (default)
# =======================================================================

class TestAllGather3D:
    def test_matches_local(self, mesh_6):
        from legoesm.grids.halo import _pad_halo_local
        from legoesm.parallel.cubesphere_exchange import explicit_pad_halo
        n = 8
        data = jax.random.normal(jax.random.PRNGKey(42), (6, n, n))
        ref = np.array(_pad_halo_local(data))
        result = np.array(explicit_pad_halo(_shard_on_face(data, mesh_6), mesh_6))
        np.testing.assert_allclose(result, ref, rtol=1e-6, atol=1e-10)

    def test_uniform(self, mesh_6):
        from legoesm.parallel.cubesphere_exchange import explicit_pad_halo
        data = jnp.ones((6, 8, 8)) * 7.0
        result = np.array(explicit_pad_halo(_shard_on_face(data, mesh_6), mesh_6))
        np.testing.assert_allclose(result[:, 1:-1, 1:-1], 7.0)
        np.testing.assert_allclose(result[:, 0, 1:-1], 7.0, rtol=1e-6)
        np.testing.assert_allclose(result[:, -1, 1:-1], 7.0, rtol=1e-6)


class TestAllGather4D:
    def test_matches_local(self, mesh_6):
        from legoesm.grids.halo import _pad_halo_local_4d
        from legoesm.parallel.cubesphere_exchange import explicit_pad_halo_4d
        data = jax.random.normal(jax.random.PRNGKey(123), (6, 8, 8, 5))
        ref = np.array(_pad_halo_local_4d(data))
        result = np.array(explicit_pad_halo_4d(_shard_on_face(data, mesh_6), mesh_6))
        np.testing.assert_allclose(result, ref, rtol=1e-6, atol=1e-10)


# =======================================================================
# ppermute backend
# =======================================================================

class TestPpermute3D:
    def test_matches_local(self, mesh_6):
        from legoesm.grids.halo import _pad_halo_local
        from legoesm.parallel.cubesphere_exchange import (
            set_ppermute_default, explicit_pad_halo, _cache,
        )
        _cache.clear()
        set_ppermute_default(True)
        try:
            n = 8
            data = jax.random.normal(jax.random.PRNGKey(42), (6, n, n))
            ref = np.array(_pad_halo_local(data))
            result = np.array(
                explicit_pad_halo(_shard_on_face(data, mesh_6), mesh_6))
            np.testing.assert_allclose(result, ref, rtol=1e-6, atol=1e-10,
                                       err_msg="ppermute 3D differs from local")
        finally:
            set_ppermute_default(False)
            _cache.clear()


class TestPpermute4D:
    def test_matches_local(self, mesh_6):
        from legoesm.grids.halo import _pad_halo_local_4d
        from legoesm.parallel.cubesphere_exchange import (
            set_ppermute_default, explicit_pad_halo_4d, _cache,
        )
        _cache.clear()
        set_ppermute_default(True)
        try:
            data = jax.random.normal(jax.random.PRNGKey(7), (6, 8, 8, 5))
            ref = np.array(_pad_halo_local_4d(data))
            result = np.array(
                explicit_pad_halo_4d(_shard_on_face(data, mesh_6), mesh_6))
            np.testing.assert_allclose(result, ref, rtol=1e-6, atol=1e-10,
                                       err_msg="ppermute 4D differs from local")
        finally:
            set_ppermute_default(False)
            _cache.clear()


# =======================================================================
# Packed multi-field exchange
# =======================================================================

class TestPackedExchange:
    def test_packed_matches_individual(self, mesh_6):
        from legoesm.parallel.cubesphere_exchange import (
            explicit_pad_halo_4d, packed_pad_halo_4d,
        )
        f1 = jax.random.normal(jax.random.PRNGKey(0), (6, 8, 8, 5))
        f2 = jax.random.normal(jax.random.PRNGKey(1), (6, 8, 8, 5))
        f3 = jax.random.normal(jax.random.PRNGKey(2), (6, 8, 8, 3))
        f1s, f2s, f3s = [_shard_on_face(f, mesh_6) for f in (f1, f2, f3)]
        ref1 = np.array(explicit_pad_halo_4d(f1s, mesh_6))
        ref2 = np.array(explicit_pad_halo_4d(f2s, mesh_6))
        ref3 = np.array(explicit_pad_halo_4d(f3s, mesh_6))
        r1, r2, r3 = [np.array(r) for r in packed_pad_halo_4d(f1s, f2s, f3s, mesh=mesh_6)]
        np.testing.assert_allclose(r1, ref1, rtol=1e-6, atol=1e-10)
        np.testing.assert_allclose(r2, ref2, rtol=1e-6, atol=1e-10)
        np.testing.assert_allclose(r3, ref3, rtol=1e-6, atol=1e-10)


# =======================================================================
# Vector (u, v) exchange
# =======================================================================

class TestVectorExchange:
    def test_matches_scalar_reference(self, mesh_6):
        """Vector SPMD exchange must match the local pad_halo_vector_4d."""
        from legoesm.grids.halo import pad_halo_vector_4d, pad_halo_4d
        from legoesm.parallel.cubesphere_exchange import (
            explicit_pad_halo_vector_4d,
        )
        n, nlev = 8, 5
        u = jax.random.normal(jax.random.PRNGKey(10), (6, n, n, nlev))
        v = jax.random.normal(jax.random.PRNGKey(11), (6, n, n, nlev))
        # Synthetic angles (identity rotation for simplicity)
        cos_a = jnp.ones((6, n, n))
        sin_a = jnp.zeros((6, n, n))
        cos_a_p = jnp.ones((6, n + 2, n + 2))
        sin_a_p = jnp.zeros((6, n + 2, n + 2))

        # Reference: local path (rotates → pads each component → rotates back)
        u_ref, v_ref = pad_halo_vector_4d(
            u, v, cos_a, sin_a, cos_a_p, sin_a_p,
        )

        # Explicit SPMD path (packs both components → one exchange)
        us, vs = _shard_on_face(u, mesh_6), _shard_on_face(v, mesh_6)
        cos_as = _shard_on_face(cos_a, mesh_6)
        sin_as = _shard_on_face(sin_a, mesh_6)
        cos_aps = _shard_on_face(cos_a_p, mesh_6)
        sin_aps = _shard_on_face(sin_a_p, mesh_6)

        u_spmd, v_spmd = explicit_pad_halo_vector_4d(
            us, vs, cos_as, sin_as, cos_aps, sin_aps, mesh_6,
        )

        np.testing.assert_allclose(
            np.array(u_spmd), np.array(u_ref), rtol=1e-6, atol=1e-10,
            err_msg="vector u mismatch")
        np.testing.assert_allclose(
            np.array(v_spmd), np.array(v_ref), rtol=1e-6, atol=1e-10,
            err_msg="vector v mismatch")


# =======================================================================
# Backend activation / dispatch integration
# =======================================================================

class TestBackendActivation:
    def test_activate_deactivate(self, mesh_6):
        from legoesm.grids.halo import get_halo_backend
        from legoesm.parallel.cubesphere_exchange import (
            activate_spmd_halo_backend, deactivate_spmd_halo_backend,
        )
        assert get_halo_backend() in ("local", "spmd")
        activate_spmd_halo_backend(mesh_6)
        assert get_halo_backend() == "spmd"
        deactivate_spmd_halo_backend()
        assert get_halo_backend() == "local"

    def test_pad_halo_dispatches_to_spmd(self, mesh_6):
        from legoesm.grids.halo import pad_halo, _pad_halo_local
        from legoesm.parallel.cubesphere_exchange import (
            activate_spmd_halo_backend, deactivate_spmd_halo_backend,
        )
        n = 8
        data = jax.random.normal(jax.random.PRNGKey(99), (6, n, n))
        ref = np.array(_pad_halo_local(data))
        activate_spmd_halo_backend(mesh_6)
        try:
            result = np.array(pad_halo(_shard_on_face(data, mesh_6)))
            np.testing.assert_allclose(result, ref, rtol=1e-6, atol=1e-10)
        finally:
            deactivate_spmd_halo_backend()
