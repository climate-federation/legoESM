"""Tests for the explicit cubed-sphere SPMD halo exchange.

Validates that the shard_map + all_gather exchange produces identical
results to the local pad_halo on a 6-device CPU mesh.
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np


def _make_6way_mesh():
    """Create a 6-device face-sharded mesh on CPU."""
    devices = jax.devices("cpu")
    if len(devices) < 6:
        pytest.skip("Need at least 6 CPU devices (set XLA_FLAGS=--xla_force_host_platform_device_count=6)")
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


class TestExplicitExchange3D:
    """Test 3D (6, n, n) halo exchange."""

    def test_matches_local_pad_halo(self, mesh_6):
        """Explicit exchange must match the local pad_halo reference."""
        from legoesm.grids.halo import _pad_halo_local
        from legoesm.parallel.cubesphere_exchange import explicit_pad_halo

        n = 8
        key = jax.random.PRNGKey(42)
        data = jax.random.normal(key, (6, n, n))

        # Reference: local (single-device) pad_halo
        ref = _pad_halo_local(data)

        # Explicit: shard data across 6 devices, exchange, gather back
        data_sharded = _shard_on_face(data, mesh_6)
        result_sharded = explicit_pad_halo(data_sharded, mesh_6)

        # Gather result to host for comparison
        result = np.array(result_sharded)
        ref_np = np.array(ref)

        np.testing.assert_allclose(result, ref_np, rtol=1e-6, atol=1e-10,
                                   err_msg="3D explicit exchange differs from local reference")

    def test_uniform_field(self, mesh_6):
        """Uniform field should have uniform halos."""
        from legoesm.parallel.cubesphere_exchange import explicit_pad_halo

        n = 8
        data = jnp.ones((6, n, n)) * 7.0
        data_sharded = _shard_on_face(data, mesh_6)
        result = np.array(explicit_pad_halo(data_sharded, mesh_6))

        # Interior should be 7.0
        np.testing.assert_allclose(result[:, 1:-1, 1:-1], 7.0)
        # Edge halos should also be 7.0 (uniform field)
        np.testing.assert_allclose(result[:, 0, 1:-1], 7.0, rtol=1e-6)
        np.testing.assert_allclose(result[:, -1, 1:-1], 7.0, rtol=1e-6)
        np.testing.assert_allclose(result[:, 1:-1, 0], 7.0, rtol=1e-6)
        np.testing.assert_allclose(result[:, 1:-1, -1], 7.0, rtol=1e-6)


class TestExplicitExchange4D:
    """Test 4D (6, n, n, nlev) halo exchange."""

    def test_matches_local_pad_halo_4d(self, mesh_6):
        """Explicit 4D exchange must match the local pad_halo_4d reference."""
        from legoesm.grids.halo import _pad_halo_local_4d
        from legoesm.parallel.cubesphere_exchange import explicit_pad_halo_4d

        n, nlev = 8, 5
        key = jax.random.PRNGKey(123)
        data = jax.random.normal(key, (6, n, n, nlev))

        ref = _pad_halo_local_4d(data)
        data_sharded = _shard_on_face(data, mesh_6)
        result = np.array(explicit_pad_halo_4d(data_sharded, mesh_6))
        ref_np = np.array(ref)

        np.testing.assert_allclose(result, ref_np, rtol=1e-6, atol=1e-10,
                                   err_msg="4D explicit exchange differs from local reference")


class TestPackedExchange:
    """Test packed multi-field exchange."""

    def test_packed_matches_individual(self, mesh_6):
        """Packed exchange of 3 fields must match individual exchanges."""
        from legoesm.parallel.cubesphere_exchange import (
            explicit_pad_halo_4d, packed_pad_halo_4d,
        )

        n, nlev = 8, 5
        key = jax.random.PRNGKey(0)
        f1 = jax.random.normal(key, (6, n, n, nlev))
        f2 = jax.random.normal(jax.random.PRNGKey(1), (6, n, n, nlev))
        f3 = jax.random.normal(jax.random.PRNGKey(2), (6, n, n, 3))

        f1s = _shard_on_face(f1, mesh_6)
        f2s = _shard_on_face(f2, mesh_6)
        f3s = _shard_on_face(f3, mesh_6)

        # Individual exchanges
        ref1 = np.array(explicit_pad_halo_4d(f1s, mesh_6))
        ref2 = np.array(explicit_pad_halo_4d(f2s, mesh_6))
        ref3 = np.array(explicit_pad_halo_4d(f3s, mesh_6))

        # Packed exchange
        results = packed_pad_halo_4d(f1s, f2s, f3s, mesh=mesh_6)
        r1, r2, r3 = [np.array(r) for r in results]

        np.testing.assert_allclose(r1, ref1, rtol=1e-6, atol=1e-10)
        np.testing.assert_allclose(r2, ref2, rtol=1e-6, atol=1e-10)
        np.testing.assert_allclose(r3, ref3, rtol=1e-6, atol=1e-10)


class TestBackendActivation:
    """Test that the SPMD backend integrates with the global pad_halo dispatch."""

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
        """After activation, pad_halo must use the explicit path."""
        from legoesm.grids.halo import pad_halo, _pad_halo_local
        from legoesm.parallel.cubesphere_exchange import (
            activate_spmd_halo_backend, deactivate_spmd_halo_backend,
        )

        n = 8
        data = jax.random.normal(jax.random.PRNGKey(99), (6, n, n))
        ref = np.array(_pad_halo_local(data))

        activate_spmd_halo_backend(mesh_6)
        try:
            data_sharded = _shard_on_face(data, mesh_6)
            result = np.array(pad_halo(data_sharded))
            np.testing.assert_allclose(result, ref, rtol=1e-6, atol=1e-10)
        finally:
            deactivate_spmd_halo_backend()
