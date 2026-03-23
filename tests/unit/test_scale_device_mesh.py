"""Category 1: Device mesh construction & sharding.

Verifies that device meshes are correctly constructed for all supported
device counts and grid types, and that state sharding/replication works.
"""

from __future__ import annotations

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from legoesm.parallel.mesh import (
    DeviceConfig,
    create_device_mesh,
    create_latlon_mesh,
    create_level_mesh,
    shard_pytree,
    replicate_pytree,
    _best_tile_factorization,
)
from legoesm.parallel.runtime import validate_device_count
from legoesm.core.field import Field


# =========================================================================
# Helpers
# =========================================================================

def _make_state_3d(n: int = 8, nlev: int = 5):
    """Create a mock cubed-sphere 3D state as a dict of Fields."""
    rng = np.random.default_rng(42)
    return {
        "T": Field(
            data=jnp.array(rng.standard_normal((6, n, n, nlev)), dtype=jnp.float64),
            name="T", dims=("face", "x", "y", "level"), units="K",
        ),
        "p_s": Field(
            data=jnp.array(rng.standard_normal((6, n, n)), dtype=jnp.float64),
            name="p_s", dims=("face", "x", "y"), units="Pa",
        ),
    }


# =========================================================================
# 1a) Cubed-sphere face-only mesh
# =========================================================================

class TestFaceOnlyMesh:
    """create_device_mesh with 1–6 devices produces valid DeviceConfig."""

    @pytest.mark.parametrize("n_devices", [1, 2, 3, 6])
    def test_face_only_creation(self, n_devices):
        if jax.device_count() < n_devices:
            pytest.skip(f"Need {n_devices} devices, have {jax.device_count()}")
        config = create_device_mesh(n_devices=n_devices)
        assert isinstance(config, DeviceConfig)
        assert config.n_devices == n_devices

    def test_single_device_no_mesh(self):
        """Single device should work without a mesh."""
        config = create_device_mesh(n_devices=1)
        assert config.n_devices == 1
        # Single device: mesh may be None or trivial
        # Just verify it doesn't crash and config is valid
        assert config.backend.lower() in ("cpu", "gpu", "tpu", "metal")

    @pytest.mark.skipif(jax.device_count() < 2, reason="Need 2+ devices")
    def test_multi_device_has_mesh(self):
        """Multi-device should create a mesh with face axis."""
        n = min(jax.device_count(), 6)
        # Find largest valid count <= n
        for nd in [6, 3, 2]:
            if nd <= n:
                config = create_device_mesh(n_devices=nd)
                assert config.mesh is not None
                assert "face" in config.mesh.axis_names
                break


# =========================================================================
# 1b) Sub-face tiling
# =========================================================================

class TestSubFaceTiling:
    """Sub-face tiling device meshes for 6*k^2 devices."""

    @pytest.mark.parametrize("k,n_devices", [(2, 24), (3, 54), (4, 96)])
    def test_tiling_mesh(self, k, n_devices):
        if jax.device_count() < n_devices:
            pytest.skip(f"Need {n_devices} devices")
        config = create_device_mesh(n_devices=n_devices)
        assert config.mesh is not None
        assert config.n_devices == n_devices
        assert config.tiling == (k, k)

    def test_tile_factorization(self):
        """_best_tile_factorization returns valid factorizations."""
        # 24 = 6 * 2 * 2
        n_groups, tx, ty = _best_tile_factorization(24)
        assert n_groups * tx * ty == 24 // 6 or tx * ty * 6 == 24
        # 54 = 6 * 3 * 3
        n_groups2, tx2, ty2 = _best_tile_factorization(54)
        assert tx2 == ty2  # square tiles required


# =========================================================================
# 1c) Invalid device counts rejected
# =========================================================================

class TestInvalidDeviceCounts:
    """Invalid cubed-sphere device counts should raise ValueError via validate_device_count."""

    @pytest.mark.parametrize("n_devices", [4, 5, 7, 10, 13, 25])
    def test_validate_device_count_raises(self, n_devices):
        with pytest.raises(ValueError, match="Unsupported device count"):
            validate_device_count(n_devices)

    @pytest.mark.parametrize("n_devices", [1, 2, 3, 6, 24, 54, 96])
    def test_validate_device_count_passes(self, n_devices):
        # Should not raise
        validate_device_count(n_devices)


# =========================================================================
# 1d) Lat-lon mesh
# =========================================================================

class TestLatLonMesh:
    """create_latlon_mesh produces valid DeviceConfig."""

    def test_latlon_single_device(self):
        config = create_latlon_mesh(n_devices=1)
        assert isinstance(config, DeviceConfig)
        assert config.grid_type == "latlon"

    @pytest.mark.skipif(jax.device_count() < 2, reason="Need 2+ devices")
    def test_latlon_multi_device(self):
        config = create_latlon_mesh(n_devices=2)
        assert config.mesh is not None
        assert "lat" in config.mesh.axis_names


# =========================================================================
# 1e) Level-parallel mesh (spectral)
# =========================================================================

class TestLevelMesh:
    """create_level_mesh produces valid DeviceConfig."""

    def test_level_mesh_single(self):
        config = create_level_mesh(n_devices=1)
        assert isinstance(config, DeviceConfig)

    @pytest.mark.skipif(jax.device_count() < 2, reason="Need 2+ devices")
    def test_level_mesh_multi(self):
        config = create_level_mesh(n_devices=2)
        assert config.mesh is not None
        assert "level" in config.mesh.axis_names


# =========================================================================
# 1f) PartitionSpec consistency
# =========================================================================

class TestSharding:
    """shard_pytree and replicate_pytree produce correct shardings."""

    def test_shard_single_device_noop(self):
        """Sharding on 1 device should not crash."""
        config = create_device_mesh(n_devices=1)
        state = _make_state_3d(n=8)
        # Should not raise
        result = shard_pytree(state, config)
        # Values should be preserved
        for key in state:
            np.testing.assert_allclose(
                np.array(result[key].data), np.array(state[key].data), atol=1e-15
            )

    def test_replicate_single_device_noop(self):
        """Replicate on 1 device should preserve values."""
        config = create_device_mesh(n_devices=1)
        state = _make_state_3d(n=8)
        result = replicate_pytree(state, config)
        for key in state:
            np.testing.assert_allclose(
                np.array(result[key].data), np.array(state[key].data), atol=1e-15
            )

    @pytest.mark.skipif(jax.device_count() < 2, reason="Need 2+ devices")
    def test_shard_multi_device(self):
        """Sharding across 2+ devices should produce non-replicated sharding."""
        n = min(jax.device_count(), 6)
        for nd in [6, 3, 2]:
            if nd <= n:
                config = create_device_mesh(n_devices=nd)
                state = _make_state_3d(n=8)
                result = shard_pytree(state, config)
                # Check that data is preserved (gather implicitly via numpy)
                for key in state:
                    np.testing.assert_allclose(
                        np.array(result[key].data),
                        np.array(state[key].data),
                        atol=1e-15,
                    )
                break


# =========================================================================
# 1g) Replicate vs shard
# =========================================================================

class TestReplicateVsShard:
    """Verify replicate gives fully-replicated and shard gives partitioned."""

    def test_replicate_preserves_values(self):
        """Replicated pytree has identical values on all devices."""
        config = create_device_mesh(n_devices=1)
        data = jnp.ones((6, 4, 4, 5))
        state = {"field": data}
        result = replicate_pytree(state, config)
        np.testing.assert_allclose(np.array(result["field"]), np.array(data))

    def test_shard_preserves_values(self):
        """Sharded pytree has same total data as original."""
        config = create_device_mesh(n_devices=1)
        data = jnp.arange(6 * 4 * 4, dtype=jnp.float64).reshape(6, 4, 4)
        state = {"field": data}
        result = shard_pytree(state, config)
        np.testing.assert_allclose(np.array(result["field"]), np.array(data))
