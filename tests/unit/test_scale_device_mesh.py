"""Category 1: Device mesh construction & sharding tests.

Tests create_device_mesh, create_latlon_mesh, create_level_mesh,
shard_pytree, replicate_pytree, and DeviceConfig for various
device counts and grid types.
"""

from __future__ import annotations

import pytest
import jax
import jax.numpy as jnp

from legoesm.parallel.mesh import (
    create_device_mesh,
    create_latlon_mesh,
    create_level_mesh,
    shard_pytree,
    replicate_pytree,
    DeviceConfig,
    _best_tile_factorization,
)


# ---------------------------------------------------------------------------
# _best_tile_factorization
# ---------------------------------------------------------------------------

class TestBestTileFactorization:
    """Unit tests for _best_tile_factorization."""

    @pytest.mark.parametrize("n,expected", [
        (1, (1, 1, 1)),
        (2, (2, 1, 1)),
        (3, (3, 1, 1)),
        (6, (6, 1, 1)),
    ])
    def test_face_only_counts(self, n, expected):
        assert _best_tile_factorization(n) == expected

    @pytest.mark.parametrize("n", [4, 5])
    def test_invalid_small_counts(self, n):
        with pytest.raises(ValueError, match="divide 6"):
            _best_tile_factorization(n)

    def test_sub_face_tiling_24(self):
        # 24 = 6 * 4 = 6 * 2^2
        n_face, tx, ty = _best_tile_factorization(24)
        assert n_face == 6
        assert tx == 2
        assert ty == 2

    def test_sub_face_tiling_54(self):
        # 54 = 6 * 9 = 6 * 3^2
        n_face, tx, ty = _best_tile_factorization(54)
        assert n_face == 6
        assert tx == 3
        assert ty == 3

    def test_non_multiple_of_6_gt6(self):
        with pytest.raises(ValueError):
            _best_tile_factorization(7)

    def test_non_square_tiles(self):
        # 12 = 6 * 2 -> tiles_per_face=2, sqrt(2) not integer
        with pytest.raises(ValueError, match="perfect square"):
            _best_tile_factorization(12)


# ---------------------------------------------------------------------------
# create_device_mesh
# ---------------------------------------------------------------------------

class TestCreateDeviceMesh:
    """Test cubed-sphere device mesh creation."""

    def test_single_device(self):
        config = create_device_mesh(n_devices=1)
        assert isinstance(config, DeviceConfig)
        assert config.n_devices == 1
        assert config.mesh is None
        assert config.face_sharding is None
        assert config.replicated_sharding is None
        assert config.tiling == (1, 1)
        assert config.grid_type == "cubed_sphere"

    def test_backend_field(self):
        config = create_device_mesh(n_devices=1)
        assert isinstance(config.backend, str)
        assert len(config.backend) > 0

    def test_is_distributed_false(self):
        config = create_device_mesh(n_devices=1)
        assert config.is_distributed is False

    def test_auto_device_count(self):
        config = create_device_mesh(n_devices="auto")
        assert config.n_devices >= 1

    def test_explicit_devices(self):
        devs = jax.devices("cpu")[:1]
        config = create_device_mesh(devices=devs)
        assert config.n_devices == 1

    def test_empty_devices_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            create_device_mesh(devices=[])

    def test_negative_n_devices_raises(self):
        with pytest.raises(ValueError, match="n_devices must be >= 1"):
            create_device_mesh(n_devices=0)


# ---------------------------------------------------------------------------
# create_latlon_mesh
# ---------------------------------------------------------------------------

class TestCreateLatlonMesh:
    def test_single_device(self):
        config = create_latlon_mesh(n_devices=1)
        assert config.n_devices == 1
        assert config.mesh is None
        assert config.grid_type == "latlon"


# ---------------------------------------------------------------------------
# create_level_mesh
# ---------------------------------------------------------------------------

class TestCreateLevelMesh:
    def test_single_device(self):
        config = create_level_mesh(n_devices=1)
        assert config.n_devices == 1
        assert config.mesh is None
        assert config.grid_type == "spectral"


# ---------------------------------------------------------------------------
# shard_pytree / replicate_pytree (single-device)
# ---------------------------------------------------------------------------

class TestShardReplicate:
    """Test pytree sharding utilities in single-device mode."""

    def test_shard_single_device_is_identity(self):
        config = create_device_mesh(n_devices=1)
        data = {"a": jnp.ones((6, 4, 4)), "b": jnp.zeros(3)}
        out = shard_pytree(data, config)
        assert jnp.array_equal(out["a"], data["a"])
        assert jnp.array_equal(out["b"], data["b"])

    def test_replicate_single_device_is_identity(self):
        config = create_device_mesh(n_devices=1)
        data = {"x": jnp.arange(12).reshape(3, 4)}
        out = replicate_pytree(data, config)
        assert jnp.array_equal(out["x"], data["x"])

    def test_shard_cubed_sphere_face_detection(self):
        """Cubed-sphere sharding recognises leading dimension of 6."""
        config = create_device_mesh(n_devices=1)
        face_arr = jnp.ones((6, 4, 4))
        non_face = jnp.ones((3, 4, 4))
        tree = {"face": face_arr, "other": non_face}
        out = shard_pytree(tree, config)
        # Single device -> same arrays
        assert out["face"].shape == (6, 4, 4)
        assert out["other"].shape == (3, 4, 4)

    def test_shard_latlon_single(self):
        config = create_latlon_mesh(n_devices=1)
        data = {"T": jnp.ones((32, 64))}
        out = shard_pytree(data, config)
        assert jnp.array_equal(out["T"], data["T"])

    def test_shard_spectral_single(self):
        config = create_level_mesh(n_devices=1)
        data = {"coeffs": jnp.ones((100, 20), dtype=jnp.complex128)}
        out = shard_pytree(data, config)
        assert jnp.array_equal(out["coeffs"], data["coeffs"])

    def test_non_array_leaves_preserved(self):
        """Non-array leaves pass through unchanged."""
        config = create_device_mesh(n_devices=1)
        tree = {"arr": jnp.ones(3), "label": "hello"}
        out = shard_pytree(tree, config)
        assert out["label"] == "hello"

    def test_replicate_non_array_preserved(self):
        config = create_device_mesh(n_devices=1)
        tree = {"arr": jnp.ones(3), "count": 42}
        out = replicate_pytree(tree, config)
        assert out["count"] == 42


# ---------------------------------------------------------------------------
# DeviceConfig NamedTuple fields
# ---------------------------------------------------------------------------

class TestDeviceConfigNamedTuple:
    def test_is_named_tuple(self):
        config = create_device_mesh(n_devices=1)
        assert hasattr(config, "_fields")
        assert "mesh" in config._fields
        assert "n_devices" in config._fields

    def test_fields_are_accessible(self):
        config = create_device_mesh(n_devices=1)
        _ = config.mesh
        _ = config.face_sharding
        _ = config.replicated_sharding
        _ = config.n_devices
        _ = config.backend
        _ = config.is_distributed
        _ = config.tiling
        _ = config.grid_type


# ---------------------------------------------------------------------------
# Issue #273: level-parallel cubed-sphere fallback for awkward device counts
# ---------------------------------------------------------------------------

class TestCubedSphereLevelFallback:
    """Issue #273: 4-GPU A100 nodes fail face-sharding divisibility
    (4 ∉ {1, 2, 3, 6, 24, ...}).  Level-parallel fallback keeps all 4
    devices busy by replicating the horizontal stencil and sharding
    the level axis."""

    def test_create_cubed_sphere_level_mesh_single_device(self):
        from legoesm.parallel.mesh import create_cubed_sphere_level_mesh
        cfg = create_cubed_sphere_level_mesh(n_devices=1)
        assert cfg.n_devices == 1
        assert cfg.grid_type == "cubed_sphere_level"
        assert cfg.mesh is None

    def test_validate_accepts_4_with_level_fallback(self):
        from legoesm.parallel.runtime import validate_device_count
        # Default validation rejects 4.
        with pytest.raises(ValueError, match="Unsupported device count 4"):
            validate_device_count(4, grid_type="cubed_sphere")
        # With level fallback enabled, accepted.
        validate_device_count(
            4, grid_type="cubed_sphere", allow_level_fallback=True,
        )

    def test_validate_accepts_arbitrary_counts_with_level_fallback(self):
        from legoesm.parallel.runtime import validate_device_count
        # Pick a handful of values that fail face-sharding divisibility.
        for n in (4, 5, 7, 9, 11, 100):
            validate_device_count(
                n, grid_type="cubed_sphere", allow_level_fallback=True,
            )

    def test_create_device_mesh_falls_back_when_allowed(self):
        """When ``allow_level_fallback=True`` and the requested device
        count fails face-sharding, the factory routes to the level-
        parallel mesh instead of raising ValueError."""
        # ``n_devices=4`` exceeds the single test device, but the
        # fallback path is exercised before any device-clamp logic.
        # We assert the path is taken by checking that the returned
        # config carries the level-mesh ``grid_type`` tag.
        from legoesm.parallel.mesh import create_device_mesh
        cfg = create_device_mesh(
            n_devices=4, allow_level_fallback=True,
        )
        # On a single-device host the level mesh degenerates to
        # ``n_devices=1``, which is still tagged as the level path.
        assert cfg.grid_type == "cubed_sphere_level"

    def test_create_device_mesh_without_fallback_still_clamps(self):
        """Backward compat: existing call sites that do NOT pass
        ``allow_level_fallback`` get the legacy behavior — either
        clamp-to-available when the requested count exceeds the host
        device count (single-device host), or raise ``ValueError``
        when the explicit count fails face-sharding divisibility
        (multi-device host).  This PR keeps the new fallback strictly
        opt-in."""
        from legoesm.parallel.mesh import create_device_mesh
        if len(jax.devices()) <= 3:
            # Single-device test host: 4 > 1 → clamp path → n_dev=1.
            cfg = create_device_mesh(n_devices=4)
            assert cfg.grid_type == "cubed_sphere"
            assert cfg.n_devices == 1
        else:
            # Multi-device host: 4 ≤ 8 (no clamp), 4 fails
            # face-divisibility → raise per Codex review contract.
            with pytest.raises(ValueError, match="divide 6"):
                create_device_mesh(n_devices=4)
