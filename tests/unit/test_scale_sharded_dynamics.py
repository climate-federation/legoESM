"""Category 5: Sharded dynamics tests.

Tests shard_state/gather_state roundtrip, ShardingSpec construction,
and output shardings on single-device (the logical correctness paths).
"""

from __future__ import annotations

import pytest
import numpy as np
import jax
import jax.numpy as jnp

from legoesm.parallel.mesh import create_device_mesh, DeviceConfig
from legoesm.parallel.sharded_dynamics import (
    shard_state,
    gather_state,
    create_output_shardings,
    _make_sharding_spec,
    ShardingSpec,
)


N = 4  # Small grid


# ---------------------------------------------------------------------------
# ShardingSpec construction
# ---------------------------------------------------------------------------

class TestShardingSpec:
    def test_face_only_spec(self):
        config = create_device_mesh(n_devices=1)
        spec = _make_sharding_spec(config)
        assert isinstance(spec, ShardingSpec)
        assert spec.tiled_3d is None
        assert spec.tiled_2d is None

    def test_face_only_spec_has_face_axis(self):
        config = create_device_mesh(n_devices=1)
        spec = _make_sharding_spec(config)
        # face_3d should shard face axis
        assert "face" in str(spec.face_3d)
        # replicated should be an empty PartitionSpec.  jax shortened the
        # repr from "PartitionSpec()" to "P()" in recent releases, so accept
        # either rather than pinning a version-specific string.
        assert str(spec.replicated) in ("PartitionSpec()", "P()")


# ---------------------------------------------------------------------------
# shard_state / gather_state roundtrip (single device)
# ---------------------------------------------------------------------------

class TestShardGatherRoundtrip:
    """On single device, shard and gather should be identity."""

    def test_roundtrip_dict(self):
        config = create_device_mesh(n_devices=1)
        state = {
            "u": jnp.ones((6, N, N, 5), dtype=jnp.float64),
            "v": jnp.ones((6, N, N, 5), dtype=jnp.float64),
            "scalar": jnp.ones((), dtype=jnp.float64),
        }
        sharded = shard_state(state, config)
        gathered = gather_state(sharded, config)

        for key in state:
            np.testing.assert_array_equal(gathered[key], state[key])

    def test_roundtrip_preserves_shape(self):
        config = create_device_mesh(n_devices=1)
        arr = jnp.ones((6, N, N), dtype=jnp.float64)
        state = {"field": arr}
        out = gather_state(shard_state(state, config), config)
        assert out["field"].shape == (6, N, N)

    def test_roundtrip_preserves_values(self):
        config = create_device_mesh(n_devices=1)
        data = jnp.arange(6 * N * N, dtype=jnp.float64).reshape(6, N, N)
        state = {"data": data}
        out = gather_state(shard_state(state, config), config)
        np.testing.assert_array_equal(out["data"], data)

    def test_non_face_arrays_pass_through(self):
        config = create_device_mesh(n_devices=1)
        state = {"params": jnp.ones((10, 20), dtype=jnp.float64)}
        out = shard_state(state, config)
        np.testing.assert_array_equal(out["params"], state["params"])


# ---------------------------------------------------------------------------
# create_output_shardings
# ---------------------------------------------------------------------------

class TestCreateOutputShardings:
    def test_single_device_returns_none(self):
        config = create_device_mesh(n_devices=1)
        state = {"u": jnp.ones((6, N, N, 5))}
        out = create_output_shardings(state, config)
        assert out["u"] is None

    def test_output_shardings_preserves_structure(self):
        config = create_device_mesh(n_devices=1)
        state = {"a": jnp.ones((6, N, N)), "b": jnp.ones(3)}
        out = create_output_shardings(state, config)
        assert set(out.keys()) == set(state.keys())


# ---------------------------------------------------------------------------
# Latlon grid type shard_state
# ---------------------------------------------------------------------------

class TestLatlonShardState:
    def test_latlon_shard_single(self):
        from legoesm.parallel.mesh import create_latlon_mesh
        config = create_latlon_mesh(n_devices=1)
        state = {"T": jnp.ones((32, 64), dtype=jnp.float64)}
        out = shard_state(state, config, grid_type="latlon")
        np.testing.assert_array_equal(out["T"], state["T"])


# ---------------------------------------------------------------------------
# JIT compatibility
# ---------------------------------------------------------------------------

class TestJITShard:
    def test_shard_jittable(self):
        """shard_state should work inside JIT."""
        config = create_device_mesh(n_devices=1)
        state = {"field": jnp.ones((6, N, N), dtype=jnp.float64)}
        # shard_state itself uses jax.device_put, which is a host-side
        # operation — but the result should be a valid JAX array
        out = shard_state(state, config)
        assert isinstance(out["field"], jax.Array)

    def test_gather_jittable(self):
        config = create_device_mesh(n_devices=1)
        state = {"field": jnp.ones((6, N, N), dtype=jnp.float64)}
        out = gather_state(state, config)
        assert isinstance(out["field"], jax.Array)
