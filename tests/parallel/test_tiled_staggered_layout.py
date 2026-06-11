"""P4 phase-1: tiled staggered-leaf layout (Pace duplicated rows).

Pins the single source of truth for how D-grid staggered face arrays
(n+1 on one horizontal axis) map to per-tile blocks under the
(face, tile_i, tile_j) mesh: ``staggered_tile_block`` (body-side
slicing, shared entry duplicated into both neighbours) and
``staggered_blocks_to_face`` (canonical-owner inverse — lower tile owns
the shared entry).  Also pins the shard_pytree policy: staggered
face-plane leaves take FACE-ONLY sharding under a tiled config instead
of raising IndivisibleError (P4 discovery probe job 8464703).
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.parallel.mesh import (
    staggered_blocks_to_face,
    staggered_tile_block,
)

KTS = (2, 3)
NL = 4


@pytest.mark.parametrize("kt", KTS)
@pytest.mark.parametrize("stag_axis", (0, 1))
def test_staggered_roundtrip_identity(kt, stag_axis):
    n = kt * NL
    shape = (n + 1, n, 3) if stag_axis == 0 else (n, n + 1, 3)
    rng = np.random.default_rng(3)
    face = jnp.asarray(rng.standard_normal(shape))

    blocks = [
        [staggered_tile_block(face, ti, tj, NL, stag_axis)
         for tj in range(kt)]
        for ti in range(kt)
    ]
    # Every block carries nl+1 on the staggered axis, nl on the other.
    want_blk = (NL + 1, NL, 3) if stag_axis == 0 else (NL, NL + 1, 3)
    assert blocks[0][0].shape == want_blk

    # Shared entries duplicated: tile (1,0)'s first staggered row ==
    # tile (0,0)'s last (axis 0 case).
    if stag_axis == 0:
        np.testing.assert_array_equal(
            np.asarray(blocks[1][0][0]), np.asarray(blocks[0][0][-1]))
    else:
        np.testing.assert_array_equal(
            np.asarray(blocks[0][1][:, 0]), np.asarray(blocks[0][0][:, -1]))

    back = staggered_blocks_to_face(blocks, kt, stag_axis)
    np.testing.assert_array_equal(np.asarray(back), np.asarray(face))


def test_shard_pytree_staggered_face_only_no_indivisible():
    """Under a tiled DeviceConfig, a staggered (6, n+1, n, C) leaf must
    shard face-only (not raise IndivisibleError) while a divisible
    (6, n, n, C) leaf takes the full tile sharding."""
    from legoesm.parallel.mesh import create_device_mesh, shard_pytree

    n_dev = len(jax.devices())
    try:
        cfg = create_device_mesh(n_devices=n_dev)
    except Exception:
        pytest.skip("no tiled-capable device mesh on this host")
    if cfg.tiling == (1, 1):
        pytest.skip("single/face-only mesh — tiled policy not exercised")

    kt = cfg.tiling[0]
    n = kt * NL
    tree = {
        "centered": jnp.zeros((6, n, n, 2)),
        "stag_i": jnp.zeros((6, n + 1, n, 2)),
        "stag_j": jnp.zeros((6, n, n + 1, 2)),
    }
    out = shard_pytree(tree, cfg)  # must not raise
    assert out["centered"].sharding.spec == jax.sharding.PartitionSpec(
        "face", "tile_i", "tile_j")
    for k in ("stag_i", "stag_j"):
        spec = out[k].sharding.spec
        assert tuple(spec)[:1] == ("face",) and all(
            s is None for s in tuple(spec)[1:]
        ), f"{k} expected face-only sharding, got {spec}"
