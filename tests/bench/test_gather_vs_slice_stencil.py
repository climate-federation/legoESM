"""Direct test of the compaction microbench's numerics (CPU, tiny grid)."""
import numpy as np
import jax.numpy as jnp

from bench_gather_vs_slice_stencil import (
    _build_gather,
    _laplacian_gather,
    _laplacian_slice,
)


def test_gather_matches_slice_on_interior():
    rng = np.random.default_rng(1)
    mask = np.ones((16, 32), dtype=bool)
    mask[:, :8] = False
    f_np = rng.standard_normal(mask.shape)
    f = jnp.asarray(f_np)
    wet, nbrs = _build_gather(mask)
    v = jnp.asarray(f_np[wet[:, 0], wet[:, 1]])
    lap_d = np.asarray(_laplacian_slice(f))
    lap_g = np.asarray(_laplacian_gather(v, nbrs))
    idx_of = -np.ones(mask.shape, dtype=np.int64)
    idx_of[wet[:, 0], wet[:, 1]] = np.arange(len(wet))
    interior = np.ones(len(wet), dtype=bool)
    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        interior &= idx_of[(wet[:, 0] + di) % 16, (wet[:, 1] + dj) % 32] >= 0
    assert interior.any()
    np.testing.assert_allclose(
        lap_g[interior], lap_d[wet[interior, 0], wet[interior, 1]],
        rtol=1e-12)


def test_dry_neighbours_map_to_self():
    mask = np.ones((8, 8), dtype=bool)
    mask[:, :4] = False
    wet, nbrs = _build_gather(mask)
    # column 4 wet cells have a dry west neighbour -> index maps to self
    west = np.asarray(nbrs[3])
    col4 = np.where(wet[:, 1] == 4)[0]
    assert (west[col4] == col4).all()
