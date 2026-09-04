"""Window layout invariants for the DUO tiled port (fv3_duo_windows).

These are layout tests, not physics: every flat cell has exactly one
owner, a scatter of a gather is the identity, a masked gather touches
nothing outside its mask, and kt=1 is the whole face byte for byte.
"""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy
jax.config.update("jax_enable_x64", True)

from legoesm.grids.fv3_duo_windows import (  # noqa: E402
    build_window_layout, gather_windows, scatter_owned, gather_masked,
    horizontal_axes, window_axis_extent)

N, NG = 24, 3


def _rand(shape, seed=0):
    return jnp.asarray(np.random.default_rng(seed).standard_normal(shape))


@pytest.mark.parametrize("partition,kt,pad", [
    ("compute", 1, 3), ("compute", 2, 5), ("compute", 2, 8),
    ("compute", 3, 5), ("padded", 2, 5), ("padded", 3, 5), ("padded", 5, 4)])
@pytest.mark.parametrize("extents", [(N + 2 * NG, N + 2 * NG),
                                     (N + 2 * NG + 1, N + 2 * NG),
                                     (N + 2 * NG, N + 2 * NG + 1),
                                     (N + 2 * NG + 1, N + 2 * NG + 1),
                                     (N, N), (N + 1, N), (N + 2, N + 2)])
def test_scatter_of_gather_is_identity(partition, kt, pad, extents):
    lay = build_window_layout(N, NG, kt, pad, partition)
    x = _rand((6,) + extents + (4,))
    xw = gather_windows(lay, x)
    assert xw.shape == (lay.nb, window_axis_extent(lay, extents[0]),
                        window_axis_extent(lay, extents[1]), 4)
    back = scatter_owned(lay, xw)
    assert np.array_equal(np.asarray(back), np.asarray(x))   # bitwise


@pytest.mark.parametrize("partition,kt", [("compute", 3), ("padded", 3),
                                          ("padded", 5)])
def test_every_flat_cell_has_exactly_one_owner(partition, kt):
    from legoesm.grids.fv3_duo_windows import _owner_mask, _index
    lay = build_window_layout(N, NG, kt, 5, partition)
    for extents in [(N + 2 * NG, N + 2 * NG + 1), (N + 1, N), (N + 2, N)]:
        count = np.zeros((6,) + extents, int)
        for w in range(lay.nb):
            idx = _index(lay, w, (6,) + extents, (1, 2))
            count[idx] += _owner_mask(lay, w, *extents)
        assert count.min() == 1 and count.max() == 1


def test_kt1_is_the_whole_face():
    lay = build_window_layout(N, NG, 1, 99)          # pad is forced to ng
    assert lay.W == N + 2 * NG and lay.nb == 6
    x = _rand((6, N + 2 * NG, N + 2 * NG + 1, 3))
    assert np.array_equal(np.asarray(gather_windows(lay, x)), np.asarray(x))


def test_gather_masked_touches_only_the_mask():
    lay = build_window_layout(N, NG, 2, 6)
    x = _rand((6, N + 2 * NG, N + 2 * NG, 2))
    xw = gather_windows(lay, x)
    y = x + 1.0
    mask = np.zeros((6, N + 2 * NG, N + 2 * NG), bool)
    mask[:, :NG, :] = True                           # the west ring
    out = np.asarray(gather_masked(lay, xw, y, mask))
    yw = np.asarray(gather_windows(lay, y))
    xw = np.asarray(xw)
    mw = np.asarray(gather_windows(lay, jnp.asarray(mask)[..., None]))[..., 0]
    assert np.array_equal(out[mw], yw[mw])
    assert np.array_equal(out[~mw], xw[~mw])
    assert mw.any() and (~mw).any()


def test_ikj_layout_round_trip():
    lay = build_window_layout(N, NG, 2, 5)
    x = _rand((6, N + 2, 7, N + 2))                  # pe: (i, k, j)
    assert horizontal_axes(lay, x.shape, 6) == (1, 3)
    xw = gather_windows(lay, x)
    assert xw.shape[2] == 7
    assert np.array_equal(np.asarray(scatter_owned(lay, xw)), np.asarray(x))


def test_window_too_wide_refused():
    with pytest.raises(ValueError, match="exceeds the padded face"):
        build_window_layout(N, NG, 2, 12)            # 12 + 24 > 30


def test_padded_partition_blocks_match_the_blocked_layout():
    """padded partition: block t = padded rows [t*nl, t*nl+nl+e) -- the
    fv3_duo_spmd.to_blocked layout; edge windows sit flush."""
    from legoesm.grids.fv3_duo_spmd import to_blocked
    lay = build_window_layout(N, NG, 3, 4, "padded")
    assert lay.nl == (N + 2 * NG) // 3 and lay.block_start(1) == lay.nl
    assert lay.origins[0][1] == 0 and lay.origins[-1][1] == lay.m_a - lay.W
    x = np.random.default_rng(1).standard_normal((6, N + 2 * NG + 1,
                                                  N + 2 * NG, 2))
    xb = to_blocked(x, 3, lay.nl)                     # (6, 3*(nl+1), 3*nl, 2)
    xw = np.asarray(gather_windows(lay, jnp.asarray(x)))
    for w, (face, oi, oj) in enumerate(lay.origins):
        ti, tj = (w % 9) // 3, w % 3
        blk = xb[face, ti * (lay.nl + 1):(ti + 1) * (lay.nl + 1),
                 tj * lay.nl:(tj + 1) * lay.nl]
        li, lj = ti * lay.nl - oi, tj * lay.nl - oj
        assert np.array_equal(xw[w, li:li + lay.nl + 1, lj:lj + lay.nl], blk)
    with pytest.raises(ValueError, match="not divisible"):
        build_window_layout(N, NG, 4, 4, "padded")   # 30 % 4
