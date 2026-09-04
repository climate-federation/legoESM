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


@pytest.mark.parametrize("kt,pad", [(1, 3), (2, 5), (2, 8), (3, 5)])
@pytest.mark.parametrize("extents", [(N + 2 * NG, N + 2 * NG),
                                     (N + 2 * NG + 1, N + 2 * NG),
                                     (N + 2 * NG, N + 2 * NG + 1),
                                     (N + 2 * NG + 1, N + 2 * NG + 1),
                                     (N, N), (N + 1, N), (N + 2, N + 2)])
def test_scatter_of_gather_is_identity(kt, pad, extents):
    lay = build_window_layout(N, NG, kt, pad)
    x = _rand((6,) + extents + (4,))
    xw = gather_windows(lay, x)
    assert xw.shape == (lay.nb, window_axis_extent(lay, extents[0]),
                        window_axis_extent(lay, extents[1]), 4)
    back = scatter_owned(lay, xw)
    assert np.array_equal(np.asarray(back), np.asarray(x))   # bitwise


def test_every_flat_cell_has_exactly_one_owner():
    from legoesm.grids.fv3_duo_windows import _owner_mask, _index
    lay = build_window_layout(N, NG, 3, 5)
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
