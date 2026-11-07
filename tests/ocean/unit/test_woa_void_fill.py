"""The harmonic data-void fill of ``_fill_source_levels_nearest_valid``.

A basin the climatology never observed (a NaN hole wider than one grid step)
sits between two water masses.  The nearest-donor stitch puts a wall of the
full contrast inside the hole; the void fill ramps across it and never leaves
the range of the surrounding data.  Cells within one step of an observation
keep their nearest donor either way.
"""

import numpy as np
import pytest

from legoesm.ocean.init_woa import _fill_source_levels_nearest_valid, _harmonic_fill_2d

N_LAT, N_LON = 20, 40
LAT = np.linspace(-9.5, 9.5, N_LAT)
LON = np.linspace(0.5, 39.5, N_LON)


def _two_water_masses_with_a_hole():
    west = np.broadcast_to(np.arange(N_LON)[None, :] < N_LON // 2, (N_LAT, N_LON))
    f = np.where(west, 10.0, 20.0)                                         # west 10, east 20
    f = np.repeat(f[:, :, None], 2, axis=2).astype(np.float64)             # (lat, lon, 2 levels)
    hole = np.zeros((N_LAT, N_LON), dtype=bool)
    hole[6:14, 15:25] = True                                               # 8 x 10 unobserved at the surface
    f[hole, 0] = np.nan                                                    # (observed at level 1: ocean, not land)
    f[8, 3, 0] = np.nan                                                    # a one-cell coastal gap
    f[0, :, :] = np.nan                                                    # a land row: never a void
    return f, hole


def _max_adjacent_jump(level, region):
    d = np.maximum(np.abs(np.diff(level, axis=1)), 0)
    dj = np.abs(np.diff(level, axis=0))
    return max(d[region[:, 1:] & region[:, :-1]].max(), dj[region[1:] & region[:-1]].max())


def test_void_fill_ramps_where_the_nearest_stitch_builds_a_wall():
    f, hole = _two_water_masses_with_a_hole()
    (stitched,), n_stitch, n_void_off = _fill_source_levels_nearest_valid([f], LAT, LON)
    (smooth,), n_fill, n_void = _fill_source_levels_nearest_valid([f], LAT, LON, void_fill=True)
    assert n_stitch == n_fill == 81 + 2 * N_LON and n_void_off == 0
    # void = missing OCEAN cells whose nearest donor is not a grid neighbour:
    # the hole minus its one-cell rim (the 6 x 8 interior); the coastal gap and
    # the land row are not void
    assert n_void == 6 * 8
    np.testing.assert_array_equal(smooth[0], stitched[0])
    interior = np.zeros_like(hole)
    interior[7:13, 16:24] = True                 # the void: hole minus its nearest-filled rim
    # nearest stitch: the full 10-unit wall runs through the void
    assert _max_adjacent_jump(stitched[:, :, 0], interior) == pytest.approx(10.0)
    # void fill: a ramp -- the middle row rises monotonically west to east in
    # steps well below the wall, every value inside the surrounding range,
    # and no non-void cell is touched
    mid = smooth[10, 16:24, 0]
    assert np.all(np.diff(mid) > 0) and np.diff(mid).max() < 2.5
    assert _max_adjacent_jump(smooth[:, :, 0], interior) < 6.0
    assert smooth[:, :, 0][hole].min() >= 10.0 - 1e-9 and smooth[:, :, 0][hole].max() <= 20.0 + 1e-9
    np.testing.assert_array_equal(smooth[~interior, 0], stitched[~interior, 0])
    assert np.all(np.isfinite(smooth))
    # the one-cell gap and the level with no hole are the nearest stitch on both paths
    assert smooth[8, 3, 0] == stitched[8, 3, 0] == 10.0
    np.testing.assert_array_equal(smooth[1:, :, 1], f[1:, :, 1])


def test_harmonic_fill_is_exact_for_a_discrete_harmonic_and_periodic_in_lon():
    j = np.arange(N_LAT)[:, None] * np.ones((1, N_LON))
    i = np.ones((N_LAT, 1)) * np.arange(N_LON)[None, :]
    # cos(2 pi i/N) cosh(kappa j) is discrete-harmonic on the periodic stencil
    # when cosh(kappa) = 2 - cos(2 pi/N); it varies in longitude, so a seam
    # handled as a wall (not periodic) would not reproduce it
    kappa = np.arccosh(2.0 - np.cos(2.0 * np.pi / N_LON))
    field = 3.0 + np.cos(2.0 * np.pi * i / N_LON) * np.cosh(kappa * j)
    void = np.zeros_like(field, dtype=bool)
    void[5:12, 10:20] = True
    out = _harmonic_fill_2d(np.where(void, 99.0, field), void)
    np.testing.assert_allclose(out, field, atol=1e-8)
    void2 = np.zeros_like(field, dtype=bool)
    void2[5:12, :3] = True
    void2[5:12, -3:] = True
    out2 = _harmonic_fill_2d(np.where(void2, 99.0, field), void2)
    np.testing.assert_allclose(out2, field, atol=1e-8)
    # the same seam void solved as if the seam were a wall is NOT the field
    wall = np.roll(np.where(void2, 99.0, field), N_LON // 2, axis=1)   # seam moved to mid-grid ...
    out3 = np.roll(_harmonic_fill_2d(wall, np.roll(void2, N_LON // 2, axis=1)), -N_LON // 2, axis=1)
    np.testing.assert_allclose(out3, field, atol=1e-8)                 # ... periodic: still exact


def test_harmonic_fill_does_not_cross_land():
    # two basins at 10 and 20 separated by a land column; a void on the
    # 10-side must be filled from the 10-side only
    field = np.where(np.arange(N_LON)[None, :] < N_LON // 2 - 1, 10.0, 20.0) * np.ones((N_LAT, 1))
    domain = np.ones((N_LAT, N_LON), dtype=bool)
    domain[:, N_LON // 2 - 1] = False                       # land column, first 20-valued column
    void = np.zeros_like(domain)
    void[5:12, N_LON // 2 - 5:N_LON // 2 - 1] = True         # void touching the wall
    out = _harmonic_fill_2d(np.where(void, 99.0, field), void, domain=domain)
    np.testing.assert_allclose(out[void], 10.0, atol=1e-9)
    out_nowall = _harmonic_fill_2d(np.where(void, 99.0, field), void)
    assert out_nowall[void].max() > 10.5                     # without the wall the 20s leak in
    # a void region with no domain neighbour at all takes the mean of what it
    # arrived with (no boundary to be harmonic against), nothing else moves
    island = np.zeros_like(domain); island[2, 2:5] = True
    dom2 = island.copy()
    arrived = field.copy(); arrived[2, 2:5] = [1.0, 2.0, 6.0]
    out3 = _harmonic_fill_2d(arrived, island, domain=dom2)
    np.testing.assert_allclose(out3[2, 2:5], 3.0)
    np.testing.assert_array_equal(out3[~island], arrived[~island])
