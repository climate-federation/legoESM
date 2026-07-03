"""Direct tests for scatter_state_latlon_2d (pure-indexing 2-D state scatter).

Foundation for the 2-D pencil dycore step: verify the rank-local block is the
correct lat x lon slice, with the v lat-face sharing its boundary ROW and the
u lon-face sharing its boundary COLUMN (the dycore stores u with n_lon+1 faces,
so a block owns n_lon_local+1 faces — the lon twin of v), and that proc_lon == 1
reproduces the full-longitude band (u keeps all n_lon+1 faces).

Regression guard: an earlier fixture built u CELL-shaped (n_lon) and asserted
``u.shape[1] == n_lon``, masking a scatter bug that dropped u's periodic-closure
face under 2-D (so proc_lon==1 did NOT reproduce the band for u).  These tests
use the REAL dycore u-face shape (n_lon+1).
"""

from __future__ import annotations

from collections import namedtuple

import numpy as np

from legoesm.parallel.latlon_mpi import (
    make_latlon_2d_layout,
    scatter_state_latlon_2d,
    slice_latlon_grid_to_block_2d,
)

St = namedtuple("St", "u v T p_s phis tracers")


def _global(n_lat=8, n_lon=12, nlev=3):
    def cell(*extra):
        shp = (n_lat, n_lon, *extra)
        return np.arange(int(np.prod(shp)), dtype=float).reshape(shp)
    # u: LON-face, n_lon+1 faces (dycore convention).  The trailing face is the
    # PERIODIC CLOSURE: face n_lon == face 0 — enforce that invariant (the
    # scatter's seam correctness relies on it), else the fixture is unfaithful.
    u = np.arange(n_lat * (n_lon + 1) * nlev, dtype=float).reshape(
        n_lat, n_lon + 1, nlev)
    u[:, n_lon, :] = u[:, 0, :]
    # v: LAT-face, n_lat+1 rows.
    v = np.arange((n_lat + 1) * n_lon * nlev, dtype=float).reshape(
        n_lat + 1, n_lon, nlev)
    return St(u=u, v=v, T=cell(nlev), p_s=cell(), phis=cell(),
              tracers={"q": cell(nlev)})


def test_scatter_2d_slices_block():
    g = _global()
    pr, pc, n_lat, n_lon = 2, 2, 8, 12
    for rank in range(pr * pc):
        L = make_latlon_2d_layout(rank, pr, pc, n_lat, n_lon)
        loc = scatter_state_latlon_2d(g, L)
        s, e, w, x = L.lat_start, L.lat_end, L.lon_start, L.lon_end
        assert np.array_equal(loc.T, g.T[s:e, w:x])
        assert np.array_equal(loc.p_s, g.p_s[s:e, w:x])
        assert np.array_equal(loc.phis, g.phis[s:e, w:x])
        assert np.array_equal(loc.tracers["q"], g.tracers["q"][s:e, w:x])
        # v lat-face: one extra row (shared boundary), lon sliced like the cells.
        assert np.array_equal(loc.v, g.v[s:e + 1, w:x])
        assert loc.v.shape[0] == loc.T.shape[0] + 1
        # u lon-face: faces [w, x] => one extra (shared east) column.
        assert np.array_equal(loc.u, g.u[s:e, w:x + 1])
        assert loc.u.shape[1] == loc.T.shape[1] + 1


def test_scatter_2d_u_shared_east_face_duplicated():
    # The east column of a block == the west column of its east neighbour
    # (the same global lon-face, held by both — the u twin of v's shared row).
    # Cover ALL lon pairs INCLUDING the periodic seam (last pcol -> pcol 0):
    # the seam works only because the global u carries the closure face
    # u[:, n_lon] == u[:, 0], so the last block's east face equals block-0's
    # west face — the case the fix relies on (codex round-2 caught the seam
    # being skipped).
    g = _global()
    pr, pc, n_lat, n_lon = 2, 3, 8, 12
    for prow in range(pr):
        for pcol in range(pc):              # incl wrap: east = (pcol+1) % pc
            rank = prow * pc + pcol
            east = prow * pc + (pcol + 1) % pc
            lo = scatter_state_latlon_2d(g, make_latlon_2d_layout(
                rank, pr, pc, n_lat, n_lon))
            le = scatter_state_latlon_2d(g, make_latlon_2d_layout(
                east, pr, pc, n_lat, n_lon))
            assert np.array_equal(lo.u[:, -1], le.u[:, 0]), (
                f"u shared east face mismatch prow={prow} pcol={pcol}->east "
                f"(seam={pcol == pc - 1})")


def test_scatter_2d_proc_lon_1_is_full_lon_band():
    g = _global()
    nproc, n_lat, n_lon = 4, 8, 12
    for rank in range(nproc):
        L = make_latlon_2d_layout(rank, nproc, 1, n_lat, n_lon)
        loc = scatter_state_latlon_2d(g, L)
        s, e = L.lat_start, L.lat_end
        # proc_lon==1 keeps ALL n_lon+1 u faces (== band), NOT n_lon.
        assert loc.u.shape[1] == n_lon + 1
        assert np.array_equal(loc.u, g.u[s:e, :])
        assert np.array_equal(loc.T, g.T[s:e, :])
        assert np.array_equal(loc.v, g.v[s:e + 1, :])


def test_slice_grid_2d_block():
    import jax.numpy as jnp
    from legoesm.grids.latlon import create_latlon_grid

    grid = create_latlon_grid(8)  # n_lat=8, n_lon=16
    pr, pc = 2, 2
    for rank in range(pr * pc):
        L = make_latlon_2d_layout(rank, pr, pc, grid.n_lat, grid.n_lon)
        g = slice_latlon_grid_to_block_2d(grid, L, skip_total_area_reduce=True)
        s, e, w, x = L.lat_start, L.lat_end, L.lon_start, L.lon_end
        assert g.n_lat == e - s and g.n_lon == x - w
        assert jnp.array_equal(g.lat, grid.lat[s:e])
        assert jnp.array_equal(g.lon, grid.lon[w:x])
        assert jnp.array_equal(g.f, grid.f[s:e, w:x])
        assert jnp.array_equal(g.area, grid.area[s:e, w:x])
        # v-face lat keeps the shared boundary row.
        assert jnp.array_equal(g.lat_v, grid.lat_v[s:e + 1])
        assert g.lat_v.shape[0] == g.n_lat + 1
        # skip_total_area_reduce => the full-sphere area (mass-fix denominator).
        # ``jnp.sum(grid.area)`` reduces in area's dtype (float32) while the
        # grid stores ``total_area`` in float64 under JAX_ENABLE_X64, so compare
        # with a float32-reduction tolerance, not exact == (a mixed-precision
        # reduction is a bug-magnet for exact equality).
        np.testing.assert_allclose(
            float(g.total_area), float(grid.total_area), rtol=1e-5)


def test_scatter_2d_blocks_tile_global_cells():
    # The union of all ranks' cell-centred T blocks must tile the globe exactly.
    g = _global()
    pr, pc, n_lat, n_lon = 2, 3, 8, 12
    covered = np.zeros((n_lat, n_lon), dtype=int)
    for rank in range(pr * pc):
        L = make_latlon_2d_layout(rank, pr, pc, n_lat, n_lon)
        covered[L.lat_start:L.lat_end, L.lon_start:L.lon_end] += 1
    assert np.all(covered == 1)  # exact partition, no gaps/overlaps
