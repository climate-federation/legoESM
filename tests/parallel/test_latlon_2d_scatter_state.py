"""Direct tests for scatter_state_latlon_2d (pure-indexing 2-D state scatter).

Foundation for the 2-D pencil dycore step: verify the rank-local block is the
correct lat x lon slice, with the v lat-face sharing its boundary row, and that
proc_lon == 1 reproduces the full-longitude band.
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
    v = np.arange((n_lat + 1) * n_lon * nlev, dtype=float).reshape(n_lat + 1, n_lon, nlev)
    return St(u=cell(nlev), v=v, T=cell(nlev), p_s=cell(), phis=cell(),
              tracers={"q": cell(nlev)})


def test_scatter_2d_slices_block():
    g = _global()
    pr, pc, n_lat, n_lon = 2, 2, 8, 12
    for rank in range(pr * pc):
        L = make_latlon_2d_layout(rank, pr, pc, n_lat, n_lon)
        loc = scatter_state_latlon_2d(g, L)
        s, e, w, x = L.lat_start, L.lat_end, L.lon_start, L.lon_end
        assert np.array_equal(loc.T, g.T[s:e, w:x])
        assert np.array_equal(loc.u, g.u[s:e, w:x])
        assert np.array_equal(loc.p_s, g.p_s[s:e, w:x])
        assert np.array_equal(loc.phis, g.phis[s:e, w:x])
        assert np.array_equal(loc.tracers["q"], g.tracers["q"][s:e, w:x])
        # v lat-face: one extra row (shared boundary), lon sliced like the rest.
        assert np.array_equal(loc.v, g.v[s:e + 1, w:x])
        assert loc.v.shape[0] == loc.T.shape[0] + 1


def test_scatter_2d_proc_lon_1_is_full_lon_band():
    g = _global()
    nproc, n_lat, n_lon = 4, 8, 12
    for rank in range(nproc):
        L = make_latlon_2d_layout(rank, nproc, 1, n_lat, n_lon)
        loc = scatter_state_latlon_2d(g, L)
        s, e = L.lat_start, L.lat_end
        assert loc.u.shape[1] == n_lon          # full longitude
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
        assert float(g.total_area) == float(grid.total_area)


def test_scatter_2d_blocks_tile_global_cells():
    # The union of all ranks' cell-centred T blocks must tile the globe exactly.
    g = _global()
    pr, pc, n_lat, n_lon = 2, 3, 8, 12
    covered = np.zeros((n_lat, n_lon), dtype=int)
    for rank in range(pr * pc):
        L = make_latlon_2d_layout(rank, pr, pc, n_lat, n_lon)
        covered[L.lat_start:L.lat_end, L.lon_start:L.lon_end] += 1
    assert np.all(covered == 1)  # exact partition, no gaps/overlaps
