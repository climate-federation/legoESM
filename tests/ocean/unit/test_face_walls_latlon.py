"""Depth-limited face walls on the lat-lon C-grid (grid.face_nlev_u/v).

A 1-degree face between two wet cells can still be blocked below a sub-grid
sill; the walls AND an open-level count into every face mask and face thickness.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.grids.latlon import create_latlon_grid, ensure_geometry  # noqa: E402
from legoesm.ocean.dynamics.latlon_cgrid_operators import (  # noqa: E402
    compute_face_masks, compute_face_masks_3d, min_cell_to_uface,
    min_cell_to_vface, walled_face_depth,
)

N_LAT, N_LON, NLEV, K_WALL = 6, 8, 6, 2


def _walls():
    fu = np.full((N_LAT, N_LON + 1), NLEV, int)
    fu[:, [0, 4, N_LON]] = K_WALL          # basins: cells 0-3 and 4-7, walled below level 2
    fu[1, 4] = 0                           # one face closed outright
    fv = np.full((N_LAT + 1, N_LON), NLEV, int)
    return fu, fv


def _geom(walls=True):
    g = ensure_geometry(create_latlon_grid(N_LAT, N_LON))
    if not walls:
        return g
    fu, fv = _walls()
    return g._replace(face_nlev_u=jnp.asarray(fu), face_nlev_v=jnp.asarray(fv))


def test_mask_builders_honour_walls_and_are_identity_without():
    act = jnp.ones((N_LAT, N_LON, NLEV))
    u0, v0 = compute_face_masks_3d(act, _geom(False))
    u1, v1 = compute_face_masks_3d(act, _geom())
    assert np.array_equal(v1, v0)
    assert float(u1[3, 4, K_WALL - 1]) == 1.0 and float(u1[3, 4, K_WALL]) == 0.0
    assert float(u1[3, 0, K_WALL]) == 0.0 and float(u1[3, N_LON, K_WALL]) == 0.0
    assert float(u1[1, 4, 0]) == 0.0 and float(u1[3, 3, NLEV - 1]) == 1.0
    m = jnp.ones((N_LAT, N_LON))
    u2, _ = compute_face_masks(m, _geom())
    assert float(u2[1, 4]) == 0.0 and float(u2[3, 4]) == 1.0
    assert np.array_equal(compute_face_masks(m, _geom(False))[0], compute_face_masks(m, None)[0])


def test_face_thickness_and_barotropic_depth():
    h = jnp.asarray(np.random.default_rng(0).uniform(50.0, 500.0, (N_LAT, N_LON, NLEV)))
    g0, g1 = _geom(False), _geom()
    hu0, hu1 = min_cell_to_uface(h, g0), min_cell_to_uface(h, g1)
    assert float(hu1[3, 4, K_WALL]) == 0.0 and float(hu1[3, 4, 0]) == float(hu0[3, 4, 0])
    assert np.array_equal(min_cell_to_vface(h, g1), min_cell_to_vface(h, g0))
    H_u = jnp.sum(hu0, -1) + 1.0          # stand-in for a min-of-totals depth
    H_v = jnp.zeros((N_LAT + 1, N_LON))
    Hu, Hv = walled_face_depth(H_u, H_v, h, g1, 0.5)
    assert float(Hu[3, 4]) == pytest.approx(float(jnp.sum(hu0[3, 4, :K_WALL])))
    assert float(Hu[3, 2]) == float(H_u[3, 2]) and np.array_equal(Hv, H_v)
    assert np.array_equal(walled_face_depth(H_u, H_v, h, g0, 0.5)[0], H_u)


def _run(walls, n_steps=12):
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    from legoesm.ocean.state import LatLonCGridOceanConfig
    from legoesm.ocean.vertical import create_ocean_z_star, create_partial_cell_coordinate
    from legoesm.runtime.precision import apply_precision
    apply_precision("fp64")      # volume closure to 1e-9 needs a 64-bit state (fp32 drifts 2e-7)
    grid = _geom(walls)
    H = np.full((N_LAT, N_LON), 3000.0)
    H[:, 4:] = 2600.0                                  # unequal partial bottom cells
    z = create_partial_cell_coordinate(create_ocean_z_star(n_levels=NLEV, H_max=3000.0),
                                       jnp.asarray(H))
    s = rest_state_latlon_cgrid_ocean(grid, z, H_max=3000.0, land_lat_threshold=89.0,
                                      H_bathy_override=jnp.asarray(H))
    T = np.full((N_LAT, N_LON, NLEV), 15.0)
    T[:, :4, K_WALL:] = 10.0                           # deep basin contrast
    T[:, 4:, K_WALL:] = 3.0
    eta = np.zeros((N_LAT, N_LON)); eta[2:4, 1:3] = 0.2   # nonzero SSH / z-star stretch
    s = s._replace(T=s.T.replace(data=jnp.asarray(T)), eta=s.eta.replace(data=jnp.asarray(eta)))
    cfg = LatLonCGridOceanConfig.from_flat(
        barotropic_solver="implicit_cn", implicit_vertical_mixing=True,
        A_h=1.0e4, enable_runtime_checks=False)
    model = LatLonCGridOceanModel(grid, z, cfg)
    area = np.asarray(grid.area_T)
    vol0 = float(np.sum(np.asarray(s.eta.data) * area * np.asarray(s.land_mask.data)))
    deep_u = []
    for _ in range(n_steps):
        s = model.step(s, 1800.0)
        deep_u.append(float(np.max(np.abs(np.asarray(s.u.data)[:, [0, 4, N_LON], K_WALL:]))))
    vol1 = float(np.sum(np.asarray(s.eta.data) * area * np.asarray(s.land_mask.data)))
    return max(deep_u), vol0, vol1, s


def test_two_basin_deep_wall_blocks_flow_and_conserves_volume():
    deep_u, vol0, vol1, s = _run(True)
    assert deep_u == 0.0
    assert np.isfinite(np.asarray(s.T.data)).all()
    deep_u_open, v0o, v1o, so = _run(False)
    assert s.eta.data.dtype == jnp.float64
    assert abs(vol1 - vol0) <= 1e-9 * abs(vol0)
    assert deep_u_open > 1e-4                           # the deep contrast does drive flow without walls
