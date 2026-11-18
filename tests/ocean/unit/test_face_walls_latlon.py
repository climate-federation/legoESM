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
    compute_face_masks, compute_face_masks_3d, face_wall_gates, min_cell_to_uface,
    min_cell_to_vface, walled_face_depth,
)

N_LAT, N_LON, NLEV, K_WALL = 6, 8, 6, 2
V_WALL = (3, slice(1, 3))                  # v-faces between rows 2|3, columns 1-2


def _walls():
    fu = np.full((N_LAT, N_LON + 1), NLEV, int)
    fu[:, [0, 4, N_LON]] = K_WALL          # basins: cells 0-3 and 4-7, walled below level 2
    fu[1, 4] = 0                           # one face closed outright
    fv = np.full((N_LAT + 1, N_LON), NLEV, int)
    fv[V_WALL] = K_WALL                    # a meridional sill
    fv[2, 5] = 0                           # one v-face closed outright
    fv[1, 6] = K_WALL                      # a sill at a polar-filtered latitude
    return fu, fv


def _geom(walls=True):
    g = ensure_geometry(create_latlon_grid(N_LAT, N_LON))
    if not walls:
        return g
    fu, fv = _walls()
    return g._replace(face_nlev_u=jnp.asarray(fu), face_nlev_v=jnp.asarray(fv))


def _setup(walls=True, H_east=3000.0, partial=True):
    """(grid, z_coord, rest state) in 64-bit; two zonal basins when ``walls``."""
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    from legoesm.ocean.vertical import create_ocean_z_star, create_partial_cell_coordinate
    from legoesm.runtime.precision import apply_precision
    apply_precision("fp64")      # closures to 1e-9 or better need a 64-bit state (fp32 drifts 2e-7)
    grid = _geom(walls)
    H = np.full((N_LAT, N_LON), 3000.0)
    H[:, 4:] = H_east
    z = create_ocean_z_star(n_levels=NLEV, H_max=3000.0)
    if partial:
        z = create_partial_cell_coordinate(z, jnp.asarray(H))
    s = rest_state_latlon_cgrid_ocean(grid, z, H_max=3000.0, land_lat_threshold=89.0,
                                      H_bathy_override=jnp.asarray(H))
    return grid, z, s


def _basin_tracers():
    """T, S uniform above the wall and uniform within each zonal basin below it."""
    T = np.full((N_LAT, N_LON, NLEV), 15.0)
    T[:, :4, K_WALL:], T[:, 4:, K_WALL:] = 10.0, 3.0
    S = np.full((N_LAT, N_LON, NLEV), 35.0)
    S[:, :4, K_WALL:], S[:, 4:, K_WALL:] = 35.5, 34.5
    return T, S


def _model(grid, z, **flat):
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
    from legoesm.ocean.state import LatLonCGridOceanConfig
    return LatLonCGridOceanModel(grid, z, LatLonCGridOceanConfig.from_flat(
        barotropic_solver="implicit_cn", implicit_vertical_mixing=True,
        enable_runtime_checks=False, **flat))


def test_mask_builders_honour_walls_and_are_identity_without():
    act = jnp.ones((N_LAT, N_LON, NLEV))
    u0, v0 = (np.asarray(m) for m in compute_face_masks_3d(act, _geom(False)))
    u1, v1 = (np.asarray(m) for m in compute_face_masks_3d(act, _geom()))
    assert float(u1[3, 4, K_WALL - 1]) == 1.0 and float(u1[3, 4, K_WALL]) == 0.0
    assert float(u1[3, 0, K_WALL]) == 0.0 and float(u1[3, N_LON, K_WALL]) == 0.0
    assert float(u1[1, 4, 0]) == 0.0 and float(u1[3, 3, NLEV - 1]) == 1.0
    assert (v1[V_WALL][:, :K_WALL] == 1.0).all() and (v1[V_WALL][:, K_WALL:] == 0.0).all()
    assert (v1[2, 5] == 0.0).all() and float(v1[4, 4, NLEV - 1]) == 1.0
    cut = ~(_walls()[1][..., None] > np.arange(NLEV))
    assert np.array_equal(v1[~cut], v0[~cut]) and (v1[cut] == 0.0).all()
    assert (v0[V_WALL] == 1.0).all()                           # the sill faces join wet cells
    m = jnp.ones((N_LAT, N_LON))
    u2, v2 = compute_face_masks(m, _geom())
    assert float(u2[1, 4]) == 0.0 and float(u2[3, 4]) == 1.0
    assert float(v2[2, 5]) == 0.0 and float(v2[3, 1]) == 1.0
    for k in (0, 1):
        assert np.array_equal(compute_face_masks(m, _geom(False))[k], compute_face_masks(m, None)[k])


def test_face_thickness_and_barotropic_depth():
    h = jnp.asarray(np.random.default_rng(0).uniform(50.0, 500.0, (N_LAT, N_LON, NLEV)))
    g0, g1 = _geom(False), _geom()
    hu0, hu1 = min_cell_to_uface(h, g0), min_cell_to_uface(h, g1)
    assert float(hu1[3, 4, K_WALL]) == 0.0 and float(hu1[3, 4, 0]) == float(hu0[3, 4, 0])
    hv0, hv1 = np.asarray(min_cell_to_vface(h, g0)), np.asarray(min_cell_to_vface(h, g1))
    assert (hv1[V_WALL][:, K_WALL:] == 0.0).all() and (hv1[2, 5] == 0.0).all()
    assert np.array_equal(hv1[V_WALL][:, :K_WALL], hv0[V_WALL][:, :K_WALL])
    assert np.array_equal(hv1[4], hv0[4]) and (hv0[V_WALL] > 0.0).all()
    H_u = jnp.sum(hu0, -1) + 1.0          # stand-ins for a min-of-totals depth
    H_v = jnp.asarray(hv0.sum(-1) + 1.0)
    Hu, Hv = walled_face_depth(H_u, H_v, h, g1, 0.5)
    assert float(Hu[3, 4]) == pytest.approx(float(jnp.sum(hu0[3, 4, :K_WALL])))
    assert float(Hu[3, 2]) == float(H_u[3, 2])
    assert float(Hv[3, 1]) == pytest.approx(float(hv0[3, 1, :K_WALL].sum()))
    assert float(Hv[2, 5]) == 0.5 and float(Hv[4, 4]) == float(H_v[4, 4])
    assert np.array_equal(walled_face_depth(H_u, H_v, h, g0, 0.5)[0], H_u)
    assert np.array_equal(walled_face_depth(H_u, H_v, h, g0, 0.5)[1], H_v)


def _run(walls, n_steps=12):
    grid, z, s = _setup(walls, H_east=2600.0)          # unequal partial bottom cells
    T, _ = _basin_tracers()
    T[3:, 1:3, K_WALL:] += 2.0                         # deep contrast across the v-sill too
    eta = np.zeros((N_LAT, N_LON)); eta[2:4, 1:3] = 0.2   # nonzero SSH / z-star stretch
    s = s._replace(T=s.T.replace(data=jnp.asarray(T)), eta=s.eta.replace(data=jnp.asarray(eta)))
    model = _model(grid, z, A_h=1.0e4)
    area = np.asarray(grid.area_T)
    vol0 = float(np.sum(np.asarray(s.eta.data) * area * np.asarray(s.land_mask.data)))
    deep_u, deep_v = [], []
    for _ in range(n_steps):
        s = model.step(s, 1800.0)
        deep_u.append(float(np.max(np.abs(np.asarray(s.u.data)[:, [0, 4, N_LON], K_WALL:]))))
        deep_v.append(float(np.max(np.abs(np.asarray(s.v.data)[V_WALL][:, K_WALL:]))))
    vol1 = float(np.sum(np.asarray(s.eta.data) * area * np.asarray(s.land_mask.data)))
    return max(deep_u), max(deep_v), vol0, vol1, s


def test_two_basin_deep_wall_blocks_flow_and_conserves_volume():
    deep_u, deep_v, vol0, vol1, s = _run(True)
    assert deep_u == 0.0 and deep_v == 0.0
    assert np.isfinite(np.asarray(s.T.data)).all()
    deep_u_open, deep_v_open, _, _, _ = _run(False)
    assert s.eta.data.dtype == jnp.float64
    assert abs(vol1 - vol0) <= 1e-9 * abs(vol0)
    # the deep contrasts do drive flow through both sills without walls
    assert deep_u_open > 1e-4 and deep_v_open > 1e-4


# ---------------------------------------------------------------- lateral tracer mixing

def _lateral_tendencies(walls, T, S, **flat):
    from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import _bc_tracer_tendencies
    from legoesm.ocean.state import LatLonCGridOceanConfig
    grid, z, s = _setup(walls)
    cfg = LatLonCGridOceanConfig.from_flat(implicit_vertical_mixing=True, **flat)
    dT, dS = _bc_tracer_tendencies(jnp.asarray(T), jnp.asarray(S), cfg, grid, s.land_mask.data,
                                   jnp.ones((N_LAT, N_LON)), z)
    return np.asarray(dT), np.asarray(dS), np.asarray(grid.area_T)


@pytest.mark.parametrize("coef", [dict(K_h=1.0e3), dict(K_bih=1.0e13),
                                  dict(K_h=1.0e3, K_bih=1.0e13)])
def test_lateral_tracer_diffusion_does_not_cross_walls(coef):
    """Budget: heat and salt.  Laplacian / biharmonic tracer fluxes are zero through a
    closed face level, so a basin-uniform deep tracer does not relax toward the
    neighbour basin.  Only what the walls change is asserted: this operator is not
    thickness-weighted and does not close the global heat budget, walls or not."""
    T, S = _basin_tracers()
    dT, dS, area = _lateral_tendencies(True, T, S, **coef)
    dT0, dS0, _ = _lateral_tendencies(False, T, S, **coef)
    assert np.abs(dT0[:, [3, 4], K_WALL:]).max() > 0.0         # open sill: the basins do mix
    assert np.abs(dS0[:, [3, 4], K_WALL:]).max() > 0.0
    assert np.abs(dT).max() == 0.0 and np.abs(dS).max() == 0.0
    if "K_bih" not in coef:
        # a meridional deep step is held by the v-sill, and mixes where there is none
        # (Laplacian only: the biharmonic reaches these cells through their open neighbours)
        T2, S2 = np.full_like(T, 15.0), np.full_like(S, 35.0)
        T2[3:, :, K_WALL:], S2[3:, :, K_WALL:] = 9.0, 34.0
        dT, dS, _ = _lateral_tendencies(True, T2, S2, **coef)
        assert np.abs(dT[2:4, 1:3, K_WALL:]).max() == 0.0
        assert np.abs(dS[2:4, 1:3, K_WALL:]).max() == 0.0
        assert np.abs(dT[2:4, 6, K_WALL:]).max() > 0.0


def _gm(walls, T, S, cfg=None):
    from legoesm.ocean.physics.lateral_mixing.config import GMRediConfig
    from legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid import (
        gm_redi_tracer_tendency_latlon)
    from legoesm.ocean.vertical import compute_layer_thickness
    grid, z, s = _setup(walls)
    dT, dS = gm_redi_tracer_tendency_latlon(
        jnp.asarray(T), jnp.asarray(S), s.eta.data, s.H_bathy.data, grid, z,
        cfg or GMRediConfig(), mask=s.land_mask.data, u_mask=s.u_mask.data,
        v_mask=s.v_mask.data)
    h = np.asarray(compute_layer_thickness(s.eta.data, s.H_bathy.data, z))
    return np.asarray(dT), np.asarray(dS), h * np.asarray(grid.area_T)[..., None]


def test_gm_redi_triads_do_not_cross_walls():
    """Budget: heat and salt.  The production triad GM/Redi (partial-cell coordinate)
    carries nothing through a closed face level and stays in flux form."""
    T, S = _basin_tracers()
    dT, dS, _ = _gm(True, T, S)
    dT0, dS0, _ = _gm(False, T, S)
    assert np.abs(dT0[:, [3, 4], K_WALL:]).max() > 0.0         # open sill: the basins do mix
    assert np.abs(dT).max() == 0.0 and np.abs(dS).max() == 0.0
    rng = np.random.default_rng(2)
    T3 = T + rng.normal(0, 0.3, T.shape) - 0.5 * np.arange(NLEV)   # stratified, sloping
    S3 = S + rng.normal(0, 0.05, S.shape)
    dT, dS, w = _gm(True, T3, S3)
    for d in (dT, dS):
        assert np.abs(d).max() > 0.0
        assert abs((d * w).sum()) <= 1e-10 * np.abs(d * w).sum()


@pytest.mark.parametrize("scheme", ["centered", "nemo_iso_lap"])
def test_walls_refuse_gm_redi_schemes_that_ignore_them(scheme):
    from legoesm.ocean.physics.lateral_mixing.config import GMRediConfig
    grid, z, _ = _setup()
    with pytest.raises(ValueError, match="slope_scheme"):
        _model(grid, z, gm_redi=GMRediConfig(slope_scheme=scheme))
    _model(grid, z, gm_redi=GMRediConfig())                    # triads: accepted
    if scheme == "centered":
        grid0, z0, _ = _setup(False)
        _model(grid0, z0, gm_redi=GMRediConfig(slope_scheme=scheme))   # no walls: unaffected


# ---------------------------------------------------------------- momentum

def _open_velocity(s, grid, seed):
    """Random u, v on open face levels, zero on closed ones."""
    rng = np.random.default_rng(seed)
    gu, gv = (np.asarray(g) for g in face_wall_gates(grid, NLEV, jnp.float64))
    u = rng.normal(0, 0.1, (N_LAT, N_LON + 1, NLEV))
    u[:, -1] = u[:, 0]
    u = u * gu * np.asarray(s.u_mask.data)[..., None]
    v = rng.normal(0, 0.1, (N_LAT + 1, N_LON, NLEV)) * gv * np.asarray(s.v_mask.data)[..., None]
    return u, v, gu, gv


def test_vertical_friction_keeps_open_column_momentum():
    """Budget: momentum.  With no drag and no stress the implicit vertical friction only
    redistributes momentum among a face's OPEN levels (sum_k dz_k u_k unchanged)."""
    grid, z, s = _setup()
    u, v, gu, gv = _open_velocity(s, grid, 3)
    s = s._replace(u=s.u.replace(data=jnp.asarray(u)), v=s.v.replace(data=jnp.asarray(v)))
    n = _model(grid, z, A_v=1.0e-2)._apply_implicit_vertical_mixing(
        s, 1800.0, None, do_tracers=False)
    dz = np.asarray(z.dz_ref)
    for old, new, g in ((u, np.asarray(n.u.data), gu), (v, np.asarray(n.v.data), gv)):
        assert np.abs(new - old).max() > 1e-6
        assert np.abs(new * (1 - g)).max() == 0.0
        assert np.abs(((new - old) * dz).sum(-1)).max() <= 1e-12 * (np.abs(old) * dz).sum(-1).max()


def test_walls_refuse_a_pure_zstar_coordinate():
    grid, z, _ = _setup(partial=False)
    with pytest.raises(ValueError, match="partial-cell"):
        _model(grid, z)
    grid0, z0, _ = _setup(False, partial=False)
    _model(grid0, z0)                                          # no walls: unaffected
