"""Lat-lon Laplacian temperature diffusion corrected to pressure surfaces.

DCMIP 2012 2-0-0 (constant-lapse-rate atmosphere at rest over a 2 km ridged
mountain) on the lat-lon C-grid grew 3.6 m/s/day at the lowest level.  The
cause was the ``A_h`` Laplacian diffusion of T taken along the terrain-following
eta surfaces: in a stratified atmosphere T varies along a sloping eta surface,
so ``A_h lap_eta T`` heats the valleys and cools the ridges (~160 K/day at the
bottom level, 72x144) with no wind present.  With the T diffusion zeroed (u,v
diffusion kept) the run stayed below 0.1 m/s.

The fix applies CAM3's partial correction to pressure surfaces (Collins et al.
2004, NCAR/TN-464+STR, eq. 3.80 with the vertical differences of eqs.
3.194-3.196): ``lap_p T = lap_eta T - (pi B dT/dp) lap(ln pi)``.  It is
OPT-IN (``A_h_pressure_surfaces``, user decision 2026-11-24): the default keeps
the eta-surface Laplacian unchanged.
"""
from __future__ import annotations

import numpy as np
import pytest

from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy


@pytest.fixture(autouse=True)
def _fp64_policy():
    prev = get_policy()
    set_policy(PrecisionPolicy.fp64())
    try:
        yield
    finally:
        set_policy(prev)


_N_LAT, _N_LON, _NLEV = 36, 72, 20
_DAY = 86400.0


def _dT_dt(stratified: bool, A_h: float, coord, pressure_surfaces: bool = True):
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
        CGridLatLonPrimitiveEquationConfig, hydrostatic_to_cgrid)
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_generic import (
        cgrid_latlon_pe_tendencies)
    from tests.test_cases.dcmip2012.rest_state_topography import (
        lapse_rate_state_topography_init_latlon, rest_state_topography_init_latlon)
    grid = create_latlon_grid(_N_LAT, _N_LON)
    init = (lapse_rate_state_topography_init_latlon if stratified
            else rest_state_topography_init_latlon)(grid, coord)
    st = hydrostatic_to_cgrid(init, grid)
    cfg = CGridLatLonPrimitiveEquationConfig(
        A_h=A_h, fix_mass=True, anchor_mass_to_initial=True,
        A_h_pressure_surfaces=pressure_surfaces)
    _du, _dv, dT, dps, _ = cgrid_latlon_pe_tendencies(st, grid, coord, cfg,
                                                     vertical_form="legacy")
    return np.asarray(dT), np.asarray(dps)


def _matrix_A_h() -> float:
    """The atmosphere matrix's lat-lon Laplacian viscosity at this n_lat."""
    import math
    from legoesm import constants
    return 0.1 * math.sqrt(constants.R_d * 300.0) * math.pi * constants.R_earth / _N_LAT


@pytest.mark.parametrize("coord_name", ["hybrid", "sigma"])
def test_stratified_rest_state_diffusion_heating_is_corrected(coord_name):
    """At rest every T tendency except the diffusion is zero, so dT/dt IS the
    diffusion.  Uncorrected eta-surface diffusion gives O(100) K/day over the
    ridges at every level; the corrected operator removes the leading term on
    the interior levels.  The top and bottom levels keep about half of it:
    CAM3's one-sided end differences (eqs. 3.194/3.196) carry a single
    interface term over 2 dp, i.e. half the derivative (KNOWN DIFFERENCE,
    reference-faithful)."""
    from legoesm.grids.vertical import create_sigma_coordinate, standard_hybrid_levels
    coord = (standard_hybrid_levels if coord_name == "hybrid"
             else create_sigma_coordinate)(_NLEV)
    dT, _ = _dT_dt(True, _matrix_A_h(), coord)
    dT_eta, _ = _dT_dt(True, _matrix_A_h(), coord, pressure_surfaces=False)
    per_level = np.abs(dT).max(axis=(0, 1)) * _DAY
    per_level_eta = np.abs(dT_eta).max(axis=(0, 1)) * _DAY
    corr = np.abs(dT - dT_eta).max(axis=(0, 1)) * _DAY
    # measured (fp64) interior peak: hybrid 3.4, sigma 3.3 K/day corrected;
    # 143 / 145 K/day uncorrected (eta-surface Laplacian).
    assert per_level[1:-1].max() < 10.0, (
        f"interior diffusion heating at rest {per_level[1:-1].max():.1f} K/day")
    # Non-vacuity: the correction must actually act (a silently zeroed
    # correction would leave the uncorrected ~140 K/day).
    assert per_level_eta[1:-1].max() > 10.0 * per_level[1:-1].max()
    # CAM's one-sided end differences are present, not dropped: the bottom
    # level is corrected by about half (measured 143 -> 74 K/day hybrid,
    # 146 -> 74 sigma) and both end levels carry a nonzero correction
    # (measured top / bottom: hybrid 3.3 / 70, sigma 44 / 72 K/day).
    assert per_level[-1] < 0.7 * per_level_eta[-1]
    assert corr[0] > 1.0 and corr[-1] > 10.0, corr


def test_default_off_is_the_eta_surface_laplacian():
    """Default (opt-out): over terrain with a stratified T the A_h tendency is
    exactly A_h lap_eta T, i.e. the operator of commit 97ddee4056 (the
    bitwise check against the base commit is out-of-tree, see the commit)."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.operators_latlon_cgrid import laplacian_cgrid
    from legoesm.grids.vertical import standard_hybrid_levels
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
        CGridLatLonPrimitiveEquationConfig, hydrostatic_to_cgrid)
    from tests.test_cases.dcmip2012.rest_state_topography import (
        lapse_rate_state_topography_init_latlon)
    assert CGridLatLonPrimitiveEquationConfig().A_h_pressure_surfaces is False
    coord = standard_hybrid_levels(_NLEV)
    dT_off, _ = _dT_dt(True, _matrix_A_h(), coord, pressure_surfaces=False)
    dT_0, _ = _dT_dt(True, 0.0, coord, pressure_surfaces=False)
    dT_on, _ = _dT_dt(True, _matrix_A_h(), coord, pressure_surfaces=True)
    grid = create_latlon_grid(_N_LAT, _N_LON)
    init = lapse_rate_state_topography_init_latlon(grid, coord)
    T = np.asarray(hydrostatic_to_cgrid(init, grid).T)
    want = _matrix_A_h() * np.asarray(laplacian_cgrid(T, grid))
    np.testing.assert_allclose(dT_off - dT_0, want, rtol=1e-9,
                               atol=1e-12 * np.abs(want).max())
    assert np.abs(dT_on - dT_off).max() * _DAY > 10.0


def test_cam_pi_B_dT_dp_rejects_a_full_level_B():
    """The vertical coefficient must be the nlev+1 half-level array."""
    import jax.numpy as jnp
    from legoesm.grids.vertical import cam_pi_B_dT_dp, create_sigma_coordinate
    coord = create_sigma_coordinate(2)          # sigma_half: 3 entries
    T = jnp.full((1, 1), 250.0)                 # nlev = 1: B[1:-1] would broadcast
    with pytest.raises(ValueError, match="nlev\\+1"):
        cam_pi_B_dT_dp(T, jnp.array([1.0e5]), coord)


def test_isothermal_rest_state_unchanged():
    """Isothermal columns have dT/dp = 0: the correction is exactly zero."""
    from legoesm.grids.vertical import standard_hybrid_levels
    dT, _ = _dT_dt(False, _matrix_A_h(), standard_hybrid_levels(_NLEV))
    assert np.abs(dT).max() < 1e-12


def test_cam_pi_B_dT_dp_matches_cam3_eqs_3_194_to_3_196():
    """Hand-evaluated CAM3 eqs. 3.194-3.196 on a 3-level hybrid column."""
    import jax.numpy as jnp
    from legoesm.grids.vertical import cam_pi_B_dT_dp, create_hybrid_coordinate
    A = jnp.array([0.0, 0.1, 0.05, 0.0])
    B = jnp.array([0.0, 0.2, 0.6, 1.0])
    coord = create_hybrid_coordinate(3, A, B)
    ps = jnp.array([9.0e4])
    T = jnp.array([[220.0, 250.0, 280.0]])
    dp = np.asarray(coord.dA * coord.p_ref + coord.dB * ps[0])
    Bh = np.asarray(coord.B_half)
    pi = 9.0e4
    want = [pi / (2 * dp[0]) * Bh[1] * 30.0,
            pi / (2 * dp[1]) * (Bh[2] * 30.0 + Bh[1] * 30.0),
            pi / (2 * dp[2]) * Bh[2] * 30.0]
    np.testing.assert_allclose(np.asarray(cam_pi_B_dT_dp(T, ps, coord))[0], want,
                               rtol=1e-12)


def test_mass_tendency_untouched():
    """The correction acts on T only: the surface-pressure tendency at rest is
    zero with or without diffusion."""
    from legoesm.grids.vertical import standard_hybrid_levels
    _, dps = _dT_dt(True, _matrix_A_h(), standard_hybrid_levels(_NLEV))
    assert np.abs(dps).max() == 0.0


def test_flat_surface_diffusion_is_the_eta_laplacian():
    """Uniform p_s: lap(ln p_s) = 0, so the operator is exactly A_h lap(T) and
    nonzero for a horizontally varying T at rest (guards against the T
    diffusion being dropped altogether)."""
    import jax.numpy as jnp
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.operators_latlon_cgrid import laplacian_cgrid
    from legoesm.grids.vertical import standard_hybrid_levels
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
        CGridLatLonPrimitiveEquationConfig, hydrostatic_to_cgrid)
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_generic import (
        cgrid_latlon_pe_tendencies)
    from tests.test_cases.dcmip2012.rest_state_topography import (
        rest_state_topography_init_latlon)
    grid = create_latlon_grid(_N_LAT, _N_LON)
    coord = standard_hybrid_levels(_NLEV)
    init = rest_state_topography_init_latlon(grid, coord, h_0=0.0)
    lon = jnp.asarray(grid.lon)[None, :, None]
    T = init.T.data + 5.0 * jnp.cos(3.0 * lon) * jnp.linspace(0.5, 1.0, _NLEV)
    st = hydrostatic_to_cgrid(init._replace(T=init.T.replace(data=T)), grid)
    out = []
    for A_h in (0.0, _matrix_A_h()):
        cfg = CGridLatLonPrimitiveEquationConfig(A_h=A_h)
        out.append(np.asarray(cgrid_latlon_pe_tendencies(
            st, grid, coord, cfg, vertical_form="legacy")[2]))
    want = _matrix_A_h() * np.asarray(laplacian_cgrid(T, grid))
    assert np.abs(want).max() > 1e-6
    np.testing.assert_allclose(out[1] - out[0], want, rtol=1e-9,
                               atol=1e-12 * np.abs(want).max())
