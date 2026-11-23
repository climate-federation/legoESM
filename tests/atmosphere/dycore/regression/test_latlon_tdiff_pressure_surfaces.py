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
3.194-3.196): ``lap_p T = lap_eta T - (pi B dT/dp) lap(ln pi)``.
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


def _dT_dt(stratified: bool, A_h: float, coord):
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
    cfg = CGridLatLonPrimitiveEquationConfig(A_h=A_h, fix_mass=True,
                                             anchor_mass_to_initial=True)
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
    per_level = np.abs(dT).max(axis=(0, 1)) * _DAY
    # measured (fp64) interior peak: hybrid 3.4, sigma 3.3 K/day corrected;
    # 136 / 144 K/day with the correction reverted.
    assert per_level[1:-1].max() < 10.0, (
        f"interior diffusion heating at rest {per_level[1:-1].max():.1f} K/day")


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
