"""Lat-lon C-grid pole rows under CROSS-POLAR flow (DCMIP 2008 rotated JW).

The rotated (alpha = 45 deg) JW steady state is an exact balance of the
continuous equations with the jet crossing both poles.  On the pole-adjacent
rows the C-grid must not treat the zero-length polar v-face (wall v = 0) as a
VALUE: v -> u-face for Coriolis and v -> cell for KE are face-length weighted,
and the pole vertex vorticity is the Stokes polar-cap circulation, not the
free-slip wall zero.  Each of those was a spurious O(f v) force on the pole
row only (rotated_steady: 44 m/s after 2 d vs 27.9 analytic).  The initial
momentum tendency on the pole rows must therefore be no larger than in the
interior (pre-fix it was ~10x larger).
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

_needs_x64 = pytest.mark.skipif(
    not jax.config.jax_enable_x64, reason="fp64 tolerances: run with JAX_ENABLE_X64=1")


def _pole_to_interior_du_ratio(alpha):
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
        CGridLatLonPrimitiveEquationConfig, CGridLatLonPrimitiveEquationModel,
        hydrostatic_to_cgrid)
    from tests.test_cases.dcmip2008.jablonowski_rotated import (
        rotated_baroclinic_init_latlon)
    grid = create_latlon_grid(36, 72, coriolis_alpha=alpha)
    sigma = create_sigma_coordinate(8)
    model = CGridLatLonPrimitiveEquationModel(
        grid, sigma, CGridLatLonPrimitiveEquationConfig(), dt=60.0)
    state = hydrostatic_to_cgrid(rotated_baroclinic_init_latlon(
        grid, sigma, perturbed=False, alpha=alpha), grid)
    du = np.abs(np.asarray(model.tendencies(state)[0]))
    return max(du[0].max(), du[-1].max()) / du[2:-2].max()


@_needs_x64
def test_rotated_jw_pole_rows_balanced():
    ratio = _pole_to_interior_du_ratio(math.pi / 4.0)
    assert ratio < 1.5, ratio


@_needs_x64
def test_unrotated_jw_pole_rows_balanced():
    # control: no cross-polar flow, pole rows were already fine
    assert _pole_to_interior_du_ratio(0.0) < 1.5


def _tilted_solid_body_pole_rel_error(dtype=jnp.float64):
    """Coriolis-like term zeta*v of a solid-body rotation about an axis tilted
    45 deg (flow crosses both poles; omega=0 so eta = zeta, known exactly:
    zeta = 2W(cos a sin lat + sin a cos lat cos lon)).  Returns the max
    relative error of cor_u on the two pole rows and on the interior."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.atmosphere.dynamics.gcm.shallow_water_latlon_cgrid import (
        absolute_vorticity_coriolis)
    g = create_latlon_grid(36, 72, omega=0.0)
    W, a, R = 1e-5, math.pi / 4.0, float(g.radius)
    lat = np.asarray(g.lat, np.float64)[:, None]
    lon_u = (np.asarray(g.lon, np.float64) - 0.5 * float(g.dlon))
    lon_u = np.append(lon_u, lon_u[0] + 2 * math.pi)[None, :]
    lat_v = np.asarray(g.lat_v, np.float64)[:, None]
    lon_c = np.asarray(g.lon, np.float64)[None, :]
    u = W * R * (math.cos(a) * np.cos(lat) - math.sin(a) * np.sin(lat) * np.cos(lon_u))
    v = W * R * math.sin(a) * np.sin(lon_c) * np.ones_like(lat_v)
    v[0] = v[-1] = 0.0                                   # pole wall faces
    cor_u, _ = absolute_vorticity_coriolis(
        jnp.asarray(u, dtype), jnp.asarray(v, dtype), g)
    zeta = 2 * W * (math.cos(a) * np.sin(lat) + math.sin(a) * np.cos(lat) * np.cos(lon_u))
    exact = zeta * W * R * math.sin(a) * np.sin(lon_u)
    err = np.abs(np.asarray(cor_u, np.float64) - exact) / np.abs(exact).max()
    return max(err[0].max(), err[-1].max()), err[1:-1].max()


@_needs_x64
def test_pole_cap_vorticity_and_v_at_u_cross_polar():
    pole, interior = _tilted_solid_body_pole_rel_error()
    # base: pole 0.75 (zero pole vorticity + halved v), interior 0.004
    assert interior < 0.01 and pole < 0.01, (pole, interior)


def test_ke_cell_v_stencil_edge_cases():
    """The pole-row extrapolation needs two REAL faces per pole: a two-row
    global grid falls back to the single real face (never the other pole's
    wall); a pole-free strip keeps the plain mean; jit-safe."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.atmosphere.dynamics.gcm.shallow_water_latlon_cgrid import (
        latlon_pole_cell_v)
    g2 = create_latlon_grid(2, 8)
    v = jnp.zeros((3, 8)).at[1].set(1.0)
    vc = 0.5 * (v[:-1] + v[1:])
    np.testing.assert_allclose(np.asarray(latlon_pole_cell_v(vc, v, g2)), 1.0)
    # a non-finite wall sample must not leak into the single-face fallback
    vnan = v.at[0].set(jnp.nan).at[2].set(jnp.nan)
    np.testing.assert_allclose(
        np.asarray(latlon_pole_cell_v(vc, vnan, g2)), 1.0)
    # same shape, but no geographic pole (regional wall): plain mean kept
    wall = g2._replace(cos_lat_v=jnp.full_like(g2.cos_lat_v, 0.5))
    np.testing.assert_array_equal(
        np.asarray(jax.jit(latlon_pole_cell_v)(vc, v, wall)), np.asarray(vc))
    # >= 3 rows: pole rows use the extrapolated interior faces only
    g = create_latlon_grid(4, 8)
    v4 = jnp.zeros((5, 8)).at[1].set(2.0).at[2].set(1.0).at[3].set(2.0)
    out = np.asarray(latlon_pole_cell_v(0.5 * (v4[:-1] + v4[1:]), v4, g))
    np.testing.assert_allclose(out[0], 2.5)   # 1.5*2 - 0.5*1, not (0+2)/2
    np.testing.assert_allclose(out[-1], 2.5)
    np.testing.assert_allclose(out[1:-1], 1.5)
    # one-cell global grid: both faces are walls -> plain mean, no wall sample
    from legoesm.atmosphere.dynamics.gcm.shallow_water_latlon_cgrid import (
        latlon_cell_ke)
    g1 = create_latlon_grid(1, 8)
    v1 = jnp.full((2, 8), jnp.nan)
    vc1 = 0.5 * (v1[:-1] + v1[1:])                 # NaN plain mean, as callers form it
    np.testing.assert_array_equal(np.asarray(latlon_pole_cell_v(vc1, v1, g1)), 0.0)
    np.testing.assert_array_equal(
        np.asarray(latlon_cell_ke(jnp.ones((1, 8)), vc1, v1, g1)), 0.5)


def test_pole_treatment_needs_global_pole_grid():
    """A wall at the pole latitude on a grid NOT built as a global pole-to-pole
    grid (regional / Mercator / Cartesian) never gets the pole treatment."""
    from legoesm.grids.latlon import create_latlon_grid, create_regional_latlon_grid
    from legoesm.atmosphere.dynamics.gcm.shallow_water_latlon_cgrid import (
        latlon_pole_cell_v, latlon_pole_end_masks)
    g = create_latlon_grid(8, 16)
    assert g.global_poles is True
    assert [bool(m) for m in latlon_pole_end_masks(g)] == [True, True]
    wall = g._replace(global_poles=False)          # same geometry, a wall
    assert [bool(m) for m in latlon_pole_end_masks(wall)] == [False, False]
    v = jnp.ones((9, 16)).at[0].set(0.0).at[-1].set(0.0)
    vc = 0.5 * (v[:-1] + v[1:])
    np.testing.assert_array_equal(np.asarray(latlon_pole_cell_v(vc, v, wall)),
                                  np.asarray(vc))
    # regional strip whose north wall is 1e-7 rad from the pole (lat in deg)
    rg, _ = create_regional_latlon_grid(4, 8, 80.0, 90.0 - 1e-7 * 180 / np.pi,
                                        periodic_x=True, lon_east=360.0)
    assert rg.global_poles is False
    assert [bool(m) for m in latlon_pole_end_masks(rg)] == [False, False]


def test_coriolis_pole_row_uses_extrapolated_v():
    """Operator level: u = 0 and zonally uniform v linear in latitude on the
    real faces (walls 0), so relative vorticity vanishes and cor_u on the pole
    rows must equal eta_at_u * v(lat_cell) exactly -- the nearest-face value
    (or the old wall-averaged one) would not."""
    from legoesm import constants
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.atmosphere.dynamics.gcm.shallow_water_latlon_cgrid import (
        absolute_vorticity_coriolis)
    g = create_latlon_grid(12, 24)
    a, b = 3.0, 2.0
    lat_v = np.asarray(g.lat_v, np.float64)
    v = (a + b * lat_v)[:, None] * np.ones((1, 24))
    v[0] = v[-1] = 0.0
    cor_u, _ = absolute_vorticity_coriolis(
        jnp.zeros((12, 25)), jnp.asarray(v), g)
    cor_u = np.asarray(cor_u)
    lat = np.asarray(g.lat, np.float64)
    f_v = 2 * constants.Omega * np.sin(0.5 * (lat[:-1] + lat[1:]))
    for row, f_pole, f_in in ((0, -2 * constants.Omega, f_v[0]),
                              (-1, 2 * constants.Omega, f_v[-1])):
        expect = 0.5 * (f_pole + f_in) * (a + b * lat[row])
        np.testing.assert_allclose(cor_u[row], expect, rtol=1e-6)


def test_cell_ke_pole_rows_value():
    """KE: u = 0, v linear in latitude on the real faces -> pole-row KE is
    0.5 * v(lat_cell)**2 (not the wall-averaged or nearest-face value)."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.atmosphere.dynamics.gcm.shallow_water_latlon_cgrid import (
        latlon_cell_ke)
    g = create_latlon_grid(12, 24)
    a, b = 3.0, 2.0
    v = (a + b * np.asarray(g.lat_v, np.float64))[:, None] * np.ones((1, 24))
    v[0] = v[-1] = 0.0
    v = jnp.asarray(v)
    ke = np.asarray(latlon_cell_ke(jnp.zeros((12, 24)), 0.5 * (v[:-1] + v[1:]), v, g))
    lat = np.asarray(g.lat, np.float64)
    for row in (0, -1):
        np.testing.assert_allclose(ke[row], 0.5 * (a + b * lat[row]) ** 2, rtol=1e-6)


@pytest.mark.parametrize("path", ["pe_legacy", "pe_sb81_theta", "sw"])
def test_every_latlon_ke_path_uses_shared_pole_ke(monkeypatch, path):
    """Enforcement: each lat-lon KE call site (legacy PE, generic PE
    sb81_theta, SW) routes through the shared ``latlon_cell_ke``."""
    import legoesm.atmosphere.dynamics.gcm.shallow_water_latlon_cgrid as swl
    import legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid as pel
    from legoesm.grids.latlon import create_latlon_grid
    calls = []
    orig = swl.latlon_cell_ke

    def spy(*a, **k):
        calls.append(1)
        return orig(*a, **k)

    monkeypatch.setattr(swl, "latlon_cell_ke", spy)
    monkeypatch.setattr(pel, "latlon_cell_ke", spy)
    g = create_latlon_grid(8, 16)
    if path == "sw":
        from legoesm.atmosphere.dynamics.gcm.shallow_water_latlon_cgrid import (
            williamson_test2_cgrid)
        from legoesm.atmosphere.dynamics.gcm.shallow_water_generic import (
            cgrid_latlon_sw_tendencies)
        cgrid_latlon_sw_tendencies(williamson_test2_cgrid(g), g)
    else:
        from legoesm.grids.vertical import create_sigma_coordinate
        from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
            CGridLatLonPrimitiveEquationConfig, hydrostatic_to_cgrid)
        from legoesm.atmosphere.dynamics.gcm.primitive_eq_generic import (
            cgrid_latlon_pe_tendencies)
        from tests.test_cases.dcmip2008.jablonowski_rotated import (
            rotated_baroclinic_init_latlon)
        sig = create_sigma_coordinate(4)
        st = hydrostatic_to_cgrid(rotated_baroclinic_init_latlon(
            g, sig, perturbed=False, alpha=0.0), g)
        form = "legacy" if path == "pe_legacy" else "sb81_theta"
        cgrid_latlon_pe_tendencies(st, g, sig, CGridLatLonPrimitiveEquationConfig(),
                                   dt=60.0, vertical_form=form)
    assert calls, f"{path}: KE did not go through latlon_cell_ke"
