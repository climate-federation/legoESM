"""Tests for prescribed ERA5 surface boundary conditions in the radiation lane.

Covers the ``sfc_lw_up``, ``sfc_sw_up``/``sfc_sw_down`` forcing keys consumed by
``make_radiation_physics``: LW up prescribes the radiative surface temperature,
SW up/down prescribes an effective column albedo above a flux threshold, and
mismatched SW keys must raise. Everything is checked against the equivalent
``T_sfc``/``sfc_albedo`` forcing semantics so the boundary condition is pinned.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from legoesm.atmosphere.dynamics.gcm.spectral_pe import isothermal_rest_state_spectral
from legoesm.atmosphere.physics.radiation.config import RadiationConfig
from legoesm.atmosphere.physics.radiation.integration import make_radiation_physics
from legoesm.grids.gaussian import create_gaussian_grid
from legoesm.grids.vertical import create_sigma_coordinate

from legoesm import constants


@pytest.fixture(scope="module")
def setup():
    grid = create_gaussian_grid(n_max=8)
    sigma = create_sigma_coordinate(n_levels=3)
    qv = jnp.full((grid.n_lat, grid.n_lon, sigma.n_levels), 5.0e-3)
    state = isothermal_rest_state_spectral(grid, sigma, tracers={"q_v": qv})
    cfg = RadiationConfig(scheme="rrtmgp", diurnal_cycle=False)
    fn = make_radiation_physics(cfg, "spectral_pe")
    ncol = grid.n_lat * grid.n_lon
    return state, grid, sigma, fn, ncol


def heating(state, grid, sigma, fn, forcing):
    out = fn(state, grid, sigma, forcing=forcing)
    return float(jnp.sum(jnp.abs(out.T_hat.data) ** 2))


def test_sfc_lw_up_changes_heating_and_varies_with_value(setup):
    state, grid, sigma, fn, ncol = setup
    h0 = heating(state, grid, sigma, fn, None)
    lw_lo = jnp.full(ncol, 200.0)
    lw_hi = jnp.full(ncol, 400.0)
    h_lo = heating(state, grid, sigma, fn, {"sfc_lw_up": lw_lo})
    h_hi = heating(state, grid, sigma, fn, {"sfc_lw_up": lw_hi})
    assert h_lo != pytest.approx(h0)
    assert h_hi != pytest.approx(h_lo)


def test_sfc_lw_up_matches_t_sfc_with_emissivity_one(setup):
    state, grid, sigma, fn, ncol = setup
    T = 289.0
    lw_up = jnp.full(ncol, constants.sigma_sb * T**4)
    h_lw = heating(state, grid, sigma, fn, {"sfc_lw_up": lw_up})
    h_tsfc = heating(
        state, grid, sigma, fn,
        {"T_sfc": jnp.full(ncol, T), "sfc_emissivity": jnp.ones(ncol)},
    )
    assert h_lw == pytest.approx(h_tsfc, rel=1e-6)


def test_sw_up_down_matches_albedo_and_threshold(setup):
    state, grid, sigma, fn, ncol = setup
    h_none = heating(state, grid, sigma, fn, None)

    sw_up = jnp.full(ncol, 0.8 * 500.0)
    sw_down = jnp.full(ncol, 500.0)
    h_sw = heating(state, grid, sigma, fn, {"sfc_sw_up": sw_up, "sfc_sw_down": sw_down})
    h_alb = heating(state, grid, sigma, fn, {"sfc_albedo": jnp.full(ncol, 0.8)})
    assert h_sw == pytest.approx(h_alb, rel=1e-6)

    # Below the 1 W/m^2 threshold the prescribed ratio is ignored.
    sw_down_lo = jnp.full(ncol, 0.5)
    h_lo = heating(
        state, grid, sigma, fn,
        {"sfc_sw_up": 0.8 * sw_down_lo, "sfc_sw_down": sw_down_lo},
    )
    assert h_lo == pytest.approx(h_none, rel=1e-6)
    # ... and keeps a per-step forcing albedo, not the config scalar.
    h_lo_alb = heating(
        state, grid, sigma, fn,
        {"sfc_sw_up": 0.8 * sw_down_lo, "sfc_sw_down": sw_down_lo,
         "sfc_albedo": jnp.full(ncol, 0.8)},
    )
    assert h_lo_alb == pytest.approx(h_alb, rel=1e-6)
    assert h_lo_alb != pytest.approx(h_none, rel=1e-6)


def test_mismatched_sw_pair_raises(setup):
    state, grid, sigma, fn, ncol = setup
    with pytest.raises(ValueError):
        heating(state, grid, sigma, fn, {"sfc_sw_up": jnp.full(ncol, 100.0)})
    with pytest.raises(ValueError):
        heating(state, grid, sigma, fn, {"sfc_sw_down": jnp.full(ncol, 500.0)})


def test_gradient_of_heating_wrt_sfc_lw_up_finite_nonzero(setup):
    state, grid, sigma, fn, ncol = setup

    def h_of_lw(lw):
        out = fn(state, grid, sigma, forcing={"sfc_lw_up": lw})
        return jnp.sum(jnp.abs(out.T_hat.data) ** 2)

    g = jax.grad(h_of_lw)(jnp.full(ncol, 300.0))
    assert g.shape == (ncol,)
    assert bool(jnp.all(jnp.isfinite(g)))
    assert float(jnp.max(jnp.abs(g))) > 0.0


def test_dark_column_fallback_accepts_a_two_dimensional_albedo(setup):
    """A build-time (n_lat, n_lon) albedo field is a supported layout; the
    dark-column fallback must flatten it like every other override."""
    state, grid, sigma, _fn, ncol = setup
    from legoesm.atmosphere.physics.radiation.config import RadiationConfig
    from legoesm.atmosphere.physics.radiation.integration import (
        make_radiation_physics,
    )

    cfg = RadiationConfig(scheme="rrtmgp", diurnal_cycle=False)
    alb2d = jnp.full((grid.n_lat, grid.n_lon), 0.45)
    fn2d = make_radiation_physics(cfg, "spectral_pe", sfc_albedo_override=alb2d)
    fn1d = make_radiation_physics(cfg, "spectral_pe",
                                  sfc_albedo_override=alb2d.reshape(-1))
    dark = {"sfc_sw_up": jnp.full(ncol, 0.4), "sfc_sw_down": jnp.full(ncol, 0.5)}
    assert heating(state, grid, sigma, fn2d, dark) == pytest.approx(
        heating(state, grid, sigma, fn1d, dark), rel=1e-9)
    assert heating(state, grid, sigma, fn2d, dark) == pytest.approx(
        heating(state, grid, sigma, fn2d, None), rel=1e-9)


def test_prescribed_radiative_fluxes_refused_on_a_scheme_that_ignores_them(setup):
    state, grid, sigma, _fn, ncol = setup
    from legoesm.atmosphere.physics.radiation.config import RadiationConfig
    from legoesm.atmosphere.physics.radiation.integration import (
        make_radiation_physics,
    )

    fn = make_radiation_physics(
        RadiationConfig(scheme="simple_lw", diurnal_cycle=False), "spectral_pe")
    with pytest.raises(ValueError, match="only consumed by the rrtmgp and gray"):
        fn(state, grid, sigma, forcing={"sfc_lw_up": jnp.full(ncol, 300.0)})


def test_scalar_prescribed_planes_broadcast_over_the_columns(setup):
    """A scalar sfc_sw_up / sfc_sw_down / sfc_lw_up (the legacy uniform-
    forcing convention) is accepted and equals the per-column version."""
    state, grid, sigma, fn, ncol = setup
    per_col = heating(state, grid, sigma, fn,
                      {"sfc_sw_up": jnp.full(ncol, 400.0),
                       "sfc_sw_down": jnp.full(ncol, 500.0),
                       "sfc_lw_up": jnp.full(ncol, 380.0)})
    scalar = heating(state, grid, sigma, fn,
                     {"sfc_sw_up": 400.0, "sfc_sw_down": 500.0,
                      "sfc_lw_up": 380.0})
    assert scalar == pytest.approx(per_col, rel=1e-9)


# --- land-coupled radiating surface (tier-5 U11-1) ---------------------------

def test_land_radiating_surface_closes_the_tile_lw_and_keeps_sea_cells():
    """``eps_g*sigma*T_g^4 + (1-eps_g)*La`` is the area sum of the two tiles'
    grey-body LW at ANY La; an f=0 cell is returned bitwise; check_grads."""
    import numpy as np
    from jax.test_util import check_grads
    from legoesm.driver.model_driver import land_radiating_surface
    sb = constants.sigma_sb
    f = jnp.array([0.0, 0.3, 0.999, 1.0])
    T_sea = jnp.array([290.0, 285.0, 271.0, 300.0])
    T_l = jnp.array([0.0, 301.0, 250.0, 310.0])      # 0: zero-filled ocean cell
    eps_l = jnp.array([0.0, 0.95, 0.986, 0.96])
    E = 0.98
    T_g, eps_g = land_radiating_surface(T_sea, f, E, T_l, eps_l)
    assert float(T_g[0]) == float(T_sea[0]) and float(eps_g[0]) == E
    for La in (150.0, 420.0):
        seen = eps_g * sb * T_g ** 4 + (1.0 - eps_g) * La
        want = ((1.0 - f) * (E * sb * T_sea ** 4 + (1.0 - E) * La)
                + f * (eps_l * sb * T_l ** 4 + (1.0 - eps_l) * La))
        np.testing.assert_allclose(seen, want, rtol=1e-13, atol=0.0)
    # a float32 land mask must not round E before the float64 fields join
    T32, e32 = land_radiating_surface(T_sea, f.astype(jnp.float32), E, T_l,
                                      eps_l)
    assert T32.dtype == T_sea.dtype and float(e32[0]) == E
    m = slice(1, 4)
    check_grads(lambda a, b, c, d: land_radiating_surface(a, b, E, c, d)[0],
                (T_sea[m], f[m], T_l[m], eps_l[m]), order=2,
                modes=("fwd", "rev"))


def test_land_radiative_bc_pair_scheme_and_identity():
    from legoesm.atmosphere.physics.radiation.integration import (
        resolve_land_radiative_bc)
    cfg = RadiationConfig(scheme="rrtmgp")
    T0 = jnp.full(3, 280.0)
    assert resolve_land_radiative_bc({}, cfg, 3, T0, None) == (T0, None)
    T, e = resolve_land_radiative_bc(
        {"T_sfc_rad": jnp.full(3, 290.0), "sfc_emissivity": jnp.full(3, 0.95)},
        cfg, 3, T0, None)
    assert float(T[0]) == 290.0 and float(e[0]) == 0.95
    with pytest.raises(ValueError, match="needs BOTH"):
        resolve_land_radiative_bc({"T_sfc_rad": T0}, cfg, 3, T0, None)
    with pytest.raises(ValueError, match="ncol=4"):
        resolve_land_radiative_bc(
            {"T_sfc_rad": T0, "sfc_emissivity": jnp.ones(3)}, cfg, 4,
            jnp.ones(4), None)
    with pytest.raises(ValueError, match="only consumed by the rrtmgp"):
        resolve_land_radiative_bc(
            {"T_sfc_rad": T0, "sfc_emissivity": jnp.ones(3)},
            RadiationConfig(scheme="gray"), 3, T0, None)
