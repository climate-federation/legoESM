"""Land 2 m air temperature (CLM5 ``t_ref2m``) from the two-leaf canopy's own
surface layer, and its use as the CMIP ``tas`` land leg.

The profile oracle below is an INDEPENDENT scalar transcription of the
similarity integral (Paulson 1970 unstable psi_h, Beljaars & Holtslag 1991
stable psi_h), not a call into the code under test."""
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.land.canopy.stability import screen_level_temperature

pytestmark = pytest.mark.skipif(not jax.config.jax_enable_x64,
                                reason="needs JAX_ENABLE_X64=1")

K = constants.kappa_von_karman


def _psi_h(zeta):
    if zeta < 0.0:
        x = (1.0 - 16.0 * zeta) ** 0.25
        return 2.0 * math.log((1.0 + x * x) / 2.0)
    a, b, c, d = 1.0, 0.667, 5.0, 0.35   # BH91 as published (b rounded)
    return (-(1.0 + 2.0 * a * zeta / 3.0) ** 1.5
            - b * (zeta - c / d) * math.exp(-d * zeta) - b * c / d + 1.0)


ZETAT = 0.465   # CLM5 heat free-convection transition (FrictionVelocityMod)


def _F(z, L, z0h):
    """Integrated heat similarity function, CLM5 FrictionVelocityMod regimes."""
    if z / L < -ZETAT:
        return (math.log(-ZETAT * L / z0h) - _psi_h(-ZETAT) + _psi_h(z0h / L)
                + 0.8 * (ZETAT ** (-1.0 / 3.0) - (-z / L) ** (-1.0 / 3.0)))
    return math.log(z / z0h) - _psi_h(z / L) + _psi_h(z0h / L)


def _oracle(theta, Tc, zl, zeta, z0h, zs=2.0):
    L = zl / zeta
    return Tc + (theta - Tc) * _F(zs + z0h, L, z0h) / _F(zl, L, z0h)


def _screen(theta, Tc, zl, zeta, z0h):
    return float(screen_level_temperature(
        jnp.asarray(theta), jnp.asarray(Tc), jnp.asarray(zl),
        jnp.asarray(zl / zeta), jnp.asarray(z0h)))


@pytest.mark.parametrize("zeta", [-20.0, -2.0, -0.3, -0.05, 0.05, 0.3])
@pytest.mark.parametrize("z0h", [0.05, 1.5])
def test_profile_matches_independent_most(zeta, z0h):
    theta, Tc, zl = 270.0, 262.0, 55.0
    got = _screen(theta, Tc, zl, zeta, z0h)
    np.testing.assert_allclose(got, _oracle(theta, Tc, zl, zeta, z0h),
                               rtol=0.0, atol=1e-9)
    assert min(theta, Tc) <= got <= max(theta, Tc)


def test_neutral_limit_is_log_law():
    theta, Tc, zl, z0h = 285.0, 289.0, 60.0, 0.8
    got = _screen(theta, Tc, zl, 1e-6, z0h)
    log_law = Tc + (theta - Tc) * math.log((2.0 + z0h) / z0h) / math.log(zl / z0h)
    np.testing.assert_allclose(got, log_law, rtol=0.0, atol=1e-5)


def test_reference_at_screen_height_returns_reference():
    z0h = 0.3
    got = _screen(271.0, 260.0, 2.0 + z0h, 0.2, z0h)
    np.testing.assert_allclose(got, 271.0, rtol=0.0, atol=1e-12)


def test_gradient_finite_and_nonzero():
    f = lambda th, tc, zeta: screen_level_temperature(
        th, tc, jnp.asarray(50.0), 50.0 / zeta, jnp.asarray(0.5))
    for zeta in (-0.3, 0.3):
        g = jax.grad(f, argnums=(0, 1, 2))(jnp.asarray(270.0), jnp.asarray(262.0),
                                          jnp.asarray(zeta))
        assert all(np.isfinite(float(x)) and abs(float(x)) > 1e-8 for x in g), g


@pytest.mark.parametrize("dT", [-6.0, 4.0])
def test_screen_profile_uses_the_flux_obukhov_length(dT):
    """temp1 of the screen profile IS the flux solve's heat coefficient:
    1 / (rah * ustar) at the Obukhov length the solve returns for its flux
    (not the post-update stability; few iterations so the two differ)."""
    from legoesm.land.canopy.stability import (
        _temperature_humidity_relation, monin_obukhov_stability)
    zl, z0m, Ta = 45.0, 0.9, 275.0
    ustar, rah, _raw, _uav, zeta, obu = monin_obukhov_stability(
        jnp.asarray(3.0), jnp.asarray(Ta), jnp.asarray(Ta), jnp.asarray(Ta + dT),
        jnp.asarray(3e-3), jnp.asarray(3e-3), jnp.asarray(zl), jnp.asarray(z0m),
        n_iters=2, zeta_cap_width=0.05, return_flux_obu=True)
    ch = float(_temperature_humidity_relation(jnp.asarray(zl), obu, jnp.asarray(z0m)))
    np.testing.assert_allclose(ch * float(ustar), 1.0 / float(rah), rtol=1e-12)
    if dT > 0:   # unstable: the flux and the post-update stability differ
        assert abs(float(zl / obu) - float(zeta)) > 1e-6


def _two_leaf(sw, lw, T_soil):
    from legoesm.core.coupling_fields import AtmToSurface
    from legoesm.land.config import MultiLayerLandConfig
    from legoesm.land.surface_scheme import TwoLeafCanopyConfig
    from legoesm.land.surface_scheme.two_leaf_canopy import (
        compute_two_leaf_canopy_fluxes)
    f = lambda v: jnp.full(1, v)
    forcing = AtmToSurface(
        sw_down=f(sw), lw_down=f(lw), precip_total=f(0.), precip_snow=f(0.),
        T_lowest=f(270.), q_lowest=f(.002), u_lowest=f(4.), v_lowest=f(0.),
        p_lowest=f(99300.), p_surface=f(100000.), rho_lowest=f(1.28),
        cos_zenith=f(.7 if sw > 0 else 0.), co2_ppmv=f(412.),
        has_radiation=f(1.), has_precipitation=f(1.), z_lowest=f(60.))
    return compute_two_leaf_canopy_fluxes(
        T_soil_top=f(T_soil), forcing=forcing,
        canopy_config=TwoLeafCanopyConfig(max_iters=30),
        land_config=MultiLayerLandConfig(), canopy_params=None,
        w_frac_rz=f(.5), wind_speed=f(4.), wind_dir_x=jnp.ones(1),
        wind_dir_y=jnp.zeros(1),
        soil_thermal_fn=lambda G, dt: jnp.full(1, T_soil),
        TgC_override=forcing.T_lowest - constants.T_freeze, dt=600.)


@pytest.mark.parametrize("sw,lw,T_soil", [(0., 200., 262.), (700., 300., 285.)])
def test_two_leaf_exports_screen_temperature_between_air_and_canopy(sw, lw, T_soil):
    out = _two_leaf(sw, lw, T_soil)
    theta = 270.0 + constants.g / constants.c_pd * 60.0
    t2, tc = float(out.T_2m[0]), float(out.T_canopy_air[0])
    assert np.isfinite(t2)
    assert abs(theta - tc) > 1.0
    assert min(theta, tc) + 1e-3 < t2 < max(theta, tc) - 1e-3, (t2, tc, theta)


def test_tas_land_leg_takes_the_land_screen_temperature():
    """Fails on the pre-fix code (no T2m_land): finite entries replace the
    ocean-law land profile exactly, non-finite ones keep it."""
    from types import SimpleNamespace as NS
    from legoesm.driver.diagnostics import DiagnosticCollector
    from legoesm.grids.vertical import create_sigma_coordinate
    nlev = 20
    coord = create_sigma_coordinate(nlev)
    dc = DiagnosticCollector(nlev=nlev, sigma_full=coord.sigma_full,
                             dsigma=coord.dsigma, vcoord=coord)
    state = NS(T=NS(data=jnp.broadcast_to(jnp.linspace(250., 280., nlev), (2, nlev))),
               u=NS(data=jnp.full((2, nlev), 5.)), v=NS(data=jnp.zeros((2, nlev))),
               p_s=NS(data=jnp.full((2,), 1.0e5)))
    q_v = jnp.full((2, nlev), 2e-3)
    kw = dict(T_land=jnp.full((2,), 262.0), land_fraction=jnp.ones((2,)))
    args = (state, q_v, jnp.full((2,), 275.0), jnp.zeros((2,)))
    old = np.asarray(dc._tas_2m(*args, T_ice=271.35, **kw))
    new = np.asarray(dc._tas_2m(*args, T_ice=271.35,
                                T2m_land=jnp.asarray([266.5, jnp.nan]), **kw))
    assert new[0] == 266.5
    assert new[1] == old[1]
