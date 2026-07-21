"""Tests for the AERK water + ice saturation-vapour-pressure curve.

``legoesm.thermo.saturation_vapor_pressure_aerk`` (and its analytic first/second
derivatives) is the Alduchov & Eskridge (1996) over-water + over-ice Magnus blend
ported to match the DifferBESS two-big-leaf canopy.  Unlike the plain over-water
``saturation_vapor_pressure`` (Bolton), it branches to ice below freezing — the
key reason for adding it.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.thermo import (
    saturation_vapor_pressure,            # Bolton over-water (reference)
    saturation_vapor_pressure_aerk,
    d_saturation_vapor_pressure_aerk,
    dd_saturation_vapor_pressure_aerk,
)

_K = constants.T_freeze


def test_aerk_positive_and_finite():
    T = jnp.linspace(-60.0, 50.0, 200) + _K
    es = saturation_vapor_pressure_aerk(T)
    assert bool(jnp.all(es > 0.0))
    assert bool(jnp.all(jnp.isfinite(es)))


def test_aerk_warm_matches_water_branch():
    """Well above freezing the blend is the AERK over-water Magnus form."""
    Tc = 25.0
    es = float(saturation_vapor_pressure_aerk(jnp.array(Tc + _K)))
    es_water = 610.94 * jnp.exp(17.625 * Tc / (Tc + 243.04))  # satcurve-ok: independent AERK Eq.21 reference (importing the fn under test would be circular)
    assert jnp.allclose(es, es_water, rtol=1e-4)


def test_aerk_cold_uses_ice_branch_below_water():
    """Below freezing the ice branch gives es well BELOW the over-water curve."""
    for Tc in (-10.0, -20.0, -40.0):
        T = jnp.array(Tc + _K)
        es_ice = float(saturation_vapor_pressure_aerk(T))
        es_water_extrap = float(saturation_vapor_pressure(T))   # Bolton over-water
        es_ice_ref = 611.21 * jnp.exp(22.587 * Tc / (Tc + 273.86))  # satcurve-ok: independent AERKi Eq.23 reference validating the ice branch
        # matches the AERKi ice form (the blend is ~pure ice this cold)
        assert jnp.allclose(es_ice, es_ice_ref, rtol=2e-3)
        # and is materially lower than extrapolating over-water
        assert es_ice < 0.97 * es_water_extrap


def test_aerk_continuous_and_smooth_at_freezing():
    """Water and ice branches nearly coincide at 0 degC -> no kink."""
    lo = float(saturation_vapor_pressure_aerk(jnp.array(_K - 0.001)))
    hi = float(saturation_vapor_pressure_aerk(jnp.array(_K + 0.001)))
    assert abs(hi - lo) < 0.1   # Pa; AERK & AERKi are 611.20 vs 611.21 at 0 C
    # first derivative finite and positive through 0 C
    g = jax.grad(lambda T: saturation_vapor_pressure_aerk(T))
    assert float(g(jnp.array(_K))) > 0.0


def test_aerk_analytic_derivatives_match_autodiff():
    g = jax.grad(lambda T: saturation_vapor_pressure_aerk(T))
    gg = jax.grad(g)
    for Tc in (-40.0, -10.0, -0.5, 0.0, 0.5, 15.0, 30.0):
        T = jnp.array(Tc + _K)
        assert jnp.allclose(d_saturation_vapor_pressure_aerk(T), g(T), rtol=1e-9, atol=1e-10)
        assert jnp.allclose(dd_saturation_vapor_pressure_aerk(T), gg(T), rtol=1e-7, atol=1e-10)
