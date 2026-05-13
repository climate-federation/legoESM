"""FV3_3D iter 847: airborne_fraction_co2_fv3.

ΔCO₂_atm = AF · E_cum / 7.81  [ppm per Gt-CO₂].

Tests
-----

1. ``test_ar6_present_day``: E=2400 Gt-CO₂, AF=0.46 → ~141 ppm.
2. ``test_zero_emission_zero``: E=0 → ΔCO₂=0.
3. ``test_full_af_extreme``: AF=1.0 → no uptake limit.
4. ``test_no_af_extreme``: AF=0 → no atmospheric gain.
5. ``test_chain_to_co2_forcing``: emission → ΔCO₂ → iter-838 ΔF_CO₂.
6. ``test_chain_to_ecs_via_concentration``: emission → ΔCO₂ → ΔF → ECS.
7. ``test_negative_emission_drawdown``: E<0 (CDR) → ΔCO₂<0.
8. ``test_shapes_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import (
    airborne_fraction_co2_fv3,
    equilibrium_climate_sensitivity_fv3,
    radiative_forcing_co2_fv3,
)


def test_ar6_present_day():
    """E_cum=2400 Gt-CO₂, AF=0.46 → ~141 ppm (1750→2020 anthropogenic)."""
    e = jnp.array([2400.0])
    delta = airborne_fraction_co2_fv3(e)
    expected = 0.46 * 2400.0 / 7.81
    np.testing.assert_allclose(np.asarray(delta), [expected], rtol=1e-12)
    assert 135.0 < float(delta[0]) < 145.0


def test_zero_emission_zero():
    """E=0 → ΔCO₂=0."""
    e = jnp.array([0.0])
    delta = airborne_fraction_co2_fv3(e)
    np.testing.assert_allclose(np.asarray(delta), [0.0], atol=1e-14)


def test_full_af_extreme():
    """AF=1.0 (no uptake) → ΔCO₂ = E/7.81."""
    e = jnp.array([100.0])
    delta = airborne_fraction_co2_fv3(e, airborne_fraction=jnp.array([1.0]))
    np.testing.assert_allclose(np.asarray(delta), [100.0/7.81], rtol=1e-12)


def test_no_af_extreme():
    """AF=0 (full uptake) → ΔCO₂=0."""
    e = jnp.array([100.0])
    delta = airborne_fraction_co2_fv3(e, airborne_fraction=jnp.array([0.0]))
    np.testing.assert_allclose(np.asarray(delta), [0.0], atol=1e-14)


def test_chain_to_co2_forcing():
    """Emission → ΔCO₂ → iter-838 ΔF_CO₂."""
    e = jnp.array([2400.0])
    delta_co2 = airborne_fraction_co2_fv3(e)
    # 278 + 141 = 419 ppm (present-day)
    c_now = 278.0 + delta_co2
    f = radiative_forcing_co2_fv3(c_now)
    # AR6 says ΔF from CO₂ alone is ~2.2 W/m² (present-day)
    assert 2.1 < float(f[0]) < 2.3


def test_chain_to_ecs_via_concentration():
    """Emission → ΔCO₂ → ΔF → ECS (concentration pathway)."""
    e = jnp.array([2400.0])
    delta_co2 = airborne_fraction_co2_fv3(e)
    c_now = 278.0 + delta_co2
    f = radiative_forcing_co2_fv3(c_now)
    lam = jnp.array([-1.4])
    ecs = equilibrium_climate_sensitivity_fv3(f, lam)
    # ΔT ≈ 2.21 / 1.4 ≈ 1.58 K (CO₂-only EQ warming, no aerosol offset)
    assert 1.4 < float(ecs[0]) < 1.8


def test_negative_emission_drawdown():
    """E<0 (CDR/CCS) → ΔCO₂<0 (atmospheric drawdown)."""
    e = jnp.array([-500.0])
    delta = airborne_fraction_co2_fv3(e)
    assert float(delta[0]) < 0.0


def test_shapes_finite():
    """3-D shapes preserved, finite."""
    rng = np.random.default_rng(seed=847)
    n_x, n_y = 6, 8
    e = jnp.asarray(rng.uniform(0.0, 5000.0, size=(n_x, n_y)))
    af = jnp.asarray(rng.uniform(0.3, 0.6, size=(n_x, n_y)))
    delta = airborne_fraction_co2_fv3(e, af)
    assert delta.shape == (n_x, n_y)
    assert jnp.all(jnp.isfinite(delta))
