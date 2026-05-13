"""FV3_3D iter 880: soil_heat_flux_g_fv3.

G = c_g · R_n  [W/m²].

Tests
-----

1. ``test_daytime_grass``: R_n=600, c_g=0.1 → G=60.
2. ``test_nighttime_release``: R_n=−70 → G<0 (soil releases).
3. ``test_zero_r_n_zero``: R_n=0 → G=0.
4. ``test_daily_mean_zero_cg``: c_g=0 → G=0 (daily-mean grass).
5. ``test_bare_soil_higher``: c_g=0.3 → G > grass c_g=0.1.
6. ``test_chain_full_et``: full chain to iter-870 with G.
7. ``test_shapes_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import (
    aerodynamic_conductance_fv3,
    net_radiation_fv3,
    penman_monteith_le_fv3,
    psychrometric_constant_fv3,
    saturation_vapor_pressure_slope_fv3,
    soil_heat_flux_g_fv3,
    vpd_from_t_rh_fv3,
)


def test_daytime_grass():
    """R_n=600, c_g=0.1 → G=60."""
    g = soil_heat_flux_g_fv3(jnp.array([600.0]))
    np.testing.assert_allclose(np.asarray(g), [60.0], rtol=1e-12)


def test_nighttime_release():
    """R_n=−70 → G<0 (soil cools, releases heat upward)."""
    g = soil_heat_flux_g_fv3(jnp.array([-70.0]))
    assert float(g[0]) < 0.0


def test_zero_r_n_zero():
    """R_n=0 → G=0."""
    g = soil_heat_flux_g_fv3(jnp.array([0.0]))
    np.testing.assert_allclose(np.asarray(g), [0.0], atol=1e-14)


def test_daily_mean_zero_cg():
    """c_g=0 (daily-mean grass) → G=0."""
    g = soil_heat_flux_g_fv3(jnp.array([400.0]), c_g=0.0)
    np.testing.assert_allclose(np.asarray(g), [0.0], atol=1e-14)


def test_bare_soil_higher():
    """c_g=0.3 (bare soil) → G > c_g=0.1 (grass)."""
    g_grass = soil_heat_flux_g_fv3(jnp.array([400.0]), c_g=0.1)
    g_soil = soil_heat_flux_g_fv3(jnp.array([400.0]), c_g=0.3)
    assert float(g_soil[0]) > float(g_grass[0])


def test_chain_full_et():
    """Full ET chain: R_n → iter-880 G → A = R_n − G → iter-870 PM."""
    r_n = net_radiation_fv3(
        sw_down=jnp.array([800.0]),
        albedo=jnp.array([0.23]),
        lw_down=jnp.array([400.0]),
        lw_up=jnp.array([450.0]),
    )
    g = soil_heat_flux_g_fv3(r_n)
    a = r_n - g
    t = jnp.array([298.15])
    rh = jnp.array([0.5])
    p = jnp.array([101325.0])
    vpd = vpd_from_t_rh_fv3(t, rh)
    gamma = psychrometric_constant_fv3(p)
    delta = saturation_vapor_pressure_slope_fv3(t)
    g_a = aerodynamic_conductance_fv3(
        jnp.array([3.0]), jnp.array([2.0]), jnp.array([2.0]),
        jnp.array([0.08]), jnp.array([0.0148]), jnp.array([0.00148]),
    )
    le = penman_monteith_le_fv3(
        a, vpd, delta, gamma, jnp.array([0.02]), g_a,
    )
    assert jnp.all(jnp.isfinite(le))
    assert float(le[0]) > 0.0


def test_shapes_finite():
    """3-D shapes preserved, finite."""
    rng = np.random.default_rng(seed=880)
    n_x, n_y = 6, 8
    r_n = jnp.asarray(rng.uniform(-100.0, 700.0, size=(n_x, n_y)))
    g = soil_heat_flux_g_fv3(r_n)
    assert g.shape == (n_x, n_y)
    assert jnp.all(jnp.isfinite(g))
