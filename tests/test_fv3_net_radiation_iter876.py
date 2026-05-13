"""FV3_3D iter 876: net_radiation_fv3.

R_n = (1 − α)·SW_down + LW_down − LW_up   [W/m²].

Tests
-----

1. ``test_tropical_noon``: SW=1000, α=0.3, LW_dn=430, LW_up=460 → 670.
2. ``test_night_loss``: SW=0, LW_dn=280, LW_up=350 → R_n=−70.
3. ``test_snow_high_albedo``: α=0.8 → low (1−α)·SW.
4. ``test_no_radiation_zero``: all inputs zero → R_n=0.
5. ``test_full_pm_chain``: R_n → A → iter-870 PM λE.
6. ``test_albedo_scaling``: ↑α → ↓R_n.
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
    vpd_from_t_rh_fv3,
)


def test_tropical_noon():
    """Tropical: SW=1000, α=0.3, LW_dn=430, LW_up=460 → R_n=670."""
    r_n = net_radiation_fv3(
        sw_down=jnp.array([1000.0]),
        albedo=jnp.array([0.3]),
        lw_down=jnp.array([430.0]),
        lw_up=jnp.array([460.0]),
    )
    # 700 + 430 − 460 = 670
    np.testing.assert_allclose(np.asarray(r_n), [670.0], rtol=1e-12)


def test_night_loss():
    """Night clear-sky: SW=0, LW_dn=280, LW_up=350 → R_n=−70."""
    r_n = net_radiation_fv3(
        sw_down=jnp.array([0.0]),
        albedo=jnp.array([0.3]),
        lw_down=jnp.array([280.0]),
        lw_up=jnp.array([350.0]),
    )
    np.testing.assert_allclose(np.asarray(r_n), [-70.0], rtol=1e-12)


def test_snow_high_albedo():
    """α=0.8 (snow) → low net SW; LW_dn=300, LW_up=320 → R_n=180."""
    r_n = net_radiation_fv3(
        sw_down=jnp.array([1000.0]),
        albedo=jnp.array([0.8]),
        lw_down=jnp.array([300.0]),
        lw_up=jnp.array([320.0]),
    )
    # 200 + 300 − 320 = 180
    np.testing.assert_allclose(np.asarray(r_n), [180.0], rtol=1e-12)


def test_no_radiation_zero():
    """All inputs zero → R_n=0."""
    r_n = net_radiation_fv3(
        sw_down=jnp.array([0.0]),
        albedo=jnp.array([0.3]),
        lw_down=jnp.array([0.0]),
        lw_up=jnp.array([0.0]),
    )
    np.testing.assert_allclose(np.asarray(r_n), [0.0], atol=1e-14)


def test_full_pm_chain():
    """R_n → A (assume G=0.1·R_n) → iter-870 PM."""
    r_n = net_radiation_fv3(
        sw_down=jnp.array([800.0]),
        albedo=jnp.array([0.23]),
        lw_down=jnp.array([400.0]),
        lw_up=jnp.array([450.0]),
    )
    g = 0.1 * r_n
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


def test_albedo_scaling():
    """↑α → ↓R_n (more reflection)."""
    r_low = net_radiation_fv3(
        sw_down=jnp.array([1000.0]),
        albedo=jnp.array([0.1]),
        lw_down=jnp.array([400.0]),
        lw_up=jnp.array([450.0]),
    )
    r_high = net_radiation_fv3(
        sw_down=jnp.array([1000.0]),
        albedo=jnp.array([0.6]),
        lw_down=jnp.array([400.0]),
        lw_up=jnp.array([450.0]),
    )
    assert float(r_high[0]) < float(r_low[0])


def test_shapes_finite():
    """3-D shapes preserved, finite."""
    rng = np.random.default_rng(seed=876)
    n_x, n_y = 6, 8
    sw = jnp.asarray(rng.uniform(0.0, 1200.0, size=(n_x, n_y)))
    alpha = jnp.asarray(rng.uniform(0.05, 0.9, size=(n_x, n_y)))
    lw_dn = jnp.asarray(rng.uniform(250.0, 450.0, size=(n_x, n_y)))
    lw_up = jnp.asarray(rng.uniform(300.0, 500.0, size=(n_x, n_y)))
    r_n = net_radiation_fv3(sw, alpha, lw_dn, lw_up)
    assert r_n.shape == (n_x, n_y)
    assert jnp.all(jnp.isfinite(r_n))
