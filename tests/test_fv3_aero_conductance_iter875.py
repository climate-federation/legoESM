"""FV3_3D iter 875: aerodynamic_conductance_fv3.

g_a = u·k²/(ln((z_m−d)/z_0m) · ln((z_h−d)/z_0h))  [m/s].

Tests
-----

1. ``test_fao56_grass_u2``: u=2, h=0.12 → g_a ≈ u_2/208 ≈ 0.0096 m/s.
2. ``test_zero_wind_zero``: u=0 → g_a=0.
3. ``test_increases_with_wind``: ↑u → ↑g_a.
4. ``test_forest_geometry``: tall canopy → larger g_a per u.
5. ``test_log_argument_floor``: (z-d)/z_0 ≤ 1 → finite (no NaN).
6. ``test_chain_with_penman_monteith``: g_a → iter-870 PM.
7. ``test_full_pm_pipeline``: full pure-JAX ET chain end-to-end.
8. ``test_shapes_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import (
    aerodynamic_conductance_fv3,
    penman_monteith_le_fv3,
    psychrometric_constant_fv3,
    saturation_vapor_pressure_slope_fv3,
    vpd_from_t_rh_fv3,
)


def test_fao56_grass_u2():
    """FAO-56 grass at u_2=2 m/s → g_a ≈ 0.0096 m/s (r_a ≈ 208/u_2)."""
    g_a = aerodynamic_conductance_fv3(
        u_z=jnp.array([2.0]),
        z_m=jnp.array([2.0]),
        z_h=jnp.array([2.0]),
        d=jnp.array([0.08]),
        z_0m=jnp.array([0.0148]),
        z_0h=jnp.array([0.00148]),
    )
    # g_a ≈ 2 / 208 = 0.00962 m/s
    assert 0.008 < float(g_a[0]) < 0.012


def test_zero_wind_zero():
    """u=0 → g_a=0 (linear in u)."""
    g_a = aerodynamic_conductance_fv3(
        u_z=jnp.array([0.0]),
        z_m=jnp.array([2.0]),
        z_h=jnp.array([2.0]),
        d=jnp.array([0.08]),
        z_0m=jnp.array([0.0148]),
        z_0h=jnp.array([0.00148]),
    )
    np.testing.assert_allclose(np.asarray(g_a), [0.0], atol=1e-14)


def test_increases_with_wind():
    """↑u → ↑g_a."""
    u = jnp.array([1.0, 2.0, 5.0, 10.0])
    z_m = jnp.full((4,), 2.0)
    z_h = jnp.full((4,), 2.0)
    d = jnp.full((4,), 0.08)
    z_0m = jnp.full((4,), 0.0148)
    z_0h = jnp.full((4,), 0.00148)
    g_a = aerodynamic_conductance_fv3(u, z_m, z_h, d, z_0m, z_0h)
    diffs = jnp.diff(g_a)
    assert jnp.all(diffs > 0.0)


def test_forest_geometry():
    """Tall forest canopy → larger g_a per unit u (rougher surface)."""
    g_grass = aerodynamic_conductance_fv3(
        u_z=jnp.array([2.0]),
        z_m=jnp.array([2.0]),
        z_h=jnp.array([2.0]),
        d=jnp.array([0.08]),
        z_0m=jnp.array([0.0148]),
        z_0h=jnp.array([0.00148]),
    )
    # 20-m forest: d=13.3, z_0m=2.46, z_0h=0.246; measurement at 25 m
    g_forest = aerodynamic_conductance_fv3(
        u_z=jnp.array([2.0]),
        z_m=jnp.array([25.0]),
        z_h=jnp.array([25.0]),
        d=jnp.array([13.3]),
        z_0m=jnp.array([2.46]),
        z_0h=jnp.array([0.246]),
    )
    assert float(g_forest[0]) > float(g_grass[0])


def test_log_argument_floor():
    """(z-d)/z_0 ≤ 1 degenerate → finite via floor."""
    g_a = aerodynamic_conductance_fv3(
        u_z=jnp.array([2.0]),
        z_m=jnp.array([0.1]),
        z_h=jnp.array([0.1]),
        d=jnp.array([0.5]),  # d > z, degenerate
        z_0m=jnp.array([0.01]),
        z_0h=jnp.array([0.001]),
    )
    assert jnp.all(jnp.isfinite(g_a))


def test_chain_with_penman_monteith():
    """g_a → iter-870 PM consistency."""
    g_a = aerodynamic_conductance_fv3(
        u_z=jnp.array([2.0]),
        z_m=jnp.array([2.0]),
        z_h=jnp.array([2.0]),
        d=jnp.array([0.08]),
        z_0m=jnp.array([0.0148]),
        z_0h=jnp.array([0.00148]),
    )
    le = penman_monteith_le_fv3(
        available_energy=jnp.array([400.0]),
        vpd_pa=jnp.array([1500.0]),
        delta_pa_k=jnp.array([180.0]),
        gamma_pa_k=jnp.array([67.0]),
        g_s=jnp.array([0.02]),
        g_a=g_a,
    )
    assert jnp.all(jnp.isfinite(le))


def test_full_pm_pipeline():
    """Full pure-JAX ET pipeline iter-871/873/874/875 → iter-870."""
    t = jnp.array([298.15])           # 25°C
    rh = jnp.array([0.6])
    p = jnp.array([101325.0])
    u = jnp.array([3.0])
    a_avail = jnp.array([400.0])
    g_s = jnp.array([0.02])
    vpd = vpd_from_t_rh_fv3(t, rh)
    gamma = psychrometric_constant_fv3(p)
    delta = saturation_vapor_pressure_slope_fv3(t)
    g_a = aerodynamic_conductance_fv3(
        u, jnp.array([2.0]), jnp.array([2.0]),
        jnp.array([0.08]), jnp.array([0.0148]), jnp.array([0.00148]),
    )
    le = penman_monteith_le_fv3(a_avail, vpd, delta, gamma, g_s, g_a)
    assert jnp.all(jnp.isfinite(le))
    assert float(le[0]) > 0.0


def test_shapes_finite():
    """3-D shapes preserved, finite."""
    rng = np.random.default_rng(seed=875)
    n_x, n_y = 6, 8
    u = jnp.asarray(rng.uniform(0.5, 10.0, size=(n_x, n_y)))
    z_m = jnp.full((n_x, n_y), 2.0)
    z_h = jnp.full((n_x, n_y), 2.0)
    d = jnp.full((n_x, n_y), 0.08)
    z_0m = jnp.full((n_x, n_y), 0.0148)
    z_0h = jnp.full((n_x, n_y), 0.00148)
    g_a = aerodynamic_conductance_fv3(u, z_m, z_h, d, z_0m, z_0h)
    assert g_a.shape == (n_x, n_y)
    assert jnp.all(jnp.isfinite(g_a))
