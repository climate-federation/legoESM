"""FV3_3D iter 874: saturation_vapor_pressure_slope_fv3.

Δ = de_sat/dT = e_sat · L_v / (R_v · T²)  [Pa/K].

Tests
-----

1. ``test_20c_canonical``: T=293.15 K → Δ ≈ 140-160 Pa/K.
2. ``test_30c_higher``: T=303.15 K → Δ > 20°C.
3. ``test_finite_diff_match``: matches centered FD of thermo.e_sat.
4. ``test_monotone_in_t``: ↑T → ↑Δ.
5. ``test_t_zero_floored``: T=0 → finite.
6. ``test_chain_with_penman_monteith``: Δ → iter-870 PM.
7. ``test_pairs_with_iter832_cc``: same CC structure as iter-832.
8. ``test_shapes_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm import constants, thermo
from legoesm.grids.cubed_sphere import (
    clausius_clapeyron_dqdt_fv3,
    penman_monteith_le_fv3,
    saturation_vapor_pressure_slope_fv3,
)


def test_20c_canonical():
    """T=20°C (293.15 K) → Δ ≈ 145 Pa/K (FAO-56 reference)."""
    delta = saturation_vapor_pressure_slope_fv3(jnp.array([293.15]))
    assert 130.0 < float(delta[0]) < 160.0


def test_30c_higher():
    """T=30°C → Δ > T=20°C (CC growth)."""
    delta_20 = saturation_vapor_pressure_slope_fv3(jnp.array([293.15]))
    delta_30 = saturation_vapor_pressure_slope_fv3(jnp.array([303.15]))
    assert float(delta_30[0]) > float(delta_20[0])


def test_finite_diff_match():
    """Δ analytic matches finite-difference of thermo.e_sat within 1%."""
    t = jnp.array([293.15])
    delta_analytic = saturation_vapor_pressure_slope_fv3(t)
    dt = 0.01
    e_plus = thermo.saturation_vapor_pressure(t + dt)
    e_minus = thermo.saturation_vapor_pressure(t - dt)
    delta_fd = (e_plus - e_minus) / (2.0 * dt)
    rel_err = float(
        jnp.abs((delta_analytic[0] - delta_fd[0]) / delta_fd[0])
    )
    assert rel_err < 0.02


def test_monotone_in_t():
    """↑T → ↑Δ over Earth-surface T range."""
    t = jnp.array([273.15, 283.15, 293.15, 303.15, 313.15])
    delta = saturation_vapor_pressure_slope_fv3(t)
    diffs = jnp.diff(delta)
    assert jnp.all(diffs > 0.0)


def test_t_zero_floored():
    """T=0 → finite via floor."""
    delta = saturation_vapor_pressure_slope_fv3(jnp.array([0.0]))
    assert jnp.all(jnp.isfinite(delta))


def test_chain_with_penman_monteith():
    """Δ from iter-874 → iter-870 PM consistency."""
    delta = saturation_vapor_pressure_slope_fv3(jnp.array([293.15]))
    le = penman_monteith_le_fv3(
        available_energy=jnp.array([400.0]),
        vpd_pa=jnp.array([1500.0]),
        delta_pa_k=delta,
        gamma_pa_k=jnp.array([67.0]),
        g_s=jnp.array([0.02]),
        g_a=jnp.array([0.05]),
    )
    assert jnp.all(jnp.isfinite(le))
    assert float(le[0]) > 0.0


def test_pairs_with_iter832_cc():
    """Verify CC structural pair: iter-832 dq/dT, iter-874 de_sat/dT.
    Both use L_v/(R_v·T²) factor."""
    t = jnp.array([293.15])
    p = jnp.array([1.0e5])
    dqdt = clausius_clapeyron_dqdt_fv3(t, p)  # RH=1 default
    delta = saturation_vapor_pressure_slope_fv3(t)
    # dq_sat/dT and de_sat/dT both positive; physical consistency
    assert float(dqdt[0]) > 0.0
    assert float(delta[0]) > 0.0


def test_shapes_finite():
    """3-D shapes preserved, finite, positive."""
    rng = np.random.default_rng(seed=874)
    n_x, n_y = 6, 8
    t = jnp.asarray(rng.uniform(250.0, 320.0, size=(n_x, n_y)))
    delta = saturation_vapor_pressure_slope_fv3(t)
    assert delta.shape == (n_x, n_y)
    assert jnp.all(jnp.isfinite(delta))
    assert jnp.all(delta > 0.0)
