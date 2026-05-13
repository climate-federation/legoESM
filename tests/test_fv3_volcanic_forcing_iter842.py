"""FV3_3D iter 842: volcanic_forcing_fv3.

ΔF_volc = − 25 · τ_strat.

Tests
-----

1. ``test_pinatubo``: τ=0.15 → ΔF ≈ −3.75 W/m².
2. ``test_tambora``: τ=0.50 → ΔF ≈ −12.5 W/m².
3. ``test_quiescent_background``: τ=0.005 → ΔF ≈ −0.125 W/m².
4. ``test_no_eruption_zero``: τ=0 → ΔF=0.
5. ``test_chain_to_ecs``: Pinatubo → ΔF → ECS ≈ −2.68 K cooling.
6. ``test_paired_natural_forcing``: Maunder + Tambora-style →
   combined natural forcing < GHG counter-forcing.
7. ``test_custom_alpha_volc``: alpha_volc=21 (AR5 low estimate)
   gives proportionally smaller |ΔF|.
8. ``test_shapes_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import (
    equilibrium_climate_sensitivity_fv3,
    solar_forcing_fv3,
    volcanic_forcing_fv3,
)


def test_pinatubo():
    """τ=0.15 (Pinatubo 1991 peak) → ΔF ≈ −3.75 W/m²."""
    tau = jnp.array([0.15])
    f = volcanic_forcing_fv3(tau)
    np.testing.assert_allclose(np.asarray(f), [-3.75], rtol=1e-12)


def test_tambora():
    """τ=0.50 (Tambora 1815 estimate) → ΔF ≈ −12.5 W/m²."""
    tau = jnp.array([0.50])
    f = volcanic_forcing_fv3(tau)
    np.testing.assert_allclose(np.asarray(f), [-12.5], rtol=1e-12)


def test_quiescent_background():
    """τ=0.005 (quiescent background) → ΔF ≈ −0.125 W/m²."""
    tau = jnp.array([0.005])
    f = volcanic_forcing_fv3(tau)
    np.testing.assert_allclose(np.asarray(f), [-0.125], rtol=1e-12)


def test_no_eruption_zero():
    """τ=0 → ΔF=0."""
    tau = jnp.array([0.0])
    f = volcanic_forcing_fv3(tau)
    np.testing.assert_allclose(np.asarray(f), [0.0], atol=1e-14)


def test_chain_to_ecs():
    """Pinatubo (τ=0.15) → ΔF=−3.75 → ECS ≈ −2.68 K equilibrium."""
    tau = jnp.array([0.15])
    f = volcanic_forcing_fv3(tau)
    lam = jnp.array([-1.4])
    # ECS = |ΔF|/|λ| = 3.75/1.4 ≈ 2.68 K
    delta_t = equilibrium_climate_sensitivity_fv3(f, lam)
    # Note iter-836 returns ΔF/|λ|; for negative ΔF, ΔT_eq < 0
    assert -2.8 < float(delta_t[0]) < -2.5


def test_paired_natural_forcing():
    """Solar+volcanic combined: Maunder ΔTSI=−1 + Tambora τ=0.50."""
    f_solar = solar_forcing_fv3(jnp.array([-1.0]))
    f_volc = volcanic_forcing_fv3(jnp.array([0.50]))
    total = f_solar + f_volc
    # −0.175 + (−12.5) = −12.675
    assert -13.0 < float(total[0]) < -12.5


def test_custom_alpha_volc():
    """alpha_volc=21 (AR5 low) → |ΔF| smaller than default 25."""
    tau = jnp.array([0.15])
    f_default = volcanic_forcing_fv3(tau)
    f_ar5_low = volcanic_forcing_fv3(tau, alpha_volc=21.0)
    assert abs(float(f_ar5_low[0])) < abs(float(f_default[0]))


def test_shapes_finite():
    """3-D shapes preserved, finite, non-positive (τ≥0)."""
    rng = np.random.default_rng(seed=842)
    n_x, n_y = 6, 8
    tau = jnp.asarray(rng.uniform(0.0, 1.0, size=(n_x, n_y)))
    f = volcanic_forcing_fv3(tau)
    assert f.shape == (n_x, n_y)
    assert jnp.all(jnp.isfinite(f))
    assert jnp.all(f <= 0.0)
