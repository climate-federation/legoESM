"""FV3_3D iter 841: solar_forcing_fv3.

ΔF_solar = (1 − α) · ΔTSI / 4.

Tests
-----

1. ``test_solar_cycle``: ΔTSI=1.0, α=0.30 → ΔF ≈ +0.175 W/m².
2. ``test_maunder_minimum_cools``: ΔTSI<0 → ΔF<0.
3. ``test_no_perturbation_zero``: ΔTSI=0 → ΔF=0.
4. ``test_albedo_one_zero_absorbed``: α=1 → ΔF=0 (full reflection).
5. ``test_disk_to_sphere_factor``: F = (1-α)·TSI/4 exact identity.
6. ``test_chain_to_ecs``: ΔTSI=1.0 → ΔF → ECS for sanity.
7. ``test_faint_young_sun``: ΔTSI=−340 → ΔF ≈ −59.5 (matches paradox).
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
)


def test_solar_cycle():
    """ΔTSI=1.0, α=0.30 → ΔF ≈ +0.175 W/m² (typical 11-yr cycle)."""
    dtsi = jnp.array([1.0])
    f = solar_forcing_fv3(dtsi)
    np.testing.assert_allclose(np.asarray(f), [0.70 / 4.0], rtol=1e-12)
    assert 0.17 < float(f[0]) < 0.18


def test_maunder_minimum_cools():
    """ΔTSI<0 (dim sun) → ΔF<0 (cooling, e.g. Maunder Minimum)."""
    dtsi = jnp.array([-1.0])
    f = solar_forcing_fv3(dtsi)
    assert float(f[0]) < 0.0


def test_no_perturbation_zero():
    """ΔTSI=0 → ΔF=0."""
    dtsi = jnp.array([0.0])
    f = solar_forcing_fv3(dtsi)
    np.testing.assert_allclose(np.asarray(f), [0.0], atol=1e-14)


def test_albedo_one_zero_absorbed():
    """α=1 (mirror planet) → ΔF=0 regardless of ΔTSI."""
    dtsi = jnp.array([5.0])
    f = solar_forcing_fv3(dtsi, albedo=jnp.array([1.0]))
    np.testing.assert_allclose(np.asarray(f), [0.0], atol=1e-14)


def test_disk_to_sphere_factor():
    """F = (1-α)·ΔTSI/4 — exact identity test."""
    dtsi = jnp.array([2.5])
    alpha = jnp.array([0.25])
    f = solar_forcing_fv3(dtsi, alpha)
    expected = (1.0 - 0.25) * 2.5 / 4.0
    np.testing.assert_allclose(np.asarray(f), [expected], rtol=1e-12)


def test_chain_to_ecs():
    """Hypothetical: doubling-equivalent ΔTSI=21.1 (gives ΔF=3.7) →
    ECS sanity at |λ|=1.4."""
    dtsi = jnp.array([21.1428])  # ΔF = 0.7·21.1428/4 = 3.7
    f = solar_forcing_fv3(dtsi)
    lam = jnp.array([-1.4])
    ecs = equilibrium_climate_sensitivity_fv3(f, lam)
    assert 2.5 < float(ecs[0]) < 2.8


def test_faint_young_sun():
    """Archean ΔTSI=−340 (25% dimmer sun): ΔF ≈ −59.5 W/m²
    (Sagan-Mullen paradox magnitude)."""
    dtsi = jnp.array([-340.0])
    f = solar_forcing_fv3(dtsi)
    # (1-0.30)·(-340)/4 = -59.5
    np.testing.assert_allclose(np.asarray(f), [-59.5], rtol=1e-12)


def test_shapes_finite():
    """3-D shapes preserved, finite."""
    rng = np.random.default_rng(seed=841)
    n_x, n_y = 6, 8
    dtsi = jnp.asarray(rng.uniform(-5.0, 5.0, size=(n_x, n_y)))
    alpha = jnp.asarray(rng.uniform(0.0, 0.9, size=(n_x, n_y)))
    f = solar_forcing_fv3(dtsi, alpha)
    assert f.shape == (n_x, n_y)
    assert jnp.all(jnp.isfinite(f))
