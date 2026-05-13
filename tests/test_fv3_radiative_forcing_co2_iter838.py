"""FV3_3D iter 838: radiative_forcing_co2_fv3.

ΔF_CO₂ = 5.35 · ln(C / C_ref).

Tests
-----

1. ``test_2xco2_ar5_canonical``: C=556, C_ref=278 → ΔF ≈ 3.71 W/m².
2. ``test_4xco2``: C=1112 → ΔF ≈ 7.42 W/m².
3. ``test_no_change_zero``: C = C_ref → ΔF = 0.
4. ``test_cooling_negative``: C < C_ref → ΔF < 0.
5. ``test_present_day``: C=420 → ΔF ≈ 2.21 W/m².
6. ``test_chain_to_ecs``: ppm → ΔF → ECS via iter-836 (≈2.64 K).
7. ``test_chain_to_tcr``: ppm → ΔF → TCR via iter-837 (≈1.76 K).
8. ``test_floor_no_nan``: C=0 → finite.
9. ``test_shapes_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import (
    equilibrium_climate_sensitivity_fv3,
    radiative_forcing_co2_fv3,
    transient_climate_response_fv3,
)


def test_2xco2_ar5_canonical():
    """C=556, C_ref=278 → ΔF = 5.35·ln(2) ≈ 3.71 W/m²."""
    c = jnp.array([556.0])
    f = radiative_forcing_co2_fv3(c)
    expected = 5.35 * float(jnp.log(2.0))
    np.testing.assert_allclose(np.asarray(f), [expected], rtol=1e-12)
    assert 3.6 < float(f[0]) < 3.8


def test_4xco2():
    """C=1112 → ΔF ≈ 7.42 W/m² (2× of 2×CO₂)."""
    c = jnp.array([1112.0])
    f = radiative_forcing_co2_fv3(c)
    assert 7.3 < float(f[0]) < 7.5


def test_no_change_zero():
    """C = C_ref → ΔF = 0."""
    c = jnp.array([278.0])
    f = radiative_forcing_co2_fv3(c)
    np.testing.assert_allclose(np.asarray(f), [0.0], atol=1e-13)


def test_cooling_negative():
    """C < C_ref → ΔF < 0 (cooling)."""
    c = jnp.array([200.0])
    f = radiative_forcing_co2_fv3(c)
    assert float(f[0]) < 0.0


def test_present_day():
    """C=420 (2025-ish), C_ref=278 → ΔF ≈ 2.21 W/m²."""
    c = jnp.array([420.0])
    f = radiative_forcing_co2_fv3(c)
    assert 2.1 < float(f[0]) < 2.3


def test_chain_to_ecs():
    """ppm → ΔF → ECS via iter-836 (2×CO₂, |λ|=1.4 → ~2.64 K)."""
    c = jnp.array([556.0])
    f = radiative_forcing_co2_fv3(c)
    lam = jnp.array([-1.4])
    ecs = equilibrium_climate_sensitivity_fv3(f, lam)
    assert 2.5 < float(ecs[0]) < 2.8


def test_chain_to_tcr():
    """ppm → ΔF → TCR via iter-837 (2×CO₂, |λ|=1.4, γ=0.7 → ~1.77 K)."""
    c = jnp.array([556.0])
    f = radiative_forcing_co2_fv3(c)
    lam = jnp.array([-1.4])
    gamma = jnp.array([0.7])
    tcr = transient_climate_response_fv3(f, lam, gamma)
    assert 1.7 < float(tcr[0]) < 1.8


def test_floor_no_nan():
    """C=0 → finite via co2_floor (no NaN)."""
    c = jnp.array([0.0])
    f = radiative_forcing_co2_fv3(c)
    assert jnp.all(jnp.isfinite(f))


def test_shapes_finite():
    """3-D shapes preserved, finite."""
    rng = np.random.default_rng(seed=838)
    n_x, n_y = 6, 8
    c = jnp.asarray(rng.uniform(150.0, 1200.0, size=(n_x, n_y)))
    f = radiative_forcing_co2_fv3(c)
    assert f.shape == (n_x, n_y)
    assert jnp.all(jnp.isfinite(f))
