"""FV3_3D iter 873: psychrometric_constant_fv3.

γ = c_p · p / (ε · λ_v)  [Pa/K].

Tests
-----

1. ``test_sea_level_fao56``: p=101325 Pa → γ ≈ 67 Pa/K.
2. ``test_high_altitude_decrease``: ↑elevation → ↓γ.
3. ``test_zero_pressure_zero``: p=0 → γ=0.
4. ``test_linear_in_pressure``: 2× p → 2× γ.
5. ``test_chain_with_penman_monteith``: γ → iter-870 PM.
6. ``test_custom_overrides``: explicit c_p, L_v, epsilon work.
7. ``test_shapes_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.grids.cubed_sphere import (
    penman_monteith_le_fv3,
    psychrometric_constant_fv3,
)


def test_sea_level_fao56():
    """p=101325 Pa → γ ≈ 67 Pa/K (FAO-56 canonical)."""
    gamma = psychrometric_constant_fv3(jnp.array([101325.0]))
    # γ = 1004.64 × 101325 / (0.622 × 2.501e6) = 65.4
    expected = (constants.c_pd * 101325.0
                / (constants.epsilon * constants.L_v))
    np.testing.assert_allclose(np.asarray(gamma), [expected], rtol=1e-12)
    # FAO-56 reports γ_sea ≈ 67; our value ~65 (slight L_v sensitivity)
    assert 60.0 < float(gamma[0]) < 70.0


def test_high_altitude_decrease():
    """↑elevation (↓p) → ↓γ."""
    gamma_sea = psychrometric_constant_fv3(jnp.array([101325.0]))
    gamma_3km = psychrometric_constant_fv3(jnp.array([70110.0]))
    gamma_5_5km = psychrometric_constant_fv3(jnp.array([50500.0]))
    assert float(gamma_3km[0]) < float(gamma_sea[0])
    assert float(gamma_5_5km[0]) < float(gamma_3km[0])


def test_zero_pressure_zero():
    """p=0 → γ=0."""
    gamma = psychrometric_constant_fv3(jnp.array([0.0]))
    np.testing.assert_allclose(np.asarray(gamma), [0.0], atol=1e-14)


def test_linear_in_pressure():
    """2× p → 2× γ."""
    g1 = psychrometric_constant_fv3(jnp.array([50000.0]))
    g2 = psychrometric_constant_fv3(jnp.array([100000.0]))
    np.testing.assert_allclose(np.asarray(g2), 2.0 * np.asarray(g1), rtol=1e-12)


def test_chain_with_penman_monteith():
    """γ → iter-870 PM consistency check."""
    gamma = psychrometric_constant_fv3(jnp.array([101325.0]))
    le = penman_monteith_le_fv3(
        available_energy=jnp.array([400.0]),
        vpd_pa=jnp.array([1500.0]),
        delta_pa_k=jnp.array([180.0]),
        gamma_pa_k=gamma,
        g_s=jnp.array([0.02]),
        g_a=jnp.array([0.05]),
    )
    assert jnp.all(jnp.isfinite(le))
    assert float(le[0]) > 0.0


def test_custom_overrides():
    """Explicit c_p, L_v, epsilon work."""
    gamma = psychrometric_constant_fv3(
        jnp.array([101325.0]),
        c_p=1010.0,
        l_v=2.5e6,
        epsilon=0.622,
    )
    expected = 1010.0 * 101325.0 / (0.622 * 2.5e6)
    np.testing.assert_allclose(np.asarray(gamma), [expected], rtol=1e-12)


def test_shapes_finite():
    """3-D shapes preserved, finite, positive."""
    rng = np.random.default_rng(seed=873)
    n_x, n_y = 6, 8
    p = jnp.asarray(rng.uniform(40000.0, 105000.0, size=(n_x, n_y)))
    gamma = psychrometric_constant_fv3(p)
    assert gamma.shape == (n_x, n_y)
    assert jnp.all(jnp.isfinite(gamma))
    assert jnp.all(gamma > 0.0)
