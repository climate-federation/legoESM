"""FV3_3D iter 793: eady_growth_rate_fv3 (σ = 0.31·|f|·|∂V_g/∂z|/N).

Composes iter-772 N + iter-778 f + iter-792 ∂V_g/∂z.

Tests
-----

1. ``test_eady_midlat_textbook``: N=0.01, |f|=1e-4, ∂u=3e-3 → σ≈9.3e-6.
2. ``test_eady_zero_shear``: ∂V_g/∂z=0 → σ=0.
3. ``test_eady_monotonic_inputs``: ↑|f|→↑σ, ↑shear→↑σ, ↑N→↓σ.
4. ``test_eady_2d_shear_magnitude``: dv_g_dz arg increases σ.
5. ``test_eady_unstratified_floored``: N=0 → finite via N_floor.
6. ``test_eady_composes_iter792``: full pipeline (∇T, lat, T) → ∂V_g/∂z → σ.
7. ``test_eady_shapes_3d_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import (
    coriolis_parameter_fv3,
    eady_growth_rate_fv3,
    thermal_wind_fv3,
)


def test_eady_midlat_textbook():
    """Mid-lat: σ = 0.31·1e-4·3e-3/0.01 = 9.3e-6 s⁻¹ → τ ≈ 1.2 days."""
    N = jnp.array([0.01])
    f = jnp.array([1.0e-4])
    du = jnp.array([3.0e-3])
    sigma = eady_growth_rate_fv3(N, f, du)
    expected = 0.31 * 1.0e-4 * 3.0e-3 / 0.01
    np.testing.assert_allclose(np.asarray(sigma), [expected], rtol=1e-12)
    # τ = 1/σ ≈ 107000 s ≈ 1.24 days
    tau_days = 1.0 / float(sigma[0]) / 86400.0
    assert 0.5 < tau_days < 3.0


def test_eady_zero_shear():
    """∂V_g/∂z=0 → σ=0."""
    N = jnp.array([0.01, 0.005, 0.02])
    f = jnp.array([1e-4, -1e-4, 5e-5])
    du = jnp.zeros((3,))
    sigma = eady_growth_rate_fv3(N, f, du)
    np.testing.assert_allclose(np.asarray(sigma), jnp.zeros((3,)), atol=1e-15)


def test_eady_monotonic_inputs():
    """↑|f| → ↑σ; ↑shear → ↑σ; ↑N → ↓σ."""
    base_N = jnp.array([0.01, 0.01, 0.01])
    base_f = jnp.array([1e-4, 1e-4, 1e-4])
    base_du = jnp.array([1e-3, 2e-3, 3e-3])
    sigma_base = eady_growth_rate_fv3(base_N, base_f, base_du)

    sigma_f_hi = eady_growth_rate_fv3(base_N, base_f * 2.0, base_du)
    assert jnp.all(sigma_f_hi > sigma_base)

    sigma_du_hi = eady_growth_rate_fv3(base_N, base_f, base_du * 2.0)
    assert jnp.all(sigma_du_hi > sigma_base)

    sigma_N_hi = eady_growth_rate_fv3(base_N * 2.0, base_f, base_du)
    assert jnp.all(sigma_N_hi < sigma_base)


def test_eady_2d_shear_magnitude():
    """With dv_g_dz arg, shear magnitude is √(du² + dv²) → larger σ."""
    N = jnp.array([0.01])
    f = jnp.array([1.0e-4])
    du = jnp.array([3.0e-3])
    dv = jnp.array([4.0e-3])
    sigma_1d = eady_growth_rate_fv3(N, f, du)
    sigma_2d = eady_growth_rate_fv3(N, f, du, dv)
    # |shear| = √(9 + 16)e-3 = 5e-3 vs 3e-3 → σ_2d = (5/3)·σ_1d
    np.testing.assert_allclose(
        np.asarray(sigma_2d), (5.0 / 3.0) * np.asarray(sigma_1d), rtol=1e-12
    )


def test_eady_unstratified_floored():
    """N=0 → σ huge but finite via N_floor."""
    N = jnp.array([0.0])
    f = jnp.array([1.0e-4])
    du = jnp.array([3.0e-3])
    sigma = eady_growth_rate_fv3(N, f, du, N_floor=1e-6)
    assert jnp.all(jnp.isfinite(sigma))
    # 0.31·1e-4·3e-3 / 1e-6 = 0.093 (huge vs unfloored ~1/0 → ∞)
    assert float(sigma[0]) > 0.01


def test_eady_composes_iter792():
    """Pipeline (∇T, lat, T) → ∂V_g/∂z → σ."""
    dT_dx = jnp.array([0.0])
    dT_dy = jnp.array([-1e-5])  # cold pole
    f = coriolis_parameter_fv3(jnp.array([45.0]), units="deg")
    T = jnp.array([250.0])
    du, dv = thermal_wind_fv3(dT_dx, dT_dy, f, T)
    N = jnp.array([0.01])
    sigma = eady_growth_rate_fv3(N, f, du, dv)
    # Expect O(1e-6) to 1e-5 s⁻¹ growth
    assert 1e-7 < float(sigma[0]) < 1e-4
    assert jnp.all(jnp.isfinite(sigma))


def test_eady_shapes_3d_finite():
    """3-D shapes preserved, finite, non-negative."""
    rng = np.random.default_rng(seed=793)
    n_x, n_y, km = 4, 5, 20
    N = jnp.asarray(rng.uniform(0.005, 0.02, size=(n_x, n_y, km)))
    f = jnp.asarray(rng.uniform(-1.5e-4, 1.5e-4, size=(n_x, n_y, km)))
    du = jnp.asarray(rng.uniform(-5e-3, 5e-3, size=(n_x, n_y, km)))
    dv = jnp.asarray(rng.uniform(-5e-3, 5e-3, size=(n_x, n_y, km)))
    sigma = eady_growth_rate_fv3(N, f, du, dv)
    assert sigma.shape == (n_x, n_y, km)
    assert jnp.all(jnp.isfinite(sigma))
    assert jnp.all(sigma >= 0.0)
