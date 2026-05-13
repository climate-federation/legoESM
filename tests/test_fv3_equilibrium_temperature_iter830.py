"""FV3_3D iter 830: equilibrium_temperature_fv3.

T_eq = ((1-α)·S_in / (ε·σ))^(1/4).

Tests
-----

1. ``test_earth_eq``: S=340.25, α=0.30 → T_eq ≈ 255 K.
2. ``test_snowball``: high α → low T_eq.
3. ``test_total_reflect``: α=1 → T_eq → 0 (floored).
4. ``test_lower_emissivity``: ε<1 → higher T_eq.
5. ``test_inverse_iter829``: round-trip with effective_radiating_temperature.
6. ``test_shapes_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.grids.cubed_sphere import (
    effective_radiating_temperature_fv3,
    equilibrium_temperature_fv3,
)


def test_earth_eq():
    """Earth (S=340.25, α=0.30, ε=1) → T_eq ≈ 255 K."""
    s_in = jnp.array([340.25])
    alpha = jnp.array([0.30])
    t_eq = equilibrium_temperature_fv3(s_in, alpha)
    expected = (0.70 * 340.25 / constants.sigma_sb) ** 0.25
    np.testing.assert_allclose(np.asarray(t_eq), [expected], rtol=1e-12)
    assert 253.0 < float(t_eq[0]) < 257.0


def test_snowball():
    """Snowball Earth (α=0.6) → T_eq much lower than Earth (~222 K)."""
    s_in = jnp.array([340.25])
    alpha_snowball = jnp.array([0.6])
    t_eq = equilibrium_temperature_fv3(s_in, alpha_snowball)
    assert 215.0 < float(t_eq[0]) < 230.0


def test_total_reflect():
    """α=1 → T_eq → 0 (floored to small positive)."""
    s_in = jnp.array([340.25])
    alpha = jnp.array([1.0])
    t_eq = equilibrium_temperature_fv3(s_in, alpha, s_floor=1e-6)
    # T_eq = (1e-6/σ)^0.25 ≈ (1e-6/5.67e-8)^0.25 ≈ 2.05 K — floor regime
    assert float(t_eq[0]) < 5.0
    assert jnp.all(jnp.isfinite(t_eq))


def test_lower_emissivity():
    """ε=0.5 → higher T_eq (less efficient emission requires hotter T)."""
    s_in = jnp.array([340.25])
    alpha = jnp.array([0.30])
    t_unit = equilibrium_temperature_fv3(s_in, alpha, emissivity=1.0)
    t_half = equilibrium_temperature_fv3(s_in, alpha, emissivity=0.5)
    assert float(t_half[0]) > float(t_unit[0])


def test_inverse_iter829():
    """Round-trip: T_eq → OLR via Planck → back to T via iter-829."""
    s_in = jnp.array([340.25])
    alpha = jnp.array([0.30])
    t_eq = equilibrium_temperature_fv3(s_in, alpha)
    # At equilibrium: OLR = (1-α)·S_in
    olr = (1.0 - alpha) * s_in
    t_recovered = effective_radiating_temperature_fv3(olr)
    np.testing.assert_allclose(
        np.asarray(t_eq), np.asarray(t_recovered), rtol=1e-12
    )


def test_shapes_finite():
    """3-D shapes preserved, finite, positive."""
    rng = np.random.default_rng(seed=830)
    n_x, n_y = 6, 8
    s_in = jnp.asarray(rng.uniform(50.0, 400.0, size=(n_x, n_y)))
    alpha = jnp.asarray(rng.uniform(0.0, 0.8, size=(n_x, n_y)))
    eps = jnp.asarray(rng.uniform(0.5, 1.0, size=(n_x, n_y)))
    t_eq = equilibrium_temperature_fv3(s_in, alpha, eps)
    assert t_eq.shape == (n_x, n_y)
    assert jnp.all(jnp.isfinite(t_eq))
    assert jnp.all(t_eq > 0.0)
