"""FV3_3D iter 735: theta_dry_fv3 port.

Dry potential temperature theta = T*(p_ref/p)^kappa.

Tests
-----

1. ``test_theta_dry_at_p_ref_matches_T``.
2. ``test_theta_dry_lifted_air_warmer``.
3. ``test_theta_dry_custom_cappa``.
4. ``test_theta_dry_inverse_via_pkz``.
5. ``test_theta_dry_shapes_3d``.
6. ``test_theta_dry_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.grids.cubed_sphere import (
    cappa_moist_fv3,
    compute_pkz_fv3,
    theta_dry_fv3,
)


def test_theta_dry_at_p_ref_matches_T():
    """p = p_ref → θ = T exactly."""
    T = jnp.full((5,), 290.0)
    p = jnp.full((5,), 1.0e5)
    theta = theta_dry_fv3(T, p)
    assert jnp.allclose(theta, T, atol=1e-12)


def test_theta_dry_lifted_air_warmer():
    """p < p_ref (high altitude) → θ > T (adiabatic lift to surface)."""
    T = jnp.full((5,), 250.0)
    p = jnp.full((5,), 5.0e4)
    theta = theta_dry_fv3(T, p)
    expected = 250.0 * (1.0e5 / 5.0e4) ** constants.kappa
    assert jnp.allclose(theta, expected, atol=1e-10)
    assert jnp.all(theta > T)


def test_theta_dry_custom_cappa():
    """Moist cappa override."""
    T = jnp.full((1,), 290.0)
    p = jnp.full((1,), 5.0e4)
    q = jnp.full((1,), 0.015)
    cappa = cappa_moist_fv3(q)
    theta = theta_dry_fv3(T, p, cappa=cappa)
    expected = 290.0 * (1.0e5 / 5.0e4) ** float(cappa[0])
    assert abs(float(theta[0]) - expected) / expected < 1e-12


def test_theta_dry_inverse_via_pkz():
    """θ = T · (p_ref/p)^κ === T/pkz when pkz = (p/p_ref)^κ.
    For thin layer at p=p_ref, pkz ≈ 1 → θ ≈ T."""
    T = jnp.array([290.0])
    pe = jnp.array([0.99e5, 1.01e5])
    peln = jnp.log(pe)
    delp = pe[1:] - pe[:-1]
    pkz = compute_pkz_fv3(delp, peln=peln, hydrostatic=True)
    p_f = delp / (peln[1:] - peln[:-1])
    theta_via_pkz = T / pkz * (1.0e5) ** constants.kappa
    theta_via_helper = theta_dry_fv3(T, p_f)
    # Both should agree to high precision for thin layer
    assert jnp.allclose(theta_via_helper, theta_via_pkz, rtol=1e-3)


def test_theta_dry_shapes_3d():
    """3-D shapes preserved."""
    rng = np.random.default_rng(seed=735)
    n_x, n_y, km = 4, 5, 20
    T = jnp.asarray(rng.uniform(220.0, 300.0, size=(n_x, n_y, km)))
    p = jnp.asarray(rng.uniform(1.0e4, 1.0e5, size=(n_x, n_y, km)))
    theta = theta_dry_fv3(T, p)
    assert theta.shape == (n_x, n_y, km)


def test_theta_dry_finite():
    """No NaN/Inf."""
    rng = np.random.default_rng(seed=736)
    T = jnp.asarray(rng.uniform(200.0, 320.0, size=(4, 30)))
    p = jnp.asarray(rng.uniform(1.0e4, 1.0e5, size=(4, 30)))
    theta = theta_dry_fv3(T, p)
    assert jnp.all(jnp.isfinite(theta))
