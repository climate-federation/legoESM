"""FV3_3D iter 736: exner_fv3 port.

Point-wise Exner function Pi = (p/p_ref)^kappa.

Tests
-----

1. ``test_exner_at_p_ref_is_one``.
2. ``test_exner_decreases_with_decreasing_p``.
3. ``test_exner_theta_dry_inverse``.
4. ``test_exner_custom_cappa``.
5. ``test_exner_shapes_3d``.
6. ``test_exner_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.grids.cubed_sphere import (
    cappa_moist_fv3,
    exner_fv3,
    theta_dry_fv3,
)


def test_exner_at_p_ref_is_one():
    """p = p_ref → Π = 1."""
    p = jnp.full((5,), 1.0e5)
    pi = exner_fv3(p)
    assert jnp.allclose(pi, 1.0, atol=1e-12)


def test_exner_decreases_with_decreasing_p():
    """Lower p (higher altitude) → smaller Π."""
    p_hi = jnp.full((5,), 1.0e5)
    p_lo = jnp.full((5,), 1.0e4)
    pi_hi = exner_fv3(p_hi)
    pi_lo = exner_fv3(p_lo)
    assert jnp.all(pi_hi > pi_lo)


def test_exner_theta_dry_inverse():
    """θ · Π = T (Exner is inverse of theta_dry).
    θ = T/(p/p_ref)^κ = T · (p_ref/p)^κ
    Π = (p/p_ref)^κ
    θ · Π = T · (p_ref/p)^κ · (p/p_ref)^κ = T."""
    T = jnp.full((5,), 290.0)
    p = jnp.linspace(2.0e4, 1.0e5, 5)
    theta = theta_dry_fv3(T, p)
    pi = exner_fv3(p)
    T_recovered = theta * pi
    assert jnp.allclose(T_recovered, T, atol=1e-10)


def test_exner_custom_cappa():
    """Moist cappa override."""
    p = jnp.full((1,), 5.0e4)
    q = jnp.full((1,), 0.015)
    cappa = cappa_moist_fv3(q)
    pi = exner_fv3(p, cappa=cappa)
    expected = (5.0e4 / 1.0e5) ** float(cappa[0])
    assert abs(float(pi[0]) - expected) / expected < 1e-12


def test_exner_shapes_3d():
    """3-D shapes preserved."""
    rng = np.random.default_rng(seed=736)
    n_x, n_y, km = 4, 5, 20
    p = jnp.asarray(rng.uniform(1.0e4, 1.0e5, size=(n_x, n_y, km)))
    pi = exner_fv3(p)
    assert pi.shape == (n_x, n_y, km)


def test_exner_finite():
    """No NaN/Inf."""
    rng = np.random.default_rng(seed=737)
    p = jnp.asarray(rng.uniform(1.0e3, 1.05e5, size=(4, 30)))
    pi = exner_fv3(p)
    assert jnp.all(jnp.isfinite(pi))
