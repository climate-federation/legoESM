"""FV3_3D iter 737: temperature_from_theta_fv3 port.

Inverse of iter-735 theta_dry_fv3:
  T = theta * (p / p_ref) ^ kappa

Tests
-----

1. ``test_T_at_p_ref_matches_theta``.
2. ``test_T_subsiding_warmer``.
3. ``test_T_round_trip_with_theta_dry``.
4. ``test_T_round_trip_via_exner``.
5. ``test_T_custom_cappa``.
6. ``test_T_shapes_3d``.
7. ``test_T_finite``.
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
    temperature_from_theta_fv3,
    theta_dry_fv3,
)


def test_T_at_p_ref_matches_theta():
    """p = p_ref → T = θ."""
    theta = jnp.full((5,), 290.0)
    p = jnp.full((5,), 1.0e5)
    T = temperature_from_theta_fv3(theta, p)
    assert jnp.allclose(T, theta, atol=1e-12)


def test_T_subsiding_warmer():
    """Descending parcel (p increases above 1e5) → T > θ."""
    theta = jnp.full((5,), 290.0)
    p = jnp.full((5,), 1.1e5)   # below sea level pressure
    T = temperature_from_theta_fv3(theta, p)
    assert jnp.all(T > theta)


def test_T_round_trip_with_theta_dry():
    """theta_dry(T, p); T_from_theta(theta, p) → recovers T."""
    rng = np.random.default_rng(seed=737)
    T_input = jnp.asarray(rng.uniform(220.0, 310.0, size=(10,)))
    p = jnp.asarray(rng.uniform(2.0e4, 1.0e5, size=(10,)))
    theta = theta_dry_fv3(T_input, p)
    T_recovered = temperature_from_theta_fv3(theta, p)
    assert jnp.allclose(T_recovered, T_input, atol=1e-10)


def test_T_round_trip_via_exner():
    """T = θ · Π (via iter-736)."""
    rng = np.random.default_rng(seed=738)
    theta = jnp.asarray(rng.uniform(280.0, 350.0, size=(10,)))
    p = jnp.asarray(rng.uniform(2.0e4, 1.0e5, size=(10,)))
    T_iter737 = temperature_from_theta_fv3(theta, p)
    T_via_exner = theta * exner_fv3(p)
    assert jnp.allclose(T_iter737, T_via_exner, atol=1e-12)


def test_T_custom_cappa():
    """Moist cappa override."""
    theta = jnp.full((1,), 320.0)
    p = jnp.full((1,), 5.0e4)
    q = jnp.full((1,), 0.015)
    cappa = cappa_moist_fv3(q)
    T = temperature_from_theta_fv3(theta, p, cappa=cappa)
    expected = 320.0 * (5.0e4 / 1.0e5) ** float(cappa[0])
    assert abs(float(T[0]) - expected) / expected < 1e-12


def test_T_shapes_3d():
    """3-D shapes preserved."""
    rng = np.random.default_rng(seed=739)
    n_x, n_y, km = 4, 5, 20
    theta = jnp.asarray(rng.uniform(280.0, 350.0, size=(n_x, n_y, km)))
    p = jnp.asarray(rng.uniform(2.0e4, 1.0e5, size=(n_x, n_y, km)))
    T = temperature_from_theta_fv3(theta, p)
    assert T.shape == (n_x, n_y, km)


def test_T_finite():
    """No NaN/Inf."""
    rng = np.random.default_rng(seed=740)
    theta = jnp.asarray(rng.uniform(250.0, 360.0, size=(4, 30)))
    p = jnp.asarray(rng.uniform(1.0e3, 1.05e5, size=(4, 30)))
    T = temperature_from_theta_fv3(theta, p)
    assert jnp.all(jnp.isfinite(T))
