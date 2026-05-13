"""FV3_3D iter 789: ekman_transport_fv3 (M_x, M_y).

M_x =  τ_y / (ρ·f),   M_y = -τ_x / (ρ·f)

Composes iter-778 ``coriolis_parameter_fv3``.

Tests
-----

1. ``test_nh_eastward_stress``: τ_x>0, f>0 → M_y<0 (rightward = south).
2. ``test_nh_northward_stress``: τ_y>0, f>0 → M_x>0 (rightward = east).
3. ``test_sh_eastward_stress``: τ_x>0, f<0 → M_y>0 (leftward = north).
4. ``test_zero_stress``: τ=0 → M=0.
5. ``test_composes_iter778``: (τ, lat) → f → M.
6. ``test_equator_floored``: f=0 → finite.
7. ``test_shapes_3d_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import (
    coriolis_parameter_fv3,
    ekman_transport_fv3,
)


def test_nh_eastward_stress():
    """NH eastward stress drives southward transport."""
    tau_x = jnp.array([0.1])
    tau_y = jnp.array([0.0])
    f = jnp.array([1.0e-4])  # NH
    rho = 1025.0
    M_x, M_y = ekman_transport_fv3(tau_x, tau_y, f, rho)
    # M_x = 0, M_y = -0.1/(1025·1e-4) ≈ -0.976
    np.testing.assert_allclose(np.asarray(M_x), [0.0], atol=1e-12)
    assert float(M_y[0]) < 0.0  # southward


def test_nh_northward_stress():
    """NH northward stress drives eastward transport."""
    tau_x = jnp.array([0.0])
    tau_y = jnp.array([0.1])
    f = jnp.array([1.0e-4])  # NH
    M_x, M_y = ekman_transport_fv3(tau_x, tau_y, f)
    assert float(M_x[0]) > 0.0  # eastward
    np.testing.assert_allclose(np.asarray(M_y), [0.0], atol=1e-12)


def test_sh_eastward_stress():
    """SH eastward stress drives northward transport (mirror NH)."""
    tau_x = jnp.array([0.1])
    tau_y = jnp.array([0.0])
    f = jnp.array([-1.0e-4])  # SH
    M_x, M_y = ekman_transport_fv3(tau_x, tau_y, f)
    np.testing.assert_allclose(np.asarray(M_x), [0.0], atol=1e-12)
    assert float(M_y[0]) > 0.0  # northward (left of east)


def test_zero_stress():
    """τ=0 → M=0."""
    zeros = jnp.zeros((3,))
    f = jnp.array([1e-4, -1e-4, 5e-5])
    M_x, M_y = ekman_transport_fv3(zeros, zeros, f)
    np.testing.assert_allclose(np.asarray(M_x), jnp.zeros((3,)), atol=1e-15)
    np.testing.assert_allclose(np.asarray(M_y), jnp.zeros((3,)), atol=1e-15)


def test_composes_iter778():
    """Pipeline (τ, lat) → f → M."""
    tau_x = jnp.array([0.1])
    tau_y = jnp.array([0.0])
    f = coriolis_parameter_fv3(jnp.array([45.0]), units="deg")
    M_x, M_y = ekman_transport_fv3(tau_x, tau_y, f)
    # |f|=2·Ω·sin(45°) ≈ 1.03e-4 → M_y ≈ -0.95 m²/s
    assert -1.5 < float(M_y[0]) < -0.5


def test_equator_floored():
    """f=0 → transport huge but finite via f_floor."""
    tau_x = jnp.array([0.1])
    tau_y = jnp.array([0.0])
    f = jnp.array([0.0])
    M_x, M_y = ekman_transport_fv3(tau_x, tau_y, f, f_floor=1e-12)
    assert jnp.all(jnp.isfinite(M_x))
    assert jnp.all(jnp.isfinite(M_y))
    # |M_y| = 0.1/(1025·1e-12) ≈ 9.8e7 — huge but finite
    assert abs(float(M_y[0])) > 1e6


def test_shapes_3d_finite():
    """3-D shapes preserved, finite, both components."""
    rng = np.random.default_rng(seed=789)
    n_x, n_y, km = 4, 5, 20
    tau_x = jnp.asarray(rng.uniform(-0.3, 0.3, size=(n_x, n_y, km)))
    tau_y = jnp.asarray(rng.uniform(-0.3, 0.3, size=(n_x, n_y, km)))
    f = jnp.asarray(rng.uniform(-1.5e-4, 1.5e-4, size=(n_x, n_y, km)))
    M_x, M_y = ekman_transport_fv3(tau_x, tau_y, f)
    assert M_x.shape == (n_x, n_y, km)
    assert M_y.shape == (n_x, n_y, km)
    assert jnp.all(jnp.isfinite(M_x))
    assert jnp.all(jnp.isfinite(M_y))
