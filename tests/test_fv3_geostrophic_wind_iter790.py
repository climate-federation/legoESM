"""FV3_3D iter 790: geostrophic_wind_fv3.

u_g = -(1/f)·∂Φ/∂y,  v_g = +(1/f)·∂Φ/∂x

Composes iter-778 ``coriolis_parameter_fv3``.

Tests
-----

1. ``test_nh_westerly_jet``: ∂Φ/∂y<0 (Φ↓ northward), f>0 → u_g>0.
2. ``test_nh_southerly_jet``: ∂Φ/∂x>0 (Φ↑ eastward), f>0 → v_g>0.
3. ``test_sh_mirror``: same gradient but f<0 → wind flips sign.
4. ``test_zero_grad_zero_wind``: ∇Φ=0 → V_g=0.
5. ``test_composes_iter778``: (gradients, lat) → f → V_g.
6. ``test_equator_floored``: f=0 → huge but finite, sign-preserving.
7. ``test_shapes_3d_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import (
    coriolis_parameter_fv3,
    geostrophic_wind_fv3,
)


def test_nh_westerly_jet():
    """NH zonal jet: ∂Φ/∂y<0 (Φ decreases northward), f>0 → u_g>0 (westerly)."""
    dphi_dx = jnp.array([0.0])
    dphi_dy = jnp.array([-1e-3])  # Φ decreases northward
    f = jnp.array([1.0e-4])
    u_g, v_g = geostrophic_wind_fv3(dphi_dx, dphi_dy, f)
    # u_g = -(1/1e-4) · (-1e-3) = +10 m/s
    np.testing.assert_allclose(np.asarray(u_g), [10.0], rtol=1e-12)
    np.testing.assert_allclose(np.asarray(v_g), [0.0], atol=1e-12)


def test_nh_southerly_jet():
    """NH meridional jet: ∂Φ/∂x>0, f>0 → v_g>0 (southerly)."""
    dphi_dx = jnp.array([1e-3])
    dphi_dy = jnp.array([0.0])
    f = jnp.array([1.0e-4])
    u_g, v_g = geostrophic_wind_fv3(dphi_dx, dphi_dy, f)
    np.testing.assert_allclose(np.asarray(u_g), [0.0], atol=1e-12)
    np.testing.assert_allclose(np.asarray(v_g), [10.0], rtol=1e-12)


def test_sh_mirror():
    """SH same gradients but f<0 → wind flips sign."""
    dphi_dx = jnp.array([1e-3])
    dphi_dy = jnp.array([-1e-3])
    f_nh = jnp.array([1.0e-4])
    f_sh = jnp.array([-1.0e-4])
    u_nh, v_nh = geostrophic_wind_fv3(dphi_dx, dphi_dy, f_nh)
    u_sh, v_sh = geostrophic_wind_fv3(dphi_dx, dphi_dy, f_sh)
    np.testing.assert_allclose(np.asarray(u_sh), -np.asarray(u_nh), rtol=1e-12)
    np.testing.assert_allclose(np.asarray(v_sh), -np.asarray(v_nh), rtol=1e-12)


def test_zero_grad_zero_wind():
    """∇Φ=0 → V_g=0."""
    zeros = jnp.zeros((3,))
    f = jnp.array([1e-4, -1e-4, 5e-5])
    u_g, v_g = geostrophic_wind_fv3(zeros, zeros, f)
    np.testing.assert_allclose(np.asarray(u_g), jnp.zeros((3,)), atol=1e-15)
    np.testing.assert_allclose(np.asarray(v_g), jnp.zeros((3,)), atol=1e-15)


def test_composes_iter778():
    """Pipeline (∇Φ, lat) → f → V_g."""
    dphi_dx = jnp.array([0.0])
    dphi_dy = jnp.array([-1e-3])
    f = coriolis_parameter_fv3(jnp.array([45.0]), units="deg")
    u_g, v_g = geostrophic_wind_fv3(dphi_dx, dphi_dy, f)
    # |f|=2·Ω·sin(45°) ≈ 1.03e-4 → u_g ≈ 9.7 m/s (westerly)
    assert 7.0 < float(u_g[0]) < 12.0
    np.testing.assert_allclose(np.asarray(v_g), [0.0], atol=1e-12)


def test_equator_floored():
    """f=0 → V_g huge but finite, preserving sign of gradients."""
    dphi_dx = jnp.array([1e-3])
    dphi_dy = jnp.array([-1e-3])
    f = jnp.array([0.0])
    u_g, v_g = geostrophic_wind_fv3(dphi_dx, dphi_dy, f, f_floor=1e-12)
    assert jnp.all(jnp.isfinite(u_g))
    assert jnp.all(jnp.isfinite(v_g))
    # u_g = -dphi_dy/f_safe = -(-1e-3)/1e-12 = +1e9 (huge but finite)
    assert float(u_g[0]) > 0.0  # sign preserved
    assert float(v_g[0]) > 0.0


def test_shapes_3d_finite():
    """3-D shapes preserved, finite, both components."""
    rng = np.random.default_rng(seed=790)
    n_x, n_y, km = 4, 5, 20
    dphi_dx = jnp.asarray(rng.uniform(-2e-3, 2e-3, size=(n_x, n_y, km)))
    dphi_dy = jnp.asarray(rng.uniform(-2e-3, 2e-3, size=(n_x, n_y, km)))
    f = jnp.asarray(rng.uniform(-1.5e-4, 1.5e-4, size=(n_x, n_y, km)))
    u_g, v_g = geostrophic_wind_fv3(dphi_dx, dphi_dy, f)
    assert u_g.shape == (n_x, n_y, km)
    assert v_g.shape == (n_x, n_y, km)
    assert jnp.all(jnp.isfinite(u_g))
    assert jnp.all(jnp.isfinite(v_g))
