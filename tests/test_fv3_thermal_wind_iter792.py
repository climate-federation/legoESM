"""FV3_3D iter 792: thermal_wind_fv3.

∂u_g/∂z = -(g/(f·T̄))·∂T/∂y
∂v_g/∂z = +(g/(f·T̄))·∂T/∂x

Composes iter-778 ``coriolis_parameter_fv3``.

Tests
-----

1. ``test_nh_baroclinic_westerly``: NH cold pole (∂T/∂y<0, f>0)
   → ∂u_g/∂z > 0 (westerly jet aloft).
2. ``test_sh_baroclinic_mirror``: SH (f<0) flips sign.
3. ``test_zero_grad_zero_shear``: ∇T=0 → ∂V_g/∂z=0.
4. ``test_meridional_shear``: ∂T/∂x>0 → ∂v_g/∂z>0.
5. ``test_composes_iter778``: (∇T, lat, T) → f → shear.
6. ``test_equator_floored``: f=0 → finite, sign preserved.
7. ``test_shapes_3d_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.grids.cubed_sphere import (
    coriolis_parameter_fv3,
    thermal_wind_fv3,
)


def test_nh_baroclinic_westerly():
    """NH cold pole (∂T/∂y<0, f>0) → ∂u_g/∂z>0 (westerly jet aloft)."""
    dT_dx = jnp.array([0.0])
    dT_dy = jnp.array([-1e-5])  # cold pole: T decreases northward
    f = jnp.array([1.0e-4])     # NH
    T_mean = jnp.array([250.0])
    du, dv = thermal_wind_fv3(dT_dx, dT_dy, f, T_mean)
    # ∂u_g/∂z = -(g/(f·T))·(-1e-5) > 0
    assert float(du[0]) > 0.0
    np.testing.assert_allclose(np.asarray(dv), [0.0], atol=1e-15)


def test_sh_baroclinic_mirror():
    """SH (f<0) flips sign of shear."""
    dT_dx = jnp.array([0.0])
    dT_dy = jnp.array([-1e-5])
    f_nh = jnp.array([1.0e-4])
    f_sh = jnp.array([-1.0e-4])
    T_mean = jnp.array([250.0])
    du_nh, _ = thermal_wind_fv3(dT_dx, dT_dy, f_nh, T_mean)
    du_sh, _ = thermal_wind_fv3(dT_dx, dT_dy, f_sh, T_mean)
    np.testing.assert_allclose(np.asarray(du_sh), -np.asarray(du_nh), rtol=1e-12)


def test_zero_grad_zero_shear():
    """∇T=0 → ∂V_g/∂z=0."""
    zeros = jnp.zeros((3,))
    f = jnp.array([1e-4, -1e-4, 5e-5])
    T = jnp.array([250.0, 250.0, 250.0])
    du, dv = thermal_wind_fv3(zeros, zeros, f, T)
    np.testing.assert_allclose(np.asarray(du), jnp.zeros((3,)), atol=1e-15)
    np.testing.assert_allclose(np.asarray(dv), jnp.zeros((3,)), atol=1e-15)


def test_meridional_shear():
    """∂T/∂x>0 (warmer east), f>0 → ∂v_g/∂z>0 (southerly jet aloft)."""
    dT_dx = jnp.array([1e-5])
    dT_dy = jnp.array([0.0])
    f = jnp.array([1.0e-4])
    T = jnp.array([250.0])
    du, dv = thermal_wind_fv3(dT_dx, dT_dy, f, T)
    np.testing.assert_allclose(np.asarray(du), [0.0], atol=1e-15)
    assert float(dv[0]) > 0.0


def test_composes_iter778():
    """Pipeline (∇T, lat, T) → f → shear."""
    dT_dx = jnp.array([0.0])
    dT_dy = jnp.array([-1e-5])
    f = coriolis_parameter_fv3(jnp.array([45.0]), units="deg")
    T = jnp.array([250.0])
    du, dv = thermal_wind_fv3(dT_dx, dT_dy, f, T)
    # |f|=1.03e-4, expected ∂u_g/∂z = g/(f·T)·1e-5 ≈ 3.8e-3 s⁻¹
    # That's a 3.8 m/s per km — typical mid-lat jet shear
    assert 1e-3 < float(du[0]) < 1e-2
    np.testing.assert_allclose(np.asarray(dv), [0.0], atol=1e-15)


def test_equator_floored():
    """f=0 → shear huge but finite, sign preserved."""
    dT_dx = jnp.array([1e-5])
    dT_dy = jnp.array([-1e-5])
    f = jnp.array([0.0])
    T = jnp.array([250.0])
    du, dv = thermal_wind_fv3(dT_dx, dT_dy, f, T, f_floor=1e-12)
    assert jnp.all(jnp.isfinite(du))
    assert jnp.all(jnp.isfinite(dv))
    # du = -(g/(f·T))·(-1e-5) > 0 huge
    assert float(du[0]) > 0.0
    assert float(dv[0]) > 0.0


def test_shapes_3d_finite():
    """3-D shapes preserved, finite, both components."""
    rng = np.random.default_rng(seed=792)
    n_x, n_y, km = 4, 5, 20
    dT_dx = jnp.asarray(rng.uniform(-2e-5, 2e-5, size=(n_x, n_y, km)))
    dT_dy = jnp.asarray(rng.uniform(-2e-5, 2e-5, size=(n_x, n_y, km)))
    f = jnp.asarray(rng.uniform(-1.5e-4, 1.5e-4, size=(n_x, n_y, km)))
    T = jnp.asarray(rng.uniform(220.0, 300.0, size=(n_x, n_y, km)))
    du, dv = thermal_wind_fv3(dT_dx, dT_dy, f, T)
    assert du.shape == (n_x, n_y, km)
    assert dv.shape == (n_x, n_y, km)
    assert jnp.all(jnp.isfinite(du))
    assert jnp.all(jnp.isfinite(dv))
