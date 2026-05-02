"""Smoke tests for ocean/physics/vertical_mixing/implicit_solver.py."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from legoesm.ocean.physics.vertical_mixing.implicit_solver import (
    build_dz_half,
    implicit_vertical_diffusion_ocean,
)


def test_build_dz_half_midpoint_distances():
    dz = jnp.array([10.0, 20.0, 30.0, 40.0])
    dz_half = build_dz_half(dz)
    assert dz_half.shape == (3,)
    expected = jnp.array([15.0, 25.0, 35.0])
    assert jnp.allclose(dz_half, expected)


def test_implicit_diffusion_no_op_with_zero_K():
    nlev = 6
    field = jnp.linspace(20.0, 5.0, nlev)
    dz = jnp.full((nlev,), 10.0)
    dz_half = build_dz_half(dz)
    K = jnp.zeros((nlev - 1,))

    out = implicit_vertical_diffusion_ocean(field, K, dz, dz_half, dt=600.0)

    assert out.shape == field.shape
    assert jnp.allclose(out, field)


def test_implicit_diffusion_relaxes_jump_to_smooth():
    nlev = 8
    # Sharp two-segment profile
    field = jnp.where(jnp.arange(nlev) < nlev // 2, 25.0, 5.0)
    dz = jnp.full((nlev,), 10.0)
    dz_half = build_dz_half(dz)
    K = jnp.full((nlev - 1,), 1.0e-2)  # 0.01 m^2/s

    out = implicit_vertical_diffusion_ocean(field.astype(jnp.float64), K, dz, dz_half, dt=86400.0)

    assert out.shape == field.shape
    assert jnp.all(jnp.isfinite(out))
    # Mass-like quantity should be conserved (zero-flux BCs)
    assert jnp.allclose(jnp.sum(out * dz), jnp.sum(field * dz), atol=1e-6)
    # Smoother than input
    var_in = jnp.var(field)
    var_out = jnp.var(out)
    assert var_out < var_in


def test_implicit_diffusion_multidim_columns():
    nlat, nlon, nlev = 3, 4, 5
    field = jnp.tile(jnp.linspace(20.0, 5.0, nlev), (nlat, nlon, 1))
    dz = jnp.full((nlev,), 10.0)
    dz_half = build_dz_half(dz)
    K_scalar = 0.01

    out = implicit_vertical_diffusion_ocean(field, K_scalar, dz, dz_half, dt=3600.0)

    assert out.shape == field.shape
    assert jnp.all(jnp.isfinite(out))


def test_implicit_diffusion_rejects_nonpositive_dt():
    field = jnp.zeros(4)
    dz = jnp.full(4, 10.0)
    dz_half = build_dz_half(dz)
    with pytest.raises(ValueError):
        implicit_vertical_diffusion_ocean(field, 0.0, dz, dz_half, dt=0.0)


def test_implicit_diffusion_one_level_noop():
    field = jnp.array([42.0])
    dz = jnp.array([10.0])
    out = implicit_vertical_diffusion_ocean(field, K=jnp.zeros((0,)), dz=dz, dz_half=jnp.zeros((0,)), dt=600.0)
    assert jnp.allclose(out, field)
