"""Mixed-precision vmix opt-in (LEGOESM_VMIX_F32_SOLVE): f32 work, f64 state.

The backward-Euler vertical-diffusion matrix is diagonally dominant (well-
conditioned), so the f32 Thomas solve is accurate; we keep f64 STATE in/out.
GPU-only speed benefit (f64=1/32 on RTX8000); CPU shows dtype-churn slowdown.
Default OFF is bit-identical to the legacy f64 solve.
"""
from __future__ import annotations

import os
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from legoesm.ocean.physics.vertical_mixing.implicit_solver import (
    implicit_vertical_diffusion_ocean,
    implicit_vertical_diffusion_ocean_pair,
)


def _setup(nlev=30):
    rng = np.random.default_rng(0)
    T = 20.0 * np.exp(-np.linspace(0, 3, nlev)) + 0.1 * rng.standard_normal(nlev)
    S = 35.0 + 0.2 * rng.standard_normal(nlev)
    K = jnp.asarray(1.0e-3 * np.ones(nlev - 1))
    dz = jnp.asarray(np.full(nlev, 10.0))
    dz_half = jnp.asarray(np.full(nlev - 1, 10.0))
    return jnp.asarray(T), jnp.asarray(S), K, dz, dz_half


def _run(field, K, dz, dz_half, f32):
    if f32:
        os.environ["LEGOESM_VMIX_F32_SOLVE"] = "1"
    else:
        os.environ.pop("LEGOESM_VMIX_F32_SOLVE", None)
    try:
        return implicit_vertical_diffusion_ocean(field, K, dz, dz_half, 3600.0)
    finally:
        os.environ.pop("LEGOESM_VMIX_F32_SOLVE", None)


def test_f32_solve_close_and_conserves():
    T, _S, K, dz, dz_half = _setup()
    x64 = _run(T, K, dz, dz_half, f32=False)
    x32 = _run(T, K, dz, dz_half, f32=True)
    assert x32.dtype == jnp.float64, "state must stay f64"
    # f32 WORK solve: close to f64, not bit-exact.
    np.testing.assert_allclose(np.asarray(x32), np.asarray(x64),
                               rtol=1e-4, atol=1e-3)
    # conservation: EXACT to f64 via the column-mass correction (codex #3).
    m0 = float(jnp.sum(T * dz))
    np.testing.assert_allclose(float(jnp.sum(x32 * dz)), m0, rtol=1e-12)


def _stiff(nlev=30):
    # Codex adversarial case: stiff surface K=0.1, dz=1, dt=3600 -> r~360.
    rng = np.random.default_rng(1)
    T = 20.0 * np.exp(-np.linspace(0, 3, nlev)) + 0.1 * rng.standard_normal(nlev)
    K = jnp.asarray(1.0e-1 * np.ones(nlev - 1))
    dz = jnp.asarray(np.full(nlev, 1.0))
    dz_half = jnp.asarray(np.full(nlev - 1, 1.0))
    return jnp.asarray(T), K, dz, dz_half


def test_f32_solve_stiff_surface():
    """Stiff K=0.1/dz=1 (codex case): f32 still close + mass-exact."""
    T, K, dz, dz_half = _stiff()
    x64 = _run(T, K, dz, dz_half, f32=False)
    x32 = _run(T, K, dz, dz_half, f32=True)
    np.testing.assert_allclose(np.asarray(x32), np.asarray(x64),
                               rtol=1e-3, atol=1e-3)
    m0 = float(jnp.sum(T * dz))
    np.testing.assert_allclose(float(jnp.sum(x32 * dz)), m0, rtol=1e-12)


def test_pair_mixed_dtype_no_f32_work():
    """Codex #1: f64 field_1 + f32 field_2, env-on -> f32-work NOT applied
    (BOTH must be f64); identical to the legacy shared solve."""
    T, S, K, dz, dz_half = _setup()
    S32 = S.astype(jnp.float32)
    os.environ.pop("LEGOESM_VMIX_F32_SOLVE", None)
    t_leg, s_leg = implicit_vertical_diffusion_ocean_pair(T, S32, K, dz, dz_half, 3600.0)
    os.environ["LEGOESM_VMIX_F32_SOLVE"] = "1"
    try:
        t_on, s_on = implicit_vertical_diffusion_ocean_pair(T, S32, K, dz, dz_half, 3600.0)
    finally:
        os.environ.pop("LEGOESM_VMIX_F32_SOLVE", None)
    np.testing.assert_array_equal(np.asarray(t_on), np.asarray(t_leg))
    np.testing.assert_array_equal(np.asarray(s_on), np.asarray(s_leg))


def test_default_off_bit_identical():
    T, _S, K, dz, dz_half = _setup()
    a = _run(T, K, dz, dz_half, f32=False)
    b = _run(T, K, dz, dz_half, f32=False)
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_pair_f32_solve_close():
    T, S, K, dz, dz_half = _setup()
    os.environ.pop("LEGOESM_VMIX_F32_SOLVE", None)
    t64, s64 = implicit_vertical_diffusion_ocean_pair(T, S, K, dz, dz_half, 3600.0)
    os.environ["LEGOESM_VMIX_F32_SOLVE"] = "1"
    try:
        t32, s32 = implicit_vertical_diffusion_ocean_pair(T, S, K, dz, dz_half, 3600.0)
    finally:
        os.environ.pop("LEGOESM_VMIX_F32_SOLVE", None)
    assert t32.dtype == jnp.float64 and s32.dtype == jnp.float64
    np.testing.assert_allclose(np.asarray(t32), np.asarray(t64), rtol=1e-4, atol=1e-3)
    np.testing.assert_allclose(np.asarray(s32), np.asarray(s64), rtol=1e-4, atol=1e-3)
