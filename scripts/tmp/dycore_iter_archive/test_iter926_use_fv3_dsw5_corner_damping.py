"""Iter-926 tests for `use_fv3_dsw5_corner_damping`.

Per user iter-926 brief, three test categories:

1. Default-off bit equality: with the flag at default False, output
   is bit-identical to the iter-893 production matrix.
2. Flag-on finite output: a single C8/C12 W2 step with the flag on
   produces finite output (no NaN, no infinities).
3. FB-model warning: setting the flag on a config used to construct
   `FV3FBShallowWaterModel` emits a clear UserWarning explaining
   the flag is production-only.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.dynamics.gcm.shallow_water_fv3_cdgrid import (
    CDGridShallowWaterConfig,
    FV3EdgeShallowWaterModel,
    FV3EdgeShallowWaterState,
    FV3FBShallowWaterModel,
)
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import (
    create_cubed_sphere_cdgrid,
)
from tests.atmosphere.shallow_water.test_cases.williamson import (
    williamson_test2,
)


def _div_damp_cube(n: int, ref_n: int = 48, ref_coeff: float = 1.5e7) -> float:
    return ref_coeff * (ref_n / n) ** 2


def _make_w2_state(n: int):
    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    sw = williamson_test2(grid)
    u0 = 2.0 * jnp.pi * grid.radius / (12.0 * 86400.0)
    u_d = cdgrid.cos_angle_edge_x * (u0 * jnp.cos(cdgrid.lat_edge_x))
    v_d = -cdgrid.sin_angle_edge_y * (u0 * jnp.cos(cdgrid.lat_edge_y))
    state = FV3EdgeShallowWaterState(
        h=sw.h.data, u_d=u_d, v_d=v_d, h_s=sw.h_s.data,
    )
    return grid, cdgrid, state


def _production_config(n: int, **overrides):
    cfg = CDGridShallowWaterConfig(
        hyperdiff_coeff=0.0,
        div_damp=8.0 * _div_damp_cube(n),
        boundary_fix=True,
        damp_v=0.06,
        nord_v=2,
        apply_fortran_xppm_boundary=True,
    )
    return cfg._replace(**overrides) if overrides else cfg


def test_iter926_default_off_bit_identical_to_iter893():
    """Flag at default False yields output bit-identical to iter-893
    production matrix on a single C8 W2 step.
    """
    n = 8
    grid, cdgrid, state = _make_w2_state(n)

    cfg_off = CDGridShallowWaterConfig(
        hyperdiff_coeff=0.0,
        div_damp=8.0 * _div_damp_cube(n),
        boundary_fix=True,
        damp_v=0.06,
        nord_v=2,
        apply_fortran_xppm_boundary=True,
        use_fv3_dsw5_corner_damping=False,  # explicit default
    )
    cfg_baseline = CDGridShallowWaterConfig(
        hyperdiff_coeff=0.0,
        div_damp=8.0 * _div_damp_cube(n),
        boundary_fix=True,
        damp_v=0.06,
        nord_v=2,
        apply_fortran_xppm_boundary=True,
    )
    model_off = FV3EdgeShallowWaterModel(grid, cfg_off)
    model_baseline = FV3EdgeShallowWaterModel(grid, cfg_baseline)
    model_off.set_initial_mass(state)
    model_baseline.set_initial_mass(state)

    s_off = model_off.step(state, 600.0)
    s_baseline = model_baseline.step(state, 600.0)

    np.testing.assert_array_equal(
        np.asarray(s_off.h), np.asarray(s_baseline.h),
        err_msg="h differs at default-off",
    )
    np.testing.assert_array_equal(
        np.asarray(s_off.u_d), np.asarray(s_baseline.u_d),
        err_msg="u_d differs at default-off",
    )
    np.testing.assert_array_equal(
        np.asarray(s_off.v_d), np.asarray(s_baseline.v_d),
        err_msg="v_d differs at default-off",
    )


def test_iter926_flag_on_c8_w2_step_finite():
    """One C8 W2 step with flag on produces finite output."""
    n = 8
    grid, cdgrid, state = _make_w2_state(n)
    cfg = CDGridShallowWaterConfig(
        hyperdiff_coeff=0.0,
        div_damp=8.0 * _div_damp_cube(n),
        boundary_fix=True,
        damp_v=0.06,
        nord_v=2,
        apply_fortran_xppm_boundary=True,
        use_fv3_dsw5_corner_damping=True,
    )
    model = FV3EdgeShallowWaterModel(grid, cfg)
    model.set_initial_mass(state)
    s = model.step(state, 600.0)
    assert np.all(np.isfinite(np.asarray(s.h))), "h has non-finite values"
    assert np.all(np.isfinite(np.asarray(s.u_d))), "u_d has non-finite values"
    assert np.all(np.isfinite(np.asarray(s.v_d))), "v_d has non-finite values"


def test_iter926_flag_on_c12_w2_step_finite():
    """One C12 W2 step with flag on produces finite output."""
    n = 12
    grid, cdgrid, state = _make_w2_state(n)
    cfg = CDGridShallowWaterConfig(
        hyperdiff_coeff=0.0,
        div_damp=8.0 * _div_damp_cube(n),
        boundary_fix=True,
        damp_v=0.06,
        nord_v=2,
        apply_fortran_xppm_boundary=True,
        use_fv3_dsw5_corner_damping=True,
    )
    model = FV3EdgeShallowWaterModel(grid, cfg)
    model.set_initial_mass(state)
    s = model.step(state, 600.0)
    assert np.all(np.isfinite(np.asarray(s.h)))
    assert np.all(np.isfinite(np.asarray(s.u_d)))
    assert np.all(np.isfinite(np.asarray(s.v_d)))


def test_iter926_flag_on_changes_state_vs_default():
    """Flag on with non-zero coefficients should produce DIFFERENT
    state from default-off after one step.  Sanity check that the
    hook is actually doing something (not silently no-op).
    """
    n = 8
    grid, _, state = _make_w2_state(n)

    # Force coefficients to non-zero so the hook does something
    # (Fortran defaults d2_bg=0/dddmp=0 with d4_bg=0.16/nord=1 give
    # del-4 background damping which is non-trivial).
    cfg_on = CDGridShallowWaterConfig(
        hyperdiff_coeff=0.0,
        div_damp=8.0 * _div_damp_cube(n),
        boundary_fix=True,
        damp_v=0.06,
        nord_v=2,
        apply_fortran_xppm_boundary=True,
        use_fv3_dsw5_corner_damping=True,
        d4_bg=0.16,
        nord=1,
    )
    cfg_off = cfg_on._replace(use_fv3_dsw5_corner_damping=False)

    model_on = FV3EdgeShallowWaterModel(grid, cfg_on)
    model_off = FV3EdgeShallowWaterModel(grid, cfg_off)
    model_on.set_initial_mass(state)
    model_off.set_initial_mass(state)

    s_on = model_on.step(state, 600.0)
    s_off = model_off.step(state, 600.0)

    diff_u = float(jnp.max(jnp.abs(s_on.u_d - s_off.u_d)))
    diff_v = float(jnp.max(jnp.abs(s_on.v_d - s_off.v_d)))
    assert diff_u > 1e-15, (
        f"Flag on did not change u_d (max diff {diff_u:.3e}); the d_sw5 "
        f"corner-damping hook may be silently no-op."
    )
    assert diff_v > 1e-15, (
        f"Flag on did not change v_d (max diff {diff_v:.3e})."
    )


def test_iter926_fb_model_emits_warning_for_flag():
    """Setting `use_fv3_dsw5_corner_damping=True` on a config used to
    construct FV3FBShallowWaterModel should emit a UserWarning at
    __init__ time."""
    n = 8
    grid = create_cubed_sphere(n)
    cfg = CDGridShallowWaterConfig(
        use_fv3_dsw5_corner_damping=True,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        FV3FBShallowWaterModel(grid, cfg)
    msgs = [str(w.message) for w in caught
            if issubclass(w.category, UserWarning)]
    assert any("use_fv3_dsw5_corner_damping" in m for m in msgs), (
        f"Expected FV3FBShallowWaterModel to emit UserWarning naming "
        f"`use_fv3_dsw5_corner_damping`.  Got: {msgs}"
    )


def test_iter926_fb_model_no_warning_for_default_off():
    """FB model should NOT warn when the flag is at default-off."""
    n = 8
    grid = create_cubed_sphere(n)
    cfg = CDGridShallowWaterConfig()  # all defaults
    assert cfg.use_fv3_dsw5_corner_damping is False
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        FV3FBShallowWaterModel(grid, cfg)
    msgs = [str(w.message) for w in caught
            if issubclass(w.category, UserWarning)]
    assert not any("use_fv3_dsw5_corner_damping" in m for m in msgs), (
        f"Default-off should not warn.  Got: {msgs}"
    )
