"""Iter-903b sentinel: verify that `fortran_faithful_ppm_left/right`
flags (iter-900/iter-903) emit a clear UserWarning when they are set
on a `CDGridShallowWaterConfig` consumed by the FB-chain
`FV3FBShallowWaterModel.step`, since the FB chain does NOT route
through `_ppm_reconstruct_1d` and the flags would otherwise be
silently ignored.

Codex iter-903 stop-time review correctly flagged this hazard:
"new shared config flag is silently ignored on non-production
model paths".
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import (
    CDGridShallowWaterConfig,
    FV3EdgeShallowWaterModel,
    FV3EdgeShallowWaterState,
    FV3FBShallowWaterModel,
)


def _make_state(grid):
    n = grid.n
    h = jnp.ones((6, n, n)) * 1.0e4
    h_s = jnp.zeros_like(h)
    u_d = jnp.zeros((6, n, n + 1))
    v_d = jnp.zeros((6, n + 1, n))
    return FV3EdgeShallowWaterState(h=h, u_d=u_d, v_d=v_d, h_s=h_s)


def test_fb_chain_warns_on_left_flag_set():
    """FB chain must emit a UserWarning when fortran_faithful_ppm_left
    is enabled (the flag has no effect there)."""
    grid = create_cubed_sphere(n=8, use_duogrid=False)
    config = CDGridShallowWaterConfig(
        fortran_faithful_ppm_left=True,
        damp_v=0.0,    # disable post-step vorticity damping for test speed
        nord_v=0,
    )
    model = FV3FBShallowWaterModel(grid, config=config)
    state = _make_state(grid)
    model.set_initial_mass(state)

    with warnings.catch_warnings(record=True) as w_list:
        warnings.simplefilter("always")
        # JIT-compile + execute one step.
        _ = model.step(state, 30.0)

    matches = [w for w in w_list
               if issubclass(w.category, UserWarning)
               and "fortran_faithful_ppm_left" in str(w.message)]
    assert matches, (
        "FV3FBShallowWaterModel.step did not emit the iter-903b "
        "UserWarning about the silently-ignored "
        "`fortran_faithful_ppm_left` flag.  "
        f"Got: {[str(w.message) for w in w_list]}")


def test_fb_chain_warns_on_right_flag_set():
    """FB chain must emit a UserWarning when fortran_faithful_ppm_right
    is enabled."""
    grid = create_cubed_sphere(n=8, use_duogrid=False)
    config = CDGridShallowWaterConfig(
        fortran_faithful_ppm_right=True,
        damp_v=0.0,
        nord_v=0,
    )
    model = FV3FBShallowWaterModel(grid, config=config)
    state = _make_state(grid)
    model.set_initial_mass(state)

    with warnings.catch_warnings(record=True) as w_list:
        warnings.simplefilter("always")
        _ = model.step(state, 30.0)

    matches = [w for w in w_list
               if issubclass(w.category, UserWarning)
               and "fortran_faithful_ppm_right" in str(w.message)]
    assert matches, (
        "FV3FBShallowWaterModel.step did not emit the iter-903b "
        "UserWarning about the silently-ignored "
        "`fortran_faithful_ppm_right` flag.  "
        f"Got: {[str(w.message) for w in w_list]}")


def test_fb_chain_silent_when_flags_default_off():
    """No warning when both flags are False (default)."""
    grid = create_cubed_sphere(n=8, use_duogrid=False)
    config = CDGridShallowWaterConfig(
        # explicit defaults
        fortran_faithful_ppm_left=False,
        fortran_faithful_ppm_right=False,
        damp_v=0.0,
        nord_v=0,
    )
    model = FV3FBShallowWaterModel(grid, config=config)
    state = _make_state(grid)
    model.set_initial_mass(state)

    with warnings.catch_warnings(record=True) as w_list:
        warnings.simplefilter("always")
        _ = model.step(state, 30.0)

    matches = [w for w in w_list
               if issubclass(w.category, UserWarning)
               and ("fortran_faithful_ppm_left" in str(w.message)
                    or "fortran_faithful_ppm_right" in str(w.message))]
    assert not matches, (
        "FV3FBShallowWaterModel.step emitted an unexpected UserWarning "
        "about iter-900/iter-903 flags when both are default-off.  "
        f"Got: {[str(w.message) for w in matches]}")


def test_production_model_does_not_warn_when_flag_set():
    """The PRODUCTION FV3EdgeShallowWaterModel (which actually consumes
    these flags) must NOT emit the FB-chain warning, since the flags
    are correctly forwarded there."""
    grid = create_cubed_sphere(n=8, use_duogrid=False)
    config = CDGridShallowWaterConfig(
        apply_fortran_xppm_boundary=True,
        fortran_faithful_ppm_left=True,
        damp_v=0.0,
        nord_v=0,
    )
    model = FV3EdgeShallowWaterModel(grid, config=config)
    state = _make_state(grid)
    model.set_initial_mass(state)

    with warnings.catch_warnings(record=True) as w_list:
        warnings.simplefilter("always")
        _ = model.step(state, 30.0)

    fb_warnings = [w for w in w_list
                    if issubclass(w.category, UserWarning)
                    and "FV3FBShallowWaterModel" in str(w.message)]
    assert not fb_warnings, (
        "FV3EdgeShallowWaterModel.step incorrectly emitted the "
        "FB-chain ignored-flag warning.  These flags are CORRECTLY "
        "consumed by the production model and should not warn.  "
        f"Got: {[str(w.message) for w in fb_warnings]}")
