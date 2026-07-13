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
from legoesm.atmosphere.dynamics.gcm.shallow_water_fv3_cdgrid import (
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
    """FB chain must emit a UserWarning at MODEL CONSTRUCTION when
    fortran_faithful_ppm_left is enabled (iter-903c: warning is in
    `__init__`, not inside `step`, so it's not JIT-trace-time only)."""
    grid = create_cubed_sphere(n=8, use_duogrid=False)
    config = CDGridShallowWaterConfig(
        fortran_faithful_ppm_left=True,
    )
    with warnings.catch_warnings(record=True) as w_list:
        warnings.simplefilter("always")
        model = FV3FBShallowWaterModel(grid, config=config)

    matches = [w for w in w_list
               if issubclass(w.category, UserWarning)
               and "fortran_faithful_ppm_left" in str(w.message)]
    assert matches, (
        "FV3FBShallowWaterModel.__init__ did not emit the iter-903b/c "
        "UserWarning about the silently-ignored "
        "`fortran_faithful_ppm_left` flag.  "
        f"Got: {[str(w.message) for w in w_list]}")


def test_fb_chain_warns_on_right_flag_set():
    """FB chain must emit a UserWarning at construction when
    fortran_faithful_ppm_right is enabled."""
    grid = create_cubed_sphere(n=8, use_duogrid=False)
    config = CDGridShallowWaterConfig(
        fortran_faithful_ppm_right=True,
    )
    with warnings.catch_warnings(record=True) as w_list:
        warnings.simplefilter("always")
        model = FV3FBShallowWaterModel(grid, config=config)

    matches = [w for w in w_list
               if issubclass(w.category, UserWarning)
               and "fortran_faithful_ppm_right" in str(w.message)]
    assert matches, (
        "FV3FBShallowWaterModel.__init__ did not emit the iter-903b/c "
        "UserWarning about the silently-ignored "
        "`fortran_faithful_ppm_right` flag.  "
        f"Got: {[str(w.message) for w in w_list]}")


def test_fb_chain_silent_when_flags_default_off():
    """No warning when both flags are False (default)."""
    grid = create_cubed_sphere(n=8, use_duogrid=False)
    config = CDGridShallowWaterConfig(
        fortran_faithful_ppm_left=False,
        fortran_faithful_ppm_right=False,
    )
    with warnings.catch_warnings(record=True) as w_list:
        warnings.simplefilter("always")
        model = FV3FBShallowWaterModel(grid, config=config)

    matches = [w for w in w_list
               if issubclass(w.category, UserWarning)
               and ("fortran_faithful_ppm_left" in str(w.message)
                    or "fortran_faithful_ppm_right" in str(w.message))]
    assert not matches, (
        "FV3FBShallowWaterModel.__init__ emitted an unexpected "
        "UserWarning about iter-900/iter-903 flags when both are "
        f"default-off.  Got: {[str(w.message) for w in matches]}")


def test_production_model_does_not_warn_when_flag_set():
    """The PRODUCTION FV3EdgeShallowWaterModel must NOT emit the
    FB-chain warning, since it correctly consumes the flag."""
    grid = create_cubed_sphere(n=8, use_duogrid=False)
    config = CDGridShallowWaterConfig(
        apply_fortran_xppm_boundary=True,
        fortran_faithful_ppm_left=True,
    )
    with warnings.catch_warnings(record=True) as w_list:
        warnings.simplefilter("always")
        model = FV3EdgeShallowWaterModel(grid, config=config)

    fb_warnings = [w for w in w_list
                    if issubclass(w.category, UserWarning)
                    and "FV3FBShallowWaterModel" in str(w.message)]
    assert not fb_warnings, (
        "FV3EdgeShallowWaterModel incorrectly emitted the FB-chain "
        "ignored-flag warning.  These flags are CORRECTLY consumed "
        f"by the production model.  Got: {fb_warnings}")


def test_iter903c_warning_fires_on_construction_not_step():
    """Iter-903c discriminating sentinel: the warning fires at
    `__init__` time, NOT at `step` time.  This ensures the guard is
    not subject to JIT-trace timing (Codex iter-903b stop-time
    finding)."""
    grid = create_cubed_sphere(n=8, use_duogrid=False)
    config = CDGridShallowWaterConfig(
        fortran_faithful_ppm_left=True,
    )

    # First: verify warning fires at construction.
    with warnings.catch_warnings(record=True) as w_list_init:
        warnings.simplefilter("always")
        model = FV3FBShallowWaterModel(grid, config=config)

    init_matches = [w for w in w_list_init
                     if "fortran_faithful_ppm_left" in str(w.message)]
    assert init_matches, "Warning did not fire on __init__"

    # Second: verify NO additional warning fires on step (because the
    # guard is no longer inside the JIT-compiled step).  This proves
    # the warning is construction-time, not JIT-trace-time.
    state = _make_state(grid)
    model.set_initial_mass(state)
    with warnings.catch_warnings(record=True) as w_list_step:
        warnings.simplefilter("always")
        _ = model.step(state, 30.0)

    step_matches = [w for w in w_list_step
                     if "fortran_faithful_ppm_left" in str(w.message)]
    assert not step_matches, (
        "Warning fired on step() — it should only fire at __init__ "
        "after iter-903c.  Got: "
        f"{[str(w.message) for w in step_matches]}")
