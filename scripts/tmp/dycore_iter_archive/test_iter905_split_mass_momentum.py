"""Iter-905 sentinels: `use_split_mass_momentum_integration` config
flag splits the time integration into mass-via-transport_step (ONCE
per dt outside RK3) and momentum-via-RK3 (with h held fixed at IC).

Implements path (3) of iter-904b's three outlined paths forward.
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
    rng = np.random.default_rng(905)
    h = jnp.asarray(1.0e4 + rng.normal(size=(6, n, n)) * 50.0)
    h_s = jnp.zeros_like(h)
    u_d = jnp.asarray(rng.normal(size=(6, n, n + 1)) * 1.0)
    v_d = jnp.asarray(rng.normal(size=(6, n + 1, n)) * 1.0)
    return FV3EdgeShallowWaterState(h=h, u_d=u_d, v_d=v_d, h_s=h_s)


def test_iter905_default_off_bit_identical_to_iter892():
    """Default `use_split_mass_momentum_integration=False` produces
    bit-identical model.step output to the iter-892 path."""
    grid = create_cubed_sphere(n=8, use_duogrid=False)
    state = _make_state(grid)

    cfg_a = CDGridShallowWaterConfig(boundary_fix=True, dddmp=0.2,
                                       damp_v=0.0, nord_v=0)
    cfg_b = CDGridShallowWaterConfig(boundary_fix=True, dddmp=0.2,
                                       damp_v=0.0, nord_v=0,
                                       use_split_mass_momentum_integration=False)

    m_a = FV3EdgeShallowWaterModel(grid, config=cfg_a)
    m_a.set_initial_mass(state)
    s_a = m_a.step(state, 30.0)

    m_b = FV3EdgeShallowWaterModel(grid, config=cfg_b)
    m_b.set_initial_mass(state)
    s_b = m_b.step(state, 30.0)

    np.testing.assert_array_equal(np.asarray(s_a.h), np.asarray(s_b.h))
    np.testing.assert_array_equal(np.asarray(s_a.u_d), np.asarray(s_b.u_d))
    np.testing.assert_array_equal(np.asarray(s_a.v_d), np.asarray(s_b.v_d))


def test_iter905_on_changes_step_output():
    """When the flag is True, model.step output must DIFFER from the
    iter-892 path (the split integration is non-trivial)."""
    grid = create_cubed_sphere(n=8, use_duogrid=False)
    state = _make_state(grid)

    cfg_off = CDGridShallowWaterConfig(boundary_fix=True, dddmp=0.2,
                                         damp_v=0.0, nord_v=0)
    cfg_on = CDGridShallowWaterConfig(boundary_fix=True, dddmp=0.2,
                                        damp_v=0.0, nord_v=0,
                                        use_split_mass_momentum_integration=True)

    m_off = FV3EdgeShallowWaterModel(grid, config=cfg_off)
    m_off.set_initial_mass(state)
    s_off = m_off.step(state, 30.0)

    m_on = FV3EdgeShallowWaterModel(grid, config=cfg_on)
    m_on.set_initial_mass(state)
    s_on = m_on.step(state, 30.0)

    assert not np.allclose(np.asarray(s_off.h), np.asarray(s_on.h),
                            atol=1e-12), "iter-905 ON path produced bit-identical h."


def test_iter905_on_step_finite():
    """ON path runs without NaN on a benign random state."""
    grid = create_cubed_sphere(n=8, use_duogrid=False)
    state = _make_state(grid)
    cfg = CDGridShallowWaterConfig(
        apply_fortran_xppm_boundary=True,
        use_split_mass_momentum_integration=True,
        damp_v=0.0, nord_v=0,
    )
    model = FV3EdgeShallowWaterModel(grid, config=cfg)
    model.set_initial_mass(state)
    s_new = model.step(state, 30.0)
    assert np.all(np.isfinite(np.asarray(s_new.h)))
    assert np.all(np.isfinite(np.asarray(s_new.u_d)))
    assert np.all(np.isfinite(np.asarray(s_new.v_d)))


def test_iter905_fb_chain_warns_on_flag_set():
    """FB chain emits __init__ UserWarning when flag is set."""
    grid = create_cubed_sphere(n=8, use_duogrid=False)
    config = CDGridShallowWaterConfig(use_split_mass_momentum_integration=True)
    with warnings.catch_warnings(record=True) as w_list:
        warnings.simplefilter("always")
        _ = FV3FBShallowWaterModel(grid, config=config)
    matches = [w for w in w_list
               if issubclass(w.category, UserWarning)
               and "use_split_mass_momentum_integration" in str(w.message)]
    assert matches, (
        "FV3FBShallowWaterModel.__init__ did not warn about the "
        "iter-905 split flag.  "
        f"Got: {[str(w.message) for w in w_list]}")


def test_iter905_default_off_in_fresh_config():
    """Default config has the iter-905 flag at False."""
    cfg = CDGridShallowWaterConfig()
    assert cfg.use_split_mass_momentum_integration is False, (
        "iter-905 use_split_mass_momentum_integration must default False")


def test_iter905_other_flags_remain_default_off():
    """iter-905's introduction must not silently flip iter-900/903/904."""
    cfg = CDGridShallowWaterConfig()
    assert cfg.fortran_faithful_ppm_left is False
    assert cfg.fortran_faithful_ppm_right is False
    assert cfg.use_fv3_dsw1_mass_transport is False
