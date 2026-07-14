"""Iter-909 sentinels: `cube_edge_softer_div_damp` config flag
multiplies the production `adaptive_coeff` by `cube_edge_div_damp_factor`
at face-boundary cells (i in [0..band-1] U [n-band..n-1] and j similarly)
and by 1.0 at deep-interior cells.
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

import warnings
import numpy as np
import jax
import jax.numpy as jnp
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
    rng = np.random.default_rng(909)
    h = jnp.asarray(1.0e4 + rng.normal(size=(6, n, n)) * 50.0)
    h_s = jnp.zeros_like(h)
    u_d = jnp.asarray(rng.normal(size=(6, n, n + 1)) * 1.0)
    v_d = jnp.asarray(rng.normal(size=(6, n + 1, n)) * 1.0)
    return FV3EdgeShallowWaterState(h=h, u_d=u_d, v_d=v_d, h_s=h_s)


def test_iter909_default_off_bit_identical_to_iter892():
    """Default off (or kwarg explicitly False) must be bit-equivalent
    to iter-892 production."""
    grid = create_cubed_sphere(n=8, use_duogrid=False)
    state = _make_state(grid)

    cfg_a = CDGridShallowWaterConfig(boundary_fix=True, dddmp=0.2,
                                       div_damp=1e8,
                                       damp_v=0.0, nord_v=0)
    cfg_b = CDGridShallowWaterConfig(boundary_fix=True, dddmp=0.2,
                                       div_damp=1e8,
                                       damp_v=0.0, nord_v=0,
                                       cube_edge_softer_div_damp=False)

    m_a = FV3EdgeShallowWaterModel(grid, config=cfg_a)
    m_a.set_initial_mass(state)
    s_a = m_a.step(state, 30.0)

    m_b = FV3EdgeShallowWaterModel(grid, config=cfg_b)
    m_b.set_initial_mass(state)
    s_b = m_b.step(state, 30.0)

    np.testing.assert_array_equal(np.asarray(s_a.h), np.asarray(s_b.h))
    np.testing.assert_array_equal(np.asarray(s_a.u_d), np.asarray(s_b.u_d))
    np.testing.assert_array_equal(np.asarray(s_a.v_d), np.asarray(s_b.v_d))


def test_iter909_on_changes_step_output_when_div_damp_active():
    """When the flag is True (and div_damp > 0 so the divergence-damping
    branch fires), model.step output must DIFFER from iter-892 default."""
    grid = create_cubed_sphere(n=8, use_duogrid=False)
    state = _make_state(grid)

    cfg_off = CDGridShallowWaterConfig(boundary_fix=True, dddmp=0.2,
                                         div_damp=1e8,
                                         damp_v=0.0, nord_v=0)
    cfg_on = CDGridShallowWaterConfig(boundary_fix=True, dddmp=0.2,
                                        div_damp=1e8,
                                        damp_v=0.0, nord_v=0,
                                        cube_edge_softer_div_damp=True,
                                        cube_edge_div_damp_factor=0.5,
                                        cube_edge_div_damp_band=2)

    m_off = FV3EdgeShallowWaterModel(grid, config=cfg_off)
    m_off.set_initial_mass(state)
    s_off = m_off.step(state, 30.0)

    m_on = FV3EdgeShallowWaterModel(grid, config=cfg_on)
    m_on.set_initial_mass(state)
    s_on = m_on.step(state, 30.0)

    assert not np.allclose(np.asarray(s_off.u_d), np.asarray(s_on.u_d),
                            atol=1e-12), (
        "iter-909 ON path produced bit-identical u_d to iter-892 — flag "
        "is not taking effect in the divergence-damping path.")


def test_iter909_no_effect_when_div_damp_zero():
    """When div_damp == 0 the entire divergence-damping branch is gated
    off — iter-909's flag must be a no-op even if True."""
    grid = create_cubed_sphere(n=8, use_duogrid=False)
    state = _make_state(grid)

    cfg_off = CDGridShallowWaterConfig(boundary_fix=True,
                                         div_damp=0.0,
                                         damp_v=0.0, nord_v=0)
    cfg_on = CDGridShallowWaterConfig(boundary_fix=True,
                                        div_damp=0.0,
                                        damp_v=0.0, nord_v=0,
                                        cube_edge_softer_div_damp=True)

    m_off = FV3EdgeShallowWaterModel(grid, config=cfg_off)
    m_off.set_initial_mass(state)
    s_off = m_off.step(state, 30.0)

    m_on = FV3EdgeShallowWaterModel(grid, config=cfg_on)
    m_on.set_initial_mass(state)
    s_on = m_on.step(state, 30.0)

    np.testing.assert_array_equal(np.asarray(s_off.u_d), np.asarray(s_on.u_d))


def test_iter909_factor_1_is_no_op():
    """factor=1.0 should multiply adaptive_coeff by 1.0 everywhere —
    bit-equivalent to flag-OFF."""
    grid = create_cubed_sphere(n=8, use_duogrid=False)
    state = _make_state(grid)

    cfg_off = CDGridShallowWaterConfig(boundary_fix=True, dddmp=0.2,
                                         div_damp=1e8,
                                         damp_v=0.0, nord_v=0)
    cfg_factor_1 = CDGridShallowWaterConfig(boundary_fix=True, dddmp=0.2,
                                              div_damp=1e8,
                                              damp_v=0.0, nord_v=0,
                                              cube_edge_softer_div_damp=True,
                                              cube_edge_div_damp_factor=1.0,
                                              cube_edge_div_damp_band=2)
    m_off = FV3EdgeShallowWaterModel(grid, config=cfg_off)
    m_off.set_initial_mass(state)
    s_off = m_off.step(state, 30.0)
    m_1 = FV3EdgeShallowWaterModel(grid, config=cfg_factor_1)
    m_1.set_initial_mass(state)
    s_1 = m_1.step(state, 30.0)
    np.testing.assert_allclose(np.asarray(s_off.u_d), np.asarray(s_1.u_d),
                                atol=1e-13)


def test_iter909_fb_chain_warns_on_flag_set():
    grid = create_cubed_sphere(n=8, use_duogrid=False)
    config = CDGridShallowWaterConfig(cube_edge_softer_div_damp=True)
    with warnings.catch_warnings(record=True) as w_list:
        warnings.simplefilter("always")
        _ = FV3FBShallowWaterModel(grid, config=config)
    matches = [w for w in w_list
               if issubclass(w.category, UserWarning)
               and "cube_edge_softer_div_damp" in str(w.message)]
    assert matches


def test_iter909_default_config_off():
    cfg = CDGridShallowWaterConfig()
    assert cfg.cube_edge_softer_div_damp is False
    assert cfg.cube_edge_div_damp_factor == 0.5
    assert cfg.cube_edge_div_damp_band == 2


def test_iter909_other_flags_remain_default_off():
    cfg = CDGridShallowWaterConfig()
    assert cfg.fortran_faithful_ppm_left is False
    assert cfg.fortran_faithful_ppm_right is False
    assert cfg.use_fv3_dsw1_mass_transport is False
    assert cfg.use_split_mass_momentum_integration is False
