"""FV3_3D iter 449: PE mirror of NH iter-448 FV3 ``Ray_fast``.

Same FV3 ``dyn_core.F90:2922-3020`` formula, applied to PE
prognostic winds u_d, v_d (PE has no w).

Reference profile: ``pfull(k) = (A_full[k] + B_full[k]) * p_ref``.

Tests
-----

1. ``test_default_rf_tau_zero``.
2. ``test_default_rf_cutoff_3000``.
3. ``test_baseline_equals_no_rf``.
4. ``test_rf_damps_top_winds``.
5. ``test_rf_no_effect_below_cutoff``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
    CDGridPrimitiveEquationConfig,
    CDGridPrimitiveEquationModel,
    hydrostatic_to_fv3,
)
from legoesm.atmosphere.held_suarez import held_suarez_init
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.grids.vertical import standard_hybrid_levels


def _build_c8():
    n = 8
    nlev = 5
    grid = create_cubed_sphere(n, use_duogrid=False)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    coord = standard_hybrid_levels(nlev)
    state_cc = held_suarez_init(grid, coord)
    state = hydrostatic_to_fv3(state_cc, cdgrid)
    rng = np.random.default_rng(seed=449)
    n_corners = n + 1
    u_p = rng.uniform(-5.0, 5.0,
                      size=(6, n_corners, n_corners, nlev))
    v_p = rng.uniform(-5.0, 5.0,
                      size=(6, n_corners, n_corners, nlev))
    state = state._replace(
        u_d=state.u_d.replace(data=jnp.asarray(u_p)),
        v_d=state.v_d.replace(data=jnp.asarray(v_p)),
    )
    return grid, coord, state


def test_default_rf_tau_zero():
    cfg = CDGridPrimitiveEquationConfig()
    assert cfg.rf_tau_days == 0.0


def test_default_rf_cutoff_3000():
    cfg = CDGridPrimitiveEquationConfig()
    assert cfg.rf_cutoff_pa == 3000.0


def test_baseline_equals_no_rf():
    grid, coord, state = _build_c8()
    cfg_d = CDGridPrimitiveEquationConfig()
    cfg_z = CDGridPrimitiveEquationConfig(rf_tau_days=0.0)
    m_d = CDGridPrimitiveEquationModel(grid, coord, cfg_d)
    m_z = CDGridPrimitiveEquationModel(grid, coord, cfg_z)
    s_d = m_d.step(state, dt=10.0)
    s_z = m_z.step(state, dt=10.0)
    np.testing.assert_allclose(
        np.asarray(s_d.u_d.data), np.asarray(s_z.u_d.data),
        rtol=1e-14, atol=1e-14,
    )


def test_rf_damps_top_winds():
    grid, coord, state = _build_c8()
    cfg_off = CDGridPrimitiveEquationConfig(rf_tau_days=0.0)
    cfg_on = CDGridPrimitiveEquationConfig(
        rf_tau_days=10.0, rf_cutoff_pa=1.0e5,
        # High cutoff so most levels damped
    )
    m_off = CDGridPrimitiveEquationModel(grid, coord, cfg_off)
    m_on = CDGridPrimitiveEquationModel(grid, coord, cfg_on)
    s_off = m_off.step(state, dt=300.0)
    s_on = m_on.step(state, dt=300.0)
    u_top_off = float(jnp.mean(jnp.abs(s_off.u_d.data[..., 0])))
    u_top_on = float(jnp.mean(jnp.abs(s_on.u_d.data[..., 0])))
    assert u_top_on < u_top_off, (
        f"PE RF did not damp top |u_d|: off={u_top_off:.3f}, "
        f"on={u_top_on:.3f}."
    )


def test_rf_no_effect_below_cutoff():
    grid, coord, state = _build_c8()
    cfg_on = CDGridPrimitiveEquationConfig(
        rf_tau_days=10.0, rf_cutoff_pa=100.0,
    )
    cfg_off = CDGridPrimitiveEquationConfig(rf_tau_days=0.0)
    m_on = CDGridPrimitiveEquationModel(grid, coord, cfg_on)
    m_off = CDGridPrimitiveEquationModel(grid, coord, cfg_off)
    s_on = m_on.step(state, dt=300.0)
    s_off = m_off.step(state, dt=300.0)
    pfull = (coord.A_full + coord.B_full) * coord.p_ref
    bottom_levs = np.where(np.asarray(pfull) >= 100.0)[0]
    if len(bottom_levs) > 0:
        np.testing.assert_allclose(
            np.asarray(s_on.u_d.data[..., bottom_levs]),
            np.asarray(s_off.u_d.data[..., bottom_levs]),
            rtol=1e-14, atol=1e-14,
        )
