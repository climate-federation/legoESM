"""FV3_3D iter 439: per-level sponge boost at k=1 / k=2 (PE).

Extends iter-438 (k=0 boost via ``d2_bg_k1``) with the k=1 +
k=2 boost via ``d2_bg_k2`` per FV3 ``dyn_core.F90:792, 802``::

    ! k=2 (1-based), our k=1 (0-based):
    if (d2_bg_k2 > 0.01)
        d2_divg = max(d2_bg, d2_bg_k2)
    ! k=3 (1-based), our k=2 (0-based):
    if (d2_bg_k2 > 0.05)
        d2_divg = max(d2_bg, 0.2 * d2_bg_k2)

Default 0.0 preserves bit-for-bit baseline.

Tests
-----

1. ``test_default_value_is_zero``.
2. ``test_threshold_at_001_no_effect`` — d2_bg_k2 = 0.005 is
   below the 0.01 threshold, so no override happens.
3. ``test_above_001_overrides_k1`` — d2_bg_k2 = 0.02 (above
   0.01, below 0.05) overrides k=1 only.
4. ``test_above_005_overrides_k1_and_k2`` — d2_bg_k2 = 2.0
   (FV3 production) overrides both k=1 and k=2.
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
from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init
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
    rng = np.random.default_rng(seed=439)
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


def test_default_value_is_zero():
    cfg = CDGridPrimitiveEquationConfig()
    assert cfg.corner_div_damp_d2_bg_k2 == 0.0


def test_threshold_at_001_no_effect():
    """d2_bg_k2 = 0.005 is below 0.01 threshold → no override."""
    grid, coord, state = _build_c8()
    cfg_base = CDGridPrimitiveEquationConfig(
        corner_div_damp_d2_bg=0.0005,
        corner_div_damp_d2_bg_k2=0.0,
    )
    cfg_below = CDGridPrimitiveEquationConfig(
        corner_div_damp_d2_bg=0.0005,
        corner_div_damp_d2_bg_k2=0.005,
    )
    m_a = CDGridPrimitiveEquationModel(grid, coord, cfg_base)
    m_b = CDGridPrimitiveEquationModel(grid, coord, cfg_below)
    s_a = m_a.step(state, dt=10.0)
    s_b = m_b.step(state, dt=10.0)
    np.testing.assert_allclose(
        np.asarray(s_a.T.data), np.asarray(s_b.T.data),
        rtol=1e-14, atol=1e-14,
    )


def test_above_001_overrides_k1():
    """d2_bg_k2 = 0.02 (between 0.01 and 0.05) → k=1 override
    but NOT k=2 override."""
    grid, coord, state = _build_c8()
    cfg_off = CDGridPrimitiveEquationConfig(
        corner_div_damp_d2_bg=0.0005,
        corner_div_damp_d2_bg_k2=0.0,
    )
    cfg_on = CDGridPrimitiveEquationConfig(
        corner_div_damp_d2_bg=0.0005,
        corner_div_damp_d2_bg_k2=0.02,
    )
    m_off = CDGridPrimitiveEquationModel(grid, coord, cfg_off)
    m_on = CDGridPrimitiveEquationModel(grid, coord, cfg_on)
    s_off = m_off.step(state, dt=10.0)
    s_on = m_on.step(state, dt=10.0)
    diff = jnp.abs(s_off.T.data - s_on.T.data)
    # k=1 should differ (override active); k=2 minimally
    # different (only via RK3/vertical coupling).
    diff_k1 = float(jnp.max(diff[..., 1]))
    assert diff_k1 > 0.0, "k=1 override is a no-op."


def test_above_005_overrides_k1_and_k2():
    """d2_bg_k2 = 2.0 (FV3 production) → k=1 + k=2 both override."""
    grid, coord, state = _build_c8()
    cfg_off = CDGridPrimitiveEquationConfig(
        corner_div_damp_d2_bg=0.0005,
        corner_div_damp_d2_bg_k2=0.0,
    )
    cfg_on = CDGridPrimitiveEquationConfig(
        corner_div_damp_d2_bg=0.0005,
        corner_div_damp_d2_bg_k2=2.0,
    )
    m_off = CDGridPrimitiveEquationModel(grid, coord, cfg_off)
    m_on = CDGridPrimitiveEquationModel(grid, coord, cfg_on)
    s_off = m_off.step(state, dt=10.0)
    s_on = m_on.step(state, dt=10.0)
    diff = jnp.abs(s_off.T.data - s_on.T.data)
    diff_k1 = float(jnp.max(diff[..., 1]))
    diff_k2 = float(jnp.max(diff[..., 2]))
    assert diff_k1 > 0.0, "k=1 override is a no-op."
    assert diff_k2 > 0.0, "k=2 override is a no-op."
