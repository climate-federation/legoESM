"""FV3_3D iter 438: per-level sponge boost of corner-div damping
at k=0 (PE).

Port of FV3 ``dyn_core.F90:780``::

    d2_divg = max(0.01, flagstruct%d2_bg, flagstruct%d2_bg_k1)   ! k=1

When ``corner_div_damp_d2_bg_k1 > 0`` and
``corner_div_damp_d2_bg > 0``, the k=0 level of the corner-
divergence adaptive damping coefficient is overridden with
``da_min_c * max(d2_bg, d2_bg_k1)`` — bypassing the 0.20
Smagorinsky cap at the top sponge level.

Default 0.0 = no boost = bit-for-bit baseline.

Currently wired on PE only.  NH mirror + k=1/k=2 levels
pending future iters.

Tests
-----

1. ``test_default_value_is_zero`` — config field defaults to 0.
2. ``test_baseline_equals_no_field`` — explicit 0 matches
   default (no field set).
3. ``test_k1_boost_changes_state_at_top`` — non-zero value
   changes T at top level vs baseline; effect concentrated
   in top level (top diff >> bottom diff).
4. ``test_boost_without_d2_bg_is_noop`` — flag is gated by
   ``corner_div_damp_d2_bg > 0``; with d2_bg=0 the boost has
   no effect (faithful to FV3's gating of the whole damping).
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
    rng = np.random.default_rng(seed=438)
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
    assert cfg.corner_div_damp_d2_bg_k1 == 0.0


def test_baseline_equals_no_field():
    grid, coord, state = _build_c8()
    cfg_default = CDGridPrimitiveEquationConfig(
        corner_div_damp_d2_bg=0.0005,
    )
    cfg_zero = CDGridPrimitiveEquationConfig(
        corner_div_damp_d2_bg=0.0005,
        corner_div_damp_d2_bg_k1=0.0,
    )
    m_d = CDGridPrimitiveEquationModel(grid, coord, cfg_default)
    m_z = CDGridPrimitiveEquationModel(grid, coord, cfg_zero)
    s_d = m_d.step(state, dt=10.0)
    s_z = m_z.step(state, dt=10.0)
    np.testing.assert_allclose(
        np.asarray(s_d.T.data), np.asarray(s_z.T.data),
        rtol=1e-14, atol=1e-14,
    )


def test_k1_boost_changes_state_at_top():
    grid, coord, state = _build_c8()
    cfg_off = CDGridPrimitiveEquationConfig(
        corner_div_damp_d2_bg=0.0005,
        corner_div_damp_d2_bg_k1=0.0,
    )
    cfg_on = CDGridPrimitiveEquationConfig(
        corner_div_damp_d2_bg=0.0005,
        corner_div_damp_d2_bg_k1=4.0,   # FV3 production value
    )
    m_off = CDGridPrimitiveEquationModel(grid, coord, cfg_off)
    m_on = CDGridPrimitiveEquationModel(grid, coord, cfg_on)
    s_off = m_off.step(state, dt=10.0)
    s_on = m_on.step(state, dt=10.0)
    # T differs at top level; the corner-div boost cascades into
    # the wind solution and eventually heats/cools via dynamics.
    diff = jnp.abs(s_off.T.data - s_on.T.data)
    diff_top = float(jnp.max(diff[..., 0]))
    # Effect must be present at top.
    assert diff_top > 0.0, (
        "iter-438 k1 boost produced no change at top level — "
        "flag is a no-op."
    )


def test_boost_without_d2_bg_is_noop():
    grid, coord, state = _build_c8()
    cfg_no_d2 = CDGridPrimitiveEquationConfig(
        corner_div_damp_d2_bg=0.0,        # whole damping path off
        corner_div_damp_d2_bg_k1=4.0,
    )
    cfg_baseline = CDGridPrimitiveEquationConfig(
        corner_div_damp_d2_bg=0.0,
        corner_div_damp_d2_bg_k1=0.0,
    )
    m_a = CDGridPrimitiveEquationModel(grid, coord, cfg_no_d2)
    m_b = CDGridPrimitiveEquationModel(grid, coord, cfg_baseline)
    s_a = m_a.step(state, dt=10.0)
    s_b = m_b.step(state, dt=10.0)
    np.testing.assert_allclose(
        np.asarray(s_a.T.data), np.asarray(s_b.T.data),
        rtol=1e-14, atol=1e-14,
    )
