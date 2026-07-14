"""FV3_3D iter 213: PE counterpart of iter-198 — quantitative test
that nord >= 1 corner-div damp reduces ``mean(|div_v|)`` MORE than
nord=0 alone.

iter-198 verified this for the NH 3D path (iter-187 smag_vort cap
in NH).  PE has the same iter-187 wiring (the smag_vort recompute
inside the d4_bg + nord > 0 branch in
``primitive_eq_cdgrid.py``).  iter 213 mirrors iter-198 for PE,
catching a silent no-op in the higher-order branch on the
hydrostatic path that the iter-187 finiteness tests would not
catch.

Tests
-----

1. ``test_pe_corner_div_damp_nord1_reduces_more_than_d2_only`` —
   nord=1 + d4_bg > 0 reduces ``mean(|div_v|)`` MORE than
   nord=0 alone (same d2_bg).
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
from legoesm.core.operators_cdgrid import (
    dgrid_to_cgrid, cgrid_divergence,
)
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.grids.vertical import standard_hybrid_levels


@pytest.fixture(scope="module")
def divergent_pe_state():
    """PE state with sinusoidal divergent perturbation in u_d.
    Mirrors iter-198's NH fixture but for the PE D-grid corner
    layout (u_d, v_d shape (6, n+1, n+1, nlev))."""
    n = 8
    nlev = 6
    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    coord = standard_hybrid_levels(nlev)
    state_cc = held_suarez_init(grid, coord)
    state = hydrostatic_to_fv3(state_cc, cdgrid)

    # Sinusoidal monopole pattern in u_d (corners shape n+1).
    n_corners = n + 1
    i_idx = jnp.arange(n_corners)
    j_idx = jnp.arange(n_corners)
    k_idx = jnp.arange(nlev)
    pattern = (
        jnp.sin(2 * jnp.pi * i_idx[None, :, None, None] / n_corners)
        * jnp.cos(2 * jnp.pi * j_idx[None, None, :, None] / n_corners)
        * jnp.ones_like(k_idx[None, None, None, :], dtype=jnp.float64)
    )
    pattern = jnp.broadcast_to(pattern, (6, n_corners, n_corners, nlev))
    U0 = 5.0
    u_perturb = jnp.asarray(U0 * pattern, dtype=jnp.float64)

    state = state._replace(
        u_d=state.u_d.replace(data=u_perturb),
        v_d=state.v_d.replace(
            data=jnp.zeros_like(state.v_d.data),
        ),
    )
    return grid, cdgrid, coord, state


def _mean_abs_div_pe(state, cdgrid):
    """Compute ``mean(|div_v|)`` for the PE state.  PE stores u_d,
    v_d at corners; project to C-grid then call cgrid_divergence."""
    u_d = state.u_d.data
    v_d = state.v_d.data
    u_c, v_c = dgrid_to_cgrid(u_d, v_d, cdgrid)
    div_v = cgrid_divergence(u_c, v_c, cdgrid)
    return float(jnp.mean(jnp.abs(div_v)))


def _step5(model, state, dt=100.0):
    s = state
    for _ in range(5):
        s = model.step(s, dt)
    return s


def test_pe_corner_div_damp_nord1_reduces_more_than_d2_only(divergent_pe_state):
    """PE: nord=1 + d4_bg > 0 reduces ``mean(|div_v|)`` MORE than
    nord=0 alone with same d2_bg (iter-198 PE counterpart).
    Catches a silent no-op in the iter-18-equivalent higher-order
    branch on the hydrostatic path."""
    grid, cdgrid, coord, state = divergent_pe_state

    common = dict(
        corner_div_damp_d2_bg=0.001,
        corner_div_damp_dddmp=0.20,
        use_conservation_fixer=False, fix_mass=False,
    )
    cfg_d2_only = CDGridPrimitiveEquationConfig(
        **common,
        corner_div_damp_d4_bg=0.0,
        corner_div_damp_nord=0,
    )
    cfg_d4_nord1 = CDGridPrimitiveEquationConfig(
        **common,
        corner_div_damp_d4_bg=1e-3,
        corner_div_damp_nord=1,
    )

    m_d2 = CDGridPrimitiveEquationModel(grid, coord, cfg_d2_only)
    m_d4 = CDGridPrimitiveEquationModel(grid, coord, cfg_d4_nord1)

    s_d2 = _step5(m_d2, state)
    s_d4 = _step5(m_d4, state)

    div_d2 = _mean_abs_div_pe(s_d2, cdgrid)
    div_d4 = _mean_abs_div_pe(s_d4, cdgrid)

    assert div_d4 < div_d2, (
        f"PE: nord=1 + d4_bg > 0 must reduce mean|div_v| MORE than "
        f"nord=0 alone: nord=0 div={div_d2:.4e}, "
        f"nord=1 div={div_d4:.4e} (higher-order PE branch did not "
        f"add damping — silent regression in d4_bg/dd8 wiring)."
    )
