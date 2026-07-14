"""FV3_3D iter 215: PE counterpart of iter-174 cell-centre
divergence-damp direction test.

iter-174 verified that NH ``div_damp_coeff > 0`` reduces
``mean(|div_v|)`` on a sinusoidal divergent IC.  PE has the same
iter-5 cell-centre div_damp wiring but no analogous direction
test — its existing tests cover the dddmp adaptive cap variations
without comparing div magnitude before/after damping.

iter 215 mirrors iter-174 for PE.

Tests
-----

1. ``test_pe_cell_centre_div_damp_reduces_divergence`` — a
   sinusoidal divergent IC + div_damp_coeff > 0 reduces
   ``mean(|div_v|)`` after 5 PE steps vs no-damping baseline.
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
    Mirrors iter-213 fixture."""
    n = 8
    nlev = 6
    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    coord = standard_hybrid_levels(nlev)
    state_cc = held_suarez_init(grid, coord)
    state = hydrostatic_to_fv3(state_cc, cdgrid)

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


def test_pe_cell_centre_div_damp_reduces_divergence(divergent_pe_state):
    """PE: div_damp_coeff > 0 reduces ``mean(|div_v|)`` vs no-damp
    baseline on a divergent IC.  Catches sign error in iter-5 PE
    wiring."""
    grid, cdgrid, coord, state = divergent_pe_state

    cfg_baseline = CDGridPrimitiveEquationConfig(
        # other damping off so we isolate cell-centre div_damp
        A_h=0.0, hyperdiff_coeff=0.0, hyperdiff_ps_coeff=0.0,
        div_damp_coeff=0.0, div_damp_dddmp=0.0,
        use_conservation_fixer=False, fix_mass=False,
    )
    cfg_active = CDGridPrimitiveEquationConfig(
        A_h=0.0, hyperdiff_coeff=0.0, hyperdiff_ps_coeff=0.0,
        # PE production uses 1e7; matrix scales by (48/n)² so at
        # C8 the equivalent is ~3.6e7.  Use 1e8 here — large enough
        # for the test to detect the damping signal above PE
        # baseline noise but small enough not to destabilize the
        # idealized HS init (1e10 like NH iter-174 over-damps PE
        # because PE dt is 10x NH).
        div_damp_coeff=1e8, div_damp_dddmp=0.0,    # constant path
        use_conservation_fixer=False, fix_mass=False,
    )

    m_baseline = CDGridPrimitiveEquationModel(grid, coord, cfg_baseline)
    m_active = CDGridPrimitiveEquationModel(grid, coord, cfg_active)

    s_baseline = _step5(m_baseline, state)
    s_active = _step5(m_active, state)

    div_baseline = _mean_abs_div_pe(s_baseline, cdgrid)
    div_active = _mean_abs_div_pe(s_active, cdgrid)

    assert div_active < div_baseline, (
        f"PE cell-centre div_damp_coeff > 0 must REDUCE mean|div_v|: "
        f"baseline={div_baseline:.4e}, active={div_active:.4e} "
        f"(damping AMPLIFIED divergence — sign error in iter-5 PE "
        f"wiring)."
    )
    rel_reduction = (div_baseline - div_active) / div_baseline
    assert rel_reduction > 0.001, (
        f"PE cell-centre div_damp must reduce mean|div_v| by at "
        f"least 0.1 % (observed {100*rel_reduction:.3f} %).  "
        f"At C8 with HS init the residual divergence baseline is "
        f"already small (~1e-6 1/s), so the % reduction is modest "
        f"even with strong damping; the direction is the key check."
    )
