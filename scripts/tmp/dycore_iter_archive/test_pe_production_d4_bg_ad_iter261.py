"""FV3_3D iter 261: AD-at-rest gradient with iter-19 PRODUCTION
d4_bg=0.02 (vs iter-184/185 umbrellas' softer 1e-3).

iter-184/185 umbrellas use corner_div_damp_d4_bg=1e-3 + nord=1
for AD-at-rest stability — tighter than the iter-19 production
sweet spot d4_bg=0.02 + nord=1 used in real C36 30-day HS.

iter-261 verifies the iter-19 PRODUCTION setting also produces
finite jax.grad at rest, with the full d_con stack engaged.
The stronger d4_bg=0.02 amplifies the iter-187 smag_vort cap
contribution; if there's a sqrt-at-zero hazard in the d4_bg
path, this stronger setting could expose it.

Tests
-----

1. ``test_pe_production_d4_bg_grad_at_rest`` — PE with iter-19
   PRODUCTION d4_bg=0.02 + nord=1 + full d_con stack at rest
   state.  jax.grad w.r.t. T finite.
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


def test_pe_production_d4_bg_grad_at_rest():
    """iter-19 production d4_bg=0.02 must be AD-safe at rest
    state with the full d_con stack."""
    n = 8
    nlev = 6
    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    coord = standard_hybrid_levels(nlev)
    state_cc = held_suarez_init(grid, coord)
    state = hydrostatic_to_fv3(state_cc, cdgrid)
    state = state._replace(
        u_d=state.u_d.replace(data=jnp.zeros_like(state.u_d.data)),
        v_d=state.v_d.replace(data=jnp.zeros_like(state.v_d.data)),
    )

    cfg = CDGridPrimitiveEquationConfig(
        # iter-19 PRODUCTION values
        corner_div_damp_d2_bg=0.0005, corner_div_damp_dddmp=0.20,
        corner_div_damp_d4_bg=0.02,    # stronger than iter-184
        corner_div_damp_nord=1,
        div_damp_coeff=1e6, div_damp_dddmp=0.20,
        damp_v=0.030, nord_v=2,
        use_fv3_a2b_zeta_corner=True,
        A_h=1e6, smagorinsky_cs=0.20,
        T_diss_coeff=0.05,
        hyperdiff_coeff=1e16, hyperdiff_ps_coeff=0.0,
        # Full PE d_con stack at FV3 production 1.0
        damp_v_d_con=1.0,
        corner_div_damp_d_con=1.0,
        div_damp_d_con=1.0,
        ah_d_con=1.0,
        delt_max=1.0,
        use_conservation_fixer=False, fix_mass=False,
        zero_mean_ps_tendency=False,
    )
    model = CDGridPrimitiveEquationModel(grid, coord, cfg)

    def loss_fn(T_data):
        s = state._replace(T=state.T.replace(data=T_data))
        for _ in range(3):
            s = model.step(s, 200.0)
        return jnp.mean(s.T.data ** 2)

    grad = jax.grad(loss_fn)(state.T.data)
    assert jnp.all(jnp.isfinite(grad)), (
        "PE iter-19 production d4_bg=0.02 + full d_con stack at "
        "rest state must produce finite jax.grad.  Stronger "
        "d4_bg amplifies the iter-187 smag_vort cap path; if "
        "this fails, check the iter-183 sqrt(0) double-where "
        "fix is intact under the stronger damping coefficient."
    )
