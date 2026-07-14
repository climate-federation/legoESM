"""FV3_3D iter 290: PE iter-19 production AD-at-rest at JAX
default float32 (mirror of iter-261 which uses jax_enable_x64).

iter-261 verified PE iter-19 production d4_bg=0.02 +
full d_con stack produces finite jax.grad at rest under
float64.  iter-290 verifies the same property holds under
JAX default float32.

If float32 AD-at-rest works, this confirms the iter-183
sqrt(0) double-where AD-safety pattern + iter-19 production
d_con stack do NOT depend on x64 precision — important
for ML-training-style differentiable workflows that
typically run at float32.

Tests
-----

1. ``test_pe_production_d_con_grad_at_rest_float32`` — PE
   iter-19 production setting + full d_con stack + AD-at-
   rest, at JAX default float32.  jax.grad w.r.t. T finite.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# DO NOT enable x64 — explicit float32 default.

from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
    CDGridPrimitiveEquationConfig,
    CDGridPrimitiveEquationModel,
    hydrostatic_to_fv3,
)
from legoesm.atmosphere.held_suarez import held_suarez_init
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.grids.vertical import standard_hybrid_levels


def test_pe_production_d_con_grad_at_rest_float32():
    """iter-19 production d4_bg=0.02 + full d_con stack must be
    AD-safe at rest state under JAX default float32."""
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
        corner_div_damp_d4_bg=0.02,
        corner_div_damp_nord=1,
        div_damp_coeff=1e6, div_damp_dddmp=0.20,
        damp_v=0.030, nord_v=2,
        use_fv3_a2b_zeta_corner=True,
        A_h=1e6, smagorinsky_cs=0.20,
        T_diss_coeff=0.05,
        hyperdiff_coeff=1e16, hyperdiff_ps_coeff=0.0,
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
        "rest state must produce finite jax.grad under JAX "
        "default float32.  Confirms the iter-183 sqrt(0) double-"
        "where AD-safety pattern doesn't depend on x64."
    )
    # Sanity-check dtype: confirm we genuinely tested float32.
    assert grad.dtype == jnp.float32, (
        f"Expected float32 grad but got {grad.dtype}.  "
        f"jax_enable_x64 may have been set inadvertently."
    )
