"""FV3_3D iter 375: AD-at-rest umbrella for PE full FV3-fidelity
stack INCLUDING iter-370 cross_face flag.

iter-355 covered PE full stack without cross_face (it was iter-
370).  iter-375 extends to include all 3 PE FV3-fidelity flags
+ duogrid + full toolkit.

Tests
-----

1. ``test_full_pe_with_cross_face_grad_at_rest`` — jax.grad
   finite through 3 PE steps with all PE flags ON + duogrid +
   full toolkit at rest.
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


def test_full_pe_with_cross_face_grad_at_rest():
    n = 8
    nlev = 5
    grid = create_cubed_sphere(n, use_duogrid=True)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    coord = standard_hybrid_levels(nlev)
    state_cc = held_suarez_init(grid, coord)
    state = hydrostatic_to_fv3(state_cc, cdgrid)
    rest = state._replace(
        u_d=state.u_d.replace(data=jnp.zeros_like(state.u_d.data)),
        v_d=state.v_d.replace(data=jnp.zeros_like(state.v_d.data)),
    )

    cfg = CDGridPrimitiveEquationConfig(
        corner_div_damp_d2_bg=0.0005, corner_div_damp_dddmp=0.20,
        corner_div_damp_d4_bg=1e-3, corner_div_damp_nord=1,
        damp_v=0.030, nord_v=1,
        use_fv3_a2b_zeta_corner=True,
        div_damp_coeff=1e6, div_damp_dddmp=0.20,
        A_h=1e6, smagorinsky_cs=0.20,
        delt_max=1.0,
        damp_v_d_con=1.0, corner_div_damp_d_con=1.0,
        div_damp_d_con=1.0, ah_d_con=1.0,
        # All 3 PE FV3-fidelity flags
        use_fv3_metric_aware_d_con=True,
        use_fv3_cross_face_du_proj=True,
        # use_fv3_a2b_zeta_corner already set above
        hyperdiff_coeff=0.0, hyperdiff_ps_coeff=0.0,
        use_conservation_fixer=False, fix_mass=False,
    )
    m = CDGridPrimitiveEquationModel(grid, coord, cfg)

    def loss(amp):
        s = rest._replace(
            u_d=rest.u_d.replace(
                data=amp * jnp.ones_like(rest.u_d.data),
            ),
        )
        for _ in range(3):
            s = m.step(s, 100.0)
        return jnp.mean(s.T.data ** 2) + jnp.mean(s.u_d.data ** 2)

    g = jax.grad(loss)(0.0)
    assert jnp.isfinite(g), (
        "FULL PE FV3-fidelity stack with cross_face flag AD "
        "grad NaN — interaction surface hazard."
    )
