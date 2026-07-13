"""FV3_3D iter 262: extend iter-245 PE C36 production stability
test from 20 steps to 50 steps with d_con stack.

iter-245 ran 20 PE × dt=200 = 1 hour at C36 production.
iter-262 extends to 50 steps × dt=200 = ~2.8 hours integrated
to expose any slow-growth instability that the 20-step test
would miss at production resolution.

Tests
-----

1. ``test_pe_c36_production_d_con_50steps_stable`` — C36 PE +
   iter-19 production toolkit + full d_con stack + delt_max=1.0,
   50 steps × dt=200.  All fields finite + max|u| < 100 m/s +
   physical T bounds.
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


def test_pe_c36_production_d_con_50steps_stable():
    """C36 PE + iter-19 production toolkit + d_con stack stable
    for 50 steps."""
    n = 36
    nlev = 6
    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    coord = standard_hybrid_levels(nlev)
    state_cc = held_suarez_init(grid, coord)
    state = hydrostatic_to_fv3(state_cc, cdgrid)

    cfg = CDGridPrimitiveEquationConfig(
        # iter-19 PRODUCTION toolkit
        corner_div_damp_d2_bg=0.0005, corner_div_damp_dddmp=0.20,
        corner_div_damp_d4_bg=0.02, corner_div_damp_nord=1,
        div_damp_coeff=1e6, div_damp_dddmp=0.20,
        damp_v=0.030, nord_v=2,
        use_fv3_a2b_zeta_corner=True,
        A_h=1e7, smagorinsky_cs=0.20,
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

    s = state
    dt = 200.0
    for _ in range(50):
        s = model.step(s, dt)

    assert jnp.all(jnp.isfinite(s.u_d.data)), (
        "PE C36 production: u_d NaN/Inf at step 50."
    )
    assert jnp.all(jnp.isfinite(s.v_d.data))
    assert jnp.all(jnp.isfinite(s.T.data))

    max_u = float(jnp.max(jnp.abs(s.u_d.data)))
    assert max_u < 100.0, (
        f"PE C36 production max|u_d|={max_u:.2f} > 100 m/s "
        f"after 50 steps — slow-growth instability at C36."
    )

    min_T = float(jnp.min(s.T.data))
    max_T = float(jnp.max(s.T.data))
    assert min_T > 100.0, f"min(T)={min_T:.1f} K — too cold."
    assert max_T < 400.0, f"max(T)={max_T:.1f} K — too hot."
