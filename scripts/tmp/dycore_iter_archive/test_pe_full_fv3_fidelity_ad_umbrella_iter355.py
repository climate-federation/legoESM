"""FV3_3D iter 355: AD-at-rest umbrella for the FULL PE
FV3-fidelity stack (duogrid + metric + a2b_zeta + full toolkit
+ all d_con).

PE counterpart of NH iter-346.  PE has 2 FV3-fidelity flags
(use_fv3_metric_aware_d_con + use_fv3_a2b_zeta_corner) vs NH's 4.

Combines:
* iter-333 duogrid wiring at PE ke_correction halo (grid)
* iter-14 a2b_ord4 zeta corner
* iter-338/347/349/351 metric-aware d_con at ALL 4 PE sites
* Full FV3 damping toolkit + 4 d_con knobs at production 1.0

Catches AD hazards at the combined PE FV3-fidelity surface.

Tests
-----

1. ``test_full_pe_fv3_fidelity_grad_at_rest`` — jax.grad finite
   through 3 PE steps with all PE flags ON + full toolkit at
   rest state.
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


def test_full_pe_fv3_fidelity_grad_at_rest():
    """jax.grad through 3 PE steps with ALL PE FV3-fidelity
    flags ON + full toolkit at rest state."""
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
        # Full FV3 damping toolkit
        corner_div_damp_d2_bg=0.0005,
        corner_div_damp_dddmp=0.20,
        corner_div_damp_d4_bg=1e-3,
        corner_div_damp_nord=1,
        damp_v=0.030, nord_v=1,
        use_fv3_a2b_zeta_corner=True,
        div_damp_coeff=1e6, div_damp_dddmp=0.20,
        A_h=1e6, smagorinsky_cs=0.20,
        delt_max=1.0,
        # All 4 PE d_con knobs at production
        damp_v_d_con=1.0,
        corner_div_damp_d_con=1.0,
        div_damp_d_con=1.0,
        ah_d_con=1.0,
        # PE FV3-fidelity flag
        use_fv3_metric_aware_d_con=True,
        # Conservation settings
        use_conservation_fixer=False, fix_mass=False,
        hyperdiff_coeff=0.0, hyperdiff_ps_coeff=0.0,
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
        "FULL PE FV3-fidelity stack (duogrid + metric_aware +"
        " a2b_zeta + full toolkit + all 4 d_con) AD-at-rest "
        "grad NaN — AD hazard at the combined-flag surface."
    )
