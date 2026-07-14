"""FV3_3D iter 245: PE iter-19 production values + d_con stack
stability test at C36 (production resolution).

iter-242 / iter-244 used iter-192-style d4_bg=1e-3 + nord=1
(softer AD-friendly setting).  iter-19 PRODUCTION uses
d4_bg=0.02 + nord=1 — much stronger nord-mode damping that
iter-19 30-day HS validated as the cube-imprint sweet spot.

iter 245 verifies the iter-19 production setting + the full
d_con stack composes stably at C36 (true production grid).
20 steps × dt=200 = 1 hour of integrated time, sufficient to
detect early-spin-up instability or NaN from over-damping.

Tests
-----

1. ``test_pe_production_iter19_with_d_con_stable_at_C36`` —
   C36 HS init + iter-19 production toolkit + all 4 PE d_con
   knobs at 1.0 + delt_max=1.0.  20 steps stable.
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


def test_pe_production_iter19_with_d_con_stable_at_C36():
    """C36 HS + iter-19 production toolkit + d_con stack ON
    is stable for 20 steps."""
    n = 36
    nlev = 6
    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    coord = standard_hybrid_levels(nlev)
    state_cc = held_suarez_init(grid, coord)
    state = hydrostatic_to_fv3(state_cc, cdgrid)

    cfg = CDGridPrimitiveEquationConfig(
        # iter-19 PRODUCTION values for HS C36 30-day cube-imprint
        # sweet spot.
        corner_div_damp_d2_bg=0.0005, corner_div_damp_dddmp=0.20,
        corner_div_damp_d4_bg=0.02,    # iter-19 production
        corner_div_damp_nord=1,
        div_damp_coeff=1e6, div_damp_dddmp=0.20,
        damp_v=0.030, nord_v=2,
        use_fv3_a2b_zeta_corner=True,
        # iter-33 production calibration: ah_x10 (10x base A_h).
        # For C36 with _laplacian_visc_cube_v2 the equivalent is
        # ~5e6 * 10 = 5e7.  Use 1e7 as a moderate value to avoid
        # over-damping the HS init transient.
        A_h=1e7, smagorinsky_cs=0.20,
        T_diss_coeff=0.05,
        hyperdiff_coeff=1e16, hyperdiff_ps_coeff=0.0,
        # All 4 PE d_con knobs at FV3 production 1.0 + delt_max=1.0
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
    dt = 200.0    # iter-19 production dt at C36
    for step in range(20):
        s = model.step(s, dt)

    # Verify all fields finite.
    assert jnp.all(jnp.isfinite(s.u_d.data)), (
        f"PE C36 production: u_d NaN/Inf at step 20."
    )
    assert jnp.all(jnp.isfinite(s.v_d.data)), (
        f"PE C36 production: v_d NaN/Inf at step 20."
    )
    assert jnp.all(jnp.isfinite(s.T.data)), (
        f"PE C36 production: T NaN/Inf at step 20."
    )

    # Sanity: bounded growth.  HS init at C36 with cube imprint
    # damping should keep max|u| in the synoptic range (~50 m/s).
    max_u = float(jnp.max(jnp.abs(s.u_d.data)))
    assert max_u < 100.0, (
        f"PE C36 production max|u_d|={max_u:.2f} > 100 m/s — "
        f"possible cube imprint blow-up at production resolution."
    )

    # Sanity: T stays in a physically-reasonable range.
    min_T = float(jnp.min(s.T.data))
    max_T = float(jnp.max(s.T.data))
    assert min_T > 100.0, f"min(T)={min_T:.1f} K — too cold."
    assert max_T < 400.0, f"max(T)={max_T:.1f} K — too hot."
