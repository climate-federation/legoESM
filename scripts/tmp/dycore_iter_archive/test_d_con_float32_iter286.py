"""FV3_3D iter 286: float32 sanity test for the PE d_con stack.

All other d_con tests use ``jax.config.update("jax_enable_x64",
True)`` for float64 precision.  Some workflows (e.g., ML
training) run with float32 default.  This test verifies the
PE d_con stack runs without NaN at float32 precision.

Tests
-----

1. ``test_pe_d_con_runs_at_float32`` — PE iter-19 toolkit +
   d_con stack at float32 precision; 5 steps stable.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# DO NOT enable x64 — test float32 default.

from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
    CDGridPrimitiveEquationConfig,
    CDGridPrimitiveEquationModel,
    hydrostatic_to_fv3,
)
from legoesm.atmosphere.held_suarez import held_suarez_init
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.grids.vertical import standard_hybrid_levels


def test_pe_d_con_runs_at_float32():
    """PE iter-19 toolkit + full d_con stack at float32 — 5
    steps × dt=100, all fields finite + bounded."""
    n = 8
    nlev = 6
    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    coord = standard_hybrid_levels(nlev)
    state_cc = held_suarez_init(grid, coord)
    state = hydrostatic_to_fv3(state_cc, cdgrid)

    rng = np.random.default_rng(seed=286)
    n_corners = n + 1
    u_p = rng.uniform(-3.0, 3.0,
                      size=(6, n_corners, n_corners, nlev))
    v_p = rng.uniform(-3.0, 3.0,
                      size=(6, n_corners, n_corners, nlev))
    state = state._replace(
        u_d=state.u_d.replace(data=jnp.asarray(u_p)),
        v_d=state.v_d.replace(data=jnp.asarray(v_p)),
    )

    cfg = CDGridPrimitiveEquationConfig(
        # iter-19-style + d_con stack
        corner_div_damp_d2_bg=0.0005, corner_div_damp_dddmp=0.20,
        corner_div_damp_d4_bg=1e-3, corner_div_damp_nord=1,
        div_damp_coeff=1e6, div_damp_dddmp=0.20,
        damp_v=0.030, nord_v=2,
        use_fv3_a2b_zeta_corner=True,
        A_h=1e6, smagorinsky_cs=0.20,
        T_diss_coeff=0.05,
        hyperdiff_coeff=0.0, hyperdiff_ps_coeff=0.0,
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
    for _ in range(5):
        s = model.step(s, 100.0)

    assert jnp.all(jnp.isfinite(s.u_d.data))
    assert jnp.all(jnp.isfinite(s.v_d.data))
    assert jnp.all(jnp.isfinite(s.T.data))

    max_u = float(jnp.max(jnp.abs(s.u_d.data)))
    assert max_u < 50.0, (
        f"PE d_con stack at float32: max|u_d|={max_u:.2f}"
    )

    # Verify dtype is float32 (test the precision actually
    # being used).
    assert s.u_d.data.dtype == jnp.float32, (
        f"Expected float32 dtype but got {s.u_d.data.dtype}.  "
        f"jax_enable_x64 may have been set inadvertently."
    )
