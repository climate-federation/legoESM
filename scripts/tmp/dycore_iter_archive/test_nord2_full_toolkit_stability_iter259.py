"""FV3_3D iter 259: multi-step stability test with FV3 production
default ``nord_v=2`` (del-6 vorticity damping).

Most tests use ``nord_v=2`` for damp_v but ``nord=1`` for the
iter-187 corner-div smag_vort cap path.  iter-187's
``test_pe_nord2_higher_order_branch_finite`` and
``test_nh_nord2_fv3_production_default_finite`` cover NORD=2
in single-step finite checks.  iter-259 extends to 20 steps
to expose any slow-growth instability in the nord=2 path.

Tests
-----

1. ``test_pe_nord2_corner_div_20step_stable`` — PE 20 steps
   with corner_div_damp_nord=2 + d4_bg=0.16 (FV3 production
   default).
2. ``test_nh_nord2_corner_div_20step_stable`` — NH 20 steps
   with same nord=2.

Both engage the FULL d_con stack (delt_max=1.0, all knobs at
1.0) to verify d_con composes with nord=2 corner-div damping.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.dynamics.gcm.compressible_euler_cdgrid import (
    CDGridCompressibleEulerConfig,
    CDGridCompressibleEulerModel,
)
from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
    CDGridPrimitiveEquationConfig,
    CDGridPrimitiveEquationModel,
    hydrostatic_to_fv3,
)
from legoesm.atmosphere.held_suarez import held_suarez_init
from legoesm.core.field import Field
from legoesm.core.state import NonHydrostaticState
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.grids.vertical import (
    compute_terrain_metric, create_height_coordinate,
    standard_hybrid_levels,
)


def test_pe_nord2_corner_div_20step_stable():
    """PE 20 steps with nord_v=2 + corner_div_damp_nord=2 +
    d4_bg=0.16 (FV3 production default for nord=2) + full
    d_con stack.  Tighter d4_bg=1e-4 used here because
    d4_bg=0.16 over-damps at C8."""
    n = 8
    nlev = 6
    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    coord = standard_hybrid_levels(nlev)
    state_cc = held_suarez_init(grid, coord)
    state = hydrostatic_to_fv3(state_cc, cdgrid)

    rng = np.random.default_rng(seed=259)
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
        # nord=2 path (FV3 production default for damp_v).
        damp_v=0.030, nord_v=2,
        corner_div_damp_d2_bg=0.0005, corner_div_damp_dddmp=0.20,
        # d4_bg=1e-4 with nord=2 (production default would over-
        # damp at C8; iter-187 test uses 1e-4 for C8).
        corner_div_damp_d4_bg=1e-4, corner_div_damp_nord=2,
        div_damp_coeff=1e6, div_damp_dddmp=0.20,
        use_fv3_a2b_zeta_corner=True,
        A_h=1e6, smagorinsky_cs=0.20,
        T_diss_coeff=0.05,
        hyperdiff_coeff=0.0,
        # Full d_con stack at FV3 production 1.0
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
    for _ in range(20):
        s = model.step(s, 100.0)

    assert jnp.all(jnp.isfinite(s.u_d.data))
    assert jnp.all(jnp.isfinite(s.v_d.data))
    assert jnp.all(jnp.isfinite(s.T.data))
    max_u = float(jnp.max(jnp.abs(s.u_d.data)))
    assert max_u < 50.0, (
        f"PE nord=2 max|u_d|={max_u:.2f} > 50 m/s after 20 "
        f"steps — possible nord=2 instability."
    )


def test_nh_nord2_corner_div_20step_stable():
    """NH 20 steps with nord_v=2 + corner_div_damp_nord=2."""
    n = 8
    nlev = 5
    z_top = 30000.0
    grid = create_cubed_sphere(n)
    height_coord = create_height_coordinate(nlev, z_top)
    terrain = jnp.zeros((6, n, n))
    terrain_metric = compute_terrain_metric(terrain, height_coord)

    dims_3d = ("face", "x", "y", "level")
    dims_w = ("face", "x", "y", "level_half")
    dims_2d = ("face", "x", "y")

    rng = np.random.default_rng(seed=2590)
    u_p = rng.uniform(-3.0, 3.0, size=(6, n, n, nlev))
    v_p = rng.uniform(-3.0, 3.0, size=(6, n, n, nlev))
    w_p = rng.uniform(-0.3, 0.3, size=(6, n, n, nlev + 1))

    state = NonHydrostaticState(
        u=Field(data=jnp.asarray(u_p), name="u",
                dims=dims_3d, units="m/s"),
        v=Field(data=jnp.asarray(v_p), name="v",
                dims=dims_3d, units="m/s"),
        w=Field(data=jnp.asarray(w_p), name="w",
                dims=dims_w, units="m/s"),
        theta_prime=Field(data=jnp.zeros((6, n, n, nlev)),
                          name="theta_prime",
                          dims=dims_3d, units="K"),
        rho_prime=Field(data=jnp.zeros((6, n, n, nlev)),
                        name="rho_prime",
                        dims=dims_3d, units="kg/m^3"),
        phis=Field(data=jnp.zeros((6, n, n)), name="phis",
                   dims=dims_2d, units="m^2/s^2"),
        tracers=Field(data=jnp.zeros((6, n, n, nlev, 0)),
                      name="tracers",
                      dims=("face", "x", "y", "level", "tracer"),
                      units="kg/kg"),
    )

    cfg = CDGridCompressibleEulerConfig(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        damp_v=0.030, nord_v=2,
        corner_div_damp_d2_bg=0.0005, corner_div_damp_dddmp=0.20,
        corner_div_damp_d4_bg=1e-4, corner_div_damp_nord=2,
        div_damp_coeff=1e6, div_damp_dddmp=0.20,
        use_fv3_a2b_zeta_corner=True,
        A_h=1e6, smagorinsky_cs=0.20,
        damp_w=0.030, nord_w=2,
        damp_w_d_con=1.0,
        damp_v_d_con=1.0,
        corner_div_damp_d_con=1.0,
        div_damp_d_con=1.0,
        ah_d_con=1.0,
        delt_max=1.0,
    )
    model = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg,
    )

    s = state
    for _ in range(20):
        s = model.step(s, 10.0)

    assert jnp.all(jnp.isfinite(s.u.data))
    assert jnp.all(jnp.isfinite(s.v.data))
    assert jnp.all(jnp.isfinite(s.w.data))
    assert jnp.all(jnp.isfinite(s.theta_prime.data))
    max_u = float(jnp.max(jnp.abs(s.u.data)))
    assert max_u < 50.0, (
        f"NH nord=2 max|u|={max_u:.2f} > 50 m/s after 20 steps."
    )
