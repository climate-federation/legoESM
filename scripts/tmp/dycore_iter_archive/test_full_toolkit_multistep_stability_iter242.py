"""FV3_3D iter 242: multi-step stability test for the full PE+NH
toolkit + d_con stack at FV3 production defaults.

iter-184/185/237 verified single-step / 3-5-step AD-at-rest /
perturbed behavior.  iter-231/232 verified energy direction at
10 steps.  iter 242 extends to 50 steps to catch slow-growth
instabilities that single-step tests can't expose.

At C8 with random perturbation IC, all FV3 toolkit knobs ON
(corner-div nord=1 + cell-centre div_damp + damp_v + damp_w +
A_h smag_cs=0.20 + delt_max=1.0 + all 5 d_con knobs at 1.0),
50 steps must complete with no NaN.

Tests
-----

1. ``test_pe_full_toolkit_50steps_stable`` — PE 50 steps.
2. ``test_nh_full_toolkit_50steps_stable`` — NH 50 steps.
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
from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init
from legoesm.core.field import Field
from legoesm.core.state import NonHydrostaticState
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.grids.vertical import (
    compute_terrain_metric, create_height_coordinate,
    standard_hybrid_levels,
)


def test_pe_full_toolkit_50steps_stable():
    """PE: full toolkit + d_con stack + delt_max=1.0 stable
    over 50 steps from random IC."""
    n = 8
    nlev = 6
    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    coord = standard_hybrid_levels(nlev)
    state_cc = held_suarez_init(grid, coord)
    state = hydrostatic_to_fv3(state_cc, cdgrid)

    rng = np.random.default_rng(seed=242)
    n_corners = n + 1
    u_p = rng.uniform(-5.0, 5.0,
                      size=(6, n_corners, n_corners, nlev))
    v_p = rng.uniform(-5.0, 5.0,
                      size=(6, n_corners, n_corners, nlev))
    state = state._replace(
        u_d=state.u_d.replace(data=jnp.asarray(u_p)),
        v_d=state.v_d.replace(data=jnp.asarray(v_p)),
    )

    cfg = CDGridPrimitiveEquationConfig(
        # iter-19 PE production toolkit
        corner_div_damp_d2_bg=0.0005, corner_div_damp_dddmp=0.20,
        corner_div_damp_d4_bg=1e-3, corner_div_damp_nord=1,
        div_damp_coeff=1e6, div_damp_dddmp=0.20,
        damp_v=0.030, nord_v=2,
        use_fv3_a2b_zeta_corner=True,
        A_h=1e6, smagorinsky_cs=0.20,
        T_diss_coeff=0.05,
        hyperdiff_coeff=0.0, hyperdiff_ps_coeff=0.0,
        # All 4 d_con knobs at FV3 production 1.0
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
    for step in range(50):
        s = model.step(s, 100.0)

    # Verify all fields finite at end.
    assert jnp.all(jnp.isfinite(s.u_d.data)), (
        f"PE: u_d NaN/Inf at step 50 with full toolkit + d_con."
    )
    assert jnp.all(jnp.isfinite(s.v_d.data)), (
        f"PE: v_d NaN/Inf at step 50."
    )
    assert jnp.all(jnp.isfinite(s.T.data)), (
        f"PE: T NaN/Inf at step 50."
    )

    # Sanity: bounded growth (no runaway).  At C8 with 5 m/s IC
    # and full damping, max|u_d| should NOT exceed 100 m/s after
    # 50 steps.
    max_u = float(jnp.max(jnp.abs(s.u_d.data)))
    assert max_u < 100.0, (
        f"PE max|u_d|={max_u:.2f} > 100 m/s at step 50 — possible "
        f"slow-growth instability."
    )


def test_nh_full_toolkit_50steps_stable():
    """NH: full toolkit + d_con stack + delt_max=1.0 stable
    over 50 steps from random IC."""
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

    rng = np.random.default_rng(seed=2420)
    u_p = rng.uniform(-5.0, 5.0, size=(6, n, n, nlev))
    v_p = rng.uniform(-5.0, 5.0, size=(6, n, n, nlev))
    w_p = rng.uniform(-0.5, 0.5, size=(6, n, n, nlev + 1))

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
        # NH production toolkit
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        corner_div_damp_d2_bg=0.0005, corner_div_damp_dddmp=0.20,
        corner_div_damp_d4_bg=1e-3, corner_div_damp_nord=1,
        damp_v=0.030, nord_v=2,
        use_fv3_a2b_zeta_corner=True,
        div_damp_coeff=1e6, div_damp_dddmp=0.20,
        A_h=1e6, smagorinsky_cs=0.20,
        damp_w=0.030, nord_w=1,
        # All 5 d_con knobs at FV3 production 1.0
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
    for step in range(50):
        s = model.step(s, 10.0)

    # Verify all fields finite.
    assert jnp.all(jnp.isfinite(s.u.data)), "NH: u NaN/Inf at step 50."
    assert jnp.all(jnp.isfinite(s.v.data)), "NH: v NaN/Inf."
    assert jnp.all(jnp.isfinite(s.w.data)), "NH: w NaN/Inf."
    assert jnp.all(jnp.isfinite(s.theta_prime.data)), (
        "NH: theta_prime NaN/Inf."
    )

    max_u = float(jnp.max(jnp.abs(s.u.data)))
    assert max_u < 100.0, (
        f"NH max|u|={max_u:.2f} > 100 m/s — possible slow-growth "
        f"instability."
    )
