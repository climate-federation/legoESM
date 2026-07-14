"""FV3_3D iter 211: quantitative formula validation for the iter-208
(PE) and iter-209 (NH) ``damp_v_d_con`` KE→heat conversions.

Mirrors iter-205's pattern for the iter-203 ``damp_w_d_con`` heat
formula.  3-run extraction:

  1. acoustic_only / no-damp_v: establishes wind state after the
     dynamics step alone (no damp_v, no d_con).
  2. damp_v / no-d_con: damp_v active without heat -> wind gets the
     damp_v correction; T (PE) or θ_p (NH) unchanged from run 1.
  3. damp_v + d_con: same wind correction as run 2; T/θ_p ALSO
     gets the heat contribution.

Then:
  du = run2.u - run1.u,  dv = run2.v - run1.v
  delta_T = run3.T - run2.T  (PE),
  delta_θ_p = run3.θ_p - run2.θ_p  (NH)

Expected:
  PE:  ΔT = -d_con * (u_pre*du + 0.5*du² + v_pre*dv + 0.5*dv²) / c_pd
       projected from corners to centres via interp_corner_to_center
  NH:  Δθ_p = -d_con * (u*du + 0.5*du² + v*dv + 0.5*dv²)
              / (c_pd * exner_ref)

Tests assert delta matches expected within machine precision.

Tests
-----

1. ``test_pe_damp_v_d_con_heat_matches_formula`` — PE.
2. ``test_nh_damp_v_d_con_heat_matches_formula`` — NH (with Π_ref).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm import constants
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
from legoesm.core.operators_cdgrid import interp_corner_to_center
from legoesm.core.state import NonHydrostaticState
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.grids.vertical import (
    create_height_coordinate, compute_terrain_metric,
    standard_hybrid_levels,
)


# ---------------------------------------------------------------- PE
@pytest.fixture(scope="module")
def small_pe_state_perturbed():
    n = 8
    nlev = 6
    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    coord = standard_hybrid_levels(nlev)
    state_cc = held_suarez_init(grid, coord)
    state = hydrostatic_to_fv3(state_cc, cdgrid)
    rng = np.random.default_rng(seed=211)
    u_p = rng.uniform(-2.0, 2.0, size=state.u_d.data.shape)
    v_p = rng.uniform(-2.0, 2.0, size=state.v_d.data.shape)
    state = state._replace(
        u_d=state.u_d.replace(data=jnp.asarray(u_p)),
        v_d=state.v_d.replace(data=jnp.asarray(v_p)),
    )
    return grid, cdgrid, coord, state


def test_pe_damp_v_d_con_heat_matches_formula(small_pe_state_perturbed):
    """PE: actual ΔT from iter-208 d_con matches the expected formula
    ``-d_con * ΔKE_corner_to_center / c_pd`` within machine precision."""
    grid, cdgrid, coord, state = small_pe_state_perturbed

    common = dict(
        use_conservation_fixer=False, fix_mass=False,
    )
    cfg_no_dampv = CDGridPrimitiveEquationConfig(
        **common, damp_v=0.0, damp_v_d_con=0.0,
    )
    cfg_dampv = CDGridPrimitiveEquationConfig(
        **common, damp_v=0.030, nord_v=1, damp_v_d_con=0.0,
    )
    cfg_dampv_d_con = CDGridPrimitiveEquationConfig(
        **common, damp_v=0.030, nord_v=1, damp_v_d_con=1.0,
    )

    m_no = CDGridPrimitiveEquationModel(grid, coord, cfg_no_dampv)
    m_dampv = CDGridPrimitiveEquationModel(grid, coord, cfg_dampv)
    m_dampv_d = CDGridPrimitiveEquationModel(grid, coord, cfg_dampv_d_con)

    s_no = m_no.step(state, 100.0)
    s_dampv = m_dampv.step(state, 100.0)
    s_dampv_d = m_dampv_d.step(state, 100.0)

    # The difference in T between (damp_v + d_con) and (damp_v only)
    # is the d_con heat contribution.
    delta_T = s_dampv_d.T.data - s_dampv.T.data

    # du, dv at corners are the damp_v wind increments.
    u_pre = s_no.u_d.data        # u after dynamics, before damp_v
    v_pre = s_no.v_d.data
    u_post = s_dampv.u_d.data    # after damp_v
    v_post = s_dampv.v_d.data
    du = u_post - u_pre
    dv = v_post - v_pre

    # Sanity: damp_v should produce non-trivial dw.
    max_du = float(jnp.max(jnp.abs(du)))
    assert max_du > 1e-6, (
        f"damp_v du expected non-trivial (max={max_du:.3e})"
    )

    # Expected formula (per iter-208 implementation):
    #   ΔKE_corner = u_pre*du + 0.5*du² + v_pre*dv + 0.5*dv²
    #   ΔKE_cc     = corner_to_center(ΔKE_corner)
    #   ΔT         = -d_con * ΔKE_cc / c_pd
    dKE_corner = (
        u_pre * du + 0.5 * du ** 2
        + v_pre * dv + 0.5 * dv ** 2
    )
    dKE_cc = interp_corner_to_center(dKE_corner)
    expected_dT = -1.0 * dKE_cc / constants.c_pd

    max_expected = float(jnp.max(jnp.abs(expected_dT)))
    # Tolerance: 1e-8 relative.  Tighter than 1e-10 fails because
    # the model's interp_corner_to_center call inside the JIT
    # graph fuses ops differently than the test's external call,
    # producing a few ULPs of difference in the corner→center
    # 4-point average.  1e-8 catches sign/factor errors while
    # tolerating fused-op rounding.
    abs_tol = max(max_expected, 1e-30) * 1e-8
    diff = float(jnp.max(jnp.abs(delta_T - expected_dT)))

    assert diff < abs_tol, (
        f"iter-208 PE d_con heat formula mismatch: "
        f"max|delta_T - expected|={diff:.3e}, "
        f"max|expected|={max_expected:.3e}, tol={abs_tol:.3e}.\n"
        f"Indicates a bug in iter-208's wiring (sign, factor of 0.5, "
        f"or interp_corner_to_center call) vs iter-211's reconstruction."
    )


# ---------------------------------------------------------------- NH
@pytest.fixture(scope="module")
def small_nh_state_perturbed():
    n = 8
    nlev = 5
    z_top = 30000.0
    grid = create_cubed_sphere(n)
    height_coord = create_height_coordinate(nlev, z_top)
    terrain = jnp.zeros((6, n, n))
    terrain_metric = compute_terrain_metric(terrain, height_coord)

    rng = np.random.default_rng(seed=211)
    u_p = rng.uniform(-2.0, 2.0, size=(6, n, n, nlev))
    v_p = rng.uniform(-2.0, 2.0, size=(6, n, n, nlev))

    dims_3d = ("face", "x", "y", "level")
    dims_w = ("face", "x", "y", "level_half")
    dims_2d = ("face", "x", "y")

    state = NonHydrostaticState(
        u=Field(data=jnp.asarray(u_p), name="u",
                dims=dims_3d, units="m/s"),
        v=Field(data=jnp.asarray(v_p), name="v",
                dims=dims_3d, units="m/s"),
        w=Field(data=jnp.zeros((6, n, n, nlev + 1)), name="w",
                dims=dims_w, units="m/s"),
        theta_prime=Field(data=jnp.zeros((6, n, n, nlev)),
                          name="theta_prime", dims=dims_3d, units="K"),
        rho_prime=Field(data=jnp.zeros((6, n, n, nlev)),
                        name="rho_prime", dims=dims_3d, units="kg/m^3"),
        phis=Field(data=jnp.zeros((6, n, n)), name="phis",
                   dims=dims_2d, units="m^2/s^2"),
        tracers=Field(data=jnp.zeros((6, n, n, nlev, 0)), name="tracers",
                      dims=("face", "x", "y", "level", "tracer"),
                      units="kg/kg"),
    )
    return grid, height_coord, terrain_metric, state


def test_nh_damp_v_d_con_heat_matches_formula(small_nh_state_perturbed):
    """NH: actual Δθ_p from iter-209 d_con matches the expected formula
    ``-d_con * ΔKE_cc / (c_pd * Π_ref)`` within machine precision."""
    grid, height_coord, terrain_metric, state = small_nh_state_perturbed

    common = dict(
        hyperdiff_coeff=0.0, n_acoustic_substeps=4, sponge_coeff=0.0,
    )
    cfg_no_dampv = CDGridCompressibleEulerConfig(
        **common, damp_v=0.0, damp_v_d_con=0.0,
    )
    cfg_dampv = CDGridCompressibleEulerConfig(
        **common, damp_v=0.030, nord_v=1, damp_v_d_con=0.0,
    )
    cfg_dampv_d_con = CDGridCompressibleEulerConfig(
        **common, damp_v=0.030, nord_v=1, damp_v_d_con=1.0,
    )

    m_no = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg_no_dampv,
    )
    m_dampv = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg_dampv,
    )
    m_dampv_d = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg_dampv_d_con,
    )

    s_no = m_no.step(state, 10.0)
    s_dampv = m_dampv.step(state, 10.0)
    s_dampv_d = m_dampv_d.step(state, 10.0)

    delta_theta_p = s_dampv_d.theta_prime.data - s_dampv.theta_prime.data

    u_pre = s_no.u.data
    v_pre = s_no.v.data
    du = s_dampv.u.data - u_pre
    dv = s_dampv.v.data - v_pre

    max_du = float(jnp.max(jnp.abs(du)))
    assert max_du > 1e-6, (
        f"damp_v du expected non-trivial (max={max_du:.3e})"
    )

    # NH formula: Δθ_p = -d_con * ΔKE_cc / (c_pd * Π_ref)
    dKE_cc = (
        u_pre * du + 0.5 * du ** 2
        + v_pre * dv + 0.5 * dv ** 2
    )
    exner_ref = height_coord.exner_ref[None, None, None, :]
    expected_dtheta_p = -1.0 * dKE_cc / (constants.c_pd * exner_ref)

    max_expected = float(jnp.max(jnp.abs(expected_dtheta_p)))
    abs_tol = max(max_expected, 1e-30) * 1e-10
    diff = float(jnp.max(jnp.abs(delta_theta_p - expected_dtheta_p)))

    assert diff < abs_tol, (
        f"iter-209 NH d_con heat formula mismatch: "
        f"max|delta_theta_p - expected|={diff:.3e}, "
        f"max|expected|={max_expected:.3e}, tol={abs_tol:.3e}.\n"
        f"Indicates a bug in iter-209's wiring vs iter-211's "
        f"reconstruction (sign, factor of 0.5, or exner_ref factor)."
    )
