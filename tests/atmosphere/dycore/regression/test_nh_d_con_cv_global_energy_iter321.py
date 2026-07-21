"""FV3_3D iter 321: global energy conservation under iter-320's
``use_fv3_d_con_cv = True`` flag.

iter-258 verified the c_pd-balance for the iter-203 damp_w_d_con
post-step block:

    Σ c_pd * Π_ref * Δθ_p_full ≈ -Σ ΔKE_w_half  (cp_air branch)

iter-320 added the opt-in ``use_fv3_d_con_cv: bool = False`` flag
that swaps c_pd → c_vd at all 5 NH d_con sites for FV3-faithful
(``cv_air`` branch in FV3 ``dyn_core.F90:1795``).  When the flag is
True, the c_vd-balance must hold instead:

    Σ c_vd * Π_ref * Δθ_p_full ≈ -Σ ΔKE_w_half  (cv_air branch)

This is the FV3-correct global energy balance for compressible NH
dynamics (constant-volume internal-energy conservation), as opposed
to the constant-pressure enthalpy form inherited from PE.

Tests
-----

1. ``test_nh_damp_w_d_con_cv_global_energy`` — flag=True NH
   damp_w_d_con conserves global energy via c_vd-balance.  Mirrors
   iter-258 but uses c_vd in the formula.
2. ``test_nh_damp_w_d_con_cv_per_cell_formula`` — per-cell formula
   match: ``Δθ_p = -d_con * heat_full / (c_vd * Π_ref)``.
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
from legoesm.core.field import Field
from legoesm.core.state import NonHydrostaticState
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.vertical import (
    compute_terrain_metric, create_height_coordinate,
)


@pytest.fixture(scope="module")
def damp_w_fixture():
    """w-perturbation IC respecting w=0 BC at top/bottom.  Mirrors
    iter-258 fixture so the cv balance is directly comparable to
    the cp balance."""
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

    rng = np.random.default_rng(seed=321)
    w_pert = np.zeros((6, n, n, nlev + 1))
    w_pert[..., 1:nlev] = rng.uniform(-2.0, 2.0,
                                      size=(6, n, n, nlev - 1))
    state = NonHydrostaticState(
        u=Field(data=jnp.zeros((6, n, n, nlev)), name="u",
                dims=dims_3d, units="m/s"),
        v=Field(data=jnp.zeros((6, n, n, nlev)), name="v",
                dims=dims_3d, units="m/s"),
        w=Field(data=jnp.asarray(w_pert), name="w",
                dims=dims_w, units="m/s"),
        theta_prime=Field(data=jnp.zeros((6, n, n, nlev)),
                          name="theta_prime", dims=dims_3d, units="K"),
        rho_prime=Field(data=jnp.zeros((6, n, n, nlev)),
                        name="rho_prime", dims=dims_3d, units="kg/m^3"),
        phis=Field(data=jnp.zeros((6, n, n)), name="phis",
                   dims=dims_2d, units="m^2/s^2"),
        tracers=Field(data=jnp.zeros((6, n, n, nlev, 0)),
                      name="tracers",
                      dims=("face", "x", "y", "level", "tracer"),
                      units="kg/kg"),
    )
    return grid, height_coord, terrain_metric, state


def test_nh_damp_w_d_con_cv_per_cell_formula(damp_w_fixture):
    """Per-cell formula match for cv flag:
    Δθ_p = -d_con * heat_full / (c_vd * Π_ref)."""
    grid, height_coord, terrain_metric, state = damp_w_fixture
    common = dict(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        damp_w=0.030, nord_w=2,
    )
    cfg_no_damp_w = CDGridCompressibleEulerConfig(
        **{k: v for k, v in common.items() if k != "damp_w"},
        damp_w=0.0, damp_w_d_con=0.0, use_fv3_d_con_cv=True,
    )
    cfg_no_dcon = CDGridCompressibleEulerConfig(
        **common, damp_w_d_con=0.0, use_fv3_d_con_cv=True,
    )
    cfg_dcon_cv = CDGridCompressibleEulerConfig(
        **common, damp_w_d_con=1.0, use_fv3_d_con_cv=True,
    )
    m_no_damp = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg_no_damp_w,
    )
    m_no = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg_no_dcon,
    )
    m_cv = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg_dcon_cv,
    )
    s_no_damp = m_no_damp.step(state, 10.0)
    s_no = m_no.step(state, 10.0)
    s_cv = m_cv.step(state, 10.0)

    dw_half = s_no.w.data - s_no_damp.w.data
    w_pre_damp = s_no_damp.w.data
    heat_half = dw_half * (w_pre_damp + 0.5 * dw_half)
    heat_full_expected = 0.5 * (heat_half[..., :-1]
                                + heat_half[..., 1:])

    exner_ref = height_coord.exner_ref
    expected_dtheta_p = -1.0 * heat_full_expected / (
        constants.c_vd * exner_ref[None, None, None, :]
    )

    dtheta_p_d_con = s_cv.theta_prime.data - s_no.theta_prime.data
    np.testing.assert_allclose(
        dtheta_p_d_con, expected_dtheta_p,
        rtol=1e-10, atol=1e-12,
        err_msg=(
            "iter-320 cv flag per-cell formula mismatch on "
            "iter-203 damp_w_d_con post-step site."
        ),
    )


def test_nh_damp_w_d_con_cv_global_energy(damp_w_fixture):
    """Global energy conservation under cv flag:
    Σ c_vd * Π_ref * Δθ_p ≈ -Σ ΔKE_w_half."""
    grid, height_coord, terrain_metric, state = damp_w_fixture
    common = dict(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        damp_w=0.030, nord_w=2,
    )
    cfg_no_damp_w = CDGridCompressibleEulerConfig(
        **{k: v for k, v in common.items() if k != "damp_w"},
        damp_w=0.0, damp_w_d_con=0.0, use_fv3_d_con_cv=True,
    )
    cfg_no_dcon = CDGridCompressibleEulerConfig(
        **common, damp_w_d_con=0.0, use_fv3_d_con_cv=True,
    )
    cfg_dcon_cv = CDGridCompressibleEulerConfig(
        **common, damp_w_d_con=1.0, use_fv3_d_con_cv=True,
    )
    m_no_damp = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg_no_damp_w,
    )
    m_no = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg_no_dcon,
    )
    m_cv = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg_dcon_cv,
    )
    s_no_damp = m_no_damp.step(state, 10.0)
    s_no = m_no.step(state, 10.0)
    s_cv = m_cv.step(state, 10.0)

    dw_half = s_no.w.data - s_no_damp.w.data
    w_pre_damp = s_no_damp.w.data
    heat_half = dw_half * (w_pre_damp + 0.5 * dw_half)

    dtheta_p_d_con = s_cv.theta_prime.data - s_no.theta_prime.data
    exner_ref = height_coord.exner_ref
    dT_eq = dtheta_p_d_con * exner_ref[None, None, None, :]

    total_heat_T_cv = float(jnp.sum(constants.c_vd * dT_eq))
    total_KE = float(jnp.sum(heat_half))
    np.testing.assert_allclose(
        total_heat_T_cv, -total_KE,
        rtol=1e-10, atol=1e-10,
        err_msg=(
            "iter-320 cv flag violates global c_vd-balance: "
            "Σ c_vd * Π_ref * Δθ_p must equal -Σ ΔKE_w_half "
            "(boundary terms vanish under w=0 BC)."
        ),
    )
    assert abs(total_heat_T_cv) > 1e-3
