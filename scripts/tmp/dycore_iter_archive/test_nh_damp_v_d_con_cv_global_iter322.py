"""FV3_3D iter 322: cv-flag global energy conservation for the
iter-209 damp_v_d_con post-step block (NH).

iter-257 verified the c_pd-balance for damp_v_d_con; iter-321
verified the c_vd-balance for damp_w_d_con under iter-320's
``use_fv3_d_con_cv = True`` flag.  iter-322 closes the symmetric
gap: c_vd-balance for damp_v_d_con under the cv flag.

Mirrors the iter-257 fixture (strong-wind IC, single NH step,
isolated damp_v block) but uses ``use_fv3_d_con_cv = True`` and
asserts:

    Σ c_vd * Π_ref * Δθ_p_d_con + Σ ΔKE_cc == 0   (cv_air balance)

Tests
-----

1. ``test_nh_damp_v_d_con_cv_per_cell_formula`` — element-wise
   ``Δθ_p = -d_con * ΔKE / (c_vd * Π_ref)`` (rtol=1e-10).
2. ``test_nh_damp_v_d_con_cv_global_energy`` — global
   ``Σ c_vd * Π_ref * Δθ_p ≈ -Σ ΔKE_cc`` (rtol=1e-10).
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
def damp_v_fixture():
    """Strong-wind IC, mirrors iter-257 fixture."""
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

    rng = np.random.default_rng(seed=322)
    u_p = rng.uniform(-30.0, 30.0, size=(6, n, n, nlev))
    v_p = rng.uniform(-30.0, 30.0, size=(6, n, n, nlev))
    state = NonHydrostaticState(
        u=Field(data=jnp.asarray(u_p), name="u",
                dims=dims_3d, units="m/s"),
        v=Field(data=jnp.asarray(v_p), name="v",
                dims=dims_3d, units="m/s"),
        w=Field(data=jnp.zeros((6, n, n, nlev + 1)),
                name="w", dims=dims_w, units="m/s"),
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
    return grid, height_coord, terrain_metric, state


def _three_runs(grid, height_coord, terrain_metric, state):
    """Triangulate damp_v contribution: no-damp / damp / damp+d_con."""
    common = dict(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        damp_v=0.030, nord_v=2,
    )
    cfg_no_damp_v = CDGridCompressibleEulerConfig(
        **{k: v for k, v in common.items() if k != "damp_v"},
        damp_v=0.0, damp_v_d_con=0.0, use_fv3_d_con_cv=True,
    )
    cfg_no_dcon = CDGridCompressibleEulerConfig(
        **common, damp_v_d_con=0.0, use_fv3_d_con_cv=True,
    )
    cfg_dcon_cv = CDGridCompressibleEulerConfig(
        **common, damp_v_d_con=1.0, use_fv3_d_con_cv=True,
    )
    m_no_damp = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg_no_damp_v,
    )
    m_no = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg_no_dcon,
    )
    m_cv = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg_dcon_cv,
    )
    return (
        m_no_damp.step(state, 10.0),
        m_no.step(state, 10.0),
        m_cv.step(state, 10.0),
    )


def test_nh_damp_v_d_con_cv_per_cell_formula(damp_v_fixture):
    """Per-cell formula: Δθ_p = -d_con * ΔKE / (c_vd * Π_ref)."""
    grid, height_coord, terrain_metric, state = damp_v_fixture
    s_no_damp, s_no, s_cv = _three_runs(
        grid, height_coord, terrain_metric, state,
    )
    du = s_no.u.data - s_no_damp.u.data
    dv = s_no.v.data - s_no_damp.v.data
    u_pre = s_no_damp.u.data
    v_pre = s_no_damp.v.data
    dKE_cc = (
        u_pre * du + 0.5 * du ** 2
        + v_pre * dv + 0.5 * dv ** 2
    )
    dtheta_p_dcon = s_cv.theta_prime.data - s_no.theta_prime.data
    exner_ref = height_coord.exner_ref
    expected = -dKE_cc / (
        constants.c_vd * exner_ref[None, None, None, :]
    )
    np.testing.assert_allclose(
        dtheta_p_dcon, expected,
        rtol=1e-10, atol=1e-12,
        err_msg=(
            "iter-322 cv flag per-cell formula mismatch on "
            "iter-209 damp_v_d_con post-step site."
        ),
    )


def test_nh_damp_v_d_con_cv_global_energy(damp_v_fixture):
    """Global c_vd-balance: Σ c_vd · Π_ref · Δθ_p ≈ -Σ ΔKE_cc."""
    grid, height_coord, terrain_metric, state = damp_v_fixture
    s_no_damp, s_no, s_cv = _three_runs(
        grid, height_coord, terrain_metric, state,
    )
    du = s_no.u.data - s_no_damp.u.data
    dv = s_no.v.data - s_no_damp.v.data
    u_pre = s_no_damp.u.data
    v_pre = s_no_damp.v.data
    dKE_cc = (
        u_pre * du + 0.5 * du ** 2
        + v_pre * dv + 0.5 * dv ** 2
    )
    dtheta_p_dcon = s_cv.theta_prime.data - s_no.theta_prime.data
    exner_ref = height_coord.exner_ref
    dT_eq = dtheta_p_dcon * exner_ref[None, None, None, :]

    total_heat_cv = float(jnp.sum(constants.c_vd * dT_eq))
    total_KE = float(jnp.sum(dKE_cc))
    np.testing.assert_allclose(
        total_heat_cv, -total_KE,
        rtol=1e-10, atol=1e-10,
        err_msg=(
            "iter-322 cv flag global c_vd-balance violation on "
            "iter-209 damp_v_d_con post-step site."
        ),
    )
    assert abs(total_heat_cv) > 1e-3
