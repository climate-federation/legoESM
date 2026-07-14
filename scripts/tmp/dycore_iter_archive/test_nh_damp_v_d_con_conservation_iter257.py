"""FV3_3D iter 257: global-energy-conservation regression for
the iter-209 damp_v_d_con post-step block (NH).

NH counterpart of PE iter-256.  iter-209 damp_v_d_con uses
cell-centre winds (NH stores u, v at cc), with the iter-207
Π_ref refinement:

    Δθ_p_cc = -damp_v_d_con * ΔKE_cc / (c_pd * Π_ref)
    where ΔKE_cc = u_cc * du_cc + 0.5*du_cc² + v_cc * dv_cc + 0.5*dv_cc²

Conservation property (in T-space):

    c_pd * Π_ref * Δθ_p = -ΔKE

Tests
-----

1. ``test_nh_damp_v_d_con_post_step_conserves_global_energy`` —
   strong-wind IC, single NH step, verify
   ``Σ c_pd * Π_ref * Δθ_p_d_con + Σ ΔKE_cc == 0`` at
   rtol=1e-10.
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


def test_nh_damp_v_d_con_post_step_conserves_global_energy():
    """NH iter-209 damp_v_d_con post-step block conserves global
    energy: Σ c_pd * Π_ref * Δθ_p + Σ ΔKE = 0."""
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

    rng = np.random.default_rng(seed=257)
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

    common = dict(
        # Only damp_v active to isolate iter-209 post-step.
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        damp_v=0.030, nord_v=2,
    )
    cfg_no_damp_v = CDGridCompressibleEulerConfig(
        **{k: v for k, v in common.items() if k != "damp_v"},
        damp_v=0.0,
        damp_v_d_con=0.0,
    )
    cfg_no_dcon = CDGridCompressibleEulerConfig(
        **common, damp_v_d_con=0.0,
    )
    cfg_dcon_on = CDGridCompressibleEulerConfig(
        **common, damp_v_d_con=1.0,
    )

    m_no_damp = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg_no_damp_v,
    )
    m_no = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg_no_dcon,
    )
    m_on = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg_dcon_on,
    )

    s_no_damp = m_no_damp.step(state, 10.0)
    s_no = m_no.step(state, 10.0)
    s_on = m_on.step(state, 10.0)

    # damp_v's wind contribution = (with damp_v) - (without).
    du_damp_v = s_no.u.data - s_no_damp.u.data
    dv_damp_v = s_no.v.data - s_no_damp.v.data

    # Pre-damp_v wind state (advection-only result).
    u_pre_damp = s_no_damp.u.data
    v_pre_damp = s_no_damp.v.data

    # ΔKE per unit mass at cell centres — iter-209 EXACT formula.
    dKE_cc = (
        u_pre_damp * du_damp_v + 0.5 * du_damp_v ** 2
        + v_pre_damp * dv_damp_v + 0.5 * dv_damp_v ** 2
    )

    # damp_v's contribution to θ_p_d_con.
    dtheta_p_d_con = s_on.theta_prime.data - s_no.theta_prime.data

    # Conversion to T-space: dT = Π_ref * dθ_p (linearization).
    exner_ref = height_coord.exner_ref    # (nlev,)
    dT_d_con_eq = (
        dtheta_p_d_con * exner_ref[None, None, None, :]
    )

    # Per-cell pointwise check (matches iter-209 internal logic).
    expected_dtheta_p = (
        -1.0 * dKE_cc / (
            constants.c_pd * exner_ref[None, None, None, :]
        )
    )
    np.testing.assert_allclose(
        dtheta_p_d_con, expected_dtheta_p,
        rtol=1e-10, atol=1e-12,
        err_msg=(
            "NH iter-209 damp_v_d_con per-cell formula must "
            "match the EXACT discrete energy budget "
            "(in θ_p-space with Π_ref factor)."
        ),
    )

    # Global conservation (T-equivalent space):
    # Σ c_pd * dT_eq + Σ ΔKE = 0.
    total_heat = float(jnp.sum(constants.c_pd * dT_d_con_eq))
    total_KE = float(jnp.sum(dKE_cc))
    np.testing.assert_allclose(
        total_heat, -total_KE,
        rtol=1e-10, atol=1e-10,
        err_msg=(
            "NH iter-209 damp_v_d_con violates global energy "
            "conservation in T-equivalent space."
        ),
    )

    assert abs(total_heat) > 1e-3
