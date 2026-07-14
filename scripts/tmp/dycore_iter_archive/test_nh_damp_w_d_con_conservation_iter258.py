"""FV3_3D iter 258: global-energy-conservation regression for
the iter-203 damp_w_d_con post-step block (NH).

iter-205 verified the per-cell heat formula matches FV3 d_sw2
exactly.  iter-258 verifies the GLOBAL conservation property:

    Σ c_pd * Π_ref * Δθ_p_full = Σ heat_full
                                = Σ (-d_con * dw * (w + 0.5*dw))_half
                                  averaged half→full

For boundary BC ``w=0`` at top/bottom:

    Σ heat_full = Σ heat_half - 0.5*heat_half[0] - 0.5*heat_half[nlev]
                ≈ Σ heat_half  (boundary terms vanish at w=0 BC)

Tests
-----

1. ``test_nh_damp_w_d_con_post_step_conserves_global_energy`` —
   w-perturbation IC respecting w=0 BC (interior-only), single
   NH step, verify Σ c_pd * Π_ref * Δθ_p ≈ -Σ ΔKE_w.
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


def test_nh_damp_w_d_con_post_step_conserves_global_energy():
    """NH iter-203 damp_w_d_con conserves global energy:
    Σ c_pd * Π_ref * Δθ_p_full ≈ -Σ ΔKE_w_half."""
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

    rng = np.random.default_rng(seed=258)
    # w perturbation that respects w=0 BC at top (k=0) and
    # bottom (k=nlev), interior values random.
    w_pert = np.zeros((6, n, n, nlev + 1))
    w_pert[..., 1:nlev] = rng.uniform(-2.0, 2.0,
                                      size=(6, n, n, nlev - 1))

    state = NonHydrostaticState(
        u=Field(data=jnp.zeros((6, n, n, nlev)),
                name="u", dims=dims_3d, units="m/s"),
        v=Field(data=jnp.zeros((6, n, n, nlev)),
                name="v", dims=dims_3d, units="m/s"),
        w=Field(data=jnp.asarray(w_pert), name="w",
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

    common = dict(
        # Only damp_w active to isolate iter-203 post-step.
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        damp_w=0.030, nord_w=2,
    )
    cfg_no_damp_w = CDGridCompressibleEulerConfig(
        **{k: v for k, v in common.items() if k != "damp_w"},
        damp_w=0.0,
        damp_w_d_con=0.0,
    )
    cfg_no_dcon = CDGridCompressibleEulerConfig(
        **common, damp_w_d_con=0.0,
    )
    cfg_dcon_on = CDGridCompressibleEulerConfig(
        **common, damp_w_d_con=1.0,
    )

    m_no_damp = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg_no_damp_w,
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

    # damp_w's contribution to w (half levels).
    dw_half = s_no.w.data - s_no_damp.w.data
    w_pre_damp = s_no_damp.w.data

    # ΔKE per half-level — iter-203 EXACT formula.
    heat_half = dw_half * (w_pre_damp + 0.5 * dw_half)

    # damp_w's contribution to θ_p.
    dtheta_p_d_con = s_on.theta_prime.data - s_no.theta_prime.data

    # Convert to T-equivalent for global sum.
    exner_ref = height_coord.exner_ref    # (nlev,)
    dT_eq = dtheta_p_d_con * exner_ref[None, None, None, :]

    # Per-cell pointwise check at full levels.
    heat_full_expected = 0.5 * (heat_half[..., :-1]
                                + heat_half[..., 1:])
    expected_dtheta_p = -1.0 * heat_full_expected / (
        constants.c_pd * exner_ref[None, None, None, :]
    )
    np.testing.assert_allclose(
        dtheta_p_d_con, expected_dtheta_p,
        rtol=1e-10, atol=1e-12,
        err_msg=(
            "iter-203 damp_w_d_con per-cell formula mismatch."
        ),
    )

    # Global conservation: with w=0 BC at boundaries,
    # Σ heat_full = Σ heat_half (boundary terms vanish).
    total_heat_T = float(jnp.sum(constants.c_pd * dT_eq))
    total_KE = float(jnp.sum(heat_half))
    np.testing.assert_allclose(
        total_heat_T, -total_KE,
        rtol=1e-10, atol=1e-10,
        err_msg=(
            "iter-203 damp_w_d_con violates global energy "
            "conservation in T-equivalent space (with w=0 BC "
            "the boundary terms must vanish)."
        ),
    )

    assert abs(total_heat_T) > 1e-3
