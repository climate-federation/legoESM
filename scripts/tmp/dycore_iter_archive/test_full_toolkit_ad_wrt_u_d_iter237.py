"""FV3_3D iter 237: AD-at-rest gradient w.r.t. u perturbation
amplitude with the FULL toolkit + d_con stack ON.

iter-184/185 verified ``jax.grad`` w.r.t. theta_prime (NH) /
T (PE) is finite at rest with the full toolkit.  iter 237
adds the orthogonal direction: differentiate w.r.t. a scalar
perturbation amplitude on (u_d, v_d) for PE / (u, v) for NH.

This direction is more sensitive to sqrt-at-zero hazards in
the wind-dependent helpers (Smagorinsky strain magnitude,
T_diss wind speed, smag_vort cap) because the perturbation
directly enters the wind state at all cells.

Tests
-----

1. ``test_pe_full_toolkit_grad_wrt_u_d_at_rest`` — PE umbrella
   + full d_con stack: ``jax.grad`` w.r.t. uniform u_d amplitude
   epsilon at rest is finite.
2. ``test_nh_full_toolkit_grad_wrt_u_at_rest`` — NH umbrella +
   full d_con stack: ``jax.grad`` w.r.t. uniform u amplitude
   epsilon at rest is finite.
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


def test_pe_full_toolkit_grad_wrt_u_d_at_rest():
    """PE: AD w.r.t. uniform u_d perturbation amplitude is finite
    at rest with the full toolkit + full d_con stack."""
    n = 8
    nlev = 6
    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    coord = standard_hybrid_levels(nlev)
    state_cc = held_suarez_init(grid, coord)
    state_rest = hydrostatic_to_fv3(state_cc, cdgrid)
    state_rest = state_rest._replace(
        u_d=state_rest.u_d.replace(
            data=jnp.zeros_like(state_rest.u_d.data),
        ),
        v_d=state_rest.v_d.replace(
            data=jnp.zeros_like(state_rest.v_d.data),
        ),
    )

    cfg = CDGridPrimitiveEquationConfig(
        # Full PE FV3 toolkit (matches iter-185 umbrella).
        A_h=1e6, smagorinsky_cs=0.20,
        hyperdiff_coeff=1e14,
        div_damp_coeff=1e6, div_damp_dddmp=0.20,
        use_fv3_a2b_zeta_corner=True,
        damp_v=0.030, nord_v=2,
        corner_div_damp_d2_bg=0.0005, corner_div_damp_dddmp=0.20,
        corner_div_damp_d4_bg=1e-3, corner_div_damp_nord=1,
        T_diss_coeff=0.05,
        # Full d_con stack at FV3 production 1.0.
        damp_v_d_con=1.0,
        delt_max=1.0,
        corner_div_damp_d_con=1.0,
        div_damp_d_con=1.0,
        ah_d_con=1.0,
        use_conservation_fixer=False, fix_mass=False,
        zero_mean_ps_tendency=False,
    )
    model = CDGridPrimitiveEquationModel(grid, coord, cfg)

    def loss_fn(eps):
        u_pert = state_rest.u_d.data + eps * jnp.ones_like(
            state_rest.u_d.data,
        )
        s = state_rest._replace(
            u_d=state_rest.u_d.replace(data=u_pert),
        )
        for _ in range(3):
            s = model.step(s, 200.0)
        return jnp.sum(s.T.data ** 2)

    g = jax.grad(loss_fn)(0.0)
    assert jnp.isfinite(g), (
        f"PE AD w.r.t. uniform u_d perturbation must be finite "
        f"at rest with the full toolkit + d_con stack ON.  "
        f"Got grad={g}.  Likely a sqrt(0) hazard in a wind-"
        f"dependent helper that the iter-184/185 T-direction "
        f"tests don't expose."
    )


def test_nh_full_toolkit_grad_wrt_u_at_rest():
    """NH: AD w.r.t. uniform u perturbation amplitude is finite
    at rest with the full NH toolkit + full d_con stack."""
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

    state_rest = NonHydrostaticState(
        u=Field(data=jnp.zeros((6, n, n, nlev)), name="u",
                dims=dims_3d, units="m/s"),
        v=Field(data=jnp.zeros((6, n, n, nlev)), name="v",
                dims=dims_3d, units="m/s"),
        w=Field(data=jnp.zeros((6, n, n, nlev + 1)), name="w",
                dims=dims_w, units="m/s"),
        theta_prime=Field(data=jnp.zeros((6, n, n, nlev)),
                          name="theta_prime", dims=dims_3d,
                          units="K"),
        rho_prime=Field(data=jnp.zeros((6, n, n, nlev)),
                        name="rho_prime", dims=dims_3d,
                        units="kg/m^3"),
        phis=Field(data=jnp.zeros((6, n, n)), name="phis",
                   dims=dims_2d, units="m^2/s^2"),
        tracers=Field(data=jnp.zeros((6, n, n, nlev, 0)),
                      name="tracers",
                      dims=("face", "x", "y", "level", "tracer"),
                      units="kg/kg"),
    )

    cfg = CDGridCompressibleEulerConfig(
        # Full NH FV3 toolkit (matches iter-184 umbrella).
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        corner_div_damp_d2_bg=0.0005, corner_div_damp_dddmp=0.20,
        corner_div_damp_d4_bg=1e-3, corner_div_damp_nord=1,
        damp_v=0.030, nord_v=2,
        use_fv3_a2b_zeta_corner=True,
        div_damp_coeff=1e6, div_damp_dddmp=0.20,
        A_h=1e6, smagorinsky_cs=0.20,
        damp_w=0.030, nord_w=1,
        # Full d_con stack at FV3 production 1.0.
        damp_w_d_con=1.0,
        damp_v_d_con=1.0,
        delt_max=1.0,
        corner_div_damp_d_con=1.0,
        div_damp_d_con=1.0,
        ah_d_con=1.0,
    )
    model = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg,
    )

    def loss_fn(eps):
        u_pert = state_rest.u.data + eps * jnp.ones_like(
            state_rest.u.data,
        )
        s = state_rest._replace(
            u=state_rest.u.replace(data=u_pert),
        )
        for _ in range(5):
            s = model.step(s, 10.0)
        return jnp.sum(s.theta_prime.data ** 2)

    g = jax.grad(loss_fn)(0.0)
    assert jnp.isfinite(g), (
        f"NH AD w.r.t. uniform u perturbation must be finite at "
        f"rest with the full toolkit + d_con stack ON.  Got "
        f"grad={g}.  Likely a sqrt(0) hazard in a wind-"
        f"dependent helper that the iter-184 θ_p-direction "
        f"test doesn't expose."
    )
