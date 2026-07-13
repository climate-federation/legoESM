"""FV3_3D iter 291: NH iter-184 production AD-at-rest at JAX
default float32 (NH counterpart of PE iter-290; mirror of NH
iter-270 which uses jax_enable_x64).

iter-270 verified NH production d4_bg=0.02 + full d_con stack
produces finite jax.grad at rest under float64.  iter-291
extends this guarantee to JAX default float32 — important for
ML-training-style differentiable workflows.

Tests
-----

1. ``test_nh_production_d_con_grad_at_rest_float32`` — NH
   iter-184 production setting + full d_con stack + AD-at-
   rest, at JAX default float32.  jax.grad w.r.t. theta_prime
   finite.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# DO NOT enable x64 — explicit float32 default.

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


def test_nh_production_d_con_grad_at_rest_float32():
    """NH iter-184 production d4_bg=0.02 + full d_con stack must
    produce finite jax.grad at rest under JAX default float32."""
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
        u=Field(data=jnp.zeros((6, n, n, nlev)),
                name="u", dims=dims_3d, units="m/s"),
        v=Field(data=jnp.zeros((6, n, n, nlev)),
                name="v", dims=dims_3d, units="m/s"),
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

    cfg = CDGridCompressibleEulerConfig(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        corner_div_damp_d2_bg=0.0005, corner_div_damp_dddmp=0.20,
        corner_div_damp_d4_bg=0.02,
        corner_div_damp_nord=1,
        damp_v=0.030, nord_v=2,
        use_fv3_a2b_zeta_corner=True,
        div_damp_coeff=1e6, div_damp_dddmp=0.20,
        A_h=1e6, smagorinsky_cs=0.20,
        damp_w=0.030, nord_w=1,
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

    def loss_fn(theta_p_data):
        s = state_rest._replace(
            theta_prime=state_rest.theta_prime.replace(
                data=theta_p_data,
            ),
        )
        for _ in range(5):
            s = model.step(s, 10.0)
        return jnp.mean(s.u.data ** 2 + s.v.data ** 2)

    grad = jax.grad(loss_fn)(state_rest.theta_prime.data)
    assert jnp.all(jnp.isfinite(grad)), (
        "NH iter-184 production d4_bg=0.02 + full d_con stack "
        "at rest must produce finite jax.grad under JAX default "
        "float32.  PE+NH iter-183 sqrt(0) AD-safety verified at "
        "both x64 (iter-270) and float32 (iter-291)."
    )
    assert grad.dtype == jnp.float32, (
        f"Expected float32 grad but got {grad.dtype}.  "
        f"jax_enable_x64 may have been set inadvertently."
    )
