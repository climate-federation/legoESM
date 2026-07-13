"""FV3_3D iter 374: AD-at-rest umbrella for NH full FV3-fidelity
stack INCLUDING iter-370 cross_face flag.

iter-346 covered NH full stack without cross_face (it was iter-
370).  iter-374 extends to include all 5 NH FV3-fidelity flags
+ duogrid + full toolkit.

Tests
-----

1. ``test_full_nh_with_cross_face_grad_at_rest`` — jax.grad
   finite through 3 NH steps with all 5 flags ON + duogrid +
   full toolkit at rest.
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
from legoesm.core.field import Field
from legoesm.core.state import NonHydrostaticState
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.vertical import (
    create_height_coordinate, compute_terrain_metric,
)


def test_full_nh_with_cross_face_grad_at_rest():
    n = 8
    nlev = 5
    z_top = 30000.0
    grid = create_cubed_sphere(n, use_duogrid=True)
    height_coord = create_height_coordinate(nlev, z_top)
    terrain = jnp.zeros((6, n, n))
    terrain_metric = compute_terrain_metric(terrain, height_coord)

    dims_3d = ("face", "x", "y", "level")
    dims_w = ("face", "x", "y", "level_half")
    dims_2d = ("face", "x", "y")

    rest = NonHydrostaticState(
        u=Field(data=jnp.zeros((6, n, n, nlev)), name="u",
                dims=dims_3d, units="m/s"),
        v=Field(data=jnp.zeros((6, n, n, nlev)), name="v",
                dims=dims_3d, units="m/s"),
        w=Field(data=jnp.zeros((6, n, n, nlev + 1)),
                name="w", dims=dims_w, units="m/s"),
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

    cfg = CDGridCompressibleEulerConfig(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        corner_div_damp_d2_bg=0.0005, corner_div_damp_dddmp=0.20,
        corner_div_damp_d4_bg=1e-3, corner_div_damp_nord=1,
        damp_v=0.030, nord_v=1,
        use_fv3_a2b_zeta_corner=True,
        div_damp_coeff=1e6, div_damp_dddmp=0.20,
        A_h=1e6, smagorinsky_cs=0.20,
        damp_w=0.030, nord_w=1,
        damp_w_d_con=1.0, damp_v_d_con=1.0,
        corner_div_damp_d_con=1.0, div_damp_d_con=1.0,
        ah_d_con=1.0, delt_max=1.0,
        # All 5 NH FV3-fidelity flags
        use_fv3_d_con_cv=True,
        use_fv3_vector_halo_uv=True,
        use_fv3_dynamic_exner=True,
        use_fv3_metric_aware_d_con=True,
        use_fv3_cross_face_du_proj=True,
    )
    m = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg,
    )

    def loss(amp):
        s = rest._replace(
            u=rest.u.replace(data=amp * jnp.ones_like(rest.u.data)),
        )
        for _ in range(3):
            s = m.step(s, 5.0)
        return jnp.mean(s.theta_prime.data ** 2) + jnp.mean(s.u.data ** 2)

    g = jax.grad(loss)(0.0)
    assert jnp.isfinite(g), (
        "FULL NH FV3-fidelity stack with cross_face flag AD grad "
        "NaN — interaction surface hazard."
    )
