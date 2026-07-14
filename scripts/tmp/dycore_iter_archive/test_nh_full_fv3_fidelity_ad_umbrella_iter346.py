"""FV3_3D iter 346: AD-at-rest umbrella for the FULL NH
FV3-fidelity stack (all 4 opt-in flags + full toolkit + d_con).

Combines:
* iter-320 ``use_fv3_d_con_cv``
* iter-325 ``use_duogrid=True`` (grid)
* iter-328 ``use_fv3_vector_halo_uv``
* iter-336/337 ``use_fv3_dynamic_exner``
* iter-339 (iter-344 fixed) ``use_fv3_metric_aware_d_con``
* Full FV3 damping toolkit + all 5 d_con sites

Uses 3 NH steps (vs iter-184's 5) since combined stack is
expensive.  Catches any future AD hazard at the
iter-320/325/328/336/339 interaction surface that iter-332
6-combo + iter-330 all-on-without-metric didn't engage.

Tests
-----

1. ``test_full_fv3_fidelity_grad_at_rest`` — jax.grad finite
   through 3 NH steps with all 5 FV3-fidelity flags ON +
   full toolkit at rest state.
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


def test_full_fv3_fidelity_grad_at_rest():
    """jax.grad through 3 NH steps with ALL FV3 fidelity flags
    ON + full toolkit + d_con stack at rest state."""
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
        # Full FV3 toolkit
        corner_div_damp_d2_bg=0.0005, corner_div_damp_dddmp=0.20,
        corner_div_damp_d4_bg=1e-3, corner_div_damp_nord=1,
        damp_v=0.030, nord_v=1,
        use_fv3_a2b_zeta_corner=True,
        div_damp_coeff=1e6, div_damp_dddmp=0.20,
        A_h=1e6, smagorinsky_cs=0.20,
        damp_w=0.030, nord_w=1,
        # All d_con knobs
        damp_w_d_con=1.0, damp_v_d_con=1.0,
        corner_div_damp_d_con=1.0, div_damp_d_con=1.0,
        ah_d_con=1.0, delt_max=1.0,
        # All 4 FV3-fidelity flags
        use_fv3_d_con_cv=True,
        use_fv3_vector_halo_uv=True,
        use_fv3_dynamic_exner=True,
        use_fv3_metric_aware_d_con=True,
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
        return jnp.mean(s.theta_prime.data ** 2 + s.u.data ** 2)

    g = jax.grad(loss)(0.0)
    assert jnp.isfinite(g), (
        "FULL FV3-fidelity stack (cv + duogrid + vector_halo + "
        "dynamic_exner + metric_aware) AD-at-rest grad NaN — "
        "indicates AD hazard at the new combined-flag surface."
    )
