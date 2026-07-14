"""FV3_3D iter 251: NH C36 production stability test with the
d_con stack (NH mirror of PE iter-245).

iter-245 verified PE iter-19 production toolkit + d_con stack
at C36 production resolution.  iter-251 mirrors for NH at the
same resolution.

Tests
-----

1. ``test_nh_production_with_d_con_stable_at_C36`` — C36 NH
   state + iter-184-style toolkit + all 5 NH d_con knobs at
   1.0 + delt_max=1.0.  10 steps × dt=10 = 100 s integrated.
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
    compute_terrain_metric, create_height_coordinate,
)


def test_nh_production_with_d_con_stable_at_C36():
    """C36 NH zero state + production toolkit + d_con stack
    is stable for 10 steps."""
    n = 36
    nlev = 5
    z_top = 30000.0
    grid = create_cubed_sphere(n)
    height_coord = create_height_coordinate(nlev, z_top)
    terrain = jnp.zeros((6, n, n))
    terrain_metric = compute_terrain_metric(terrain, height_coord)

    dims_3d = ("face", "x", "y", "level")
    dims_w = ("face", "x", "y", "level_half")
    dims_2d = ("face", "x", "y")

    rng = np.random.default_rng(seed=251)
    # Modest IC at production resolution.
    u_p = rng.uniform(-2.0, 2.0, size=(6, n, n, nlev))
    v_p = rng.uniform(-2.0, 2.0, size=(6, n, n, nlev))
    w_p = rng.uniform(-0.2, 0.2, size=(6, n, n, nlev + 1))

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
        # NH production toolkit (matches iter-184 umbrella)
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        corner_div_damp_d2_bg=0.0005, corner_div_damp_dddmp=0.20,
        corner_div_damp_d4_bg=1e-3, corner_div_damp_nord=1,
        damp_v=0.030, nord_v=2,
        use_fv3_a2b_zeta_corner=True,
        div_damp_coeff=1e6, div_damp_dddmp=0.20,
        A_h=1e6, smagorinsky_cs=0.20,
        damp_w=0.030, nord_w=1,
        # All 5 NH d_con knobs at FV3 production 1.0
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
    dt = 10.0
    for _ in range(10):
        s = model.step(s, dt)

    # Verify all fields finite + bounded.
    assert jnp.all(jnp.isfinite(s.u.data)), (
        "NH C36 production: u NaN/Inf at step 10."
    )
    assert jnp.all(jnp.isfinite(s.v.data))
    assert jnp.all(jnp.isfinite(s.w.data))
    assert jnp.all(jnp.isfinite(s.theta_prime.data))

    max_u = float(jnp.max(jnp.abs(s.u.data)))
    assert max_u < 50.0, (
        f"NH C36 production max|u|={max_u:.2f} > 50 m/s — "
        f"possible instability at production resolution."
    )

    # theta_prime should stay in physically reasonable range.
    max_dtheta = float(jnp.max(jnp.abs(s.theta_prime.data)))
    assert max_dtheta < 100.0, (
        f"NH C36 max|θ_p|={max_dtheta:.2f} > 100 K — possible "
        f"thermodynamic instability."
    )
