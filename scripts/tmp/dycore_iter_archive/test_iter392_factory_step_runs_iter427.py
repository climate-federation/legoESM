"""FV3_3D iter 427: integration test — factory configs from
iter-392 must actually drive a finite dycore step.

iter-417/418 only checks factory CONSTRUCTION; iter-426 verifies
factory IMPORT.  Neither runs the model.  iter-427 closes the
loop: build factory config, build model, run ONE dt step at C8,
assert every output field finite.  Catches the failure mode where
a factory flag is wired in defaults but actually crashes the
step path (e.g., a flag that introduces NaN, division by zero,
shape mismatch, etc.).

Tests
-----

1. ``test_factory_nh_step_finite`` — NH factory config + 1 dt
   step at C8 produces finite u, v, w, theta_prime, rho_prime.
2. ``test_factory_pe_step_finite`` — PE factory config + 1 dt
   step at C8 produces finite u_d, v_d, T, p_s.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.dynamics.gcm.compressible_euler_cdgrid import (
    CDGridCompressibleEulerModel,
    make_fv3_faithful_nh_config,
)
from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
    CDGridPrimitiveEquationModel,
    hydrostatic_to_fv3,
    make_fv3_faithful_pe_config,
)
from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init
from legoesm.core.field import Field
from legoesm.core.state import NonHydrostaticState
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.grids.vertical import (
    create_height_coordinate,
    compute_terrain_metric,
    standard_hybrid_levels,
)


def test_factory_nh_step_finite():
    n = 8
    nlev = 5
    z_top = 30000.0
    grid = create_cubed_sphere(n, use_duogrid=True)
    height_coord = create_height_coordinate(nlev, z_top)
    terrain = jnp.zeros((6, n, n))
    terrain_metric = compute_terrain_metric(terrain, height_coord)
    rng = np.random.default_rng(seed=427)
    u_p = rng.uniform(-1.0, 1.0, size=(6, n, n, nlev))
    v_p = rng.uniform(-1.0, 1.0, size=(6, n, n, nlev))
    dims_3d = ("face", "x", "y", "level")
    dims_w = ("face", "x", "y", "level_half")
    dims_2d = ("face", "x", "y")
    state = NonHydrostaticState(
        u=Field(data=jnp.asarray(u_p), name="u",
                dims=dims_3d, units="m/s"),
        v=Field(data=jnp.asarray(v_p), name="v",
                dims=dims_3d, units="m/s"),
        w=Field(data=jnp.zeros((6, n, n, nlev + 1)),
                name="w", dims=dims_w, units="m/s"),
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
    cfg = make_fv3_faithful_nh_config(
        n_acoustic_substeps=4,
        damp_v=0.030, nord_v=1, damp_v_d_con=1.0,
        corner_div_damp_d2_bg=0.0005,
        corner_div_damp_d_con=1.0,
        div_damp_coeff=1e6, div_damp_d_con=1.0,
        damp_w=0.030, damp_w_d_con=1.0,
    )
    model = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg,
    )
    new_state = model.step(state, dt=10.0)
    for f in (new_state.u, new_state.v, new_state.w,
              new_state.theta_prime, new_state.rho_prime):
        assert jnp.all(jnp.isfinite(f.data)), (
            f"Factory-driven NH step produced non-finite {f.name}"
        )


def test_factory_pe_step_finite():
    n = 8
    nlev = 5
    grid = create_cubed_sphere(n, use_duogrid=True)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    coord = standard_hybrid_levels(nlev)
    state_cc = held_suarez_init(grid, coord)
    state = hydrostatic_to_fv3(state_cc, cdgrid)
    cfg = make_fv3_faithful_pe_config(
        damp_v=0.030, nord_v=1, damp_v_d_con=1.0,
        corner_div_damp_d2_bg=0.0005,
        corner_div_damp_d_con=1.0,
        div_damp_coeff=1e6, div_damp_d_con=1.0,
    )
    model = CDGridPrimitiveEquationModel(grid, coord, cfg)
    new_state = model.step(state, dt=10.0)
    for f in (new_state.u_d, new_state.v_d, new_state.T,
              new_state.p_s):
        assert jnp.all(jnp.isfinite(f.data)), (
            f"Factory-driven PE step produced non-finite {f.name}"
        )
