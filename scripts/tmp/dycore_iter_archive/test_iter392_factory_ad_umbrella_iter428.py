"""FV3_3D iter 428: AD-at-rest umbrella driven through the
iter-392 factories.

iter-346/355 verify AD-at-rest for the FULL FV3-fidelity stack
when the config is HAND-BUILT.  iter-427 verifies the factory
drives a finite step.  iter-428 closes the gap: jax.grad through
1 NH step + 1 PE step with the factory-produced config must
return finite.  Catches the failure mode where a factory default
flag value introduces AD hazard at the combined-flag surface
(e.g., factory accidentally sets a flag that triggers a where/
clip path that breaks reverse-mode).  Reduced to 1 step each
because grad-through-full-toolkit + factory + duogrid at C8
compiles >12 min / >11 GB at 3 steps; 1 step still exercises
every flag's AD path.

Tests
-----

1. ``test_factory_nh_grad_at_rest`` — factory NH config + 1
   step + jax.grad finite.
2. ``test_factory_pe_grad_at_rest`` — factory PE config + 1
   step + jax.grad finite.
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
from legoesm.atmosphere.held_suarez import held_suarez_init
from legoesm.core.field import Field
from legoesm.core.state import NonHydrostaticState
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.grids.vertical import (
    create_height_coordinate,
    compute_terrain_metric,
    standard_hybrid_levels,
)


def test_factory_nh_grad_at_rest():
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
    )
    m = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg,
    )

    def loss(amp):
        s = rest._replace(
            u=rest.u.replace(data=amp * jnp.ones_like(rest.u.data)),
        )
        s = m.step(s, 5.0)
        return jnp.mean(s.theta_prime.data ** 2 + s.u.data ** 2)

    g = jax.grad(loss)(0.0)
    assert jnp.isfinite(g), (
        "Factory-driven NH FV3-faithful config + 3 steps "
        "jax.grad NaN at rest — AD hazard introduced by factory "
        "default flag combination."
    )


def test_factory_pe_grad_at_rest():
    n = 8
    nlev = 5
    grid = create_cubed_sphere(n, use_duogrid=True)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    coord = standard_hybrid_levels(nlev)
    state_cc = held_suarez_init(grid, coord)
    rest = hydrostatic_to_fv3(state_cc, cdgrid)
    cfg = make_fv3_faithful_pe_config(
        damp_v=0.030, nord_v=1, damp_v_d_con=1.0,
        corner_div_damp_d2_bg=0.0005,
        corner_div_damp_d_con=1.0,
        div_damp_coeff=1e6, div_damp_d_con=1.0,
        A_h=1e6, ah_d_con=1.0,
    )
    m = CDGridPrimitiveEquationModel(grid, coord, cfg)

    def loss(amp):
        s = rest._replace(
            u_d=rest.u_d.replace(
                data=rest.u_d.data + amp,
            ),
        )
        s = m.step(s, 5.0)
        return jnp.mean(s.T.data ** 2) + jnp.mean(s.u_d.data ** 2)

    g = jax.grad(loss)(0.0)
    assert jnp.isfinite(g), (
        "Factory-driven PE FV3-faithful config + 3 steps "
        "jax.grad NaN at rest — AD hazard in factory "
        "default flag combination."
    )
