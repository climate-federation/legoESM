"""FV3_3D iter 429: multi-step stability for factory-driven
dycores.

iter-427 verifies factory + 1-step finite.  iter-428 verifies
factory + 1-step grad finite.  Neither catches multi-step
instability where the first step is finite but the trajectory
diverges by step 3.  iter-429 runs 3 NH steps + 3 PE steps with
the factory config and asserts state norms stay bounded.

Cheap: no jax.grad, no jit-around-segments, just sequential
``model.step`` calls in eager mode.  Catches the failure mode
where a factory flag combination produces O(1) state but slow
exponential growth that 1-step misses.

Tests
-----

1. ``test_factory_nh_three_step_bounded`` — NH factory + 3
   steps, max-abs(u) and max-abs(θ′) stay below growth cap.
2. ``test_factory_pe_three_step_bounded`` — PE factory + 3
   steps, max-abs(T) stays below growth cap.
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


def test_factory_nh_three_step_bounded():
    n = 8
    nlev = 5
    z_top = 30000.0
    grid = create_cubed_sphere(n, use_duogrid=True)
    height_coord = create_height_coordinate(nlev, z_top)
    terrain = jnp.zeros((6, n, n))
    terrain_metric = compute_terrain_metric(terrain, height_coord)
    rng = np.random.default_rng(seed=429)
    u_p = rng.uniform(-1.0, 1.0, size=(6, n, n, nlev))
    v_p = rng.uniform(-1.0, 1.0, size=(6, n, n, nlev))
    dims_3d = ("face", "x", "y", "level")
    dims_w = ("face", "x", "y", "level_half")
    dims_2d = ("face", "x", "y")
    s = NonHydrostaticState(
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
    u0 = float(jnp.max(jnp.abs(s.u.data)))
    for _ in range(3):
        s = model.step(s, dt=5.0)
        assert jnp.all(jnp.isfinite(s.u.data))
        assert jnp.all(jnp.isfinite(s.theta_prime.data))
    u3 = float(jnp.max(jnp.abs(s.u.data)))
    th3 = float(jnp.max(jnp.abs(s.theta_prime.data)))
    assert u3 < 100.0 * u0, (
        f"NH factory + 3 steps: |u| grew {u0:.3f} → {u3:.3f} "
        f"(>100× initial)."
    )
    assert th3 < 50.0, (
        f"NH factory + 3 steps: |θ′| reached {th3:.3f} K "
        f"(>50 K from rest)."
    )


def test_factory_pe_three_step_bounded():
    n = 8
    nlev = 5
    grid = create_cubed_sphere(n, use_duogrid=True)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    coord = standard_hybrid_levels(nlev)
    state_cc = held_suarez_init(grid, coord)
    s = hydrostatic_to_fv3(state_cc, cdgrid)
    cfg = make_fv3_faithful_pe_config(
        damp_v=0.030, nord_v=1, damp_v_d_con=1.0,
        corner_div_damp_d2_bg=0.0005,
        corner_div_damp_d_con=1.0,
        div_damp_coeff=1e6, div_damp_d_con=1.0,
    )
    model = CDGridPrimitiveEquationModel(grid, coord, cfg)
    T0_max = float(jnp.max(jnp.abs(s.T.data)))
    for _ in range(3):
        s = model.step(s, dt=5.0)
        assert jnp.all(jnp.isfinite(s.T.data))
        assert jnp.all(jnp.isfinite(s.u_d.data))
    T3_max = float(jnp.max(jnp.abs(s.T.data)))
    assert T3_max < 2.0 * T0_max, (
        f"PE factory + 3 steps: |T| grew {T0_max:.1f} → "
        f"{T3_max:.1f} K (>2× initial)."
    )
