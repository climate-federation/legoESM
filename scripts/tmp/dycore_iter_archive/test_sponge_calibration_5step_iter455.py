"""FV3_3D iter 455: 5-step stability check at legoESM-scale
sponge calibration values.

Per iter-452 finding, FV3 namelist values
``d2_bg_k1=4.0``/``d2_bg_k2=2.0`` are NOT directly portable to
legoESM (cause blow-up).  legoESM-compatible scale is
~1e-4 paired with ``d2_bg=5e-4``.

iter-455 pins a 5-step C8 run at this calibration as a
stability regression: if a future change makes the
``d2_bg_k1`` semantics inadvertently more aggressive, this
test catches the resulting blow-up.

Tests
-----

1. ``test_nh_sponge_calibration_5step_stable`` — 5 NH steps
   with d2_bg=5e-4, d2_bg_k1=1e-4, sponge_damp_w=True,
   sponge_damp_v=True → all fields finite, |u| bounded.
2. ``test_pe_sponge_calibration_5step_stable`` — same for PE.
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


def test_nh_sponge_calibration_5step_stable():
    n = 8
    nlev = 5
    z_top = 30000.0
    grid = create_cubed_sphere(n, use_duogrid=False)
    hc = create_height_coordinate(nlev, z_top)
    tm = compute_terrain_metric(jnp.zeros((6, n, n)), hc)
    rng = np.random.default_rng(seed=455)
    u_p = rng.uniform(-3.0, 3.0, size=(6, n, n, nlev))
    v_p = rng.uniform(-3.0, 3.0, size=(6, n, n, nlev))
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
    u0 = float(jnp.max(jnp.abs(state.u.data)))
    cfg = CDGridCompressibleEulerConfig(
        n_acoustic_substeps=4,
        damp_v=0.030, nord_v=1, damp_v_d_con=1.0,
        # iter-455 legoESM-scale calibration:
        corner_div_damp_d2_bg=5.0e-4,
        corner_div_damp_d2_bg_k1=1.0e-4,
        corner_div_damp_d_con=1.0,
        div_damp_coeff=1e6, div_damp_d_con=1.0,
        damp_w=0.030, damp_w_d_con=1.0,
        use_fv3_sponge_damp_w=True,
        use_fv3_sponge_damp_v=True,
    )
    model = CDGridCompressibleEulerModel(grid, hc, tm, cfg)
    s = state
    for _ in range(5):
        s = model.step(s, dt=10.0)
        assert jnp.all(jnp.isfinite(s.u.data))
        assert jnp.all(jnp.isfinite(s.theta_prime.data))
    u5 = float(jnp.max(jnp.abs(s.u.data)))
    assert u5 < 100.0 * u0, (
        f"5-step NH calibration check: |u| grew {u0:.2f}→{u5:.2f} "
        f"(>100× — blow-up).  Calibration may be too aggressive."
    )


def test_pe_sponge_calibration_5step_stable():
    n = 8
    nlev = 5
    grid = create_cubed_sphere(n, use_duogrid=False)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    coord = standard_hybrid_levels(nlev)
    state_cc = held_suarez_init(grid, coord)
    state = hydrostatic_to_fv3(state_cc, cdgrid)
    rng = np.random.default_rng(seed=455)
    n_corners = n + 1
    u_p = rng.uniform(-5.0, 5.0,
                      size=(6, n_corners, n_corners, nlev))
    v_p = rng.uniform(-5.0, 5.0,
                      size=(6, n_corners, n_corners, nlev))
    state = state._replace(
        u_d=state.u_d.replace(data=jnp.asarray(u_p)),
        v_d=state.v_d.replace(data=jnp.asarray(v_p)),
    )
    u0 = float(jnp.max(jnp.abs(state.u_d.data)))
    cfg = CDGridPrimitiveEquationConfig(
        damp_v=0.030, nord_v=1, damp_v_d_con=1.0,
        corner_div_damp_d2_bg=5.0e-4,
        corner_div_damp_d2_bg_k1=1.0e-4,
        corner_div_damp_d_con=1.0,
        div_damp_coeff=1e6, div_damp_d_con=1.0,
        use_fv3_sponge_damp_v=True,
    )
    model = CDGridPrimitiveEquationModel(grid, coord, cfg)
    s = state
    for _ in range(5):
        s = model.step(s, dt=10.0)
        assert jnp.all(jnp.isfinite(s.u_d.data))
        assert jnp.all(jnp.isfinite(s.T.data))
    u5 = float(jnp.max(jnp.abs(s.u_d.data)))
    assert u5 < 100.0 * u0, (
        f"5-step PE calibration check: |u_d| grew {u0:.2f}→"
        f"{u5:.2f} (>100×)."
    )
