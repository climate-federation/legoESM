"""FV3_3D iter 384: independence test for iter-370 cross_face
flag + iter-338/339 metric flag.

Verifies the two flags affect state through INDEPENDENT
mechanisms (no double-cancellation):

- metric flag changes the d_con HEAT formula (cosa_s + rsin2
  correction)
- cross_face flag changes the WIND halo at cube edges

Both flags together should produce state different from either
alone.

Tests
-----

1. ``test_pe_flags_combine_independently`` — at PE,
   max|s_both - s_metric_only| > 0 AND
   max|s_both - s_cross_only| > 0.
2. ``test_nh_flags_combine_independently`` — same for NH.
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
    compute_terrain_metric, create_height_coordinate,
    standard_hybrid_levels,
)


def test_pe_flags_combine_independently():
    n = 8
    nlev = 5
    grid = create_cubed_sphere(n, use_duogrid=True)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    coord = standard_hybrid_levels(nlev)
    state_cc = held_suarez_init(grid, coord)
    state = hydrostatic_to_fv3(state_cc, cdgrid)
    rng = np.random.default_rng(seed=384)
    u_p = rng.uniform(-20.0, 20.0, size=(6, n + 1, n + 1, nlev))
    v_p = rng.uniform(-20.0, 20.0, size=(6, n + 1, n + 1, nlev))
    state = state._replace(
        u_d=state.u_d.replace(data=jnp.asarray(u_p)),
        v_d=state.v_d.replace(data=jnp.asarray(v_p)),
    )

    def cfg(metric, cross):
        return CDGridPrimitiveEquationConfig(
            damp_v=0.030, nord_v=1, damp_v_d_con=1.0,
            corner_div_damp_d2_bg=0.0, A_h=0.0,
            hyperdiff_coeff=0.0, hyperdiff_ps_coeff=0.0,
            div_damp_coeff=0.0,
            use_conservation_fixer=False, fix_mass=False,
            use_fv3_metric_aware_d_con=metric,
            use_fv3_cross_face_du_proj=cross,
        )

    s_metric = CDGridPrimitiveEquationModel(
        grid, coord, cfg(True, False),
    ).step(state, 100.0)
    s_cross = CDGridPrimitiveEquationModel(
        grid, coord, cfg(False, True),
    ).step(state, 100.0)
    s_both = CDGridPrimitiveEquationModel(
        grid, coord, cfg(True, True),
    ).step(state, 100.0)

    # cross_face affects u_d (post-step wind state via
    # du_corner), metric affects T (d_con heat formula).
    # Compare BOTH fields to confirm independent effects.
    diff_vs_metric = float(np.max(np.abs(
        np.asarray(s_both.u_d.data) - np.asarray(s_metric.u_d.data),
    )))
    diff_vs_cross = float(np.max(np.abs(
        np.asarray(s_both.T.data) - np.asarray(s_cross.T.data),
    )))
    assert diff_vs_metric > 1e-12, (
        "PE both-flags state matches metric-only — cross_face "
        "flag has no effect when metric is also ON?"
    )
    assert diff_vs_cross > 1e-12, (
        "PE both-flags state matches cross-only — metric flag "
        "has no effect when cross_face is also ON?"
    )


def test_nh_flags_combine_independently():
    n = 8
    nlev = 5
    z_top = 30000.0
    grid = create_cubed_sphere(n, use_duogrid=True)
    hc = create_height_coordinate(nlev, z_top)
    terrain = jnp.zeros((6, n, n))
    tm = compute_terrain_metric(terrain, hc)
    rng = np.random.default_rng(seed=384)
    u_p = rng.uniform(-15.0, 15.0, size=(6, n, n, nlev))
    v_p = rng.uniform(-15.0, 15.0, size=(6, n, n, nlev))
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

    def cfg(metric, cross):
        return CDGridCompressibleEulerConfig(
            hyperdiff_coeff=1e14, n_acoustic_substeps=4,
            damp_v=0.030, nord_v=1, damp_v_d_con=1.0,
            use_fv3_metric_aware_d_con=metric,
            use_fv3_cross_face_du_proj=cross,
        )

    s_m = CDGridCompressibleEulerModel(
        grid, hc, tm, cfg(True, False),
    ).step(state, 5.0)
    s_c = CDGridCompressibleEulerModel(
        grid, hc, tm, cfg(False, True),
    ).step(state, 5.0)
    s_b = CDGridCompressibleEulerModel(
        grid, hc, tm, cfg(True, True),
    ).step(state, 5.0)

    diff_vs_m = float(np.max(np.abs(
        np.asarray(s_b.u.data) - np.asarray(s_m.u.data),
    )))
    diff_vs_c = float(np.max(np.abs(
        np.asarray(s_b.theta_prime.data)
        - np.asarray(s_c.theta_prime.data),
    )))
    assert diff_vs_m > 1e-12
    assert diff_vs_c > 1e-12
