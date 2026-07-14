"""FV3_3D iter 345: cube-edge concentration test for iter-338 PE
+ iter-339 NH metric-aware d_con form.

The metric correction's ``cosa_s`` factor is non-zero at cube
edges (where the grid is non-orthogonal) and ZERO at the face
interior (where the grid is locally orthogonal).  Therefore the
diff between metric and simple forms must CONCENTRATE at panel
edges.

iter-338 had an edge-concentration test but it tested under the
old (buggy) scaling.  iter-344 fixed the scaling.  iter-345
re-pins edge concentration on the FIXED metric form.

Tests
-----

1. ``test_pe_metric_concentration_at_edges`` — PE diff
   ``T(metric) - T(simple)`` concentrates at panel edges
   (edge_max > 1.2 × interior_max).
2. ``test_nh_metric_concentration_at_edges`` — same for NH θ_p.
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


def test_pe_metric_concentration_at_edges():
    n = 8
    nlev = 5
    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    coord = standard_hybrid_levels(nlev)
    state_cc = held_suarez_init(grid, coord)
    state = hydrostatic_to_fv3(state_cc, cdgrid)
    rng = np.random.default_rng(seed=345)
    n_corners = n + 1
    u_p = rng.uniform(-25.0, 25.0,
                      size=(6, n_corners, n_corners, nlev))
    v_p = rng.uniform(-25.0, 25.0,
                      size=(6, n_corners, n_corners, nlev))
    state = state._replace(
        u_d=state.u_d.replace(data=jnp.asarray(u_p)),
        v_d=state.v_d.replace(data=jnp.asarray(v_p)),
    )

    def make_cfg(metric):
        return CDGridPrimitiveEquationConfig(
            damp_v=0.030, nord_v=1, damp_v_d_con=1.0,
            corner_div_damp_d2_bg=0.0, A_h=0.0,
            hyperdiff_coeff=0.0, hyperdiff_ps_coeff=0.0,
            div_damp_coeff=0.0,
            use_conservation_fixer=False, fix_mass=False,
            use_fv3_metric_aware_d_con=metric,
        )

    m_simple = CDGridPrimitiveEquationModel(grid, coord, make_cfg(False))
    m_metric = CDGridPrimitiveEquationModel(grid, coord, make_cfg(True))
    s_simple = m_simple.step(state, 100.0)
    s_metric = m_metric.step(state, 100.0)
    diff_T = np.asarray(s_metric.T.data) - np.asarray(s_simple.T.data)

    edge_width = 2
    i_idx = np.arange(n)
    edge_i = (i_idx < edge_width) | (i_idx >= n - edge_width)
    edge_mask = edge_i[:, None] | edge_i[None, :]
    interior_mask = ~edge_mask
    edge_max = float(np.max(np.abs(diff_T[:, edge_mask, :])))
    interior_max = float(np.max(np.abs(diff_T[:, interior_mask, :])))
    assert edge_max > interior_max * 1.2, (
        f"PE metric correction not concentrated at panel edges: "
        f"edge_max={edge_max:.3e}, interior_max={interior_max:.3e}."
    )


def test_nh_metric_concentration_at_edges():
    n = 8
    nlev = 5
    z_top = 30000.0
    grid = create_cubed_sphere(n)
    height_coord = create_height_coordinate(nlev, z_top)
    terrain = jnp.zeros((6, n, n))
    terrain_metric = compute_terrain_metric(terrain, height_coord)

    rng = np.random.default_rng(seed=345)
    u_p = rng.uniform(-25.0, 25.0, size=(6, n, n, nlev))
    v_p = rng.uniform(-25.0, 25.0, size=(6, n, n, nlev))

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

    def make_cfg(metric):
        return CDGridCompressibleEulerConfig(
            hyperdiff_coeff=1e14, n_acoustic_substeps=4,
            damp_v=0.030, nord_v=1, damp_v_d_con=1.0,
            use_fv3_metric_aware_d_con=metric,
        )

    m_simple = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, make_cfg(False),
    )
    m_metric = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, make_cfg(True),
    )
    s_simple = m_simple.step(state, 5.0)
    s_metric = m_metric.step(state, 5.0)
    diff_theta = (
        np.asarray(s_metric.theta_prime.data)
        - np.asarray(s_simple.theta_prime.data)
    )

    edge_width = 2
    i_idx = np.arange(n)
    edge_i = (i_idx < edge_width) | (i_idx >= n - edge_width)
    edge_mask = edge_i[:, None] | edge_i[None, :]
    interior_mask = ~edge_mask
    edge_max = float(np.max(np.abs(diff_theta[:, edge_mask, :])))
    interior_max = float(np.max(np.abs(diff_theta[:, interior_mask, :])))
    assert edge_max > interior_max * 1.2, (
        f"NH metric correction not concentrated at panel edges: "
        f"edge_max={edge_max:.3e}, interior_max={interior_max:.3e}."
    )
