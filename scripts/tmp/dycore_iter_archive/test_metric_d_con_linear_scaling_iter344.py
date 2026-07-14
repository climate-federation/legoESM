"""FV3_3D iter 344: linearity-in-damp_v_d_con scaling test for
iter-338 (PE) + iter-339 (NH) metric-aware d_con forms.

The FV3 metric form ``heat = -0.25 * d_con * rsin2 * (sum_edges
+ 2*sum(gy,gx) - cosa_s*cross)`` is LINEAR in ``d_con``.  Mirrors
iter-230 NH scaling test pattern.

Tests
-----

1. ``test_pe_metric_linear_in_damp_v_d_con`` — at iter-338 PE
   site, ``Δθ_p(d=2.0) - Δθ_p(d=0.5) == 3.0 * (Δθ_p(d=1.0) -
   Δθ_p(d=0.5))`` (rtol=1e-10).
2. ``test_nh_metric_linear_in_damp_v_d_con`` — same for iter-339
   NH site.

Catches non-linear coupling errors in the metric form's
formula (e.g., d_con² coupling, accidental clamp).
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


def test_pe_metric_linear_in_damp_v_d_con():
    """PE iter-338 metric form: ΔT scales linearly with d_con."""
    n = 8
    nlev = 5
    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    coord = standard_hybrid_levels(nlev)
    state_cc = held_suarez_init(grid, coord)
    state = hydrostatic_to_fv3(state_cc, cdgrid)
    rng = np.random.default_rng(seed=344)
    n_corners = n + 1
    u_p = rng.uniform(-25.0, 25.0,
                      size=(6, n_corners, n_corners, nlev))
    v_p = rng.uniform(-25.0, 25.0,
                      size=(6, n_corners, n_corners, nlev))
    state = state._replace(
        u_d=state.u_d.replace(data=jnp.asarray(u_p)),
        v_d=state.v_d.replace(data=jnp.asarray(v_p)),
    )

    def make_cfg(d):
        return CDGridPrimitiveEquationConfig(
            damp_v=0.030, nord_v=1, damp_v_d_con=d,
            corner_div_damp_d2_bg=0.0, A_h=0.0,
            hyperdiff_coeff=0.0, hyperdiff_ps_coeff=0.0,
            div_damp_coeff=0.0,
            use_conservation_fixer=False, fix_mass=False,
            use_fv3_metric_aware_d_con=True,
        )

    m_05 = CDGridPrimitiveEquationModel(grid, coord, make_cfg(0.5))
    m_10 = CDGridPrimitiveEquationModel(grid, coord, make_cfg(1.0))
    m_20 = CDGridPrimitiveEquationModel(grid, coord, make_cfg(2.0))
    s_05 = m_05.step(state, 100.0)
    s_10 = m_10.step(state, 100.0)
    s_20 = m_20.step(state, 100.0)
    delta_10 = np.asarray(s_10.T.data) - np.asarray(s_05.T.data)
    delta_20 = np.asarray(s_20.T.data) - np.asarray(s_05.T.data)
    np.testing.assert_allclose(
        delta_20, 3.0 * delta_10, rtol=1e-10, atol=1e-12,
    )
    assert float(np.max(np.abs(delta_10))) > 1e-10


def test_nh_metric_linear_in_damp_v_d_con():
    """NH iter-339 metric form: Δθ_p scales linearly with d_con."""
    n = 8
    nlev = 5
    z_top = 30000.0
    grid = create_cubed_sphere(n)
    height_coord = create_height_coordinate(nlev, z_top)
    terrain = jnp.zeros((6, n, n))
    terrain_metric = compute_terrain_metric(terrain, height_coord)

    rng = np.random.default_rng(seed=344)
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

    def make_cfg(d):
        return CDGridCompressibleEulerConfig(
            hyperdiff_coeff=1e14, n_acoustic_substeps=4,
            damp_v=0.030, nord_v=1, damp_v_d_con=d,
            use_fv3_metric_aware_d_con=True,
        )

    m_05 = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, make_cfg(0.5),
    )
    m_10 = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, make_cfg(1.0),
    )
    m_20 = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, make_cfg(2.0),
    )
    s_05 = m_05.step(state, 5.0)
    s_10 = m_10.step(state, 5.0)
    s_20 = m_20.step(state, 5.0)
    delta_10 = (
        np.asarray(s_10.theta_prime.data)
        - np.asarray(s_05.theta_prime.data)
    )
    delta_20 = (
        np.asarray(s_20.theta_prime.data)
        - np.asarray(s_05.theta_prime.data)
    )
    np.testing.assert_allclose(
        delta_20, 3.0 * delta_10, rtol=1e-10, atol=1e-12,
    )
    assert float(np.max(np.abs(delta_10))) > 1e-10
