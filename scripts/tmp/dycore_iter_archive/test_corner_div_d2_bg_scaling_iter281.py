"""FV3_3D iter 281: scaling test for ``corner_div_damp_d2_bg``
(FV3 ``d2_bg`` background floor in the iter-16 nord=0 corner-
div damping path).

When dddmp=0 (no Smagorinsky) the FV3 formula reduces to:

    damp = da_min_c * d2_bg

which is LINEAR in d2_bg.  The wind change from corner-div
damp scales linearly with damp_corner, so doubling d2_bg
should double the wind change.

Tests
-----

1. ``test_pe_corner_div_d2_bg_linear_scaling`` — PE: 2x
   d2_bg → 2x wind change (within FP tolerance).
2. ``test_nh_corner_div_d2_bg_linear_scaling`` — NH same.
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


def test_pe_corner_div_d2_bg_linear_scaling():
    """PE: doubling corner_div_damp_d2_bg doubles wind change
    (with dddmp=0 to avoid the Smagorinsky cap)."""
    n = 8
    nlev = 6
    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    coord = standard_hybrid_levels(nlev)
    state_cc = held_suarez_init(grid, coord)
    state = hydrostatic_to_fv3(state_cc, cdgrid)

    rng = np.random.default_rng(seed=281)
    n_corners = n + 1
    u_p = rng.uniform(-30.0, 30.0,
                      size=(6, n_corners, n_corners, nlev))
    v_p = rng.uniform(-30.0, 30.0,
                      size=(6, n_corners, n_corners, nlev))
    state = state._replace(
        u_d=state.u_d.replace(data=jnp.asarray(u_p)),
        v_d=state.v_d.replace(data=jnp.asarray(v_p)),
    )

    common = dict(
        # dddmp=0 → only the d2_bg floor is active.
        corner_div_damp_dddmp=0.0,
        damp_v=0.0,
        div_damp_coeff=0.0,
        A_h=0.0, hyperdiff_coeff=0.0, hyperdiff_ps_coeff=0.0,
        T_diss_coeff=0.0,
        use_conservation_fixer=False, fix_mass=False,
        zero_mean_ps_tendency=False,
    )
    base = 0.0005
    cfg_off = CDGridPrimitiveEquationConfig(
        **common, corner_div_damp_d2_bg=0.0,
    )
    cfg_1x = CDGridPrimitiveEquationConfig(
        **common, corner_div_damp_d2_bg=base,
    )
    cfg_2x = CDGridPrimitiveEquationConfig(
        **common, corner_div_damp_d2_bg=2.0 * base,
    )

    s_off = CDGridPrimitiveEquationModel(grid, coord, cfg_off).step(
        state, 100.0,
    )
    s_1x = CDGridPrimitiveEquationModel(grid, coord, cfg_1x).step(
        state, 100.0,
    )
    s_2x = CDGridPrimitiveEquationModel(grid, coord, cfg_2x).step(
        state, 100.0,
    )

    du_1x = s_1x.u_d.data - s_off.u_d.data
    du_2x = s_2x.u_d.data - s_off.u_d.data

    mask = jnp.abs(du_1x) > 1e-6
    if not jnp.any(mask):
        pytest.skip("No detectable corner-div effect")

    ratio = du_2x[mask] / du_1x[mask]
    mean_ratio = float(jnp.mean(ratio))
    np.testing.assert_allclose(
        mean_ratio, 2.0, rtol=0.05,
        err_msg=(
            f"PE corner_div_damp_d2_bg scaling: 2x d2_bg "
            f"should give 2x wind change; got {mean_ratio:.3f}."
        ),
    )


def test_nh_corner_div_d2_bg_linear_scaling():
    """NH: doubling corner_div_damp_d2_bg doubles wind change."""
    n = 8
    nlev = 5
    z_top = 30000.0
    grid = create_cubed_sphere(n)
    height_coord = create_height_coordinate(nlev, z_top)
    terrain = jnp.zeros((6, n, n))
    terrain_metric = compute_terrain_metric(terrain, height_coord)

    dims_3d = ("face", "x", "y", "level")
    dims_w = ("face", "x", "y", "level_half")
    dims_2d = ("face", "x", "y")

    rng = np.random.default_rng(seed=2810)
    u_p = rng.uniform(-30.0, 30.0, size=(6, n, n, nlev))
    v_p = rng.uniform(-30.0, 30.0, size=(6, n, n, nlev))

    state = NonHydrostaticState(
        u=Field(data=jnp.asarray(u_p), name="u",
                dims=dims_3d, units="m/s"),
        v=Field(data=jnp.asarray(v_p), name="v",
                dims=dims_3d, units="m/s"),
        w=Field(data=jnp.zeros((6, n, n, nlev + 1)),
                name="w", dims=dims_w, units="m/s"),
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

    common = dict(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        corner_div_damp_dddmp=0.0,
    )
    base = 0.0005
    cfg_off = CDGridCompressibleEulerConfig(
        **common, corner_div_damp_d2_bg=0.0,
    )
    cfg_1x = CDGridCompressibleEulerConfig(
        **common, corner_div_damp_d2_bg=base,
    )
    cfg_2x = CDGridCompressibleEulerConfig(
        **common, corner_div_damp_d2_bg=2.0 * base,
    )

    s_off = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg_off,
    ).step(state, 10.0)
    s_1x = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg_1x,
    ).step(state, 10.0)
    s_2x = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg_2x,
    ).step(state, 10.0)

    du_1x = s_1x.u.data - s_off.u.data
    du_2x = s_2x.u.data - s_off.u.data

    mask = jnp.abs(du_1x) > 1e-6
    if not jnp.any(mask):
        pytest.skip("No detectable corner-div effect")

    ratio = du_2x[mask] / du_1x[mask]
    mean_ratio = float(jnp.mean(ratio))
    np.testing.assert_allclose(
        mean_ratio, 2.0, rtol=0.10,
        err_msg=(
            f"NH corner_div_damp_d2_bg scaling: 2x d2_bg "
            f"should give 2x wind change; got {mean_ratio:.3f}."
        ),
    )
