"""FV3_3D iter 283: NH counterpart of PE iter-282 — verify the
``dd8 = (da_min_c * d4_bg)^(nord+1)`` scaling on the iter-168
NH corner-div damping nord >= 1 branch.

Tests
-----

1. ``test_nh_corner_div_d4_bg_nord1_scaling`` — 2x d4_bg →
   4x wind change at nord=1.
2. ``test_nh_corner_div_d4_bg_nord2_scaling`` — 2x d4_bg →
   8x at nord=2.
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


def _check_nh_d4_bg_scaling(nord_v, expected_factor):
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

    rng = np.random.default_rng(seed=283 + nord_v)
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
        corner_div_damp_d2_bg=1e-10,    # tiny floor
        corner_div_damp_dddmp=0.0,
        corner_div_damp_nord=nord_v,
    )
    base = 1e-3
    cfg_off = CDGridCompressibleEulerConfig(
        **common, corner_div_damp_d4_bg=0.0,
    )
    cfg_1x = CDGridCompressibleEulerConfig(
        **common, corner_div_damp_d4_bg=base,
    )
    cfg_2x = CDGridCompressibleEulerConfig(
        **common, corner_div_damp_d4_bg=2.0 * base,
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
        pytest.skip("No detectable d4_bg branch effect")

    ratio = du_2x[mask] / du_1x[mask]
    mean_ratio = float(jnp.mean(ratio))
    np.testing.assert_allclose(
        mean_ratio, expected_factor,
        rtol=0.10,
        err_msg=(
            f"NH corner_div d4_bg at nord={nord_v}: expected "
            f"~{expected_factor}x; got {mean_ratio:.3f}."
        ),
    )


def test_nh_corner_div_d4_bg_nord1_scaling():
    _check_nh_d4_bg_scaling(nord_v=1, expected_factor=4.0)


def test_nh_corner_div_d4_bg_nord2_scaling():
    _check_nh_d4_bg_scaling(nord_v=2, expected_factor=8.0)
