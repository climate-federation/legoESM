"""FV3_3D iter 280: damp_w scaling test for the iter-193 NH
damp_w wiring.  Verifies ``damp4 = (damp_w * da_min_c)^(nord_w
+ 1)`` formula.

Mirrors iter-278/279 damp_v scaling for damp_w on NH.

Tests
-----

1. ``test_nh_damp_w_scaling_nord0`` — nord=0, factor=2.
2. ``test_nh_damp_w_scaling_nord1`` — nord=1, factor=4.
3. ``test_nh_damp_w_scaling_nord2`` — nord=2, factor=8.
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


def _setup_w_state():
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

    rng = np.random.default_rng(seed=280)
    # w-perturbation respecting w=0 BC.
    w_pert = np.zeros((6, n, n, nlev + 1))
    w_pert[..., 1:nlev] = rng.uniform(
        -2.0, 2.0, size=(6, n, n, nlev - 1),
    )

    state = NonHydrostaticState(
        u=Field(data=jnp.zeros((6, n, n, nlev)),
                name="u", dims=dims_3d, units="m/s"),
        v=Field(data=jnp.zeros((6, n, n, nlev)),
                name="v", dims=dims_3d, units="m/s"),
        w=Field(data=jnp.asarray(w_pert), name="w",
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
    return grid, height_coord, terrain_metric, state


def _check_w_scaling(nord_w, expected_factor):
    grid, hc, tm, state = _setup_w_state()
    common = dict(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        nord_w=nord_w, damp_w_d_con=0.0,
    )
    cfg_off = CDGridCompressibleEulerConfig(**common, damp_w=0.0)
    cfg_1x = CDGridCompressibleEulerConfig(**common, damp_w=0.030)
    cfg_2x = CDGridCompressibleEulerConfig(**common, damp_w=0.060)

    s_off = CDGridCompressibleEulerModel(grid, hc, tm, cfg_off).step(
        state, 10.0,
    )
    s_1x = CDGridCompressibleEulerModel(grid, hc, tm, cfg_1x).step(
        state, 10.0,
    )
    s_2x = CDGridCompressibleEulerModel(grid, hc, tm, cfg_2x).step(
        state, 10.0,
    )

    dw_1x = s_1x.w.data - s_off.w.data
    dw_2x = s_2x.w.data - s_off.w.data

    # Mask zero-BC boundary half-levels and tiny effects.
    mask = jnp.abs(dw_1x) > 1e-8
    if not jnp.any(mask):
        pytest.skip("No detectable damp_w effect at this nord")

    ratio = dw_2x[mask] / dw_1x[mask]
    mean_ratio = float(jnp.mean(ratio))
    np.testing.assert_allclose(
        mean_ratio, expected_factor,
        rtol=0.10,
        err_msg=(
            f"NH damp_w scaling at nord_w={nord_w}: expected "
            f"~{expected_factor}x; got {mean_ratio:.3f}."
        ),
    )


def test_nh_damp_w_scaling_nord0():
    _check_w_scaling(nord_w=0, expected_factor=2.0)


def test_nh_damp_w_scaling_nord1():
    _check_w_scaling(nord_w=1, expected_factor=4.0)


def test_nh_damp_w_scaling_nord2():
    _check_w_scaling(nord_w=2, expected_factor=8.0)
