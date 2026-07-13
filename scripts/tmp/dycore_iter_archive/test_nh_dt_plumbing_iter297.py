"""FV3_3D iter 297: NH iter-189 dt-plumbing regression guard
(NH counterpart of PE iter-296).

iter-189 plumbed the actual integration ``dt`` from
``model.step(state, dt)`` through to the corner-div damping
adaptive cap on BOTH PE and NH paths.  iter-296 pinned PE;
iter-297 mirrors it for NH (default ``corner_div_damp_dt_proxy``
is 10.0 in NH vs 200.0 in PE — separate plumbing).

Two NH models differing ONLY in ``corner_div_damp_dt_proxy``
(2.0 vs 50.0) produce bit-for-bit identical state from
``model.step(state, dt=10.0)`` — confirms the actual dt wins.

Tests
-----

1. ``test_nh_step_uses_dt_actual_over_proxy`` — bit-for-bit
   identical step output across different
   ``corner_div_damp_dt_proxy`` values when ``model.step`` is
   used.
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


def _make_state(seed):
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

    rng = np.random.default_rng(seed=seed)
    u_p = rng.uniform(-3.0, 3.0, size=(6, n, n, nlev))
    v_p = rng.uniform(-3.0, 3.0, size=(6, n, n, nlev))
    w_p = rng.uniform(-0.3, 0.3, size=(6, n, n, nlev + 1))

    state = NonHydrostaticState(
        u=Field(data=jnp.asarray(u_p), name="u",
                dims=dims_3d, units="m/s"),
        v=Field(data=jnp.asarray(v_p), name="v",
                dims=dims_3d, units="m/s"),
        w=Field(data=jnp.asarray(w_p), name="w",
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


def _build_cfg(dt_proxy):
    return CDGridCompressibleEulerConfig(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        corner_div_damp_d2_bg=0.0005, corner_div_damp_dddmp=0.20,
        corner_div_damp_d4_bg=0.02, corner_div_damp_nord=1,
        corner_div_damp_dt_proxy=dt_proxy,
        damp_v=0.030, nord_v=2,
        use_fv3_a2b_zeta_corner=True,
        div_damp_coeff=1e6, div_damp_dddmp=0.20,
        A_h=1e6, smagorinsky_cs=0.20,
        damp_w=0.030, nord_w=1,
        damp_w_d_con=1.0,
        damp_v_d_con=1.0,
        corner_div_damp_d_con=1.0,
        div_damp_d_con=1.0,
        ah_d_con=1.0,
        delt_max=1.0,
    )


def test_nh_step_uses_dt_actual_over_proxy():
    """``model.step(state, dt)`` (NH) must use real dt, not
    ``corner_div_damp_dt_proxy`` from the config.

    Two configs differ ONLY in ``corner_div_damp_dt_proxy``
    (2.0 vs 50.0).  Both must produce identical step output
    when ``model.step`` is called with the same dt — confirms
    iter-189 dt plumbing on the NH path.
    """
    grid, height_coord, terrain_metric, state = _make_state(seed=297)

    cfg_proxy_2 = _build_cfg(dt_proxy=2.0)
    cfg_proxy_50 = _build_cfg(dt_proxy=50.0)

    model_2 = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg_proxy_2,
    )
    model_50 = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg_proxy_50,
    )

    dt = 10.0
    s_2 = model_2.step(state, dt)
    s_50 = model_50.step(state, dt)

    np.testing.assert_array_equal(
        np.asarray(s_2.u.data), np.asarray(s_50.u.data),
    )
    np.testing.assert_array_equal(
        np.asarray(s_2.v.data), np.asarray(s_50.v.data),
    )
    np.testing.assert_array_equal(
        np.asarray(s_2.w.data), np.asarray(s_50.w.data),
    )
    np.testing.assert_array_equal(
        np.asarray(s_2.theta_prime.data),
        np.asarray(s_50.theta_prime.data),
    )
