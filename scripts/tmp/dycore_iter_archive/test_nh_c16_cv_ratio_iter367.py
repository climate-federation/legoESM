"""FV3_3D iter 367: C16 cv-vs-cp heating ratio test for NH
damp_v_d_con + damp_w_d_con post-acoustic sites.

iter-320 verified the c_p/c_v ≈ 1.40 element-wise ratio at C8.
iter-367 confirms the ratio holds at C16 with non-zero θ' and ρ'
state (where dynamic Exner perturbation is non-trivial).

Tests
-----

1. ``test_nh_c16_cv_ratio_post_acoustic`` — at the post-acoustic
   sites, mean(|Δθ_p_cv|) / mean(|Δθ_p_cp|) ≈ c_pd/c_vd at C16.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.atmosphere.dynamics.gcm.compressible_euler_cdgrid import (
    CDGridCompressibleEulerConfig,
    CDGridCompressibleEulerModel,
)
from legoesm.core.field import Field
from legoesm.core.state import NonHydrostaticState
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.vertical import (
    create_height_coordinate, compute_terrain_metric,
)


def test_nh_c16_cv_ratio_post_acoustic():
    n = 16
    nlev = 5
    z_top = 30000.0
    grid = create_cubed_sphere(n)
    height_coord = create_height_coordinate(nlev, z_top)
    terrain = jnp.zeros((6, n, n))
    terrain_metric = compute_terrain_metric(terrain, height_coord)
    rng = np.random.default_rng(seed=367)
    u_p = rng.uniform(-10.0, 10.0, size=(6, n, n, nlev))
    v_p = rng.uniform(-10.0, 10.0, size=(6, n, n, nlev))
    w_p = rng.uniform(-1.0, 1.0, size=(6, n, n, nlev + 1))
    w_p[..., 0] = 0.0
    w_p[..., -1] = 0.0
    dims_3d = ("face", "x", "y", "level")
    dims_w = ("face", "x", "y", "level_half")
    dims_2d = ("face", "x", "y")
    state = NonHydrostaticState(
        u=Field(data=jnp.asarray(u_p), name="u",
                dims=dims_3d, units="m/s"),
        v=Field(data=jnp.asarray(v_p), name="v",
                dims=dims_3d, units="m/s"),
        w=Field(data=jnp.asarray(w_p), name="w",
                dims=dims_w, units="m/s"),
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

    common = dict(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        damp_v=0.030, nord_v=1,
        damp_w=0.030, nord_w=1,
    )
    cfg_no = CDGridCompressibleEulerConfig(
        **common, damp_v_d_con=0.0, damp_w_d_con=0.0,
    )
    cfg_cp = CDGridCompressibleEulerConfig(
        **common, damp_v_d_con=1.0, damp_w_d_con=1.0,
        use_fv3_d_con_cv=False,
    )
    cfg_cv = CDGridCompressibleEulerConfig(
        **common, damp_v_d_con=1.0, damp_w_d_con=1.0,
        use_fv3_d_con_cv=True,
    )

    m_no = CDGridCompressibleEulerModel(grid, height_coord, terrain_metric, cfg_no)
    m_cp = CDGridCompressibleEulerModel(grid, height_coord, terrain_metric, cfg_cp)
    m_cv = CDGridCompressibleEulerModel(grid, height_coord, terrain_metric, cfg_cv)
    s_no = m_no.step(state, 5.0)
    s_cp = m_cp.step(state, 5.0)
    s_cv = m_cv.step(state, 5.0)

    dt_cp = np.asarray(s_cp.theta_prime.data) - np.asarray(s_no.theta_prime.data)
    dt_cv = np.asarray(s_cv.theta_prime.data) - np.asarray(s_no.theta_prime.data)
    ratio = float(np.mean(np.abs(dt_cv))) / float(np.mean(np.abs(dt_cp)))
    expected = constants.c_pd / constants.c_vd
    np.testing.assert_allclose(ratio, expected, rtol=1e-4)
