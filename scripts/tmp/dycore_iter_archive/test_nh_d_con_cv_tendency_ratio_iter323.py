"""FV3_3D iter 323: tendency-level c_pd/c_vd ratio for the 3 NH
slow-tendency d_con sites under iter-320's
``use_fv3_d_con_cv = True`` flag.

iter-230 verified linearity in d_con for the 3 NH slow-tendency
sites (corner_div / div_damp / A_h) at the
``cdgrid_compressible_euler_slow_tendencies`` output level.
iter-321 / iter-322 verified the c_vd-balance for the 2
post-acoustic d_con sites (damp_w / damp_v) at the full ``step``
level.

iter-323 closes the symmetric coverage at the slow-tendency level:
the ratio between cv-flagged and cp-default ``dθ_p/dt`` must equal
exactly ``c_pd / c_vd ≈ 1.40`` element-wise (rtol=1e-12) at every
slow-tendency d_con site.  Acoustic feedback does not contaminate
this test because we extract ``dθ_p/dt`` BEFORE acoustic substeps.

This is the cleanest possible iter-320 verification at the 3 sites
that are otherwise hardest to isolate (cdd / dd / ah feed
acoustic, so full-step state comparisons see feedback noise).

Tests
-----

1. ``test_corner_div_damp_d_con_cv_ratio`` — iter-222 site.
2. ``test_div_damp_d_con_cv_ratio`` — iter-224 site.
3. ``test_ah_d_con_cv_ratio`` — iter-226 site.

Each runs slow_tendencies twice (cp, cv) at d_con=1.0 and asserts
``c_pd · dθ_p/dt|_cp == c_vd · dθ_p/dt|_cv`` (since the only
difference between the two paths is the heat-capacity denominator;
this is the bit-for-bit invariant of iter-320).
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
    cdgrid_compressible_euler_slow_tendencies,
)
from legoesm.core.field import Field
from legoesm.core.state import NonHydrostaticState
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.grids.vertical import (
    compute_terrain_metric, create_height_coordinate,
)


@pytest.fixture(scope="module")
def small_nh_state_with_winds():
    n = 8
    nlev = 5
    z_top = 30000.0
    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    height_coord = create_height_coordinate(nlev, z_top)
    terrain = jnp.zeros((6, n, n))
    terrain_metric = compute_terrain_metric(terrain, height_coord)

    dims_3d = ("face", "x", "y", "level")
    dims_w = ("face", "x", "y", "level_half")
    dims_2d = ("face", "x", "y")

    rng = np.random.default_rng(seed=323)
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
    return grid, cdgrid, height_coord, terrain_metric, state


def _check_cv_ratio(make_cfg, state, grid, height_coord,
                    terrain_metric, cdgrid):
    """Run slow_tendencies twice (cp, cv) at d_con=1.0; verify the
    bit-for-bit c_pd · dθ_p/dt|_cp == c_vd · dθ_p/dt|_cv invariant."""
    dt = 10.0
    cfg_no = make_cfg(0.0, False)
    cfg_cp = make_cfg(1.0, False)
    cfg_cv = make_cfg(1.0, True)
    tend_no = cdgrid_compressible_euler_slow_tendencies(
        state, grid, height_coord, terrain_metric, cdgrid,
        cfg_no, dt_actual=dt,
    )
    tend_cp = cdgrid_compressible_euler_slow_tendencies(
        state, grid, height_coord, terrain_metric, cdgrid,
        cfg_cp, dt_actual=dt,
    )
    tend_cv = cdgrid_compressible_euler_slow_tendencies(
        state, grid, height_coord, terrain_metric, cdgrid,
        cfg_cv, dt_actual=dt,
    )

    dtheta_dt_cp_dcon = (
        tend_cp.dtheta_prime_dt.data - tend_no.dtheta_prime_dt.data
    )
    dtheta_dt_cv_dcon = (
        tend_cv.dtheta_prime_dt.data - tend_no.dtheta_prime_dt.data
    )

    np.testing.assert_allclose(
        constants.c_pd * dtheta_dt_cp_dcon,
        constants.c_vd * dtheta_dt_cv_dcon,
        rtol=1e-12,
        err_msg=(
            "iter-320 cv flag invariant violation at slow-tendency "
            "level: c_pd · dθ_p/dt|_cp must equal c_vd · "
            "dθ_p/dt|_cv element-wise (only the denominator differs)."
        ),
    )
    # Sanity: the d_con contribution is non-zero.
    assert float(jnp.max(jnp.abs(dtheta_dt_cp_dcon))) > 1e-10


def test_corner_div_damp_d_con_cv_ratio(small_nh_state_with_winds):
    """iter-222 NH corner-div d_con: cv ratio invariant."""
    grid, cdgrid, hc, tm, state = small_nh_state_with_winds

    def make_cfg(d, cv):
        return CDGridCompressibleEulerConfig(
            hyperdiff_coeff=1e14, n_acoustic_substeps=4,
            corner_div_damp_d2_bg=0.0005,
            corner_div_damp_dddmp=0.20,
            corner_div_damp_d_con=d,
            use_fv3_d_con_cv=cv,
        )

    _check_cv_ratio(make_cfg, state, grid, hc, tm, cdgrid)


def test_div_damp_d_con_cv_ratio(small_nh_state_with_winds):
    """iter-224 NH cell-centre div_damp d_con: cv ratio invariant."""
    grid, cdgrid, hc, tm, state = small_nh_state_with_winds

    def make_cfg(d, cv):
        return CDGridCompressibleEulerConfig(
            hyperdiff_coeff=1e14, n_acoustic_substeps=4,
            div_damp_coeff=1e6, div_damp_dddmp=0.20,
            div_damp_d_con=d,
            use_fv3_d_con_cv=cv,
        )

    _check_cv_ratio(make_cfg, state, grid, hc, tm, cdgrid)


def test_ah_d_con_cv_ratio(small_nh_state_with_winds):
    """iter-226 NH Smagorinsky-A_h d_con: cv ratio invariant."""
    grid, cdgrid, hc, tm, state = small_nh_state_with_winds

    def make_cfg(d, cv):
        return CDGridCompressibleEulerConfig(
            hyperdiff_coeff=1e14, n_acoustic_substeps=4,
            A_h=1e6, smagorinsky_cs=0.20,
            ah_d_con=d,
            use_fv3_d_con_cv=cv,
        )

    _check_cv_ratio(make_cfg, state, grid, hc, tm, cdgrid)
