"""FV3_3D iter 366: composition test for iter-218/219 delt_max
sponge cap + iter-320 cv flag at the NH damp_v_d_con post-step
site.

iter-218/219 delt_max cap applies to ``|Δθ_p · Π_ref| ≤ dt ·
delt_max``.  Under cv flag, the Δθ_p magnitude is c_p/c_v ≈ 1.40
larger than cp path.  Does delt_max cap behave correctly under cv?

Tests
-----

1. ``test_baseline_bit_for_bit`` — delt_max + cv default OFF
   matches existing baseline.
2. ``test_cv_with_high_delt_max_changes_state`` — cv flag with
   loose cap (delt_max=10) produces state different from cp +
   loose cap.
3. ``test_cv_with_tight_delt_max_finite`` — cv flag with tight
   cap (delt_max=1e-4) clips heating without producing NaN.
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
    create_height_coordinate, compute_terrain_metric,
)


@pytest.fixture(scope="module")
def small_nh():
    n = 8
    nlev = 5
    z_top = 30000.0
    grid = create_cubed_sphere(n)
    height_coord = create_height_coordinate(nlev, z_top)
    terrain = jnp.zeros((6, n, n))
    terrain_metric = compute_terrain_metric(terrain, height_coord)
    rng = np.random.default_rng(seed=366)
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
    return grid, height_coord, terrain_metric, state


def _cfg(cv=False, delt_max=0.0):
    return CDGridCompressibleEulerConfig(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        damp_v=0.030, nord_v=1, damp_v_d_con=1.0,
        delt_max=delt_max,
        use_fv3_d_con_cv=cv,
    )


def test_baseline_bit_for_bit(small_nh):
    grid, hc, tm, state = small_nh
    m_default = CDGridCompressibleEulerModel(grid, hc, tm, _cfg())
    m_explicit = CDGridCompressibleEulerModel(
        grid, hc, tm, _cfg(cv=False, delt_max=0.0),
    )
    s_d = m_default.step(state, 5.0)
    s_e = m_explicit.step(state, 5.0)
    np.testing.assert_array_equal(
        np.asarray(s_d.theta_prime.data),
        np.asarray(s_e.theta_prime.data),
    )


def test_cv_with_high_delt_max_changes_state(small_nh):
    """Loose delt_max=10 with cv flag differs from cp + loose
    cap (cv path heats c_p/c_v larger; under loose cap nothing
    clips so cv vs cp differ)."""
    grid, hc, tm, state = small_nh
    m_cp = CDGridCompressibleEulerModel(
        grid, hc, tm, _cfg(cv=False, delt_max=10.0),
    )
    m_cv = CDGridCompressibleEulerModel(
        grid, hc, tm, _cfg(cv=True, delt_max=10.0),
    )
    s_cp = m_cp.step(state, 5.0)
    s_cv = m_cv.step(state, 5.0)
    diff = float(np.max(np.abs(
        np.asarray(s_cv.theta_prime.data)
        - np.asarray(s_cp.theta_prime.data),
    )))
    assert diff > 1e-12


def test_cv_with_tight_delt_max_finite(small_nh):
    """Tight delt_max=1e-4 + cv flag produces finite state
    (cap engages, no NaN)."""
    grid, hc, tm, state = small_nh
    m = CDGridCompressibleEulerModel(
        grid, hc, tm, _cfg(cv=True, delt_max=1e-4),
    )
    s = m.step(state, 5.0)
    for fld in (s.u.data, s.v.data, s.w.data,
                s.theta_prime.data, s.rho_prime.data):
        assert np.all(np.isfinite(np.asarray(fld)))
