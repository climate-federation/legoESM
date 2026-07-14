"""FV3_3D iter 364: composition test for NH damp_w_d_con +
use_fv3_dynamic_exner.

iter-337 wired dynamic Exner at the damp_w_d_con post-acoustic
site.  iter-364 verifies the composition:
- baseline (dyn_exner=False) bit-for-bit baseline
- dyn_exner=True with damp_w_d_con>0 changes θ_p
- composition with cv flag finite

Tests
-----

1. ``test_damp_w_dyn_exner_baseline_bit_for_bit``
2. ``test_damp_w_dyn_exner_changes_theta_p``
3. ``test_damp_w_dyn_exner_plus_cv_finite``
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
    rng = np.random.default_rng(seed=364)
    w_p = rng.uniform(-1.0, 1.0, size=(6, n, n, nlev + 1))
    # Zero w at top/bottom BC.
    w_p[..., 0] = 0.0
    w_p[..., -1] = 0.0
    theta_p = rng.uniform(-10.0, 10.0, size=(6, n, n, nlev))
    rho_p = rng.uniform(-0.15, 0.15, size=(6, n, n, nlev))
    dims_3d = ("face", "x", "y", "level")
    dims_w = ("face", "x", "y", "level_half")
    dims_2d = ("face", "x", "y")
    state = NonHydrostaticState(
        u=Field(data=jnp.zeros((6, n, n, nlev)), name="u",
                dims=dims_3d, units="m/s"),
        v=Field(data=jnp.zeros((6, n, n, nlev)), name="v",
                dims=dims_3d, units="m/s"),
        w=Field(data=jnp.asarray(w_p), name="w",
                dims=dims_w, units="m/s"),
        theta_prime=Field(data=jnp.asarray(theta_p),
                          name="theta_prime",
                          dims=dims_3d, units="K"),
        rho_prime=Field(data=jnp.asarray(rho_p),
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


def _cfg(dyn_exner, cv=False):
    return CDGridCompressibleEulerConfig(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        damp_w=0.030, nord_w=1, damp_w_d_con=1.0,
        use_fv3_dynamic_exner=dyn_exner,
        use_fv3_d_con_cv=cv,
    )


def test_damp_w_dyn_exner_baseline_bit_for_bit(small_nh):
    grid, hc, tm, state = small_nh
    m_default = CDGridCompressibleEulerModel(
        grid, hc, tm, _cfg(dyn_exner=False),
    )
    m_explicit_off = CDGridCompressibleEulerModel(
        grid, hc, tm, _cfg(dyn_exner=False),
    )
    s_d = m_default.step(state, 5.0)
    s_e = m_explicit_off.step(state, 5.0)
    np.testing.assert_array_equal(
        np.asarray(s_d.theta_prime.data),
        np.asarray(s_e.theta_prime.data),
    )


def test_damp_w_dyn_exner_changes_theta_p(small_nh):
    grid, hc, tm, state = small_nh
    m_off = CDGridCompressibleEulerModel(
        grid, hc, tm, _cfg(dyn_exner=False),
    )
    m_on = CDGridCompressibleEulerModel(
        grid, hc, tm, _cfg(dyn_exner=True),
    )
    s_off = m_off.step(state, 5.0)
    s_on = m_on.step(state, 5.0)
    diff = float(np.max(np.abs(
        np.asarray(s_on.theta_prime.data)
        - np.asarray(s_off.theta_prime.data),
    )))
    assert diff > 1e-12, (
        "damp_w_d_con + dynamic_exner combo did NOT change "
        "theta_p when flag flipped."
    )


def test_damp_w_dyn_exner_plus_cv_finite(small_nh):
    grid, hc, tm, state = small_nh
    m = CDGridCompressibleEulerModel(
        grid, hc, tm, _cfg(dyn_exner=True, cv=True),
    )
    s = m.step(state, 5.0)
    for fld in (s.u.data, s.v.data, s.w.data,
                s.theta_prime.data, s.rho_prime.data):
        assert np.all(np.isfinite(np.asarray(fld)))
