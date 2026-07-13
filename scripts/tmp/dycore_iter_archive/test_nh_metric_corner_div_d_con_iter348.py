"""FV3_3D iter 348: NH mirror of iter-347 metric-aware d_con flag
at corner_div_damp_d_con slow-tendency site.

Tests
-----

1. ``test_baseline_bit_for_bit``
2. ``test_metric_changes_theta_p``
3. ``test_metric_finite``
4. ``test_metric_differentiable_at_rest``
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

    rng = np.random.default_rng(seed=348)
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
    return grid, height_coord, terrain_metric, state


def _cfg(metric=False):
    return CDGridCompressibleEulerConfig(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        corner_div_damp_d2_bg=0.0005, corner_div_damp_dddmp=0.20,
        corner_div_damp_d_con=1.0,
        damp_v=0.0,
        use_fv3_metric_aware_d_con=metric,
    )


def test_baseline_bit_for_bit(small_nh):
    grid, hc, tm, state = small_nh
    m_d = CDGridCompressibleEulerModel(grid, hc, tm, _cfg())
    m_e = CDGridCompressibleEulerModel(grid, hc, tm, _cfg(metric=False))
    s_d = m_d.step(state, 5.0)
    s_e = m_e.step(state, 5.0)
    np.testing.assert_array_equal(
        np.asarray(s_d.theta_prime.data),
        np.asarray(s_e.theta_prime.data),
    )


def test_metric_changes_theta_p(small_nh):
    grid, hc, tm, state = small_nh
    m_off = CDGridCompressibleEulerModel(grid, hc, tm, _cfg(metric=False))
    m_on = CDGridCompressibleEulerModel(grid, hc, tm, _cfg(metric=True))
    s_off = m_off.step(state, 5.0)
    s_on = m_on.step(state, 5.0)
    diff = float(np.max(np.abs(
        np.asarray(s_on.theta_prime.data)
        - np.asarray(s_off.theta_prime.data),
    )))
    assert diff > 1e-12


def test_metric_finite(small_nh):
    grid, hc, tm, state = small_nh
    m = CDGridCompressibleEulerModel(grid, hc, tm, _cfg(metric=True))
    s = m.step(state, 5.0)
    for fld in (s.u.data, s.v.data, s.theta_prime.data):
        assert np.all(np.isfinite(np.asarray(fld)))


def test_metric_differentiable_at_rest(small_nh):
    grid, hc, tm, _ = small_nh
    n = 8
    nlev = 5
    dims_3d = ("face", "x", "y", "level")
    dims_w = ("face", "x", "y", "level_half")
    dims_2d = ("face", "x", "y")
    rest = NonHydrostaticState(
        u=Field(data=jnp.zeros((6, n, n, nlev)), name="u",
                dims=dims_3d, units="m/s"),
        v=Field(data=jnp.zeros((6, n, n, nlev)), name="v",
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
    m = CDGridCompressibleEulerModel(grid, hc, tm, _cfg(metric=True))

    def loss(amp):
        s = rest._replace(
            u=rest.u.replace(data=amp * jnp.ones_like(rest.u.data)),
        )
        for _ in range(2):
            s = m.step(s, 5.0)
        return jnp.mean(s.theta_prime.data ** 2)

    g = jax.grad(loss)(0.0)
    assert jnp.isfinite(g)
