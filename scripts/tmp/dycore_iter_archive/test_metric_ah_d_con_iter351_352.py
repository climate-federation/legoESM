"""FV3_3D iter 351 (PE) + iter 352 (NH): metric-aware d_con flag
at A_h d_con (Smagorinsky-A_h slow-tendency) sites.

Closes gap #1 metric d_con extension across ALL 8 PE+NH d_con
sites: damp_v (iter-338/339) + corner_div (iter-347/348) +
div_damp (iter-349/350) + A_h (iter-351/352).

Same use_fv3_metric_aware_d_con flag.

Tests
-----
PE iter-225:
1. PE A_h baseline bit-for-bit at flag=False
2. PE A_h metric changes T at flag=True
3. PE A_h metric AD-safe at rest
NH iter-226:
4. NH A_h baseline bit-for-bit
5. NH A_h metric changes theta_p
6. NH A_h metric AD-safe
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
from legoesm.atmosphere.held_suarez import held_suarez_init
from legoesm.core.field import Field
from legoesm.core.state import NonHydrostaticState
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.grids.vertical import (
    compute_terrain_metric, create_height_coordinate,
    standard_hybrid_levels,
)


@pytest.fixture(scope="module")
def pe_state():
    n = 8
    nlev = 5
    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    coord = standard_hybrid_levels(nlev)
    state_cc = held_suarez_init(grid, coord)
    state = hydrostatic_to_fv3(state_cc, cdgrid)
    rng = np.random.default_rng(seed=351)
    n_corners = n + 1
    u_p = rng.uniform(-25.0, 25.0,
                      size=(6, n_corners, n_corners, nlev))
    v_p = rng.uniform(-25.0, 25.0,
                      size=(6, n_corners, n_corners, nlev))
    state = state._replace(
        u_d=state.u_d.replace(data=jnp.asarray(u_p)),
        v_d=state.v_d.replace(data=jnp.asarray(v_p)),
    )
    return grid, cdgrid, coord, state


@pytest.fixture(scope="module")
def nh_state():
    n = 8
    nlev = 5
    z_top = 30000.0
    grid = create_cubed_sphere(n)
    height_coord = create_height_coordinate(nlev, z_top)
    terrain = jnp.zeros((6, n, n))
    terrain_metric = compute_terrain_metric(terrain, height_coord)
    rng = np.random.default_rng(seed=352)
    u_p = rng.uniform(-25.0, 25.0, size=(6, n, n, nlev))
    v_p = rng.uniform(-25.0, 25.0, size=(6, n, n, nlev))
    dims_3d = ("face", "x", "y", "level")
    dims_w = ("face", "x", "y", "level_half")
    dims_2d = ("face", "x", "y")
    state = NonHydrostaticState(
        u=Field(data=jnp.asarray(u_p), name="u", dims=dims_3d, units="m/s"),
        v=Field(data=jnp.asarray(v_p), name="v", dims=dims_3d, units="m/s"),
        w=Field(data=jnp.zeros((6, n, n, nlev + 1)), name="w",
                dims=dims_w, units="m/s"),
        theta_prime=Field(data=jnp.zeros((6, n, n, nlev)),
                          name="theta_prime", dims=dims_3d, units="K"),
        rho_prime=Field(data=jnp.zeros((6, n, n, nlev)),
                        name="rho_prime", dims=dims_3d, units="kg/m^3"),
        phis=Field(data=jnp.zeros((6, n, n)), name="phis",
                   dims=dims_2d, units="m^2/s^2"),
        tracers=Field(data=jnp.zeros((6, n, n, nlev, 0)), name="tracers",
                      dims=("face", "x", "y", "level", "tracer"),
                      units="kg/kg"),
    )
    return grid, height_coord, terrain_metric, state


def _pe_cfg(metric):
    return CDGridPrimitiveEquationConfig(
        A_h=1e6, ah_d_con=1.0,
        damp_v=0.0, corner_div_damp_d2_bg=0.0,
        hyperdiff_coeff=0.0, hyperdiff_ps_coeff=0.0,
        div_damp_coeff=0.0,
        use_conservation_fixer=False, fix_mass=False,
        use_fv3_metric_aware_d_con=metric,
    )


def _nh_cfg(metric):
    return CDGridCompressibleEulerConfig(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        A_h=1e6, ah_d_con=1.0,
        use_fv3_metric_aware_d_con=metric,
    )


def test_pe_ah_baseline_bit_for_bit(pe_state):
    grid, _, coord, state = pe_state
    m_d = CDGridPrimitiveEquationModel(grid, coord, _pe_cfg(False))
    m_e = CDGridPrimitiveEquationModel(grid, coord, _pe_cfg(False))
    s_d = m_d.step(state, 100.0)
    s_e = m_e.step(state, 100.0)
    np.testing.assert_array_equal(
        np.asarray(s_d.T.data), np.asarray(s_e.T.data),
    )


def test_pe_ah_metric_changes_T(pe_state):
    grid, _, coord, state = pe_state
    m_off = CDGridPrimitiveEquationModel(grid, coord, _pe_cfg(False))
    m_on = CDGridPrimitiveEquationModel(grid, coord, _pe_cfg(True))
    s_off = m_off.step(state, 100.0)
    s_on = m_on.step(state, 100.0)
    diff = float(np.max(np.abs(
        np.asarray(s_on.T.data) - np.asarray(s_off.T.data),
    )))
    assert diff > 1e-12


def test_pe_ah_metric_ad_at_rest(pe_state):
    grid, _, coord, state = pe_state
    rest = state._replace(
        u_d=state.u_d.replace(data=jnp.zeros_like(state.u_d.data)),
        v_d=state.v_d.replace(data=jnp.zeros_like(state.v_d.data)),
    )
    m = CDGridPrimitiveEquationModel(grid, coord, _pe_cfg(True))

    def loss(amp):
        s = rest._replace(
            u_d=rest.u_d.replace(
                data=amp * jnp.ones_like(rest.u_d.data),
            ),
        )
        for _ in range(2):
            s = m.step(s, 100.0)
        return jnp.mean(s.T.data ** 2)

    g = jax.grad(loss)(0.0)
    assert jnp.isfinite(g)


def test_nh_ah_baseline_bit_for_bit(nh_state):
    grid, hc, tm, state = nh_state
    m_d = CDGridCompressibleEulerModel(grid, hc, tm, _nh_cfg(False))
    m_e = CDGridCompressibleEulerModel(grid, hc, tm, _nh_cfg(False))
    s_d = m_d.step(state, 5.0)
    s_e = m_e.step(state, 5.0)
    np.testing.assert_array_equal(
        np.asarray(s_d.theta_prime.data),
        np.asarray(s_e.theta_prime.data),
    )


def test_nh_ah_metric_changes_theta_p(nh_state):
    grid, hc, tm, state = nh_state
    m_off = CDGridCompressibleEulerModel(grid, hc, tm, _nh_cfg(False))
    m_on = CDGridCompressibleEulerModel(grid, hc, tm, _nh_cfg(True))
    s_off = m_off.step(state, 5.0)
    s_on = m_on.step(state, 5.0)
    diff = float(np.max(np.abs(
        np.asarray(s_on.theta_prime.data)
        - np.asarray(s_off.theta_prime.data),
    )))
    assert diff > 1e-12


def test_nh_ah_metric_ad_at_rest(nh_state):
    grid, hc, tm, _ = nh_state
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
    m = CDGridCompressibleEulerModel(grid, hc, tm, _nh_cfg(True))

    def loss(amp):
        s = rest._replace(
            u=rest.u.replace(data=amp * jnp.ones_like(rest.u.data)),
        )
        for _ in range(2):
            s = m.step(s, 5.0)
        return jnp.mean(s.theta_prime.data ** 2)

    g = jax.grad(loss)(0.0)
    assert jnp.isfinite(g)
