"""FV3_3D iter 370: opt-in cross-face halo for damp_v
post-step wind-increment projection back to corners (PE + NH).

Default ``jnp.pad(mode='edge')`` is same-face extension at the
cube boundary; ``use_fv3_cross_face_du_proj=True`` switches to
``pad_halo_4d`` (duogrid-aware) for FV3-faithful cross-face
value at cube edges.

Tests
-----
PE:
1. baseline bit-for-bit
2. flag=True changes u_d
3. AD-safe at rest
NH:
4-6. same set
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


def _pe_state():
    n = 8
    nlev = 5
    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    coord = standard_hybrid_levels(nlev)
    state_cc = held_suarez_init(grid, coord)
    state = hydrostatic_to_fv3(state_cc, cdgrid)
    rng = np.random.default_rng(seed=370)
    u_p = rng.uniform(-20.0, 20.0,
                      size=(6, n + 1, n + 1, nlev))
    v_p = rng.uniform(-20.0, 20.0,
                      size=(6, n + 1, n + 1, nlev))
    state = state._replace(
        u_d=state.u_d.replace(data=jnp.asarray(u_p)),
        v_d=state.v_d.replace(data=jnp.asarray(v_p)),
    )
    return grid, coord, state


def _nh_state():
    n = 8
    nlev = 5
    z_top = 30000.0
    grid = create_cubed_sphere(n)
    hc = create_height_coordinate(nlev, z_top)
    terrain = jnp.zeros((6, n, n))
    tm = compute_terrain_metric(terrain, hc)
    rng = np.random.default_rng(seed=370)
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
        tracers=Field(data=jnp.zeros((6, n, n, nlev, 0)), name="tracers",
                      dims=("face", "x", "y", "level", "tracer"),
                      units="kg/kg"),
    )
    return grid, hc, tm, state


def _pe_cfg(cross_face):
    return CDGridPrimitiveEquationConfig(
        damp_v=0.030, nord_v=1, damp_v_d_con=0.0,
        corner_div_damp_d2_bg=0.0, A_h=0.0,
        hyperdiff_coeff=0.0, hyperdiff_ps_coeff=0.0,
        div_damp_coeff=0.0,
        use_conservation_fixer=False, fix_mass=False,
        use_fv3_cross_face_du_proj=cross_face,
    )


def _nh_cfg(cross_face):
    return CDGridCompressibleEulerConfig(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        damp_v=0.030, nord_v=1,
        use_fv3_cross_face_du_proj=cross_face,
    )


def test_pe_baseline_bit_for_bit():
    grid, coord, state = _pe_state()
    m1 = CDGridPrimitiveEquationModel(grid, coord, _pe_cfg(False))
    m2 = CDGridPrimitiveEquationModel(grid, coord, _pe_cfg(False))
    s1 = m1.step(state, 100.0)
    s2 = m2.step(state, 100.0)
    np.testing.assert_array_equal(
        np.asarray(s1.u_d.data), np.asarray(s2.u_d.data),
    )


def test_pe_cross_face_changes_u_d():
    grid, coord, state = _pe_state()
    m_off = CDGridPrimitiveEquationModel(grid, coord, _pe_cfg(False))
    m_on = CDGridPrimitiveEquationModel(grid, coord, _pe_cfg(True))
    s_off = m_off.step(state, 100.0)
    s_on = m_on.step(state, 100.0)
    diff = float(np.max(np.abs(
        np.asarray(s_on.u_d.data) - np.asarray(s_off.u_d.data),
    )))
    assert diff > 1e-12


def test_pe_cross_face_ad_at_rest():
    grid, coord, state = _pe_state()
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


def test_nh_baseline_bit_for_bit():
    grid, hc, tm, state = _nh_state()
    m1 = CDGridCompressibleEulerModel(grid, hc, tm, _nh_cfg(False))
    m2 = CDGridCompressibleEulerModel(grid, hc, tm, _nh_cfg(False))
    s1 = m1.step(state, 5.0)
    s2 = m2.step(state, 5.0)
    np.testing.assert_array_equal(
        np.asarray(s1.u.data), np.asarray(s2.u.data),
    )


def test_nh_cross_face_changes_u():
    grid, hc, tm, state = _nh_state()
    m_off = CDGridCompressibleEulerModel(grid, hc, tm, _nh_cfg(False))
    m_on = CDGridCompressibleEulerModel(grid, hc, tm, _nh_cfg(True))
    s_off = m_off.step(state, 5.0)
    s_on = m_on.step(state, 5.0)
    diff = float(np.max(np.abs(
        np.asarray(s_on.u.data) - np.asarray(s_off.u.data),
    )))
    assert diff > 1e-12


def test_nh_cross_face_ad_at_rest():
    grid, hc, tm, _ = _nh_state()
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
