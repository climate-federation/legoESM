"""FV3_3D iter 225: KE→heat ``d_con`` for the iter-57/58
Smagorinsky-A_h Laplacian on (u_d, v_d) (PE).

Heat tendency formula (per second, leading order in dt):

    dKE/dt_corner = u_d * du_d_dt_ah + v_d * dv_d_dt_ah
    dT/dt += -ah_d_con * (dKE/dt) / c_pd

Tests
-----

1. ``test_ah_d_con_off_baseline`` — default 0.0 bit-for-bit.
2. ``test_ah_d_con_changes_T`` — d_con > 0 changes T when A_h
   active and winds non-zero.
3. ``test_ah_d_con_no_op_when_ah_off`` — gated INSIDE A_h > 0.
4. ``test_ah_d_con_differentiable_at_rest`` — AD-safe.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
    CDGridPrimitiveEquationConfig,
    CDGridPrimitiveEquationModel,
    hydrostatic_to_fv3,
)
from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.grids.vertical import standard_hybrid_levels


@pytest.fixture(scope="module")
def small_pe_state():
    n = 8
    nlev = 6
    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    coord = standard_hybrid_levels(nlev)
    state_cc = held_suarez_init(grid, coord)
    state = hydrostatic_to_fv3(state_cc, cdgrid)

    rng = np.random.default_rng(seed=225)
    n_corners = n + 1
    u_p = rng.uniform(-30.0, 30.0,
                      size=(6, n_corners, n_corners, nlev))
    v_p = rng.uniform(-30.0, 30.0,
                      size=(6, n_corners, n_corners, nlev))
    state = state._replace(
        u_d=state.u_d.replace(data=jnp.asarray(u_p)),
        v_d=state.v_d.replace(data=jnp.asarray(v_p)),
    )
    return grid, cdgrid, coord, state


def _step5(model, state, dt=100.0):
    s = state
    for _ in range(5):
        s = model.step(s, dt)
    return s


def test_ah_d_con_off_baseline(small_pe_state):
    grid, cdgrid, coord, state = small_pe_state
    common = dict(
        A_h=1e6, smagorinsky_cs=0.20,
        hyperdiff_coeff=0.0, hyperdiff_ps_coeff=0.0,
        div_damp_coeff=0.0, damp_v=0.0,
        use_conservation_fixer=False, fix_mass=False,
    )
    cfg_explicit_off = CDGridPrimitiveEquationConfig(
        **common, ah_d_con=0.0,
    )
    cfg_default = CDGridPrimitiveEquationConfig(**common)

    s_explicit = _step5(CDGridPrimitiveEquationModel(
        grid, coord, cfg_explicit_off,
    ), state)
    s_default = _step5(CDGridPrimitiveEquationModel(
        grid, coord, cfg_default,
    ), state)

    np.testing.assert_array_equal(s_explicit.T.data, s_default.T.data)


def test_ah_d_con_changes_T(small_pe_state):
    grid, cdgrid, coord, state = small_pe_state
    common = dict(
        A_h=1e6, smagorinsky_cs=0.20,
        hyperdiff_coeff=0.0, hyperdiff_ps_coeff=0.0,
        div_damp_coeff=0.0, damp_v=0.0,
        use_conservation_fixer=False, fix_mass=False,
    )
    cfg_off = CDGridPrimitiveEquationConfig(**common, ah_d_con=0.0)
    cfg_on = CDGridPrimitiveEquationConfig(**common, ah_d_con=1.0)

    s_off = _step5(CDGridPrimitiveEquationModel(
        grid, coord, cfg_off,
    ), state)
    s_on = _step5(CDGridPrimitiveEquationModel(
        grid, coord, cfg_on,
    ), state)

    diff = float(jnp.max(jnp.abs(s_on.T.data - s_off.T.data)))
    assert diff > 1e-8, (
        f"ah_d_con=1.0 must change T relative to d_con=0; got "
        f"max|ΔT|={diff:.4e}."
    )


def test_ah_d_con_no_op_when_ah_off(small_pe_state):
    grid, cdgrid, coord, state = small_pe_state
    common = dict(
        A_h=0.0,    # OFF
        smagorinsky_cs=0.0,
        hyperdiff_coeff=0.0, hyperdiff_ps_coeff=0.0,
        div_damp_coeff=0.0, damp_v=0.0,
        use_conservation_fixer=False, fix_mass=False,
    )
    cfg_a = CDGridPrimitiveEquationConfig(**common, ah_d_con=0.0)
    cfg_b = CDGridPrimitiveEquationConfig(**common, ah_d_con=1.0)
    s_a = CDGridPrimitiveEquationModel(grid, coord, cfg_a).step(
        state, 100.0,
    )
    s_b = CDGridPrimitiveEquationModel(grid, coord, cfg_b).step(
        state, 100.0,
    )
    np.testing.assert_array_equal(s_a.T.data, s_b.T.data)


def test_ah_d_con_differentiable_at_rest():
    n = 8
    nlev = 6
    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    coord = standard_hybrid_levels(nlev)
    state_cc = held_suarez_init(grid, coord)
    state_rest = hydrostatic_to_fv3(state_cc, cdgrid)
    state_rest = state_rest._replace(
        u_d=state_rest.u_d.replace(
            data=jnp.zeros_like(state_rest.u_d.data),
        ),
        v_d=state_rest.v_d.replace(
            data=jnp.zeros_like(state_rest.v_d.data),
        ),
    )

    cfg = CDGridPrimitiveEquationConfig(
        A_h=1e6, smagorinsky_cs=0.20, ah_d_con=1.0,
        hyperdiff_coeff=0.0, hyperdiff_ps_coeff=0.0,
        div_damp_coeff=0.0, damp_v=0.0,
        use_conservation_fixer=False, fix_mass=False,
    )
    model = CDGridPrimitiveEquationModel(grid, coord, cfg)

    def loss(eps):
        u_d_pert = state_rest.u_d.data + eps * jnp.ones_like(
            state_rest.u_d.data,
        )
        s = state_rest._replace(
            u_d=state_rest.u_d.replace(data=u_d_pert),
        )
        s_new = model.step(s, 100.0)
        return jnp.sum(s_new.T.data ** 2)

    g = jax.grad(loss)(0.0)
    assert jnp.isfinite(g), (
        f"PE ah_d_con must be AD-safe at rest; got grad={g}"
    )
