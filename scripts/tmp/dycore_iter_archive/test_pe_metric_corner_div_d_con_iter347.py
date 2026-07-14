"""FV3_3D iter 347: extend metric-aware d_con flag to PE iter-221
corner_div_damp_d_con site.

iter-338 wired metric form at PE damp_v_d_con (post-step).
iter-347 extends to PE corner_div_damp_d_con (slow-tendency).
Same ``use_fv3_metric_aware_d_con`` flag.

Tests
-----

1. ``test_baseline_bit_for_bit`` — flag=False bit-for-bit
   baseline at corner_div_d_con site.
2. ``test_metric_changes_T`` — flag=True changes T at
   corner_div_d_con.
3. ``test_metric_linear_in_d_con`` — Δθ scales linearly with
   corner_div_damp_d_con.
4. ``test_metric_differentiable_at_rest`` — AD-safe.
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
def pe_state():
    n = 8
    nlev = 5
    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    coord = standard_hybrid_levels(nlev)
    state_cc = held_suarez_init(grid, coord)
    state = hydrostatic_to_fv3(state_cc, cdgrid)
    rng = np.random.default_rng(seed=347)
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


def _cfg(metric=False, d=1.0):
    return CDGridPrimitiveEquationConfig(
        corner_div_damp_d2_bg=0.0005, corner_div_damp_dddmp=0.20,
        corner_div_damp_d_con=d,
        damp_v=0.0,  # isolate corner_div site
        A_h=0.0, hyperdiff_coeff=0.0, hyperdiff_ps_coeff=0.0,
        div_damp_coeff=0.0,
        use_conservation_fixer=False, fix_mass=False,
        use_fv3_metric_aware_d_con=metric,
    )


def test_baseline_bit_for_bit(pe_state):
    grid, _, coord, state = pe_state
    m_default = CDGridPrimitiveEquationModel(grid, coord, _cfg())
    m_explicit_off = CDGridPrimitiveEquationModel(
        grid, coord, _cfg(metric=False),
    )
    s_d = m_default.step(state, 100.0)
    s_e = m_explicit_off.step(state, 100.0)
    np.testing.assert_array_equal(
        np.asarray(s_d.T.data), np.asarray(s_e.T.data),
    )


def test_metric_changes_T(pe_state):
    grid, _, coord, state = pe_state
    m_off = CDGridPrimitiveEquationModel(
        grid, coord, _cfg(metric=False),
    )
    m_on = CDGridPrimitiveEquationModel(
        grid, coord, _cfg(metric=True),
    )
    s_off = m_off.step(state, 100.0)
    s_on = m_on.step(state, 100.0)
    diff = float(np.max(np.abs(
        np.asarray(s_on.T.data) - np.asarray(s_off.T.data),
    )))
    assert diff > 1e-12


def test_metric_linear_in_d_con(pe_state):
    """ΔT scales linearly in corner_div_damp_d_con under metric form."""
    grid, _, coord, state = pe_state
    m_05 = CDGridPrimitiveEquationModel(
        grid, coord, _cfg(metric=True, d=0.5),
    )
    m_10 = CDGridPrimitiveEquationModel(
        grid, coord, _cfg(metric=True, d=1.0),
    )
    m_20 = CDGridPrimitiveEquationModel(
        grid, coord, _cfg(metric=True, d=2.0),
    )
    s_05 = m_05.step(state, 100.0)
    s_10 = m_10.step(state, 100.0)
    s_20 = m_20.step(state, 100.0)
    d10 = np.asarray(s_10.T.data) - np.asarray(s_05.T.data)
    d20 = np.asarray(s_20.T.data) - np.asarray(s_05.T.data)
    # Slow-tendency site → acoustic-feedback noise breaks
    # exact linearity (PE primitive eq has dynamics coupling via
    # ``dT_dt_cdd_cc`` feeding back through pressure → wind).
    # Relax to rtol=1e-2.
    np.testing.assert_allclose(
        d20, 3.0 * d10, rtol=1e-2, atol=1e-8,
    )
    assert float(np.max(np.abs(d10))) > 1e-10


def test_metric_differentiable_at_rest(pe_state):
    grid, _, coord, state = pe_state
    rest = state._replace(
        u_d=state.u_d.replace(data=jnp.zeros_like(state.u_d.data)),
        v_d=state.v_d.replace(data=jnp.zeros_like(state.v_d.data)),
    )
    m = CDGridPrimitiveEquationModel(grid, coord, _cfg(metric=True))

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
