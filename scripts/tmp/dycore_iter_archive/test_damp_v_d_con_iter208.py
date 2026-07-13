"""FV3_3D iter 208: KE→heat conversion (``d_con``) for the iter-12
PE ``damp_v`` post-step damping.

Faithful (simplified) port of FV3 ``sw_core.F90:1953-1990`` d_con
block, restricted to the damp_v contribution alone.  When
``damp_v`` removes KE from (u_d, v_d) via the post-step
(du_corner, dv_corner) wind increments, the lost KE is converted
to heat in T (energy conservation):

    ΔKE_per_mass = u_d * du + 0.5*du² + v_d * dv + 0.5*dv²
    ΔT = -damp_v_d_con * ΔKE / c_pd

Computed at corners then projected to cell centres via the
4-point ``interp_corner_to_center`` helper for the T (cell-
centre) update.

Tests
-----

1. ``test_damp_v_d_con_off_baseline`` — ``damp_v_d_con=0.0`` (and
   field unset) is bit-for-bit baseline (Python-static gate).
2. ``test_damp_v_d_con_increases_T_when_wind_damped`` — when
   ``damp_v > 0`` removes KE from a perturbed (u_d, v_d), the
   T field gains heat (positive δT where ΔKE > 0).
3. ``test_damp_v_d_con_differentiable_at_rest`` — ``jax.grad``
   through 3 PE steps with ``damp_v_d_con > 0`` at rest stays
   finite.
4. ``test_damp_v_d_con_no_op_when_damp_v_off`` — even with
   ``damp_v_d_con > 0``, when ``damp_v == 0`` the heat source is
   gated off (the d_con block is INSIDE the ``damp_v > 0`` block).
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
from legoesm.atmosphere.held_suarez import held_suarez_init
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
    return grid, cdgrid, coord, state


def test_damp_v_d_con_off_baseline(small_pe_state):
    """damp_v_d_con=0.0 is bit-for-bit baseline (gated off)."""
    grid, cdgrid, coord, state = small_pe_state

    n = grid.n
    nlev = state.u_d.data.shape[-1]
    rng = np.random.default_rng(seed=208)
    u_p = rng.uniform(-3.0, 3.0, size=(6, n + 1, n + 1, nlev))
    v_p = rng.uniform(-3.0, 3.0, size=(6, n + 1, n + 1, nlev))
    s = state._replace(
        u_d=state.u_d.replace(data=jnp.asarray(u_p)),
        v_d=state.v_d.replace(data=jnp.asarray(v_p)),
    )

    cfg_baseline = CDGridPrimitiveEquationConfig(
        damp_v=0.030, nord_v=1,
        # damp_v_d_con=0.0 default — no heat
        use_conservation_fixer=False, fix_mass=False,
    )
    cfg_d_con_zero = CDGridPrimitiveEquationConfig(
        damp_v=0.030, nord_v=1, damp_v_d_con=0.0,
        use_conservation_fixer=False, fix_mass=False,
    )

    m_baseline = CDGridPrimitiveEquationModel(grid, coord, cfg_baseline)
    m_d_con_zero = CDGridPrimitiveEquationModel(grid, coord, cfg_d_con_zero)

    s_baseline = m_baseline.step(s, 100.0)
    s_d_con_zero = m_d_con_zero.step(s, 100.0)

    np.testing.assert_array_equal(
        np.asarray(s_baseline.T.data),
        np.asarray(s_d_con_zero.T.data),
    )


def test_damp_v_d_con_increases_T_when_wind_damped(small_pe_state):
    """With damp_v_d_con > 0, T should differ from the no-d_con
    baseline (heat from KE removal added)."""
    grid, cdgrid, coord, state = small_pe_state

    n = grid.n
    nlev = state.u_d.data.shape[-1]
    rng = np.random.default_rng(seed=208)
    u_p = rng.uniform(-3.0, 3.0, size=(6, n + 1, n + 1, nlev))
    v_p = rng.uniform(-3.0, 3.0, size=(6, n + 1, n + 1, nlev))
    s = state._replace(
        u_d=state.u_d.replace(data=jnp.asarray(u_p)),
        v_d=state.v_d.replace(data=jnp.asarray(v_p)),
    )

    cfg_no_d_con = CDGridPrimitiveEquationConfig(
        damp_v=0.030, nord_v=1, damp_v_d_con=0.0,
        use_conservation_fixer=False, fix_mass=False,
    )
    cfg_d_con = CDGridPrimitiveEquationConfig(
        damp_v=0.030, nord_v=1, damp_v_d_con=1.0,    # FV3 default
        use_conservation_fixer=False, fix_mass=False,
    )

    m_no = CDGridPrimitiveEquationModel(grid, coord, cfg_no_d_con)
    m_d = CDGridPrimitiveEquationModel(grid, coord, cfg_d_con)

    s_no = m_no.step(s, 100.0)
    s_d = m_d.step(s, 100.0)

    diff = float(jnp.max(jnp.abs(s_no.T.data - s_d.T.data)))
    assert diff > 1e-10, (
        f"damp_v_d_con > 0 must change T (diff={diff:.3e})"
    )

    # damp_v removes KE on average; mean T should rise (or at
    # least not fall) under d_con > 0.  (The change is small over
    # one step since damp_v is mild, but the sign is robust.)
    mean_no = float(jnp.mean(s_no.T.data))
    mean_d = float(jnp.mean(s_d.T.data))
    assert mean_d + 1e-6 >= mean_no, (
        f"damp_v_d_con > 0 must not REDUCE mean T (heat added "
        f"on average): mean(no_d_con)={mean_no:.4e}, "
        f"mean(d_con)={mean_d:.4e} (sign error in iter-208 wiring)."
    )


def test_damp_v_d_con_differentiable_at_rest(small_pe_state):
    """``jax.grad`` through 3 PE steps with damp_v + damp_v_d_con
    active at rest stays finite."""
    grid, cdgrid, coord, state = small_pe_state

    rest = state._replace(
        u_d=state.u_d.replace(data=jnp.zeros_like(state.u_d.data)),
        v_d=state.v_d.replace(data=jnp.zeros_like(state.v_d.data)),
    )

    cfg = CDGridPrimitiveEquationConfig(
        damp_v=0.030, nord_v=1, damp_v_d_con=1.0,
        use_conservation_fixer=False, fix_mass=False,
    )
    model = CDGridPrimitiveEquationModel(grid, coord, cfg)

    def loss_fn(T_data):
        s = rest._replace(T=rest.T.replace(data=T_data))
        for _ in range(3):
            s = model.step(s, 100.0)
        return jnp.mean(s.T.data ** 2)

    grad = jax.grad(loss_fn)(rest.T.data)
    assert jnp.all(jnp.isfinite(grad))


def test_damp_v_d_con_no_op_when_damp_v_off(small_pe_state):
    """When damp_v == 0, damp_v_d_con must NOT add heat (gated
    INSIDE damp_v block)."""
    grid, cdgrid, coord, state = small_pe_state

    cfg_unset = CDGridPrimitiveEquationConfig(
        use_conservation_fixer=False, fix_mass=False,
    )
    cfg_d_con_only = CDGridPrimitiveEquationConfig(
        damp_v=0.0, damp_v_d_con=1.0,    # d_con set but damp_v off
        use_conservation_fixer=False, fix_mass=False,
    )

    m_unset = CDGridPrimitiveEquationModel(grid, coord, cfg_unset)
    m_d_con_only = CDGridPrimitiveEquationModel(grid, coord, cfg_d_con_only)

    s_unset = m_unset.step(state, 100.0)
    s_d_con_only = m_d_con_only.step(state, 100.0)

    np.testing.assert_array_equal(
        np.asarray(s_unset.T.data),
        np.asarray(s_d_con_only.T.data),
    )
