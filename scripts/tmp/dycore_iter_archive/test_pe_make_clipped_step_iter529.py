"""FV3_3D iter 529: ``make_clipped_step`` on PE — symmetry +
mass + AD validation.

iter-526 validated the helper on NH.  iter-529 mirrors on PE
to confirm full symmetry of the iter-505/526 helper stack
across both dycores.

Tests
-----

1. ``test_pe_make_clipped_step_runs`` — basic invocation.
2. ``test_pe_make_clipped_step_conserves_mass`` — sum(p_s ·
   area) preserved to ~1e-6 over 10 steps.
3. ``test_pe_make_clipped_step_differentiable`` — jax.grad
   flows through.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
    CDGridPrimitiveEquationModel,
    hydrostatic_to_fv3,
    make_legoesm_pe_min_edge_config,
)
from legoesm.atmosphere.held_suarez import held_suarez_init
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.grids.halo import make_clipped_step
from legoesm.grids.vertical import standard_hybrid_levels


def _build_pe_state(n=8, nlev=5):
    grid = create_cubed_sphere(n, use_duogrid=True)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    coord = standard_hybrid_levels(nlev)
    state_hs = held_suarez_init(grid, coord)
    state = hydrostatic_to_fv3(state_hs, cdgrid)
    return grid, coord, state


def _make_pe_model(grid, coord):
    kw = dict(
        damp_v=0.030, damp_v_d_con=1.0,
        corner_div_damp_d2_bg=5e-4,
        corner_div_damp_d_con=1.0,
        div_damp_coeff=1e6, div_damp_d_con=1.0,
    )
    cfg = make_legoesm_pe_min_edge_config(**kw)
    return CDGridPrimitiveEquationModel(grid, coord, cfg)


def test_pe_make_clipped_step_runs():
    grid, coord, state = _build_pe_state()
    m = _make_pe_model(grid, coord)
    step = make_clipped_step(m, state, dt=10.0, slack=0.5)
    new_state = step(state, 10.0)
    assert new_state.T.data.shape == state.T.data.shape
    assert jnp.all(jnp.isfinite(new_state.T.data))
    assert jnp.all(jnp.isfinite(new_state.p_s.data))


def test_pe_make_clipped_step_conserves_mass(capsys):
    grid, coord, state = _build_pe_state()
    m = _make_pe_model(grid, coord)
    step = make_clipped_step(m, state, dt=10.0, slack=0.5)

    def _total_mass(s):
        return float(np.sum(
            np.asarray(s.p_s.data) * np.asarray(grid.area)
        ))

    m0 = _total_mass(state)
    s = state
    for _ in range(10):
        s = step(s, 10.0)
    m10 = _total_mass(s)
    rel = abs((m10 - m0) / m0)
    with capsys.disabled():
        print(
            f"\n[iter-529 PE make_clipped_step mass: "
            f"rel drift over 10 steps = {rel:.2e}]"
        )
    assert rel < 1e-6, (
        f"PE mass should be conserved under make_clipped_step: "
        f"rel drift = {rel:.2e}"
    )


def test_pe_make_clipped_step_differentiable():
    grid, coord, state = _build_pe_state()
    m = _make_pe_model(grid, coord)
    step = make_clipped_step(m, state, dt=10.0, slack=0.5)

    T0 = state.T.data

    def loss_fn(T_init):
        s = state._replace(T=state.T.replace(data=T_init))
        new_state = step(s, 10.0)
        return jnp.mean(new_state.T.data ** 2)

    grad_T = jax.grad(loss_fn)(T0)
    assert jnp.all(jnp.isfinite(grad_T))
    assert float(jnp.linalg.norm(grad_T)) > 1e-12
