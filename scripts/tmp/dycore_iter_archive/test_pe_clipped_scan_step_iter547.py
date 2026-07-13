"""FV3_3D iter 547: PE make_clipped_scan_step validation.

iter-544 added make_clipped_scan_step on NH.  Mirror to PE
to confirm symmetry of the multi-step scan API.

Tests
-----

1. ``test_pe_scan_step_matches_loop`` — PE scan output
   matches Python-loop output bit-for-bit.
2. ``test_pe_scan_step_grad`` — jax.grad works.
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
from legoesm.grids.halo import make_clipped_step, make_clipped_scan_step
from legoesm.grids.vertical import standard_hybrid_levels


def _build_pe_state(n=8, nlev=5):
    grid = create_cubed_sphere(n, use_duogrid=True)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    coord = standard_hybrid_levels(nlev)
    state_hs = held_suarez_init(grid, coord)
    state = hydrostatic_to_fv3(state_hs, cdgrid)
    # Promote all field data to float64 so jax.lax.scan body has
    # stable carry types (iter-544 dtype-match requirement).
    state = state._replace(
        u_d=state.u_d.replace(data=state.u_d.data.astype(jnp.float64)),
        v_d=state.v_d.replace(data=state.v_d.data.astype(jnp.float64)),
        T=state.T.replace(data=state.T.data.astype(jnp.float64)),
        p_s=state.p_s.replace(data=state.p_s.data.astype(jnp.float64)),
        phis=state.phis.replace(data=state.phis.data.astype(jnp.float64)),
    )
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


def test_pe_scan_step_matches_loop():
    grid, coord, state = _build_pe_state()

    # Python loop
    m_loop = _make_pe_model(grid, coord)
    step_loop = make_clipped_step(m_loop, state, dt=10.0, slack=0.5)
    s_loop = state
    for _ in range(5):
        s_loop = step_loop(s_loop, 10.0)

    # scan_step
    m_scan = _make_pe_model(grid, coord)
    scan_step = make_clipped_scan_step(
        m_scan, state, dt=10.0, n_steps=5, slack=0.5,
    )
    s_scan = scan_step(state)

    for fld in ("u_d", "v_d", "T", "p_s"):
        a = getattr(s_loop, fld).data
        b = getattr(s_scan, fld).data
        diff = float(jnp.abs(a - b).max())
        assert diff < 1e-10, (
            f"PE scan_step diverges from loop on {fld}: "
            f"max|diff|={diff}"
        )


def test_pe_scan_step_grad():
    grid, coord, state = _build_pe_state()
    m = _make_pe_model(grid, coord)
    scan_step = make_clipped_scan_step(
        m, state, dt=10.0, n_steps=3, slack=0.5,
    )

    T0 = state.T.data

    def loss_fn(T_init):
        s = state._replace(T=state.T.replace(data=T_init))
        final = scan_step(s)
        return jnp.mean(final.T.data ** 2)

    g = jax.grad(loss_fn)(T0)
    assert jnp.all(jnp.isfinite(g))
    assert float(jnp.linalg.norm(g)) > 1e-12
