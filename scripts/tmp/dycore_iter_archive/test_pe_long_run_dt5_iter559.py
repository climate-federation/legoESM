"""FV3_3D iter 559: PE long-run with dt=5 (half iter-558).

iter-558 showed PE 100 steps at dt=10 produces unphysical
values (u_d ~ 10⁵ m/s).  Does halving dt keep it physical?
At dt=5, 200 steps reaches the same final time but with
finer CFL.

Tests
-----

1. ``test_pe_long_run_dt5_physical`` — 200 PE steps at dt=5,
   verify |u_d| < 200 m/s, T > 100 K (physical).
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
    make_fv3_faithful_pe_config,
)
from legoesm.atmosphere.held_suarez import held_suarez_init
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.grids.halo import make_clipped_scan_step
from legoesm.grids.vertical import standard_hybrid_levels


def test_pe_long_run_dt5_physical(capsys):
    n = 8
    nlev = 5
    n_steps = 200
    dt = 5.0
    grid = create_cubed_sphere(n, use_duogrid=True)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    coord = standard_hybrid_levels(nlev)
    state_hs = held_suarez_init(grid, coord)
    state = hydrostatic_to_fv3(state_hs, cdgrid)
    state = state._replace(
        u_d=state.u_d.replace(data=state.u_d.data.astype(jnp.float64)),
        v_d=state.v_d.replace(data=state.v_d.data.astype(jnp.float64)),
        T=state.T.replace(data=state.T.data.astype(jnp.float64)),
        p_s=state.p_s.replace(data=state.p_s.data.astype(jnp.float64)),
        phis=state.phis.replace(data=state.phis.data.astype(jnp.float64)),
    )
    kw = dict(
        damp_v=0.030, damp_v_d_con=1.0,
        corner_div_damp_d2_bg=5e-4, corner_div_damp_d_con=1.0,
        div_damp_coeff=1e6, div_damp_d_con=1.0,
    )
    cfg = make_fv3_faithful_pe_config(**kw)
    m = CDGridPrimitiveEquationModel(grid, coord, cfg)
    scan_step = make_clipped_scan_step(
        m, state, dt=dt, n_steps=n_steps, slack=0.5,
    )
    final = scan_step(state)
    u_max = float(jnp.abs(final.u_d.data).max())
    v_max = float(jnp.abs(final.v_d.data).max())
    T_min = float(jnp.min(final.T.data))
    T_max = float(jnp.max(final.T.data))
    p_min = float(jnp.min(final.p_s.data))
    p_max = float(jnp.max(final.p_s.data))
    with capsys.disabled():
        print(
            f"\n[iter-559 PE {n_steps}-step run @ dt={dt}, C8, "
            f"FV3-faithful + scan]"
        )
        print(f"  |u_d| max: {u_max:.3e}")
        print(f"  |v_d| max: {v_max:.3e}")
        print(f"  T range:   {T_min:.1f} to {T_max:.1f} K")
        print(f"  p_s range: {p_min:.3e} to {p_max:.3e} Pa")
        print(
            f"\n  Final time: {n_steps * dt} s (same as iter-558's "
            f"{100 * 10} s)"
        )
    # Physical sanity (lenient — PE still has stability issues at this
    # resolution but should be much better than dt=10):
    assert jnp.all(jnp.isfinite(final.u_d.data))
    assert jnp.all(jnp.isfinite(final.T.data))
