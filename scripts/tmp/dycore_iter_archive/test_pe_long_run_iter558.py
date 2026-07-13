"""FV3_3D iter 558: PE long-run stability test.

iter-539 showed NH @ C8 blows up at step 43 (stability limit).
Does PE stay stable longer (PE doesn't have acoustic modes
that cause CFL violations)?

Tests
-----

1. ``test_pe_long_run_stability`` — 100 PE steps at C8,
   Held-Suarez IC, FV3-faithful + scan_step.  Check for NaN.
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


def test_pe_long_run_stability(capsys):
    n = 8
    nlev = 5
    n_steps = 100
    grid = create_cubed_sphere(n, use_duogrid=True)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    coord = standard_hybrid_levels(nlev)
    state_hs = held_suarez_init(grid, coord)
    state = hydrostatic_to_fv3(state_hs, cdgrid)
    # Cast to float64 for scan body type stability
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
        m, state, dt=10.0, n_steps=n_steps, slack=0.5,
    )
    final = scan_step(state)
    # Check finiteness of all fields
    all_finite = True
    for fld_name in ("u_d", "v_d", "T", "p_s", "phis"):
        d = getattr(final, fld_name).data
        if not jnp.all(jnp.isfinite(d)):
            all_finite = False
            break
    with capsys.disabled():
        print(
            f"\n[iter-558 PE {n_steps}-step run @ C8 Held-Suarez "
            f"(FV3-faithful + scan_step)]"
        )
        for fld_name in ("u_d", "v_d", "T", "p_s"):
            d = np.asarray(getattr(final, fld_name).data)
            if np.all(np.isfinite(d)):
                print(
                    f"  {fld_name:6s}: max={d.max():.3e}, "
                    f"min={d.min():.3e}"
                )
            else:
                print(f"  {fld_name:6s}: contains non-finite values")
        print(f"\n  all finite: {all_finite}")
    assert all_finite, "PE 100-step run produced non-finite values"
