"""FV3_3D iter 524: PE mass conservation under clip context.

Mirror of iter-523 on the hydrostatic PE dycore.  PE
prognostic is p_s (surface pressure) and T; total dry mass
∝ sum(p_s · area).  This iter verifies sum(p_s · area) is
conserved over 10 PE steps under min-edge + clip context.

Tests
-----

1. ``test_pe_total_mass_drift_iter524`` — measure
   sum(p_s · area) at step 0, 5, 10.  Report relative drift.
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
from legoesm.grids.halo import monotone_halo_clip_context
from legoesm.grids.vertical import standard_hybrid_levels


def _total_dry_mass(state, grid):
    """sum(p_s * area) — proportional to total dry mass."""
    p_s = np.asarray(state.p_s.data)        # (6, n, n)
    area = np.asarray(grid.area)             # (6, n, n)
    return float(np.sum(p_s * area))


def test_pe_total_mass_drift_iter524(capsys):
    n = 8
    nlev = 5
    grid = create_cubed_sphere(n, use_duogrid=True)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    coord = standard_hybrid_levels(nlev)
    state_hs = held_suarez_init(grid, coord)
    state = hydrostatic_to_fv3(state_hs, cdgrid)
    kw = dict(
        damp_v=0.030, damp_v_d_con=1.0,
        corner_div_damp_d2_bg=5e-4,
        corner_div_damp_d_con=1.0,
        div_damp_coeff=1e6, div_damp_d_con=1.0,
    )
    cfg = make_legoesm_pe_min_edge_config(**kw)
    mass_history = [(0, _total_dry_mass(state, grid))]
    with monotone_halo_clip_context(slack=0.5):
        m = CDGridPrimitiveEquationModel(grid, coord, cfg)
        s = state
        for step in range(1, 11):
            s = m.step(s, dt=10.0)
            if step in (5, 10):
                mass_history.append((step, _total_dry_mass(s, grid)))
    m0 = mass_history[0][1]
    with capsys.disabled():
        print(
            f"\n[iter-524 PE mass conservation @ C8 + duogrid, "
            f"min-edge + clip, 10 steps]"
        )
        print(f"  {'step':>5s}: {'sum(p_s*area)':>16s}  {'rel drift':>10s}")
        for step, mass in mass_history:
            rel = (mass - m0) / m0 if m0 != 0 else float("nan")
            print(f"  {step:5d}: {mass:16.6e}  {rel:+10.2e}")
        max_rel = max(
            abs((m - m0) / m0) if m0 != 0 else 0
            for _, m in mass_history
        )
        print(f"\n  max |rel drift|: {max_rel:.2e}")
        if max_rel < 1e-10:
            print(f"  → mass conserved to machine precision")
        elif max_rel < 1e-6:
            print(f"  → mass nearly conserved (1ppm)")
        else:
            print(f"  → mass drifts (>1ppm)")
    max_rel = max(
        abs((m - m0) / m0) if m0 != 0 else 0
        for _, m in mass_history
    )
    assert max_rel < 1e-2, (
        f"PE mass should not drift by more than 1%: got {max_rel:.2e}"
    )
