"""FV3_3D iter 603: PE TE boundary-work sign-fix regression test.

Iter 598 had inverted-sign boundary work:
    iter-598 (buggy): te = pe_top·phi_top - pe_sfc·phi_sfc
    iter-603 (fixed): te = pe_sfc·phi_sfc - pe_top·phi_top  (FV3-faithful)

The bug masked itself in iter-598 tests because they only checked
positivity and θ-monotonicity, which are dominated by Σ cp·T·delp.

This test locks in the sign by adding surface elevation (raising
phis above 0): with the CORRECT sign, TE should INCREASE because
phi_sfc grows.

Tests
-----

1. ``test_raising_phis_increases_te`` — locks in fix sign.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
    hydrostatic_to_fv3,
)
from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init
from legoesm.diagnostics import compute_total_energy_pe
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.grids.vertical import standard_hybrid_levels


def test_raising_phis_increases_te():
    """Surface geopotential ↑ → boundary-work term ↑ → TE ↑.

    Confirms FV3-faithful sign: te = pe_sfc·phi_sfc - pe_top·phi_top.
    With iter-598's reversed sign this test would FAIL (Δ < 0).
    """
    n = 8
    nlev = 5
    grid = create_cubed_sphere(n, use_duogrid=True)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    coord = standard_hybrid_levels(nlev)
    state_hs = held_suarez_init(grid, coord)
    state_flat = hydrostatic_to_fv3(state_hs, cdgrid)
    # Raise surface: phis = 1000 m²/s² uniformly
    state_raised = state_flat._replace(
        phis=state_flat.phis.replace(
            data=jnp.full_like(state_flat.phis.data, 1000.0),
        ),
    )
    _, te_flat = compute_total_energy_pe(state_flat, grid, coord)
    _, te_raised = compute_total_energy_pe(state_raised, grid, coord)
    delta = te_raised - te_flat
    # With FV3-faithful sign, both ϕ_sfc and ϕ_top increase by 1000.
    # Δte/area = pe_sfc · 1000 - pe_top · 1000 = 1000 · (pe_sfc - pe_top)
    # which is POSITIVE for any physical atmosphere.
    # iter-598 buggy sign would give Δte ≈ -1000·(pe_sfc - pe_top) < 0.
    assert delta > 0, (
        f"Surface elevation ↑ should INCREASE TE under FV3-faithful sign; "
        f"got Δ={delta:.3e} (iter-598 buggy sign would give Δ<0)"
    )
