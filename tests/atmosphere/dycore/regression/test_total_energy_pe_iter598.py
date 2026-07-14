"""FV3_3D iter 598: PE hydrostatic total-energy diagnostic test.

Tests
-----

1. ``test_te_pe_held_suarez_finite_positive``.
2. ``test_te_pe_increases_with_temperature``.
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
from legoesm.atmosphere.held_suarez import held_suarez_init
from legoesm.diagnostics import compute_total_energy_pe
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.grids.vertical import standard_hybrid_levels


def _build_pe_state(n=8, nlev=5):
    grid = create_cubed_sphere(n, use_duogrid=True)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    coord = standard_hybrid_levels(nlev)
    state_hs = held_suarez_init(grid, coord)
    state = hydrostatic_to_fv3(state_hs, cdgrid)
    return grid, coord, state


def test_te_pe_held_suarez_finite_positive():
    grid, coord, state = _build_pe_state()
    col, total = compute_total_energy_pe(state, grid, coord)
    assert col.shape == (6, 8, 8)
    assert np.isfinite(total)
    assert total > 0, (
        f"TE for Held-Suarez should be positive (cp·T·delp dominates); "
        f"got {total}"
    )


def test_te_pe_increases_with_temperature():
    grid, coord, state_cold = _build_pe_state()
    # Build warmer state: T + 10K
    state_warm = state_cold._replace(
        T=state_cold.T.replace(data=state_cold.T.data + 10.0),
    )
    _, te_cold = compute_total_energy_pe(state_cold, grid, coord)
    _, te_warm = compute_total_energy_pe(state_warm, grid, coord)
    delta = te_warm - te_cold
    # ΔTE_ideal = cp · ΔT · total_mass.  total mass = sum(delp·area)/g
    # ≈ (ps·area_earth)/g.  ΔT=10, so ΔTE > 0 strongly.
    assert delta > 0, f"+10K T should increase TE; got Δ={delta:.3e}"
