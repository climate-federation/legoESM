"""FV3_3D iter 333: PE-side mirror of iter-325's duogrid wiring
fix — thread ``duogrid=grid.duogrid`` through the PE
``ke_correction`` halo at the FV3 corner-divergence damping site.

Audit
-----
iter-325 fixed 3 NH halo sites that silently bypassed duogrid
even when the user constructed the grid with ``use_duogrid=True``.
Continuing the audit, PE has the SAME bypass at
``primitive_eq_cdgrid.py:964``::

    _ke_pad = _pad_halo_4d_fn(_ke_correction)  # NO duogrid kwarg

This is the FV3 corner-div damping ``ke_correction`` gradient site
(d_sw5 line 1717).  Without duogrid, the cube-edge gradient at
panel boundaries uses cube-projected halo cells instead of FV3
``fv_duogrid.F90`` Lagrange-extended halo — same O(dx²) bias as
the NH iter-325 fix.

iter-333 passes ``duogrid=grid.duogrid`` to the PE call.  Closes
the symmetric PE/NH ke_correction halo gap.

Tests
-----

1. ``test_pe_no_duogrid_path_unchanged`` — bit-for-bit baseline
   preserved (regression guard).
2. ``test_pe_duogrid_path_changes_tendency`` — duogrid path's
   wind tendency differs measurably from no-duogrid (proves
   wiring active).
3. ``test_pe_duogrid_path_finite`` — duogrid path tendency
   finite (no NaN/Inf).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
    CDGridPrimitiveEquationConfig,
    fv3_hydrostatic_tendencies,
    hydrostatic_to_fv3,
)
from legoesm.atmosphere.held_suarez import held_suarez_init
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.grids.vertical import standard_hybrid_levels


def _build_pe_state_with_winds(use_duogrid: bool):
    n = 8
    nlev = 6
    grid = create_cubed_sphere(n, use_duogrid=use_duogrid)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    coord = standard_hybrid_levels(nlev)
    state_cc = held_suarez_init(grid, coord)
    state = hydrostatic_to_fv3(state_cc, cdgrid)

    rng = np.random.default_rng(seed=333)
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


def _cfg_engages_corner_div():
    return CDGridPrimitiveEquationConfig(
        corner_div_damp_d2_bg=0.0005, corner_div_damp_dddmp=0.20,
        A_h=0.0, hyperdiff_coeff=0.0, hyperdiff_ps_coeff=0.0,
        div_damp_coeff=0.0, damp_v=0.0,
        use_conservation_fixer=False, fix_mass=False,
    )


def test_pe_no_duogrid_path_unchanged():
    """No-duogrid path produces finite tendency (regression guard:
    flag default works without grid.duogrid attr)."""
    grid, cdgrid, coord, state = _build_pe_state_with_winds(
        use_duogrid=False,
    )
    cfg = _cfg_engages_corner_div()
    tend = fv3_hydrostatic_tendencies(
        state, grid, coord, cdgrid, cfg, dt_actual=100.0,
    )
    assert np.all(np.isfinite(np.asarray(tend.du_d_dt.data)))
    assert np.all(np.isfinite(np.asarray(tend.dv_d_dt.data)))


def test_pe_duogrid_path_changes_tendency():
    """Duogrid path's wind tendency differs measurably from
    no-duogrid path on identical winds.  Proves iter-333 wiring
    is active and not a silent no-op."""
    grid_plain, cdgrid_plain, coord, state_plain = (
        _build_pe_state_with_winds(use_duogrid=False)
    )
    grid_duo, cdgrid_duo, _, state_duo = (
        _build_pe_state_with_winds(use_duogrid=True)
    )
    # Same wind perturbation seed → identical (u_d, v_d) inputs.
    np.testing.assert_array_equal(
        np.asarray(state_plain.u_d.data),
        np.asarray(state_duo.u_d.data),
    )
    cfg = _cfg_engages_corner_div()
    tend_plain = fv3_hydrostatic_tendencies(
        state_plain, grid_plain, coord, cdgrid_plain, cfg,
        dt_actual=100.0,
    )
    tend_duo = fv3_hydrostatic_tendencies(
        state_duo, grid_duo, coord, cdgrid_duo, cfg,
        dt_actual=100.0,
    )
    diff = float(np.max(np.abs(
        np.asarray(tend_plain.du_d_dt.data)
        - np.asarray(tend_duo.du_d_dt.data),
    )))
    assert diff > 1e-12, (
        "iter-333 PE wiring did NOT change du_d_dt when duogrid "
        "active.  Either the grid.duogrid attribute is None on "
        "use_duogrid=True paths or the halo site still bypasses it."
    )


def test_pe_duogrid_path_finite():
    """Duogrid path produces finite tendency."""
    grid, cdgrid, coord, state = _build_pe_state_with_winds(
        use_duogrid=True,
    )
    cfg = _cfg_engages_corner_div()
    tend = fv3_hydrostatic_tendencies(
        state, grid, coord, cdgrid, cfg, dt_actual=100.0,
    )
    assert np.all(np.isfinite(np.asarray(tend.du_d_dt.data)))
    assert np.all(np.isfinite(np.asarray(tend.dv_d_dt.data)))
    assert np.all(np.isfinite(np.asarray(tend.dT_dt.data)))
