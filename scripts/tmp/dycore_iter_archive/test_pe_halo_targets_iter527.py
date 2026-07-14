"""FV3_3D iter 527: extend ``monotone_halo_clip_context`` to PE
import aliases.

PE dycore imports 4 distinct halo aliases not covered by
iter-505-526 (which targeted NH + operator modules):
- ``_pad_halo_4d`` (line 57)
- ``_pad_halo_4d_module`` (line 83)
- ``pad_halo_vector`` (3D, line 84)
- ``pad_halo_vector_4d`` (line 85)

iter-527 adds these as 3 additional patch targets bringing
total to 15.

Tests
-----

1. ``test_pe_aliases_patchable`` — verify patching works on
   the new PE targets.
2. ``test_pe_mass_still_conserved`` — iter-524 mass
   conservation test still passes with extra patches active.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid as pe_mod
from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
    CDGridPrimitiveEquationModel,
    hydrostatic_to_fv3,
    make_legoesm_pe_min_edge_config,
)
from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.grids.halo import monotone_halo_clip_context
from legoesm.grids.vertical import standard_hybrid_levels


def test_pe_aliases_patchable():
    """Inside context, all 4 PE aliases must be patched to partials."""
    import functools
    orig_pad_halo_4d = pe_mod._pad_halo_4d
    orig_pad_halo_4d_module = pe_mod._pad_halo_4d_module
    orig_pad_halo_vector = pe_mod.pad_halo_vector
    orig_pad_halo_vector_4d = pe_mod.pad_halo_vector_4d
    with monotone_halo_clip_context(slack=0.5):
        # Inside: must be partial of original
        assert isinstance(pe_mod._pad_halo_4d, functools.partial), (
            "PE _pad_halo_4d not patched in iter-527"
        )
        assert isinstance(pe_mod._pad_halo_4d_module, functools.partial)
        assert isinstance(pe_mod.pad_halo_vector, functools.partial)
        assert isinstance(pe_mod.pad_halo_vector_4d, functools.partial)
    # Outside: restored
    assert pe_mod._pad_halo_4d is orig_pad_halo_4d
    assert pe_mod._pad_halo_4d_module is orig_pad_halo_4d_module
    assert pe_mod.pad_halo_vector is orig_pad_halo_vector
    assert pe_mod.pad_halo_vector_4d is orig_pad_halo_vector_4d


def test_pe_mass_still_conserved():
    """PE mass conservation under expanded clip context (iter-527)."""
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

    def _total_mass(s):
        p_s = np.asarray(s.p_s.data)
        area = np.asarray(grid.area)
        return float(np.sum(p_s * area))

    m0 = _total_mass(state)
    with monotone_halo_clip_context(slack=0.5):
        m = CDGridPrimitiveEquationModel(grid, coord, cfg)
        s = state
        for _ in range(10):
            s = m.step(s, dt=10.0)
    m10 = _total_mass(s)
    rel = abs((m10 - m0) / m0)
    assert rel < 1e-6, (
        f"PE mass should be conserved under iter-527 expanded "
        f"clip context: rel drift = {rel:.2e}"
    )
