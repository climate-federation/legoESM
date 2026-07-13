"""FV3_3D iter 484: PE mirror of NH iter-483 aggressive
edge-min factory.

Tests
-----

1. ``test_factory_importable``.
2. ``test_factory_disables_hurting_flags``.
3. ``test_factory_boosts_d2_bg``.
4. ``test_factory_overrides_take_precedence``.
5. ``test_factory_step_finite``.
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
    make_legoesm_pe_min_edge_aggressive_config,
)
from legoesm.atmosphere.held_suarez import held_suarez_init
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.grids.vertical import standard_hybrid_levels


def test_factory_importable():
    assert callable(make_legoesm_pe_min_edge_aggressive_config)


def test_factory_disables_hurting_flags():
    cfg = make_legoesm_pe_min_edge_aggressive_config()
    assert cfg.use_fv3_metric_aware_d_con is False
    assert cfg.heat_source_del2_iters == 0
    assert cfg.d_con_top_zero_levels == 0


def test_factory_boosts_d2_bg():
    cfg = make_legoesm_pe_min_edge_aggressive_config()
    assert cfg.corner_div_damp_d2_bg == 5e-2


def test_factory_overrides_take_precedence():
    cfg = make_legoesm_pe_min_edge_aggressive_config(
        corner_div_damp_d2_bg=1e-3,
        use_fv3_metric_aware_d_con=True,
    )
    assert cfg.corner_div_damp_d2_bg == 1e-3
    assert cfg.use_fv3_metric_aware_d_con is True


def test_factory_step_finite():
    n = 8
    nlev = 5
    grid = create_cubed_sphere(n, use_duogrid=True)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    coord = standard_hybrid_levels(nlev)
    state = hydrostatic_to_fv3(held_suarez_init(grid, coord), cdgrid)
    cfg = make_legoesm_pe_min_edge_aggressive_config(
        damp_v=0.030, damp_v_d_con=1.0,
        corner_div_damp_d_con=1.0,
        div_damp_coeff=1e6, div_damp_d_con=1.0,
    )
    model = CDGridPrimitiveEquationModel(grid, coord, cfg)
    new_state = model.step(state, dt=10.0)
    for f in (new_state.u_d, new_state.v_d, new_state.T,
              new_state.p_s):
        assert jnp.all(jnp.isfinite(f.data))
