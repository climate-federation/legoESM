"""FV3_3D iter 359: PE full FV3-fidelity stack changes state
measurably at C16.

iter-357 verified PE metric flag at C16 doesn't AMPLIFY imprint;
iter-359 verifies full PE FV3-fidelity stack (duogrid + metric +
a2b_zeta) produces measurably DIFFERENT state from default flags
+ same toolkit at C16.  Confirms wiring active at production
resolution (not just bit-for-bit at C8).

Tests
-----

1. ``test_pe_c16_full_fv3_changes_T`` — full PE FV3 stack
   differs measurably from default at C16 in T field.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
    CDGridPrimitiveEquationConfig,
    CDGridPrimitiveEquationModel,
    hydrostatic_to_fv3,
)
from legoesm.atmosphere.held_suarez import held_suarez_init
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.grids.vertical import standard_hybrid_levels


def _build_c16(use_duogrid):
    n = 16
    nlev = 5
    grid = create_cubed_sphere(n, use_duogrid=use_duogrid)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    coord = standard_hybrid_levels(nlev)
    state_cc = held_suarez_init(grid, coord)
    state = hydrostatic_to_fv3(state_cc, cdgrid)
    rng = np.random.default_rng(seed=359)
    n_corners = n + 1
    u_p = rng.uniform(-5.0, 5.0,
                      size=(6, n_corners, n_corners, nlev))
    v_p = rng.uniform(-5.0, 5.0,
                      size=(6, n_corners, n_corners, nlev))
    state = state._replace(
        u_d=state.u_d.replace(data=jnp.asarray(u_p)),
        v_d=state.v_d.replace(data=jnp.asarray(v_p)),
    )
    return grid, coord, state


def _cfg(all_flags):
    return CDGridPrimitiveEquationConfig(
        corner_div_damp_d2_bg=0.0005, corner_div_damp_dddmp=0.20,
        corner_div_damp_d_con=1.0,
        damp_v=0.030, nord_v=1,
        damp_v_d_con=1.0,
        use_fv3_a2b_zeta_corner=all_flags,
        div_damp_coeff=1e6, div_damp_dddmp=0.20,
        div_damp_d_con=1.0,
        A_h=1e6, ah_d_con=1.0,
        delt_max=1.0,
        hyperdiff_coeff=0.0, hyperdiff_ps_coeff=0.0,
        use_conservation_fixer=False, fix_mass=False,
        use_fv3_metric_aware_d_con=all_flags,
    )


def test_pe_c16_full_fv3_changes_T():
    grid_d, coord, state = _build_c16(use_duogrid=False)
    grid_f, _, state_f = _build_c16(use_duogrid=True)
    m_default = CDGridPrimitiveEquationModel(grid_d, coord, _cfg(False))
    m_full = CDGridPrimitiveEquationModel(grid_f, coord, _cfg(True))
    s_d = m_default.step(state, 100.0)
    s_f = m_full.step(state_f, 100.0)
    diff = float(np.max(np.abs(
        np.asarray(s_d.T.data) - np.asarray(s_f.T.data),
    )))
    assert diff > 1e-6, (
        f"PE full FV3-fidelity stack at C16 did NOT measurably "
        f"change T vs default: diff={diff:.3e}.  Wiring may be "
        f"silent no-op at production resolution."
    )
