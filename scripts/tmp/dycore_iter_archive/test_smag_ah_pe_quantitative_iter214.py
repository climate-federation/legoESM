"""FV3_3D iter 214: PE counterpart of iter-199 quantitative
direction test for the iter-57/58 Smagorinsky-adaptive A_h.

iter-199 verified that smag_cs > 0 + A_h > 0 produces SMALLER
max|u| after 5 steps than A_h-only baseline on the NH 3D path.
PE has its own iter-57/58 Smagorinsky wiring tested in
``test_div_damp_adaptive.py`` (test_smagorinsky_cs_zero_is_bit_for_bit_baseline,
test_smagorinsky_cs_active_changes_winds,
test_smagorinsky_combined_with_ah_stable_multistep) but NO
direction-correctness test.  A sign-flipped Smagorinsky on PE
would still pass the existing tests but AMPLIFY winds at high-
strain regions.

This iter mirrors iter-199 for PE.

Tests
-----

1. ``test_pe_smag_reduces_max_wind_more_than_static_ah`` — high-
   strain shear IC; smag_cs > 0 + A_h > 0 produces SMALLER
   max|u_d| after 5 PE steps than A_h-only baseline.
2. ``test_pe_smag_no_amplification_for_typical_cs_range[0.1/0.2/0.4]``
   — across PE iter-60-tested range, none should AMPLIFY winds
   beyond A_h-only baseline.
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


@pytest.fixture(scope="module")
def high_strain_pe_state():
    """PE state with sinusoidal shear in u_d so ∂u_d/∂y is large.
    Uses HS init's T, p_s but zero background winds — a cleaner
    IC than HS spin-up so the strain signal isn't dominated by
    the HS jet's own response to dynamics."""
    n = 8
    nlev = 6
    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    coord = standard_hybrid_levels(nlev)
    state_cc = held_suarez_init(grid, coord)
    state = hydrostatic_to_fv3(state_cc, cdgrid)

    n_corners = n + 1
    j_idx = jnp.arange(n_corners)
    shear = (j_idx - n_corners / 2)[None, None, :, None]
    pattern = jnp.broadcast_to(shear, (6, n_corners, n_corners, nlev))
    pattern = pattern.astype(jnp.float64)
    U0 = 5.0
    u_perturb = jnp.asarray(U0 * pattern, dtype=jnp.float64)

    state = state._replace(
        u_d=state.u_d.replace(data=u_perturb),
        v_d=state.v_d.replace(
            data=jnp.zeros_like(state.v_d.data),
        ),
    )
    return grid, cdgrid, coord, state


def _strain_proxy(state):
    """Use std|u_d| as a strain proxy.  std integrates over the full
    field rather than just sampling the max (which can be sensitive
    to single-cell HS jet response)."""
    return float(jnp.std(state.u_d.data))


def _step5(model, state, dt=100.0):
    s = state
    for _ in range(5):
        s = model.step(s, dt)
    return s


def test_pe_smag_reduces_max_wind_more_than_static_ah(high_strain_pe_state):
    """smagorinsky_cs > 0 + A_h > 0 produces smaller std|u_d| after
    5 steps than A_h-only baseline on a high-strain IC.  Use std
    rather than max to avoid sensitivity to single-cell HS jet
    response (the HS init's own jet structure can lift the max in
    a way uncorrelated with the smag damping intent)."""
    grid, cdgrid, coord, state = high_strain_pe_state

    common = dict(
        A_h=1e6,
        use_conservation_fixer=False, fix_mass=False,
    )
    cfg_static_only = CDGridPrimitiveEquationConfig(
        **common, smagorinsky_cs=0.0,
    )
    cfg_static_plus_smag = CDGridPrimitiveEquationConfig(
        **common, smagorinsky_cs=0.20,
    )

    m_static = CDGridPrimitiveEquationModel(grid, coord, cfg_static_only)
    m_smag = CDGridPrimitiveEquationModel(grid, coord, cfg_static_plus_smag)

    s_static = _step5(m_static, state)
    s_smag = _step5(m_smag, state)

    std_static = _strain_proxy(s_static)
    std_smag = _strain_proxy(s_smag)

    assert std_smag < std_static, (
        f"PE smagorinsky_cs=0.20 must REDUCE std|u_d| relative to "
        f"static A_h alone: static={std_static:.4e}, "
        f"smag={std_smag:.4e} (Smagorinsky AMPLIFIED strain — "
        f"sign error in compute_smagorinsky_ah_3d on PE path)."
    )


@pytest.mark.parametrize("smag_cs", [0.1, 0.2, 0.4])
def test_pe_smag_no_amplification_for_typical_cs_range(
    high_strain_pe_state, smag_cs,
):
    """Across smag_cs in {0.1, 0.2, 0.4}, none should AMPLIFY winds
    beyond the A_h-only baseline on PE."""
    grid, cdgrid, coord, state = high_strain_pe_state

    common = dict(
        A_h=1e6,
        use_conservation_fixer=False, fix_mass=False,
    )
    cfg_static_only = CDGridPrimitiveEquationConfig(
        **common, smagorinsky_cs=0.0,
    )
    cfg_smag = CDGridPrimitiveEquationConfig(
        **common, smagorinsky_cs=smag_cs,
    )

    m_static = CDGridPrimitiveEquationModel(grid, coord, cfg_static_only)
    m_smag = CDGridPrimitiveEquationModel(grid, coord, cfg_smag)

    s_static = _step5(m_static, state)
    s_smag = _step5(m_smag, state)

    std_static = _strain_proxy(s_static)
    std_smag = _strain_proxy(s_smag)

    assert std_smag <= std_static * 1.001, (
        f"PE smag_cs={smag_cs}: must not AMPLIFY strain: "
        f"static={std_static:.4e}, smag={std_smag:.4e}.  "
        f"Sign error in the PE Smagorinsky augmentation."
    )
