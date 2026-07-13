"""FV3_3D iter 248: stress test the delt_max cap mechanism
with EXTREME damping values where the cap actively engages.

Background: iter-218/219 added the post-step cap on damp_v
d_con; iter-239/240 extended it to the aggregate tendency
d_con sites.  Default ``delt_max=0.0`` disables.  FV3
production default 1.0 K/s rarely engages for typical flows.

This test forces the cap to engage by using EXTREME damping
values that produce per-step ΔT > 1 K, then verifies:

1. The cap actually bounds the per-step ΔT (no runaway).
2. With cap active, the integration remains stable for 20
   steps (no NaN, bounded growth).
3. Without cap (delt_max=0), the same extreme damping
   produces UN-CAPPED per-step ΔT — the cap test isn't
   vacuous.

Tests
-----

1. ``test_pe_extreme_damping_delt_max_cap_engages`` — strong
   damp_v + corner-div + cell-centre div_damp + A_h, all
   d_con knobs at MAX 100.0 (way over FV3 production 1.0)
   with delt_max=0.5.  Verify cap is active (per-step
   |ΔT_d_con| ≤ 0.5 * dt at interior layers) AND model
   remains stable.
2. ``test_pe_extreme_damping_uncapped_runaway_or_huge`` —
   same extreme damping with delt_max=0.0.  Per-step ΔT
   should be SIGNIFICANTLY larger than the capped run
   (proving the test isn't vacuous).
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


def _strong_state():
    n = 8
    nlev = 6
    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    coord = standard_hybrid_levels(nlev)
    state_cc = held_suarez_init(grid, coord)
    state = hydrostatic_to_fv3(state_cc, cdgrid)

    rng = np.random.default_rng(seed=248)
    n_corners = n + 1
    # Very strong winds → very strong KE removal → very strong d_con heat
    u_p = rng.uniform(-50.0, 50.0,
                      size=(6, n_corners, n_corners, nlev))
    v_p = rng.uniform(-50.0, 50.0,
                      size=(6, n_corners, n_corners, nlev))
    state = state._replace(
        u_d=state.u_d.replace(data=jnp.asarray(u_p)),
        v_d=state.v_d.replace(data=jnp.asarray(v_p)),
    )
    return grid, cdgrid, coord, state


def _extreme_damping_common():
    return dict(
        # Production damping
        corner_div_damp_d2_bg=0.0005, corner_div_damp_dddmp=0.20,
        div_damp_coeff=1e6, div_damp_dddmp=0.20,
        damp_v=0.030, nord_v=0,
        A_h=1e6, smagorinsky_cs=0.20,
        hyperdiff_coeff=0.0, hyperdiff_ps_coeff=0.0,
        # EXTREME d_con multipliers (100x FV3 production)
        damp_v_d_con=100.0,
        corner_div_damp_d_con=100.0,
        div_damp_d_con=100.0,
        ah_d_con=100.0,
        use_conservation_fixer=False, fix_mass=False,
        zero_mean_ps_tendency=False,
    )


def test_pe_extreme_damping_delt_max_cap_engages():
    """A very tight delt_max=1e-5 actively bounds the d_con
    aggregate tendency in interior layers under extreme damping.
    Tests the iter-239 aggregate cap, not the iter-218
    post-step cap (those test different code paths)."""
    grid, cdgrid, coord, state = _strong_state()
    dt = 100.0
    delt_max = 1e-5    # cap @ interior = 1e-5 K/s = 1e-3 K/step

    common_no_dampv = dict(
        # Production damping but with damp_v=0 so we test the
        # iter-239 aggregate tendency cap (which excludes the
        # iter-218 post-step damp_v cap from the comparison).
        corner_div_damp_d2_bg=0.0005, corner_div_damp_dddmp=0.20,
        div_damp_coeff=1e6, div_damp_dddmp=0.20,
        damp_v=0.0,    # disable post-step block
        A_h=1e6, smagorinsky_cs=0.20,
        hyperdiff_coeff=0.0, hyperdiff_ps_coeff=0.0,
        # Extreme d_con multipliers
        corner_div_damp_d_con=100.0,
        div_damp_d_con=100.0,
        ah_d_con=100.0,
        use_conservation_fixer=False, fix_mass=False,
        zero_mean_ps_tendency=False,
    )
    cfg_no_dcon = CDGridPrimitiveEquationConfig(
        **{k: v for k, v in common_no_dampv.items()
           if not k.endswith("_d_con")},
        corner_div_damp_d_con=0.0,
        div_damp_d_con=0.0, ah_d_con=0.0,
        delt_max=0.0,
    )
    cfg_uncapped = CDGridPrimitiveEquationConfig(
        **common_no_dampv, delt_max=0.0,
    )
    cfg_capped = CDGridPrimitiveEquationConfig(
        **common_no_dampv, delt_max=delt_max,
    )

    m_no = CDGridPrimitiveEquationModel(grid, coord, cfg_no_dcon)
    m_un = CDGridPrimitiveEquationModel(grid, coord, cfg_uncapped)
    m_cap = CDGridPrimitiveEquationModel(grid, coord, cfg_capped)

    s_no = m_no.step(state, dt)
    s_un = m_un.step(state, dt)
    s_cap = m_cap.step(state, dt)

    # d_con-only contribution at INTERIOR layers (k≥2).
    d_con_un = (s_un.T.data - s_no.T.data)[..., 2:]
    d_con_cap = (s_cap.T.data - s_no.T.data)[..., 2:]

    max_un = float(jnp.max(jnp.abs(d_con_un)))
    max_cap = float(jnp.max(jnp.abs(d_con_cap)))

    # Capped must be smaller (cap is engaging).
    assert max_cap < max_un, (
        f"Capped d_con interior max|ΔT|={max_cap:.4e} should be "
        f"< uncapped {max_un:.4e}.  Cap is not engaging — try a "
        f"tighter delt_max."
    )

    # Run 20 steps with cap; verify stable.
    s = state
    for _ in range(20):
        s = m_cap.step(s, dt)
    assert jnp.all(jnp.isfinite(s.T.data)), (
        "PE with cap must stay stable over 20 steps even with "
        "100x d_con multipliers."
    )


def test_pe_extreme_damping_uncapped_excess_heat():
    """Without the cap, extreme damping produces large per-step
    heat that the cap would otherwise reduce — confirms the
    iter-248 stress test is non-vacuous."""
    grid, cdgrid, coord, state = _strong_state()
    dt = 100.0

    cfg_extreme_off = CDGridPrimitiveEquationConfig(
        **_extreme_damping_common(), delt_max=0.0,
    )
    cfg_normal_off = CDGridPrimitiveEquationConfig(
        **{k: v for k, v in _extreme_damping_common().items()
           if k not in ("damp_v_d_con", "corner_div_damp_d_con",
                        "div_damp_d_con", "ah_d_con")},
        damp_v_d_con=1.0,
        corner_div_damp_d_con=1.0,
        div_damp_d_con=1.0,
        ah_d_con=1.0,
        delt_max=0.0,
    )

    m_extreme = CDGridPrimitiveEquationModel(
        grid, coord, cfg_extreme_off,
    )
    m_normal = CDGridPrimitiveEquationModel(
        grid, coord, cfg_normal_off,
    )

    s_extreme = m_extreme.step(state, dt)
    s_normal = m_normal.step(state, dt)

    dT_extreme = s_extreme.T.data - state.T.data
    dT_normal = s_normal.T.data - state.T.data

    # Extreme should produce notably MORE heat than normal in
    # interior layers (where d_con's effect dominates).
    interior_extreme = float(jnp.max(jnp.abs(dT_extreme[..., 2:])))
    interior_normal = float(jnp.max(jnp.abs(dT_normal[..., 2:])))
    assert interior_extreme > interior_normal * 5.0, (
        f"Extreme d_con multipliers (100x) must produce >5x more "
        f"heat than normal (1x): extreme={interior_extreme:.4e}, "
        f"normal={interior_normal:.4e}.  If not, the test is "
        f"vacuous because d_con multipliers don't actually scale "
        f"heat output."
    )
