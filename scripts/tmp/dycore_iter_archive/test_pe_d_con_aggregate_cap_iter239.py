"""FV3_3D iter 239: sponge-aware ``delt_max`` cap on the
aggregate tendency-based d_con stack (PE).

iter-218/219 implemented ``delt_max`` cap on the iter-208
damp_v_d_con post-step block.  The tendency-based d_con sites
(iter-221 corner-div, iter-223 cell-centre div_damp, iter-225
A_h) bypassed that cap.  iter-239 closes the gap: the 3
tendency-based contributions are now AGGREGATED and the cap
applies to the SUM, mirroring FV3 ``dyn_core.F90:1764-1779``
which accumulates ``heat_source`` from all sources then caps
once.

Cap policy (per FV3 + iter-219 sponge-awareness, in tendency
form for PE):

* Top 2 sponge layers (k=0,1): UNCAPPED (jnp.inf).
* Interior layers (k>=2): |dT_dt_d_con_total| ≤ ``delt_max`` K/s,
  equivalent to per-step ΔT cap of ``dt * delt_max``.

Tests
-----

1. ``test_pe_d_con_aggregate_off_when_delt_max_zero`` —
   delt_max=0 (default) preserves baseline (no cap, all
   tendency-based d_con contributions sum unchanged).
2. ``test_pe_d_con_aggregate_caps_interior_to_delt_max`` —
   strong damping + small delt_max: the d_con-only contribution
   to dT_dt at INTERIOR levels (k>=2) must satisfy
   ``|dT_dt| <= delt_max``.
3. ``test_pe_d_con_top_layers_uncapped`` — k=0,1 are not
   capped (FV3 cp_air sponge skip).
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
from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.grids.vertical import standard_hybrid_levels


@pytest.fixture(scope="module")
def small_pe_state_with_winds():
    n = 8
    nlev = 6
    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    coord = standard_hybrid_levels(nlev)
    state_cc = held_suarez_init(grid, coord)
    state = hydrostatic_to_fv3(state_cc, cdgrid)

    rng = np.random.default_rng(seed=239)
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


def test_pe_d_con_aggregate_off_when_delt_max_zero(small_pe_state_with_winds):
    """delt_max=0 preserves uncapped baseline."""
    grid, cdgrid, coord, state = small_pe_state_with_winds
    common = dict(
        corner_div_damp_d2_bg=0.0005, corner_div_damp_dddmp=0.20,
        div_damp_coeff=1e6, div_damp_dddmp=0.20,
        A_h=1e6, smagorinsky_cs=0.20,
        hyperdiff_coeff=0.0, hyperdiff_ps_coeff=0.0,
        damp_v=0.0,
        use_conservation_fixer=False, fix_mass=False,
    )
    cfg_explicit_zero = CDGridPrimitiveEquationConfig(
        **common,
        corner_div_damp_d_con=1.0,
        div_damp_d_con=1.0, ah_d_con=1.0,
        delt_max=0.0,
    )
    cfg_default_zero = CDGridPrimitiveEquationConfig(
        **common,
        corner_div_damp_d_con=1.0,
        div_damp_d_con=1.0, ah_d_con=1.0,
    )
    dt = 100.0
    t1 = fv3_hydrostatic_tendencies(
        state, grid, coord, cdgrid, cfg_explicit_zero, dt_actual=dt,
    )
    t2 = fv3_hydrostatic_tendencies(
        state, grid, coord, cdgrid, cfg_default_zero, dt_actual=dt,
    )
    np.testing.assert_array_equal(t1.dT_dt.data, t2.dT_dt.data)


def test_pe_d_con_aggregate_caps_interior_to_delt_max(small_pe_state_with_winds):
    """Interior layers (k>=2) bounded to ``delt_max`` per second
    when delt_max > 0 + strong damping engages all tendency
    d_con sites."""
    grid, cdgrid, coord, state = small_pe_state_with_winds
    common = dict(
        corner_div_damp_d2_bg=0.0005, corner_div_damp_dddmp=0.20,
        div_damp_coeff=1e6, div_damp_dddmp=0.20,
        A_h=1e6, smagorinsky_cs=0.20,
        hyperdiff_coeff=0.0, hyperdiff_ps_coeff=0.0,
        damp_v=0.0,
        use_conservation_fixer=False, fix_mass=False,
    )
    delt_max = 1e-6   # very tight cap to force activation
    cfg_no_dcon = CDGridPrimitiveEquationConfig(
        **common,
        corner_div_damp_d_con=0.0,
        div_damp_d_con=0.0, ah_d_con=0.0,
        delt_max=0.0,
    )
    cfg_capped = CDGridPrimitiveEquationConfig(
        **common,
        corner_div_damp_d_con=1.0,
        div_damp_d_con=1.0, ah_d_con=1.0,
        delt_max=delt_max,
    )
    dt = 100.0
    t_no = fv3_hydrostatic_tendencies(
        state, grid, coord, cdgrid, cfg_no_dcon, dt_actual=dt,
    )
    t_cap = fv3_hydrostatic_tendencies(
        state, grid, coord, cdgrid, cfg_capped, dt_actual=dt,
    )

    # d_con contribution = capped run - no-d_con baseline.
    dT_dt_d_con = t_cap.dT_dt.data - t_no.dT_dt.data

    # Interior k>=2 must respect the cap.
    interior = dT_dt_d_con[..., 2:]
    max_interior = float(jnp.max(jnp.abs(interior)))
    assert max_interior <= delt_max + 1e-12, (
        f"d_con aggregate INTERIOR max|dT_dt|={max_interior:.4e} "
        f"> cap={delt_max:.4e} — sponge-aware aggregate cap is "
        f"not bounding interior layers."
    )


def test_pe_d_con_top_layers_uncapped(small_pe_state_with_winds):
    """Top 2 sponge layers (k=0,1) must NOT be capped — FV3
    cp_air branch sw_core.F90:1764-1773 skips the cap entirely
    for these layers."""
    grid, cdgrid, coord, state = small_pe_state_with_winds
    common = dict(
        corner_div_damp_d2_bg=0.0005, corner_div_damp_dddmp=0.20,
        div_damp_coeff=1e6, div_damp_dddmp=0.20,
        A_h=1e6, smagorinsky_cs=0.20,
        hyperdiff_coeff=0.0, hyperdiff_ps_coeff=0.0,
        damp_v=0.0,
        use_conservation_fixer=False, fix_mass=False,
    )
    delt_max = 1e-10   # cap so tight even tiny d_con tendencies
                       # would clip at interior; ensures we can
                       # detect the asymmetry between k<2 and k>=2.
    cfg_uncapped = CDGridPrimitiveEquationConfig(
        **common,
        corner_div_damp_d_con=1.0,
        div_damp_d_con=1.0, ah_d_con=1.0,
        delt_max=0.0,
    )
    cfg_capped = CDGridPrimitiveEquationConfig(
        **common,
        corner_div_damp_d_con=1.0,
        div_damp_d_con=1.0, ah_d_con=1.0,
        delt_max=delt_max,
    )
    dt = 100.0
    t_un = fv3_hydrostatic_tendencies(
        state, grid, coord, cdgrid, cfg_uncapped, dt_actual=dt,
    )
    t_cap = fv3_hydrostatic_tendencies(
        state, grid, coord, cdgrid, cfg_capped, dt_actual=dt,
    )

    # Top 2 layers must be IDENTICAL (no cap applied).
    np.testing.assert_array_equal(
        t_cap.dT_dt.data[..., 0],
        t_un.dT_dt.data[..., 0],
        err_msg=(
            "PE k=0 (top sponge) d_con aggregate must not be "
            "capped — FV3 cp_air branch sw_core.F90:1764-1773 "
            "skips the cap entirely for k<3 (FV3 1-based)."
        ),
    )
    np.testing.assert_array_equal(
        t_cap.dT_dt.data[..., 1],
        t_un.dT_dt.data[..., 1],
        err_msg="PE k=1 (sponge) must not be capped.",
    )
