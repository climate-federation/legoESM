"""FV3_3D iter 282: scaling test for ``corner_div_damp_d4_bg``
in the iter-18/iter-187 ``nord >= 1`` higher-order branch.

When nord_v >= 1 + d4_bg > 0, the FV3 formula at sw_core.F90:
1809 / 1811 sets:

    dd8 = (da_min_c * d4_bg)^(nord+1)

The ke_correction has a ``dd8 * divg_d_iter`` term that's
LINEAR in dd8.  So dd8 ∝ d4_bg^(nord+1) → wind change
contribution from d4_bg branch ∝ d4_bg^(nord+1).

Tests
-----

1. ``test_pe_corner_div_d4_bg_nord1_scaling`` — PE: 2x d4_bg
   at nord=1 → 4x wind contribution from d4_bg branch.
2. ``test_pe_corner_div_d4_bg_nord2_scaling`` — PE: 2x d4_bg
   at nord=2 → 8x.
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
from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.grids.vertical import standard_hybrid_levels


def _check_d4_bg_scaling(nord_v, expected_factor):
    """Verify dd8 = (da_min_c * d4_bg)^(nord+1) scaling.

    Strategy: with d2_bg=0 + dddmp=0 + nord+d4_bg active, the
    ke_correction reduces to ``dd8 * divg_d_iter`` — linear in
    dd8.  So 2x d4_bg → 2^(nord+1)x wind change.
    """
    n = 8
    nlev = 6
    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    coord = standard_hybrid_levels(nlev)
    state_cc = held_suarez_init(grid, coord)
    state = hydrostatic_to_fv3(state_cc, cdgrid)

    rng = np.random.default_rng(seed=282)
    n_corners = n + 1
    u_p = rng.uniform(-30.0, 30.0,
                      size=(6, n_corners, n_corners, nlev))
    v_p = rng.uniform(-30.0, 30.0,
                      size=(6, n_corners, n_corners, nlev))
    state = state._replace(
        u_d=state.u_d.replace(data=jnp.asarray(u_p)),
        v_d=state.v_d.replace(data=jnp.asarray(v_p)),
    )

    common = dict(
        # Need d2_bg > 0 to enter the corner-div block (gating
        # condition) but want to ISOLATE the d4_bg contribution.
        # Strategy: use a TINY d2_bg so it contributes
        # negligibly compared to d4_bg.
        corner_div_damp_d2_bg=1e-10,
        corner_div_damp_dddmp=0.0,    # no Smagorinsky cap
        corner_div_damp_nord=nord_v,
        damp_v=0.0,
        div_damp_coeff=0.0,
        A_h=0.0, hyperdiff_coeff=0.0, hyperdiff_ps_coeff=0.0,
        T_diss_coeff=0.0,
        use_conservation_fixer=False, fix_mass=False,
        zero_mean_ps_tendency=False,
    )
    base = 1e-3
    cfg_off = CDGridPrimitiveEquationConfig(
        **common, corner_div_damp_d4_bg=0.0,
    )
    cfg_1x = CDGridPrimitiveEquationConfig(
        **common, corner_div_damp_d4_bg=base,
    )
    cfg_2x = CDGridPrimitiveEquationConfig(
        **common, corner_div_damp_d4_bg=2.0 * base,
    )

    s_off = CDGridPrimitiveEquationModel(grid, coord, cfg_off).step(
        state, 100.0,
    )
    s_1x = CDGridPrimitiveEquationModel(grid, coord, cfg_1x).step(
        state, 100.0,
    )
    s_2x = CDGridPrimitiveEquationModel(grid, coord, cfg_2x).step(
        state, 100.0,
    )

    du_1x = s_1x.u_d.data - s_off.u_d.data
    du_2x = s_2x.u_d.data - s_off.u_d.data

    mask = jnp.abs(du_1x) > 1e-6
    if not jnp.any(mask):
        pytest.skip("No detectable d4_bg branch effect")

    ratio = du_2x[mask] / du_1x[mask]
    mean_ratio = float(jnp.mean(ratio))
    np.testing.assert_allclose(
        mean_ratio, expected_factor,
        rtol=0.10,
        err_msg=(
            f"corner_div_damp_d4_bg scaling at nord_v={nord_v}: "
            f"expected ~{expected_factor}x; got "
            f"{mean_ratio:.3f}.  dd8 = (da_min_c * d4_bg)^"
            f"(nord+1) exponent may be wrong."
        ),
    )


def test_pe_corner_div_d4_bg_nord1_scaling():
    """nord=1: dd8 ∝ d4_bg² → 2x d4_bg → 4x wind change."""
    _check_d4_bg_scaling(nord_v=1, expected_factor=4.0)


def test_pe_corner_div_d4_bg_nord2_scaling():
    """nord=2: dd8 ∝ d4_bg³ → 2x d4_bg → 8x wind change."""
    _check_d4_bg_scaling(nord_v=2, expected_factor=8.0)
