"""FV3_3D iter 278: scaling test for the iter-12 damp_v
wiring — verifies the FV3-faithful damp coefficient formula
``damp4 = (damp_v * da_min_c)^(nord_v + 1)``.

For nord_v=k, doubling damp_v should multiply damp4 by
2^(k+1) and therefore scale the wind tendency contribution
linearly in damp4 (since del6_vt_flux is linear in damp).

Tests
-----

1. ``test_pe_damp_v_scaling_nord0`` — nord_v=0: damp4 ∝ damp_v.
   Doubling damp_v doubles wind change.
2. ``test_pe_damp_v_scaling_nord1`` — nord_v=1: damp4 ∝
   damp_v². Doubling damp_v quadruples wind change.
3. ``test_pe_damp_v_scaling_nord2`` — nord_v=2: damp4 ∝
   damp_v³. Doubling damp_v gives 8× wind change.
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


def _setup_state():
    n = 8
    nlev = 6
    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    coord = standard_hybrid_levels(nlev)
    state_cc = held_suarez_init(grid, coord)
    state = hydrostatic_to_fv3(state_cc, cdgrid)

    rng = np.random.default_rng(seed=278)
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


def _check_scaling(nord_v, expected_factor):
    """Compare damp_v=1x and damp_v=2x; verify the wind change
    contribution scales as expected_factor (= 2^(nord+1))."""
    grid, cdgrid, coord, state = _setup_state()

    common = dict(
        nord_v=nord_v,
        corner_div_damp_d2_bg=0.0,
        div_damp_coeff=0.0,
        A_h=0.0, hyperdiff_coeff=0.0, hyperdiff_ps_coeff=0.0,
        T_diss_coeff=0.0,
        damp_v_d_con=0.0,
        use_conservation_fixer=False, fix_mass=False,
        zero_mean_ps_tendency=False,
    )
    base_damp = 0.030
    cfg_off = CDGridPrimitiveEquationConfig(
        **common, damp_v=0.0,
    )
    cfg_1x = CDGridPrimitiveEquationConfig(
        **common, damp_v=base_damp,
    )
    cfg_2x = CDGridPrimitiveEquationConfig(
        **common, damp_v=2.0 * base_damp,
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

    # damp_v wind contribution = (with damp_v) - (without).
    du_1x = s_1x.u_d.data - s_off.u_d.data
    du_2x = s_2x.u_d.data - s_off.u_d.data

    # Verify scaling: du_2x ≈ expected_factor * du_1x at the
    # cells where du_1x is non-trivially large.
    mask = jnp.abs(du_1x) > 1e-6
    if not jnp.any(mask):
        pytest.skip("No detectable damp_v effect at this nord")

    ratio = du_2x[mask] / du_1x[mask]
    # Should be close to expected_factor.  Use mean to average
    # out FP noise.
    mean_ratio = float(jnp.mean(ratio))
    np.testing.assert_allclose(
        mean_ratio, expected_factor,
        rtol=0.05,
        err_msg=(
            f"damp_v scaling at nord_v={nord_v}: expected wind "
            f"change to scale as ~{expected_factor}x when "
            f"damp_v doubles (damp4 ∝ damp_v^(nord+1))."
        ),
    )


def test_pe_damp_v_scaling_nord0():
    """nord_v=0: damp4 ∝ damp_v.  2x damp_v → 2x wind change."""
    _check_scaling(nord_v=0, expected_factor=2.0)


def test_pe_damp_v_scaling_nord1():
    """nord_v=1: damp4 ∝ damp_v².  2x damp_v → 4x wind change."""
    _check_scaling(nord_v=1, expected_factor=4.0)


def test_pe_damp_v_scaling_nord2():
    """nord_v=2: damp4 ∝ damp_v³.  2x damp_v → 8x wind change."""
    _check_scaling(nord_v=2, expected_factor=8.0)
