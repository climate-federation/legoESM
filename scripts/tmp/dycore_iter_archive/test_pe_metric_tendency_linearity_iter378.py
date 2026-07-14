"""FV3_3D iter 378: tendency-level bit-for-bit linearity for the
3 PE SLOW-TENDENCY d_con sites under metric form (iter-347/349/
351).

iter-347 had to relax linearity rtol to 1e-2 at the step()
level because the slow-tendency d_con feeds acoustic
substepping (feedback noise).  iter-378 extracts
``dT_dt`` BEFORE acoustic via direct
``fv3_hydrostatic_tendencies`` calls and pins exact linearity
(rtol=1e-12).

Tests
-----

1. ``test_pe_metric_corner_div_d_con_linear_tendency``
2. ``test_pe_metric_div_damp_d_con_linear_tendency``
3. ``test_pe_metric_ah_d_con_linear_tendency``
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
def state_setup():
    n = 8
    nlev = 5
    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    coord = standard_hybrid_levels(nlev)
    state_cc = held_suarez_init(grid, coord)
    state = hydrostatic_to_fv3(state_cc, cdgrid)
    rng = np.random.default_rng(seed=378)
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


def _check_tendency_linearity(make_cfg, state, grid, coord, cdgrid):
    """Verify dT/dt scales linearly with d_con under metric form."""
    cfg_05 = make_cfg(0.5)
    cfg_10 = make_cfg(1.0)
    cfg_20 = make_cfg(2.0)
    dt = 100.0
    tend_05 = fv3_hydrostatic_tendencies(
        state, grid, coord, cdgrid, cfg_05, dt_actual=dt,
    )
    tend_10 = fv3_hydrostatic_tendencies(
        state, grid, coord, cdgrid, cfg_10, dt_actual=dt,
    )
    tend_20 = fv3_hydrostatic_tendencies(
        state, grid, coord, cdgrid, cfg_20, dt_actual=dt,
    )
    d10 = np.asarray(tend_10.dT_dt.data) - np.asarray(tend_05.dT_dt.data)
    d20 = np.asarray(tend_20.dT_dt.data) - np.asarray(tend_05.dT_dt.data)
    np.testing.assert_allclose(
        d20, 3.0 * d10, rtol=1e-10, atol=1e-12,
    )
    assert float(np.max(np.abs(d10))) > 1e-12


def test_pe_metric_corner_div_d_con_linear_tendency(state_setup):
    grid, cdgrid, coord, state = state_setup

    def make_cfg(d):
        return CDGridPrimitiveEquationConfig(
            corner_div_damp_d2_bg=0.0005, corner_div_damp_dddmp=0.20,
            corner_div_damp_d_con=d,
            damp_v=0.0, A_h=0.0,
            hyperdiff_coeff=0.0, hyperdiff_ps_coeff=0.0,
            div_damp_coeff=0.0,
            use_conservation_fixer=False, fix_mass=False,
            use_fv3_metric_aware_d_con=True,
        )

    _check_tendency_linearity(make_cfg, state, grid, coord, cdgrid)


def test_pe_metric_div_damp_d_con_linear_tendency(state_setup):
    grid, cdgrid, coord, state = state_setup

    def make_cfg(d):
        return CDGridPrimitiveEquationConfig(
            div_damp_coeff=1e6, div_damp_dddmp=0.20,
            div_damp_d_con=d,
            damp_v=0.0, corner_div_damp_d2_bg=0.0, A_h=0.0,
            hyperdiff_coeff=0.0, hyperdiff_ps_coeff=0.0,
            use_conservation_fixer=False, fix_mass=False,
            use_fv3_metric_aware_d_con=True,
        )

    _check_tendency_linearity(make_cfg, state, grid, coord, cdgrid)


def test_pe_metric_ah_d_con_linear_tendency(state_setup):
    grid, cdgrid, coord, state = state_setup

    def make_cfg(d):
        return CDGridPrimitiveEquationConfig(
            A_h=1e6, ah_d_con=d,
            damp_v=0.0, corner_div_damp_d2_bg=0.0,
            hyperdiff_coeff=0.0, hyperdiff_ps_coeff=0.0,
            div_damp_coeff=0.0,
            use_conservation_fixer=False, fix_mass=False,
            use_fv3_metric_aware_d_con=True,
        )

    _check_tendency_linearity(make_cfg, state, grid, coord, cdgrid)
