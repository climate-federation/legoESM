"""FV3_3D iter 296: PE iter-189 dt-plumbing regression.

iter-189 plumbed the actual integration ``dt`` from
``model.step(state, dt)`` through to the corner-div damping
adaptive cap (FV3 sw_core.F90:1720).  Direct callers of the
tendency function that don't pass ``dt_actual`` fall back to
``config.corner_div_damp_dt_proxy`` (default 200.0), preserving
backward compatibility for unit tests.

If the dt-plumbing ever regresses (e.g., ``model.step`` stops
passing dt_actual to the tendency fn), the cap will silently
fall back to ``corner_div_damp_dt_proxy`` — likely with no
visible test failure since most tests use the default proxy
=200 anyway.

This iter pins the plumbing by:
1. Building two PE models with intentionally DIFFERENT
   ``corner_div_damp_dt_proxy`` values (50 vs 500).
2. Stepping both with the SAME ``dt`` (=100).
3. Asserting outputs are bit-for-bit identical — confirms
   the actual integration dt wins over the config proxy.

Tests
-----

1. ``test_pe_step_uses_dt_actual_over_proxy`` — bit-for-bit
   identical step output across different
   ``corner_div_damp_dt_proxy`` values when ``model.step`` is
   used.
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


def _make_state(seed):
    n = 8
    nlev = 6
    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    coord = standard_hybrid_levels(nlev)
    state_cc = held_suarez_init(grid, coord)
    state = hydrostatic_to_fv3(state_cc, cdgrid)
    rng = np.random.default_rng(seed=seed)
    n_corners = n + 1
    u_p = rng.uniform(-3.0, 3.0,
                      size=(6, n_corners, n_corners, nlev))
    v_p = rng.uniform(-3.0, 3.0,
                      size=(6, n_corners, n_corners, nlev))
    state = state._replace(
        u_d=state.u_d.replace(data=jnp.asarray(u_p)),
        v_d=state.v_d.replace(data=jnp.asarray(v_p)),
    )
    return grid, coord, state


def _build_cfg(dt_proxy):
    return CDGridPrimitiveEquationConfig(
        corner_div_damp_d2_bg=0.0005,
        corner_div_damp_dddmp=0.20,        # cap is active
        corner_div_damp_d4_bg=0.02,
        corner_div_damp_nord=1,            # iter-187 cap path
        corner_div_damp_dt_proxy=dt_proxy, # the proxy under test
        div_damp_coeff=1e6, div_damp_dddmp=0.20,
        damp_v=0.030, nord_v=2,
        use_fv3_a2b_zeta_corner=True,
        A_h=1e6, smagorinsky_cs=0.20,
        T_diss_coeff=0.05,
        hyperdiff_coeff=1e16, hyperdiff_ps_coeff=0.0,
        damp_v_d_con=1.0,
        corner_div_damp_d_con=1.0,
        div_damp_d_con=1.0,
        ah_d_con=1.0,
        delt_max=1.0,
        use_conservation_fixer=False, fix_mass=False,
        zero_mean_ps_tendency=False,
    )


def test_pe_step_uses_dt_actual_over_proxy():
    """``model.step(state, dt)`` must use real dt, not
    ``corner_div_damp_dt_proxy`` from the config.

    Two configs differ ONLY in ``corner_div_damp_dt_proxy``
    (50 vs 500).  Both must produce identical step output when
    ``model.step`` is called with the same dt — confirms iter-
    189 dt plumbing wins over the config proxy.
    """
    grid, coord, state = _make_state(seed=296)

    cfg_proxy_50 = _build_cfg(dt_proxy=50.0)
    cfg_proxy_500 = _build_cfg(dt_proxy=500.0)

    model_50 = CDGridPrimitiveEquationModel(grid, coord, cfg_proxy_50)
    model_500 = CDGridPrimitiveEquationModel(grid, coord, cfg_proxy_500)

    # Same dt for both — must beat the divergent proxy values.
    dt = 100.0
    s_50 = model_50.step(state, dt)
    s_500 = model_500.step(state, dt)

    # Outputs must be bit-for-bit identical.  If the real dt
    # plumbing has regressed and the cap is reading the proxy,
    # the divergence in proxy values (50 vs 500) propagates into
    # different damping → different state.
    np.testing.assert_array_equal(
        np.asarray(s_50.u_d.data), np.asarray(s_500.u_d.data),
    )
    np.testing.assert_array_equal(
        np.asarray(s_50.v_d.data), np.asarray(s_500.v_d.data),
    )
    np.testing.assert_array_equal(
        np.asarray(s_50.T.data), np.asarray(s_500.T.data),
    )
