"""FV3_3D iter 298: PE iter-189 dt-proxy fallback semantics.

iter-296 pinned that ``model.step(state, dt)`` plumbs the
actual integration ``dt`` through to the corner-div damping
adaptive cap, BYPASSING ``config.corner_div_damp_dt_proxy``.

iter-298 pins the COMPLEMENT: when callers invoke
``fv3_hydrostatic_tendencies`` directly WITHOUT
``dt_actual=...``, the proxy IS used (preserving the
backward-compat fallback documented at
``primitive_eq_cdgrid.py:830``).

Together iter-296 + iter-298 form a complete contract:

    dt_actual present  → real dt wins (iter-296)
    dt_actual absent   → proxy wins (iter-298)

If a future refactor accidentally hardcodes a single dt path
(e.g., always proxy or always None), one of these two
regression guards catches it.

Tests
-----

1. ``test_pe_tendency_uses_proxy_when_no_dt_actual`` — call
   ``fv3_hydrostatic_tendencies`` directly with two configs
   that differ ONLY in ``corner_div_damp_dt_proxy``.  When
   no ``dt_actual`` is passed, tendencies must DIFFER (proxy
   is being read).
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
    return grid, cdgrid, coord, state


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


def test_pe_tendency_uses_proxy_when_no_dt_actual():
    """Direct ``fv3_hydrostatic_tendencies`` calls without
    ``dt_actual`` must read ``corner_div_damp_dt_proxy``.

    Two configs differ only in proxy (50 vs 500); call the
    tendency fn directly.  If proxies are read, tendencies
    differ; if proxies are silently ignored, tendencies are
    identical and this test fails — catching a regression in
    the fallback semantics.
    """
    grid, cdgrid, coord, state = _make_state(seed=298)

    cfg_proxy_50 = _build_cfg(dt_proxy=50.0)
    cfg_proxy_500 = _build_cfg(dt_proxy=500.0)

    tend_50 = fv3_hydrostatic_tendencies(
        state, grid, coord, cdgrid, cfg_proxy_50,
        # no dt_actual — fallback path engages
    )
    tend_500 = fv3_hydrostatic_tendencies(
        state, grid, coord, cdgrid, cfg_proxy_500,
    )

    diff = float(jnp.max(jnp.abs(
        tend_50.du_d_dt.data - tend_500.du_d_dt.data,
    )))
    base = float(jnp.max(jnp.abs(tend_50.du_d_dt.data)))
    assert diff > 1e-10 * max(base, 1e-10), (
        f"Direct tendency call without dt_actual must read "
        f"corner_div_damp_dt_proxy.  Got identical tendencies "
        f"with proxy 50 vs 500 (diff={diff:.3e}, base="
        f"{base:.3e}); proxy fallback path may be broken."
    )
    assert jnp.all(jnp.isfinite(tend_50.du_d_dt.data))
    assert jnp.all(jnp.isfinite(tend_500.du_d_dt.data))
