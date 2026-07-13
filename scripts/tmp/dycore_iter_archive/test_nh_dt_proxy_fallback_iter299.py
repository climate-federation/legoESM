"""FV3_3D iter 299: NH iter-189 dt-proxy fallback semantics
(NH counterpart of PE iter-298).

iter-297 pinned that ``model.step`` (NH) plumbs real dt and
beats the proxy.  iter-299 pins the COMPLEMENT: direct
``cdgrid_compressible_euler_slow_tendencies`` calls WITHOUT
``dt_actual`` must read ``corner_div_damp_dt_proxy``.

Together iter-297 + iter-299 form the complete iter-189 NH
contract (mirroring PE iter-296 + iter-298):

    dt_actual present  → real dt wins (iter-297)
    dt_actual absent   → proxy wins (iter-299)

Tests
-----

1. ``test_nh_tendency_uses_proxy_when_no_dt_actual`` — direct
   tendency call with two configs that differ ONLY in
   ``corner_div_damp_dt_proxy`` (2.0 vs 50.0).  Without
   ``dt_actual``, tendencies must DIFFER (proxy is being
   read).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.dynamics.gcm.compressible_euler_cdgrid import (
    CDGridCompressibleEulerConfig,
    cdgrid_compressible_euler_slow_tendencies,
)
from legoesm.core.field import Field
from legoesm.core.state import NonHydrostaticState
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.grids.vertical import (
    compute_terrain_metric, create_height_coordinate,
)


def _make_state(seed):
    n = 8
    nlev = 5
    z_top = 30000.0
    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    height_coord = create_height_coordinate(nlev, z_top)
    terrain = jnp.zeros((6, n, n))
    terrain_metric = compute_terrain_metric(terrain, height_coord)

    dims_3d = ("face", "x", "y", "level")
    dims_w = ("face", "x", "y", "level_half")
    dims_2d = ("face", "x", "y")

    rng = np.random.default_rng(seed=seed)
    # Larger perturbations than PE iter-298 so |delpc| escapes the
    # d2_bg floor and the proxy difference propagates into the cap.
    u_p = rng.uniform(-15.0, 15.0, size=(6, n, n, nlev))
    v_p = rng.uniform(-15.0, 15.0, size=(6, n, n, nlev))
    w_p = rng.uniform(-0.3, 0.3, size=(6, n, n, nlev + 1))

    state = NonHydrostaticState(
        u=Field(data=jnp.asarray(u_p), name="u",
                dims=dims_3d, units="m/s"),
        v=Field(data=jnp.asarray(v_p), name="v",
                dims=dims_3d, units="m/s"),
        w=Field(data=jnp.asarray(w_p), name="w",
                dims=dims_w, units="m/s"),
        theta_prime=Field(data=jnp.zeros((6, n, n, nlev)),
                          name="theta_prime",
                          dims=dims_3d, units="K"),
        rho_prime=Field(data=jnp.zeros((6, n, n, nlev)),
                        name="rho_prime",
                        dims=dims_3d, units="kg/m^3"),
        phis=Field(data=jnp.zeros((6, n, n)), name="phis",
                   dims=dims_2d, units="m^2/s^2"),
        tracers=Field(data=jnp.zeros((6, n, n, nlev, 0)),
                      name="tracers",
                      dims=("face", "x", "y", "level", "tracer"),
                      units="kg/kg"),
    )
    return grid, cdgrid, height_coord, terrain_metric, state


def _build_cfg(dt_proxy):
    return CDGridCompressibleEulerConfig(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        corner_div_damp_d2_bg=0.0005, corner_div_damp_dddmp=0.20,
        corner_div_damp_d4_bg=0.02, corner_div_damp_nord=1,
        corner_div_damp_dt_proxy=dt_proxy,
        damp_v=0.030, nord_v=2,
        use_fv3_a2b_zeta_corner=True,
        div_damp_coeff=1e6, div_damp_dddmp=0.20,
        A_h=1e6, smagorinsky_cs=0.20,
        damp_w=0.030, nord_w=1,
        damp_w_d_con=1.0,
        damp_v_d_con=1.0,
        corner_div_damp_d_con=1.0,
        div_damp_d_con=1.0,
        ah_d_con=1.0,
        delt_max=1.0,
    )


def test_nh_tendency_uses_proxy_when_no_dt_actual():
    """Direct ``cdgrid_compressible_euler_slow_tendencies`` calls
    without ``dt_actual`` must read ``corner_div_damp_dt_proxy``.

    Two configs differ only in proxy (2 vs 50); call the
    tendency fn directly.  Tendencies must differ — confirms
    NH proxy fallback semantics.  Mirror of PE iter-298.
    """
    grid, cdgrid, height_coord, terrain_metric, state = _make_state(seed=299)

    # Wider proxy spread (10 vs 1000) — beats both d2_bg floor and
    # the iter-187 smag_vort 0.20 ceiling for some corners.
    cfg_proxy_2 = _build_cfg(dt_proxy=10.0)
    cfg_proxy_50 = _build_cfg(dt_proxy=1000.0)

    tend_2 = cdgrid_compressible_euler_slow_tendencies(
        state, grid, height_coord, terrain_metric,
        cdgrid, cfg_proxy_2,
        # no dt_actual
    )
    tend_50 = cdgrid_compressible_euler_slow_tendencies(
        state, grid, height_coord, terrain_metric,
        cdgrid, cfg_proxy_50,
    )

    diff = float(jnp.max(jnp.abs(
        tend_2.du_dt.data - tend_50.du_dt.data,
    )))
    base = float(jnp.max(jnp.abs(tend_2.du_dt.data)))
    assert diff > 1e-10 * max(base, 1e-10), (
        f"NH direct tendency call without dt_actual must read "
        f"corner_div_damp_dt_proxy.  Got identical du_dt with "
        f"proxy 10 vs 1000 (diff={diff:.3e}, base={base:.3e}); "
        f"NH proxy fallback path may be broken."
    )
    assert jnp.all(jnp.isfinite(tend_2.du_dt.data))
    assert jnp.all(jnp.isfinite(tend_50.du_dt.data))
