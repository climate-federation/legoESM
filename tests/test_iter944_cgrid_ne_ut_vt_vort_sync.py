"""Iter-944 sentinel: pin the FB chain 1-day NaN-free milestone from
adding CGRID_NE syncs of (ut, vt) at step 1 and (fx_vort, fy_vort) at
step 7 inside `_d_sw_native`.

Cumulative FB chain step survival on W2 C36 dt=300 s (target 1-day =
288 steps), default damping (d4_bg=0.16, nord=1, damp_v=0.06, nord_v=2):

| iter      | survived | improvement |
|-----------|----------|-------------|
| baseline  | 41       | —           |
| iter-941  | 63       | +54 %       |
| iter-942  | 209      | +410 %      |
| iter-944  | 288      | +602 %      |

iter-944 added two CGRID_NE syncs (`synchronize_cgrid_fluxes`) inside
`_d_sw_native`:

1. ut, vt sync after `_d_sw1_recompute_ut_vt` (step 1).  ut, vt are
   contravariant transport velocities at C-grid u-face / v-face
   positions; the face-local upwind sin_sg boundary overrides leave
   cube-edge cells inconsistent across face pairs.

2. fx_vort, fy_vort sync after the vorticity-flux fv_tp_2d call
   (step 7).  iter-864 set `apply_cgrid_flux_sync=False` to match
   Fortran's commented-out averaging block (dyn_core.F90:1124-1207),
   but the duogrid gate inside `fv_tp_2d` would have skipped the sync
   for non-duogrid grids regardless.  Forcing the sync explicitly
   (bypassing the duogrid gate) closes the cube-edge inconsistency
   that the d_sw6 wind update propagates into a structural growth
   mode.

Caveat: 288-step "1-day NaN-free" is NUMERICAL stability, not W2
acceptance.  Velocities at 1 day reach |v|≈2300 m/s (analytical ≈0),
heights drift to ~41 km (analytical ~3 km).  iter-944 closes the
NaN blocker; W2 acceptance (v_ll_Linf ≤ 0.119 m/s per user iter-938
brief) is the iter-945+ territory.

Production impact: ZERO.  `_d_sw_native` is FB-chain-only; production
`fv3_sw_tendencies` (`FV3EdgeShallowWaterModel` default) does not
invoke this code path.  Production W2 sentinel (iter-921) and
rest-state sentinel (iter-925) bit-identical post-iter-944.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import warnings

import jax.numpy as jnp
import numpy as np

from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import (
    CDGridShallowWaterConfig,
    FV3EdgeShallowWaterState,
    FV3FBShallowWaterModel,
)
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from tests.atmosphere.shallow_water.test_cases.williamson import (
    williamson_test2,
)


def _fb_chain_survival(N: int, dt: float, max_steps: int) -> int:
    grid = create_cubed_sphere(N)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    sw = williamson_test2(grid)
    u0 = 2.0 * jnp.pi * grid.radius / (12.0 * 86400.0)
    u_d = cdgrid.cos_angle_edge_x * (u0 * jnp.cos(cdgrid.lat_edge_x))
    v_d = -cdgrid.sin_angle_edge_y * (u0 * jnp.cos(cdgrid.lat_edge_y))
    state = FV3EdgeShallowWaterState(
        h=sw.h.data, u_d=u_d, v_d=v_d, h_s=sw.h_s.data,
    )
    cfg = CDGridShallowWaterConfig(
        hyperdiff_coeff=0.0,
        d2_bg=0.0, dddmp=0.0, d4_bg=0.16, nord=1,
        damp_v=0.06, nord_v=2,
    )
    model = FV3FBShallowWaterModel(grid, cfg)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for k in range(max_steps):
            state = model.step(state, dt)
            if not bool(np.all(np.isfinite(np.asarray(state.h)))):
                return k
    return max_steps


def test_iter944_fb_chain_c36_reaches_1_day_nan_free():
    """Post-iter-944 FB chain at C36 dt=300 s must reach the 1-day target
    (288 steps) NaN-free.  Pre-iter-944 baseline: 209 steps (iter-942).

    This is the structural-growth-NaN-blocker close.  W2 acceptance
    (v_ll_Linf ≤ 0.119 m/s) is NOT achieved at 1 day — velocities
    still drift to ~3000 m/s.  Firing this gate signals a regression
    to the iter-944 CGRID_NE syncs of ut/vt or fx_vort/fy_vort.
    """
    n = _fb_chain_survival(36, 300.0, max_steps=288)
    assert n >= 288, (
        f"FB chain at C36 dt=300 s survived only {n} steps; iter-944 "
        f"baseline expected >= 288 (1-day NaN-free target).  iter-942 "
        f"baseline was 209."
    )
