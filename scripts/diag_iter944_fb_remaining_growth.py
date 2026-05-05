#!/usr/bin/env python
"""Iter-944 diagnostic: FB chain step-survival + 1-day NaN-free milestone.

iter-942 reached 209 FB chain steps on W2 C36 dt=300 s (target 1-day = 288).
iter-943 found that syncing `ubbtemp, vbb` (vector or scalar) and pre-d_sw5
ke_corner sync did NOT help.  iter-944 located the remaining gap in the
CGRID_NE flux/Courant sync at FB chain step 1 (ut, vt) and step 7
(fx_vort, fy_vort).  Both syncs were already implemented in
`synchronize_cgrid_fluxes` but were either guarded by the `apply_cgrid_flux_sync=False`
kwarg (vorticity flux, iter-864) or the duogrid gate inside `fv_tp_2d`
(non-duogrid grids skipped the sync regardless of the kwarg).

Probe results at C36 dt=300 s W2 1-day target:

| state                                              | survived |
|---------------------------------------------------|----------|
| iter-942 baseline                                 | 208–209  |
| + apply_cgrid_flux_sync=True on vort flux only    | 208      |  ← duogrid gate skips
| + sync ke_damping before add (probe B, redundant) | 208      |
| + force vort flux sync (bypass gate, probe C)     | 270      |  ← +29 %
| + ungate fv_tp_2d sync only (probe D, no probe C) | 187      |  ← regresses
| + probes C + D (vort + mass flux sync)            | 279      |
| **+ probes C + E (vort sync + ut/vt sync at step 1)** | **288** |  ← **1-day NaN-free**

Caveat: 288-step "1-day NaN-free" is NUMERICAL stability, not W2
acceptance.  At 1 day:
  - h range:    [-323, 41796] m  (W2 IC: ~1000..3000 m)
  - |u_max|:    3044 m/s         (analytical: 40 m/s)
  - |v_max|:    2278 m/s         (analytical: ~0 m/s)

The FB chain runs through 1 day without producing NaN, but the
W2 v_ll_Linf acceptance criterion (≤ 0.119 m/s per user iter-938
brief) is still failed by 4 orders of magnitude.  iter-944 closes
the structural NaN-blocker; iter-945+ work is to close the W2
fidelity gap (probably PPM hord=9 boundary handling, the
operator-split sweep order in `_bgrid_ke_transport`, or the cube-
vertex halo for u_d/v_d themselves).

Run:

    JAX_ENABLE_X64=1 .venv/bin/python scripts/diag_iter944_fb_remaining_growth.py
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

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


def fb_survival(N: int, dt: float, max_steps: int):
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
                return k, state
    return max_steps, state


def main() -> None:
    print("=== iter-944 FB chain step-survival probe at C36 dt=300 s ===")
    n, state = fb_survival(36, 300.0, max_steps=288)
    print(f"\nFB chain survived: {n} steps (target 1-day = 288)")
    h = np.asarray(state.h)
    u = np.asarray(state.u_d)
    v = np.asarray(state.v_d)
    print(f"final h range: [{h.min():.3f}, {h.max():.3f}] m  "
          f"(W2 IC: 1000..3000 m)")
    print(f"final |u_max|: {np.abs(u).max():.2f} m/s  "
          f"(W2 IC: |u_max|=40 m/s)")
    print(f"final |v_max|: {np.abs(v).max():.2f} m/s  "
          f"(W2 IC: |v_max|=0 m/s)")
    if n >= 288:
        print("  -> 1-DAY NAN-FREE (FB chain reaches W2 1-day target).")
        print("     W2 acceptance v_ll_Linf still failed (analytical |v|≈0).")
    elif n >= 209:
        print("  -> matches/exceeds iter-942 baseline (209).")
    elif n >= 41:
        print(f"  -> regression vs iter-942 baseline (209).")
    else:
        print("  -> catastrophic regression vs iter-942 baseline.")


if __name__ == "__main__":
    main()
