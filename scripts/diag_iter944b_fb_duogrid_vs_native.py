#!/usr/bin/env python
"""Iter-944b diagnostic: FB chain step survival on duogrid vs non-duogrid.

iter-944b reverted iter-942's `synchronize_corner_scalar(ke_corner)`
unconditional sync and iter-944's CGRID_NE syncs of (ut, vt) and
(fx_vort, fy_vort), and gated iter-941's BGRID_NE sync on `duogrid`.
All three matched Fortran reference source on the audit:

- BGRID_NE sync (`dyn_core.F90:968-1011`) is INSIDE `if (duogrid)`.
- ke_corner sync (`dyn_core.F90:1029-1055, 1180-1207`) is COMMENTED OUT
  even within the duogrid branch.
- vorticity-flux CGRID_NE sync (`dyn_core.F90:1124-1207`) is COMMENTED
  OUT even within the duogrid branch.
- (ut, vt) standalone CGRID_NE sync: NOT in Fortran reference at all.

This script measures FB chain step survival on C36 W2 dt=300 s for both
the non-duogrid grid (where no syncs should fire per Fortran) and the
duogrid grid (where the BGRID_NE sync fires).

Run:

    JAX_ENABLE_X64=1 .venv/bin/python scripts/diag_iter944b_fb_duogrid_vs_native.py
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


def fb_survival(N: int, dt: float, max_steps: int, *, use_duogrid: bool):
    grid = create_cubed_sphere(N, use_duogrid=use_duogrid)
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


def measure(label: str, *, use_duogrid: bool):
    print(f"\n=== {label} ===")
    n, state = fb_survival(36, 300.0, max_steps=288,
                           use_duogrid=use_duogrid)
    h = np.asarray(state.h)
    u = np.asarray(state.u_d)
    v = np.asarray(state.v_d)
    print(f"  step survival: {n}/288")
    print(f"  final h range:  [{h.min():9.3f}, {h.max():9.3f}] m  "
          f"(W2 IC: ~1000..3000)")
    print(f"  final |u_max|:  {np.abs(u).max():9.2f} m/s  "
          f"(W2 IC: 40)")
    print(f"  final |v_max|:  {np.abs(v).max():9.2f} m/s  "
          f"(W2 IC: ~0)")


def main() -> None:
    print("iter-944b FB chain step-survival probe at C36 dt=300 s")
    measure("non-duogrid (Fortran: NO syncs fire)", use_duogrid=False)
    measure("duogrid (Fortran: BGRID_NE sync fires)", use_duogrid=True)


if __name__ == "__main__":
    main()
