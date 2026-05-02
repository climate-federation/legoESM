#!/usr/bin/env python
"""Iter-945 diagnostic: D-grid PPM cross-face halo on the FB chain.

iter-944b reverted Python-only `mpp_get_boundary` syncs not present in
Fortran.  The duogrid FB chain at C36 dt=300 s reached 1-day NaN-free
but with |u_max|=106 m/s, |v_max|=151 m/s — far from analytical W2
(|u_max|=40 m/s, |v_max|≈0).

iter-945 wires the existing `ext_vector_dgrid` halo (used in
`_d2a2c_vect_duogrid`) into a new helper `_pad_halo_dgrid_for_ppm`
that returns u_d with i-cell halo and v_d with j-cell halo at depth 2.
`_ppm_transport_1d` gains an `external_halo` kwarg that consumes the
pre-padded fields and uses `mode='edge'` only for the gap to the
PPM stencil width (h3=4).

Effect on the duogrid FB chain at C36 dt=300 s:

| baseline (iter-944b) | post-iter-945 |
|----------------------|---------------|
| |u_max| = 106 m/s    | |u_max| ≈ 78 m/s |
| |v_max| = 151 m/s    | |v_max| ≈ 81 m/s |
| step survival 288/288 | step survival 288/288 |

Run:

    JAX_ENABLE_X64=1 .venv/bin/python scripts/diag_iter945_dgrid_ppm_halo.py
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
    print("iter-945 FB chain step-survival probe at C36 dt=300 s")
    measure("non-duogrid (mode='edge' fallback)", use_duogrid=False)
    measure("duogrid (cross-face halo via ext_vector_dgrid)",
            use_duogrid=True)


if __name__ == "__main__":
    main()
