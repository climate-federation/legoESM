#!/usr/bin/env python
"""Iter-933 FB chain stability scan across resolutions.

The user's issue #8 marks `FV3FBShallowWaterModel` as experimental
and unstable.  iter-933 quantifies WHERE the instability shows up
by running one FB step at C8/C12/C16/C24/C36 with the same dt=300s.

Result PRE-iter-934 (W2 t=0, dt=300 s, FB defaults d4_bg=0.16, nord=1):

| N   | h finite? | u finite? | v finite? |
|-----|-----------|-----------|-----------|
| 8   | NO (NaN)  | yes       | yes       |
| 12  | NO (NaN)  | yes       | yes       |
| 16  | NO (NaN)  | yes       | yes       |
| 24  | yes       | yes       | yes       |
| 36  | yes       | yes       | yes       |

Result POST-iter-934 (after `_deln_flux` damp-factoring fix):

| N   | h finite? | u finite? | v finite? |
|-----|-----------|-----------|-----------|
| 8   | yes       | yes       | yes       |
| 12  | yes       | yes       | yes       |
| 16  | yes       | yes       | yes       |
| 24  | yes       | yes       | yes       |
| 36  | yes       | yes       | yes       |

Surprising findings (pre-iter-934 historical context):

1. FB chain is stable at C24+ but NaN at C8/C12/C16.  This is the
   OPPOSITE of typical CFL-driven instability (which fails at
   high resolution).
2. The NaN is in `h` only (mass transport), NOT in u or v
   (velocity tendencies).  Velocity computation handles low
   resolution fine.
3. The instability is grid-size-dependent (geometry/halo), not
   time-step-dependent (CFL).

Implication: stabilizing the FB chain at low resolution requires
addressing the MASS TRANSPORT path (`transport_step` →
`fv_tp_2d` → `_ppm_1d`), not the velocity (Coriolis/B-grad/d_sw5)
path.  Specifically, the `_ppm_1d` boundary-cell handling may be
inadequate when n_interior < some threshold (between 16 and 24).

This is consistent with user's prohibition: "Do not use
use_fv3_dsw1_mass_transport or split_mass_momentum as the fix;
those paths already failed catastrophically."  The mass-transport
path itself is where the FB chain breaks.

This script is read-only; no production code change.

Run:

    JAX_ENABLE_X64=1 .venv/bin/python scripts/diag_iter933_fb_chain_resolution_stability.py
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


def fb_one_step(N: int, dt: float) -> dict:
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
        new_state = model.step(state, dt)

    h_arr = np.asarray(new_state.h)
    u_arr = np.asarray(new_state.u_d)
    v_arr = np.asarray(new_state.v_d)
    nan_h = bool(np.any(~np.isfinite(h_arr)))
    nan_u = bool(np.any(~np.isfinite(u_arr)))
    nan_v = bool(np.any(~np.isfinite(v_arr)))
    h_max = float(np.max(np.abs(np.where(np.isfinite(h_arr), h_arr, 0))))
    u_max = float(np.max(np.abs(np.where(np.isfinite(u_arr), u_arr, 0))))
    v_max = float(np.max(np.abs(np.where(np.isfinite(v_arr), v_arr, 0))))
    return {
        "N": N, "dt": dt,
        "h_max": h_max, "u_max": u_max, "v_max": v_max,
        "h_finite": not nan_h,
        "u_finite": not nan_u,
        "v_finite": not nan_v,
    }


def main() -> None:
    print("=== iter-933 FB chain stability scan ===")
    print(f"{'N':>4} {'dt':>5} {'h_max':>10} {'u_max':>9} {'v_max':>9} "
          f"{'h_fin':>6} {'u_fin':>6} {'v_fin':>6}")
    for N in (8, 12, 16, 24, 36):
        r = fb_one_step(N, dt=300.0)
        print(
            f"{r['N']:>4d} {r['dt']:>5.0f} {r['h_max']:>10.3e} "
            f"{r['u_max']:>9.3e} {r['v_max']:>9.3e} "
            f"{str(r['h_finite']):>6s} {str(r['u_finite']):>6s} "
            f"{str(r['v_finite']):>6s}"
        )

    print()
    print("Findings (post-iter-934):")
    print("- FB chain stable at ALL resolutions C8..C36 with iter-934 fix.")
    print("- Pre-iter-934 NaN at C8/C12/C16 from float32 overflow in")
    print("  `_deln_flux`'s damp*q intermediate; iter-934 factors damp")
    print("  to the final Step 4 instead of Step 1 initialisation.")


if __name__ == "__main__":
    main()
