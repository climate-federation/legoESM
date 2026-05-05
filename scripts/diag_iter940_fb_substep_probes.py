#!/usr/bin/env python
"""Iter-940 substep probes — localise FB chain structural growth.

iter-935 found FB chain blows up at step 41 on C36 1-day target.
iter-936 found velocities grow first (|u|, |v| double at step 31
before h goes out of range).  iter-940 narrows further by
temporarily ZEROING individual contributions inside `_d_sw_native`
and measuring step survival.

Probe results (W2 C36 dt=300 s, default damp_v=0.06, nord_v=2,
d4_bg=0.16, nord=1):

| modification                                 | survived |
|----------------------------------------------|----------|
| BASELINE (full FB chain)                     | 41 steps |
| vorticity flux sync ON (was OFF per iter-864)| 41 steps |
| `d4_bg=0` (no d_sw5 corner damping)          | 41 steps |
| `damp_v=0` (no del6_vt post-step)            | 41 steps |
| `damp_v=2.0` (huge del6_vt — destabilises)   |  3 steps |
| `fy_vort=fx_vort=0` (zero vorticity flux)    | 43 steps |
| **`u_d_new = u_d` (skip d_sw6 wind update)** | **120+** |
| `ke_diff_u/v_scaled=0` (skip KE-grad ONLY)   | **120+** |

**Key finding**: zeroing the d_sw6 KE-gradient contribution
(`ke_corner[i] - ke_corner[i+1]`) makes the FB chain stable for
120+ steps.  Vorticity transport contributes a small growth (43 vs
41) but the dominant structural blow-up source is the KE-gradient.

**Localisation**: `_d_sw_native` step 6 (`ke_diff_u_scaled =
ke_corner[:, :-1, :] - ke_corner[:, 1:, :]`) wired into the wind
update at lines 2426-2427 of `fv3_sw_core.py`.

`ke_corner` itself is computed by `_bgrid_ke_transport(u_d, v_d,
uc, vc, cdgrid, dt)` (step 4).  The bug is either in:

1. `_bgrid_ke_transport` — wrong KE staggering, halo, or
   advection scheme on the cubed sphere; or
2. The d_sw6 formula at lines 2397-2398 + 2426-2427 — wrong
   gradient sign, scaling, or stagger on our staggered grid.

Future iter (941+) should:
1. Compare our `_bgrid_ke_transport` against Fortran sw_core.F90:
   1201-1388 (d_sw3 B-grid KE transport).  Check stagger and halo.
2. Verify the `(ke_corner[:, :-1, :] - ke_corner[:, 1:, :]) / dx_u`
   formula matches Fortran's `(ke(i,j) - ke(i+1,j))` interpretation.

This script is read-only — the probes are restored to the source
file's bit-identical baseline.  Reproducible by re-running the
experiments documented above (each requires manual edits to
`_d_sw_native` to zero the relevant contribution).

Run:

    JAX_ENABLE_X64=1 .venv/bin/python scripts/diag_iter940_fb_substep_probes.py
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


def fb_survival(N: int, dt: float, max_steps: int = 80, **cfg_kwargs) -> int:
    grid = create_cubed_sphere(N)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    sw = williamson_test2(grid)
    u0 = 2.0 * jnp.pi * grid.radius / (12.0 * 86400.0)
    u_d = cdgrid.cos_angle_edge_x * (u0 * jnp.cos(cdgrid.lat_edge_x))
    v_d = -cdgrid.sin_angle_edge_y * (u0 * jnp.cos(cdgrid.lat_edge_y))
    state = FV3EdgeShallowWaterState(
        h=sw.h.data, u_d=u_d, v_d=v_d, h_s=sw.h_s.data,
    )
    base = dict(
        hyperdiff_coeff=0.0,
        d2_bg=0.0, dddmp=0.0, d4_bg=0.16, nord=1,
        damp_v=0.06, nord_v=2,
    )
    base.update(cfg_kwargs)
    cfg = CDGridShallowWaterConfig(**base)
    model = FV3FBShallowWaterModel(grid, cfg)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for k in range(max_steps):
            state = model.step(state, dt)
            if not bool(np.all(np.isfinite(np.asarray(state.h)))):
                return k
    return max_steps


def main() -> None:
    print("=== iter-940 baseline-survival sweep at C36 dt=300 s ===")
    print(f"{'config':<55s} {'survived':>10s}")
    sweeps = [
        ("default (d4=0.16, nord=1, damp_v=0.06)", {}),
        ("d4_bg=0 (no d_sw5 corner damping)", dict(d4_bg=0.0)),
        ("damp_v=0 (no del6_vt post-step)", dict(damp_v=0.0)),
        ("d4_bg=0 + damp_v=0 (no damping at all)",
         dict(d4_bg=0.0, damp_v=0.0)),
    ]
    for label, override in sweeps:
        n = fb_survival(36, 300.0, max_steps=50, **override)
        print(f"{label:<55s} {n:>10d}")

    print()
    print("Key finding (verified by manual probes — see docstring):")
    print("  Zeroing d_sw6 KE-gradient → FB chain survives 120+ steps.")
    print("  Localises structural growth to `_bgrid_ke_transport` or")
    print("  the KE-gradient stencil at fv3_sw_core.py:2397-2398.")


if __name__ == "__main__":
    main()
