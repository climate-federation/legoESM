#!/usr/bin/env python
"""Iter-935 FB chain long-term instability characterisation.

iter-934 fixed `_deln_flux`'s float32 overflow → FB chain produces
finite output at step 1 for all resolutions C8..C36.  iter-935
asks the next question: does FB chain stay finite over a full
W2 1-day integration?

Result: NO.  FB chain blows up after 40-90 steps at every resolution:

| N   | dt  | survived steps | h_max at end |
|-----|-----|----------------|---------------|
| 8   | 1350 | 49             | 1.46e+10      |
| 12  | 900  | 67             | 1.52e+20      |
| 16  | 675  | 84             | 1.12e+08      |
| 24  | 450  | 42             | 1.69e+13      |
| 36  | 300  | 41             | 5.13e+19      |

This is EXPLOSIVE growth (h_max from ~3000 m → 1e10..1e20 m), not
a slow-mode underdamped instability.  Damping coefficient sweeps
at C36 confirm this is NOT a damping-insufficiency:

| damping config                          | survived |
|-----------------------------------------|----------|
| default (d4=0.16, nord=1, damp_v=0.06)  | 41 / 50  |
| stronger del-4 (d4=0.5)                 |  9 / 50  |
| add d2_bg=0.05                          | 41 / 50  |
| aggressive Smag (dddmp=0.4)             | 41 / 50  |
| higher damp_v=0.2, nord_v=2             | 18 / 50  |

Higher damping makes it WORSE (d4=0.5 → 9 steps; damp_v=0.2 →
18 steps).  This refutes "underdamped slow mode" and points to a
STRUCTURAL bug in one of the d_sw1/d_sw4/d_sw5/d_sw6 sub-operators.

Implication: iter-934's fix is necessary but not sufficient for
FB chain production-readiness.  Closing the gap requires deeper
debugging — tracing which operator step contributes to the
explosive growth between steps 30 and 50.

This script is read-only.  Output is text; no PNG.
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


def fb_survival(N: int, dt: float, max_steps: int = 300, **cfg_kwargs) -> dict:
    grid = create_cubed_sphere(N)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    sw = williamson_test2(grid)
    u0 = 2.0 * jnp.pi * grid.radius / (12.0 * 86400.0)
    u_d = cdgrid.cos_angle_edge_x * (u0 * jnp.cos(cdgrid.lat_edge_x))
    v_d = -cdgrid.sin_angle_edge_y * (u0 * jnp.cos(cdgrid.lat_edge_y))
    state = FV3EdgeShallowWaterState(
        h=sw.h.data, u_d=u_d, v_d=v_d, h_s=sw.h_s.data,
    )
    base_cfg = dict(
        hyperdiff_coeff=0.0,
        d2_bg=0.0, dddmp=0.0, d4_bg=0.16, nord=1,
        damp_v=0.06, nord_v=2,
    )
    base_cfg.update(cfg_kwargs)
    cfg = CDGridShallowWaterConfig(**base_cfg)
    model = FV3FBShallowWaterModel(grid, cfg)
    h_max_log = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for k in range(max_steps):
            state = model.step(state, dt)
            h = np.asarray(state.h)
            if not np.all(np.isfinite(h)):
                return {"survived": k, "h_max_end": (h_max_log[-1]
                                                      if h_max_log else None)}
            h_max_log.append(float(np.max(np.abs(h))))
    return {"survived": max_steps, "h_max_end": h_max_log[-1]}


def main() -> None:
    print("=== iter-935 FB chain long-term instability ===")
    print()
    print("Resolution scan (default damping, target 1-day):")
    print(f"{'N':>4} {'dt':>5} {'target steps':>13} {'survived':>10} "
          f"{'h_max@end':>11}")
    cases = [(8, 1350.0, 64), (12, 900.0, 96), (16, 675.0, 128),
             (24, 450.0, 192), (36, 300.0, 288)]
    for N, dt, target in cases:
        r = fb_survival(N, dt, max_steps=target)
        end = r["h_max_end"]
        end_str = f"{end:.3e}" if end is not None else "N/A"
        print(f"{N:>4d} {dt:>5.0f} {target:>13d} {r['survived']:>10d} "
              f"{end_str:>11s}")

    print()
    print("Damping sweep at C36 (50 steps target):")
    print(f"{'config':<48s} {'survived':>10s}")
    sweeps = [
        ("default (d4=0.16, nord=1, damp_v=0.06)", {}),
        ("stronger del-4 (d4=0.5)",
         dict(d4_bg=0.5)),
        ("add d2_bg=0.05",
         dict(d2_bg=0.05)),
        ("aggressive Smag (dddmp=0.4)",
         dict(dddmp=0.4)),
        ("higher damp_v=0.2",
         dict(damp_v=0.2)),
    ]
    for label, override in sweeps:
        r = fb_survival(36, 300.0, max_steps=50, **override)
        print(f"{label:<48s} {r['survived']:>10d}")

    print()
    print("Interpretation:")
    print("- Higher damping makes instability WORSE (d4=0.5 → 9 steps).")
    print("- This refutes 'underdamped slow mode' hypothesis.")
    print("- Points to STRUCTURAL bug in one of d_sw1/d_sw4/d_sw5/d_sw6")
    print("  sub-operators that takes 30-50 steps to amplify.")


if __name__ == "__main__":
    main()
