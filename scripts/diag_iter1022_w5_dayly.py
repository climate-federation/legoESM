"""Iter-1022: W5 day-by-day stability with iter-1021 calibration.

Confirms iter-1021 is artifact-free through day 5:
  day 1-4: speed < 40 m/s (Rossby wave development)
  day 5: speed = 53.3 m/s (still under 80 threshold)
  day 6: speed = 184 m/s (instability onset)
  day 9: h goes negative (-28 m)

W5 day-6+ instability is structural and independent of
(div_factor, damp_v) calibration — same family as iter-989's
FB-chain finding.
"""
from __future__ import annotations

import os
import warnings

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np

from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import (
    FV3EdgeShallowWaterModel,
    FV3EdgeShallowWaterState,
    iter1009_dual_target_config,
)
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from tests.atmosphere.shallow_water.test_cases.williamson import (
    williamson_test5,
)


def main():
    N = 36
    DT = 300.0
    grid = create_cubed_sphere(N)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    sw = williamson_test5(grid)
    u_d = cdgrid.cos_angle_edge_x * (
        20.0 * jnp.cos(cdgrid.lat_edge_x))
    v_d = -cdgrid.sin_angle_edge_y * (
        20.0 * jnp.cos(cdgrid.lat_edge_y))
    state = FV3EdgeShallowWaterState(
        h=sw.h.data, u_d=u_d, v_d=v_d, h_s=sw.h_s.data,
    )
    cfg = iter1009_dual_target_config(N)
    model = FV3EdgeShallowWaterModel(grid, cfg)
    model.set_initial_mass(state)

    print("=== Iter-1022 W5 day-by-day with iter-1021 calibration ===")
    print(f"{'day':>3} | h_min   h_max  | speed_max | OK?")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for day in range(1, 11):
            for _ in range(int(86400 / DT)):
                state = model.step(state, DT)
            h = np.asarray(state.h)
            u = np.asarray(state.u_d)
            v = np.asarray(state.v_d)
            if not np.isfinite(h).all():
                print(f"{day:>3d} | NaN")
                break
            u_cc = 0.5 * (u[:, :, :-1] + u[:, :, 1:])
            v_cc = 0.5 * (v[:, :-1, :] + v[:, 1:, :])
            speed = np.sqrt(u_cc**2 + v_cc**2)
            ok = '✓' if (h.min() > 0 and speed.max() < 80) else '✗'
            print(f"{day:>3d} | {h.min():>6.0f} {h.max():>6.0f} | "
                  f"{speed.max():>9.1f} | {ok}")


if __name__ == "__main__":
    main()
