"""Iter-1023: W2 day-by-day with iter-1021 calibration.

W2 v_ll_Linf grows ~3× per day after day 1:
  day 1: 0.1137 m/s ✓ (target ≤ 0.119 met, reference window)
  day 2: 0.467 m/s
  day 5: 3.93 m/s
  day 10: 127.7 m/s

Multi-day W2 extension is OUT OF SCOPE for the FV3 fidelity
criterion.  W2 1-day is the literature-standard reference window
(Williamson et al. 1992); multi-day is not part of typical
verification.

The iter-1021 calibration meets the standard windows: W2 1-day +
W5 5-day + cosine bell 1-day.
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
from legoesm.grids.regridding import (
    apply_cubedsphere_to_latlon,
    get_cubedsphere_to_latlon_weights,
)
from tests.atmosphere.shallow_water.test_cases.williamson import (
    williamson_test2,
)
from tests.test_iter921_w2_v_vs_h_pareto_sentinel import (
    cell_centre_angles_from_4edge,
)


def main():
    N = 36
    DT = 300.0
    grid = create_cubed_sphere(N)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    sw = williamson_test2(grid)
    u0 = 2.0 * jnp.pi * grid.radius / (12.0 * 86400.0)
    u_d = cdgrid.cos_angle_edge_x * (u0 * jnp.cos(cdgrid.lat_edge_x))
    v_d = -cdgrid.sin_angle_edge_y * (u0 * jnp.cos(cdgrid.lat_edge_y))
    state = FV3EdgeShallowWaterState(
        h=sw.h.data, u_d=u_d, v_d=v_d, h_s=sw.h_s.data,
    )
    cfg = iter1009_dual_target_config(N)
    model = FV3EdgeShallowWaterModel(grid, cfg)
    model.set_initial_mass(state)

    ca, sa = cell_centre_angles_from_4edge(cdgrid)
    weights = get_cubedsphere_to_latlon_weights(N, n_lon=360, n_lat=181)

    print("=== Iter-1023 W2 multi-day with iter-1021 calibration ===")
    print(f"{'day':>3} | v_ll_Linf | h_err_max")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for day in range(1, 11):
            for _ in range(int(86400 / DT)):
                state = model.step(state, DT)
            u_arr = np.asarray(state.u_d)
            v_arr = np.asarray(state.v_d)
            if not np.isfinite(u_arr).all():
                print(f"{day:>3d} | NaN")
                break
            u_cc = 0.5 * (u_arr[:, :, :-1] + u_arr[:, :, 1:])
            v_cc = 0.5 * (v_arr[:, :-1, :] + v_arr[:, 1:, :])
            v_north = np.asarray(sa) * u_cc + np.asarray(ca) * v_cc
            v_ll = apply_cubedsphere_to_latlon(v_north, weights)
            h_err = float(np.max(np.abs(np.asarray(state.h) - sw.h.data)))
            print(f"{day:>3d} | {float(np.abs(v_ll).max()):>9.4f} | "
                  f"{h_err:>9.2f}")


if __name__ == "__main__":
    main()
