"""Iter-989: resolution scaling on FB chain W2 1-day.

Tests whether the FB chain v_ll_Linf=55.6 m/s gap is a resolved
numerical error (scales as (C/N)^2) or an unresolved grid-scale
instability (constant or growing with N).

Result: error is non-monotonic and GROWS at N=48, confirming the
architectural instability documented in `FV3FBShallowWaterModel`'s
docstring.

| N  | dt  | v_ll_Linf | seam_max |
|----|-----|-----------|----------|
| 18 | 600 |   50.60   |  44.63   |
| 24 | 450 |   57.30   |  46.53   |
| 36 | 300 |   55.61   |  56.50   |
| 48 | 225 |   88.89   | 122.29   |

A resolved 2nd-order scheme would show ~4× reduction per ~2× N;
this shows growth at fine N — unresolved grid-scale mode.
"""
from __future__ import annotations

import os
import time
import warnings

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np

from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import (
    CDGridShallowWaterConfig,
    FV3EdgeShallowWaterState,
    FV3FBShallowWaterModel,
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


def run(N, dt):
    grid = create_cubed_sphere(N, use_duogrid=True)
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
        d4_bg=0.16, nord=1, damp_v=0.06, nord_v=2,
    )
    model = FV3FBShallowWaterModel(grid, cfg)
    n_steps = int(86400 / dt)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(n_steps):
            state = model.step(state, dt)
    ca, sa = cell_centre_angles_from_4edge(cdgrid)
    u_d_arr = np.asarray(state.u_d)
    v_d_arr = np.asarray(state.v_d)
    u_cc = 0.5 * (u_d_arr[:, :, :-1] + u_d_arr[:, :, 1:])
    v_cc = 0.5 * (v_d_arr[:, :-1, :] + v_d_arr[:, 1:, :])
    v_north = np.asarray(sa) * u_cc + np.asarray(ca) * v_cc
    seam_max = float(np.max(np.abs(v_north[:, 0, :])))
    weights = get_cubedsphere_to_latlon_weights(N, n_lon=360, n_lat=181)
    v_ll = apply_cubedsphere_to_latlon(v_north, weights)
    return float(np.abs(v_ll).max()), seam_max


def main():
    print("=== Iter-989 resolution scaling on FB chain W2 1-day ===")
    print(f"{'N':>4} {'dt':>6} {'v_ll_Linf':>10} {'seam_max':>10} "
          f"{'time(s)':>8}")
    for N, dt in [(18, 600.0), (24, 450.0), (36, 300.0), (48, 225.0)]:
        t0 = time.time()
        v_ll, seam = run(N, dt)
        elapsed = time.time() - t0
        print(f"{N:>4} {dt:>6} {v_ll:>10.4f} {seam:>10.4f} "
              f"{elapsed:>8.1f}")


if __name__ == "__main__":
    main()
