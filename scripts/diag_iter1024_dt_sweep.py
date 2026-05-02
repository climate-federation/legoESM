"""Iter-1024: dt scaling with iter-1021 calibration.

Result: dt ∈ [150, 450] satisfies both W2 ≤ 0.119 AND W5 day-5
artifact-free.  dt=300 is optimal but dt=450 (33% fewer steps)
also works with similar quality.  dt < 150 misses W2 acceptance
threshold (temporal aliasing).
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
    williamson_test5,
)
from tests.test_iter921_w2_v_vs_h_pareto_sentinel import (
    cell_centre_angles_from_4edge,
)


def run_w2(N, dt, cfg):
    grid = create_cubed_sphere(N)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    sw = williamson_test2(grid)
    u0 = 2.0 * jnp.pi * grid.radius / (12.0 * 86400.0)
    u_d = cdgrid.cos_angle_edge_x * (u0 * jnp.cos(cdgrid.lat_edge_x))
    v_d = -cdgrid.sin_angle_edge_y * (u0 * jnp.cos(cdgrid.lat_edge_y))
    state = FV3EdgeShallowWaterState(
        h=sw.h.data, u_d=u_d, v_d=v_d, h_s=sw.h_s.data,
    )
    model = FV3EdgeShallowWaterModel(grid, cfg)
    model.set_initial_mass(state)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(int(86400 / dt)):
            state = model.step(state, dt)
    ca, sa = cell_centre_angles_from_4edge(cdgrid)
    u_arr = np.asarray(state.u_d)
    v_arr = np.asarray(state.v_d)
    if not np.isfinite(u_arr).all():
        return float("nan")
    u_cc = 0.5 * (u_arr[:, :, :-1] + u_arr[:, :, 1:])
    v_cc = 0.5 * (v_arr[:, :-1, :] + v_arr[:, 1:, :])
    v_north = np.asarray(sa) * u_cc + np.asarray(ca) * v_cc
    weights = get_cubedsphere_to_latlon_weights(N, n_lon=360, n_lat=181)
    v_ll = apply_cubedsphere_to_latlon(v_north, weights)
    return float(np.abs(v_ll).max())


def run_w5(N, dt, cfg, days=5):
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
    model = FV3EdgeShallowWaterModel(grid, cfg)
    model.set_initial_mass(state)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(int(days * 86400 / dt)):
            state = model.step(state, dt)
    h = np.asarray(state.h)
    u = np.asarray(state.u_d)
    v = np.asarray(state.v_d)
    if not np.isfinite(h).all():
        return None
    u_cc = 0.5 * (u[:, :, :-1] + u[:, :, 1:])
    v_cc = 0.5 * (v[:, :-1, :] + v[:, 1:, :])
    speed = np.sqrt(u_cc**2 + v_cc**2)
    return float(h.min()), float(speed.max())


def main():
    N = 36
    cfg = iter1009_dual_target_config(N)
    print("=== Iter-1024 dt scaling with iter-1021 calibration ===")
    print(f"{'dt':>5} | W2 v_ll | W5 day-5")
    for dt in [600, 450, 300, 240, 200, 150, 100]:
        v_ll = run_w2(N, dt, cfg)
        r5 = run_w5(N, dt, cfg, 5)
        w2_ok = (not np.isnan(v_ll)) and v_ll <= 0.119
        w5_ok = r5 is not None and r5[0] > 0 and r5[1] < 80
        w2 = f"{v_ll:.4f}" if not np.isnan(v_ll) else "NaN"
        w5 = f"({r5[0]:.0f}, {r5[1]:.1f})" if r5 else "NaN"
        flags = ('W2' + ('✓' if w2_ok else '✗') + ' W5' +
                  ('✓' if w5_ok else '✗'))
        print(f"{dt:>5d} | {w2:>7s} | {w5:>14s} | {flags}")


if __name__ == "__main__":
    main()
