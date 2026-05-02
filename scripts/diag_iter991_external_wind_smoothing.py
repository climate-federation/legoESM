"""Iter-991/992: external del-2 wind smoothing post FB step.

NEGATIVE RESULT: external smoothing reduces v_ll_Linf to 10.7 m/s
but corrupts h field (3839 m error vs analytical 0).  Geostrophic
balance is destroyed by the post-step wind smoothing.

The iter-990 v_ll_Linf=43.77 m/s floor remains the best calibration
result that preserves W2 solution structure.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

from legoesm.core.fv3_sw_core import fv3_fb_sw_step
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


def lap_2d(arr):
    pi = jnp.pad(arr, [(0, 0), (1, 1), (0, 0)], mode='edge')
    li = pi[:, :-2, :] - 2 * pi[:, 1:-1, :] + pi[:, 2:, :]
    pj = jnp.pad(arr, [(0, 0), (0, 0), (1, 1)], mode='edge')
    lj = pj[:, :, :-2] - 2 * pj[:, :, 1:-1] + pj[:, :, 2:]
    return li + lj


def make_step(cdgrid, dt, damp_post):
    @jax.jit
    def step(h, u_d, v_d, h_s):
        h_new, u_new, v_new = fv3_fb_sw_step(
            h, u_d, v_d, h_s, cdgrid, dt,
            d2_bg=0.09, dddmp=0.45, d4_bg=0.16, nord=1,
            damp_v=0.06, nord_v=2,
        )
        if damp_post > 0:
            u_new = u_new + damp_post * lap_2d(u_new)
            v_new = v_new + damp_post * lap_2d(v_new)
        return h_new, u_new, v_new
    return step


def run(N, dt, damp_post):
    grid = create_cubed_sphere(N, use_duogrid=True)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    sw = williamson_test2(grid)
    u0 = 2.0 * jnp.pi * grid.radius / (12.0 * 86400.0)
    u_d = cdgrid.cos_angle_edge_x * (u0 * jnp.cos(cdgrid.lat_edge_x))
    v_d = -cdgrid.sin_angle_edge_y * (u0 * jnp.cos(cdgrid.lat_edge_y))
    h = sw.h.data
    h_s = sw.h_s.data
    h_ic = np.asarray(h)
    n_steps = int(86400 / dt)
    step = make_step(cdgrid, dt, damp_post)
    for _ in range(n_steps):
        h, u_d, v_d = step(h, u_d, v_d, h_s)
    u_d_arr = np.asarray(u_d)
    v_d_arr = np.asarray(v_d)
    if not (np.isfinite(u_d_arr).all()
             and np.isfinite(v_d_arr).all()):
        return float("nan"), float("nan")
    ca, sa = cell_centre_angles_from_4edge(cdgrid)
    u_cc = 0.5 * (u_d_arr[:, :, :-1] + u_d_arr[:, :, 1:])
    v_cc = 0.5 * (v_d_arr[:, :-1, :] + v_d_arr[:, 1:, :])
    v_north = np.asarray(sa) * u_cc + np.asarray(ca) * v_cc
    weights = get_cubedsphere_to_latlon_weights(N, n_lon=360, n_lat=181)
    v_ll = apply_cubedsphere_to_latlon(v_north, weights)
    h_err = float(np.max(np.abs(np.asarray(h) - h_ic)))
    return float(np.abs(v_ll).max()), h_err


def main():
    N, dt = 36, 300.0
    print("=== Iter-991/992 external del-2 wind smoothing ===")
    print(f"{'damp_post':>10} | v_ll_Linf | h_err_max")
    for d in [0.0, 0.05, 0.10, 0.20, 0.265]:
        v_ll, h_err = run(N, dt, d)
        if np.isnan(v_ll):
            tag = "NaN"
        else:
            tag = f"{v_ll:>9.4f} | {h_err:>9.2f}"
        print(f"{d:>10.4f} | {tag}")


if __name__ == "__main__":
    main()
