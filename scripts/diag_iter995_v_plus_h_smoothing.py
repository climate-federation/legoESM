"""Iter-995: combined v + h smoothing to reduce h_err residual.

Adds a mass-conserving del-2 smoother to h on top of iter-994's
meridional v smoother.

Result: damp_h=0.05 halves h_err (2567 → 1589 m) while keeping
v_ll near the iter-994 residual (8.08 → 8.33 m/s).  Higher damp_h
catastrophically grows v_ll due to balance loss.
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


def smooth_v_north_npass(u_d, v_d, ca_cc, sa_cc, damp, n_pass):
    u_cc = 0.5 * (u_d[:, :, :-1] + u_d[:, :, 1:])
    v_cc = 0.5 * (v_d[:, :-1, :] + v_d[:, 1:, :])
    u_east = ca_cc * u_cc - sa_cc * v_cc
    v_north = sa_cc * u_cc + ca_cc * v_cc
    for _ in range(n_pass):
        v_north = v_north + damp * lap_2d(v_north)
    u_cc_new = ca_cc * u_east + sa_cc * v_north
    v_cc_new = -sa_cc * u_east + ca_cc * v_north
    u_cc_delta = u_cc_new - u_cc
    v_cc_delta = v_cc_new - v_cc
    u_d_pad = jnp.pad(u_cc_delta, [(0, 0), (0, 0), (1, 0)], mode='edge')
    u_d_pad = jnp.concatenate([u_d_pad, u_cc_delta[:, :, -1:]], axis=2)
    u_d_corr = 0.5 * (u_d_pad[:, :, :-1] + u_d_pad[:, :, 1:])
    u_d_new = u_d + u_d_corr
    v_d_pad = jnp.concatenate([v_cc_delta[:, :1, :], v_cc_delta], axis=1)
    v_d_pad = jnp.concatenate([v_d_pad, v_cc_delta[:, -1:, :]], axis=1)
    v_d_corr = 0.5 * (v_d_pad[:, :-1, :] + v_d_pad[:, 1:, :])
    v_d_new = v_d + v_d_corr
    return u_d_new, v_d_new


def smooth_h_mass_conserving(h, area, total_area, damp_h):
    if damp_h == 0:
        return h
    mass_before = jnp.sum(h * area)
    h_smooth = h + damp_h * lap_2d(h)
    mass_after = jnp.sum(h_smooth * area)
    correction = (mass_before - mass_after) / total_area
    return h_smooth + correction


def make_step(cdgrid, dt, damp_v, n_pass_v, damp_h, ca_cc, sa_cc):
    area = cdgrid.base.area
    total_area = jnp.sum(area)

    @jax.jit
    def step(h, u_d, v_d, h_s):
        h_new, u_new, v_new = fv3_fb_sw_step(
            h, u_d, v_d, h_s, cdgrid, dt,
            d2_bg=0.09, dddmp=0.45, d4_bg=0.16, nord=1,
            damp_v=0.06, nord_v=2,
        )
        if damp_v > 0:
            u_new, v_new = smooth_v_north_npass(
                u_new, v_new, ca_cc, sa_cc, damp_v, n_pass_v)
        if damp_h > 0:
            h_new = smooth_h_mass_conserving(
                h_new, area, total_area, damp_h)
        return h_new, u_new, v_new
    return step


def run(N, dt, damp_v, n_pass_v, damp_h):
    grid = create_cubed_sphere(N, use_duogrid=True)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    sw = williamson_test2(grid)
    u0 = 2.0 * jnp.pi * grid.radius / (12.0 * 86400.0)
    u_d = cdgrid.cos_angle_edge_x * (u0 * jnp.cos(cdgrid.lat_edge_x))
    v_d = -cdgrid.sin_angle_edge_y * (u0 * jnp.cos(cdgrid.lat_edge_y))
    h = sw.h.data
    h_s = sw.h_s.data
    h_ic = np.asarray(h)
    ca_cc, sa_cc = cell_centre_angles_from_4edge(cdgrid)
    ca_cc, sa_cc = jnp.asarray(ca_cc), jnp.asarray(sa_cc)
    n_steps = int(86400 / dt)
    step = make_step(cdgrid, dt, damp_v, n_pass_v, damp_h, ca_cc, sa_cc)
    for _ in range(n_steps):
        h, u_d, v_d = step(h, u_d, v_d, h_s)
    u_d_arr = np.asarray(u_d)
    v_d_arr = np.asarray(v_d)
    h_arr = np.asarray(h)
    if not (np.isfinite(u_d_arr).all() and np.isfinite(v_d_arr).all()):
        return float("nan"), float("nan")
    ca_n, sa_n = cell_centre_angles_from_4edge(cdgrid)
    u_cc = 0.5 * (u_d_arr[:, :, :-1] + u_d_arr[:, :, 1:])
    v_cc = 0.5 * (v_d_arr[:, :-1, :] + v_d_arr[:, 1:, :])
    v_north = np.asarray(sa_n) * u_cc + np.asarray(ca_n) * v_cc
    weights = get_cubedsphere_to_latlon_weights(N, n_lon=360, n_lat=181)
    v_ll = apply_cubedsphere_to_latlon(v_north, weights)
    h_err = float(np.max(np.abs(h_arr - h_ic)))
    return float(np.abs(v_ll).max()), h_err


def main():
    print("=== Iter-995 v + mass-conserving h smoothing ===")
    print(f"{'damp_v':>7} {'np_v':>5} {'damp_h':>7} | v_ll | h_err")
    for dv, npv, dh in [
        (0.30, 3, 0.00),
        (0.30, 3, 0.05),
        (0.30, 3, 0.10),
        (0.30, 3, 0.20),
    ]:
        v_ll, h_err = run(36, 300.0, dv, npv, dh)
        if np.isnan(v_ll):
            tag = "NaN"
        else:
            tag = f"{v_ll:>7.4f} | {h_err:>8.2f}"
        print(f"{dv:>7.3f} {npv:>5d} {dh:>7.3f} | {tag}")


if __name__ == "__main__":
    main()
