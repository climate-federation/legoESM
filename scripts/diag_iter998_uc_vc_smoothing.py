"""Iter-998: smooth uc/vc between p_grad_c and d_sw inside FB step.

Tests whether pre-smoothing the C-grid winds suppresses the seam
mode at source.  Result: NO gain — v_ll INCREASES slightly with
damp_uvc.  The seam mode is generated within d_sw itself, not
inherited from c_sw + p_grad_c.

Confirms the iter-994 residual at v_ll≈8.08 m/s is structural.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

from legoesm.core.fv3_sw_core import _c_sw, _p_grad_c, _d_sw_native
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


def smooth_uc_vc_meridional(uc, vc, damp, ca_cc, sa_cc):
    uc_cc = 0.5 * (uc[:, :-1, :] + uc[:, 1:, :])
    vc_cc = 0.5 * (vc[:, :, :-1] + vc[:, :, 1:])
    u_east = ca_cc * uc_cc - sa_cc * vc_cc
    v_north = sa_cc * uc_cc + ca_cc * vc_cc
    v_north = v_north + damp * lap_2d(v_north)
    uc_cc_new = ca_cc * u_east + sa_cc * v_north
    vc_cc_new = -sa_cc * u_east + ca_cc * v_north
    uc_delta_cc = uc_cc_new - uc_cc
    vc_delta_cc = vc_cc_new - vc_cc
    uc_pad = jnp.concatenate(
        [uc_delta_cc[:, :1, :], uc_delta_cc, uc_delta_cc[:, -1:, :]], axis=1)
    uc_corr = 0.5 * (uc_pad[:, :-1, :] + uc_pad[:, 1:, :])
    vc_pad = jnp.concatenate(
        [vc_delta_cc[:, :, :1], vc_delta_cc, vc_delta_cc[:, :, -1:]], axis=2)
    vc_corr = 0.5 * (vc_pad[:, :, :-1] + vc_pad[:, :, 1:])
    return uc + uc_corr, vc + vc_corr


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
    v_d_pad = jnp.concatenate([v_cc_delta[:, :1, :], v_cc_delta], axis=1)
    v_d_pad = jnp.concatenate([v_d_pad, v_cc_delta[:, -1:, :]], axis=1)
    v_d_corr = 0.5 * (v_d_pad[:, :-1, :] + v_d_pad[:, 1:, :])
    return u_d + u_d_corr, v_d + v_d_corr


def make_step(cdgrid, dt, damp_uvc, damp_post, n_pass_post,
               ca_cc, sa_cc):
    @jax.jit
    def step(h, u_d, v_d, h_s):
        h_star, uc, vc, ua, va = _c_sw(
            h, u_d, v_d, h_s, cdgrid, dt, 9.80616)
        dp_x, dp_y = _p_grad_c(h_star, h_s, cdgrid, dt * 0.5, 9.80616)
        uc = uc + dp_x
        vc = vc + dp_y
        if damp_uvc > 0:
            uc, vc = smooth_uc_vc_meridional(
                uc, vc, damp_uvc, ca_cc, sa_cc)
        h_new, u_new, v_new = _d_sw_native(
            h, u_d, v_d, h_s, uc, vc, ua, va, cdgrid, dt, 9.80616,
            d2_bg=0.09, dddmp=0.45, d4_bg=0.16, nord=1,
            damp_v=0.06, nord_v=2,
        )
        if damp_post > 0:
            u_new, v_new = smooth_v_north_npass(
                u_new, v_new, ca_cc, sa_cc, damp_post, n_pass_post)
        return h_new, u_new, v_new
    return step


def run(N, dt, damp_uvc, damp_post=0.30, n_pass_post=3):
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
    step = make_step(
        cdgrid, dt, damp_uvc, damp_post, n_pass_post, ca_cc, sa_cc)
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
    print("=== Iter-998 smooth uc/vc between p_grad_c and d_sw ===")
    print(f"{'damp_uvc':>8} | v_ll | h_err")
    for d in [0.0, 0.05, 0.10, 0.20, 0.40]:
        v_ll, h_err = run(36, 300.0, d)
        if np.isnan(v_ll):
            tag = "NaN"
        else:
            tag = f"{v_ll:>7.4f} | {h_err:>8.2f}"
        print(f"{d:>8.3f} | {tag}")


if __name__ == "__main__":
    main()
