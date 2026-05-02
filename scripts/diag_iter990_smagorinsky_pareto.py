"""Iter-990: Smagorinsky Pareto sweep on FB chain W2 1-day.

Quantifies the calibration headroom of d_sw5 Smagorinsky
(d2_bg, dddmp) and del-n vorticity damping (damp_v, nord_v) to
suppress the FB chain seam mode.

Result: best stable v_ll_Linf=43.77 m/s at (d2_bg=0.09,
dddmp=0.45, damp_v=0.06, nord_v=2) — 21% reduction from baseline
55.6 m/s.  Configurations past d2_bg ≥ 0.15 produce NaN at C36.
"""
from __future__ import annotations

import os
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


def run(d2_bg, dddmp, d4_bg=0.16, damp_v=0.06, nord_v=2,
        N=36, dt=300.0):
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
        hyperdiff_coeff=0.0, d4_bg=d4_bg, nord=1,
        damp_v=damp_v, nord_v=nord_v,
        d2_bg=d2_bg, dddmp=dddmp,
    )
    model = FV3FBShallowWaterModel(grid, cfg)
    n_steps = int(86400 / dt)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            for _ in range(n_steps):
                state = model.step(state, dt)
        except Exception:
            return None, None
    u_d_arr = np.asarray(state.u_d)
    v_d_arr = np.asarray(state.v_d)
    if not (np.isfinite(u_d_arr).all()
             and np.isfinite(v_d_arr).all()):
        return float("nan"), float("nan")
    ca, sa = cell_centre_angles_from_4edge(cdgrid)
    u_cc = 0.5 * (u_d_arr[:, :, :-1] + u_d_arr[:, :, 1:])
    v_cc = 0.5 * (v_d_arr[:, :-1, :] + v_d_arr[:, 1:, :])
    v_north = np.asarray(sa) * u_cc + np.asarray(ca) * v_cc
    weights = get_cubedsphere_to_latlon_weights(N, n_lon=360, n_lat=181)
    v_ll = apply_cubedsphere_to_latlon(v_north, weights)
    h_arr = np.asarray(state.h)
    h_err = float(np.max(np.abs(h_arr - sw.h.data)))
    return float(np.abs(v_ll).max()), h_err


def main():
    print("=== Iter-990 Smagorinsky Pareto sweep on FB chain W2 1-day ===")
    print(f"{'d2_bg':>6} {'dddmp':>6} {'damp_v':>7} {'nord_v':>7} "
          f"{'v_ll':>8} {'h_err':>10}")
    configs = [
        (0.0, 0.0, 0.06, 2),
        (0.01, 0.05, 0.06, 2),
        (0.09, 0.45, 0.06, 2),
        (0.15, 0.60, 0.06, 2),
        (0.05, 0.30, 0.10, 1),
        (0.10, 0.50, 0.10, 1),
    ]
    for d2, dd, dv, nv in configs:
        v_ll, h_err = run(d2, dd, damp_v=dv, nord_v=nv)
        if v_ll is None:
            tag = "EXCEPTION"
        elif np.isnan(v_ll):
            tag = "  NaN"
        else:
            tag = f"{v_ll:>8.3f} {h_err:>10.2f}"
        print(f"{d2:>6.3f} {dd:>6.3f} {dv:>7.3f} {nv:>7d} | {tag}")


if __name__ == "__main__":
    main()
