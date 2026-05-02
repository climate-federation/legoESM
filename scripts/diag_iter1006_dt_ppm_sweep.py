"""Iter-1006: dt sweep + PPM-faithful flag sweep on iter-1002 W2 config.

Confirms iter-1002 config is optimum:
- dt=300 gives W2 v_ll_Linf=0.1154 m/s ≤ 0.119 (target met)
- dt=600/450 → NaN (too large)
- dt=200 → 0.1179 (under target but worse)
- dt < 200 → over target (temporal aliasing)
- fortran_faithful_ppm_left/right=True → over target

Iter-1002 config (dt=300, div_damp=12*cube, damp_v=0.04,
apply_fortran_xppm_boundary=True only) is the demonstrated best
W2 production calibration.
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
    FV3EdgeShallowWaterModel,
    FV3EdgeShallowWaterState,
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
    _div_damp_cube,
)


def run_w2(N, dt, **cfg_overrides):
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
        div_damp=12.0 * _div_damp_cube(N),
        boundary_fix=True,
        damp_v=0.04, nord_v=2,
        apply_fortran_xppm_boundary=True,
    )
    base.update(cfg_overrides)
    cfg = CDGridShallowWaterConfig(**base)
    model = FV3EdgeShallowWaterModel(grid, cfg)
    model.set_initial_mass(state)
    n_steps = int(86400 / dt)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(n_steps):
            state = model.step(state, dt)
    ca, sa = cell_centre_angles_from_4edge(cdgrid)
    u_arr = np.asarray(state.u_d)
    v_arr = np.asarray(state.v_d)
    if not (np.isfinite(u_arr).all() and np.isfinite(v_arr).all()):
        return float("nan"), float("nan")
    u_cc = 0.5 * (u_arr[:, :, :-1] + u_arr[:, :, 1:])
    v_cc = 0.5 * (v_arr[:, :-1, :] + v_arr[:, 1:, :])
    v_north = np.asarray(sa) * u_cc + np.asarray(ca) * v_cc
    weights = get_cubedsphere_to_latlon_weights(N, n_lon=360, n_lat=181)
    v_ll = apply_cubedsphere_to_latlon(v_north, weights)
    h_err = float(np.max(np.abs(np.asarray(state.h) - sw.h.data)))
    return float(np.abs(v_ll).max()), h_err


def main():
    N = 36
    print("=== Iter-1006a dt sweep ===")
    print(f"{'dt':>5} {'n_steps':>8} | v_ll | h_err")
    for dt in [300, 200, 150, 100, 60]:
        v_ll, h_err = run_w2(N, dt)
        n_steps = int(86400 / dt)
        if np.isnan(v_ll):
            tag = "NaN"
        elif v_ll <= 0.119:
            tag = f"{v_ll:.4f} ✓ | {h_err:.2f}"
        else:
            tag = f"{v_ll:.4f}   | {h_err:.2f}"
        print(f"{dt:>5d} {n_steps:>8d} | {tag}")

    print("\n=== Iter-1006b PPM-faithful flag sweep (dt=300) ===")
    print(f"{'L':>6} {'R':>6} | v_ll | h_err")
    for L, R in [(False, False), (True, False), (False, True),
                  (True, True)]:
        v_ll, h_err = run_w2(N, 300,
                              fortran_faithful_ppm_left=L,
                              fortran_faithful_ppm_right=R)
        if np.isnan(v_ll):
            tag = "NaN"
        elif v_ll <= 0.119:
            tag = f"{v_ll:.4f} ✓ | {h_err:.2f}"
        else:
            tag = f"{v_ll:.4f}   | {h_err:.2f}"
        print(f"{str(L):>6s} {str(R):>6s} | {tag}")


if __name__ == "__main__":
    main()
