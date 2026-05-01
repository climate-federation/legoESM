"""Iter-1005: nord_v (vort damping order) sweep for W5 stability.

Result: nord_v=2 (del-6, default) gives best W5 long-term stability
among tested configs.  No nord_v / damp_v combination achieves
speed_max < 80 m/s at day 7.  W5 long-term instability is robust
across calibration knobs.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import warnings

import jax.numpy as jnp
import numpy as np

from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import (
    CDGridShallowWaterConfig,
    FV3EdgeShallowWaterModel,
    FV3EdgeShallowWaterState,
)
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from tests.atmosphere.shallow_water.test_cases.williamson import (
    williamson_test5,
)
from tests.test_iter921_w2_v_vs_h_pareto_sentinel import _div_damp_cube


def run(N, dt, nord_v, damp_v, days):
    grid = create_cubed_sphere(N)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    sw = williamson_test5(grid)
    u0 = 20.0
    u_d = cdgrid.cos_angle_edge_x * (u0 * jnp.cos(cdgrid.lat_edge_x))
    v_d = -cdgrid.sin_angle_edge_y * (u0 * jnp.cos(cdgrid.lat_edge_y))
    state = FV3EdgeShallowWaterState(
        h=sw.h.data, u_d=u_d, v_d=v_d, h_s=sw.h_s.data,
    )
    cfg = CDGridShallowWaterConfig(
        hyperdiff_coeff=0.0,
        div_damp=12.0 * _div_damp_cube(N),
        boundary_fix=True,
        damp_v=damp_v, nord_v=nord_v,
        apply_fortran_xppm_boundary=True,
    )
    model = FV3EdgeShallowWaterModel(grid, cfg)
    model.set_initial_mass(state)
    n_steps = int(days * 86400 / dt)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(n_steps):
            state = model.step(state, dt)
    h = np.asarray(state.h)
    u = np.asarray(state.u_d)
    v = np.asarray(state.v_d)
    if not (np.isfinite(h).all()
             and np.isfinite(u).all()
             and np.isfinite(v).all()):
        return None
    u_cc = 0.5 * (u[:, :, :-1] + u[:, :, 1:])
    v_cc = 0.5 * (v[:, :-1, :] + v[:, 1:, :])
    speed = np.sqrt(u_cc**2 + v_cc**2)
    return float(h.min()), float(h.max()), float(speed.max())


def main():
    print("=== Iter-1005 W5 nord_v / damp_v sweep ===")
    print(f"{'nord_v':>6} {'damp_v':>7} {'days':>5} | h_min | h_max | speed")
    configs = [
        (0, 0.06, 7),
        (1, 0.06, 7),
        (2, 0.06, 7),
        (1, 0.10, 7),
        (1, 0.15, 7),
    ]
    for nv, dv, days in configs:
        r = run(36, 300.0, nv, dv, days)
        if r is None:
            print(f"{nv:>6d} {dv:>7.2f} {days:>5d} | NaN")
        else:
            h_min, h_max, spd = r
            ok = "✓" if (h_min > 0 and spd < 80) else "✗"
            print(f"{nv:>6d} {dv:>7.2f} {days:>5d} | "
                  f"h=[{h_min:>6.0f}, {h_max:>6.0f}] "
                  f"spd={spd:>6.1f} {ok}")


if __name__ == "__main__":
    main()
