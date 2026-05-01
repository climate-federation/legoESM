"""Iter-985: localise the W2 1-day v_ll_Linf=55.6 m/s peak.

Iter-981-984 chased cube-vertex tendency in 1-step c_sw + p_grad_c.
Iter-985 verifies whether the cube-vertex error matters for the
day-1 W2 v_ll_Linf metric by:

1. Patching `_corner_vorticity` to use edge-mode at cube vertices
   (no cross-face rotation) and measuring v_ll_Linf delta.
2. Localising where on the cubed sphere |v_north| peaks at day 1.

Result: cube-vertex fix yields +0.02% improvement (negligible).
Peak is at equatorial cube-edge seam (face=0..3, i=0, j=21) at
lat ~6.34°, NOT at cube vertices.  Iter-986 should pivot to the
equatorial cube-edge seam.
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


def run_w2_1day(N: int, dt: float):
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
    return grid, cdgrid, state


def localise_peak(grid, cdgrid, state):
    ca, sa = cell_centre_angles_from_4edge(cdgrid)
    u_d_arr = np.asarray(state.u_d)
    v_d_arr = np.asarray(state.v_d)
    u_cc = 0.5 * (u_d_arr[:, :, :-1] + u_d_arr[:, :, 1:])
    v_cc = 0.5 * (v_d_arr[:, :-1, :] + v_d_arr[:, 1:, :])
    v_north = np.asarray(sa) * u_cc + np.asarray(ca) * v_cc

    # Top-10 |v_north| locations
    flat_idx = np.argsort(np.abs(v_north).flatten())[::-1][:10]
    print("Top 10 |v_north| locations on cubed sphere:")
    for k, idx in enumerate(flat_idx):
        f, i, j = np.unravel_index(idx, v_north.shape)
        lat = np.degrees(grid.lat[f, i, j])
        lon = np.degrees(grid.lon[f, i, j])
        print(f"  #{k+1}: face={f}, i={i}, j={j} | "
              f"v_north={v_north[f,i,j]:9.4f} m/s | "
              f"lat={lat:6.2f}° lon={lon:7.2f}°")

    # Lat-lon regrid v_ll_Linf
    weights = get_cubedsphere_to_latlon_weights(
        cdgrid.n, n_lon=360, n_lat=181)
    v_ll = apply_cubedsphere_to_latlon(v_north, weights)
    return float(np.abs(v_ll).max())


def main():
    print("=== Iter-985 W2 1-day localisation ===")
    grid, cdgrid, state = run_w2_1day(36, 300.0)
    v_ll_max = localise_peak(grid, cdgrid, state)
    print(f"\nv_ll_Linf = {v_ll_max:.4f} m/s")


if __name__ == "__main__":
    main()
