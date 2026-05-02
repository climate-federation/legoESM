"""Iter-986: damping sweep + west/east cube-edge asymmetry probe.

Iter-985 localised the v_ll_Linf=55.6 m/s peak to equatorial
cube-edge seams at face=*, i=0, j=21.  Iter-986 sweeps damping
parameters to characterise the seam mode and probes the asymmetry
between west (i=0, upwind) and east (i=N-1, downwind) edges.

Findings:
- Seam mode IS damp_v-sensitive (damp_v=0.12 → 44.6; damp_v=0 → 63.4).
- d_sw5 (d4_bg=0.16) damps seam and a separate interior mode at
  lat=1°.  Setting d4_bg=0 cuts seam to 17.9 m/s but unleashes
  the interior mode to 86 m/s.
- Asymmetry: face=1 i=0 (west, upwind) at -56.5 m/s; face=0 i=35
  (east, downwind) at -23.4 m/s.  PPM duogrid bypass of boundary
  corrections is the prime suspect.

Per-step Δv at j=21 is roughly uniform across i (-0.23 m/s/step
everywhere); the i=0 vs i=N-1 day-1 asymmetry comes from
asymmetric damping behaviour at the boundary, not from per-step
forcing.
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


def run_w2_1day(N=36, dt=300.0, **cfg_overrides):
    """FB chain C36 W2 1-day → (v_ll_Linf, seam_peak, interior_peak)."""
    grid = create_cubed_sphere(N, use_duogrid=True)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    sw = williamson_test2(grid)
    u0 = 2.0 * jnp.pi * grid.radius / (12.0 * 86400.0)
    u_d = cdgrid.cos_angle_edge_x * (u0 * jnp.cos(cdgrid.lat_edge_x))
    v_d = -cdgrid.sin_angle_edge_y * (u0 * jnp.cos(cdgrid.lat_edge_y))
    state = FV3EdgeShallowWaterState(
        h=sw.h.data, u_d=u_d, v_d=v_d, h_s=sw.h_s.data,
    )
    cfg_full = dict(
        hyperdiff_coeff=0.0,
        d4_bg=0.16, nord=1, damp_v=0.06, nord_v=2,
    )
    cfg_full.update(cfg_overrides)
    cfg = CDGridShallowWaterConfig(**cfg_full)
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
    seam = float(np.abs(v_north[1, 0, 21]))
    interior = float(np.abs(v_north[1, 1, 21]))
    weights = get_cubedsphere_to_latlon_weights(N, n_lon=360, n_lat=181)
    v_ll = apply_cubedsphere_to_latlon(v_north, weights)
    return float(np.abs(v_ll).max()), seam, interior


def main():
    print("=== Iter-986 damping sweep on FB chain C36 W2 1-day ===")
    fmt = "{label:30s} | v_ll_Linf | seam[1,0,21] | interior[1,1,21]"
    print(fmt.format(label="config"))
    print("-" * 80)
    configs = [
        ("baseline", {}),
        ("damp_v=0.12 (2x)", dict(damp_v=0.12)),
        ("damp_v=0", dict(damp_v=0.0)),
        ("d4_bg=0 (no d_sw5)", dict(d4_bg=0.0)),
        ("nord_v=1 (del-4 vort)", dict(nord_v=1)),
    ]
    for label, ckwargs in configs:
        v_ll, peak, interior = run_w2_1day(**ckwargs)
        print(f"{label:30s} |  {v_ll:7.3f}  |    {peak:6.2f}    |"
              f"     {interior:6.2f}")


if __name__ == "__main__":
    main()
