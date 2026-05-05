"""Iter-754b visual inspection: W2 v-wind snapshot with del6 enabled.

Compare the baseline (hyperdiff-only) snapshot vs the del6-enabled
snapshot at t=1 d for the mode-4 polar band pattern.

Usage:
    JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu .venv/bin/python \
        scripts/diag_iter754b_del6_visual.py
"""
import os, sys
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import (
    create_cubed_sphere_cdgrid, cell_centre_angles_from_4edge)
from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import (
    FV3EdgeShallowWaterModel, FV3EdgeShallowWaterState,
    CDGridShallowWaterConfig)
from tests.test_cases.williamson import williamson_test2
from scripts.run_atmosphere_test_matrix import (
    _hyperdiff_cube, _div_damp_cube, _regrid_2d)


def compute_v_ll(state, grid, cdgrid):
    ca_4edge, sa_4edge = cell_centre_angles_from_4edge(cdgrid)
    ca_np = np.asarray(ca_4edge, dtype=np.float64)
    sa_np = np.asarray(sa_4edge, dtype=np.float64)
    u_cc = 0.5 * (np.asarray(state.u_d, dtype=np.float64)[:, :, :-1]
                  + np.asarray(state.u_d, dtype=np.float64)[:, :, 1:])
    v_cc = 0.5 * (np.asarray(state.v_d, dtype=np.float64)[:, :-1, :]
                  + np.asarray(state.v_d, dtype=np.float64)[:, 1:, :])
    v_north_face = sa_np * u_cc + ca_np * v_cc
    lon_deg = np.asarray(grid.lon, dtype=np.float64) * 180 / np.pi
    lat_deg = np.asarray(grid.lat, dtype=np.float64) * 180 / np.pi
    v_ll = _regrid_2d(v_north_face, lon_deg, lat_deg, "cube")
    return v_ll


def run(damp_v, nord_v, hyperdiff, label):
    n = 36
    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    dt = 300.0
    config = CDGridShallowWaterConfig(
        hyperdiff_coeff=hyperdiff,
        div_damp=_div_damp_cube(n),
        boundary_fix=True,
        damp_v=damp_v, nord_v=nord_v,
    )
    model = FV3EdgeShallowWaterModel(grid, config)
    cdgrid_out = model.cdgrid

    sw = williamson_test2(grid)
    u0 = 2.0 * np.pi * float(grid.radius) / (12.0 * 86400.0)
    u_d = cdgrid_out.cos_angle_edge_x * (u0 * jnp.cos(cdgrid_out.lat_edge_x))
    v_d = -cdgrid_out.sin_angle_edge_y * (u0 * jnp.cos(cdgrid_out.lat_edge_y))
    state = FV3EdgeShallowWaterState(
        h=sw.h.data, u_d=u_d, v_d=v_d, h_s=sw.h_s.data)
    model.set_initial_mass(state)

    for _ in range(int(86400 / dt)):
        state = model.step(state, dt)

    v_ll = compute_v_ll(state, grid, cdgrid_out)
    print(f"  {label}: max|v_ll|={np.max(np.abs(v_ll)):.3e} m/s")
    return v_ll


hyp = _hyperdiff_cube(36)

v_ll_baseline = run(0.0, 1, hyp, "baseline")
v_ll_del6 = run(0.06, 1, 0.0, "del6 damp_v=0.06 (replaces hyperdiff)")
v_ll_both = run(0.06, 1, hyp, "both damp_v=0.06 + hyperdiff")

# Plot side-by-side.
lat = np.linspace(-90, 90, v_ll_baseline.shape[0])
lon = np.linspace(-180, 180, v_ll_baseline.shape[1], endpoint=False)

fig, axes = plt.subplots(3, 1, figsize=(10, 9))
vmin, vmax = -0.3, 0.3
cmap = "RdBu_r"

for ax, v_ll, title in [(axes[0], v_ll_baseline, "baseline (hyperdiff only)"),
                        (axes[1], v_ll_del6, "del6 damp_v=0.06 nord_v=1 (NO hyperdiff)"),
                        (axes[2], v_ll_both, "both del6 + hyperdiff")]:
    im = ax.pcolormesh(lon, lat, v_ll, vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_title(f"{title}  max|v|={np.max(np.abs(v_ll)):.2e} m/s")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_yticks([-90, -60, -30, 0, 30, 60, 90])
    fig.colorbar(im, ax=ax, label="v_ll (m/s)")

fig.tight_layout()
outpath = "diagnostics/fv3_visual/iter754_del6_w2_comparison.png"
os.makedirs(os.path.dirname(outpath), exist_ok=True)
fig.savefig(outpath, dpi=120)
print(f"\nSaved: {outpath}")
