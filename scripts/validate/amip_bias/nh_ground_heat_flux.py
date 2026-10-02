#!/usr/bin/env python3
"""December conductive ground heat flux from checkpoints, with the land's OWN
conductivity (``compute_thermal_conductivity`` on the deployed hydraulics/thermal
config) and the same interface geometry ``solve_soil_thermal`` assembles
(harmonic-mean interface conductivity over ``dz_interface``).

Flux UP toward the surface across interface k (positive = heat leaving the soil
upward into the layer above / the surface):  F_k = k_half (T[k+1] - T[k]) / dz_if.
Instantaneous 00Z checkpoint states averaged over the window; driver built for
setup only (``land_deployed_lai_stress.build_driver``).  Also prints k_eff at the
top node and at ~0.6 m, the deployed solid conductivity and the per-PFT solid-
conductivity scale (k_solid / texture-only ``soil_solid_conductivity``).

Usage: nh_ground_heat_flux.py <run> <scratch> --days 335 365
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.environ.get("LEGOESM_AMIP_RUNS",
                      "/work/bd1083/b309178/diffESM/legoesm_pg/amip_runs")
sys.path.insert(0, os.environ.get(
    "LEGOESM_DIAG_LAND_PROBES",
    "/work/bd1083/b309178/diffESM/legoesm_pg/wt_diag_land/scripts/validate/amip_bias"))
from land_deployed_lai_stress import build_driver  # noqa: E402

REGIONS = {"45-70N": (45, 70, 0, 360), "Siberia": (50, 70, 60, 140)}


def wmean(x, w):
    on = w > 0
    if not np.isfinite(np.asarray(x)[on]).all():
        raise SystemExit("FATAL: non-finite inside region")
    return float(np.sum(np.where(on, x, 0.0) * w) / np.sum(w))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("scratch")
    ap.add_argument("--days", nargs=2, type=int, required=True)
    a = ap.parse_args()
    import jax.numpy as jnp
    from legoesm.land.clm_surface_map import load_clm_surface
    from legoesm.land.soil_grid import make_soil_grid
    from legoesm.land.soil_texture import soil_solid_conductivity
    from legoesm.land.soil_thermal import compute_thermal_conductivity

    d = build_driver(a.run, a.scratch)
    c = d.physics.land_ml_cfg
    g = make_soil_grid(c.soil_grid)
    z, dzi = np.asarray(g.z_node), np.asarray(g.dz_interface)
    lat = np.rad2deg(np.asarray(d.grid.latCell))
    lon = np.rad2deg(np.asarray(d.grid.lonCell)) % 360
    f_land = np.asarray(d._f_land).reshape(-1)
    area = np.asarray(d.grid.areaCell)
    sm = load_clm_surface(d.config.clm_surfdata_path, lat, lon)
    k_solid = np.asarray(c.thermal.k_solid).reshape(-1)
    k_tex = np.asarray(soil_solid_conductivity(sm["pct_sand"], sm["pct_clay"])).reshape(-1)
    glac = np.asarray(sm["glacier_frac"]).reshape(-1)
    i_02 = int(np.argmin(np.abs(0.5 * (z[:-1] + z[1:]) - 0.2)))
    i_06 = int(np.argmin(np.abs(z - 0.6)))
    print(f"interfaces: top {0.5 * (z[0] + z[1]):.4f} m, ~0.2 m -> {0.5 * (z[i_02] + z[i_02 + 1]):.3f}"
          f" m; k at node {z[0]:.3f} m and {z[i_06]:.3f} m; freeze_thaw "
          f"{c.thermal.enable_freeze_thaw}")
    acc = {}
    days = range(a.days[0], a.days[1] + 1)
    for day in days:
        ck = np.load(f"{ROOT}/{a.run}/checkpoint_day_{day:04d}.npz")
        T, th = np.asarray(ck["land_ml_T_soil"]), np.asarray(ck["land_ml_theta_soil"])
        k = np.asarray(compute_thermal_conductivity(jnp.asarray(th), c.hydraulics, c.thermal))
        kh = 2.0 * k[:, :-1] * k[:, 1:] / (k[:, :-1] + k[:, 1:] + 1e-20)
        F = kh * (T[:, 1:] - T[:, :-1]) / dzi
        for key, v in (("F_top", F[:, 0]), ("F_02", F[:, i_02]), ("k_top", k[:, 0]),
                       ("k_06", k[:, i_06])):
            acc.setdefault(key, []).append(v)
    mean = {k_: np.mean(v, 0) for k_, v in acc.items()}
    for name, (a0, a1, b0, b1) in REGIONS.items():
        w = area * f_land * ((f_land > 0.5) & (glac < 0.5) & (lat >= a0) & (lat <= a1)
                             & (lon >= b0) & (lon <= b1))
        print(f"[{name}] cells {int((w > 0).sum())}, days {a.days[0]}-{a.days[1]}: "
              f"upward conductive flux top {wmean(mean['F_top'], w):+.1f} W/m2, "
              f"~0.2 m {wmean(mean['F_02'], w):+.1f} W/m2 | k_eff top "
              f"{wmean(mean['k_top'], w):.3f}, ~0.6 m {wmean(mean['k_06'], w):.3f} W/m/K | "
              f"k_solid {wmean(k_solid, w):.3f} (texture-only {wmean(k_tex, w):.3f}), "
              f"PFT scale {wmean(k_solid / k_tex, w):.3f}")


if __name__ == "__main__":
    main()
