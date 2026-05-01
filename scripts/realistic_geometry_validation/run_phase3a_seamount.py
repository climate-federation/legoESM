#!/usr/bin/env python
"""Phase 3a of the realistic-geometry plan: Beckmann-Haidvogel seamount.

The canonical PGF stress test for z-coordinate ocean models
(Beckmann & Haidvogel 1993, JPO).  Stratified rest state in a
zonally-periodic basin with a single Gaussian seamount.  Zero forcing,
zero initial flow.  Any spurious velocity that develops is a numerical
artefact from the horizontal pressure gradient computed over the steep
bathymetry.

Setup
-----
- Grid: 36x72 lat-lon (5 deg resolution).
- Vertical: 20 levels, H_max=4000 m, dz_surface=10 m, dz_deep=500 m.
- Bathymetry: Gaussian seamount centred at (lat=0, lon=180),
  peak height 3800 m above the basin floor, sigma=10 deg (~1100 km e-fold).
  Optional N passes of Laplacian smoothing (--smoothing-passes).
- Stratification: linear T(z), 20C surface -> 2C deep.  Salinity uniform.
- Forcing: NONE (no wind, no SST restoring).
- Integration: 30 sim-days, dt=600 s (4320 steps), implicit-CN solver.
- Coord: pure z* (--coord zstar) OR z* + partial bottom cells
  (--coord partial).  Partial cells pass the test at much higher
  r-factor (steeper, less-smoothed bathymetry).

Pass criterion
--------------
``max|u|`` after 30 days <= 5 mm/s.

Outputs
-------
results/realistic_geometry_validation/phase3a_seamount_<coord>_smooth{N}/
  - umax_timeseries.png:  max|u| as a function of time
  - velocity_snapshot.png: surface speed map at day 30
  - bathymetry.png:        zonal section through the seamount
  - run.log:              per-day diagnostics

Usage
-----
    JAX_ENABLE_X64=1 python scripts/realistic_geometry_validation/run_phase3a_seamount.py \\
        --coord partial --smoothing-passes 0
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
os.environ.setdefault("JAX_ENABLE_X64", "1")

from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.vertical import (
    create_ocean_z_star,
    create_partial_cell_coordinate,
    compute_centroid_depth,
)
from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
from legoesm.ocean.state import LatLonCGridOceanConfig
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
from legoesm.ocean.bathymetry import _laplacian_smooth_2d
from legoesm.ocean.eos import scale_depth as _SCALE_DEPTH


N_LAT, N_LON = 36, 72
N_LEVELS = 20
H_MAX = 4000.0
DZ_SURFACE = 10.0
DZ_DEEP = 500.0

SEAMOUNT_LAT_DEG = 0.0
SEAMOUNT_LON_DEG = 180.0
SEAMOUNT_HEIGHT_M = 3800.0
SEAMOUNT_SIGMA_DEG = 10.0

TOTAL_DAYS = 30.0
DT_SECONDS = 600.0
RECORD_EVERY_DAYS = 1.0


def _build_seamount_bathymetry(grid, smoothing_passes,
                                 height_m=SEAMOUNT_HEIGHT_M,
                                 sigma_deg=SEAMOUNT_SIGMA_DEG):
    lat_deg = np.asarray(grid.lat2d) * 180.0 / np.pi
    lon_deg = np.asarray(grid.lon2d) * 180.0 / np.pi
    dlon = lon_deg - SEAMOUNT_LON_DEG
    dlon = np.where(dlon > 180.0, dlon - 360.0, dlon)
    dlon = np.where(dlon < -180.0, dlon + 360.0, dlon)
    dlat = lat_deg - SEAMOUNT_LAT_DEG
    r2 = dlat ** 2 + dlon ** 2
    seamount = height_m * np.exp(-r2 / (2 * sigma_deg ** 2))
    H_bathy = H_MAX - seamount
    if smoothing_passes > 0:
        H_bathy = _laplacian_smooth_2d(H_bathy, smoothing_passes,
                                         is_cubed=False)
    return jnp.asarray(H_bathy)


def _make_step_block(model, dt):
    def scan_body(state, _):
        return model.step(state, dt), None

    @partial(jax.jit, static_argnames=("n_inner",))
    def block_fn(state, n_inner: int):
        state, _ = jax.lax.scan(scan_body, state, None, length=n_inner)
        return state

    return block_fn


def _r_factor_max(H_bathy):
    H = np.asarray(H_bathy)
    r_e = np.abs(H - np.roll(H, -1, axis=1)) / np.fmax(H, np.roll(H, -1, axis=1))
    r_w = np.abs(H - np.roll(H, +1, axis=1)) / np.fmax(H, np.roll(H, +1, axis=1))
    r_n = np.zeros_like(H); r_s = np.zeros_like(H)
    r_n[:-1, :] = np.abs(H[:-1, :] - H[1:, :]) / np.fmax(H[:-1, :], H[1:, :])
    r_s[1:, :] = np.abs(H[1:, :] - H[:-1, :]) / np.fmax(H[1:, :], H[:-1, :])
    r = np.fmax(np.fmax(r_e, r_w), np.fmax(r_n, r_s))
    return float(r.max())


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--coord", type=str, default="partial",
                        choices=["zstar", "partial"],
                        help="Vertical coord: legacy z* or z* + partial bottom cells.")
    parser.add_argument("--smoothing-passes", type=int, default=0,
                        help="Laplacian smoothing passes on the seamount.")
    parser.add_argument("--seamount-height", type=float,
                        default=SEAMOUNT_HEIGHT_M)
    parser.add_argument("--seamount-sigma", type=float,
                        default=SEAMOUNT_SIGMA_DEG)
    parser.add_argument("--days", type=float, default=TOTAL_DAYS)
    parser.add_argument("--bottom-drag-r", type=float, default=0.0,
                        help="Linear bottom drag coefficient [1/s]. "
                        "Beckmann-Haidvogel 1993 used 1e-3 m/s linear "
                        "drag; for our linear-r convention, ~3e-7 1/s.")
    parser.add_argument("--bbl-thickness", type=float, default=0.0,
                        help="Bottom boundary layer thickness [m] for "
                        "distributed BBL drag (0 = legacy single-level).")
    parser.add_argument("--pgf-scheme", type=str, default="adcroft",
                        choices=["adcroft", "smc03"],
                        help="Pressure-gradient force scheme on partial "
                        "cells.  ``adcroft``: existing Adcroft & Campin "
                        "2004 face correction.  ``smc03``: Shchepetkin & "
                        "McWilliams 2003 density-Jacobian PGF.")
    args = parser.parse_args()

    drag_tag = "_drag{:.0e}".format(args.bottom_drag_r) if args.bottom_drag_r > 0 else ""
    pgf_tag = "" if args.pgf_scheme == "adcroft" else f"_pgf-{args.pgf_scheme}"
    tag = (
        f"{args.coord}_smooth{args.smoothing_passes}"
        f"_h{int(args.seamount_height)}_s{int(args.seamount_sigma)}"
        f"{drag_tag}{pgf_tag}"
    )
    output_dir = Path(
        f"results/realistic_geometry_validation/phase3a_seamount_{tag}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    grid = create_latlon_grid(N_LAT, N_LON)
    z_coord_base = create_ocean_z_star(
        n_levels=N_LEVELS, H_max=H_MAX,
        dz_surface=DZ_SURFACE, dz_deep=DZ_DEEP,
    )

    H_bathy = _build_seamount_bathymetry(
        grid, args.smoothing_passes,
        height_m=args.seamount_height, sigma_deg=args.seamount_sigma,
    )
    r_max = _r_factor_max(H_bathy)

    if args.coord == "partial":
        z_coord = create_partial_cell_coordinate(z_coord_base, H_bathy)
    else:
        z_coord = z_coord_base

    print(f"=== Phase 3a Beckmann-Haidvogel seamount ===")
    print(f"  Coord:           {args.coord}")
    print(f"  PGF scheme:      {args.pgf_scheme}")
    print(f"  Grid: {N_LAT}x{N_LON} (5deg), {N_LEVELS} levels, H_max={H_MAX} m")
    print(f"  Seamount: height {args.seamount_height} m, sigma={args.seamount_sigma}")
    print(f"  Smoothing passes: {args.smoothing_passes}")
    print(f"  r-factor max:    {r_max:.4f}  (B-H stable: < 0.2)")
    print(f"  H_bathy range:   [{float(H_bathy.min()):.0f}, "
          f"{float(H_bathy.max()):.0f}] m")
    print(f"  dt = {DT_SECONDS} s, total = {args.days} sim-days")
    print(f"  Output: {output_dir}")
    print()

    state = rest_state_latlon_cgrid_ocean(
        grid, z_coord_base,
        T_surface=20.0, T_deep=2.0, S_uniform=35.0,
        H_bathy_override=H_bathy,
    )

    if args.coord == "partial":
        # Centroid-aware T initialisation: each cell sits at its actual
        # geometric centroid, which differs from z_full_ref at the
        # partial bottom cells.  Without this, the rest-state PGF has a
        # ~1e-6 m/s^2 residual from the centroid mismatch alone.
        centroid = compute_centroid_depth(
            jnp.zeros_like(H_bathy), H_bathy, z_coord,
        )
        T_per_cell = 2.0 + (20.0 - 2.0) * jnp.exp(-centroid / _SCALE_DEPTH)
        T_per_cell = jnp.where(z_coord.is_active, T_per_cell, 2.0)
        state = state._replace(T=state.T.replace(data=T_per_cell))

    cfg = LatLonCGridOceanConfig(
        barotropic_solver="implicit_cn",
        physics=None,
        bottom_drag_r=args.bottom_drag_r,
        bottom_drag_bbl_thickness=args.bbl_thickness,
        pgf_scheme=args.pgf_scheme,
    )
    model = LatLonCGridOceanModel(grid, z_coord, cfg)
    block_fn = _make_step_block(model, DT_SECONDS)

    n_total_steps = int(args.days * 86400.0 / DT_SECONDS)
    steps_per_record = int(RECORD_EVERY_DAYS * 86400.0 / DT_SECONDS)

    print(f"Starting integration ({n_total_steps} steps, "
          f"{steps_per_record} per record)")
    t0 = time.time()

    times_days = [0.0]
    umax_history = [float(jnp.max(jnp.abs(state.u.data)))]
    eta_max_history = [float(jnp.max(jnp.abs(state.eta.data)))]

    s = state
    for record_idx in range(int(args.days / RECORD_EVERY_DAYS)):
        s = block_fn(s, steps_per_record)
        jax.block_until_ready(s.eta.data)
        day = (record_idx + 1) * RECORD_EVERY_DAYS
        u_max = float(jnp.max(jnp.abs(s.u.data)))
        eta_max = float(jnp.max(jnp.abs(s.eta.data)))
        T_max = float(jnp.max(s.T.data))
        T_min = float(jnp.min(s.T.data))
        times_days.append(day)
        umax_history.append(u_max)
        eta_max_history.append(eta_max)
        elapsed = time.time() - t0
        eta_remaining = elapsed / day * (args.days - day) if day > 0 else 0
        finite = bool(jnp.all(jnp.isfinite(s.u.data)) and
                      jnp.all(jnp.isfinite(s.T.data)))
        print(f"  Day {day:5.1f}/{args.days:.0f}  "
              f"|u|max={u_max*1000:7.3f} mm/s  "
              f"|eta|max={eta_max:.3e} m  "
              f"T=[{T_min:.2f},{T_max:.2f}]  "
              f"finite={finite}  ETA {eta_remaining/60:.1f} min")
        if not finite:
            print("  BLEW UP -- aborting integration")
            break

    wall = time.time() - t0
    print(f"\nIntegration complete in {wall:.0f}s ({wall/60:.1f} min)")

    times_days = np.array(times_days)
    umax_history = np.array(umax_history)
    eta_max_history = np.array(eta_max_history)
    final_umax_mm = umax_history[-1] * 1000

    pass_threshold_mm = 5.0
    finite_final = bool(jnp.all(jnp.isfinite(s.u.data)))
    passed = finite_final and (final_umax_mm <= pass_threshold_mm)
    status = "PASS" if passed else "FAIL"
    print(f"\n--- Phase 3a result ---")
    print(f"Coord:         {args.coord}")
    print(f"Final max|u|:  {final_umax_mm:.3f} mm/s")
    print(f"Threshold:     {pass_threshold_mm:.1f} mm/s")
    print(f"Status:        {status}")
    print(f"r-factor max:  {r_max:.4f}")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(times_days, umax_history * 1000, "o-", color="C0", ms=4)
    ax.axhline(pass_threshold_mm, color="C3", ls="--",
                label=f"Pass threshold ({pass_threshold_mm} mm/s)")
    ax.set_xlabel("Sim day")
    ax.set_ylabel("max|u| (mm/s)")
    ax.set_yscale("log")
    ax.set_title(
        f"Phase 3a Beckmann-Haidvogel seamount  "
        f"(coord={args.coord}, smoothing={args.smoothing_passes}, r_max={r_max:.3f})\n"
        f"Final max|u|={final_umax_mm:.3f} mm/s -- {status}"
    )
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "umax_timeseries.png", dpi=130)
    plt.close()
    print(f"Saved {output_dir / 'umax_timeseries.png'}")

    u_c = 0.5 * (np.asarray(s.u.data[:, :-1, 0]) + np.asarray(s.u.data[:, 1:, 0]))
    v_c = 0.5 * (np.asarray(s.v.data[:-1, :, 0]) + np.asarray(s.v.data[1:, :, 0]))
    speed_sfc = np.sqrt(u_c**2 + v_c**2)
    lat_deg = np.asarray(grid.lat) * 180.0 / np.pi
    lon_deg = np.asarray(grid.lon) * 180.0 / np.pi
    fig, ax = plt.subplots(figsize=(11, 5))
    speed_plot = np.where(np.isfinite(speed_sfc), speed_sfc, 0.0)
    im = ax.pcolormesh(lon_deg, lat_deg, speed_plot * 1000,
                        cmap="hot_r", shading="auto")
    plt.colorbar(im, ax=ax, label="Surface |U| (mm/s)", fraction=0.025)
    H_np = np.asarray(H_bathy)
    cs = ax.contour(lon_deg, lat_deg, H_np / 1000.0,
                     levels=[1, 2, 3], colors="white",
                     linewidths=0.7, alpha=0.7)
    ax.clabel(cs, inline=True, fontsize=8, fmt="%g km")
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.set_title(
        f"Phase 3a Beckmann-Haidvogel:  surface |U| at day {args.days:.0f}  "
        f"(coord={args.coord}, smoothing={args.smoothing_passes})\n"
        f"White contours: bathymetry (km depth)"
    )
    plt.tight_layout()
    plt.savefig(output_dir / "velocity_snapshot.png", dpi=130)
    plt.close()
    print(f"Saved {output_dir / 'velocity_snapshot.png'}")

    seamount_lat_idx = int(np.argmin(np.abs(lat_deg - SEAMOUNT_LAT_DEG)))
    H_section = np.asarray(H_bathy)[seamount_lat_idx, :]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.fill_between(lon_deg, H_section, H_MAX, color="saddlebrown", alpha=0.4,
                     label="Seamount")
    ax.plot(lon_deg, H_section, "-", color="saddlebrown", lw=1.5)
    ax.invert_yaxis()
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Depth (m)")
    ax.set_title(
        f"Bathymetry zonal section at lat={SEAMOUNT_LAT_DEG} deg  "
        f"(coord={args.coord}, smoothing={args.smoothing_passes}, r_max={r_max:.3f})"
    )
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "bathymetry.png", dpi=130)
    plt.close()
    print(f"Saved {output_dir / 'bathymetry.png'}")

    log_path = output_dir / "run.log"
    with open(log_path, "w") as f:
        f.write(f"Phase 3a Beckmann-Haidvogel seamount\n")
        f.write(f"coord = {args.coord}\n")
        f.write(f"pgf_scheme = {args.pgf_scheme}\n")
        f.write(f"smoothing_passes = {args.smoothing_passes}\n")
        f.write(f"r_max = {r_max:.6f}\n")
        f.write(f"days = {args.days}\n")
        f.write(f"final_max_u_mm_per_s = {final_umax_mm:.6f}\n")
        f.write(f"pass_threshold_mm_per_s = {pass_threshold_mm:.1f}\n")
        f.write(f"status = {status}\n")
        f.write(f"finite_final = {finite_final}\n")
        f.write(f"wall_time_s = {wall:.1f}\n")
        f.write(f"\nDay-by-day:\n")
        for d, u, e in zip(times_days, umax_history, eta_max_history):
            f.write(f"  day={d:5.1f}  |u|max={u*1000:.4f} mm/s  "
                    f"|eta|max={e:.3e} m\n")
    print(f"Saved {log_path}")


if __name__ == "__main__":
    main()
