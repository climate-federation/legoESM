"""Phase 6 of the SMC03 plan: ETOPO 30-day rest-state stress test.

Loads real ETOPO bathymetry, regrids to a lat-lon grid, applies
modest Laplacian smoothing, and integrates the rest-state for 30
days under implicit-CN barotropic + linear bottom drag, with the
SMC03 density-Jacobian PGF.

This is the headline real-bathymetry check.  Pass criterion (per
``docs/ocean_experiments/density_jacobian_pgf_plan.md`` §3 Phase 6):

- Model integrates 30 sim-days without NaN.
- ``|u|max < 50 mm/s``.
- Spurious flow not concentrated at any particular bathymetric
  feature (visual check on the saved snapshot).

Defaults: 2° resolution (90×180), 20 levels, 30 sim-days.  The
plan's spec called for 1° but 2° captures the same partial-cell
diversity and runs in ~10× less wall time.  Use ``--n-lat 180
--n-lon 360 --n-levels 30`` to run the full plan-spec config.

Usage:

    JAX_ENABLE_X64=1 python scripts/realistic_geometry_validation/run_phase6_etopo.py
    JAX_ENABLE_X64=1 python scripts/realistic_geometry_validation/run_phase6_etopo.py --pgf-scheme adcroft   # baseline
"""

from __future__ import annotations

import argparse
import os
import subprocess
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
from legoesm.ocean.bathymetry import BathymetryConfig, init_ocean_bathymetry
from legoesm.ocean.vertical import (
    create_ocean_z_star,
    create_partial_cell_coordinate,
    compute_centroid_depth,
)
from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
from legoesm.ocean.state import LatLonCGridOceanConfig
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
from legoesm.ocean.eos import scale_depth as _SCALE_DEPTH


DATA_DIR = Path("data/bathymetry")
ETOPO_FILE = DATA_DIR / "etopo_1deg.nc"
ETOPO_URL = (
    "https://upwell.pfeg.noaa.gov/erddap/griddap/etopo180.nc?"
    "altitude%5B(-90):60:(90)%5D%5B(-180):60:(180)%5D"
)


def _ensure_etopo():
    if ETOPO_FILE.exists():
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading ETOPO → {ETOPO_FILE}")
    subprocess.run(
        ["curl", "-fsS", "-o", str(ETOPO_FILE), ETOPO_URL], check=True,
    )


def _make_step_block(model, dt):
    def scan_body(state, _):
        return model.step(state, dt), None

    @partial(jax.jit, static_argnames=("n_inner",))
    def block_fn(state, n_inner: int):
        state, _ = jax.lax.scan(scan_body, state, None, length=n_inner)
        return state

    return block_fn


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-lat", type=int, default=90)
    parser.add_argument("--n-lon", type=int, default=180)
    parser.add_argument("--n-levels", type=int, default=20)
    parser.add_argument("--H-max", type=float, default=5000.0)
    parser.add_argument("--dz-surface", type=float, default=20.0)
    parser.add_argument("--dz-deep", type=float, default=500.0)
    parser.add_argument("--smoothing-passes", type=int, default=5)
    parser.add_argument("--H-min", type=float, default=50.0,
                        help="Minimum ocean depth [m]; cells shallower "
                        "than this become land.  Default 50 m: avoids "
                        "degenerate partial cells thinner than ~2x "
                        "dz_surface that destabilise the implicit-CN "
                        "barotropic on steep continental shelves.")
    parser.add_argument("--bottom-drag-r", type=float, default=1.0e-3)
    parser.add_argument("--bbl-thickness", type=float, default=100.0)
    parser.add_argument("--A-h", type=float, default=1.0e4,
                        help="Horizontal Laplacian viscosity [m²/s].  "
                        "Default 1e4 (LatLonCGridOceanConfig default); "
                        "ETOPO at 2° benefits from 5e4 to suppress "
                        "coastal computational modes.")
    parser.add_argument("--days", type=float, default=30.0)
    parser.add_argument("--dt", type=float, default=600.0)
    parser.add_argument("--record-every-days", type=float, default=1.0)
    parser.add_argument("--pgf-scheme", type=str, default="smc03",
                        choices=["adcroft", "smc03"])
    parser.add_argument("--coord", type=str, default="partial",
                        choices=["zstar", "partial"])
    args = parser.parse_args()

    _ensure_etopo()

    pgf_tag = "" if args.pgf_scheme == "adcroft" else f"_pgf-{args.pgf_scheme}"
    output_dir = Path(
        f"results/realistic_geometry_validation/"
        f"phase6_etopo_{args.coord}_{args.n_lat}x{args.n_lon}_{args.n_levels}lev"
        f"_smooth{args.smoothing_passes}_drag{args.bottom_drag_r:.0e}{pgf_tag}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    grid = create_latlon_grid(args.n_lat, args.n_lon)
    z_coord_base = create_ocean_z_star(
        n_levels=args.n_levels, H_max=args.H_max,
        dz_surface=args.dz_surface, dz_deep=args.dz_deep,
    )

    bathy_cfg = BathymetryConfig(
        source="file",
        path=str(ETOPO_FILE),
        H_max=args.H_max,
        H_min=args.H_min,
        smoothing_passes=args.smoothing_passes,
        enforce_straits=True,
        strait_width_factor=1.0,
        fill_isolated_basins=True,
        depth_is_negative=True,
    )
    H_bathy, ocean_mask = init_ocean_bathymetry(grid, bathy_cfg)
    # Match the rest_state working dtype (float32 by default) to avoid
    # dtype-mismatch carry errors inside the scan-based step block.
    H_bathy = H_bathy.astype(jnp.float32)
    ocean_mask = ocean_mask.astype(jnp.float32)

    if args.coord == "partial":
        z_coord = create_partial_cell_coordinate(z_coord_base, H_bathy)
    else:
        z_coord = z_coord_base

    n_ocean = int(np.sum(ocean_mask))
    n_total = int(ocean_mask.size)
    print(f"=== Phase 6 ETOPO 30-day rest-state ===")
    print(f"  Grid:            {args.n_lat}x{args.n_lon} "
          f"(~{180.0/args.n_lat:.2f}° lat, {360.0/args.n_lon:.2f}° lon)")
    print(f"  Levels:          {args.n_levels}, H_max={args.H_max} m, "
          f"dz=[{args.dz_surface}, {args.dz_deep}]")
    print(f"  Coord:           {args.coord}")
    print(f"  PGF scheme:      {args.pgf_scheme}")
    print(f"  Smoothing:       {args.smoothing_passes} Laplacian passes")
    print(f"  Bottom drag:     r={args.bottom_drag_r:.1e} 1/s, "
          f"BBL={args.bbl_thickness} m")
    print(f"  Wet cells:       {n_ocean}/{n_total} "
          f"({100.0*n_ocean/n_total:.1f}%)")
    print(f"  H_bathy range:   [{float(H_bathy[ocean_mask>0].min()):.0f}, "
          f"{float(H_bathy.max()):.0f}] m")
    print(f"  dt = {args.dt} s, total = {args.days} sim-days")
    print(f"  Output: {output_dir}")
    print()

    state = rest_state_latlon_cgrid_ocean(
        grid, z_coord_base,
        T_surface=20.0, T_deep=2.0, S_uniform=35.0,
        H_bathy_override=H_bathy,
        land_mask_override=ocean_mask,
    )

    if args.coord == "partial":
        # Centroid-aware T initialization to keep the rest-state PGF clean.
        centroid = compute_centroid_depth(
            jnp.zeros_like(H_bathy), H_bathy, z_coord,
        )
        T_per_cell = 2.0 + (20.0 - 2.0) * jnp.exp(-centroid / _SCALE_DEPTH)
        T_per_cell = jnp.where(z_coord.is_active, T_per_cell, 2.0)
        T_per_cell = T_per_cell * state.land_mask.data[..., jnp.newaxis]
        state = state._replace(
            T=state.T.replace(data=T_per_cell.astype(state.T.data.dtype)),
        )

    cfg = LatLonCGridOceanConfig(
        barotropic_solver="implicit_cn",
        physics=None,
        A_h=args.A_h,
        bottom_drag_r=args.bottom_drag_r,
        bottom_drag_bbl_thickness=args.bbl_thickness,
        pgf_scheme=args.pgf_scheme,
    )
    model = LatLonCGridOceanModel(grid, z_coord, cfg)
    block_fn = _make_step_block(model, args.dt)

    n_total_steps = int(args.days * 86400.0 / args.dt)
    steps_per_record = int(args.record_every_days * 86400.0 / args.dt)
    n_records = max(1, int(args.days / args.record_every_days))

    print(f"Starting integration ({n_total_steps} steps, "
          f"{steps_per_record} per record)")
    t0 = time.time()

    times_days = [0.0]
    umax_history = [float(jnp.max(jnp.abs(state.u.data)))]
    eta_max_history = [float(jnp.max(jnp.abs(state.eta.data)))]

    s = state
    blew_up = False
    for record_idx in range(n_records):
        s = block_fn(s, steps_per_record)
        jax.block_until_ready(s.eta.data)
        day = (record_idx + 1) * args.record_every_days
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
              f"|u|max={u_max*1000:8.3f} mm/s  "
              f"|eta|max={eta_max:.3e} m  "
              f"T=[{T_min:.2f},{T_max:.2f}]  "
              f"finite={finite}  ETA {eta_remaining/60:.1f} min")
        if not finite:
            print("  BLEW UP — aborting integration")
            blew_up = True
            break

    wall = time.time() - t0
    print(f"\nIntegration complete in {wall:.0f}s ({wall/60:.1f} min)")

    times_days = np.array(times_days)
    umax_history = np.array(umax_history)
    eta_max_history = np.array(eta_max_history)
    final_umax_mm = umax_history[-1] * 1000

    pass_threshold_mm = 50.0
    finite_final = (not blew_up) and bool(jnp.all(jnp.isfinite(s.u.data)))
    passed = finite_final and (final_umax_mm <= pass_threshold_mm)
    status = "PASS" if passed else "FAIL"
    print(f"\n--- Phase 6 result ---")
    print(f"Coord:         {args.coord}")
    print(f"PGF scheme:    {args.pgf_scheme}")
    print(f"Final max|u|:  {final_umax_mm:.3f} mm/s")
    print(f"Threshold:     {pass_threshold_mm:.1f} mm/s")
    print(f"Status:        {status}")

    # |u|max time series
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(times_days, umax_history * 1000, "o-", color="C0", ms=4)
    ax.axhline(pass_threshold_mm, color="C3", ls="--",
               label=f"Pass threshold ({pass_threshold_mm:.0f} mm/s)")
    ax.set_xlabel("Sim day")
    ax.set_ylabel("max|u| (mm/s)")
    ax.set_yscale("log")
    ax.set_title(
        f"Phase 6 ETOPO {args.n_lat}×{args.n_lon} "
        f"(coord={args.coord}, pgf={args.pgf_scheme}, smooth={args.smoothing_passes})\n"
        f"Final max|u|={final_umax_mm:.3f} mm/s — {status}"
    )
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "umax_timeseries.png", dpi=130)
    plt.close()
    print(f"Saved {output_dir / 'umax_timeseries.png'}")

    # Surface speed snapshot
    if finite_final:
        u_c = 0.5 * (np.asarray(s.u.data[:, :-1, 0]) + np.asarray(s.u.data[:, 1:, 0]))
        v_c = 0.5 * (np.asarray(s.v.data[:-1, :, 0]) + np.asarray(s.v.data[1:, :, 0]))
        speed_sfc = np.sqrt(u_c ** 2 + v_c ** 2)
        lat_deg = np.asarray(grid.lat) * 180.0 / np.pi
        lon_deg = np.asarray(grid.lon) * 180.0 / np.pi
        speed_plot = np.where(np.isfinite(speed_sfc), speed_sfc, 0.0)
        # Mask land cells for clarity.
        land = (np.asarray(ocean_mask) < 0.5)
        speed_plot = np.ma.masked_where(land, speed_plot)
        fig, ax = plt.subplots(figsize=(11, 5))
        im = ax.pcolormesh(lon_deg, lat_deg, speed_plot * 1000,
                           cmap="hot_r", shading="auto")
        plt.colorbar(im, ax=ax, label="Surface |U| (mm/s)", fraction=0.025)
        H_np = np.asarray(H_bathy)
        H_plot = np.where(land, np.nan, H_np)
        cs = ax.contour(lon_deg, lat_deg, H_plot / 1000.0,
                        levels=[1, 2, 3, 4], colors="white",
                        linewidths=0.6, alpha=0.6)
        ax.clabel(cs, inline=True, fontsize=7, fmt="%g km")
        ax.set_xlabel("Longitude (deg)")
        ax.set_ylabel("Latitude (deg)")
        ax.set_title(
            f"Phase 6 ETOPO surface |U| at day {args.days:.0f}  "
            f"(coord={args.coord}, pgf={args.pgf_scheme})\n"
            f"White contours: bathymetry (km depth)"
        )
        plt.tight_layout()
        plt.savefig(output_dir / "velocity_snapshot.png", dpi=130)
        plt.close()
        print(f"Saved {output_dir / 'velocity_snapshot.png'}")

    log_path = output_dir / "run.log"
    with open(log_path, "w") as f:
        f.write(f"Phase 6 ETOPO 30-day rest-state\n")
        f.write(f"coord = {args.coord}\n")
        f.write(f"pgf_scheme = {args.pgf_scheme}\n")
        f.write(f"grid = {args.n_lat}x{args.n_lon}\n")
        f.write(f"n_levels = {args.n_levels}\n")
        f.write(f"smoothing_passes = {args.smoothing_passes}\n")
        f.write(f"bottom_drag_r = {args.bottom_drag_r}\n")
        f.write(f"bbl_thickness = {args.bbl_thickness}\n")
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
