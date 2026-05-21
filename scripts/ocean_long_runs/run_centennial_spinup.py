"""Centennial OMIP-2 ocean spin-up driver with auto-resume + AMOC tracking.

Builds on :mod:`scripts.ocean_long_runs.run_omip2` (annual loop + RPE +
tracer budget + yearly restart) and adds:

* Auto-resume from the latest ``restart_year_YYYY.npz`` in the output
  directory — re-running with ``--years 200`` after a year-100 restart
  continues from year 101.
* Per-year AMOC@26.5°N tracking (Atlantic basin if a mask is
  provided; global MOC fallback otherwise).
* ``SpinupHealth`` time series dumped as ``spinup_history.json`` +
  ``spinup_history.csv``.
* :func:`legoesm.ocean.spinup.is_converged` check at the end + after
  every 50 years; emits ``CONVERGED`` line when criteria are met.
* Optional Bryan-Lewis 3-phase accelerated tracer dt via ``--bryan``.

USAGE
-----
::

    JAX_ENABLE_X64=1 python scripts/ocean_long_runs/run_centennial_spinup.py \
        --grid latlon --resolution 36x72 --years 1000 \
        --output results/spinup_1000yr --bryan

Driver imports the (jax) ocean stack on demand so ``--help`` works
without a CUDA device.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np


def _load_history_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cast = {}
            for k, v in row.items():
                try:
                    cast[k] = float(v)
                except (TypeError, ValueError):
                    cast[k] = v
            out.append(cast)
    return out


def _append_history_csv(path: Path, row: dict) -> None:
    file_exists = path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--grid", choices=["latlon", "mpas"], default="latlon")
    p.add_argument("--resolution", type=str, default="36x72")
    p.add_argument("--years", type=int, default=200)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--restart-from", type=Path, default=None,
                   help="Force start from a specific restart; otherwise the "
                        "driver auto-resumes from the latest restart in "
                        "--output.")
    p.add_argument("--jra55-cache", type=Path, default=None)
    p.add_argument("--dt", type=float, default=1800.0)
    p.add_argument("--bryan", action="store_true",
                   help="Use Bryan-Lewis 3-phase accelerated tracer dt.")
    p.add_argument("--bryan-ratio", type=float, default=10.0)
    p.add_argument("--bryan-phase1-years", type=int, default=200)
    p.add_argument("--bryan-phase2-years", type=int, default=100)
    p.add_argument("--convergence-check-every", type=int, default=50,
                   help="Check convergence criteria every N years.")
    p.add_argument("--amoc-target-lat", type=float, default=26.5,
                   help="Latitude (°N) for AMOC monitoring (default 26.5).")
    p.add_argument("--amoc-basin", choices=["atlantic", "global"],
                   default="atlantic",
                   help="Basin filter for AMOC streamfunction.")
    p.add_argument("--amoc-lon-min-deg", type=float, default=-75.0)
    p.add_argument("--amoc-lon-max-deg", type=float, default=15.0)
    # --- OMIP-2 SSS restoring ---
    p.add_argument("--sss-restoring", action="store_true",
                   help="Enable OMIP-2 surface salinity restoring to WOA "
                        "climatology with region-aware tau.")
    p.add_argument("--sss-tau-days", type=float, default=365.0,
                   help="Default interior SSS restoring timescale [days].")
    p.add_argument("--sss-z1-m", type=float, default=10.0,
                   help="Surface restoring layer thickness [m].")
    p.add_argument("--sss-cache", type=Path, default=None,
                   help="WOA SSS NetCDF cache directory; falls back to "
                        "synthetic climatology when missing.")
    p.add_argument("--smoke", action="store_true",
                   help="Run a single model day to exercise code paths.")
    args = p.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    scripts_dir = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(scripts_dir))

    from legoesm.ocean import restart as _restart
    from legoesm.ocean.rpe import compute_rpe, rpe_drift_rate_per_m2
    from legoesm.ocean.budgets import (
        compute_energy_budget, compute_tracer_budget,
    )
    from legoesm.ocean.forcing import load_jra55_do
    from legoesm.ocean.spinup import (
        SpinupHealth, evaluate_health, is_converged,
        ConvergenceCriteria, compute_amoc_timeseries,
        find_latest_restart, bryan_accelerated_dt,
        compute_amoc_from_state,
        compute_amoc_from_state_mpas,
    )
    from legoesm.ocean.vertical import compute_layer_thickness
    from legoesm.ocean.forcing.sss_restoring import (
        SSSRestoringConfig, interp_woa_sss_to_grid,
    )
    from legoesm.ocean.forcing.woa_sss import load_woa_sss
    from legoesm.ocean.coupler import (
        apply_sss_restoring_step,
        apply_sss_restoring_step_mpas,
    )
    from run_omip2 import _build_state  # type: ignore
    import jax

    print(f"==> Building global rest-state on {args.grid}/{args.resolution}")
    state, grid, z_coord, model = _build_state(
        args.grid, args.resolution, scripts_dir=scripts_dir,
    )

    # --- OMIP-2 SSS restoring setup (post-grid) -----------------------
    # Supported on lat-lon and MPAS Voronoi.  WOA SSS lives on a 1°
    # regular lat-lon grid; bilinear interp lands it on either target.
    sss_config = None
    S_target_on_grid = None
    if args.sss_restoring:
        sss_config = SSSRestoringConfig(
            enabled=True,
            tau_restore_days_default=args.sss_tau_days,
            z1_m=args.sss_z1_m,
        )
        print(
            f"==> Loading WOA SSS climatology "
            f"(cache: {args.sss_cache or 'synthetic'})"
        )
        sss_woa, lat_woa, lon_woa = load_woa_sss(cache_dir=args.sss_cache)
        import jax.numpy as _jnp
        if args.grid == "latlon":
            lat_deg = np.degrees(np.asarray(grid.lat))
            lon_deg = np.degrees(np.asarray(grid.lon))
            lat2d = np.broadcast_to(
                lat_deg[:, None], (lat_deg.size, lon_deg.size),
            )
            lon2d = np.broadcast_to(
                lon_deg[None, :], (lat_deg.size, lon_deg.size),
            )
            S_target_on_grid = np.asarray(interp_woa_sss_to_grid(
                _jnp.asarray(sss_woa),
                _jnp.asarray(lat_woa),
                _jnp.asarray(lon_woa),
                _jnp.asarray(lat2d),
                _jnp.asarray(lon2d),
            ))
        elif args.grid == "mpas":
            lat_cell_deg = np.degrees(np.asarray(grid.latCell))
            lon_cell_deg = np.degrees(np.asarray(grid.lonCell))
            S_target_on_grid = np.asarray(interp_woa_sss_to_grid(
                _jnp.asarray(sss_woa),
                _jnp.asarray(lat_woa),
                _jnp.asarray(lon_woa),
                _jnp.asarray(lat_cell_deg),
                _jnp.asarray(lon_cell_deg),
            ))
        else:
            print(
                f"==> WARNING: --sss-restoring not supported on grid="
                f"{args.grid!r}; disabling."
            )
            sss_config = None
            S_target_on_grid = None
        if S_target_on_grid is not None:
            print(
                f"   SSS target range: "
                f"{float(np.min(S_target_on_grid)):.2f}"
                f" – {float(np.max(S_target_on_grid)):.2f} PSU"
            )

    # Auto-resume: prefer ``--restart-from``, fallback to latest in
    # ``--output``.
    resume_path = args.restart_from
    if resume_path is None:
        resume_path = find_latest_restart(args.output)
    start_year = 0
    if resume_path is not None and resume_path.exists():
        print(f"==> Loading restart from {resume_path}")
        state = _restart.load_restart(resume_path, state)
        # Parse year number from filename (best-effort).
        import re
        m = re.search(r"restart_year_(\d+)\.npz$", resume_path.name)
        if m is not None:
            start_year = int(m.group(1))
            print(f"   Resuming at year {start_year + 1}")

    # Initial diagnostics — captured at year 0 on a fresh run, or
    # loaded from the persisted JSON on a resumed run.  The drift
    # baselines (tb0_*) MUST come from the original year-0 state on
    # both code paths so reported drift is consistent across the
    # spin-up regardless of resume boundaries.
    history_csv = args.output / "spinup_history.csv"
    if start_year == 0:
        rpe0 = compute_rpe(state, z_coord, grid_type=args.grid, grid=grid)
        eb0 = compute_energy_budget(state, z_coord, grid_type=args.grid, grid=grid)
        tb0 = compute_tracer_budget(state, z_coord, grid_type=args.grid, grid=grid)
        tb0_volume = float(tb0.volume)
        tb0_heat = float(tb0.heat_content)
        tb0_salt = float(tb0.salt_mass)
        area0 = float(eb0.area_total)
        with open(args.output / "initial_diagnostics.json", "w") as f:
            json.dump({
                "rpe0": float(rpe0),
                "KE0": float(eb0.KE), "APE0": float(eb0.APE),
                "area_total": area0,
                "volume_0": tb0_volume,
                "heat_content_0": tb0_heat,
                "salt_mass_0": tb0_salt,
            }, f, indent=2)
    else:
        with open(args.output / "initial_diagnostics.json", "r") as f:
            init = json.load(f)
        rpe0 = init["rpe0"]
        tb0_volume = init["volume_0"]
        tb0_heat = init["heat_content_0"]
        tb0_salt = init["salt_mass_0"]
        area0 = init["area_total"]

    # Load existing history if resuming.
    history_rows = _load_history_csv(history_csv)
    history: list[SpinupHealth] = []
    for row in history_rows:
        history.append(SpinupHealth(
            year=int(row["year"]),
            amoc_Sv=float(row.get("amoc_Sv", float("nan"))),
            rpe_drift_W_per_m2=float(row["rpe_drift_W_per_m2"]),
            volume_drift_frac=float(row["volume_drift_frac"]),
            heat_drift_frac=float(row["heat_drift_frac"]),
            salt_drift_frac=float(row["salt_drift_frac"]),
        ))

    from legoesm.ocean.coupler import apply_omip2_surface_fluxes

    days_per_year = 1 if args.smoke else 365
    base_dt = float(args.dt)
    wall_t0 = time.time()
    # ``last_rpe`` seeds the per-year drift derivative.  On a fresh
    # run this is the genuine year-0 RPE.  On a resumed run the
    # spin-up has already drifted; using ``rpe0`` here would
    # collapse all accumulated change since year 0 into the first
    # resumed year's drift rate.  Compute RPE on the restart state
    # directly so the first resumed year reports the actual one-
    # year change.
    if start_year == 0:
        last_rpe = rpe0
    else:
        last_rpe = compute_rpe(state, z_coord, grid_type=args.grid, grid=grid)
    last_year = start_year

    # Early-exit guard: a re-run where the latest restart is already
    # at or past the requested target must not silently advance.
    if args.years <= start_year:
        print(
            f"==> Latest restart is at year {start_year}; --years="
            f"{args.years} requested ≤ existing.  Nothing to do."
        )
        return 0
    target_year = args.years
    for y in range(start_year, target_year):
        if args.bryan:
            dt_mom, dt_tra = bryan_accelerated_dt(
                y,
                dt_physical_s=base_dt,
                dt_tracer_ratio=args.bryan_ratio,
                phase1_years=args.bryan_phase1_years,
                phase2_years=args.bryan_phase2_years,
            )
        else:
            dt_mom = base_dt
            dt_tra = base_dt
        # The current model.step does not yet take separate
        # dt_momentum/dt_tracer; use the momentum dt as the effective
        # timestep and log the tracer dt for the bookkeeping record.
        dt = dt_mom
        steps_per_year = int(days_per_year * 86400.0 / dt)

        print(
            f"==> Year {y + 1}/{target_year} ({steps_per_year} steps, "
            f"dt_mom={dt_mom:.0f}s, dt_tra={dt_tra:.0f}s)"
        )
        forcing = load_jra55_do(
            year=(2000 + (y % 60)) if not args.smoke else 0,
            cache_dir=args.jra55_cache,
        )
        n_forc = forcing.u10.shape[0]
        for step in range(steps_per_year):
            idx_t = (step * n_forc) // steps_per_year
            state = apply_omip2_surface_fluxes(
                state, forcing=forcing, idx_t=idx_t,
                z_coord=z_coord, grid=grid, grid_type=args.grid,
                dt=dt,
            )
            # OMIP-2 SSS restoring (when enabled).  Ocean-only driver
            # passes explicit zero ice-fraction; coupled-ice driver
            # should plumb the live ``ice_state.concentration`` so
            # restoring is suppressed under ice and the brine flux
            # (TileResponse.salt_flux) drives the budget there.
            if sss_config is not None and S_target_on_grid is not None:
                ice_open = np.zeros_like(S_target_on_grid)
                if args.grid == "latlon":
                    state = apply_sss_restoring_step(
                        state,
                        S_target=S_target_on_grid,
                        ice_concentration=ice_open,
                        config=sss_config,
                        grid=grid,
                        z_coord=z_coord,
                        dt=dt,
                    )
                elif args.grid == "mpas":
                    state = apply_sss_restoring_step_mpas(
                        state,
                        S_target=S_target_on_grid,
                        ice_concentration=ice_open,
                        config=sss_config,
                        mesh=grid,
                        dt=dt,
                    )
            state = model.step(state, dt)
        state = jax.block_until_ready(state)

        # Yearly diagnostics.
        rpe_y = compute_rpe(state, z_coord, grid_type=args.grid, grid=grid)
        eb_y = compute_energy_budget(state, z_coord, grid_type=args.grid, grid=grid)
        tb_y = compute_tracer_budget(state, z_coord, grid_type=args.grid, grid=grid)
        delta_t_s = float(steps_per_year * dt)
        rpe_flux = rpe_drift_rate_per_m2(
            last_rpe, rpe_y, delta_t_s, eb_y.area_total,
        )
        last_rpe = rpe_y

        # AMOC@target-lat — computed on the lat-lon C-grid and MPAS
        # Voronoi paths.  Other grids fall back to NaN and the
        # convergence helper auto-skips the criterion.
        if args.grid == "latlon":
            h_partial_now = compute_layer_thickness(
                state.eta.data, state.H_bathy.data, z_coord,
            )
            amoc_Sv = compute_amoc_from_state(
                v_face=state.v.data,
                h_partial=h_partial_now,
                land_mask=state.land_mask.data,
                grid=grid,
                target_lat_deg=args.amoc_target_lat,
                basin=args.amoc_basin,
                basin_lon_min_deg=args.amoc_lon_min_deg,
                basin_lon_max_deg=args.amoc_lon_max_deg,
            )
        elif args.grid == "mpas":
            h_cell_now = compute_layer_thickness(
                state.eta.data, state.H_bathy.data, z_coord,
            )
            amoc_Sv = compute_amoc_from_state_mpas(
                u_edge=state.u.data,
                h_cell=h_cell_now,
                mesh=grid,
                target_lat_deg=args.amoc_target_lat,
                basin=args.amoc_basin,
                basin_lon_min_deg=args.amoc_lon_min_deg,
                basin_lon_max_deg=args.amoc_lon_max_deg,
            )
        else:
            amoc_Sv = float("nan")

        # ``tb0_*`` baselines were captured at year 0 (fresh) or
        # loaded from ``initial_diagnostics.json`` (resume); drift
        # is measured against the genuine start of the spin-up.
        health = evaluate_health(
            year=y + 1,
            amoc_Sv=amoc_Sv,
            rpe_drift_W_per_m2=float(rpe_flux),
            volume_now=float(tb_y.volume),
            volume_init=float(tb0_volume),
            heat_now=float(tb_y.heat_content),
            heat_init=float(tb0_heat),
            salt_now=float(tb_y.salt_mass),
            salt_init=float(tb0_salt),
        )
        history.append(health)
        _append_history_csv(history_csv, health._asdict())
        last_year = y + 1

        if np.isfinite(amoc_Sv):
            amoc_str = f"{amoc_Sv:.2f} Sv"
        else:
            amoc_str = "n/a"
        print(
            f"   RPE_flux = {rpe_flux:.3e} W/m²  |  KE = {eb_y.KE:.3e}  "
            f"|  vol_drift = {health.volume_drift_frac:+.3e}  "
            f"|  AMOC@{args.amoc_target_lat:.1f}°N({args.amoc_basin}) = {amoc_str}"
        )

        # Yearly restart.
        _restart.save_restart(
            state, args.output / f"restart_year_{y + 1:04d}.npz",
            time_s=delta_t_s * (y + 1),
            step=steps_per_year * (y + 1),
        )

        # Convergence check.
        if ((y + 1) % args.convergence_check_every == 0
                and len(history) >= 30):
            verdict = is_converged(history, ConvergenceCriteria())
            print(f"   CONVERGENCE @ year {y + 1}: {verdict['converged']}  "
                  f"(amoc_ok={verdict['amoc_ok']}, rpe_ok={verdict['rpe_ok']}, "
                  f"vol_ok={verdict['volume_ok']}, heat_ok={verdict['heat_ok']}, "
                  f"salt_ok={verdict['salt_ok']})")
            with open(args.output / f"convergence_year_{y + 1:04d}.json", "w") as f:
                json.dump(verdict, f, indent=2, default=str)

    # Final report.
    if history:
        final_verdict = is_converged(history, ConvergenceCriteria())
        with open(args.output / "spinup_summary.json", "w") as f:
            json.dump({
                "final_year": last_year,
                "n_years_run": last_year - start_year,
                "wall_time_s": time.time() - wall_t0,
                "convergence": final_verdict,
                "amoc_summary": compute_amoc_timeseries(
                    [h.amoc_Sv for h in history],
                ),
            }, f, indent=2, default=str)
        print(
            f"==> Spin-up complete: {last_year} years, "
            f"converged={final_verdict['converged']}, "
            f"wall={time.time() - wall_t0:.1f}s"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
