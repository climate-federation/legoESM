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

    JAX_ENABLE_X64=1 python scripts/run/ocean_long_runs/run_centennial_spinup.py \
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

from legoesm import constants


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
                   help="WOA SSS NetCDF cache directory; missing -> error "
                        "unless --allow-synthetic.")
    # --- Dai-Trenberth river runoff ---
    p.add_argument("--runoff", action="store_true",
                   help="Enable Dai-Trenberth global river runoff.")
    p.add_argument("--runoff-cache", type=Path, default=None,
                   help="Dai-Trenberth NetCDF cache directory; missing -> "
                        "error unless --allow-synthetic.")
    # --- Ice-shelf basal melt ---
    p.add_argument("--ice-shelf", action="store_true",
                   help="Enable Holland-Jenkins ice-shelf basal melt "
                        "(requires --ice-shelf-mask + --ice-draft NPY).")
    p.add_argument("--ice-shelf-mask", type=Path, default=None,
                   help="NPY file with the {0,1} cavity mask on the run grid: "
                        "(n_lat, n_lon) for --grid latlon, (nCells,) for "
                        "--grid mpas.")
    p.add_argument("--ice-draft", type=Path, default=None,
                   help="NPY file with ice-base depth [m] on the run grid: "
                        "(n_lat, n_lon) for --grid latlon, (nCells,) for "
                        "--grid mpas.")
    p.add_argument("--ice-shelf-scheme",
                   choices=["three_equation", "linear"],
                   default="three_equation")
    # --- Tidal mixing (Jayne-StLaurent abyssal κ + per-step tracer mixing) ---
    p.add_argument("--tidal-mixing", action="store_true",
                   help="Enable Jayne-StLaurent abyssal tidal κ + per-step "
                        "implicit-Euler vertical tracer mixing of T + S.")
    p.add_argument("--smoke", action="store_true",
                   help="Run a single model day to exercise code paths.")
    p.add_argument("--allow-synthetic", action="store_true",
                   help="Smoke/CI only: allow analytic stand-ins when a forcing "
                        "or observation cache is missing (default: fail).")
    args = p.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    scripts_dir = Path(__file__).resolve().parents[2] / "matrix"
    sys.path.insert(0, str(scripts_dir))

    from legoesm import constants
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
        apply_runoff_step,
        apply_runoff_step_mpas,
        apply_ice_shelf_basal_step,
        apply_ice_shelf_basal_step_mpas,
        apply_tidal_mixing_step,
    )
    from legoesm.ocean.forcing.dai_trenberth import (
        load_dai_trenberth, project_runoff_to_grid,
        project_runoff_to_mpas_cells,
    )
    from legoesm.ocean.physics.ice_shelf import IceShelfConfig
    from legoesm.ocean.physics.vertical_mixing.tidal import (
        TidalMixingConfig,
        brunt_vaisala_cell_from_eos,
        synthetic_baroclinic_tide_energy_from_bathy,
    )
    import jax.numpy as jnp
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
            f"(cache: {args.sss_cache or 'default cache'})"
        )
        sss_woa, lat_woa, lon_woa = load_woa_sss(
            cache_dir=args.sss_cache, allow_synthetic=args.allow_synthetic)
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

    # --- Dai-Trenberth runoff setup ----------------------------------
    runoff_on_grid = None
    runoff_area = None
    if args.runoff:
        if args.grid not in ("latlon", "mpas"):
            print(
                f"==> WARNING: --runoff not supported on grid="
                f"{args.grid!r}; ignoring."
            )
        else:
            print(
                f"==> Loading Dai-Trenberth rivers "
                f"(cache: {args.runoff_cache or 'default cache'})"
            )
            rivers = load_dai_trenberth(
                cache_dir=args.runoff_cache,
                allow_synthetic=args.allow_synthetic)
            ocean_mask = np.asarray(state.land_mask.data, dtype=np.int32)
            if args.grid == "mpas":
                # Unstructured mesh: bin river mouths onto cell centres by
                # great-circle nearest-ocean-cell (project_runoff_to_mpas_cells),
                # the counterpart of the lat-lon binning below. Cell area is the
                # mesh's own areaCell (no cos(lat) fallback needed).
                lat_cell = np.degrees(np.asarray(grid.latCell))
                lon_cell = np.degrees(np.asarray(grid.lonCell))
                runoff_area = np.asarray(grid.areaCell, dtype=np.float64)
                runoff_on_grid = project_runoff_to_mpas_cells(
                    rivers,
                    lat_cell_deg=lat_cell,
                    lon_cell_deg=lon_cell,
                    area_cell_m2=runoff_area,
                    month=None,
                    ocean_mask=ocean_mask,
                )
            else:
                lat_deg = np.degrees(np.asarray(grid.lat))
                lon_deg = np.degrees(np.asarray(grid.lon))
                cell_area = np.asarray(getattr(grid, "area", None))
                if cell_area is None or cell_area.shape != (
                    lat_deg.size, lon_deg.size
                ):
                    # Fallback: cos(lat)-weighted nominal cell area.
                    R_e = float(getattr(grid, "radius", constants.R_earth))
                    dlon_g = 2.0 * np.pi / lon_deg.size
                    dlat_g = np.pi / lat_deg.size
                    cell_area = (
                        R_e * R_e * dlon_g * dlat_g
                        * np.cos(np.deg2rad(lat_deg))[:, None]
                        * np.ones((1, lon_deg.size))
                    )
                runoff_area = cell_area
                runoff_on_grid = project_runoff_to_grid(
                    rivers,
                    grid_lat_deg=lat_deg,
                    grid_lon_deg=lon_deg,
                    cell_area_m2=cell_area,
                    month=None,
                    ocean_mask=ocean_mask,
                )
            # Report the freshwater actually landed vs the river source, so any
            # runoff DROPPED for lack of a nearby ocean cell is visible rather
            # than a silent mass sink (both projectors drop; codex).
            source_kg_s = float(rivers.monthly_flux_kg_s.mean(axis=0).sum())
            landed_kg_s = float((runoff_on_grid * runoff_area).sum())
            dropped_kg_s = max(source_kg_s - landed_kg_s, 0.0)
            dropped_pct = 100.0 * dropped_kg_s / max(source_kg_s, 1.0)
            # Always report the dropped fraction (codex): any freshwater with no
            # ocean cell in range is a mass sink and must be visible, however
            # small.
            print(
                f"   Runoff landed: {landed_kg_s:.3e} kg/s of "
                f"{source_kg_s:.3e} kg/s source "
                f"(dropped {dropped_kg_s:.3e} kg/s = {dropped_pct:.2f}% "
                f"with no ocean cell in range)"
            )

    # --- Ice-shelf setup ---------------------------------------------
    ice_shelf_config = None
    ice_shelf_mask_arr = None
    ice_draft_arr = None
    if args.ice_shelf:
        if args.grid not in ("latlon", "mpas"):
            # Both grids have a tested basal-melt apply
            # (apply_ice_shelf_basal_step / _mpas); anything else is unsupported.
            print(
                f"==> WARNING: --ice-shelf not supported on grid="
                f"{args.grid!r}; ignoring."
            )
        elif args.ice_shelf_mask is None or args.ice_draft is None:
            print(
                "==> WARNING: --ice-shelf requires --ice-shelf-mask + "
                "--ice-draft NPY files; ignoring."
            )
        else:
            ice_shelf_config = IceShelfConfig(
                enabled=True, scheme=args.ice_shelf_scheme,
            )
            ice_shelf_mask_arr = np.asarray(
                np.load(args.ice_shelf_mask), dtype=np.float64,
            )
            ice_draft_arr = np.asarray(
                np.load(args.ice_draft), dtype=np.float64,
            )
            print(
                f"==> Ice-shelf cavity coupling enabled "
                f"(scheme={args.ice_shelf_scheme}, "
                f"{int(ice_shelf_mask_arr.sum())} cavity cells)"
            )

    # --- Tidal mixing setup (Jayne-StLaurent abyssal κ + tracer mix) -
    # Pre-compute the static fields ONCE: E_BT (depends only on
    # bathymetry), layer-depth column (depends only on z_coord +
    # nlev), and the H_bathy snapshot.  Per-step we then recompute
    # h_partial + N² from the current state, evaluate K_tidal, and
    # apply implicit Euler vertical tracer mixing to (T, S).
    tidal_config = None
    tidal_static = None
    if args.tidal_mixing:
        if args.grid not in ("latlon", "mpas"):
            print(
                f"==> WARNING: --tidal-mixing not supported on "
                f"grid={args.grid!r}; ignoring."
            )
        else:
            # Grid-agnostic: the whole tidal-mixing chain
            # (synthetic_baroclinic_tide_energy_from_bathy ->
            # compute_layer_thickness -> the N²/F(z)/K_tidal per-step block ->
            # apply_tidal_mixing_step) operates purely over the trailing
            # vertical axis (`...`, `axis=-1`, `[..., None]`), so it runs on
            # lat-lon (n_lat, n_lon, nlev) AND MPAS (nCells, nlev) unchanged.
            # The ONLY shape-specific piece is the layer-depth broadcast below,
            # which now derives its spatial shape from E_BT_arr rather than
            # assuming 2-D.
            tidal_config = TidalMixingConfig(enabled=True)
            H_bathy_arr = np.asarray(state.H_bathy.data, dtype=np.float64)
            E_BT_arr = np.asarray(
                synthetic_baroclinic_tide_energy_from_bathy(
                    jnp.asarray(H_bathy_arr),
                )
            )
            nlev_static = int(np.asarray(z_coord.dz_ref).shape[0])
            dz_ref_arr = np.asarray(z_coord.dz_ref, dtype=np.float64)[:nlev_static]
            z_edges = np.concatenate([[0.0], np.cumsum(dz_ref_arr)])
            layer_depths_1d = 0.5 * (z_edges[:-1] + z_edges[1:])
            # Broadcast to (*spatial, nlev): (n_lat, n_lon, nlev) on lat-lon,
            # (nCells, nlev) on MPAS -- spatial shape comes from E_BT_arr, which
            # is elementwise in H_bathy and so already carries the grid shape.
            layer_depths_nd = np.broadcast_to(
                layer_depths_1d, E_BT_arr.shape + (nlev_static,),
            ).copy()
            tidal_static = {
                "E_BT": E_BT_arr,
                "layer_depths": layer_depths_nd,
                "H_bathy": H_bathy_arr,
            }
            print(
                f"==> Tidal mixing enabled (per-step tracer mixing); "
                f"E_BT global mean: {float(np.mean(E_BT_arr)):.3e} W/m²"
            )

    # Build the tidal-N² EOS ONCE from the SAME EOS the model integrates
    # (default "wright"), so the tidal diffusivity's stratification is
    # consistent with the dynamical core rather than a hardcoded linear
    # ρ-anomaly (issue #1111). make_eos_fn returns a light closure; the
    # per-step N² evaluates it on the host T/S arrays.
    tidal_eos_fn = None
    tidal_rho_0 = None
    tidal_g = None
    if tidal_static is not None:
        from legoesm.ocean.eos import make_eos_fn
        # Source ρ₀ / g from the SAME model config the dynamical core uses (not
        # module constants). Build the EOS EXACTLY as the core does so tidal
        # density matches the integrated density: the geometric-depth NEMO path
        # passes rho0=config.rho_0 (ocean_pe_latlon_cgrid), the default "insitu"
        # path and MPAS pass none — mirror that conditional here.
        tidal_rho_0 = float(model.config.rho_0)
        tidal_g = float(model.config.g)
        _eos_kw = ({"rho0": tidal_rho_0}
                   if getattr(model.config, "eos_depth", "insitu") == "geometric"
                   else {})
        tidal_eos_fn = make_eos_fn(
            model.config.eos, getattr(model.config, "eos_linear", None),
            **_eos_kw,
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
            allow_synthetic=args.allow_synthetic,
            cycle_years=not args.smoke,   # OMIP-2 repeats the cache's year window
        )
        n_forc = forcing.u10.shape[0]
        # Yearly tidal-κ accumulators (mean/max diagnostic only).
        K_tidal_year_sum = 0.0
        K_tidal_year_max = 0.0
        K_tidal_year_steps = 0
        for step in range(steps_per_year):
            idx_t = (step * n_forc) // steps_per_year
            state = apply_omip2_surface_fluxes(
                state, forcing=forcing, idx_t=idx_t,
                z_coord=z_coord, grid=grid, grid_type=args.grid,
                dt=dt,
            )
            # Dai-Trenberth runoff (when enabled). Grid-dispatched like the
            # SSS / ice-shelf steps: the mpas apply shares the same virtual-salt
            # + eta-rise convention. Both R fields carry the (spatial,) shape
            # their projector produced -- (n_lat, n_lon) or (nCells,).
            if runoff_on_grid is not None:
                if args.grid == "mpas":
                    state = apply_runoff_step_mpas(
                        state,
                        R_kg_m2_s=runoff_on_grid,
                        z_coord=z_coord,
                        dt=dt,
                    )
                else:
                    state = apply_runoff_step(
                        state,
                        R_kg_m2_s=runoff_on_grid,
                        z_coord=z_coord,
                        dt=dt,
                    )

            # Ice-shelf basal melt (when enabled). Grid-dispatched exactly like
            # the SSS-restoring / runoff steps above: the lat-lon and MPAS apply
            # functions share the same three-equation basal-melt convention and
            # both are unit-tested. The MPAS variant was implemented and tested
            # but previously unreachable -- the driver warned "not supported on
            # grid=mpas" for a step it already had.
            if (ice_shelf_config is not None
                    and ice_shelf_mask_arr is not None
                    and ice_draft_arr is not None):
                if args.grid == "mpas":
                    state, _ = apply_ice_shelf_basal_step_mpas(
                        state,
                        ice_shelf_mask=ice_shelf_mask_arr,
                        ice_draft_m=ice_draft_arr,
                        z_coord=z_coord,
                        dt=dt,
                        config=ice_shelf_config,
                    )
                else:
                    state, _ = apply_ice_shelf_basal_step(
                        state,
                        ice_shelf_mask=ice_shelf_mask_arr,
                        ice_draft_m=ice_draft_arr,
                        z_coord=z_coord,
                        dt=dt,
                        config=ice_shelf_config,
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

            # Jayne-StLaurent tidal vertical mixing — applied AFTER
            # the dycore's own step so it layers on top of any other
            # vertical diffusion already inside ``model.step``.  The
            # whole tidal path runs in NumPy on host: ``state.T``
            # and ``state.S`` are already pulled to host in the
            # implicit solver inside ``apply_tidal_mixing_step``,
            # so we keep N², F(z), and K_tidal in NumPy too rather
            # than ping-ponging arrays across the host/device
            # boundary every step.
            if tidal_config is not None and tidal_static is not None:
                h_partial_now = np.asarray(compute_layer_thickness(
                    state.eta.data, state.H_bathy.data, z_coord,
                ), dtype=np.float64)
                T_arr = np.asarray(state.T.data, dtype=np.float64)
                S_arr = np.asarray(state.S.data, dtype=np.float64)
                rho_0_loc = tidal_rho_0
                N2_min = float(tidal_config.N_squared_min)
                # N² from the model-selected EOS (issue #1111): locally-
                # referenced density difference at the shared interface
                # pressure, cell-centred + floored. Evaluated on device via
                # the EOS closure, then pulled back to host for the NumPy
                # tidal path below.
                N2_cell = np.asarray(
                    brunt_vaisala_cell_from_eos(
                        T_arr, S_arr, h_partial_now, tidal_eos_fn,
                        rho_0=rho_0_loc, n_squared_min=N2_min, g=tidal_g,
                    ),
                    dtype=np.float64,
                )

                # Numpy mirror of ``compute_tidal_diffusivity``:
                # K_tidal = Γ q E_BT F(z) / (ρ_0 N²), clipped to K_max,
                # with F(z) = bottom-intensified exp-decay normalised
                # to ``Σ F·h = 1`` per column.
                E_BT_arr = tidal_static["E_BT"]
                layer_depths = tidal_static["layer_depths"]
                H_bathy_loc = tidal_static["H_bathy"]
                h_decay = max(float(tidal_config.h_decay_m), 1.0e-6)
                dist_from_bottom = np.maximum(
                    H_bathy_loc[..., None] - layer_depths, 0.0,
                )
                F_raw = np.exp(-dist_from_bottom / h_decay)
                integral = np.sum(
                    F_raw * h_partial_now, axis=-1, keepdims=True,
                )
                F_norm = np.where(
                    integral > 1.0e-12, F_raw / np.where(
                        integral > 1.0e-12, integral, 1.0,
                    ), 0.0,
                )
                K_tidal = (
                    float(tidal_config.Gamma)
                    * float(tidal_config.q_local)
                    * E_BT_arr[..., None]
                    * F_norm
                ) / (rho_0_loc * N2_cell)
                K_tidal = np.clip(
                    K_tidal, 0.0, float(tidal_config.K_max),
                )

                state = apply_tidal_mixing_step(
                    state,
                    K_tidal=K_tidal,
                    h_partial=h_partial_now,
                    dt=dt,
                )
                K_tidal_year_sum += float(np.mean(K_tidal))
                K_tidal_year_max = max(
                    K_tidal_year_max, float(np.max(K_tidal)),
                )
                K_tidal_year_steps += 1
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

        # Yearly tidal-κ summary from the per-step accumulator.
        if tidal_config is not None and K_tidal_year_steps > 0:
            K_year_mean = K_tidal_year_sum / K_tidal_year_steps
            print(
                f"   tidal-κ: year-mean(<K>) = {K_year_mean:.3e} m²/s, "
                f"year-max = {K_tidal_year_max:.3e} m²/s "
                f"({K_tidal_year_steps} steps)"
            )

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
