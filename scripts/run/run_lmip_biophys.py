#!/usr/bin/env python
"""Global **biophysics-only** land driver forced by CRU-JRA reanalysis (LMIP).

This is the M3 driver of the LMIP forcing workplan (``docs/land/lmip_s3_scope.md``).
It is a copy of the ``run_lmip_smoke.py`` template with the *idealised* per-step
forcing replaced by **real CRU-JRA reanalysis** (CLM datm format), disaggregated
from 6-hourly to the model timestep and streamed through ``lax.scan`` as an
explicit per-step input (SegmentForcing doctrine).  ``run_lmip_smoke.py`` is kept
untouched as the synthetic-forcing smoke test.

Configuration matches ``run_lmip_smoke`` exactly: ``MultiLayerLandConfig`` with
prescribed seasonal LAI (CLM5 monthly climatology, one-year cycle) and
**carbon disabled** (``carbon="none"``) — energy + water + snow + soil
temperature only.  No NBP; the carbon cycle is a later workstream.

Default timestep is **1 h** (``--dt 3600``); pass ``--dt 1800`` for 30-min steps.
Default grid is **latlon ~2°** (``--grid-type latlon --resolution 90`` -> 90x180).

Usage::

    # synthetic forcing (no data needed), quick CI-scale smoke
    JAX_ENABLE_X64=1 python scripts/run/run_lmip_biophys.py \\
        --surfdata data/legoesm_surfdata_v1.nc --resolution 24 --n-steps 48

    # real CRU-JRA on Derecho
    JAX_ENABLE_X64=1 python scripts/run/run_lmip_biophys.py \\
        --surfdata data/legoesm_surfdata_v1.nc --resolution 90 \\
        --forcing-dir $SCRATCH/crujra --year 2023 --start-doy 196 \\
        --n-steps 240 --dt 3600 --output lmip_biophys
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from legoesm.land.config import MultiLayerLandConfig, LandConfig, resolve_land_config
from legoesm.land.soil_grid import SoilGridConfig
from legoesm.land.canopy import CanopyConfig
from legoesm.land.surface_scheme import SimpleSEBConfig
from legoesm.land.multilayer_land import step_multilayer_land, init_multilayer_land_state
from legoesm.land.slab_land import step_land
from legoesm.land.boundary_data import init_land_surface_data, make_step_land_params_updater
from legoesm.land.forcing import stage_forcing, stage_forcing_years
from legoesm.land.output_tapes import (
    accumulate_tape_step, build_slot_indices, finalize_tape,
    init_tape_accumulator, load_output_config,
)
from legoesm.land.restart import load_land_restart, save_land_restart

U_MIN = 1.0
_SEC_PER_DAY = 86400.0


def make_grid(grid_type: str, resolution: int):
    """Model grid with the ModelDriver/run_amip ``--resolution N`` convention
    (latlon -> N x 2N; cubed_sphere -> CN; gaussian -> TN).  Mirrors
    ``run_lmip_smoke.make_grid`` (kept separate so that smoke driver is untouched).
    """
    from legoesm.driver.config import normalize_grid_type
    gt = normalize_grid_type(grid_type)
    if gt == "cubed_sphere":
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        return create_cubed_sphere(resolution)
    if gt == "gaussian":
        from legoesm.grids.gaussian import create_gaussian_grid
        return create_gaussian_grid(resolution)
    if gt == "latlon":
        from legoesm.grids.latlon import create_latlon_grid
        return create_latlon_grid(resolution)
    raise ValueError(f"unsupported grid_type {grid_type!r}")


def grid_latlon_rad(grid):
    """Per-column ``(lat, lon)`` in radians (the surfdata loader's order)."""
    if hasattr(grid, "lat2d") and hasattr(grid, "lon2d"):
        lat, lon = np.asarray(grid.lat2d), np.asarray(grid.lon2d)
    elif hasattr(grid, "latCell") and hasattr(grid, "lonCell"):
        lat, lon = np.asarray(grid.latCell), np.asarray(grid.lonCell)
    else:
        lat, lon = np.asarray(grid.lat), np.asarray(grid.lon)
    return jnp.asarray(lat.ravel()), jnp.asarray(lon.ravel())


def build_model_times(start_doy: float, dt: float, n_steps: int, *, synthetic: bool):
    """Model step times in seconds since the forcing-year start.

    Synthetic forcing starts its clock at 0, so synthetic runs start at day 0
    regardless of ``--start-doy`` (a warning is printed)."""
    t0 = 0.0 if synthetic else float(start_doy) * _SEC_PER_DAY
    return t0 + dt * np.arange(n_steps, dtype=np.float64)


def _args_from_config(cfg, cli_args) -> argparse.Namespace:
    """Build the internal argument namespace from a validated LMIPConfig.
    CLI still supplies ``--output-dir`` and (optional) ``--restart-from``
    overrides so a chained run doesn't need a config edit."""
    ns = argparse.Namespace(
        output=cli_args.output_dir,
        grid_type=cfg.grid["type"],
        resolution=int(cfg.grid["resolution"]),
        land_mode=cfg.physics["land_mode"],
        surface_scheme=cfg.physics["surface_scheme"],
        bulk=cfg.physics["bulk_scheme"],
        surfdata=cfg.surfdata["path"],
        forcing_dir=cfg.forcing.get("data_dir", ""),
        prefix=cfg.forcing.get("prefix", ""),
        suffix=cfg.forcing.get("suffix", ""),
        year=int(cfg.forcing["year_start"]),
        year_end=int(cfg.forcing["year_end"]),
        start_doy=float(cfg.time["start_doy"]),
        dt=float(cfg.time["dt"]),
        n_steps=int(cfg.time["n_steps"]),
        k_neighbors=int(cfg.forcing.get("k_neighbors", 4)),
        land_mask_file=cfg.land_mask_file,
        land_frac_min=cfg.land_frac_min,
        # CLI overrides the config here for chaining ergonomics.
        restart_from=cli_args.restart_from or cfg.restart.get("from", ""),
        output_config="",                        # embedded output block is used directly
        _cfg_output_tapes=cfg.output,            # -> load_output_config indirection below
    )
    return ns


def run(args) -> int:
    out_dir = Path(args.output); out_dir.mkdir(parents=True, exist_ok=True)
    grid = make_grid(args.grid_type, args.resolution)
    lat_rad, lon_rad = grid_latlon_rad(grid)
    ncol = lat_rad.shape[0]

    # --- land config: SAME as run_lmip_smoke (carbon stays at its default
    #     "none"); --surface-scheme picks two-leaf canopy or SimpleSEB. ---
    surf = (CanopyConfig(max_iters=50, tol=1e-2)
            if args.surface_scheme == "two_leaf_canopy" else SimpleSEBConfig())
    if args.land_mode == "multilayer":
        base_cfg = MultiLayerLandConfig(
            surface_scheme=surf, soil_grid=SoilGridConfig(),
            bulk_scheme=args.bulk, snow_albedo_feedback=True)
        step_fn = step_multilayer_land
    else:
        base_cfg = LandConfig(surface_scheme=surf)
        step_fn = step_land
    base_cfg = resolve_land_config(args.land_mode, base_cfg)
    is_multilayer = (args.land_mode == "multilayer")

    config, _params_nominal, gsd = init_land_surface_data(
        args.surfdata, grid, base_cfg, args.start_doy)

    # --- CRU-JRA forcing: load -> regrid -> disaggregate to the model steps. ---
    # Year range: --year-end defaults to --year (single-year, backward-compat).
    # A larger --year-end triggers multi-year contiguous forcing.
    year_start = int(args.year)
    year_end = int(args.year_end) if args.year_end is not None else year_start
    if year_end < year_start:
        raise SystemExit(f"--year-end ({year_end}) < --year ({year_start})")
    multi_year = year_end > year_start

    # Per-year forcing-file check.  When a data_dir is set, EVERY year in the
    # range must have its Solr file staged.  Silent fallback to synthetic for
    # a missing intermediate year would load a fake full-year climatology and
    # blow up device memory (~30 GB on GPU for one year of 6h global fake
    # forcing).  Fail fast with a clean list + the exact fix command.
    if args.forcing_dir:
        missing = []
        for y in range(year_start, year_end + 1):
            p = Path(args.forcing_dir) / f"{args.prefix}.Solr.{y}{args.suffix}.nc"
            if not p.exists():
                missing.append((y, p))
        if missing:
            print("ERROR: CRU-JRA forcing not staged for the requested years:",
                  file=sys.stderr)
            for y, p in missing:
                print(f"  year {y}: missing {p}", file=sys.stderr)
            print("\nStage them first from a login node (repo root):",
                  file=sys.stderr)
            for y, _ in missing:
                print(f"  ./scripts/data/download_lmip_data.sh --year {y}",
                      file=sys.stderr)
            return 2
        synthetic = False
    else:
        synthetic = True
        if args.start_doy != 0.0:
            print("(synthetic forcing starts at day 0; --start-doy ignored)")

    dt = float(args.dt)
    model_times_s = build_model_times(args.start_doy, dt, args.n_steps, synthetic=synthetic)
    forcing_desc = ("synthetic" if synthetic
                    else f"CRU-JRA {year_start}"
                    + (f"-{year_end}" if multi_year else ""))
    print(f"grid={args.grid_type} | {ncol} columns | surface={args.surface_scheme} | "
          f"carbon={config.carbon.scheme} | dt={dt:.0f}s | n_steps={args.n_steps} | "
          f"forcing={forcing_desc}")
    # allow_synthetic=False when we have real data: any surprise missing file
    # (permission error, corrupted symlink, etc.) surfaces immediately instead
    # of silently substituting fake data.
    allow_syn = synthetic
    if multi_year:
        forcing_xs = stage_forcing_years(
            lat_rad, lon_rad, model_times_s,
            year_start=year_start, year_end=year_end,
            data_dir=(None if synthetic else args.forcing_dir),
            prefix=args.prefix, suffix=args.suffix,
            k_neighbors=args.k_neighbors, allow_synthetic=allow_syn)
    else:
        forcing_xs = stage_forcing(
            lat_rad, lon_rad, model_times_s,
            year=year_start, data_dir=(None if synthetic else args.forcing_dir),
            prefix=args.prefix, suffix=args.suffix,
            k_neighbors=args.k_neighbors, allow_synthetic=allow_syn)
    doy_xs = jnp.asarray(model_times_s / _SEC_PER_DAY)

    # --- initial state (soil/skin T seeded from the first forcing step). ---
    T0 = forcing_xs.T_lowest[0]
    if args.land_mode == "slab":
        from legoesm.core.field import Field
        from legoesm.land.state import LandState
        z = lambda: jnp.zeros(ncol)
        state = LandState(
            T_soil=Field(T0, name="T_soil", units="K"),
            W_bucket=Field(jnp.full(ncol, 100.0), name="W_bucket", units="kg/m2"),
            snow_depth=Field(z(), name="snow_depth", units="kg/m2"),
            snow_age=Field(z(), name="snow_age", units="s"),
            runoff=z(),
        )
    else:
        if args.restart_from:
            # Warm start from a prior end-state — bypass the cold-init T_soil
            # broadcast so the loaded profile survives verbatim.
            state, restart_meta = load_land_restart(
                args.restart_from,
                expected_land_mode="multilayer",
                expected_ncol=ncol,
                expected_n_layers=config.soil_grid.n_layers)
            print(f"restart: loaded state from {args.restart_from} "
                  f"(t_end_s={restart_meta['t_end_s']:.1f}, "
                  f"steps_completed={restart_meta['n_steps_completed']})")
        else:
            state = init_multilayer_land_state(ncol, config, T_init=288.0)
            state = state._replace(T_soil=jnp.broadcast_to(T0[:, None], state.T_soil.shape))

    update_land_params = make_step_land_params_updater(gsd, config.surface_scheme)

    # ----- output tapes (CLM-style history streams; see output_tapes.py) -----
    if getattr(args, "_cfg_output_tapes", None) is not None:
        # Config-driven path: build TapeSpec list from the embedded output block.
        from legoesm.land.output_tapes import TapeSpec
        tape_specs = [TapeSpec(name=t["name"], freq=t["freq"],
                                average=t.get("average", "mean"),
                                vars=tuple(t["vars"]))
                      for t in args._cfg_output_tapes["tapes"]]
    else:
        tape_specs = load_output_config(args.output_config or None)
    tape_slots = {}                                # (slot_idx, n_slots, slot_times) per tape
    tape_accums = {}
    for tape in tape_specs:
        slot_idx, n_slots, slot_times = build_slot_indices(model_times_s, tape.freq)
        tape_slots[tape.name] = (jnp.asarray(slot_idx), n_slots, slot_times)
        tape_accums[tape.name] = init_tape_accumulator(tape, n_slots, ncol)
    print("tapes: " + " | ".join(
        f"{t.name}(freq={t.freq},avg={t.average},vars={len(t.vars)})" for t in tape_specs))

    _ZEROS = jnp.zeros(ncol)                       # slab-mode placeholder for multilayer-only vars

    # ----- scan body: (state, tape_accums) -> next; no per-step output returned.
    def _step_body(carry, xs):
        state, accums = carry
        forcing_t, doy_t, per_tape_slot = xs
        theta_top_t = (state.theta_soil[:, 0] if is_multilayer else jnp.full(ncol, 0.2))
        land_params_t, lai_diag = update_land_params(theta_top_t, doy_t)
        new_state, resp, _ = step_fn(state, forcing_t, config, U_MIN, dt,
                                     lat=lat_rad, land_params=land_params_t, doy=doy_t)
        # Available variables per step -> selected by each tape's spec.
        values = {
            "T_sfc": resp.T_sfc, "albedo": resp.albedo,
            "shflx": resp.shflx, "lhflx": resp.lhflx,
            "runoff": resp.freshwater_flux,
            "precip": forcing_t.precip_total,
            "LAI": lai_diag,
        }
        if is_multilayer:
            values["T_soil_top"] = new_state.T_soil[:, 0]
            values["theta_soil_top"] = new_state.theta_soil[:, 0]
            values["snow_depth"] = new_state.snow_depth
        else:
            values["T_soil_top"] = _ZEROS
            values["theta_soil_top"] = _ZEROS
            values["snow_depth"] = _ZEROS
        new_accums = {}
        for tape in tape_specs:                    # unrolled at trace time
            new_accums[tape.name] = accumulate_tape_step(
                accums[tape.name], tape, per_tape_slot[tape.name],
                {v: values[v] for v in tape.vars})
        return (new_state, new_accums), None

    print(f"stepping {args.n_steps} timestep(s) (lax.scan) ...")
    slot_idx_xs = {name: t[0] for name, t in tape_slots.items()}
    (state, tape_accums), _ = jax.lax.scan(
        _step_body, (state, tape_accums), (forcing_xs, doy_xs, slot_idx_xs))

    # --- land mask + NaN-over-land validation (the smoke PASS/FAIL). ---
    def cover1d(a):
        a = np.asarray(a)
        return a[0] if a.ndim == 2 else a
    if args.land_mask_file:
        from legoesm.grids.topography import load_land_fraction
        land_fraction = np.asarray(load_land_fraction(grid, args.land_mask_file)).ravel()
    else:
        land_fraction = (cover1d(gsd.f_land) + cover1d(gsd.f_lake)
                         + cover1d(gsd.f_glacier))
    land = land_fraction >= args.land_frac_min

    # --- PASS/FAIL: final soil top-layer T must be finite over land. ---
    T_final = np.asarray(state.T_soil[:, 0] if is_multilayer else _ZEROS).ravel()
    nan_land = int(np.isnan(T_final[land]).sum())
    finite = np.all(np.isfinite(T_final[land]))
    status = "PASS" if (nan_land == 0 and finite) else "FAIL"
    print(f"land cells: {int(land.sum())} | NaN final T_soil_top over land: {nan_land} -> {status}")
    if is_multilayer:
        def rng(a):
            return f"[{np.nanmin(a[land]):.2f}, {np.nanmax(a[land]):.2f}]"
        print(f"  final T_soil_top {rng(T_final)} K | "
              f"theta_top {rng(np.asarray(state.theta_soil[:, 0]))} | "
              f"snow_depth {rng(np.asarray(state.snow_depth))} kg/m2")

    # --- per-tape NetCDF writers ---
    #
    # Latlon output uses the standard (time, lat, lon) rectangular layout, so
    # tools like ``xr.plot`` / ncview / panoply just work.  Non-rectangular
    # grids (cubed-sphere, MPAS Voronoi, gaussian) can't be reshape'd cleanly,
    # so they fall back to (time, ncol) with lat/lon as coord vars on ncol.
    lat_deg = np.rad2deg(np.asarray(lat_rad)); lon_deg = np.rad2deg(np.asarray(lon_rad))
    is_latlon = args.grid_type == "latlon"
    if is_latlon:
        nlat, nlon = args.resolution, 2 * args.resolution
        assert nlat * nlon == ncol, f"latlon reshape mismatch: {nlat}*{nlon} != {ncol}"
        lat_1d = lat_deg.reshape(nlat, nlon)[:, 0]
        lon_1d = lon_deg.reshape(nlat, nlon)[0, :]

    try:
        import xarray as xr
        masked = lambda a: np.where(land, np.asarray(a, np.float64), np.nan)
        for tape in tape_specs:
            finalized = finalize_tape(tape_accums[tape.name], tape)   # var -> (n_slots, ncol)
            _, n_slots, slot_times = tape_slots[tape.name]
            dims = ("time", "lat", "lon") if is_latlon else ("time", "ncol")

            def pack(arr):
                arr2 = np.stack([masked(arr[i]) for i in range(n_slots)])
                return arr2.reshape(n_slots, nlat, nlon) if is_latlon else arr2

            data_vars = {v: (dims, pack(finalized[v])) for v in tape.vars}
            coords = {"time": (("time",), slot_times / _SEC_PER_DAY)}   # doy since year_start
            if is_latlon:
                coords.update({"lat": (("lat",), lat_1d), "lon": (("lon",), lon_1d)})
            else:
                coords.update({"lat": (("ncol",), lat_deg), "lon": (("ncol",), lon_deg)})
            attrs = {
                "forcing": "synthetic" if synthetic else f"CRU-JRA {year_start}"
                           + (f"-{year_end}" if year_end > year_start else ""),
                "dt": dt, "start_doy": args.start_doy,
                "grid_type": args.grid_type, "surface_scheme": args.surface_scheme,
                "carbon": config.carbon.scheme,
                "tape_name": tape.name, "tape_freq": tape.freq, "tape_average": tape.average,
                "time_units": "days since year_start Jan 1 (noleap)",
            }
            ds = xr.Dataset(data_vars, coords=coords, attrs=attrs)
            nc = out_dir / f"lmip_biophys.{tape.name}.nc"
            ds.to_netcdf(nc)
            print(f"wrote {nc} ({n_slots} {tape.freq} slots, "
                  f"layout={'lat,lon' if is_latlon else 'ncol'})")
    except Exception as e:  # noqa: BLE001
        print(f"(netcdf write skipped: {e})")

    # --- auto-save the end-of-run state as a chained-run seed (Phase C). ---
    #
    # Filename embeds the model time the state represents so a directory of
    # end-states is an audit trail (chronological on `ls`).  Format:
    #   restart_<YEAR>_d<DDD>h<HH>.npz   (noleap 365-day calendar)
    # where YEAR = year_start + full_365-day-years elapsed, DDD is day-of-year
    # in that year (0-364), HH is hour-of-day (0-23).
    if is_multilayer:
        try:
            t_end_s = float(model_times_s[-1] + dt)
            days_since_start = t_end_s / _SEC_PER_DAY
            year_offset = int(days_since_start // 365)
            year_final = year_start + year_offset
            doy_float = days_since_start - year_offset * 365.0
            doy_int = int(doy_float)
            hour_of_day = int(round((doy_float - doy_int) * 24.0)) % 24
            restart_name = f"restart_{year_final:04d}_d{doy_int:03d}h{hour_of_day:02d}.npz"
            restart_meta = {
                "grid_type": args.grid_type, "resolution": args.resolution,
                "surface_scheme": args.surface_scheme, "bulk_scheme": args.bulk,
                "year": year_start, "year_end": year_end, "dt": dt,
                "n_steps": args.n_steps, "start_doy": args.start_doy,
                "forcing": ("synthetic" if synthetic else "CRU-JRA"),
                "year_final": year_final, "doy_final": doy_int,
                "hour_final": hour_of_day,
            }
            rp = save_land_restart(
                out_dir / restart_name, state,
                land_mode="multilayer", t_end_s=t_end_s,
                n_steps_completed=args.n_steps, metadata=restart_meta)
            print(f"wrote {rp}")
        except Exception as e:  # noqa: BLE001
            print(f"(restart write skipped: {e})")

    return 0 if status == "PASS" else 1


def build_parser() -> argparse.ArgumentParser:
    """Minimal CLI: --config points at a fully-resolved YAML config (typically
    generated by scripts/run/init_experiment.py).  --output-dir + optional
    --restart-from are the only knobs kept outside the YAML — they change every
    run and belong on the command line for chaining ergonomics."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--config", required=True,
                    help="resolved YAML config (see templates/land/ + "
                         "scripts/run/init_experiment.py)")
    ap.add_argument("--output-dir", required=True,
                    help="experiment output directory (auto-created)")
    ap.add_argument("--restart-from", default="",
                    help="override the config's restart.from field — the most "
                         "common per-run change (chained warm starts)")
    return ap


def main() -> None:
    from legoesm.land.lmip_config import load_config
    cli_args = build_parser().parse_args()
    cfg = load_config(cli_args.config)
    sys.exit(run(_args_from_config(cfg, cli_args)))


if __name__ == "__main__":
    main()
