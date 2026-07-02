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
from legoesm.land.forcing import stage_forcing

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
    # Detect real data by the actual Solr stream file (prefix + year), not just
    # the directory, so a prefix/year mismatch warns loudly instead of silently
    # falling back to synthetic.
    solr_file = (Path(args.forcing_dir) / f"{args.prefix}.Solr.{args.year}{args.suffix}.nc"
                 if args.forcing_dir else None)
    synthetic = not (solr_file and solr_file.exists())
    if args.forcing_dir and synthetic:
        print(f"(CRU-JRA Solr file not found: {solr_file}; using synthetic forcing)")
    if synthetic and args.start_doy != 0.0:
        print("(synthetic forcing starts at day 0; --start-doy ignored)")
    dt = float(args.dt)
    model_times_s = build_model_times(args.start_doy, dt, args.n_steps, synthetic=synthetic)
    print(f"grid={args.grid_type} | {ncol} columns | surface={args.surface_scheme} | "
          f"carbon={config.carbon.scheme} | dt={dt:.0f}s | n_steps={args.n_steps} | "
          f"forcing={'synthetic' if synthetic else 'CRU-JRA ' + str(args.year)}")
    forcing_xs = stage_forcing(
        lat_rad, lon_rad, model_times_s,
        year=args.year, data_dir=(None if synthetic else args.forcing_dir),
        prefix=args.prefix, suffix=args.suffix,
        k_neighbors=args.k_neighbors, allow_synthetic=True)
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
        state = init_multilayer_land_state(ncol, config, T_init=288.0)
        state = state._replace(T_soil=jnp.broadcast_to(T0[:, None], state.T_soil.shape))

    update_land_params = make_step_land_params_updater(gsd, config.surface_scheme)

    # ----- scan body: state -> state' + diagnostics, scanning pre-staged forcing.
    def _step_body(state, xs):
        forcing_t, doy_t = xs
        theta_top_t = (state.theta_soil[:, 0] if is_multilayer else jnp.full(ncol, 0.2))
        land_params_t, lai_diag = update_land_params(theta_top_t, doy_t)
        new_state, resp, _ = step_fn(state, forcing_t, config, U_MIN, dt,
                                     lat=lat_rad, land_params=land_params_t, doy=doy_t)
        diag = (resp.T_sfc, resp.shflx, resp.lhflx, lai_diag, resp.albedo, doy_t)
        if is_multilayer:
            diag = diag + (new_state.theta_soil[:, 0], new_state.T_soil[:, 0],
                           new_state.snow_depth)
        return new_state, diag

    print(f"stepping {args.n_steps} timestep(s) (lax.scan) ...")
    state, scan_out = jax.lax.scan(_step_body, state, (forcing_xs, doy_xs))

    if is_multilayer:
        (T_sfc_t, shflx_t, lhflx_t, LAI_t, alb_t, doy_t_arr,
         theta_top_t, T_soil_top_t, snow_depth_t) = scan_out
    else:
        (T_sfc_t, shflx_t, lhflx_t, LAI_t, alb_t, doy_t_arr) = scan_out

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

    T_last = np.asarray(T_sfc_t[-1]).ravel()
    sh_last = np.asarray(shflx_t[-1]).ravel()
    lh_last = np.asarray(lhflx_t[-1]).ravel()
    nan_land = int(np.isnan(T_last[land]).sum())
    finite = np.all(np.isfinite(T_last[land])) and np.all(np.isfinite(sh_last[land]))
    status = "PASS" if (nan_land == 0 and finite) else "FAIL"
    print(f"land cells: {int(land.sum())} | NaN T_sfc over land: {nan_land} -> {status}")
    def rng(a):
        a = a[land]
        return f"[{np.nanmin(a):.2f}, {np.nanmax(a):.2f}]"
    print(f"  T_sfc {rng(T_last)} K | SH {rng(sh_last)} W/m2 | LH {rng(lh_last)} W/m2")

    # --- time-series NetCDF (per-step land diagnostics). ---
    #
    # Latlon output uses the standard (time, lat, lon) rectangular layout, so
    # tools like ``xr.plot`` / ncview / panoply just work.  Non-rectangular
    # grids (cubed-sphere, MPAS Voronoi, gaussian) can't be reshape'd cleanly,
    # so they fall back to (time, ncol) with lat/lon as coord vars on ncol.
    lat_deg = np.rad2deg(np.asarray(lat_rad)); lon_deg = np.rad2deg(np.asarray(lon_rad))
    is_latlon = args.grid_type == "latlon"
    if is_latlon:
        nlat, nlon = args.resolution, 2 * args.resolution   # matches create_latlon_grid
        assert nlat * nlon == ncol, f"latlon reshape mismatch: {nlat}*{nlon} != {ncol}"
        lat_1d = lat_deg.reshape(nlat, nlon)[:, 0]          # unique lat per row
        lon_1d = lon_deg.reshape(nlat, nlon)[0, :]          # unique lon per column
    try:
        import xarray as xr
        masked = lambda a: np.where(land, np.asarray(a, np.float64), np.nan)

        def pack(series):
            """(n_steps, ncol) -> (n_steps, lat, lon) for latlon, else (n_steps, ncol)."""
            arr = np.stack([masked(series[i]) for i in range(args.n_steps)])   # (time, ncol)
            return arr.reshape(args.n_steps, nlat, nlon) if is_latlon else arr

        dims = ("time", "lat", "lon") if is_latlon else ("time", "ncol")
        ts = {"T_sfc": (dims, pack(T_sfc_t)),
              "shflx": (dims, pack(shflx_t)),
              "lhflx": (dims, pack(lhflx_t)),
              "LAI":   (dims, pack(LAI_t))}
        if is_multilayer:
            ts["T_soil_top"] = (dims, pack(T_soil_top_t))
            ts["snow_depth"] = (dims, pack(snow_depth_t))

        coords = {"time": (("time",), np.asarray(doy_t_arr))}
        if is_latlon:
            coords.update({"lat": (("lat",), lat_1d), "lon": (("lon",), lon_1d)})
        else:
            coords.update({"lat": (("ncol",), lat_deg), "lon": (("ncol",), lon_deg)})

        ds = xr.Dataset(
            ts, coords=coords,
            attrs={"forcing": "synthetic" if synthetic else f"CRU-JRA {args.year}",
                   "dt": dt, "start_doy": args.start_doy, "grid_type": args.grid_type,
                   "surface_scheme": args.surface_scheme, "carbon": config.carbon.scheme},
        )
        nc = out_dir / "lmip_biophys.nc"
        ds.to_netcdf(nc)
        print(f"wrote {nc} ({args.n_steps} steps, layout={'lat,lon' if is_latlon else 'ncol'})")
    except Exception as e:  # noqa: BLE001
        print(f"(netcdf write skipped: {e})")

    return 0 if status == "PASS" else 1


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--surfdata", default="data/legoesm_surfdata_c250617.nc",
                    help="harmonized surfdata NetCDF (staged from Zenodo by "
                         "download_lmip_data.sh)")
    ap.add_argument("--land-mode", default="multilayer", choices=["multilayer", "slab"])
    ap.add_argument("--surface-scheme", default="two_leaf_canopy",
                    choices=["two_leaf_canopy", "simple_seb"])
    ap.add_argument("--bulk", default="most", choices=["constant", "most"],
                    help="surface bulk-flux scheme (default MOST — stability-dependent "
                         "exchange; 'constant' uses fixed Cd/Ch as in earlier smoke runs)")
    ap.add_argument("--grid-type", default="latlon",
                    choices=["latlon", "gaussian", "cubed_sphere"])
    ap.add_argument("--resolution", type=int, default=90,
                    help="grid size N (latlon -> N x 2N; ~2deg at N=90)")
    ap.add_argument("--forcing-dir", default="data/crujra",
                    help="directory with CRU-JRA CLM streams (staged by "
                         "download_lmip_data.sh); missing files -> synthetic forcing")
    ap.add_argument("--prefix", default="clmforc.CRUJRAv2.5_filled_antarct_and_grnlnd_0.5x0.5",
                    help="CRU-JRA CLM filename prefix "
                         "(<prefix>.{Solr,Prec,TPQWL}.<year><suffix>.nc)")
    ap.add_argument("--suffix", default="",
                    help="optional filename suffix after the year (CLM naming variants)")
    ap.add_argument("--year", type=int, default=1920, help="CRU-JRA forcing year")
    ap.add_argument("--start-doy", type=float, default=0.0,
                    help="start day-of-year (real forcing); ignored for synthetic")
    ap.add_argument("--dt", type=float, default=3600.0, help="timestep [s] (1h default; 1800 for 30min)")
    ap.add_argument("--n-steps", type=int, default=48, help="number of land steps")
    ap.add_argument("--k-neighbors", type=int, default=4, help="forcing regrid IDW neighbours")
    ap.add_argument("--land-mask-file", default="")
    ap.add_argument("--land-frac-min", type=float, default=0.5)
    ap.add_argument("--output", default="lmip_biophys")
    return ap


def main() -> None:
    sys.exit(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
