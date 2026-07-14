#!/usr/bin/env python
"""Global single-timestep land **smoke test** (LMIP).

Runs **one timestep of the global land model** (multilayer / slab; two-leaf
canopy or SimpleSEB) driven by the harmonized surface dataset
(``legoesm_surfdata`` — CLM5 cover/PFT/LAI + HWSD soil; see
``legoesm.land.surface_data``).  Purpose: a fast end-to-end smoke check that the
surface-data loader -> param-provider -> land-step path runs on any grid/scheme
and produces finite fields over land.  It is NOT a production run (one step,
idealised forcing); the production global driver is ``run_lmip.py``.

Surface data wiring (all from the surfdata, per column):
  - soil hydraulics  <- Cosby pedotransfer on HWSD sand/clay (per-column
    Clapp-Hornberger; sandy fallback where HWSD has no soil);
  - PFT photosynthesis/aero params <- dominant CLM5 PFT (canopy biome tables);
  - LAI + canopy height <- surfdata monthly climatology at the run day-of-year;
  - background albedo (ALB_VIS/NIR) <- CLM soil-colour class + top-layer wetness.

Atmospheric forcing is idealised (solar geometry + latitudinal/seasonal/diurnal
temperature) — no ERA5 needed for a single diagnostic step.

Usage::

    JAX_ENABLE_X64=1 python scripts/run/run_lmip_smoke.py \\
        --surfdata data/legoesm_surfdata_v1.nc --grid-type latlon --resolution 48 \\
        --doy 196 --hour 12 --output lmip_smoke
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

from legoesm import constants
from legoesm.thermo import saturation_mixing_ratio
from legoesm.core.coupling_fields import AtmToSurface
from legoesm.land.config import MultiLayerLandConfig, LandConfig, resolve_land_config
from legoesm.land.soil_grid import SoilGridConfig
from legoesm.land.canopy import CanopyConfig
from legoesm.land.surface_scheme import SimpleSEBConfig
from legoesm.land.multilayer_land import step_multilayer_land, init_multilayer_land_state
from legoesm.land.slab_land import step_land
from legoesm.land.global_surface_data import interp_monthly
from legoesm.land.soil_albedo import soil_albedo_broadband
from legoesm.land.boundary_data import (
    dominant_pft_index, glacier_mask, surface_data_to_land_params,
    init_land_surface_data, fill_land_param_gaps,
    make_step_land_params_updater)

U_MIN = 1.0


def make_grid(grid_type: str, resolution: int):
    """Build a model grid with the SAME factories + ``--resolution N``
    convention ModelDriver/run_amip use (see driver.cli_resolution):

      * cubed_sphere → ``CN``            (N cells per face edge)
      * latlon       → ``N x 2N``        (n_lon defaults to 2*n_lat)
      * gaussian     → ``TN`` truncation (n_max = N, e.g. T106)
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
        return create_latlon_grid(resolution)        # n_lon = 2*resolution
    raise ValueError(f"unsupported grid_type {grid_type!r}")


def grid_latlon_rad(grid):
    """Per-column ``(lat, lon)`` in **radians** for any grid (the loader's order)."""
    if hasattr(grid, "lat2d") and hasattr(grid, "lon2d"):
        lat, lon = np.asarray(grid.lat2d), np.asarray(grid.lon2d)
    elif hasattr(grid, "latCell") and hasattr(grid, "lonCell"):
        lat, lon = np.asarray(grid.latCell), np.asarray(grid.lonCell)
    else:
        lat, lon = np.asarray(grid.lat), np.asarray(grid.lon)
    return jnp.asarray(lat.ravel()), jnp.asarray(lon.ravel())


def make_global_forcing(lat_rad, lon_rad, doy, hour, *, dtype=jnp.float64):
    """Idealised per-column atmospheric forcing (shape (ncol,))."""
    lat = jnp.asarray(lat_rad, dtype=dtype)
    lon_deg = jnp.asarray(lon_rad, dtype=dtype) * 180.0 / jnp.pi
    decl = 23.45 * jnp.pi / 180.0 * jnp.sin(2.0 * jnp.pi * (doy - 80.0) / 365.0)
    local_hour = hour + lon_deg / 15.0
    ha = (local_hour - 12.0) * 15.0 * jnp.pi / 180.0
    cos_sza = jnp.maximum(
        jnp.sin(lat) * jnp.sin(decl) + jnp.cos(lat) * jnp.cos(decl) * jnp.cos(ha), 0.0
    )
    sw_down = constants.S_0 * cos_sza

    T_base = 288.0 - 30.0 * jnp.abs(lat) / (jnp.pi / 2.0)
    T_season = 15.0 * jnp.abs(lat) / (jnp.pi / 2.0) * jnp.cos(2.0 * jnp.pi * (doy - 200.0) / 365.0)
    T_diurnal = 3.0 * jnp.cos(2.0 * jnp.pi * (local_hour - 14.0) / 24.0)
    T_atm = (T_base + T_season + T_diurnal).astype(dtype)

    p_sfc = jnp.full_like(T_atm, 1.0e5)
    q_atm = (0.6 * saturation_mixing_ratio(T_atm, p_sfc)).astype(dtype)
    precip = jnp.zeros_like(T_atm)
    one = jnp.ones_like(T_atm)
    return AtmToSurface(
        sw_down=sw_down.astype(dtype),
        lw_down=(0.75 * constants.sigma_sb * T_atm ** 4).astype(dtype),
        precip_total=precip, precip_snow=precip,
        T_lowest=T_atm, q_lowest=q_atm,
        u_lowest=3.0 * one, v_lowest=2.0 * one,
        p_lowest=0.95e5 * one, p_surface=p_sfc, rho_lowest=1.2 * one,
        cos_zenith=cos_sza.astype(dtype), co2_ppmv=412.0 * one,
        has_radiation=True, has_precipitation=False,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--surfdata", default="data/legoesm_surfdata_v1.nc")
    # Scheme is specified the same way the drivers' config does: land_mode +
    # land_config (here the land_config's surface_scheme via --surface-scheme).
    ap.add_argument("--land-mode", default="multilayer", choices=["multilayer", "slab"],
                    help="land model (CoupledConfig land_mode convention)")
    ap.add_argument("--surface-scheme", default="two_leaf_canopy",
                    choices=["two_leaf_canopy", "simple_seb"],
                    help="surface scheme inside the land config")
    ap.add_argument("--grid-type", default="latlon",
                    choices=["latlon", "gaussian", "cubed_sphere"],
                    help="model grid (same factories as ModelDriver)")
    ap.add_argument("--resolution", type=int, default=48,
                    help="grid size N (run_amip convention): latlon -> N x 2N; "
                         "cubed_sphere -> CN; gaussian -> TN truncation (e.g. 106)")
    ap.add_argument("--doy", type=float, default=196.0,
                    help="starting day-of-year (LAI + solar)")
    ap.add_argument("--hour", type=float, default=12.0, help="starting UTC hour")
    ap.add_argument("--dt", type=float, default=1800.0)
    ap.add_argument("--n-steps", type=int, default=1,
                    help="number of land steps to take (>1 enables the time loop with "
                         "per-step LAI interpolation from gsd.lai_monthly and per-step "
                         "atmospheric forcing — solar geometry, seasonal/diurnal T)")
    ap.add_argument("--output-every", type=int, default=1,
                    help="save a time-series snapshot every N steps (n-steps>1)")
    ap.add_argument("--output", default="lmip_global")
    ap.add_argument("--land-mask-file", default="",
                    help="land-sea mask NetCDF (CMIP6 sftlf / ERA5 lsm), as ModelDriver "
                         "uses it; authoritative. When empty, the surfdata's own "
                         "land fraction is used as a standalone fallback.")
    ap.add_argument("--land-frac-min", type=float, default=0.5,
                    help="cells with land fraction below this are masked as ocean in output")
    ap.add_argument("--no-plot", dest="plot", action="store_false",
                    help="skip the maps PNG")
    args = ap.parse_args()

    out_dir = Path(args.output); out_dir.mkdir(parents=True, exist_ok=True)
    grid = make_grid(args.grid_type, args.resolution)
    lat_rad, lon_rad = grid_latlon_rad(grid)
    ncol = lat_rad.shape[0]
    print(f"grid={args.grid_type} | {ncol} columns | "
          f"land_mode={args.land_mode} surface={args.surface_scheme}")
    forcing = make_global_forcing(lat_rad, lon_rad, args.doy, args.hour)

    # --- land config from (land_mode, land_config), same convention as the
    #     drivers; --surface-scheme sets the land_config's surface scheme. ---
    surf = (CanopyConfig(max_iters=50, tol=1e-2)
            if args.surface_scheme == "two_leaf_canopy" else SimpleSEBConfig())
    if args.land_mode == "multilayer":
        base_cfg = MultiLayerLandConfig(surface_scheme=surf, soil_grid=SoilGridConfig())
        step_fn = step_multilayer_land
    else:                                                  # slab
        base_cfg = LandConfig(surface_scheme=surf)
        step_fn = step_land
    base_cfg = resolve_land_config(args.land_mode, base_cfg)

    # --- run the surface-data loader at simulation start: regrid to this grid,
    #     derive scheme-appropriate config (soil hydraulics) + land params. ---
    config, _params_nominal, gsd = init_land_surface_data(
        args.surfdata, grid, base_cfg, args.doy)

    # --- state (soil/skin T initialised near the local air temperature) ---
    if args.land_mode == "slab":
        from legoesm.core.field import Field
        from legoesm.land.state import LandState
        z = lambda: jnp.zeros(ncol)
        # Seed ``runoff`` as a zero array (default is ``None``); step_land returns
        # a real array, so leaving it ``None`` makes the input/output pytree
        # structures differ and lax.scan refuses to run.  TgC stays ``None`` for
        # SimpleSEB and is preserved by the step.
        state = LandState(
            T_soil=Field(forcing.T_lowest, name="T_soil", units="K"),
            W_bucket=Field(jnp.full(ncol, 100.0), name="W_bucket", units="kg/m2"),
            snow_depth=Field(z(), name="snow_depth", units="kg/m2"),
            snow_age=Field(z(), name="snow_age", units="s"),
            runoff=z(),
        )
        theta_top = jnp.full(ncol, 0.2)               # slab has no soil profile
    else:
        state = init_multilayer_land_state(ncol, config, T_init=288.0)
        state = state._replace(
            T_soil=jnp.broadcast_to(forcing.T_lowest[:, None], state.T_soil.shape))
        theta_top = state.theta_soil[:, 0]

    # JAX-pure per-step LAI/albedo updater.  Captures the static PFT lookups,
    # soil_color, glacier/covered masks once; per step it just recomputes LAI
    # (from monthly climatology at the traced doy) and the soil-colour
    # background albedo (from the evolving top-layer wetness).  This lets the
    # entire time loop run inside a single ``lax.scan``.
    update_land_params = make_step_land_params_updater(gsd, config.surface_scheme)
    # Idealised smoke uses single-year surfdata; interp_annual returns the one
    # slice for any year, so cover is constant across the run.
    cover_year = jnp.asarray(float(np.asarray(gsd.years)[0]))

    # ----- scan body: state -> state' + diagnostics, fully JAX-traceable -----
    is_multilayer = (args.land_mode == "multilayer")
    dt = float(args.dt)
    dt_days = dt / 86400.0
    dt_hours = dt / 3600.0
    doy_start = jnp.asarray(float(args.doy))
    hour_start = jnp.asarray(float(args.hour))

    def _step_body(state, step_i):
        doy_t = doy_start + step_i.astype(doy_start.dtype) * dt_days
        hour_t = jnp.mod(hour_start + step_i.astype(hour_start.dtype) * dt_hours, 24.0)
        forcing_t = make_global_forcing(lat_rad, lon_rad, doy_t, hour_t)
        theta_top_t = (state.theta_soil[:, 0] if is_multilayer
                       else jnp.full(ncol, 0.2))
        land_params_t, lai_diag = update_land_params(theta_top_t, doy_t, cover_year)
        new_state, resp, _ = step_fn(state, forcing_t, config, U_MIN, dt,
                                     lat=lat_rad,
                                     land_params=land_params_t, doy=doy_t)
        diag = (resp.T_sfc, resp.shflx, resp.lhflx, lai_diag, resp.albedo, doy_t)
        if is_multilayer:
            diag = diag + (new_state.theta_soil[:, 0],
                           new_state.T_soil[:, 0],
                           new_state.snow_depth)
        return new_state, diag

    print(f"stepping {args.n_steps} timestep(s) (dt={dt:.0f}s, "
          f"doy_start={float(args.doy):.2f}, lax.scan) ...")
    scan_steps = jnp.arange(args.n_steps)
    state, scan_out = jax.lax.scan(_step_body, state, scan_steps)

    # Unpack scan outputs (each element has leading axis = n_steps).
    if is_multilayer:
        (T_sfc_t, shflx_t, lhflx_t, LAI_t, alb_t, doy_t_arr,
         theta_top_t, T_soil_top_t, snow_depth_t) = scan_out
    else:
        (T_sfc_t, shflx_t, lhflx_t, LAI_t, alb_t, doy_t_arr) = scan_out

    # Subsample with --output-every.
    sel = np.arange(0, args.n_steps, args.output_every)
    if sel[-1] != args.n_steps - 1:
        sel = np.append(sel, args.n_steps - 1)
    diagnostics = {
        "time_doy": [float(x) for x in np.asarray(doy_t_arr)[sel]],
        "T_sfc":  [np.asarray(T_sfc_t[i])  for i in sel],
        "shflx":  [np.asarray(shflx_t[i])  for i in sel],
        "lhflx":  [np.asarray(lhflx_t[i])  for i in sel],
        "LAI":    [np.asarray(LAI_t[i])    for i in sel],
        "albedo": [np.asarray(alb_t[i])    for i in sel],
    }
    if is_multilayer:
        diagnostics["theta_top"]   = [np.asarray(theta_top_t[i])   for i in sel]
        diagnostics["T_soil_top"]  = [np.asarray(T_soil_top_t[i])  for i in sel]
        diagnostics["snow_depth"]  = [np.asarray(snow_depth_t[i])  for i in sel]

    # Use the LAST scan step's outputs for the legacy single-snapshot maps —
    # cheaper than running an extra step and keeps n_steps=1 bit-equivalent
    # to the prior single-step path.
    class _LastResp:                              # minimal stand-in for the
        T_sfc = T_sfc_t[-1]                       # response NamedTuple
        shflx = shflx_t[-1]
        lhflx = lhflx_t[-1]
        albedo = alb_t[-1]
    resp = _LastResp
    new_state = state                              # final post-scan state

    # --- authoritative land-sea mask (matches ModelDriver), per column ---
    # With a mask file (sftlf / ERA5 lsm) it is the authority, exactly as
    # ModelDriver uses load_land_fraction(grid, land_mask_path).  Standalone (no
    # file), fall back to the surfdata's own land fraction = soil/veg+lake+glacier.
    def cover1d(a):
        a = np.asarray(a)
        return (a[0] if a.ndim == 2 else a)
    f_soil_veg = cover1d(gsd.f_land)
    if args.land_mask_file:
        from legoesm.grids.topography import load_land_fraction
        land_fraction = np.asarray(load_land_fraction(grid, args.land_mask_file)).ravel()
    else:
        land_fraction = f_soil_veg + cover1d(gsd.f_lake) + cover1d(gsd.f_glacier)
    land = land_fraction >= args.land_frac_min                # (ncol,)

    import warnings
    def col(a):                                               # mask to land (ncol,)
        return np.where(land, np.asarray(a, dtype=np.float64).ravel(), np.nan)
    def layer_mean(a):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN soil cols
            return np.nanmean(np.asarray(a), axis=1)

    # Surfdata-derived spatial inputs (scheme-agnostic — straight from gsd).
    dom = dominant_pft_index(gsd)
    lai_dom = np.asarray(interp_monthly(gsd.lai_monthly, jnp.asarray(args.doy)))[
        np.arange(ncol), dom]
    soil_bg = np.asarray(soil_albedo_broadband(
        jnp.asarray(np.asarray(gsd.soil_color)), jnp.full(ncol, 0.2)))
    fields = {
        "sand_pct":     (col(layer_mean(gsd.sand_frac) * 100.0), "YlOrBr", "sand %"),
        "clay_pct":     (col(layer_mean(gsd.clay_frac) * 100.0), "BuPu", "clay %"),
        "organic":      (col(layer_mean(gsd.organic)), "YlGn", "organic"),
        "bulk_density": (col(layer_mean(gsd.bulk_density)), "cividis", "bulk density kg/m3"),
        "soil_color":   (col(gsd.soil_color), "viridis", "soil colour class"),
        "LAI":          (col(lai_dom), "YlGn", "LAI (dominant PFT)"),
        "dominant_pft": (col(dom), "tab20", "dominant CLM5 PFT index"),
        "albedo_bg":    (col(soil_bg), "Greys_r", "soil background albedo (broadband)"),
        "land_fraction": (np.asarray(land_fraction, np.float64), "Blues", "land fraction"),
        "glacier":      (col(glacier_mask(gsd).astype(float)), "cool", "glacier (ice) mask"),
        "albedo_out":   (col(resp.albedo), "Greys_r", "surface albedo (model)"),
        "T_sfc":        (col(resp.T_sfc), "magma", "surface T [K]"),
        "shflx":        (col(resp.shflx), "RdBu_r", "sensible heat [W/m2]"),
        "lhflx":        (col(resp.lhflx), "viridis", "latent heat [W/m2]"),
    }

    n_land = int(land.sum())
    nan_land = int(np.isnan(fields["T_sfc"][0][land]).sum())
    status = "PASS" if nan_land == 0 else "FAIL"
    print(f"land cells: {n_land} | NaN T_sfc over land: {nan_land} -> {status}")
    rng = lambda a: f"[{np.nanmin(a):.2f}, {np.nanmax(a):.2f}]"
    for k in ("sand_pct", "clay_pct", "bulk_density", "LAI", "T_sfc"):
        print(f"  {k:12s} {rng(fields[k][0])}")

    lat_deg = np.rad2deg(np.asarray(lat_rad)); lon_deg = np.rad2deg(np.asarray(lon_rad))
    try:
        import xarray as xr
        ds = xr.Dataset(
            {k: (("ncol",), v[0]) for k, v in fields.items()},
            coords={"lat": (("ncol",), lat_deg), "lon": (("ncol",), lon_deg)},
            attrs={"doy": args.doy, "hour": args.hour, "surfdata": args.surfdata,
                   "grid_type": args.grid_type, "land_mode": args.land_mode,
                   "surface_scheme": args.surface_scheme},
        )
        nc = out_dir / "lmip_global_step.nc"
        ds.to_netcdf(nc)
        print(f"wrote {nc}")

        if args.n_steps > 1:
            ts_vars = {k: (("time", "ncol"), np.stack(v, axis=0))
                       for k, v in diagnostics.items() if k != "time_doy"}
            ts_ds = xr.Dataset(
                ts_vars,
                coords={"time": (("time",), np.asarray(diagnostics["time_doy"])),
                        "lat": (("ncol",), lat_deg), "lon": (("ncol",), lon_deg)},
                attrs={"n_steps": args.n_steps, "dt": args.dt,
                       "doy_start": args.doy,
                       "doy_end": float(np.asarray(doy_t_arr[-1])),
                       "output_every": args.output_every,
                       "grid_type": args.grid_type, "land_mode": args.land_mode,
                       "surface_scheme": args.surface_scheme},
            )
            ts_nc = out_dir / "land_spinup.nc"
            ts_ds.to_netcdf(ts_nc)
            print(f"wrote {ts_nc} ({len(diagnostics['time_doy'])} snapshots)")
    except Exception as e:  # noqa: BLE001
        print(f"(netcdf write skipped: {e})")

    if args.plot:
        _plot_maps(fields, lat_deg, lon_deg, out_dir / "lmip_global_maps.png",
                   doy=args.doy, grid_type=args.grid_type)

    if status == "FAIL":
        sys.exit(1)


def _column_to_dataarray(arr, lat_deg, lon_deg):
    """Per-column field -> 2D (lat, lon) ``xr.DataArray`` for xarray plotting.

    Regular lat-lon / Gaussian grids (ncol == n_uniq_lat * n_uniq_lon) are
    reshaped exactly by scattering each column into its (lat, lon) cell.
    Irregular grids (cubed-sphere) are interpolated to a 1° display grid.
    """
    import xarray as xr

    arr = np.asarray(arr)
    lon = np.where(lon_deg > 180.0, lon_deg - 360.0, lon_deg)   # [-180,180)
    ulat, ilat = np.unique(np.round(lat_deg, 4), return_inverse=True)
    ulon, ilon = np.unique(np.round(lon, 4), return_inverse=True)
    if ulat.size * ulon.size == arr.size:                       # regular grid
        grid = np.full((ulat.size, ulon.size), np.nan, dtype=float)
        grid[ilat, ilon] = arr
        return xr.DataArray(grid, coords={"lat": ulat, "lon": ulon},
                            dims=("lat", "lon"))
    from scipy.interpolate import griddata                      # cubed-sphere etc.
    dlat = np.arange(-89.5, 90.0, 1.0)
    dlon = np.arange(-179.5, 180.0, 1.0)
    LON, LAT = np.meshgrid(dlon, dlat)
    z = griddata((lon, lat_deg), arr, (LON, LAT), method="nearest")
    return xr.DataArray(z, coords={"lat": dlat, "lon": dlon}, dims=("lat", "lon"))


def _plot_maps(fields, lat_deg, lon_deg, path, *, doy, grid_type):
    """Filled maps via xarray ``.plot`` (pcolormesh); any grid."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    items = list(fields.items())
    ncols = 3
    nrows = (len(items) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 2.8 * nrows))
    for ax, (name, (arr, cmap, label)) in zip(axes.ravel(), items):
        da = _column_to_dataarray(arr, lat_deg, lon_deg)
        da.plot(ax=ax, cmap=cmap, add_labels=False,
                cbar_kwargs={"shrink": 0.8, "label": ""})
        ax.set_title(label, fontsize=10)
        ax.set_xlim(-180, 180); ax.set_ylim(-90, 90)
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes.ravel()[len(items):]:
        ax.axis("off")
    fig.suptitle(f"run_lmip one step | grid={grid_type} | doy {doy:.0f}", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=95)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
