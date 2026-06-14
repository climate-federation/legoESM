#!/usr/bin/env python
"""Global single-timestep land driver (LMIP).

Runs **one timestep of the global multilayer-land + two-leaf-canopy model**,
driven by the harmonized surface dataset (``legoesm_surfdata`` — CLM5 cover/PFT/
LAI + HWSD soil; see ``legoesm.land.surface_data``).  This is the gridded
counterpart to the single-point FLUXNET driver (``run_fluxnet``, separate).

Surface data wiring (all from the surfdata, per column):
  - soil hydraulics  <- Cosby pedotransfer on HWSD sand/clay (per-column
    Clapp-Hornberger; sandy fallback where HWSD has no soil);
  - PFT photosynthesis/aero params <- dominant CLM5 PFT (canopy biome tables);
  - LAI + canopy height <- surfdata monthly climatology at the run day-of-year;
  - background albedo (ALB_VIS/NIR) <- CLM soil-colour class + top-layer wetness.

Atmospheric forcing is idealised (solar geometry + latitudinal/seasonal/diurnal
temperature) — no ERA5 needed for a single diagnostic step.

Usage::

    JAX_ENABLE_X64=1 python scripts/run_lmip.py \\
        --surfdata data/legoesm_surfdata_v1.nc --nlat 48 --nlon 96 \\
        --doy 196 --hour 12 --output lmip_global
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from legoesm import constants
from legoesm.thermo import saturation_mixing_ratio
from legoesm.coupler.coupling_fields import AtmToSurface
from legoesm.land.config import MultiLayerLandConfig, LandConfig
from legoesm.land.soil_grid import SoilGridConfig
from legoesm.land.canopy import CanopyConfig
from legoesm.land.surface_scheme import SimpleSEBConfig
from legoesm.land.multilayer_land import step_multilayer_land, init_multilayer_land_state
from legoesm.land.slab_land import step_land
from legoesm.land.global_surface_data import interp_monthly
from legoesm.land.soil_albedo import soil_albedo_broadband
from legoesm.land.surface_data.land_inputs import (
    dominant_pft_index, glacier_mask, surface_data_to_land_params,
    init_land_surface_data, fill_land_param_gaps)

U_MIN = 1.0


def make_grid(grid_type: str, n_lat: int, n_lon: int, resolution: int):
    """Build a model grid with the SAME factories ModelDriver uses."""
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
        return create_latlon_grid(n_lat, n_lon)
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
    ap.add_argument("--land-scheme", default="multilayer-canopy",
                    choices=["multilayer-canopy", "multilayer-seb", "slab"],
                    help="land surface scheme to drive with the surfdata")
    ap.add_argument("--grid-type", default="latlon",
                    choices=["latlon", "gaussian", "cubed_sphere"],
                    help="model grid (same factories as ModelDriver)")
    ap.add_argument("--nlat", type=int, default=48, help="latlon: latitude points")
    ap.add_argument("--nlon", type=int, default=96, help="latlon: longitude points")
    ap.add_argument("--resolution", type=int, default=48,
                    help="cubed_sphere: cells/face (n); gaussian: spectral truncation (n_max)")
    ap.add_argument("--doy", type=float, default=196.0, help="day-of-year (LAI + solar)")
    ap.add_argument("--hour", type=float, default=12.0, help="UTC hour")
    ap.add_argument("--dt", type=float, default=1800.0)
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
    grid = make_grid(args.grid_type, args.nlat, args.nlon, args.resolution)
    lat_rad, lon_rad = grid_latlon_rad(grid)
    ncol = lat_rad.shape[0]
    print(f"grid={args.grid_type} | {ncol} columns | scheme={args.land_scheme}")
    forcing = make_global_forcing(lat_rad, lon_rad, args.doy, args.hour)

    # --- base config + step function for the chosen scheme ---
    if args.land_scheme == "slab":
        base_cfg = LandConfig()
        step_fn = step_land
    else:
        surf = (CanopyConfig(max_iters=50, tol=1e-2)
                if args.land_scheme == "multilayer-canopy" else SimpleSEBConfig())
        base_cfg = MultiLayerLandConfig(surface_scheme=surf, soil_grid=SoilGridConfig())
        step_fn = step_multilayer_land

    # --- run the surface-data loader at simulation start: regrid to this grid,
    #     derive scheme-appropriate config (soil hydraulics) + land params. ---
    config, _params_nominal, gsd = init_land_surface_data(
        args.surfdata, grid, base_cfg, args.doy)

    # --- state (soil/skin T initialised near the local air temperature) ---
    if args.land_scheme == "slab":
        from legoesm.core.field import Field
        from legoesm.land.state import LandState
        z = lambda: jnp.zeros(ncol)
        state = LandState(
            T_soil=Field(forcing.T_lowest, name="T_soil", units="K"),
            W_bucket=Field(jnp.full(ncol, 100.0), name="W_bucket", units="kg/m2"),
            snow_depth=Field(z(), name="snow_depth", units="kg/m2"),
            snow_age=Field(z(), name="snow_age", units="s"),
        )
        theta_top = jnp.full(ncol, 0.2)               # slab has no soil profile
    else:
        state = init_multilayer_land_state(ncol, config, T_init=288.0)
        state = state._replace(
            T_soil=jnp.broadcast_to(forcing.T_lowest[:, None], state.T_soil.shape))
        theta_top = state.theta_soil[:, 0]

    # land params with the actual top-layer wetness (accurate soil-colour albedo),
    # then reconcile with the authoritative land mask: any cell the mask calls land
    # but the surfdata doesn't cover falls back to bare soil (finite everywhere).
    land_params = surface_data_to_land_params(gsd, config.surface_scheme, args.doy, theta_top)
    land_params = fill_land_param_gaps(land_params, gsd)

    print("stepping one timestep ...")
    step = jax.jit(lambda s, f: step_fn(
        s, f, config, U_MIN, args.dt, lat=lat_rad, land_params=land_params, doy=args.doy))
    new_state, resp, _ = step(state, forcing)

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
        "T_sfc":        (col(resp.T_surface), "magma", "surface T [K]"),
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
                   "grid_type": args.grid_type, "land_scheme": args.land_scheme},
        )
        nc = out_dir / "lmip_global_step.nc"
        ds.to_netcdf(nc)
        print(f"wrote {nc}")
    except Exception as e:  # noqa: BLE001
        print(f"(netcdf write skipped: {e})")

    if args.plot:
        _plot_maps(fields, lat_deg, lon_deg, out_dir / "lmip_global_maps.png",
                   doy=args.doy, grid_type=args.grid_type)

    if status == "FAIL":
        sys.exit(1)


def _plot_maps(fields, lat_deg, lon_deg, path, *, doy, grid_type):
    """Per-column scatter maps (works on any grid: lat-lon, gaussian, cubed-sphere)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lon = np.where(lon_deg > 180.0, lon_deg - 360.0, lon_deg)   # -> [-180,180) for display
    items = list(fields.items())
    ncols = 3
    nrows = (len(items) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 2.8 * nrows))
    for ax, (name, (arr, cmap, label)) in zip(axes.ravel(), items):
        im = ax.scatter(lon, lat_deg, c=np.asarray(arr), s=4, cmap=cmap, marker="s")
        ax.set_title(label, fontsize=10)
        ax.set_xlim(-180, 180); ax.set_ylim(-90, 90)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.8)
    for ax in axes.ravel()[len(items):]:
        ax.axis("off")
    fig.suptitle(f"run_lmip one step | grid={grid_type} | doy {doy:.0f}", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=95)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
