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
from legoesm.land.config import MultiLayerLandConfig
from legoesm.land.soil_grid import SoilGridConfig
from legoesm.land.canopy import CanopyConfig
from legoesm.land.multilayer_land import step_multilayer_land, init_multilayer_land_state
from legoesm.land.global_surface_data import get_surfdata_preset, load_global_surface_data
from legoesm.land.surface_data.land_inputs import (
    build_canopy_params, dominant_pft_index, glacier_mask)
from legoesm.land.pedotransfer import soil_hydraulics_config_from_texture

U_MIN = 1.0


class _LatLonGrid:
    """Minimal regular lat-lon grid for the surfdata loader (radians)."""

    def __init__(self, nlat: int, nlon: int):
        self.lat_deg = np.linspace(-89.0, 89.0, nlat)
        self.lon_deg = np.linspace(0.0, 360.0, nlon, endpoint=False)
        lon2d, lat2d = np.meshgrid(np.deg2rad(self.lon_deg), np.deg2rad(self.lat_deg))
        self.lat2d, self.lon2d = lat2d, lon2d
        self.nlat, self.nlon, self.ncol = nlat, nlon, nlat * nlon
        # cos-lat cell-area proxy (only used as a fallback; absolute scale irrelevant here)
        area = np.cos(lat2d).clip(min=0.0) * (constants.R_earth ** 2)
        self.grid_area = jnp.asarray(area.ravel(), dtype=jnp.float32)


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
    ap.add_argument("--nlat", type=int, default=48)
    ap.add_argument("--nlon", type=int, default=96)
    ap.add_argument("--doy", type=float, default=196.0, help="day-of-year (LAI + solar)")
    ap.add_argument("--hour", type=float, default=12.0, help="UTC hour")
    ap.add_argument("--dt", type=float, default=1800.0)
    ap.add_argument("--output", default="lmip_global")
    ap.add_argument("--land-frac-min", type=float, default=0.5,
                    help="cells with f_land below this are masked as ocean in output")
    ap.add_argument("--no-plot", dest="plot", action="store_false",
                    help="skip the maps PNG")
    args = ap.parse_args()

    out_dir = Path(args.output); out_dir.mkdir(parents=True, exist_ok=True)
    grid = _LatLonGrid(args.nlat, args.nlon)
    print(f"grid {grid.nlat}x{grid.nlon} = {grid.ncol} columns; loading {args.surfdata}")

    # --- surface data on the model grid ---
    cfg_sd = get_surfdata_preset("legoesm_surfdata")._replace(surf_path=args.surfdata)
    gsd = load_global_surface_data(cfg_sd, grid)

    # --- config: Cosby soil hydraulics + two-leaf canopy scheme ---
    # v1 uses a global-mean (uniform) texture: solve_richards mixes column-level
    # (ncol,) and profile (ncol,nlayer) ops, so per-column (ncol,1) hydraulic
    # params don't broadcast cleanly yet.  Spatially-varying soil (the tested
    # land_inputs.build_soil_hydraulics) awaits a richards shape-hardening pass.
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_sand = float(np.nanmean(np.asarray(gsd.sand_frac) * 100.0))
        mean_clay = float(np.nanmean(np.asarray(gsd.clay_frac) * 100.0))
    soil_hyd = soil_hydraulics_config_from_texture(mean_sand, mean_clay)
    print(f"soil (global-mean texture): sand={mean_sand:.0f}% clay={mean_clay:.0f}%")
    config = MultiLayerLandConfig(
        surface_scheme=CanopyConfig(max_iters=50, tol=1e-2),
        soil_grid=SoilGridConfig(n_layers=gsd.sand_frac.shape[1]),
        hydraulics=soil_hyd,
    )

    # --- forcing + state (soil initialised near the local air temperature) ---
    lat_rad = jnp.asarray(grid.lat2d.ravel())
    lon_rad = jnp.asarray(grid.lon2d.ravel())
    forcing = make_global_forcing(lat_rad, lon_rad, args.doy, args.hour)
    state = init_multilayer_land_state(grid.ncol, config, T_init=288.0)
    state = state._replace(
        T_soil=jnp.broadcast_to(forcing.T_lowest[:, None], state.T_soil.shape)
    )

    # --- canopy params (LAI/height/albedo/PFT) at this day-of-year ---
    theta_top = state.theta_soil[:, 0]
    canopy = build_canopy_params(gsd, args.doy, theta_top)

    print("stepping one timestep ...")
    step = jax.jit(lambda s, f: step_multilayer_land(
        s, f, config, U_MIN, args.dt, lat=lat_rad, land_params=canopy, doy=args.doy))
    new_state, resp, _ = step(state, forcing)

    # --- land mask, reshape, validate, write ---
    # The land-SEA mask is the total land fraction = soil/veg + lake + glacier
    # (so ice sheets and lake-rich regions count as land).  gsd.f_land here is the
    # CLM natveg+crop (soil/veg) fraction only — NOT the land mask.
    def cover1d(a):
        a = np.asarray(a)
        return (a[0] if a.ndim == 2 else a)
    f_soil_veg = cover1d(gsd.f_land)
    land_frac_flat = f_soil_veg + cover1d(gsd.f_lake) + cover1d(gsd.f_glacier)
    land_fraction = land_frac_flat.reshape(grid.nlat, grid.nlon)
    land = land_fraction >= args.land_frac_min

    import warnings
    def grid2d(a):
        return np.where(land, np.asarray(a, dtype=np.float64).reshape(grid.nlat, grid.nlon), np.nan)
    def layer_mean(a):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN soil cols
            return np.nanmean(np.asarray(a), axis=1)

    # Surfdata-derived spatial inputs (these vary per column) + model outputs.
    dom = dominant_pft_index(gsd)
    fields = {
        # --- surface inputs from the surfdata pipeline ---
        "sand_pct":     (grid2d(layer_mean(gsd.sand_frac) * 100.0), "YlOrBr", "sand %"),
        "clay_pct":     (grid2d(layer_mean(gsd.clay_frac) * 100.0), "BuPu", "clay %"),
        "organic":      (grid2d(layer_mean(gsd.organic)), "YlGn", "organic"),
        "bulk_density": (grid2d(layer_mean(gsd.bulk_density)), "cividis", "bulk density kg/m3"),
        "soil_color":   (grid2d(gsd.soil_color), "viridis", "soil colour class"),
        "LAI":          (grid2d(canopy.LAI), "YlGn", "LAI (dominant PFT)"),
        "dominant_pft": (grid2d(dom), "tab20", "dominant CLM5 PFT index"),
        "albedo_vis":   (grid2d(canopy.ALB_VIS), "Greys_r", "soil background albedo (VIS)"),
        "land_fraction": (land_fraction, "Blues", "land fraction (veg+lake+glacier)"),
        "f_soil_veg":   (grid2d(f_soil_veg), "YlGn", "soil/veg fraction (natveg+crop)"),
        "glacier":      (grid2d(glacier_mask(gsd).astype(float)), "cool", "glacier (ice) mask"),
        # --- model outputs (one step) ---
        "albedo_out":   (grid2d(resp.albedo), "Greys_r", "surface albedo (model)"),
        "T_sfc":        (grid2d(resp.T_surface), "magma", "surface T [K]"),
        "shflx":        (grid2d(resp.shflx), "RdBu_r", "sensible heat [W/m2]"),
        "lhflx":        (grid2d(resp.lhflx), "viridis", "latent heat [W/m2]"),
    }

    n_land = int(land.sum())
    nan_land = int(np.isnan(fields["T_sfc"][0][land]).sum())
    status = "PASS" if nan_land == 0 else "FAIL"
    print(f"land cells: {n_land} | NaN T_sfc over land: {nan_land} -> {status}")
    rng = lambda a: f"[{np.nanmin(a):.2f}, {np.nanmax(a):.2f}]"
    for k in ("sand_pct", "clay_pct", "bulk_density", "LAI", "T_sfc"):
        print(f"  {k:12s} {rng(fields[k][0])}")

    try:
        import xarray as xr
        ds = xr.Dataset(
            {k: (("lat", "lon"), v[0]) for k, v in fields.items()},
            coords={"lat": grid.lat_deg, "lon": grid.lon_deg},
            attrs={"doy": args.doy, "hour": args.hour, "surfdata": args.surfdata},
        )
        nc = out_dir / "lmip_global_step.nc"
        ds.to_netcdf(nc)
        print(f"wrote {nc}")
    except Exception as e:  # noqa: BLE001
        print(f"(netcdf write skipped: {e})")

    if args.plot:
        _plot_maps(fields, grid.lat_deg, grid.lon_deg, out_dir / "lmip_global_maps.png",
                   doy=args.doy)

    if status == "FAIL":
        sys.exit(1)


def _plot_maps(fields, lat, lon, path, *, doy):
    """Multi-panel global maps of the surfdata inputs + one-step outputs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    items = list(fields.items())
    ncols = 3
    nrows = (len(items) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 2.8 * nrows))
    for ax, (name, (arr, cmap, label)) in zip(axes.ravel(), items):
        im = ax.pcolormesh(lon, lat, arr, shading="auto", cmap=cmap)
        ax.set_title(label, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.8)
    for ax in axes.ravel()[len(items):]:
        ax.axis("off")
    fig.suptitle(f"run_lmip global one step (day-of-year {doy:.0f})", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=95)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
