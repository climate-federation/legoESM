"""Regrid a spun-up land soil state onto another model grid.

WHY.  The published LMIP soil state (PR #1624, Zenodo record 21986853) is ten
years of spun-up soil moisture, temperature and snow on the 2-degree lat-lon
grid.  The coupled AMIP runs on an MPAS mesh, and its land otherwise COLD
STARTS: soil moisture seeded from the atmosphere's own humidity, which is both
a spin-up shock and least realistic exactly where the land matters most.  The
strict restart loader (rightly) refuses a state whose column count is not the
run's own, so the state must be regridded once, offline, and the result is a
normal land IC any run can name with --land-ic-path.

HOW.  Nearest LAND neighbour on the sphere.  Soil state is a per-column
quantity, not a conserved flux, so nearest-neighbour is the right operation;
what matters is never mixing ocean placeholders into land columns.  "Land" on
the source is the spin-up's OWN definition — the surfdata's land+lake+glacier
cover on the source grid, through the same loader the spin-up used — not a
re-derived mask.  Every target column receives its nearest land source column;
target ocean columns carry coastal values the land model never reads (the
driver applies its own mask downstream).

The output carries the soil-column stamp (layer thicknesses) so the loader can
refuse it on any run whose soil column differs — the source file predates the
stamp, and adding it here is what makes the strict check able to work.

KNOWN CARRIED DEFECTS of the published state, documented in its runbook and
deliberately NOT altered here: episodic runoff spikes in ~12% of land cells
(finite values), and an unbounded ice-sheet snowpack.  Altering data in a
regridder would be a hidden choice; if those need fixing it happens upstream,
visibly.

Usage:
    regrid_land_ic.py --source data/lmip_soil_ic/restart_1985_d000h00.npz \
        --surfdata data/legoesm_surfdata_c260716.nc \
        --target-grid mpas --target-resolution 4 --out <path.npz> \
        --source-soil-column 10,3.0,2.0 \
        --source-soil-hydraulics clapp_hornberger surfdata_cosby \
            data/legoesm_surfdata_c260716.nc

    The soil-hydraulics attestation is needed while the source predates the
    stamp.  To stamp an IC already on its grid without regridding it (arrays
    unchanged), add --stamp-only and give only --source, --out and
    --source-soil-hydraulics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess

import numpy as np

_STATE_FIELDS = ("T_soil", "psi_soil", "theta_soil", "runoff_surface",
                 "runoff_subsurface", "snow_depth", "snow_age")
# Per-file fields copied unchanged.  Anything else a source carries (the optional
# water reservoirs, canopy water, TgC, carbon pools, frozen fraction) is
# per-column state this script does not remap, so such a source is refused
# rather than written with another grid's columns attached.
_COPIED_FIELDS = ("restart_version", "land_mode", "t_end_s", "n_steps_completed",
                  "metadata_json", "soil_dz", "soil_z_interface",
                  "soil_hydraulics_json")
# Belongs to the SOURCE grid's columns; a regridded state is converted in full
# on load, so it is dropped rather than attached to the wrong columns.
_DROPPED_FIELDS = ("soil_hydraulics_column_sig",)

# ERA5 (IFS) soil layer interfaces [m]: 0-7, 7-28, 28-100, 100-289 cm (IFS
# documentation Part IV, land surface).  The GRIB stamps layer 4 as 100-255 cm
# (8-bit field limit); the model's base is 2.89 m.  Below it the column takes
# layer 4's value.
_ERA5_SOIL_BOUNDS_M = (0.0, 0.07, 0.28, 1.0, 2.89)
_ERA5_LAND_LSM_MIN = 0.5
_DONOR_FAR_KM = 100.0
# cdo names a GRIB field varNNN after its parameter id: stl1..stl4, lsm.
_ERA5_STL_VARS = ("var139", "var170", "var183", "var236")
_ERA5_LSM_VAR = "var172"
# Refuse a soil temperature outside this range [K]: catches degC or a wrong param.
_ERA5_STL_RANGE_K = (180.0, 340.0)
# ERA5 volumetric soil water swvl1..swvl4 (params 39-42) and snow depth of
# water equivalent sd (param 141, m): the --era5-soil-w / --era5-snow extras.
_ERA5_SWVL_VARS = ("var39", "var40", "var41", "var42")
_ERA5_SD_VAR = "var141"
# swvl is a fraction; small negatives are regridding artefacts (clipped later),
# anything beyond this is a wrong field.  sd is capped by ECMWF at 10 m on ice.
_ERA5_SWVL_RANGE = (-0.05, 1.0)
_ERA5_SD_RANGE_M = (0.0, 10.5)


def overlap_weights(soil_dz, bounds=_ERA5_SOIL_BOUNDS_M) -> np.ndarray:
    """(n_model, n_src) depth-overlap weights, rows summing to 1.

    Each model layer takes the thickness-weighted mean of the source layers it
    overlaps; the part of a model layer below the deepest source interface is
    credited to the deepest source layer.
    """
    z = np.concatenate([[0.0], np.cumsum(np.asarray(soil_dz, np.float64))])
    lo = np.asarray(bounds[:-1], np.float64)
    hi = np.asarray(bounds[1:], np.float64).copy()
    hi[-1] = np.inf
    top, bot = z[:-1, None], z[1:, None]
    w = np.clip(np.minimum(bot, hi) - np.maximum(top, lo), 0.0, None)
    return w / w.sum(axis=1, keepdims=True)


def _era5_land_donors(dst_lat, dst_lon, replace, src_lat, src_lon, src_land):
    """(replaced column indices, their nearest ERA5 LAND point, distance [km]
    for every column, NaN where not replaced)."""
    from legoesm.coupler.grid_remap import nearest_column_map

    from legoesm import constants
    idx = np.flatnonzero(replace)
    near = nearest_column_map(src_lat, src_lon, dst_lat[idx], dst_lon[idx],
                              src_valid=src_land)
    cosd = (np.sin(dst_lat[idx]) * np.sin(src_lat[near])
            + np.cos(dst_lat[idx]) * np.cos(src_lat[near])
            * np.cos(dst_lon[idx] - src_lon[near]))
    dist = np.full(dst_lat.shape[0], np.nan)
    dist[idx] = np.arccos(np.clip(cosd, -1.0, 1.0)) * constants.R_earth / 1e3
    return idx, near, dist


def era5_soil_profile(x_ic, soil_dz, dst_lat, dst_lon, replace,
                      src_lat, src_lon, layers, src_land):
    """Replace ``x_ic`` on ``replace`` columns with the depth-mapped ERA5
    4-layer profile ``layers`` (4, n_src) of the nearest ERA5 LAND point.
    Returns the new (ncol, n_layers) field and the donor distance [km]."""
    idx, near, dist = _era5_land_donors(dst_lat, dst_lon, replace,
                                        src_lat, src_lon, src_land)
    x = np.array(x_ic, dtype=np.float64, copy=True)
    x[idx] = np.asarray(layers, np.float64)[:, near].T @ overlap_weights(soil_dz).T
    return x, dist


def era5_soil_temperature(T_ic, soil_dz, dst_lat, dst_lon, replace,
                          src_lat, src_lon, stl, src_land):
    """Replace ``T_ic`` on ``replace`` columns with depth-mapped ERA5 soil T.

    ``stl`` is (4, n_src) = stl1..stl4 [K] on source points (radians lat/lon);
    each replaced column takes its nearest ERA5 LAND point.  Returns the new
    (ncol, n_layers) temperature and the donor great-circle distance [km] for
    every column (NaN where not replaced).
    """
    return era5_soil_profile(T_ic, soil_dz, dst_lat, dst_lon, replace,
                             src_lat, src_lon, stl, src_land)


def era5_soil_water(theta_ic, psi_ic, soil_dz, dst_lat, dst_lon, replace,
                    src_lat, src_lon, swvl, src_land, hydraulics):
    """Replace soil water on ``replace`` columns with depth-mapped ERA5 swvl1-4.

    The mapped water content is clipped into the Richards solver's own
    admissible range for that (column, layer) -- its dry floor
    ``theta_from_psi(psi_dry_floor)`` and the Cosby ``theta_sat`` -- and the
    matric potential is recomputed from it with the model's retention curve,
    so the pair (theta, psi) is consistent on load.  Returns (theta, psi,
    donor distance [km], number of clipped (column, layer) cells).
    """
    from legoesm.land.richards import psi_dry_floor
    from legoesm.land.soil_hydraulics import psi_from_theta, theta_from_psi
    theta, dist = era5_soil_profile(theta_ic, soil_dz, dst_lat, dst_lon, replace,
                                    src_lat, src_lon, swvl, src_land)
    lo = np.asarray(theta_from_psi(psi_dry_floor(hydraulics), hydraulics), np.float64)
    hi = np.asarray(hydraulics.theta_sat, np.float64)
    idx = np.flatnonzero(replace)
    raw = theta[idx]
    clipped = np.clip(raw, np.broadcast_to(lo, theta.shape)[idx],
                      np.broadcast_to(hi, theta.shape)[idx])
    n_clipped = int(np.sum(clipped != raw))
    theta[idx] = clipped
    psi = np.array(psi_ic, dtype=np.float64, copy=True)
    psi_all = np.asarray(psi_from_theta(theta, hydraulics), np.float64)
    psi[idx] = psi_all[idx]
    return theta, psi, dist, n_clipped


def era5_snow(snow_ic, age_ic, dst_lat, dst_lon, replace,
              src_lat, src_lon, sd_m, src_land):
    """Replace the snow water equivalent [kg/m2] on ``replace`` columns with
    ERA5 ``sd`` [m of water] of the nearest ERA5 LAND point.  A column that
    gains snow it did not have starts with age 0 (fresh); otherwise the source
    age is kept.  Returns (snow_depth, snow_age, donor distance [km])."""
    from legoesm import constants
    idx, near, dist = _era5_land_donors(dst_lat, dst_lon, replace,
                                        src_lat, src_lon, src_land)
    snow = np.array(snow_ic, dtype=np.float64, copy=True)
    age = np.array(age_ic, dtype=np.float64, copy=True)
    new = np.asarray(sd_m, np.float64)[near] * constants.rho_water
    fresh = (new > 0.0) & (snow[idx] <= 0.0)
    snow[idx] = new
    age[idx] = np.where(fresh, 0.0, age[idx])
    return snow, age, dist


def _cols_rad(grid) -> tuple[np.ndarray, np.ndarray]:
    """Per-column (lat, lon) in radians, in the surfdata loader's order.

    Mirrors run_lmip_biophys.grid_latlon_rad — the function that defined the
    source state's column order — so source and target are addressed the same
    way. Duplicated here rather than imported because run scripts are not
    importable modules; a drift test pins the two (test_regrid_land_ic.py).
    """
    if hasattr(grid, "lat2d") and hasattr(grid, "lon2d"):
        lat, lon = np.asarray(grid.lat2d), np.asarray(grid.lon2d)
    elif hasattr(grid, "latCell") and hasattr(grid, "lonCell"):
        lat, lon = np.asarray(grid.latCell), np.asarray(grid.lonCell)
    else:
        lat, lon = np.asarray(grid.lat), np.asarray(grid.lon)
    return lat.ravel().astype(np.float64), lon.ravel().astype(np.float64)


def _ic_grid(grid_type: str, resolution: int):
    """The grid an IC's columns live on (target or source), in that run's order.

    ``fv3_duo`` is the FV3 duo COLUMN lane: its columns are the duo's six-face
    A-grid centres, face-major -- the mesh the driver builds for that lane
    (create_fv3_duo_grid + build_duo_column_mesh) -- not the generic cubed
    sphere, which has the same column count at the same C number (so the
    loader's count check cannot tell them apart) but other centres (up to
    1.6 deg away) in another order (same-index centres median 273 km apart at
    C24, measured 2026-10-10).
    """
    if grid_type == "fv3_duo":
        from legoesm.atmosphere.dynamics.gcm.fv3_duo_column import build_duo_column_mesh
        from legoesm.grids.factory import create_fv3_duo_grid
        bnd = create_fv3_duo_grid(resolution)
        return build_duo_column_mesh(bnd.ctx_np, bnd.n, bnd.ng)
    from legoesm.grids.factory import create_grid
    return create_grid(grid_type, resolution=resolution)


def _source_land_mask(surfdata: str, grid, land_ncol: int) -> np.ndarray:
    """The spin-up's own land definition on its own grid.

    f_land + f_lake + f_glacier > 0 from the harmonized surfdata regridded by
    the SAME loader the spin-up used (run_lmip_biophys.py's no-mask-file
    branch), so the mask cannot disagree with the state it selects from.
    """
    return _land_and_glacier(surfdata, grid, land_ncol)[0]


def _land_and_glacier(surfdata: str, grid, ncol: int):
    """(land, glacier) masks from the harmonized surfdata on ``grid``.

    land = f_land + f_lake + f_glacier > 0 through the spin-up's own loader;
    glacier = the builders' dominant-glacier mask (ice-sheet columns).
    """
    from legoesm.land.boundary_data import init_land_surface_data
    from legoesm.land.boundary_data.builders import glacier_mask
    from legoesm.land.config import MultiLayerLandConfig

    _, _, gsd = init_land_surface_data(
        surfdata, grid, MultiLayerLandConfig(), 0.0)

    def cover1d(a):
        a = np.asarray(a)
        return a[0] if a.ndim == 2 else a

    frac = (cover1d(gsd.f_land) + cover1d(gsd.f_lake)
            + cover1d(gsd.f_glacier)).ravel()
    if frac.size != ncol:
        raise SystemExit(
            f"surfdata mask has {frac.size} columns but the soil state has "
            f"{ncol}: wrong grid.")
    return frac > 0.0, np.asarray(glacier_mask(gsd)).ravel()


def _calibrated_hydraulics(surfdata: str, grid, soil_dz):
    """Per-(column, layer) soil hydraulics exactly as the run builds them: the
    surfdata texture remapped to the CALIBRATED soil grid by the run's loader,
    through the Cosby pedotransfer.  Refuses a grid that differs from the IC's
    stamped column."""
    from legoesm.land.boundary_data import init_land_surface_data
    from legoesm.land.config import MultiLayerLandConfig, biophysics_lmip_two_leaf_setup
    from legoesm.land.soil_grid import make_soil_grid
    cal = biophysics_lmip_two_leaf_setup()["soil_grid"]
    dz = np.asarray(make_soil_grid(cal).dz, np.float64)
    if dz.shape != np.shape(soil_dz) or not np.allclose(dz, soil_dz):
        raise SystemExit("the IC's soil column differs from the calibrated "
                         "column; refusing to derive hydraulics for it.")
    land_cfg, _, _ = init_land_surface_data(
        surfdata, grid, MultiLayerLandConfig(soil_grid=cal), 0.0)
    return land_cfg.hydraulics


def _run_hydraulics_clm(clm_surfdata: str, grid):
    """Per-column van-Genuchten hydraulics exactly as the AMIP run builds them
    from its CLM surfdata (``clm_surfdata_path``): the same loader and the same
    builder, on the target columns."""
    from legoesm.land.clm_surface_map import clm_hydraulics_config, load_clm_surface
    lat, lon = _cols_rad(grid)
    return clm_hydraulics_config(
        load_clm_surface(clm_surfdata, np.degrees(lat), np.degrees(lon)))


def _era5_field(path, var):
    """(values (n,), lat_rad (n,), lon_rad (n,), times) of a one-step ERA5 nc.

    ``var`` is the cdo name the file must carry (varNNN = GRIB param id), so a
    file in the wrong slot is refused instead of permuting the profile.
    """
    import xarray as xr
    ds = xr.open_dataset(path)
    names = [k for k in ds.data_vars if k.startswith("var")]
    if names != [var]:
        raise SystemExit(f"{path}: expected field {var}, got {names}.")
    a = ds[var]
    if a.dims[-2:] != ("lat", "lon"):
        raise SystemExit(f"{path}: expected (..., lat, lon) dims, got {a.dims}.")
    if a.size != ds.sizes["lat"] * ds.sizes["lon"]:
        raise SystemExit(f"{path}: expected ONE time step on one level.")
    lat, lon = np.meshgrid(np.deg2rad(ds["lat"].values),
                           np.deg2rad(ds["lon"].values), indexing="ij")
    times = [str(t) for t in np.atleast_1d(ds["time"].values)] if "time" in ds else []
    return np.asarray(a.values, np.float64).ravel(), lat.ravel(), lon.ravel(), times


def main_era5_soil_t(args) -> int:
    """Replace the soil temperature (and, with --era5-soil-w / --era5-snow,
    the soil water and snow) of an IC already on the target grid.

    Non-glacier land columns take ERA5 stl1-4 from the files' single time step
    (nearest ERA5 land point, lsm >= 0.5; depth-overlap onto the IC's stamped
    soil column).  Glacier columns and every other field stay byte-identical:
    ERA5 stl under an ice sheet is not a soil temperature.
    """
    src = np.load(args.source, allow_pickle=False)
    if "soil_dz" not in src.files:
        raise SystemExit("source IC has no soil_dz stamp; regrid it first.")
    if "soil_hydraulics_json" not in src.files:
        raise SystemExit(
            "source IC has no soil-hydraulics stamp, so the output would be "
            "refused by the model; stamp it first (--stamp-only).")
    ncol = src["T_soil"].shape[0]
    grid = _ic_grid(args.target_grid, args.target_resolution)
    lat, lon = _cols_rad(grid)
    if lat.size != ncol:
        raise SystemExit(f"grid has {lat.size} columns, the IC has {ncol}.")
    land, glacier = _land_and_glacier(args.surfdata, grid, ncol)
    replace = land & ~glacier

    fields = [_era5_field(f, v) for f, v in zip(args.era5_soil_t, _ERA5_STL_VARS)]
    w_fields = ([_era5_field(f, v) for f, v in zip(args.era5_soil_w, _ERA5_SWVL_VARS)]
                if args.era5_soil_w else [])
    s_fields = [_era5_field(args.era5_snow, _ERA5_SD_VAR)] if args.era5_snow else []
    lsm, llat, llon, _ = _era5_field(args.era5_lsm, _ERA5_LSM_VAR)
    glat, glon = fields[0][1], fields[0][2]
    for _, a, o, _ in fields[1:] + w_fields + s_fields + [(None, llat, llon, None)]:
        if not (np.array_equal(a, glat) and np.array_equal(o, glon)):
            raise SystemExit("ERA5 soil layers / land-sea mask are not on one grid.")
    times = {tuple(f[3]) for f in fields + w_fields + s_fields}
    if len(times) != 1:
        raise SystemExit(f"ERA5 soil layers are at different times: {times}.")
    ftime = next(iter(times))
    if (len(ftime) != 1
            or np.datetime64(ftime[0]) != np.datetime64(args.era5_time)):
        raise SystemExit(f"ERA5 soil layers are at {list(ftime)}, not the "
                         f"requested --era5-time {args.era5_time}.")
    stl = np.stack([f[0] for f in fields])
    if not np.all(np.isfinite(stl)):
        raise SystemExit("ERA5 soil temperature has non-finite values; refusing.")
    lo, hi = _ERA5_STL_RANGE_K
    if stl.min() < lo or stl.max() > hi:
        raise SystemExit(f"ERA5 soil temperature spans {stl.min():.1f}-"
                         f"{stl.max():.1f} K, outside {lo}-{hi} K; refusing.")
    if not replace.any():
        raise SystemExit("no non-glacier land column to replace; wrong surfdata?")

    T, dist = era5_soil_temperature(src["T_soil"], src["soil_dz"], lat, lon,
                                    replace, glat, glon, stl,
                                    lsm >= _ERA5_LAND_LSM_MIN)
    d = dist[replace]
    far = int(np.sum(d > _DONOR_FAR_KM))
    print(f"replaced {int(replace.sum())} land columns, kept "
          f"{int((land & glacier).sum())} glacier columns; donor distance max "
          f"{d.max():.1f} km, p99 {np.percentile(d, 99):.1f} km, "
          f"{far} columns > {_DONOR_FAR_KM:.0f} km")

    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, check=True).stdout.strip()
    except Exception:
        sha = "unknown"
    md5 = lambda p: hashlib.md5(pathlib.Path(p).read_bytes()).hexdigest()
    meta = json.loads(str(src["metadata_json"]))
    meta.update({
        "soil_t_source": "ERA5 stl1-4 (GRIB params 139,170,183,236)",
        "soil_t_time": list(ftime),
        "soil_t_files": {v: {"path": str(f), "md5": md5(f)}
                         for f, v in zip(args.era5_soil_t, _ERA5_STL_VARS)},
        "soil_t_lsm": {_ERA5_LSM_VAR: {"path": str(args.era5_lsm),
                                       "md5": md5(args.era5_lsm)}},
        "soil_t_surfdata": {"path": str(args.surfdata), "md5": md5(args.surfdata)},
        "soil_t_target_grid": f"{args.target_grid} {args.target_resolution}",
        "soil_t_from_ic": pathlib.Path(args.source).name,
        "soil_t_from_ic_md5": md5(args.source),
        "soil_t_choices": (
            "T_soil only (moisture/snow/runoff from the source IC); nearest ERA5 "
            "land point lsm>=0.5; depth-overlap on 0,.07,.28,1.0,2.89 m, layer 4 "
            "below; stl1 for the top layer (not skt); glacier columns keep the "
            "source IC"),
        "soil_t_donor_km_max": float(d.max()),
        "soil_t_donor_km_p99": float(np.percentile(d, 99)),
        "soil_t_donor_n_over_100km": far,
        "soil_t_git_sha": sha,
        "soil_t_script": "scripts/data/regrid_land_ic.py --era5-soil-t",
    })
    out = {k: src[k] for k in src.files}
    out["T_soil"] = T.astype(src["T_soil"].dtype)
    src_land = lsm >= _ERA5_LAND_LSM_MIN
    if w_fields:
        swvl = np.stack([f[0] for f in w_fields])
        if not np.all(np.isfinite(swvl)):
            raise SystemExit("ERA5 soil water has non-finite values; refusing.")
        lo, hi = _ERA5_SWVL_RANGE
        if swvl.min() < lo or swvl.max() > hi:
            raise SystemExit(f"ERA5 soil water spans {swvl.min():.3f}-"
                             f"{swvl.max():.3f}, outside {lo}-{hi}; refusing.")
        stamp = json.loads(str(src["soil_hydraulics_json"]))
        if args.soil_w_hydraulics == "clm-map":
            # The run converts the file's water onto THESE hydraulics at load
            # (restart.convert_ic_soil_water): clip into their range so no
            # column is refused as drier than the run's floor.
            hyd = _run_hydraulics_clm(args.clm_surfdata, grid)
            hyd_desc = f"clm-map van Genuchten from {pathlib.Path(args.clm_surfdata).name}"
        else:
            if (stamp.get("retention_curve") != "clapp_hornberger"
                    or stamp.get("parameter_source") != "surfdata_cosby"):
                raise SystemExit("the IC's hydraulics stamp is not the surfdata "
                                 f"Cosby / Clapp-Hornberger pair ({stamp}); the "
                                 "rebuilt matric potential would not match it.")
            hyd = _calibrated_hydraulics(args.surfdata, grid, src["soil_dz"])
            hyd_desc = "surfdata Cosby Clapp-Hornberger (the IC's stamp)"
        theta, psi, dist_w, n_clip = era5_soil_water(
            src["theta_soil"], src["psi_soil"], src["soil_dz"], lat, lon,
            replace, glat, glon, swvl, src_land, hyd)
        if not (np.all(np.isfinite(theta)) and np.all(np.isfinite(psi))):
            raise SystemExit("soil water / matric potential non-finite after "
                             "the ERA5 replacement; refusing.")
        out["theta_soil"] = theta.astype(src["theta_soil"].dtype)
        out["psi_soil"] = psi.astype(src["psi_soil"].dtype)
        print(f"soil water: replaced {int(replace.sum())} columns from ERA5 "
              f"swvl1-4; {n_clip} (column, layer) cells clipped into the "
              "solver's [dry floor, theta_sat] range")
        meta.update({
            "soil_w_source": "ERA5 swvl1-4 (GRIB params 39,40,41,42)",
            "soil_w_files": {v: {"path": str(f), "md5": md5(f)}
                             for f, v in zip(args.era5_soil_w, _ERA5_SWVL_VARS)},
            "soil_w_hydraulics": hyd_desc,
            "soil_w_choices": (
                "same donors and depth-overlap as T_soil; theta clipped to the "
                "Richards dry floor / theta_sat of the chosen hydraulics "
                f"({hyd_desc}); psi = psi_from_theta of the clipped theta on "
                "those hydraulics; glacier columns keep the source IC"),
            "soil_w_n_clipped_cells": n_clip,
        })
    if s_fields:
        sd = s_fields[0][0]
        if not np.all(np.isfinite(sd)):
            raise SystemExit("ERA5 snow depth has non-finite values; refusing.")
        lo, hi = _ERA5_SD_RANGE_M
        if sd.min() < lo or sd.max() > hi:
            raise SystemExit(f"ERA5 snow depth spans {sd.min():.2f}-{sd.max():.2f} m, "
                             f"outside {lo}-{hi} m; refusing.")
        snow, age, _ = era5_snow(src["snow_depth"], src["snow_age"], lat, lon,
                                 replace, glat, glon, sd, src_land)
        out["snow_depth"] = snow.astype(src["snow_depth"].dtype)
        out["snow_age"] = age.astype(src["snow_age"].dtype)
        print(f"snow: replaced {int(replace.sum())} columns from ERA5 sd; "
              f"{int(np.sum(snow[replace] > 0))} of them carry snow")
        meta.update({
            "snow_source": "ERA5 sd (GRIB param 141) x rho_water",
            "snow_files": {_ERA5_SD_VAR: {"path": str(args.era5_snow),
                                          "md5": md5(args.era5_snow)}},
            "snow_choices": (
                "nearest ERA5 land point; snow_age 0 where the column gains "
                "snow it did not have, else the source age; glacier columns "
                "keep the source IC"),
        })
    out["metadata_json"] = np.str_(json.dumps(meta))
    np.savez_compressed(args.out, **out)
    print(f"wrote {args.out}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="spun-up land restart .npz")
    ap.add_argument("--surfdata", default=None,
                    help="harmonized surfdata that defines the source land mask")
    ap.add_argument("--target-grid", default=None,
                    help="target grid type (e.g. mpas, latlon; fv3_duo = the "
                         "FV3 duo column lane's own mesh, not cubed_sphere)")
    ap.add_argument("--target-resolution", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--era5-soil-t", nargs=4, metavar="STL",
                    help="ERA5 stl1..stl4 netCDFs (one time step, regular "
                         "grid, e.g. cdo -f nc -setgridtype,regular "
                         "-seltimestep,1). Switches to SOIL-T mode: --source is "
                         "an IC already on the target grid and ONLY its soil "
                         "temperature is replaced on non-glacier land columns.")
    ap.add_argument("--era5-soil-w", nargs=4, metavar="SWVL",
                    help="with --era5-soil-t: ERA5 swvl1..swvl4 netCDFs (same "
                         "grid and instant). Soil water of the replaced columns "
                         "is taken from ERA5 too (depth-overlap, clipped to the "
                         "solver's admissible range, matric potential rebuilt "
                         "with the model's retention curve).")
    ap.add_argument("--soil-w-hydraulics", choices=("surfdata-cosby", "clm-map"),
                    help="with --era5-soil-w (required): whose dry floor / "
                         "saturation the ERA5 soil water is clipped into. "
                         "'clm-map' = the hydraulics the AMIP run builds from "
                         "its CLM surfdata (van Genuchten, needs --clm-surfdata); "
                         "'surfdata-cosby' = the offline land spin-up's Cosby "
                         "hydraulics from --surfdata. A state below the RUN's "
                         "floor is refused at load, so pick the run's.")
    ap.add_argument("--clm-surfdata", help="the run's clm_surfdata_path "
                                           "(for --soil-w-hydraulics clm-map)")
    ap.add_argument("--era5-snow", metavar="SD",
                    help="with --era5-soil-t: ERA5 sd netCDF (snow depth of "
                         "water equivalent [m], same grid and instant). Snow of "
                         "the replaced columns is taken from ERA5 (x rho_water).")
    ap.add_argument("--era5-time", help="the instant the --era5-soil-t files "
                                       "must hold (e.g. 1979-01-01T00:00); "
                                       "required with --era5-soil-t")
    ap.add_argument("--era5-lsm", help="ERA5 land-sea mask netCDF (param 172), "
                                       "same grid; required with --era5-soil-t")
    ap.add_argument("--source-soil-column", default=None,
                    help="attest the SOURCE state's soil column as "
                         "'n_layers,depth_m,growth' (e.g. '10,3.0,2.0'). The "
                         "source file predates the soil-column stamp, so its "
                         "column cannot be read off the file; the operator "
                         "must state it, it is checked against the calibrated "
                         "column, and it is recorded in provenance. Without "
                         "this a source on a same-layer-count but different-"
                         "depth column would be silently certified as "
                         "calibrated (codex).")
    ap.add_argument("--source-soil-hydraulics", nargs=3, default=None,
                    metavar=("CURVE", "SOURCE", "PARAMETER_FILE"),
                    help="attest the soil hydraulics an UNSTAMPED source was "
                         "evolved under, e.g. clapp_hornberger surfdata_cosby "
                         "data/legoesm_surfdata_c260716.nc. Refused when the "
                         "source already carries a stamp. Recorded as attested, "
                         "with the source's md5.")
    ap.add_argument("--stamp-only", action="store_true",
                    help="do not regrid: write --source unchanged plus the "
                         "attested soil-hydraulics stamp to --out (for an IC "
                         "already on its target grid).")
    args = ap.parse_args(argv)
    if args.era5_soil_t:
        missing = [f for f in ("era5_lsm", "era5_time", "surfdata", "target_grid",
                               "target_resolution") if getattr(args, f) is None]
        if missing:
            ap.error("--era5-soil-t needs " + ", ".join(
                "--" + m.replace("_", "-") for m in missing))
        if args.stamp_only or args.source_soil_hydraulics:
            ap.error("--era5-soil-t takes a stamped --source; --stamp-only / "
                     "--source-soil-hydraulics apply to regridding or stamping")
        if args.era5_soil_w and args.soil_w_hydraulics is None:
            ap.error("--era5-soil-w needs --soil-w-hydraulics")
        if args.soil_w_hydraulics == "clm-map" and not args.clm_surfdata:
            ap.error("--soil-w-hydraulics clm-map needs --clm-surfdata")
        return main_era5_soil_t(args)
    if args.era5_soil_w or args.era5_snow:
        ap.error("--era5-soil-w / --era5-snow extend --era5-soil-t")
    if not args.stamp_only:
        missing = [f for f in ("surfdata", "target_grid", "target_resolution",
                               "source_soil_column")
                   if getattr(args, f) is None]
        if missing:
            ap.error("regridding needs " + ", ".join(
                "--" + m.replace("_", "-") for m in missing))

    src = np.load(args.source, allow_pickle=False)
    extra = sorted(set(src.files) - set(_STATE_FIELDS) - set(_COPIED_FIELDS)
                   - set(_DROPPED_FIELDS))
    if extra and not args.stamp_only:
        raise SystemExit(
            f"source carries per-column fields this script does not remap: "
            f"{extra}. Refusing rather than attaching them to the wrong columns.")
    from legoesm.land.restart import file_md5, soil_hydraulics_stamp
    hyd_json = None
    if "soil_hydraulics_json" in src.files:
        if args.source_soil_hydraulics:
            raise SystemExit(
                "the source already records its soil hydraulics; "
                "--source-soil-hydraulics would overrule the file.")
    else:
        if not args.source_soil_hydraulics:
            raise SystemExit(
                "the source records no soil hydraulics; state them with "
                "--source-soil-hydraulics CURVE SOURCE PARAMETER_FILE.")
        curve, source, pfile = args.source_soil_hydraulics
        stamp = soil_hydraulics_stamp(curve, source, pfile)
        stamp.update(attested=True, attested_source_md5=file_md5(args.source))
        hyd_json = np.array(json.dumps(stamp))
    if args.stamp_only:
        if hyd_json is None:
            raise SystemExit("--stamp-only: the source is already stamped.")
        out = {k: src[k] for k in src.files}
        out["soil_hydraulics_json"] = hyd_json
        np.savez_compressed(args.out, **out)
        print(f"wrote {args.out}: {args.source} unchanged plus its "
              "soil-hydraulics stamp")
        return 0
    meta = json.loads(str(src["metadata_json"]))
    ncol_src = src["T_soil"].shape[0]

    src_grid = _ic_grid(meta["grid_type"], int(meta["resolution"]))
    src_lat, src_lon = _cols_rad(src_grid)
    if src_lat.size != ncol_src:
        raise SystemExit(
            f"reconstructed source grid has {src_lat.size} columns, the state "
            f"has {ncol_src}: the metadata does not describe this file.")

    dst_grid = _ic_grid(args.target_grid, args.target_resolution)
    dst_lat, dst_lon = _cols_rad(dst_grid)

    from legoesm.coupler.grid_remap import nearest_column_map
    land = _source_land_mask(args.surfdata, src_grid, ncol_src)
    n_land = int(land.sum())
    nearest = nearest_column_map(src_lat, src_lon, dst_lat, dst_lon,
                                 src_valid=land)

    out = {k: src[k] for k in src.files
           if k not in _STATE_FIELDS and k not in _DROPPED_FIELDS}
    if hyd_json is not None:
        out["soil_hydraulics_json"] = hyd_json
    for k in _STATE_FIELDS:
        out[k] = np.asarray(src[k])[nearest]

    # Refuse to write garbage rather than stamp it as an IC.
    for k in ("T_soil", "psi_soil", "theta_soil"):
        if not np.all(np.isfinite(out[k])):
            raise SystemExit(f"{k}: non-finite values after regrid; refusing.")

    # The soil-column stamp: her template's own column, computed by the model's
    # own grid builder, never typed in by hand.
    from legoesm.land.config import biophysics_lmip_two_leaf_setup
    from legoesm.land.soil_grid import SoilGridConfig, make_soil_grid
    cal_grid = biophysics_lmip_two_leaf_setup()["soil_grid"]
    n_str, d_str, g_str = args.source_soil_column.split(",")
    attested = SoilGridConfig(n_layers=int(n_str), total_depth=float(d_str),
                              growth_factor=float(g_str))
    if (attested.n_layers != cal_grid.n_layers
            or attested.total_depth != cal_grid.total_depth
            or attested.growth_factor != cal_grid.growth_factor):
        raise SystemExit(
            f"attested source column {args.source_soil_column} is not the "
            f"calibrated column ({cal_grid.n_layers},{cal_grid.total_depth},"
            f"{cal_grid.growth_factor}): refusing to stamp a state as a "
            "column it is not on.")
    if src["T_soil"].shape[1] != attested.n_layers:
        raise SystemExit(
            f"source has {src['T_soil'].shape[1]} layers but the attestation "
            f"says {attested.n_layers}.")
    out["soil_dz"] = np.asarray(make_soil_grid(cal_grid).dz, dtype=np.float64)

    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, check=True).stdout.strip()
    except Exception:
        sha = "unknown"
    meta.update({
        "grid_type": args.target_grid,
        "resolution": args.target_resolution,
        "regridded_from": pathlib.Path(args.source).name,
        "regridded_from_md5": hashlib.md5(
            pathlib.Path(args.source).read_bytes()).hexdigest(),
        "regrid_script": "scripts/data/regrid_land_ic.py",
        "regrid_git_sha": sha,
        "regrid_n_source_land": n_land,
        "source_soil_column_attested": args.source_soil_column,
    })
    out["metadata_json"] = np.str_(json.dumps(meta))

    np.savez_compressed(args.out, **out)
    print(f"wrote {args.out}: {dst_lat.size} columns from {n_land} source land "
          f"columns ({meta['grid_type']} {meta['resolution']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
