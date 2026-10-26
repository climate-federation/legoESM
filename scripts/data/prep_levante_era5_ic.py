#!/usr/bin/env python
"""Prepare ERA5 initial-condition Zarr store from Levante pool data.

Reads ERA5 GRIB files from /pool/data/ERA5/E5/ (Levante), regrids to a
regular 1-degree lat-lon grid via CDO, converts units, and writes a
single-snapshot Zarr store compatible with ``load_era5_ic`` in
``legoesm.training.era5_to_state``.

Required variables (CDO ECMWF parameter codes):
  Pressure levels (pl/an/1D/):
    130 temperature     → temperature       [K]
    131 u-wind          → u_component_of_wind [m/s]
    132 v-wind          → v_component_of_wind [m/s]
    133 specific hum.   → specific_humidity  [kg/kg]
  Surface (sf/an/1D/):
    134 surface press.  → surface_pressure   [Pa]
    235 skin temp.      → skin_temperature   [K]
  Surface invariant (sf/an/IV/):
    129 geopotential    → geopotential_at_surface [m²/s²]

Usage::

    python scripts/prep_levante_era5_ic.py \
        --year 1979 --month 1 --day 1 \
        --out /scratch/b/b309178/era5_ic_1979-01-01.zarr
"""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import xarray as xr

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

ERA5_ROOT = Path("/pool/data/ERA5/E5")
CDO = "/sw/spack-levante/cdo-2.2.2-4z4icb/bin/cdo"

# ECMWF parameter code → (subdir, WB2-compatible variable name)
_PL_VARS = {
    130: "temperature",
    131: "u_component_of_wind",
    132: "v_component_of_wind",
    133: "specific_humidity",
}
_SF_VARS = {
    134: "surface_pressure",
    235: "skin_temperature",
}
_INV_VAR = 129  # surface geopotential (invariant)


def _grib_path(code: int, year: int, month: int, kind: str = "pl") -> Path:
    """Return path to a monthly ERA5 GRIB file.

    kind : "pl" = pressure level, "sf" = surface, "inv" = invariant surface
    """
    ym = f"{year:04d}-{month:02d}"
    if kind == "inv":
        return ERA5_ROOT / "sf" / "an" / "IV" / str(code) / f"E5sf00_IV_INVARIANT_{code}.grb"
    subdir = "pl" if kind == "pl" else "sf"
    prefix = "E5pl00" if kind == "pl" else "E5sf00"
    return ERA5_ROOT / subdir / "an" / "1D" / str(code) / f"{prefix}_1D_{ym}_{code}.grb"


def _cdo_to_nc4(src: Path, dst: Path, day_index: int = 1) -> None:
    """Run CDO to extract one time step and regrid to 1° regular lat-lon.

    Uses r360x180 (180 lat × 360 lon, no poles) so the point count
    matches the Gaussian proxy expected by era5_to_cubedsphere_carry:
      n_max = 360 // 2 - 1 = 179  →  n_lat = 180, n_lon = 360 (linear dealiasing)
      src_flat_size = 180 * 360 = 64800
    """
    cmd = [
        CDO, "-f", "nc4",
        "-remapnn,r360x180",
        f"-seltimestep,{day_index}",
        str(src), str(dst),
    ]
    log.info(f"  CDO: {src.name} → {dst.name}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"CDO failed for {src}:\n{result.stderr}")


def _open_and_rename(path: Path, varname: str) -> xr.Dataset:
    """Open CDO-produced NetCDF, rename var code to WB2 name, fix pressure units."""
    ds = xr.open_dataset(str(path), engine="netcdf4")

    # CDO names pressure-level variables as 'var<code>'
    raw = [v for v in ds.data_vars if v.startswith("var")]
    if len(raw) == 1:
        ds = ds.rename({raw[0]: varname})
    elif varname not in ds.data_vars:
        raise ValueError(f"Cannot find variable in {path}: {list(ds.data_vars)}")

    # Rename plev → level and convert Pa → hPa
    if "plev" in ds.dims:
        plev_Pa = ds["plev"].values
        plev_hPa = (plev_Pa / 100.0).astype(np.float32)
        ds = ds.rename({"plev": "level"})
        ds = ds.assign_coords(level=("level", plev_hPa))
        ds["level"].attrs["units"] = "hPa"

    # CDO names surface fields 'plev_bnds' sometimes – drop if present
    drop = [c for c in ds.coords if c not in ("time", "lat", "lon", "level")]
    ds = ds.drop_vars(drop, errors="ignore")

    return ds


def build_era5_ic_zarr(
    year: int,
    month: int,
    day: int,
    out_path: str | Path,
) -> Path:
    """Build a single-snapshot ERA5 Zarr store for use as AMIP initial conditions.

    Parameters
    ----------
    year, month, day : int
        Date of initial conditions.
    out_path : str or Path
        Output Zarr store path.

    Returns
    -------
    Path to the written Zarr store.
    """
    out_path = Path(out_path)
    if out_path.exists():
        import shutil
        log.info(f"Removing existing Zarr at {out_path}")
        shutil.rmtree(out_path)

    with tempfile.TemporaryDirectory(prefix="era5ic_", dir=out_path.parent) as tmpdir:
        tmp = Path(tmpdir)

        datasets: list[xr.Dataset] = []

        # --- Pressure-level variables ---
        for code, varname in _PL_VARS.items():
            src = _grib_path(code, year, month, kind="pl")
            if not src.exists():
                raise FileNotFoundError(f"ERA5 GRIB not found: {src}")
            dst = tmp / f"var{code}.nc"
            _cdo_to_nc4(src, dst, day_index=day)
            ds = _open_and_rename(dst, varname)
            # Keep only (time, level, lat, lon) — drop extras
            keep_dims = {"time", "level", "lat", "lon"}
            drop_coords = [c for c in ds.coords if c not in keep_dims]
            ds = ds.drop_vars(drop_coords, errors="ignore")
            # Drop time dim — we want a single snapshot
            ds = ds.isel(time=0, drop=True)
            datasets.append(ds[[varname]])

        # --- Surface variables ---
        for code, varname in _SF_VARS.items():
            src = _grib_path(code, year, month, kind="sf")
            if not src.exists():
                raise FileNotFoundError(f"ERA5 GRIB not found: {src}")
            dst = tmp / f"var{code}.nc"
            _cdo_to_nc4(src, dst, day_index=day)
            ds = _open_and_rename(dst, varname)
            drop_coords = [c for c in ds.coords if c not in {"lat", "lon"}]
            ds = ds.drop_vars(drop_coords, errors="ignore")
            # Squeeze out time, level dims if present
            for dim in ("time", "level", "plev"):
                if dim in ds.dims:
                    ds = ds.isel({dim: 0}, drop=True)
            datasets.append(ds[[varname]])

        # --- Surface geopotential (invariant) ---
        src_inv = _grib_path(_INV_VAR, year, month, kind="inv")
        if src_inv.exists():
            dst_inv = tmp / f"var{_INV_VAR}.nc"
            # Invariant file: extract first (only) time step
            _cdo_to_nc4(src_inv, dst_inv, day_index=1)
            ds_inv = _open_and_rename(dst_inv, "geopotential_at_surface")
            drop_coords = [c for c in ds_inv.coords if c not in {"lat", "lon"}]
            ds_inv = ds_inv.drop_vars(drop_coords, errors="ignore")
            for dim in ("time", "level", "plev"):
                if dim in ds_inv.dims:
                    ds_inv = ds_inv.isel({dim: 0}, drop=True)
            datasets.append(ds_inv[["geopotential_at_surface"]])
        else:
            log.warning(f"Surface geopotential invariant not found at {src_inv} — using zeros")

        # --- Merge and add a dummy time coordinate ---
        import pandas as pd
        merged = xr.merge(datasets)
        t0 = pd.Timestamp(year=year, month=month, day=day)
        merged = merged.expand_dims("time").assign_coords(
            time=("time", [np.datetime64(t0)])
        )

        # Ensure lat/lon are float32 and in expected ranges
        if "lat" in merged.coords:
            lat = merged.lat.values.astype(np.float32)
            lon = merged.lon.values.astype(np.float32)
            # Shift lon from 0..359 → -180..179 if needed
            # (era5_to_state._open_era5_zarr handles both conventions)
            merged = merged.assign_coords(lat=lat, lon=lon)

        log.info(f"Writing Zarr to {out_path} …")
        merged.to_zarr(str(out_path), mode="w")
        log.info(f"Done. Variables: {list(merged.data_vars)}")
        log.info(f"  Pressure levels (hPa): {list(merged.level.values[:5])} … {list(merged.level.values[-3:])}")
        log.info(f"  Grid: {merged.lat.size} lat × {merged.lon.size} lon")

    return out_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--year", type=int, default=1979)
    parser.add_argument("--month", type=int, default=1)
    parser.add_argument("--day", type=int, default=1,
                        help="Day-of-month (CDO time step index within the monthly file)")
    parser.add_argument("--out", type=str,
                        default="/scratch/b/b309178/era5_ic_1979-01-01.zarr",
                        help="Output Zarr store path")
    args = parser.parse_args(argv)

    out = build_era5_ic_zarr(args.year, args.month, args.day, args.out)
    print(f"ERA5 IC Zarr written to: {out}")


if __name__ == "__main__":
    main()
