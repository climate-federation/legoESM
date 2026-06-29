#!/usr/bin/env python
"""Prepare ERA5 initial-condition Zarr store from Levante pool data.

Reads ERA5 GRIB files from /pool/data/ERA5/E5/ (Levante), regrids to a
regular 1-degree lat-lon grid via CDO, converts units, and writes a
single-snapshot Zarr store compatible with ``load_era5_ic`` in
``legoesm.training.era5_to_state``.

Required variables (ECMWF parameter codes):
  Pressure levels (pl/an/1H/):
    130 temperature          → temperature              [K]
    131 u-wind               → u_component_of_wind      [m/s]
    132 v-wind               → v_component_of_wind      [m/s]
    133 specific humidity    → specific_humidity         [kg/kg]
  Surface (sf/an/1H/):
    134 surface pressure     → surface_pressure          [Pa]
    235 skin temperature     → skin_temperature          [K]
  Surface invariant (sf/an/IV/):
    129 surface geopotential → geopotential_at_surface   [m²/s²]

Usage::

    python scripts/data/prep_era5_ic.py \\
        --year 1979 --month 1 --day 1 --hour 0 \\
        --out /scratch/b/b309178/era5_ic_1979-01-01.zarr

The output Zarr is then passed to run_amip.py via --ic era5 --ic-path <path>.
"""
from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import xarray as xr

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

ERA5_ROOT = Path("/pool/data/ERA5/E5")
CDO = "/sw/spack-levante/cdo-2.2.2-4z4icb/bin/cdo"

# ECMWF parameter code → WB2-compatible variable name
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
_INV_CODE = 129  # surface geopotential (time-invariant)


def _grib_path_1h(code: int, year: int, month: int, day: int,
                  kind: str = "pl") -> Path:
    """Return path to an hourly ERA5 GRIB file for a specific day.

    Files under pl/an/1H/ and sf/an/1H/ contain 24 hourly snapshots per
    day (one file per calendar date).  The invariant surface geopotential
    (kind='inv') lives in sf/an/IV/ and is time-independent.
    """
    ymd = f"{year:04d}-{month:02d}-{day:02d}"
    if kind == "inv":
        return ERA5_ROOT / "sf" / "an" / "IV" / str(code) / \
               f"E5sf00_IV_INVARIANT_{code}.grb"
    if kind == "pl":
        return ERA5_ROOT / "pl" / "an" / "1H" / str(code) / \
               f"E5pl00_1H_{ymd}_{code}.grb"
    # sf
    return ERA5_ROOT / "sf" / "an" / "1H" / str(code) / \
           f"E5sf00_1H_{ymd}_{code}.grb"


def _cdo_to_nc4(src: Path, dst: Path, timestep: int = 1) -> None:
    """Extract one time step from a GRIB file and regrid to 1° regular lat-lon.

    Uses r360x180 (180 lat × 360 lon, no poles) so the point count
    matches the Gaussian proxy expected by era5_to_cubedsphere_carry:
      n_lon=360  →  n_max = 360//2 - 1 = 179  →  n_lat=180 (linear dealiasing)
      flat size = 180 × 360 = 64800

    timestep is 1-indexed: 1 = first hour (00:00 UTC), 2 = 01:00 UTC, etc.
    """
    cmd = [
        CDO, "-f", "nc4",
        "-remapnn,r360x180",
        f"-seltimestep,{timestep}",
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
        raise ValueError(
            f"Cannot find variable in {path}: {list(ds.data_vars)}"
        )

    # Rename plev → level and convert Pa → hPa
    if "plev" in ds.dims:
        plev_hPa = (ds["plev"].values / 100.0).astype(np.float32)
        ds = ds.rename({"plev": "level"})
        ds = ds.assign_coords(level=("level", plev_hPa))
        ds["level"].attrs["units"] = "hPa"

    # Drop auxiliary coordinates that confuse xr.merge
    keep = {"time", "lat", "lon", "level"}
    ds = ds.drop_vars([c for c in ds.coords if c not in keep], errors="ignore")
    return ds


def build_era5_ic_zarr(
    year: int,
    month: int,
    day: int,
    hour: int = 0,
    out_path: str | Path = "/scratch/b/b309178/era5_ic_1979-01-01.zarr",
) -> Path:
    """Build a single-snapshot ERA5 Zarr store for use as AMIP initial conditions.

    Parameters
    ----------
    year, month, day : int
        Date of initial conditions.
    hour : int
        UTC hour (0–23).  Default 0 = 00:00 UTC.
    out_path : str or Path
        Output Zarr store path.  Overwritten if it already exists.

    Returns
    -------
    Path to the written Zarr store.
    """
    out_path = Path(out_path)
    if out_path.exists():
        log.info(f"Removing existing Zarr at {out_path}")
        shutil.rmtree(out_path)

    # CDO time step is 1-indexed; hour 0 = step 1
    timestep = hour + 1

    tmpdir_base = Path("/scratch/b/b309178")
    tmpdir_base.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="era5ic_", dir=str(tmpdir_base)) as tmpdir:
        tmp = Path(tmpdir)
        datasets: list[xr.Dataset] = []

        # --- Pressure-level variables ---
        for code, varname in _PL_VARS.items():
            src = _grib_path_1h(code, year, month, day, kind="pl")
            if not src.exists():
                raise FileNotFoundError(f"ERA5 GRIB not found: {src}")
            dst = tmp / f"var{code}.nc"
            _cdo_to_nc4(src, dst, timestep=timestep)
            ds = _open_and_rename(dst, varname)
            # Drop time dim — single snapshot
            if "time" in ds.dims:
                ds = ds.isel(time=0, drop=True)
            datasets.append(ds[[varname]])

        # --- Surface variables ---
        for code, varname in _SF_VARS.items():
            src = _grib_path_1h(code, year, month, day, kind="sf")
            if not src.exists():
                raise FileNotFoundError(f"ERA5 GRIB not found: {src}")
            dst = tmp / f"var{code}.nc"
            _cdo_to_nc4(src, dst, timestep=timestep)
            ds = _open_and_rename(dst, varname)
            for dim in ("time", "level", "plev"):
                if dim in ds.dims:
                    ds = ds.isel({dim: 0}, drop=True)
            datasets.append(ds[[varname]])

        # --- Surface geopotential (time-invariant orography) ---
        src_inv = _grib_path_1h(_INV_CODE, year, month, day, kind="inv")
        if src_inv.exists():
            dst_inv = tmp / f"var{_INV_CODE}.nc"
            _cdo_to_nc4(src_inv, dst_inv, timestep=1)
            ds_inv = _open_and_rename(dst_inv, "geopotential_at_surface")
            for dim in ("time", "level", "plev"):
                if dim in ds_inv.dims:
                    ds_inv = ds_inv.isel({dim: 0}, drop=True)
            datasets.append(ds_inv[["geopotential_at_surface"]])
        else:
            log.warning(
                f"Surface geopotential invariant not found at {src_inv} — "
                "using zeros.  ERA5 IC will have no orography."
            )

        # --- Merge and add a single-element time coordinate ---
        import pandas as pd
        merged = xr.merge(datasets)
        t0 = np.datetime64(
            pd.Timestamp(year=year, month=month, day=day, hour=hour)
        )
        merged = merged.expand_dims("time").assign_coords(
            time=("time", [t0])
        )

        # Ensure lat/lon are float32
        if "lat" in merged.coords:
            merged = merged.assign_coords(
                lat=merged.lat.values.astype(np.float32),
                lon=merged.lon.values.astype(np.float32),
            )

        log.info(f"Writing Zarr to {out_path} ...")
        merged.to_zarr(str(out_path), mode="w")

        log.info(f"Done. Variables: {list(merged.data_vars)}")
        if "level" in merged.dims:
            levs = list(merged.level.values)
            log.info(f"  Pressure levels (hPa): {levs[:4]} ... {levs[-3:]}")
        log.info(f"  Grid: {merged.lat.size} lat × {merged.lon.size} lon")

    return out_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--year", type=int, default=1979)
    parser.add_argument("--month", type=int, default=1)
    parser.add_argument("--day", type=int, default=1)
    parser.add_argument("--hour", type=int, default=0,
                        help="UTC hour (0-23); default 0 = 00:00 UTC")
    parser.add_argument(
        "--out", type=str,
        default="/scratch/b/b309178/era5_ic_1979-01-01.zarr",
        help="Output Zarr store path",
    )
    args = parser.parse_args(argv)

    out = build_era5_ic_zarr(args.year, args.month, args.day, args.hour, args.out)
    print(f"ERA5 IC Zarr written to: {out}")


if __name__ == "__main__":
    main()
