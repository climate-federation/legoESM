#!/usr/bin/env python
"""Extract structural metadata from CMIP6 NetCDF forcing files on HPC."""

from __future__ import annotations

import glob
import json
import os
import sys

import xarray as xr


ROOT = "/pool/data/ICON/grids/public/mpim"

FILE_PATTERNS = {
    "solar_swflux_14band": "common/solar_radiation/swflux_14band_cmip6_1850-2299-v3.2.nc",
    "ghg_historical_plus": "independent/greenhouse_gases/greenhouse_historical_plus.nc",
    "aerosol_kinne_sw_b14_fin": "common/aerosol_kinne/aeropt_kinne_sw_b14_fin_*_rast.nc",
    "aerosol_kinne_lw_b16_coa": "common/aerosol_kinne/aeropt_kinne_lw_b16_coa_rast.nc",
    "aerosol_kinne_sw_b14_coa": "common/aerosol_kinne/aeropt_kinne_sw_b14_coa_rast.nc",
    "volcanic_aerosol_cmip6": "common/aerosol_volcanic_cmip6/bc_aeropt_cmip6_volc_lw_b16_sw_b14_*.nc",
    "ozone_cmip6_historical": "common/ozone_cmip6_forcing/historical/vmro3_input4MIPs_ozone_CMIP_UReading-CCMI-1-0_gn_*.nc",
}

BASIC_COORDS = {
    "lat", "latitude", "lon", "longitude", "time", "time_bnds",
    "lat_bnds", "lon_bnds", "time_bounds", "lat_bounds", "lon_bounds",
    "bnds", "nv",
}


def resolve_path(rel_pattern: str) -> str | None:
    full_pattern = os.path.join(ROOT, rel_pattern)
    if "*" in full_pattern:
        matches = sorted(glob.glob(full_pattern))
        return matches[0] if matches else None
    return full_pattern if os.path.exists(full_pattern) else None


def inspect_file(path: str) -> dict:
    out: dict = {"path": path}
    try:
        ds = xr.open_dataset(path, decode_times=False)
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
        return out

    try:
        out["dimensions"] = {d: int(s) for d, s in ds.dims.items()}

        coord_sizes: dict = {}
        for cname, cvar in ds.coords.items():
            try:
                arr = cvar.values
                entry = {
                    "size": int(cvar.size),
                    "dtype": str(arr.dtype),
                    "dims": list(cvar.dims),
                }
                if arr.ndim == 1 and arr.size > 0 and arr.size <= 250:
                    try:
                        entry["first"] = arr.flat[0].item() if hasattr(arr.flat[0], "item") else str(arr.flat[0])
                        entry["last"] = arr.flat[-1].item() if hasattr(arr.flat[-1], "item") else str(arr.flat[-1])
                    except Exception:
                        entry["first"] = str(arr.flat[0])
                        entry["last"] = str(arr.flat[-1])
                units = cvar.attrs.get("units")
                if units is not None:
                    entry["units"] = str(units)
                coord_sizes[cname] = entry
            except Exception as exc:
                coord_sizes[cname] = {"error": f"{type(exc).__name__}: {exc}"}
        out["coords"] = coord_sizes

        variables: dict = {}
        for vname, var in ds.data_vars.items():
            if vname in BASIC_COORDS:
                continue
            v_entry = {
                "dims": list(var.dims),
                "shape": [int(s) for s in var.shape],
                "dtype": str(var.dtype),
            }
            for ak in ("units", "long_name", "standard_name"):
                if ak in var.attrs:
                    v_entry[ak] = str(var.attrs[ak])
            variables[vname] = v_entry
        out["variables"] = variables

        global_attrs: dict = {}
        for ak in ("title", "source", "institution", "experiment_id",
                   "frequency", "table_id", "Conventions"):
            if ak in ds.attrs:
                global_attrs[ak] = str(ds.attrs[ak])
        if global_attrs:
            out["global_attrs"] = global_attrs
    finally:
        ds.close()

    return out


def main() -> int:
    result: dict = {}
    for key, pattern in FILE_PATTERNS.items():
        resolved = resolve_path(pattern)
        if resolved is None:
            result[key] = {
                "pattern": os.path.join(ROOT, pattern),
                "error": "no_match",
            }
            continue
        entry = inspect_file(resolved)
        entry["pattern"] = os.path.join(ROOT, pattern)
        result[key] = entry

    json.dump(result, sys.stdout, separators=(",", ":"), default=str)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
