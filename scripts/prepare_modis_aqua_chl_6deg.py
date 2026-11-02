#!/usr/bin/env python3
"""Prepare MODIS-Aqua chlorophyll monthly climatology with a 6-degree filter.

Source dataset (OPeNDAP):
http://apdrc.soest.hawaii.edu:80/dods/public_data/satellite_product/MODIS_Aqua/chla_mapped_daily_9km

Processing steps:
1) Load daily chlor_a from OPeNDAP.
2) Select requested years (clamped to available data range).
3) Compute monthly climatology (12 months).
4) Apply NaN-aware 6-degree spatial box filter.
5) Interpolate to the model grid from bathy_from_woa.nc.
6) Save NetCDF with variable chlor_a(month, lat, lon).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.ndimage import uniform_filter

DEFAULT_URL = "http://apdrc.soest.hawaii.edu:80/dods/public_data/satellite_product/MODIS_Aqua/chla_mapped_daily_9km"


def open_opendap_with_retry(url: str, retries: int = 6, backoff_s: float = 10.0) -> xr.Dataset:
    candidates = [url]
    if ":80/" in url:
        candidates.append(url.replace(":80", ""))
    if url.startswith("http://"):
        candidates.append(url.replace("http://", "https://"))

    last_err: Exception | None = None
    for candidate in candidates:
        for attempt in range(1, retries + 1):
            try:
                print(f"Opening OPeNDAP dataset: {candidate} (attempt {attempt}/{retries})", flush=True)
                return xr.open_dataset(candidate)
            except Exception as exc:  # pragma: no cover - network/server transient handling
                last_err = exc
                if attempt < retries:
                    wait_s = backoff_s * attempt
                    print(f"Open failed ({exc}); retrying in {wait_s:.0f}s", flush=True)
                    time.sleep(wait_s)
                else:
                    print(f"Open failed on candidate {candidate}: {exc}", flush=True)

    assert last_err is not None
    raise last_err


def nan_aware_uniform_filter_2d(field: np.ndarray, size_lat: int, size_lon: int) -> np.ndarray:
    valid = np.isfinite(field)
    num = uniform_filter(np.where(valid, field, 0.0), size=(size_lat, size_lon), mode=("nearest", "wrap"))
    den = uniform_filter(valid.astype(np.float64), size=(size_lat, size_lon), mode=("nearest", "wrap"))
    out = np.where(den > 0.0, num / den, np.nan)
    return out


def to_0360_lon(da: xr.DataArray) -> xr.DataArray:
    lon = ((da["lon"] + 360.0) % 360.0)
    return da.assign_coords(lon=lon).sortby("lon")


def add_wrap_lon_column(da: xr.DataArray) -> xr.DataArray:
    first = da.isel(lon=0)
    first = first.assign_coords(lon=float(first["lon"].values) + 360.0)
    return xr.concat([da, first], dim="lon")


def compute_monthly_climatology_streaming(
    chl: xr.DataArray,
    start_year: int,
    end_year: int,
) -> xr.DataArray:
    nlat = chl.sizes["lat"]
    nlon = chl.sizes["lon"]

    monthly_out = np.full((12, nlat, nlon), np.nan, dtype=np.float64)

    for month in range(1, 13):
        sum_field = np.zeros((nlat, nlon), dtype=np.float64)
        cnt_field = np.zeros((nlat, nlon), dtype=np.float64)
        total_days = 0

        for year in range(start_year, end_year + 1):
            t0 = f"{year:04d}-{month:02d}-01"
            t1 = f"{year:04d}-{month:02d}-31"
            sub = chl.sel(time=slice(t0, t1))
            ntime = sub.sizes.get("time", 0)
            if ntime == 0:
                continue

            arr = np.asarray(sub.values, dtype=np.float64)
            valid = np.isfinite(arr)
            sum_field += np.nansum(arr, axis=0)
            cnt_field += np.sum(valid, axis=0)
            total_days += int(ntime)

        if total_days == 0:
            print(f"month {month:02d}: no samples found", flush=True)
            continue

        monthly_out[month - 1] = np.where(cnt_field > 0.0, sum_field / cnt_field, np.nan)
        print(f"month {month:02d}: aggregated {total_days} daily fields", flush=True)

    return xr.DataArray(
        monthly_out,
        dims=("month", "lat", "lon"),
        coords={
            "month": np.arange(1, 13, dtype=np.int32),
            "lat": chl["lat"].values,
            "lon": chl["lon"].values,
        },
        name="chlor_a",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare MODIS-Aqua chlor_a with 6-degree spatial filtering")
    parser.add_argument("--url", type=str, default=DEFAULT_URL)
    parser.add_argument("--start-year", type=int, default=1997)
    parser.add_argument("--end-year", type=int, default=2014)
    parser.add_argument("--filter-deg", type=float, default=6.0)
    parser.add_argument("--open-retries", type=int, default=6)
    parser.add_argument("--open-backoff", type=float, default=10.0)
    parser.add_argument(
        "--source-stride",
        type=int,
        default=12,
        help="Stride applied on source lat/lon before processing (12 ~= 1 degree from 9 km grid)",
    )
    parser.add_argument("--model-grid", type=str, default="bathy_from_woa.nc")
    parser.add_argument("--out", type=str, default="results/chl_satellite_fit/modis_aqua_chlor_a_monthly_clim_6deg_on_woa5deg.nc")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ds = open_opendap_with_retry(
        args.url,
        retries=max(1, int(args.open_retries)),
        backoff_s=max(1.0, float(args.open_backoff)),
    )
    if "chlor_a" not in ds:
        raise KeyError("Variable 'chlor_a' not found in dataset")

    available_start = int(ds["time"].dt.year.min().values)
    available_end = int(ds["time"].dt.year.max().values)
    use_start = max(args.start_year, available_start)
    use_end = min(args.end_year, available_end)
    if use_start > use_end:
        raise ValueError(
            f"Requested period {args.start_year}-{args.end_year} has no overlap with available MODIS period {available_start}-{available_end}"
        )

    if use_start != args.start_year or use_end != args.end_year:
        print(
            f"Requested period {args.start_year}-{args.end_year} was clamped to available {use_start}-{use_end}"
        )

    stride = max(1, int(args.source_stride))
    chl = ds["chlor_a"].astype(np.float64).isel(lat=slice(None, None, stride), lon=slice(None, None, stride))
    print(
        f"Using source stride={stride}: working grid is "
        f"{chl.sizes['lat']} x {chl.sizes['lon']}"
    )
    monthly = compute_monthly_climatology_streaming(chl, use_start, use_end)

    dlat = float(np.abs(monthly["lat"].values[1] - monthly["lat"].values[0]))
    dlon = float(np.abs(monthly["lon"].values[1] - monthly["lon"].values[0]))
    win_lat = max(3, int(round(args.filter_deg / dlat)))
    win_lon = max(3, int(round(args.filter_deg / dlon)))
    if win_lat % 2 == 0:
        win_lat += 1
    if win_lon % 2 == 0:
        win_lon += 1
    print(f"Applying NaN-aware {args.filter_deg:.1f} degree filter with windows lat={win_lat}, lon={win_lon}")

    filtered = np.empty(monthly.shape, dtype=np.float64)
    for i in range(12):
        arr = np.asarray(monthly.isel(month=i).values, dtype=np.float64)
        filtered[i] = nan_aware_uniform_filter_2d(arr, win_lat, win_lon)

    filtered_da = xr.DataArray(
        filtered,
        dims=("month", "lat", "lon"),
        coords={"month": monthly["month"].values, "lat": monthly["lat"].values, "lon": monthly["lon"].values},
        name="chlor_a",
    )

    grid = xr.open_dataset(args.model_grid)
    target_lat = np.asarray(grid["lat"].values, dtype=np.float64)
    target_lon = np.asarray(grid["lon"].values, dtype=np.float64)

    filtered_da = to_0360_lon(filtered_da)
    filtered_da = add_wrap_lon_column(filtered_da)

    interp = filtered_da.interp(lat=target_lat, lon=target_lon, method="linear")

    out_ds = xr.Dataset(
        {
            "chlor_a": (("month", "lat", "lon"), np.asarray(interp.values, dtype=np.float32)),
        },
        coords={
            "month": np.arange(1, 13, dtype=np.int32),
            "lat": target_lat,
            "lon": target_lon,
        },
        attrs={
            "source_url": args.url,
            "source_variable": "chlor_a",
            "source_time_range_used": f"{use_start}-{use_end}",
            "requested_time_range": f"{args.start_year}-{args.end_year}",
            "spatial_filter": f"nan-aware box filter, {args.filter_deg} degrees",
            "note": "MODIS Aqua begins in 2002; earlier requested years are unavailable",
        },
    )

    out_ds.to_netcdf(out_path)
    print(f"Saved filtered monthly climatology to {out_path}")


if __name__ == "__main__":
    main()
