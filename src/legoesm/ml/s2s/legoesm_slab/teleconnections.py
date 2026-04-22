"""Lightweight teleconnection diagnostics for initialized slab runs.

These helpers intentionally stay conservative.  They expose simple regional
time series that are useful for conditioned skill analysis without claiming a
full ENSO or MJO verification pipeline.
"""

from __future__ import annotations

import numpy as np
import xarray as xr


def _normalize_lon_bounds(
    lon: xr.DataArray,
    lon_range: tuple[float, float],
) -> xr.DataArray:
    lon_values = np.mod(np.asarray(lon, dtype=float), 360.0)
    lon_min, lon_max = (float(bound) % 360.0 for bound in lon_range)
    if lon_min <= lon_max:
        mask = (lon_values >= lon_min) & (lon_values <= lon_max)
    else:
        mask = (lon_values >= lon_min) | (lon_values <= lon_max)
    return xr.DataArray(mask, coords={lon.dims[0]: lon}, dims=lon.dims)


def weighted_region_mean(
    field: xr.DataArray,
    *,
    lat_range: tuple[float, float],
    lon_range: tuple[float, float],
    latitude_name: str = "latitude",
    longitude_name: str = "longitude",
) -> xr.DataArray:
    """Return a cosine-weighted regional mean over a lat-lon box."""
    latitude = field[latitude_name]
    longitude = field[longitude_name]
    lat_mask = (latitude >= float(lat_range[0])) & (latitude <= float(lat_range[1]))
    lon_mask = _normalize_lon_bounds(longitude, lon_range)
    region = field.where(lat_mask & lon_mask, drop=True)
    weights = np.cos(np.deg2rad(region[latitude_name]))
    return region.weighted(weights).mean((latitude_name, longitude_name))


def compute_nino34_series(
    dataset: xr.Dataset,
    *,
    sst_var: str = "sea_surface_temperature",
    lat_range: tuple[float, float] = (-5.0, 5.0),
    lon_range: tuple[float, float] = (190.0, 240.0),
) -> xr.DataArray:
    """Return a raw Niño3.4-region SST series."""
    if sst_var not in dataset:
        raise KeyError(f"{sst_var!r} is missing from the supplied dataset")
    return weighted_region_mean(
        dataset[sst_var],
        lat_range=lat_range,
        lon_range=lon_range,
    ).rename("nino34_sst")


def classify_enso_phase(
    series: xr.DataArray,
    *,
    warm_threshold: float = 0.5,
    cold_threshold: float = -0.5,
) -> xr.DataArray:
    """Classify a scalar SST anomaly series into warm/neutral/cold phases."""
    values = np.asarray(series, dtype=float)
    labels = np.full(values.shape, "neutral", dtype=object)
    labels[values >= float(warm_threshold)] = "warm"
    labels[values <= float(cold_threshold)] = "cold"
    return xr.DataArray(labels, coords=series.coords, dims=series.dims, name="enso_phase")


def compute_mjo_wind_shear_proxy(
    dataset: xr.Dataset,
    *,
    u_var: str = "u_component_of_wind",
    lower_level: int = 850,
    upper_level: int = 200,
    lat_range: tuple[float, float] = (-15.0, 15.0),
    lon_range: tuple[float, float] = (40.0, 180.0),
    level_name: str = "level",
) -> xr.DataArray:
    """Return a simple tropical zonal-wind shear proxy for MJO analysis."""
    if u_var not in dataset:
        raise KeyError(f"{u_var!r} is missing from the supplied dataset")
    zonal = dataset[u_var]
    u850 = zonal.sel({level_name: lower_level}, method="nearest")
    u200 = zonal.sel({level_name: upper_level}, method="nearest")
    shear = u850 - u200
    return weighted_region_mean(
        shear,
        lat_range=lat_range,
        lon_range=lon_range,
    ).rename("mjo_wind_shear_proxy")


__all__ = [
    "classify_enso_phase",
    "compute_mjo_wind_shear_proxy",
    "compute_nino34_series",
    "weighted_region_mean",
]
