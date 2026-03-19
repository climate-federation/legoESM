"""ARCO surface-forcing preparation for SFNO slab-ocean coupling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import xarray as xr

from legoesm.ml.sfno_s2s.config import CHAOSBENCH_DATA_DIR
from legoesm.ml.sfno_s2s.regrid import build_target_grid, regrid_channels_to_gaussian


DEFAULT_ARCO_ERA5_STORE = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"


@dataclass(frozen=True)
class ArcoSurfaceForcingConfig:
    """Configuration for preparing SFNO slab-ocean surface forcing from ARCO ERA5."""

    era5_store: str = DEFAULT_ARCO_ERA5_STORE
    time_name: str = "time"
    latitude_name: str = "latitude"
    longitude_name: str = "longitude"
    surface_pressure: str = "surface_pressure"
    sea_surface_temperature: str = "sea_surface_temperature"
    sea_ice_cover: str = "sea_ice_cover"
    sw_down: str = "surface_solar_radiation_downwards"
    lw_down: str = "surface_thermal_radiation_downwards"


@dataclass(frozen=True)
class ArcoSSTCacheConfig:
    """Configuration for preprocessing ARCO daily SST for SFNO training."""

    era5_store: str = DEFAULT_ARCO_ERA5_STORE
    time_name: str = "time"
    latitude_name: str = "latitude"
    longitude_name: str = "longitude"
    sea_surface_temperature: str = "sea_surface_temperature"
    chaosbench_data_dir: str = CHAOSBENCH_DATA_DIR
    oras5_reference_name: str = "oras5_full_1.5deg_19790101.zarr"


def _open_era5_store(store: str) -> xr.Dataset:
    storage_options = {"token": "anon"} if store.startswith("gs://") else None
    return xr.open_zarr(store, chunks=None, consolidated=None, storage_options=storage_options)


def _coerce_time(value: str | np.datetime64) -> np.datetime64:
    return value if isinstance(value, np.datetime64) else np.datetime64(value)


def _lead_times(start_time: np.datetime64, forecast_days: int) -> list[np.datetime64]:
    return [start_time + np.timedelta64(day, "D") for day in range(1, forecast_days + 1)]


def _daily_times(start_time: np.datetime64, end_time: np.datetime64) -> np.ndarray:
    if end_time < start_time:
        raise ValueError(f"end_time {end_time} precedes start_time {start_time}")
    return np.arange(start_time, end_time + np.timedelta64(1, "D"), np.timedelta64(1, "D"))


def _fill_nan_nearest(data: xr.DataArray) -> xr.DataArray:
    if data["latitude"].values[0] > data["latitude"].values[-1]:
        data = data.sortby("latitude")
    if data["longitude"].values[0] > data["longitude"].values[-1]:
        data = data.sortby("longitude")
    filled = data.interpolate_na(dim="longitude", method="nearest", fill_value="extrapolate")
    filled = filled.interpolate_na(dim="latitude", method="nearest", fill_value="extrapolate")
    values = np.asarray(filled.values, dtype=np.float32)
    if not np.isfinite(values).all():
        finite = values[np.isfinite(values)]
        fill_value = float(finite.mean()) if finite.size else 0.0
        values = np.where(np.isfinite(values), values, fill_value).astype(np.float32, copy=False)
        filled = xr.DataArray(values, coords=filled.coords, dims=filled.dims, attrs=filled.attrs)
    return filled


def _reference_lat_lon(
    *,
    config: ArcoSSTCacheConfig,
    reference_path: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if reference_path is None:
        reference_path = (
            Path(config.chaosbench_data_dir)
            / "oras5"
            / config.oras5_reference_name
        )
    ds = xr.open_dataset(reference_path, engine="zarr")
    try:
        latitude = np.asarray(ds["latitude"].values, dtype=np.float32)
        longitude = np.asarray(ds["longitude"].values, dtype=np.float32)
    finally:
        ds.close()
    return latitude, longitude


def _interp_to_reference_grid(
    field: xr.DataArray,
    *,
    latitude: np.ndarray,
    longitude: np.ndarray,
    config: ArcoSSTCacheConfig,
) -> xr.DataArray:
    data = field
    if data[config.latitude_name].values[0] > data[config.latitude_name].values[-1]:
        data = data.sortby(config.latitude_name)
    data = data.assign_coords(
        {
            config.longitude_name: np.mod(
                np.asarray(data[config.longitude_name].values, dtype=np.float64),
                360.0,
            )
        }
    ).sortby(config.longitude_name)
    return data.interp(
        {
            config.latitude_name: latitude.astype(np.float64),
            config.longitude_name: longitude.astype(np.float64),
        },
        method="linear",
    )


def _date_strings(times: Sequence[np.datetime64]) -> np.ndarray:
    return np.asarray([str(value)[:10] for value in times], dtype="U10")


def _regrid_time_field(
    field: xr.DataArray,
    *,
    target_n_max: int,
    preserve_nan_mask: bool = False,
) -> np.ndarray:
    target = build_target_grid(target_n_max)
    source_lat = np.asarray(field["latitude"].values, dtype=np.float64)
    source_lon = np.asarray(field["longitude"].values, dtype=np.float64)
    values = np.asarray(field.values, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError(f"Expected (time, lat, lon), got {values.shape}")

    if preserve_nan_mask:
        valid_mask = np.isfinite(values).astype(np.float32)
        filled = _fill_nan_nearest(field)
        regridded_values = regrid_channels_to_gaussian(
            np.asarray(filled.values, dtype=np.float32),
            source_lat=source_lat,
            source_lon=source_lon,
            target=target,
        )
        regridded_mask = regrid_channels_to_gaussian(
            valid_mask,
            source_lat=source_lat,
            source_lon=source_lon,
            target=target,
        )
        out = np.moveaxis(regridded_values, 0, 0).astype(np.float32, copy=False)
        out[regridded_mask < 0.5] = np.nan
        return out

    filled = _fill_nan_nearest(field)
    regridded = regrid_channels_to_gaussian(
        np.asarray(filled.values, dtype=np.float32),
        source_lat=source_lat,
        source_lon=source_lon,
        target=target,
    )
    return np.moveaxis(regridded, 0, 0).astype(np.float32, copy=False)


def _daily_flux_series(
    dataset: xr.Dataset,
    *,
    start_time: np.datetime64,
    forecast_days: int,
    config: ArcoSurfaceForcingConfig,
) -> xr.Dataset:
    start = start_time + np.timedelta64(1, "h")
    end = start_time + np.timedelta64(forecast_days, "D")
    hourly = dataset[[config.sw_down, config.lw_down]].sel({config.time_name: slice(start, end)})
    expected = 24 * forecast_days
    count = int(hourly.sizes.get(config.time_name, 0))
    if count != expected:
        raise ValueError(
            f"Expected {expected} hourly radiation samples starting at {start_time!s}, got {count}"
        )
    daily = hourly.coarsen({config.time_name: 24}, boundary="exact").sum() / 86400.0
    daily = daily.rename({config.sw_down: "sw_down", config.lw_down: "lw_down"})
    daily = daily.assign_coords({config.time_name: _lead_times(start_time, forecast_days)})
    return daily


def _sst_to_forcing_units(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size and float(np.nanmean(finite)) > 200.0:
        return values - np.float32(273.15)
    return values


def prepare_arco_sst_cache(
    *,
    start_time: str | np.datetime64,
    end_time: str | np.datetime64,
    config: ArcoSSTCacheConfig | None = None,
    output_path: str | Path,
    stats_output_path: str | Path | None = None,
    reference_path: str | Path | None = None,
    chunk_days: int = 31,
) -> tuple[xr.Dataset, xr.Dataset]:
    """Prepare a daily ARCO SST cache and matching scalar normalization stats.

    The cache is written on the ChaosBench ORAS5 1.5-degree lat-lon grid using:
    - `sosstsst` in Celsius-like forcing units
    - `ocean_mask` as a finite-value mask derived from the interpolated SST
    """
    prep = config or ArcoSSTCacheConfig()
    start = _coerce_time(start_time)
    end = _coerce_time(end_time)
    times = _daily_times(start, end)
    if chunk_days <= 0:
        raise ValueError(f"chunk_days must be positive, got {chunk_days}")

    latitude, longitude = _reference_lat_lon(config=prep, reference_path=reference_path)
    era5 = _open_era5_store(prep.era5_store)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if stats_output_path is not None:
        stats_output_path = Path(stats_output_path)
        stats_output_path.parent.mkdir(parents=True, exist_ok=True)

    sum_value = 0.0
    sumsq_value = 0.0
    count_value = 0
    first_chunk = True

    # Stream ARCO in small date chunks so the cache build is resumable and
    # does not have to materialize an entire multi-year range in memory.
    for start_index in range(0, len(times), chunk_days):
        chunk_times = times[start_index : start_index + chunk_days]
        sst = era5[prep.sea_surface_temperature].sel({prep.time_name: chunk_times})
        sst = _interp_to_reference_grid(
            sst,
            latitude=latitude,
            longitude=longitude,
            config=prep,
        )
        sst_values = _sst_to_forcing_units(np.asarray(sst.values, dtype=np.float32))
        ocean_mask = np.isfinite(sst_values)

        finite = sst_values[ocean_mask]
        if finite.size:
            sum_value += float(finite.sum(dtype=np.float64))
            sumsq_value += float(np.square(finite, dtype=np.float64).sum(dtype=np.float64))
            count_value += int(finite.size)

        chunk_ds = xr.Dataset(
            data_vars={
                "sosstsst": (
                    ("time", "latitude", "longitude"),
                    sst_values.astype(np.float32, copy=False),
                ),
                "ocean_mask": (
                    ("time", "latitude", "longitude"),
                    ocean_mask.astype(np.int8, copy=False),
                ),
            },
            coords={
                "time": chunk_times.astype("datetime64[ns]"),
                "date": ("time", _date_strings(chunk_times)),
                "latitude": latitude.astype(np.float32, copy=False),
                "longitude": longitude.astype(np.float32, copy=False),
            },
            attrs={
                "surface_source": "arco",
                "era5_store": prep.era5_store,
                "units": "degC_like",
            },
        )

        mode = "w" if first_chunk else "a"
        append_dim = None if first_chunk else "time"
        chunk_ds.to_zarr(output_path, mode=mode, append_dim=append_dim)
        first_chunk = False

    if count_value <= 0:
        raise ValueError("No finite SST values were found while building the ARCO cache.")

    mean = np.float32(sum_value / count_value)
    variance = max(sumsq_value / count_value - float(mean) ** 2, 1.0e-12)
    sigma = np.float32(np.sqrt(variance))
    stats_ds = xr.Dataset(
        data_vars={
            "mean": (("param",), np.asarray([mean], dtype=np.float32)),
            "sigma": (("param",), np.asarray([sigma], dtype=np.float32)),
        },
        coords={"param": np.asarray(["sosstsst"], dtype="U16")},
        attrs={
            "source": "arco",
            "start_time": str(start),
            "end_time": str(end),
            "era5_store": prep.era5_store,
            "count": int(count_value),
        },
    )
    if stats_output_path is not None:
        stats_ds.to_zarr(stats_output_path, mode="w")
    cache_ds = xr.open_dataset(output_path, engine="zarr")
    return cache_ds, stats_ds


def prepare_arco_surface_forcing(
    *,
    start_time: str | np.datetime64,
    forecast_days: int,
    gaussian_n_max: int,
    config: ArcoSurfaceForcingConfig | None = None,
    output_path: str | Path | None = None,
) -> xr.Dataset:
    """Prepare daily ARCO surface forcing on the SFNO Gaussian grid.

    The returned dataset contains daily lead-time snapshots on the target SFNO grid:
    - `surface_pressure`
    - `sea_surface_temperature`
    - `sea_ice_cover`
    - `sw_down`
    - `lw_down`

    It also includes:
    - `initial_sea_surface_temperature`
    - `initial_sea_ice_cover`

    This dataset is the surface contract consumed by the slab-coupled SFNO
    inference path. The SST and sea-ice fields keep their ocean mask so the
    coupler can distinguish ocean points from land points explicitly.
    """
    prep = config or ArcoSurfaceForcingConfig()
    start = _coerce_time(start_time)
    era5 = _open_era5_store(prep.era5_store)
    target = build_target_grid(gaussian_n_max)
    lead_times = _lead_times(start, forecast_days)

    initial_surface = era5[[prep.sea_surface_temperature, prep.sea_ice_cover]].sel({prep.time_name: start})
    future_surface = era5[
        [prep.surface_pressure, prep.sea_surface_temperature, prep.sea_ice_cover]
    ].sel({prep.time_name: lead_times})
    radiation = _daily_flux_series(era5, start_time=start, forecast_days=forecast_days, config=prep)

    initial_sst = _regrid_time_field(
        initial_surface[prep.sea_surface_temperature].expand_dims({prep.time_name: [start]}),
        target_n_max=gaussian_n_max,
        preserve_nan_mask=True,
    )[0]
    initial_sic = _regrid_time_field(
        initial_surface[prep.sea_ice_cover].expand_dims({prep.time_name: [start]}),
        target_n_max=gaussian_n_max,
        preserve_nan_mask=False,
    )[0]

    sst = _regrid_time_field(
        future_surface[prep.sea_surface_temperature],
        target_n_max=gaussian_n_max,
        preserve_nan_mask=True,
    )
    sic = np.clip(
        _regrid_time_field(
            future_surface[prep.sea_ice_cover],
            target_n_max=gaussian_n_max,
            preserve_nan_mask=False,
        ),
        0.0,
        1.0,
    )
    sp = _regrid_time_field(
        future_surface[prep.surface_pressure],
        target_n_max=gaussian_n_max,
        preserve_nan_mask=False,
    )
    sw_down = _regrid_time_field(radiation["sw_down"], target_n_max=gaussian_n_max, preserve_nan_mask=False)
    lw_down = _regrid_time_field(radiation["lw_down"], target_n_max=gaussian_n_max, preserve_nan_mask=False)

    sst = _sst_to_forcing_units(sst)
    initial_sst = _sst_to_forcing_units(initial_sst)

    ds = xr.Dataset(
        data_vars={
            "surface_pressure": (("lead_day", "latitude", "longitude"), sp.astype(np.float32, copy=False)),
            "sea_surface_temperature": (("lead_day", "latitude", "longitude"), sst.astype(np.float32, copy=False)),
            "sea_ice_cover": (("lead_day", "latitude", "longitude"), sic.astype(np.float32, copy=False)),
            "sw_down": (("lead_day", "latitude", "longitude"), sw_down.astype(np.float32, copy=False)),
            "lw_down": (("lead_day", "latitude", "longitude"), lw_down.astype(np.float32, copy=False)),
            "initial_sea_surface_temperature": (
                ("latitude", "longitude"),
                initial_sst.astype(np.float32, copy=False),
            ),
            "initial_sea_ice_cover": (
                ("latitude", "longitude"),
                np.clip(initial_sic, 0.0, 1.0).astype(np.float32, copy=False),
            ),
        },
        coords={
            "lead_day": np.arange(1, forecast_days + 1, dtype=np.int32),
            "target_date": ("lead_day", np.asarray([str(value)[:10] for value in lead_times], dtype="U10")),
            "latitude": target.latitude_deg.astype(np.float32, copy=False),
            "longitude": target.longitude_deg.astype(np.float32, copy=False),
        },
        attrs={
            "start_time": str(start),
            "gaussian_n_max": int(gaussian_n_max),
            "surface_source": "arco",
            "era5_store": prep.era5_store,
        },
    )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ds.to_netcdf(output_path)
    return ds


__all__ = [
    "ArcoSSTCacheConfig",
    "ArcoSurfaceForcingConfig",
    "DEFAULT_ARCO_ERA5_STORE",
    "prepare_arco_sst_cache",
    "prepare_arco_surface_forcing",
]
