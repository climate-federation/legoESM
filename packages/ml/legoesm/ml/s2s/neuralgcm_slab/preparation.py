"""Real-data preparation utilities for NeuralGCM slab-ocean experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import xarray as xr

from legoesm.ml.s2s.neuralgcm_slab.neuralgcm_backend import (
    DEFAULT_NEURALGCM_CHECKPOINT,
    NeuralGCMBackend,
    NeuralGCMBackendConfig,
)

DEFAULT_ARCO_ERA5_STORE = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
DEFAULT_PREP_CHECKPOINT = DEFAULT_NEURALGCM_CHECKPOINT
DEFAULT_RADIATION_VARIABLES = {
    "sw_down": "surface_solar_radiation_downwards",
    "lw_down": "surface_thermal_radiation_downwards",
}


@dataclass(frozen=True)
class PreparationConfig:
    """Configuration for real-data NeuralGCM case preparation."""

    era5_store: str = DEFAULT_ARCO_ERA5_STORE
    checkpoint: str = DEFAULT_PREP_CHECKPOINT
    time_name: str = "time"
    latitude_name: str = "latitude"
    longitude_name: str = "longitude"
    level_name: str = "level"
    surface_pressure: str = "surface_pressure"
    sst_variable: str = "sea_surface_temperature"
    sea_ice_variable: str = "sea_ice_cover"
    sw_down_variable: str = DEFAULT_RADIATION_VARIABLES["sw_down"]
    lw_down_variable: str = DEFAULT_RADIATION_VARIABLES["lw_down"]


def _open_era5_store(store: str) -> xr.Dataset:
    storage_options = {"token": "anon"} if store.startswith("gs://") else None
    try:
        return xr.open_zarr(store, chunks=None, consolidated=None, storage_options=storage_options)
    except (ImportError, ModuleNotFoundError) as exc:
        dependency_hint = "Opening ERA5 Zarr stores requires zarr/fsspec"
        if store.startswith("gs://"):
            dependency_hint += " and gcsfs for gs:// access"
        raise ImportError(
            f"{dependency_hint}. Install legoesm[data] before running NeuralGCM preparation "
            f"(store={store!r})."
        ) from exc


def _import_dinosaur() -> tuple[Any, Any, Any]:
    try:
        from dinosaur import horizontal_interpolation, spherical_harmonic, xarray_utils
    except ImportError as exc:
        raise ImportError(
            "NeuralGCM preparation requires the dinosaur package. Install the NeuralGCM "
            "preparation dependencies before running prepare or ensemble-inference."
        ) from exc

    return horizontal_interpolation, spherical_harmonic, xarray_utils


def build_source_grid(dataset: xr.Dataset, config: PreparationConfig) -> Any:
    _, spherical_harmonic, _ = _import_dinosaur()
    return spherical_harmonic.Grid(
        longitude_nodes=int(dataset.sizes[config.longitude_name]),
        latitude_nodes=int(dataset.sizes[config.latitude_name]),
        latitude_spacing="equiangular_with_poles",
    )


def _make_regridder(source_grid: Any, target_grid: Any) -> Any:
    horizontal_interpolation, _, _ = _import_dinosaur()
    return horizontal_interpolation.ConservativeRegridder(source_grid, target_grid)


def _regrid_field_by_slice(field: xr.DataArray, regridder: Any) -> xr.DataArray:
    _, _, xarray_utils = _import_dinosaur()
    spatial_dims = tuple(dim for dim in field.dims if dim in ("longitude", "latitude"))
    non_spatial_dims = tuple(dim for dim in field.dims if dim not in spatial_dims)
    if not non_spatial_dims:
        filled = xarray_utils.fill_nan_with_nearest(field)
        return xarray_utils.regrid_horizontal(filled, regridder)

    stacked = field.stack(_non_spatial_slice=non_spatial_dims)
    regridded_slices: list[xr.DataArray] = []
    for index in range(int(stacked.sizes["_non_spatial_slice"])):
        slice_field = stacked.isel(_non_spatial_slice=index, drop=True)
        filled_slice = xarray_utils.fill_nan_with_nearest(slice_field)
        regridded_slice = xarray_utils.regrid_horizontal(filled_slice, regridder)
        coord_value = stacked["_non_spatial_slice"].values[index]
        regridded_slices.append(
            regridded_slice.expand_dims({"_non_spatial_slice": [coord_value]})
        )

    combined = xr.concat(regridded_slices, dim="_non_spatial_slice")
    if len(non_spatial_dims) == 1:
        dim_name = non_spatial_dims[0]
        combined = combined.rename({"_non_spatial_slice": dim_name}).assign_coords(
            {dim_name: field[dim_name]}
        )
    else:
        combined = combined.unstack("_non_spatial_slice")
    combined = combined.transpose(*field.dims)
    combined.attrs = dict(field.attrs)
    return combined.rename(field.name)


def _fill_and_regrid(field: xr.DataArray, regridder: Any) -> xr.DataArray:
    _, _, xarray_utils = _import_dinosaur()
    try:
        filled = xarray_utils.fill_nan_with_nearest(field)
    except ValueError as exc:
        if "NaN mask is not fixed across non-spatial dimensions" not in str(exc):
            raise
        return _regrid_field_by_slice(field, regridder)
    return xarray_utils.regrid_horizontal(filled, regridder)


def regrid_dataset_to_model_grid(dataset: xr.Dataset, *, regridder: Any) -> xr.Dataset:
    regridded = {
        name: _fill_and_regrid(dataset[name], regridder)
        for name in dataset.data_vars
    }
    return xr.Dataset(regridded, attrs=dataset.attrs)


def _coerce_time(value: str | np.datetime64) -> np.datetime64:
    return np.datetime64(value)


def _daily_times(start_time: np.datetime64, forecast_days: int) -> list[np.datetime64]:
    return [start_time + np.timedelta64(day, "D") for day in range(forecast_days)]


def _truth_times(start_time: np.datetime64, forecast_days: int) -> list[np.datetime64]:
    return [start_time + np.timedelta64(day, "D") for day in range(1, forecast_days + 1)]


def _save_dataset(dataset: xr.Dataset, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".zarr":
        dataset.to_zarr(path, mode="w")
    else:
        dataset.to_netcdf(path)
    return path


def _select_snapshot(
    dataset: xr.Dataset,
    variables: Sequence[str],
    when: np.datetime64,
    *,
    config: PreparationConfig,
) -> xr.Dataset:
    snapshot = dataset[list(variables)].sel({config.time_name: when})
    return snapshot.expand_dims({config.time_name: [when]})


def _select_snapshots(
    dataset: xr.Dataset,
    variables: Sequence[str],
    times: Sequence[np.datetime64],
    *,
    config: PreparationConfig,
) -> xr.Dataset:
    return dataset[list(variables)].sel({config.time_name: list(times)})


def _daily_flux_series(
    dataset: xr.Dataset,
    start_time: np.datetime64,
    forecast_days: int,
    *,
    config: PreparationConfig,
) -> xr.Dataset:
    start = start_time + np.timedelta64(1, "h")
    end = start_time + np.timedelta64(forecast_days, "D")
    hourly = dataset[[config.sw_down_variable, config.lw_down_variable]].sel(
        {config.time_name: slice(start, end)}
    )
    expected = 24 * int(forecast_days)
    count = int(hourly.sizes.get(config.time_name, 0))
    if count != expected:
        raise ValueError(
            f"Expected {expected} hourly radiation samples starting at {start_time}, got {count}"
        )
    daily = (hourly.coarsen({config.time_name: 24}, boundary="exact").sum() / 86400.0).rename(
        {
            config.sw_down_variable: "sw_down",
            config.lw_down_variable: "lw_down",
        }
    )
    return daily.assign_coords({config.time_name: _daily_times(start_time, forecast_days)})


def prepare_neuralgcm_case(
    *,
    start_time: str,
    forecast_days: int,
    output_dir: str | Path,
    config: PreparationConfig | None = None,
) -> dict[str, Path]:
    prep = config or PreparationConfig()
    start = _coerce_time(start_time)
    output_dir = Path(output_dir)

    backend = NeuralGCMBackend(NeuralGCMBackendConfig(checkpoint=prep.checkpoint))
    target_grid = backend.model.data_coords.horizontal

    era5 = _open_era5_store(prep.era5_store)
    source_grid = build_source_grid(era5, prep)
    regridder = _make_regridder(source_grid, target_grid)

    state_variables = list(dict.fromkeys([*backend.required_variables, prep.surface_pressure]))
    forcing_variables = [prep.sst_variable, prep.sea_ice_variable, prep.surface_pressure]
    truth_variables = list(
        dict.fromkeys(
            (
                "temperature",
                "specific_humidity",
                "geopotential",
                "u_component_of_wind",
                "v_component_of_wind",
                prep.sst_variable,
                prep.sea_ice_variable,
                prep.surface_pressure,
            )
        )
    )

    initial = regrid_dataset_to_model_grid(
        _select_snapshot(era5, state_variables, start, config=prep),
        regridder=regridder,
    )
    forcing = regrid_dataset_to_model_grid(
        _select_snapshots(era5, forcing_variables, _daily_times(start, forecast_days), config=prep),
        regridder=regridder,
    )
    radiation = regrid_dataset_to_model_grid(
        _daily_flux_series(era5, start, forecast_days, config=prep),
        regridder=regridder,
    )
    truth = regrid_dataset_to_model_grid(
        _select_snapshots(era5, truth_variables, _truth_times(start, forecast_days), config=prep),
        regridder=regridder,
    )

    paths = {
        "initial": _save_dataset(initial, output_dir / "initial.nc"),
        "forcing": _save_dataset(forcing, output_dir / "forcing.nc"),
        "radiation": _save_dataset(radiation, output_dir / "radiation.nc"),
        "truth": _save_dataset(truth, output_dir / "truth.nc"),
    }
    return paths


__all__ = [
    "DEFAULT_ARCO_ERA5_STORE",
    "DEFAULT_PREP_CHECKPOINT",
    "DEFAULT_RADIATION_VARIABLES",
    "PreparationConfig",
    "prepare_neuralgcm_case",
]
