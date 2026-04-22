"""ARCO/ERA5 case preparation for initialized legoESM slab forecasts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Sequence

import jax.numpy as jnp
import numpy as np
import xarray as xr

from legoesm.thermo import saturation_specific_humidity
from legoesm.grids.cubed_sphere import rotate_winds_geo_to_grid, rotate_winds_grid_to_geo
from legoesm.grids.edge_blending import (
    blend_scalar_cube_edges,
    blend_scalar_cube_edges_2d,
    blend_vector_cube_edges,
)
from legoesm.core.field import Field
from legoesm.core.state import HydrostaticState
from legoesm.driver.config import ExperimentConfig, experiment_config_to_dict
from legoesm.atmosphere.dynamics.primitive_eq_cdgrid import (
    hydrostatic_to_fv3,
    fv3_to_hydrostatic,
)
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.io.restart import save_restart
from legoesm.ml.s2s.legoesm_slab.campaign import surface_forcing_filename
from legoesm.training.vertical_interp import (
    compute_model_pressures_hybrid,
    compute_model_pressures_sigma,
    interp_pressure_to_hybrid,
    interp_pressure_to_sigma,
)
from legoesm.grids.regridding import (
    _pad_field_for_regrid,
    apply_cubedsphere_to_latlon,
    get_cubedsphere_to_latlon_weights,
)
from legoesm.grids.vertical import compute_geopotential, compute_geopotential_hybrid

DEFAULT_ARCO_ERA5_STORE = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
DEFAULT_PRESSURE_LEVELS = (200, 500, 700, 850)

_VAR_CANDIDATES = {
    "temperature": ("temperature", "t"),
    "specific_humidity": ("specific_humidity", "q"),
    "u_component_of_wind": ("u_component_of_wind", "u"),
    "v_component_of_wind": ("v_component_of_wind", "v"),
    "surface_pressure": ("surface_pressure", "sp"),
    "sea_surface_temperature": ("sea_surface_temperature", "sst"),
    "land_surface_temperature": ("land_surface_temperature", "skin_temperature", "skt"),
    "land_sea_mask": ("land_sea_mask", "lsm"),
    "sea_ice_cover": ("sea_ice_cover", "siconc", "sic"),
    "geopotential": ("geopotential", "z"),
    "surface_geopotential": ("geopotential_at_surface", "surface_geopotential", "z_sfc"),
    "total_precipitation": ("total_precipitation", "tp"),
    "specific_cloud_liquid_water_content": ("specific_cloud_liquid_water_content", "clwc"),
}


@dataclass(frozen=True)
class PreparationConfig:
    """Configuration for one initialized legoESM slab case."""

    era5_store: str = DEFAULT_ARCO_ERA5_STORE
    time_name: str = "time"
    latitude_name: str = "latitude"
    longitude_name: str = "longitude"
    level_name: str = "level"
    forcing_resolution_deg: float = 1.0
    evaluation_resolution_deg: float = 5.0
    pressure_levels: tuple[int, ...] = DEFAULT_PRESSURE_LEVELS
    seed_cloud_liquid_from_era5: bool = False
    cdgrid_balance_steps: int = 5
    cubedsphere_edge_blend_strength: float = 0.0
    cubedsphere_edge_blend_width: int = 0
    vertical_first_remap: bool = False
    mask_subsurface_pressure_levels: bool = False


@dataclass(frozen=True)
class PreparedCaseMetadata:
    """Serializable metadata for one prepared initialized case."""

    start_time: str
    forecast_days: int
    forcing_start_day: float
    forcing_resolution_deg: float
    evaluation_resolution_deg: float
    pressure_levels: tuple[int, ...]
    experiment_config: dict[str, Any]
    coupled: dict[str, Any]


def _open_era5_store(store: str) -> xr.Dataset:
    storage_options = {"token": "anon"} if store.startswith("gs://") else None
    try:
        return xr.open_zarr(
            store,
            chunks=None,
            consolidated=None,
            storage_options=storage_options,
        )
    except (ImportError, ModuleNotFoundError) as exc:
        dependency_hint = "Opening ERA5 Zarr stores requires zarr/fsspec"
        if store.startswith("gs://"):
            dependency_hint += " and gcsfs for gs:// access"
        raise ImportError(
            f"{dependency_hint}. Install legoesm[data] before running legoesm_slab preparation "
            f"(store={store!r})."
        ) from exc


def _find_variable(dataset: xr.Dataset, key: str) -> xr.DataArray:
    for candidate in _VAR_CANDIDATES[key]:
        if candidate in dataset:
            return dataset[candidate]
    available = ", ".join(sorted(set(dataset.data_vars) | set(dataset.coords)))
    raise KeyError(
        f"Could not resolve {key!r} in the ERA5 dataset. Available variables: {available}"
    )


def _find_optional_variable(dataset: xr.Dataset, key: str) -> xr.DataArray | None:
    for candidate in _VAR_CANDIDATES.get(key, ()):
        if candidate in dataset:
            return dataset[candidate]
    return None


def _sorted_era5_dataset(dataset: xr.Dataset, config: PreparationConfig) -> xr.Dataset:
    lon_name = config.longitude_name
    lat_name = config.latitude_name
    dataset = dataset.assign_coords(
        {lon_name: np.mod(np.asarray(dataset[lon_name], dtype=float), 360.0)}
    )
    return dataset.sortby(lat_name).sortby(lon_name)


def _target_grid_coords(resolution_deg: float) -> tuple[np.ndarray, np.ndarray]:
    n_lon = int(round(360.0 / float(resolution_deg)))
    n_lat = int(round(180.0 / float(resolution_deg))) + 1
    lon = np.linspace(0.0, 360.0, n_lon, endpoint=False, dtype=float) + 180.0 / n_lon
    lat = np.linspace(-90.0, 90.0, n_lat, dtype=float)
    return lat, lon


def _virtual_temperature(temperature: np.ndarray, specific_humidity: np.ndarray) -> np.ndarray:
    """Approximate virtual temperature from temperature and specific humidity."""
    return np.asarray(temperature, dtype=np.float64) * (
        1.0 + 0.61 * np.asarray(specific_humidity, dtype=np.float64)
    )


def _datetime64_sequence(start_time: str, *, days: Sequence[int]) -> list[np.datetime64]:
    start = np.datetime64(start_time)
    return [start + np.timedelta64(int(day), "D") for day in days]


def _absolute_start_day(start_time: str) -> float:
    timestamp = datetime.fromisoformat(start_time)
    return (
        float(timestamp.timetuple().tm_yday - 1)
        + float(timestamp.hour) / 24.0
        + float(timestamp.minute) / 1440.0
        + float(timestamp.second) / 86400.0
    )


def _create_grid_and_sigma(config: ExperimentConfig):
    if config.grid.grid_type != "cubed_sphere":
        raise NotImplementedError(
            "legoesm_slab v1 currently supports cubed-sphere initialized forecasts only"
        )
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.grids.vertical import create_sigma_coordinate, make_hybrid_levels

    grid = create_cubed_sphere(config.grid.resolution)
    if config.grid.vertical_coord == "hybrid":
        sigma = make_hybrid_levels(
            config.grid.nlev,
            p_top_Pa=config.grid.p_top_Pa,
            stretching=config.grid.stretching,
        )
    else:
        sigma = create_sigma_coordinate(config.grid.nlev)
    return grid, sigma


def _interp_latlon_to_cubedsphere_2d(
    field: np.ndarray,
    *,
    source_lat: np.ndarray,
    source_lon: np.ndarray,
    target_lat_rad: np.ndarray,
    target_lon_rad: np.ndarray,
    k_neighbors: int | None = None,
) -> np.ndarray:
    from scipy.spatial import cKDTree

    source_lat = np.asarray(source_lat, dtype=np.float64)
    source_lon = np.asarray(source_lon, dtype=np.float64)
    values = np.asarray(field, dtype=np.float64)
    lon2d, lat2d = np.meshgrid(np.deg2rad(source_lon), np.deg2rad(source_lat))
    src_xyz = np.stack(
        [
            np.cos(lat2d) * np.cos(lon2d),
            np.cos(lat2d) * np.sin(lon2d),
            np.sin(lat2d),
        ],
        axis=-1,
    ).reshape(-1, 3)
    tgt_lat = np.asarray(target_lat_rad, dtype=np.float64)
    tgt_lon = np.asarray(target_lon_rad, dtype=np.float64)
    tgt_xyz = np.stack(
        [
            np.cos(tgt_lat) * np.cos(tgt_lon),
            np.cos(tgt_lat) * np.sin(tgt_lon),
            np.sin(tgt_lat),
        ],
        axis=-1,
    ).reshape(-1, 3)

    tree = cKDTree(src_xyz)
    requested_neighbors = 16 if k_neighbors is None else int(k_neighbors)
    k_neighbors = max(1, min(requested_neighbors, src_xyz.shape[0]))
    distances, indices = tree.query(tgt_xyz, k=k_neighbors)
    if k_neighbors == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    distances = np.maximum(distances, 1.0e-12)
    weights = 1.0 / distances
    weights /= weights.sum(axis=-1, keepdims=True)

    source_flat = values.reshape(-1)
    remapped = np.sum(source_flat[indices] * weights, axis=-1)
    return remapped.reshape(target_lat_rad.shape).astype(np.float32)


def _fill_nan_nearest_latlon(
    field: xr.DataArray,
    *,
    config: PreparationConfig,
) -> xr.DataArray:
    """Fill NaNs on a lat-lon field with nearest-neighbor support."""
    lat_name = config.latitude_name
    lon_name = config.longitude_name
    filled = field.interpolate_na(dim=lon_name, method="nearest", fill_value="extrapolate")
    filled = filled.interpolate_na(dim=lat_name, method="nearest", fill_value="extrapolate")
    if np.isnan(np.asarray(filled)).any():
        filled = filled.fillna(field.mean(skipna=True))
    return filled


def _interp_latlon_to_cubedsphere_masked_2d(
    field: xr.DataArray,
    *,
    source_lat: np.ndarray,
    source_lon: np.ndarray,
    target_lat_rad: np.ndarray,
    target_lon_rad: np.ndarray,
    config: PreparationConfig,
) -> np.ndarray:
    """Interpolate a partially masked lat-lon field without bleeding NaNs."""
    field_values = np.asarray(field, dtype=np.float32)
    valid = np.isfinite(field_values).astype(np.float32)
    numerator = _interp_latlon_to_cubedsphere_2d(
        np.where(np.isfinite(field_values), field_values, 0.0),
        source_lat=source_lat,
        source_lon=source_lon,
        target_lat_rad=target_lat_rad,
        target_lon_rad=target_lon_rad,
    )
    denominator = _interp_latlon_to_cubedsphere_2d(
        valid,
        source_lat=source_lat,
        source_lon=source_lon,
        target_lat_rad=target_lat_rad,
        target_lon_rad=target_lon_rad,
    )
    fallback = _interp_latlon_to_cubedsphere_2d(
        np.asarray(_fill_nan_nearest_latlon(field, config=config), dtype=np.float32),
        source_lat=source_lat,
        source_lon=source_lon,
        target_lat_rad=target_lat_rad,
        target_lon_rad=target_lon_rad,
    )
    return np.where(denominator > 1.0e-6, numerator / np.maximum(denominator, 1.0e-6), fallback).astype(np.float32)


def _interp_latlon_to_cubedsphere_3d(
    field: np.ndarray,
    *,
    source_lat: np.ndarray,
    source_lon: np.ndarray,
    target_lat_rad: np.ndarray,
    target_lon_rad: np.ndarray,
) -> np.ndarray:
    levels = np.asarray(field, dtype=np.float32).shape[-1]
    slices = [
        _interp_latlon_to_cubedsphere_2d(
            np.asarray(field[..., level_index], dtype=np.float32),
            source_lat=source_lat,
            source_lon=source_lon,
            target_lat_rad=target_lat_rad,
            target_lon_rad=target_lon_rad,
        )
        for level_index in range(levels)
    ]
    return np.stack(slices, axis=-1)


def _blend_cubedsphere_scalar(
    field: np.ndarray,
    *,
    config: PreparationConfig,
) -> np.ndarray:
    """Relax scalar discontinuities across connected cube-face edges."""
    strength = float(config.cubedsphere_edge_blend_strength)
    width = int(config.cubedsphere_edge_blend_width)
    if strength <= 0.0 or width <= 0:
        return np.asarray(field, dtype=np.float32)
    return np.asarray(
        blend_scalar_cube_edges(
            jnp.asarray(field, dtype=jnp.float32),
            strength=strength,
            width=width,
        ),
        dtype=np.float32,
    )


def _blend_cubedsphere_vector(
    u: np.ndarray,
    v: np.ndarray,
    *,
    grid,
    config: PreparationConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Relax vector discontinuities across connected cube-face edges."""
    strength = float(config.cubedsphere_edge_blend_strength)
    width = int(config.cubedsphere_edge_blend_width)
    if strength <= 0.0 or width <= 0 or not hasattr(grid, "angle"):
        return np.asarray(u, dtype=np.float32), np.asarray(v, dtype=np.float32)
    u_blend, v_blend = blend_vector_cube_edges(
        jnp.asarray(u, dtype=jnp.float32),
        jnp.asarray(v, dtype=jnp.float32),
        jnp.cos(jnp.asarray(grid.angle, dtype=jnp.float32)),
        jnp.sin(jnp.asarray(grid.angle, dtype=jnp.float32)),
        strength=strength,
        width=width,
    )
    return np.asarray(u_blend, dtype=np.float32), np.asarray(v_blend, dtype=np.float32)


def _coerce_pressure_levels(level_values: np.ndarray, field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    levels = np.asarray(level_values, dtype=np.float64)
    data = np.asarray(field, dtype=np.float32)
    if np.nanmax(levels) < 2_000.0:
        levels = levels * 100.0
    if levels[0] > levels[-1]:
        levels = levels[::-1]
        data = data[..., ::-1]
    return levels, data


def _mask_subsurface_pressure_levels(
    field_plev: np.ndarray,
    *,
    plev_pa: np.ndarray,
    surface_pressure: np.ndarray,
) -> np.ndarray:
    """Clamp source pressure-level values below the local surface.

    ERA5 pressure-level products can carry extrapolated values at levels below
    the surface over high terrain. Those values can contaminate the coarse
    sigma/hybrid mapping when bracketing near-surface targets. This helper
    replaces all below-surface source levels with the lowest above-ground value
    in each source column, so later interpolation behaves like constant
    extrapolation from the last physically supported level.
    """
    field = np.asarray(field_plev, dtype=np.float32)
    surface_pressure = np.asarray(surface_pressure, dtype=np.float64)
    level_axis = field.ndim - 1
    plev_shape = (1,) * level_axis + (int(np.asarray(plev_pa).size),)
    valid = np.asarray(plev_pa, dtype=np.float64).reshape(plev_shape) <= surface_pressure[..., None]
    last_valid_idx = np.maximum(valid.sum(axis=-1) - 1, 0).astype(np.int64)
    last_valid_val = np.take_along_axis(field, last_valid_idx[..., None], axis=-1)
    return np.where(valid, field, last_valid_val).astype(np.float32)


def _snapshot_to_state(
    snapshot: xr.Dataset,
    *,
    grid,
    sigma,
    config: PreparationConfig,
) -> tuple[HydrostaticState, jnp.ndarray]:
    lat_name = config.latitude_name
    lon_name = config.longitude_name
    level_name = config.level_name
    latitude = np.asarray(snapshot[lat_name], dtype=np.float64)
    longitude = np.asarray(snapshot[lon_name], dtype=np.float64)

    temperature = _find_variable(snapshot, "temperature").transpose(lat_name, lon_name, level_name)
    humidity = _find_variable(snapshot, "specific_humidity").transpose(lat_name, lon_name, level_name)
    zonal = _find_variable(snapshot, "u_component_of_wind").transpose(lat_name, lon_name, level_name)
    meridional = _find_variable(snapshot, "v_component_of_wind").transpose(lat_name, lon_name, level_name)
    surface_pressure = _find_variable(snapshot, "surface_pressure").transpose(lat_name, lon_name)
    surface_geopotential = _find_variable(snapshot, "surface_geopotential").transpose(lat_name, lon_name)

    plev_pa, temperature_ll = _coerce_pressure_levels(temperature[level_name].values, temperature.values)
    _, humidity_ll = _coerce_pressure_levels(humidity[level_name].values, humidity.values)
    _, zonal_ll = _coerce_pressure_levels(zonal[level_name].values, zonal.values)
    _, meridional_ll = _coerce_pressure_levels(meridional[level_name].values, meridional.values)
    if config.mask_subsurface_pressure_levels:
        surface_pressure_ll = np.asarray(surface_pressure.values, dtype=np.float32)
        temperature_ll = _mask_subsurface_pressure_levels(
            temperature_ll,
            plev_pa=plev_pa,
            surface_pressure=surface_pressure_ll,
        )
        humidity_ll = _mask_subsurface_pressure_levels(
            humidity_ll,
            plev_pa=plev_pa,
            surface_pressure=surface_pressure_ll,
        )
        zonal_ll = _mask_subsurface_pressure_levels(
            zonal_ll,
            plev_pa=plev_pa,
            surface_pressure=surface_pressure_ll,
        )
        meridional_ll = _mask_subsurface_pressure_levels(
            meridional_ll,
            plev_pa=plev_pa,
            surface_pressure=surface_pressure_ll,
        )

    p_s_cs = _interp_latlon_to_cubedsphere_2d(
        surface_pressure.values,
        source_lat=latitude,
        source_lon=longitude,
        target_lat_rad=grid.grid_lat,
        target_lon_rad=grid.grid_lon,
    )
    p_s_cs = _blend_cubedsphere_scalar(p_s_cs, config=config)
    phis_cs = _interp_latlon_to_cubedsphere_2d(
        surface_geopotential.values,
        source_lat=latitude,
        source_lon=longitude,
        target_lat_rad=grid.grid_lat,
        target_lon_rad=grid.grid_lon,
    )
    phis_cs = _blend_cubedsphere_scalar(phis_cs, config=config)

    p_s_jax = jnp.asarray(p_s_cs)
    plev_jax = jnp.asarray(plev_pa)
    p_s_source_jax = jnp.asarray(surface_pressure.values)
    if config.vertical_first_remap:
        if hasattr(sigma, "A_full") and hasattr(sigma, "B_full"):
            T_latlon = interp_pressure_to_hybrid(
                jnp.asarray(temperature_ll),
                plev_jax,
                p_s_source_jax,
                jnp.asarray(sigma.A_full),
                jnp.asarray(sigma.B_full),
                p_ref=float(getattr(sigma, "p_ref", 1.0e5)),
            )
            u_latlon = interp_pressure_to_hybrid(
                jnp.asarray(zonal_ll),
                plev_jax,
                p_s_source_jax,
                jnp.asarray(sigma.A_full),
                jnp.asarray(sigma.B_full),
                p_ref=float(getattr(sigma, "p_ref", 1.0e5)),
            )
            v_latlon = interp_pressure_to_hybrid(
                jnp.asarray(meridional_ll),
                plev_jax,
                p_s_source_jax,
                jnp.asarray(sigma.A_full),
                jnp.asarray(sigma.B_full),
                p_ref=float(getattr(sigma, "p_ref", 1.0e5)),
            )
            q_latlon = interp_pressure_to_hybrid(
                jnp.asarray(humidity_ll),
                plev_jax,
                p_s_source_jax,
                jnp.asarray(sigma.A_full),
                jnp.asarray(sigma.B_full),
                p_ref=float(getattr(sigma, "p_ref", 1.0e5)),
            )
        else:
            sigma_full = jnp.asarray(sigma.sigma_full)
            T_latlon = interp_pressure_to_sigma(jnp.asarray(temperature_ll), plev_jax, p_s_source_jax, sigma_full)
            u_latlon = interp_pressure_to_sigma(jnp.asarray(zonal_ll), plev_jax, p_s_source_jax, sigma_full)
            v_latlon = interp_pressure_to_sigma(jnp.asarray(meridional_ll), plev_jax, p_s_source_jax, sigma_full)
            q_latlon = interp_pressure_to_sigma(jnp.asarray(humidity_ll), plev_jax, p_s_source_jax, sigma_full)

        T_model = jnp.asarray(
            _blend_cubedsphere_scalar(
                _interp_latlon_to_cubedsphere_3d(
                    np.asarray(T_latlon, dtype=np.float32),
                    source_lat=latitude,
                    source_lon=longitude,
                    target_lat_rad=grid.grid_lat,
                    target_lon_rad=grid.grid_lon,
                ),
                config=config,
            ),
        )
        q_model = jnp.asarray(
            _blend_cubedsphere_scalar(
                _interp_latlon_to_cubedsphere_3d(
                    np.asarray(q_latlon, dtype=np.float32),
                    source_lat=latitude,
                    source_lon=longitude,
                    target_lat_rad=grid.grid_lat,
                    target_lon_rad=grid.grid_lon,
                ),
                config=config,
            ),
        )
        u_cs = _interp_latlon_to_cubedsphere_3d(
            np.asarray(u_latlon, dtype=np.float32),
            source_lat=latitude,
            source_lon=longitude,
            target_lat_rad=grid.grid_lat,
            target_lon_rad=grid.grid_lon,
        )
        v_cs = _interp_latlon_to_cubedsphere_3d(
            np.asarray(v_latlon, dtype=np.float32),
            source_lat=latitude,
            source_lon=longitude,
            target_lat_rad=grid.grid_lat,
            target_lon_rad=grid.grid_lon,
        )
        if hasattr(grid, "angle"):
            u_cs_jax, v_cs_jax = rotate_winds_geo_to_grid(
                jnp.asarray(u_cs),
                jnp.asarray(v_cs),
                jnp.asarray(grid.angle)[..., None],
            )
            u_cs = np.asarray(u_cs_jax, dtype=np.float32)
            v_cs = np.asarray(v_cs_jax, dtype=np.float32)
        u_model_np, v_model_np = _blend_cubedsphere_vector(u_cs, v_cs, grid=grid, config=config)
        u_model = jnp.asarray(u_model_np)
        v_model = jnp.asarray(v_model_np)
    else:
        T_cs = _interp_latlon_to_cubedsphere_3d(
            temperature_ll,
            source_lat=latitude,
            source_lon=longitude,
            target_lat_rad=grid.grid_lat,
            target_lon_rad=grid.grid_lon,
        )
        T_cs = _blend_cubedsphere_scalar(T_cs, config=config)
        q_cs = _interp_latlon_to_cubedsphere_3d(
            humidity_ll,
            source_lat=latitude,
            source_lon=longitude,
            target_lat_rad=grid.grid_lat,
            target_lon_rad=grid.grid_lon,
        )
        q_cs = _blend_cubedsphere_scalar(q_cs, config=config)
        u_cs = _interp_latlon_to_cubedsphere_3d(
            zonal_ll,
            source_lat=latitude,
            source_lon=longitude,
            target_lat_rad=grid.grid_lat,
            target_lon_rad=grid.grid_lon,
        )
        v_cs = _interp_latlon_to_cubedsphere_3d(
            meridional_ll,
            source_lat=latitude,
            source_lon=longitude,
            target_lat_rad=grid.grid_lat,
            target_lon_rad=grid.grid_lon,
        )
        if hasattr(grid, "angle"):
            u_cs_jax, v_cs_jax = rotate_winds_geo_to_grid(
                jnp.asarray(u_cs),
                jnp.asarray(v_cs),
                jnp.asarray(grid.angle)[..., None],
            )
            u_cs = np.asarray(u_cs_jax, dtype=np.float32)
            v_cs = np.asarray(v_cs_jax, dtype=np.float32)
        u_cs, v_cs = _blend_cubedsphere_vector(u_cs, v_cs, grid=grid, config=config)
        if hasattr(sigma, "A_full") and hasattr(sigma, "B_full"):
            T_model = interp_pressure_to_hybrid(
                jnp.asarray(T_cs),
                plev_jax,
                p_s_jax,
                jnp.asarray(sigma.A_full),
                jnp.asarray(sigma.B_full),
                p_ref=float(getattr(sigma, "p_ref", 1.0e5)),
            )
            u_model = interp_pressure_to_hybrid(
                jnp.asarray(u_cs),
                plev_jax,
                p_s_jax,
                jnp.asarray(sigma.A_full),
                jnp.asarray(sigma.B_full),
                p_ref=float(getattr(sigma, "p_ref", 1.0e5)),
            )
            v_model = interp_pressure_to_hybrid(
                jnp.asarray(v_cs),
                plev_jax,
                p_s_jax,
                jnp.asarray(sigma.A_full),
                jnp.asarray(sigma.B_full),
                p_ref=float(getattr(sigma, "p_ref", 1.0e5)),
            )
            q_model = interp_pressure_to_hybrid(
                jnp.asarray(q_cs),
                plev_jax,
                p_s_jax,
                jnp.asarray(sigma.A_full),
                jnp.asarray(sigma.B_full),
                p_ref=float(getattr(sigma, "p_ref", 1.0e5)),
            )
        else:
            sigma_full = jnp.asarray(sigma.sigma_full)
            T_model = interp_pressure_to_sigma(jnp.asarray(T_cs), plev_jax, p_s_jax, sigma_full)
            u_model = interp_pressure_to_sigma(jnp.asarray(u_cs), plev_jax, p_s_jax, sigma_full)
            v_model = interp_pressure_to_sigma(jnp.asarray(v_cs), plev_jax, p_s_jax, sigma_full)
            q_model = interp_pressure_to_sigma(jnp.asarray(q_cs), plev_jax, p_s_jax, sigma_full)

    dims_3d = ("face", "x", "y", "level")
    dims_2d = ("face", "x", "y")
    state = HydrostaticState(
        u=Field(data=u_model, name="u", dims=dims_3d, units="m/s"),
        v=Field(data=v_model, name="v", dims=dims_3d, units="m/s"),
        T=Field(data=T_model, name="T", dims=dims_3d, units="K"),
        p_s=Field(data=p_s_jax, name="p_s", dims=dims_2d, units="Pa"),
        phis=Field(data=jnp.asarray(phis_cs), name="phis", dims=dims_2d, units="m^2/s^2"),
    )
    return state, jnp.maximum(q_model, 0.0)


def _project_state_to_cdgrid_subspace(
    state: HydrostaticState,
    *,
    grid,
    experiment_config: ExperimentConfig,
) -> HydrostaticState:
    """Project initialized winds into the cubed-sphere CD-grid representable subspace.

    The FV3-style cubed-sphere dycore prognoses winds on D-grid corners and the
    legacy HydrostaticState wrapper reconstructs those corner winds from
    cell-centre ``u``/``v`` on entry.  Directly remapped ERA5 cell-centre winds can
    therefore experience a large one-step adjustment that is mostly this
    representation mismatch rather than physical evolution.

    For initialized S2S cases on the cubed-sphere CD-grid, pre-project the
    restart winds through ``HydrostaticState -> FV3HydrostaticState ->
    HydrostaticState`` so the saved restart already lies close to the dycore's
    own representable cell-centre subspace.
    """
    if str(getattr(experiment_config.grid, "grid_type", "")) != "cubed_sphere":
        return state
    if str(getattr(experiment_config.dycore, "discretization", "")) != "cdgrid":
        return state
    cdgrid = create_cubed_sphere_cdgrid(grid)
    fv3_state = hydrostatic_to_fv3(state, cdgrid)
    return fv3_to_hydrostatic(fv3_state, cdgrid)


def _balance_initialized_cdgrid_state(
    state: HydrostaticState,
    q_v: jax.Array,
    *,
    grid,
    sigma,
    experiment_config: ExperimentConfig,
    n_steps: int,
) -> tuple[HydrostaticState, jax.Array]:
    """Dry-balance an initialized cubed-sphere CD-grid state for a few steps.

    ERA5-remapped initialized states on the coarse cubed sphere can still
    carry substantial discrete imbalance even after the wind-representation
    projection. A short dycore-only adjustment phase damps the startup shock
    before the forecast proper begins.
    """
    if int(n_steps) <= 0:
        return state, q_v
    if str(getattr(experiment_config.grid, "grid_type", "")) != "cubed_sphere":
        return state, q_v
    if str(getattr(experiment_config.dycore, "discretization", "")) != "cdgrid":
        return state, q_v

    from legoesm.driver.component_factory import create_atmosphere_dycore

    tracer_field = state.T.replace(data=jnp.asarray(q_v), name="q_v", units="kg/kg")
    balanced = state._replace(tracers={"q_v": tracer_field})
    model = create_atmosphere_dycore(experiment_config, grid, sigma)
    dt = float(experiment_config.dycore.dt)
    for _ in range(int(n_steps)):
        balanced = model.step(balanced, dt)

    q_v_balanced = balanced.tracers.get("q_v") if balanced.tracers is not None else None
    q_v_data = q_v_balanced.data if q_v_balanced is not None else q_v
    return balanced._replace(tracers=None), jnp.maximum(jnp.asarray(q_v_data), 0.0)


def _interp_pressure_field_to_model_levels(
    field_plev: xr.DataArray,
    *,
    grid,
    sigma,
    surface_pressure: jax.Array,
    config: PreparationConfig,
) -> jax.Array:
    lat_name = config.latitude_name
    lon_name = config.longitude_name
    level_name = config.level_name
    latitude = np.asarray(field_plev[lat_name], dtype=np.float64)
    longitude = np.asarray(field_plev[lon_name], dtype=np.float64)
    plev_pa, field_ll = _coerce_pressure_levels(field_plev[level_name].values, field_plev.values)
    field_cs = _interp_latlon_to_cubedsphere_3d(
        field_ll,
        source_lat=latitude,
        source_lon=longitude,
        target_lat_rad=grid.grid_lat,
        target_lon_rad=grid.grid_lon,
    )
    field_cs = _blend_cubedsphere_scalar(field_cs, config=config)
    plev_jax = jnp.asarray(plev_pa)
    if hasattr(sigma, "A_full") and hasattr(sigma, "B_full"):
        return interp_pressure_to_hybrid(
            jnp.asarray(field_cs),
            plev_jax,
            surface_pressure,
            jnp.asarray(sigma.A_full),
            jnp.asarray(sigma.B_full),
            p_ref=float(getattr(sigma, "p_ref", 1.0e5)),
        )
    return interp_pressure_to_sigma(
        jnp.asarray(field_cs),
        plev_jax,
        surface_pressure,
        jnp.asarray(sigma.sigma_full),
    )


def _compute_model_full_pressure(
    surface_pressure: jax.Array,
    *,
    sigma,
) -> jax.Array:
    if hasattr(sigma, "A_full") and hasattr(sigma, "B_full"):
        return compute_model_pressures_hybrid(
            surface_pressure,
            jnp.asarray(sigma.A_full),
            jnp.asarray(sigma.B_full),
            p_ref=float(getattr(sigma, "p_ref", 1.0e5)),
        )
    return compute_model_pressures_sigma(surface_pressure, jnp.asarray(sigma.sigma_full))


def _validate_monotonic_model_interfaces(
    surface_pressure: jax.Array,
    *,
    sigma,
) -> None:
    """Fail fast if the chosen vertical coordinate yields non-monotonic interfaces."""
    p_half = np.asarray(sigma.pressure_at_half(surface_pressure), dtype=np.float64)
    dp = np.diff(p_half, axis=-1)
    bad = dp <= 0.0
    if not np.any(bad):
        return
    bad_columns = np.any(bad, axis=-1)
    bad_count = int(np.sum(bad_columns))
    p_s = np.asarray(surface_pressure, dtype=np.float64)
    bad_ps = p_s[bad_columns]
    min_dp = float(np.min(dp))
    raise ValueError(
        "Prepared initialized case has non-monotonic model interface pressures "
        f"for {bad_count} columns (min dp={min_dp:.3f} Pa; "
        f"bad-column surface pressure range={float(np.min(bad_ps)):.1f}-{float(np.max(bad_ps)):.1f} Pa). "
        "This vertical coordinate/setup is not safe for initialized forecasts at the chosen resolution. "
        "Use `--vertical-coord sigma` or increase `--nlev`."
    )


def _initial_condensate_seed(
    snapshot: xr.Dataset,
    *,
    state: HydrostaticState,
    q_v: jax.Array,
    grid,
    sigma,
    config: PreparationConfig,
) -> tuple[jax.Array, jax.Array]:
    """Seed restart hydrometeors from ERA5 condensate, falling back to RH excess."""
    cloud_liquid = _find_optional_variable(snapshot, "specific_cloud_liquid_water_content")
    if cloud_liquid is not None:
        q_c = _interp_pressure_field_to_model_levels(
            cloud_liquid.transpose(
                config.latitude_name,
                config.longitude_name,
                config.level_name,
            ),
            grid=grid,
            sigma=sigma,
            surface_pressure=state.p_s.data,
            config=config,
        )
        q_c = jnp.minimum(jnp.maximum(q_c, 0.0), jnp.maximum(q_v, 0.0))
    else:
        p_full = _compute_model_full_pressure(state.p_s.data, sigma=sigma)
        q_sat = saturation_specific_humidity(state.T.data, p_full)
        q_c = jnp.clip(q_v - q_sat, 0.0, None)
    q_r = jnp.zeros_like(q_c)
    return q_c, q_r


def _interp_regular_field(
    field: xr.DataArray,
    *,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
    config: PreparationConfig,
) -> xr.DataArray:
    lat_name = config.latitude_name
    lon_name = config.longitude_name
    linear = field.interp(
        {lat_name: target_lat, lon_name: target_lon},
        method="linear",
    )
    nearest = field.interp(
        {lat_name: target_lat, lon_name: target_lon},
        method="nearest",
    )
    return linear.fillna(nearest)


def _interp_mask_field(
    field: xr.DataArray,
    *,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
    config: PreparationConfig,
) -> xr.DataArray:
    """Interpolate categorical mask-like fields with nearest-neighbor only."""
    lat_name = config.latitude_name
    lon_name = config.longitude_name
    return field.interp(
        {lat_name: target_lat, lon_name: target_lon},
        method="nearest",
    )


def _select_pressure_levels(
    field: xr.DataArray,
    pressure_levels: Sequence[int],
    *,
    config: PreparationConfig,
) -> xr.DataArray:
    level_name = config.level_name
    level_values = np.asarray(field[level_name], dtype=float)
    selector = np.asarray(pressure_levels, dtype=float)
    if np.nanmax(level_values) > 2_000.0:
        selector = selector * 100.0
    selected = field.sel({level_name: selector.tolist()}, method="nearest")
    if np.nanmax(level_values) > 2_000.0:
        selected = selected.assign_coords(
            {level_name: np.asarray(selected[level_name], dtype=float) / 100.0}
        )
    return selected


def _build_truth_dataset(
    dataset: xr.Dataset,
    *,
    start_time: str,
    forecast_days: int,
    grid,
    config: PreparationConfig,
) -> xr.Dataset:
    target_lat, target_lon = _target_grid_coords(config.evaluation_resolution_deg)
    times = _datetime64_sequence(start_time, days=tuple(range(1, int(forecast_days) + 1)))
    truth = dataset.sel({config.time_name: times})
    source_lat = np.asarray(dataset[config.latitude_name], dtype=np.float64)
    source_lon = np.asarray(dataset[config.longitude_name], dtype=np.float64)
    weights = get_cubedsphere_to_latlon_weights(
        grid.n,
        n_lon=target_lon.size,
        n_lat=target_lat.size,
    )

    def _roundtrip_scalar_field(
        field: xr.DataArray,
        *,
        masked: bool = False,
        k_neighbors: int | None = None,
        preserve_mask: bool = False,
    ) -> xr.DataArray:
        lat_name = config.latitude_name
        lon_name = config.longitude_name
        ordered_dims = [dim for dim in field.dims if dim not in (lat_name, lon_name)] + [lat_name, lon_name]
        transposed = field.transpose(*ordered_dims)
        values = np.asarray(transposed, dtype=np.float32)
        flat_values = values.reshape((-1,) + values.shape[-2:])
        out = np.empty((flat_values.shape[0], target_lat.size, target_lon.size), dtype=np.float32)
        for index, slice_2d in enumerate(flat_values):
            valid_ll = None
            if masked:
                source_slice = xr.DataArray(
                    slice_2d,
                    coords={lat_name: source_lat, lon_name: source_lon},
                    dims=(lat_name, lon_name),
                )
                cs_field = _interp_latlon_to_cubedsphere_masked_2d(
                    source_slice,
                    source_lat=source_lat,
                    source_lon=source_lon,
                    target_lat_rad=grid.grid_lat,
                    target_lon_rad=grid.grid_lon,
                    config=config,
                )
                if preserve_mask:
                    valid_cs = _interp_latlon_to_cubedsphere_2d(
                        np.isfinite(slice_2d).astype(np.float32),
                        source_lat=source_lat,
                        source_lon=source_lon,
                        target_lat_rad=grid.grid_lat,
                        target_lon_rad=grid.grid_lon,
                    )
                    valid_ll = _apply_cubedsphere_to_latlon_masked(
                        _blend_cubedsphere_scalar(valid_cs, config=config),
                        weights,
                    ).astype(np.float32)
            else:
                cs_field = _interp_latlon_to_cubedsphere_2d(
                    slice_2d,
                    source_lat=source_lat,
                    source_lon=source_lon,
                    target_lat_rad=grid.grid_lat,
                    target_lon_rad=grid.grid_lon,
                    k_neighbors=k_neighbors,
                )
            cs_field = _blend_cubedsphere_scalar(cs_field, config=config)
            out_slice = _apply_cubedsphere_to_latlon_masked(cs_field, weights).astype(np.float32)
            if valid_ll is not None:
                out_slice = np.where(valid_ll > 0.5, out_slice, np.nan).astype(np.float32)
            out[index] = out_slice
        out = out.reshape(values.shape[:-2] + (target_lat.size, target_lon.size))
        return xr.DataArray(
            out,
            coords={
                **{dim: transposed.coords[dim] for dim in transposed.dims if dim not in (lat_name, lon_name)},
                lat_name: target_lat,
                lon_name: target_lon,
            },
            dims=transposed.dims,
        ).transpose(*field.dims)

    def _roundtrip_vector_fields(
        u_field: xr.DataArray,
        v_field: xr.DataArray,
    ) -> tuple[xr.DataArray, xr.DataArray]:
        lat_name = config.latitude_name
        lon_name = config.longitude_name
        ordered_dims = [dim for dim in u_field.dims if dim not in (lat_name, lon_name)] + [lat_name, lon_name]
        u_transposed = u_field.transpose(*ordered_dims)
        v_transposed = v_field.transpose(*ordered_dims)
        u_values = np.asarray(u_transposed, dtype=np.float32)
        v_values = np.asarray(v_transposed, dtype=np.float32)
        flat_u = u_values.reshape((-1,) + u_values.shape[-2:])
        flat_v = v_values.reshape((-1,) + v_values.shape[-2:])
        out_u = np.empty((flat_u.shape[0], target_lat.size, target_lon.size), dtype=np.float32)
        out_v = np.empty_like(out_u)
        for index, (u_slice, v_slice) in enumerate(zip(flat_u, flat_v, strict=True)):
            u_geo_cs = _interp_latlon_to_cubedsphere_2d(
                u_slice,
                source_lat=source_lat,
                source_lon=source_lon,
                target_lat_rad=grid.grid_lat,
                target_lon_rad=grid.grid_lon,
            )
            v_geo_cs = _interp_latlon_to_cubedsphere_2d(
                v_slice,
                source_lat=source_lat,
                source_lon=source_lon,
                target_lat_rad=grid.grid_lat,
                target_lon_rad=grid.grid_lon,
            )
            u_grid_cs, v_grid_cs = rotate_winds_geo_to_grid(
                jnp.asarray(u_geo_cs, dtype=jnp.float32),
                jnp.asarray(v_geo_cs, dtype=jnp.float32),
                jnp.asarray(grid.angle, dtype=jnp.float32),
            )
            u_grid_np, v_grid_np = _blend_cubedsphere_vector(
                np.asarray(u_grid_cs, dtype=np.float32),
                np.asarray(v_grid_cs, dtype=np.float32),
                grid=grid,
                config=config,
            )
            u_geo_back, v_geo_back = rotate_winds_grid_to_geo(
                jnp.asarray(u_grid_np, dtype=jnp.float32),
                jnp.asarray(v_grid_np, dtype=jnp.float32),
                jnp.asarray(grid.angle, dtype=jnp.float32),
            )
            out_u[index] = _apply_cubedsphere_to_latlon_masked(
                np.asarray(u_geo_back, dtype=np.float32),
                weights,
            ).astype(np.float32)
            out_v[index] = _apply_cubedsphere_to_latlon_masked(
                np.asarray(v_geo_back, dtype=np.float32),
                weights,
            ).astype(np.float32)
        coords = {
            **{dim: u_transposed.coords[dim] for dim in u_transposed.dims if dim not in (lat_name, lon_name)},
            lat_name: target_lat,
            lon_name: target_lon,
        }
        out_shape = u_values.shape[:-2] + (target_lat.size, target_lon.size)
        return (
            xr.DataArray(out_u.reshape(out_shape), coords=coords, dims=u_transposed.dims).transpose(*u_field.dims),
            xr.DataArray(out_v.reshape(out_shape), coords=coords, dims=v_transposed.dims).transpose(*v_field.dims),
        )

    fields: dict[str, xr.DataArray] = {}
    for key in ("temperature", "specific_humidity", "geopotential"):
        field = _find_variable(truth, key)
        field = _select_pressure_levels(field, config.pressure_levels, config=config)
        fields[key] = _roundtrip_scalar_field(field)

    u_field = _select_pressure_levels(
        _find_variable(truth, "u_component_of_wind"),
        config.pressure_levels,
        config=config,
    )
    v_field = _select_pressure_levels(
        _find_variable(truth, "v_component_of_wind"),
        config.pressure_levels,
        config=config,
    )
    fields["u_component_of_wind"], fields["v_component_of_wind"] = _roundtrip_vector_fields(
        u_field,
        v_field,
    )

    for key in ("sea_surface_temperature", "sea_ice_cover"):
        field = _find_variable(truth, key)
        fields[key] = _roundtrip_scalar_field(
            field,
            masked=True,
            preserve_mask=True,
        )

    fields["surface_pressure"] = _roundtrip_scalar_field(
        _find_variable(truth, "surface_pressure"),
    )

    if any(candidate in truth for candidate in _VAR_CANDIDATES["total_precipitation"]):
        fields["total_precipitation"] = _roundtrip_scalar_field(
            _find_variable(truth, "total_precipitation"),
        )

    result = xr.Dataset(fields).rename(
        {
            config.latitude_name: "latitude",
            config.longitude_name: "longitude",
            config.level_name: "level",
            config.time_name: "time",
        }
    )
    return result.assign_coords(
        {"lead_day": ("time", np.arange(1, int(forecast_days) + 1, dtype=int))}
    )


def _build_initial_dataset(
    dataset: xr.Dataset,
    *,
    start_time: str,
    config: PreparationConfig,
) -> xr.Dataset:
    target_lat, target_lon = _target_grid_coords(config.evaluation_resolution_deg)
    snapshot = dataset.sel({config.time_name: np.datetime64(start_time)})

    fields: dict[str, xr.DataArray] = {}
    for key in (
        "temperature",
        "specific_humidity",
        "u_component_of_wind",
        "v_component_of_wind",
        "geopotential",
    ):
        field = _find_variable(snapshot, key)
        field = _select_pressure_levels(field, config.pressure_levels, config=config)
        fields[key] = _interp_regular_field(
            field,
            target_lat=target_lat,
            target_lon=target_lon,
            config=config,
        )

    for key in ("sea_surface_temperature", "sea_ice_cover"):
        field = _find_variable(snapshot, key)
        fields[key] = _interp_regular_field(
            field,
            target_lat=target_lat,
            target_lon=target_lon,
            config=config,
        )

    fields["surface_pressure"] = _interp_regular_field(
        _find_variable(snapshot, "surface_pressure"),
        target_lat=target_lat,
        target_lon=target_lon,
        config=config,
    )
    if any(candidate in snapshot for candidate in _VAR_CANDIDATES["land_surface_temperature"]):
        fields["land_surface_temperature"] = _interp_regular_field(
            _find_variable(snapshot, "land_surface_temperature"),
            target_lat=target_lat,
            target_lon=target_lon,
            config=config,
        )
    if any(candidate in snapshot for candidate in _VAR_CANDIDATES["land_sea_mask"]):
        fields["land_sea_mask"] = _interp_regular_field(
            _find_variable(snapshot, "land_sea_mask"),
            target_lat=target_lat,
            target_lon=target_lon,
            config=config,
        )

    return xr.Dataset(fields).rename(
        {
            config.latitude_name: "latitude",
            config.longitude_name: "longitude",
            config.level_name: "level",
        }
    ).expand_dims(time=[np.datetime64(start_time)])


def _interpolate_model_field_to_pressure_levels(
    field_model: np.ndarray,
    pressure_model: np.ndarray,
    pressure_levels: Sequence[int],
) -> np.ndarray:
    """Interpolate one model-level field to requested pressure levels in hPa."""
    p_target = np.asarray(pressure_levels, dtype=np.float64) * 100.0
    field = np.asarray(field_model, dtype=np.float64)
    pressure = np.asarray(pressure_model, dtype=np.float64)
    flat_field = field.reshape(-1, field.shape[-1])
    flat_pressure = pressure.reshape(-1, pressure.shape[-1])
    output = np.full((flat_field.shape[0], len(p_target)), np.nan, dtype=np.float64)

    for column_index in range(flat_field.shape[0]):
        source_pressure = np.maximum(flat_pressure[column_index], 1.0e-10)
        source_values = flat_field[column_index]
        order = np.argsort(source_pressure)
        source_pressure = source_pressure[order]
        source_values = source_values[order]
        log_source_pressure = np.log(source_pressure)
        for target_index, target_pressure in enumerate(p_target):
            if target_pressure < source_pressure[0] or target_pressure > source_pressure[-1]:
                continue
            idx_hi = int(np.searchsorted(source_pressure, target_pressure, side="left"))
            idx_hi = min(max(idx_hi, 1), source_pressure.size - 1)
            idx_lo = idx_hi - 1
            p_lo = log_source_pressure[idx_lo]
            p_hi = log_source_pressure[idx_hi]
            denom = p_hi - p_lo
            alpha = 0.0 if denom == 0.0 else float(np.clip((np.log(target_pressure) - p_lo) / denom, 0.0, 1.0))
            output[column_index, target_index] = (
                source_values[idx_lo] + alpha * (source_values[idx_hi] - source_values[idx_lo])
            )
    return output.reshape(field.shape[:-1] + (len(p_target),)).astype(np.float32)


def _apply_cubedsphere_to_latlon_masked(
    field_faces: np.ndarray,
    weights,
) -> np.ndarray:
    """Regrid one 2-D cubed-sphere field without expanding NaN masks."""
    n = weights.n
    field = np.asarray(field_faces, dtype=np.float64).reshape(6, n, n)
    valid = np.isfinite(field).astype(np.float64)
    filled = np.where(np.isfinite(field), field, 0.0)

    padded_value = _pad_field_for_regrid(filled, n)
    padded_valid = _pad_field_for_regrid(valid, n)

    i1 = weights.i0 + 1
    j1 = weights.j0 + 1
    c00 = (1.0 - weights.wi) * (1.0 - weights.wj)
    c10 = weights.wi * (1.0 - weights.wj)
    c01 = (1.0 - weights.wi) * weights.wj
    c11 = weights.wi * weights.wj

    v00 = padded_value[weights.face, weights.i0, weights.j0]
    v10 = padded_value[weights.face, i1, weights.j0]
    v01 = padded_value[weights.face, weights.i0, j1]
    v11 = padded_value[weights.face, i1, j1]

    m00 = padded_valid[weights.face, weights.i0, weights.j0]
    m10 = padded_valid[weights.face, i1, weights.j0]
    m01 = padded_valid[weights.face, weights.i0, j1]
    m11 = padded_valid[weights.face, i1, j1]

    numerator = v00 * c00 + v10 * c10 + v01 * c01 + v11 * c11
    denominator = m00 * c00 + m10 * c10 + m01 * c01 + m11 * c11
    result = np.full_like(numerator, np.nan, dtype=np.float64)
    valid_target = denominator > 0.0
    result[valid_target] = numerator[valid_target] / denominator[valid_target]
    return result.reshape(weights.n_lat, weights.n_lon)


def _apply_cubedsphere_to_latlon_3d_masked(
    field_faces: np.ndarray,
    weights,
) -> np.ndarray:
    """Regrid a 3-D cubed-sphere field without spreading NaNs between targets."""
    arr = np.asarray(field_faces, dtype=np.float64)
    n = weights.n
    if arr.ndim == 2:
        nlev = arr.shape[-1]
        arr = arr.reshape(6, n, n, nlev)
    elif arr.ndim == 4:
        nlev = arr.shape[-1]
    else:
        raise ValueError(f"Expected 2-D or 4-D input, got shape {arr.shape}")

    out = np.empty((weights.n_lat, weights.n_lon, nlev), dtype=np.float64)
    for k in range(nlev):
        out[..., k] = _apply_cubedsphere_to_latlon_masked(arr[..., k], weights)
    return out


def _build_initial_dataset_from_state(
    state: HydrostaticState,
    q_v: jax.Array,
    *,
    grid,
    sigma,
    model_surface_forcing: xr.Dataset,
    start_time: str,
    config: PreparationConfig,
) -> xr.Dataset:
    """Export the actual initialized model state and day-0 forcing to eval lat-lon."""
    target_lat, target_lon = _target_grid_coords(config.evaluation_resolution_deg)
    weights = get_cubedsphere_to_latlon_weights(
        grid.n,
        n_lon=target_lon.size,
        n_lat=target_lat.size,
    )

    p_s = np.asarray(state.p_s.data, dtype=np.float32)
    T_virtual = _virtual_temperature(state.T.data, q_v)
    if hasattr(sigma, "A_full") and hasattr(sigma, "B_full"):
        pressure_model = compute_model_pressures_hybrid(
            jnp.asarray(p_s),
            jnp.asarray(sigma.A_full),
            jnp.asarray(sigma.B_full),
            p_ref=float(getattr(sigma, "p_ref", 1.0e5)),
        )
        geopotential_model = compute_geopotential_hybrid(
            T_virtual,
            state.p_s.data,
            sigma,
            state.phis.data,
        )
    else:
        pressure_model = compute_model_pressures_sigma(
            jnp.asarray(p_s),
            jnp.asarray(sigma.sigma_full),
        )
        geopotential_model = compute_geopotential(
            T_virtual,
            state.p_s.data,
            sigma,
            state.phis.data,
        )

    u_geo, v_geo = rotate_winds_grid_to_geo(
        jnp.asarray(state.u.data),
        jnp.asarray(state.v.data),
        jnp.asarray(grid.angle)[..., None],
    )

    fields = {
        "temperature": _apply_cubedsphere_to_latlon_3d_masked(
            _interpolate_model_field_to_pressure_levels(
                np.asarray(state.T.data, dtype=np.float32),
                np.asarray(pressure_model, dtype=np.float32),
                config.pressure_levels,
            ),
            weights,
        ).astype(np.float32),
        "specific_humidity": _apply_cubedsphere_to_latlon_3d_masked(
            _interpolate_model_field_to_pressure_levels(
                np.asarray(q_v, dtype=np.float32),
                np.asarray(pressure_model, dtype=np.float32),
                config.pressure_levels,
            ),
            weights,
        ).astype(np.float32),
        "u_component_of_wind": _apply_cubedsphere_to_latlon_3d_masked(
            _interpolate_model_field_to_pressure_levels(
                np.asarray(u_geo, dtype=np.float32),
                np.asarray(pressure_model, dtype=np.float32),
                config.pressure_levels,
            ),
            weights,
        ).astype(np.float32),
        "v_component_of_wind": _apply_cubedsphere_to_latlon_3d_masked(
            _interpolate_model_field_to_pressure_levels(
                np.asarray(v_geo, dtype=np.float32),
                np.asarray(pressure_model, dtype=np.float32),
                config.pressure_levels,
            ),
            weights,
        ).astype(np.float32),
        "geopotential": _apply_cubedsphere_to_latlon_3d_masked(
            _interpolate_model_field_to_pressure_levels(
                np.asarray(geopotential_model, dtype=np.float32),
                np.asarray(pressure_model, dtype=np.float32),
                config.pressure_levels,
            ),
            weights,
        ).astype(np.float32),
        "surface_pressure": apply_cubedsphere_to_latlon(
            np.asarray(state.p_s.data, dtype=np.float32),
            weights,
        ).astype(np.float32),
    }

    day0 = model_surface_forcing.isel(time=0)
    fields["sea_surface_temperature"] = _apply_cubedsphere_to_latlon_masked(
        np.where(
            np.asarray(day0["land_fraction"], dtype=np.float32) < 0.5,
            np.asarray(day0["fixed_ocean_sea_surface_temperature"], dtype=np.float32),
            np.nan,
        ),
        weights,
    ).astype(np.float32)
    fields["sea_ice_cover"] = _apply_cubedsphere_to_latlon_masked(
        np.where(
            np.asarray(day0["land_fraction"], dtype=np.float32) < 0.5,
            np.asarray(day0["sea_ice_cover"], dtype=np.float32),
            np.nan,
        ),
        weights,
    ).astype(np.float32)
    fields["land_surface_temperature"] = _apply_cubedsphere_to_latlon_masked(
        np.where(
            np.asarray(day0["land_fraction"], dtype=np.float32) >= 0.5,
            np.asarray(day0["land_surface_temperature"], dtype=np.float32),
            np.nan,
        ),
        weights,
    ).astype(np.float32)
    fields["land_sea_mask"] = apply_cubedsphere_to_latlon(
        np.asarray(day0["land_fraction"], dtype=np.float32),
        weights,
    ).astype(np.float32)

    return xr.Dataset(
        {
            "temperature": xr.DataArray(
                fields["temperature"],
                coords={"latitude": target_lat, "longitude": target_lon, "level": np.asarray(config.pressure_levels, dtype=np.int32)},
                dims=("latitude", "longitude", "level"),
            ).transpose("level", "latitude", "longitude"),
            "specific_humidity": xr.DataArray(
                fields["specific_humidity"],
                coords={"latitude": target_lat, "longitude": target_lon, "level": np.asarray(config.pressure_levels, dtype=np.int32)},
                dims=("latitude", "longitude", "level"),
            ).transpose("level", "latitude", "longitude"),
            "u_component_of_wind": xr.DataArray(
                fields["u_component_of_wind"],
                coords={"latitude": target_lat, "longitude": target_lon, "level": np.asarray(config.pressure_levels, dtype=np.int32)},
                dims=("latitude", "longitude", "level"),
            ).transpose("level", "latitude", "longitude"),
            "v_component_of_wind": xr.DataArray(
                fields["v_component_of_wind"],
                coords={"latitude": target_lat, "longitude": target_lon, "level": np.asarray(config.pressure_levels, dtype=np.int32)},
                dims=("latitude", "longitude", "level"),
            ).transpose("level", "latitude", "longitude"),
            "geopotential": xr.DataArray(
                fields["geopotential"],
                coords={"latitude": target_lat, "longitude": target_lon, "level": np.asarray(config.pressure_levels, dtype=np.int32)},
                dims=("latitude", "longitude", "level"),
            ).transpose("level", "latitude", "longitude"),
            "surface_pressure": xr.DataArray(
                fields["surface_pressure"],
                coords={"latitude": target_lat, "longitude": target_lon},
                dims=("latitude", "longitude"),
            ),
            "sea_surface_temperature": xr.DataArray(
                fields["sea_surface_temperature"],
                coords={"latitude": target_lat, "longitude": target_lon},
                dims=("latitude", "longitude"),
            ),
            "sea_ice_cover": xr.DataArray(
                fields["sea_ice_cover"],
                coords={"latitude": target_lat, "longitude": target_lon},
                dims=("latitude", "longitude"),
            ),
            "land_surface_temperature": xr.DataArray(
                fields["land_surface_temperature"],
                coords={"latitude": target_lat, "longitude": target_lon},
                dims=("latitude", "longitude"),
            ),
            "land_sea_mask": xr.DataArray(
                fields["land_sea_mask"],
                coords={"latitude": target_lat, "longitude": target_lon},
                dims=("latitude", "longitude"),
            ),
        }
    ).expand_dims(time=[np.datetime64(start_time)])


def _align_truth_masks_to_reference(
    truth: xr.Dataset,
    reference: xr.Dataset,
) -> xr.Dataset:
    """Force truth masks onto the same model-space support as one reference export."""
    aligned = truth.copy()
    for name, truth_field in aligned.data_vars.items():
        if name not in reference:
            continue
        ref_field = reference[name]
        if "time" in ref_field.dims and ref_field.sizes.get("time", 0) == 1 and truth_field.sizes.get("time", 0) >= 1:
            ref_field = ref_field.isel(time=0, drop=True).expand_dims(time=truth_field.coords["time"])
        ref_field = ref_field.broadcast_like(truth_field)
        aligned[name] = truth_field.where(np.isfinite(np.asarray(ref_field)))
    return aligned


def _build_observed_surface_dataset(
    dataset: xr.Dataset,
    *,
    start_time: str,
    forecast_days: int,
    resolution_deg: float,
    config: PreparationConfig,
) -> xr.Dataset:
    target_lat, target_lon = _target_grid_coords(resolution_deg)
    times = _datetime64_sequence(start_time, days=tuple(range(0, int(forecast_days) + 1)))
    observed = dataset.sel({config.time_name: times})
    initial_snapshot = dataset.sel({config.time_name: np.datetime64(start_time)})
    land_mask = _find_variable(initial_snapshot, "land_sea_mask")
    if config.time_name in land_mask.dims:
        land_mask = land_mask.isel({config.time_name: 0}, drop=True)
    variables = {
        "sea_surface_temperature": _interp_regular_field(
            _find_variable(observed, "sea_surface_temperature"),
            target_lat=target_lat,
            target_lon=target_lon,
            config=config,
        ),
        "land_surface_temperature": _interp_regular_field(
            _find_variable(observed, "land_surface_temperature"),
            target_lat=target_lat,
            target_lon=target_lon,
            config=config,
        ),
        "sea_ice_cover": _interp_regular_field(
            _find_variable(observed, "sea_ice_cover"),
            target_lat=target_lat,
            target_lon=target_lon,
            config=config,
        ),
        "land_sea_mask": _interp_mask_field(
            land_mask,
            target_lat=target_lat,
            target_lon=target_lon,
            config=config,
        ),
    }
    return xr.Dataset(variables).rename(
        {
            config.latitude_name: "latitude",
            config.longitude_name: "longitude",
            config.time_name: "time",
        }
    )


def _combine_surface_temperature(
    ocean_surface_temperature: xr.DataArray,
    land_surface_temperature: xr.DataArray,
    land_sea_mask: xr.DataArray,
) -> xr.DataArray:
    land_fraction = land_sea_mask.clip(min=0.0, max=1.0)
    ocean_filled = ocean_surface_temperature.fillna(land_surface_temperature)
    land_filled = land_surface_temperature.fillna(ocean_filled)
    return (1.0 - land_fraction) * ocean_filled + land_fraction * land_filled


def build_control_surface_forcing_dataset(observed_surface: xr.Dataset) -> xr.Dataset:
    """Return fixed-day0 ocean SST with daily land temperature and sea ice."""
    time_values = observed_surface["time"].values
    land_mask = observed_surface["land_sea_mask"]
    initial_ocean_sst = observed_surface["sea_surface_temperature"].isel(time=0, drop=True)
    fixed_ocean_sst = xr.concat(
        [initial_ocean_sst.expand_dims(time=[time_value]) for time_value in time_values],
        dim="time",
    )
    combined_surface_temperature = _combine_surface_temperature(
        fixed_ocean_sst,
        observed_surface["land_surface_temperature"],
        land_mask,
    )
    forcing = xr.Dataset(
        {
            "sea_surface_temperature": combined_surface_temperature.transpose("time", "latitude", "longitude"),
            "sea_ice_cover": ((1.0 - land_mask) * observed_surface["sea_ice_cover"].fillna(0.0)).transpose("time", "latitude", "longitude"),
            "initial_sea_surface_temperature": _combine_surface_temperature(
                initial_ocean_sst,
                observed_surface["land_surface_temperature"].isel(time=0, drop=True),
                land_mask,
            ).transpose("latitude", "longitude"),
            "initial_ocean_sea_surface_temperature": initial_ocean_sst.transpose("latitude", "longitude"),
            "fixed_ocean_sea_surface_temperature": fixed_ocean_sst.transpose("time", "latitude", "longitude"),
            "land_surface_temperature": observed_surface["land_surface_temperature"].transpose("time", "latitude", "longitude"),
            "land_sea_mask": land_mask.transpose("latitude", "longitude"),
        }
    )
    forcing.attrs["surface_source"] = "arco_era5"
    forcing.attrs["sst_mode"] = "fixed_day0_ocean"
    forcing.attrs["land_temperature_mode"] = "daily_observed"
    forcing.attrs["sea_ice_mode"] = "daily_observed"
    return forcing


def _build_model_surface_forcing_dataset(
    dataset: xr.Dataset,
    *,
    start_time: str,
    forecast_days: int,
    grid,
    config: PreparationConfig,
) -> xr.Dataset:
    lat_name = config.latitude_name
    lon_name = config.longitude_name
    target_dims = ("face", "x", "y")
    source_lat = np.asarray(dataset[lat_name], dtype=np.float64)
    source_lon = np.asarray(dataset[lon_name], dtype=np.float64)
    target_lat_rad = np.asarray(grid.grid_lat, dtype=np.float64)
    target_lon_rad = np.asarray(grid.grid_lon, dtype=np.float64)
    times = _datetime64_sequence(start_time, days=tuple(range(0, int(forecast_days) + 1)))
    observed = dataset.sel({config.time_name: times})

    initial_snapshot = dataset.sel({config.time_name: np.datetime64(start_time)})
    land_mask_source = _find_variable(initial_snapshot, "land_sea_mask")
    if config.time_name in land_mask_source.dims:
        land_mask_source = land_mask_source.isel({config.time_name: 0}, drop=True)
    land_fraction = _interp_latlon_to_cubedsphere_2d(
        np.asarray(land_mask_source, dtype=np.float32),
        source_lat=source_lat,
        source_lon=source_lon,
        target_lat_rad=target_lat_rad,
        target_lon_rad=target_lon_rad,
        k_neighbors=1,
    )
    land_fraction = np.clip(
        _blend_cubedsphere_scalar(land_fraction, config=config),
        0.0,
        1.0,
    ).astype(np.float32)

    ocean_series: list[np.ndarray] = []
    land_series: list[np.ndarray] = []
    ice_series: list[np.ndarray] = []
    for time_index in range(len(times)):
        ocean_source = _find_variable(observed.isel({config.time_name: time_index}), "sea_surface_temperature")
        land_source = _find_variable(observed.isel({config.time_name: time_index}), "land_surface_temperature")
        ice_source = _find_variable(observed.isel({config.time_name: time_index}), "sea_ice_cover")

        ocean_series.append(
            _blend_cubedsphere_scalar(
                _interp_latlon_to_cubedsphere_masked_2d(
                    ocean_source,
                    source_lat=source_lat,
                    source_lon=source_lon,
                    target_lat_rad=target_lat_rad,
                    target_lon_rad=target_lon_rad,
                    config=config,
                ),
                config=config,
            )
        )
        land_series.append(
            _blend_cubedsphere_scalar(
                _interp_latlon_to_cubedsphere_masked_2d(
                    land_source,
                    source_lat=source_lat,
                    source_lon=source_lon,
                    target_lat_rad=target_lat_rad,
                    target_lon_rad=target_lon_rad,
                    config=config,
                ),
                config=config,
            )
        )
        ice_series.append(
            _blend_cubedsphere_scalar(
                _interp_latlon_to_cubedsphere_masked_2d(
                    ice_source.fillna(0.0),
                    source_lat=source_lat,
                    source_lon=source_lon,
                    target_lat_rad=target_lat_rad,
                    target_lon_rad=target_lon_rad,
                    config=config,
                ),
                config=config,
            )
        )

    observed_ocean = np.stack(ocean_series, axis=0).astype(np.float32)
    land_temperature = np.stack(land_series, axis=0).astype(np.float32)
    sea_ice_cover = np.stack(ice_series, axis=0).astype(np.float32)
    fixed_ocean = np.broadcast_to(observed_ocean[:1], observed_ocean.shape).copy()
    uncoupled_surface = (
        (1.0 - land_fraction[None, ...]) * fixed_ocean
        + land_fraction[None, ...] * land_temperature
    ).astype(np.float32)
    sea_ice_cover = ((1.0 - land_fraction[None, ...]) * np.clip(sea_ice_cover, 0.0, 1.0)).astype(np.float32)

    return xr.Dataset(
        {
            "uncoupled_surface_temperature": xr.DataArray(
                uncoupled_surface,
                coords={"time": np.asarray(times), "face": np.arange(6), "x": np.arange(grid.n), "y": np.arange(grid.n)},
                dims=("time",) + target_dims,
            ),
            "fixed_ocean_sea_surface_temperature": xr.DataArray(
                fixed_ocean,
                coords={"time": np.asarray(times), "face": np.arange(6), "x": np.arange(grid.n), "y": np.arange(grid.n)},
                dims=("time",) + target_dims,
            ),
            "land_surface_temperature": xr.DataArray(
                land_temperature,
                coords={"time": np.asarray(times), "face": np.arange(6), "x": np.arange(grid.n), "y": np.arange(grid.n)},
                dims=("time",) + target_dims,
            ),
            "sea_ice_cover": xr.DataArray(
                sea_ice_cover,
                coords={"time": np.asarray(times), "face": np.arange(6), "x": np.arange(grid.n), "y": np.arange(grid.n)},
                dims=("time",) + target_dims,
            ),
            "land_fraction": xr.DataArray(
                land_fraction,
                coords={"face": np.arange(6), "x": np.arange(grid.n), "y": np.arange(grid.n)},
                dims=target_dims,
            ),
        }
    )


def _save_dataset(dataset: xr.Dataset, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_netcdf(path)
    return path


def _save_metadata(path: Path, metadata: PreparedCaseMetadata) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(asdict(metadata), handle, indent=2, default=str)
    return path


def load_case_metadata(path: str | Path) -> PreparedCaseMetadata:
    """Load one prepared-case metadata JSON file."""
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    raw["pressure_levels"] = tuple(int(level) for level in raw["pressure_levels"])
    return PreparedCaseMetadata(**raw)


def prepare_legoesm_case(
    *,
    start_time: str,
    forecast_days: int,
    case_dir: str | Path,
    experiment_config: ExperimentConfig,
    config: PreparationConfig | None = None,
    coupled: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Prepare one initialized ARCO/ERA5 case bundle for legoESM slab S2S."""
    prep = config or PreparationConfig()
    case_dir = Path(case_dir)
    prepared_dir = case_dir / "_prepared"
    prepared_dir.mkdir(parents=True, exist_ok=True)

    era5 = _sorted_era5_dataset(_open_era5_store(prep.era5_store), prep)
    initial_snapshot = era5.sel({prep.time_name: np.datetime64(start_time)})

    grid, sigma = _create_grid_and_sigma(experiment_config)
    state, q_v = _snapshot_to_state(initial_snapshot, grid=grid, sigma=sigma, config=prep)
    state = _project_state_to_cdgrid_subspace(
        state,
        grid=grid,
        experiment_config=experiment_config,
    )
    state, q_v = _balance_initialized_cdgrid_state(
        state,
        q_v,
        grid=grid,
        sigma=sigma,
        experiment_config=experiment_config,
        n_steps=int(prep.cdgrid_balance_steps),
    )
    _validate_monotonic_model_interfaces(state.p_s.data, sigma=sigma)
    if prep.seed_cloud_liquid_from_era5:
        q_c, q_r = _initial_condensate_seed(
            initial_snapshot,
            state=state,
            q_v=q_v,
            grid=grid,
            sigma=sigma,
            config=prep,
        )
    else:
        q_c = jnp.zeros_like(q_v)
        q_r = jnp.zeros_like(q_v)
    forcing_start_day = _absolute_start_day(start_time)

    restart_path = prepared_dir / "initial_restart.npz"
    save_restart(
        restart_path,
        state,
        q_v,
        step=0,
        day=forcing_start_day,
        config=experiment_config,
        q_c=np.asarray(q_c),
        q_r=np.asarray(q_r),
    )

    observed_surface = _build_observed_surface_dataset(
        era5,
        start_time=start_time,
        forecast_days=int(forecast_days),
        resolution_deg=prep.forcing_resolution_deg,
        config=prep,
    )
    surface_forcing = build_control_surface_forcing_dataset(observed_surface)
    model_surface_forcing = _build_model_surface_forcing_dataset(
        era5,
        start_time=start_time,
        forecast_days=int(forecast_days),
        grid=grid,
        config=prep,
    )
    truth = _build_truth_dataset(
        era5,
        start_time=start_time,
        forecast_days=int(forecast_days),
        grid=grid,
        config=prep,
    )
    initial = _build_initial_dataset_from_state(
        state,
        q_v,
        grid=grid,
        sigma=sigma,
        model_surface_forcing=model_surface_forcing,
        start_time=start_time,
        config=prep,
    )
    truth = _align_truth_masks_to_reference(truth, initial)

    forcing_path = prepared_dir / surface_forcing_filename(
        start_time=start_time,
        forecast_days=int(forecast_days),
    )
    model_forcing_path = prepared_dir / "surface_forcing_model.nc"
    observed_surface_path = prepared_dir / "observed_surface.nc"
    truth_path = prepared_dir / "truth.nc"
    initial_path = prepared_dir / "initial.nc"
    config_path = prepared_dir / "experiment_config.json"
    metadata_path = prepared_dir / "metadata.json"

    experiment_record = experiment_config_to_dict(experiment_config)
    metadata = PreparedCaseMetadata(
        start_time=start_time,
        forecast_days=int(forecast_days),
        forcing_start_day=float(forcing_start_day),
        forcing_resolution_deg=float(prep.forcing_resolution_deg),
        evaluation_resolution_deg=float(prep.evaluation_resolution_deg),
        pressure_levels=tuple(int(level) for level in prep.pressure_levels),
        experiment_config=experiment_record,
        coupled=dict(coupled or {"ocean_mode": "slab", "land_mode": "slab", "f_land_mode": "analytical"}),
    )

    outputs = {
        "restart": restart_path,
        "surface_forcing": _save_dataset(surface_forcing, forcing_path),
        "model_surface_forcing": _save_dataset(model_surface_forcing, model_forcing_path),
        "observed_surface": _save_dataset(observed_surface, observed_surface_path),
        "truth": _save_dataset(truth, truth_path),
        "initial": _save_dataset(initial, initial_path),
    }
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(experiment_record, handle, indent=2, default=str)
    outputs["experiment_config"] = config_path
    outputs["metadata"] = _save_metadata(metadata_path, metadata)
    return outputs


__all__ = [
    "DEFAULT_ARCO_ERA5_STORE",
    "DEFAULT_PRESSURE_LEVELS",
    "PreparedCaseMetadata",
    "PreparationConfig",
    "build_control_surface_forcing_dataset",
    "load_case_metadata",
    "prepare_legoesm_case",
]
