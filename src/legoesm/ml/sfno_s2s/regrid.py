"""Regridding helpers from ChaosBench lat-lon to the local Gaussian grid."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import numpy as np

from legoesm.grids.gaussian import GaussianGrid, create_gaussian_grid


@dataclass(frozen=True)
class TargetGridSpec:
    """Precomputed Gaussian-grid metadata for S2S training."""

    grid: GaussianGrid
    latitude_deg: np.ndarray
    longitude_deg: np.ndarray


def build_target_grid(n_max: int = 79) -> TargetGridSpec:
    """Construct the Gaussian target grid used by the local SFNO.

    ``n_max=79`` yields a ``120 x 240`` quadratic grid, which lines up with the
    ChaosBench longitude count and keeps the latitude remap lightweight.
    """
    if not jax.config.jax_enable_x64:
        jax.config.update("jax_enable_x64", True)
    grid = create_gaussian_grid(n_max)
    return TargetGridSpec(
        grid=grid,
        latitude_deg=np.rad2deg(np.asarray(grid.lat)),
        longitude_deg=np.rad2deg(np.asarray(grid.lon)),
    )


def regrid_channels_to_gaussian(
    channels: np.ndarray,
    *,
    source_lat: np.ndarray,
    source_lon: np.ndarray,
    target: TargetGridSpec,
) -> np.ndarray:
    """Regrid ``(channels, lat, lon)`` fields onto the Gaussian target grid.

    Latitude is remapped linearly and longitude is remapped with periodic
    1D interpolation so reduced Gaussian grids can be used for smoke tests.
    """
    if channels.ndim != 3:
        raise ValueError(f"Expected channels with shape (n_channels, n_lat, n_lon), got {channels.shape}")
    if channels.size == 0:
        return np.empty((0, target.latitude_deg.size, target.longitude_deg.size), dtype=np.float32)

    lat = np.asarray(source_lat, dtype=np.float64)
    lon = np.asarray(source_lon, dtype=np.float64)
    values = np.asarray(channels, dtype=np.float32)
    if lat[0] > lat[-1]:
        lat = lat[::-1]
        values = values[:, ::-1, :]

    upper = np.searchsorted(lat, target.latitude_deg, side="left")
    upper = np.clip(upper, 1, lat.size - 1)
    lower = upper - 1
    lower_lat = lat[lower]
    upper_lat = lat[upper]
    denom = np.where(np.abs(upper_lat - lower_lat) > 1.0e-12, upper_lat - lower_lat, 1.0)
    weight_upper = ((target.latitude_deg - lower_lat) / denom).astype(np.float32)
    weight_lower = 1.0 - weight_upper

    lower_values = values[:, lower, :]
    upper_values = values[:, upper, :]
    regridded = (
        lower_values * weight_lower[None, :, None]
        + upper_values * weight_upper[None, :, None]
    )

    if lon[0] > lon[-1]:
        lon = lon[::-1]
        regridded = regridded[..., ::-1]

    lon = np.mod(lon, 360.0)
    target_lon = np.mod(np.asarray(target.longitude_deg, dtype=np.float64), 360.0)
    sort_idx = np.argsort(lon)
    lon_sorted = lon[sort_idx]
    data_sorted = regridded[..., sort_idx]

    # Add a wrapped endpoint so interpolation remains periodic across Greenwich.
    lon_ext = np.concatenate([lon_sorted, lon_sorted[:1] + 360.0])
    data_ext = np.concatenate([data_sorted, data_sorted[..., :1]], axis=-1)
    target_lon_ext = np.where(target_lon < lon_ext[0], target_lon + 360.0, target_lon)

    lon_regridded = np.empty(
        (regridded.shape[0], regridded.shape[1], target_lon_ext.size),
        dtype=np.float32,
    )
    for channel_index in range(regridded.shape[0]):
        for lat_index in range(regridded.shape[1]):
            lon_regridded[channel_index, lat_index] = np.interp(
                target_lon_ext,
                lon_ext,
                data_ext[channel_index, lat_index],
            )
    return lon_regridded.astype(np.float32, copy=False)
