"""ChaosBench daily-file loader for local SFNO S2S training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np
import xarray as xr

from legoesm.ml.s2s.sfno_slab.config import (
    ChaosBenchS2SConfig,
    LAND_SEA_MASK_VAR,
)
from legoesm.ml.s2s.sfno_slab.regrid import TargetGridSpec, build_target_grid, regrid_channels_to_gaussian

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
except ImportError:  # pragma: no cover - handled at runtime when torch is unavailable
    torch = None
    DataLoader = None
    Dataset = object


_PERSISTENT_ZARR_DATASETS: dict[str, xr.Dataset] = {}


@dataclass(frozen=True)
class NormalizationBundle:
    """Per-channel normalization arrays for atmosphere/land/ocean inputs."""

    atmos_mean: np.ndarray
    atmos_sigma: np.ndarray
    land_mean: np.ndarray
    land_sigma: np.ndarray
    ocean_mean: np.ndarray
    ocean_sigma: np.ndarray


class SFNOS2SDataset(Dataset):
    """Torch-compatible dataset that preserves the local SFNO S2S sample contract."""

    def __init__(self, config: ChaosBenchS2SConfig) -> None:
        self.config = config
        self.path_map = _build_path_map(config)
        self.dates = _sorted_common_dates(self.path_map)
        self.norms = _load_normalization_bundle(config)
        self.target_grid = _build_worker_safe_target_grid(config.gaussian_n_max)

        self.n_samples = len(self.dates) - config.lead_time - config.n_steps + 1
        if self.n_samples <= 0:
            raise ValueError(
                "Not enough daily files for the requested S2S configuration: "
                f"{len(self.dates)} dates for lead_time={config.lead_time}, n_steps={config.n_steps}"
            )

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, sample_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return load_s2s_sample(
            self.config,
            sample_index=int(sample_index),
            path_map=self.path_map,
            dates=self.dates,
            target_grid=self.target_grid,
            norms=self.norms,
        )


def _build_worker_safe_target_grid(n_max: int) -> TargetGridSpec:
    full = build_target_grid(n_max)
    return TargetGridSpec(
        grid=SimpleNamespace(
            n_lat=int(full.grid.n_lat),
            n_lon=int(full.grid.n_lon),
            n_sh=int(full.grid.n_sh),
        ),
        latitude_deg=np.asarray(full.latitude_deg, dtype=np.float64),
        longitude_deg=np.asarray(full.longitude_deg, dtype=np.float64),
    )


def available_s2s_dates(config: ChaosBenchS2SConfig) -> list[str]:
    """Return the sorted common daily dates available across required datasets."""
    return _sorted_common_dates(_build_path_map(config))


def resolve_s2s_sample_dates(
    config: ChaosBenchS2SConfig,
    *,
    sample_index: int,
) -> tuple[str, list[str]]:
    """Return the initial date and target dates for one S2S sample index."""
    dates = available_s2s_dates(config)
    max_index = len(dates) - config.lead_time - config.n_steps + 1
    if sample_index < 0 or sample_index >= max_index:
        raise IndexError(f"sample_index {sample_index} is out of range for {max_index} samples")
    input_date = dates[sample_index]
    target_dates = dates[
        sample_index + config.lead_time : sample_index + config.lead_time + config.n_steps
    ]
    return input_date, target_dates


def load_normalization_bundle(config: ChaosBenchS2SConfig) -> NormalizationBundle:
    """Load the climatology normalization bundle used by the S2S loader."""
    return _load_normalization_bundle(config)


def normalize_atmospheric_channels(
    channels: np.ndarray,
    config: ChaosBenchS2SConfig,
    *,
    norms: NormalizationBundle | None = None,
) -> np.ndarray:
    """Normalize atmosphere channels stored on the last axis."""
    norms = norms or _load_normalization_bundle(config)
    return _normalize_last_axis(channels, norms.atmos_mean[:, 0, 0], norms.atmos_sigma[:, 0, 0], config)


def denormalize_atmospheric_channels(
    channels: np.ndarray,
    config: ChaosBenchS2SConfig,
    *,
    norms: NormalizationBundle | None = None,
) -> np.ndarray:
    """Denormalize atmosphere channels stored on the last axis."""
    norms = norms or _load_normalization_bundle(config)
    return _denormalize_last_axis(channels, norms.atmos_mean[:, 0, 0], norms.atmos_sigma[:, 0, 0], config)


def normalize_forcing_channels(
    channels: np.ndarray,
    config: ChaosBenchS2SConfig,
    *,
    norms: NormalizationBundle | None = None,
) -> np.ndarray:
    """Normalize forcing channels stored on the last axis."""
    norms = norms or _load_normalization_bundle(config)
    forcing_mean = np.concatenate(
        [norms.land_mean[:, 0, 0], norms.ocean_mean[:, 0, 0]],
        axis=0,
    ).astype(np.float32, copy=False)
    forcing_sigma = np.concatenate(
        [norms.land_sigma[:, 0, 0], norms.ocean_sigma[:, 0, 0]],
        axis=0,
    ).astype(np.float32, copy=False)
    return _normalize_last_axis(channels, forcing_mean, forcing_sigma, config)


def denormalize_forcing_channels(
    channels: np.ndarray,
    config: ChaosBenchS2SConfig,
    *,
    norms: NormalizationBundle | None = None,
) -> np.ndarray:
    """Denormalize forcing channels stored on the last axis."""
    norms = norms or _load_normalization_bundle(config)
    forcing_mean = np.concatenate(
        [norms.land_mean[:, 0, 0], norms.ocean_mean[:, 0, 0]],
        axis=0,
    ).astype(np.float32, copy=False)
    forcing_sigma = np.concatenate(
        [norms.land_sigma[:, 0, 0], norms.ocean_sigma[:, 0, 0]],
        axis=0,
    ).astype(np.float32, copy=False)
    return _denormalize_last_axis(channels, forcing_mean, forcing_sigma, config)


def load_surface_sequence(
    config: ChaosBenchS2SConfig,
    *,
    dates: list[str],
    collection: str,
    variables: tuple[str, ...],
    target_grid: TargetGridSpec | None = None,
) -> np.ndarray:
    """Load and regrid a surface-variable sequence onto the Gaussian grid.

    Returns arrays with shape ``(time, n_lat, n_lon, n_channels)`` in physical units.
    """
    if not variables:
        target_grid = target_grid or build_target_grid(config.gaussian_n_max)
        return np.empty((len(dates), target_grid.grid.n_lat, target_grid.grid.n_lon, 0), dtype=np.float32)

    root = Path(config.data_dir) / collection
    years = {date[:4] for date in dates}
    path_dict = _collect_daily_files(root, years)
    target_grid = target_grid or build_target_grid(config.gaussian_n_max)
    pieces: list[np.ndarray] = []
    for date in dates:
        source_lat, source_lon, channels = _load_surface_channels(path_dict, date, variables)
        regridded = regrid_channels_to_gaussian(
            channels,
            source_lat=source_lat,
            source_lon=source_lon,
            target=target_grid,
        )
        pieces.append(np.moveaxis(regridded, 0, -1))
    return np.stack(pieces, axis=0).astype(np.float32, copy=False)


def atmospheric_param_labels(config: ChaosBenchS2SConfig) -> list[str]:
    """Return flattened atmosphere channel labels in file/climatology order."""
    return [
        f"{var}-{level}"
        for var in config.atmosphere_vars
        for level in config.pressure_levels
    ]


def forcing_channel_labels(config: ChaosBenchS2SConfig) -> list[str]:
    """Return the input-only forcing channel labels."""
    return [*config.land_vars, *config.ocean_vars]


def create_s2s_training_iterator(
    config: ChaosBenchS2SConfig,
    *,
    batch_size: int = 2,
    seed: int = 0,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    persistent_workers: bool | None = None,
):
    """Yield normalized and regridded S2S batches for local SFNO training."""
    yield from _torch_training_iterator(
        config,
        batch_size=batch_size,
        seed=seed,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
    )


def create_s2s_dataloader(
    config: ChaosBenchS2SConfig,
    *,
    batch_size: int = 2,
    seed: int = 0,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    persistent_workers: bool | None = None,
    drop_last: bool = True,
):
    """Build a Torch DataLoader around the local SFNO S2S dataset."""
    if DataLoader is None or torch is None:
        raise ImportError("Torch is required for the SFNO DataLoader path.")

    dataset = SFNOS2SDataset(config)
    if batch_size > len(dataset):
        raise ValueError(
            f"Requested batch_size={batch_size} but only {len(dataset)} S2S samples are available."
        )

    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": drop_last,
        "generator": torch.Generator().manual_seed(seed),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = (
            True if persistent_workers is None else bool(persistent_workers)
        )
        loader_kwargs["prefetch_factor"] = int(prefetch_factor)
        loader_kwargs["multiprocessing_context"] = "spawn"

    return DataLoader(dataset, **loader_kwargs)


def _torch_training_iterator(
    config: ChaosBenchS2SConfig,
    *,
    batch_size: int,
    seed: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    prefetch_factor: int,
    persistent_workers: bool | None,
):
    loader = create_s2s_dataloader(
        config,
        batch_size=batch_size,
        seed=seed,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
    )

    while True:
        for batch in loader:
            yield tuple(
                _to_jax_float32(_to_numpy(component))
                for component in batch
            )


def _to_numpy(value):
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _to_jax_float32(value):
    import jax.numpy as jnp

    return jnp.asarray(value, dtype=jnp.float32)


def load_s2s_sample(
    config: ChaosBenchS2SConfig,
    *,
    sample_index: int,
    path_map: dict[str, dict[str, Path]] | None = None,
    dates: list[str] | None = None,
    target_grid: TargetGridSpec | None = None,
    norms: NormalizationBundle | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load one S2S sample as local-SFNO input, target, and forcing arrays."""
    path_map = path_map or _build_path_map(config)
    dates = dates or _sorted_common_dates(path_map)
    target_grid = target_grid or build_target_grid(config.gaussian_n_max)
    norms = norms or _load_normalization_bundle(config)

    max_index = len(dates) - config.lead_time - config.n_steps + 1
    if sample_index < 0 or sample_index >= max_index:
        raise IndexError(f"sample_index {sample_index} is out of range for {max_index} samples")

    input_date = dates[sample_index]
    target_dates = dates[
        sample_index + config.lead_time : sample_index + config.lead_time + config.n_steps
    ]

    source_lat, source_lon, input_atmos = _load_atmospheric_channels(path_map["era5"][input_date], config)
    _, _, input_land = _load_surface_channels(path_map.get("lra5"), input_date, config.land_vars)
    _, _, input_ocean = _load_surface_channels(path_map.get("ocean"), input_date, config.ocean_vars)

    input_atmos = _normalize_if_enabled(input_atmos, norms.atmos_mean, norms.atmos_sigma, config)
    input_land = _normalize_if_enabled(input_land, norms.land_mean, norms.land_sigma, config)
    input_ocean = _normalize_if_enabled(input_ocean, norms.ocean_mean, norms.ocean_sigma, config)

    input_forcing = _concat_forcing_channels(
        input_land,
        input_ocean,
        n_lat=source_lat.size,
        n_lon=source_lon.size,
    )
    input_combined = regrid_channels_to_gaussian(
        np.concatenate([input_atmos, input_forcing], axis=0),
        source_lat=source_lat,
        source_lon=source_lon,
        target=target_grid,
    )

    target_atmos_steps: list[np.ndarray] = []
    forcing_steps: list[np.ndarray] = []
    for date in target_dates:
        step_source_lat, step_source_lon, target_atmos = _load_atmospheric_channels(path_map["era5"][date], config)
        _, _, target_land = _load_surface_channels(path_map.get("lra5"), date, config.land_vars)
        _, _, target_ocean = _load_surface_channels(path_map.get("ocean"), date, config.ocean_vars)

        target_atmos = _normalize_if_enabled(target_atmos, norms.atmos_mean, norms.atmos_sigma, config)
        target_land = _normalize_if_enabled(target_land, norms.land_mean, norms.land_sigma, config)
        target_ocean = _normalize_if_enabled(target_ocean, norms.ocean_mean, norms.ocean_sigma, config)

        forcing_channels = _concat_forcing_channels(
            target_land,
            target_ocean,
            n_lat=step_source_lat.size,
            n_lon=step_source_lon.size,
        )
        target_atmos_steps.append(
            regrid_channels_to_gaussian(
                target_atmos,
                source_lat=step_source_lat,
                source_lon=step_source_lon,
                target=target_grid,
            )
        )
        forcing_steps.append(
            regrid_channels_to_gaussian(
                forcing_channels,
                source_lat=step_source_lat,
                source_lon=step_source_lon,
                target=target_grid,
            )
        )

    return (
        np.moveaxis(input_combined, 0, -1),
        np.moveaxis(np.stack(target_atmos_steps, axis=0), 1, -1),
        np.moveaxis(np.stack(forcing_steps, axis=0), 1, -1),
    )


def _build_path_map(config: ChaosBenchS2SConfig) -> dict[str, dict[str, Path]]:
    root = Path(config.data_dir)
    years = {str(year) for year in config.years}
    path_map: dict[str, dict[str, Path]] = {
        "era5": _collect_daily_files(root / "era5", years),
    }
    if config.land_vars:
        path_map["lra5"] = _collect_daily_files(root / "lra5", years)
    if config.ocean_vars:
        if config.ocean_source == "oras5":
            path_map["ocean"] = _collect_daily_files(root / "oras5", years)
        elif config.ocean_source == "arco_sst":
            path_map["ocean"] = _collect_cached_sst_dates(
                Path(config.arco_sst_cache_path),
                years,
            )
        else:
            raise ValueError(f"Unsupported ocean_source={config.ocean_source!r}")
    return path_map


def _collect_daily_files(directory: Path, years: set[str]) -> dict[str, Path]:
    pattern = re.compile(r"(\d{8})")
    files: dict[str, Path] = {}
    for path in sorted(directory.glob("*.zarr")):
        match = pattern.search(path.name)
        if match is None:
            continue
        date = match.group(1)
        if date[:4] in years:
            files[date] = path
    return files


def _collect_cached_sst_dates(cache_path: Path, years: set[str]) -> dict[str, Path]:
    ds = xr.open_zarr(cache_path, consolidated=False)
    try:
        if "date" in ds:
            dates = [str(value) for value in np.asarray(ds["date"].values)]
        else:
            dates = [str(value)[:10] for value in np.asarray(ds["time"].values)]
    finally:
        ds.close()
    out: dict[str, Path] = {}
    for date in dates:
        ymd = date.replace("-", "")[:8]
        if ymd[:4] in years:
            out[ymd] = cache_path
    return out


def _sorted_common_dates(path_map: dict[str, dict[str, Path]]) -> list[str]:
    available = [set(paths.keys()) for paths in path_map.values() if paths]
    common = set.intersection(*available) if available else set()
    return sorted(common)


def _open_dataset(path: Path) -> xr.Dataset:
    key = str(path)
    if "arco_sst_daily_" in path.name:
        ds = _PERSISTENT_ZARR_DATASETS.get(key)
        if ds is None:
            ds = xr.open_zarr(path, consolidated=False)
            _PERSISTENT_ZARR_DATASETS[key] = ds
        return ds
    return xr.open_dataset(path, engine="zarr")


def _load_atmospheric_channels(
    path: Path,
    config: ChaosBenchS2SConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ds = _open_dataset(path)
    arrays = []
    for var in config.atmosphere_vars:
        data = ds[var].sel({config.level_name: list(config.pressure_levels)})
        data = data.transpose(config.level_name, config.latitude_name, config.longitude_name)
        arrays.append(np.asarray(data.values, dtype=np.float32))
    packed = np.stack(arrays, axis=0).reshape(-1, ds.sizes[config.latitude_name], ds.sizes[config.longitude_name])
    source_lat = np.asarray(ds[config.latitude_name].values, dtype=np.float64)
    source_lon = np.asarray(ds[config.longitude_name].values, dtype=np.float64)
    ds.close()
    return source_lat, source_lon, packed


def _load_surface_channels(
    path_dict: dict[str, Path] | None,
    date: str,
    variables: tuple[str, ...],
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray]:
    if not variables:
        return None, None, np.empty((0, 0, 0), dtype=np.float32)
    if path_dict is None:
        raise KeyError(f"Missing path dictionary for forcing variables {variables!r}")
    path = path_dict[date]
    ds = _open_dataset(path)
    arrays = []
    for var in variables:
        arrays.append(
            np.asarray(
                _select_surface_variable(ds, var, date).transpose("latitude", "longitude").values,
                dtype=np.float32,
            )
        )
    packed = np.stack(arrays, axis=0)
    source_lat = np.asarray(ds["latitude"].values, dtype=np.float64)
    source_lon = np.asarray(ds["longitude"].values, dtype=np.float64)
    if "arco_sst_daily_" not in path.name:
        ds.close()
    return source_lat, source_lon, packed


def _select_surface_variable(ds: xr.Dataset, var: str, date: str) -> xr.DataArray:
    if var == LAND_SEA_MASK_VAR:
        if "ocean_mask" in ds:
            data = xr.ones_like(ds["ocean_mask"], dtype=np.float32) - ds["ocean_mask"].astype(np.float32)
        elif "sosstsst" in ds:
            data = xr.where(ds["sosstsst"].notnull(), np.float32(0.0), np.float32(1.0))
        else:
            raise KeyError("Cannot derive land_sea_mask without ocean_mask or sosstsst.")
    else:
        data = ds[var]
    if "time" in data.dims:
        time_value = np.datetime64(f"{date[:4]}-{date[4:6]}-{date[6:8]}")
        data = data.sel(time=time_value)
    return data


def _load_normalization_bundle(config: ChaosBenchS2SConfig) -> NormalizationBundle:
    root = Path(config.data_dir) / "climatology"

    atmos = xr.open_dataset(root / "climatology_era5.zarr", engine="zarr")
    atmos_labels = atmospheric_param_labels(config)
    atmos_mean = np.asarray(atmos["mean"].sel(param=atmos_labels).values, dtype=np.float32)[:, None, None]
    atmos_sigma = np.asarray(atmos["sigma"].sel(param=atmos_labels).values, dtype=np.float32)[:, None, None]
    atmos.close()

    land_mean = np.empty((0, 1, 1), dtype=np.float32)
    land_sigma = np.empty((0, 1, 1), dtype=np.float32)
    if config.land_vars:
        land = xr.open_dataset(root / "climatology_lra5.zarr", engine="zarr")
        land_mean = np.asarray(land["mean"].sel(param=list(config.land_vars)).values, dtype=np.float32)[:, None, None]
        land_sigma = np.asarray(land["sigma"].sel(param=list(config.land_vars)).values, dtype=np.float32)[:, None, None]
        land.close()

    ocean_mean = np.empty((0, 1, 1), dtype=np.float32)
    ocean_sigma = np.empty((0, 1, 1), dtype=np.float32)
    if config.ocean_vars:
        real_ocean_vars = [var for var in config.ocean_vars if var != LAND_SEA_MASK_VAR]
        stats_path = (
            Path(config.arco_sst_stats_path)
            if config.ocean_source == "arco_sst"
            else root / "climatology_oras5.zarr"
        )
        ocean_stats = None
        if real_ocean_vars:
            ocean_stats = xr.open_dataset(stats_path, engine="zarr")
        mean_values: list[float] = []
        sigma_values: list[float] = []
        for var in config.ocean_vars:
            if var == LAND_SEA_MASK_VAR:
                mean_values.append(0.0)
                sigma_values.append(1.0)
            else:
                assert ocean_stats is not None
                mean_values.append(float(ocean_stats["mean"].sel(param=var).values))
                sigma_values.append(max(float(ocean_stats["sigma"].sel(param=var).values), 1.0e-6))
        if ocean_stats is not None:
            ocean_stats.close()
        ocean_mean = np.asarray(mean_values, dtype=np.float32)[:, None, None]
        ocean_sigma = np.asarray(sigma_values, dtype=np.float32)[:, None, None]

    return NormalizationBundle(
        atmos_mean=atmos_mean,
        atmos_sigma=np.maximum(atmos_sigma, 1.0e-6),
        land_mean=land_mean,
        land_sigma=np.maximum(land_sigma, 1.0e-6),
        ocean_mean=ocean_mean,
        ocean_sigma=np.maximum(ocean_sigma, 1.0e-6),
    )


def _normalize_if_enabled(
    channels: np.ndarray,
    mean: np.ndarray,
    sigma: np.ndarray,
    config: ChaosBenchS2SConfig,
) -> np.ndarray:
    if channels.size == 0:
        return channels.astype(np.float32, copy=False)
    if config.normalize:
        out = ((channels - mean) / sigma).astype(np.float32, copy=False)
    else:
        out = channels.astype(np.float32, copy=False)
    return np.where(np.isfinite(out), out, np.float32(0.0)).astype(np.float32, copy=False)


def _normalize_last_axis(
    channels: np.ndarray,
    mean: np.ndarray,
    sigma: np.ndarray,
    config: ChaosBenchS2SConfig,
) -> np.ndarray:
    if channels.size == 0 or not config.normalize:
        return np.asarray(channels, dtype=np.float32)
    values = np.asarray(channels, dtype=np.float32)
    mean = np.asarray(mean, dtype=np.float32)
    sigma = np.asarray(sigma, dtype=np.float32)
    reshape = (1,) * (values.ndim - 1) + (mean.size,)
    return ((values - mean.reshape(reshape)) / sigma.reshape(reshape)).astype(np.float32, copy=False)


def _denormalize_last_axis(
    channels: np.ndarray,
    mean: np.ndarray,
    sigma: np.ndarray,
    config: ChaosBenchS2SConfig,
) -> np.ndarray:
    if channels.size == 0 or not config.normalize:
        return np.asarray(channels, dtype=np.float32)
    values = np.asarray(channels, dtype=np.float32)
    mean = np.asarray(mean, dtype=np.float32)
    sigma = np.asarray(sigma, dtype=np.float32)
    reshape = (1,) * (values.ndim - 1) + (mean.size,)
    return (values * sigma.reshape(reshape) + mean.reshape(reshape)).astype(np.float32, copy=False)


def _concat_forcing_channels(
    land_channels: np.ndarray,
    ocean_channels: np.ndarray,
    *,
    n_lat: int,
    n_lon: int,
) -> np.ndarray:
    pieces = []
    if land_channels.size:
        pieces.append(land_channels)
    if ocean_channels.size:
        pieces.append(ocean_channels)
    if not pieces:
        return np.empty((0, n_lat, n_lon), dtype=np.float32)
    return np.concatenate(pieces, axis=0).astype(np.float32, copy=False)
