"""Paired coupled-vs-uncoupled rollout helpers for legoESM slab S2S."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Callable, Sequence

import jax.numpy as jnp
import numpy as np
import xarray as xr

from legoesm import constants
from legoesm.driver.config import ExperimentConfig, experiment_config_from_dict
from legoesm.driver.coupled_config import CoupledConfig
from legoesm.driver.coupled_esm_driver import CoupledESMDriver
from legoesm.driver.model_driver import ModelDriver
from legoesm.grids.cubed_sphere import rotate_winds_grid_to_geo
from legoesm.grids.regridding import (
    _pad_field_for_regrid,
    apply_cubedsphere_to_latlon,
    get_cubedsphere_to_latlon_weights,
)
from legoesm.grids.vertical import compute_geopotential, compute_geopotential_hybrid
from legoesm.ml.s2s.legoesm_slab.campaign import surface_forcing_filename
from legoesm.ml.s2s.legoesm_slab.preparation import PreparedCaseMetadata, load_case_metadata
from legoesm.ocean.simple_ocean import SimpleOceanConfig

DEFAULT_EXPORT_PRESSURE_LEVELS = (200, 500, 700, 850)


@dataclass(frozen=True)
class ForecastExportConfig:
    """Lat-lon export settings for one paired case rollout."""

    pressure_levels: tuple[int, ...] = DEFAULT_EXPORT_PRESSURE_LEVELS
    evaluation_resolution_deg: float = 5.0


@dataclass(frozen=True)
class CaseRunSummary:
    """Compact JSON-serializable summary for one paired case run."""

    start_time: str
    forecast_days: int
    forcing_start_day: float
    coupled_status: str
    uncoupled_status: str
    pressure_levels: tuple[int, ...]
    evaluation_resolution_deg: float


@dataclass(frozen=True)
class ModelSurfaceForcing:
    """Prepared model-grid surface forcing for land-aware initialized runs."""

    times: jnp.ndarray
    uncoupled_surface_temperature: jnp.ndarray
    fixed_ocean_surface_temperature: jnp.ndarray
    land_surface_temperature: jnp.ndarray
    sea_ice_cover: jnp.ndarray
    land_fraction: jnp.ndarray


def wrap_relative_forcing_getter(
    getter: Callable[[float], tuple[Any, Any]],
    forcing_start_day: float,
) -> Callable[[float], tuple[Any, Any]]:
    """Shift an absolute simulation day into a forcing-relative day."""

    def wrapped(day: float) -> tuple[Any, Any]:
        return getter(float(day) - float(forcing_start_day))

    return wrapped


def _load_model_surface_forcing(path: str | Path) -> ModelSurfaceForcing:
    dataset = xr.open_dataset(path)
    try:
        time_values = np.asarray(dataset["time"].values, dtype="datetime64[s]")
        time_days = ((time_values - time_values[0]) / np.timedelta64(1, "D")).astype(np.float32)
        return ModelSurfaceForcing(
            times=jnp.asarray(time_days),
            uncoupled_surface_temperature=jnp.asarray(np.asarray(dataset["uncoupled_surface_temperature"], dtype=np.float32)),
            fixed_ocean_surface_temperature=jnp.asarray(np.asarray(dataset["fixed_ocean_sea_surface_temperature"], dtype=np.float32)),
            land_surface_temperature=jnp.asarray(np.asarray(dataset["land_surface_temperature"], dtype=np.float32)),
            sea_ice_cover=jnp.asarray(np.asarray(dataset["sea_ice_cover"], dtype=np.float32)),
            land_fraction=jnp.asarray(np.asarray(dataset["land_fraction"], dtype=np.float32)),
        )
    finally:
        dataset.close()


def _interp_time_series(values: jnp.ndarray, times: jnp.ndarray, day: float) -> jnp.ndarray:
    ntime = int(times.shape[0])
    if ntime <= 1:
        return values[0]
    day_value = jnp.asarray(float(day), dtype=jnp.float32)
    # Initialized forecast forcing should clamp at the prepared window
    # endpoints, not wrap cyclically back to day 0 at the final timestamp.
    day_value = jnp.clip(day_value, times[0], times[-1])
    idx = jnp.searchsorted(times, day_value, side="right") - 1
    idx = jnp.clip(idx, 0, ntime - 2)
    idx_next = idx + 1
    dt = jnp.maximum(times[idx_next] - times[idx], 1.0e-6)
    weight = (day_value - times[idx]) / dt
    return (1.0 - weight) * values[idx] + weight * values[idx_next]


def build_surface_boundary_getter(
    forcing: ModelSurfaceForcing,
    *,
    ocean_temperature_getter: Callable[[float], jnp.ndarray] | None = None,
) -> Callable[[float], tuple[jnp.ndarray, jnp.ndarray]]:
    """Return a land-aware surface boundary getter on the model grid."""

    land_fraction = jnp.clip(forcing.land_fraction, 0.0, 1.0)

    def getter(day: float) -> tuple[jnp.ndarray, jnp.ndarray]:
        sic = _interp_time_series(forcing.sea_ice_cover, forcing.times, day)
        sic = (1.0 - land_fraction) * jnp.clip(sic, 0.0, 1.0)
        if ocean_temperature_getter is None:
            sst = _interp_time_series(
                forcing.uncoupled_surface_temperature,
                forcing.times,
                day,
            )
            return sst, sic
        ocean_temperature = jnp.asarray(ocean_temperature_getter(day), dtype=forcing.land_fraction.dtype)
        land_temperature = _interp_time_series(forcing.land_surface_temperature, forcing.times, day)
        sst = (1.0 - land_fraction) * ocean_temperature + land_fraction * land_temperature
        return sst, sic

    return getter


def _target_grid_shape(resolution_deg: float) -> tuple[int, int]:
    n_lon = int(round(360.0 / float(resolution_deg)))
    n_lat = int(round(180.0 / float(resolution_deg))) + 1
    return n_lat, n_lon


def _forecast_longitude_coords(weights) -> np.ndarray:
    """Return forecast-export longitudes in the same 0..360 ordering as prep truth."""
    return np.mod(np.asarray(weights.lon_cent, dtype=np.float32) + 180.0, 360.0)


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
            if denom == 0.0:
                alpha = 0.0
            else:
                alpha = float(np.clip((np.log(target_pressure) - p_lo) / denom, 0.0, 1.0))
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


def _compute_model_pressures(state, sigma) -> np.ndarray:
    p_s = np.asarray(state.p_s.data, dtype=np.float64)
    if hasattr(sigma, "A_full") and hasattr(sigma, "B_full"):
        return (
            np.asarray(sigma.A_full, dtype=np.float64) * float(getattr(sigma, "p_ref", 1.0e5))
            + np.asarray(sigma.B_full, dtype=np.float64) * p_s[..., None]
        )
    return np.asarray(sigma.sigma_full, dtype=np.float64) * p_s[..., None]


def _virtual_temperature(temperature: np.ndarray, specific_humidity: np.ndarray) -> np.ndarray:
    """Approximate virtual temperature from temperature and specific humidity."""
    return np.asarray(temperature, dtype=np.float64) * (
        1.0 + 0.61 * np.asarray(specific_humidity, dtype=np.float64)
    )


def _state_specific_humidity(state) -> np.ndarray:
    """Extract specific humidity from either legacy or tracer-backed state."""
    if hasattr(state, "q_v") and state.q_v is not None:
        return np.asarray(state.q_v.data, dtype=np.float64)
    tracers = getattr(state, "tracers", None) or {}
    if "q_v" in tracers:
        field = tracers["q_v"]
        return np.asarray(field.data if hasattr(field, "data") else field, dtype=np.float64)
    return np.zeros_like(np.asarray(state.T.data, dtype=np.float64))


def _compute_model_geopotential(state, sigma) -> np.ndarray:
    T_virtual = _virtual_temperature(state.T.data, _state_specific_humidity(state))
    if hasattr(sigma, "A_full") and hasattr(sigma, "B_full"):
        phi = compute_geopotential_hybrid(
            T_virtual,
            state.p_s.data,
            sigma,
            state.phis.data,
        )
    else:
        phi = compute_geopotential(
            T_virtual,
            state.p_s.data,
            sigma,
            state.phis.data,
        )
    return np.asarray(phi, dtype=np.float32)


def _winds_to_geographic(grid, u_grid: np.ndarray, v_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotate cubed-sphere winds from grid axes to geographic east/north."""
    if not hasattr(grid, "angle"):
        return np.asarray(u_grid, dtype=np.float32), np.asarray(v_grid, dtype=np.float32)
    u_geo, v_geo = rotate_winds_grid_to_geo(
        jnp.asarray(u_grid),
        jnp.asarray(v_grid),
        jnp.asarray(grid.angle)[..., None],
    )
    return np.asarray(u_geo, dtype=np.float32), np.asarray(v_geo, dtype=np.float32)


def _surface_fields_from_driver(driver, day: float, *, ocean_state=None) -> tuple[np.ndarray, np.ndarray]:
    if ocean_state is not None:
        _, sic = driver.get_sst_sic(day)
        return np.asarray(ocean_state.T_sfc.data, dtype=np.float32), np.asarray(sic, dtype=np.float32)
    sst, sic = driver.get_sst_sic(day)
    return np.asarray(sst, dtype=np.float32), np.asarray(sic, dtype=np.float32)


class DailyForecastCollector:
    """Collect and export daily forecast fields on a common lat-lon grid."""

    def __init__(
        self,
        *,
        grid,
        sigma,
        start_time: str,
        forecast_days: int,
        forcing_start_day: float,
        config: ForecastExportConfig,
    ) -> None:
        if not hasattr(grid, "n"):
            raise NotImplementedError("legoesm_slab export currently supports cubed-sphere grids only")
        self.grid = grid
        self.sigma = sigma
        self.start_time = np.datetime64(start_time)
        self.forecast_days = int(forecast_days)
        self.forcing_start_day = float(forcing_start_day)
        self.config = config
        n_lat, n_lon = _target_grid_shape(config.evaluation_resolution_deg)
        self._weights = get_cubedsphere_to_latlon_weights(grid.n, n_lon=n_lon, n_lat=n_lat)
        self._lead_days: list[int] = []
        self._times: list[np.datetime64] = []
        self._records: dict[str, list[np.ndarray]] = {
            "temperature": [],
            "specific_humidity": [],
            "u_component_of_wind": [],
            "v_component_of_wind": [],
            "geopotential": [],
            "surface_pressure": [],
            "sea_surface_temperature": [],
            "sea_ice_cover": [],
            "precipitation": [],
            "rlut": [],
        }

    def record(self, driver, day: float, dt_segment: float, *, ocean_state=None) -> None:
        """Collect one daily forecast slice."""
        lead_day = int(round(float(day) - self.forcing_start_day))
        if lead_day <= 0 or lead_day > self.forecast_days:
            return
        if lead_day in self._lead_days:
            return

        state = driver.state
        q_v = np.asarray(driver.q_v, dtype=np.float32)
        surface_pressure = np.asarray(state.p_s.data, dtype=np.float32)
        pressure_model = _compute_model_pressures(state, self.sigma)
        geopotential_model = _compute_model_geopotential(state, self.sigma)
        u_geo_model, v_geo_model = _winds_to_geographic(
            self.grid,
            np.asarray(state.u.data, dtype=np.float32),
            np.asarray(state.v.data, dtype=np.float32),
        )

        self._records["temperature"].append(
            _apply_cubedsphere_to_latlon_3d_masked(
                _interpolate_model_field_to_pressure_levels(
                    np.asarray(state.T.data, dtype=np.float32),
                    pressure_model,
                    self.config.pressure_levels,
                ),
                self._weights,
            ).astype(np.float32)
        )
        self._records["specific_humidity"].append(
            _apply_cubedsphere_to_latlon_3d_masked(
                _interpolate_model_field_to_pressure_levels(
                    q_v,
                    pressure_model,
                    self.config.pressure_levels,
                ),
                self._weights,
            ).astype(np.float32)
        )
        self._records["u_component_of_wind"].append(
            _apply_cubedsphere_to_latlon_3d_masked(
                _interpolate_model_field_to_pressure_levels(
                    u_geo_model,
                    pressure_model,
                    self.config.pressure_levels,
                ),
                self._weights,
            ).astype(np.float32)
        )
        self._records["v_component_of_wind"].append(
            _apply_cubedsphere_to_latlon_3d_masked(
                _interpolate_model_field_to_pressure_levels(
                    v_geo_model,
                    pressure_model,
                    self.config.pressure_levels,
                ),
                self._weights,
            ).astype(np.float32)
        )
        self._records["geopotential"].append(
            _apply_cubedsphere_to_latlon_3d_masked(
                _interpolate_model_field_to_pressure_levels(
                    geopotential_model,
                    pressure_model,
                    self.config.pressure_levels,
                ),
                self._weights,
            ).astype(np.float32)
        )
        self._records["surface_pressure"].append(
            apply_cubedsphere_to_latlon(surface_pressure, self._weights).astype(np.float32)
        )

        sst, sic = _surface_fields_from_driver(driver, day, ocean_state=ocean_state)
        land_fraction = np.asarray(
            getattr(driver, "_f_land", np.zeros_like(state.p_s.data)),
            dtype=np.float32,
        )
        sst_ocean_only = np.where(land_fraction < 0.5, sst, np.nan).astype(np.float32)
        aux = getattr(driver, "_carry_aux", {})
        seg_precip = np.asarray(aux.get("seg_precip", jnp.zeros_like(state.p_s.data)), dtype=np.float32)
        rlut = np.asarray(aux.get("held_lw_up_toa", jnp.zeros_like(state.p_s.data)), dtype=np.float32)
        precip_rate = seg_precip / max(float(dt_segment), 1.0) * 86400.0

        self._records["sea_surface_temperature"].append(
            _apply_cubedsphere_to_latlon_masked(sst_ocean_only, self._weights).astype(np.float32)
        )
        self._records["sea_ice_cover"].append(
            apply_cubedsphere_to_latlon(sic, self._weights).astype(np.float32)
        )
        self._records["precipitation"].append(
            apply_cubedsphere_to_latlon(precip_rate, self._weights).astype(np.float32)
        )
        self._records["rlut"].append(
            apply_cubedsphere_to_latlon(rlut, self._weights).astype(np.float32)
        )
        self._lead_days.append(lead_day)
        self._times.append(self.start_time + np.timedelta64(lead_day, "D"))

    def to_dataset(self) -> xr.Dataset:
        """Return the collected forecast slices as one xarray Dataset."""
        if not self._lead_days:
            raise ValueError("No daily forecast slices were collected before dataset export")
        latitude = xr.DataArray(self._weights.lat_cent, dims=("latitude",), name="latitude")
        longitude = xr.DataArray(_forecast_longitude_coords(self._weights), dims=("longitude",), name="longitude")
        level = xr.DataArray(np.asarray(self.config.pressure_levels, dtype=int), dims=("level",), name="level")
        lead_day = np.asarray(self._lead_days, dtype=int)
        time = np.asarray(self._times, dtype="datetime64[s]")
        dataset = xr.Dataset(
            {
                "temperature": xr.DataArray(
                    np.stack(self._records["temperature"], axis=0),
                    coords={"lead_day": lead_day, "time": ("lead_day", time), "latitude": latitude, "longitude": longitude, "level": level},
                    dims=("lead_day", "latitude", "longitude", "level"),
                ).transpose("lead_day", "level", "latitude", "longitude"),
                "specific_humidity": xr.DataArray(
                    np.stack(self._records["specific_humidity"], axis=0),
                    coords={"lead_day": lead_day, "time": ("lead_day", time), "latitude": latitude, "longitude": longitude, "level": level},
                    dims=("lead_day", "latitude", "longitude", "level"),
                ).transpose("lead_day", "level", "latitude", "longitude"),
                "u_component_of_wind": xr.DataArray(
                    np.stack(self._records["u_component_of_wind"], axis=0),
                    coords={"lead_day": lead_day, "time": ("lead_day", time), "latitude": latitude, "longitude": longitude, "level": level},
                    dims=("lead_day", "latitude", "longitude", "level"),
                ).transpose("lead_day", "level", "latitude", "longitude"),
                "v_component_of_wind": xr.DataArray(
                    np.stack(self._records["v_component_of_wind"], axis=0),
                    coords={"lead_day": lead_day, "time": ("lead_day", time), "latitude": latitude, "longitude": longitude, "level": level},
                    dims=("lead_day", "latitude", "longitude", "level"),
                ).transpose("lead_day", "level", "latitude", "longitude"),
                "geopotential": xr.DataArray(
                    np.stack(self._records["geopotential"], axis=0),
                    coords={"lead_day": lead_day, "time": ("lead_day", time), "latitude": latitude, "longitude": longitude, "level": level},
                    dims=("lead_day", "latitude", "longitude", "level"),
                ).transpose("lead_day", "level", "latitude", "longitude"),
                "sea_surface_temperature": xr.DataArray(
                    np.stack(self._records["sea_surface_temperature"], axis=0),
                    coords={"lead_day": lead_day, "time": ("lead_day", time), "latitude": latitude, "longitude": longitude},
                    dims=("lead_day", "latitude", "longitude"),
                ),
                "surface_pressure": xr.DataArray(
                    np.stack(self._records["surface_pressure"], axis=0),
                    coords={"lead_day": lead_day, "time": ("lead_day", time), "latitude": latitude, "longitude": longitude},
                    dims=("lead_day", "latitude", "longitude"),
                ),
                "sea_ice_cover": xr.DataArray(
                    np.stack(self._records["sea_ice_cover"], axis=0),
                    coords={"lead_day": lead_day, "time": ("lead_day", time), "latitude": latitude, "longitude": longitude},
                    dims=("lead_day", "latitude", "longitude"),
                ),
                "precipitation": xr.DataArray(
                    np.stack(self._records["precipitation"], axis=0),
                    coords={"lead_day": lead_day, "time": ("lead_day", time), "latitude": latitude, "longitude": longitude},
                    dims=("lead_day", "latitude", "longitude"),
                ),
                "rlut": xr.DataArray(
                    np.stack(self._records["rlut"], axis=0),
                    coords={"lead_day": lead_day, "time": ("lead_day", time), "latitude": latitude, "longitude": longitude},
                    dims=("lead_day", "latitude", "longitude"),
                ),
            }
        )
        dataset.attrs["pressure_levels_hpa"] = ",".join(str(level_value) for level_value in self.config.pressure_levels)
        dataset.attrs["evaluation_resolution_deg"] = float(self.config.evaluation_resolution_deg)
        return dataset

    def has_records(self) -> bool:
        """Return whether at least one daily forecast slice was collected."""
        return bool(self._lead_days)


class InitializedCoupledSlabDriver(CoupledESMDriver):
    """Coupled slab driver with relative forcing and daily forecast export."""

    def __init__(
        self,
        atm_config: ExperimentConfig,
        *,
        forcing_start_day: float,
        collector: DailyForecastCollector | None,
        coupled_config: CoupledConfig,
        surface_forcing: ModelSurfaceForcing,
        output_dir: str | Path,
    ) -> None:
        super().__init__(atm_config, coupled_config=coupled_config, output_dir=output_dir)
        self._forcing_start_day = float(forcing_start_day)
        self._collector = collector
        self._surface_forcing = surface_forcing

    def setup(self) -> None:
        """Initialize the coupled system with a forcing-relative SST getter."""
        self._atm.setup()
        self._atm.get_sst_sic = build_surface_boundary_getter(
            self._surface_forcing,
        )
        self._atm._f_land = self._surface_forcing.land_fraction
        if getattr(self._atm, "physics", None) is not None:
            self._atm.physics.land_fraction = self._surface_forcing.land_fraction
        self._init_ocean()
        self._init_coupler()
        self._init_carbon()
        self._override_sst()

    def _init_ocean(self) -> None:
        """Initialize the slab ocean from the ocean-only day-0 SST field.

        The generic coupled driver seeds the slab from
        ``atm.get_sst_sic(0.0)``, which is correct for standard AMIP-style
        atmosphere forcing but wrong for this land-aware initialized S2S
        workflow because ``get_sst_sic`` is intentionally blended with land
        skin temperature over land points.  The slab ocean must start from the
        ocean-only SST field, not the blended lower-boundary temperature.
        """
        from legoesm.core.field import Field
        from legoesm.ocean.simple_ocean import init_slab_state, make_ocean

        cfg = self.coupled_cfg
        shape_2d = self._atm.grid.grid_shape_2d

        sst_init = jnp.asarray(
            self._surface_forcing.fixed_ocean_surface_temperature[0],
            dtype=jnp.float32,
        )
        T_sfc_mean = float(jnp.mean(sst_init))

        self._ocean_state = init_slab_state(shape_2d, T_sfc_init=T_sfc_mean)
        self._ocean_state = self._ocean_state._replace(
            T_sfc=Field(
                data=jnp.asarray(
                    sst_init, dtype=self._ocean_state.T_sfc.data.dtype,
                ),
                name="T_sfc",
                dims=self._ocean_state.T_sfc.dims,
                units="K",
            ),
        )
        self._ocean_step = make_ocean(cfg.ocean_config)

    def _override_sst(self) -> None:
        """Use slab SST over ocean and prescribed land skin temperature over land."""
        self._original_get_sst_sic = self._atm.get_sst_sic
        ocean_getter = lambda day: self._ocean_state.T_sfc.data
        self._atm.get_sst_sic = build_surface_boundary_getter(
            self._surface_forcing,
            ocean_temperature_getter=ocean_getter,
        )

    def _segment_hook(self, driver, day, dt_segment):
        super()._segment_hook(driver, day, dt_segment)
        if self._collector is not None:
            self._collector.record(self._atm, day, dt_segment, ocean_state=self._ocean_state)

    def run(
        self,
        start_step: int = 0,
        start_day: float | None = None,
        *,
        compiled: bool = True,
    ) -> str:
        """Run the coupled integration with optional uncompiled atmosphere stepping."""
        return self._atm.run(
            start_step=start_step,
            start_day=start_day,
            compiled=compiled,
            segment_callback=self._segment_hook,
        )


def _build_uncoupled_driver(
    config: ExperimentConfig,
    *,
    surface_forcing: ModelSurfaceForcing,
    output_dir: str | Path,
) -> ModelDriver:
    driver = ModelDriver(config, output_dir=output_dir)
    # Initialized S2S uncoupled runs must inject the prepared land mask
    # before setup so the atmosphere physics path does not bootstrap with
    # the flat-topography ocean-everywhere default.
    driver._f_land = surface_forcing.land_fraction
    driver.setup()
    driver._f_land = surface_forcing.land_fraction
    if getattr(driver, "physics", None) is not None:
        driver.physics.land_fraction = surface_forcing.land_fraction
    driver.get_sst_sic = build_surface_boundary_getter(surface_forcing)
    return driver


def _build_coupled_driver(
    config: ExperimentConfig,
    *,
    forcing_start_day: float,
    collector: DailyForecastCollector | None,
    coupled_metadata: dict[str, Any],
    surface_forcing: ModelSurfaceForcing,
    output_dir: str | Path,
) -> InitializedCoupledSlabDriver:
    ocean_h_mix = float(coupled_metadata.get("ocean_h_mix", 50.0))
    coupling_dt = float(coupled_metadata.get("coupling_dt", 3600.0))
    coupled_config = CoupledConfig(
        ocean_mode="slab",
        ocean_config=SimpleOceanConfig(mode="slab", h_mix=ocean_h_mix),
        land_mode=str(coupled_metadata.get("land_mode", "none")),
        f_land_mode=str(coupled_metadata.get("f_land_mode", "zero")),
        coupling_dt=coupling_dt,
    )
    driver = InitializedCoupledSlabDriver(
        config,
        forcing_start_day=forcing_start_day,
        collector=collector,
        coupled_config=coupled_config,
        surface_forcing=surface_forcing,
        output_dir=output_dir,
    )
    # Initialized S2S slab runs carry their own land mask from ARCO/ERA5
    # preparation. Inject it before setup so both the atmosphere physics
    # pipeline and the coupled tile fractions see the same land geometry.
    driver._atm._f_land = surface_forcing.land_fraction
    driver.setup()
    if getattr(driver._atm, "physics", None) is not None:
        driver._atm.physics.land_fraction = surface_forcing.land_fraction
    return driver


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
    return path


def _run_experiment(
    experiment: str,
    *,
    case_dir: Path,
    metadata: PreparedCaseMetadata,
    export_config: ForecastExportConfig,
    compiled: bool,
) -> tuple[str, Path]:
    config_path = case_dir / "_prepared" / "experiment_config.json"
    if config_path.exists():
        config = experiment_config_from_dict(
            json.loads(config_path.read_text(encoding="utf-8"))
        )
    else:
        config = experiment_config_from_dict(metadata.experiment_config)
    output_dir = case_dir / experiment
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_name in ("forecast.nc", "run_summary.json"):
        stale_path = output_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()
    model_surface_forcing = _load_model_surface_forcing(case_dir / "_prepared" / "surface_forcing_model.nc")
    prepared_forcing_path = case_dir / "_prepared" / surface_forcing_filename(
        start_time=metadata.start_time,
        forecast_days=int(metadata.forecast_days),
    )
    config = config._replace(
        output=config.output._replace(output_dir=str(output_dir), diag_days=1),
        start_day=float(metadata.forcing_start_day),
        days=int(metadata.forecast_days),
        dataset="custom",
        forcing_path=str(prepared_forcing_path),
        sst_var="sea_surface_temperature",
        sic_var="sea_ice_cover",
        time_var="time",
        lat_var="latitude",
        lon_var="longitude",
    )
    if experiment == "coupled":
        driver = _build_coupled_driver(
            config,
            forcing_start_day=float(metadata.forcing_start_day),
            collector=None,
            coupled_metadata=metadata.coupled,
            surface_forcing=model_surface_forcing,
            output_dir=output_dir,
        )
        collector = DailyForecastCollector(
            grid=driver._atm.grid,
            sigma=driver._atm.sigma,
            start_time=metadata.start_time,
            forecast_days=int(metadata.forecast_days),
            forcing_start_day=float(metadata.forcing_start_day),
            config=export_config,
        )
        driver._collector = collector
    else:
        driver = _build_uncoupled_driver(
            config,
            surface_forcing=model_surface_forcing,
            output_dir=output_dir,
        )
        collector = DailyForecastCollector(
            grid=driver.grid,
            sigma=driver.sigma,
            start_time=metadata.start_time,
            forecast_days=int(metadata.forecast_days),
            forcing_start_day=float(metadata.forcing_start_day),
            config=export_config,
        )

    restart_path = case_dir / "_prepared" / "initial_restart.npz"
    start_step, start_day = (
        driver._atm.load_checkpoint(restart_path)
        if experiment == "coupled"
        else driver.load_checkpoint(restart_path)
    )

    if experiment == "coupled":
        status = driver.run(start_step=start_step, start_day=start_day, compiled=compiled)
    else:
        status = driver.run(
            start_step=start_step,
            start_day=start_day,
            compiled=compiled,
            segment_callback=lambda active_driver, day, dt_segment: collector.record(
                active_driver,
                day,
                dt_segment,
            ),
        )

    forecast_path: Path | None = None
    if collector.has_records():
        forecast_path = output_dir / "forecast.nc"
        dataset = collector.to_dataset()
        dataset.attrs["experiment"] = experiment
        dataset.attrs["run_status"] = status
        dataset.to_netcdf(forecast_path)
    _write_json(
        output_dir / "run_summary.json",
        {
            "experiment": experiment,
            "run_status": status,
            "forecast_path": str(forecast_path) if forecast_path is not None else None,
        },
    )
    return status, forecast_path


def run_legoesm_case(
    *,
    case_dir: str | Path,
    experiments: Sequence[str] = ("coupled", "uncoupled"),
    export_config: ForecastExportConfig | None = None,
    compiled: bool = False,
) -> dict[str, Path]:
    """Run the paired coupled-vs-uncoupled initialized legoESM slab case."""
    case_dir = Path(case_dir)
    metadata = load_case_metadata(case_dir / "_prepared" / "metadata.json")
    export = export_config or ForecastExportConfig(
        pressure_levels=tuple(metadata.pressure_levels),
        evaluation_resolution_deg=float(metadata.evaluation_resolution_deg),
    )
    requested = tuple(experiments)
    outputs: dict[str, Path] = {}
    statuses: dict[str, str] = {}
    for experiment in ("coupled", "uncoupled"):
        summary_path = case_dir / experiment / "run_summary.json"
        if summary_path.exists():
            with summary_path.open("r", encoding="utf-8") as handle:
                statuses[experiment] = json.load(handle).get("run_status", "UNKNOWN")
    for experiment in requested:
        if experiment not in {"coupled", "uncoupled"}:
            raise ValueError(f"Unknown experiment {experiment!r}; expected coupled or uncoupled")
        status, forecast_path = _run_experiment(
            experiment,
            case_dir=case_dir,
            metadata=metadata,
            export_config=export,
            compiled=compiled,
        )
        if forecast_path is not None:
            outputs[f"{experiment}_forecast"] = forecast_path
        statuses[experiment] = status

    summary = CaseRunSummary(
        start_time=metadata.start_time,
        forecast_days=int(metadata.forecast_days),
        forcing_start_day=float(metadata.forcing_start_day),
        coupled_status=statuses.get("coupled", "SKIPPED"),
        uncoupled_status=statuses.get("uncoupled", "SKIPPED"),
        pressure_levels=tuple(export.pressure_levels),
        evaluation_resolution_deg=float(export.evaluation_resolution_deg),
    )
    outputs["run_summary"] = _write_json(case_dir / "run_summary.json", asdict(summary))
    return outputs


__all__ = [
    "CaseRunSummary",
    "DEFAULT_EXPORT_PRESSURE_LEVELS",
    "DailyForecastCollector",
    "ForecastExportConfig",
    "InitializedCoupledSlabDriver",
    "run_legoesm_case",
    "wrap_relative_forcing_getter",
]
