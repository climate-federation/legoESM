"""Daily slab-ocean coupling for NeuralGCM forecasts."""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from typing import Any, Protocol

import jax.numpy as jnp
import numpy as np
import xarray as xr

from legoesm import constants
from legoesm.core.field import Field
from legoesm.core.coupling_fields import AtmToSurface
from legoesm.ocean.simple_ocean import SimpleOceanConfig, SlabOceanState, make_ocean


@dataclass(frozen=True)
class CoupledFieldNames:
    """Canonical variable names for the coupled NeuralGCM workflow."""

    sst: str = "sea_surface_temperature"
    sea_ice: str = "sea_ice_cover"
    surface_pressure: str = "surface_pressure"
    temperature: str = "temperature"
    specific_humidity: str = "specific_humidity"
    u_component_of_wind: str = "u_component_of_wind"
    v_component_of_wind: str = "v_component_of_wind"
    sw_down: str = "sw_down"
    lw_down: str = "lw_down"
    sst_tendency: str = "sst_tendency"
    ocean_u_sfc: str = "ocean_u_sfc"
    ocean_v_sfc: str = "ocean_v_sfc"


@dataclass(frozen=True)
class SlabCouplingConfig:
    """Configuration for daily slab-ocean coupling."""

    ocean: SimpleOceanConfig = dataclass_field(
        default_factory=lambda: SimpleOceanConfig(mode="slab")
    )
    field_names: CoupledFieldNames = dataclass_field(default_factory=CoupledFieldNames)
    lowest_level_hpa: int = 1000
    dt_seconds: float = 86400.0
    sea_ice_threshold: float = 0.15


class AtmosphereBackend(Protocol):
    """Protocol for the backend methods used by the coupler."""

    config: Any

    def prepare_initial_state(self, dataset: xr.Dataset): ...

    def build_forcing_dataset(
        self,
        template: xr.Dataset,
        sea_surface_temperature: xr.DataArray,
        sea_ice_cover: xr.DataArray,
    ) -> xr.Dataset: ...

    def run_one_day(self, state, forcing_dataset: xr.Dataset, *, start_with_input: bool = True): ...


def _drop_singleton_time(
    data: xr.Dataset | xr.DataArray,
    time_name: str,
) -> xr.Dataset | xr.DataArray:
    if time_name in data.dims and int(data.sizes.get(time_name, 0)) == 1:
        return data.squeeze(time_name, drop=True)
    return data


def _ensure_dataarray(value: xr.DataArray | np.ndarray | float, *, name: str) -> xr.DataArray:
    if isinstance(value, xr.DataArray):
        return value.rename(name)
    return xr.DataArray(np.asarray(value), name=name)


def _assign_coords_like(
    value: xr.DataArray | np.ndarray | float,
    reference: xr.DataArray,
    *,
    name: str,
) -> xr.DataArray:
    array = _ensure_dataarray(value, name=name)
    data = np.asarray(array, dtype=float)
    if data.shape == reference.shape:
        return xr.DataArray(data, coords=reference.coords, dims=reference.dims, name=name)
    if data.shape == ():
        data = np.broadcast_to(data, reference.shape)
        return xr.DataArray(data, coords=reference.coords, dims=reference.dims, name=name)
    broadcast = np.broadcast_to(data, reference.shape)
    return xr.DataArray(broadcast, coords=reference.coords, dims=reference.dims, name=name)


def _coerce_to_reference(
    value: xr.DataArray | np.ndarray | float,
    reference: xr.DataArray,
    *,
    name: str,
) -> xr.DataArray:
    array = _ensure_dataarray(value, name=name)
    if set(reference.dims).issubset(set(array.dims)):
        selection = {dim: reference[dim] for dim in reference.dims}
        return array.sel(selection, method="nearest").transpose(*reference.dims).rename(name)
    return _assign_coords_like(array, reference, name=name)


def _align_dataset_to_reference_coords(
    dataset: xr.Dataset,
    reference: xr.DataArray,
) -> xr.Dataset:
    aligned = dataset
    for dim in reference.dims:
        if dim not in aligned.dims:
            continue
        ref_coord = reference[dim]
        if int(aligned.sizes[dim]) == int(ref_coord.size):
            aligned = aligned.assign_coords({dim: ref_coord})
        else:
            aligned = aligned.sel({dim: ref_coord}, method="nearest")
    return aligned


def _select_daily(
    data: xr.Dataset | xr.DataArray,
    day_index: int,
    *,
    time_name: str,
) -> xr.Dataset | xr.DataArray:
    if time_name not in data.dims:
        return data
    return data.isel({time_name: slice(day_index, day_index + 1)})


def init_slab_state_from_field(
    sea_surface_temperature: xr.DataArray,
    *,
    deep_temperature: float,
) -> SlabOceanState:
    data = jnp.asarray(np.asarray(sea_surface_temperature, dtype=np.float32))
    dims = tuple(str(dim) for dim in sea_surface_temperature.dims)
    deep = jnp.full(data.shape, float(deep_temperature), dtype=data.dtype)
    return SlabOceanState(
        T_sfc=Field(data, name="T_sfc", dims=dims, units="K"),
        T_deep=Field(deep, name="T_deep", dims=dims, units="K"),
    )


def apply_sea_ice_mask(
    sea_surface_temperature: xr.DataArray,
    sea_ice_cover: xr.DataArray,
    *,
    threshold: float,
    freeze_temperature: float = constants.T_freeze,
) -> xr.DataArray:
    masked = xr.where(sea_ice_cover >= float(threshold), float(freeze_temperature), sea_surface_temperature)
    return masked.rename(sea_surface_temperature.name)


def extract_surface_forcing_from_dataset(
    atmospheric_output: xr.Dataset,
    *,
    radiation_dataset: xr.Dataset | None = None,
    surface_pressure: xr.DataArray | None = None,
    config: SlabCouplingConfig = SlabCouplingConfig(),
) -> AtmToSurface:
    field_names = config.field_names
    level_name = "level"
    if hasattr(atmospheric_output, "dims") and field_names.temperature in atmospheric_output:
        level_name = (
            "level"
            if "level" in atmospheric_output[field_names.temperature].dims
            else level_name
        )

    temperature = atmospheric_output[field_names.temperature].sel(
        {level_name: config.lowest_level_hpa},
        method="nearest",
    )
    specific_humidity = atmospheric_output[field_names.specific_humidity].sel(
        {level_name: config.lowest_level_hpa},
        method="nearest",
    )
    u_wind = atmospheric_output[field_names.u_component_of_wind].sel(
        {level_name: config.lowest_level_hpa},
        method="nearest",
    )
    v_wind = atmospheric_output[field_names.v_component_of_wind].sel(
        {level_name: config.lowest_level_hpa},
        method="nearest",
    )

    level_value = float(temperature[level_name].item())
    p_lowest_pa = 100.0 * level_value if abs(level_value) < 2_000.0 else level_value
    p_lowest = _assign_coords_like(p_lowest_pa, temperature, name="p_lowest")
    if surface_pressure is None:
        if field_names.surface_pressure not in atmospheric_output:
            raise ValueError(
                "surface pressure is required: pass surface_pressure= or include "
                f"{field_names.surface_pressure!r} in the atmospheric output "
                "(no fallback to the lowest model level)."
            )
        surface_pressure = atmospheric_output[field_names.surface_pressure]
    p_surface = _coerce_to_reference(surface_pressure, temperature, name="p_surface")

    if radiation_dataset is None:
        raise ValueError(
            "radiation_dataset (sw_down, lw_down) is required; surface radiation "
            "is not zero-filled."
        )
    sw_down = _coerce_to_reference(
        _drop_singleton_time(radiation_dataset[field_names.sw_down], "time"),
        temperature,
        name=field_names.sw_down,
    )
    lw_down = _coerce_to_reference(
        _drop_singleton_time(radiation_dataset[field_names.lw_down], "time"),
        temperature,
        name=field_names.lw_down,
    )

    temperature_np = np.asarray(temperature, dtype=float)
    humidity_np = np.asarray(specific_humidity, dtype=float)
    # Canonical virtual-temperature coefficient 1/epsilon - 1 (~0.608), not the
    # rounded 0.61 literal (~0.4% drift) — matches atmosphere.physics._shared.
    virtual_temperature = temperature_np * (1.0 + (1.0 / constants.epsilon - 1.0) * humidity_np)
    rho_lowest = _assign_coords_like(
        np.asarray(p_lowest, dtype=float) / (constants.R_d * virtual_temperature),
        temperature,
        name="rho_lowest",
    )
    zeros = xr.zeros_like(temperature)
    ones = xr.ones_like(temperature)

    return AtmToSurface(
        sw_down=jnp.asarray(np.asarray(sw_down, dtype=np.float32)),
        lw_down=jnp.asarray(np.asarray(lw_down, dtype=np.float32)),
        precip_total=jnp.asarray(np.asarray(zeros, dtype=np.float32)),
        precip_snow=jnp.asarray(np.asarray(zeros, dtype=np.float32)),
        T_lowest=jnp.asarray(np.asarray(temperature, dtype=np.float32)),
        q_lowest=jnp.asarray(np.asarray(specific_humidity, dtype=np.float32)),
        u_lowest=jnp.asarray(np.asarray(u_wind, dtype=np.float32)),
        v_lowest=jnp.asarray(np.asarray(v_wind, dtype=np.float32)),
        p_lowest=jnp.asarray(np.asarray(p_lowest, dtype=np.float32)),
        p_surface=jnp.asarray(np.asarray(p_surface, dtype=np.float32)),
        rho_lowest=jnp.asarray(np.asarray(rho_lowest, dtype=np.float32)),
        cos_zenith=jnp.asarray(np.asarray(ones, dtype=np.float32)),
        co2_ppmv=jnp.asarray(np.asarray(420.0 * ones, dtype=np.float32)),
        has_radiation=jnp.asarray(np.asarray(ones, dtype=np.float32)),
        has_precipitation=jnp.asarray(np.asarray(zeros, dtype=np.float32)),
    )


def advance_slab_ocean_one_day(
    state: SlabOceanState,
    forcing: AtmToSurface,
    *,
    reference_sst: xr.DataArray,
    sea_ice_cover: xr.DataArray,
    config: SlabCouplingConfig = SlabCouplingConfig(),
) -> tuple[SlabOceanState, xr.Dataset]:
    ocean_step = make_ocean(config.ocean)
    next_state, next_sst, u_sfc, v_sfc = ocean_step(state, forcing, float(config.dt_seconds))
    masked_sst = apply_sea_ice_mask(
        _assign_coords_like(next_sst, reference_sst, name=config.field_names.sst),
        sea_ice_cover,
        threshold=config.sea_ice_threshold,
        freeze_temperature=config.ocean.T_freeze,
    )
    tendency = (np.asarray(masked_sst, dtype=float) - np.asarray(reference_sst, dtype=float)) / float(
        config.dt_seconds
    )
    tendency_field = _assign_coords_like(
        tendency,
        reference_sst,
        name=config.field_names.sst_tendency,
    )
    u_field = _assign_coords_like(u_sfc, reference_sst, name=config.field_names.ocean_u_sfc)
    v_field = _assign_coords_like(v_sfc, reference_sst, name=config.field_names.ocean_v_sfc)
    updated_state = next_state._replace(
        T_sfc=next_state.T_sfc.replace(data=jnp.asarray(np.asarray(masked_sst, dtype=np.float32)))
    )
    diagnostics = xr.Dataset(
        {
            config.field_names.sst: masked_sst,
            config.field_names.sst_tendency: tendency_field,
            config.field_names.ocean_u_sfc: u_field,
            config.field_names.ocean_v_sfc: v_field,
        }
    )
    return updated_state, diagnostics


def rollout_coupled_daily(
    backend: AtmosphereBackend,
    initial_dataset: xr.Dataset,
    forcing_dataset: xr.Dataset,
    radiation_dataset: xr.Dataset,
    *,
    forecast_days: int,
    config: SlabCouplingConfig = SlabCouplingConfig(),
    coupled: bool,
) -> xr.Dataset:
    """Run a daily coupled or fixed-SST forecast."""
    time_name = getattr(backend.config, "time_name", "time")
    field_names = config.field_names

    prepared_initial = initial_dataset.copy(deep=True)
    state = backend.prepare_initial_state(prepared_initial)

    first_forcing = _select_daily(forcing_dataset, 0, time_name=time_name)
    first_sea_ice = _drop_singleton_time(first_forcing[field_names.sea_ice], time_name)
    fixed_sea_ice = _coerce_to_reference(
        first_sea_ice,
        _drop_singleton_time(prepared_initial[field_names.sst], time_name),
        name=field_names.sea_ice,
    )
    initial_sst = apply_sea_ice_mask(
        _drop_singleton_time(prepared_initial[field_names.sst], time_name),
        fixed_sea_ice,
        threshold=config.sea_ice_threshold,
        freeze_temperature=config.ocean.T_freeze,
    )
    fixed_sst = initial_sst.copy(deep=True)
    current_sst = initial_sst.copy(deep=True)
    ocean_state = init_slab_state_from_field(
        current_sst,
        deep_temperature=float(config.ocean.T_deep_ref),
    )

    outputs: list[xr.Dataset] = []
    valid_times: list[np.datetime64] = []

    for day_index in range(int(forecast_days)):
        forcing_daily = _select_daily(forcing_dataset, day_index, time_name=time_name)
        radiation_daily = _select_daily(radiation_dataset, day_index, time_name=time_name)
        sea_ice = _coerce_to_reference(
            _drop_singleton_time(forcing_daily[field_names.sea_ice], time_name),
            current_sst,
            name=field_names.sea_ice,
        )
        forcing_sea_ice = sea_ice if coupled else fixed_sea_ice
        forcing_sst = apply_sea_ice_mask(
            current_sst if coupled else fixed_sst,
            forcing_sea_ice,
            threshold=config.sea_ice_threshold,
            freeze_temperature=config.ocean.T_freeze,
        )
        forcing_input = backend.build_forcing_dataset(
            xr.merge([forcing_daily, radiation_daily], compat="override"),
            forcing_sst,
            forcing_sea_ice,
        )
        state, atmospheric_output = backend.run_one_day(state, forcing_input)
        atmospheric_output = _drop_singleton_time(atmospheric_output, time_name)
        if time_name in atmospheric_output.coords and time_name not in atmospheric_output.dims:
            atmospheric_output = atmospheric_output.drop_vars(time_name)
        atmospheric_output = _align_dataset_to_reference_coords(
            atmospheric_output,
            current_sst,
        )

        surface_pressure = None
        if field_names.surface_pressure in forcing_daily.data_vars:
            surface_pressure = _drop_singleton_time(
                forcing_daily[field_names.surface_pressure],
                time_name,
            )

        if coupled:
            surface_forcing = extract_surface_forcing_from_dataset(
                atmospheric_output,
                radiation_dataset=radiation_daily,
                surface_pressure=surface_pressure,
                config=config,
            )
            ocean_state, ocean_output = advance_slab_ocean_one_day(
                ocean_state,
                surface_forcing,
                reference_sst=forcing_sst,
                sea_ice_cover=sea_ice,
                config=config,
            )
            current_sst = ocean_output[field_names.sst]
        else:
            zeros = xr.zeros_like(fixed_sst)
            ocean_output = xr.Dataset(
                {
                    field_names.sst: fixed_sst.copy(deep=True),
                    field_names.sst_tendency: zeros.rename(field_names.sst_tendency),
                    field_names.ocean_u_sfc: zeros.rename(field_names.ocean_u_sfc),
                    field_names.ocean_v_sfc: zeros.rename(field_names.ocean_v_sfc),
                }
            )
            current_sst = fixed_sst.copy(deep=True)

        combined = xr.merge(
            [atmospheric_output, ocean_output],
            compat="override",
            join="override",
        )
        combined = combined.expand_dims({"lead_day": [day_index + 1]})
        outputs.append(combined)

        if time_name in forcing_daily.coords:
            start_value = np.asarray(forcing_daily[time_name]).reshape(-1)[0]
            valid_times.append(np.datetime64(start_value) + np.timedelta64(1, "D"))

    forecast = xr.concat(outputs, dim="lead_day")
    forecast = forecast.assign_coords({"lead_day": np.arange(1, int(forecast_days) + 1, dtype=int)})
    if len(valid_times) == int(forecast_days):
        forecast = forecast.assign_coords({"valid_time": ("lead_day", np.asarray(valid_times))})
    forecast.attrs["experiment"] = "coupled" if coupled else "uncoupled"
    forecast.attrs["forecast_days"] = int(forecast_days)
    return forecast


__all__ = [
    "AtmosphereBackend",
    "CoupledFieldNames",
    "SlabCouplingConfig",
    "advance_slab_ocean_one_day",
    "apply_sea_ice_mask",
    "extract_surface_forcing_from_dataset",
    "init_slab_state_from_field",
    "rollout_coupled_daily",
]
