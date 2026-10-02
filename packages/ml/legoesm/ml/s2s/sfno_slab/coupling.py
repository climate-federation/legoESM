"""Daily slab-ocean coupling for the local SFNO S2S workflow."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr

from legoesm import constants
from legoesm.core.field import Field
from legoesm.core.coupling_fields import AtmToSurface
from legoesm.ml.s2s.sfno_slab.data import (
    atmospheric_param_labels,
    denormalize_atmospheric_channels,
    denormalize_forcing_channels,
    forcing_channel_labels,
    load_normalization_bundle,
    load_s2s_sample,
    load_surface_sequence,
    normalize_forcing_channels,
    resolve_s2s_sample_dates,
)
from legoesm.ml.s2s.neuralgcm_slab.slab_coupling import (
    SlabCouplingConfig,
    extract_surface_forcing_from_dataset,
)
from legoesm.ml.s2s.sfno_slab.regrid import build_target_grid
from legoesm.ml.s2s.sfno_slab.training import S2SStochasticConfig, predict_next_atmosphere
from legoesm.ocean.simple_ocean import SimpleOceanConfig, SlabOceanState, make_ocean


@dataclass(frozen=True)
class S2SSlabCouplingConfig:
    """Configuration for daily SFNO slab coupling."""

    ocean: SimpleOceanConfig = SimpleOceanConfig(mode="slab")
    lowest_level_hpa: int = 1000
    dt_seconds: float = 86400.0
    sea_ice_threshold: float = 0.15
    sw_down_var: str = "ssrd"
    lw_down_var: str = "strd"
    surface_pressure_var: str = "sp"
    sea_ice_var: str = "ileadfra"
    sea_ice_thickness_var: str = "iicethic"


def _channel_index(labels: list[str], name: str) -> int:
    if name not in labels:
        raise KeyError(f"Required channel {name!r} not found.")
    return labels.index(name)


def _fill_nan_with_mean(field: np.ndarray, *, default: float) -> np.ndarray:
    values = np.asarray(field, dtype=np.float32)
    if np.isfinite(values).all():
        return values
    finite = values[np.isfinite(values)]
    fill_value = float(finite.mean()) if finite.size else float(default)
    return np.where(np.isfinite(values), values, fill_value).astype(np.float32, copy=False)


def _sea_ice_cover_from_oras(
    leadfra: np.ndarray | None,
    thickness: np.ndarray | None,
) -> np.ndarray | None:
    del leadfra
    if thickness is None:
        return None
    cover = (np.asarray(thickness, dtype=np.float32) > 0.05).astype(np.float32)
    return np.where(np.isfinite(cover), np.clip(cover, 0.0, 1.0), 0.0).astype(np.float32, copy=False)


def _sst_state_from_masked_array(
    sst: np.ndarray,
    ocean_mask: np.ndarray,
    *,
    freeze_temperature: float,
) -> np.ndarray:
    return np.where(ocean_mask, np.asarray(sst, dtype=np.float32), np.float32(freeze_temperature))


def _masked_output(field: np.ndarray, ocean_mask: np.ndarray) -> np.ndarray:
    return np.where(ocean_mask, np.asarray(field, dtype=np.float32), np.nan).astype(np.float32, copy=False)


def _sst_uses_celsius_units(sst: np.ndarray, ocean_mask: np.ndarray) -> bool:
    ocean_values = np.asarray(sst, dtype=np.float32)[ocean_mask]
    if ocean_values.size == 0:
        return False
    return float(np.nanmean(ocean_values)) < 200.0


def _sst_to_ocean_units(sst: np.ndarray, *, uses_celsius: bool) -> np.ndarray:
    values = np.asarray(sst, dtype=np.float32)
    return values + np.float32(constants.T_freeze) if uses_celsius else values


def _sst_from_ocean_units(sst: np.ndarray, *, uses_celsius: bool) -> np.ndarray:
    values = np.asarray(sst, dtype=np.float32)
    return values - np.float32(constants.T_freeze) if uses_celsius else values


def _build_atm_to_surface(
    prediction_physical: np.ndarray,
    aux_surface: dict[str, np.ndarray],
    *,
    channel_labels: list[str],
    pressure_levels: tuple[int, ...],
    config: S2SSlabCouplingConfig,
) -> AtmToSurface:
    """Slab-ocean forcing from one SFNO prediction via the shared S2S builder.

    Packs the lowest-level T/q/u/v channels and the surface fields into the
    datasets :func:`extract_surface_forcing_from_dataset` (the NeuralGCM-slab
    builder) consumes: density from the virtual temperature at the model-level
    pressure.  A missing channel or surface field, or a non-finite value in
    one, raises; nothing is filled.  Precipitation is not an SFNO output, so the
    builder reports none (has_precipitation=0); the slab ocean does not read it.
    """
    level = min(pressure_levels, key=lambda lev: abs(lev - config.lowest_level_hpa))
    for name in ("sp", "sw_down", "lw_down"):
        if not np.isfinite(aux_surface[name]).all():
            raise ValueError(f"non-finite surface field {name!r} in SFNO slab forcing")
    shape = prediction_physical.shape[:-1]
    coords = {"latitude": np.arange(shape[0]), "longitude": np.arange(shape[1])}
    names = SlabCouplingConfig().field_names
    atmosphere = xr.Dataset(
        {
            name: (("level", "latitude", "longitude"),
                   prediction_physical[None, ..., _channel_index(channel_labels, f"{var}-{level}")])
            for var, name in (
                ("t", names.temperature),
                ("q", names.specific_humidity),
                ("u", names.u_component_of_wind),
                ("v", names.v_component_of_wind),
            )
        },
        coords={"level": [level], **coords},
    )
    if not all(np.isfinite(da.values).all() for da in atmosphere.data_vars.values()):
        raise ValueError(f"non-finite lowest-level channel (level {level}) in SFNO slab forcing")
    radiation = xr.Dataset(
        {
            names.sw_down: (("latitude", "longitude"), aux_surface["sw_down"]),
            names.lw_down: (("latitude", "longitude"), aux_surface["lw_down"]),
        },
        coords=coords,
    )
    return extract_surface_forcing_from_dataset(
        atmosphere,
        radiation_dataset=radiation,
        surface_pressure=aux_surface["sp"],
        config=SlabCouplingConfig(lowest_level_hpa=config.lowest_level_hpa),
    )


def coupled_rollout_to_dataset(
    model,
    data_config,
    *,
    sample_index: int,
    coupled: bool = True,
    surface_forcing: xr.Dataset | None = None,
    config: S2SSlabCouplingConfig = S2SSlabCouplingConfig(),
    stochastic_seed: int = 0,
    stochastic_config: S2SStochasticConfig = S2SStochasticConfig(),
    tendency_prediction: bool = False,
) -> xr.Dataset:
    """Run a daily coupled or uncoupled rollout using the slab ocean.

    The ``uncoupled`` branch is a fixed-SST control:
    - ocean points keep the initial-condition SST for the full rollout
    - land points are filled with a finite reference value before normalization

    This avoids feeding land NaNs back into the SFNO input state while still
    preserving an ocean-only SST diagnostic field in the saved dataset.
    """
    target_grid = build_target_grid(data_config.gaussian_n_max)
    norms = load_normalization_bundle(data_config)
    channel_labels = atmospheric_param_labels(data_config)
    input_date, target_dates = resolve_s2s_sample_dates(data_config, sample_index=sample_index)
    initial_input, target, true_forcing = load_s2s_sample(
        data_config,
        sample_index=sample_index,
        target_grid=target_grid,
        norms=norms,
    )

    forcing_labels = forcing_channel_labels(data_config)
    if "sosstsst" not in forcing_labels:
        raise KeyError("Coupled rollout requires ocean forcing channel 'sosstsst'.")
    sst_index = forcing_labels.index("sosstsst")

    if surface_forcing is None:
        lra5 = load_surface_sequence(
            data_config,
            dates=target_dates,
            collection="lra5",
            variables=(config.surface_pressure_var, config.sw_down_var, config.lw_down_var),
            target_grid=target_grid,
        )
        oras5 = load_surface_sequence(
            data_config,
            dates=target_dates,
            collection="oras5",
            variables=(config.sea_ice_var, config.sea_ice_thickness_var),
            target_grid=target_grid,
        )
        surface_source = "chaosbench"
    else:
        lra5 = None
        oras5 = None
        surface_source = str(surface_forcing.attrs.get("surface_source", "external"))

    initial_atmosphere = initial_input[..., : target.shape[-1]]
    initial_forcing = initial_input[..., target.shape[-1] :]
    current_state = np.asarray(initial_input, dtype=np.float32)
    current_forcing = np.asarray(initial_forcing, dtype=np.float32)
    current_forcing_physical = denormalize_forcing_channels(current_forcing, data_config, norms=norms)
    target_forcing_physical = denormalize_forcing_channels(true_forcing, data_config, norms=norms)

    surface_sst_raw = None
    surface_sst_filled = None
    if surface_forcing is not None:
        surface_forcing = surface_forcing.copy()
        surface_sst_raw = np.asarray(surface_forcing["sea_surface_temperature"].values, dtype=np.float32)
        surface_sst_filled = np.stack(
            [_fill_nan_with_mean(surface_sst_raw[day], default=0.0) for day in range(surface_sst_raw.shape[0])],
            axis=0,
        )
        initial_sst_raw = np.asarray(surface_forcing["initial_sea_surface_temperature"].values, dtype=np.float32)
        initial_sst_filled = _fill_nan_with_mean(initial_sst_raw, default=0.0)
        current_forcing_physical[..., sst_index] = initial_sst_filled
        target_forcing_physical = np.asarray(target_forcing_physical, dtype=np.float32).copy()
        target_forcing_physical[..., sst_index] = surface_sst_filled
        current_forcing = normalize_forcing_channels(current_forcing_physical, data_config, norms=norms)
        current_state = np.concatenate([initial_atmosphere, current_forcing], axis=-1).astype(np.float32, copy=False)
        current_sst = initial_sst_raw
    else:
        current_sst = np.asarray(current_forcing_physical[..., sst_index], dtype=np.float32)
    target_physical = denormalize_atmospheric_channels(target, data_config, norms=norms)
    initial_atmosphere_physical = denormalize_atmospheric_channels(initial_atmosphere, data_config, norms=norms)
    initial_forcing_physical = np.asarray(current_forcing_physical, dtype=np.float32)

    ocean_mask = np.isfinite(current_sst)
    sst_uses_celsius = _sst_uses_celsius_units(current_sst, ocean_mask)
    current_sst_ocean = _sst_to_ocean_units(current_sst, uses_celsius=sst_uses_celsius)
    # Keep a finite SST field for the uncoupled control. The diagnostic output
    # is still masked back to ocean points later, but the model input itself
    # must remain finite everywhere.
    fixed_uncoupled_sst_state = _sst_state_from_masked_array(
        current_sst_ocean,
        ocean_mask,
        freeze_temperature=config.ocean.T_freeze,
    )
    fixed_uncoupled_sst = _sst_from_ocean_units(
        fixed_uncoupled_sst_state,
        uses_celsius=sst_uses_celsius,
    )
    current_sst_state = _sst_state_from_masked_array(
        current_sst_ocean,
        ocean_mask,
        freeze_temperature=config.ocean.T_freeze,
    )
    ocean_state = SlabOceanState(
        T_sfc=Field(jnp.asarray(current_sst_state, dtype=float), name="T_sfc", dims=("latitude", "longitude"), units="K"),
        T_deep=Field(
            jnp.full_like(jnp.asarray(current_sst_state, dtype=float), config.ocean.T_deep_ref),
            name="T_deep",
            dims=("latitude", "longitude"),
            units="K",
        ),
    )
    ocean_step = make_ocean(config.ocean)

    predictions: list[np.ndarray] = []
    forcings_used: list[np.ndarray] = []
    sst_series: list[np.ndarray] = []
    step_rng = jax.random.PRNGKey(int(stochastic_seed))

    for day in range(target.shape[0]):
        step_rng, day_key = jax.random.split(step_rng)
        prediction = np.asarray(
            predict_next_atmosphere(
                model,
                jnp.asarray(current_state[..., : target.shape[-1]], dtype=jnp.float32),
                jnp.asarray(current_state[..., target.shape[-1] :], dtype=jnp.float32),
                target_grid.grid,
            stochastic_config=stochastic_config,
            rng_key=day_key,
            lead_index=day,
            n_steps=target.shape[0],
            tendency_prediction=tendency_prediction,
        ),
            dtype=np.float32,
        )
        prediction_physical = denormalize_atmospheric_channels(prediction, data_config, norms=norms)

        if surface_forcing is None:
            sp = lra5[day, ..., 0]
            sw_down = lra5[day, ..., 1] / np.float32(config.dt_seconds)
            lw_down = lra5[day, ..., 2] / np.float32(config.dt_seconds)
            leadfra = oras5[day, ..., 0] if oras5.shape[-1] >= 1 else None
            thickness = oras5[day, ..., 1] if oras5.shape[-1] >= 2 else None
            sea_ice_cover = _sea_ice_cover_from_oras(leadfra, thickness)
        else:
            sp = np.asarray(surface_forcing["surface_pressure"].isel(lead_day=day).values, dtype=np.float32)
            sw_down = np.asarray(surface_forcing["sw_down"].isel(lead_day=day).values, dtype=np.float32)
            lw_down = np.asarray(surface_forcing["lw_down"].isel(lead_day=day).values, dtype=np.float32)
            sea_ice_cover = np.asarray(surface_forcing["sea_ice_cover"].isel(lead_day=day).values, dtype=np.float32)

        forcing = _build_atm_to_surface(
            prediction_physical,
            {"sp": sp, "sw_down": sw_down, "lw_down": lw_down},
            channel_labels=channel_labels,
            pressure_levels=data_config.pressure_levels,
            config=config,
        )
        state_new, sst_new, _, _ = ocean_step(ocean_state, forcing, config.dt_seconds)
        sst_new = np.asarray(sst_new, dtype=np.float32)
        sst_prev = np.asarray(ocean_state.T_sfc.data, dtype=np.float32)
        sst_new = np.where(np.isfinite(sst_new), sst_new, sst_prev)
        if sea_ice_cover is not None:
            sst_new = np.where(
                sea_ice_cover >= config.sea_ice_threshold,
                np.float32(config.ocean.T_freeze),
                sst_new,
            )
        sst_new = _sst_state_from_masked_array(
            sst_new,
            ocean_mask,
            freeze_temperature=config.ocean.T_freeze,
        )
        sst_new_physical = _sst_from_ocean_units(sst_new, uses_celsius=sst_uses_celsius)
        ocean_state = SlabOceanState(
            T_sfc=state_new.T_sfc.replace(data=jnp.asarray(sst_new, dtype=float)),
            T_deep=state_new.T_deep,
        )

        next_forcing_physical = np.asarray(target_forcing_physical[day], dtype=np.float32).copy()
        if coupled:
            next_forcing_physical[..., sst_index] = np.asarray(sst_new_physical, dtype=np.float32)
            sst_output = _masked_output(sst_new_physical, ocean_mask)
        else:
            next_forcing_physical[..., sst_index] = fixed_uncoupled_sst
            sst_output = _masked_output(fixed_uncoupled_sst, ocean_mask)
        forcing_used_physical = np.asarray(next_forcing_physical, dtype=np.float32)
        forcing_used_normalized = normalize_forcing_channels(forcing_used_physical, data_config, norms=norms)
        current_state = np.concatenate([prediction, forcing_used_normalized], axis=-1).astype(np.float32, copy=False)

        predictions.append(prediction_physical.astype(np.float32, copy=False))
        forcings_used.append(forcing_used_physical.astype(np.float32, copy=False))
        sst_series.append(np.asarray(sst_output, dtype=np.float32))

    prediction_array = np.stack(predictions, axis=0)
    forcing_array = np.stack(forcings_used, axis=0)
    sst_array = np.stack(sst_series, axis=0)
    if surface_sst_raw is not None:
        target_sst_array = np.asarray(surface_sst_raw, dtype=np.float32)
    else:
        target_sst_array = np.asarray(target_forcing_physical[..., sst_index], dtype=np.float32)

    ds = xr.Dataset(
        data_vars={
            "prediction": (("lead_day", "latitude", "longitude", "channel"), prediction_array),
            "target": (("lead_day", "latitude", "longitude", "channel"), target_physical.astype(np.float32, copy=False)),
            "forcing": (("lead_day", "latitude", "longitude", "forcing_channel"), forcing_array),
            "target_forcing": (
                ("lead_day", "latitude", "longitude", "forcing_channel"),
                target_forcing_physical.astype(np.float32, copy=False),
            ),
            "sea_surface_temperature": (("lead_day", "latitude", "longitude"), sst_array),
            "target_sea_surface_temperature": (
                ("lead_day", "latitude", "longitude"),
                target_sst_array.astype(np.float32, copy=False),
            ),
            "initial_atmosphere": (
                ("latitude", "longitude", "channel"),
                initial_atmosphere_physical.astype(np.float32, copy=False),
            ),
            "initial_forcing": (
                ("latitude", "longitude", "forcing_channel"),
                initial_forcing_physical.astype(np.float32, copy=False),
            ),
            "ocean_mask": (("latitude", "longitude"), ocean_mask.astype(np.int8, copy=False)),
        },
        coords={
            "lead_day": np.arange(1, prediction_array.shape[0] + 1, dtype=np.int32),
            "target_date": ("lead_day", np.asarray(target_dates, dtype="U8")),
            "latitude": target_grid.latitude_deg.astype(np.float32, copy=False),
            "longitude": target_grid.longitude_deg.astype(np.float32, copy=False),
            "channel": np.asarray(channel_labels, dtype="U16"),
            "forcing_channel": np.asarray(forcing_labels, dtype="U16"),
        },
        attrs={
            "init_date": input_date,
            "sample_index": int(sample_index),
            "gaussian_n_max": int(data_config.gaussian_n_max),
            "normalize": int(bool(data_config.normalize)),
            "coupled": int(bool(coupled)),
            "surface_forcing_source": surface_source,
            "stochastic_seed": int(stochastic_seed),
            "tendency_prediction": int(bool(tendency_prediction)),
        },
    )
    return ds


__all__ = ["S2SSlabCouplingConfig", "coupled_rollout_to_dataset"]
