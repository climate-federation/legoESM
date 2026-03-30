"""Optional NeuralGCM backend wrapper.

This module keeps the NeuralGCM integration isolated behind an import guard so
other legoESM workflows can be imported without the external dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
import importlib
import pickle
from pathlib import Path
from typing import Any
import warnings

import numpy as np
import xarray as xr

from legoesm import constants

DEFAULT_NEURALGCM_CHECKPOINT = "gs://neuralgcm/models/v1/stochastic_1_4_deg.pkl"

_SURFACE_PRESSURE_GIN = """
inputs_to_units_mapping = {
    'geopotential': 'm^2 s^-2',
    'specific_humidity': 'kg kg^-1',
    'temperature': 'K',
    'u_component_of_wind': 'm s^-1',
    'v_component_of_wind': 'm s^-1',
    'specific_cloud_ice_water_content': 'kg kg^-1',
    'specific_cloud_liquid_water_content': 'kg kg^-1',
    'surface_pressure': 'Pa',
}
gin_bindings = [
    'StochasticPhysicsParameterizationStep.diagnostics_module = @NodalModelDiagnosticsDecoder',
    'NodalModelDiagnosticsDecoder.decoder_module = @SurfacePressureDiagnostics',
    'Model.step_filters = (@NoFilter, @ClipFilter, @ExponentialLeapfrogFilter, @FixGlobalMeanFilter)',
]
"""


def neuralgcm_available() -> bool:
    return importlib.util.find_spec("neuralgcm") is not None


@dataclass(frozen=True)
class NeuralGCMBackendConfig:
    """Configuration for the optional NeuralGCM backend."""

    checkpoint: str = DEFAULT_NEURALGCM_CHECKPOINT
    patch_surface_pressure: bool = True
    add_stability_filter: bool = True
    forcing_time_shift: str | None = None
    daily_timedelta_hours: int = 24
    level_name: str = "level"
    latitude_name: str = "latitude"
    longitude_name: str = "longitude"
    time_name: str = "time"
    sst_variable: str = "sea_surface_temperature"
    sea_ice_variable: str = "sea_ice_cover"
    rng_seed: int = 0


def _load_pickle(checkpoint: str) -> Any:
    if checkpoint.startswith(("gs://", "s3://")):
        try:
            import gcsfs
        except ImportError as exc:
            raise ImportError(
                "Remote checkpoints require fsspec/gcsfs. Install legoesm[data] first."
            ) from exc
        if checkpoint.startswith("gs://"):
            fs = gcsfs.GCSFileSystem(token="anon")
            with fs.open(checkpoint, "rb") as handle:
                return pickle.load(handle)
        import fsspec
        with fsspec.open(checkpoint, "rb") as handle:
            return pickle.load(handle)
    with Path(checkpoint).expanduser().open("rb") as handle:
        return pickle.load(handle)


def patch_checkpoint_for_surface_pressure(
    checkpoint: dict[str, Any],
    *,
    add_stability_filter: bool,
) -> dict[str, Any]:
    patched = copy.deepcopy(checkpoint)
    config_str = str(patched.get("model_config_str", ""))
    if "surface_pressure" not in config_str:
        config_str = config_str.rstrip() + "\n" + _SURFACE_PRESSURE_GIN.strip() + "\n"
    if not add_stability_filter:
        config_str = config_str.replace(", @FixGlobalMeanFilter", "").replace(
            "(@NoFilter, @ClipFilter, @ExponentialLeapfrogFilter, @FixGlobalMeanFilter)",
            "(@NoFilter, @ClipFilter, @ExponentialLeapfrogFilter)",
        )
    patched["model_config_str"] = config_str
    return patched


def _drop_singleton_time(data: xr.Dataset | xr.DataArray, time_name: str) -> xr.Dataset | xr.DataArray:
    if time_name in data.dims and int(data.sizes.get(time_name, 0)) == 1:
        return data.squeeze(time_name, drop=True)
    return data


def _with_time_like(
    field: xr.DataArray,
    template: xr.Dataset,
    *,
    time_name: str,
) -> xr.DataArray:
    if time_name in field.dims:
        return field
    if time_name in template.coords:
        return field.expand_dims({time_name: template[time_name]})
    return field.expand_dims({time_name: [0]})


def _squeeze_initial_time_axis(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    squeezed: dict[str, np.ndarray] = {}
    for name, value in data.items():
        array = np.asarray(value)
        if array.ndim > 0 and array.shape[0] == 1:
            array = array[0]
        squeezed[name] = array
    return squeezed


class NeuralGCMBackend:
    """Thin adapter around the external NeuralGCM checkpoint/model APIs."""

    def __init__(
        self,
        config: NeuralGCMBackendConfig | None = None,
        *,
        model: Any = None,
    ):
        self.config = config or NeuralGCMBackendConfig()
        self._model = model

    def _import_neuralgcm(self):
        if not neuralgcm_available():
            raise ImportError(
                "neuralgcm is not installed in the active environment. Install it before running the NeuralGCM driver."
            )
        import neuralgcm
        return neuralgcm

    def load_checkpoint(self):
        if self._model is not None:
            return self._model
        neuralgcm = self._import_neuralgcm()
        checkpoint = _load_pickle(self.config.checkpoint)
        if self.config.patch_surface_pressure:
            checkpoint = patch_checkpoint_for_surface_pressure(
                checkpoint,
                add_stability_filter=self.config.add_stability_filter,
            )
        self._model = neuralgcm.PressureLevelModel.from_checkpoint(checkpoint)
        return self._model

    @property
    def model(self):
        return self.load_checkpoint()

    @property
    def required_variables(self) -> list[str]:
        model = self.model
        required = list(getattr(model, "input_variables", ())) + list(
            getattr(model, "forcing_variables", ())
        )
        return list(dict.fromkeys(required))

    def prepare_initial_state(self, dataset: xr.Dataset):
        model = self.model
        initial = dataset.copy()
        inputs = _squeeze_initial_time_axis(model.inputs_from_xarray(initial))
        forcings = _squeeze_initial_time_axis(model.forcings_from_xarray(initial))
        try:
            import jax
        except ImportError:
            jax = None
        if jax is not None:
            rng_key = jax.random.PRNGKey(self.config.rng_seed)
            try:
                return model.encode(inputs, forcings, rng_key)
            except TypeError:
                pass
        return model.encode(inputs, forcings)

    def build_forcing_dataset(
        self,
        template: xr.Dataset,
        sea_surface_temperature: xr.DataArray,
        sea_ice_cover: xr.DataArray,
    ) -> xr.Dataset:
        forcing = template.copy(deep=True)
        sst = _with_time_like(
            sea_surface_temperature,
            forcing,
            time_name=self.config.time_name,
        )
        sic = _with_time_like(
            sea_ice_cover,
            forcing,
            time_name=self.config.time_name,
        )
        forcing[self.config.sst_variable] = sst
        forcing[self.config.sea_ice_variable] = sic
        return forcing

    def run_one_day(
        self,
        state,
        forcing_dataset: xr.Dataset,
        *,
        start_with_input: bool = True,
    ):
        model = self.model
        forcings = model.forcings_from_xarray(forcing_dataset)
        next_state, prediction = model.unroll(
            state,
            forcings,
            steps=1,
            timedelta=np.timedelta64(self.config.daily_timedelta_hours, "h"),
            start_with_input=start_with_input,
        )
        output = model.data_to_xarray(
            prediction,
            times=np.asarray([self.config.daily_timedelta_hours]),
        )
        output = _drop_singleton_time(output, self.config.time_name)
        return next_state, output

    def to_eval_subset(
        self,
        forecast: xr.Dataset,
        *,
        levels_hpa: tuple[int, ...] = (500, 700, 850),
        level_name: str | None = None,
    ) -> xr.Dataset:
        level_name = level_name or self.config.level_name
        required = {
            "z": "geopotential",
            "q": "specific_humidity",
            "t": "temperature",
            "u": "u_component_of_wind",
            "v": "v_component_of_wind",
        }
        missing = [src for src in required.values() if src not in forecast]
        if missing:
            raise KeyError(
                f"Forecast dataset is missing required eval variables: {missing}"
            )
        subset = xr.Dataset()
        selector = {level_name: list(levels_hpa)}
        subset["z"] = forecast["geopotential"].sel(selector, method="nearest") / constants.g
        subset["q"] = forecast["specific_humidity"].sel(selector, method="nearest")
        subset["t"] = forecast["temperature"].sel(selector, method="nearest")
        subset["u"] = forecast["u_component_of_wind"].sel(selector, method="nearest")
        subset["v"] = forecast["v_component_of_wind"].sel(selector, method="nearest")
        return subset

    def prepare_dataset_for_model(self, dataset: xr.Dataset) -> xr.Dataset:
        required = self.required_variables
        missing = [name for name in required if name not in dataset]
        if missing:
            raise KeyError(f"Dataset is missing required NeuralGCM variables: {missing}")
        prepared = dataset[required].copy().fillna(0.0)
        if self.config.forcing_time_shift:
            warnings.warn(
                "forcing_time_shift is configured but not applied automatically. Provide pre-shifted forcing data when running the real NeuralGCM pipeline.",
                stacklevel=2,
            )
        return prepared


__all__ = [
    "DEFAULT_NEURALGCM_CHECKPOINT",
    "NeuralGCMBackend",
    "NeuralGCMBackendConfig",
    "neuralgcm_available",
    "patch_checkpoint_for_surface_pressure",
]
