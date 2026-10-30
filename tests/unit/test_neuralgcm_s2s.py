from __future__ import annotations

import argparse
import csv
import importlib.util
import builtins
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from legoesm.ml.s2s.neuralgcm_slab import preparation as neuralgcm_preparation
from legoesm.ml.s2s.neuralgcm_slab.campaign import (
    filter_start_times_to_year,
    generate_semimonthly_start_times,
)
from legoesm.ml.s2s.neuralgcm_slab.evaluation import LeadTimeWindow
from legoesm.ml.s2s.neuralgcm_slab.ensemble import (
    HeadlineField,
    apply_initial_perturbations,
    build_ensemble_daily_crps_table,
)
from legoesm.ml.s2s.neuralgcm_slab.postprocess import postprocess_campaign, postprocess_case
from legoesm.ml.s2s.neuralgcm_slab.slab_coupling import rollout_coupled_daily


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_script_module(module_name: str, relative_path: str):
    if relative_path.startswith(("src/legoesm/", "legoesm/")):
        # carve-aware: ml/ may live in a uv-workspace member (packages/legoesm-ml).
        from tests.legoesm_paths import legoesm_source_path

        path = legoesm_source_path(relative_path)
    else:
        path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeBackendConfig:
    time_name = "time"


class _RecordingBackend:
    def __init__(self):
        self.config = _FakeBackendConfig()
        self.sst_inputs: list[np.ndarray] = []
        self.sea_ice_inputs: list[np.ndarray] = []

    def prepare_initial_state(self, dataset: xr.Dataset):
        return 0

    def build_forcing_dataset(
        self,
        template: xr.Dataset,
        sea_surface_temperature: xr.DataArray,
        sea_ice_cover: xr.DataArray,
    ) -> xr.Dataset:
        forcing = template.copy(deep=True)
        if "time" not in sea_surface_temperature.dims:
            sea_surface_temperature = sea_surface_temperature.expand_dims(
                {"time": template["time"].values}
            )
        if "time" not in sea_ice_cover.dims:
            sea_ice_cover = sea_ice_cover.expand_dims({"time": template["time"].values})
        forcing["sea_surface_temperature"] = sea_surface_temperature
        forcing["sea_ice_cover"] = sea_ice_cover
        return forcing

    def run_one_day(self, state, forcing_dataset: xr.Dataset, *, start_with_input: bool = True):
        sst = forcing_dataset["sea_surface_temperature"].squeeze("time", drop=True)
        sea_ice = forcing_dataset["sea_ice_cover"].squeeze("time", drop=True)
        self.sst_inputs.append(np.asarray(sst, dtype=float))
        self.sea_ice_inputs.append(np.asarray(sea_ice, dtype=float))
        level = xr.DataArray(np.asarray([1000.0]), dims=("level",), name="level")
        latitude = sst["latitude"]
        longitude = sst["longitude"]
        shape_3d = (1, latitude.size, longitude.size)
        forecast = xr.Dataset(
            {
                "temperature": xr.DataArray(
                    np.full(shape_3d, 280.0 + len(self.sst_inputs), dtype=np.float32),
                    coords={"level": level, "latitude": latitude, "longitude": longitude},
                    dims=("level", "latitude", "longitude"),
                ),
                "specific_humidity": xr.DataArray(
                    np.full(shape_3d, 0.005, dtype=np.float32),
                    coords={"level": level, "latitude": latitude, "longitude": longitude},
                    dims=("level", "latitude", "longitude"),
                ),
                "u_component_of_wind": xr.DataArray(
                    np.full(shape_3d, 2.0, dtype=np.float32),
                    coords={"level": level, "latitude": latitude, "longitude": longitude},
                    dims=("level", "latitude", "longitude"),
                ),
                "v_component_of_wind": xr.DataArray(
                    np.full(shape_3d, 1.0, dtype=np.float32),
                    coords={"level": level, "latitude": latitude, "longitude": longitude},
                    dims=("level", "latitude", "longitude"),
                ),
            }
        )
        return state + 1, forecast


def _surface_dataset(
    start: str,
    values: list[float],
    *,
    sea_ice_values: list[float] | None = None,
) -> xr.Dataset:
    times = np.asarray(
        [np.datetime64(start) + np.timedelta64(index, "D") for index in range(len(values))]
    )
    latitude = xr.DataArray(np.asarray([-10.0, 10.0]), dims=("latitude",), name="latitude")
    longitude = xr.DataArray(np.asarray([0.0, 180.0]), dims=("longitude",), name="longitude")
    sst = np.stack(
        [np.full((latitude.size, longitude.size), value, dtype=np.float32) for value in values],
        axis=0,
    )
    if sea_ice_values is None:
        sea_ice_values = [0.0] * len(values)
    sea_ice = np.stack(
        [np.full((latitude.size, longitude.size), value, dtype=np.float32) for value in sea_ice_values],
        axis=0,
    )
    return xr.Dataset(
        {
            "sea_surface_temperature": xr.DataArray(
                sst,
                coords={"time": times, "latitude": latitude, "longitude": longitude},
                dims=("time", "latitude", "longitude"),
            ),
            "sea_ice_cover": xr.DataArray(
                sea_ice,
                coords={"time": times, "latitude": latitude, "longitude": longitude},
                dims=("time", "latitude", "longitude"),
            ),
            "surface_pressure": xr.DataArray(
                np.full_like(sst, 100000.0, dtype=np.float32),
                coords={"time": times, "latitude": latitude, "longitude": longitude},
                dims=("time", "latitude", "longitude"),
            ),
        }
    )


def _radiation_dataset(start: str, n_days: int) -> xr.Dataset:
    times = np.asarray(
        [np.datetime64(start) + np.timedelta64(index, "D") for index in range(n_days)]
    )
    latitude = xr.DataArray(np.asarray([-10.0, 10.0]), dims=("latitude",), name="latitude")
    longitude = xr.DataArray(np.asarray([0.0, 180.0]), dims=("longitude",), name="longitude")
    shape = (n_days, latitude.size, longitude.size)
    return xr.Dataset(
        {
            "sw_down": xr.DataArray(
                np.full(shape, 200.0, dtype=np.float32),
                coords={"time": times, "latitude": latitude, "longitude": longitude},
                dims=("time", "latitude", "longitude"),
            ),
            "lw_down": xr.DataArray(
                np.full(shape, 300.0, dtype=np.float32),
                coords={"time": times, "latitude": latitude, "longitude": longitude},
                dims=("time", "latitude", "longitude"),
            ),
        }
    )


def _truth_dataset(start: str, n_leads: int) -> xr.Dataset:
    times = np.asarray(
        [np.datetime64(start) + np.timedelta64(index + 1, "D") for index in range(n_leads)]
    )
    levels = xr.DataArray(np.asarray([500.0, 700.0, 850.0]), dims=("level",), name="level")
    latitude = xr.DataArray(np.asarray([-10.0, 10.0]), dims=("latitude",), name="latitude")
    longitude = xr.DataArray(np.asarray([0.0, 180.0]), dims=("longitude",), name="longitude")
    shape_4d = (n_leads, levels.size, latitude.size, longitude.size)
    shape_3d = (n_leads, latitude.size, longitude.size)
    base = np.arange(n_leads, dtype=np.float32)[:, None, None, None]
    truth = xr.Dataset(
        {
            "temperature": xr.DataArray(
                280.0 + base + np.zeros(shape_4d, dtype=np.float32),
                coords={"time": times, "level": levels, "latitude": latitude, "longitude": longitude},
                dims=("time", "level", "latitude", "longitude"),
            ),
            "geopotential": xr.DataArray(
                5000.0 + 10.0 * base + np.zeros(shape_4d, dtype=np.float32),
                coords={"time": times, "level": levels, "latitude": latitude, "longitude": longitude},
                dims=("time", "level", "latitude", "longitude"),
            ),
            "specific_humidity": xr.DataArray(
                0.002 + 1.0e-4 * base + np.zeros(shape_4d, dtype=np.float32),
                coords={"time": times, "level": levels, "latitude": latitude, "longitude": longitude},
                dims=("time", "level", "latitude", "longitude"),
            ),
            "u_component_of_wind": xr.DataArray(
                5.0 + np.zeros(shape_4d, dtype=np.float32),
                coords={"time": times, "level": levels, "latitude": latitude, "longitude": longitude},
                dims=("time", "level", "latitude", "longitude"),
            ),
            "v_component_of_wind": xr.DataArray(
                2.0 + np.zeros(shape_4d, dtype=np.float32),
                coords={"time": times, "level": levels, "latitude": latitude, "longitude": longitude},
                dims=("time", "level", "latitude", "longitude"),
            ),
            "sea_surface_temperature": xr.DataArray(
                290.0 + np.arange(n_leads, dtype=np.float32)[:, None, None] + np.zeros(shape_3d, dtype=np.float32),
                coords={"time": times, "latitude": latitude, "longitude": longitude},
                dims=("time", "latitude", "longitude"),
            ),
            "sea_ice_cover": xr.DataArray(
                np.zeros(shape_3d, dtype=np.float32),
                coords={"time": times, "latitude": latitude, "longitude": longitude},
                dims=("time", "latitude", "longitude"),
            ),
            "surface_pressure": xr.DataArray(
                np.full(shape_3d, 100000.0, dtype=np.float32),
                coords={"time": times, "latitude": latitude, "longitude": longitude},
                dims=("time", "latitude", "longitude"),
            ),
        }
    )
    return truth


def _forecast_from_truth(truth: xr.Dataset, *, offset: float) -> xr.Dataset:
    lead_days = np.arange(1, int(truth.sizes["time"]) + 1, dtype=int)
    forecast = xr.Dataset(
        {
            "temperature": truth["temperature"].rename({"time": "lead_day"}) + offset,
            "geopotential": truth["geopotential"].rename({"time": "lead_day"}) + 10.0 * offset,
            "specific_humidity": truth["specific_humidity"].rename({"time": "lead_day"}) + 1.0e-4 * offset,
            "u_component_of_wind": truth["u_component_of_wind"].rename({"time": "lead_day"}),
            "v_component_of_wind": truth["v_component_of_wind"].rename({"time": "lead_day"}),
            "sea_surface_temperature": truth["sea_surface_temperature"].rename({"time": "lead_day"}) + offset,
        }
    )
    forecast = forecast.assign_coords({"lead_day": lead_days})
    return forecast


def _write_case_forecasts(case_dir: Path, truth: xr.Dataset) -> None:
    for experiment, base_offset in (("coupled", 0.1), ("uncoupled", 1.0)):
        for member_index, member_offset in enumerate((-0.1, 0.1)):
            forecast_dir = case_dir / f"seed_{member_index:02d}"
            forecast_dir.mkdir(parents=True, exist_ok=True)
            forecast = _forecast_from_truth(truth, offset=base_offset + member_offset)
            forecast.to_netcdf(forecast_dir / f"{experiment}.nc")


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_filter_start_times_to_year_keeps_november_15_and_drops_december_1() -> None:
    start_times = generate_semimonthly_start_times(2022)
    filtered = filter_start_times_to_year(start_times, forecast_days=42, year=2022)
    assert "2022-11-15T00:00:00" in filtered
    assert "2022-12-01T00:00:00" not in filtered


def test_uncoupled_rollout_uses_fixed_initial_surface_state() -> None:
    backend = _RecordingBackend()
    initial = _surface_dataset(
        "2022-11-15T00:00:00",
        [290.0],
        sea_ice_values=[0.0],
    ).isel(time=slice(0, 1))
    forcing = _surface_dataset(
        "2022-11-15T00:00:00",
        [290.0, 291.0, 292.0],
        sea_ice_values=[0.0, 1.0, 1.0],
    )
    radiation = _radiation_dataset("2022-11-15T00:00:00", 3)

    forecast = rollout_coupled_daily(
        backend,
        initial,
        forcing,
        radiation,
        forecast_days=3,
        coupled=False,
    )

    assert len(backend.sst_inputs) == 3
    assert len(backend.sea_ice_inputs) == 3
    expected_sst = np.full((2, 2), 290.0, dtype=float)
    expected_sea_ice = np.zeros((2, 2), dtype=float)
    for recorded in backend.sst_inputs:
        np.testing.assert_allclose(recorded, expected_sst)
    for recorded in backend.sea_ice_inputs:
        np.testing.assert_allclose(recorded, expected_sea_ice)
    for lead_index in range(3):
        np.testing.assert_allclose(
            np.asarray(forecast["sea_surface_temperature"].isel(lead_day=lead_index), dtype=float),
            expected_sst,
        )
        np.testing.assert_allclose(
            np.asarray(forecast["sst_tendency"].isel(lead_day=lead_index), dtype=float),
            0.0,
        )


class _PerturbedCoordBackend(_RecordingBackend):
    def run_one_day(self, state, forcing_dataset: xr.Dataset, *, start_with_input: bool = True):
        state, forecast = super().run_one_day(state, forcing_dataset, start_with_input=start_with_input)
        forecast = forecast.assign_coords(
            {
                "latitude": xr.DataArray(
                    np.asarray(forecast["latitude"], dtype=np.float64) + np.asarray([0.0, 1.0e-6]),
                    dims=("latitude",),
                ),
                "longitude": xr.DataArray(
                    np.asarray(forecast["longitude"], dtype=np.float64) + np.asarray([0.0, 1.0e-6]),
                    dims=("longitude",),
                ),
            }
        )
        return state, forecast


def test_rollout_coerces_backend_coords_to_reference_grid() -> None:
    backend = _PerturbedCoordBackend()
    initial = _surface_dataset("2022-11-15T00:00:00", [290.0]).isel(time=slice(0, 1))
    forcing = _surface_dataset("2022-11-15T00:00:00", [290.0])
    radiation = _radiation_dataset("2022-11-15T00:00:00", 1)

    forecast = rollout_coupled_daily(
        backend,
        initial,
        forcing,
        radiation,
        forecast_days=1,
        coupled=False,
    )

    assert forecast.sizes["longitude"] == initial.sizes["longitude"]
    assert forecast.sizes["latitude"] == initial.sizes["latitude"]
    np.testing.assert_allclose(
        np.asarray(forecast["longitude"], dtype=float),
        np.asarray(initial["longitude"], dtype=float),
    )
    np.testing.assert_allclose(
        np.asarray(forecast["latitude"], dtype=float),
        np.asarray(initial["latitude"], dtype=float),
    )


def test_apply_initial_perturbations_handles_longitude_latitude_order() -> None:
    truth = _truth_dataset("2022-11-15T00:00:00", 2)
    initial = truth.isel(time=slice(0, 1)).copy()
    initial = initial.transpose("time", "level", "longitude", "latitude")
    initial["sea_surface_temperature"] = initial["sea_surface_temperature"].transpose("time", "longitude", "latitude")
    initial["sea_ice_cover"] = initial["sea_ice_cover"].transpose("time", "longitude", "latitude")
    initial["surface_pressure"] = initial["surface_pressure"].transpose("time", "longitude", "latitude")

    perturbed, metadata = apply_initial_perturbations(initial, member_index=1)

    assert perturbed["temperature"].dims == ("time", "level", "longitude", "latitude")
    assert perturbed["sea_surface_temperature"].dims == ("time", "longitude", "latitude")
    assert metadata["variables"]["temperature"]["delta_std"] > 0.0
    assert metadata["variables"]["sea_surface_temperature"]["delta_std"] > 0.0


def test_fill_and_regrid_falls_back_to_slicewise_fill_for_varying_nan_masks(
    monkeypatch,
) -> None:
    call_dims: list[tuple[str, ...]] = []

    class _FakeXarrayUtils:
        @staticmethod
        def fill_nan_with_nearest(field: xr.DataArray) -> xr.DataArray:
            call_dims.append(field.dims)
            if "time" in field.dims:
                raise ValueError("NaN mask is not fixed across non-spatial dimensions")
            values = np.asarray(field.values, dtype=np.float32)
            fill_value = float(np.nanmean(values)) if np.isnan(values).any() else 0.0
            if np.isnan(fill_value):
                fill_value = 0.0
            filled = np.where(np.isnan(values), fill_value, values)
            return xr.DataArray(
                filled,
                coords=field.coords,
                dims=field.dims,
                name=field.name,
                attrs=field.attrs,
            )

        @staticmethod
        def regrid_horizontal(field: xr.DataArray, regridder: object) -> xr.DataArray:
            return field.copy(deep=True)

    monkeypatch.setattr(
        neuralgcm_preparation,
        "_import_dinosaur",
        lambda: (None, None, _FakeXarrayUtils),
    )

    field = xr.DataArray(
        np.asarray(
            [
                [[np.nan, 290.0], [291.0, 292.0]],
                [[293.0, 294.0], [np.nan, 295.0]],
            ],
            dtype=np.float32,
        ),
        coords={
            "time": np.asarray([np.datetime64("2022-09-01"), np.datetime64("2022-09-02")]),
            "latitude": np.asarray([-10.0, 10.0], dtype=np.float32),
            "longitude": np.asarray([0.0, 180.0], dtype=np.float32),
        },
        dims=("time", "latitude", "longitude"),
        name="sea_surface_temperature",
        attrs={"units": "K"},
    )

    result = neuralgcm_preparation._fill_and_regrid(field, regridder=object())

    assert result.dims == field.dims
    assert result.name == field.name
    assert result.attrs["units"] == "K"
    assert not np.isnan(np.asarray(result.values)).any()
    assert call_dims[0] == ("time", "latitude", "longitude")
    assert call_dims.count(("latitude", "longitude")) == 2


def test_build_ensemble_daily_crps_handles_swapped_longitude_latitude_order() -> None:
    truth = _truth_dataset("2022-11-15T00:00:00", 3)
    forecasts = []
    for offset in (-0.1, 0.1):
        forecast = _forecast_from_truth(truth, offset=offset)
        forecast["temperature"] = forecast["temperature"].transpose(
            "lead_day", "level", "longitude", "latitude"
        )
        forecast["geopotential"] = forecast["geopotential"].transpose(
            "lead_day", "level", "longitude", "latitude"
        )
        forecast["specific_humidity"] = forecast["specific_humidity"].transpose(
            "lead_day", "level", "longitude", "latitude"
        )
        forecast["u_component_of_wind"] = forecast["u_component_of_wind"].transpose(
            "lead_day", "level", "longitude", "latitude"
        )
        forecast["v_component_of_wind"] = forecast["v_component_of_wind"].transpose(
            "lead_day", "level", "longitude", "latitude"
        )
        forecast["sea_surface_temperature"] = forecast["sea_surface_temperature"].transpose(
            "lead_day", "longitude", "latitude"
        )
        forecasts.append(forecast)

    fields = (
        HeadlineField("t850", "temperature", 850),
        HeadlineField("sst", "sea_surface_temperature", None),
    )
    crps = build_ensemble_daily_crps_table(forecasts, truth, fields=fields)

    assert set(crps) == {"t850", "sst"}
    assert np.all(np.isfinite(crps["t850"]))
    assert np.all(np.isfinite(crps["sst"]))


def test_postprocess_case_and_campaign_write_expected_metric_tables(tmp_path: Path) -> None:
    case_dir = tmp_path / "case"
    prepared_dir = case_dir / "_prepared"
    prepared_dir.mkdir(parents=True, exist_ok=True)

    truth = _truth_dataset("2022-11-15T00:00:00", 4)
    truth.to_netcdf(prepared_dir / "truth.nc")
    initial = truth.isel(time=slice(0, 1)).copy()
    initial = initial.assign_coords({"time": [np.datetime64("2022-11-15T00:00:00")]})
    initial.to_netcdf(prepared_dir / "initial.nc")
    _write_case_forecasts(case_dir, truth)

    windows = (
        LeadTimeWindow("wk1", 1, 2),
        LeadTimeWindow("wk2", 3, 4),
    )
    postprocess_case(
        case_dir,
        n_members=2,
        windows=windows,
        plot_member_indices=(0,),
        snapshot_days=(1, 4),
        make_snapshots=False,
        make_gifs=False,
    )

    ensemble_rows = _read_csv_rows(case_dir / "metrics" / "window_ensemble_metrics.csv")
    assert ensemble_rows
    assert {"experiment", "field", "window", "rmse", "mae", "crps", "n_members"}.issubset(
        ensemble_rows[0].keys()
    )

    campaign_dir = tmp_path / "campaign"
    postprocess_campaign([case_dir], output_dir=campaign_dir, windows=windows)
    campaign_rows = _read_csv_rows(campaign_dir / "metrics" / "case_ensemble_daily_metrics.csv")
    assert campaign_rows
    assert {"init_time", "experiment", "field", "metric", "lead_day", "value"}.issubset(
        campaign_rows[0].keys()
    )


def test_open_era5_store_raises_friendly_dependency_error(monkeypatch) -> None:
    def _boom(*args, **kwargs):
        raise ModuleNotFoundError("No module named 'gcsfs'")

    monkeypatch.setattr(xr, "open_zarr", _boom)

    with pytest.raises(ImportError, match="Install legoesm\\[data\\]"):
        neuralgcm_preparation._open_era5_store("gs://example-bucket/test.zarr")


def test_import_dinosaur_raises_friendly_dependency_error(monkeypatch) -> None:
    original_import = builtins.__import__

    def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "dinosaur":
            raise ModuleNotFoundError("No module named 'dinosaur'")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    with pytest.raises(ImportError, match="requires the dinosaur package"):
        neuralgcm_preparation._import_dinosaur()


def test_resolve_case_dirs_accepts_campaign_dir(tmp_path: Path) -> None:
    neuralgcm_cli = _load_script_module(
        "neuralgcm_cli_test",
        "src/legoesm/ml/s2s/neuralgcm_slab/cli.py",
    )
    campaign_dir = tmp_path / "campaign"
    first = campaign_dir / "inference_20220101"
    second = campaign_dir / "inference_20220115"
    first.mkdir(parents=True)
    second.mkdir(parents=True)

    args = argparse.Namespace(
        case_dir=[],
        campaign_dir=[campaign_dir],
        year=None,
        months="1,15",
        days="1,15",
        forecast_days=42,
        checkpoint="dummy",
        base_output_dir=tmp_path,
    )

    assert neuralgcm_cli._resolve_case_dirs(args) == [first, second]


def test_submit_runtime_selector_rejects_python_and_conda_env() -> None:
    module = _load_script_module(
        "submit_neuralgcm_campaign_test",
        "scripts/run/s2s/submit_neuralgcm_campaign.py",
    )

    with pytest.raises(ValueError, match="Use only one of --python or --conda-env"):
        module._resolve_runtime_selector(
            python_executable="/usr/bin/python",
            conda_env="legoesm",
        )


def test_plot_coupling_diagnostics_validates_inputs() -> None:
    module = _load_script_module(
        "plot_coupling_diagnostics_test",
        "scripts/run/s2s/plot_coupling_diagnostics.py",
    )
    coupled = xr.Dataset(
        {
            "sea_surface_temperature": xr.DataArray(
                np.ones((2, 2, 2), dtype=np.float32),
                dims=("time", "latitude", "longitude"),
            )
        }
    )

    with pytest.raises(ValueError, match="lead_day"):
        module._validate_plot_inputs(coupled, uncoupled=None, truth=None)


def test_surface_forcing_builder_refuses_missing_inputs() -> None:
    from legoesm.ml.s2s.neuralgcm_slab.slab_coupling import extract_surface_forcing_from_dataset

    coords = {"level": [1000], "latitude": [0.0, 1.0], "longitude": [0.0, 1.0]}
    atm = xr.Dataset(
        {name: (("level", "latitude", "longitude"), np.full((1, 2, 2), val))
         for name, val in (("temperature", 290.0), ("specific_humidity", 0.01),
                           ("u_component_of_wind", 1.0), ("v_component_of_wind", 1.0))},
        coords=coords,
    )
    rad = xr.Dataset(
        {n: (("latitude", "longitude"), np.full((2, 2), 100.0)) for n in ("sw_down", "lw_down")},
        coords={"latitude": [0.0, 1.0], "longitude": [0.0, 1.0]},
    )
    with pytest.raises(ValueError, match="radiation_dataset"):
        extract_surface_forcing_from_dataset(atm, surface_pressure=np.full((2, 2), 1e5))
    with pytest.raises(ValueError, match="surface pressure is required"):
        extract_surface_forcing_from_dataset(atm, radiation_dataset=rad)
    out = extract_surface_forcing_from_dataset(
        atm, radiation_dataset=rad, surface_pressure=np.full((2, 2), 1e5))
    assert np.allclose(np.asarray(out.sw_down), 100.0)
