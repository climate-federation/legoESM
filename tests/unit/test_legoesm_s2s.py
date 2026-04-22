from __future__ import annotations

import csv
from pathlib import Path
import types

import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr

from legoesm.driver.config import DycoreConfig, ExperimentConfig, GridConfig, OutputConfig, experiment_config_to_dict
from legoesm.driver.compiled_segments import pack_forcing
from legoesm.grids.cubed_sphere import create_cubed_sphere, rotate_winds_geo_to_grid
from legoesm.grids.regridding import get_cubedsphere_to_latlon_weights
from legoesm.grids.vertical import compute_geopotential, create_sigma_coordinate, make_hybrid_levels
from legoesm.forcing.amip import AMIPForcing, AMIPForcingConfig, get_forcing_at_time
from legoesm.io.restart import load_restart
from legoesm.core.field import Field
from legoesm.core.state import HydrostaticState
from legoesm.ml.s2s.legoesm_slab.campaign import (
    filter_start_times_to_year,
    generate_semimonthly_start_times,
)
from legoesm.ml.s2s.legoesm_slab.cli import build_arg_parser
from legoesm.ml.s2s.legoesm_slab.metrics import weighted_rmse
from legoesm.ml.s2s.legoesm_slab.postprocess import postprocess_case
from legoesm.ml.s2s.legoesm_slab.postprocess import _fill_plot_gaps
from legoesm.ml.s2s.legoesm_slab.preparation import (
    PreparationConfig,
    _build_truth_dataset,
    _balance_initialized_cdgrid_state,
    _blend_cubedsphere_scalar,
    _blend_cubedsphere_vector,
    _mask_subsurface_pressure_levels,
    _project_state_to_cdgrid_subspace,
    _snapshot_to_state,
    _validate_monotonic_model_interfaces,
    _build_model_surface_forcing_dataset,
    build_control_surface_forcing_dataset,
    prepare_legoesm_case,
)
from legoesm.ml.s2s.plotting import normalize_longitude_data
from legoesm.ml.s2s.legoesm_slab.rollout import wrap_relative_forcing_getter
from legoesm.ml.s2s.legoesm_slab.rollout import _build_coupled_driver
from legoesm.ml.s2s.legoesm_slab.rollout import _build_uncoupled_driver
from legoesm.ml.s2s.legoesm_slab.rollout import _run_experiment, ForecastExportConfig
from legoesm.ml.s2s.legoesm_slab.rollout import _apply_cubedsphere_to_latlon_masked
from legoesm.ml.s2s.legoesm_slab.rollout import _apply_cubedsphere_to_latlon_3d_masked
from legoesm.ml.s2s.legoesm_slab.rollout import _compute_model_geopotential
from legoesm.ml.s2s.legoesm_slab.rollout import _forecast_longitude_coords
from legoesm.ml.s2s.legoesm_slab.rollout import _interpolate_model_field_to_pressure_levels
from legoesm.ml.s2s.legoesm_slab.rollout import _winds_to_geographic
from legoesm.ml.s2s.legoesm_slab.rollout import ModelSurfaceForcing
from legoesm.ml.s2s.legoesm_slab.preparation import PreparedCaseMetadata
from legoesm.ml.s2s.legoesm_slab.teleconnections import compute_nino34_series
from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import RRTMGP
from legoesm.atmosphere.dynamics.primitive_eq_cdgrid import hydrostatic_to_fv3, fv3_to_hydrostatic
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.grids.halo import CONNECTIVITY, WEST, EAST, SOUTH, NORTH


def _synthetic_era5_dataset() -> xr.Dataset:
    time = np.asarray(
        [
            "2022-01-01T00:00:00",
            "2022-01-02T00:00:00",
            "2022-01-03T00:00:00",
        ],
        dtype="datetime64[s]",
    )
    level = xr.DataArray(np.asarray([200, 500, 700, 850]), dims=("level",), name="level")
    latitude = xr.DataArray(np.asarray([-30.0, 30.0]), dims=("latitude",), name="latitude")
    longitude = xr.DataArray(np.asarray([45.0, 135.0]), dims=("longitude",), name="longitude")
    shape_4d = (time.size, latitude.size, longitude.size, level.size)
    shape_3d = (time.size, latitude.size, longitude.size)
    base = np.arange(time.size, dtype=np.float32)[:, None, None, None]
    return xr.Dataset(
        {
            "temperature": xr.DataArray(
                260.0 + base + np.zeros(shape_4d, dtype=np.float32),
                coords={"time": time, "latitude": latitude, "longitude": longitude, "level": level},
                dims=("time", "latitude", "longitude", "level"),
            ),
            "specific_humidity": xr.DataArray(
                0.004 + 1.0e-4 * base + np.zeros(shape_4d, dtype=np.float32),
                coords={"time": time, "latitude": latitude, "longitude": longitude, "level": level},
                dims=("time", "latitude", "longitude", "level"),
            ),
            "specific_cloud_liquid_water_content": xr.DataArray(
                2.5e-4 + 1.0e-5 * base + np.zeros(shape_4d, dtype=np.float32),
                coords={"time": time, "latitude": latitude, "longitude": longitude, "level": level},
                dims=("time", "latitude", "longitude", "level"),
            ),
            "u_component_of_wind": xr.DataArray(
                5.0 + np.zeros(shape_4d, dtype=np.float32),
                coords={"time": time, "latitude": latitude, "longitude": longitude, "level": level},
                dims=("time", "latitude", "longitude", "level"),
            ),
            "v_component_of_wind": xr.DataArray(
                1.0 + np.zeros(shape_4d, dtype=np.float32),
                coords={"time": time, "latitude": latitude, "longitude": longitude, "level": level},
                dims=("time", "latitude", "longitude", "level"),
            ),
            "surface_pressure": xr.DataArray(
                np.full(shape_3d, 101325.0, dtype=np.float32),
                coords={"time": time, "latitude": latitude, "longitude": longitude},
                dims=("time", "latitude", "longitude"),
            ),
            "sea_surface_temperature": xr.DataArray(
                np.asarray(
                    [
                        [[300.0, np.nan], [301.0, np.nan]],
                        [[302.0, np.nan], [303.0, np.nan]],
                        [[304.0, np.nan], [305.0, np.nan]],
                    ],
                    dtype=np.float32,
                ),
                coords={"time": time, "latitude": latitude, "longitude": longitude},
                dims=("time", "latitude", "longitude"),
            ),
            "land_surface_temperature": xr.DataArray(
                np.asarray(
                    [
                        [[295.0, 285.0], [296.0, 286.0]],
                        [[297.0, 287.0], [298.0, 288.0]],
                        [[299.0, 289.0], [300.0, 290.0]],
                    ],
                    dtype=np.float32,
                ),
                coords={"time": time, "latitude": latitude, "longitude": longitude},
                dims=("time", "latitude", "longitude"),
            ),
            "land_sea_mask": xr.DataArray(
                np.asarray([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32),
                coords={"latitude": latitude, "longitude": longitude},
                dims=("latitude", "longitude"),
            ),
            "sea_ice_cover": xr.DataArray(
                np.asarray(
                    [
                        [[0.1, np.nan], [0.2, np.nan]],
                        [[0.3, np.nan], [0.4, np.nan]],
                        [[0.5, np.nan], [0.6, np.nan]],
                    ],
                    dtype=np.float32,
                ),
                coords={"time": time, "latitude": latitude, "longitude": longitude},
                dims=("time", "latitude", "longitude"),
            ),
            "geopotential": xr.DataArray(
                50000.0 + 10.0 * base + np.zeros(shape_4d, dtype=np.float32),
                coords={"time": time, "latitude": latitude, "longitude": longitude, "level": level},
                dims=("time", "latitude", "longitude", "level"),
            ),
            "surface_geopotential": xr.DataArray(
                np.full(shape_3d, 25.0, dtype=np.float32),
                coords={"time": time, "latitude": latitude, "longitude": longitude},
                dims=("time", "latitude", "longitude"),
            ),
            "total_precipitation": xr.DataArray(
                np.full(shape_3d, 1.0e-4, dtype=np.float32),
                coords={"time": time, "latitude": latitude, "longitude": longitude},
                dims=("time", "latitude", "longitude"),
            ),
        }
    )


def _synthetic_truth_dataset() -> xr.Dataset:
    time = np.asarray(["2022-01-02T00:00:00", "2022-01-03T00:00:00"], dtype="datetime64[s]")
    level = xr.DataArray(np.asarray([200, 500, 700, 850]), dims=("level",), name="level")
    latitude = xr.DataArray(np.asarray([-5.0, 5.0]), dims=("latitude",), name="latitude")
    longitude = xr.DataArray(np.asarray([200.0, 220.0]), dims=("longitude",), name="longitude")
    shape_4d = (time.size, level.size, latitude.size, longitude.size)
    shape_3d = (time.size, latitude.size, longitude.size)
    base = np.arange(time.size, dtype=np.float32)[:, None, None, None]
    return xr.Dataset(
        {
            "temperature": xr.DataArray(
                280.0 + base + np.zeros(shape_4d, dtype=np.float32),
                coords={"time": time, "level": level, "latitude": latitude, "longitude": longitude},
                dims=("time", "level", "latitude", "longitude"),
            ),
            "geopotential": xr.DataArray(
                50000.0 + 50.0 * base + np.zeros(shape_4d, dtype=np.float32),
                coords={"time": time, "level": level, "latitude": latitude, "longitude": longitude},
                dims=("time", "level", "latitude", "longitude"),
            ),
            "specific_humidity": xr.DataArray(
                0.005 + 1.0e-4 * base + np.zeros(shape_4d, dtype=np.float32),
                coords={"time": time, "level": level, "latitude": latitude, "longitude": longitude},
                dims=("time", "level", "latitude", "longitude"),
            ),
            "u_component_of_wind": xr.DataArray(
                4.0 + np.zeros(shape_4d, dtype=np.float32),
                coords={"time": time, "level": level, "latitude": latitude, "longitude": longitude},
                dims=("time", "level", "latitude", "longitude"),
            ),
            "v_component_of_wind": xr.DataArray(
                1.0 + np.zeros(shape_4d, dtype=np.float32),
                coords={"time": time, "level": level, "latitude": latitude, "longitude": longitude},
                dims=("time", "level", "latitude", "longitude"),
            ),
            "sea_surface_temperature": xr.DataArray(
                300.0 + np.arange(time.size, dtype=np.float32)[:, None, None] + np.zeros(shape_3d, dtype=np.float32),
                coords={"time": time, "latitude": latitude, "longitude": longitude},
                dims=("time", "latitude", "longitude"),
            ),
            "sea_ice_cover": xr.DataArray(
                np.zeros(shape_3d, dtype=np.float32),
                coords={"time": time, "latitude": latitude, "longitude": longitude},
                dims=("time", "latitude", "longitude"),
            ),
        }
    )


def _forecast_from_truth(truth: xr.Dataset, *, offset: float) -> xr.Dataset:
    lead_day = np.arange(1, int(truth.sizes["time"]) + 1, dtype=int)
    dataset = truth.rename({"time": "lead_day"}).assign_coords({"lead_day": lead_day}).copy(deep=True)
    dataset["temperature"] = dataset["temperature"] + offset
    dataset["geopotential"] = dataset["geopotential"] + 10.0 * offset
    dataset["specific_humidity"] = dataset["specific_humidity"] + 1.0e-4 * offset
    dataset["sea_surface_temperature"] = dataset["sea_surface_temperature"] + offset
    dataset["precipitation"] = xr.DataArray(
        np.full((lead_day.size, truth.sizes["latitude"], truth.sizes["longitude"]), 1.0 + offset, dtype=np.float32),
        coords={"lead_day": lead_day, "latitude": truth["latitude"], "longitude": truth["longitude"]},
        dims=("lead_day", "latitude", "longitude"),
    )
    dataset["rlut"] = xr.DataArray(
        np.full((lead_day.size, truth.sizes["latitude"], truth.sizes["longitude"]), 240.0, dtype=np.float32),
        coords={"lead_day": lead_day, "latitude": truth["latitude"], "longitude": truth["longitude"]},
        dims=("lead_day", "latitude", "longitude"),
    )
    dataset = dataset.assign_coords({"time": ("lead_day", truth["time"].values)})
    return dataset


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _mean_unique_edge_rmse(field: np.ndarray) -> float:
    from legoesm.grids.halo import CONNECTIVITY, WEST, EAST, SOUTH, NORTH

    def edge_strip(arr: np.ndarray, face: int, edge: int) -> np.ndarray:
        if edge == WEST:
            return arr[face, 0, :]
        if edge == EAST:
            return arr[face, -1, :]
        if edge == SOUTH:
            return arr[face, :, 0]
        if edge == NORTH:
            return arr[face, :, -1]
        raise ValueError(edge)

    seen = set()
    diffs: list[float] = []
    for face in range(6):
        for edge, (nbr_face, nbr_edge, reversed_idx) in CONNECTIVITY[face].items():
            key = tuple(sorted(((face, edge), (nbr_face, nbr_edge))))
            if key in seen:
                continue
            seen.add(key)
            x = edge_strip(field, face, edge)
            y = edge_strip(field, nbr_face, nbr_edge)
            if reversed_idx:
                y = y[::-1]
            diffs.append(float(np.sqrt(np.mean((x - y) ** 2))))
    return float(np.mean(diffs))


def test_legoesm_s2s_parser_defaults() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(["prepare-case", "--start-time", "2022-01-01T00:00:00"])
    assert args.forecast_days == 42
    assert args.resolution == 16
    assert args.nlev == 20
    assert args.radiation == "rrtmgp"
    assert args.convection == "sbm"
    assert args.sbm_tau_c == 7200.0
    assert args.sbm_RH_ref == 0.7
    assert args.sbm_cape_threshold == 70.0
    assert args.sigma_b == 0.7
    assert args.k_BL_max_per_day == 1.0
    assert args.k_free_per_day == 0.1
    assert args.A_h_scale == 0.0
    assert args.hyperdiff_scale == 0.0
    assert args.div_damp_scale == 1.0
    assert args.seed_cloud_liquid_from_era5 is True
    assert args.cdgrid_balance_steps == 5
    assert args.cubedsphere_edge_blend_strength == 0.0
    assert args.cubedsphere_edge_blend_width == 0


def test_legoesm_s2s_parser_accepts_sbm_overrides() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "prepare-case",
            "--start-time",
            "2022-01-01T00:00:00",
            "--sbm-tau-c",
            "10800",
            "--sbm-RH-ref",
            "0.8",
            "--sbm-cape-threshold",
            "150",
        ]
    )
    assert args.sbm_tau_c == 10800.0
    assert args.sbm_RH_ref == 0.8
    assert args.sbm_cape_threshold == 150.0


def test_legoesm_s2s_parser_accepts_drag_overrides() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "prepare-case",
            "--start-time",
            "2022-01-01T00:00:00",
            "--sigma-b",
            "0.5",
            "--k-BL-max-per-day",
            "0.0",
            "--k-free-per-day",
            "0.0",
        ]
    )
    assert args.sigma_b == 0.5
    assert args.k_BL_max_per_day == 0.0
    assert args.k_free_per_day == 0.0


def test_legoesm_s2s_parser_exposes_sat_adjust_without_microphysics() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "prepare-case",
            "--start-time",
            "2022-01-01T00:00:00",
            "--sat-adjust-without-microphysics",
        ]
    )
    assert args.sat_adjust_without_microphysics is True

    args = parser.parse_args(
        [
            "prepare-case",
            "--start-time",
            "2022-01-01T00:00:00",
        ]
    )
    assert args.sat_adjust_without_microphysics is False


def test_legoesm_s2s_parser_accepts_surface_exchange_overrides() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "prepare-case",
            "--start-time",
            "2022-01-01T00:00:00",
            "--C-H",
            "0.0",
            "--C-E",
            "0.0",
        ]
    )
    assert args.C_H == 0.0
    assert args.C_E == 0.0


def test_legoesm_s2s_parser_accepts_sponge_overrides() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "prepare-case",
            "--start-time",
            "2022-01-01T00:00:00",
            "--sponge-sigma",
            "0.0",
            "--sponge-tau-sec",
            "0.0",
        ]
    )
    assert args.sponge_sigma == 0.0
    assert args.sponge_tau_sec == 0.0


def test_build_coupled_driver_injects_prepared_land_mask_before_setup(monkeypatch) -> None:
    recorded = {}

    class _FakePhysics:
        def __init__(self):
            self.land_fraction = None

    class _FakeAtm:
        def __init__(self):
            self._f_land = None
            self.physics = _FakePhysics()

    class _FakeDriver:
        def __init__(self, *args, **kwargs):
            self._atm = _FakeAtm()

        def setup(self):
            recorded["land_before_setup"] = self._atm._f_land

    monkeypatch.setattr(
        "legoesm.ml.s2s.legoesm_slab.rollout.InitializedCoupledSlabDriver",
        _FakeDriver,
    )

    forcing = ModelSurfaceForcing(
        times=jnp.asarray([0.0], dtype=jnp.float32),
        uncoupled_surface_temperature=jnp.zeros((1, 6, 2, 2), dtype=jnp.float32),
        fixed_ocean_surface_temperature=jnp.zeros((1, 6, 2, 2), dtype=jnp.float32),
        land_surface_temperature=jnp.zeros((1, 6, 2, 2), dtype=jnp.float32),
        sea_ice_cover=jnp.zeros((1, 6, 2, 2), dtype=jnp.float32),
        land_fraction=jnp.full((6, 2, 2), 0.5, dtype=jnp.float32),
    )

    config = ExperimentConfig(
        grid=GridConfig(grid_type="cubed_sphere", resolution=2, nlev=4),
        dycore=DycoreConfig(discretization="cdgrid", dt=300.0),
        output=OutputConfig(output_dir="/tmp"),
        days=1,
    )

    driver = _build_coupled_driver(
        config,
        forcing_start_day=0.0,
        collector=None,
        coupled_metadata={},
        surface_forcing=forcing,
        output_dir="/tmp",
    )

    np.testing.assert_allclose(recorded["land_before_setup"], forcing.land_fraction)
    np.testing.assert_allclose(driver._atm.physics.land_fraction, forcing.land_fraction)


def test_build_uncoupled_driver_injects_prepared_land_mask_before_setup(monkeypatch) -> None:
    recorded = {}

    class _FakePhysics:
        def __init__(self):
            self.land_fraction = None

    class _FakeDriver:
        def __init__(self, *args, **kwargs):
            self._f_land = None
            self.physics = _FakePhysics()
            self.get_sst_sic = None

        def setup(self):
            recorded["land_before_setup"] = self._f_land

    monkeypatch.setattr(
        "legoesm.ml.s2s.legoesm_slab.rollout.ModelDriver",
        _FakeDriver,
    )

    forcing = ModelSurfaceForcing(
        times=jnp.asarray([0.0], dtype=jnp.float32),
        uncoupled_surface_temperature=jnp.zeros((1, 6, 2, 2), dtype=jnp.float32),
        fixed_ocean_surface_temperature=jnp.zeros((1, 6, 2, 2), dtype=jnp.float32),
        land_surface_temperature=jnp.zeros((1, 6, 2, 2), dtype=jnp.float32),
        sea_ice_cover=jnp.zeros((1, 6, 2, 2), dtype=jnp.float32),
        land_fraction=jnp.full((6, 2, 2), 0.5, dtype=jnp.float32),
    )

    config = ExperimentConfig(
        grid=GridConfig(grid_type="cubed_sphere", resolution=2, nlev=4),
        dycore=DycoreConfig(discretization="cdgrid", dt=300.0),
        output=OutputConfig(output_dir="/tmp"),
        days=1,
    )

    driver = _build_uncoupled_driver(
        config,
        surface_forcing=forcing,
        output_dir="/tmp",
    )

    np.testing.assert_allclose(recorded["land_before_setup"], forcing.land_fraction)
    np.testing.assert_allclose(driver.physics.land_fraction, forcing.land_fraction)


def test_cdgrid_projection_reduces_restart_wind_representation_error() -> None:
    synthetic = _synthetic_era5_dataset()
    snapshot = synthetic.sel(time=np.datetime64("2022-01-01T00:00:00"))
    grid = create_cubed_sphere(2)
    sigma = create_sigma_coordinate(4)
    config = ExperimentConfig(
        grid=GridConfig(grid_type="cubed_sphere", resolution=2, nlev=4),
        dycore=DycoreConfig(discretization="cdgrid", dt=300.0),
        output=OutputConfig(output_dir="/tmp"),
        days=1,
    )
    state, _ = _snapshot_to_state(snapshot, grid=grid, sigma=sigma, config=PreparationConfig())
    cdgrid = create_cubed_sphere_cdgrid(grid)

    def _wind_roundtrip_error(s):
        s_rt = fv3_to_hydrostatic(hydrostatic_to_fv3(s, cdgrid), cdgrid)
        du = np.asarray(s_rt.u.data - s.u.data, dtype=np.float64)
        dv = np.asarray(s_rt.v.data - s.v.data, dtype=np.float64)
        return float(np.sqrt(np.mean(du**2) + np.mean(dv**2)))

    err_before = _wind_roundtrip_error(state)
    projected = _project_state_to_cdgrid_subspace(
        state,
        grid=grid,
        experiment_config=config,
    )
    err_after = _wind_roundtrip_error(projected)
    assert err_after < err_before


def test_hydrostatic_to_fv3_enforces_shared_edge_wind_continuity() -> None:
    grid = create_cubed_sphere(4)
    sigma = create_sigma_coordinate(3)
    cdgrid = create_cubed_sphere_cdgrid(grid)

    u_east = jnp.ones((6, grid.n, grid.n, sigma.n_levels), dtype=jnp.float32) * 20.0
    v_north = jnp.zeros_like(u_east)
    u_grid, v_grid = rotate_winds_geo_to_grid(u_east, v_north, grid.angle[..., None])
    state = HydrostaticState(
        u=Field(data=u_grid, name="u", dims=("face", "x", "y", "level"), units="m/s"),
        v=Field(data=v_grid, name="v", dims=("face", "x", "y", "level"), units="m/s"),
        T=Field(
            data=jnp.ones((6, grid.n, grid.n, sigma.n_levels), dtype=jnp.float32) * 280.0,
            name="T",
            dims=("face", "x", "y", "level"),
            units="K",
        ),
        p_s=Field(
            data=jnp.ones((6, grid.n, grid.n), dtype=jnp.float32) * 1.0e5,
            name="p_s",
            dims=("face", "x", "y"),
            units="Pa",
        ),
        phis=Field(
            data=jnp.zeros((6, grid.n, grid.n), dtype=jnp.float32),
            name="phis",
            dims=("face", "x", "y"),
            units="m^2/s^2",
        ),
    )

    fv3 = hydrostatic_to_fv3(state, cdgrid)
    ca = cdgrid.cos_angle_corner[..., None]
    sa = cdgrid.sin_angle_corner[..., None]
    u_east_corner = ca * fv3.u_d.data - sa * fv3.v_d.data
    v_north_corner = sa * fv3.u_d.data + ca * fv3.v_d.data

    def _get_strip(arr, face, edge):
        if edge == WEST:
            return arr[face, 0, :, :]
        if edge == EAST:
            return arr[face, grid.n, :, :]
        if edge == SOUTH:
            return arr[face, :, 0, :]
        return arr[face, :, grid.n, :]

    diffs = []
    for face in range(6):
        for edge in (WEST, EAST, SOUTH, NORTH):
            nbr_face, nbr_edge, is_reversed = CONNECTIVITY[face][edge]
            if nbr_face >= face:
                continue
            u_a = _get_strip(u_east_corner, face, edge)
            v_a = _get_strip(v_north_corner, face, edge)
            u_b = _get_strip(u_east_corner, nbr_face, nbr_edge)
            v_b = _get_strip(v_north_corner, nbr_face, nbr_edge)
            if is_reversed:
                u_b = u_b[::-1, :]
                v_b = v_b[::-1, :]
            diffs.append(np.asarray(u_a - u_b, dtype=np.float64).ravel())
            diffs.append(np.asarray(v_a - v_b, dtype=np.float64).ravel())

    diff = np.concatenate(diffs)
    assert float(np.sqrt(np.mean(diff**2))) < 1.0e-6
    assert float(np.max(np.abs(diff))) < 1.0e-5


def test_snapshot_to_state_vertical_first_remap_produces_finite_state() -> None:
    synthetic = _synthetic_era5_dataset()
    snapshot = synthetic.sel(time=np.datetime64("2022-01-01T00:00:00"))
    grid = create_cubed_sphere(2)
    sigma = create_sigma_coordinate(4)
    state, q_v = _snapshot_to_state(
        snapshot,
        grid=grid,
        sigma=sigma,
        config=PreparationConfig(vertical_first_remap=True),
    )
    for array in (
        state.u.data,
        state.v.data,
        state.T.data,
        state.p_s.data,
        state.phis.data,
        q_v,
    ):
        assert np.isfinite(np.asarray(array)).all()


def test_mask_subsurface_pressure_levels_clamps_below_ground_values() -> None:
    plev = np.asarray([50000.0, 70000.0, 85000.0, 100000.0], dtype=np.float32)
    surface_pressure = np.asarray([[90000.0]], dtype=np.float32)
    field = np.asarray([[[250.0, 260.0, 270.0, 999.0]]], dtype=np.float32)
    masked = _mask_subsurface_pressure_levels(
        field,
        plev_pa=plev,
        surface_pressure=surface_pressure,
    )
    np.testing.assert_allclose(masked[0, 0], np.asarray([250.0, 260.0, 270.0, 270.0], dtype=np.float32))


def test_fill_plot_gaps_replaces_nan_holes_for_display_only() -> None:
    field = xr.DataArray(
        np.asarray([[1.0, np.nan], [3.0, 4.0]], dtype=np.float32),
        coords={"latitude": np.asarray([-5.0, 5.0]), "longitude": np.asarray([0.0, 10.0])},
        dims=("latitude", "longitude"),
    )
    filled = _fill_plot_gaps(field)
    assert np.isfinite(np.asarray(filled)).all()
    assert float(filled.sel(latitude=-5.0, longitude=0.0)) == 1.0


def test_forecast_longitude_coords_match_preparation_ordering() -> None:
    weights = types.SimpleNamespace(
        lon_cent=np.asarray([-177.5, -172.5, -2.5, 2.5, 172.5, 177.5], dtype=np.float32)
    )
    lon = _forecast_longitude_coords(weights)
    np.testing.assert_allclose(
        lon,
        np.asarray([2.5, 7.5, 177.5, 182.5, 352.5, 357.5], dtype=np.float32),
    )


def test_balance_initialized_cdgrid_state_updates_qv_and_strips_tracers(monkeypatch) -> None:
    grid = create_cubed_sphere(2)
    sigma = create_sigma_coordinate(4)
    config = ExperimentConfig(
        grid=GridConfig(grid_type="cubed_sphere", resolution=2, nlev=4),
        dycore=DycoreConfig(discretization="cdgrid", dt=300.0),
        output=OutputConfig(output_dir="/tmp"),
        days=1,
    )
    state = HydrostaticState(
        u=Field(jnp.ones((6, 2, 2, 4), dtype=jnp.float32), "u", ("face", "x", "y", "level"), "m/s"),
        v=Field(jnp.ones((6, 2, 2, 4), dtype=jnp.float32), "v", ("face", "x", "y", "level"), "m/s"),
        T=Field(280.0 * jnp.ones((6, 2, 2, 4), dtype=jnp.float32), "T", ("face", "x", "y", "level"), "K"),
        p_s=Field(9.5e4 * jnp.ones((6, 2, 2), dtype=jnp.float32), "p_s", ("face", "x", "y"), "Pa"),
        phis=Field(jnp.zeros((6, 2, 2), dtype=jnp.float32), "phis", ("face", "x", "y"), "m^2/s^2"),
    )
    q_v = 0.01 * jnp.ones((6, 2, 2, 4), dtype=jnp.float32)

    class _FakeModel:
        def step(self, s, dt):
            updated = dict(s.tracers or {})
            updated["q_v"] = updated["q_v"].replace(data=updated["q_v"].data + 0.001)
            return s._replace(
                u=s.u.replace(data=s.u.data + 1.0),
                tracers=updated,
            )

    monkeypatch.setattr(
        "legoesm.driver.component_factory.create_atmosphere_dycore",
        lambda cfg, grid_obj, sigma_obj: _FakeModel(),
        raising=False,
    )

    balanced_state, balanced_qv = _balance_initialized_cdgrid_state(
        state,
        q_v,
        grid=grid,
        sigma=sigma,
        experiment_config=config,
        n_steps=2,
    )

    assert balanced_state.tracers is None
    np.testing.assert_allclose(np.asarray(balanced_state.u.data), np.asarray(state.u.data + 2.0))
    np.testing.assert_allclose(np.asarray(balanced_qv), np.asarray(q_v + 0.002))


def test_validate_monotonic_model_interfaces_rejects_bad_hybrid_column() -> None:
    sigma = make_hybrid_levels(8, p_top_Pa=200.0, stretching=2.0)
    with pytest.raises(ValueError, match="non-monotonic model interface pressures"):
        _validate_monotonic_model_interfaces(jnp.full((1, 1, 1), 5.8e4), sigma=sigma)


def test_prepare_legoesm_case_writes_case_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    synthetic = _synthetic_era5_dataset()

    from legoesm.ml.s2s.legoesm_slab import preparation as prep_module

    monkeypatch.setattr(prep_module, "_open_era5_store", lambda store: synthetic)

    case_dir = tmp_path / "inference_20220101"
    config = ExperimentConfig(
        grid=GridConfig(grid_type="cubed_sphere", resolution=2, nlev=4, vertical_coord="sigma"),
        dycore=DycoreConfig(discretization="cdgrid", dt=450.0),
        output=OutputConfig(output_dir=str(case_dir), diag_days=1),
        days=2,
        start_day=0.0,
        dataset="custom",
        forcing_path=str(case_dir / "_prepared" / "surface_forcing_20220101_2d.nc"),
        sst_var="sea_surface_temperature",
        sic_var="sea_ice_cover",
        time_var="time",
        lat_var="latitude",
        lon_var="longitude",
        radiation="rrtmgp",
        cloud_scheme="sundqvist",
        microphysics="sundqvist",
        convection="mass_flux",
        turbulence="louis",
        topography="flat",
        start_year=2022,
    )

    outputs = prepare_legoesm_case(
        start_time="2022-01-01T00:00:00",
        forecast_days=2,
        case_dir=case_dir,
        experiment_config=config,
        config=PreparationConfig(
            era5_store="synthetic",
            forcing_resolution_deg=5.0,
            evaluation_resolution_deg=10.0,
            pressure_levels=(200, 500, 700, 850),
            seed_cloud_liquid_from_era5=True,
        ),
    )

    assert outputs["restart"].exists()
    assert outputs["surface_forcing"].exists()
    assert outputs["model_surface_forcing"].exists()
    assert outputs["truth"].exists()
    assert outputs["initial"].exists()
    assert outputs["metadata"].exists()

    forcing = xr.open_dataset(outputs["surface_forcing"])
    model_surface_forcing = xr.open_dataset(outputs["model_surface_forcing"])
    truth = xr.open_dataset(outputs["truth"])
    initial = xr.open_dataset(outputs["initial"])
    try:
        restart_grid = create_cubed_sphere(config.grid.resolution)
        restart_sigma = create_sigma_coordinate(config.grid.nlev)
        _, _, _, _, _, _, restart_q_c, restart_q_r, _, _ = load_restart(
            outputs["restart"],
            restart_grid,
            restart_sigma,
            config=config,
        )
        np.testing.assert_allclose(
            forcing["sea_surface_temperature"].isel(time=0).values[..., 0],
            forcing["sea_surface_temperature"].isel(time=1).values[..., 0],
        )
        assert np.isfinite(model_surface_forcing["land_fraction"]).all()
        assert np.isfinite(model_surface_forcing["uncoupled_surface_temperature"]).all()
        assert restart_q_c is not None
        assert np.isfinite(np.asarray(restart_q_c)).all()
        assert np.nanmax(np.asarray(restart_q_c)) > 0.0
        assert restart_q_r is not None
        assert np.allclose(np.asarray(restart_q_r), 0.0)
        assert not np.allclose(
            forcing["sea_ice_cover"].isel(time=0),
            forcing["sea_ice_cover"].isel(time=1),
        )
        np.testing.assert_array_equal(truth["lead_day"].values, np.asarray([1, 2]))
        assert str(initial["time"].values[0]).startswith("2022-01-01T00:00:00")
    finally:
        forcing.close()
        model_surface_forcing.close()
        truth.close()
        initial.close()


def test_build_truth_dataset_keeps_raw_pressure_level_fields_for_level_first_inputs() -> None:
    time = xr.DataArray(np.asarray(["2022-01-02T00:00:00"], dtype="datetime64[s]"), dims=("time",), name="time")
    level = xr.DataArray(np.asarray([200, 500, 700, 850], dtype=np.int32), dims=("level",), name="level")
    latitude = xr.DataArray(np.asarray([-5.0, 5.0], dtype=np.float32), dims=("latitude",), name="latitude")
    longitude = xr.DataArray(np.asarray([45.0, 135.0], dtype=np.float32), dims=("longitude",), name="longitude")
    shape_4d = (time.size, level.size, latitude.size, longitude.size)
    shape_3d = (time.size, latitude.size, longitude.size)
    dataset = xr.Dataset(
        {
            "temperature": xr.DataArray(
                280.0 + np.zeros(shape_4d, dtype=np.float32),
                coords={"time": time, "level": level, "latitude": latitude, "longitude": longitude},
                dims=("time", "level", "latitude", "longitude"),
            ),
            "specific_humidity": xr.DataArray(
                0.005 + np.zeros(shape_4d, dtype=np.float32),
                coords={"time": time, "level": level, "latitude": latitude, "longitude": longitude},
                dims=("time", "level", "latitude", "longitude"),
            ),
            "u_component_of_wind": xr.DataArray(
                np.zeros(shape_4d, dtype=np.float32),
                coords={"time": time, "level": level, "latitude": latitude, "longitude": longitude},
                dims=("time", "level", "latitude", "longitude"),
            ),
            "v_component_of_wind": xr.DataArray(
                np.zeros(shape_4d, dtype=np.float32),
                coords={"time": time, "level": level, "latitude": latitude, "longitude": longitude},
                dims=("time", "level", "latitude", "longitude"),
            ),
            "geopotential": xr.DataArray(
                50000.0 + np.zeros(shape_4d, dtype=np.float32),
                coords={"time": time, "level": level, "latitude": latitude, "longitude": longitude},
                dims=("time", "level", "latitude", "longitude"),
            ),
            "surface_pressure": xr.DataArray(
                np.full(shape_3d, 80000.0, dtype=np.float32),
                coords={"time": time, "latitude": latitude, "longitude": longitude},
                dims=("time", "latitude", "longitude"),
            ),
            "sea_surface_temperature": xr.DataArray(
                300.0 + np.zeros(shape_3d, dtype=np.float32),
                coords={"time": time, "latitude": latitude, "longitude": longitude},
                dims=("time", "latitude", "longitude"),
            ),
            "sea_ice_cover": xr.DataArray(
                np.zeros(shape_3d, dtype=np.float32),
                coords={"time": time, "latitude": latitude, "longitude": longitude},
                dims=("time", "latitude", "longitude"),
            ),
        }
    )
    config = ExperimentConfig(
        grid=GridConfig(grid_type="cubed_sphere", resolution=2, nlev=4, vertical_coord="sigma"),
        dycore=DycoreConfig(discretization="cdgrid", dt=300.0),
        output=OutputConfig(output_dir="/tmp"),
        days=1,
    )
    grid, _ = create_cubed_sphere(config.grid.resolution), create_sigma_coordinate(config.grid.nlev)
    truth = _build_truth_dataset(
        dataset,
        start_time="2022-01-01T00:00:00",
        forecast_days=1,
        grid=grid,
        config=PreparationConfig(
            evaluation_resolution_deg=10.0,
            pressure_levels=(200, 500, 700, 850),
        ),
    )
    assert np.isfinite(np.asarray(truth["temperature"].isel(time=0).sel(level=850))).all()


def test_control_surface_forcing_keeps_day0_sst_and_daily_sic() -> None:
    time = np.asarray(["2022-01-01T00:00:00", "2022-01-02T00:00:00"], dtype="datetime64[s]")
    latitude = xr.DataArray(np.asarray([-5.0, 5.0]), dims=("latitude",), name="latitude")
    longitude = xr.DataArray(np.asarray([0.0, 10.0]), dims=("longitude",), name="longitude")
    observed = xr.Dataset(
        {
            "sea_surface_temperature": xr.DataArray(
                np.asarray(
                    [
                        [[300.0, np.nan], [302.0, np.nan]],
                        [[310.0, np.nan], [312.0, np.nan]],
                    ],
                    dtype=np.float32,
                ),
                coords={"time": time, "latitude": latitude, "longitude": longitude},
                dims=("time", "latitude", "longitude"),
            ),
            "land_surface_temperature": xr.DataArray(
                np.asarray(
                    [
                        [[295.0, 280.0], [296.0, 281.0]],
                        [[297.0, 282.0], [298.0, 283.0]],
                    ],
                    dtype=np.float32,
                ),
                coords={"time": time, "latitude": latitude, "longitude": longitude},
                dims=("time", "latitude", "longitude"),
            ),
            "land_sea_mask": xr.DataArray(
                np.asarray([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32),
                coords={"latitude": latitude, "longitude": longitude},
                dims=("latitude", "longitude"),
            ),
            "sea_ice_cover": xr.DataArray(
                np.asarray(
                    [
                        [[0.1, np.nan], [0.3, np.nan]],
                        [[0.5, np.nan], [0.7, np.nan]],
                    ],
                    dtype=np.float32,
                ),
                coords={"time": time, "latitude": latitude, "longitude": longitude},
                dims=("time", "latitude", "longitude"),
            ),
        }
    )
    forcing = build_control_surface_forcing_dataset(observed)
    np.testing.assert_allclose(
        forcing["sea_surface_temperature"].isel(time=0).values[:, 0],
        np.asarray([300.0, 302.0], dtype=np.float32),
    )
    np.testing.assert_allclose(
        forcing["sea_surface_temperature"].isel(time=1).values[:, 0],
        np.asarray([300.0, 302.0], dtype=np.float32),
    )
    np.testing.assert_allclose(
        forcing["sea_surface_temperature"].isel(time=0).values[:, 1],
        np.asarray([280.0, 281.0], dtype=np.float32),
    )
    np.testing.assert_allclose(
        forcing["sea_surface_temperature"].isel(time=1).values[:, 1],
        np.asarray([282.0, 283.0], dtype=np.float32),
    )
    np.testing.assert_allclose(
        forcing["sea_ice_cover"].fillna(0.0).isel(longitude=0).transpose("time", "latitude").values,
        np.asarray([[0.1, 0.3], [0.5, 0.7]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        forcing["sea_ice_cover"].fillna(0.0).isel(longitude=1).transpose("time", "latitude").values,
        np.zeros((2, 2), dtype=np.float32),
    )


def test_model_surface_forcing_keeps_ocean_field_distinct_from_land_temperature() -> None:
    dataset = _synthetic_era5_dataset()
    grid = create_cubed_sphere(2)
    forcing = _build_model_surface_forcing_dataset(
        dataset,
        start_time="2022-01-01T00:00:00",
        forecast_days=1,
        grid=grid,
        config=PreparationConfig(era5_store="synthetic"),
    )
    land_fraction = np.asarray(forcing["land_fraction"], dtype=np.float32)
    fixed_ocean = np.asarray(forcing["fixed_ocean_sea_surface_temperature"].isel(time=0), dtype=np.float32)
    land_temperature = np.asarray(forcing["land_surface_temperature"].isel(time=0), dtype=np.float32)
    pure_land = land_fraction > 0.95
    assert pure_land.any()
    # The fixed ocean field may carry benign finite fill values over land,
    # but it should no longer just inherit the land temperature there.
    assert not np.allclose(fixed_ocean[pure_land], land_temperature[pure_land])


def test_relative_forcing_wrapper_shifts_absolute_day() -> None:
    calls: list[float] = []

    def getter(day: float):
        calls.append(day)
        return day, day + 1.0

    wrapped = wrap_relative_forcing_getter(getter, 120.0)
    assert wrapped(121.5) == (1.5, 2.5)
    assert calls == [1.5]


def test_run_experiment_forces_prepared_custom_boundary_contract(tmp_path: Path, monkeypatch) -> None:
    case_dir = tmp_path / "case"
    prepared_dir = case_dir / "_prepared"
    prepared_dir.mkdir(parents=True)
    config = ExperimentConfig(
        grid=GridConfig(grid_type="cubed_sphere", resolution=2, nlev=4),
        dycore=DycoreConfig(discretization="cdgrid", dt=300.0),
        output=OutputConfig(output_dir=str(case_dir)),
        days=1,
        dataset="analytical",
        radiation="gray",
        cloud_scheme="none",
        microphysics="none",
        convection="none",
        turbulence="none",
    )
    (prepared_dir / "experiment_config.json").write_text(
        __import__("json").dumps(experiment_config_to_dict(config)),
        encoding="utf-8",
    )

    captured = {}

    class _FakeDriver:
        def __init__(self):
            self.grid = "grid"
            self.sigma = "sigma"

        def load_checkpoint(self, path):
            captured["restart_path"] = str(path)
            return 0, 0.0

        def run(self, **kwargs):
            captured["run_kwargs"] = kwargs
            return "COMPLETED"

    class _FakeCollector:
        def __init__(self, *args, **kwargs):
            pass

        def record(self, *args, **kwargs):
            return None

        def has_records(self):
            return False

    def _fake_build_uncoupled_driver(config, **kwargs):
        captured["config"] = config
        return _FakeDriver()

    monkeypatch.setattr(
        "legoesm.ml.s2s.legoesm_slab.rollout._build_uncoupled_driver",
        _fake_build_uncoupled_driver,
    )
    monkeypatch.setattr(
        "legoesm.ml.s2s.legoesm_slab.rollout._load_model_surface_forcing",
        lambda *_args, **_kwargs: "forcing",
    )
    monkeypatch.setattr(
        "legoesm.ml.s2s.legoesm_slab.rollout.DailyForecastCollector",
        _FakeCollector,
    )

    metadata = PreparedCaseMetadata(
        start_time="2022-01-01T00:00:00",
        forecast_days=1,
        forcing_start_day=0.0,
        forcing_resolution_deg=1.0,
        evaluation_resolution_deg=5.0,
        pressure_levels=(200, 500, 700, 850),
        experiment_config=experiment_config_to_dict(config),
        coupled={"ocean_mode": "slab", "land_mode": "none", "f_land_mode": "prepared"},
    )

    status, forecast_path = _run_experiment(
        "uncoupled",
        case_dir=case_dir,
        metadata=metadata,
        export_config=ForecastExportConfig(),
        compiled=False,
    )
    assert status == "COMPLETED"
    assert forecast_path is None
    assert captured["config"].dataset == "custom"
    assert captured["config"].forcing_path.endswith("surface_forcing_20220101_1d.nc")
    assert captured["config"].sst_var == "sea_surface_temperature"
    assert captured["config"].sic_var == "sea_ice_cover"
    assert captured["config"].time_var == "time"
    assert captured["config"].lat_var == "latitude"
    assert captured["config"].lon_var == "longitude"


def test_weighted_rmse_ignores_partial_nan_overlap() -> None:
    latitude = xr.DataArray(np.asarray([-30.0, 30.0]), dims=("latitude",), name="latitude")
    longitude = xr.DataArray(np.asarray([0.0, 10.0]), dims=("longitude",), name="longitude")
    forecast = xr.DataArray(
        np.asarray([[1.0, np.nan], [3.0, 5.0]], dtype=np.float32),
        coords={"latitude": latitude, "longitude": longitude},
        dims=("latitude", "longitude"),
    )
    truth = xr.DataArray(
        np.asarray([[0.0, 2.0], [1.0, np.nan]], dtype=np.float32),
        coords={"latitude": latitude, "longitude": longitude},
        dims=("latitude", "longitude"),
    )
    score = weighted_rmse(forecast, truth)
    assert np.isfinite(score)
    assert score > 0.0


def test_pressure_level_interpolation_masks_out_of_range_targets() -> None:
    field = np.asarray([[[[10.0, 20.0, 30.0]]]], dtype=np.float32)
    pressure = np.asarray([[[[20000.0, 50000.0, 90000.0]]]], dtype=np.float32)
    interpolated = _interpolate_model_field_to_pressure_levels(field, pressure, [100, 500, 1000])
    assert np.isnan(interpolated[..., 0]).all()
    np.testing.assert_allclose(interpolated[..., 1], 20.0)
    assert np.isnan(interpolated[..., 2]).all()


def test_exported_geopotential_uses_virtual_temperature() -> None:
    sigma = create_sigma_coordinate(2)
    phis = np.zeros((6, 2, 2), dtype=np.float32)
    temperature = np.full((6, 2, 2, 2), 280.0, dtype=np.float32)
    qv_dry = np.zeros((6, 2, 2, 2), dtype=np.float32)
    qv_moist = np.full((6, 2, 2, 2), 0.02, dtype=np.float32)
    state_dry = types.SimpleNamespace(
        T=types.SimpleNamespace(data=temperature),
        q_v=types.SimpleNamespace(data=qv_dry),
        p_s=types.SimpleNamespace(data=np.full((6, 2, 2), 1.0e5, dtype=np.float32)),
        phis=types.SimpleNamespace(data=phis),
    )
    state_moist = types.SimpleNamespace(
        T=types.SimpleNamespace(data=temperature),
        q_v=types.SimpleNamespace(data=qv_moist),
        p_s=types.SimpleNamespace(data=np.full((6, 2, 2), 1.0e5, dtype=np.float32)),
        phis=types.SimpleNamespace(data=phis),
    )
    phi_dry = _compute_model_geopotential(state_dry, sigma)
    phi_moist = _compute_model_geopotential(state_moist, sigma)
    assert np.all(phi_moist > phi_dry)


def test_geopotential_accepts_optional_specific_humidity() -> None:
    sigma = create_sigma_coordinate(2)
    temperature = jnp.full((1, 1, 1, 2), 280.0, dtype=jnp.float32)
    qv = jnp.full((1, 1, 1, 2), 0.02, dtype=jnp.float32)
    p_s = jnp.full((1, 1, 1), 1.0e5, dtype=jnp.float32)
    phis = jnp.zeros((1, 1, 1), dtype=jnp.float32)
    phi_dry = compute_geopotential(temperature, p_s, sigma, phis)
    phi_moist = compute_geopotential(temperature, p_s, sigma, phis, q_v=qv)
    assert np.all(np.asarray(phi_moist) > np.asarray(phi_dry))


def test_masked_cubedsphere_regrid_preserves_partial_support() -> None:
    weights = get_cubedsphere_to_latlon_weights(2, n_lon=36, n_lat=19)
    field = np.ones((6, 2, 2, 1), dtype=np.float32)
    field[0, 0, 0, 0] = np.nan
    regridded = _apply_cubedsphere_to_latlon_3d_masked(field, weights)
    assert np.isfinite(regridded[..., 0]).any()
    assert not np.isnan(regridded[..., 0]).all()


def test_relative_forcing_interpolation_clamps_at_endpoint() -> None:
    forcing = jnp.asarray([10.0, 20.0, 30.0], dtype=jnp.float32)
    times = jnp.asarray([0.0, 1.0, 2.0], dtype=jnp.float32)
    from legoesm.ml.s2s.legoesm_slab.rollout import _interp_time_series

    assert float(_interp_time_series(forcing, times, 0.0)) == pytest.approx(10.0)
    assert float(_interp_time_series(forcing, times, 1.0)) == pytest.approx(20.0)
    assert float(_interp_time_series(forcing, times, 2.0)) == pytest.approx(30.0)
    assert float(_interp_time_series(forcing, times, 3.0)) == pytest.approx(30.0)


def test_pack_forcing_preserves_missing_inline_radiation_overrides() -> None:
    forcing = pack_forcing(
        sst=jnp.ones((2, 2), dtype=jnp.float32),
        sic=jnp.zeros((2, 2), dtype=jnp.float32),
        day_of_year=1.0,
        seconds_of_day=0.0,
        solar_weights=jnp.zeros((0,), dtype=jnp.float32),
        s_0=1361.0,
        o3_vmr=None,
        aerosol_od=None,
        ghg_vmr=None,
    )
    assert forcing.o3_vmr is None
    assert forcing.aerosol_od is None
    assert forcing.ghg_vmr.shape == (0,)


def test_masked_regrid_does_not_bleed_land_values_into_ocean_sst() -> None:
    weights = get_cubedsphere_to_latlon_weights(2, n_lon=36, n_lat=19)
    sst = np.full((6, 2, 2), 280.0, dtype=np.float32)
    land_fraction = np.zeros((6, 2, 2), dtype=np.float32)
    sst[0, 0, 0] = 330.0
    land_fraction[0, 0, 0] = 1.0
    ocean_only = np.where(land_fraction < 0.5, sst, np.nan).astype(np.float32)
    regridded = _apply_cubedsphere_to_latlon_masked(ocean_only, weights)
    assert np.isfinite(regridded).any()
    assert float(np.nanmax(regridded)) < 300.0


def test_cubedsphere_wind_rotation_roundtrip_preserves_geographic_components() -> None:
    grid = create_cubed_sphere(2)
    u_east = jnp.full((6, 2, 2, 1), 5.0, dtype=jnp.float32)
    v_north = jnp.full((6, 2, 2, 1), -2.0, dtype=jnp.float32)
    u_grid, v_grid = rotate_winds_geo_to_grid(u_east, v_north, grid.angle[..., None])
    u_geo, v_geo = _winds_to_geographic(
        grid,
        np.asarray(u_grid, dtype=np.float32),
        np.asarray(v_grid, dtype=np.float32),
    )
    np.testing.assert_allclose(u_geo, np.asarray(u_east), atol=1.0e-6)
    np.testing.assert_allclose(v_geo, np.asarray(v_north), atol=1.0e-6)


def test_cubedsphere_edge_blending_reduces_face_discontinuities() -> None:
    grid = create_cubed_sphere(2)
    config = PreparationConfig(cubedsphere_edge_blend_strength=0.5, cubedsphere_edge_blend_width=1)

    scalar = np.stack(
        [np.full((2, 2), float(face), dtype=np.float32) for face in range(6)],
        axis=0,
    )
    scalar_blend = _blend_cubedsphere_scalar(scalar, config=config)
    assert _mean_unique_edge_rmse(scalar_blend) < _mean_unique_edge_rmse(scalar)

    u = np.stack(
        [np.full((2, 2, 1), 2.0 + face, dtype=np.float32) for face in range(6)],
        axis=0,
    )
    v = np.stack(
        [np.full((2, 2, 1), -1.0 - face, dtype=np.float32) for face in range(6)],
        axis=0,
    )
    u_blend, v_blend = _blend_cubedsphere_vector(u, v, grid=grid, config=config)
    u_geo_before, v_geo_before = _winds_to_geographic(grid, u, v)
    u_geo_after, v_geo_after = _winds_to_geographic(grid, u_blend, v_blend)
    assert _mean_unique_edge_rmse(u_geo_after[..., 0]) < _mean_unique_edge_rmse(u_geo_before[..., 0])
    assert _mean_unique_edge_rmse(v_geo_after[..., 0]) < _mean_unique_edge_rmse(v_geo_before[..., 0])


def test_custom_amip_forcing_clamps_instead_of_wrapping_at_endpoint() -> None:
    forcing = AMIPForcing(
        times=jnp.asarray([0.0, 1.0], dtype=jnp.float32),
        sst=jnp.asarray(
            [
                [[300.0, 301.0]],
                [[310.0, 311.0]],
            ],
            dtype=jnp.float32,
        ),
        sic=jnp.asarray(
            [
                [[0.1, 0.2]],
                [[0.3, 0.4]],
            ],
            dtype=jnp.float32,
        ),
        config=AMIPForcingConfig(dataset="custom", path="synthetic.nc"),
    )

    sst_0, sic_0 = get_forcing_at_time(forcing, 0.0)
    sst_1, sic_1 = get_forcing_at_time(forcing, 1.0)
    sst_2, sic_2 = get_forcing_at_time(forcing, 2.0)

    np.testing.assert_allclose(np.asarray(sst_0), np.asarray([[300.0, 301.0]], dtype=np.float32))
    np.testing.assert_allclose(np.asarray(sst_1), np.asarray([[310.0, 311.0]], dtype=np.float32))
    np.testing.assert_allclose(np.asarray(sst_2), np.asarray([[310.0, 311.0]], dtype=np.float32))
    np.testing.assert_allclose(np.asarray(sic_1), np.asarray([[0.3, 0.4]], dtype=np.float32))
    np.testing.assert_allclose(np.asarray(sic_2), np.asarray([[0.3, 0.4]], dtype=np.float32))


def test_postprocess_case_writes_metrics_and_plots(tmp_path: Path) -> None:
    case_dir = tmp_path / "inference_20220101"
    prepared_dir = case_dir / "_prepared"
    coupled_dir = case_dir / "coupled"
    uncoupled_dir = case_dir / "uncoupled"
    prepared_dir.mkdir(parents=True)
    coupled_dir.mkdir()
    uncoupled_dir.mkdir()

    truth = _synthetic_truth_dataset()
    truth.to_netcdf(prepared_dir / "truth.nc")
    _forecast_from_truth(truth, offset=0.1).to_netcdf(coupled_dir / "forecast.nc")
    _forecast_from_truth(truth, offset=1.0).to_netcdf(uncoupled_dir / "forecast.nc")
    (case_dir / "run_summary.json").write_text("{}", encoding="utf-8")

    outputs = postprocess_case(case_dir, snapshot_days=(1, 2))
    assert outputs["daily_coupled_metrics"].exists()
    assert outputs["window_uncoupled_metrics"].exists()
    assert outputs["headline_rmse_timeseries"].exists()
    assert outputs["enso_series"].exists()

    coupled_rows = _read_csv(outputs["daily_coupled_metrics"])
    uncoupled_rows = _read_csv(outputs["daily_uncoupled_metrics"])
    assert coupled_rows and uncoupled_rows


def test_postprocess_case_aligns_sst_mask_after_longitude_normalization(tmp_path: Path) -> None:
    case_dir = tmp_path / "inference_20220101"
    prepared_dir = case_dir / "_prepared"
    coupled_dir = case_dir / "coupled"
    uncoupled_dir = case_dir / "uncoupled"
    prepared_dir.mkdir(parents=True)
    coupled_dir.mkdir()
    uncoupled_dir.mkdir()

    truth = _synthetic_truth_dataset().assign_coords(
        longitude=xr.DataArray(np.asarray([20.0, 200.0]), dims=("longitude",), name="longitude")
    )
    truth["sea_surface_temperature"] = xr.DataArray(
        np.asarray(
            [
                [[np.nan, 300.0], [np.nan, 301.0]],
                [[np.nan, 302.0], [np.nan, 303.0]],
            ],
            dtype=np.float32,
        ),
        coords={
            "time": truth["time"],
            "latitude": truth["latitude"],
            "longitude": truth["longitude"],
        },
        dims=("time", "latitude", "longitude"),
    )
    truth.to_netcdf(prepared_dir / "truth.nc")

    coupled = normalize_longitude_data(_forecast_from_truth(truth, offset=0.1))
    uncoupled = normalize_longitude_data(_forecast_from_truth(truth, offset=1.0))
    coupled.to_netcdf(coupled_dir / "forecast.nc")
    uncoupled.to_netcdf(uncoupled_dir / "forecast.nc")
    (case_dir / "run_summary.json").write_text("{}", encoding="utf-8")

    outputs = postprocess_case(case_dir, snapshot_days=(1,))
    coupled_rows = _read_csv(outputs["daily_coupled_metrics"])
    uncoupled_rows = _read_csv(outputs["daily_uncoupled_metrics"])
    assert float(coupled_rows[0]["rmse.sea_surface_temperature"]) == pytest.approx(0.1, abs=1.0e-4)
    assert float(uncoupled_rows[0]["rmse.sea_surface_temperature"]) == pytest.approx(1.0, abs=1.0e-4)


def test_filter_start_times_to_year_keeps_november_15_and_drops_december_1() -> None:
    start_times = generate_semimonthly_start_times(2022)
    filtered = filter_start_times_to_year(start_times, forecast_days=42, year=2022)
    assert "2022-11-15T00:00:00" in filtered
    assert "2022-12-01T00:00:00" not in filtered


def test_compute_nino34_series_returns_time_series() -> None:
    truth = _synthetic_truth_dataset()
    series = compute_nino34_series(truth)
    assert series.ndim == 1
    assert series.sizes["time"] == truth.sizes["time"]


def test_rrtmgp_solver_uses_runtime_solar_constant(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, float] = {}

    def _fake_solve_lw(*args, **kwargs):
        atmos_state = args[4]
        captured["irrad"] = float(np.asarray(atmos_state.irrad).reshape(-1)[0])
        pressure = args[0]
        return {
            "flux_up": jnp.zeros_like(pressure),
            "flux_down": jnp.zeros_like(pressure),
            "flux_net": jnp.zeros_like(pressure),
        }

    def _fake_solve_sw(*args, **kwargs):
        pressure = args[0]
        return {
            "flux_up": jnp.zeros_like(pressure),
            "flux_down": jnp.zeros_like(pressure),
            "flux_net": jnp.zeros_like(pressure),
        }

    def _fake_heating_rate(flux_net, pressure, dp=None):
        del pressure, dp
        return jnp.zeros_like(flux_net)

    from legoesm.atmosphere.physics.radiation.rrtmgp import rrtmgp as rrtmgp_module

    monkeypatch.setattr(rrtmgp_module.two_stream, "solve_lw", _fake_solve_lw)
    monkeypatch.setattr(rrtmgp_module.two_stream, "solve_sw", _fake_solve_sw)
    monkeypatch.setattr(rrtmgp_module.two_stream, "compute_heating_rate", _fake_heating_rate)

    solver = object.__new__(RRTMGP)
    solver._config = types.SimpleNamespace(
        sfc_albedo=0.06,
        sfc_emissivity=0.98,
        S_0=1361.0,
        include_clouds=False,
        use_scan=False,
        aerosol_ssa=0.9,
        aerosol_g=0.7,
    )
    solver.optics_lib = types.SimpleNamespace(
        gas_optics_lw=types.SimpleNamespace(kmajor=jnp.asarray([0.0], dtype=jnp.float32)),
        n_gpt_sw=0,
    )
    solver.atmospheric_state = types.SimpleNamespace(vmr=None)

    solver.solve_columns(
        T=jnp.full((1, 2), 280.0, dtype=jnp.float32),
        p_full=jnp.asarray([[80000.0, 95000.0]], dtype=jnp.float32),
        p_half=jnp.asarray([[70000.0, 90000.0, 100000.0]], dtype=jnp.float32),
        sfc_temperature=jnp.asarray([290.0], dtype=jnp.float32),
        q_v=jnp.asarray([[0.005, 0.01]], dtype=jnp.float32),
        cos_zenith=jnp.asarray([0.5], dtype=jnp.float32),
        solar_constant=1401.0,
    )

    assert captured["irrad"] == pytest.approx(1401.0)
