"""Direct controls for the committed WW3 input/metric adapter."""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from scripts.validate.validate_wave_ww3 import configure
from scripts.validate.wave_fidelity.case_io import moments, read_config
from scripts.validate.wave_fidelity.compare import errors, circular_errors


def test_metrics_detect_a_planted_error_and_zero_reference():
    reference = np.asarray([0., 1., 2.])
    prediction = reference.copy()
    prediction[1] += 1.
    score = errors(prediction, reference, np.ones(3, dtype=bool))
    assert score["max_absolute"] == 1.
    assert score["relative_l2"] > 0.
    assert score["max_absolute_index"] == [1]
    assert errors(prediction, np.zeros(3), np.ones(3, dtype=bool))["relative_l2"] is None
    assert circular_errors(np.asarray([1.]), np.asarray([359.]), True)["max_absolute"] == 2.


def test_common_moments_include_lowest_frequency_bin():
    spectrum = np.zeros((2, 3), dtype=float)
    spectrum[0, 0] = 1.
    result = moments(spectrum, np.asarray([0., 180.]), np.asarray([.05, .055, .0605]))
    assert result["swh"] > 0
    assert np.isfinite(result["mwp"])


@pytest.mark.parametrize("name", ["tp1_1", "tp1_2", "tp1_4", "tp2_1"])
def test_oracle_case_configs_select_explicit_no_source_dynamics(name):
    path = Path(__file__).parents[3]/"scripts/validate/wave_fidelity/configs"/(name+".json")
    config = read_config(path)
    case = dict(x=np.arange(3.), y=np.arange(3.), theta=np.asarray([0., 180.]),
                freq=np.asarray([.05, .055, .0605]), depth=np.ones((3, 3))*50,
                initial=np.ones((3, 3, 2, 3)), wet=np.ones((3, 3), bool),
                boundary_mask=np.zeros((3, 3), bool))
    cfg, state, boundary, fixed = configure(case, config, jnp.float32)
    cfg.validate()
    assert cfg.network.parameterization == "none"
    assert cfg.network.use_transport_networks is False
    assert cfg.grid.boundary_mode == "physical"
    assert cfg.grid.coordinate_type == config["coordinate_type"]
    assert boundary.shape == state.spectra.shape


def test_manifest_config_serialisation_keeps_field_names():
    from legoesm.wave.config import (
        WaveConfig, WaveGridConfig, WaveNetworkConfig)
    from scripts.validate.validate_wave_ww3 import _plain

    config = WaveConfig(
        grid=WaveGridConfig(
            lon_deg=(0.0, 1.0), lat_deg=(1.0, 0.0),
            freq_hz=(0.05, 0.06), theta_rad=(0.0, np.pi)),
        network=WaveNetworkConfig(
            mlp_input_channels=16, transport_input_channels=6,
            transport_output_vectors=4, mlp_hidden_channels_0=2,
            mlp_hidden_channels_1=2, mlp_hidden_channels_2=2,
            transport_hidden_channels=2),
    )
    plain = _plain(config)
    assert isinstance(plain, dict)
    assert set(plain) >= {"grid", "time", "physics", "network", "gse"}
    assert plain["time"]["dt_global_s"] == config.time.dt_global_s
    assert plain["grid"]["lon_deg"] == [0.0, 1.0]
    assert plain["network"]["parameterization"] == "mlp"
    assert isinstance(plain["gse"], dict) and "enabled" in plain["gse"]


def test_free_mask_excludes_boundaries_and_land():
    from scripts.validate.validate_wave_ww3 import free_mask

    case = dict(wet=np.array([True, True, False, True]),
                boundary_mask=np.array([False, True, False, False]))
    np.testing.assert_array_equal(
        free_mask(case), [True, False, False, True])


def test_tp2_1_max_absolute_gate_is_enforced():
    from scripts.validate.validate_wave_ww3 import case_passed

    assert case_passed("tp2.1", {"rmse": 1e-6, "max_absolute": 1e-3})
    assert not case_passed("tp2.1", {"rmse": 1e-6, "max_absolute": 3e-3})
    assert not case_passed("tp1.1", {"rmse": 2e-4, "max_absolute": 0.0})
    assert case_passed("tp1.1", {"rmse": 1e-5, "max_absolute": 1.0})


def test_oracle_adapter_reads_the_ww3_axes_and_scoring_masks(tmp_path):
    """Synthetic netCDF: a transposed efth axis or scoring mask must fail."""
    import netCDF4 as nc
    from scripts.validate.wave_fidelity.case_io import load_case

    n_lat, n_lon, n_dir, n_freq = 3, 4, 4, 3
    folder = tmp_path / "case" / "input"
    folder.mkdir(parents=True)
    mapsta = np.asarray([[0, 1, 1, 0],
                         [1, 1, 2, 1],
                         [0, 1, 1, 0]], dtype=np.int32)
    hs = np.asarray([[1.0 + 10.0*y + x for x in range(n_lon)]
                     for y in range(n_lat)])[None]
    freq = np.asarray([0.05, 0.055, 0.0605])
    direction = np.asarray([90.0, 180.0, 270.0, 0.0])

    with nc.Dataset(folder / "grid.nc", "w") as ds:
        ds.createDimension("time", None)
        ds.createDimension("latitude", n_lat)
        ds.createDimension("longitude", n_lon)
        t = ds.createVariable("time", "i4", ("time",))
        t.units = "hours since 2000-01-01 00:00:00"
        t.calendar = "standard"
        t[:] = [0, 1]
        ds.createVariable("latitude", "f8", ("latitude",))[:] = np.arange(n_lat)
        ds.createVariable("longitude", "f8", ("longitude",))[:] = np.arange(n_lon)
        ds.createVariable("x", "f8", ("longitude",))[:] = np.arange(n_lon)
        ds.createVariable("y", "f8", ("latitude",))[:] = np.arange(n_lat)
        ds.createVariable("MAPSTA", "i4", ("latitude", "longitude"))[:] = mapsta
        ds.createVariable("dpt", "f8", ("time", "latitude", "longitude"))[:] = (
            np.full((1, n_lat, n_lon), 50.0))
        for name in ("hs", "dir", "t01"):
            ds.createVariable(name, "f8", ("time", "latitude", "longitude"))[:] = (
                hs if name == "hs" else np.zeros_like(hs))
        ds.WAVEWATCH_III_version_number = "6.07"
        ds.WAVEWATCH_III_switches = "TEST"

    spec = np.zeros((1, 1, n_freq, n_dir))
    spec[0, 0, 1, 2] = 1.0
    spec[0, 0, 0, 0] = 0.25
    with nc.Dataset(folder / "spec.nc", "w") as ds:
        ds.createDimension("time", None)
        ds.createDimension("station", 1)
        ds.createDimension("frequency", n_freq)
        ds.createDimension("direction", n_dir)
        ds.createDimension("name_len", 8)
        t = ds.createVariable("time", "i4", ("time",))
        t.units = "hours since 2000-01-01 00:00:00"
        t.calendar = "standard"
        t[:] = [0]
        ds.createVariable("frequency", "f8", ("frequency",))[:] = freq
        d = ds.createVariable("direction", "f8", ("direction",))
        d.standard_name = "sea_surface_wave_to_direction"
        d[:] = direction
        ds.createVariable(
            "efth", "f8",
            ("time", "station", "frequency", "direction"))[:] = spec
        ds.createVariable("x", "f8", ("time", "station"))[:] = [[1.0]]
        ds.createVariable("y", "f8", ("time", "station"))[:] = [[1.0]]
        names = np.zeros((1, 8), dtype="S1")
        names[0, :3] = [b"s", b"t", b"0"]
        ds.createVariable(
            "station_name", "S1", ("station", "name_len"))[:] = names

    config = dict(
        case="tp1.1", data_directory="case", grid_file="grid.nc",
        coordinate_type="Cartesian", bathymetry={"constant_m": 50.0},
        spectrum_files=["spec.nc"], station_coordinate_scale=1.0,
        reference_revision="test")
    case = load_case(config, tmp_path)

    assert case["initial"].shape == (n_lon, n_lat, n_dir, n_freq)
    assert case["wet"][1, 1] and not case["wet"][0, 0]
    assert case["boundary_mask"][2, 1] and not case["boundary_mask"][1, 1]
    assert case["depth"][1, 1] == 50.0 and case["depth"][0, 0] == 0.0
    # Initial spectrum = t=0 hs envelope * the station shape, (dir, freq).
    template = spec[0, 0].T
    m0 = moments(template, direction, freq)["m0"]
    expected = hs[0, 1, 1]**2 / (16.0*m0) * template
    np.testing.assert_allclose(case["initial"][1, 1], expected, rtol=1e-12)
    np.testing.assert_array_equal(case["initial"][0, 0], 0.0)
    np.testing.assert_array_equal(case["station_ix"], [1])
    np.testing.assert_array_equal(case["station_iy"], [1])
