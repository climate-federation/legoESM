"""Tests for the WB2 ERA5 case builder (#919) — synthetic in-memory dataset only.

No network: a small xarray Dataset with the WB2 headline variable names drives
both the case-builder unit checks (headline mapping, climatology, forcing,
fail-loud) and a tiny T-truncation end-to-end run of the eval CLI. Requires
JAX_ENABLE_X64 (spectral SH analysis of the ERA5 IC).
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from legoesm.ml.channel_packing import WB2_PRESSURE_LEVELS
from legoesm.training.era5_to_state import TrainingERA5Config

from evaluations.wb_era5_cases import (
    build_forecast_cases,
    climatology_from_cases,
)
from legoesm import constants

_ROOT = Path(__file__).resolve().parents[2]

# Known synthetic values so the headline mapping is exact after regridding.
_T850_K = 250.0
_Z500_M = 100.0           # geopotential@500 = (_Z500_M + time_idx) * g -> z500 = _Z500_M + t
_U10, _V10 = 3.0, 4.0     # -> wind_speed_10m = 5


def _synthetic_era5(n_time=3, n_lat=24, n_lon=48, start="2020-01-01", cadence_h=6):
    """WB2-style ERA5 dataset (all 13 WB2 levels, ascending lat) with constant/
    analytic headline fields. geopotential@500 varies with time so climatology
    (the mean over verification snapshots) is non-trivial."""
    lat = np.linspace(-90.0, 90.0, n_lat)                 # ascending S->N
    lon = np.linspace(0.0, 360.0, n_lon, endpoint=False)  # [0, 360)
    levels = np.array(WB2_PRESSURE_LEVELS, dtype=np.float64)  # hPa, descending
    time = np.array(
        [np.datetime64(start) + np.timedelta64(cadence_h * i, "h") for i in range(n_time)],
        dtype="datetime64[ns]")
    nt, nl, ny, nx = n_time, len(levels), n_lat, n_lon

    temp = np.full((nt, nl, ny, nx), _T850_K)             # isothermal -> t850 = 250 K
    geop = np.zeros((nt, nl, ny, nx))
    i500 = int(np.where(levels == 500)[0][0])
    for ti in range(nt):
        geop[ti, i500] = (_Z500_M + ti) * constants.g     # z500 = _Z500_M + time_idx
    zeros4 = np.zeros((nt, nl, ny, nx))

    def _surf(val):
        return np.full((nt, ny, nx), float(val))

    return xr.Dataset(
        {
            "temperature": (("time", "level", "lat", "lon"), temp),
            "u_component_of_wind": (("time", "level", "lat", "lon"), zeros4),
            "v_component_of_wind": (("time", "level", "lat", "lon"), zeros4),
            "specific_humidity": (("time", "level", "lat", "lon"), np.full((nt, nl, ny, nx), 1e-4)),
            "geopotential": (("time", "level", "lat", "lon"), geop),
            "surface_pressure": (("time", "lat", "lon"), _surf(1.0e5)),
            "mean_sea_level_pressure": (("time", "lat", "lon"), _surf(1.013e5)),
            "2m_temperature": (("time", "lat", "lon"), _surf(288.0)),
            "10m_u_component_of_wind": (("time", "lat", "lon"), _surf(_U10)),
            "10m_v_component_of_wind": (("time", "lat", "lon"), _surf(_V10)),
            "geopotential_at_surface": (("lat", "lon"), np.zeros((ny, nx))),
        },
        coords={"time": time, "level": levels, "lat": lat, "lon": lon},
    )


def _grid_sigma(n_max=8, nlev=4):
    from legoesm.grids.gaussian import create_gaussian_grid
    from legoesm.grids.vertical import create_sigma_coordinate
    return create_gaussian_grid(n_max), create_sigma_coordinate(nlev)


def _cfg():
    return TrainingERA5Config(zarr_store="synthetic", dt_hours=6)


def test_headline_mapping():
    """constant temperature@850 -> t850=250; geopotential@500=(100+t)*g -> z500=100+t;
    hypot(u10,v10) -> wind_speed_10m=5. Regridding a constant field is exact."""
    ds = _synthetic_era5()
    grid, sigma = _grid_sigma()
    cases = build_forecast_cases(
        _cfg(), grid, sigma, leads_hours=[6, 12], eval_year=2020, n_inits=1,
        init_stride_hours=6, resolution_deg=6.0, ds=ds)
    assert len(cases) == 1
    # lead 6 h from init 0 (i_ic=0) verifies at time index 1 -> z500 = 100 + 1.
    v6 = cases[0].verif_by_lead[6]["fields"]
    v12 = cases[0].verif_by_lead[12]["fields"]
    assert np.allclose(v6["t850"], _T850_K)
    assert np.allclose(v6["z500"], _Z500_M + 1.0)          # geopotential/g
    assert np.allclose(v12["z500"], _Z500_M + 2.0)         # time index 2
    assert np.allclose(v6["wind_speed_10m"], 5.0)          # hypot(3, 4)
    assert np.allclose(v6["u10"], _U10) and np.allclose(v6["v10"], _V10)
    # every headline key present + all-True valid mask (scorer intersects it).
    from evaluations.wb_forecast import HEADLINE_FIELD_KEYS
    assert set(v6) == set(HEADLINE_FIELD_KEYS)
    assert all(cases[0].verif_by_lead[6]["valid"][k].all() for k in HEADLINE_FIELD_KEYS)


def test_climatology_is_mean_of_verifs():
    ds = _synthetic_era5()
    grid, sigma = _grid_sigma()
    cases = build_forecast_cases(
        _cfg(), grid, sigma, leads_hours=[6, 12], eval_year=2020, n_inits=1,
        init_stride_hours=6, resolution_deg=6.0, ds=ds)
    clim = climatology_from_cases(cases)
    # z500 verifs are 101 and 102 -> mean 101.5.
    assert np.allclose(clim["z500"], _Z500_M + 1.5)
    # explicit: clim == elementwise mean of the two verification snapshots.
    v6 = cases[0].verif_by_lead[6]["fields"]["z500"]
    v12 = cases[0].verif_by_lead[12]["fields"]["z500"]
    assert np.allclose(clim["z500"], 0.5 * (np.asarray(v6) + np.asarray(v12)))


def test_forcing_structure():
    """forcing = flat ncol T_sfc/sic + 1-based integer doy + seconds-of-day
    (scale_build calendar convention); init at 2020-01-01T00 -> doy 1, 0 s."""
    ds = _synthetic_era5()
    grid, sigma = _grid_sigma()
    cases = build_forecast_cases(
        _cfg(), grid, sigma, leads_hours=[6], eval_year=2020, n_inits=1,
        init_stride_hours=6, resolution_deg=6.0, ds=ds)
    f = cases[0].forcing
    ncol = int(grid.n_lat) * int(grid.n_lon)
    assert f["T_sfc"].shape == (ncol,)
    assert f["sic"].shape == (ncol,) and float(np.max(np.abs(np.asarray(f["sic"])))) == 0.0
    assert float(f["day_of_year"]) == 1.0            # 1-based integer day
    assert float(f["seconds_of_day"]) == 0.0
    # init_state is a spectral state (has the prognostic spectral fields).
    assert hasattr(cases[0].init_state, "vor_hat")


def test_raises_on_bad_lead():
    ds = _synthetic_era5()
    grid, sigma = _grid_sigma()
    with pytest.raises(ValueError, match="multiple of the ERA5 cadence"):
        build_forecast_cases(
            _cfg(), grid, sigma, leads_hours=[4], eval_year=2020, n_inits=1,
            init_stride_hours=6, resolution_deg=6.0, ds=ds)   # 4 h not a multiple of 6 h


def test_raises_on_lead_beyond_store():
    ds = _synthetic_era5(n_time=3)                    # time indices 0,1,2
    grid, sigma = _grid_sigma()
    with pytest.raises(ValueError, match="beyond the ERA5 store"):
        build_forecast_cases(
            _cfg(), grid, sigma, leads_hours=[18], eval_year=2020, n_inits=1,
            init_stride_hours=6, resolution_deg=6.0, ds=ds)   # i_verif = 0 + 3 = 3 >= 3


def test_raises_on_init_beyond_store():
    ds = _synthetic_era5(n_time=3)
    grid, sigma = _grid_sigma()
    with pytest.raises(ValueError, match="beyond the ERA5 store"):
        build_forecast_cases(
            _cfg(), grid, sigma, leads_hours=[6], eval_year=2020, n_inits=10,
            init_stride_hours=6, resolution_deg=6.0, ds=ds)   # init 3 -> i_ic=3 >= 3


def test_raises_on_missing_variable():
    ds = _synthetic_era5().drop_vars("mean_sea_level_pressure")
    grid, sigma = _grid_sigma()
    with pytest.raises(ValueError, match="mean_sea_level_pressure"):
        build_forecast_cases(
            _cfg(), grid, sigma, leads_hours=[6], eval_year=2020, n_inits=1,
            init_stride_hours=6, resolution_deg=6.0, ds=ds)


def _load_cli():
    entry = _ROOT / "scripts" / "validate" / "run_weatherbench_eval.py"
    spec = importlib.util.spec_from_file_location("run_wb_eval_e2e", entry)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    return cli


def test_end_to_end_finite_scorecard(tmp_path):
    """T-small end-to-end: build components from a tiny YAML, serialize the
    (random-init) params as a checkpoint, run the driver against the synthetic
    dataset, and assert every scorecard entry is finite."""
    import json
    import types

    import equinox as eqx
    import yaml
    from legoesm.training.scale_build import build_mode_components

    yml = {
        "n_lat": 32, "n_lon": 64, "nlev": 4,
        "era5_zarr": "synthetic", "eval_years": [2020], "era5_cadence_hours": 6,
        "mode": "physics",
        "spectral": {"n_max": 10, "dt": 3600.0, "semi_implicit": True},
        "loss": {},
    }
    yaml_path = tmp_path / "tiny.yaml"
    yaml_path.write_text(yaml.safe_dump(yml))

    # Build the components + serialize the params skeleton as the checkpoint the
    # driver will deserialize (same structure -> round-trips exactly).
    mode_cfg = types.SimpleNamespace(mode="physics", training_core="spectral")
    _, _grid, _sigma, params, _mrs, _lc, _dt = build_mode_components(mode_cfg, yml)
    ckpt = tmp_path / "epoch_0000.eqx"
    eqx.tree_serialise_leaves(str(ckpt), params)

    ds = _synthetic_era5(n_time=3, n_lat=24, n_lon=48)
    out_path = tmp_path / "scorecard.json"
    cli = _load_cli()
    result = cli.main(
        ["--config", str(yaml_path), "--mode", "physics", "--checkpoint", str(ckpt),
         "--leads", "6,12", "--n-inits", "1", "--init-stride-hours", "6",
         "--resolution-deg", "12.0", "--out", str(out_path)],
        ds=ds)

    # scorecard.json written + the three sections present.
    on_disk = json.loads(out_path.read_text())
    assert set(on_disk) == {"meta", "model", "persistence", "climatology"}
    assert on_disk["meta"]["mode"] == "physics"
    assert "NOT comparable" in on_disk["meta"]["climatology_note"]

    from evaluations.wb_forecast import HEADLINE_FIELD_KEYS
    for section in ("model", "persistence", "climatology"):
        sc = on_disk[section]
        for key in HEADLINE_FIELD_KEYS:
            for lead in ("6", "12"):
                entry = sc[key][lead]
                for m in ("rmse", "acc", "bias"):
                    assert np.isfinite(entry[m]), (section, key, lead, m)
    # in-memory result matches the flat orchestrator keying via _nest.
    assert result["model"]["z500"]["6"]["rmse"] >= 0.0
