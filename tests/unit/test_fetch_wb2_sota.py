"""Login-safe unit tests for scripts/data/fetch_wb2_sota.py (JAX-free, no network).

Mirrors ``test_run_weatherbench_eval_cli.py``: the module's top level must NOT
import jax / xarray / gcsfs, so it loads on a login node and this test exercises
only the arg-parse layer and the CSV writer. The CSV round-trips through the real
``evaluations.baselines.load_sota_headline`` consumer.
"""
import importlib.util
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]


def _load_by_path(name, rel):
    spec = importlib.util.spec_from_file_location(name, _ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# Load both modules directly by FILE PATH so neither the fetch script's top level
# NOR evaluations/__init__.py (which imports jax) is pulled in — keeps this test
# JAX-free and login-node safe. baselines.py itself imports only csv/math/pathlib.
mod = _load_by_path("fetch_wb2_sota", "scripts/data/fetch_wb2_sota.py")
_baselines = _load_by_path("wb2_baselines", "evaluations/baselines.py")
load_sota_headline = _baselines.load_sota_headline


# ---------------------------------------------------------------- arg-parse ---
def test_argparse_defaults():
    cfg = mod.build_fetch_config_from_args([])
    assert cfg.source == mod.WB2_RESULTS_BUCKET
    assert cfg.resolution == mod.WB2_RESOLUTION
    assert cfg.year == mod.WB2_YEAR
    assert cfg.models == mod.DEFAULT_MODELS
    assert cfg.leads_hours == mod.DEFAULT_LEADS_HOURS
    assert cfg.out == "config/wb/sota/wb2_headline_rmse.csv"


def test_argparse_overrides():
    cfg = mod.build_fetch_config_from_args(
        ["--source", "gs://my/benchmark_results", "--resolution", "64x32",
         "--year", "2018", "--models", "hres,graphcast", "--leads", "24,120",
         "--out", "out/wb2.csv"])
    assert cfg.source == "gs://my/benchmark_results"
    assert cfg.resolution == "64x32"
    assert cfg.year == 2018
    assert cfg.models == ("hres", "graphcast")
    assert cfg.leads_hours == (24, 120)
    assert cfg.out == "out/wb2.csv"


def test_local_source_override_accepted():
    cfg = mod.build_fetch_config_from_args(["--source", "/data/wb2_results"])
    assert cfg.source == "/data/wb2_results"


def test_unknown_model_hard_errors():
    """A typo'd/unknown reference model is a hard SystemExit, not a silent skip."""
    with pytest.raises(SystemExit, match="unknown reference model"):
        mod.build_fetch_config_from_args(["--models", "graphcast,BogusNet"])


# -------------------------------------------------------- filename template ---
def test_result_filename_template():
    """The bucket filename is {model}_vs_era5_{resolution}_{year}.nc (flat)."""
    assert mod.result_filename("graphcast", "240x121", 2020) == (
        "graphcast_vs_era5_240x121_2020.nc")
    assert mod.result_filename("neuralgcm_hres", "240x121", 2020) == (
        "neuralgcm_hres_vs_era5_240x121_2020.nc")
    assert mod.result_filename("climatology", "64x32", 2018) == (
        "climatology_vs_era5_64x32_2018.nc")


def test_bad_leads_rejected():
    with pytest.raises(SystemExit):
        mod.build_fetch_config_from_args(["--leads", "0,24"])
    with pytest.raises(SystemExit):
        mod.build_fetch_config_from_args(["--leads", ""])


# ------------------------------------------------------------- CSV writer -----
# Synthetic records use the REAL emitted schema: 'variable' holds the WB2 LONG
# name and 'level' the hPa int (that is what load_sota_headline + the plotter's
# FIELD_KEY_TO_SOTA key on), and 'model' the display name.
def _records():
    return [
        {"model": "IFS-HRES", "variable": "geopotential", "level": 500,
         "lead_hours": 24, "rmse": 152.3},
        {"model": "IFS-HRES", "variable": "temperature", "level": 850,
         "lead_hours": 72, "rmse": 1.35},
        {"model": "GraphCast", "variable": "specific_humidity", "level": 700,
         "lead_hours": 24, "rmse": 0.00071},
        {"model": "GraphCast", "variable": "u_component_of_wind", "level": 850,
         "lead_hours": 120, "rmse": 2.4},
    ]


def test_write_csv_roundtrips_through_baselines(tmp_path):
    out = tmp_path / "sota.csv"
    n = mod.write_sota_csv(_records(), out)
    assert n == 4
    assert out.exists()

    # exact column order + header
    header = out.read_text().splitlines()[0]
    assert header == "model,variable,level,lead_hours,rmse"

    loaded = load_sota_headline(out)          # the REAL consumer
    assert set(loaded) == {"IFS-HRES", "GraphCast"}
    assert loaded["IFS-HRES"][("geopotential", 500, 24)] == pytest.approx(152.3)
    assert loaded["IFS-HRES"][("temperature", 850, 72)] == pytest.approx(1.35)
    assert loaded["GraphCast"][("specific_humidity", 700, 24)] == pytest.approx(0.00071)
    assert loaded["GraphCast"][("u_component_of_wind", 850, 120)] == pytest.approx(2.4)


def test_write_csv_duplicate_key_raises(tmp_path):
    dup = _records() + [
        {"model": "IFS-HRES", "variable": "z500", "level": 500,
         "lead_hours": 24, "rmse": 200.0}]          # same (model,var,level,lead)
    with pytest.raises(ValueError, match="duplicate SOTA key"):
        mod.write_sota_csv(dup, tmp_path / "dup.csv")
    # nothing written on failure
    assert not (tmp_path / "dup.csv").exists()


def test_write_csv_nonfinite_rmse_raises(tmp_path):
    bad = [{"model": "m", "variable": "z500", "level": 500,
            "lead_hours": 24, "rmse": float("nan")}]
    with pytest.raises(ValueError, match="non-finite rmse"):
        mod.write_sota_csv(bad, tmp_path / "nan.csv")
    assert not (tmp_path / "nan.csv").exists()


def test_write_csv_nonpositive_rmse_raises(tmp_path):
    bad = [{"model": "m", "variable": "z500", "level": 500,
            "lead_hours": 24, "rmse": 0.0}]
    with pytest.raises(ValueError, match="non-positive rmse"):
        mod.write_sota_csv(bad, tmp_path / "zero.csv")


def test_write_csv_empty_raises(tmp_path):
    with pytest.raises(ValueError, match="no SOTA records"):
        mod.write_sota_csv([], tmp_path / "empty.csv")


# --------------------------------------------------- mapping self-consistency -
# The CRUX of the overlay lining up: each emitted (csv_variable, csv_level) must
# match a value in plot_wb_scorecard.FIELD_KEY_TO_SOTA (which the plotter uses to
# select SOTA rows for a scorecard field_key). The plotter is import-light + JAX-
# free, so we load it by file path (like the fetch script) to read the real map.
_plotter = _load_by_path("plot_wb_scorecard", "scripts/plot/plot_wb_scorecard.py")


def test_emitted_var_level_pairs_match_plotter_mapping():
    """Every (csv_variable, csv_level) we emit overlays a real scorecard field.

    The plotter maps field_key -> (wb2_variable, level) in FIELD_KEY_TO_SOTA and
    matches SOTA rows by that pair. If a row we write is not a value in that map,
    the overlay would silently drop it. (t500 is intentionally exempt: it is not a
    scorecard headline field, just a harmless extra upper-air row.)
    """
    valid_pairs = set(_plotter.FIELD_KEY_TO_SOTA.values())
    for hv in mod.HEADLINE_VARS:
        pair = (hv.csv_variable, hv.csv_level)
        if hv.scorecard_field_key == "t500":
            assert pair not in valid_pairs   # sanity: t500 truly is not mapped
            continue
        assert pair in valid_pairs, (hv.scorecard_field_key, pair)
        # and the declared scorecard_field_key resolves to exactly this pair
        assert _plotter.FIELD_KEY_TO_SOTA[hv.scorecard_field_key] == pair


def test_headline_vars_are_upper_air_only():
    """Surface fields (t2m/mslp/10m wind) are dropped — none present in HEADLINE_VARS."""
    surface = {"2m_temperature", "mean_sea_level_pressure",
               "10m_wind_speed", "10m_u_component_of_wind",
               "10m_v_component_of_wind"}
    for hv in mod.HEADLINE_VARS:
        assert hv.wb2_var not in surface, hv.wb2_var
        assert hv.csv_level != 0, hv          # no surface (level-0) rows


def test_display_names_cover_default_models():
    for m in mod.DEFAULT_MODELS:
        assert m in mod.MODEL_DISPLAY_NAMES
