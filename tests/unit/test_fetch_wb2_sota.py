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
    assert cfg.resolution_dir == mod.WB2_RESOLUTION_DIR
    assert cfg.models == mod.DEFAULT_MODELS
    assert cfg.leads_hours == mod.DEFAULT_LEADS_HOURS
    assert cfg.out == "config/wb/sota/wb2_headline_rmse.csv"


def test_argparse_overrides():
    cfg = mod.build_fetch_config_from_args(
        ["--source", "gs://my/results/", "--resolution-dir", "64x32",
         "--models", "IFS-HRES,GraphCast", "--leads", "24,120",
         "--out", "out/wb2.csv"])
    assert cfg.source == "gs://my/results/"
    assert cfg.resolution_dir == "64x32"
    assert cfg.models == ("IFS-HRES", "GraphCast")
    assert cfg.leads_hours == (24, 120)
    assert cfg.out == "out/wb2.csv"


def test_local_source_override_accepted():
    cfg = mod.build_fetch_config_from_args(["--source", "/data/wb2_results"])
    assert cfg.source == "/data/wb2_results"


def test_unknown_model_hard_errors():
    """A typo'd/unknown reference model is a hard SystemExit, not a silent skip."""
    with pytest.raises(SystemExit, match="unknown reference model"):
        mod.build_fetch_config_from_args(["--models", "GraphCast,BogusNet"])


def test_bad_leads_rejected():
    with pytest.raises(SystemExit):
        mod.build_fetch_config_from_args(["--leads", "0,24"])
    with pytest.raises(SystemExit):
        mod.build_fetch_config_from_args(["--leads", ""])


# ------------------------------------------------------------- CSV writer -----
def _records():
    return [
        {"model": "IFS-HRES", "variable": "z500", "level": 500,
         "lead_hours": 24, "rmse": 152.3},
        {"model": "IFS-HRES", "variable": "t850", "level": 850,
         "lead_hours": 72, "rmse": 1.35},
        {"model": "GraphCast", "variable": "t2m", "level": 0,
         "lead_hours": 24, "rmse": 0.71},
        {"model": "GraphCast", "variable": "wind_speed_10m", "level": 0,
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
    assert loaded["IFS-HRES"][("z500", 500, 24)] == pytest.approx(152.3)
    assert loaded["IFS-HRES"][("t850", 850, 72)] == pytest.approx(1.35)
    assert loaded["GraphCast"][("t2m", 0, 24)] == pytest.approx(0.71)
    assert loaded["GraphCast"][("wind_speed_10m", 0, 120)] == pytest.approx(2.4)


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
# evaluations.wb_forecast imports jax at module top, so it cannot be imported on
# a login node. Its HEADLINE_FIELD_KEYS is a stable public contract; mirror the
# relevant subset here (this literal is the thing under test, and a drift check
# against the source lives in the full-suite run, not this login-safe test).
_EXPECTED_HEADLINE_KEYS = frozenset({
    "z500", "t850", "q700",
    "u850", "v850", "u700", "v700", "u500", "v500", "u250", "v250",
    "mslp", "t2m", "u10", "v10", "wind_speed_10m",
})


def test_headline_var_keys_are_real_scorecard_keys():
    """Every CSV variable maps to a legoESM WB2 headline scorecard key.

    Uses a mirrored literal of ``wb_forecast.HEADLINE_FIELD_KEYS`` to stay
    JAX-free (login-safe); the source module is not imported here.
    """
    for hv in mod.HEADLINE_VARS:
        assert hv.csv_variable in _EXPECTED_HEADLINE_KEYS, hv.csv_variable


def test_model_prefixes_cover_default_models():
    for m in mod.DEFAULT_MODELS:
        assert m in mod.MODEL_PREFIXES
