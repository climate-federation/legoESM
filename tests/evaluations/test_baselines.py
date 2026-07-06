"""Tests for baseline forecasts + SOTA loader (Stage 0, Task 8). stdlib/numpy only."""
import numpy as np
import pytest

from evaluations.baselines import (
    persistence_forecast,
    climatology_forecast,
    load_sota_headline,
)


def test_persistence_is_lead_independent():
    ic = {"z500": np.full((4, 8), 5500.0)}
    p = persistence_forecast(ic, [6, 24, 120])
    assert set(p) == {6, 24, 120}
    assert np.array_equal(p[6]["z500"], p[120]["z500"])       # persisted, unchanged


def test_climatology_is_lead_independent():
    clim = {"t850": np.full((4, 8), 280.0)}
    c = climatology_forecast(clim, [24, 240])
    assert set(c) == {24, 240}
    assert np.array_equal(c[24]["t850"], c[240]["t850"])


def test_load_sota_headline(tmp_path):
    csv_path = tmp_path / "sota.csv"
    csv_path.write_text(
        "model,variable,level,lead_hours,rmse\n"
        "ifs_hres,geopotential,500,72,150.0\n"
        "graphcast,geopotential,500,72,120.0\n"
        "graphcast,temperature,850,72,1.1\n"
    )
    sota = load_sota_headline(csv_path)
    assert sota["ifs_hres"][("geopotential", 500, 72)] == 150.0
    assert sota["graphcast"][("geopotential", 500, 72)] == 120.0
    assert sota["graphcast"][("temperature", 850, 72)] == 1.1


def test_load_sota_headline_rejects_duplicates(tmp_path):
    dup = tmp_path / "dup.csv"
    dup.write_text(
        "model,variable,level,lead_hours,rmse\n"
        "ifs_hres,geopotential,500,72,150.0\n"
        "ifs_hres,geopotential,500,72,151.0\n"   # duplicate (model,var,level,lead)
    )
    with pytest.raises(ValueError):
        load_sota_headline(dup)


def test_load_sota_headline_bad_numeric_has_row_context(tmp_path):
    bad = tmp_path / "badnum.csv"
    bad.write_text(
        "model,variable,level,lead_hours,rmse\n"
        "ifs_hres,geopotential,500,72,not_a_number\n"
    )
    with pytest.raises(ValueError, match="line 2"):
        load_sota_headline(bad)


def test_load_sota_headline_rejects_nonfinite_rmse(tmp_path):
    bad = tmp_path / "nonfinite.csv"
    bad.write_text(
        "model,variable,level,lead_hours,rmse\n"
        "ifs_hres,geopotential,500,72,nan\n"
    )
    with pytest.raises(ValueError, match="non-finite"):
        load_sota_headline(bad)


def test_load_sota_headline_missing_columns(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("model,rmse\nx,1.0\n")
    with pytest.raises(ValueError):
        load_sota_headline(bad)


def test_load_sota_headline_missing_file():
    with pytest.raises(FileNotFoundError):
        load_sota_headline("/nonexistent/does_not_exist.csv")
