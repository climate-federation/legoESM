from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="module")
def capstone():
    path = (Path(__file__).resolve().parents[3] / "scripts/validate/ocean_fidelity/"
            "dino_1226/multi_year_climate_equivalence.py")
    spec = importlib.util.spec_from_file_location("_capstone_score", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_snapshot_schema_is_derived_from_per_day_keys(capstone):
    keys = [f"{field}_day{day}" for field in capstone.FIELDS
            for day in capstone.EXPECTED_DAYS]
    got = capstone.snapshot_days_from_keys(keys)
    assert all(days == capstone.EXPECTED_DAYS for days in got.values())


def test_snapshot_schema_missing_day_fails_loudly(capstone):
    keys = [f"{field}_day{day}" for field in capstone.FIELDS
            for day in capstone.EXPECTED_DAYS]
    keys.remove("S3d_day7200")
    with pytest.raises(RuntimeError, match="per-field snapshot day sets disagree"):
        capstone.snapshot_days_from_keys(keys)


def test_weighted_quantile_uses_first_cumulative_crossing(capstone):
    values = np.array([30.0, 10.0, 20.0])
    weights = np.array([1.0, 8.0, 1.0])
    assert capstone.weighted_quantile(values, weights, 0.10) == 10.0
    assert capstone.weighted_quantile(values, weights, 0.50) == 10.0
    assert capstone.weighted_quantile(values, weights, 0.90) == 20.0


def test_monthly_summary_removes_registered_climatology(capstone):
    climatology = np.arange(12, dtype=np.float64)
    series = np.tile(climatology, 5)
    got = capstone.monthly_summary(series)
    np.testing.assert_array_equal(got["climatology"], climatology)
    assert got["seasonal_amplitude"] == 11.0
    assert got["deseasonalized_std"] == 0.0
    assert got["interannual_std"] == 0.0


def test_classifier_plants_reach_all_frozen_branches(capstone):
    receipts = capstone.controls()
    assert all(receipts.values())
