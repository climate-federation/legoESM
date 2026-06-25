"""Tests for WeatherBench2 integration: data loading, metrics, and evaluation."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
import jax.numpy as jnp

# ---------------------------------------------------------------------------
# ERA5 loader tests (no network)
# ---------------------------------------------------------------------------


class TestERA5Config:
    """Test ERA5Config defaults and creation."""

    def test_defaults(self):
        from legoesm.ml.data.era5_loader import ERA5Config, WB2_ERA5_ZARR

        cfg = ERA5Config()
        assert cfg.zarr_store == WB2_ERA5_ZARR
        assert "geopotential" in cfg.variables
        assert cfg.dt_hours == 6

    def test_climatology_config_defaults(self):
        from legoesm.ml.data.era5_loader import ERA5ClimatologyConfig, WB2_CLIMATOLOGY_ZARR

        cfg = ERA5ClimatologyConfig()
        assert cfg.zarr_store == WB2_CLIMATOLOGY_ZARR

    def test_override(self):
        from legoesm.ml.data.era5_loader import ERA5Config

        cfg = ERA5Config(zarr_store="/tmp/test.zarr", dt_hours=12)
        assert cfg.zarr_store == "/tmp/test.zarr"
        assert cfg.dt_hours == 12


class TestDatasetToArray:
    """Test _dataset_to_array with synthetic xarray datasets."""

    def _make_mock_ds(self, n_lat=8, n_lon=16, n_levels=3):
        """Create a synthetic xarray Dataset mimicking WB2 format."""
        import xarray as xr

        ds = xr.Dataset(
            {
                "geopotential": xr.DataArray(
                    np.random.randn(n_levels, n_lat, n_lon).astype(np.float32),
                    dims=["level", "lat", "lon"],
                ),
                "temperature": xr.DataArray(
                    np.random.randn(n_levels, n_lat, n_lon).astype(np.float32),
                    dims=["level", "lat", "lon"],
                ),
                "u_component_of_wind": xr.DataArray(
                    np.random.randn(n_levels, n_lat, n_lon).astype(np.float32),
                    dims=["level", "lat", "lon"],
                ),
            },
        )
        return ds

    def test_3d_packing_shape(self):
        from legoesm.ml.data.era5_loader import _dataset_to_array, ERA5Config

        ds = self._make_mock_ds(n_lat=8, n_lon=16, n_levels=3)
        cfg = ERA5Config(
            variables=("geopotential", "temperature", "u_component_of_wind"),
            levels=(1000, 500, 250),
        )
        arr = _dataset_to_array(ds, cfg)

        # 3 vars * 3 levels = 9 channels
        assert arr.shape == (8, 16, 9)
        assert arr.dtype == jnp.float32

    def test_2d_surface_variable(self):
        import xarray as xr
        from legoesm.ml.data.era5_loader import _dataset_to_array, ERA5Config

        ds = xr.Dataset(
            {
                "mean_sea_level_pressure": xr.DataArray(
                    np.random.randn(8, 16).astype(np.float32),
                    dims=["lat", "lon"],
                ),
            },
        )
        cfg = ERA5Config(variables=("mean_sea_level_pressure",), levels=())
        arr = _dataset_to_array(ds, cfg)
        assert arr.shape == (8, 16, 1)

    def test_missing_variable_raises(self):
        import pytest
        from legoesm.ml.data.era5_loader import _dataset_to_array, ERA5Config

        ds = self._make_mock_ds()
        cfg = ERA5Config(
            variables=("geopotential", "nonexistent_variable"),
            levels=(1000, 500, 250),
        )
        # An unresolvable variable now fails loud (was silently skipped, which
        # yielded a short-channel array surfacing as a shape error far from the
        # cause — PR C era5 fix).
        with pytest.raises(ValueError, match="(?i)not found"):
            _dataset_to_array(ds, cfg)


class TestVariableResolution:
    """Test WB2 variable name alias resolution."""

    def test_short_to_long_alias(self):
        import xarray as xr
        from legoesm.ml.data.era5_loader import _resolve_variables

        ds = xr.Dataset(
            {"geopotential": xr.DataArray(np.zeros(1))}
        )
        resolved = _resolve_variables(ds, ("z",))
        assert resolved == ["geopotential"]

    def test_direct_match(self):
        import xarray as xr
        from legoesm.ml.data.era5_loader import _resolve_variables

        ds = xr.Dataset(
            {"temperature": xr.DataArray(np.zeros(1))}
        )
        resolved = _resolve_variables(ds, ("temperature",))
        assert resolved == ["temperature"]


# ---------------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------------


class TestRMSE:
    """Test RMSE metric with known arrays."""

    def test_zero_error(self):
        from evaluations.metrics import rmse

        pred = jnp.ones((4, 8))
        weights = jnp.ones(4) / 4.0
        result = rmse(pred, pred, weights)
        assert float(result) == pytest.approx(0.0, abs=1e-7)

    def test_known_value(self):
        from evaluations.metrics import rmse

        pred = jnp.zeros((4, 8))
        target = jnp.ones((4, 8))
        weights = jnp.ones(4) / 4.0
        result = rmse(pred, target, weights)
        assert float(result) == pytest.approx(1.0, abs=1e-6)

    def test_weights_matter(self):
        from evaluations.metrics import rmse

        pred = jnp.zeros((4, 8))
        target = jnp.ones((4, 8))
        # Weight only the first latitude
        weights = jnp.array([1.0, 0.0, 0.0, 0.0])
        result = rmse(pred, target, weights)
        assert float(result) == pytest.approx(1.0, abs=1e-6)


class TestACC:
    """Test Anomaly Correlation Coefficient."""

    def test_perfect_forecast(self):
        from evaluations.metrics import acc

        target = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        clim = jnp.zeros_like(target)
        weights = jnp.ones(2) / 2.0
        result = acc(target, target, clim, weights)
        assert float(result) == pytest.approx(1.0, abs=1e-6)

    def test_opposite_anomalies(self):
        from evaluations.metrics import acc

        pred = jnp.array([[1.0, 1.0], [1.0, 1.0]])
        target = jnp.array([[-1.0, -1.0], [-1.0, -1.0]])
        clim = jnp.zeros((2, 2))
        weights = jnp.ones(2) / 2.0
        result = acc(pred, target, clim, weights)
        assert float(result) == pytest.approx(-1.0, abs=1e-6)


class TestBias:
    """Test area-weighted bias."""

    def test_zero_bias(self):
        from evaluations.metrics import bias

        x = jnp.ones((4, 8))
        weights = jnp.ones(4) / 4.0
        result = bias(x, x, weights)
        assert float(result) == pytest.approx(0.0, abs=1e-7)

    def test_known_bias(self):
        from evaluations.metrics import bias

        pred = jnp.ones((4, 8)) * 3.0
        target = jnp.ones((4, 8)) * 1.0
        weights = jnp.ones(4) / 4.0
        result = bias(pred, target, weights)
        assert float(result) == pytest.approx(2.0, abs=1e-6)


class TestSpreadSkillRatio:
    """Test ensemble spread-skill ratio."""

    def test_single_member(self):
        from evaluations.metrics import spread_skill_ratio

        preds = jnp.ones((1, 4, 8))
        target = jnp.zeros((4, 8))
        weights = jnp.ones(4) / 4.0
        result = spread_skill_ratio(preds, target, weights)
        # Zero spread / nonzero skill → 0
        assert float(result) == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Scorecard tests
# ---------------------------------------------------------------------------


class TestScorecard:
    """Test scorecard computation."""

    def test_basic_scorecard(self):
        from evaluations.metrics import compute_scorecard

        weights = jnp.ones(4) / 4.0
        pred_ds = {
            ("temperature", 500, 6): jnp.ones((4, 8)),
            ("temperature", 500, 24): jnp.ones((4, 8)) * 1.5,
        }
        target_ds = {
            ("temperature", 500, 6): jnp.zeros((4, 8)),
            ("temperature", 500, 24): jnp.zeros((4, 8)),
        }
        clim_ds = {
            ("temperature", 500): jnp.ones((4, 8)) * 0.5,
        }
        sc = compute_scorecard(
            pred_ds, target_ds, clim_ds, weights,
            variables=["temperature"], levels=[500],
        )
        assert "temperature" in sc
        assert 500 in sc["temperature"]
        assert 6 in sc["temperature"][500]
        assert "rmse" in sc["temperature"][500][6]
        assert "acc" in sc["temperature"][500][6]


# ---------------------------------------------------------------------------
# Config loading tests
# ---------------------------------------------------------------------------


class TestConfigLoading:
    """Test YAML config loading."""

    def test_load_weatherbench2_config(self, tmp_path):
        from evaluations.weatherbench import load_eval_config

        yaml_content = """
dataset: "gs://weatherbench2/datasets/era5/test.zarr"
climatology: "gs://weatherbench2/datasets/era5-hourly-climatology/test.zarr"
eval_period: ["2020-01-01", "2020-06-30"]
lead_times_hours: [6, 24]
variables:
  - geopotential
  - temperature
levels: [500, 850]
output_dir: "results/test"
"""
        yaml_path = tmp_path / "test_config.yaml"
        yaml_path.write_text(yaml_content)

        cfg = load_eval_config(str(yaml_path))
        assert cfg.eval_period == ("2020-01-01", "2020-06-30")
        assert cfg.lead_times_hours == (6, 24)
        assert "geopotential" in cfg.variables
        assert 500 in cfg.levels
        assert cfg.era5_config.zarr_store.endswith("test.zarr")


# ---------------------------------------------------------------------------
# Loss function tests
# ---------------------------------------------------------------------------


class TestLossFunctions:
    """Test new loss functions."""

    def test_weighted_mae(self):
        from legoesm.ml.loss import weighted_mae

        pred = jnp.ones((4, 8, 2)) * 3.0
        target = jnp.ones((4, 8, 2)) * 1.0
        weights = jnp.ones(4) / 4.0
        result = weighted_mae(pred, target, weights)
        # PR C: weighted_mae now applies the n_lat/Σw resolution correction, so
        # the area-weighted MAE of a CONSTANT error 2.0 is 2.0 (the constant)
        # regardless of the weights — was an uncorrected 0.5 = mean(2·0.25).
        assert float(result) == pytest.approx(2.0, abs=1e-6)

    def test_spectral_loss_zero(self):
        from legoesm.ml.loss import spectral_loss

        x = jnp.ones(10) + 0j
        result = spectral_loss(x, x)
        assert float(result) == pytest.approx(0.0, abs=1e-7)

    def test_spectral_loss_nonzero(self):
        from legoesm.ml.loss import spectral_loss

        pred = jnp.ones(10) + 0j
        target = jnp.zeros(10) + 0j
        result = spectral_loss(pred, target)
        assert float(result) == pytest.approx(1.0, abs=1e-7)


# ---------------------------------------------------------------------------
# GCS streaming test (skipped without network)
# ---------------------------------------------------------------------------


def _has_gcsfs() -> bool:
    try:
        import gcsfs
        return True
    except ImportError:
        return False


@pytest.mark.skipif(
    not _has_gcsfs(),
    reason="gcsfs not installed or no network",
)
def test_gcs_streaming():
    """Integration test: verify GCS streaming works with WB2 ERA5 data."""
    from legoesm.ml.data.era5_loader import create_era5_dataset, ERA5Config

    cfg = ERA5Config(time_range=("2020-01-01", "2020-01-02"))
    ds = create_era5_dataset(cfg)
    assert "time" in ds.dims
    assert len(ds.time) > 0
