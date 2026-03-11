"""Unit tests for monthly-mean diagnostics and AMIP validation (Task 12).

Tests cover:
- MonthlyAccumulator: day-to-month mapping, 2D/3D/scalar accumulation,
  zonal binning, finalization, multi-month accumulation, save/load
- Validation script: target checking
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import numpy.testing as npt

from legoesm.diagnostics.monthly_means import MonthlyAccumulator


class TestDayToMonth:
    """Tests for MonthlyAccumulator.day_to_month."""

    def test_january_first(self):
        assert MonthlyAccumulator.day_to_month(1.0) == 1

    def test_january_end(self):
        assert MonthlyAccumulator.day_to_month(31.0) == 1

    def test_february_start(self):
        assert MonthlyAccumulator.day_to_month(32.0) == 2

    def test_march(self):
        # Day 60 = Feb 28 (day 59), so day 60 = March 1
        assert MonthlyAccumulator.day_to_month(60.0) == 3

    def test_december(self):
        assert MonthlyAccumulator.day_to_month(365.0) == 12

    def test_mid_year(self):
        # Day 183 should be July (days 182-212)
        assert MonthlyAccumulator.day_to_month(183.0) == 7


class TestMonthlyAccumulator2D:
    """Tests for 2D field accumulation."""

    def _make_grid(self, n=4):
        """Create a simple test grid."""
        lat = np.linspace(-80, 80, 6 * n * n).reshape(6, n, n)
        return lat

    def test_add_2d_creates_zonal(self):
        """Adding 2D fields should create zonal-mean entries."""
        accum = MonthlyAccumulator(nlev=10, n_lat_bins=18)
        lat = self._make_grid()
        fields = {'T_low': np.full((6, 4, 4), 280.0)}
        accum.add_2d(15.0, 0, fields, lat)

        result = accum.finalize()
        assert 'zonal_T_low' in result
        assert result['zonal_T_low'].shape == (1, 18)

    def test_zonal_mean_value(self):
        """Uniform field should give uniform zonal mean."""
        accum = MonthlyAccumulator(nlev=10, n_lat_bins=18)
        lat = self._make_grid()
        fields = {'T_low': np.full((6, 4, 4), 280.0)}
        accum.add_2d(15.0, 0, fields, lat)

        result = accum.finalize()
        zonal = result['zonal_T_low'][0]
        valid = ~np.isnan(zonal)
        assert np.any(valid)
        npt.assert_allclose(zonal[valid], 280.0, atol=1e-10)

    def test_scalar_accumulation(self):
        """Global mean should be stored as scalar."""
        accum = MonthlyAccumulator(nlev=10, n_lat_bins=18)
        lat = self._make_grid()
        fields = {'T_low': np.full((6, 4, 4), 280.0)}
        accum.add_2d(15.0, 0, fields, lat)

        result = accum.finalize()
        assert 'scalar_T_low' in result
        npt.assert_allclose(result['scalar_T_low'][0], 280.0, atol=1e-10)

    def test_multi_sample_averaging(self):
        """Multiple adds in same month should average correctly."""
        accum = MonthlyAccumulator(nlev=10, n_lat_bins=18)
        lat = self._make_grid()

        accum.add_2d(5.0, 0, {'T_low': np.full((6, 4, 4), 270.0)}, lat)
        accum.add_2d(15.0, 0, {'T_low': np.full((6, 4, 4), 290.0)}, lat)

        result = accum.finalize()
        zonal = result['zonal_T_low'][0]
        valid = ~np.isnan(zonal)
        npt.assert_allclose(zonal[valid], 280.0, atol=1e-10)

    def test_different_months(self):
        """Adds in different months should produce separate entries."""
        accum = MonthlyAccumulator(nlev=10, n_lat_bins=18)
        lat = self._make_grid()

        accum.add_2d(15.0, 0, {'T_low': np.full((6, 4, 4), 270.0)}, lat)  # Jan
        accum.add_2d(45.0, 0, {'T_low': np.full((6, 4, 4), 280.0)}, lat)  # Feb
        accum.add_2d(75.0, 0, {'T_low': np.full((6, 4, 4), 290.0)}, lat)  # Mar

        result = accum.finalize()
        assert result['zonal_T_low'].shape[0] == 3


class TestMonthlyAccumulator3D:
    """Tests for 3D profile accumulation."""

    def test_add_3d_creates_profiles(self):
        """Adding 3D fields should create profile entries."""
        accum = MonthlyAccumulator(nlev=5, n_lat_bins=18)
        lat = np.linspace(-80, 80, 6 * 4 * 4).reshape(6, 4, 4)
        T = np.full((6, 4, 4, 5), 250.0)
        accum.add_3d(15.0, 0, {'T': T}, lat)

        result = accum.finalize()
        assert 'profile_T' in result
        assert result['profile_T'].shape == (1, 18, 5)

    def test_profile_values(self):
        """Uniform 3D field should give uniform profiles."""
        accum = MonthlyAccumulator(nlev=5, n_lat_bins=18)
        lat = np.linspace(-80, 80, 6 * 4 * 4).reshape(6, 4, 4)
        T = np.full((6, 4, 4, 5), 260.0)
        accum.add_3d(15.0, 0, {'T': T}, lat)

        result = accum.finalize()
        profile = result['profile_T'][0]
        valid = ~np.isnan(profile)
        npt.assert_allclose(profile[valid], 260.0, atol=1e-10)


class TestMonthlyAccumulatorScalar:
    """Tests for scalar accumulation."""

    def test_add_scalar(self):
        """Scalar diagnostics should be stored per month."""
        accum = MonthlyAccumulator(nlev=10)
        accum.add_scalar(15.0, 0, {'T_atm': 260.0, 'precip': 3.0})

        result = accum.finalize()
        assert 'scalar_T_atm' in result
        assert 'scalar_precip' in result
        npt.assert_allclose(result['scalar_T_atm'][0], 260.0, atol=1e-10)

    def test_scalar_averaging(self):
        """Multiple scalar adds should average."""
        accum = MonthlyAccumulator(nlev=10)
        accum.add_scalar(5.0, 0, {'T_atm': 250.0})
        accum.add_scalar(15.0, 0, {'T_atm': 270.0})

        result = accum.finalize()
        npt.assert_allclose(result['scalar_T_atm'][0], 260.0, atol=1e-10)


class TestMonthlyAccumulatorMultiYear:
    """Tests for multi-year accumulation."""

    def test_year_month_keys(self):
        """Different years should create separate monthly bins."""
        accum = MonthlyAccumulator(nlev=10)
        accum.add_scalar(15.0, 0, {'T': 260.0})  # Year 0, Jan
        accum.add_scalar(15.0, 1, {'T': 265.0})  # Year 1, Jan

        result = accum.finalize()
        assert len(result['months']) == 2
        assert result['months'][0] == (0, 1)
        assert result['months'][1] == (1, 1)
        npt.assert_allclose(result['scalar_T'][0], 260.0, atol=1e-10)
        npt.assert_allclose(result['scalar_T'][1], 265.0, atol=1e-10)


class TestMonthlyAccumulatorSave:
    """Tests for save/load functionality."""

    def test_save_and_load(self):
        """Save and reload should preserve data."""
        accum = MonthlyAccumulator(nlev=5, n_lat_bins=18)
        lat = np.linspace(-80, 80, 96).reshape(6, 4, 4)

        accum.add_2d(15.0, 0, {'T_low': np.full((6, 4, 4), 280.0)}, lat)
        accum.add_3d(15.0, 0, {'T': np.full((6, 4, 4, 5), 260.0)}, lat)
        accum.add_scalar(15.0, 0, {'precip': 2.5})

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "monthly_test.npz"
            accum.save(path, sigma=np.linspace(0.05, 0.95, 5))

            data = dict(np.load(str(path)))
            assert 'lat' in data
            assert 'years' in data
            assert 'month_nums' in data
            assert 'sigma' in data
            assert 'zonal_T_low' in data
            assert 'profile_T' in data
            assert 'scalar_precip' in data

    def test_empty_finalize(self):
        """Finalize with no data should return empty structure."""
        accum = MonthlyAccumulator(nlev=10)
        result = accum.finalize()
        assert result['months'] == []
        assert len(result['lat']) == 90  # default


class TestValidation:
    """Tests for the validation target checking."""

    def test_check_target_pass(self):
        """Value within range should pass."""
        from scripts.validate_amip import check_target
        msg, ok = check_target("test", 288.0, {"min": 287.0, "max": 289.0, "observed": 288.0, "unit": "K"})
        assert ok
        assert "PASS" in msg

    def test_check_target_fail(self):
        """Value outside range should fail."""
        from scripts.validate_amip import check_target
        msg, ok = check_target("test", 300.0, {"min": 287.0, "max": 289.0, "observed": 288.0, "unit": "K"})
        assert not ok
        assert "FAIL" in msg

    def test_generate_report_missing_dir(self):
        """Should handle missing directory gracefully."""
        from scripts.validate_amip import generate_report
        report = generate_report(Path("/nonexistent/path"))
        assert "AMIP Validation Report" in report
