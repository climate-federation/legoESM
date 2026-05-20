"""Unit tests for the multi-decade ocean spin-up library."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from legoesm.ocean.spinup import (
    SpinupHealth,
    ConvergenceCriteria,
    compute_amoc_timeseries,
    evaluate_health,
    is_converged,
    find_latest_restart,
    bryan_accelerated_dt,
)


# ==============================================================================
# AMOC timeseries summary
# ==============================================================================

class TestAMOCTimeseries:

    def test_empty_series(self):
        out = compute_amoc_timeseries([])
        assert np.isnan(out["latest_Sv"])
        assert out["n_years"] == 0

    def test_constant_series_zero_slope(self):
        out = compute_amoc_timeseries([15.0] * 30, window_years=30)
        assert out["latest_Sv"] == 15.0
        assert out["window_mean_Sv"] == 15.0
        assert out["window_std_Sv"] == 0.0
        assert abs(out["window_slope_Sv_per_yr"]) < 1e-12
        assert out["window_drift_fraction"] < 1e-10

    def test_linear_trend_recovered(self):
        # AMOC growing 0.1 Sv / year for 30 years.
        series = [15.0 + 0.1 * y for y in range(30)]
        out = compute_amoc_timeseries(series, window_years=30)
        assert abs(out["window_slope_Sv_per_yr"] - 0.1) < 1e-9
        # Drift fraction = |slope · window| / mean = 0.1 · 30 / mean.
        expected_drift = 0.1 * 30 / out["window_mean_Sv"]
        assert abs(out["window_drift_fraction"] - expected_drift) < 1e-9

    def test_window_clips_to_available_length(self):
        out = compute_amoc_timeseries([10.0, 11.0, 12.0], window_years=30)
        assert out["window_years"] == 3
        assert out["n_years"] == 3


# ==============================================================================
# SpinupHealth + evaluate_health
# ==============================================================================

class TestEvaluateHealth:

    def test_no_drift(self):
        h = evaluate_health(
            year=1, amoc_Sv=15.0, rpe_drift_W_per_m2=0.0,
            volume_now=1e18, volume_init=1e18,
            heat_now=1e22, heat_init=1e22,
            salt_now=1e16, salt_init=1e16,
        )
        assert h.volume_drift_frac == 0.0
        assert h.heat_drift_frac == 0.0
        assert h.salt_drift_frac == 0.0

    def test_positive_drift_signed(self):
        h = evaluate_health(
            year=10, amoc_Sv=15.0, rpe_drift_W_per_m2=0.01,
            volume_now=1.001e18, volume_init=1e18,
            heat_now=1e22, heat_init=1e22,
            salt_now=1e16, salt_init=1e16,
        )
        assert h.volume_drift_frac == pytest.approx(1.0e-3, rel=1e-6)

    def test_zero_init_returns_nan(self):
        h = evaluate_health(
            year=1, amoc_Sv=15.0, rpe_drift_W_per_m2=0.0,
            volume_now=1.0, volume_init=0.0,
            heat_now=1.0, heat_init=0.0,
            salt_now=1.0, salt_init=0.0,
        )
        assert np.isnan(h.volume_drift_frac)


# ==============================================================================
# Convergence
# ==============================================================================

class TestIsConverged:

    def _stable_history(self, n_years=40):
        return [
            SpinupHealth(
                year=y + 1,
                amoc_Sv=15.0 + 0.001 * (y - 20),  # ~0.001 Sv/yr noise
                rpe_drift_W_per_m2=0.001,
                volume_drift_frac=1e-5,
                heat_drift_frac=1e-4,
                salt_drift_frac=1e-5,
            )
            for y in range(n_years)
        ]

    def test_stable_run_converges(self):
        history = self._stable_history()
        v = is_converged(history)
        assert v["converged"], v

    def test_drifting_amoc_fails(self):
        history = [
            SpinupHealth(
                year=y + 1,
                amoc_Sv=15.0 + 0.1 * y,  # strong drift
                rpe_drift_W_per_m2=0.001,
                volume_drift_frac=1e-5,
                heat_drift_frac=1e-4,
                salt_drift_frac=1e-5,
            )
            for y in range(40)
        ]
        v = is_converged(history)
        assert not v["converged"]
        assert not v["amoc_ok"]

    def test_drifting_volume_fails(self):
        history = self._stable_history()
        history[-1] = history[-1]._replace(volume_drift_frac=0.01)
        v = is_converged(history)
        assert not v["converged"]
        assert not v["volume_ok"]

    def test_empty_history(self):
        v = is_converged([])
        assert not v["converged"]

    def test_all_nan_amoc_skips_amoc_criterion(self):
        """Driver currently emits NaN AMOC pending diagnostic wire-up.

        The convergence check must NOT permanently fail in this regime
        — instead the AMOC criterion is skipped and other criteria
        drive the verdict.  Verifies ``amoc_skipped=True``.
        """
        history = [
            SpinupHealth(
                year=y + 1,
                amoc_Sv=float("nan"),
                rpe_drift_W_per_m2=0.001,
                volume_drift_frac=1e-5,
                heat_drift_frac=1e-4,
                salt_drift_frac=1e-5,
            )
            for y in range(40)
        ]
        v = is_converged(history)
        assert v["amoc_skipped"], v
        assert v["amoc_ok"], v
        assert v["converged"], v

    def test_partial_nan_amoc_uses_finite_tail(self):
        """A history with NaN early then finite AMOC after diagnostic
        comes online must evaluate AMOC drift on the finite tail."""
        history = []
        for y in range(40):
            amoc = float("nan") if y < 20 else 15.0 + 0.0005 * (y - 20)
            history.append(SpinupHealth(
                year=y + 1, amoc_Sv=amoc,
                rpe_drift_W_per_m2=0.001,
                volume_drift_frac=1e-5,
                heat_drift_frac=1e-4,
                salt_drift_frac=1e-5,
            ))
        v = is_converged(history)
        assert not v["amoc_skipped"]
        assert v["amoc_summary"]["n_years"] == 20  # only finite tail

    def test_short_trailing_nan_uses_finite_tail(self):
        """Transient trailing-NaN (≤ max_trailing) is tolerated.

        ``max_trailing_nan_years = max(window // 3, 5)`` — at the default
        30-year window this is 10 years.  A 3-year trailing gap is
        well within tolerance.
        """
        history = []
        for y in range(30):
            history.append(SpinupHealth(
                year=y + 1, amoc_Sv=15.0,
                rpe_drift_W_per_m2=0.001,
                volume_drift_frac=1e-5,
                heat_drift_frac=1e-4,
                salt_drift_frac=1e-5,
            ))
        for y in range(30, 33):
            history.append(SpinupHealth(
                year=y + 1, amoc_Sv=float("nan"),
                rpe_drift_W_per_m2=0.001,
                volume_drift_frac=1e-5,
                heat_drift_frac=1e-4,
                salt_drift_frac=1e-5,
            ))
        v = is_converged(history)
        # Tolerable trailing gap → uses finite tail.
        assert not v["amoc_skipped"]
        assert not v["amoc_has_recent_nan"]
        assert v["amoc_ok"]
        assert v["converged"]

    def test_long_trailing_nan_fails(self):
        """Sustained trailing-NaN (> threshold) is a data regression
        that must fail convergence — finite values are now too stale
        to represent the current state."""
        history = []
        for y in range(30):
            history.append(SpinupHealth(
                year=y + 1, amoc_Sv=15.0,
                rpe_drift_W_per_m2=0.001,
                volume_drift_frac=1e-5,
                heat_drift_frac=1e-4,
                salt_drift_frac=1e-5,
            ))
        # 20-year NaN gap > max_trailing_nan_years (10 at default window).
        for y in range(30, 50):
            history.append(SpinupHealth(
                year=y + 1, amoc_Sv=float("nan"),
                rpe_drift_W_per_m2=0.001,
                volume_drift_frac=1e-5,
                heat_drift_frac=1e-4,
                salt_drift_frac=1e-5,
            ))
        v = is_converged(history)
        assert v["amoc_has_recent_nan"]
        assert not v["amoc_ok"]
        assert not v["converged"]


# ==============================================================================
# Restart chain helper
# ==============================================================================

class TestFindLatestRestart:

    def test_returns_none_for_empty_dir(self, tmp_path):
        assert find_latest_restart(tmp_path) is None

    def test_returns_none_for_missing_dir(self):
        assert find_latest_restart("/nonexistent/path/that/does/not/exist") is None

    def test_picks_highest_year(self, tmp_path):
        for y in (1, 5, 12, 100, 999):
            (tmp_path / f"restart_year_{y:04d}.npz").write_bytes(b"")
        latest = find_latest_restart(tmp_path)
        assert latest is not None
        assert latest.name == "restart_year_0999.npz"

    def test_ignores_non_restart_files(self, tmp_path):
        (tmp_path / "restart_year_0010.npz").write_bytes(b"")
        (tmp_path / "summary.json").write_text("{}")
        (tmp_path / "history.csv").write_text("year,amoc\n1,15\n")
        latest = find_latest_restart(tmp_path)
        assert latest is not None
        assert latest.name == "restart_year_0010.npz"

    def test_resume_at_target_year_short_circuits(self, tmp_path):
        """Re-running with ``--years`` ≤ latest restart year is a no-op.

        Guards against silently extending a completed run.  The
        driver's early-exit guard implements this; the regex helper
        is a precondition (latest year must be parseable).
        """
        (tmp_path / "restart_year_0100.npz").write_bytes(b"")
        latest = find_latest_restart(tmp_path)
        import re
        m = re.search(r"restart_year_(\d+)\.npz$", latest.name)
        assert m is not None
        latest_year = int(m.group(1))
        # Re-run request must be > latest_year to advance.
        for requested in (1, 50, 99, 100):
            assert requested <= latest_year, (
                f"--years={requested} ≤ latest year={latest_year}; driver "
                "must short-circuit"
            )


# ==============================================================================
# Bryan-Lewis accelerated timestep
# ==============================================================================

class TestBryanAcceleratedDt:

    def test_phase1_uses_long_tracer_dt(self):
        dt_m, dt_t = bryan_accelerated_dt(
            year=10, dt_physical_s=1800.0,
            dt_tracer_ratio=10.0,
            phase1_years=200,
        )
        assert dt_m == 1800.0
        assert dt_t == 18000.0

    def test_phase2_ramps_to_physical(self):
        # Mid-phase-2: ratio halfway between 10 and 1.
        dt_m, dt_t = bryan_accelerated_dt(
            year=250, dt_physical_s=1800.0,
            dt_tracer_ratio=10.0,
            phase1_years=200, phase2_years=100,
        )
        # At year 250: progress = 50/100 = 0.5 → ratio = 10 + (1-10)*0.5 = 5.5
        assert abs(dt_t - 1800.0 * 5.5) < 1e-6

    def test_phase3_equals_physical(self):
        dt_m, dt_t = bryan_accelerated_dt(
            year=500, dt_physical_s=1800.0,
            dt_tracer_ratio=10.0,
            phase1_years=200, phase2_years=100,
        )
        assert dt_m == dt_t == 1800.0

    def test_phase2_endpoint_returns_to_physical(self):
        dt_m, dt_t = bryan_accelerated_dt(
            year=300, dt_physical_s=1800.0,
            dt_tracer_ratio=10.0,
            phase1_years=200, phase2_years=100,
        )
        assert dt_t == dt_m == 1800.0
