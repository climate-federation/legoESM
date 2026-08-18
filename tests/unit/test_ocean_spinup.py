"""Unit tests for the multi-decade ocean spin-up library."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from legoesm import constants
from legoesm.ocean.spinup import (
    SpinupHealth,
    ConvergenceCriteria,
    compute_amoc_timeseries,
    evaluate_health,
    is_converged,
    find_latest_restart,
    bryan_accelerated_dt,
    compute_amoc_from_state,
    compute_acc_from_state,
    compute_mht_from_state,
    atlantic_basin_mask,
    _grid_lat_v_deg,
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


# ==============================================================================
# AMOC computation from state
# ==============================================================================

class _FakeGrid:
    """Minimal LatLonGrid-like stub for moc_streamfunction + AMOC tests."""

    def __init__(self, n_lat=36, n_lon=72, radius=constants.R_earth):
        self.n_lat = n_lat
        self.n_lon = n_lon
        self.radius = radius
        # Uniform lat in [-87.5, 87.5], lon in [0, 360-d).
        dlat = np.pi / n_lat
        dlon = 2.0 * np.pi / n_lon
        self.dlat = float(dlat)
        self.dlon = float(dlon)
        self.lat = np.linspace(-np.pi / 2 + 0.5 * dlat,
                                np.pi / 2 - 0.5 * dlat, n_lat)
        self.lon = np.linspace(0.0, 2.0 * np.pi - dlon, n_lon)


class TestGridLatVHelpers:

    def test_lat_v_from_grid_lat_dlat(self):
        g = _FakeGrid(n_lat=10, n_lon=20)
        lat_v_deg = _grid_lat_v_deg(g, n_lat_v=g.n_lat + 1)
        assert lat_v_deg.shape == (11,)
        # First v-face = lat[0] - dlat/2 = south boundary near -90°.
        assert lat_v_deg[0] == pytest.approx(-90.0, abs=1e-6)
        assert lat_v_deg[-1] == pytest.approx(90.0, abs=1e-6)


class TestAtlanticBasinMask:

    def test_default_band_picks_atlantic_longitudes(self):
        g = _FakeGrid(n_lat=18, n_lon=36)  # 10° resolution
        mask = atlantic_basin_mask(g)
        lon_deg = np.degrees(np.asarray(g.lon))
        lon_wrapped = ((lon_deg + 180.0) % 360.0) - 180.0
        # All True cells must sit inside the band.
        for i in range(len(mask)):
            if bool(mask[i]):
                assert -75.0 <= lon_wrapped[i] <= 15.0
        # 0 °E (prime meridian) must be inside.
        idx0 = int(np.argmin(np.abs(lon_wrapped)))
        assert bool(mask[idx0])
        # 180 °E must be outside.
        idx180 = int(np.argmin(np.abs(lon_wrapped - 180.0)))
        # Could be 180 or -180 after wrap; test the diametric point.
        for i in (idx180, len(mask) - idx180):
            if 0 <= i < len(mask):
                lwi = lon_wrapped[i]
                if abs(abs(lwi) - 180.0) < 1e-6:
                    assert not bool(mask[i])

    def test_wraparound_band(self):
        g = _FakeGrid(n_lat=10, n_lon=36)
        mask = atlantic_basin_mask(g, lon_min_deg=170.0, lon_max_deg=-170.0)
        lon_deg = np.degrees(np.asarray(g.lon))
        lon_wrapped = ((lon_deg + 180.0) % 360.0) - 180.0
        for i in range(len(mask)):
            inside_pred = (lon_wrapped[i] >= 170.0) or (lon_wrapped[i] <= -170.0)
            assert bool(mask[i]) == inside_pred


class TestComputeAMOCFromState:

    def _make_synthetic(self, *, sign=+1.0, amplitude=0.01,
                        n_lat=36, n_lon=72, nlev=10):
        """Build a synthetic v field whose zonal-mean is non-zero only at
        cell j_target (a single latitude row) so the AMOC max sits there.

        Returns ``(v_face, h_partial, mask, grid, j_target)``.
        """
        grid = _FakeGrid(n_lat=n_lat, n_lon=n_lon)
        v = np.zeros((n_lat + 1, n_lon, nlev), dtype=np.float64)
        # Set a uniform v at one v-face row.
        lat_v_deg = _grid_lat_v_deg(grid, n_lat + 1)
        j_target = int(np.argmin(np.abs(lat_v_deg - 26.5)))
        v[j_target, :, :nlev // 2] = sign * amplitude  # upper-half only
        # Uniform 50-m layers.
        h = np.full((n_lat, n_lon, nlev), 50.0, dtype=np.float64)
        mask = np.ones((n_lat, n_lon), dtype=np.float64)
        return v, h, mask, grid, j_target

    def test_returns_finite_value_for_uniform_v(self):
        v, h, mask, grid, _ = self._make_synthetic()
        amoc = compute_amoc_from_state(
            v, h, mask, grid, target_lat_deg=26.5, basin="global",
        )
        assert np.isfinite(amoc)
        assert amoc != 0.0

    def test_sign_convention_positive_v_gives_positive_amoc(self):
        """Northward upper transport (v > 0 at v-face) corresponds to
        the classical Atlantic-style overturning cell.  With the
        ``moc_streamfunction`` convention ``psi = -cumsum(V_zonal)``
        this drives ψ negative at the interface depth; the AMOC
        helper returns ``-min(profile)``, which must come back
        POSITIVE.
        """
        v_pos, h, mask, grid, _ = self._make_synthetic(sign=+1.0)
        a_pos = compute_amoc_from_state(
            v_pos, h, mask, grid, target_lat_deg=26.5, basin="global",
        )
        assert a_pos > 0.0, (
            f"Positive v should give positive (RAPID-style) AMOC; "
            f"got {a_pos}"
        )

    def test_reversed_cell_reports_same_magnitude(self):
        """SIGN-AGNOSTIC semantics (2026-08-11): psi's sign encodes grid
        orientation, not physical direction -- this suite's own old
        docstring said a reversed cell makes psi POSITIVE here, while on
        the production tripole state the NORMAL cell's psi is positive
        (which made the old -min() report -0.13 Sv against a
        cross-validated 9.8 Sv).  The diagnostic now returns the
        surface-referenced peak MAGNITUDE (amoc_core semantics), so a
        reversed cell reports the same positive strength and direction is
        not inferable from this scalar."""
        v_pos, h, mask, grid, _ = self._make_synthetic(sign=+1.0)
        v_neg, _, _, _, _ = self._make_synthetic(sign=-1.0)
        a_pos = compute_amoc_from_state(
            v_pos, h, mask, grid, target_lat_deg=26.5, basin="global",
        )
        a_neg = compute_amoc_from_state(
            v_neg, h, mask, grid, target_lat_deg=26.5, basin="global",
        )
        assert a_pos > 0.0
        assert a_neg == a_pos, (
            f"magnitude must be orientation-invariant: {a_pos} vs {a_neg}"
        )

    def test_target_lat_out_of_range_returns_nan(self):
        v, h, mask, grid, _ = self._make_synthetic()
        amoc = compute_amoc_from_state(
            v, h, mask, grid, target_lat_deg=200.0, basin="global",
            lat_tol_deg=5.0,
        )
        assert np.isnan(amoc)

    def test_atlantic_basin_excludes_pacific_signal(self):
        """Place a strong signal at Pacific longitudes only;
        Atlantic basin mask should suppress it."""
        grid = _FakeGrid(n_lat=18, n_lon=36)  # 10° lon
        n_lat, n_lon, nlev = 18, 36, 6
        v = np.zeros((n_lat + 1, n_lon, nlev), dtype=np.float64)
        lat_v_deg = _grid_lat_v_deg(grid, n_lat + 1)
        j = int(np.argmin(np.abs(lat_v_deg - 26.5)))
        lon_deg = np.degrees(np.asarray(grid.lon))
        lon_wrapped = ((lon_deg + 180.0) % 360.0) - 180.0
        # Signal at 150 °E (Pacific).
        pac_i = np.argmin(np.abs(lon_wrapped - 150.0))
        v[j, pac_i, :3] = 0.1
        h = np.full((n_lat, n_lon, nlev), 100.0, dtype=np.float64)
        mask = np.ones((n_lat, n_lon), dtype=np.float64)

        amoc_global = compute_amoc_from_state(
            v, h, mask, grid, target_lat_deg=26.5, basin="global",
        )
        amoc_atlantic = compute_amoc_from_state(
            v, h, mask, grid, target_lat_deg=26.5, basin="atlantic",
        )
        # Atlantic-only AMOC sees no signal — close to zero.
        # Global sees the Pacific peak.
        assert abs(amoc_atlantic) < 1.0e-3, (
            f"Atlantic basin should suppress Pacific signal, got "
            f"{amoc_atlantic} Sv"
        )
        assert abs(amoc_global) > abs(amoc_atlantic)

    def test_unknown_basin_raises(self):
        v, h, mask, grid, _ = self._make_synthetic()
        with pytest.raises(ValueError):
            compute_amoc_from_state(
                v, h, mask, grid, basin="indian",
            )


# ==============================================================================
# Drake-Passage ACC from a lat-lon C-grid state
# ==============================================================================

class _FakeGrid2DLat(_FakeGrid):
    """_FakeGrid but with a 2-D (n_lat, n_lon) ``lat`` (constant per row) to
    exercise compute_acc_from_state's curvilinear-lat row-reduction path."""

    def __init__(self, n_lat=36, n_lon=72, radius=constants.R_earth):
        super().__init__(n_lat=n_lat, n_lon=n_lon, radius=radius)
        self.lat = np.broadcast_to(
            np.asarray(self.lat)[:, None], (n_lat, n_lon)).copy()


class TestComputeACCFromState:
    """compute_acc_from_state = barotropic_streamfunction + acc_transport glue
    (Sv<->m3/s round-trip + 2-D-lat reduction).  The streamfunction/transport
    cores are already unit-tested elsewhere; here we pin the glue."""

    def _make(self, *, u0=0.1, n_lat=60, n_lon=72, nlev=8, grid_cls=_FakeGrid):
        grid = grid_cls(n_lat=n_lat, n_lon=n_lon)
        # uniform eastward zonal flow at u-faces (n_lat, n_lon+1, nlev)
        u = np.full((n_lat, n_lon + 1, nlev), u0, dtype=np.float64)
        h = np.full((n_lat, n_lon, nlev), 100.0, dtype=np.float64)
        mask = np.ones((n_lat, n_lon), dtype=np.float64)
        return u, h, mask, grid

    def test_uniform_eastward_gives_finite_nonzero(self):
        u, h, mask, grid = self._make(u0=0.1)
        acc = compute_acc_from_state(u, h, mask, grid)
        assert np.isfinite(acc)
        assert abs(acc) > 0.0

    def test_zero_flow_zero_acc(self):
        u, h, mask, grid = self._make(u0=0.0)
        acc = compute_acc_from_state(u, h, mask, grid)
        assert abs(acc) < 1e-12

    def test_scales_linearly_with_velocity(self):
        u1, h, mask, grid = self._make(u0=0.05)
        u2, _, _, _ = self._make(u0=0.10)
        a1 = compute_acc_from_state(u1, h, mask, grid)
        a2 = compute_acc_from_state(u2, h, mask, grid)
        assert abs(a2 - 2.0 * a1) < 1e-9 * max(1.0, abs(a2))

    def test_2d_lat_matches_1d(self):
        """A 2-D (row-constant) lat must reduce to the same ACC as 1-D lat."""
        u, h, mask, g1 = self._make(u0=0.1, grid_cls=_FakeGrid)
        _, _, _, g2 = self._make(u0=0.1, grid_cls=_FakeGrid2DLat)
        a1 = compute_acc_from_state(u, h, mask, g1)
        a2 = compute_acc_from_state(u, h, mask, g2)
        assert abs(a1 - a2) < 1e-9 * max(1.0, abs(a1))

    def test_drake_column_excludes_remote_flow(self):
        """The single-meridian section method (vs whole-band max-min) must NOT
        count zonal transport at a longitude far from the Drake meridian: flow
        only near 90 E -> ~0 ACC at the Drake (-68) column."""
        n_lat, n_lon, nlev = 60, 72, 4
        grid = _FakeGrid(n_lat=n_lat, n_lon=n_lon)
        u = np.zeros((n_lat, n_lon + 1, nlev), dtype=np.float64)
        lon_deg = np.degrees(np.asarray(grid.lon))
        i_far = int(np.argmin(np.abs(lon_deg - 90.0)))      # remote from -68=292
        u[:, i_far, :] = 0.5                                 # strong remote flow
        h = np.full((n_lat, n_lon, nlev), 100.0, dtype=np.float64)
        mask = np.ones((n_lat, n_lon), dtype=np.float64)
        acc = compute_acc_from_state(u, h, mask, grid, drake_lon_deg=-68.0)
        assert abs(acc) < 1e-9       # nothing crosses the Drake meridian

    def test_dlon_sentinel_estimates_spacing(self):
        """A dlon=0 sentinel grid (tripole-style) still selects the Drake column
        via the estimated local spacing -> remote-flow exclusion still holds and
        the half-cell U-face shift is applied."""
        n_lat, n_lon, nlev = 60, 72, 4
        grid = _FakeGrid(n_lat=n_lat, n_lon=n_lon)
        grid.dlon = 0.0                                      # tripole sentinel
        u = np.zeros((n_lat, n_lon + 1, nlev), dtype=np.float64)
        i_far = int(np.argmin(np.abs(np.degrees(np.asarray(grid.lon)) - 90.0)))
        u[:, i_far, :] = 0.5
        h = np.full((n_lat, n_lon, nlev), 100.0, dtype=np.float64)
        mask = np.ones((n_lat, n_lon), dtype=np.float64)
        acc = compute_acc_from_state(u, h, mask, grid, drake_lon_deg=-68.0)
        assert abs(acc) < 1e-9


class TestComputeMHTFromState:
    """compute_mht_from_state (latlon/tripole) = meridional_heat_transport curve
    reduced to NH-peak / SH-min.  Pin sign + zero (the geometry core is tested
    via the diagnostics module)."""

    def _make(self, *, v0=0.1, theta=10.0, n_lat=40, n_lon=72, nlev=6):
        grid = _FakeGrid(n_lat=n_lat, n_lon=n_lon)
        v = np.full((n_lat + 1, n_lon, nlev), v0, dtype=np.float64)  # v-faces
        th = np.full((n_lat, n_lon, nlev), theta, dtype=np.float64)
        h = np.full((n_lat, n_lon, nlev), 100.0, dtype=np.float64)
        mask = np.ones((n_lat, n_lon), dtype=np.float64)
        return v, th, h, mask, grid

    def test_northward_warm_positive(self):
        v, th, h, mask, grid = self._make(v0=0.1, theta=10.0)
        r = compute_mht_from_state(v, th, h, mask, grid)
        assert r["nh_peak_PW"] > 0.0
        assert r["nh_peak_lat"] > 0.0

    def test_sign_flips_with_velocity(self):
        north = compute_mht_from_state(*self._make(v0=0.1, theta=10.0))
        south = compute_mht_from_state(*self._make(v0=-0.1, theta=10.0))
        # Northward warm -> poleward (NH +); southward warm -> SH transport
        # negative.  (nh_peak is the NH maximum, which a uniform field pads with 0
        # at the pole rows, so the robust sign discriminator is the SH min.)
        assert north["nh_peak_PW"] > 0.0
        assert south["sh_min_PW"] < 0.0
        assert north["nh_peak_PW"] > south["nh_peak_PW"]

    def test_zero_velocity_zero(self):
        r = compute_mht_from_state(*self._make(v0=0.0, theta=10.0))
        assert abs(r["nh_peak_PW"]) < 1e-12 and abs(r["sh_min_PW"]) < 1e-12
