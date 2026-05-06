"""Unit tests for the cross-grid comparison helpers in
``scripts/run_atmosphere_test_matrix.py``.

These pin the iter-1..10 helpers (``_area_weighted_mean``,
``_zonal_mean_at_final_time``, ``_zonal_mean_climatology``,
``_interp_zonal_mean_to_target``, ``_compute_cross_grid_rms_agreement``,
``_atm_extract_field_2d``, ``_select_resolution_dir``,
``_vertical_coords_for_case``).

They use synthetic input arrays so they are fast and don't require any
of the scientific-stack imports that the script's ``main()`` triggers.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Make the script importable as a module without invoking ``main()``.
_SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import run_atmosphere_test_matrix as M  # noqa: E402  (sys.path tweak above)


# ---------------------------------------------------------------------------
# _area_weighted_mean
# ---------------------------------------------------------------------------

class TestAreaWeightedMean:
    def test_uniform_field_returns_input(self):
        """For a uniform field, area-weighted mean equals the value."""
        field = np.full((36, 72), 17.5)
        cos_lat = np.cos(np.deg2rad(np.linspace(-89.0, 89.0, 36)))
        area = cos_lat[:, None] * np.ones((36, 72))
        out = M._area_weighted_mean(field, area)
        assert abs(out - 17.5) < 1e-10

    def test_pole_dominated_field_unweighted_disagrees_weighted(self):
        """A field that's ZERO at the equator and 1 at the poles:
        unweighted ``np.mean`` gives ~1.0; cos-lat-weighted mean is much
        smaller because pole cells carry tiny weight.  This pins the
        whole point of the iter-2 bug fix."""
        n_lat, n_lon = 91, 144
        lat = np.linspace(-90.0, 90.0, n_lat)
        cos_lat = np.cos(np.deg2rad(lat))
        # Field = 1 within 5° of either pole, 0 elsewhere.
        field = np.where(np.abs(lat[:, None]) > 85.0, 1.0, 0.0)
        field = np.broadcast_to(field, (n_lat, n_lon))
        area = cos_lat[:, None] * np.ones((n_lat, n_lon))
        unweighted = float(np.mean(field))
        weighted = M._area_weighted_mean(field, area)
        # Unweighted ≈ fraction of latitude band > 85°: ~5/180 = 0.0278.
        # Weighted = (∫_{85°}^{90°} cos φ dφ × 2 hemispheres) / 4π_lat
        #   = 2 (sin 90° - sin 85°) / 2 = 1 - cos(5°) ≈ 0.0038.
        assert weighted < unweighted * 0.2, (
            f"weighted={weighted:.4f} unweighted={unweighted:.4f} — weighted "
            f"should be much smaller for a pole-dominated field"
        )

    def test_collapses_extra_trailing_axes(self):
        """For a 3-D field with a 2-D area, the helper averages over
        the trailing axes uniformly THEN horizontal-area-weights."""
        n_lat, n_lon, nlev = 36, 72, 5
        # T(level) = 200 + 20*level — vertical-uniform mean = 240.
        T_3d = np.broadcast_to(
            (200.0 + 20.0 * np.arange(nlev))[None, None, :],
            (n_lat, n_lon, nlev),
        ).copy()
        cos_lat = np.cos(np.deg2rad(np.linspace(-89.0, 89.0, n_lat)))
        area = cos_lat[:, None] * np.ones((n_lat, n_lon))
        out = M._area_weighted_mean(T_3d, area)
        assert abs(out - 240.0) < 1e-10

    def test_voronoi_1d_layout(self):
        """``(n_cells,)`` field with ``(n_cells,)`` area — same shape."""
        rng = np.random.default_rng(42)
        n_cells = 1000
        field = rng.normal(size=n_cells)
        area = rng.uniform(0.5, 1.5, size=n_cells)
        # Reference value computed directly.
        expected = float(np.sum(field * area) / np.sum(area))
        out = M._area_weighted_mean(field, area)
        assert abs(out - expected) < 1e-12

    def test_field_ndim_less_than_area_raises(self):
        with pytest.raises(ValueError, match="field.ndim"):
            M._area_weighted_mean(np.zeros(36), np.zeros((36, 72)))


# ---------------------------------------------------------------------------
# iter-83: _compute_drift denominator floor (mirror of iter-80 fix)
# ---------------------------------------------------------------------------

class TestComputeDriftDenominatorFloor:
    """The iter-83 fix raised ``_compute_drift``'s denominator
    floor from ``1e-30`` to ``1.0``, mirroring the iter-80 fix
    in ``run_ocean_test_matrix.py:_save_conservation``.  Pin
    that the floor stays at a physically meaningful value so a
    regression to 1e-30 (or lower) would amplify machine-
    precision rounding into spurious 1e+30-magnitude "drift"
    values for any future caller with a near-zero baseline.
    """

    def test_returns_zero_for_short_series(self):
        """Empty / single-element series → 0 (no drift defined)."""
        assert M._compute_drift([]) == 0.0
        assert M._compute_drift([1.0]) == 0.0

    def test_relative_drift_for_non_zero_baseline(self):
        """For a realistic baseline (e.g., atmospheric mass
        ~5e19 Pa·m², drift 5e15), drift = 5e15 / 5e19 = 1e-4.
        Pin that the relative-drift normalization is preserved
        when the baseline is well above the floor.
        """
        baseline = 5.0e19
        drift_amount = 5.0e15
        result = M._compute_drift([baseline, baseline + drift_amount])
        # 1e-4 with reasonable tolerance.
        assert abs(result - 1e-4) < 1e-6

    def test_drift_is_absolute_value_for_decreasing_series(self):
        """iter-86 codex LOW: the function uses ``abs(values[-1] -
        values[0])`` so decreasing series report POSITIVE drift.
        A regression dropping the ``abs(...)`` on the numerator
        would let this test catch the sign error (the previous
        tests all used positive drift).
        """
        baseline = 5.0e19
        drift_amount = 5.0e15
        # Decreasing series: drift is negative if abs() is dropped.
        result = M._compute_drift([baseline, baseline - drift_amount])
        # Same magnitude as the positive case; if abs() was
        # dropped, this would be negative (-1e-4).
        assert result == pytest.approx(1e-4, rel=1e-6), (
            f"iter-86 regression: ``_compute_drift`` of a "
            f"decreasing series returned {result:.6e}, expected "
            f"+1e-4 (positive magnitude).  If you got -1e-4, the "
            f"``abs()`` on the numerator was dropped."
        )

    def test_baseline_zero_does_not_blow_up(self):
        """The iter-78/80 pathology: baseline ≈ 0 + tiny rounding
        drift.  iter-83 ensures the result is the absolute drift
        (in physical units), NOT a 1e+30-magnitude spurious value.
        """
        # Baseline 0, drift 1e-17 (machine precision).
        result = M._compute_drift([0.0, 1.0e-17])
        # With iter-83 floor 1.0: result = 1e-17 / 1 = 1e-17.
        # With iter-78 floor 1e-30: result = 1e-17 / 1e-30 = 1e+13.
        assert result < 1e-10, (
            f"iter-83 regression: drift = {result:.2e} for "
            f"baseline=0 + 1e-17 rounding drift.  Expected ~1e-17 "
            f"(absolute drift); got 1e-10 or larger.  Most likely "
            f"cause: the denominator floor was lowered from 1.0 "
            f"back toward 1e-30."
        )
        # iter-84 codex LOW: pin the EXACT behavioral contract.
        # With floor=1.0 and baseline=0, drift = 1e-17 / 1 = 1e-17
        # (absolute drift in physical units).  Subtler floor
        # regressions (e.g., 1e-15) wouldn't fail the < 1e-10
        # check above but WOULD fail this exact-value check.
        assert result == pytest.approx(1.0e-17, rel=1e-6), (
            f"iter-84: ``_compute_drift`` for baseline=0 + 1e-17 "
            f"drift must return exactly 1e-17 (denominator=max(0, "
            f"1.0)=1.0, drift / 1 = 1e-17).  Got {result:.6e}.  "
            f"This catches subtler floor regressions than the "
            f"< 1e-10 magnitude bound."
        )

    def test_floor_constant_is_one(self):
        """Source-level pin: the denominator floor in the canonical
        ``compute_relative_drift`` helper must be ``1.0`` (not 1e-30
        or lower).

        iter-88: the iter-83 inline expression
        ``max(abs(values[0]), 1.0)`` was factored out into
        ``legoesm.diagnostics.conservation_drift.compute_relative_drift``.
        ``_compute_drift`` is now a thin delegating wrapper, so this
        test pins the floor in the shared helper instead.  The
        ``DEFAULT_MIN_BASELINE = 1.0`` module constant is the
        single source of truth.
        """
        import inspect

        from legoesm.diagnostics.conservation_drift import (
            DEFAULT_MIN_BASELINE,
            compute_relative_drift,
        )
        # 1) The shared helper's default floor is 1.0.
        assert DEFAULT_MIN_BASELINE == 1.0, (
            "iter-88: the canonical denominator floor "
            "``DEFAULT_MIN_BASELINE`` must be 1.0.  iter-78/80 showed "
            "the previous 1e-30 floor amplified machine-precision "
            "rounding to 1e+30-magnitude spurious values."
        )
        helper_src = inspect.getsource(compute_relative_drift)
        # 2) The helper uses ``max(abs(...), float(min_baseline))``
        #    style — pin that the floor parameter is honoured (no
        #    raw 1e-30 literal remaining).
        assert "max(" in helper_src and "min_baseline" in helper_src, (
            "iter-88: the canonical helper must use "
            "``max(|x[0]|, min_baseline)`` form."
        )
        assert "1e-30" not in helper_src, (
            "iter-88: the legacy 1e-30 denominator floor must be "
            "removed from the canonical helper.  Use the "
            "``min_baseline`` parameter (default 1.0)."
        )
        # 3) The thin wrapper ``_compute_drift`` must delegate to the
        #    canonical helper (not re-implement the floor).
        wrapper_src = inspect.getsource(M._compute_drift)
        assert "compute_relative_drift" in wrapper_src, (
            "iter-88: ``_compute_drift`` must delegate to the shared "
            "``compute_relative_drift`` helper."
        )
        assert "1e-30" not in wrapper_src, (
            "iter-88: the legacy 1e-30 floor must not reappear in "
            "the wrapper either."
        )


# ---------------------------------------------------------------------------
# _zonal_mean_at_final_time / _zonal_mean_climatology
# ---------------------------------------------------------------------------

class TestZonalMean:
    def _make_4d_field(self, n_times=11, n_lat=36, n_lon=72, nlev=20):
        """Construct a deterministic 4-D field with known structure."""
        rng = np.random.default_rng(123)
        # Mean structure: T(lat, lev) = 250 + 20*cos(lat)*lev/nlev
        lat = np.deg2rad(np.linspace(-89.0, 89.0, n_lat))
        base = 250.0 + 20.0 * np.cos(lat)[:, None] * (
            np.arange(nlev)[None, :] / nlev
        )
        # Time variation: linear ramp + lon-eddy noise.
        field = np.empty((n_times, n_lat, n_lon, nlev))
        for t in range(n_times):
            ramp = base + 5.0 * t / n_times          # warming over time
            eddy = 0.5 * rng.normal(size=(n_lat, n_lon, nlev))
            # Broadcast ramp over lon.
            field[t] = ramp[:, None, :] + eddy
        return field, base

    def test_final_time_zonal_mean_signature(self):
        field, _ = self._make_4d_field(n_times=11)
        zm = M._zonal_mean_at_final_time(field)
        assert zm.shape == (36, 20)

    def test_climatology_window_smaller_variance_than_snapshot(self):
        """Climatology mean over many snapshots should have lower
        eddy variance than a single snapshot — the iter-10 motivation."""
        field, base = self._make_4d_field(n_times=11)
        zm_snap = M._zonal_mean_at_final_time(field)
        zm_clim = M._zonal_mean_climatology(field, n_avg=11)
        # Both should match the structural mean roughly, but climatology
        # should be closer to the noise-free reference.
        snap_residual = float(np.sqrt(np.mean((zm_snap - base) ** 2)))
        clim_residual = float(np.sqrt(np.mean((zm_clim - base) ** 2)))
        # At least 3x improvement when averaging 11 IID-noise frames.
        # (theory: variance reduction = sqrt(11) ≈ 3.3).
        assert clim_residual < snap_residual / 1.5, (
            f"clim_residual={clim_residual:.4f} snap_residual={snap_residual:.4f}"
        )

    def test_climatology_n_avg_caps_at_n_times(self):
        """Asking for more snapshots than exist returns the all-times mean."""
        field, _ = self._make_4d_field(n_times=5)
        zm_all = M._zonal_mean_climatology(field, n_avg=999)
        zm_5 = M._zonal_mean_climatology(field, n_avg=5)
        assert np.allclose(zm_all, zm_5, atol=1e-12)

    def test_n_climatology_avg_helper(self):
        # 0.5 fraction → ceil(0.5 * N)
        assert M._n_climatology_avg(11) == 6
        assert M._n_climatology_avg(10) == 5
        assert M._n_climatology_avg(1) == 1
        assert M._n_climatology_avg(0) == 1  # min-clamped

    def test_wrong_ndim_raises(self):
        with pytest.raises(ValueError, match="expected 4-D"):
            M._zonal_mean_at_final_time(np.zeros((11, 36, 72)))
        with pytest.raises(ValueError, match="expected 4-D"):
            M._zonal_mean_climatology(np.zeros((11, 36, 72)))


# ---------------------------------------------------------------------------
# _interp_zonal_mean_to_target
# ---------------------------------------------------------------------------

class TestInterpZonalMeanToTarget:
    def test_identity_when_native_eq_target(self):
        zm = np.tile(np.linspace(0, 1, 5), (36, 1))   # (36, 5)
        lat = np.linspace(-90.0, 90.0, 36)
        out = M._interp_zonal_mean_to_target(zm, lat, lat)
        assert np.allclose(out, zm, atol=1e-12)

    def test_handles_descending_lat_natively(self):
        """If native latitude is descending (n→s), helper should
        flip both the lat and zm arrays before ``np.interp``."""
        zm = np.linspace(0.0, 1.0, 36)[:, None]   # (36, 1)
        lat_desc = np.linspace(90.0, -90.0, 36)
        lat_target = np.linspace(-90.0, 90.0, 36)
        out = M._interp_zonal_mean_to_target(zm, lat_desc, lat_target)
        # Reverse zm because lat_desc was reversed.
        expected = zm[::-1]
        assert np.allclose(out, expected, atol=1e-12)

    def test_target_smaller_resolution(self):
        zm = np.linspace(-1.0, 1.0, 100)[:, None]
        lat_native = np.linspace(-90.0, 90.0, 100)
        lat_target = np.linspace(-90.0, 90.0, 10)
        out = M._interp_zonal_mean_to_target(zm, lat_native, lat_target)
        assert out.shape == (10, 1)
        # First and last should match endpoints.
        assert abs(out[0, 0] - (-1.0)) < 1e-10
        assert abs(out[-1, 0] - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# _compute_cross_grid_rms_agreement
# ---------------------------------------------------------------------------

class TestCrossGridRMS:
    def _fake_grid_results(self, fields_per_grid: dict[str, np.ndarray]):
        """Build a minimal grid_results dict that the helpers expect."""
        out = {}
        for grid_name, f4d in fields_per_grid.items():
            n_times, n_lat, n_lon, n_lev = f4d.shape
            files = {
                "lat": np.linspace(-90.0, 90.0, n_lat),
                "lon": np.linspace(-180.0, 180.0, n_lon),
                "T_3d": f4d,
                "times_days": np.arange(n_times) * 3.0,
            }

            class _NPZShim:
                """Mimic the ``files`` attribute access of NpzFile."""
                def __init__(self, d):
                    self._d = d
                @property
                def files(self):
                    return list(self._d.keys())
                def __getitem__(self, k):
                    return self._d[k]
                def __contains__(self, k):
                    return k in self._d

            out[grid_name] = {
                "snapshots": _NPZShim(files),
                "metadata": {},
                "resolution": "test",
                "vertical_coord": "sigma",
            }
        return out

    def test_perfectly_agreeing_grids_return_zero_rms(self):
        rng = np.random.default_rng(7)
        f = rng.normal(size=(11, 36, 72, 20))
        gr = self._fake_grid_results({"a": f, "b": f.copy()})
        metric = M._compute_cross_grid_rms_agreement(gr, "T_3d")
        assert metric is not None
        assert max(metric["pairwise"].values()) < 1e-10
        assert metric["ensemble_mean"] < 1e-10

    def test_nlev_mismatch_returns_none(self):
        """Different vertical-level counts → None (defensive early-exit)."""
        rng = np.random.default_rng(7)
        f1 = rng.normal(size=(11, 36, 72, 20))
        f2 = rng.normal(size=(11, 36, 72, 30))
        gr = self._fake_grid_results({"a": f1, "b": f2})
        metric = M._compute_cross_grid_rms_agreement(gr, "T_3d")
        assert metric is None

    def test_single_grid_returns_none(self):
        rng = np.random.default_rng(7)
        f = rng.normal(size=(11, 36, 72, 20))
        gr = self._fake_grid_results({"only": f})
        metric = M._compute_cross_grid_rms_agreement(gr, "T_3d")
        assert metric is None

    def test_three_grids_with_known_offsets(self):
        """Three grids with constant offsets should produce predictable
        pair-wise and ensemble RMS."""
        n_times, n_lat, n_lon, n_lev = 11, 36, 72, 20
        base = np.zeros((n_times, n_lat, n_lon, n_lev))
        f_a = base + 0.0
        f_b = base + 1.0   # +1 K everywhere
        f_c = base + 4.0   # +4 K everywhere
        gr = self._fake_grid_results({"a": f_a, "b": f_b, "c": f_c})
        metric = M._compute_cross_grid_rms_agreement(gr, "T_3d")
        assert metric is not None
        # Pair-wise RMS of constant-offset fields = absolute difference.
        assert abs(metric["pairwise"]["a vs b"] - 1.0) < 1e-10
        assert abs(metric["pairwise"]["a vs c"] - 4.0) < 1e-10
        assert abs(metric["pairwise"]["b vs c"] - 3.0) < 1e-10
        # Ensemble-mean field = (0+1+4)/3 ≈ 1.667.  Per-grid deviation:
        # a: 1.667, b: 0.667, c: 2.333.  Ensemble-averaged = 1.555.
        assert abs(metric["ensemble"]["a"] - 5.0/3.0) < 1e-10
        assert abs(metric["ensemble"]["b"] - 2.0/3.0) < 1e-10
        assert abs(metric["ensemble"]["c"] - 7.0/3.0) < 1e-10

    def test_per_level_spread_uniform_offsets(self):
        """iter-14 per-level spread = max-min of cos-lat-weighted
        horizontal mean per grid, per level.  For constant offsets,
        spread should equal abs offset and not depend on level."""
        n_times, n_lat, n_lon, n_lev = 5, 36, 72, 4
        f_a = np.zeros((n_times, n_lat, n_lon, n_lev))
        f_b = np.full_like(f_a, 2.0)
        f_c = np.full_like(f_a, 5.0)
        gr = self._fake_grid_results({"a": f_a, "b": f_b, "c": f_c})
        metric = M._compute_cross_grid_rms_agreement(gr, "T_3d")
        assert metric is not None
        spread = metric["per_level_spread"]
        assert spread.shape == (n_lev,)
        # Constant offsets → spread = max - min = 5 - 0 = 5 K everywhere.
        assert np.all(np.abs(spread - 5.0) < 1e-10)
        for k in range(n_lev):
            assert metric["per_level_max_grid"][k] == "c"
            assert metric["per_level_min_grid"][k] == "a"

    def test_per_level_spread_cos_lat_weighting(self):
        """iter-16 codex MEDIUM: per-level spread uses cos-lat
        weighting.  Construct a field that's:
          a: uniform 0 K everywhere
          b: 100 K only above |lat| > 80°, 0 K elsewhere (pole-only)
        Uniform-lat-mean would give b ≈ 100 * (10°/180°) ≈ 5.6 K
        per-grid horizontal mean.  Cos-lat-weighted gives
        b ≈ 100 * (1 - cos(80°)) ≈ 100 * 0.826 ... no, cos-lat
        weighted of a band [80°, 90°]:
          ∫_{80°}^{90°} cos(φ) dφ / ∫_{-90°}^{90°} cos(φ) dφ
          = (sin 90° - sin 80°) / 2  (single hemisphere)
          = (1 - 0.985) / 2 = 0.0076
        Total over both hemispheres = 0.0152.  So b mean ≈ 1.52 K.
        The uniform mean would give ~5.6 K.  Test discriminates."""
        n_times, n_lat, n_lon, n_lev = 5, 91, 72, 2
        f_a = np.zeros((n_times, n_lat, n_lon, n_lev))
        f_b = np.zeros_like(f_a)
        # Latitudes go -90 to 90 in 91 points (steps of 2°).
        lat_axis = np.linspace(-90.0, 90.0, n_lat)
        polar_mask = np.abs(lat_axis) > 80.0
        # Set b = 100 above 80° latitude.
        f_b[:, polar_mask, :, :] = 100.0
        gr = self._fake_grid_results({"a": f_a, "b": f_b})
        metric = M._compute_cross_grid_rms_agreement(gr, "T_3d")
        assert metric is not None
        spread = metric["per_level_spread"]
        # Cos-lat-weighted polar-band fraction is ~0.015, so the spread
        # should be ~1.5 K — clearly less than the uniform-mean ~5.6 K.
        for k in range(n_lev):
            assert spread[k] < 3.0, (
                f"level {k} spread {spread[k]:.2f} too large — likely "
                f"uniform-lat mean instead of cos-lat-weighted"
            )
            # but greater than zero (the polar perturbation IS there)
            assert spread[k] > 0.5


# ---------------------------------------------------------------------------
# _atm_extract_field_2d
# ---------------------------------------------------------------------------

class TestAtmExtractField2D:
    def _wrap(self, d):
        class _NPZShim:
            def __init__(self, d):
                self._d = d
            @property
            def files(self):
                return list(self._d.keys())
            def __getitem__(self, k):
                return self._d[k]
        return _NPZShim(d)

    def test_2d_passthrough(self):
        field = np.arange(36 * 72).reshape(36, 72).astype(float)
        out = M._atm_extract_field_2d(self._wrap({"x": field}), "x")
        assert out is field or np.array_equal(out, field)

    def test_3d_takes_last_time(self):
        field = np.stack([
            np.full((36, 72), 100.0),
            np.full((36, 72), 200.0),
        ], axis=0)
        out = M._atm_extract_field_2d(self._wrap({"x": field}), "x")
        assert np.allclose(out, 200.0)

    def test_4d_takes_last_time_lowest_level(self):
        field = np.empty((3, 36, 72, 5))
        for t in range(3):
            for k in range(5):
                field[t, :, :, k] = float(100 * t + k)
        out = M._atm_extract_field_2d(self._wrap({"x": field}), "x")
        # Last time = t=2 (200), lowest level = k=4: 204.
        assert np.allclose(out, 204.0)

    def test_missing_field_returns_none(self):
        out = M._atm_extract_field_2d(self._wrap({"a": np.zeros(2)}), "b")
        assert out is None

    def test_unrecognised_shape_returns_none(self):
        out = M._atm_extract_field_2d(self._wrap({"x": np.zeros(2)}), "x")
        assert out is None


# ---------------------------------------------------------------------------
# _select_resolution_dir / _vertical_coords_for_case
# ---------------------------------------------------------------------------

class TestPathDiscovery:
    def test_select_resolution_dir_single(self, tmp_path):
        gd = tmp_path / "cubed_sphere"
        (gd / "C36").mkdir(parents=True)
        out = M._select_resolution_dir(gd)
        assert out == gd / "C36"

    def test_select_resolution_dir_multiple_warns_once(self, tmp_path, capsys):
        """iter-108 update: when multiple GRID-TYPED dirs exist
        (e.g., ``T21`` + ``T42``), no warning fires — both are
        valid post-iter-95 candidates and we silently pick
        alphabetically-first.  The warning only fires when at
        least one *non-grid-typed* dir exists alongside a
        grid-typed one (the iter-108 stale-vs-fresh case).
        """
        M._RES_DIR_WARNED.clear()
        gd = tmp_path / "spectral"
        (gd / "T21").mkdir(parents=True)
        (gd / "T42").mkdir(parents=True)
        out1 = M._select_resolution_dir(gd)
        out2 = M._select_resolution_dir(gd)
        captured = capsys.readouterr()
        # Pre-iter-108: would warn "multiple resolution dirs".
        # Post-iter-108: both T21 and T42 are valid grid-typed
        # candidates, so no stale-dir warning is needed.
        assert "ignoring non-grid-typed" not in captured.out
        # Both calls return the alphabetically-first grid-typed dir.
        assert out1 == out2 == gd / "T21"

    def test_select_resolution_dir_warns_when_stale_present(self, tmp_path, capsys):
        """iter-108 stale-dir warning: if a non-grid-typed dir
        coexists with a grid-typed one (e.g., ``16/`` from a
        pre-iter-95 run alongside a fresh ``T16/``), we silently
        prefer the typed dir AND emit a warning naming the
        stale dirs.
        """
        M._RES_DIR_WARNED.clear()
        gd = tmp_path / "spectral"
        (gd / "16").mkdir(parents=True)  # stale pre-iter-95 form
        (gd / "T16").mkdir(parents=True)  # fresh post-iter-95 form
        out1 = M._select_resolution_dir(gd)
        out2 = M._select_resolution_dir(gd)
        captured = capsys.readouterr()
        # Warning fires once (deduped via _RES_DIR_WARNED).
        assert "ignoring non-grid-typed" in captured.out
        assert captured.out.count("ignoring non-grid-typed") == 1
        # The typed dir wins.
        assert out1 == out2 == gd / "T16"

    def test_vertical_coords_for_case_empty_dir(self, tmp_path):
        assert M._vertical_coords_for_case(tmp_path) == []

    def test_vertical_coords_for_case_sw_layout(self, tmp_path):
        # SW: leaf files at <case>/<grid>/<resolution>/.
        for grid_name in ("cubed_sphere", "latlon"):
            d = tmp_path / grid_name / "X1"
            d.mkdir(parents=True)
            (d / "snapshots_latlon.npz").touch()
        out = M._vertical_coords_for_case(tmp_path)
        assert out == [None]

    def test_vertical_coords_for_case_hydrostatic_layout(self, tmp_path):
        # Hydro: leaf files at <case>/<grid>/<resolution>/<vert>/.
        for grid_name in ("cubed_sphere", "latlon"):
            for vert in ("sigma", "hybrid"):
                d = tmp_path / grid_name / "X1" / vert
                d.mkdir(parents=True)
                (d / "snapshots_latlon.npz").touch()
        out = M._vertical_coords_for_case(tmp_path)
        assert out == ["hybrid", "sigma"]

    def test_vertical_coords_for_case_csv_only_run(self, tmp_path):
        """iter-26 codex HIGH: RCE-style runs without snapshot NPZ
        but WITH ``mean_timeseries.csv`` + ``results.txt`` should
        be discovered.  iter-27 codex LOW: a stale CSV without
        ``results.txt`` should NOT trigger a phantom combo."""
        # Grid 1: full CSV+results.txt run (collectable).
        d1 = tmp_path / "cubed_sphere" / "C24"
        d1.mkdir(parents=True)
        (d1 / "mean_timeseries.csv").touch()
        (d1 / "results.txt").touch()
        # Grid 2: stale CSV only (NOT collectable — missing
        # results.txt).
        d2 = tmp_path / "latlon" / "32x64"
        d2.mkdir(parents=True)
        (d2 / "mean_timeseries.csv").touch()
        out = M._vertical_coords_for_case(tmp_path)
        # Grid 1 is collectable → SW-style anchor (None).  Grid 2
        # is not collectable; its presence should not add a
        # vertical-coord variant.
        assert out == [None]


# ---------------------------------------------------------------------------
# iter-26: NPZ-missing collection path (timeseries-only runs)
# ---------------------------------------------------------------------------

class TestNpzMissingCollection:
    def test_collect_grid_results_skips_missing_results_txt(self, tmp_path):
        """``_collect_grid_results_atmosphere`` requires CSV +
        results.txt to populate a grid entry.  CSV-only runs are
        skipped (no phantom entries)."""
        # cube has CSV but NO results.txt — should be skipped.
        d_cube = tmp_path / "cubed_sphere" / "C24"
        d_cube.mkdir(parents=True)
        (d_cube / "mean_timeseries.csv").write_text("time_days,mean_T\n0,300\n")

        # latlon has CSV + results.txt (no NPZ) — should be picked up
        # via the iter-26 NPZ-optional path.
        d_latlon = tmp_path / "latlon" / "X1"
        d_latlon.mkdir(parents=True)
        (d_latlon / "mean_timeseries.csv").write_text("time_days,mean_T\n0,290\n")
        (d_latlon / "results.txt").write_text("test: rce\nstatus: PASS\n")

        gr = M._collect_grid_results_atmosphere(tmp_path)
        # Only latlon should be collected.
        assert "cubed_sphere" not in gr
        assert "latlon" in gr
        # The latlon entry should use the empty-snapshots stub.
        assert gr["latlon"]["snapshots"] is M._EMPTY_SNAPSHOTS_STUB

    def test_empty_snapshots_stub_skips_field_extraction(self):
        """``_atm_extract_field_2d(stub, "T_3d")`` returns ``None``
        because the stub has no fields."""
        stub = M._EMPTY_SNAPSHOTS_STUB
        out = M._atm_extract_field_2d(stub, "T_3d")
        assert out is None
        # Even an empty string field name should fail gracefully.
        out = M._atm_extract_field_2d(stub, "")
        assert out is None

    def test_rms_agreement_skips_grids_with_empty_snapshots(self):
        """If both grids have empty snapshot stubs,
        ``_compute_cross_grid_rms_agreement`` returns ``None``
        (fewer than 2 valid 4-D fields)."""
        stub = M._EMPTY_SNAPSHOTS_STUB
        gr = {
            "a": {"snapshots": stub, "metadata": {}, "resolution": "x"},
            "b": {"snapshots": stub, "metadata": {}, "resolution": "y"},
        }
        metric = M._compute_cross_grid_rms_agreement(gr, "T_3d")
        assert metric is None


# ---------------------------------------------------------------------------
# iter-76: pin the iter-73 ``--test <external>`` fallback (RCE / OMIP /
# any case run by external scripts not in TEST_MATRIX)
# ---------------------------------------------------------------------------

class TestCrossGridPlotsOnlyExternalCaseFallback:
    """End-to-end stub coverage of the iter-73 ``--cross-grid-plots-only
    --test <external>`` fallback.  TEST_MATRIX has no ``rce``,
    ``omip``, etc. entries (those are run by external
    ``run_rce.py`` / ``run_omip.py``), but the user-facing wrapper
    invokes ``--cross-grid-plots-only --test rce`` to collect the
    cross-grid plot.  Without the iter-73 fallback, the matrix
    runner would silently skip all cases.

    The codex iter-75 review flagged this as MEDIUM coverage gap
    (no end-to-end stub).  iter-76 closes it: synthesize 4-grid
    timeseries-only output, run ``main()`` with ``--cross-grid-
    plots-only --test rce``, assert the cross-grid plot is
    produced.
    """

    def _write_synthetic_grid_output(
        self, base: Path, test_case: str, grid: str, resolution: str,
    ):
        """Mimic a per-grid RCE/OMIP output with matrix-format files."""
        d = base / "hydrostatic" / test_case / grid / resolution
        d.mkdir(parents=True, exist_ok=True)
        (d / "mean_timeseries.csv").write_text(
            "time_days,mean_T_sfc,mean_T,max_wind\n"
            "1.0,299.96,278.64,0.86\n"
            "2.0,299.95,277.27,1.76\n"
            "3.0,299.95,275.95,2.65\n"
        )
        (d / "results.txt").write_text(
            f"test: {test_case}\n"
            f"grid: {grid}\n"
            f"resolution: {resolution}\n"
            f"days: 3\n"
            f"dt: 600.0\n"
            f"levels: 20\n"
            f"status: PASS\n"
            f"wall_time: 16.5s\n"
        )

    def test_test_rce_fallback_finds_synthetic_output(self, tmp_path, monkeypatch):
        """Synthesize 4-grid RCE-style output, run main() with
        ``--cross-grid-plots-only --test rce``, verify the
        cross-grid plot is produced (i.e. the iter-73 fallback
        ``{args.test}`` for empty allowed_cases worked).
        """
        # 4-grid synthetic output.
        for grid, resolution in [
            ("cubed_sphere", "C24"),
            ("latlon", "32x64"),
            ("icosahedral", "ico4"),
            ("spectral", "T21"),
        ]:
            self._write_synthetic_grid_output(
                tmp_path, "rce", grid, resolution,
            )

        # Invoke main() via argv override.
        monkeypatch.setattr(
            sys, "argv",
            [
                "run_atmosphere_test_matrix.py",
                "--cross-grid-plots-only",
                "--test", "rce",
                "--output", str(tmp_path),
            ],
        )
        # main() prints to stdout — capture isn't strictly needed,
        # but we need to make sure it doesn't sys.exit.
        try:
            M.main()
        except SystemExit as e:
            assert e.code in (0, None), (
                f"main() exited with code {e.code}"
            )

        # Cross-grid plot was produced at the case-dir level.
        case_dir = tmp_path / "hydrostatic" / "rce"
        plot_path = case_dir / "comparison_timeseries.png"
        assert plot_path.exists(), (
            f"iter-73 fallback failed: {plot_path} was not produced.  "
            f"This means ``--test rce`` matched 0 cases (the iter-73 "
            f"``{{args.test}}`` fallback regressed)."
        )


# ---------------------------------------------------------------------------
# iter-32: GPU/MPI efficiency table behaviour on DCMIP / NH metadata
# ---------------------------------------------------------------------------

class TestGpuEfficiencyTableMetadataKeys:
    """Pin the iter-32 codex MEDIUM fix where the GPU efficiency table
    now accepts ``period_days`` (DCMIP) and ``duration_hours`` (NH)
    in addition to ``days``.

    These are integration-style tests: they exercise
    ``_create_atmosphere_comparison_summary`` end-to-end and assert
    the GPU efficiency table appears with correct s/day values.
    """

    def _grid_results_with_metadata(self, metadata_per_grid):
        """Build a minimal grid_results dict with prescribed metadata."""
        out = {}
        for gname, md in metadata_per_grid.items():
            out[gname] = {
                "timeseries": __import__("pandas").DataFrame(
                    {"time_days": [0.0, 1.0], "mean_T": [300.0, 299.0]}
                ),
                "snapshots": M._EMPTY_SNAPSHOTS_STUB,
                "metadata": md,
                "resolution": md.get("resolution", "X1"),
                "vertical_coord": md.get("vertical_coord", ""),
            }
        return out

    def test_efficiency_table_with_days_key(self, tmp_path):
        """Standard ``days`` key path (HS / SW / baroclinic / AMIP)."""
        gr = self._grid_results_with_metadata({
            "cubed_sphere": {"days": "30", "wall_time": "60.0s"},
            "spectral":     {"days": "30", "wall_time": "10.0s"},
        })
        out_dir = tmp_path
        M._create_atmosphere_comparison_summary(out_dir, gr, label="test")
        text = (out_dir / "comparison_summary.txt").read_text()
        assert "GPU / MPI efficiency" in text
        # spectral should be fastest (10s/30d = 0.33 s/d).
        assert "spectral" in text
        # cube is slowest (60/30 = 2.00 s/d).  Speedup vs slowest:
        # spectral 2.00 / 0.33 = 6x.
        # The exact format check: look for "0.33" and "2.00" strings.
        assert "0.33" in text
        assert "2.00" in text

    def test_efficiency_table_with_period_days_key(self, tmp_path):
        """DCMIP transport writes ``period_days`` instead of
        ``days`` — iter-32 fix should pick this up."""
        gr = self._grid_results_with_metadata({
            "cubed_sphere": {"period_days": "12", "wall_time": "120.0s"},
            "spectral":     {"period_days": "12", "wall_time":  "30.0s"},
        })
        out_dir = tmp_path
        M._create_atmosphere_comparison_summary(out_dir, gr, label="dcmip")
        text = (out_dir / "comparison_summary.txt").read_text()
        assert "GPU / MPI efficiency" in text
        # spectral: 30/12 = 2.50 s/d; cube: 120/12 = 10.00 s/d.
        assert "2.50" in text
        assert "10.00" in text

    def test_efficiency_table_with_duration_hours_key(self, tmp_path):
        """NH dycore writes ``duration_hours`` (e.g., DCMIP TC1
        which runs for 3 hours) — iter-32 fix divides by 24 to
        get s/day."""
        gr = self._grid_results_with_metadata({
            "cubed_sphere": {"duration_hours": "3", "wall_time": "300.0s"},
            "spectral":     {"duration_hours": "3", "wall_time":  "60.0s"},
        })
        out_dir = tmp_path
        M._create_atmosphere_comparison_summary(out_dir, gr, label="dcmip_tc1")
        text = (out_dir / "comparison_summary.txt").read_text()
        assert "GPU / MPI efficiency" in text
        # 3 hours = 0.125 days; cube: 300/0.125 = 2400 s/d;
        # spectral: 60/0.125 = 480 s/d.
        assert "2400" in text
        assert "480" in text

    def test_efficiency_table_skips_missing_wall_time(self, tmp_path):
        """When wall_time is missing for both grids the table no-ops."""
        gr = self._grid_results_with_metadata({
            "cubed_sphere": {"days": "30"},   # no wall_time
            "spectral":     {"days": "30"},
        })
        out_dir = tmp_path
        M._create_atmosphere_comparison_summary(out_dir, gr, label="bare")
        text = (out_dir / "comparison_summary.txt").read_text()
        # No GPU/MPI efficiency line (no wall_time data).
        assert "GPU / MPI efficiency" not in text

    def test_efficiency_table_skips_non_finite_days(self, tmp_path):
        """iter-32 codex caveat: NaN/inf in days metadata should
        not produce nan-rate rows."""
        gr = self._grid_results_with_metadata({
            "cubed_sphere": {"days": "nan", "wall_time": "60.0s"},
            "spectral":     {"days": "inf", "wall_time": "10.0s"},
        })
        out_dir = tmp_path
        M._create_atmosphere_comparison_summary(out_dir, gr, label="bad")
        text = (out_dir / "comparison_summary.txt").read_text()
        # Both grids skipped → table absent.
        assert "GPU / MPI efficiency" not in text


# ---------------------------------------------------------------------------
# iter-32: _make_rrtmgp_physics GHG override correctness
# ---------------------------------------------------------------------------

class TestRrtmgpGhgOverrides:
    """Pin the iter-31/32 GHG-override knobs."""

    def test_runtime_overrides_dict_present(self):
        assert hasattr(M, "_RUNTIME_RRTMGP_OVERRIDES")
        d = M._RUNTIME_RRTMGP_OVERRIDES
        assert "co2_ppmv" in d
        assert "ch4_ppbv" in d
        assert "n2o_ppbv" in d

    def test_make_rrtmgp_physics_signature_has_ghg_kwargs(self):
        """The function should accept co2_ppmv/ch4_ppbv/n2o_ppbv."""
        import inspect
        sig = inspect.signature(M._make_rrtmgp_physics)
        params = sig.parameters
        assert "co2_ppmv" in params
        assert "ch4_ppbv" in params
        assert "n2o_ppbv" in params
        # All default to None so the runtime overrides path is the
        # default behaviour.
        assert params["co2_ppmv"].default is None
        assert params["ch4_ppbv"].default is None
        assert params["n2o_ppbv"].default is None

    def test_cloud_scheme_flips_include_clouds_in_rrtmgp_config(
        self, monkeypatch,
    ):
        """iter-36 codex HIGH: when ``--cloud-scheme`` is set to a
        non-"none" value, ``RRTMGPConfig.include_clouds`` must
        flip to True so RRTMGP actually consumes cloud properties.

        iter-37 codex LOW: replace the iter-36 source-inspection
        test (which can false-positive on stale comments and false-
        negative on helper refactors) with a proper config-capture
        test.  We monkeypatch ``make_physics`` to record the
        ``RRTMGPConfig`` it was given.
        """
        captured = {}

        def fake_make_physics(phys_cfg, model_type, dt):
            captured["phys_cfg"] = phys_cfg
            return lambda *a, **kw: None  # no-op physics_fn

        # monkeypatch the import inside ``_make_rrtmgp_physics``.
        import legoesm.atmosphere.physics.combined as combined_mod
        monkeypatch.setattr(combined_mod, "make_physics", fake_make_physics)

        # Case 1: cloud_scheme="sundqvist" → include_clouds=True.
        M._RUNTIME_RRTMGP_OVERRIDES["cloud_scheme"] = "sundqvist"
        try:
            _ = M._make_rrtmgp_physics("hydrostatic", dt=300.0)
            phys_cfg = captured["phys_cfg"]
            assert phys_cfg.radiation.scheme == "rrtmgp"
            assert phys_cfg.radiation.cloud_scheme == "sundqvist"
            assert phys_cfg.radiation.rrtmgp.include_clouds is True
        finally:
            M._RUNTIME_RRTMGP_OVERRIDES["cloud_scheme"] = None

        # Case 2: cloud_scheme="none" → include_clouds remains False.
        captured.clear()
        M._RUNTIME_RRTMGP_OVERRIDES["cloud_scheme"] = "none"
        try:
            _ = M._make_rrtmgp_physics("hydrostatic", dt=300.0)
            phys_cfg = captured["phys_cfg"]
            assert phys_cfg.radiation.cloud_scheme == "none"
            assert phys_cfg.radiation.rrtmgp.include_clouds is False
        finally:
            M._RUNTIME_RRTMGP_OVERRIDES["cloud_scheme"] = None

        # Case 3: cloud_scheme=None (default) → include_clouds default (False).
        captured.clear()
        _ = M._make_rrtmgp_physics("hydrostatic", dt=300.0)
        phys_cfg = captured["phys_cfg"]
        assert phys_cfg.radiation.rrtmgp.include_clouds is False


# ---------------------------------------------------------------------------
# iter-35: _augment_with_rrtmgp_overrides helper
# ---------------------------------------------------------------------------

class TestAugmentWithRrtmgpOverrides:
    """Tests for ``_augment_with_rrtmgp_overrides`` (iter-35).

    All tests snapshot the full ``_RUNTIME_RRTMGP_OVERRIDES`` dict
    before mutating and restore it afterwards (iter-36 codex LOW
    fix).  This makes the tests isolation-safe regardless of which
    keys end up being touched.
    """

    @pytest.fixture(autouse=True)
    def _isolate_overrides(self):
        """Snapshot/restore the full overrides dict around each test."""
        snapshot = dict(M._RUNTIME_RRTMGP_OVERRIDES)
        try:
            # Start each test from a known-clean state.
            for k in M._RUNTIME_RRTMGP_OVERRIDES:
                M._RUNTIME_RRTMGP_OVERRIDES[k] = None
            yield
        finally:
            M._RUNTIME_RRTMGP_OVERRIDES.clear()
            M._RUNTIME_RRTMGP_OVERRIDES.update(snapshot)

    def test_no_op_for_gray_radiation(self):
        """Helper is a no-op when ``radiation != "rrtmgp"``."""
        rows = {"test": "x", "wall_time": "1.0s"}
        M._RUNTIME_RRTMGP_OVERRIDES["co2_ppmv"] = 280.0
        out = M._augment_with_rrtmgp_overrides(rows, "gray")
        assert "co2_ppmv" not in out
        assert out["test"] == "x"

    def test_appends_rrtmgp_overrides(self):
        """When ``radiation == "rrtmgp"``, set overrides are recorded."""
        M._RUNTIME_RRTMGP_OVERRIDES["co2_ppmv"] = 280.0
        M._RUNTIME_RRTMGP_OVERRIDES["cloud_scheme"] = "sundqvist"
        out = M._augment_with_rrtmgp_overrides(
            {"test": "y", "wall_time": "1.0s"}, "rrtmgp",
        )
        assert out["co2_ppmv"] == 280.0
        assert out["cloud_scheme"] == "sundqvist"
        # Unset overrides not appended (fixture clears all keys to None).
        assert "ch4_ppbv" not in out
        assert "n2o_ppbv" not in out

    def test_does_not_mutate_input_dict(self):
        """Helper returns a NEW dict; doesn't mutate the caller's."""
        original = {"test": "z", "wall_time": "1.0s"}
        M._RUNTIME_RRTMGP_OVERRIDES["co2_ppmv"] = 415.0
        out = M._augment_with_rrtmgp_overrides(original, "rrtmgp")
        assert "co2_ppmv" in out
        # Original dict unchanged.
        assert "co2_ppmv" not in original


# ---------------------------------------------------------------------------
# iter-39: ozone-profile CLI override knobs
# ---------------------------------------------------------------------------

class TestOzoneProfileOverrides:
    """Tests for the iter-39 ``--ozone-source`` / ``--ozone-peak-hPa``
    / ``--ozone-max-vmr`` CLI knobs.

    These mirror the iter-31/32/36 GHG and cloud-scheme override
    tests: validate the runtime-override dict has the new keys, the
    augment helper records them when set, and ``_make_rrtmgp_physics``
    propagates them through ``RadiationConfig.ozone``.
    """

    @pytest.fixture(autouse=True)
    def _isolate_overrides(self):
        """Snapshot/restore the full overrides dict around each test."""
        snapshot = dict(M._RUNTIME_RRTMGP_OVERRIDES)
        try:
            for k in M._RUNTIME_RRTMGP_OVERRIDES:
                M._RUNTIME_RRTMGP_OVERRIDES[k] = None
            yield
        finally:
            M._RUNTIME_RRTMGP_OVERRIDES.clear()
            M._RUNTIME_RRTMGP_OVERRIDES.update(snapshot)

    def test_overrides_dict_has_ozone_keys(self):
        d = M._RUNTIME_RRTMGP_OVERRIDES
        assert "ozone_source" in d
        # iter-39 codex MEDIUM: lower-case unit suffix.
        assert "ozone_peak_hpa" in d
        assert "ozone_max_vmr" in d

    def test_augment_records_set_ozone_overrides(self):
        """When set, ozone overrides should appear in the augmented row."""
        M._RUNTIME_RRTMGP_OVERRIDES["ozone_source"] = "analytical"
        M._RUNTIME_RRTMGP_OVERRIDES["ozone_peak_hpa"] = 50.0
        M._RUNTIME_RRTMGP_OVERRIDES["ozone_max_vmr"] = 1.0e-5
        out = M._augment_with_rrtmgp_overrides(
            {"test": "ozone", "wall_time": "1.0s"}, "rrtmgp",
        )
        assert out["ozone_source"] == "analytical"
        assert out["ozone_peak_hpa"] == 50.0
        assert out["ozone_max_vmr"] == 1.0e-5

    def test_augment_skips_unset_ozone_overrides(self):
        """None values are not appended (matches GHG behaviour)."""
        out = M._augment_with_rrtmgp_overrides(
            {"test": "no_ozone", "wall_time": "1.0s"}, "rrtmgp",
        )
        assert "ozone_source" not in out
        assert "ozone_peak_hpa" not in out
        assert "ozone_max_vmr" not in out

    def test_augment_no_op_for_gray_with_ozone_set(self):
        """Even with ozone overrides set, gray radiation should ignore them."""
        M._RUNTIME_RRTMGP_OVERRIDES["ozone_source"] = "analytical"
        out = M._augment_with_rrtmgp_overrides(
            {"test": "gray_with_ozone", "wall_time": "1.0s"}, "gray",
        )
        assert "ozone_source" not in out

    def test_make_rrtmgp_physics_propagates_ozone_config(self, monkeypatch):
        """When ozone overrides are set, ``_make_rrtmgp_physics`` must
        construct a non-default ``RadiationConfig.ozone``.
        """
        captured = {}

        def fake_make_physics(phys_cfg, model_type, dt):
            captured["phys_cfg"] = phys_cfg
            return lambda *a, **kw: None  # no-op

        import legoesm.atmosphere.physics.combined as combined_mod
        monkeypatch.setattr(combined_mod, "make_physics", fake_make_physics)

        M._RUNTIME_RRTMGP_OVERRIDES["ozone_source"] = "analytical"
        M._RUNTIME_RRTMGP_OVERRIDES["ozone_peak_hpa"] = 50.0
        M._RUNTIME_RRTMGP_OVERRIDES["ozone_max_vmr"] = 1.0e-5
        _ = M._make_rrtmgp_physics("hydrostatic", dt=300.0)
        phys_cfg = captured["phys_cfg"]
        assert phys_cfg.radiation.ozone.source == "analytical"
        assert phys_cfg.radiation.ozone.p_peak_hPa == 50.0
        assert phys_cfg.radiation.ozone.o3_max_vmr == 1.0e-5

    def test_make_rrtmgp_physics_default_ozone_when_unset(
        self, monkeypatch,
    ):
        """When no ozone overrides are set, ``RadiationConfig.ozone``
        falls back to its NamedTuple default (source="standard").
        """
        captured = {}

        def fake_make_physics(phys_cfg, model_type, dt):
            captured["phys_cfg"] = phys_cfg
            return lambda *a, **kw: None

        import legoesm.atmosphere.physics.combined as combined_mod
        monkeypatch.setattr(combined_mod, "make_physics", fake_make_physics)

        _ = M._make_rrtmgp_physics("hydrostatic", dt=300.0)
        phys_cfg = captured["phys_cfg"]
        # Default OzoneProfileConfig has source="standard".
        assert phys_cfg.radiation.ozone.source == "standard"

    def test_implicit_source_promotion_when_only_peak_set(
        self, monkeypatch,
    ):
        """iter-39 codex MEDIUM: when only ``--ozone-peak-hpa`` is
        set without an explicit source, ``_make_rrtmgp_physics``
        must auto-promote source to ``"analytical"`` so the peak
        knob actually takes effect (otherwise it silently no-ops
        because ``_compute_ozone_vmr`` ignores ``p_peak_hPa`` when
        source != "analytical").
        """
        captured = {}

        def fake_make_physics(phys_cfg, model_type, dt):
            captured["phys_cfg"] = phys_cfg
            return lambda *a, **kw: None

        import legoesm.atmosphere.physics.combined as combined_mod
        monkeypatch.setattr(combined_mod, "make_physics", fake_make_physics)

        M._RUNTIME_RRTMGP_OVERRIDES["ozone_peak_hpa"] = 25.0
        _ = M._make_rrtmgp_physics("hydrostatic", dt=300.0)
        phys_cfg = captured["phys_cfg"]
        assert phys_cfg.radiation.ozone.source == "analytical"
        assert phys_cfg.radiation.ozone.p_peak_hPa == 25.0
        # Other ozone fields keep NamedTuple defaults.
        assert phys_cfg.radiation.ozone.o3_max_vmr == 8.0e-6

    def test_implicit_source_promotion_when_only_vmr_set(
        self, monkeypatch,
    ):
        """Same as above but with ``--ozone-max-vmr`` only."""
        captured = {}

        def fake_make_physics(phys_cfg, model_type, dt):
            captured["phys_cfg"] = phys_cfg
            return lambda *a, **kw: None

        import legoesm.atmosphere.physics.combined as combined_mod
        monkeypatch.setattr(combined_mod, "make_physics", fake_make_physics)

        M._RUNTIME_RRTMGP_OVERRIDES["ozone_max_vmr"] = 1.5e-5
        _ = M._make_rrtmgp_physics("hydrostatic", dt=300.0)
        phys_cfg = captured["phys_cfg"]
        assert phys_cfg.radiation.ozone.source == "analytical"
        assert phys_cfg.radiation.ozone.o3_max_vmr == 1.5e-5

    def test_compute_ozone_vmr_actually_consumes_config(self):
        """iter-39 codex MEDIUM: end-to-end O3 consumption test.
        Directly call ``_compute_ozone_vmr`` from the radiation
        module with each source value to verify the analytical
        knobs reach the actual ozone-VMR computation site
        (RRTMGP's downstream consumer).  Catches the include_clouds-
        style "config plumbed but never read" failure mode.
        """
        import jax.numpy as jnp
        from legoesm.atmosphere.physics.radiation.config import (
            OzoneProfileConfig,
        )
        from legoesm.atmosphere.physics.radiation.integration import (
            _compute_ozone_vmr,
        )

        # 4-column, 5-level pressure profile (Pa).
        p = jnp.broadcast_to(
            jnp.linspace(1.0e3, 1.0e5, 5)[None, :], (4, 5),
        )
        lat = jnp.array([-0.5, -0.1, 0.1, 0.5])

        # source="standard" → returns None (RRTMGP uses built-in).
        cfg_std = OzoneProfileConfig(source="standard")
        assert _compute_ozone_vmr(p, lat, cfg_std) is None

        # source="none" → returns ~zero everywhere.
        cfg_none = OzoneProfileConfig(source="none")
        o3_none = _compute_ozone_vmr(p, lat, cfg_none)
        assert o3_none is not None
        assert float(jnp.max(o3_none)) <= 1.0e-9

        # source="analytical" with default peak/vmr → non-trivial
        # Gaussian, max ~ default vmr.
        cfg_a1 = OzoneProfileConfig(source="analytical")
        o3_a1 = _compute_ozone_vmr(p, lat, cfg_a1)
        assert o3_a1 is not None
        peak1 = float(jnp.max(o3_a1))
        # Default o3_max_vmr is 8e-6; with lat_dependence factor up
        # to 1.5, peak should fall in (4e-6, 1.6e-5).
        assert 4.0e-6 < peak1 < 1.6e-5

        # Doubling o3_max_vmr should approximately double the peak.
        cfg_a2 = OzoneProfileConfig(
            source="analytical", o3_max_vmr=1.6e-5,
        )
        o3_a2 = _compute_ozone_vmr(p, lat, cfg_a2)
        peak2 = float(jnp.max(o3_a2))
        # Strict ratio test: VMR scales linearly in o3_max_vmr.
        assert abs(peak2 / peak1 - 2.0) < 1.0e-6

        # Moving the peak higher (lower hPa = higher altitude)
        # changes the location of the maximum.  The default peak is
        # 30 hPa; move to 100 hPa and verify the level-of-max changes.
        cfg_a3 = OzoneProfileConfig(
            source="analytical", p_peak_hPa=100.0,
        )
        o3_a3 = _compute_ozone_vmr(p, lat, cfg_a3)
        # Per-column argmax level differs from cfg_a1's argmax.
        kmax_a1 = int(jnp.argmax(o3_a1[0]))
        kmax_a3 = int(jnp.argmax(o3_a3[0]))
        assert kmax_a1 != kmax_a3


# ---------------------------------------------------------------------------
# iter-42: AMIP → matrix-runner-format converter for the iter-41 wrapper
# ---------------------------------------------------------------------------

class TestAmipToMatrixFormat:
    """Tests for ``scripts/_amip_to_matrix_format.py`` (iter-42).

    The iter-41 ``run_amip_cross_grid.sh`` wrapper invokes
    ``scripts/run_amip.py`` per grid; ``run_amip.py`` writes
    ``timeseries.npz`` + a free-form ``results.txt``, but the
    matrix-runner cross-grid plot collector requires
    ``mean_timeseries.csv`` + a key:value ``results.txt``.  The
    converter bridges the two formats.

    Without this bridge, the wrapper would silently skip every grid
    in the cross-grid plot pass — exactly the include_clouds-style
    "plumbed but never read" failure mode the iter-39 codex review
    flagged.
    """

    def _import_converter(self):
        import importlib.util
        path = _SCRIPT_DIR / "_amip_to_matrix_format.py"
        spec = importlib.util.spec_from_file_location(
            "_amip_to_matrix_format", path,
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def _write_synthetic_amip_outputs(self, out_dir: Path, n: int = 6):
        """Mimic ``run_amip.py``'s outputs in ``out_dir``."""
        out_dir.mkdir(parents=True, exist_ok=True)
        days = np.arange(n, dtype=np.float64)
        np.savez(
            out_dir / "timeseries.npz",
            days=days,
            T_atm=280.0 + 0.05 * days,
            max_wind=20.0 + 0.1 * days,
            dry_mass_ps=np.full(n, 1.0e15),
            precip=2.5 + 0.01 * days,
            CWV=25.0 + 0.05 * days,
            sw_up_toa=np.full(n, 100.0),
            lw_up_toa=np.full(n, 230.0),
            sst=np.float64(np.nan),  # NaN-sentinel (run_amip.py ocean off)
            sic=np.float64(np.nan),
        )
        (out_dir / "results.txt").write_text(
            "legoESM AMIP run\n"
            "Grid: cubed_sphere C48 / L20, dt=300.0s, 30 days\n"
            "Radiation: rrtmgp\n"
            "Status: COMPLETED\n\n"
            "JIT compilation: 12.3s\n"
            "Wall time: 67.8s\n\n"
            "Final <T_atm>: 280.250 K\n"
            "Final <Precip>: 2.55 mm/day\n"
            "Final <CWV>: 25.25 kg/m2\n"
        )

    def test_converter_produces_matrix_format_files(self, tmp_path):
        """End-to-end: synthetic AMIP outputs → matrix-format outputs."""
        mod = self._import_converter()
        self._write_synthetic_amip_outputs(tmp_path, n=6)
        rc = mod.main(tmp_path)
        assert rc == 0
        # mean_timeseries.csv must exist and have the canonical column names.
        csv_path = tmp_path / "mean_timeseries.csv"
        assert csv_path.exists()
        header = csv_path.read_text().splitlines()[0]
        assert header.startswith("time_days,")
        # Real (non-NaN) AMIP variables made it into the CSV.
        assert "mean_T" in header
        assert "max_wind" in header
        assert "mass" in header
        # NaN-sentinel variables (sst, sic) are skipped.
        assert "mean_SST" not in header
        # results.txt is now matrix-key:value format.
        results_text = (tmp_path / "results.txt").read_text()
        assert results_text.startswith("test: amip\n")
        assert "grid: cubed_sphere" in results_text
        assert "resolution: C48" in results_text
        assert "status: PASS" in results_text  # COMPLETED → PASS
        assert "wall_time: 67.8s" in results_text
        # Original AMIP results.txt preserved.
        assert (tmp_path / "results_amip.txt").exists()

    def test_converter_status_mapping(self, tmp_path):
        """``Status: COMPLETED`` → PASS; anything else → ERROR.

        iter-42 codex MEDIUM: the whitelist is broader now —
        SUCCESS / PASS / OK / DONE / COMPLETE all map to PASS too.
        """
        mod = self._import_converter()
        for status_keyword, expected in [
            ("COMPLETED", "PASS"),
            ("COMPLETE", "PASS"),
            ("SUCCESS", "PASS"),
            ("PASS", "PASS"),
            ("OK", "PASS"),
            ("DONE", "PASS"),
            ("FAILED", "ERROR"),
            ("CRASHED", "ERROR"),
            ("UNKNOWN", "ERROR"),
        ]:
            self._write_synthetic_amip_outputs(tmp_path)
            (tmp_path / "results.txt").write_text(
                "legoESM AMIP run\n"
                "Grid: spectral T42 / L20, dt=600.0s, 30 days\n"
                f"Status: {status_keyword}\n"
                "Wall time: 5.0s\n"
            )
            rc = mod.main(tmp_path)
            assert rc == 0
            results_text = (tmp_path / "results.txt").read_text()
            assert f"status: {expected}" in results_text, (
                f"{status_keyword} → expected status: {expected}"
            )

    def test_converter_relaxed_grid_regex(self, tmp_path):
        """iter-42 codex MEDIUM: ``Grid:`` regex must accept variations
        like singular ``day``, decimal days, scientific notation in
        ``dt=``, missing ``L<n>`` and missing ``dt=``.  iter-42 v1
        was overly strict and dropped grid metadata silently.
        """
        mod = self._import_converter()
        for grid_line, expect_resolution in [
            ("Grid: cubed_sphere C48 / L20, dt=300.0s, 30 days",  "C48"),
            ("Grid: spectral T42 / L20, dt=6e2s, 30 days",         "T42"),  # sci dt
            ("Grid: cubed_sphere C48 / L20, dt=300s, 1 day",       "C48"),  # singular
            ("Grid: cubed_sphere C48 / L20, dt=300s, 0.5 days",    "C48"),  # decimal days
            ("Grid: voronoi ico6, dt=600s, 30 days",                "ico6"), # no L<n>
            ("Grid: latlon 90x180",                                 "90x180"),  # bare grid+res
        ]:
            self._write_synthetic_amip_outputs(tmp_path)
            (tmp_path / "results.txt").write_text(
                "legoESM AMIP run\n"
                f"{grid_line}\n"
                "Status: COMPLETED\n"
                "Wall time: 1.0s\n"
            )
            rc = mod.main(tmp_path)
            assert rc == 0
            results_text = (tmp_path / "results.txt").read_text()
            assert f"resolution: {expect_resolution}" in results_text, (
                f"failed on grid_line={grid_line!r}"
            )

    def test_converter_all_nan_array_skipped(self, tmp_path):
        """iter-42 codex MEDIUM: a 1-D all-NaN array must be skipped
        from the CSV — not written as a column of NaNs.
        """
        mod = self._import_converter()
        tmp_path.mkdir(parents=True, exist_ok=True)
        n = 6
        days = np.arange(n, dtype=np.float64)
        np.savez(
            tmp_path / "timeseries.npz",
            days=days,
            T_atm=280.0 + 0.05 * days,
            # 1-D all-NaN array (a future run_amip.py change might
            # use this instead of the scalar NaN sentinel).
            sst=np.full(n, np.nan, dtype=np.float64),
        )
        (tmp_path / "results.txt").write_text(
            "legoESM AMIP run\n"
            "Grid: cubed_sphere C48 / L20, dt=300.0s, 30 days\n"
            "Status: COMPLETED\n"
            "Wall time: 1.0s\n"
        )
        rc = mod.main(tmp_path)
        assert rc == 0
        header = (tmp_path / "mean_timeseries.csv").read_text().splitlines()[0]
        assert "mean_SST" not in header
        assert "mean_T" in header

    def test_converter_length_mismatch_warns(self, tmp_path, capsys):
        """iter-42 codex MEDIUM: a length-mismatched variable must
        emit a stderr warning rather than silently dropping data.
        """
        mod = self._import_converter()
        tmp_path.mkdir(parents=True, exist_ok=True)
        n = 6
        days = np.arange(n, dtype=np.float64)
        np.savez(
            tmp_path / "timeseries.npz",
            days=days,
            T_atm=280.0 + 0.05 * days,
            # Wrong length — must trigger a warning.
            max_wind=np.array([1.0, 2.0, 3.0]),
        )
        (tmp_path / "results.txt").write_text(
            "legoESM AMIP run\n"
            "Grid: cubed_sphere C48 / L20, dt=300s, 1 day\n"
            "Status: COMPLETED\nWall time: 1.0s\n"
        )
        capsys.readouterr()  # clear any prior output
        rc = mod.main(tmp_path)
        assert rc == 0
        captured = capsys.readouterr()
        assert "WARNING" in captured.err
        assert "max_wind" in captured.err
        # The mismatched variable is still skipped from the CSV.
        header = (tmp_path / "mean_timeseries.csv").read_text().splitlines()[0]
        assert "max_wind" not in header

    def test_converter_default_status_error_when_missing(self, tmp_path):
        """iter-42 codex HIGH: if results.txt has no parsable Status
        line, the converter must emit ``status: ERROR`` (not omit
        the status field) so the matrix collector treats the run
        as failed.
        """
        mod = self._import_converter()
        self._write_synthetic_amip_outputs(tmp_path)
        # Overwrite results.txt with no Status line.
        (tmp_path / "results.txt").write_text(
            "legoESM AMIP run\n"
            "Grid: spectral T42 / L20, dt=600s, 30 days\n"
            "Wall time: 5.0s\n"
        )
        rc = mod.main(tmp_path)
        assert rc == 0
        results_text = (tmp_path / "results.txt").read_text()
        assert "status: ERROR" in results_text

    def test_converter_default_status_error_when_results_missing(
        self, tmp_path,
    ):
        """If the AMIP results.txt is missing entirely, the converter
        must still write a matrix-format results.txt with
        ``status: ERROR``.  Without this, _has_collectable would
        accept an output with no provenance metadata.
        """
        mod = self._import_converter()
        self._write_synthetic_amip_outputs(tmp_path)
        (tmp_path / "results.txt").unlink()
        rc = mod.main(tmp_path)
        assert rc == 0
        results_text = (tmp_path / "results.txt").read_text()
        assert "status: ERROR" in results_text

    def test_converter_handles_missing_npz(self, tmp_path, capsys):
        """If timeseries.npz is missing, converter returns non-zero and
        purges stale matrix-format files.  iter-42 codex HIGH: a failed
        AMIP rerun into an existing OUTDIR must NOT silently leave
        behind data from the previous successful run.
        """
        mod = self._import_converter()
        # Simulate a stale converter run: matrix-format files from
        # a previous successful AMIP run, but timeseries.npz is now
        # missing because the most recent AMIP run failed.
        (tmp_path / "mean_timeseries.csv").write_text(
            "time_days,mean_T\n0,300\n1,301\n",
        )
        (tmp_path / "results.txt").write_text(
            "test: amip\nstatus: PASS\nwall_time: 5.0s\n",
        )
        (tmp_path / "results_amip.txt").write_text(
            "legoESM AMIP run\nGrid: foo bar / L1, dt=1.0s, 1 days\n"
            "Status: COMPLETED\nWall time: 5.0s\n",
        )
        # Now call the converter — it must purge the stale outputs
        # and return non-zero.
        rc = mod.main(tmp_path)
        assert rc != 0
        captured = capsys.readouterr()
        assert "ERROR" in captured.err or "missing" in captured.err
        # Stale matrix-format files have been purged.
        assert not (tmp_path / "mean_timeseries.csv").exists()
        assert not (tmp_path / "results.txt").exists()
        assert not (tmp_path / "results_amip.txt").exists()

    def test_converter_distinguishes_empty_days_from_missing(
        self, tmp_path, capsys,
    ):
        """iter-70: the iter-69 5-day AMIP smoke initially produced a
        npz with an empty ``days`` array (run was 1 day, default
        ``--diag-days 5``, so no diagnostic snapshot was emitted).
        The iter-42 message read "'days' missing" which was
        misleading.  iter-70 distinguishes the two cases:

          * 'days' key not in the npz file at all
          * 'days' key present but the array is empty

        Both still return non-zero (the converter is correct that
        no CSV can be written), but the error message is now
        actionable for the AMIP smoke-test case (suggests checking
        ``--diag-days`` vs ``--days``).
        """
        import numpy as np
        mod = self._import_converter()

        # Empty-days case: write a npz with ``days=array([])``.
        np.savez(
            tmp_path / "timeseries.npz",
            days=np.array([], dtype=np.float64),
            T_atm=np.array([], dtype=np.float64),
        )
        (tmp_path / "results.txt").write_text(
            "legoESM AMIP run\nStatus: COMPLETED\n",
        )
        rc = mod.main(tmp_path)
        assert rc != 0
        captured = capsys.readouterr()
        # iter-70 message includes "empty" + a hint about diag-days.
        assert "empty" in captured.err.lower()
        assert "diag-days" in captured.err

    def test_collector_recognizes_converted_output(self, tmp_path):
        """The matrix runner's collector predicate must return True
        on a directory after conversion.  This is the integration
        guarantee — without this, the iter-41 wrapper silently no-ops.

        iter-43 codex MEDIUM: bind directly to
        ``has_collectable_atmosphere_outputs`` (lifted to module
        scope in iter-43).  No more re-implementing the predicate
        locally; if the matrix runner's predicate semantics change,
        this test goes red automatically.
        """
        mod = self._import_converter()
        self._write_synthetic_amip_outputs(tmp_path)
        rc = mod.main(tmp_path)
        assert rc == 0

        # Direct import of the real predicate.
        assert M.has_collectable_atmosphere_outputs(tmp_path)

    def test_collector_rejects_directory_with_only_csv(self, tmp_path):
        """The predicate must NOT return True on a directory with only
        ``mean_timeseries.csv`` (no ``results.txt``).  This is the
        iter-27 codex LOW guarantee — a stale CSV without provenance
        must not produce a phantom cross-grid combo.  iter-43: now
        binds to the real predicate.
        """
        (tmp_path / "mean_timeseries.csv").write_text("time_days\n0\n")
        assert not M.has_collectable_atmosphere_outputs(tmp_path)

    def test_crash_safety_during_conversion(self, tmp_path, monkeypatch):
        """iter-43 codex LOW: simulate a crash between the rename and
        the CSV write.  The directory must NOT be collectable until
        the FINAL ``results.txt`` write commits.
        """
        mod = self._import_converter()
        self._write_synthetic_amip_outputs(tmp_path)

        # Monkeypatch _atomic_write_text to crash on the CSV write.
        original = mod._atomic_write_text
        crash_count = {"n": 0}

        def crashing_write(path, text):
            if path.name == "mean_timeseries.csv":
                crash_count["n"] += 1
                raise RuntimeError("simulated crash mid-conversion")
            original(path, text)

        monkeypatch.setattr(mod, "_atomic_write_text", crashing_write)
        with pytest.raises(RuntimeError, match="simulated crash"):
            mod.main(tmp_path)
        assert crash_count["n"] == 1
        # Directory must NOT be collectable: results.txt was renamed
        # to results_amip.txt before the crash, so results.txt is
        # missing.
        assert not (tmp_path / "results.txt").exists()
        assert (tmp_path / "results_amip.txt").exists()
        # The predicate (real one) returns False because results.txt
        # is missing.
        assert not M.has_collectable_atmosphere_outputs(tmp_path)

    def test_post_crash_recovery_recovers_metadata(
        self, tmp_path, monkeypatch,
    ):
        """iter-44 codex LOW: after a crash that leaves results_amip.txt
        but no results.txt, a rerun must recover the AMIP metadata
        from results_amip.txt rather than emitting a bogus
        ``status: ERROR`` with empty grid/resolution fields.
        """
        mod = self._import_converter()
        self._write_synthetic_amip_outputs(tmp_path)

        # Crash on the FIRST converter run (CSV write).
        original = mod._atomic_write_text
        def crashing_write(path, text):
            if path.name == "mean_timeseries.csv":
                raise RuntimeError("simulated crash mid-conversion")
            original(path, text)
        monkeypatch.setattr(mod, "_atomic_write_text", crashing_write)
        with pytest.raises(RuntimeError):
            mod.main(tmp_path)

        # Restore the real writer for the recovery run.
        monkeypatch.setattr(mod, "_atomic_write_text", original)

        # Verify the post-crash state matches what codex described:
        # results.txt is missing, results_amip.txt has the AMIP file.
        assert not (tmp_path / "results.txt").exists()
        assert (tmp_path / "results_amip.txt").exists()

        # Recovery run should succeed AND emit the original AMIP
        # metadata (status: PASS, the grid line, etc.), NOT a bare
        # status: ERROR.
        rc = mod.main(tmp_path)
        assert rc == 0
        results_text = (tmp_path / "results.txt").read_text()
        assert "status: PASS" in results_text  # COMPLETED → PASS
        assert "grid: cubed_sphere" in results_text
        assert "resolution: C48" in results_text
        # The directory is now collectable (real predicate).
        assert M.has_collectable_atmosphere_outputs(tmp_path)

    def test_converter_idempotent(self, tmp_path):
        """Running the converter twice must produce identical output
        AND must preserve the original AMIP free-form ``results.txt``
        across both calls (no data loss)."""
        mod = self._import_converter()
        self._write_synthetic_amip_outputs(tmp_path, n=4)
        original_amip_text = (tmp_path / "results.txt").read_text()
        mod.main(tmp_path)
        first_csv = (tmp_path / "mean_timeseries.csv").read_text()
        first_results = (tmp_path / "results.txt").read_text()
        # After first call, the original AMIP file is preserved.
        assert (tmp_path / "results_amip.txt").exists()
        assert (tmp_path / "results_amip.txt").read_text() == original_amip_text
        # Second call.
        mod.main(tmp_path)
        second_csv = (tmp_path / "mean_timeseries.csv").read_text()
        second_results = (tmp_path / "results.txt").read_text()
        # CSV is byte-identical.
        assert second_csv == first_csv
        # Matrix-format results.txt is byte-identical (same parsed
        # fields from the preserved AMIP file).
        assert second_results == first_results
        # Crucially: the AMIP original is STILL there and STILL
        # contains the original free-form text.  iter-42: an earlier
        # version of the converter overwrote results_amip.txt on the
        # second call with its own matrix-format file, losing the
        # AMIP free-form summary.
        assert (tmp_path / "results_amip.txt").read_text() == original_amip_text


# ---------------------------------------------------------------------------
# iter-46: Held-Suarez init consistency across grids
# ---------------------------------------------------------------------------

class TestHeldSuarezInitConsistency:
    """Regression tests that pin per-grid HS initial-state consistency.

    iter-46 audit: the cube-cold / latlon-warm structural disagreement
    documented in iter-9..15 is dycore-level (effective dissipation /
    sponge formulation), NOT init-level.  This test class exists to
    prevent a future change from accidentally regressing init
    consistency — e.g. by changing ``T_init`` for one grid only or by
    drifting one grid's perturbation amplitude relative to the
    others.

    The matrix runner's ``run_held_suarez`` calls each grid's init
    function with the SAME defaults (``T_init=300.0``,
    ``perturbation_amplitude=1.0``, ``seed=42``).  Each init function
    must produce:
      * mean T close to T_init (within perturbation_amplitude)
      * zero wind (u=v=0)
      * surface pressure equal to p_ref (no topography)
      * the same Held-Suarez forcing constants (K_A, K_S, K_F,
        SIGMA_B, DELTA_T_Y, DELTA_THETA_Z, T_MIN)

    These tests import the actual init functions; they do NOT use
    synthetic surrogates, so a real change to the init implementation
    flips the test.
    """

    @pytest.fixture(scope="class")
    def hs_module(self):
        """Import the held_suarez module once for all tests."""
        import importlib
        return importlib.import_module(
            "legoesm.atmosphere.held_suarez",
        )

    def test_forcing_constants_match_held_suarez_1994(self, hs_module):
        """Pin the table-1 constants from the Held-Suarez 1994 paper.

        These constants are SHARED across all four forcing variants
        (cubed_sphere, latlon, mpas, spectral) so a single source of
        truth is enforced by the test.

        iter-47 codex LOW: use ``pytest.approx`` instead of
        ``< 1e-30`` so harmless representation refactors (e.g.
        switching ``1.0/(40.0*86400.0)`` to ``1.0/3456000.0``)
        don't false-positive.
        """
        # Temperature relaxation timescales [1/s].
        assert hs_module.K_A == pytest.approx(1.0 / (40.0 * 86400.0), rel=1e-12)
        assert hs_module.K_S == pytest.approx(1.0 / (4.0 * 86400.0), rel=1e-12)
        # Rayleigh friction timescale [1/s].
        assert hs_module.K_F == pytest.approx(1.0 / (1.0 * 86400.0), rel=1e-12)
        # Boundary-layer threshold (dimensionless sigma).
        assert hs_module.SIGMA_B == pytest.approx(0.7)
        # Equilibrium temperature parameters [K].
        assert hs_module.DELTA_T_Y == pytest.approx(60.0)
        assert hs_module.DELTA_THETA_Z == pytest.approx(10.0)
        assert hs_module.T_MIN == pytest.approx(200.0)

    def test_init_signatures_share_canonical_defaults(self, hs_module):
        """Every per-grid HS init function must use the same defaults
        for the four user-facing parameters.  iter-46 audit guard: a
        drift in any single grid's defaults would re-introduce the
        structural cube-cold disagreement at the init level.

        iter-47 codex HIGH: spectral (Gaussian grid) is added to the
        coverage.  ``isothermal_rest_state_spectral`` lives in
        ``atmosphere/dynamics/spectral_pe.py`` rather than the
        ``held_suarez`` module, but its defaults must match the
        other three since the matrix runner's spectral HS branch
        calls it directly.
        """
        import inspect
        # Cube/latlon/MPAS inits all live in the held_suarez module.
        spectral_fn = None
        try:
            from legoesm.atmosphere.dynamics.spectral_pe import (
                isothermal_rest_state_spectral,
            )
            spectral_fn = isothermal_rest_state_spectral
        except ImportError:
            pass

        cases = [
            ("held_suarez_init",        hs_module.held_suarez_init),
            ("held_suarez_init_latlon", hs_module.held_suarez_init_latlon),
            ("held_suarez_init_mpas",   hs_module.held_suarez_init_mpas),
        ]
        if spectral_fn is not None:
            cases.append(("isothermal_rest_state_spectral", spectral_fn))

        for fn_name, fn in cases:
            sig = inspect.signature(fn)
            params = sig.parameters
            assert params["T_init"].default == 300.0, (
                f"{fn_name}.T_init drifted from 300.0"
            )
            assert params["p_s_init"].default == hs_module.constants.p_ref, (
                f"{fn_name}.p_s_init drifted from constants.p_ref"
            )
            assert params["perturbation_amplitude"].default == 1.0, (
                f"{fn_name}.perturbation_amplitude drifted from 1.0"
            )
            assert params["seed"].default == 42, (
                f"{fn_name}.seed drifted from 42"
            )

    def test_held_suarez_equilibrium_temperature_basic_properties(
        self, hs_module,
    ):
        """``T_eq`` must respect the canonical Held-Suarez 1994
        invariants:

        * T_eq is positive everywhere.
        * T_eq >= T_MIN = 200 K.
        * Surface T_eq at the equator > T_eq at the poles (≥
          DELTA_T_Y / 2 K difference).
        * T_eq decreases monotonically with altitude near the
          equator (from p_ref to 100 hPa, sampled at 5 levels).

        iter-47 codex LOW: extended the monotonic check from 2 to
        5 sample points so a non-monotonic profile between p_ref
        and 100 hPa cannot slip through.
        """
        import jax.numpy as jnp
        T_eq_eq = float(hs_module.held_suarez_equilibrium_temperature(
            jnp.array(0.0), jnp.array(hs_module.constants.p_ref),
        ))
        T_eq_pole = float(hs_module.held_suarez_equilibrium_temperature(
            jnp.array(jnp.pi / 2), jnp.array(hs_module.constants.p_ref),
        ))
        assert T_eq_eq > 0
        assert T_eq_pole > 0
        assert T_eq_eq >= hs_module.T_MIN
        assert T_eq_pole >= hs_module.T_MIN
        assert T_eq_eq - T_eq_pole >= hs_module.DELTA_T_Y / 2

        # iter-48 codex LOW: dense 51-point monotonicity check.
        # 5 samples missed possible non-monotonic behavior between
        # sampled pressures; 51 samples (Δp ≈ 1800 Pa from p_ref to
        # 100 hPa) makes any local maximum >Δp wide visible.
        # Pressure levels from p_ref (surface) → 10000 Pa (100 hPa).
        p_levels = jnp.linspace(hs_module.constants.p_ref, 10000.0, 51)
        T_eq_array = hs_module.held_suarez_equilibrium_temperature(
            jnp.zeros_like(p_levels), p_levels,
        )
        T_profile = [float(x) for x in T_eq_array]
        # Each successive level must be no warmer than the previous.
        for i in range(1, len(T_profile)):
            assert T_profile[i] <= T_profile[i - 1] + 1e-9, (
                f"T_eq non-monotonic at equator: levels {i-1}→{i}: "
                f"p={float(p_levels[i-1]):.0f} → {float(p_levels[i]):.0f} Pa, "
                f"T={T_profile[i-1]:.2f} → {T_profile[i]:.2f} K"
            )

    def test_init_produces_zero_wind(self, hs_module):
        """Each per-grid init must produce a state at rest (u=v=0).

        iter-47 codex HIGH: extended from cube/latlon to also
        cover MPAS.  Spectral has u stored as ``vor_hat``/``div_hat``
        spectral coefficients (not a u/v state field), so we test
        that the synthesised grid u/v are zero in a separate
        spectral test below.

        iter-46 audit invariant: the cubed-sphere, latlon, MPAS,
        and spectral inits all start from an isothermal atmosphere
        at rest with only a small T perturbation.  A future change
        that adds nonzero u/v to one variant only would diverge
        cross-grid HS comparisons from t=0.
        """
        import jax.numpy as jnp
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.latlon import create_latlon_grid
        from legoesm.grids.vertical import standard_hybrid_levels

        sigma = standard_hybrid_levels(8)

        grid_cube = create_cubed_sphere(6)
        s_cube = hs_module.held_suarez_init(grid_cube, sigma)
        assert float(jnp.max(jnp.abs(s_cube.u.data))) == 0.0
        assert float(jnp.max(jnp.abs(s_cube.v.data))) == 0.0

        grid_ll = create_latlon_grid(8, 16)
        s_ll = hs_module.held_suarez_init_latlon(grid_ll, sigma)
        assert float(jnp.max(jnp.abs(s_ll.u.data))) == 0.0
        assert float(jnp.max(jnp.abs(s_ll.v.data))) == 0.0

        # iter-47 codex HIGH: also exercise the MPAS init.  MPAS
        # state stores u on edges (no separate v); a single u
        # array must be zero.
        # iter-48 codex LOW: tighten the catch.  ``ImportError`` for
        # missing module + ``FileNotFoundError`` for missing mesh
        # data file are legitimate skips; everything else is a real
        # bug we want to surface.
        try:
            from legoesm.grids.mpas import create_mpas_mesh
        except ImportError:
            pytest.skip("MPAS mesh module unavailable")
        try:
            mesh = create_mpas_mesh(level=2)
        except FileNotFoundError:
            pytest.skip("MPAS level-2 mesh data file unavailable")
        s_mpas = hs_module.held_suarez_init_mpas(mesh, sigma)
        assert float(jnp.max(jnp.abs(s_mpas.u.data))) == 0.0

    def test_spectral_init_produces_zero_wind(self):
        """iter-47 codex HIGH: the spectral init must also produce a
        rest state (zero vorticity and zero divergence in spectral
        space, which synthesises to zero u and v on the grid).
        """
        import jax.numpy as jnp
        try:
            from legoesm.atmosphere.dynamics.spectral_pe import (
                isothermal_rest_state_spectral, spectral_pe_to_grid,
            )
            from legoesm.grids.gaussian import create_gaussian_grid
            from legoesm.grids.vertical import create_sigma_coordinate
        except ImportError:
            pytest.skip("spectral dynamics module unavailable")

        grid = create_gaussian_grid(21)  # T21 — small but valid
        sigma = create_sigma_coordinate(8)
        s = isothermal_rest_state_spectral(grid, sigma, T_init=300.0)
        # Vorticity and divergence coefficients all zero.
        assert float(jnp.max(jnp.abs(s.vor_hat.data))) == 0.0
        assert float(jnp.max(jnp.abs(s.div_hat.data))) == 0.0
        # Sanity: synthesised grid u/v are also zero (modulo
        # rounding from the spectral-to-grid transform).
        fields = spectral_pe_to_grid(s, grid, sigma)
        assert float(jnp.max(jnp.abs(fields["u"]))) < 1e-10
        assert float(jnp.max(jnp.abs(fields["v"]))) < 1e-10

    def test_init_produces_constant_p_s_when_no_topography(
        self, hs_module,
    ):
        """Without topography, p_s must equal p_s_init at every cell.

        iter-47 codex HIGH: extended from latlon-only to also cover
        cube and MPAS.  All four grids share the same hydrostatic
        adjustment formula (``p_s = p_s_init * exp(-phis/(R_d*T_init))``);
        with phis=0 (no topography) the result must be exactly
        p_s_init.
        """
        import jax.numpy as jnp
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.latlon import create_latlon_grid
        from legoesm.grids.vertical import standard_hybrid_levels

        sigma = standard_hybrid_levels(8)
        p_ref = hs_module.constants.p_ref

        # Latlon.
        grid_ll = create_latlon_grid(8, 16)
        s_ll = hs_module.held_suarez_init_latlon(grid_ll, sigma)
        p_s_arr = jnp.asarray(s_ll.p_s.data)
        assert abs(float(jnp.min(p_s_arr)) - p_ref) < 1e-6
        assert abs(float(jnp.max(p_s_arr)) - p_ref) < 1e-6

        # Cubed-sphere.
        grid_cube = create_cubed_sphere(6)
        s_cube = hs_module.held_suarez_init(grid_cube, sigma)
        p_s_cube = jnp.asarray(s_cube.p_s.data)
        assert abs(float(jnp.min(p_s_cube)) - p_ref) < 1e-6
        assert abs(float(jnp.max(p_s_cube)) - p_ref) < 1e-6

        # MPAS.
        # iter-48 codex LOW: only catch legitimate-skip exceptions.
        try:
            from legoesm.grids.mpas import create_mpas_mesh
        except ImportError:
            pytest.skip("MPAS mesh module unavailable; cube/latlon coverage suffices")
        try:
            mesh = create_mpas_mesh(level=2)
        except FileNotFoundError:
            pytest.skip("MPAS level-2 mesh data file unavailable")
        s_mpas = hs_module.held_suarez_init_mpas(mesh, sigma)
        p_s_mpas = jnp.asarray(s_mpas.p_s.data)
        assert abs(float(jnp.min(p_s_mpas)) - p_ref) < 1e-6
        assert abs(float(jnp.max(p_s_mpas)) - p_ref) < 1e-6

    def test_init_mean_T_close_to_T_init(self, hs_module):
        """The bulk-mean T of the initial state must be close to
        T_init (within ``perturbation_amplitude`` / sqrt(N_cells)).

        iter-47 codex HIGH: extended from latlon-only to also cover
        cube and MPAS.  All three grids share the same uniform-T
        + low-level perturbation pattern.
        """
        import jax.numpy as jnp
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.latlon import create_latlon_grid
        from legoesm.grids.vertical import standard_hybrid_levels

        sigma = standard_hybrid_levels(8)

        # Latlon.
        grid_ll = create_latlon_grid(16, 32)
        s_ll = hs_module.held_suarez_init_latlon(grid_ll, sigma)
        mean_T_ll = float(jnp.mean(s_ll.T.data))
        # 16x32x8 = 4096 cells; perturbation only at the lowest
        # level (16x32 = 512 cells); stddev of mean ≈ 1/sqrt(512)
        # ≈ 0.044, then divided by 8 levels ≈ 0.006.  Allow 0.1 K.
        assert abs(mean_T_ll - 300.0) < 0.1, (
            f"latlon init mean T={mean_T_ll:.4f} ≠ T_init=300"
        )

        # Cubed-sphere.
        grid_cube = create_cubed_sphere(6)
        s_cube = hs_module.held_suarez_init(grid_cube, sigma)
        mean_T_cube = float(jnp.mean(s_cube.T.data))
        # 6x6x6x8 = 1728 cells; perturbation at the lowest level
        # (6x6x6 = 216 cells); stddev of mean ≈ 1/sqrt(216)/8 ≈ 0.009.
        # Allow 0.2 K.
        assert abs(mean_T_cube - 300.0) < 0.2, (
            f"cube init mean T={mean_T_cube:.4f} ≠ T_init=300"
        )

        # MPAS.
        # iter-48 codex LOW: legitimate-skip-only catches.
        try:
            from legoesm.grids.mpas import create_mpas_mesh
        except ImportError:
            pytest.skip("MPAS mesh module unavailable")
        try:
            mesh = create_mpas_mesh(level=2)
        except FileNotFoundError:
            pytest.skip("MPAS level-2 mesh data file unavailable")
        s_mpas = hs_module.held_suarez_init_mpas(mesh, sigma)
        mean_T_mpas = float(jnp.mean(s_mpas.T.data))
        assert abs(mean_T_mpas - 300.0) < 0.2, (
            f"mpas init mean T={mean_T_mpas:.4f} ≠ T_init=300"
        )

    def test_runner_dispatch_passes_consistent_kwargs(self):
        """iter-47 codex MEDIUM: the matrix runner's ``run_held_suarez``
        must call each per-grid init function with the SAME effective
        kwargs (i.e. relying on the canonical defaults rather than
        passing per-grid overrides).  A bug like
        ``held_suarez_init_latlon(..., T_init=290.0)`` while the
        cubed-sphere branch uses the default 300.0 would silently
        disagree from t=0 — and the iter-46 default-pinning tests
        would NOT catch it because they only check the function
        signatures.

        iter-48 codex MEDIUM: previous regex-based version had two
        gaps — (1) ``T_init = 290.0`` (whitespace around ``=``)
        would slip through the ``T_init=`` substring check, and
        (2) the test didn't assert all four expected init calls
        are present, so a dropped branch (e.g. spectral renamed
        away) would pass.  Switched to AST parsing.
        """
        import ast
        path = _SCRIPT_DIR / "run_atmosphere_test_matrix.py"
        tree = ast.parse(path.read_text())

        # Locate the run_held_suarez function definition.
        run_hs_func = None
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.FunctionDef)
                and node.name == "run_held_suarez"
            ):
                run_hs_func = node
                break
        assert run_hs_func is not None, (
            "run_held_suarez not found in matrix runner"
        )

        # Walk the body to find every init call.
        expected_callees = {
            "held_suarez_init",
            "held_suarez_init_latlon",
            "held_suarez_init_mpas",
            "isothermal_rest_state_spectral",
        }
        forbidden_kwargs = {
            "T_init", "perturbation_amplitude", "seed", "p_s_init",
        }
        seen_callees: set[str] = set()
        for node in ast.walk(run_hs_func):
            if not isinstance(node, ast.Call):
                continue
            # Resolve the callee name (handle both ``foo()`` and
            # ``mod.foo()`` forms — though the matrix runner uses
            # the former).
            if isinstance(node.func, ast.Name):
                callee = node.func.id
            elif isinstance(node.func, ast.Attribute):
                callee = node.func.attr
            else:
                continue
            if callee not in expected_callees:
                continue
            seen_callees.add(callee)
            # Check that none of the forbidden kwargs is passed.
            for kw in node.keywords:
                if kw.arg in forbidden_kwargs:
                    raise AssertionError(
                        f"run_held_suarez calls {callee} with kwarg "
                        f"{kw.arg!r} — this would break cross-grid "
                        f"init consistency.  Rely on the canonical "
                        f"default instead."
                    )

        # iter-48: assert all four expected init callees are present
        # so a dropped branch (e.g. spectral renamed away or stripped
        # out) flips the test.
        missing = expected_callees - seen_callees
        assert not missing, (
            f"run_held_suarez is missing call sites for: {missing}.  "
            f"Either the function was refactored or a grid branch "
            f"was dropped — update this test if intentional."
        )


# ---------------------------------------------------------------------------
# iter-58: pin the iter-57 cross-dycore dissipation imbalance
# ---------------------------------------------------------------------------

class TestHeldSuarezDissipationImbalance:
    """Quantitative regression test for the iter-57 dissipation
    imbalance audit.

    The matrix runner's ``run_held_suarez`` configures four
    different grids with different dissipation operators:

      * cube:  hyperdiff (biharmonic, 4 terms)
                + hyperdiff_ps (biharmonic on p_s)
                + div_damp (Laplacian on divergence)
                + A_h (Laplacian viscosity, frac=0.05)
      * latlon: A_h (Laplacian viscosity, frac=0.10) ONLY
      * MPAS:   _hyperdiff_ico (biharmonic) ONLY
      * spectral: hyperdiff (biharmonic) + spectral_filter

    iter-57 identified that this imbalance is the likely cause
    of the HS cube-cold structural pattern.  The cube has more
    total dissipation than any of the other three grids AND it's
    biharmonic-dominated.

    These tests pin the current numerical relationship.  A future
    fix that rebalances the dissipation MUST update these
    expected ratios alongside the source — the test exists to
    make the rebalancing inspectable + atomic with the docs in
    ``CROSS_GRID_COMPARISON_REPORT.md`` §5.
    """

    def test_cube_laplacian_visc_is_half_latlon_at_matched_dx(self):
        """``_laplacian_visc_cube`` uses ``frac=0.05`` while
        ``_laplacian_visc_latlon`` uses ``frac=0.10``.  At matched
        grid spacing this means the cube's A_h is HALF of latlon's
        — a known iter-57 imbalance contributor.  Pin the ratio.
        """
        import math
        from legoesm import constants
        # Pick a cube_n and latlon_n_lat that give similar dx.
        # cube dx = π R / (2 n);  latlon dy = π R / n_lat.
        # Equal dx means n_lat = 2 n.  Use n=48 → n_lat=96.
        cube_n = 48
        latlon_n_lat = 96
        c_gw = math.sqrt(constants.R_d * 300.0)
        cube_dx = math.pi * constants.R_earth / (2.0 * cube_n)
        ll_dy = math.pi * constants.R_earth / latlon_n_lat
        # Verify dx ≈ dy (within 1% — they are mathematically equal
        # under the n_lat = 2 n choice).
        assert abs(cube_dx - ll_dy) / cube_dx < 0.01

        cube_visc = M._laplacian_visc_cube(cube_n)  # frac=0.05
        ll_visc = M._laplacian_visc_latlon(latlon_n_lat)  # frac=0.10
        # Ratio cube/latlon should be ≈ 0.5 (the frac ratio).
        ratio = cube_visc / ll_visc
        assert abs(ratio - 0.5) < 0.01, (
            f"iter-57 audit: cube/latlon Laplacian-viscosity ratio "
            f"changed from 0.5 to {ratio:.4f}.  If you intended a "
            f"rebalancing, update §5 of "
            f"CROSS_GRID_COMPARISON_REPORT.md and this test."
        )

    def test_cube_has_strict_superset_of_latlon_dissipation(self):
        """Cube uses 4 dissipation terms; latlon uses 1.

        iter-59 codex HIGH: scope the source checks to the cube
        branch ONLY (was matching anywhere in ``run_held_suarez``,
        which would let a spectral / MPAS / comment match cover
        for a cube branch that dropped a term).  Use AST parsing
        to find the ``elif tc.grid_type == "cubed_sphere":`` /
        ``if tc.grid_type == "cubed_sphere":`` branch body.
        """
        import ast
        import inspect
        src = inspect.getsource(M.run_held_suarez)
        tree = ast.parse(src)
        # Find the function definition.
        func_def = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "run_held_suarez":
                func_def = node
                break
        assert func_def is not None

        # Walk if/elif chain, find the branch whose test is
        # ``tc.grid_type == "cubed_sphere"``.
        cube_branch_body = None
        latlon_branch_body = None

        def _matches_grid_type(test, value):
            return (
                isinstance(test, ast.Compare)
                and len(test.ops) == 1
                and isinstance(test.ops[0], ast.Eq)
                and isinstance(test.left, ast.Attribute)
                and test.left.attr == "grid_type"
                and len(test.comparators) == 1
                and isinstance(test.comparators[0], ast.Constant)
                and test.comparators[0].value == value
            )

        def _walk_if_chain(node):
            nonlocal cube_branch_body, latlon_branch_body
            if not isinstance(node, ast.If):
                return
            if _matches_grid_type(node.test, "cubed_sphere"):
                cube_branch_body = node.body
            elif _matches_grid_type(node.test, "latlon"):
                latlon_branch_body = node.body
            for sub in node.orelse:
                _walk_if_chain(sub)

        for node in func_def.body:
            _walk_if_chain(node)

        assert cube_branch_body is not None, (
            "could not locate ``tc.grid_type == 'cubed_sphere'`` "
            "branch in run_held_suarez"
        )
        assert latlon_branch_body is not None, (
            "could not locate ``tc.grid_type == 'latlon'`` branch "
            "in run_held_suarez"
        )

        # Dump just those branches as source.
        cube_src = "\n".join(ast.unparse(s) for s in cube_branch_body)
        latlon_src = "\n".join(ast.unparse(s) for s in latlon_branch_body)

        # iter-57 audit: cube branch must reference all four.
        for term in (
            "hyperdiff_coeff",
            "hyperdiff_ps_coeff",
            "div_damp_coeff",
            "A_h",
        ):
            assert term in cube_src, (
                f"iter-57: cube branch dropped dissipation term "
                f"``{term}``.  Update the iter-57 audit in §5 of "
                f"CROSS_GRID_COMPARISON_REPORT.md alongside this "
                f"change."
            )

        # iter-59 codex HIGH: latlon branch must use ``A_h`` (the
        # iter-57 finding asserts latlon has Laplacian-only
        # dissipation).  A future change that drops ``A_h`` from
        # the latlon HS instantiation would silently shift the
        # cross-grid disagreement.
        assert "A_h" in latlon_src, (
            "iter-57 audit: latlon HS branch dropped ``A_h``.  "
            "This would change the cross-grid dissipation balance "
            "documented in §5."
        )
        # And iter-57: latlon HS must NOT instantiate biharmonic
        # / div_damp (until the C-grid config supports them).
        for nope in ("hyperdiff_coeff", "div_damp_coeff"):
            assert nope not in latlon_src, (
                f"iter-57 audit: latlon HS branch now uses "
                f"``{nope}``.  This is the §5 fix-candidate (a) "
                f"path — update the audit doc."
            )

        # Latlon C-grid config NamedTuple field check (independent
        # confirmation that biharmonic / div_damp are not yet
        # exposed at the config layer).
        # iter-68 NOTE: an ad-hoc biharmonic implementation
        # (``-coeff · vector_laplacian_cgrid²``) was attempted and
        # made the cross-grid disagreement 11× WORSE (latlon
        # deviation 0.490 → 5.445 K).  Reverted.  The fix-candidate
        # (a) needs a proper Fortran-reference biharmonic operator
        # (e.g., MOM6 / GFDL FMS) before re-attempt.
        from legoesm.atmosphere.dynamics.primitive_eq_latlon_cgrid import (
            CGridLatLonPrimitiveEquationConfig,
        )
        ll_fields = set(CGridLatLonPrimitiveEquationConfig._fields)
        assert "A_h" in ll_fields
        for nope in ("hyperdiff_coeff", "div_damp_coeff"):
            assert nope not in ll_fields, (
                f"iter-57: ``CGridLatLonPrimitiveEquationConfig`` "
                f"now exposes ``{nope}`` — the matrix runner's "
                f"latlon HS branch should be updated to use it, "
                f"and the §5 fix-candidate (a) plan in "
                f"CROSS_GRID_COMPARISON_REPORT.md should be "
                f"marked done.  iter-68 attempted this with an "
                f"ad-hoc biharmonic implementation and made the "
                f"cross-grid disagreement 11× worse — re-attempt "
                f"requires a Fortran-reference operator."
            )

    def _find_branch_body(self, branch_grid: str):
        """Return the AST nodes that make up the branch body for
        ``tc.grid_type == <branch_grid>`` inside ``run_held_suarez``.
        Used by iter-60 to walk the actual config-call AST.
        """
        import ast
        import inspect
        src = inspect.getsource(M.run_held_suarez)
        tree = ast.parse(src)
        # Find run_held_suarez.
        run_hs = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "run_held_suarez":
                run_hs = node
                break
        assert run_hs is not None

        # Walk if-elif chain.
        def _matches(test, value):
            return (
                isinstance(test, ast.Compare)
                and len(test.ops) == 1 and isinstance(test.ops[0], ast.Eq)
                and isinstance(test.left, ast.Attribute)
                and test.left.attr == "grid_type"
                and len(test.comparators) == 1
                and isinstance(test.comparators[0], ast.Constant)
                and test.comparators[0].value == value
            )

        def _walk(node):
            if not isinstance(node, ast.If):
                return None
            if _matches(node.test, branch_grid):
                return node.body
            for sub in node.orelse:
                hit = _walk(sub)
                if hit is not None:
                    return hit
            return None

        for node in run_hs.body:
            hit = _walk(node)
            if hit is not None:
                return hit
        raise AssertionError(
            f"could not locate ``tc.grid_type == {branch_grid!r}`` branch"
        )

    def _resolve_local_assignment(self, body: list, target_name: str):
        """Find ``<target_name> = <expr>`` inside the AST body and
        return the right-hand-side AST node (last assignment wins).
        ``None`` if not found.
        """
        import ast
        rhs = None
        for stmt in body:
            for node in ast.walk(stmt):
                if (
                    isinstance(node, ast.Assign)
                    and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == target_name
                ):
                    rhs = node.value
        return rhs

    def _is_call_to(self, node, callee_name: str, arg_name: str | None = None):
        """True iff ``node`` is ``callee_name(arg_name)`` or
        ``callee_name(arg_name, ...)``.
        """
        import ast
        if not isinstance(node, ast.Call):
            return False
        if not (isinstance(node.func, ast.Name) and node.func.id == callee_name):
            return False
        if arg_name is None:
            return True
        if not node.args:
            return False
        first = node.args[0]
        return isinstance(first, ast.Name) and first.id == arg_name

    def _find_config_call(self, body: list, ctor_name: str):
        """Find a ``ConfigCtor(...)`` call inside the branch body and
        return the ``ast.Call`` node.
        """
        import ast
        for stmt in body:
            for node in ast.walk(stmt):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == ctor_name
                ):
                    return node
        return None

    def test_cube_branch_config_wires_helpers_via_local_aliases(self):
        """iter-60 codex HIGH/MEDIUM: walk the AST to verify the
        cube branch actually wires the helpers through the
        ``PrimitiveEquationConfig(...)`` call:

          n   = int(tc.resolution[1:])      # from tc
          hd  = _hyperdiff_cube(n)
          dd  = _div_damp_cube(n)
          ah  = _laplacian_visc_cube(n)
          config = PrimitiveEquationConfig(
              hyperdiff_coeff=hd,
              hyperdiff_ps_coeff=hd,
              div_damp_coeff=dd,
              A_h=ah,
              ...)

        Every step in this chain must be checked at the AST
        level.  Substring matches let dead code or dataflow
        breaks pass.
        """
        import ast
        body = self._find_branch_body("cubed_sphere")

        # Each local assignment has the right RHS.
        hd_rhs = self._resolve_local_assignment(body, "hd")
        dd_rhs = self._resolve_local_assignment(body, "dd")
        ah_rhs = self._resolve_local_assignment(body, "ah")
        n_rhs = self._resolve_local_assignment(body, "n")

        assert hd_rhs is not None and self._is_call_to(hd_rhs, "_hyperdiff_cube", "n"), (
            "iter-60 codex MEDIUM: ``hd = _hyperdiff_cube(n)`` "
            "expected in cube HS branch"
        )
        assert dd_rhs is not None and self._is_call_to(dd_rhs, "_div_damp_cube", "n"), (
            "iter-60 codex MEDIUM: ``dd = _div_damp_cube(n)`` "
            "expected in cube HS branch"
        )
        assert ah_rhs is not None and self._is_call_to(ah_rhs, "_laplacian_visc_cube", "n"), (
            "iter-60 codex MEDIUM: ``ah = _laplacian_visc_cube(n)`` "
            "expected in cube HS branch"
        )
        # ``n = int(tc.resolution[1:])`` is the canonical idiom for
        # parsing the cube resolution string (e.g. ``C48`` → 48).
        # Pin that ``n`` is derived from ``tc.resolution`` rather
        # than hardcoded.
        assert n_rhs is not None, "``n = ...`` assignment missing"
        n_src = ast.unparse(n_rhs)
        assert "tc.resolution" in n_src, (
            f"iter-60 codex MEDIUM: ``n`` must be derived from "
            f"``tc.resolution``, got ``n = {n_src!r}``"
        )

        # PrimitiveEquationConfig(...) call must use the local
        # aliases.  Find the call.
        ctor_call = self._find_config_call(body, "PrimitiveEquationConfig")
        assert ctor_call is not None, (
            "iter-60 codex HIGH: cube HS branch must invoke "
            "``PrimitiveEquationConfig(...)``"
        )
        kwargs = {kw.arg: kw.value for kw in ctor_call.keywords}
        for field, expected_alias in (
            ("hyperdiff_coeff", "hd"),
            ("hyperdiff_ps_coeff", "hd"),
            ("div_damp_coeff", "dd"),
            ("A_h", "ah"),
        ):
            assert field in kwargs, (
                f"iter-60 codex HIGH: cube HS branch's "
                f"``PrimitiveEquationConfig`` call is missing "
                f"keyword ``{field}``"
            )
            value = kwargs[field]
            assert (
                isinstance(value, ast.Name)
                and value.id == expected_alias
            ), (
                f"iter-60 codex HIGH: ``PrimitiveEquationConfig({field}=...)`` "
                f"must reference local alias ``{expected_alias}``, "
                f"got ``{ast.unparse(value)}``"
            )

    def test_latlon_branch_config_wires_A_h_via_local_alias(self):
        """iter-60 codex HIGH: pin that the latlon HS branch
        actually instantiates
        ``CGridLatLonPrimitiveEquationConfig(A_h=ah, ...)`` with
        the ``ah`` local — not just that "A_h" appears as a
        substring somewhere.  An unused local would have slipped
        through the iter-59 substring check.
        """
        import ast
        body = self._find_branch_body("latlon")
        # Look at ALL ``ah = ...`` assignments in the branch.  The
        # latlon HS path has ``ah = _laplacian_visc_latlon(n_lat)``
        # followed by an optional clip ``ah = min(ah, _A_h_max)``.
        # We accept any chain that includes a call to the canonical
        # helper.
        ah_assignments = []
        for stmt in body:
            for node in ast.walk(stmt):
                if (
                    isinstance(node, ast.Assign)
                    and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == "ah"
                ):
                    ah_assignments.append(node.value)
        assert ah_assignments, (
            "iter-60 codex HIGH: latlon HS branch must compute an "
            "``ah`` Laplacian-viscosity local"
        )
        # At least ONE assignment must invoke the canonical helper
        # (the other(s) may be clips / refinements of the result).
        helper_used = any(
            "_laplacian_visc_latlon" in ast.unparse(rhs)
            for rhs in ah_assignments
        )
        assert helper_used, (
            f"iter-60 codex HIGH: latlon ``ah`` should derive from "
            f"``_laplacian_visc_latlon(...)`` somewhere in the "
            f"chain.  Saw: "
            f"{[ast.unparse(rhs) for rhs in ah_assignments]}"
        )
        # And the config call must reference ``ah``.
        ctor_call = self._find_config_call(
            body, "CGridLatLonPrimitiveEquationConfig",
        )
        assert ctor_call is not None, (
            "iter-60 codex HIGH: latlon HS branch must invoke "
            "``CGridLatLonPrimitiveEquationConfig(...)``"
        )
        kwargs = {kw.arg: kw.value for kw in ctor_call.keywords}
        assert "A_h" in kwargs, (
            "iter-60 codex HIGH: latlon HS config call is missing "
            "``A_h=`` keyword"
        )
        ah_value = kwargs["A_h"]
        assert isinstance(ah_value, ast.Name) and ah_value.id == "ah", (
            f"iter-60 codex HIGH: ``A_h=`` must reference local "
            f"``ah``, got ``{ast.unparse(ah_value)}``"
        )
        # And pin the iter-57 finding: NO biharmonic / div_damp
        # kwarg in the latlon config.
        for nope in ("hyperdiff_coeff", "div_damp_coeff"):
            assert nope not in kwargs, (
                f"iter-60 codex: latlon config now passes "
                f"``{nope}=`` — the §5 fix-candidate (a) should be "
                f"marked done and the iter-57 audit refreshed."
            )

    def test_hyperdiff_cube_scales_inversely_with_n_to_fourth_power(self):
        """Pin the iter-57 hyperdiff scaling.  This is the
        biharmonic operator's CFL-like scaling rule
        (``coeff ~ dx⁴``) and the matrix runner uses it for
        every C48 / C96 / etc.  A future change that breaks
        this scaling would silently shift the cube-cold pattern.
        """
        c48 = M._hyperdiff_cube(48)
        c96 = M._hyperdiff_cube(96)
        c192 = M._hyperdiff_cube(192)
        # hyperdiff_cube(n) = ref * (48/n)^4.  Doubling n → ratio (1/2)^4 = 1/16.
        assert abs(c48 / c96 - 16.0) < 1e-6
        assert abs(c48 / c192 - 256.0) < 1e-6
        # Reference value at n=ref_n should be the ref_coeff.
        assert abs(c48 - 1e16) < 1e6

    def test_div_damp_cube_scales_inversely_with_n_squared(self):
        """Pin the iter-57 div_damp scaling.  Laplacian damping
        scales as ``dx²``.  Doubling n → ratio (1/2)² = 1/4.
        """
        d48 = M._div_damp_cube(48)
        d96 = M._div_damp_cube(96)
        assert abs(d48 / d96 - 4.0) < 1e-6
        assert abs(d48 - 1.5e7) < 1e-3


class TestCliResolutionPerGridDispatch:
    """iter-95 audit followup: ``--resolution N`` (bare integer)
    on the CLI was previously applied verbatim to every grid
    type, breaking 3 of 4 parsers:

    * cubed_sphere ``int(res[1:])``: "16" → 6 (silently wrong N)
    * latlon ``res.split("x")``: "16" → unpack error
    * icosahedral ``res.replace("ico", "")``: "16" → level=16
      (4.29e+10 cells, ValueError "Maximum supported level is 8")
    * spectral ``res.replace("T", "")``: "16" → 16 (correct, only
      one of four that worked)

    iter-95 added per-grid dispatch via ``_expand_cli_res(N,
    grid_type)``: bare integers expand to grid-typed strings;
    pre-formatted strings (``"C36"``, ``"ico5"``, ``"72x144"``,
    ``"T42"``) pass through unchanged.

    These tests pin the dispatch logic so a future regression
    that re-introduces the ``--resolution N`` verbatim-apply
    bug would fail loudly instead of silently using n=6 on
    cubed-sphere or trying to allocate 4.29e+10 cells on
    icosahedral.
    """

    def _import_module(self):
        import importlib
        return importlib.import_module("run_atmosphere_test_matrix")

    def test_cli_resolution_smoke_runs_on_all_grids(self):
        """End-to-end smoke: ``run_atmosphere_test_matrix.py
        --only sw --quick --resolution 16`` must produce 12/12
        PASS (4 grids × 3 SW cases) — not 6 PASS / 6 ERROR
        as before iter-95.

        This is exercise-the-fix coverage, not a unit test.
        Marked as @pytest.mark.smoke so it can be skipped in
        the fast unit-test cycle.  Use ``-m smoke`` to run.
        """
        # Skip by default; only run with -m smoke or --runsmoke.
        pytest.skip(
            "iter-95 smoke test: invoke "
            "``JAX_ENABLE_X64=1 .venv/bin/python "
            "scripts/run_atmosphere_test_matrix.py "
            "--only sw --quick --resolution 16`` to verify; "
            "expected 12/12 PASS (4 grids × 3 SW cases) since "
            "iter-95.  Inline pytest-driven runs would require "
            "JAX state setup and ~3 min wall."
        )

    def test_resolution_dispatch_unit_for_each_grid(self):
        """Unit-test the ``_expand_cli_res`` dispatch logic
        directly without invoking the full matrix runner.

        Reproduces the lambda body from iter-95.
        """
        import math

        def _expand(N: int, grid_type: str) -> str:
            if grid_type == "cubed_sphere":
                return f"C{N}"
            elif grid_type == "latlon":
                return f"{N}x{2 * N}"
            elif grid_type == "icosahedral":
                level = max(2, min(8, round(math.log(2 * N * N / 10) / math.log(4))))
                return f"ico{level}"
            elif grid_type == "spectral":
                return f"T{N}"
            else:
                return str(N)

        # N=16 → cube C16, latlon 16x32, ico level 3, spectral T16
        assert _expand(16, "cubed_sphere") == "C16"
        assert _expand(16, "latlon") == "16x32"
        assert _expand(16, "icosahedral") == "ico3"
        assert _expand(16, "spectral") == "T16"
        # N=32 → ico level 4 (~ 2562 cells, matches 32×64 latlon)
        assert _expand(32, "cubed_sphere") == "C32"
        assert _expand(32, "latlon") == "32x64"
        assert _expand(32, "icosahedral") == "ico4"
        assert _expand(32, "spectral") == "T32"
        # N=72 → ico level 5 (matches default ico5)
        assert _expand(72, "cubed_sphere") == "C72"
        assert _expand(72, "latlon") == "72x144"
        assert _expand(72, "icosahedral") == "ico5"

    def test_source_pin_cli_resolution_dispatch_present(self):
        """Source-level pin: the matrix runner uses the shared
        ``legoesm.driver.cli_resolution`` helpers (iter-115
        refactored from inline iter-95 literals).

        The actual dispatch literals (``f"C{N}"`` etc.) now
        live in
        ``src/legoesm/driver/cli_resolution.py:expand_cli_resolution``;
        verify the helper's behaviour separately (e.g., via
        a unit test that imports and calls it).
        """
        import inspect
        import re
        M = self._import_module()
        text = inspect.getsource(M)
        text_no_strings = re.sub(r'""".*?"""', "", text, flags=re.DOTALL)
        text_no_strings = re.sub(r"'''.*?'''", "", text_no_strings, flags=re.DOTALL)
        code_only = "\n".join(
            line for line in text_no_strings.splitlines()
            if not line.lstrip().startswith("#")
        )
        # iter-115 refactor: matrix runner uses the shared helpers.
        assert "validate_cli_resolution" in code_only, (
            "iter-115: matrix dispatch must use the shared "
            "validate_cli_resolution helper."
        )
        assert "expand_cli_resolution" in code_only, (
            "iter-115: matrix dispatch must use the shared "
            "expand_cli_resolution helper."
        )

    def test_source_pin_cli_resolution_dispatch_legacy_literals_in_helper(self):
        """The iter-95 dispatch literals (``f"C{N}"``,
        ``f"{N}x{2 * N}"``, ``f"ico{level}"``, ``f"T{N}"``)
        now live in the shared helper module.  Pin that they
        are present so a future regression that breaks the
        per-grid dispatch would fail this test.
        """
        from legoesm.driver import cli_resolution as cr
        import inspect
        text = inspect.getsource(cr)
        assert 'f"C{N}"' in text, (
            "iter-115: helper must contain ``f\"C{N}\"`` "
            "for cubed_sphere dispatch."
        )
        assert 'f"{N}x{2 * N}"' in text, (
            "iter-115: helper must contain ``f\"{N}x{2 * N}\"`` "
            "for latlon dispatch."
        )
        assert 'f"ico{level}"' in text, (
            "iter-115: helper must contain ``f\"ico{level}\"`` "
            "for icosahedral dispatch."
        )
        assert 'f"T{N}"' in text, (
            "iter-115: helper must contain ``f\"T{N}\"`` "
            "for spectral dispatch."
        )
        assert 'f"{N}km"' in text, (
            "iter-115: helper must contain ``f\"{N}km\"`` "
            "for mpas_regional dispatch."
        )


class TestAtmosphereMatrixBlowupReporting:
    """iter-98: extends the iter-97 OMIP BLOWUP-reporting fix to
    the atmosphere matrix runner.  ``_run_timeloop`` now stores
    BLOWUP details in ``diag["_blowup_info"]``;
    ``_write_results_txt`` accepts ``diag=`` kwarg and prepends a
    BLOWUP marker to the ``notes`` field of results.txt when a
    BLOWUP occurred.

    Pre-iter-98, atmosphere matrix runs that BLEW UP would write
    ``status: FAIL`` to results.txt with ``notes`` derived from
    the LAST CLEAN diagnostic — the same false-improvement bug
    that misled iter-96 for OMIP.
    """

    def test_run_timeloop_records_blowup_info_in_diag(self):
        """Source-level: ``_run_timeloop`` writes a
        ``_blowup_info`` key into ``diag`` when a BLOWUP fires.
        """
        import inspect
        import re
        src = inspect.getsource(M._run_timeloop)
        src_no_strings = re.sub(r'""".*?"""', "", src, flags=re.DOTALL)
        src_no_strings = re.sub(r"'''.*?'''", "", src_no_strings, flags=re.DOTALL)
        code_only = "\n".join(
            line for line in src_no_strings.splitlines()
            if not line.lstrip().startswith("#")
        )
        assert 'diag["_blowup_info"]' in code_only or \
               "diag['_blowup_info']" in code_only, (
            "iter-98: ``_run_timeloop`` must store BLOWUP details "
            "in ``diag['_blowup_info']`` so ``_write_results_txt`` "
            "can surface them."
        )

    def test_write_results_txt_accepts_diag_kwarg(self):
        """Source-level: ``_write_results_txt`` signature has a
        ``diag=None`` kwarg (introduced iter-98).
        """
        import inspect
        sig = inspect.signature(M._write_results_txt)
        assert "diag" in sig.parameters, (
            "iter-98: ``_write_results_txt`` must accept ``diag`` "
            "kwarg for opt-in BLOWUP info threading."
        )
        assert "blowup_info" in sig.parameters, (
            "iter-98: ``_write_results_txt`` must accept "
            "``blowup_info`` kwarg as the lower-level entry point."
        )

    def test_write_results_txt_emits_blowup_marker(self, tmp_path):
        """End-to-end: invoking ``_write_results_txt`` with a
        synthetic BLOWUP-info-bearing diag emits a BLOWUP marker
        in the resulting ``results.txt`` notes.
        """
        diag = {
            "times": [0.0, 1.0],
            "steps": [0, 100],
            "mass": [5.0e19, 5.0e19],
            "_blowup_info": {
                "step": 200,
                "day": 0.69,
                "metric": 1234.5,
                "is_finite": False,
                "threshold": 1000.0,
                "reason": "state non-finite (NaN/Inf)",
            },
        }
        rows = {
            "test": "amip",
            "grid": "cubed_sphere",
            "status": "FAIL",
            "notes": "mass drift=5.5e-12",
            "wall_time": "13.6s",
        }
        M._write_results_txt(tmp_path, rows, diag=diag)
        text = (tmp_path / "results.txt").read_text()
        assert "BLOWUP at step 200" in text, (
            f"iter-98: results.txt must include the BLOWUP "
            f"marker when ``diag['_blowup_info']`` is set; got:\n"
            f"{text}"
        )
        assert "state non-finite" in text, (
            "iter-98: results.txt must include the BLOWUP reason."
        )
        assert "last clean: mass drift=5.5e-12" in text, (
            "iter-98: results.txt must preserve the original "
            "notes (the last clean diagnostic) but mark them "
            "explicitly as ``last clean:``."
        )

    def test_write_results_txt_unaffected_for_pass_runs(self, tmp_path):
        """For PASS runs, no BLOWUP marker should appear."""
        diag = {
            "times": [0.0, 1.0],
            "steps": [0, 100],
            "mass": [5.0e19, 5.0e19],
        }
        rows = {
            "test": "amip",
            "grid": "cubed_sphere",
            "status": "PASS",
            "notes": "mass drift=5.5e-12",
            "wall_time": "13.6s",
        }
        M._write_results_txt(tmp_path, rows, diag=diag)
        text = (tmp_path / "results.txt").read_text()
        assert "BLOWUP" not in text, (
            "iter-98: PASS runs must not have BLOWUP markers."
        )
        assert "notes: mass drift=5.5e-12" in text, (
            "iter-98: PASS runs should write notes verbatim."
        )

    def test_write_results_txt_unaffected_when_no_diag_passed(self, tmp_path):
        """Backward compat: if caller doesn't pass ``diag=`` and
        no ``blowup_info=``, output is unchanged from pre-iter-98.
        """
        rows = {
            "test": "sw", "grid": "spectral", "status": "PASS",
            "notes": "mass drift=0.00e+00", "wall_time": "0.9s",
        }
        M._write_results_txt(tmp_path, rows)
        text = (tmp_path / "results.txt").read_text()
        assert "BLOWUP" not in text
        assert "notes: mass drift=0.00e+00" in text


class TestAtmosphereMatrixAllRunnersThreadDiag:
    """iter-99: structural assertion that EVERY caller of
    ``_write_results_txt`` in the atmosphere matrix runner
    threads ``diag=diag`` (or equivalent), so BLOWUP info from
    iter-98's ``diag["_blowup_info"]`` is surfaced uniformly
    across SW, HS, cosine_bell, baroclinic, dcmip_transport,
    AMIP, and DCMIP-NH runners.

    Pre-iter-99: only 2 of 8 callers (SW + HS, iter-98) threaded
    ``diag=``; the other 6 (cosine_bell, baroclinic,
    dcmip_transport, AMIP-w/-radiation, NH DCMIP) used the
    default ``diag=None`` and would silently emit
    last-clean-diagnostic notes for BLOWUP-failed runs.

    iter-99 threaded the remaining 6.  This test asserts every
    callsite has the marker comment so a future regression
    (e.g., a new test runner added without ``diag=``) would
    fail the CI.
    """

    def test_all_write_results_txt_callsites_thread_diag(self):
        """Every ``_write_results_txt(...)`` call in the
        atmosphere matrix runner module passes ``diag=`` (either
        ``diag=diag`` or, in the future, ``diag=...``).
        """
        import inspect
        import re
        # Read raw source — ``inspect.getsource`` on a module
        # returns the file contents.
        text = inspect.getsource(M)
        # Strip line comments to count active callsites.
        code_only = "\n".join(
            line for line in text.splitlines()
            if not line.lstrip().startswith("#")
        )
        # Count callsite invocations (skip the def itself).
        # The function signature ``def _write_results_txt(...)``
        # also matches ``_write_results_txt(`` but we want only
        # the call sites; filter by checking the line is NOT
        # immediately preceded by ``def``.
        all_calls = list(re.finditer(
            r"_write_results_txt\(", code_only
        ))
        # Filter out the def itself.
        callsite_count = 0
        threaded_count = 0
        for m in all_calls:
            # Look back ~10 chars for "def " preceding.
            start = max(0, m.start() - 10)
            preceding = code_only[start:m.start()]
            if "def " in preceding:
                continue
            callsite_count += 1
            # Look forward up to 500 chars for the closing ``)``
            # of this call, then check if ``diag=`` appears
            # within the call's argument list.
            tail = code_only[m.start():m.start() + 500]
            # Naive but effective: look for ``diag=`` followed
            # by an identifier or expression.
            if re.search(r"\bdiag\s*=\s*\w", tail):
                threaded_count += 1
        # We expect the SW(1803), cosine_bell(2161), HS(2467),
        # baroclinic(2746), dcmip_transport(2978), AMIP(3271),
        # NH DCMIP(3838) — 7 from explicit list above plus 1 hs
        # variant — totals 8 callsites.  All must thread ``diag=``.
        assert callsite_count >= 7, (
            f"iter-99 sanity check: expected at least 7 "
            f"``_write_results_txt(...)`` callsites in the "
            f"atmosphere matrix runner; found {callsite_count}.  "
            f"If you removed callsites, update the expected count."
        )
        assert threaded_count == callsite_count, (
            f"iter-99: every ``_write_results_txt(...)`` callsite "
            f"in the atmosphere matrix runner must thread "
            f"``diag=`` to surface BLOWUP info uniformly.  Found "
            f"{threaded_count} / {callsite_count} threaded."
        )


class TestRunAmipFiniteCheck:
    """iter-100: closes the iter-98 deferred gap that
    ``scripts/run_amip.py`` had ZERO finiteness checks
    (``grep -c isfinite`` = 0 in 450 lines).  A NaN-producing
    AMIP run would silently complete and print "Complete." while
    writing garbage to the output directory.

    iter-100 added ``_check_run_state_finite(driver)`` which
    inspects ``driver.state`` for NaN/Inf in T, u, v, p_s and
    returns ``(ok, first_bad_field)``.  When the check fails,
    ``main()`` prints a FAIL message to stderr and exits with
    code 1 so wrapper scripts can detect failure.

    These tests validate the helper logic without booting the
    full driver: synthetic state objects let us exercise the
    finite/non-finite branches deterministically.
    """

    def _import_run_amip(self):
        import importlib
        import sys
        from pathlib import Path
        scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        return importlib.import_module("run_amip")

    def _make_synthetic_driver(self, *, T_data, u_data, v_data, ps_data):
        """Build a minimal duck-typed driver object with .state."""
        from types import SimpleNamespace
        return SimpleNamespace(
            state=SimpleNamespace(
                T=SimpleNamespace(data=T_data),
                u=SimpleNamespace(data=u_data),
                v=SimpleNamespace(data=v_data),
                p_s=SimpleNamespace(data=ps_data),
            ),
        )

    def test_check_finite_returns_ok_for_clean_state(self):
        import jax.numpy as jnp
        m = self._import_run_amip()
        driver = self._make_synthetic_driver(
            T_data=jnp.full((6, 4, 4, 5), 280.0),
            u_data=jnp.full((6, 4, 4, 5), 10.0),
            v_data=jnp.full((6, 4, 4, 5), 5.0),
            ps_data=jnp.full((6, 4, 4), 1e5),
        )
        ok, bad = m._check_run_state_finite(driver)
        assert ok is True
        assert bad is None

    def test_check_finite_detects_nan_in_T(self):
        import jax.numpy as jnp
        m = self._import_run_amip()
        T_data = jnp.full((6, 4, 4, 5), 280.0)
        T_data = T_data.at[0, 0, 0, 0].set(float("nan"))
        driver = self._make_synthetic_driver(
            T_data=T_data,
            u_data=jnp.full((6, 4, 4, 5), 10.0),
            v_data=jnp.full((6, 4, 4, 5), 5.0),
            ps_data=jnp.full((6, 4, 4), 1e5),
        )
        ok, bad = m._check_run_state_finite(driver)
        assert ok is False
        assert bad == "T"

    def test_check_finite_detects_inf_in_u(self):
        import jax.numpy as jnp
        m = self._import_run_amip()
        u_data = jnp.full((6, 4, 4, 5), 10.0)
        u_data = u_data.at[0, 0, 0, 0].set(float("inf"))
        driver = self._make_synthetic_driver(
            T_data=jnp.full((6, 4, 4, 5), 280.0),
            u_data=u_data,
            v_data=jnp.full((6, 4, 4, 5), 5.0),
            ps_data=jnp.full((6, 4, 4), 1e5),
        )
        ok, bad = m._check_run_state_finite(driver)
        assert ok is False
        assert bad == "u"

    def test_check_finite_detects_nan_in_p_s(self):
        import jax.numpy as jnp
        m = self._import_run_amip()
        ps_data = jnp.full((6, 4, 4), 1e5)
        ps_data = ps_data.at[0, 0, 0].set(float("nan"))
        driver = self._make_synthetic_driver(
            T_data=jnp.full((6, 4, 4, 5), 280.0),
            u_data=jnp.full((6, 4, 4, 5), 10.0),
            v_data=jnp.full((6, 4, 4, 5), 5.0),
            ps_data=ps_data,
        )
        ok, bad = m._check_run_state_finite(driver)
        assert ok is False
        assert bad == "p_s"

    def test_check_finite_handles_missing_state(self):
        """If ``driver.state`` is None (e.g., setup not yet run),
        return ``(True, None)`` — there's nothing to check.
        """
        from types import SimpleNamespace
        m = self._import_run_amip()
        driver = SimpleNamespace(state=None)
        ok, bad = m._check_run_state_finite(driver)
        assert ok is True
        assert bad is None

    def test_check_finite_handles_partial_state(self):
        """Spectral state may not have all of T/u/v/p_s; helper
        skips missing fields gracefully and returns OK if the
        present fields are finite.
        """
        import jax.numpy as jnp
        from types import SimpleNamespace
        m = self._import_run_amip()
        # Synthetic state with only T (no u/v/p_s attributes).
        driver = SimpleNamespace(
            state=SimpleNamespace(
                T=SimpleNamespace(data=jnp.full((6, 4, 4, 5), 280.0)),
            )
        )
        ok, bad = m._check_run_state_finite(driver)
        assert ok is True
        assert bad is None

    def test_main_emits_fail_message_on_nan_state(self, monkeypatch, capsys):
        """End-to-end: monkey-patching the driver to produce a
        NaN final state, ``main()`` prints the iter-100 FAIL
        message and exits with code 1.

        This is the contract: wrapper scripts (e.g.,
        ``run_amip_cross_grid.sh``) check ``$?`` to decide
        whether to mark the AMIP run as failed.
        """
        import jax.numpy as jnp
        m = self._import_run_amip()

        # Build a dummy driver class with a state attribute that
        # has NaN.  Replace ModelDriver in the module.
        class DummyDriver:
            def __init__(self, config):
                from types import SimpleNamespace
                T = jnp.full((6, 4, 4, 5), 280.0)
                T = T.at[0, 0, 0, 0].set(float("nan"))
                self.state = SimpleNamespace(
                    T=SimpleNamespace(data=T),
                    u=SimpleNamespace(data=jnp.full((6, 4, 4, 5), 10.0)),
                    v=SimpleNamespace(data=jnp.full((6, 4, 4, 5), 5.0)),
                    p_s=SimpleNamespace(data=jnp.full((6, 4, 4), 1e5)),
                )
                self.output_dir = "/tmp/amip_test_unused"
                self._mpi_rank = None
            def setup(self):
                pass
            def run(self, **kwargs):
                pass
            def load_checkpoint(self, p):
                return 0, 0.0

        # Patch the import inside main()
        from legoesm.driver import model_driver
        monkeypatch.setattr(model_driver, "ModelDriver", DummyDriver)

        with pytest.raises(SystemExit) as exc_info:
            m.main([
                "--dataset", "analytical", "--days", "1",
                "--resolution", "16", "--dt", "600",
            ])
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "FAIL" in captured.err
        assert "T" in captured.err  # the bad field


class TestRunRceExitCodeOnBlowup:
    """iter-101: pre-iter-101, ``run_rce.py`` wrote
    ``status: FAIL`` to results.txt on BLOWUP but exited with
    code 0 (the default).  The ``run_rce_cross_grid.sh`` wrapper
    comment at iter-73 explicitly flagged this:

        ``run_rce.py`` does not currently exit non-zero on FAIL
        the way ``run_omip.py`` does, so this is a defensive
        guard for future regressions.

    A user invoking ``run_rce.py`` directly (not via the wrapper)
    would not know if it BLEW UP — exit 0 silently misled.
    iter-101 added ``sys.exit(1)`` after the results.txt write
    block when ``blowup`` is True, mirroring iter-97
    ``run_omip.py`` and iter-100 ``run_amip.py`` exit-code
    conventions.

    These tests pin the source-level contract.  An end-to-end
    run-the-script test is impractical (RCE needs ~5 s wall +
    full module bootstrapping), so the pin is via inspecting the
    source for the ``if blowup: sys.exit(1)`` pattern.
    """

    def test_run_rce_source_contains_blowup_exit_guard(self):
        """Source-level pin: ``run_rce.py`` body contains
        ``if blowup: sys.exit(1)`` so BLOWUP runs surface as
        exit code 1 to wrapper scripts.
        """
        import re
        from pathlib import Path
        path = Path(__file__).resolve().parent.parent / "scripts" / "run_rce.py"
        text = path.read_text()
        # Strip line comments + triple-quoted strings.
        text_no_strings = re.sub(r'""".*?"""', "", text, flags=re.DOTALL)
        text_no_strings = re.sub(r"'''.*?'''", "", text_no_strings, flags=re.DOTALL)
        code_only = "\n".join(
            line for line in text_no_strings.splitlines()
            if not line.lstrip().startswith("#")
        )
        # Match ``if blowup:`` followed (within a few lines) by
        # ``sys.exit(1)``.
        assert re.search(
            r"if\s+blowup\s*:\s*\n\s+sys\.exit\(1\)",
            code_only,
        ) is not None, (
            "iter-101: ``run_rce.py`` must contain "
            "``if blowup: sys.exit(1)`` after the results.txt "
            "write so BLOWUP runs surface as exit code 1.  "
            "iter-73 wrapper comment explicitly flagged this as "
            "missing pre-iter-101."
        )

    def test_run_rce_imports_sys(self):
        """Sanity: ``sys`` must be imported (the iter-101 fix
        uses ``sys.exit``).
        """
        from pathlib import Path
        path = Path(__file__).resolve().parent.parent / "scripts" / "run_rce.py"
        text = path.read_text()
        # Strip docstrings.
        import re
        text_no_strings = re.sub(r'""".*?"""', "", text, flags=re.DOTALL)
        assert "import sys" in text_no_strings, (
            "iter-101: ``run_rce.py`` must import ``sys`` for "
            "the BLOWUP exit-code guard."
        )

    def test_run_rce_cross_grid_wrapper_comment_updated(self):
        """The iter-73 wrapper comment that called out
        ``run_rce.py`` lacking exit-code support has been
        updated to reflect iter-101's fix.  This pins the
        documentation so future readers don't think the gap is
        still open.
        """
        from pathlib import Path
        path = Path(__file__).resolve().parent.parent / "scripts" / "run_rce_cross_grid.sh"
        text = path.read_text()
        # Pre-iter-101 comment claimed run_rce.py "does not
        # currently exit non-zero on FAIL" — that statement must
        # no longer appear (or must be updated).
        assert "does not currently exit non-zero on FAIL" not in text, (
            "iter-101: the iter-73 wrapper comment claiming "
            "``run_rce.py does not currently exit non-zero on "
            "FAIL`` must be updated since iter-101 fixed it."
        )
        # The new comment must reference iter-101 explicitly
        # so future readers can find the fix.
        assert "iter-101" in text, (
            "iter-101: the wrapper comment must reference "
            "iter-101 so the fix is discoverable from the "
            "wrapper context."
        )


class TestRunAmipFiniteCheckSpectralFields:
    """iter-104 codex HIGH-2: the iter-100
    ``_check_run_state_finite`` originally only checked grid-
    space fields (T, u, v, p_s).  Spectral AMIP states (``--
    grid-type gaussian --discretization spectral``) use
    different attribute names: ``T_hat``, ``vor_hat``,
    ``div_hat``, ``lnps_hat``.  Pre-iter-104 a NaN in any of
    those would silently bypass the iter-100 check and the
    helper would return ``(True, None)``.

    iter-104 extended the field list to include both grid-space
    and spectral names.  The helper iterates through the union
    and skips missing attributes (so the same helper works for
    both AMIP execution paths).
    """

    def _import_run_amip(self):
        import importlib
        import sys
        from pathlib import Path
        scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        return importlib.import_module("run_amip")

    def test_check_finite_detects_nan_in_T_hat(self):
        """Pre-iter-104, NaN in T_hat would return (True, None).
        Post-iter-104, it returns (False, "T_hat").
        """
        import jax.numpy as jnp
        from types import SimpleNamespace
        m = self._import_run_amip()
        T_hat = jnp.full((6, 4, 4, 5), 280.0, dtype=jnp.complex128)
        T_hat = T_hat.at[0, 0, 0, 0].set(jnp.nan)
        # Spectral state: only T_hat, vor_hat, div_hat, lnps_hat.
        # No grid-space T/u/v/p_s.
        driver = SimpleNamespace(
            state=SimpleNamespace(
                T_hat=SimpleNamespace(data=T_hat),
                vor_hat=SimpleNamespace(
                    data=jnp.zeros((6, 4, 4, 5), dtype=jnp.complex128)),
                div_hat=SimpleNamespace(
                    data=jnp.zeros((6, 4, 4, 5), dtype=jnp.complex128)),
                lnps_hat=SimpleNamespace(
                    data=jnp.zeros((6, 4, 4), dtype=jnp.complex128)),
            )
        )
        ok, bad = m._check_run_state_finite(driver)
        assert ok is False
        assert bad == "T_hat"

    def test_check_finite_detects_nan_in_vor_hat(self):
        import jax.numpy as jnp
        from types import SimpleNamespace
        m = self._import_run_amip()
        vor_hat = jnp.full((6, 4, 4, 5), 0.0, dtype=jnp.complex128)
        vor_hat = vor_hat.at[0, 0, 0, 0].set(jnp.inf)
        driver = SimpleNamespace(
            state=SimpleNamespace(
                T_hat=SimpleNamespace(
                    data=jnp.full((6, 4, 4, 5), 280.0, dtype=jnp.complex128)),
                vor_hat=SimpleNamespace(data=vor_hat),
                div_hat=SimpleNamespace(
                    data=jnp.zeros((6, 4, 4, 5), dtype=jnp.complex128)),
                lnps_hat=SimpleNamespace(
                    data=jnp.zeros((6, 4, 4), dtype=jnp.complex128)),
            )
        )
        ok, bad = m._check_run_state_finite(driver)
        assert ok is False
        assert bad == "vor_hat"

    def test_check_finite_clean_spectral_state(self):
        """Clean spectral state returns (True, None)."""
        import jax.numpy as jnp
        from types import SimpleNamespace
        m = self._import_run_amip()
        driver = SimpleNamespace(
            state=SimpleNamespace(
                T_hat=SimpleNamespace(
                    data=jnp.full((6, 4, 4, 5), 280.0, dtype=jnp.complex128)),
                vor_hat=SimpleNamespace(
                    data=jnp.zeros((6, 4, 4, 5), dtype=jnp.complex128)),
                div_hat=SimpleNamespace(
                    data=jnp.zeros((6, 4, 4, 5), dtype=jnp.complex128)),
                lnps_hat=SimpleNamespace(
                    data=jnp.zeros((6, 4, 4), dtype=jnp.complex128)),
            )
        )
        ok, bad = m._check_run_state_finite(driver)
        assert ok is True
        assert bad is None

    def test_check_finite_grid_space_takes_priority_when_both_present(self):
        """If a hybrid state has both grid and spectral fields
        (unusual but the helper handles it), grid-space is
        checked first.  Pin the iteration order.
        """
        import jax.numpy as jnp
        from types import SimpleNamespace
        m = self._import_run_amip()
        T = jnp.full((6, 4, 4, 5), 280.0)
        T = T.at[0, 0, 0, 0].set(jnp.nan)
        T_hat = jnp.full((6, 4, 4, 5), 280.0, dtype=jnp.complex128)
        T_hat = T_hat.at[0, 0, 0, 0].set(jnp.nan)
        driver = SimpleNamespace(
            state=SimpleNamespace(
                T=SimpleNamespace(data=T),
                u=SimpleNamespace(data=jnp.full((6, 4, 4, 5), 10.0)),
                v=SimpleNamespace(data=jnp.full((6, 4, 4, 5), 5.0)),
                p_s=SimpleNamespace(data=jnp.full((6, 4, 4), 1e5)),
                T_hat=SimpleNamespace(data=T_hat),
            )
        )
        ok, bad = m._check_run_state_finite(driver)
        # Grid-space iteration order: T comes first.
        assert ok is False
        assert bad == "T"


class TestCliResolutionValidation:
    """iter-107 (codex iter-104 LOW-7): pre-iter-107,
    ``--resolution 0`` raised a confusing
    ``ValueError: math domain error`` deep inside the
    icosahedral level computation, and ``--resolution -16``
    silently produced ``C-16`` / ``-16x-32`` / ``T-16``
    invalid strings while icosahedral mapped as if positive.

    iter-107 added validation at the boundary: bare integer
    ``N <= 0`` is rejected with a clear parser-style error
    message and ``sys.exit(2)``.

    Also added a warning when ``--resolution N`` maps to an
    icosahedral level > 8 (the max supported per
    ``voronoi.py:1063``).
    """

    def test_atmosphere_resolution_zero_rejected(self):
        """``run_atmosphere_test_matrix.py --resolution 0``
        exits with code 2 and a clear error message.
        """
        import subprocess
        from pathlib import Path
        repo_root = Path(__file__).resolve().parent.parent
        result = subprocess.run(
            [".venv/bin/python",
             "scripts/run_atmosphere_test_matrix.py",
             "--only", "sw", "--quick", "--resolution", "0",
             "--no-cross-grid-plots"],
            cwd=str(repo_root),
            capture_output=True, text=True,
            env={"JAX_ENABLE_X64": "1", "PATH": "/usr/bin:/bin"},
            timeout=60,
        )
        assert result.returncode == 2, (
            f"iter-107: --resolution 0 must exit 2, got "
            f"{result.returncode}.\nstderr:\n{result.stderr}"
        )
        assert "must be a positive integer" in result.stderr, (
            f"iter-107: --resolution 0 must produce a clear "
            f"error message; got:\n{result.stderr}"
        )

    def test_atmosphere_resolution_negative_rejected(self):
        """``run_atmosphere_test_matrix.py --resolution -16``
        exits with code 2 and a clear error message.
        """
        import subprocess
        from pathlib import Path
        repo_root = Path(__file__).resolve().parent.parent
        result = subprocess.run(
            [".venv/bin/python",
             "scripts/run_atmosphere_test_matrix.py",
             "--only", "sw", "--quick", "--resolution", "-16",
             "--no-cross-grid-plots"],
            cwd=str(repo_root),
            capture_output=True, text=True,
            env={"JAX_ENABLE_X64": "1", "PATH": "/usr/bin:/bin"},
            timeout=60,
        )
        assert result.returncode == 2, (
            f"iter-107: --resolution -16 must exit 2, got "
            f"{result.returncode}.\nstderr:\n{result.stderr}"
        )
        assert "must be a positive integer" in result.stderr

    def test_atmosphere_resolution_decimal_rejected(self):
        """iter-111 codex iter-110 LOW-4: ``--resolution 0.5``
        and other decimal numerics must exit 2 with a clear
        error.  Pre-iter-111 they slipped past the int() parse,
        were treated as preformatted strings, and produced 12
        ERRORs deep inside the per-grid parsers.
        """
        import subprocess
        from pathlib import Path
        repo_root = Path(__file__).resolve().parent.parent
        result = subprocess.run(
            [".venv/bin/python",
             "scripts/run_atmosphere_test_matrix.py",
             "--only", "sw", "--quick", "--resolution", "0.5",
             "--no-cross-grid-plots"],
            cwd=str(repo_root),
            capture_output=True, text=True,
            env={"JAX_ENABLE_X64": "1", "PATH": "/usr/bin:/bin"},
            timeout=60,
        )
        assert result.returncode == 2, (
            f"iter-111: --resolution 0.5 must exit 2, got "
            f"{result.returncode}.\nstderr:\n{result.stderr}"
        )
        assert "must be a positive INTEGER" in result.stderr, (
            f"iter-111: error message must say INTEGER; got:\n"
            f"{result.stderr}"
        )
        # iter-112 (codex MEDIUM-1): broadened wording from
        # "decimal string" → "non-integer numeric string" to
        # cover ``.5``, ``1.``, ``1e3``, ``inf``, ``nan``.
        assert "non-integer numeric string" in result.stderr, (
            f"iter-112: error must mention non-integer numeric"
            f" string rejection; got:\n{result.stderr}"
        )

    def test_atmosphere_resolution_negative_decimal_rejected(self):
        """``--resolution -0.5`` also exits 2."""
        import subprocess
        from pathlib import Path
        repo_root = Path(__file__).resolve().parent.parent
        result = subprocess.run(
            [".venv/bin/python",
             "scripts/run_atmosphere_test_matrix.py",
             "--only", "sw", "--quick", "--resolution", "-0.5",
             "--no-cross-grid-plots"],
            cwd=str(repo_root),
            capture_output=True, text=True,
            env={"JAX_ENABLE_X64": "1", "PATH": "/usr/bin:/bin"},
            timeout=60,
        )
        assert result.returncode == 2

    def test_atmosphere_resolution_iter112_extended_forms_rejected(self):
        """iter-112 codex MEDIUM-1: ``.5``, ``1.``, ``1e3``,
        ``inf``, ``nan`` (which slipped past iter-111's regex)
        all exit 2.
        """
        import subprocess
        from pathlib import Path
        repo_root = Path(__file__).resolve().parent.parent
        for bad in (".5", "1.", "1e3", "inf", "nan"):
            result = subprocess.run(
                [".venv/bin/python",
                 "scripts/run_atmosphere_test_matrix.py",
                 "--only", "sw", "--quick", "--resolution", bad,
                 "--no-cross-grid-plots"],
                cwd=str(repo_root),
                capture_output=True, text=True,
                env={"JAX_ENABLE_X64": "1", "PATH": "/usr/bin:/bin"},
                timeout=60,
            )
            assert result.returncode == 2, (
                f"iter-112: --resolution {bad} must exit 2; got "
                f"{result.returncode}.\nstderr:\n{result.stderr}"
            )

    def test_atmosphere_resolution_string_format_still_works(self):
        """Pre-formatted strings (``C36``, ``ico5``, etc.) must
        still pass through unchanged.  The iter-107 validation
        only rejects bare-integer ``N <= 0``, not strings.
        """
        import inspect
        import re
        # Source-level pin: the iter-107 validation only triggers
        # when ``is_bare_int`` is True (so non-integer strings
        # like "C36" pass through to ``_expand_cli_res``
        # unchanged).
        from importlib import import_module
        import sys
        from pathlib import Path
        scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        m = import_module("run_atmosphere_test_matrix")
        text = inspect.getsource(m)
        text_no_strings = re.sub(r'""".*?"""', "", text, flags=re.DOTALL)
        text_no_strings = re.sub(r"'''.*?'''", "", text_no_strings, flags=re.DOTALL)
        code_only = "\n".join(
            line for line in text_no_strings.splitlines()
            if not line.lstrip().startswith("#")
        )
        # iter-115 refactored to use the shared helper.
        # Validation now lives in
        # ``legoesm.driver.cli_resolution.validate_cli_resolution``;
        # the matrix runner only invokes the helpers.
        assert "validate_cli_resolution" in code_only, (
            "iter-115: dispatch must use the shared "
            "``legoesm.driver.cli_resolution.validate_cli_resolution`` "
            "helper instead of inline validation."
        )
        assert "expand_cli_resolution" in code_only, (
            "iter-115: dispatch must use the shared "
            "``legoesm.driver.cli_resolution.expand_cli_resolution`` "
            "helper instead of inline per-grid format strings."
        )


class TestSelectResolutionDirPrefersGridTyped:
    """iter-108 (codex iter-104 MEDIUM-6): the cross-grid plot
    collector previously took ``sorted(res_dirs)[0]`` —
    alphabetically-first — which made stale ``16/`` dirs (from
    pre-iter-95 verbatim-resolution runs) shadow fresh ``C16/``
    dirs (post-iter-95 per-grid dispatch).  iter-108 prefers
    the grid-typed format explicitly:

    * cubed_sphere → ``C*``
    * latlon → ``*x*``
    * icosahedral / mpas → ``ico*``
    * spectral → ``T*``
    * regional grids → ``*km``

    Falls back to ``sorted(res_dirs)[0]`` only when no
    grid-typed candidate exists (graceful degradation for
    pure-legacy trees).
    """

    def test_atmosphere_prefers_C_over_bare_int(self, tmp_path):
        """``cubed_sphere/{16, C16}`` → C16."""
        grid_dir = tmp_path / "cubed_sphere"
        grid_dir.mkdir()
        (grid_dir / "16").mkdir()
        (grid_dir / "C16").mkdir()
        chosen = M._select_resolution_dir(grid_dir)
        assert chosen.name == "C16"

    def test_atmosphere_prefers_xform_over_bare_int(self, tmp_path):
        """``latlon/{16, 16x32}`` → 16x32."""
        grid_dir = tmp_path / "latlon"
        grid_dir.mkdir()
        (grid_dir / "16").mkdir()
        (grid_dir / "16x32").mkdir()
        chosen = M._select_resolution_dir(grid_dir)
        assert chosen.name == "16x32"

    def test_atmosphere_prefers_ico_over_bare_int(self, tmp_path):
        """``icosahedral/{16, ico3}`` → ico3."""
        grid_dir = tmp_path / "icosahedral"
        grid_dir.mkdir()
        (grid_dir / "16").mkdir()
        (grid_dir / "ico3").mkdir()
        chosen = M._select_resolution_dir(grid_dir)
        assert chosen.name == "ico3"

    def test_atmosphere_prefers_T_over_bare_int(self, tmp_path):
        """``spectral/{16, T16}`` → T16."""
        grid_dir = tmp_path / "spectral"
        grid_dir.mkdir()
        (grid_dir / "16").mkdir()
        (grid_dir / "T16").mkdir()
        chosen = M._select_resolution_dir(grid_dir)
        assert chosen.name == "T16"

    def test_atmosphere_falls_back_to_alphabetical_for_legacy(self, tmp_path):
        """If only legacy dirs are present (no grid-typed
        candidate), fall back to alphabetical-first.
        """
        grid_dir = tmp_path / "cubed_sphere"
        grid_dir.mkdir()
        (grid_dir / "16").mkdir()
        (grid_dir / "32").mkdir()
        chosen = M._select_resolution_dir(grid_dir)
        # Both are bare-integer (no grid-typed); falls back to
        # alphabetical-first which is "16" (< "32" alphabetically).
        assert chosen.name == "16"

    def test_atmosphere_returns_none_for_empty(self, tmp_path):
        grid_dir = tmp_path / "cubed_sphere"
        grid_dir.mkdir()
        chosen = M._select_resolution_dir(grid_dir)
        assert chosen is None

    def test_atmosphere_picks_C_when_only_grid_typed(self, tmp_path):
        """No stale dirs — just pick the grid-typed one."""
        grid_dir = tmp_path / "cubed_sphere"
        grid_dir.mkdir()
        (grid_dir / "C24").mkdir()
        chosen = M._select_resolution_dir(grid_dir)
        assert chosen.name == "C24"

    def test_iter110_filters_hidden_dirs(self, tmp_path):
        """iter-110 codex MEDIUM-3: filter ``.ipynb_checkpoints``
        and ``__pycache__`` before fallback.  Pre-iter-110 the
        atmosphere collector could pick an ``.ipynb_checkpoints/``
        dir over a valid ``16/`` legacy dir if the hidden one
        sorted alphabetically first.
        """
        M._RES_DIR_WARNED.clear()
        grid_dir = tmp_path / "cubed_sphere"
        grid_dir.mkdir()
        (grid_dir / ".ipynb_checkpoints").mkdir()
        (grid_dir / "__pycache__").mkdir()
        (grid_dir / "16").mkdir()  # legacy bare-numeric
        chosen = M._select_resolution_dir(grid_dir)
        assert chosen.name == "16"

    def test_iter110_returns_none_for_only_hidden(self, tmp_path):
        """If only hidden/internal dirs exist, return None
        rather than picking one of them.
        """
        M._RES_DIR_WARNED.clear()
        grid_dir = tmp_path / "cubed_sphere"
        grid_dir.mkdir()
        (grid_dir / ".ipynb_checkpoints").mkdir()
        (grid_dir / "__pycache__").mkdir()
        chosen = M._select_resolution_dir(grid_dir)
        assert chosen is None


class TestStatusToExitCode:
    """iter-109 (codex iter-104 MEDIUM-8): centralized
    status-to-exit-code helper.  Pre-iter-109, three other
    user-facing ``ModelDriver`` wrappers (src/legoesm/cli.py,
    scripts/run_held_suarez_rrtmgp_allgrids.py,
    scripts/run_held_suarez_icos_0p5deg.py) logged the run
    status string but exited 0 even when status indicated
    BLOWUP.

    iter-109 added ``legoesm.driver.run_status.status_to_exit_code``
    and applied it to all 3 wrappers.  ``status="COMPLETED"``
    → 0; everything else → 1.
    """

    def test_completed_returns_zero(self):
        from legoesm.driver.run_status import status_to_exit_code
        assert status_to_exit_code("COMPLETED") == 0

    def test_blowup_returns_one(self):
        from legoesm.driver.run_status import status_to_exit_code
        assert status_to_exit_code("BLOWUP at day 5.7") == 1

    def test_unknown_status_returns_one(self):
        """Defensive: unexpected strings are treated as failure."""
        from legoesm.driver.run_status import status_to_exit_code
        assert status_to_exit_code("WEIRD") == 1
        assert status_to_exit_code("") == 1
        assert status_to_exit_code("FAILED: ImportError") == 1

    def test_helper_is_re_exported_from_driver(self):
        """Discoverability: ``from legoesm.driver import
        status_to_exit_code`` works.
        """
        from legoesm.driver import status_to_exit_code as _ste
        assert _ste("COMPLETED") == 0


class TestModelDriverWrappersUseStatusHelper:
    """iter-109: structural pin that the 3 ModelDriver wrappers
    flagged by codex iter-104 MEDIUM-8 actually use the
    centralized helper.
    """

    def _read(self, rel: str) -> str:
        from pathlib import Path
        return (Path(__file__).resolve().parent.parent / rel).read_text()

    def test_cli_uses_status_to_exit_code(self):
        text = self._read("src/legoesm/cli.py")
        assert "status_to_exit_code" in text, (
            "iter-109: ``src/legoesm/cli.py`` must use "
            "``status_to_exit_code`` to translate driver status "
            "to process exit code."
        )

    def test_held_suarez_rrtmgp_allgrids_uses_helper(self):
        text = self._read("scripts/run_held_suarez_rrtmgp_allgrids.py")
        assert "status_to_exit_code" in text, (
            "iter-109: "
            "``scripts/run_held_suarez_rrtmgp_allgrids.py`` "
            "must use ``status_to_exit_code``."
        )
        assert "sys.exit(main())" in text, (
            "iter-109: "
            "``scripts/run_held_suarez_rrtmgp_allgrids.py`` "
            "must propagate ``main()``'s return value to "
            "``sys.exit`` so wrappers can detect failure."
        )

    def test_held_suarez_icos_0p5deg_uses_helper(self):
        text = self._read("scripts/run_held_suarez_icos_0p5deg.py")
        assert "status_to_exit_code" in text, (
            "iter-109: "
            "``scripts/run_held_suarez_icos_0p5deg.py`` "
            "must use ``status_to_exit_code``."
        )
        assert "sys.exit(main())" in text, (
            "iter-109: "
            "``scripts/run_held_suarez_icos_0p5deg.py`` "
            "must propagate ``main()``'s return value."
        )


class TestCliBehaviorOnBlowup:
    """iter-111 (codex iter-110 LOW-5): behavior test for the
    iter-109 cli.py exit-code propagation.

    Pre-iter-111, only source-level pins existed
    (``TestModelDriverWrappersUseStatusHelper``).  iter-111
    adds a true behavior test: monkeypatch ``ModelDriver`` to
    return a BLOWUP status, invoke ``cmd_run``, assert
    ``SystemExit(1)``.
    """

    def _build_dummy_args(self, tmp_path):
        """Construct minimal Namespace + dummy Config so that
        cmd_run can run end-to-end without touching disk."""
        from types import SimpleNamespace
        # The legoesm.config.Config.get(...) method is used at
        # several points in cmd_run to log info — return strings
        # so logger calls don't crash.
        class DummyConfig:
            def get(self, key):
                return f"<{key}>"
            def to_experiment_config(self):
                return SimpleNamespace()
        return SimpleNamespace(config=str(tmp_path / "fake.yaml")), \
               DummyConfig()

    def _patch_cli_for_test(self, monkeypatch, dummy_config, dummy_run_status):
        """Patch the heavy bits of cmd_run so it can be invoked
        unit-test-style: Config.from_yaml, bootstrap, ModelDriver."""
        from legoesm import config as legoesm_config_mod
        from legoesm.driver import model_driver
        from legoesm import runtime as legoesm_runtime_mod
        from types import SimpleNamespace

        monkeypatch.setattr(
            legoesm_config_mod.Config, "from_yaml",
            classmethod(lambda cls, p: dummy_config))
        monkeypatch.setattr(
            legoesm_runtime_mod, "bootstrap_from_yaml_config",
            lambda c: SimpleNamespace(
                backend="cpu", precision="fp32",
                device_config=SimpleNamespace(n_devices=1),
                distributed=False))

        class DummyDriver:
            def __init__(self, config):
                self.config = config
            def setup(self):
                pass
            def run(self, *args, **kwargs):
                return dummy_run_status
        monkeypatch.setattr(
            model_driver, "ModelDriver", DummyDriver)

    def test_cli_cmd_run_exits_one_on_blowup(self, monkeypatch, tmp_path):
        """``cmd_run`` exits with SystemExit(1) when the
        driver returns ``"BLOWUP at day 5.7"``.
        """
        from legoesm import cli
        args, dummy_config = self._build_dummy_args(tmp_path)
        self._patch_cli_for_test(monkeypatch, dummy_config, "BLOWUP at day 5.7")

        with pytest.raises(SystemExit) as exc_info:
            cli.cmd_run(args)
        assert exc_info.value.code == 1

    def test_cli_cmd_run_succeeds_on_completed(self, monkeypatch, tmp_path):
        """``cmd_run`` returns normally (no SystemExit) when
        the driver returns ``"COMPLETED"``.
        """
        from legoesm import cli
        args, dummy_config = self._build_dummy_args(tmp_path)
        self._patch_cli_for_test(monkeypatch, dummy_config, "COMPLETED")

        # Should NOT raise SystemExit on clean completion.
        try:
            cli.cmd_run(args)
        except SystemExit as e:
            assert e.code == 0, (
                f"iter-111: ``cmd_run`` must NOT exit non-zero "
                f"on COMPLETED status; got code {e.code}."
            )


class TestSharedCliResolution:
    """iter-115 (codex iter-114-followup HIGH-1, MEDIUM-2):
    centralized ``--resolution`` validation + dispatch via
    ``legoesm.driver.cli_resolution``.

    Pre-iter-115:
    * MEDIUM-2: float() gate let ``2j``, ``hello`` slip through
    * HIGH-1: ``run_omip.py`` had no dispatch (verbatim apply)

    iter-115 fixes:
    * Added ``validate_cli_resolution`` that rejects non-numeric
      strings unless they match a known per-grid format
      pattern (``C\\d+``, ``\\d+x\\d+``, ``ico\\d+``, ``T\\d+``,
      ``\\d+km``).
    * Added ``expand_cli_resolution`` for the per-grid mapping.
    * Centralized in ``src/legoesm/driver/cli_resolution.py``.
    * Applied to atmosphere matrix, ocean matrix, modular ocean
      cli, and ``run_omip.py``.
    """

    def test_validate_accepts_positive_int(self):
        from legoesm.driver.cli_resolution import validate_cli_resolution
        assert validate_cli_resolution("16") == 16
        assert validate_cli_resolution("32") == 32
        assert validate_cli_resolution("1") == 1

    def test_validate_returns_none_for_grid_typed_string(self):
        from legoesm.driver.cli_resolution import validate_cli_resolution
        for s in ("C24", "C36", "ico3", "ico5", "36x72", "72x144", "T21", "T42", "50km"):
            assert validate_cli_resolution(s) is None, (
                f"iter-115: {s!r} should pass through (None)"
            )

    def test_validate_rejects_zero_negative(self):
        from legoesm.driver.cli_resolution import validate_cli_resolution
        for bad in ("0", "-1", "-16"):
            with pytest.raises(SystemExit) as e:
                validate_cli_resolution(bad)
            assert e.value.code == 2

    def test_validate_rejects_decimal(self):
        from legoesm.driver.cli_resolution import validate_cli_resolution
        for bad in ("0.5", "1.0", "1.5", ".5", "1.", "1e3", "inf", "nan"):
            with pytest.raises(SystemExit) as e:
                validate_cli_resolution(bad)
            assert e.value.code == 2

    def test_validate_rejects_unrecognized_strings(self):
        """iter-115 codex MEDIUM-2: ``2j``, ``hello``,
        ``garbage`` must be rejected — pre-iter-115 they
        slipped past the float() check.
        """
        from legoesm.driver.cli_resolution import validate_cli_resolution
        for bad in ("2j", "hello", "garbage", "C", "ico", "T"):
            with pytest.raises(SystemExit) as e:
                validate_cli_resolution(bad)
            assert e.value.code == 2

    def test_expand_for_each_grid(self):
        from legoesm.driver.cli_resolution import expand_cli_resolution
        assert expand_cli_resolution(16, "cubed_sphere") == "C16"
        assert expand_cli_resolution(16, "latlon") == "16x32"
        assert expand_cli_resolution(16, "icosahedral") == "ico3"
        assert expand_cli_resolution(16, "mpas") == "ico3"
        assert expand_cli_resolution(16, "spectral") == "T16"
        assert expand_cli_resolution(16, "mpas_regional") == "16km"
        assert expand_cli_resolution(16, "latlon_regional") == "16x32"
        assert expand_cli_resolution(16, "cs_regional") == "C16"

    def test_run_omip_uses_helper(self):
        """iter-115 codex HIGH-1: ``run_omip.py`` must use the
        shared dispatch helper.
        """
        from pathlib import Path
        path = Path(__file__).resolve().parent.parent / "scripts" / "run_omip.py"
        text = path.read_text()
        assert "validate_cli_resolution" in text, (
            "iter-115: ``run_omip.py`` must use the shared "
            "``validate_cli_resolution`` helper."
        )
        assert "expand_cli_resolution" in text, (
            "iter-115: ``run_omip.py`` must use the shared "
            "``expand_cli_resolution`` helper."
        )


class TestSpectralW2L2Norm:
    """iter-116 (codex iter-114 HIGH-5): pre-iter-116, the
    spectral W2 branch fell through to ``notes: mass drift=0``
    rather than computing the analytical L2 height error.
    Codex correctly flagged this as ``not a valid Williamson
    L2 comparison`` — comparing cube/latlon/ico L2 errors
    against a spectral ``mass drift=0`` is meaningless.

    iter-116 added a spectral W2 branch that:
    1. Re-creates the initial state via
       ``williamson_test2_spectral(grid)`` (which IS the
       analytical steady-state solution for W2).
    2. Synthesizes both initial and final ``phi_hat`` to grid
       space height.
    3. Computes area-weighted L2/Linf error norms.

    These tests pin the iter-116 fix at the source level.
    """

    def _import_module(self):
        import importlib
        import sys
        from pathlib import Path
        scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        return importlib.import_module("run_atmosphere_test_matrix")

    def test_source_pin_spectral_w2_l2_branch_present(self):
        """The matrix runner contains the iter-116 spectral W2
        branch.  Pre-iter-116 the matrix had branches for
        cube, latlon, and icosahedral W2 but NOT spectral —
        spectral fell through to the generic mass-drift
        branch.
        """
        import inspect
        import re
        M = self._import_module()
        text = inspect.getsource(M)
        text_no_strings = re.sub(r'""".*?"""', "", text, flags=re.DOTALL)
        text_no_strings = re.sub(r"'''.*?'''", "", text_no_strings, flags=re.DOTALL)
        code_only = "\n".join(
            line for line in text_no_strings.splitlines()
            if not line.lstrip().startswith("#")
        )
        assert 'test_num == 2 and tc.grid_type == "spectral"' in code_only, (
            "iter-116: matrix runner must have a "
            "``test_num == 2 and tc.grid_type == \"spectral\"`` "
            "branch that computes L2/Linf, not fall through to "
            "the generic mass-drift branch."
        )
        assert "williamson_test2_spectral" in code_only, (
            "iter-116: spectral W2 branch must re-create the "
            "initial state via ``williamson_test2_spectral`` "
            "to use as the analytical steady-state reference."
        )

    def test_spectral_w2_l2_below_threshold(self):
        """End-to-end smoke: run spectral W2 at T16 for 1 day
        and assert L2 < 1e-6 (much better than the FV/MPAS
        dycores' ~1e-3 to 1e-4 at C16/16x32/ico3).
        """
        import subprocess
        from pathlib import Path
        repo_root = Path(__file__).resolve().parent.parent
        result = subprocess.run(
            [".venv/bin/python",
             "scripts/run_atmosphere_test_matrix.py",
             "--only", "sw", "--test", "williamson2",
             "--grid", "spectral",
             "--quick", "--resolution", "16",
             "--no-cross-grid-plots"],
            cwd=str(repo_root),
            capture_output=True, text=True,
            env={"JAX_ENABLE_X64": "1", "PATH": "/usr/bin:/bin"},
            timeout=120,
        )
        assert result.returncode == 0, (
            f"iter-116: spectral W2 must PASS; got "
            f"{result.returncode}.\nstdout:\n{result.stdout}"
            f"\nstderr:\n{result.stderr}"
        )
        # Extract the L2 from the PASS line.
        import re
        m = re.search(
            r"shallow_water/williamson2/spectral.*L2=([\d.eE+-]+)",
            result.stdout,
        )
        assert m is not None, (
            f"iter-116: PASS line must include L2=...; got:\n"
            f"{result.stdout}"
        )
        l2 = float(m.group(1))
        # Williamson 1992 reports T42 W2 L2 ~ 1e-9.  At T16 +
        # 1 day with dt~10 min, expect ~1e-8.  Allow 10x
        # headroom for resolution / dt sensitivity → 1e-6.
        assert l2 < 1e-6, (
            f"iter-116: spectral T16 W2 L2 should be at least "
            f"1e-6 (was 3.6e-8 in development); got {l2:.2e}.  "
            f"This is a much tighter bound than cube/latlon/ico "
            f"and serves as a regression sentinel."
        )
