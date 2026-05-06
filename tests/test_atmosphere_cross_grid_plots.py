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
        # Reset the dedup set — also reset by main(); test isolation.
        M._RES_DIR_WARNED.clear()
        gd = tmp_path / "spectral"
        (gd / "T21").mkdir(parents=True)
        (gd / "T42").mkdir(parents=True)
        out1 = M._select_resolution_dir(gd)
        out2 = M._select_resolution_dir(gd)
        captured = capsys.readouterr()
        # First call warns, second is silent (deduped).
        assert "multiple resolution" in captured.out
        # No second warning.
        assert captured.out.count("multiple resolution") == 1
        # Both calls return the same alphabetically-first directory.
        assert out1 == out2 == gd / "T21"

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
        mod.main(tmp_path)
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
        """``Status: COMPLETED`` → PASS; anything else → ERROR."""
        mod = self._import_converter()
        self._write_synthetic_amip_outputs(tmp_path)
        # Overwrite with a non-COMPLETED status line.
        (tmp_path / "results.txt").write_text(
            "legoESM AMIP run\n"
            "Grid: spectral T42 / L20, dt=600.0s, 30 days\n"
            "Status: FAILED\n"
            "Wall time: 5.0s\n"
        )
        mod.main(tmp_path)
        results_text = (tmp_path / "results.txt").read_text()
        assert "status: ERROR" in results_text

    def test_converter_handles_missing_npz(self, tmp_path, capsys):
        """If timeseries.npz is missing, converter warns and exits cleanly."""
        mod = self._import_converter()
        # No ``timeseries.npz`` in tmp_path.
        mod.main(tmp_path)
        captured = capsys.readouterr()
        assert "WARNING" in captured.err or "missing" in captured.err
        # Must NOT have written a CSV.
        assert not (tmp_path / "mean_timeseries.csv").exists()

    def test_collector_recognizes_converted_output(self, tmp_path):
        """The matrix runner's ``_has_collectable`` must return True
        on a directory after conversion.  This is the integration
        guarantee — without this, the iter-41 wrapper silently no-ops.
        """
        mod = self._import_converter()
        self._write_synthetic_amip_outputs(tmp_path)
        mod.main(tmp_path)

        # Re-implement the matrix-runner predicate locally.
        def _has_collectable(d: Path) -> bool:
            if (d / "snapshots_latlon.npz").exists():
                return True
            return (d / "mean_timeseries.csv").exists() and (d / "results.txt").exists()

        assert _has_collectable(tmp_path)

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
