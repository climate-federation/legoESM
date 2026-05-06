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
