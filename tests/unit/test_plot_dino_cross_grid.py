"""Unit tests for the DINO cross-grid comparison plotter helpers.

Covers the pure logic (snapshot selection, stats) without building a mesh or
regridding — those go through the heavy ``create_regional_voronoi_mesh`` /
``regrid_curv_to_latlon`` paths, which have their own tests and are exercised
by the live ``scripts/plot/plot_dino_cross_grid.py`` run.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

# Load the plotter module directly (scripts/ is not a package).
_SPEC = importlib.util.spec_from_file_location(
    "plot_dino_cross_grid",
    Path(__file__).resolve().parents[2] / "scripts" / "plot" / "plot_dino_cross_grid.py",
)
pdx = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(pdx)


def _write_snap(d: Path, idx: int, day: float, t_surf):
    """Write a minimal snapshot npz (T (ncol,1), land_mask, time_days). The
    land_mask is required by the ocean-masked finite check in _snap_for_day."""
    T = np.asarray(t_surf, dtype=float)[:, None]
    mask = np.ones(T.shape[0], dtype=float)  # all-ocean stub
    np.savez(d / f"snapshot_{idx:05d}.npz",
             time_days=np.float64(day), T=T, land_mask=mask)


class TestStats:
    # _stats requires >10 common finite cells (guard for real data), so use
    # arrays comfortably larger than that.
    def test_identical_fields(self):
        a = np.arange(20.0).reshape(4, 5)
        corr, rms = pdx._stats(a, a.copy())
        assert corr == pytest.approx(1.0)
        assert rms == pytest.approx(0.0)

    def test_known_offset(self):
        a = np.arange(20.0).reshape(4, 5)
        b = a + 2.0  # constant offset: perfect corr, RMS = 2
        corr, rms = pdx._stats(a, b)
        assert corr == pytest.approx(1.0)
        assert rms == pytest.approx(2.0)

    def test_nan_masked(self):
        a = np.arange(20.0).reshape(4, 5)
        b = a.copy(); b[0, 0] = np.nan  # 19 common-finite of 20 (> 10 guard)
        corr, rms = pdx._stats(a, b)    # only common-finite entries compared
        assert rms == pytest.approx(0.0)
        assert corr == pytest.approx(1.0)


class TestSnapForDay:
    def test_picks_closest_day(self, tmp_path):
        snaps = tmp_path / "snapshots"; snaps.mkdir()
        for i, day in enumerate([0.0, 30.0, 60.0, 90.0]):
            _write_snap(snaps, i, day, np.full(6, 10.0 + day))
        snap, day = pdx._snap_for_day(snaps, 58.0)
        assert day == pytest.approx(60.0)
        assert snap.name == "snapshot_00002.npz"

    def test_none_skips_nan_tail(self, tmp_path):
        """day=None must return the last snapshot with FINITE surface T, i.e.
        skip a NaN-contaminated blown-up tail (the DINO MPAS day>130 case)."""
        snaps = tmp_path / "snapshots"; snaps.mkdir()
        _write_snap(snaps, 0, 0.0, np.full(6, 10.0))
        _write_snap(snaps, 1, 90.0, np.full(6, 12.0))     # last good
        _write_snap(snaps, 2, 130.0, np.full(6, np.nan))  # blew up
        snap, day = pdx._snap_for_day(snaps, None)
        assert day == pytest.approx(90.0)
        assert snap.name == "snapshot_00001.npz"


class TestDayOf:
    def test_reads_time_days(self, tmp_path):
        snaps = tmp_path / "snapshots"; snaps.mkdir()
        _write_snap(snaps, 0, 42.0, np.full(6, 11.0))
        assert pdx._day_of(snaps / "snapshot_00000.npz") == pytest.approx(42.0)
