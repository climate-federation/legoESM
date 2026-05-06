"""Unit tests for the cross-grid comparison + diagnostic helpers in
``scripts/run_ocean_test_matrix.py`` and the modular copy at
``scripts/ocean_test_matrix/diagnostic_io.py``.

Pins the iter-19 latlon-C-grid staggered-velocity averaging fix and the
iter-21 defensive shape-validation refinement.

Counterpart to ``tests/test_atmosphere_cross_grid_plots.py``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


_SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))


# ---------------------------------------------------------------------------
# Pandas import sanity check (iter-19)
# ---------------------------------------------------------------------------

class TestImports:
    def test_run_ocean_test_matrix_imports_pandas(self):
        """iter-19: ``pd.read_csv`` was used at lines 4617 and 6106 but
        pandas was never imported.  Test that the import is now
        present and that the module loads without error.
        """
        # Import the script module (will trigger the top-level imports).
        import importlib
        import run_ocean_test_matrix as M
        assert hasattr(M, "pd"), (
            "scripts/run_ocean_test_matrix.py is missing the iter-19 "
            "``import pandas as pd`` — cross-grid comparison block "
            "will silently fail with NameError"
        )
        # pandas namespace check.
        assert hasattr(M.pd, "read_csv")


# ---------------------------------------------------------------------------
# Latlon C-grid staggered velocity averaging logic (iter-19, iter-21)
# ---------------------------------------------------------------------------

class TestStaggeredVelocityAveraging:
    """Pin the iter-19 patch + iter-21 refinement for the latlon-ocean
    quiver overlay.  The patch lives directly inside the plotting
    function so we can't unit-test the function in isolation; instead
    test the averaging FORMULA on synthetic input arrays.
    """

    def _stagger_average(self, u, v):
        """Mirrors the in-source patch.  Returns ``(u_cc, v_cc, ok)``
        where ``ok`` is ``False`` when the shapes still don't match
        after the canonical face→cell averaging (iter-21 defensive
        skip)."""
        if u.shape != v.shape:
            if u.shape[1] == v.shape[1] + 1:
                u = 0.5 * (u[:, :-1] + u[:, 1:])
            if v.shape[0] == u.shape[0] + 1:
                v = 0.5 * (v[:-1, :] + v[1:, :])
        return u, v, (u.shape == v.shape)

    def test_canonical_latlon_cgrid(self):
        """``u(n_lat, n_lon+1)`` and ``v(n_lat+1, n_lon)`` average to
        the same cell-centred ``(n_lat, n_lon)`` shape with a simple
        average of adjacent face values."""
        n_lat, n_lon = 5, 7
        # Make distinguishable face values.
        u = np.arange(n_lat * (n_lon + 1), dtype=float).reshape(n_lat, n_lon + 1)
        v = np.arange((n_lat + 1) * n_lon, dtype=float).reshape(n_lat + 1, n_lon)
        u_cc, v_cc, ok = self._stagger_average(u, v)
        assert ok
        assert u_cc.shape == (n_lat, n_lon)
        assert v_cc.shape == (n_lat, n_lon)
        # Verify the actual averaging.
        assert np.allclose(u_cc[2, 3], 0.5 * (u[2, 3] + u[2, 4]))
        assert np.allclose(v_cc[2, 3], 0.5 * (v[2, 3] + v[3, 3]))

    def test_already_cell_centred(self):
        """When u and v already share the same shape (cubed-sphere /
        gaussian / mpas extracted to cell centres), the averaging
        path is a no-op."""
        n_lat, n_lon = 6, 8
        u = np.full((n_lat, n_lon), 3.0)
        v = np.full((n_lat, n_lon), -2.0)
        u_cc, v_cc, ok = self._stagger_average(u, v)
        assert ok
        assert u_cc is u
        assert v_cc is v

    def test_non_canonical_mismatch_skips(self):
        """iter-21 codex MEDIUM: if shapes don't match the canonical
        C-grid stagger relation, the patch returns ``ok=False`` so the
        caller can skip the quiver overlay rather than silently
        truncate."""
        # u missing only 1 column on lon axis (canonical) but v also
        # missing 1 column → non-canonical.
        u = np.zeros((5, 8))   # would average to (5, 7)
        v = np.zeros((6, 7))   # canonical: would average to (5, 7)
        u_cc, v_cc, ok = self._stagger_average(u, v)
        # u didn't match the +1 condition (8 != 7+1=8 — actually it
        # DOES match), so let's pick a truly non-canonical case.

        # Pick u shape = (5, 9), v shape = (7, 7) — neither matches
        # the canonical +1 relation.
        u = np.zeros((5, 9))
        v = np.zeros((7, 7))
        u_cc, v_cc, ok = self._stagger_average(u, v)
        # Neither face→cell averaging applies; shapes stay different.
        assert not ok

    def test_constant_field_preserves_value(self):
        """Averaging a constant field returns the same constant —
        sanity check on the 0.5 averaging weights."""
        n_lat, n_lon = 4, 6
        u = np.full((n_lat, n_lon + 1), 7.5)
        v = np.full((n_lat + 1, n_lon), -1.25)
        u_cc, v_cc, ok = self._stagger_average(u, v)
        assert ok
        assert np.allclose(u_cc, 7.5)
        assert np.allclose(v_cc, -1.25)


# ---------------------------------------------------------------------------
# Modular copy at scripts/ocean_test_matrix/diagnostic_io.py
# ---------------------------------------------------------------------------

class TestModularCopyParity:
    def test_modular_diagnostic_io_loads(self):
        """iter-21 LOW: the modular copy
        ``scripts/ocean_test_matrix/diagnostic_io.py`` had the same
        unfixed staggered-velocity bug.  Verify it imports cleanly."""
        import importlib
        # Direct import via package path.
        sys.path.insert(0, str(_SCRIPT_DIR / "ocean_test_matrix"))
        try:
            import diagnostic_io  # noqa: F401
        finally:
            sys.path.pop(0)


# ---------------------------------------------------------------------------
# iter-49: relaxed _collect_grid_results for timeseries-only OMIP runs
# ---------------------------------------------------------------------------

class TestRelaxedCollectorAcceptsTimeseriesOnly:
    """Pin the iter-49 relaxation of ``_collect_grid_results``.

    Previously the ocean cross-grid collector required ALL THREE of
    ``mean_timeseries.csv`` + ``snapshots_latlon.npz`` + ``results.txt``;
    OMIP runs (which don't emit snapshots_latlon.npz) were silently
    skipped from the cross-grid plot pass.  iter-49 relaxes the
    predicate to match the iter-26 atmosphere-matrix pattern: accept
    EITHER ``snapshots_latlon.npz`` OR (``mean_timeseries.csv`` AND
    ``results.txt``).
    """

    def _import_matrix_module(self):
        import importlib
        return importlib.import_module("run_ocean_test_matrix")

    def _write_synthetic_timeseries_only(self, dir_path: Path):
        """Mimic an OMIP per-grid output: CSV + results.txt, no npz."""
        dir_path.mkdir(parents=True, exist_ok=True)
        (dir_path / "mean_timeseries.csv").write_text(
            "time_days,mean_eta,max_speed\n"
            "0.0,0.0,0.0\n"
            "1.0,0.001,0.05\n"
            "2.0,0.002,0.10\n"
        )
        (dir_path / "results.txt").write_text(
            "test: omip\n"
            "grid: cubed_sphere\n"
            "resolution: C24\n"
            "status: PASS\n"
            "wall_time: 12.3s\n"
        )

    def _write_synthetic_snapshots_only(self, dir_path: Path):
        """Snapshots-only output (legacy path, less common)."""
        dir_path.mkdir(parents=True, exist_ok=True)
        eta = np.zeros((2, 4, 6), dtype=np.float64)
        np.savez(dir_path / "snapshots_latlon.npz", eta=eta)

    def test_collector_accepts_timeseries_only(self, tmp_path):
        """A grid directory with CSV + results.txt (but no
        snapshots_latlon.npz) must be picked up by the collector.
        """
        M = self._import_matrix_module()
        case_dir = tmp_path / "omip"
        grid_dir = case_dir / "cubed_sphere" / "C24"
        self._write_synthetic_timeseries_only(grid_dir)

        results = M._collect_grid_results(case_dir)
        assert "cubed_sphere" in results
        entry = results["cubed_sphere"]
        # Timeseries was loaded.
        assert entry["timeseries"] is not None
        # Snapshots is None (no npz file).
        assert entry["snapshots"] is None
        # Metadata parsed from results.txt.
        assert entry["metadata"]["test"] == "omip"
        assert entry["metadata"]["status"] == "PASS"

    def test_collector_accepts_snapshots_only(self, tmp_path):
        """A grid directory with snapshots_latlon.npz only (no CSV /
        results.txt) must still be picked up — the iter-26 pattern
        accepts EITHER half."""
        M = self._import_matrix_module()
        case_dir = tmp_path / "rest_state"
        grid_dir = case_dir / "latlon" / "36x72"
        self._write_synthetic_snapshots_only(grid_dir)

        results = M._collect_grid_results(case_dir)
        assert "latlon" in results
        entry = results["latlon"]
        # Snapshots loaded.
        assert entry["snapshots"] is not None
        # Timeseries is None (no CSV).
        assert entry["timeseries"] is None

    def test_collector_rejects_empty_grid_dir(self, tmp_path):
        """A grid directory with NEITHER half is rejected (no
        phantom collection of runs that didn't produce output)."""
        M = self._import_matrix_module()
        case_dir = tmp_path / "barotropic_wave"
        grid_dir = case_dir / "spectral" / "T21"
        grid_dir.mkdir(parents=True, exist_ok=True)
        # Touch an unrelated file — must NOT count.
        (grid_dir / "stale.log").write_text("nothing here\n")

        results = M._collect_grid_results(case_dir)
        assert "spectral" not in results

    def test_create_cross_grid_comparisons_handles_timeseries_only(
        self, tmp_path,
    ):
        """``_create_cross_grid_comparisons`` must NOT crash when all
        grids in the test case have only timeseries data (no snapshots).
        It should emit ONLY the timeseries comparison plot.
        """
        M = self._import_matrix_module()
        case_dir = tmp_path / "omip"
        for grid in ("cubed_sphere", "latlon"):
            grid_dir = case_dir / grid / "C24"
            self._write_synthetic_timeseries_only(grid_dir)
        # Override the grid name in the metadata so they look distinct.
        (case_dir / "latlon" / "C24" / "results.txt").write_text(
            "test: omip\n"
            "grid: latlon\n"
            "resolution: 90x180\n"
            "status: PASS\n"
            "wall_time: 18.4s\n"
        )

        results = M._collect_grid_results(case_dir)
        assert len(results) == 2
        # Should not crash even though no grid has snapshots.
        M._create_cross_grid_comparisons(case_dir, results)
        # Only the timeseries comparison was emitted; no
        # comparison_snapshots_*.png.
        assert (case_dir / "comparison_timeseries.png").exists()
        for f in case_dir.iterdir():
            assert not f.name.startswith("comparison_snapshots_"), (
                f"unexpected snapshot plot {f.name} for "
                f"timeseries-only test case"
            )

    def _write_synthetic_full_run(self, dir_path: Path):
        """A run with BOTH timeseries CSV and snapshots npz."""
        self._write_synthetic_timeseries_only(dir_path)
        # Add a 2-time-step eta snapshot for plotting.
        eta = np.zeros((2, 4, 6), dtype=np.float64)
        np.savez(
            dir_path / "snapshots_latlon.npz",
            eta=eta, times_days=np.array([0.0, 2.0]),
        )

    def test_mixed_fixture_only_grids_with_snapshots_get_snapshot_plots(
        self, tmp_path,
    ):
        """iter-50 codex MEDIUM: when one grid has snapshots and
        another is timeseries-only, the snapshot plotters must
        receive ONLY the grids that contributed snapshots — not
        both.

        Specifically: with N=2 grids total but only 1 with
        snapshots, ``len(grids_with_snapshots) < 2`` so the
        cross-grid snapshot plot is skipped (would otherwise be
        a misleading single-grid "cross-grid" plot).
        """
        M = self._import_matrix_module()
        case_dir = tmp_path / "omip"
        # Grid A: full run (snapshots + timeseries).
        self._write_synthetic_full_run(case_dir / "cubed_sphere" / "C24")
        # Grid B: timeseries-only.
        self._write_synthetic_timeseries_only(case_dir / "latlon" / "C24")
        (case_dir / "latlon" / "C24" / "results.txt").write_text(
            "test: omip\n"
            "grid: latlon\n"
            "resolution: 90x180\n"
            "status: PASS\n"
            "wall_time: 18.4s\n"
        )

        results = M._collect_grid_results(case_dir)
        assert len(results) == 2
        M._create_cross_grid_comparisons(case_dir, results)
        # Timeseries plot was emitted (both grids contribute).
        assert (case_dir / "comparison_timeseries.png").exists()
        # Snapshot plots are SKIPPED because only 1 grid has
        # snapshots — a single-grid "cross-grid" plot would be
        # misleading.
        for f in case_dir.iterdir():
            assert not f.name.startswith("comparison_snapshots_"), (
                f"unexpected snapshot plot {f.name} for mixed "
                f"fixture (1 snapshots, 1 timeseries-only)"
            )

    def test_mixed_fixture_two_with_snapshots_does_emit_snapshot_plots(
        self, tmp_path,
    ):
        """If 2+ grids have snapshots, the cross-grid snapshot plot
        SHOULD be emitted — even if other grids are timeseries-only.
        """
        M = self._import_matrix_module()
        case_dir = tmp_path / "omip"
        # 2 grids with snapshots.
        self._write_synthetic_full_run(case_dir / "cubed_sphere" / "C24")
        self._write_synthetic_full_run(case_dir / "latlon" / "C24")
        (case_dir / "latlon" / "C24" / "results.txt").write_text(
            "test: omip\nstatus: PASS\nwall_time: 1s\n"
        )
        # 1 timeseries-only grid (extra context but doesn't add to
        # the snapshot panel).
        self._write_synthetic_timeseries_only(case_dir / "spectral" / "T21")
        (case_dir / "spectral" / "T21" / "results.txt").write_text(
            "test: omip\nstatus: PASS\nwall_time: 1s\n"
        )

        results = M._collect_grid_results(case_dir)
        assert len(results) == 3
        M._create_cross_grid_comparisons(case_dir, results)
        # Snapshot plot emitted for eta (2 grids contribute).
        assert (case_dir / "comparison_snapshots_eta.png").exists()
        assert (case_dir / "comparison_timeseries.png").exists()

    def test_corrupt_npz_falls_back_to_timeseries(self, tmp_path):
        """iter-50 codex MEDIUM: per-artifact load isolation.  A
        corrupt ``snapshots_latlon.npz`` must NOT drop the whole
        run; the CSV + results.txt are still usable for the
        timeseries cross-grid comparison.
        """
        M = self._import_matrix_module()
        case_dir = tmp_path / "rest_state"
        grid_dir = case_dir / "cubed_sphere" / "C24"
        self._write_synthetic_timeseries_only(grid_dir)
        # Write a corrupt npz file (random bytes that aren't a
        # valid numpy archive).
        (grid_dir / "snapshots_latlon.npz").write_bytes(
            b"NOT_A_VALID_NPZ_FILE\x00\x01\x02\x03"
        )

        results = M._collect_grid_results(case_dir)
        # Run is still collected (timeseries half is intact).
        assert "cubed_sphere" in results
        entry = results["cubed_sphere"]
        # Snapshots load failed → None.
        assert entry["snapshots"] is None
        # Timeseries half is preserved.
        assert entry["timeseries"] is not None

    def test_snapshots_only_skips_empty_timeseries_plot(self, tmp_path):
        """iter-50 codex LOW: when all collected grids are snapshots-
        only (no timeseries CSV), ``_create_comparison_timeseries``
        must early-return so we don't emit an empty
        ``comparison_timeseries.png`` with no curves.
        """
        M = self._import_matrix_module()
        case_dir = tmp_path / "rest_state"
        for grid in ("cubed_sphere", "latlon"):
            self._write_synthetic_snapshots_only(case_dir / grid / "C24")

        results = M._collect_grid_results(case_dir)
        assert len(results) == 2
        # All entries are snapshots-only.
        for entry in results.values():
            assert entry["timeseries"] is None
            assert entry["snapshots"] is not None
        M._create_comparison_timeseries(case_dir, results)
        # No timeseries.png emitted (no timeseries data).
        assert not (case_dir / "comparison_timeseries.png").exists()
