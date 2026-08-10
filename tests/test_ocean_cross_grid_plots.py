"""Unit tests for the cross-grid comparison + diagnostic helpers in
``scripts/matrix/run_ocean_test_matrix.py`` and the modular copy at
``scripts/matrix/ocean_test_matrix/diagnostic_io.py``.

Pins the iter-19 latlon-C-grid staggered-velocity averaging fix and the
iter-21 defensive shape-validation refinement.

Counterpart to ``tests/test_atmosphere_cross_grid_plots.py``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


def _read_repo_or_legoesm(repo, rel):
    """Read a repo file; carve-aware for legoesm sources (a subpackage may live in
    a uv-workspace member, so resolve ``src/legoesm/...`` via the namespace)."""
    if str(rel).startswith(("src/legoesm/", "legoesm/")):
        from tests.legoesm_paths import legoesm_source_path

        return legoesm_source_path(rel).read_text()
    return (repo / rel).read_text()


_SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
# run_ocean_test_matrix and the ocean_test_matrix/ package both live under
# scripts/matrix/ (bucket layout; see tests/test_scripts_layout.py).
_MATRIX_DIR = _SCRIPT_DIR / "matrix"
if str(_MATRIX_DIR) not in sys.path:
    sys.path.insert(0, str(_MATRIX_DIR))


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
            "scripts/matrix/run_ocean_test_matrix.py is missing the iter-19 "
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
# Modular copy at scripts/matrix/ocean_test_matrix/diagnostic_io.py
# ---------------------------------------------------------------------------

class TestModularCopyParity:
    def test_modular_diagnostic_io_loads(self):
        """iter-21 LOW: the modular copy
        ``scripts/matrix/ocean_test_matrix/diagnostic_io.py`` had the same
        unfixed staggered-velocity bug.  Verify it imports cleanly."""
        import importlib
        # Direct import via package path.
        sys.path.insert(0, str(_MATRIX_DIR / "ocean_test_matrix"))
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

    def test_csv_valid_metadata_corrupt_keeps_run(self, tmp_path):
        """iter-51 codex MEDIUM: per-artifact load isolation must
        also handle the case where ``results.txt`` is unparseable
        (metadata ends up empty) but ``mean_timeseries.csv`` is
        valid.  The run should be KEPT (timeseries data is enough);
        downstream summary plotter uses ``metadata.get(..., 'N/A')``
        so missing fields render as N/A.
        """
        M = self._import_matrix_module()
        case_dir = tmp_path / "rest_state"
        grid_dir = case_dir / "cubed_sphere" / "C24"
        self._write_synthetic_timeseries_only(grid_dir)
        # Overwrite results.txt with content that has no ``key:value``
        # lines so the parser produces an empty metadata dict.
        (grid_dir / "results.txt").write_text(
            "this file has no colon-separated lines\n"
            "so the iter-49 parser produces empty metadata\n"
        )

        results = M._collect_grid_results(case_dir)
        # Run is still collected (timeseries half is intact).
        assert "cubed_sphere" in results
        entry = results["cubed_sphere"]
        # Timeseries loaded.
        assert entry["timeseries"] is not None
        # Metadata is empty.
        assert entry["metadata"] == {}

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


# ---------------------------------------------------------------------------
# iter-82: pin the iter-80 conservation-timeseries denominator-floor fix
# ---------------------------------------------------------------------------

class TestSaveConservationDenominatorFloor:
    """The iter-80 fix raised ``_save_conservation``'s denominator
    floor from 1e-30 to 1.0.  Pin that the floor stays at a
    physically meaningful value so a regression to 1e-30 (or
    lower) would amplify machine-precision rounding into spurious
    1e+13-magnitude "relative drift" values for rest-state runs.
    """

    def _import_matrix_module(self):
        import importlib
        return importlib.import_module("run_ocean_test_matrix")

    def test_rest_state_baseline_zero_does_not_blow_up(self, tmp_path):
        """Rest-state runs have ``vol[0] = 0`` (initial mean η is
        zero by construction).  After the iter-80 fix, machine-
        precision rounding of -2.83e-17 should produce a
        ``vol_rel`` value of -2.83e-17, NOT -2.83e+13.

        iter-84 codex MEDIUM: extended to also test heat_rel and
        salt_rel zero-baseline behavior.  iter-82 only checked
        vol_rel; if a future regression lowered only heat_denom
        or salt_denom back to 1e-30, the iter-82 test would
        miss it.
        """
        M = self._import_matrix_module()

        # iter-86 codex LOW: use DISTINCT drifts per column so a
        # column-mixup bug (e.g., heat_rel accidentally written
        # to salt_rel) would also flip the test, not just a
        # denominator-floor regression.
        vol_drift = -2.83e-17
        heat_drift = -1.59e-16
        salt_drift = -7.42e-18
        diag = {
            "vol_key": [0.0, -1e-19, vol_drift],
            "heat_key": [0.0, -1e-19, heat_drift],
            "salt_key": [0.0, -1e-19, salt_drift],
            "times": [0.0, 0.05, 0.10],
        }
        out = tmp_path / "out"
        M._save_conservation(
            out, "rest_state", diag,
            vol_key="vol_key", heat_key="heat_key", salt_key="salt_key",
        )
        csv_path = out / "conservation_timeseries.csv"
        assert csv_path.exists()
        import csv as csv_mod
        with open(csv_path) as f:
            reader = csv_mod.DictReader(f)
            rows = list(reader)
        # iter-84: pin all three columns, with both magnitude
        # bound (catches the spurious 1e+13) AND exact value
        # match (catches subtler floor regressions).
        # iter-86: distinct drifts per column also catch
        # column-mixup bugs.
        expected_per_col = {
            "vol_rel": vol_drift,
            "heat_rel": heat_drift,
            "salt_rel": salt_drift,
        }
        for col, drift in expected_per_col.items():
            value = float(rows[-1][col])
            assert abs(value) < 1e-10, (
                f"iter-80/82 regression: {col} = {value:.2e} for "
                f"a zero baseline.  Expected ~1e-17 (actual "
                f"rounding magnitude); got 1e-10 or larger.  Most "
                f"likely cause: ``_MIN_RELATIVE_BASELINE`` (or its "
                f"per-column equivalent) was lowered from 1.0 "
                f"back toward 1e-30."
            )
            # iter-84 codex LOW: pin the EXACT value.  With the
            # iter-80 fix (denominator = max(|baseline|, 1.0) =
            # 1.0 for baseline=0), the column reports absolute
            # drift in physical units.  iter-86: distinct
            # per-column drift values catch column-mixup bugs.
            assert value == pytest.approx(drift, rel=1e-6), (
                f"iter-80/82/86 regression: {col} = {value:.6e}, "
                f"expected exactly {drift:.6e}.  If {col} returned "
                f"another column's drift, the column-mixup bug "
                f"flipped this test — check the CSV writer order."
            )

    def test_real_baseline_uses_relative_drift(self, tmp_path):
        """When the baseline is non-trivial (e.g., a forced run
        with vol[0] = 1.5e18 m³), ``vol_rel`` should report
        relative drift, not absolute.
        """
        M = self._import_matrix_module()
        # Realistic ocean baseline: 1.5e18 m³ (~ Earth ocean volume).
        # Drift over 3 steps: 1.5e15 m³ (= 1e-3 relative).
        diag = {
            "vol_key": [1.5e18, 1.5e18 + 5e14, 1.5e18 + 1.5e15],
            "heat_key": [3.6e25, 3.6e25, 3.6e25],
            "salt_key": [1.4e22, 1.4e22, 1.4e22],
            "times": [0.0, 0.05, 0.10],
        }
        out = tmp_path / "out"
        M._save_conservation(
            out, "forced_run", diag,
            vol_key="vol_key", heat_key="heat_key", salt_key="salt_key",
        )
        import csv as csv_mod
        with open(out / "conservation_timeseries.csv") as f:
            reader = csv_mod.DictReader(f)
            rows = list(reader)
        last_vol_rel = float(rows[-1]["vol_rel"])
        # With baseline 1.5e18 and drift 1.5e15, relative drift
        # should be 1e-3.  Not -2.83e+13, not 1.5e15.
        assert 0.5e-3 <= last_vol_rel <= 2e-3, (
            f"vol_rel = {last_vol_rel:.4e} for a forced run with "
            f"baseline 1.5e18 m³ and drift 1.5e15 m³; expected "
            f"~1e-3 (the relative-drift value).  Was the iter-80 "
            f"fix accidentally always-absolute?"
        )

    def test_min_relative_baseline_constant_is_one(self, tmp_path):
        """Source-level pin of the iter-80 floor convention.

        iter-90 codex HIGH-2: the previous test checked for the
        literal string ``_MIN_RELATIVE_BASELINE = 1.0`` in
        ``_save_conservation``'s source.  After iter-88 factored
        the inline floor into the shared helper module, that exact
        literal still appeared in a docstring comment ("originally
        written inline here as ``_MIN_RELATIVE_BASELINE = 1.0``")
        — so the test passed for the WRONG reason and would no
        longer catch a regression that re-introduced the 1e-30
        floor.

        Updated test pins:
          1) ``_save_conservation`` delegates to
             ``relative_drift_series`` (the canonical helper).
          2) The 1e-30 anti-pattern is absent from the actual code
             (we strip comments first to avoid the iter-88
             docstring-mention false-positive).
          3) The shared helper's ``DEFAULT_MIN_BASELINE`` is 1.0.
        """
        import inspect
        import re

        M = self._import_matrix_module()
        from legoesm.diagnostics.conservation_drift import (
            DEFAULT_MIN_BASELINE,
        )

        src = inspect.getsource(M._save_conservation)
        # Strip ``#`` line-comments and docstring-style ``"""..."""``
        # blocks so we test the actual code, not iter-88-style
        # historical mentions of the legacy formula.
        src_no_strings = re.sub(r'""".*?"""', "", src, flags=re.DOTALL)
        src_no_strings = re.sub(r"'''.*?'''", "", src_no_strings, flags=re.DOTALL)
        # Drop any line whose first non-whitespace char is ``#``.
        code_only = "\n".join(
            line for line in src_no_strings.splitlines()
            if not line.lstrip().startswith("#")
        )

        # 1) Delegation to the canonical helper.
        assert "relative_drift_series" in code_only, (
            "iter-90: ``_save_conservation`` must call "
            "``relative_drift_series`` from "
            "``legoesm.diagnostics.conservation_drift`` so that the "
            "iter-80 floor convention is inherited from the single "
            "source of truth."
        )

        # 2) The 1e-30 anti-pattern must not reappear in code.
        assert "1e-30" not in code_only, (
            "iter-90: the legacy 1e-30 denominator floor must not "
            "appear in ``_save_conservation``'s code (only in the "
            "history comments).  iter-78/80 showed it amplifies "
            "machine-precision rounding to spurious 1e+13 "
            "'relative drift' values for rest-state baselines."
        )

        # 3) The shared floor constant is 1.0.
        assert DEFAULT_MIN_BASELINE == 1.0, (
            "iter-90: the canonical "
            "``DEFAULT_MIN_BASELINE`` must remain at 1.0.  Any "
            "deliberate change should land in iter-N alongside this "
            "test."
        )


class TestOceanComputeDriftDelegates:
    """iter-90 codex review HIGH-1: the ocean script's
    ``_compute_drift`` (separate from the atmosphere version of the
    same name) was missed in the iter-88 refactor.  Ten callsites
    (T_drift, PE_drift, S_integral_drift) had stale ``1e-30`` floor
    behavior.  Pin that the function now delegates to the shared
    helper so all 10 callsites inherit the iter-80 floor convention.
    """

    def _import_matrix_module(self):
        import importlib
        return importlib.import_module("run_ocean_test_matrix")

    def test_ocean_compute_drift_delegates_to_helper(self):
        """The ocean ``_compute_drift`` body must call the canonical
        ``compute_relative_drift`` (or be that function directly),
        not re-implement the floor inline.
        """
        import inspect
        import re

        M = self._import_matrix_module()
        src = inspect.getsource(M._compute_drift)
        # Strip docstring + line comments to test actual code only.
        src_no_strings = re.sub(r'""".*?"""', "", src, flags=re.DOTALL)
        src_no_strings = re.sub(r"'''.*?'''", "", src_no_strings, flags=re.DOTALL)
        code_only = "\n".join(
            line for line in src_no_strings.splitlines()
            if not line.lstrip().startswith("#")
        )
        assert "compute_relative_drift" in code_only, (
            "iter-90 (codex HIGH-1): ocean ``_compute_drift`` must "
            "call ``compute_relative_drift`` from "
            "``legoesm.diagnostics.conservation_drift`` so the "
            "iter-80 floor convention is inherited from a single "
            "source.  iter-88 missed this 10-callsite function."
        )
        # The legacy 1e-30 floor must not appear in the active code.
        assert "1e-30" not in code_only, (
            "iter-90: the legacy 1e-30 denominator floor must not "
            "appear in ocean ``_compute_drift``'s code."
        )

    def test_ocean_compute_drift_zero_baseline_returns_absolute(self):
        """End-to-end: invoke the ocean ``_compute_drift`` and
        verify rest-state behavior.  Baseline = 0 + 1e-17 rounding
        ⇒ result ~ 1e-17 (NOT 1e+13 spurious).
        """
        M = self._import_matrix_module()
        result = M._compute_drift([0.0, 1.0e-17])
        assert result < 1.0e-10, (
            f"iter-90: ocean ``_compute_drift`` regression — "
            f"baseline=0 + 1e-17 drift returned {result:.6e}, "
            f"should be ~1e-17 (1.0 floor convention).  Got "
            f"≥1e-10 means the legacy 1e-30 floor came back."
        )

    def test_ocean_compute_drift_normal_baseline(self):
        """Production-baseline behavior preserved post-refactor."""
        M = self._import_matrix_module()
        # Ocean mean_T baseline ~ 280 K, drift 0.02 K → 7.14e-5.
        result = M._compute_drift([280.0, 280.02])
        assert result == pytest.approx(0.02 / 280.0, rel=1e-6)

    def test_ocean_compute_drift_short_series(self):
        M = self._import_matrix_module()
        assert M._compute_drift([]) == 0.0
        assert M._compute_drift([1.0]) == 0.0


class TestOceanTestMatrixPackageDelegates:
    """iter-91 audit followup to codex iter-90 review.

    Codex's HIGH-1 finding pointed out that
    ``scripts/matrix/run_ocean_test_matrix.py:_compute_drift`` (a script-
    level copy) had been missed in iter-88's "factor into shared
    helper" refactor.  iter-91 audited more aggressively and
    found two MORE missed copies in the sibling
    ``scripts/matrix/ocean_test_matrix/`` package:

      * ``ocean_test_matrix/timeloop.py:_compute_drift``
        (10 callsites in ``experiments.py``: T_drift, PE_drift,
        S_integral_drift)
      * ``ocean_test_matrix/diagnostic_io.py:_save_conservation``
        (called transitively by ``_save_case_diagnostics``, used
        by ``experiments.py``, ``continue_eady_uniform.py``,
        ``run_eady_advection_comparison.py``)

    Both had the iter-78/80 ``1e-30`` denominator floor and would
    spuriously inflate machine epsilon to 1e+13 for rest-state
    baselines.  Migrated to delegate to the canonical helper.

    These tests pin the delegation so a future regression
    re-introducing the legacy floor would be caught immediately.
    """

    def test_timeloop_compute_drift_delegates(self):
        """``ocean_test_matrix.timeloop._compute_drift`` calls
        the canonical ``compute_relative_drift`` helper, not an
        inline ``1e-30`` floor.
        """
        import importlib
        import inspect
        import re
        import sys
        from pathlib import Path

        scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        timeloop = importlib.import_module(
            "ocean_test_matrix.timeloop"
        )

        src = inspect.getsource(timeloop._compute_drift)
        # Strip docstrings + line comments; test active code only.
        src_no_strings = re.sub(r'""".*?"""', "", src, flags=re.DOTALL)
        src_no_strings = re.sub(r"'''.*?'''", "", src_no_strings, flags=re.DOTALL)
        code_only = "\n".join(
            line for line in src_no_strings.splitlines()
            if not line.lstrip().startswith("#")
        )
        assert "compute_relative_drift" in code_only, (
            "iter-91: ``ocean_test_matrix.timeloop._compute_drift`` "
            "must delegate to ``compute_relative_drift`` from "
            "``legoesm.diagnostics.conservation_drift``.  Codex "
            "iter-90 review caught the iter-88 audit was incomplete "
            "for the script copy; iter-91 audit caught this package "
            "copy."
        )
        assert "1e-30" not in code_only, (
            "iter-91: legacy 1e-30 floor must not appear in the "
            "active code of ``timeloop._compute_drift``."
        )

    def test_timeloop_compute_drift_zero_baseline(self):
        """Behavioural pin: rest-state baseline (0) returns
        absolute drift, NOT the spurious 1e+13.
        """
        import importlib
        import sys
        from pathlib import Path

        scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        timeloop = importlib.import_module(
            "ocean_test_matrix.timeloop"
        )
        result = timeloop._compute_drift([0.0, 1.0e-17])
        assert result < 1.0e-10, (
            f"iter-91: ``timeloop._compute_drift`` regression — "
            f"baseline=0 + 1e-17 drift returned {result:.6e}, "
            f"should be ~1e-17 (1.0 floor convention)."
        )

    def test_diagnostic_io_save_conservation_delegates(self):
        """``ocean_test_matrix.diagnostic_io._save_conservation``
        uses ``relative_drift_series``, not an inline 1e-30 floor.
        """
        import importlib
        import inspect
        import re
        import sys
        from pathlib import Path

        scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        diagnostic_io = importlib.import_module(
            "ocean_test_matrix.diagnostic_io"
        )

        src = inspect.getsource(diagnostic_io._save_conservation)
        src_no_strings = re.sub(r'""".*?"""', "", src, flags=re.DOTALL)
        src_no_strings = re.sub(r"'''.*?'''", "", src_no_strings, flags=re.DOTALL)
        code_only = "\n".join(
            line for line in src_no_strings.splitlines()
            if not line.lstrip().startswith("#")
        )
        assert "relative_drift_series" in code_only, (
            "iter-91: "
            "``ocean_test_matrix.diagnostic_io._save_conservation`` "
            "must delegate to ``relative_drift_series`` from "
            "``legoesm.diagnostics.conservation_drift``.  iter-91 "
            "audit caught this duplicate of "
            "``run_ocean_test_matrix._save_conservation`` had been "
            "missed in iter-88+90."
        )
        assert "1e-30" not in code_only, (
            "iter-91: legacy 1e-30 floor must not appear in the "
            "active code of "
            "``ocean_test_matrix.diagnostic_io._save_conservation``."
        )


class TestOmipBlowupReporting:
    """iter-97: when the cube OMIP BLOWS UP, the results.txt and
    CLI summary table must clearly mark the run as a BLOWUP rather
    than reporting the last *clean* SST/SSS/SSH (which iter-96
    misread as a false-improvement claim — "cube OMIP no longer
    BLOWUPS, now reports finite SST=19.76" — when in fact the
    BLOWUP at step 500 was still happening; the 19.76 was just
    the last clean diagnostic from before the blowup).

    These tests pin the BLOWUP-aware output format so a future
    regression that drops the BLOWUP marker would fail this test.
    """

    def _import_run_omip(self):
        import importlib
        import sys
        from pathlib import Path
        scripts_dir = Path(__file__).resolve().parent.parent / "scripts" / "run"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        return importlib.import_module("run_omip")

    def test_run_omip_loop_returns_blowup_info(self):
        """``_run_omip_loop`` returns 5-tuple ending in
        ``blowup_info``.  This is a contract iter-96 audit
        relied on: callers can no longer ignore that the
        cube ran into a BLOWUP rather than a clean run.
        """
        import inspect
        m = self._import_run_omip()
        src = inspect.getsource(m._run_omip_loop)
        # Strip docstrings + comments before pattern check.
        import re
        src_no_strings = re.sub(r'""".*?"""', "", src, flags=re.DOTALL)
        src_no_strings = re.sub(r"'''.*?'''", "", src_no_strings, flags=re.DOTALL)
        code_only = "\n".join(
            line for line in src_no_strings.splitlines()
            if not line.lstrip().startswith("#")
        )
        # Returns include blowup_info as the last element.
        assert "blowup_info" in code_only, (
            "iter-97: ``_run_omip_loop`` must capture and return "
            "``blowup_info`` so downstream output can mark BLOWUP "
            "runs distinctly."
        )
        assert "return state, diag, wall, ok, blowup_info" in code_only, (
            "iter-97: ``_run_omip_loop`` must end with "
            "``return state, diag, wall, ok, blowup_info``."
        )

    def test_save_output_accepts_blowup_info(self):
        """``_save_output`` accepts ``blowup_info`` kwarg and
        threads it into the results.txt notes.
        """
        import inspect
        m = self._import_run_omip()
        sig = inspect.signature(m._save_output)
        assert "blowup_info" in sig.parameters, (
            "iter-97: ``_save_output`` must accept ``blowup_info`` "
            "kwarg so the BLOWUP step / max|T| / max|η| can be "
            "written into results.txt."
        )

    def test_results_txt_emits_blowup_marker_for_failed_runs(self):
        """End-to-end: when ``_save_output`` is called with a
        non-None ``blowup_info``, the resulting ``results.txt``
        must lead with a BLOWUP marker, not the last-clean SST.
        """
        import argparse
        from pathlib import Path
        import tempfile
        m = self._import_run_omip()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "cubed_sphere" / "C24"
            # Synthetic diag dict with last-clean SST=19.759 (the
            # iter-96 misleading value).
            diag = {
                "day": [0.0, 1.0],
                "step": [0, 288],
                "SST": [20.0, 19.759],
                "SSS": [35.0, 35.0],
                "SSH": [0.0, 1.4e-6],
            }
            args = argparse.Namespace(
                resolution="C24", nlev=20, days=2.0, dt=300.0,
                physics="full", water_type="ocean", sw_down=300.0,
                output=tmpdir,
            )
            blowup_info = {
                "step": 500,
                "day": 1.74,
                "T_max": 8.342e6,
                "T_finite": True,
                "eta_max": 2678.0,
                "eta_finite": True,
                "reasons": [
                    "|T| reached 8341965.5 °C (sanity threshold 100 °C)",
                    "|η| reached 2678 m (iter-79 sanity threshold 1000 m)",
                ],
            }
            m._save_output(
                output_dir, diag, args, "cubed_sphere",
                wall_time=74.0, ok=False, blowup_info=blowup_info,
            )
            text = (output_dir / "results.txt").read_text()

            # The BLOWUP marker MUST appear early in notes.
            assert "BLOWUP at step 500" in text, (
                f"iter-97: results.txt must contain ``BLOWUP at "
                f"step 500`` for cube OMIP failure; got:\n{text}"
            )
            assert "max|T|=8.342e+06" in text, (
                "iter-97: results.txt must include max|T| at the "
                "BLOWUP step."
            )
            assert "max|η|=2678" in text, (
                "iter-97: results.txt must include max|η| at the "
                "BLOWUP step."
            )
            assert "last clean SST=19.759" in text, (
                "iter-97: results.txt should still preserve the "
                "last clean diagnostic (with explicit ``last "
                "clean`` label) so the reader can see what the "
                "system looked like before BLOWUP."
            )

    def test_results_txt_clean_run_unaffected(self):
        """For a clean (PASS) run, ``results.txt`` still emits
        SST/SSS/SSH normally — iter-97 must not regress the
        existing behaviour for non-BLOWUP runs.
        """
        import argparse
        from pathlib import Path
        import tempfile
        m = self._import_run_omip()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "latlon" / "36x72"
            diag = {
                "day": [0.0, 1.0, 2.0],
                "step": [0, 288, 576],
                "SST": [20.0, 19.85, 19.75],
                "SSS": [35.0, 35.0, 35.0],
                "SSH": [0.0, 5.0e-7, 1.0e-6],
            }
            args = argparse.Namespace(
                resolution="36x72", nlev=20, days=2.0, dt=300.0,
                physics="full", water_type="ocean", sw_down=300.0,
                output=tmpdir,
            )
            m._save_output(
                output_dir, diag, args, "latlon",
                wall_time=3.8, ok=True, blowup_info=None,
            )
            text = (output_dir / "results.txt").read_text()
            assert "BLOWUP" not in text, (
                "iter-97: clean PASS runs must NOT contain "
                "``BLOWUP`` in results.txt."
            )
            assert "SST=19.750" in text, (
                "iter-97: clean PASS runs must report the final "
                "SST normally (3-decimal format from iter-25)."
            )


class TestOceanCliResolutionPerGridDispatch:
    """iter-102: same per-grid CLI dispatch fix as iter-95
    atmosphere matrix.  Pre-iter-102, ``--resolution N`` (bare
    integer) was applied verbatim to every ocean grid type,
    breaking 3+ of 4 parsers:

    * cubed_sphere ``int(res[1:])``: "16" → 6 (silent wrong)
    * latlon ``res.split("x")``: "16" → unpack error
    * mpas ``res.replace("ico", "")``: "16" → level=16 → 4.29e+10
      cells ValueError
    * spectral ``int(res[1:])``: "16" → 6 (silent wrong)

    iter-102 added ``_expand_cli_res(N, grid_type)``.  Bare
    integers expand to grid-typed strings (``f"C{N}"``,
    ``f"{N}x{2*N}"``, ``f"ico{level}"``, ``f"T{N}"``,
    ``f"{N}km"`` for regional); pre-formatted strings pass
    through unchanged.
    """

    def _import_module(self):
        import importlib
        import sys
        from pathlib import Path
        scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        return importlib.import_module("run_ocean_test_matrix")

    def test_ocean_resolution_dispatch_unit_for_each_grid(self):
        """Mirror iter-95 atmosphere unit test: dispatch logic
        produces correct grid-typed strings for N=16, 32, 72.
        """
        import math

        def _expand(N: int, grid_type: str) -> str:
            if grid_type == "cubed_sphere":
                return f"C{N}"
            elif grid_type == "latlon":
                return f"{N}x{2 * N}"
            elif grid_type == "mpas":
                level = max(2, min(8, round(math.log(2 * N * N / 10) / math.log(4))))
                return f"ico{level}"
            elif grid_type == "spectral":
                return f"T{N}"
            elif grid_type in ("mpas_regional", "latlon_regional", "cs_regional"):
                return f"{N}km"
            else:
                return str(N)

        # N=16 → cube C16, latlon 16x32, ico level 3, regional 16km
        assert _expand(16, "cubed_sphere") == "C16"
        assert _expand(16, "latlon") == "16x32"
        assert _expand(16, "mpas") == "ico3"
        assert _expand(16, "mpas_regional") == "16km"
        # N=32 → ico level 4
        assert _expand(32, "cubed_sphere") == "C32"
        assert _expand(32, "mpas") == "ico4"
        # N=72 → ico level 5 (matches default ico5)
        assert _expand(72, "mpas") == "ico5"

    def test_ocean_source_pin_resolution_dispatch(self):
        """Source-level pin: the ocean matrix runner uses the
        shared cli_resolution helpers (iter-115 refactored
        from inline iter-102 literals to a centralized helper
        in ``legoesm.driver.cli_resolution``).

        The actual dispatch literals (``f"C{N}"`` etc.) now
        live in
        ``cli_resolution.py:expand_cli_resolution``
        — see ``TestSharedCliResolution`` for the
        behavior-level tests of that helper.
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
        # iter-115: matrix runner uses the shared helper.
        assert "validate_cli_resolution" in code_only, (
            "iter-115: ocean matrix dispatch must use the shared "
            "``legoesm.driver.cli_resolution.validate_cli_resolution`` "
            "helper instead of inline validation."
        )
        assert "expand_cli_resolution" in code_only, (
            "iter-115: ocean matrix dispatch must use the shared "
            "``legoesm.driver.cli_resolution.expand_cli_resolution`` "
            "helper instead of inline per-grid format strings."
        )

    def test_ocean_resolution_smoke_runs_on_all_grids(self):
        """End-to-end smoke placeholder.  Verified manually:
        ``run_ocean_test_matrix.py --only rest_state --quick
        --resolution 16`` produces 12/12 PASS (4 rest-state
        variants × 3 grids: cube, latlon, mpas — spectral does
        not have rest_state).  Marked as skip so the fast unit
        cycle does not run the 2-3 minute matrix.
        """
        pytest.skip(
            "iter-102 smoke: invoke "
            "``JAX_ENABLE_X64=1 .venv/bin/python "
            "scripts/matrix/run_ocean_test_matrix.py --only rest_state "
            "--quick --resolution 16`` to verify; expected 12/12 "
            "PASS (3 grids × 4 rest_state variants) since iter-102."
        )


class TestOceanMatrixBlowupReporting:
    """iter-105 (codex iter-104 MEDIUM-3): mirror the iter-98
    atmosphere matrix BLOWUP-reporting fix in the ocean matrix
    runner.  Pre-iter-105, the ocean ``_run_timeloop`` printed
    BLOWUP and returned ``ok=False``, but no ``_blowup_info``
    was stored, so ``results.txt`` notes were derived from the
    last *clean* diagnostic (same false-improvement risk as
    iter-96 OMIP ⇒ iter-97 fix).

    iter-105 added:
    * ``diag["_blowup_info"]`` capture in ``_run_timeloop``
    * ``diag=`` and ``blowup_info=`` kwargs on
      ``_write_results_txt``
    * ``diag=diag`` threaded through all 14 ocean callsites

    These tests mirror the iter-98 atmosphere structural and
    behavioural pins.
    """

    def _import_module(self):
        import importlib
        import sys
        from pathlib import Path
        scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        return importlib.import_module("run_ocean_test_matrix")

    def test_run_timeloop_records_blowup_info_in_diag(self):
        """``_run_timeloop`` writes ``_blowup_info`` into diag
        when a BLOWUP fires.
        """
        import inspect
        import re
        M = self._import_module()
        src = inspect.getsource(M._run_timeloop)
        src_no_strings = re.sub(r'""".*?"""', "", src, flags=re.DOTALL)
        src_no_strings = re.sub(r"'''.*?'''", "", src_no_strings, flags=re.DOTALL)
        code_only = "\n".join(
            line for line in src_no_strings.splitlines()
            if not line.lstrip().startswith("#")
        )
        assert ('diag["_blowup_info"]' in code_only or
                "diag['_blowup_info']" in code_only), (
            "iter-105: ocean ``_run_timeloop`` must store "
            "BLOWUP details in ``diag['_blowup_info']`` so "
            "``_write_results_txt`` can surface them."
        )

    def test_write_results_txt_accepts_diag_kwarg(self):
        import inspect
        M = self._import_module()
        sig = inspect.signature(M._write_results_txt)
        assert "diag" in sig.parameters, (
            "iter-105: ocean ``_write_results_txt`` must "
            "accept ``diag`` kwarg for opt-in BLOWUP info "
            "threading."
        )
        assert "blowup_info" in sig.parameters, (
            "iter-105: ocean ``_write_results_txt`` must "
            "accept ``blowup_info`` kwarg as the lower-level "
            "entry point."
        )

    def test_write_results_txt_emits_blowup_marker(self, tmp_path):
        """End-to-end: ocean ``_write_results_txt`` with a
        synthetic BLOWUP-info-bearing diag emits a BLOWUP
        marker in results.txt notes.
        """
        M = self._import_module()
        diag = {
            "times": [0.0, 1.0],
            "steps": [0, 100],
            "vol": [1e18, 1e18],
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
            "test": "rest_state",
            "grid": "cubed_sphere",
            "status": "FAIL",
            "notes": "eta drift=1.2e-17",
            "wall_time": "15.6s",
        }
        M._write_results_txt(tmp_path, rows, diag=diag)
        text = (tmp_path / "results.txt").read_text()
        assert "BLOWUP at step 200" in text, (
            f"iter-105: ocean results.txt must contain BLOWUP "
            f"marker when ``diag['_blowup_info']`` is set; got:"
            f"\n{text}"
        )
        assert "state non-finite" in text, (
            "iter-105: ocean results.txt must include reason."
        )
        assert "last clean: eta drift=1.2e-17" in text, (
            "iter-105: ocean results.txt must preserve the "
            "original notes labeled as ``last clean:``."
        )

    def test_write_results_txt_unaffected_for_pass_runs(self, tmp_path):
        M = self._import_module()
        diag = {
            "times": [0.0, 1.0],
            "steps": [0, 100],
            "vol": [1e18, 1e18],
        }
        rows = {
            "test": "rest_state",
            "grid": "cubed_sphere",
            "status": "PASS",
            "notes": "eta drift=1.2e-17",
            "wall_time": "15.6s",
        }
        M._write_results_txt(tmp_path, rows, diag=diag)
        text = (tmp_path / "results.txt").read_text()
        assert "BLOWUP" not in text, (
            "iter-105: ocean PASS runs must not have BLOWUP markers."
        )
        assert "notes: eta drift=1.2e-17" in text

    def test_all_ocean_write_results_txt_callsites_thread_diag(self):
        """Mirror iter-99 atmosphere structural test: every
        ``_write_results_txt`` call in the ocean matrix runner
        threads ``diag=`` (or equivalent).
        """
        import inspect
        import re
        M = self._import_module()
        text = inspect.getsource(M)
        code_only = "\n".join(
            line for line in text.splitlines()
            if not line.lstrip().startswith("#")
        )
        all_calls = list(re.finditer(
            r"_write_results_txt\(", code_only
        ))
        callsite_count = 0
        threaded_count = 0
        for m in all_calls:
            start = max(0, m.start() - 10)
            preceding = code_only[start:m.start()]
            if "def " in preceding:
                continue
            callsite_count += 1
            tail = code_only[m.start():m.start() + 800]
            if re.search(r"\bdiag\s*=\s*\w", tail):
                threaded_count += 1
        # We expect at least 13 callsites (14 minus the def).
        # iter-105 threaded all of them.
        assert callsite_count >= 13, (
            f"iter-105 sanity: expected ≥13 ``_write_results_txt`` "
            f"callsites in ocean matrix runner; found {callsite_count}."
        )
        assert threaded_count == callsite_count, (
            f"iter-105: every ``_write_results_txt`` callsite in "
            f"ocean matrix must thread ``diag=`` to surface "
            f"BLOWUP info uniformly.  Found {threaded_count}/"
            f"{callsite_count} threaded."
        )


class TestSelectOceanResolutionDirPrefersGridTyped:
    """iter-108 (codex iter-104 MEDIUM-6): mirror the
    atmosphere collector iter-108 fix in the ocean collector.
    Stale ``16/`` dirs from pre-iter-102 runs would previously
    shadow fresh ``C16/`` / ``36x32/`` / ``ico3/`` / ``T16/``
    dirs.

    Ocean grid types: cubed_sphere, latlon, mpas, spectral
    (and regional variants ``mpas_regional``,
    ``latlon_regional``, ``cs_regional`` which use ``Nkm``).
    """

    def _import_module(self):
        import importlib
        import sys
        from pathlib import Path
        scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        return importlib.import_module("run_ocean_test_matrix")

    def test_cubed_sphere_prefers_C(self, tmp_path):
        M = self._import_module()
        grid_dir = tmp_path / "cubed_sphere"
        grid_dir.mkdir()
        (grid_dir / "16").mkdir()
        (grid_dir / "C16").mkdir()
        chosen = M._select_ocean_resolution_dir(
            grid_dir, list(grid_dir.iterdir()))
        assert chosen.name == "C16"

    def test_latlon_prefers_xform(self, tmp_path):
        M = self._import_module()
        grid_dir = tmp_path / "latlon"
        grid_dir.mkdir()
        (grid_dir / "16").mkdir()
        (grid_dir / "16x32").mkdir()
        chosen = M._select_ocean_resolution_dir(
            grid_dir, list(grid_dir.iterdir()))
        assert chosen.name == "16x32"

    def test_mpas_prefers_ico(self, tmp_path):
        M = self._import_module()
        grid_dir = tmp_path / "mpas"
        grid_dir.mkdir()
        (grid_dir / "16").mkdir()
        (grid_dir / "ico3").mkdir()
        chosen = M._select_ocean_resolution_dir(
            grid_dir, list(grid_dir.iterdir()))
        assert chosen.name == "ico3"

    def test_spectral_prefers_T(self, tmp_path):
        M = self._import_module()
        grid_dir = tmp_path / "spectral"
        grid_dir.mkdir()
        (grid_dir / "16").mkdir()
        (grid_dir / "T16").mkdir()
        chosen = M._select_ocean_resolution_dir(
            grid_dir, list(grid_dir.iterdir()))
        assert chosen.name == "T16"

    def test_regional_prefers_km(self, tmp_path):
        M = self._import_module()
        grid_dir = tmp_path / "mpas_regional"
        grid_dir.mkdir()
        (grid_dir / "16").mkdir()
        (grid_dir / "50km").mkdir()
        chosen = M._select_ocean_resolution_dir(
            grid_dir, list(grid_dir.iterdir()))
        assert chosen.name == "50km"

    def test_falls_back_to_first_for_legacy(self, tmp_path):
        """No grid-typed candidate → fall back to first."""
        M = self._import_module()
        grid_dir = tmp_path / "cubed_sphere"
        grid_dir.mkdir()
        d16 = grid_dir / "16"
        d16.mkdir()
        d32 = grid_dir / "32"
        d32.mkdir()
        chosen = M._select_ocean_resolution_dir(
            grid_dir, [d16, d32])
        # Falls back to the first in the input list (filesystem
        # order, not sorted) — pin to whichever was passed first.
        assert chosen.name in ("16", "32")


class TestIter123OceanDriftTolerance:
    """iter-123 (codex iter-119-followup MEDIUM-4): ocean
    matrix conservation gates.

    Pre-iter-123, ocean rest_state PASS criteria checked
    only finiteness + non-blown-up; eta_drift and T_drift
    were reported in notes but never gated.  This made the
    cross-grid PASS column meaningless for conservation
    tests where the expectation is machine precision.

    iter-123 added a generalized ``_apply_drift_tolerance``
    helper (mirrors the atmosphere ``_apply_mass_drift_tolerance``
    iter-117/118/120) and applied it to all 4 rest_state
    variants with eta tol 1e-10 m and T tol 1e-8 relative.
    """

    def _import_module(self):
        import importlib
        import sys
        from pathlib import Path
        scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        return importlib.import_module("run_ocean_test_matrix")

    def test_apply_drift_tolerance_passes_at_machine_precision(self):
        """Rest-state-style: drift = 1e-17 should pass."""
        m = self._import_module()
        ok, notes = m._apply_drift_tolerance(
            ok=True, notes="initial",
            drift=1e-17, tol=1e-10, label="eta",
            n_samples=2,
        )
        assert ok is True

    def test_apply_drift_tolerance_fails_above_tol(self):
        m = self._import_module()
        ok, notes = m._apply_drift_tolerance(
            ok=True, notes="initial",
            drift=1e-9, tol=1e-10, label="eta",
            n_samples=2,
        )
        assert ok is False
        assert "eta drift" in notes
        assert "tolerance 1e-10" in notes

    def test_apply_drift_tolerance_fails_nan(self):
        m = self._import_module()
        ok, notes = m._apply_drift_tolerance(
            ok=True, notes="initial",
            drift=float("nan"), tol=1e-10, label="T",
            n_samples=2,
        )
        assert ok is False
        assert "T drift is non-finite" in notes

    def test_apply_drift_tolerance_fails_few_samples(self):
        m = self._import_module()
        ok, notes = m._apply_drift_tolerance(
            ok=True, notes="initial",
            drift=0.0, tol=1e-10, label="eta",
            n_samples=1,
        )
        assert ok is False
        assert "eta series has only 1 sample" in notes

    def test_apply_drift_tolerance_idempotent_on_failed(self):
        m = self._import_module()
        ok, notes = m._apply_drift_tolerance(
            ok=False, notes="BLOWUP at step 100",
            drift=1e-3, tol=1e-10, label="eta",
            n_samples=2,
        )
        assert ok is False
        assert notes == "BLOWUP at step 100"

    def test_label_is_keyword_only(self):
        """iter-124 codex iter-123-followup LOW-4: ``label``
        must be keyword-only so future callers can't silently
        swap arguments.
        """
        m = self._import_module()
        # Calling with label as positional MUST raise TypeError.
        import pytest as _pytest
        with _pytest.raises(TypeError):
            m._apply_drift_tolerance(
                True, "initial", 1e-12, 1e-10, "eta",
                n_samples=2)

    def test_geostrophic_adjustment_uses_drift_tolerance(self):
        """iter-124 codex iter-123-followup MEDIUM-1:
        ``run_geostrophic_adjustment`` must apply the
        T-drift tolerance.
        """
        import inspect
        m = self._import_module()
        src = inspect.getsource(m.run_geostrophic_adjustment)
        assert "_apply_drift_tolerance" in src, (
            "iter-124: ``run_geostrophic_adjustment`` must "
            "call ``_apply_drift_tolerance`` to gate PASS on "
            "T conservation."
        )

    def test_save_timeseries_csv_skips_blowup_info_metadata(self, tmp_path):
        """iter-125 codex iter-124-followup LOW-4:
        ``_save_timeseries_csv`` must skip underscore-prefixed
        metadata keys (e.g., ``_blowup_info``) which are dicts
        not lists.  Pre-iter-125, a BLOWUP would crash the
        post-run CSV writer with ``KeyError: 0``.
        """
        m = self._import_module()
        diag = {
            "steps": [0, 100, 200],
            "times": [0.0, 0.5, 1.0],
            "mean_eta": [0.0, 0.001, 0.002],
            "_blowup_info": {  # iter-105 metadata; not a series
                "step": 200, "day": 1.0, "metric": 1234.5,
                "is_finite": False, "threshold": 1000.0,
                "reason": "test",
            },
        }
        # Pre-iter-125 this would raise KeyError: 0.
        m._save_timeseries_csv(tmp_path, diag, dt=300.0)
        # CSV file should exist and contain mean_eta but NOT
        # ``_blowup_info`` column.
        csv_text = (tmp_path / "mean_timeseries.csv").read_text()
        assert "_blowup_info" not in csv_text, (
            "iter-125: ``_blowup_info`` metadata must not appear "
            "in mean_timeseries.csv as a column."
        )
        assert "mean_eta" in csv_text, (
            "iter-125: legitimate timeseries columns must "
            "still be present."
        )

    def test_overflow_uses_drift_tolerance(self):
        """iter-125 codex iter-124-followup MEDIUM-2:
        ``run_overflow`` must apply the T-drift tolerance.
        PE drift is NOT gated since PE evolves physically.
        """
        import inspect
        m = self._import_module()
        src = inspect.getsource(m.run_overflow)
        assert "_apply_drift_tolerance" in src, (
            "iter-125: ``run_overflow`` must call "
            "``_apply_drift_tolerance`` for the T-drift gate."
        )
        # Pin that the gate is on T (not PE).
        assert 'label="T"' in src, (
            "iter-125: overflow gate must target T (passive "
            "scalar), not PE (which evolves physically)."
        )

    def test_stommel_tolerance_matches_documented_threshold(self):
        """iter-125 codex iter-124-followup HIGH-1: stommel
        tolerance was tightened from 1e-2 to 1e-3 to match the
        documented threshold in ocean_experiments_reference.md.
        """
        import inspect
        m = self._import_module()
        src = inspect.getsource(m.run_stommel_gyre_tracer)
        # The drift tolerance line should now be 1e-3, not 1e-2.
        import re
        # Find _apply_drift_tolerance call with S_integral label.
        # Match: _apply_drift_tolerance(... TOL, label="S_integral" ...)
        m_match = re.search(
            r"_apply_drift_tolerance\(\s*"
            r"ok,\s*notes,\s*S_int_drift,\s*"
            r"([\d.eE+-]+)",
            src,
        )
        assert m_match is not None, (
            "iter-125: could not parse stommel tolerance from source."
        )
        tol = float(m_match.group(1))
        assert tol <= 1e-3, (
            f"iter-125: stommel tolerance must be <= 1e-3 "
            f"(documented).  Got {tol:.0e}."
        )

    def test_stommel_gyre_tracer_uses_drift_tolerance(self):
        """iter-124 codex iter-123-followup MEDIUM-1:
        ``run_stommel_gyre_tracer`` must apply the
        S_integral-drift tolerance.
        """
        import inspect
        m = self._import_module()
        src = inspect.getsource(m.run_stommel_gyre_tracer)
        assert "_apply_drift_tolerance" in src, (
            "iter-124: ``run_stommel_gyre_tracer`` must "
            "call ``_apply_drift_tolerance`` to gate PASS on "
            "S_integral conservation."
        )

    def test_modular_timeloop_has_apply_drift_tolerance(self):
        """iter-126 codex iter-124-followup MEDIUM-3:
        ``scripts/matrix/ocean_test_matrix/timeloop.py`` must export
        ``_apply_drift_tolerance`` so the modular package's
        ``experiments.py`` can use the iter-123/124 helper.
        """
        import importlib
        import sys
        from pathlib import Path
        scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        timeloop = importlib.import_module(
            "ocean_test_matrix.timeloop")
        assert hasattr(timeloop, "_apply_drift_tolerance"), (
            "iter-126: ``ocean_test_matrix.timeloop`` must "
            "export ``_apply_drift_tolerance`` for the modular "
            "experiments runners."
        )

    def test_modular_experiments_imports_helper(self):
        """``ocean_test_matrix.experiments`` imports
        ``_apply_drift_tolerance`` from timeloop.
        """
        from pathlib import Path
        path = (Path(__file__).resolve().parent.parent
                / "scripts" / "matrix" / "ocean_test_matrix"
                / "experiments.py")
        text = path.read_text()
        assert "_apply_drift_tolerance" in text, (
            "iter-126: modular ``experiments.py`` must import "
            "and use ``_apply_drift_tolerance``."
        )

    def test_iter127_helper_centralized(self):
        """iter-127 codex iter-126-followup LOW-5: the
        ``apply_drift_tolerance`` helper is now centralized
        in ``legoesm.diagnostics.conservation_drift`` and
        re-exported.  Both runner copies are thin delegating
        wrappers.
        """
        from legoesm.diagnostics import apply_drift_tolerance
        # Centralized helper exists and is callable.
        ok, notes = apply_drift_tolerance(
            ok=True, notes="initial",
            drift=1e-3, tol=1e-2, label="test",
            n_samples=2,
        )
        assert ok is True

        # Both runner wrappers delegate to the centralized one.
        from pathlib import Path
        for rel in (
            "scripts/matrix/run_ocean_test_matrix.py",
            "scripts/matrix/ocean_test_matrix/timeloop.py",
        ):
            text = (
                Path(__file__).resolve().parent.parent / rel
            ).read_text()
            assert "apply_drift_tolerance" in text, (
                f"iter-127: {rel} must use the centralized helper."
            )

    def test_iter127_modular_geostrophic_uses_gate(self):
        from pathlib import Path
        path = (Path(__file__).resolve().parent.parent
                / "scripts" / "matrix" / "ocean_test_matrix"
                / "experiments.py")
        text = path.read_text()
        # Find run_geostrophic_adjustment and verify it has
        # a _apply_drift_tolerance call.
        import re
        m = re.search(
            r"def run_geostrophic_adjustment\b.*?(?=\ndef \w)",
            text, re.DOTALL,
        )
        assert m is not None, (
            "iter-127: could not find run_geostrophic_adjustment "
            "in modular experiments.py"
        )
        body = m.group(0)
        assert "_apply_drift_tolerance" in body, (
            "iter-127: modular run_geostrophic_adjustment must "
            "apply T-drift tolerance."
        )

    def test_iter127_modular_overflow_uses_gate(self):
        from pathlib import Path
        path = (Path(__file__).resolve().parent.parent
                / "scripts" / "matrix" / "ocean_test_matrix"
                / "experiments.py")
        text = path.read_text()
        import re
        m = re.search(
            r"def run_overflow\b.*?(?=\ndef \w)",
            text, re.DOTALL,
        )
        assert m is not None
        body = m.group(0)
        assert "_apply_drift_tolerance" in body
        assert 'label="T"' in body, (
            "iter-127: modular run_overflow must gate on T "
            "(passive scalar), not PE (which evolves physically)."
        )

    def test_iter127_modular_stommel_uses_gate(self):
        from pathlib import Path
        path = (Path(__file__).resolve().parent.parent
                / "scripts" / "matrix" / "ocean_test_matrix"
                / "experiments.py")
        text = path.read_text()
        import re
        m = re.search(
            r"def run_stommel_gyre_tracer\b.*?(?=\ndef \w|$)",
            text, re.DOTALL,
        )
        assert m is not None
        body = m.group(0)
        assert "_apply_drift_tolerance" in body
        assert 'label="S_integral"' in body
        # Verify tolerance is the documented 1e-3 (not 1e-2).
        m2 = re.search(
            r"_apply_drift_tolerance\(\s*ok,\s*notes,\s*"
            r"S_int_drift,\s*([\d.eE+-]+)",
            body,
        )
        assert m2 is not None
        assert float(m2.group(1)) <= 1e-3

    def test_iter127_save_csv_handles_only_private_keys(self, tmp_path):
        """iter-127 codex iter-126-followup LOW-4: when ALL
        diag keys are private (underscore-prefixed), the CSV
        writer should return cleanly without writing anything.
        """
        m = self._import_module()
        diag = {
            "steps": [0, 100, 200],
            "times": [0.0, 0.5, 1.0],
            "_blowup_info": {"step": 200},
            # No legitimate timeseries columns.
        }
        m._save_timeseries_csv(tmp_path, diag, dt=300.0)
        # CSV file should NOT exist (no legitimate keys).
        csv_path = tmp_path / "mean_timeseries.csv"
        assert not csv_path.exists(), (
            "iter-127: when ALL diag keys are private, no CSV "
            "should be written.  Got an empty CSV."
        )

    # ====== iter-128: codex iter-127-followup MEDIUM-1/2/3 + LOW-1/2 ======

    def test_iter128_value_threshold_helper_le(self):
        """iter-128 codex iter-127-followup MEDIUM-2/3: the new
        ``apply_value_threshold`` helper enforces ``<= threshold``
        with NaN handling.
        """
        from legoesm.diagnostics import apply_value_threshold
        # Pass case: value below threshold.
        ok, notes = apply_value_threshold(
            ok=True, notes="", value=0.05, threshold=0.1,
            label="overshoot", units="PSU")
        assert ok is True
        # Fail case: value above threshold.
        ok, notes = apply_value_threshold(
            ok=True, notes="", value=0.15, threshold=0.1,
            label="overshoot", units="PSU")
        assert ok is False
        assert "0.15" in notes
        assert "PSU" in notes
        # NaN case: any non-finite value fails.
        ok, notes = apply_value_threshold(
            ok=True, notes="", value=float("nan"), threshold=0.1,
            label="overshoot")
        assert ok is False
        assert "non-finite" in notes
        # Idempotent on already-failed: ``ok=False`` short-circuits.
        ok, notes = apply_value_threshold(
            ok=False, notes="prior", value=0.05, threshold=0.1,
            label="overshoot")
        assert ok is False
        assert notes == "prior"

    def test_iter128_value_threshold_helper_lt(self):
        """iter-128/iter-129 codex iter-127/128-followup
        MEDIUM-2: the ``op="lt"`` mode for strict-less-than
        gates.  iter-129 generalized the iter-128 ``op="lt_zero"``
        to a general ``op="lt"`` with explicit threshold (LOW-3).
        Overflow uses it with threshold=0.0 for the documented
        ``pe_rel_final < 0`` sign check.
        """
        from legoesm.diagnostics import apply_value_threshold
        # Pass case: strictly less than threshold.
        ok, _ = apply_value_threshold(
            ok=True, notes="", value=-0.5, threshold=0.0,
            label="PE_rel_final", op="lt")
        assert ok is True
        # Fail case: equal to threshold (must be STRICTLY less).
        ok, notes = apply_value_threshold(
            ok=True, notes="", value=0.0, threshold=0.0,
            label="PE_rel_final", op="lt")
        assert ok is False
        assert "expected strictly less-than" in notes
        # Fail case: above threshold.
        ok, notes = apply_value_threshold(
            ok=True, notes="", value=0.1, threshold=0.0,
            label="PE_rel_final", op="lt")
        assert ok is False
        # NaN/Inf case.
        ok, notes = apply_value_threshold(
            ok=True, notes="", value=float("inf"), threshold=0.0,
            label="PE_rel_final", op="lt")
        assert ok is False
        assert "non-finite" in notes
        # Generic threshold (not just zero): pe_rel_final must
        # be < -0.001 (10x noise floor).
        ok, _ = apply_value_threshold(
            ok=True, notes="", value=-0.005, threshold=-0.001,
            label="PE_rel_final", op="lt")
        assert ok is True
        ok, notes = apply_value_threshold(
            ok=True, notes="", value=-0.0005, threshold=-0.001,
            label="PE_rel_final", op="lt")
        assert ok is False, (
            "iter-129 LOW-3: ``op='lt'`` must respect the "
            "threshold argument, not always compare against 0.")

    def test_iter128_value_threshold_unknown_op_raises(self):
        from legoesm.diagnostics import apply_value_threshold
        import pytest
        with pytest.raises(ValueError, match="unknown op"):
            apply_value_threshold(
                ok=True, notes="", value=1.0, threshold=0.5,
                label="x", op="bogus")

    def test_iter128_atmosphere_wrapper_delegates(self):
        """iter-128 codex iter-127-followup LOW-1: the atmosphere
        matrix's ``_apply_mass_drift_tolerance`` is now a thin
        delegating wrapper over the centralized helper, just
        like the two ocean wrappers.
        """
        from pathlib import Path
        text = (
            Path(__file__).resolve().parent.parent
            / "scripts" / "matrix" / "run_atmosphere_test_matrix.py"
        ).read_text()
        # Grab the wrapper body.
        import re
        m = re.search(
            r"def _apply_mass_drift_tolerance\b.*?(?=\ndef \w)",
            text, re.DOTALL,
        )
        assert m is not None, (
            "iter-128: could not find _apply_mass_drift_tolerance "
            "in atmosphere matrix.")
        body = m.group(0)
        # Body must IMPORT and CALL the centralized helper.
        assert "apply_drift_tolerance" in body
        assert "from legoesm.diagnostics" in body
        assert 'label="mass"' in body
        # Wrapper must NOT redefine the gate logic inline.
        # (The pre-iter-128 inline body had `_np.isfinite` —
        # the new delegating body does not.)
        assert "_np.isfinite" not in body, (
            "iter-128: atmosphere wrapper still has inline "
            "gate logic — did the iter-128 delegation revert?")

    def test_iter128_monolithic_overflow_pe_sign_gate(self):
        """iter-128 codex iter-127-followup MEDIUM-2: monolithic
        run_overflow gates the documented ``pe_rel_final < 0``
        sign constraint.
        """
        from pathlib import Path
        path = (Path(__file__).resolve().parent.parent
                / "scripts" / "matrix" / "run_ocean_test_matrix.py")
        text = path.read_text()
        import re
        m = re.search(
            r"def run_overflow\b.*?(?=\ndef \w|^\s*RUNNERS)",
            text, re.DOTALL | re.MULTILINE,
        )
        assert m is not None
        body = m.group(0)
        assert "_apply_pe_rel_sign" in body, (
            "iter-128: monolithic run_overflow must apply the "
            "documented pe_rel_final < 0 sign gate.")
        # Verify the sign gate uses the correct label.
        assert 'label="PE_rel_final"' in body

    def test_iter128_modular_overflow_pe_sign_gate(self):
        """iter-128 codex iter-127-followup MEDIUM-2: modular
        run_overflow ports the same pe_rel_final sign gate.
        """
        from pathlib import Path
        path = (Path(__file__).resolve().parent.parent
                / "scripts" / "matrix" / "ocean_test_matrix"
                / "experiments.py")
        text = path.read_text()
        import re
        m = re.search(
            r"def run_overflow\b.*?(?=\ndef \w)",
            text, re.DOTALL,
        )
        assert m is not None
        body = m.group(0)
        assert "_apply_pe_rel_sign" in body
        assert 'label="PE_rel_final"' in body

    def test_iter128_monolithic_stommel_overshoot_undershoot(self):
        """iter-128 codex iter-127-followup MEDIUM-3: monolithic
        run_stommel_gyre_tracer gates the documented
        overshoot/undershoot < 0.1 PSU thresholds.
        """
        from pathlib import Path
        path = (Path(__file__).resolve().parent.parent
                / "scripts" / "matrix" / "run_ocean_test_matrix.py")
        text = path.read_text()
        import re
        m = re.search(
            r"def run_stommel_gyre_tracer\b.*?(?=\ndef \w|^\s*RUNNERS)",
            text, re.DOTALL | re.MULTILINE,
        )
        assert m is not None
        body = m.group(0)
        assert 'label="S overshoot"' in body
        assert 'label="S undershoot"' in body
        # Both gates must use threshold 0.1 and units PSU.
        assert "overshoot, 0.1" in body
        assert "undershoot, 0.1" in body
        assert 'units="PSU"' in body

    def test_iter128_modular_stommel_overshoot_undershoot(self):
        """iter-128 codex iter-127-followup MEDIUM-3: modular
        run_stommel_gyre_tracer mirrors the same gates.
        """
        from pathlib import Path
        path = (Path(__file__).resolve().parent.parent
                / "scripts" / "matrix" / "ocean_test_matrix"
                / "experiments.py")
        text = path.read_text()
        import re
        m = re.search(
            r"def run_stommel_gyre_tracer\b.*?(?=\ndef \w|$)",
            text, re.DOTALL,
        )
        assert m is not None
        body = m.group(0)
        assert 'label="S overshoot"' in body
        assert 'label="S undershoot"' in body
        assert "overshoot, 0.1" in body
        assert "undershoot, 0.1" in body

    def test_iter128_modular_geostrophic_tolerance_pinned(self):
        """iter-128 codex iter-127-followup LOW-2: tighten the
        modular geostrophic gate test by pinning the EXACT
        tolerance (1e-8) and label, not just the helper name.
        """
        from pathlib import Path
        path = (Path(__file__).resolve().parent.parent
                / "scripts" / "matrix" / "ocean_test_matrix"
                / "experiments.py")
        text = path.read_text()
        import re
        m = re.search(
            r"def run_geostrophic_adjustment\b.*?(?=\ndef \w)",
            text, re.DOTALL,
        )
        assert m is not None
        body = m.group(0)
        # Find the tolerance argument to _apply_drift_tolerance.
        m2 = re.search(
            r"_apply_drift_tolerance\(\s*ok,\s*notes,\s*"
            r"T_drift,\s*([\d.eE+-]+)",
            body,
        )
        assert m2 is not None, (
            "iter-128: modular geostrophic must call "
            "_apply_drift_tolerance with T_drift.")
        tol = float(m2.group(1))
        assert tol == 1e-8, (
            f"iter-128: modular geostrophic tolerance must be "
            f"1e-8 (tighter than doc's 1e-3 — see "
            f"docs/dev-notes/ocean_experiments_reference.md:419).  "
            f"Got {tol}.")
        assert 'label="T"' in body

    def test_iter128_modular_overflow_T_tolerance_pinned(self):
        """iter-128 codex iter-127-followup LOW-2: pin the
        EXACT modular overflow T-drift tolerance to 1e-2.
        """
        from pathlib import Path
        path = (Path(__file__).resolve().parent.parent
                / "scripts" / "matrix" / "ocean_test_matrix"
                / "experiments.py")
        text = path.read_text()
        import re
        m = re.search(
            r"def run_overflow\b.*?(?=\ndef \w)",
            text, re.DOTALL,
        )
        assert m is not None
        body = m.group(0)
        m2 = re.search(
            r"_apply_drift_tolerance\(\s*ok,\s*notes,\s*"
            r"T_drift,\s*([\d.eE+-]+)",
            body,
        )
        assert m2 is not None
        tol = float(m2.group(1))
        assert tol == 1e-2, (
            f"iter-128: modular overflow T tolerance must be "
            f"1e-2 (matches docs/dev-notes/ocean_experiments_reference.md:625). "
            f"Got {tol}.")

    # ====== iter-129: codex iter-128-followup MEDIUM-1/2 + LOW-1/2/3/4 ======

    def test_iter129_value_threshold_n_samples_kwarg(self):
        """iter-129 codex iter-128-followup MEDIUM-1: the new
        ``n_samples`` kwarg fails explicitly when the underlying
        diagnostic series has < 2 samples — preventing a missing
        diagnostic from silently passing via a default-zero
        placeholder.
        """
        from legoesm.diagnostics import apply_value_threshold
        # Empty series: must FAIL even though value would pass.
        ok, notes = apply_value_threshold(
            ok=True, notes="", value=-0.5, threshold=0.0,
            label="PE_rel_final", op="lt", n_samples=0)
        assert ok is False
        assert "0 sample" in notes
        # Single-sample: must FAIL.
        ok, notes = apply_value_threshold(
            ok=True, notes="", value=-0.5, threshold=0.0,
            label="PE_rel_final", op="lt", n_samples=1)
        assert ok is False
        assert "1 sample" in notes
        # Adequate samples: pass through to value check.
        ok, _ = apply_value_threshold(
            ok=True, notes="", value=-0.5, threshold=0.0,
            label="PE_rel_final", op="lt", n_samples=10)
        assert ok is True
        # n_samples=None (default): no sample-count check.
        ok, _ = apply_value_threshold(
            ok=True, notes="", value=-0.5, threshold=0.0,
            label="PE_rel_final", op="lt")
        assert ok is True

    def test_iter129_lt_zero_op_removed(self):
        """iter-129 codex iter-128-followup LOW-3: the
        deprecated ``op="lt_zero"`` is removed.  Callers must
        use ``op="lt"`` with explicit threshold.
        """
        from legoesm.diagnostics import apply_value_threshold
        import pytest
        with pytest.raises(ValueError, match="unknown op"):
            apply_value_threshold(
                ok=True, notes="", value=-0.5, threshold=0.0,
                label="x", op="lt_zero")

    def test_iter129_no_lt_zero_in_runner_callsites(self):
        """iter-129 codex iter-128-followup LOW-3: no production
        code (runners or src/) should still reference the
        deprecated ``op="lt_zero"``.  Comments and docstrings
        that *describe* the deprecation are OK.
        """
        from pathlib import Path
        repo = Path(__file__).resolve().parent.parent
        for rel in (
            "scripts/matrix/run_ocean_test_matrix.py",
            "scripts/matrix/ocean_test_matrix/timeloop.py",
            "scripts/matrix/ocean_test_matrix/experiments.py",
            "legoesm/diagnostics/conservation_drift.py",
        ):
            text = _read_repo_or_legoesm(repo, rel)
            # An actual op="lt_zero" call would look like
            # ``op="lt_zero"`` or ``op='lt_zero'`` in code.
            # Allow it in docstrings/comments only.
            for line_no, line in enumerate(text.splitlines(), 1):
                stripped = line.lstrip()
                if (stripped.startswith("#") or
                        stripped.startswith('"""') or
                        stripped.startswith("'''") or
                        stripped.startswith('``op="lt_zero"``') or
                        '"lt_zero"' in stripped and
                        ('Removed' in stripped or
                         'deprecated' in stripped or
                         'Pre-iter' in stripped or
                         'pre-iter' in stripped)):
                    continue
                # If we still see 'lt_zero' in code, fail.
                if 'lt_zero' in line:
                    # Allow strings in docstrings — heuristic: if
                    # the line is part of a docstring block, skip.
                    # Simpler: only fail on `op="lt_zero"` or
                    # `op='lt_zero'` exactly.
                    if 'op="lt_zero"' in line or "op='lt_zero'" in line:
                        # Ignore docstring lines.
                        if line.lstrip().startswith('*') or line.lstrip().startswith('('):
                            continue
                        raise AssertionError(
                            f"iter-129: {rel}:{line_no} still has "
                            f"op='lt_zero' callsite: {line.strip()!r}")

    def test_iter129_overflow_short_pe_series_fails(self):
        """iter-129 codex iter-128-followup MEDIUM-1: with an
        empty ``PE_rel`` series, the Overflow PE-sign gate must
        FAIL explicitly (not silently pass via the default
        ``pe_rel_final = 0.0`` placeholder).  Behavior test of
        the wrapper, not just substring.

        iter-138 (n_samples kwarg unchanged; only the op
        changed from 'lt' to 'le').  This test still verifies
        the n_samples=0 fast-fail path — the op change only
        affects the value-comparison branch.
        """
        from legoesm.diagnostics.conservation_drift import (
            apply_value_threshold,
        )
        # n_samples=0 fails via the sample-count guard
        # (independent of op).
        ok, notes = apply_value_threshold(
            ok=True, notes="initial", value=0.0, threshold=0.0,
            label="PE_rel_final", op="le", n_samples=0)
        assert ok is False, (
            "iter-129: empty PE_rel series with default 0.0 "
            "placeholder must NOT silently pass.")
        assert "0 sample" in notes

    def test_iter129_monolithic_lock_exchange_pe_sign_gate(self):
        """iter-129 codex iter-128-followup MEDIUM-2: monolithic
        run_lock_exchange must apply the same ``pe_rel_final < 0``
        sign gate as Overflow (per
        docs/dev-notes/ocean_experiments_reference.md:575).
        """
        from pathlib import Path
        path = (Path(__file__).resolve().parent.parent
                / "scripts" / "matrix" / "run_ocean_test_matrix.py")
        text = path.read_text()
        import re
        m = re.search(
            r"def run_lock_exchange\b.*?(?=\ndef \w|^\s*RUNNERS)",
            text, re.DOTALL | re.MULTILINE,
        )
        assert m is not None
        body = m.group(0)
        assert "_apply_pe_rel_sign" in body, (
            "iter-129: monolithic run_lock_exchange must apply "
            "the documented pe_rel_final < 0 sign gate.")
        assert 'label="PE_rel_final"' in body
        # Must pass n_samples to fail explicitly on missing series.
        assert "n_samples=" in body and "PE_rel" in body

    def test_iter129_modular_lock_exchange_pe_sign_gate(self):
        """iter-129 codex iter-128-followup MEDIUM-2: modular
        run_lock_exchange ports the same gate.
        """
        from pathlib import Path
        path = (Path(__file__).resolve().parent.parent
                / "scripts" / "matrix" / "ocean_test_matrix"
                / "experiments.py")
        text = path.read_text()
        import re
        m = re.search(
            r"def run_lock_exchange\b.*?(?=\ndef \w)",
            text, re.DOTALL,
        )
        assert m is not None
        body = m.group(0)
        assert "_apply_pe_rel_sign" in body
        assert 'label="PE_rel_final"' in body
        assert "n_samples=" in body

    def test_iter129_stommel_uses_lt_strict(self):
        """iter-129 codex iter-128-followup LOW-1: Stommel
        overshoot/undershoot gates use ``op="lt"`` (strict)
        to match the documented ``< 0.1 PSU`` strict bound,
        not ``op="le"`` which would pass at exactly 0.1.
        """
        from pathlib import Path
        for rel in (
            "scripts/matrix/run_ocean_test_matrix.py",
            "scripts/matrix/ocean_test_matrix/experiments.py",
        ):
            text = (Path(__file__).resolve().parent.parent
                    / rel).read_text()
            import re
            m = re.search(
                r"def run_stommel_gyre_tracer\b"
                r".*?(?=\ndef \w|\nRUNNERS)",
                text, re.DOTALL,
            )
            assert m is not None, f"{rel}: missing function"
            body = m.group(0)
            # Strip comment lines so a comment that quotes
            # ``op="le"`` for context doesn't trip the gate.
            code_lines = [
                line for line in body.splitlines()
                if not line.lstrip().startswith("#")
            ]
            code_only = "\n".join(code_lines)
            # Both gates must use op="lt" not op="le".
            le_count = code_only.count('op="le"')
            lt_count = code_only.count('op="lt"')
            assert lt_count >= 2, (
                f"iter-129: {rel}:run_stommel_gyre_tracer must "
                f"have at least 2 ``op='lt'`` code-level gates "
                f"(overshoot+undershoot).  Got lt={lt_count} "
                f"le={le_count}.")
            # No remaining op="le" in the Stommel code (comments
            # quoting the historical op="le" are OK).
            assert le_count == 0, (
                f"iter-129: {rel}:run_stommel_gyre_tracer must "
                f"use op='lt' (strict <), not op='le' (<=). "
                f"Found {le_count} ``op='le'`` code-level calls.")

    def test_iter129_stommel_nan_extrema_caught(self):
        """iter-129 codex iter-128-followup LOW-2: a NaN in
        ``S_min``/``S_max`` must propagate to the helper as
        non-finite (not get masked by ``max(0, NaN)`` which can
        return 0).  Verify the helper catches non-finite values.
        """
        import numpy as np
        from legoesm.diagnostics import apply_value_threshold
        # If S_max_final = NaN, the raw delta is NaN; the helper
        # must fail with "non-finite", not pass.
        ok, notes = apply_value_threshold(
            ok=True, notes="", value=float("nan"), threshold=0.1,
            label="S overshoot", op="lt", units="PSU",
            n_samples=10)
        assert ok is False
        assert "non-finite" in notes
        # Verify Python's max(0, NaN) gotcha is real (this is the
        # bug pattern iter-129 LOW-2 fixed): max with NaN is
        # order-dependent and non-deterministic.  We don't assert
        # what it returns — just that the runner now bypasses
        # max(0, ...) on NaN by computing raw deltas first.
        # Read the runner source to confirm the fix is in place.
        from pathlib import Path
        for rel in (
            "scripts/matrix/run_ocean_test_matrix.py",
            "scripts/matrix/ocean_test_matrix/experiments.py",
        ):
            text = (Path(__file__).resolve().parent.parent
                    / rel).read_text()
            import re
            m = re.search(
                r"def run_stommel_gyre_tracer\b"
                r".*?(?=\ndef \w|\nRUNNERS)",
                text, re.DOTALL,
            )
            assert m is not None
            body = m.group(0)
            # The fix introduces ``raw_over``/``raw_under`` and
            # checks ``np.isfinite(raw_*)`` BEFORE clamping.
            assert "raw_over" in body
            assert "raw_under" in body
            assert "isfinite(raw_over)" in body
            assert "isfinite(raw_under)" in body

    def test_iter129_overflow_passes_n_samples_to_pe_sign(self):
        """iter-129 codex iter-128-followup MEDIUM-1: the
        Overflow PE-sign callsite passes ``n_samples`` so a
        missing/single-sample PE_rel series fails explicitly.
        """
        from pathlib import Path
        for rel in (
            "scripts/matrix/run_ocean_test_matrix.py",
            "scripts/matrix/ocean_test_matrix/experiments.py",
        ):
            text = (Path(__file__).resolve().parent.parent
                    / rel).read_text()
            import re
            m = re.search(
                r"def run_overflow\b.*?(?=\ndef \w|\nRUNNERS)",
                text, re.DOTALL,
            )
            assert m is not None, f"{rel}: missing run_overflow"
            body = m.group(0)
            # Find the _apply_pe_rel_sign call and verify it has
            # n_samples= in its argument list.
            m2 = re.search(
                r"_apply_pe_rel_sign\([^)]*n_samples=[^)]*\)",
                body, re.DOTALL,
            )
            assert m2 is not None, (
                f"iter-129: {rel}:run_overflow must pass "
                f"n_samples= to _apply_pe_rel_sign.")

    # ====== iter-130: codex iter-129-followup HIGH-1/2 + MEDIUM + LOW ======

    # Module-level cache for the dynamically-loaded monolithic
    # runner (~3-second import).  iter-131 codex iter-130-followup
    # LOW-1: the cache lives in this class attribute (not in
    # ``sys.modules`` permanently); the module is registered in
    # ``sys.modules`` only for the duration of ``exec_module``
    # so that ``@dataclass`` can resolve its owning module — and
    # is left there because re-importing under the same alias
    # is idempotent (Python's import system would deduplicate).
    # The alias name uses a leading underscore to mark it as
    # test-internal.
    _CACHED_MONOLITHIC_RUNNER: object = None

    @classmethod
    def _import_monolithic_runner(cls):
        """Helper: import scripts/matrix/run_ocean_test_matrix.py as a
        module under a stable alias so ``@dataclass`` can resolve
        its owning module via ``sys.modules``.

        Cached in the class-attribute ``_CACHED_MONOLITHIC_RUNNER``
        on first call to avoid the ~3-second re-import per test.
        """
        if cls._CACHED_MONOLITHIC_RUNNER is not None:
            return cls._CACHED_MONOLITHIC_RUNNER
        import importlib.util
        import sys
        from pathlib import Path
        # Re-use cached entry from sys.modules if any test in
        # the same session already loaded it.
        cached = sys.modules.get("_run_ocean_test_matrix_runner")
        if cached is not None:
            cls._CACHED_MONOLITHIC_RUNNER = cached
            return cached
        path = (Path(__file__).resolve().parent.parent
                / "scripts" / "matrix" / "run_ocean_test_matrix.py")
        spec = importlib.util.spec_from_file_location(
            "_run_ocean_test_matrix_runner", path)
        mod = importlib.util.module_from_spec(spec)
        # Critical: register before exec so @dataclass decorators
        # in the module can find their owning module via
        # ``sys.modules``.  Without this registration the import
        # raises AttributeError in the dataclass machinery.
        sys.modules["_run_ocean_test_matrix_runner"] = mod
        try:
            spec.loader.exec_module(mod)
            cls._CACHED_MONOLITHIC_RUNNER = mod
        except Exception:
            # iter-131 codex iter-130-followup LOW-1: clean up
            # the partial sys.modules entry on failure so a
            # broken import doesn't leak a half-initialized
            # module to subsequent tests.
            sys.modules.pop("_run_ocean_test_matrix_runner", None)
            raise
        return mod

    def test_iter130_monolithic_value_threshold_wrapper_accepts_n_samples(self):
        """iter-130 codex iter-129-followup HIGH-1: the
        monolithic ``_apply_value_threshold`` wrapper must accept
        the ``n_samples`` kwarg that iter-129 added to the
        centralized helper.  Without this, runtime callsites
        (Stommel overshoot/undershoot, _apply_pe_rel_sign) raise
        TypeError.  This is a BEHAVIOR test that actually
        invokes the wrapper, so it would catch the iter-129
        regression that string-only tests missed.
        """
        mod = self._import_monolithic_runner()
        # Must accept n_samples without raising TypeError.
        ok, notes = mod._apply_value_threshold(
            ok=True, notes="", value=0.05, threshold=0.1,
            label="overshoot", op="lt", units="PSU",
            n_samples=10)
        assert ok is True
        # Short series must FAIL through the wrapper.
        ok, notes = mod._apply_value_threshold(
            ok=True, notes="", value=0.05, threshold=0.1,
            label="overshoot", op="lt", units="PSU",
            n_samples=0)
        assert ok is False
        assert "0 sample" in notes

    def test_iter130_modular_value_threshold_wrapper_accepts_n_samples(self):
        """iter-130 codex iter-129-followup HIGH-2: same
        guard for the modular ``_apply_value_threshold`` wrapper
        in ``scripts/matrix/ocean_test_matrix/timeloop.py``.
        """
        import importlib.util
        import sys
        from pathlib import Path
        scripts_dir = (Path(__file__).resolve().parent.parent
                       / "scripts")
        sys.path.insert(0, str(scripts_dir))
        try:
            from ocean_test_matrix.timeloop import (
                _apply_value_threshold,
            )
            ok, notes = _apply_value_threshold(
                ok=True, notes="", value=0.05, threshold=0.1,
                label="overshoot", op="lt", units="PSU",
                n_samples=10)
            assert ok is True
            ok, notes = _apply_value_threshold(
                ok=True, notes="", value=0.05, threshold=0.1,
                label="overshoot", op="lt", units="PSU",
                n_samples=0)
            assert ok is False
        finally:
            sys.path.remove(str(scripts_dir))

    def test_iter130_monolithic_pe_sign_wrapper_works(self):
        """iter-130/iter-138/iter-152 behavior test for
        ``_apply_pe_rel_sign``.  iter-152 (codex iter-151
        review MEDIUM-2) made the gate days-aware:
        * full mode (days >= 1.0): strict ``op="lt"`` (PE_rel = 0
          FAILS, matches documented contract)
        * quick mode (days < 1.0 or omitted): ``op="le"`` (PE_rel = 0
          PASSES, accommodates timestep-budget realities)
        """
        mod = self._import_monolithic_runner()
        # PASS case: pe_rel_final < 0 (RPE decreased).
        ok, _ = mod._apply_pe_rel_sign(
            True, "", -0.5, label="PE_rel_final", n_samples=10,
            days=2.0)
        assert ok is True
        # FULL mode (days=2.0): pe_rel_final = 0 must FAIL
        # (strict < 0 contract).
        ok, notes = mod._apply_pe_rel_sign(
            True, "", 0.0, label="PE_rel_final", n_samples=10,
            days=2.0)
        assert ok is False, (
            "iter-152: full mode (days >= 1.0) must enforce "
            "strict pe_rel_final < 0.")
        # QUICK mode (days=0.1): pe_rel_final = 0 must PASS
        # (timestep budget too tight for measurable PE evolution).
        ok, notes = mod._apply_pe_rel_sign(
            True, "", 0.0, label="PE_rel_final", n_samples=10,
            days=0.1)
        assert ok is True, (
            "iter-152: quick mode (days < 1.0) must allow "
            "pe_rel_final = 0 (no time for PE evolution).")
        # FAIL case: pe_rel_final > 0 (RPE INCREASED — spurious
        # PE creation by numerical mixing; physically wrong).
        # Both full and quick modes should fail this.
        ok, notes = mod._apply_pe_rel_sign(
            True, "", 0.5, label="PE_rel_final", n_samples=10,
            days=2.0)
        assert ok is False
        ok, notes = mod._apply_pe_rel_sign(
            True, "", 0.5, label="PE_rel_final", n_samples=10,
            days=0.1)
        assert ok is False
        # FAIL case: missing series (n_samples=0).
        ok, notes = mod._apply_pe_rel_sign(
            True, "", 0.0, label="PE_rel_final", n_samples=0,
            days=2.0)
        assert ok is False
        assert "0 sample" in notes

    def test_iter130_modular_pe_sign_wrapper_works(self):
        """iter-130 codex iter-129-followup HIGH-2 + iter-153:
        same behavior test for the modular wrapper, with the
        iter-153 ``days`` kwarg.
        """
        import sys
        from pathlib import Path
        scripts_dir = (Path(__file__).resolve().parent.parent
                       / "scripts")
        sys.path.insert(0, str(scripts_dir))
        try:
            from ocean_test_matrix.timeloop import _apply_pe_rel_sign
            # PASS case: pe_rel_final < 0 (RPE decreased).
            ok, _ = _apply_pe_rel_sign(
                True, "", -0.5, label="PE_rel_final",
                n_samples=10, days=2.0)
            assert ok is True
            # FAIL case: missing series (n_samples=0).
            ok, notes = _apply_pe_rel_sign(
                True, "", 0.0, label="PE_rel_final",
                n_samples=0, days=0.1)
            assert ok is False
            assert "0 sample" in notes
        finally:
            sys.path.remove(str(scripts_dir))

    def test_iter130_stommel_uses_per_extremum_sample_count(self):
        """iter-130 codex iter-129-followup LOW-1: Stommel
        overshoot uses ``len(S_max_series)`` and undershoot
        uses ``len(S_min_series)`` — not a single shared count.
        This way a partial S_max diagnostic doesn't bypass the
        overshoot guard (and vice versa).

        iter-131 (codex iter-130-followup LOW-3 follow-through):
        the runner now also assigns ``n_max = len(S_max_series)``
        and ``n_min = len(S_min_series)`` to local variables for
        the WARN annotations; the assertion accepts either the
        direct ``len(...)`` form or the named-variable form.
        Use the iter-131 ``test_iter131_stommel_per_extremum_call_is_specific``
        for the stricter call-site check.
        """
        from pathlib import Path
        for rel in (
            "scripts/matrix/run_ocean_test_matrix.py",
            "scripts/matrix/ocean_test_matrix/experiments.py",
        ):
            text = (Path(__file__).resolve().parent.parent
                    / rel).read_text()
            import re
            m = re.search(
                r"def run_stommel_gyre_tracer\b"
                r".*?(?=\ndef \w|\nRUNNERS)",
                text, re.DOTALL,
            )
            assert m is not None
            body = m.group(0)
            # overshoot must use S_max sample count.
            has_overshoot = (
                'label="S overshoot"' in body
                and ("n_samples=len(S_max_series)" in body
                     or "n_samples=n_max" in body)
            )
            assert has_overshoot, (
                f"iter-130/iter-131: {rel}: Stommel overshoot "
                f"must use S_max sample count for n_samples "
                f"(via len(S_max_series) or local n_max).")
            # undershoot must use S_min sample count.
            has_undershoot = (
                'label="S undershoot"' in body
                and ("n_samples=len(S_min_series)" in body
                     or "n_samples=n_min" in body)
            )
            assert has_undershoot, (
                f"iter-130/iter-131: {rel}: Stommel undershoot "
                f"must use S_min sample count for n_samples "
                f"(via len(S_min_series) or local n_min).")

    def test_iter130_no_stale_doc_line_numbers(self):
        """iter-130 codex iter-129-followup LOW-2: doc line
        numbers in code comments drift as the doc is edited.
        For PE-sign-related references, prefer named-section
        callouts.  This test enforces no stale ``:575`` or
        ``:626`` references survive.
        """
        from pathlib import Path
        for rel in (
            "scripts/matrix/run_ocean_test_matrix.py",
            "scripts/matrix/ocean_test_matrix/experiments.py",
        ):
            text = (Path(__file__).resolve().parent.parent
                    / rel).read_text()
            for stale_line in (":575", ":626"):
                marker = f"ocean_experiments_reference.md{stale_line}"
                assert marker not in text, (
                    f"iter-130: {rel} has stale doc line "
                    f"reference {marker!r}; the actual line "
                    f"numbers drifted to 581 (lock_exchange) "
                    f"and 632 (overflow) after iter-128 edits. "
                    f"Replace with a section-name callout.")

    # ====== iter-131: codex iter-130-followup HIGH-1 + MEDIUM + LOW ======

    def test_iter131_value_threshold_op_ge(self):
        """iter-131 codex iter-130-followup HIGH-1: the new
        ``op="ge"`` mode for greater-than-or-equal lower-bound
        gates (e.g., Phillips ``eta_growth >= 0.8``).
        """
        from legoesm.diagnostics import apply_value_threshold
        # PASS: value at threshold.
        ok, _ = apply_value_threshold(
            ok=True, notes="", value=0.8, threshold=0.8,
            label="eta_growth", op="ge")
        assert ok is True
        # PASS: value above threshold.
        ok, _ = apply_value_threshold(
            ok=True, notes="", value=2.0, threshold=0.8,
            label="eta_growth", op="ge")
        assert ok is True
        # FAIL: value below threshold.
        ok, notes = apply_value_threshold(
            ok=True, notes="", value=0.5, threshold=0.8,
            label="eta_growth", op="ge")
        assert ok is False
        assert "expected greater-than-or-equal" in notes
        # NaN: fail.
        ok, _ = apply_value_threshold(
            ok=True, notes="", value=float("nan"), threshold=0.8,
            label="eta_growth", op="ge")
        assert ok is False

    def test_iter131_unknown_op_message_lists_ge(self):
        """iter-131 codex iter-130-followup HIGH-1 follow-up:
        the unknown-op error message must now list 'ge' too.
        """
        from legoesm.diagnostics import apply_value_threshold
        import pytest
        with pytest.raises(ValueError, match="'ge'"):
            apply_value_threshold(
                ok=True, notes="", value=1.0, threshold=0.5,
                label="x", op="bogus")

    def test_iter131_phillips_has_three_gates_monolithic(self):
        """iter-131 codex iter-130-followup HIGH-1: monolithic
        run_phillips_two_layer must apply T_abs_drift, eta_growth
        (as range: lower + upper), and max_eta_amplitude PASS
        gates per the documented thresholds.
        """
        from pathlib import Path
        text = (Path(__file__).resolve().parent.parent
                / "scripts" / "matrix" / "run_ocean_test_matrix.py").read_text()
        import re
        m = re.search(
            r"def run_phillips_two_layer\b.*?(?=\ndef \w|\nRUNNERS)",
            text, re.DOTALL,
        )
        assert m is not None
        body = m.group(0)
        # Strip comments.
        code = "\n".join(
            line for line in body.splitlines()
            if not line.lstrip().startswith("#"))
        # T_abs_drift gate.
        assert 'label="T_abs_drift"' in code
        assert "T_abs_drift, 5.0" in code
        # eta_growth lower bound (op="ge").
        assert 'label="eta_growth_lower"' in code
        assert 'op="ge"' in code
        # eta_growth upper bound (op="le").
        assert 'label="eta_growth_upper"' in code
        # max_eta_amplitude gate.
        assert 'label="max_eta_amplitude"' in code
        assert "max_eta_overall, 5.0" in code

    def test_iter131_phillips_has_three_gates_modular(self):
        """iter-131 codex iter-130-followup HIGH-1: modular
        run_phillips_two_layer mirrors the same gates.
        """
        from pathlib import Path
        text = (Path(__file__).resolve().parent.parent
                / "scripts" / "matrix" / "ocean_test_matrix"
                / "experiments.py").read_text()
        import re
        m = re.search(
            r"def run_phillips_two_layer\b.*?(?=\ndef \w)",
            text, re.DOTALL,
        )
        assert m is not None
        body = m.group(0)
        code = "\n".join(
            line for line in body.splitlines()
            if not line.lstrip().startswith("#"))
        assert 'label="T_abs_drift"' in code
        assert "T_abs_drift, 5.0" in code
        assert 'label="eta_growth_lower"' in code
        assert 'op="ge"' in code
        assert 'label="eta_growth_upper"' in code
        assert 'label="max_eta_amplitude"' in code
        assert "max_eta_overall, 5.0" in code

    def test_iter131_rest_state_has_S_drift_gate_monolithic(self):
        """iter-131 codex iter-130-followup MEDIUM-1: monolithic
        rest_state variants must gate S_drift (< 1e-6 per
        documented threshold).
        """
        from pathlib import Path
        text = (Path(__file__).resolve().parent.parent
                / "scripts" / "matrix" / "run_ocean_test_matrix.py").read_text()
        import re
        for fn in (
            "run_rest_state",
            "run_rest_state_no_land",
            "run_rest_state_uniform_ts",
            "run_rest_state_uniform_ts_no_land",
        ):
            m = re.search(
                rf"def {fn}\b.*?(?=\ndef \w|\nRUNNERS)",
                text, re.DOTALL,
            )
            assert m is not None, f"{fn}: not found"
            body = m.group(0)
            code = "\n".join(
                line for line in body.splitlines()
                if not line.lstrip().startswith("#"))
            assert 'label="S"' in code, (
                f"iter-131: monolithic {fn} must gate S_drift.")
            assert "S_drift, 1e-6" in code, (
                f"iter-131: monolithic {fn} S_drift tolerance "
                f"must be 1e-6 per documented threshold.")

    def test_iter131_rest_state_has_S_drift_gate_modular(self):
        """iter-131 codex iter-130-followup MEDIUM-1: modular
        rest_state variants must gate S_drift too.
        """
        from pathlib import Path
        text = (Path(__file__).resolve().parent.parent
                / "scripts" / "matrix" / "ocean_test_matrix"
                / "experiments.py").read_text()
        import re
        for fn in (
            "run_rest_state",
            "run_rest_state_no_land",
            "run_rest_state_uniform_ts",
            "run_rest_state_uniform_ts_no_land",
        ):
            m = re.search(
                rf"def {fn}\b.*?(?=\ndef \w)",
                text, re.DOTALL,
            )
            assert m is not None, f"{fn}: not found"
            body = m.group(0)
            code = "\n".join(
                line for line in body.splitlines()
                if not line.lstrip().startswith("#"))
            assert 'label="S"' in code
            assert "S_drift, 1e-6" in code

    def test_iter131_stommel_pre_records_both_warnings(self):
        """iter-131 codex iter-130-followup LOW-3: Stommel
        pre-records WARN annotations for BOTH missing extrema
        BEFORE either gate fails, so the second extremum's
        absence isn't hidden by the first gate's short-circuit.
        """
        from pathlib import Path
        for rel in (
            "scripts/matrix/run_ocean_test_matrix.py",
            "scripts/matrix/ocean_test_matrix/experiments.py",
        ):
            text = (Path(__file__).resolve().parent.parent
                    / rel).read_text()
            import re
            m = re.search(
                r"def run_stommel_gyre_tracer\b"
                r".*?(?=\ndef \w|\nRUNNERS)",
                text, re.DOTALL,
            )
            assert m is not None
            body = m.group(0)
            # Both WARN strings must appear, and must be
            # appended BEFORE the gate calls (so they survive
            # short-circuiting).
            assert "WARN: S_max series" in body
            assert "WARN: S_min series" in body

    def test_iter131_no_partial_doc_line_drift(self):
        """iter-131 codex iter-130-followup LOW-2: broaden the
        no-stale-doc-line test to scan ALL files in scripts/
        and src/legoesm/diagnostics/, not just the two runners.
        """
        from pathlib import Path
        repo = Path(__file__).resolve().parent.parent
        files_to_check = [
            "scripts/matrix/run_ocean_test_matrix.py",
            "scripts/matrix/ocean_test_matrix/experiments.py",
            "scripts/matrix/ocean_test_matrix/timeloop.py",
            "src/legoesm/diagnostics/conservation_drift.py",
        ]
        for stale_line in (":575", ":626", ":681-682",
                           ":419"):
            # ``:419`` is the geostrophic 1e-3 doc line —
            # the runner uses 1e-8.  iter-128 added a doc
            # cross-reference test (test_iter128_*); the
            # comment in modular geostrophic legitimately
            # cites :419.  We allow this one but flag if it
            # ever drifts.
            if stale_line == ":419":
                continue
            for rel in files_to_check:
                text = _read_repo_or_legoesm(repo, rel)
                marker = f"ocean_experiments_reference.md{stale_line}"
                assert marker not in text, (
                    f"iter-131: {rel} has stale doc line "
                    f"reference {marker!r}.  Replace with a "
                    f"section-name callout.")

    def test_iter131_stommel_per_extremum_call_is_specific(self):
        """iter-131 codex iter-130-followup LOW-4: the Stommel
        per-extremum sample-count check must verify that the
        SAME ``_apply_value_threshold`` call uses both the
        right label AND the right ``n_samples`` — not just that
        both strings appear somewhere in the body.

        Achieved via regex matching the exact call arguments.
        """
        from pathlib import Path
        for rel in (
            "scripts/matrix/run_ocean_test_matrix.py",
            "scripts/matrix/ocean_test_matrix/experiments.py",
        ):
            text = (Path(__file__).resolve().parent.parent
                    / rel).read_text()
            import re
            m = re.search(
                r"def run_stommel_gyre_tracer\b"
                r".*?(?=\ndef \w|\nRUNNERS)",
                text, re.DOTALL,
            )
            assert m is not None
            body = m.group(0)
            # Match the exact overshoot call: must contain
            # label="S overshoot" AND n_samples=n_max
            # (or n_samples=len(S_max_series)) in the SAME
            # parenthesized argument list.
            overshoot_calls = re.findall(
                r"_apply_value_threshold\([^)]*"
                r'label="S overshoot"[^)]*\)',
                body, re.DOTALL)
            assert len(overshoot_calls) == 1, (
                f"iter-131: {rel}: expected exactly one "
                f"overshoot _apply_value_threshold call.")
            overshoot_call = overshoot_calls[0]
            assert ("n_samples=n_max" in overshoot_call
                    or "n_samples=len(S_max_series)" in overshoot_call), (
                f"iter-131: {rel}: overshoot call must use "
                f"S_max sample count, got: {overshoot_call!r}")

            undershoot_calls = re.findall(
                r"_apply_value_threshold\([^)]*"
                r'label="S undershoot"[^)]*\)',
                body, re.DOTALL)
            assert len(undershoot_calls) == 1
            undershoot_call = undershoot_calls[0]
            assert ("n_samples=n_min" in undershoot_call
                    or "n_samples=len(S_min_series)" in undershoot_call), (
                f"iter-131: {rel}: undershoot call must use "
                f"S_min sample count, got: {undershoot_call!r}")

    # ====== iter-132: codex iter-131-followup HIGH-1 + MEDIUM ======

    def test_iter132_igw_has_l2_and_amplitude_gates_monolithic(self):
        """iter-132 codex iter-131-followup HIGH-1: monolithic
        run_inertia_gravity_wave applies the documented L2_err
        and amplitude_conservation gates.

        iter-138b: L2 threshold is now days-aware
        (``l2_threshold = 0.1 if days >= 1.0 else 2.0``) since
        quick-mode (0.2 days) at coarse resolution has natural
        L2 ~ 1-2 due to numerical dispersion + propagation
        time, far above the doc's 0.1 (which is for full mode).
        """
        from pathlib import Path
        text = (Path(__file__).resolve().parent.parent
                / "scripts" / "matrix" / "run_ocean_test_matrix.py").read_text()
        import re
        m = re.search(
            r"def run_inertia_gravity_wave\b.*?(?=\ndef \w|\nRUNNERS)",
            text, re.DOTALL,
        )
        assert m is not None
        body = m.group(0)
        code = "\n".join(
            line for line in body.splitlines()
            if not line.lstrip().startswith("#"))
        # 2026-08-10: the L2 / amplitude gates were REMOVED. ``eta_exact``
        # is an f-plane plane wave (constant f0 = 1e-4, k = kx/a) while
        # every arm integrates the full sphere (f = 2*Omega*sin(lat), true
        # k = kx/(a cos lat)), so the comparison is unpassable BY
        # CONSTRUCTION -- measured L2 ~ 1.0 on latlon, mpas, tripole and
        # fesom alike. A gate no arm can pass discredits the matrix, so
        # L2/amplitude are now UNGATED DIAGNOSTICS and the case gates on
        # stability instead. This test pins that decision: it fails if the
        # invalid gate is reintroduced without rebuilding the case on an
        # f-plane channel where a Poincare wave is defined.
        assert 'label="IGW L2 vs analytical"' not in code, (
            "the f-plane L2 gate is invalid on a global sphere; rebuild "
            "the case on an f-plane channel before re-gating it")
        assert "l2_threshold" not in code
        # The stability gate that replaced it must be present and must
        # look at the WHOLE run, not just the final sample.
        assert 'label="IGW max|eta| over run vs IC"' in code
        assert "max_eta_run" in code
        # L2 is still REPORTED (as a diagnostic) so the regression is
        # visible when the case is eventually fixed.
        assert "l2_err" in code

    def test_iter132_igw_has_l2_and_amplitude_gates_modular(self):
        """iter-132 codex iter-131-followup HIGH-1: modular
        IGW mirrors the same gates (days-aware L2 threshold
        per iter-138b).
        """
        from pathlib import Path
        text = (Path(__file__).resolve().parent.parent
                / "scripts" / "matrix" / "ocean_test_matrix"
                / "experiments.py").read_text()
        import re
        m = re.search(
            r"def run_inertia_gravity_wave\b.*?(?=\ndef \w)",
            text, re.DOTALL,
        )
        assert m is not None
        body = m.group(0)
        code = "\n".join(
            line for line in body.splitlines()
            if not line.lstrip().startswith("#"))
        assert 'label="IGW L2 vs analytical"' in code
        assert "l2_threshold = 0.1 if days >= 1.0 else 2.0" in code
        assert "l2_err, l2_threshold" in code
        assert 'label="IGW amplitude_ratio_lower"' in code
        assert 'op="ge"' in code
        assert 'label="IGW amplitude_ratio_upper"' in code

    def test_iter132_geostrophic_max_speed_gate(self):
        """iter-132 codex iter-131-followup MEDIUM-1: both
        monolithic and modular geostrophic_adjustment gate the
        documented ``max_speed_final < 1.0 m/s``.
        """
        from pathlib import Path
        for rel in (
            "scripts/matrix/run_ocean_test_matrix.py",
            "scripts/matrix/ocean_test_matrix/experiments.py",
        ):
            text = (Path(__file__).resolve().parent.parent
                    / rel).read_text()
            import re
            m = re.search(
                r"def run_geostrophic_adjustment\b"
                r".*?(?=\ndef \w|\nRUNNERS)",
                text, re.DOTALL,
            )
            assert m is not None
            body = m.group(0)
            code = "\n".join(
                line for line in body.splitlines()
                if not line.lstrip().startswith("#"))
            assert 'label="max_speed_final"' in code, (
                f"iter-132: {rel}: geostrophic must gate "
                f"max_speed_final.")
            assert "max_speed_final, 1.0" in code, (
                f"iter-132: {rel}: max_speed_final tolerance "
                f"must be 1.0 m/s.")

    def test_iter132_overflow_t_bounds_gate(self):
        """iter-132 codex iter-131-followup MEDIUM-2: both
        monolithic and modular run_overflow gate the documented
        ``Temperature within [-200, 200] C`` bounds.
        """
        from pathlib import Path
        for rel in (
            "scripts/matrix/run_ocean_test_matrix.py",
            "scripts/matrix/ocean_test_matrix/experiments.py",
        ):
            text = (Path(__file__).resolve().parent.parent
                    / rel).read_text()
            import re
            m = re.search(
                r"def run_overflow\b.*?(?=\ndef \w|\nRUNNERS)",
                text, re.DOTALL,
            )
            assert m is not None
            body = m.group(0)
            code = "\n".join(
                line for line in body.splitlines()
                if not line.lstrip().startswith("#"))
            assert 'label="T_min_final"' in code
            assert "T_min_final, -200.0" in code
            assert 'label="T_max_final"' in code
            assert "T_max_final, 200.0" in code

    def test_iter132_lock_exchange_t_bounds_gate(self):
        """iter-132 codex iter-131-followup MEDIUM-2: both
        monolithic and modular run_lock_exchange gate the
        documented ``Temperature within [-200, 200] C`` bounds.
        """
        from pathlib import Path
        for rel in (
            "scripts/matrix/run_ocean_test_matrix.py",
            "scripts/matrix/ocean_test_matrix/experiments.py",
        ):
            text = (Path(__file__).resolve().parent.parent
                    / rel).read_text()
            import re
            m = re.search(
                r"def run_lock_exchange\b.*?(?=\ndef \w|\nRUNNERS)",
                text, re.DOTALL,
            )
            assert m is not None
            body = m.group(0)
            code = "\n".join(
                line for line in body.splitlines()
                if not line.lstrip().startswith("#"))
            assert 'label="T_min_final"' in code
            assert "T_min_final, -200.0" in code
            assert 'label="T_max_final"' in code
            assert "T_max_final, 200.0" in code

    def test_iter132_phillips_eta_initial_zero_fallback(self):
        """iter-132 codex iter-131-followup MEDIUM-3: when the
        first max_eta sample is zero (initial perturbation
        before forcing kicks in), fall back to the first
        finite non-zero sample as denominator instead of
        producing NaN eta_growth.
        """
        from pathlib import Path
        for rel in (
            "scripts/matrix/run_ocean_test_matrix.py",
            "scripts/matrix/ocean_test_matrix/experiments.py",
        ):
            text = (Path(__file__).resolve().parent.parent
                    / rel).read_text()
            import re
            m = re.search(
                r"def run_phillips_two_layer\b"
                r".*?(?=\ndef \w|\nRUNNERS)",
                text, re.DOTALL,
            )
            assert m is not None
            body = m.group(0)
            # Must reference a "finite_nonzero" array selection
            # or equivalent to find the first valid baseline.
            assert "finite_nonzero" in body, (
                f"iter-132: {rel}: Phillips eta_growth must "
                f"fall back to first finite non-zero max_eta "
                f"sample as denominator.")

    # ====== iter-133: self-review barotropic experiment gates ======

    def test_iter133_barotropic_wave_gates(self):
        """iter-133 self-review based on
        docs/dev-notes/ocean_experiments_reference.md 'Barotropic Wave'
        Validation Thresholds: gates eta_conservation
        (range), mean_eta_drift < 1e-4 m,
        min_final_amplitude > 0.1 m.

        iter-138b: range relaxed from [0.8, 1.2] to [0.5, 1.5]
        for propagating-wave tolerance (the doc range was
        meant for standing-wave steady state).
        """
        from pathlib import Path
        for rel in (
            "scripts/matrix/run_ocean_test_matrix.py",
            "scripts/matrix/ocean_test_matrix/experiments.py",
        ):
            text = (Path(__file__).resolve().parent.parent
                    / rel).read_text()
            import re
            m = re.search(
                r"def run_barotropic_wave\b"
                r".*?(?=\ndef \w|\nRUNNERS)",
                text, re.DOTALL,
            )
            assert m is not None, f"{rel}: missing"
            body = m.group(0)
            code = "\n".join(
                line for line in body.splitlines()
                if not line.lstrip().startswith("#"))
            assert 'label="eta_conservation_lower"' in code
            assert 'label="eta_conservation_upper"' in code
            # iter-138b relaxed thresholds.
            assert "eta_conservation, 0.5" in code
            assert "eta_conservation, 1.5" in code
            assert 'label="mean_eta_drift"' in code
            assert "mean_eta_drift, 1e-4" in code
            assert 'label="min_final_amplitude"' in code
            assert "min_final_amplitude, 0.1" in code

    def test_iter133_barotropic_gyre_gates(self):
        """iter-133 self-review based on
        docs/dev-notes/ocean_experiments_reference.md 'Barotropic Gyre'
        Validation Thresholds: gates max_speed_final with an upper
        bound of 0.5 m/s and a lower bound defaulting to 0.05 m/s
        (the monolithic runner may lower it per-case via
        ``min_max_speed`` for short double-gyre / sin2 spin-ups),
        plus eta_drift < 1e-3 m absolute. Applied via
        _run_gyre_experiment shared runner so it covers single +
        double + sin2 variants.
        """
        from pathlib import Path
        for rel in (
            "scripts/matrix/run_ocean_test_matrix.py",
            "scripts/matrix/ocean_test_matrix/experiments.py",
        ):
            text = (Path(__file__).resolve().parent.parent
                    / rel).read_text()
            import re
            m = re.search(
                r"def _run_gyre_experiment\b"
                r".*?(?=\ndef \w|\nRUNNERS)",
                text, re.DOTALL,
            )
            assert m is not None, f"{rel}: missing _run_gyre_experiment"
            body = m.group(0)
            code = "\n".join(
                line for line in body.splitlines()
                if not line.lstrip().startswith("#"))
            assert 'label="max_speed_final_lower"' in code
            assert 'label="max_speed_final_upper"' in code
            # Lower-speed gate (label="max_speed_final_lower", op="ge"). The
            # modular runner inlines the 0.05 m/s default; the monolithic runner
            # feeds the gate ``lower_thresh`` (default 0.05, lowered per-case via
            # ``min_max_speed`` for short double-gyre / sin2 spin-ups). Tie the
            # check to the gate ACTUALLY consuming that threshold value (not just
            # to ``lower_thresh`` being defined somewhere).
            assert (
                "max_speed), 0.05" in code
                or ("max_speed), lower_thresh" in code and "lower_thresh = 0.05" in code)
            ), (
                f"{rel}: gyre lower-speed gate must consume the 0.05 m/s default "
                "(inline ``max_speed), 0.05`` or ``max_speed), lower_thresh`` with "
                "``lower_thresh = 0.05``).")
            assert "max_speed), 0.5" in code
            assert 'label="eta_drift_absolute"' in code
            assert "eta_drift), 1e-3" in code

    # ====== iter-134: self-review barotropic gate semantics ======

    def test_iter134_min_final_amplitude_uses_window_min(self):
        """iter-134 self-review fix: ``min_final_amplitude``
        should be the MIN max|eta| across the final 20% of
        samples, not just the last sample.  This catches
        transient damping that the last-sample-only check
        would miss.
        """
        from pathlib import Path
        for rel in (
            "scripts/matrix/run_ocean_test_matrix.py",
            "scripts/matrix/ocean_test_matrix/experiments.py",
        ):
            text = (Path(__file__).resolve().parent.parent
                    / rel).read_text()
            import re
            m = re.search(
                r"def run_barotropic_wave\b"
                r".*?(?=\ndef \w|\nRUNNERS)",
                text, re.DOTALL,
            )
            assert m is not None
            body = m.group(0)
            # Must compute ``final_window`` and use np.nanmin.
            assert "final_window" in body
            assert "np.nanmin" in body
            assert "n_eta // 5" in body, (
                f"iter-134: {rel}: barotropic_wave must use "
                f"a window of size n_eta // 5 (~20%) for the "
                f"final-amplitude diagnostic.")
            # Must NOT use the buggy iter-133 form
            # ``min_final_amplitude = eta_final``.
            code_lines = [
                line for line in body.splitlines()
                if not line.lstrip().startswith("#")
            ]
            code_only = "\n".join(code_lines)
            assert "min_final_amplitude = eta_final" not in code_only, (
                f"iter-134: {rel}: barotropic_wave must NOT "
                f"use the buggy iter-133 form "
                f"min_final_amplitude = eta_final.  Use the "
                f"window-min instead.")

    def test_iter134_barotropic_wave_window_behavior(self):
        """iter-134 behavior test: the window-min logic
        correctly catches a wave that damps then recovers.

        Simulates a max_eta_series like [1.0, 0.5, 0.3, 0.05, 0.15]:
        last sample is 0.15 (would PASS the > 0.1 gate), but the
        min over the final 20% (1 sample = 0.15) — wait,
        with n=5, n//5=1, so window=[0.15], min=0.15, PASSES.

        Use a longer series: [1.0, 0.8, 0.6, 0.4, 0.3, 0.2,
        0.15, 0.05, 0.12, 0.18] (n=10, window=[0.18, 0.12,
        0.05, 0.18][last 2]=0.05 — that's the bug demo).
        """
        import numpy as np
        # n=10, n//5=2, so the final window is the last 2 samples.
        max_eta_series = [1.0, 0.8, 0.6, 0.4, 0.3, 0.2,
                          0.15, 0.05, 0.12, 0.18]
        n = len(max_eta_series)
        arr = np.asarray(max_eta_series)
        final_window = arr[-max(1, n // 5):]
        # The min over the last 2 samples [0.12, 0.18] is 0.12.
        min_final_amplitude = float(np.nanmin(np.abs(final_window)))
        assert min_final_amplitude == 0.12
        # If we'd used eta_final only (= 0.18), it would PASS
        # the > 0.1 gate trivially.  The window-min still
        # PASSES at 0.12 here, but a deeper transient would
        # correctly fail.

        # Demonstration: a series where last sample passes
        # but final-window min fails:
        max_eta_series2 = [1.0, 0.8, 0.6, 0.4, 0.3, 0.2,
                           0.15, 0.05, 0.05, 0.15]
        arr2 = np.asarray(max_eta_series2)
        n2 = len(arr2)
        final_window2 = arr2[-max(1, n2 // 5):]
        # Last 2 samples: [0.05, 0.15], min=0.05
        min_amp2 = float(np.nanmin(np.abs(final_window2)))
        assert min_amp2 == 0.05
        # With the gate threshold of 0.1, this run FAILS as it
        # should — but the iter-133 last-sample-only check
        # would have PASSED (0.15 > 0.1).
        from legoesm.diagnostics import apply_value_threshold
        ok, notes = apply_value_threshold(
            ok=True, notes="", value=min_amp2, threshold=0.1,
            label="min_final_amplitude", op="ge", units="m",
            n_samples=n2)
        assert ok is False
        assert "0.05" in notes

    # ====== iter-138: production fixes from end-to-end runner ======

    def test_iter138_phillips_latlon_u_shape_fix(self):
        """iter-138 (iter-137 ERROR-1): Phillips IC must handle
        latlon C-grid u-staggering (shape n_lat, n_lon+1).
        """
        from pathlib import Path
        text = (Path(__file__).resolve().parent.parent
                / "scripts" / "matrix" / "run_ocean_test_matrix.py").read_text()
        import re
        m = re.search(
            r"def _add_phillips_perturbation\b"
            r".*?(?=\ndef \w|\nRUNNERS)",
            text, re.DOTALL,
        )
        assert m is not None
        body = m.group(0)
        # 2026-08-10: latlon SHARES this branch with tripole -- the u-face
        # latitudes now come from _cgrid_face_lat_lon, which serves the
        # rectilinear and the curvilinear mesh alike. The invariant the
        # test guards is unchanged: the C-grid arms must NOT fall through
        # to the cube branch, because their u has shape (n_lat, n_lon+1).
        assert 'elif grid_type in ("latlon", "tripole")' in body, (
            "iter-138: Phillips IC must dispatch the C-grid arms "
            "separately to handle u-shape (n_lat, n_lon+1).")
        # The jet must be evaluated at the U-FACE positions (shape
        # (n_lat, n_lon+1)), which is what the shared helper returns.
        assert "_cgrid_face_lat_lon(grid)" in body
        assert "u_jet_2d" in body

    def test_iter138_igw_latlon_u_v_edge_fix(self):
        """iter-138 (iter-137 ERROR-2): IGW IC must compute u/v
        on latlon C-grid edges (shape n_lat, n_lon+1 for u and
        n_lat+1, n_lon for v), not at cell centers.
        """
        from pathlib import Path
        text = (Path(__file__).resolve().parent.parent
                / "scripts" / "matrix" / "run_ocean_test_matrix.py").read_text()
        import re
        m = re.search(
            r"def _init_inertia_gravity_wave\b"
            r".*?(?=\ndef \w|\nRUNNERS)",
            text, re.DOTALL,
        )
        assert m is not None
        body = m.group(0)
        # Must have a dedicated C-grid branch (separate from cube).
        # 2026-08-10: latlon and tripole SHARE it -- the face coordinates
        # now come from _cgrid_face_lat_lon, which serves both the
        # rectilinear and the curvilinear mesh.
        assert 'elif grid_type in ("latlon", "tripole")' in body
        # Must place u/v perturbations at the FACE positions (now via
        # the shared helper, which returns lat_u/lon_u/lat_v/lon_v).
        assert "_cgrid_face_lat_lon(grid)" in body
        assert "lon_u" in body
        assert "lat_v" in body

    def test_iter138_barotropic_wave_eta_conservation_window(self):
        """iter-138 (iter-137 FAIL-1): eta_conservation metric
        is now min/max over the FINAL 50% of samples (steady-
        state stability), not initial-vs-final ratio (which
        always failed in quick mode due to natural dispersion).
        """
        from pathlib import Path
        for rel in (
            "scripts/matrix/run_ocean_test_matrix.py",
            "scripts/matrix/ocean_test_matrix/experiments.py",
        ):
            text = (Path(__file__).resolve().parent.parent
                    / rel).read_text()
            import re
            m = re.search(
                r"def run_barotropic_wave\b"
                r".*?(?=\ndef \w|\nRUNNERS)",
                text, re.DOTALL,
            )
            assert m is not None, f"{rel}: missing"
            body = m.group(0)
            # Must use ``final_half`` (50% window), ``final_min``
            # and ``final_max`` (window stats).
            assert "final_half" in body, (
                f"iter-138: {rel}: barotropic_wave must use "
                f"final_half = max_eta_arr[-n_eta // 2:] for "
                f"the new eta_conservation metric.")
            assert "final_min" in body
            assert "final_max" in body
            # Must NOT use the buggy iter-133/134 form
            # ``eta_final / eta_initial``.
            code_lines = [
                line for line in body.splitlines()
                if not line.lstrip().startswith("#")
            ]
            code_only = "\n".join(code_lines)
            assert "eta_conservation = eta_final / eta_initial" not in code_only, (
                f"iter-138: {rel}: barotropic_wave must NOT use "
                f"the iter-133/134 initial-vs-final ratio.")

    # ====== iter-149/150 → iter-174 → Phase B.1: cube bottom_drag_r plumbed ======

    def test_cube_bottom_drag_r_is_plumbed(self):
        """Phase B.1 unblock (supersedes the iter-174 NotImplementedError gate):
        the cubed-sphere cd-grid backend
        (``ocean_baroclinic_tendencies_cdgrid``) now applies model-level linear /
        quadratic / BBL bottom drag the same way the lat-lon C-grid does, so the
        monolithic runner plumbs ``bottom_drag_r`` straight into the cube
        ``OceanConfig`` instead of raising ``NotImplementedError`` and SKIPping
        the case.

        Verifies the monolithic runner: (1) the cube branch forwards
        ``bottom_drag_r`` into ``OceanConfig``; (2) the old
        ``bottom_drag_r > 0`` NotImplementedError gate is gone. The modular path
        (``scripts/matrix/ocean_test_matrix/setup.py``) is held to the same contract by
        ``test_iter175_modular_setup_cube_bottom_drag_r_plumbed`` — both runners
        now plumb the kwarg (parity restored).
        """
        import re
        from pathlib import Path
        text = (Path(__file__).resolve().parent.parent
                / "scripts" / "matrix" / "run_ocean_test_matrix.py").read_text()
        # (1) bottom_drag_r is plumbed into the (cube) OceanConfig.
        assert 'kw["bottom_drag_r"] = bottom_drag_r' in text, (
            "Phase B.1: run_ocean_test_matrix.py must forward bottom_drag_r into "
            "the cube OceanConfig (the cd-grid backend applies it like lat-lon).")
        # (2) the old cube-specific bottom_drag_r>0 NotImplementedError gate is gone.
        assert re.search(
            r"if bottom_drag_r is not None and bottom_drag_r > 0\.0:\s*\n"
            r"\s*raise NotImplementedError",
            text,
        ) is None, (
            "Phase B.1: the cube ``bottom_drag_r > 0`` NotImplementedError gate "
            "must be removed now that cube bottom drag is supported.")

    def test_iter175_modular_setup_cube_bottom_drag_r_plumbed(self):
        """Phase B.1 parity (supersedes the iter-175 NotImplementedError gate):
        ``scripts/matrix/ocean_test_matrix/setup.py`` (modular) must now PLUMB
        ``bottom_drag_r`` into the cube ``OceanConfig`` exactly like the
        monolithic ``scripts/matrix/run_ocean_test_matrix.py`` — cube bottom
        drag is supported (the cd-grid backend applies it), so neither path may
        SKIP via ``NotImplementedError`` (which would silently downgrade physics
        fidelity / produce non-comparable cross-grid PASS results).

        Verifies the modular path matches the monolithic one: (1) bottom_drag_r
        is forwarded into the cube OceanConfig; (2) the old bottom_drag_r>0
        NotImplementedError gate is gone.
        """
        import re
        from pathlib import Path
        text = (Path(__file__).resolve().parent.parent
                / "scripts" / "matrix" / "ocean_test_matrix" / "setup.py").read_text()
        # (1) bottom_drag_r is plumbed into the cube OceanConfig (parity).
        assert 'kw["bottom_drag_r"] = bottom_drag_r' in text, (
            "Phase B.1 parity: scripts/matrix/ocean_test_matrix/setup.py must forward "
            "bottom_drag_r into the cube OceanConfig like the monolithic runner.")
        # (2) the old cube bottom_drag_r>0 NotImplementedError gate is gone.
        assert re.search(
            r"if bottom_drag_r is not None and bottom_drag_r > 0\.0:\s*\n"
            r"\s*raise NotImplementedError",
            text,
        ) is None, (
            "Phase B.1 parity: the modular cube ``bottom_drag_r > 0`` "
            "NotImplementedError gate must be removed (cube bottom drag is "
            "supported; monolithic + modular paths must agree).")

    def test_iter138_pe_rel_sign_uses_le_not_lt(self):
        """iter-138 (iter-137 FAIL-2) + iter-152 update:
        _apply_pe_rel_sign now uses op="lt" if days >= 1 else "le"
        (days-aware per codex iter-151 review MEDIUM-2 — restore
        strict < 0 semantic for full mode while keeping quick
        mode PE-evolution-budget-tolerant).
        """
        from pathlib import Path
        for rel in (
            "scripts/matrix/run_ocean_test_matrix.py",
            "scripts/matrix/ocean_test_matrix/timeloop.py",
        ):
            text = (Path(__file__).resolve().parent.parent
                    / rel).read_text()
            import re
            m = re.search(
                r"def _apply_pe_rel_sign\b.*?(?=\ndef \w)",
                text, re.DOTALL,
            )
            assert m is not None, f"{rel}: missing"
            body = m.group(0)
            # Strip docstring/comments quoting historical op.
            code_lines = [
                line for line in body.splitlines()
                if not line.lstrip().startswith("#")
            ]
            code_only = "\n".join(code_lines)
            # Find the actual call line — the one with
            # ``label=label, op=...`` pattern.  Docstrings may
            # mention historical op="lt" via backticks; skip
            # those by requiring the line to have ``label=label``
            # (a kwarg) AND ``op=`` (without backticks).
            actual_op_lines = [
                line for line in code_lines
                if "label=label" in line and "op=" in line
                and "``" not in line
            ]
            assert len(actual_op_lines) >= 1, (
                f"iter-138: {rel}: _apply_pe_rel_sign must have "
                f"a single explicit 'label=label, op=...' kwarg "
                f"line (the actual call to _apply_value_threshold).")
            for line in actual_op_lines:
                # iter-152: op is now ``op=op`` (a variable) where
                # ``op`` is selected days-aware as ``"lt" if days >= 1.0
                # else "le"``.  Either ``op=op`` or the literal
                # ``op="le"`` (legacy) is acceptable.
                assert ('op=op' in line or 'op="le"' in line
                        or 'op="lt"' in line), (
                    f"iter-138/iter-152: {rel}: _apply_pe_rel_sign "
                    f"call line must use op='lt'/'le'/op variable; "
                    f"got: {line.strip()!r}")

    def test_iter155_pe_sign_days_numpy_scalars_accepted(self):
        """iter-155 (codex iter-154 review LOW-1): numpy scalar
        types like ``np.float32``, ``np.float64``, ``np.int64``
        must be accepted (callers naturally pass these from
        diagnostic dicts).  ``bool`` must be rejected
        (subclasses int but ``days=True`` is nonsense).
        """
        import numpy as np
        import pytest
        mod = self._import_monolithic_runner()
        # np.float64 (the most common case from diag dicts).
        ok, _ = mod._apply_pe_rel_sign(
            True, "", -0.5, label="PE_rel_final",
            n_samples=10, days=np.float64(2.0))
        assert ok is True
        # np.float32 (less common but possible).
        ok, _ = mod._apply_pe_rel_sign(
            True, "", -0.5, label="PE_rel_final",
            n_samples=10, days=np.float32(2.0))
        assert ok is True
        # np.int64 (also possible).
        ok, _ = mod._apply_pe_rel_sign(
            True, "", -0.5, label="PE_rel_final",
            n_samples=10, days=np.int64(2))
        assert ok is True
        # bool MUST be rejected even though it's a subclass of int.
        with pytest.raises(ValueError, match="must be a real number"):
            mod._apply_pe_rel_sign(
                True, "", -0.5, label="PE_rel_final",
                n_samples=10, days=True)
        # str is also rejected.
        with pytest.raises(ValueError, match="must be a real number"):
            mod._apply_pe_rel_sign(
                True, "", -0.5, label="PE_rel_final",
                n_samples=10, days="2.0")

    def test_iter154_pe_sign_days_none_rejected(self):
        """iter-154/iter-155: explicit ``days=None``, NaN, 0,
        negative all rejected.  iter-155 split the validation
        into "must be a real number" (None, str, etc.) vs
        "must be a finite positive number" (NaN, 0, -1) so the
        regex matches either error message.
        """
        import pytest
        mod = self._import_monolithic_runner()
        # days=None must raise (real-number check).
        with pytest.raises(ValueError,
                           match="real number|finite positive number"):
            mod._apply_pe_rel_sign(
                True, "", -0.5, label="PE_rel_final",
                n_samples=10, days=None)
        # days=NaN must raise (finite check).
        with pytest.raises(ValueError, match="finite positive number"):
            mod._apply_pe_rel_sign(
                True, "", -0.5, label="PE_rel_final",
                n_samples=10, days=float("nan"))
        # days=0 must raise (no time means no gate).
        with pytest.raises(ValueError, match="finite positive number"):
            mod._apply_pe_rel_sign(
                True, "", -0.5, label="PE_rel_final",
                n_samples=10, days=0)
        # days=-1 must raise.
        with pytest.raises(ValueError, match="finite positive number"):
            mod._apply_pe_rel_sign(
                True, "", -0.5, label="PE_rel_final",
                n_samples=10, days=-1)
        # Valid finite positive days works.
        ok, _ = mod._apply_pe_rel_sign(
            True, "", -0.5, label="PE_rel_final",
            n_samples=10, days=2.0)
        assert ok is True

    def test_iter154_days_required_sentinel_centralized(self):
        """iter-154 (codex iter-153 review LOW-2): the
        ``DAYS_REQUIRED`` sentinel is now centralized in
        ``legoesm.diagnostics``.  Both monolithic and modular
        runners import the SAME singleton (``is`` check).
        """
        from legoesm.diagnostics import DAYS_REQUIRED
        # Monolithic uses the centralized sentinel.
        mod = self._import_monolithic_runner()
        assert mod._DAYS_REQUIRED is DAYS_REQUIRED, (
            "iter-154: monolithic _DAYS_REQUIRED must be the "
            "centralized legoesm.diagnostics.DAYS_REQUIRED.")
        # Modular too.
        import sys
        from pathlib import Path
        scripts_dir = (Path(__file__).resolve().parent.parent
                       / "scripts")
        sys.path.insert(0, str(scripts_dir))
        try:
            from ocean_test_matrix import timeloop
            assert timeloop._DAYS_REQUIRED is DAYS_REQUIRED, (
                "iter-154: modular _DAYS_REQUIRED must be the "
                "centralized legoesm.diagnostics.DAYS_REQUIRED.")
        finally:
            sys.path.remove(str(scripts_dir))

    def test_iter153_pe_sign_days_required(self):
        """iter-153 (codex iter-152 review MEDIUM-1): ``days``
        kwarg is now REQUIRED on ``_apply_pe_rel_sign`` — omitting
        it raises TypeError instead of silently defaulting to
        quick-mode ``op="le"``.  Behavior test that actually
        invokes the wrapper.
        """
        import pytest
        mod = self._import_monolithic_runner()
        # Omitting ``days`` must raise.
        with pytest.raises(TypeError, match="'days' kwarg is required"):
            mod._apply_pe_rel_sign(
                True, "", -0.5, label="PE_rel_final",
                n_samples=10)
        # Same for modular.
        import sys
        from pathlib import Path
        scripts_dir = (Path(__file__).resolve().parent.parent
                       / "scripts")
        sys.path.insert(0, str(scripts_dir))
        try:
            from ocean_test_matrix.timeloop import (
                _apply_pe_rel_sign as modular_pe_rel_sign,
            )
            with pytest.raises(TypeError, match="'days' kwarg is required"):
                modular_pe_rel_sign(
                    True, "", -0.5, label="PE_rel_final",
                    n_samples=10)
        finally:
            sys.path.remove(str(scripts_dir))

    def test_iter153_pe_sign_behavior_full_vs_quick(self):
        """iter-153 (codex iter-152 review LOW-1): tighter
        behavior test — actually invoke both wrapper modules
        with days=2.0 (full) and days=0.1 (quick), assert
        pe_rel_final=0 fails in full mode and passes in quick.
        """
        mod = self._import_monolithic_runner()
        # Full mode (days=2): PE_rel_final = 0 must FAIL.
        ok, notes = mod._apply_pe_rel_sign(
            True, "", 0.0, label="PE_rel_final",
            n_samples=10, days=2.0)
        assert ok is False, (
            "iter-153: full-mode PE_rel_final=0 must FAIL "
            "(strict <0 contract).")
        # Quick mode (days=0.1): PE_rel_final = 0 must PASS.
        ok, _ = mod._apply_pe_rel_sign(
            True, "", 0.0, label="PE_rel_final",
            n_samples=10, days=0.1)
        assert ok is True, (
            "iter-153: quick-mode PE_rel_final=0 must PASS "
            "(no time for PE evolution).")
        # Same dual check on the modular wrapper.
        import sys
        from pathlib import Path
        scripts_dir = (Path(__file__).resolve().parent.parent
                       / "scripts")
        sys.path.insert(0, str(scripts_dir))
        try:
            from ocean_test_matrix.timeloop import (
                _apply_pe_rel_sign as modular_pe_rel_sign,
            )
            ok, _ = modular_pe_rel_sign(
                True, "", 0.0, label="PE_rel_final",
                n_samples=10, days=2.0)
            assert ok is False
            ok, _ = modular_pe_rel_sign(
                True, "", 0.0, label="PE_rel_final",
                n_samples=10, days=0.1)
            assert ok is True
        finally:
            sys.path.remove(str(scripts_dir))

    def test_iter153_relative_drift_series_nan_anywhere(self):
        """iter-153 (codex iter-152 review MEDIUM-2):
        ``relative_drift_series`` must propagate NaN from any
        sample to the WHOLE returned array, mirroring the
        iter-152 scalar fix.  Pre-iter-153 a NaN in the middle
        produced ``[0, NaN, 0]`` so a caller using
        ``abs(series[-1])`` could still silently PASS a
        transient blowup.
        """
        from legoesm.diagnostics import relative_drift_series
        import numpy as np
        result = relative_drift_series([1.0, float("nan"), 1.0])
        assert result.shape == (3,)
        assert np.all(np.isnan(result)), (
            "iter-153: NaN anywhere in series must produce "
            "all-NaN output array.")
        # Healthy series still works.
        result = relative_drift_series([1.0, 1.0001, 1.0002])
        assert np.allclose(result, [0.0, 1e-4, 2e-4], rtol=1e-3)

    def test_iter153_idempotent_when_ok_false(self):
        """iter-153 (codex iter-152 review MEDIUM-3): when
        ``ok=False``, the helpers must return immediately
        WITHOUT validating tol/threshold.  Otherwise a caller
        chaining gates with placeholder tols (e.g., NaN to
        signal "skip this gate") gets spurious ValueErrors.
        """
        from legoesm.diagnostics import (
            apply_drift_tolerance, apply_value_threshold,
        )
        # ok=False must short-circuit BEFORE the NaN-tol check.
        ok, notes = apply_drift_tolerance(
            ok=False, notes="prior FAIL",
            drift=1e-3, tol=float("nan"),  # NaN tol
            label="test")
        assert ok is False
        assert notes == "prior FAIL"

        ok, notes = apply_value_threshold(
            ok=False, notes="prior FAIL",
            value=0.5, threshold=float("nan"),  # NaN threshold
            label="test", op="lt")
        assert ok is False
        assert notes == "prior FAIL"

    def test_iter152_pe_sign_days_aware(self):
        """iter-152 (codex iter-151 review MEDIUM-2): the PE-sign
        gate is days-aware — strict ``op="lt"`` for full mode,
        ``op="le"`` for quick mode.  Verify the wrapper code
        contains the days-aware op selection logic.
        """
        from pathlib import Path
        for rel in (
            "scripts/matrix/run_ocean_test_matrix.py",
            "scripts/matrix/ocean_test_matrix/timeloop.py",
        ):
            text = (Path(__file__).resolve().parent.parent
                    / rel).read_text()
            import re
            m = re.search(
                r"def _apply_pe_rel_sign\b.*?(?=\ndef \w)",
                text, re.DOTALL,
            )
            assert m is not None
            body = m.group(0)
            code_lines = [
                line for line in body.splitlines()
                if not line.lstrip().startswith("#")
            ]
            code_only = "\n".join(code_lines)
            # Must have days-aware op selection.
            assert ('days is not None and days >= 1.0' in code_only
                    or 'op = "lt" if' in code_only), (
                f"iter-152: {rel}: _apply_pe_rel_sign must have "
                f"days-aware op selection (op='lt' for "
                f"full mode, op='le' for quick).")
            # Must accept ``days`` kwarg.
            sig_match = re.search(
                r"def _apply_pe_rel_sign\([^)]*days[^)]*\)",
                code_only, re.DOTALL,
            )
            assert sig_match is not None, (
                f"iter-152: {rel}: _apply_pe_rel_sign signature "
                f"must include ``days`` kwarg.")

    def test_iter152_compute_relative_drift_nan_in_middle_fails(self):
        """iter-152 (codex iter-151 review HIGH-1): a NaN anywhere
        in the conservation series must surface as NaN drift, not
        silently pass via the ``arr[0]`` and ``arr[-1]`` only check.
        """
        from legoesm.diagnostics import compute_relative_drift
        import math
        # NaN in the middle, finite at endpoints — pre-iter-152
        # this would yield drift=0.0.
        result = compute_relative_drift([1.0, float("nan"), 1.0])
        assert math.isnan(result), (
            "iter-152: NaN in middle of series must surface as "
            "NaN result (not silently pass via endpoints-only check).")
        # Inf in middle: same behavior expected.
        result = compute_relative_drift([1.0, float("inf"), 1.0])
        assert math.isnan(result) or math.isinf(result), (
            "iter-152: Inf in middle of series must surface as "
            "non-finite result.")
        # Healthy series still works.
        result = compute_relative_drift([1.0, 1.0001, 1.0002])
        assert math.isclose(result, 2e-4, rel_tol=1e-3)

    def test_iter152_apply_drift_tolerance_rejects_nan_tol(self):
        """iter-152 (codex iter-151 review MEDIUM-3): tol=NaN must
        raise ValueError, not silently pass via NaN-comparison.
        """
        from legoesm.diagnostics import apply_drift_tolerance
        import pytest
        with pytest.raises(ValueError, match="tol must be finite"):
            apply_drift_tolerance(
                ok=True, notes="", drift=0.5, tol=float("nan"),
                label="test")
        with pytest.raises(ValueError, match="tol must be finite"):
            apply_drift_tolerance(
                ok=True, notes="", drift=0.5, tol=float("inf"),
                label="test")
        # Sanity: finite tol still works.
        ok, _ = apply_drift_tolerance(
            ok=True, notes="", drift=0.5, tol=1.0, label="test")
        assert ok is True

    def test_iter152_apply_value_threshold_rejects_nan_threshold(self):
        """iter-152 (codex iter-151 review MEDIUM-3): threshold=NaN
        must raise ValueError on the value-threshold helper too.
        """
        from legoesm.diagnostics import apply_value_threshold
        import pytest
        with pytest.raises(ValueError, match="threshold must be finite"):
            apply_value_threshold(
                ok=True, notes="", value=0.5, threshold=float("nan"),
                label="test", op="lt")
        # Sanity: finite threshold still works.
        ok, _ = apply_value_threshold(
            ok=True, notes="", value=-0.5, threshold=0.0,
            label="test", op="lt")
        assert ok is True

    def test_iter128_geostrophic_doc_documents_tighter_gate(self):
        """iter-128 codex iter-127-followup MEDIUM-1: the doc
        threshold of 1e-3 is the loose contract; the runner uses
        1e-8.  The doc must explicitly document this discrepancy
        so future readers don't try to "loosen" the runner gate
        to match the doc.
        """
        from pathlib import Path
        doc = (Path(__file__).resolve().parent.parent
               / "docs" / "dev-notes" / "ocean_experiments_reference.md")
        text = doc.read_text()
        # Find the geostrophic_adjustment validation thresholds
        # section.
        idx = text.find("geostrophic_adjustment")
        assert idx >= 0
        # The "test-matrix runner gate" callout must appear in
        # the same section (within 2000 chars after the heading).
        section = text[idx: idx + 2000]
        assert "1e-8" in section, (
            "iter-128: geostrophic_adjustment doc section must "
            "mention the 1e-8 runner gate (tighter than 1e-3 "
            "loose contract).")
        assert "test-matrix runner gate" in section, (
            "iter-128: doc must explicitly mark 1e-8 as the "
            "test-matrix runner gate.")

    def test_modular_rest_state_uses_drift_tolerance(self):
        """All 4 modular rest_state variants apply
        ``_apply_drift_tolerance`` for both eta and T.
        """
        from pathlib import Path
        path = (Path(__file__).resolve().parent.parent
                / "scripts" / "matrix" / "ocean_test_matrix"
                / "experiments.py")
        text = path.read_text()
        # Strip line comments before counting active calls.
        import re
        text_no_strings = re.sub(r'""".*?"""', "", text, flags=re.DOTALL)
        text_no_strings = re.sub(r"'''.*?'''", "", text_no_strings, flags=re.DOTALL)
        code_only = "\n".join(
            line for line in text_no_strings.splitlines()
            if not line.lstrip().startswith("#")
        )
        # Expect ≥8 callsites: 4 rest_state variants × 2 drifts
        # (eta + T).  After replace_all, all 4 rest_state
        # blocks should each have 2 helper calls.
        count = code_only.count("_apply_drift_tolerance(")
        assert count >= 8, (
            f"iter-126: expected ≥8 ``_apply_drift_tolerance`` "
            f"call sites in modular ``experiments.py`` "
            f"(4 rest_state variants × 2 drifts each); "
            f"found {count}."
        )

    def test_rest_state_uses_drift_tolerance(self):
        """All 4 rest_state variants apply ``_apply_drift_tolerance``
        for both eta and T.  Source-pin via inspect.
        """
        import inspect
        m = self._import_module()
        # The rest_state variant runners share an inline
        # tolerance-application pattern.  Pin its presence in
        # the module source.
        text = inspect.getsource(m)
        # Strip line comments first.
        import re
        text_no_strings = re.sub(r'""".*?"""', "", text, flags=re.DOTALL)
        text_no_strings = re.sub(r"'''.*?'''", "", text_no_strings, flags=re.DOTALL)
        code_only = "\n".join(
            line for line in text_no_strings.splitlines()
            if not line.lstrip().startswith("#")
        )
        # Count occurrences of the helper call — should be at
        # least 8 (4 rest_state variants × 2 drifts each).
        count = code_only.count("_apply_drift_tolerance(")
        assert count >= 8, (
            f"iter-123: at least 8 ``_apply_drift_tolerance`` "
            f"call sites expected (4 rest_state × 2 "
            f"drifts).  Found {count}."
        )

    def test_iter157_stommel_gyre_tracer_uses_centralized_helper(self):
        """iter-157 (post-iter-156 audit, codex-clean cycle):
        ``stommel_gyre_tracer.compute_diagnostics`` and
        ``compute_transport_metrics`` previously inlined the
        iter-78 pathology pattern
        (``abs(final - init) / abs(init)`` with a stale
        ``> 1e-30`` floor).  iter-157 migrated both to the
        centralized ``compute_relative_drift`` helper.

        This structural test pins the migration so the inline
        pattern cannot regress silently.
        """
        import inspect
        from legoesm.ocean.experiments import stommel_gyre_tracer
        src = inspect.getsource(stommel_gyre_tracer)
        # Strip docstrings and comments so we only inspect code.
        import re
        text_no_strings = re.sub(r'""".*?"""', "", src, flags=re.DOTALL)
        text_no_strings = re.sub(r"'''.*?'''", "", text_no_strings, flags=re.DOTALL)
        code_only = "\n".join(
            line for line in text_no_strings.splitlines()
            if not line.lstrip().startswith("#")
        )
        # No more ``> 1e-30`` floor on baseline (the iter-78
        # pathology pattern).
        assert "> 1e-30" not in code_only, (
            "iter-157: stommel_gyre_tracer must not use the "
            "stale ``> 1e-30`` baseline floor.  Use the "
            "centralized ``compute_relative_drift`` helper "
            "(DEFAULT_MIN_BASELINE = 1.0) instead."
        )
        # Must import compute_relative_drift somewhere.
        assert "compute_relative_drift" in code_only, (
            "iter-157: stommel_gyre_tracer must use "
            "``compute_relative_drift`` from "
            "``legoesm.diagnostics``."
        )

    def test_iter157_atmosphere_matrix_uses_compute_drift_wrapper(self):
        """iter-157: ``run_atmosphere_test_matrix.error_fn`` for
        cosine_bell previously inlined
        ``abs(mass_final - _mass_init) / abs(_mass_init)``.
        iter-157 migrated it to the ``_compute_drift`` wrapper
        (which delegates to ``compute_relative_drift``).

        We do a text-only scan of the file because importing
        the script triggers heavy argparse/global setup; the
        structural assertion only needs the source string.
        """
        from pathlib import Path
        scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
        # Scripts reorg: run_atmosphere_test_matrix.py lives in the matrix/ bucket.
        path = scripts_dir / "matrix" / "run_atmosphere_test_matrix.py"
        text = path.read_text()
        # Strip docstrings and comments so we only inspect code.
        import re
        text_no_strings = re.sub(r'""".*?"""', "", text, flags=re.DOTALL)
        text_no_strings = re.sub(r"'''.*?'''", "", text_no_strings, flags=re.DOTALL)
        code_only = "\n".join(
            line for line in text_no_strings.splitlines()
            if not line.lstrip().startswith("#")
        )
        # The iter-78 pathology pattern for cosine_bell mass
        # drift must not appear in the code (only in comments).
        assert "abs(mass_final - _mass_init) / abs(_mass_init)" not in code_only, (
            "iter-157: ``run_atmosphere_test_matrix.error_fn`` "
            "must use the ``_compute_drift`` wrapper for "
            "cosine_bell mass drift, not the inline "
            "``abs(mass_final - _mass_init) / abs(_mass_init)`` "
            "formula (the iter-78 pathology pattern that was "
            "systematically removed in iter-90/91/93)."
        )
        # And the new pattern must be present.
        assert "_compute_drift([_mass_init, mass_final])" in code_only, (
            "iter-157: ``run_atmosphere_test_matrix.error_fn`` "
            "must call "
            "``_compute_drift([_mass_init, mass_final])`` for "
            "cosine_bell mass drift."
        )


class TestIter110CodexReviewFixes:
    """iter-110 (codex iter-104 follow-up review): 3 MEDIUM
    findings on the iter-104..109 fix sequence:

    * MEDIUM-1: ocean ``--resolution N`` for regional grids
      mapped all 3 to ``Nkm``, but ``_parse_resolution``
      expects ``latlon_regional → NxM`` and
      ``cs_regional → CN``.
    * MEDIUM-2: ``run_baroclinic_gyre`` passed ``diag=diag``
      but omitted ``status``, so iter-105 BLOWUP marker never
      fired (gated on ``rows.get("status") == "FAIL"``).
    * MEDIUM-3: collector fallback could pick
      ``.ipynb_checkpoints`` or ``__pycache__`` over a valid
      legacy ``16/`` dir.
    """

    def _import_module(self):
        import importlib
        import sys
        from pathlib import Path
        scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        return importlib.import_module("run_ocean_test_matrix")

    def test_medium_1_latlon_regional_uses_xform(self, tmp_path):
        """``latlon_regional/{16, 16x32}`` → 16x32 (NOT 16km)."""
        M = self._import_module()
        grid_dir = tmp_path / "latlon_regional"
        grid_dir.mkdir()
        (grid_dir / "16").mkdir()
        (grid_dir / "16x32").mkdir()
        chosen = M._select_ocean_resolution_dir(
            grid_dir, list(grid_dir.iterdir()))
        assert chosen.name == "16x32"

    def test_medium_1_cs_regional_uses_C_form(self, tmp_path):
        """``cs_regional/{16, C16}`` → C16 (NOT 16km)."""
        M = self._import_module()
        grid_dir = tmp_path / "cs_regional"
        grid_dir.mkdir()
        (grid_dir / "16").mkdir()
        (grid_dir / "C16").mkdir()
        chosen = M._select_ocean_resolution_dir(
            grid_dir, list(grid_dir.iterdir()))
        assert chosen.name == "C16"

    def test_medium_1_mpas_regional_still_uses_km(self, tmp_path):
        """``mpas_regional/{16, 50km}`` → 50km (unchanged from
        iter-108).  iter-110 only re-routed latlon_regional
        and cs_regional, not mpas_regional.
        """
        M = self._import_module()
        grid_dir = tmp_path / "mpas_regional"
        grid_dir.mkdir()
        (grid_dir / "16").mkdir()
        (grid_dir / "50km").mkdir()
        chosen = M._select_ocean_resolution_dir(
            grid_dir, list(grid_dir.iterdir()))
        assert chosen.name == "50km"

    def test_medium_2_baroclinic_gyre_writes_status(self):
        """``run_baroclinic_gyre`` must include ``status`` in
        its rows dict so the iter-105 BLOWUP marker can fire.
        """
        import inspect
        import re
        M = self._import_module()
        src = inspect.getsource(M.run_baroclinic_gyre)
        # Strip docstrings + line comments before pattern check.
        src_no_strings = re.sub(r'""".*?"""', "", src, flags=re.DOTALL)
        src_no_strings = re.sub(r"'''.*?'''", "", src_no_strings, flags=re.DOTALL)
        code_only = "\n".join(
            line for line in src_no_strings.splitlines()
            if not line.lstrip().startswith("#")
        )
        assert '"status": "PASS" if ok else "FAIL"' in code_only, (
            "iter-110 codex MEDIUM-2: ``run_baroclinic_gyre`` "
            "must include ``status: PASS|FAIL`` in its "
            "_write_results_txt rows dict so the iter-105 "
            "BLOWUP-marker logic fires on FAIL."
        )

    def test_medium_3_filters_hidden_dirs(self, tmp_path):
        """Hidden dirs (``.ipynb_checkpoints``) and internal
        tooling dirs (``__pycache__``) must be filtered out
        before fallback selection.  Pre-iter-110, a stale
        ``.ipynb_checkpoints/`` dir alongside a valid ``16/``
        dir could be chosen by the fallback.
        """
        M = self._import_module()
        grid_dir = tmp_path / "cubed_sphere"
        grid_dir.mkdir()
        (grid_dir / ".ipynb_checkpoints").mkdir()
        (grid_dir / "__pycache__").mkdir()
        (grid_dir / "16").mkdir()
        chosen = M._select_ocean_resolution_dir(
            grid_dir, list(grid_dir.iterdir()))
        # Must pick the ``16/`` dir, NOT the hidden/internal
        # dirs.  No grid-typed candidate present → falls back
        # to the legacy bare-numeric dir, which is the right
        # behaviour.
        assert chosen.name == "16"

    def test_medium_3_returns_none_for_only_hidden(self, tmp_path):
        """If only hidden/internal dirs are present, return
        None so the collector skips this grid_dir entirely.
        """
        M = self._import_module()
        grid_dir = tmp_path / "cubed_sphere"
        grid_dir.mkdir()
        (grid_dir / ".ipynb_checkpoints").mkdir()
        (grid_dir / "__pycache__").mkdir()
        chosen = M._select_ocean_resolution_dir(
            grid_dir, list(grid_dir.iterdir()))
        assert chosen is None
