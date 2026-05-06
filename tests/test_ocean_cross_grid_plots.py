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
    ``scripts/run_ocean_test_matrix.py:_compute_drift`` (a script-
    level copy) had been missed in iter-88's "factor into shared
    helper" refactor.  iter-91 audited more aggressively and
    found two MORE missed copies in the sibling
    ``scripts/ocean_test_matrix/`` package:

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
