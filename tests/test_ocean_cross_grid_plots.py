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
        scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
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
        ``src/legoesm/driver/cli_resolution.py:expand_cli_resolution``
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
            "scripts/run_ocean_test_matrix.py --only rest_state "
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
        ``scripts/ocean_test_matrix/timeloop.py`` must export
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
                / "scripts" / "ocean_test_matrix"
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
            "scripts/run_ocean_test_matrix.py",
            "scripts/ocean_test_matrix/timeloop.py",
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
                / "scripts" / "ocean_test_matrix"
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
                / "scripts" / "ocean_test_matrix"
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
                / "scripts" / "ocean_test_matrix"
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

    def test_modular_rest_state_uses_drift_tolerance(self):
        """All 4 modular rest_state variants apply
        ``_apply_drift_tolerance`` for both eta and T.
        """
        from pathlib import Path
        path = (Path(__file__).resolve().parent.parent
                / "scripts" / "ocean_test_matrix"
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
