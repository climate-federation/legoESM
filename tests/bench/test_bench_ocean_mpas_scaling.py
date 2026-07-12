"""Direct tests for the MPAS/Voronoi OCEAN scaling lane (audit item 6).

Covers the pure helpers (weak-level quantization, owned-cell gather
coverage), the CLI rejections, and a single-process end-to-end main() run
on the tiny L2 mesh with BOTH gates armed — locking the record schema
(shared metadata v2 + partition fields) that downstream aggregation reads.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_BENCH = (Path(__file__).resolve().parents[2]
          / "scripts" / "bench" / "bench_ocean_mpas_scaling.py")
_spec = importlib.util.spec_from_file_location("bench_ocean_mpas", _BENCH)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def test_weak_level_quantization():
    # L4 = 2,562 cells: 1 rank at the default target picks L4 exactly.
    assert mod.weak_level_for(2562, 1) == 4
    # 4 ranks at the same per-rank target need 4x the cells -> L5.
    assert mod.weak_level_for(2562, 4) == 5
    # Ratio metric: 4x-quantized levels pick the CLOSEST ratio, never 0/inf.
    lv = mod.weak_level_for(1000, 3)
    assert lv in mod.WEAK_LEVELS


def test_ncells_formula():
    assert mod._ncells(2) == 162
    assert mod._ncells(4) == 2562


def test_main_rejects_bad_timing_window(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["bench", "--steps", "0"])
    with pytest.raises(SystemExit):
        mod.main()
    monkeypatch.setattr(sys, "argv", ["bench", "--steps", "4",
                                      "--warmup", "4"])
    with pytest.raises(SystemExit):
        mod.main()


def test_main_rejects_bad_fused_schedule(monkeypatch):
    # --warmup 0 so the timing-window guard cannot fire first — these
    # must exercise the fused-schedule guards specifically.
    for extra, msg in ((["--block-steps", "-1"], "block-steps"),
                       (["--blocks", "0"], "blocks"),
                       (["--probe-steps", "-2"], "probe-steps")):
        monkeypatch.setattr(
            sys, "argv", ["bench", "--steps", "2", "--warmup", "0"] + extra)
        with pytest.raises(SystemExit, match=msg):
            mod.main()


def test_main_rejects_per_step_halo_single_rank(monkeypatch):
    # Dispatch hardening: an explicit per_step request that cannot be
    # honored (no partition layout single-rank) is a hard error, never a
    # silent no-op.
    # --warmup 0 so the timing-window guard cannot fire first — this
    # must exercise the halo-refresh guard specifically.
    monkeypatch.setattr(sys, "argv", [
        "bench", "--steps", "2", "--warmup", "0",
        "--halo-refresh", "per_step"])
    with pytest.raises(SystemExit, match="per_step"):
        mod.main()


def test_m1_lane_flags_exist():
    # Wiring tripwire (OL gate pattern): a rename would only break at
    # runtime on a cluster.
    src = _BENCH.read_text()
    for flag in ("--barotropic-solver", "--halo-refresh", "--block-steps",
                 "--blocks", "--probe-steps", "--parity-gate",
                 "--check-conservation"):
        assert flag in src, flag
    # Unknown solver literals must be rejected by argparse choices.
    import argparse  # noqa: F401  (documents the surface under test)
    assert 'choices=["explicit_substep", "implicit_cn"]' in src
    # implicit_cn parity at n_ranks>1 is impossible (the serial reference
    # trips the solver's layout-less multi-rank refusal) — the bench must
    # refuse the combination loudly, never crash mid-reference.
    assert "layout-less multi-rank refusal" in src


def test_main_rejects_long_parity_window(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "bench", "--steps", "30", "--parity-gate"])
    with pytest.raises(SystemExit):
        mod.main()


def test_main_single_rank_writes_gated_record(tmp_path, monkeypatch):
    out = tmp_path / "rec.jsonl"
    monkeypatch.setattr(sys, "argv", [
        "bench", "--subdivision", "2", "--nlev", "3", "--steps", "3",
        "--warmup", "1", "--parity-gate", "--check-conservation",
        "--block-steps", "2", "--blocks", "1", "--probe-steps", "1",
        "--out", str(out)])
    assert mod.main() == 0
    rec = json.loads(out.read_text().splitlines()[-1])
    assert rec["component"] == "ocean"
    assert rec["grid"] == "voronoi"
    assert rec["n_cells"] == 162
    md = rec["metadata"]
    # Shared self-describing record (schema v2) — the anti-fake-scaling
    # fields a row must carry.
    assert md["schema_version"] >= 2
    for k in ("transport", "virtual_cpu_devices", "launcher", "backend",
              "decomposition", "precision_knobs", "cells_per_rank"):
        assert k in md, k
    assert md["decomposition"] == "none"  # single rank
    assert "_incomplete" not in md
    # --- M1 lane fields (scaling-M3d increment-1) ---
    assert rec["barotropic_solver"] == "explicit_substep"
    assert rec["halo_refresh"] == "none"          # auto @ np=1
    assert rec["stage_halo_correct"] is True      # no partition @ np=1
    assert rec["stage_halo_note"] is None
    # Fused-scan M1 contract: headline + separate dispatch-latency probe.
    fused = rec["fused"]
    assert fused["block_steps"] == 2 and fused["n_blocks"] == 1
    assert fused["fused_step_ms"] > 0
    assert fused["step_latency_ms"] > 0
    assert fused["rank_imbalance"] == 1.0         # single process
    # Wet-cell metrics: internally consistent (land presence at L2
    # depends on mesh orientation vs land_lat_threshold — not asserted).
    wet = rec["wet_cell"]
    total = rec["n_cells"] * 3
    assert 0 < wet["wet_cell_levels"] <= total
    assert wet["wet_equals_total"] == (wet["wet_cell_levels"] == total)
    assert wet["wet_cell_levels_per_device"] == wet["wet_cell_levels"]
    assert wet["wet_cell_levels_per_device_min"] == wet["wet_cell_levels"]
    # explicit_substep: no iterative solve, honestly-null residual.
    assert rec["solver_iters"] is None
    assert "explicit_substep" in rec["solver_iters_mode"]
    assert rec["zero_forcing_probe_measured"] is False
    assert rec["zero_forcing_probe_residual"] is None
    for k in ("halo_refresh", "barotropic_solver", "block_steps"):
        assert k in md["extra"], k


def test_main_single_rank_implicit_cn_residual_probe(tmp_path, monkeypatch):
    out = tmp_path / "rec_impl.jsonl"
    monkeypatch.setattr(sys, "argv", [
        "bench", "--subdivision", "2", "--nlev", "3", "--steps", "2",
        "--warmup", "0", "--barotropic-solver", "implicit_cn",
        "--block-steps", "0", "--probe-steps", "0",
        "--out", str(out)])
    assert mod.main() == 0
    rec = json.loads(out.read_text().splitlines()[-1])
    assert rec["barotropic_solver"] == "implicit_cn"
    # --block-steps 0 disables the fused measurement (honest null).
    assert rec["fused"] is None
    # Zero-forcing Helmholtz residual probe: measured, finite, converged
    # to the solver's ballpark on the tiny mesh (health evidence).
    assert rec["zero_forcing_probe_measured"] is True
    res = rec["zero_forcing_probe_residual"]
    assert res is not None and np.isfinite(res)
    assert res < 1e-3, f"implicit-CN probe residual suspiciously large: {res}"
    # Single-rank stock CG does not expose an iteration count.
    assert rec["solver_iters"] is None
    assert "adaptive_stock_cg" in rec["solver_iters_mode"]
    assert rec["metadata"]["solver_variant"] == "mpas_ocean_implicit_cn"


def test_gather_owned_cells_coverage_check():
    class _Part:
        n_owned_cells = 2
        local_cells = np.array([0, 2, 1])  # owned: 0,2 (halo: 1)

    class _Comm:
        def gather(self, piece, root=0):
            return [piece]

        def Get_rank(self):
            return 0

    # Owned cells 0,2 of a 3-cell global leave cell 1 UNCOVERED -> the
    # NaN-fill coverage check must raise (partition-coverage tripwire).
    with pytest.raises(RuntimeError, match="unowned"):
        mod.gather_owned_cells(np.array([1.0, 3.0, 2.0]), _Part(), _Comm(), 3)

    class _Part2(_Part):
        n_owned_cells = 3

    out = mod.gather_owned_cells(
        np.array([1.0, 3.0, 2.0]), _Part2(), _Comm(), 3)
    np.testing.assert_array_equal(out, [1.0, 2.0, 3.0])
