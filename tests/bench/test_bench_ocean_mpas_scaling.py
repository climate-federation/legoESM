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
