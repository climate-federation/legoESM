"""Direct tests for the Voronoi partition-method quality bench (item 8)."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_BENCH = (Path(__file__).resolve().parents[2]
          / "scripts" / "bench" / "bench_voronoi_partition_methods.py")
_spec = importlib.util.spec_from_file_location("bench_vor_part", _BENCH)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def _mesh(level=2):
    from legoesm.grids.voronoi import create_voronoi_mesh

    return create_voronoi_mesh(subdivision_level=level)


def test_partition_quality_metrics_shape_and_sanity():
    mesh = _mesh()
    owner = mod.owner_for(mesh, "geometric", 4)
    q = mod.partition_quality(mesh, owner, 4)
    assert q["cells_per_rank_min"] >= 1
    assert q["cells_per_rank_max"] >= q["cells_per_rank_min"]
    assert q["load_imbalance_max_over_mean"] >= 1.0
    assert 0 < q["edge_cut"] < int(mesh.nEdges)
    assert 0.0 < q["edge_cut_fraction"] < 1.0
    assert q["halo_cells_max"] >= q["halo_cells_mean"] > 0
    assert 1 <= q["neighbor_ranks_max"] < 4


def test_partition_quality_rejects_bad_owner():
    mesh = _mesh()
    n = int(mesh.nCells)
    with pytest.raises(AssertionError, match="out of range"):
        mod.partition_quality(mesh, np.full(n, 7), 4)
    with pytest.raises(AssertionError, match="shape"):
        mod.partition_quality(mesh, np.zeros(n - 1, dtype=int), 4)


def test_owner_for_unknown_method_raises():
    mesh = _mesh()
    with pytest.raises(ValueError, match="unknown partition method"):
        mod.owner_for(mesh, "voodoo", 4)


def test_method_available_never_substitutes():
    # geometric/sfc are dependency-free; metis truthfully reports.
    assert mod.method_available("geometric") is True
    assert mod.method_available("sfc") is True
    try:
        import pymetis  # noqa: F401

        assert mod.method_available("metis") is True
    except Exception:
        assert mod.method_available("metis") is False


def test_main_writes_quality_table(tmp_path, monkeypatch):
    out = tmp_path / "q.json"
    monkeypatch.setattr(sys, "argv", [
        "bench", "--subdivision", "2", "--rank-counts", "2,4",
        "--methods", "geometric,sfc", "--out", str(out)])
    assert mod.main() == 0
    payload = json.loads(out.read_text())
    rows = [r for r in payload["rows"] if r.get("available")]
    assert {(r["method"], r["n_ranks"]) for r in rows} == {
        ("geometric", 2), ("geometric", 4), ("sfc", 2), ("sfc", 4)}
    assert payload["auto_resolves_to"] in ("metis", "geometric")
    md = payload["metadata"]
    assert md["transport"] == "none"
    assert "_incomplete" not in md


def test_main_rejects_bad_args(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["bench", "--methods", "voodoo"])
    with pytest.raises(SystemExit):
        mod.main()
    monkeypatch.setattr(sys, "argv", ["bench", "--rank-counts", "1"])
    with pytest.raises(SystemExit):
        mod.main()
