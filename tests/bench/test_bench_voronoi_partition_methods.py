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


class _SyntheticMesh:
    """4-cell ring: cells 0-1-2-3 cyclic (each cell has 2 neighbors).

    Edges: (0,1) (1,2) (2,3) (3,0) + one INVALID edge (-1,-1) to lock the
    valid-edge masking in the edge-cut denominator.
    """
    nCells = 4
    nEdges = 5
    maxEdges = 2
    cellsOnEdge = np.array([[0, 1, 2, 3, -1],
                            [1, 2, 3, 0, -1]])
    cellsOnCell = np.array([[1, 2, 3, 0],    # neighbor k=0
                            [3, 0, 1, 2]])   # neighbor k=1


def test_metric_definitions_locked_on_synthetic_mesh():
    """Exact edge cut / halo / neighbor values on a hand-built ring —
    a denominator or halo-construction drift fails HERE, not in a range
    check (codex finding 3)."""
    mesh = _SyntheticMesh()
    owner = np.array([0, 0, 1, 1])  # cells 0,1 -> rank0; 2,3 -> rank1
    q = mod.partition_quality(mesh, owner, 2, halo_depth=1)
    # Cut edges: (1,2) and (3,0) -> 2 of 4 VALID edges (invalid edge
    # excluded from the denominator).
    assert q["edge_cut"] == 2
    assert q["edge_cut_fraction"] == pytest.approx(0.5)
    # halo_depth=1: each rank's halo = the 2 cells of the other rank that
    # touch it (ring: both of them).
    assert q["halo_cells_max"] == 2
    assert q["halo_cells_mean"] == pytest.approx(2.0)
    assert q["halo_owned_ratio_max"] == pytest.approx(1.0)
    assert q["neighbor_ranks_max"] == 1
    assert q["cells_per_rank_min"] == q["cells_per_rank_max"] == 2
    assert q["load_imbalance_max_over_mean"] == pytest.approx(1.0)


def test_empty_rank_rejected():
    mesh = _SyntheticMesh()
    owner = np.array([0, 0, 0, 0])  # rank 1 skipped
    with pytest.raises(AssertionError, match="empty rank"):
        mod.partition_quality(mesh, owner, 2)


def test_halo_matches_runtime_partition():
    """Real-mesh lock: the bench's halo size equals the RUNTIME partition's
    (n_local - n_owned) for the same owner array — the bench reports the
    runtime's halos, not an estimate (codex finding 3)."""
    from legoesm.parallel.voronoi_partition import partition_voronoi_mesh

    mesh = _mesh()
    owner = mod.owner_for(mesh, "geometric", 4)
    q = mod.partition_quality(mesh, owner, 4, halo_depth=2)
    halos = []
    for r in range(4):
        part = partition_voronoi_mesh(
            mesh, 4, r, method="geometric", halo_depth=2, cell_owner=owner)
        halos.append(int(part.n_local_cells) - int(part.n_owned_cells))
    assert q["halo_cells_max"] == max(halos)
    assert q["halo_cells_mean"] == pytest.approx(float(np.mean(halos)))
