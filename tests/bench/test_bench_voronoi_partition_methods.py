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


# --- SPMD halo-schedule depth (--schedule-cost) ---------------------------


def test_schedule_cost_row_reports_rounds_against_their_lower_bound():
    """``schedule_cost_row`` returns the REAL schedule depth and the bound it
    must be read against."""
    mesh = _mesh()
    sc = mod.schedule_cost_row(mesh, "geometric", 2)
    # max_degree is the graph's own lower bound on a proper edge colouring,
    # so a schedule can never beat it.  -1 is the scorer's "not reported"
    # sentinel and would make the gap meaningless.
    assert sc["max_degree"] >= 1
    assert sc["n_rounds"] >= sc["max_degree"]
    assert sc["coloring_gap"] == sc["n_rounds"] - sc["max_degree"]
    assert sc["n_rounds_greedy"] >= sc["n_rounds"]
    assert sc["score_seconds"] >= 0.0


@pytest.mark.parametrize(
    "n_rounds, max_degree, gap, hmin, hmax, proven",
    [
        # Provably optimal: no proper edge colouring beats max_degree.
        (12, 12, 0, 0, 0, True),
        # Vizing allows the true optimum to BE max_degree+1, so a gap of 1
        # spans [0, 1] — inconclusive.  This is the case that a naive
        # "gap > 0 means recolour" rule would over-claim.
        (13, 12, 1, 0, 1, False),
        # Guaranteed to remove at least gap-1 = 3, at most gap = 4.
        (14, 10, 4, 3, 4, False),
    ])
def test_coloring_gap_and_headroom_are_derived_not_assumed(
        monkeypatch, n_rounds, max_degree, gap, hmin, hmax, proven):
    """Gap and the Vizing-bounded headroom must be COMPUTED, not assumed.

    Non-vacuity, the hard way: on every mesh small enough to test quickly the
    real gap is 0 (measured L2/L3/L4 x {geometric,sfc} x nd 2-16, and s6
    lloyd=0 at np8/np16 — the colourer lands exactly on ``max_degree`` every
    time), so a real-mesh assertion cannot tell a correct subtraction from a
    hardcoded ``0``; that exact mutation passed the first version of this
    test.  Stubbing the production scorer with KNOWN values is what makes
    the assertion able to fail.

    The ``gap == 1`` row is the one that matters: the schedule is a proper
    EDGE colouring and ``max_degree`` is that graph's max vertex degree, so
    Vizing gives ``Delta <= chi' <= Delta + 1``.  A gap of 1 is therefore
    indistinguishable from optimal (Class 2), and claiming recolouring
    headroom there would be an over-claim.
    """
    stub = {
        "n_rounds": n_rounds, "max_degree": max_degree,
        "n_rounds_greedy": n_rounds + 1, "coloring_method": "stub",
        "resolved_method": "geometric", "halo_depth": 3,
        "cells_per_device": 99_999, "production_strategy": "ppermute",
        "reorder_target": 8, "already_reordered": False,
    }
    import legoesm.parallel.sharded_dynamics as sd
    monkeypatch.setattr(sd, "spmd_schedule_cost", lambda *a, **k: stub)

    sc = mod.schedule_cost_row(object(), "geometric", 8)
    assert sc["coloring_gap"] == gap
    # The headroom is an INTERVAL: Vizing pins the optimum to
    # {Delta, Delta+1}, so gap-1 is guaranteed and gap is the best case.
    assert sc["coloring_headroom_rounds_min"] == hmin
    assert sc["coloring_headroom_rounds_max"] == hmax
    assert sc["coloring_optimal_proven"] is proven


def test_schedule_rounds_never_beat_the_vizing_floor_on_a_real_mesh():
    """Real-mesh sanity on the bound itself: a proper edge colouring can
    never use fewer rounds than the graph's max degree, and the multi-start
    search should not overshoot Vizing's ``Delta + 1`` either.  If this ever
    fires, ``max_degree`` is not the degree of the graph being coloured and
    every gap-based conclusion built on it is void."""
    for method in ("geometric", "sfc"):
        for n_ranks in (2, 4, 8):
            sc = mod.schedule_cost_row(_mesh(level=3), method, n_ranks)
            assert sc["max_degree"] <= sc["n_rounds"] <= sc["max_degree"] + 1, (
                f"{method} np={n_ranks}: rounds={sc['n_rounds']} "
                f"max_degree={sc['max_degree']}")


def test_schedule_cost_scores_the_spmd_depth_not_the_mpi_halo_depth():
    """The schedule is scored at the SPMD production halo depth, NOT this
    script's ``--halo-depth`` (the MPI lane's 2).  Scoring at 2 would colour
    a different graph and quietly report the wrong lane's cost."""
    from legoesm.parallel.sharded_dynamics import SPMD_HALO_DEPTH

    sc = mod.schedule_cost_row(_mesh(), "geometric", 2)
    assert sc["schedule_halo_depth"] == SPMD_HALO_DEPTH


def test_schedule_cost_matches_the_production_scorer_exactly():
    """Lock: the wrapper reports what the production scorer returns — it is
    a passthrough, not a re-derivation (the whole point: a 1-ring lookalike
    reports 8 rounds where the real depth-3 graph reports 12-14)."""
    from legoesm.parallel.sharded_dynamics import spmd_schedule_cost

    mesh = _mesh()
    ref = spmd_schedule_cost(mesh, 2, method="sfc")
    sc = mod.schedule_cost_row(mesh, "sfc", 2)
    for key in ("n_rounds", "max_degree", "n_rounds_greedy",
                "coloring_method", "resolved_method", "cells_per_device",
                "production_strategy"):
        assert sc[key] == ref[key], key


def test_schedule_cost_flag_is_opt_in_and_lands_in_the_json(
        tmp_path, monkeypatch):
    """Off by default (it is the expensive layer); on, every scored row
    carries the schedule block and the run records that it ran."""
    out = tmp_path / "off.json"
    monkeypatch.setattr(sys, "argv", [
        "bench", "--subdivision", "2", "--rank-counts", "2",
        "--methods", "geometric", "--out", str(out)])
    assert mod.main() == 0
    payload = json.loads(out.read_text())
    assert "schedule" not in payload["rows"][0]
    assert payload["metadata"]["extra"]["schedule_cost"] is False

    out2 = tmp_path / "on.json"
    monkeypatch.setattr(sys, "argv", [
        "bench", "--subdivision", "2", "--rank-counts", "2",
        "--methods", "geometric", "--schedule-cost", "--out", str(out2)])
    assert mod.main() == 0
    payload2 = json.loads(out2.read_text())
    row = payload2["rows"][0]
    assert row["schedule"]["n_rounds"] >= row["schedule"]["max_degree"] >= 1
    assert payload2["metadata"]["extra"]["schedule_cost"] is True
    # Scorer provenance per row: a copied row must show it scored a mesh
    # partitioned for THIS device count, not one reordered for another.
    assert row["schedule"]["reorder_target"] == row["n_ranks"]
    assert row["schedule"]["already_reordered"] is False


def test_lloyd_flag_reaches_the_mesh_builder(tmp_path, monkeypatch):
    """``--lloyd`` must actually select the mesh, not just be recorded.

    Non-vacuity: asserting only the recorded default (50) passes even if the
    flag is never wired into ``create_voronoi_mesh`` — codex caught exactly
    that.  This spies on the builder, so dropping ``lloyd_iterations=
    args.lloyd`` fails here, and it checks a NON-default value so the
    assertion cannot be satisfied by the default.
    """
    import legoesm.grids.voronoi as vor

    seen = {}
    real = vor.create_voronoi_mesh

    def spy(*a, **k):
        seen.update(k)
        # lloyd=0 is cheap and is what the scaling meshes actually use.
        return real(*a, **k)

    monkeypatch.setattr(vor, "create_voronoi_mesh", spy)
    out = tmp_path / "lloyd0.json"
    monkeypatch.setattr(sys, "argv", [
        "bench", "--subdivision", "2", "--rank-counts", "2",
        "--methods", "geometric", "--lloyd", "0", "--out", str(out)])
    assert mod.main() == 0
    assert seen.get("lloyd_iterations") == 0, seen
    # And it is recorded, so a synthetic mesh cannot be read back as a
    # production SCVT receipt.
    payload = json.loads(out.read_text())
    assert payload["metadata"]["extra"]["lloyd_iterations"] == 0


def test_expect_rounds_gate_fails_loudly_and_never_vacuously(
        tmp_path, monkeypatch):
    """The instrument check must FAIL on a wrong expectation and on a pair
    that was never scored — a gate that can only pass is not a gate."""
    base = ["bench", "--subdivision", "2", "--rank-counts", "2",
            "--methods", "geometric", "--schedule-cost"]

    # Truth first: read what this configuration really scores.
    out = tmp_path / "truth.json"
    monkeypatch.setattr(sys, "argv", base + ["--out", str(out)])
    assert mod.main() == 0
    truth = json.loads(out.read_text())["rows"][0]["schedule"]["n_rounds"]

    # Matching expectation -> pass, and the check is recorded.
    ok = tmp_path / "ok.json"
    monkeypatch.setattr(sys, "argv", base + [
        "--expect-rounds", f"geometric:2={truth}", "--out", str(ok)])
    assert mod.main() == 0
    assert json.loads(ok.read_text())["expected_rounds_check"]["passed"] is True

    # Wrong expectation -> non-zero exit.
    bad = tmp_path / "bad.json"
    monkeypatch.setattr(sys, "argv", base + [
        "--expect-rounds", f"geometric:2={truth + 7}", "--out", str(bad)])
    assert mod.main() == 1
    assert json.loads(bad.read_text())["expected_rounds_check"]["failures"]

    # A pair that was never scored is a FAILURE, not a silent skip —
    # otherwise a sweep missing a method still reports a clean gate.
    missing = tmp_path / "missing.json"
    monkeypatch.setattr(sys, "argv", base + [
        "--expect-rounds", "sfc:2=3", "--out", str(missing)])
    assert mod.main() == 1
    fails = json.loads(missing.read_text())["expected_rounds_check"]["failures"]
    assert any("NOT SCORED" in f for f in fails), fails


def test_expect_rounds_refuses_to_pass_vacuously_without_scoring(monkeypatch):
    """Without --schedule-cost nothing is scored, so the gate would pass on
    an empty comparison. It must refuse instead."""
    monkeypatch.setattr(sys, "argv", [
        "bench", "--subdivision", "2", "--rank-counts", "2",
        "--methods", "geometric", "--expect-rounds", "geometric:2=1",
        "--out", "/dev/null"])
    with pytest.raises(SystemExit, match="needs --schedule-cost"):
        mod.main()


@pytest.mark.parametrize("spec", ["geometric:2", "geometric=2", "voodoo:2=3"])
def test_expect_rounds_rejects_malformed_specs(spec):
    """A typo'd expectation must raise, never be dropped — a silently
    skipped expectation turns the gate into a no-op."""
    with pytest.raises(ValueError, match="expect-rounds"):
        mod.parse_expect_rounds(spec)


def test_expect_rounds_rejects_duplicate_and_empty_specs():
    """Two more ways the gate could be silently bypassed (codex round 2).

    A duplicate key would let the later expectation overwrite the earlier,
    so a listed-but-wrong expectation is discarded and the gate passes. A
    non-empty spec that parses to nothing (``",,,"``) would make main() skip
    the check entirely while the caller believes it ran.
    """
    with pytest.raises(ValueError, match="duplicate expectation"):
        mod.parse_expect_rounds("geometric:2=999,geometric:2=13")
    with pytest.raises(ValueError, match="NO expectations"):
        mod.parse_expect_rounds(",,,")
    # Whitespace-only is a value the caller PASSED; reading it as "no gate"
    # is the same bypass (codex round 3 — the first fix guarded on
    # spec.strip() and let this through).
    for blank in ("   ", "\t", " , , "):
        with pytest.raises(ValueError, match="NO expectations"):
            mod.parse_expect_rounds(blank)
    # A genuinely empty spec is the "no gate requested" default, not an error.
    assert mod.parse_expect_rounds("") == {}


@pytest.mark.parametrize("flag, value", [
    ("--lloyd", "-1"), ("--subdivision", "-1"), ("--halo-depth", "-1")])
def test_negative_numeric_args_refused_as_false_provenance(
        monkeypatch, flag, value):
    """All three behave exactly like 0 in the underlying code but would be
    RECORDED under the negative value — a level-0 mesh filed as 'L-1'."""
    argv = ["bench", "--subdivision", "2", "--rank-counts", "2",
            "--methods", "geometric", "--out", "/dev/null"]
    # Replace the flag if already present, else append.
    if flag in argv:
        argv[argv.index(flag) + 1] = value
    else:
        argv += [flag, value]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit, match=r"must be >= 0"):
        mod.main()


def test_expect_rounds_fails_when_a_method_is_unavailable(
        tmp_path, monkeypatch):
    """An expectation naming a method that reported UNAVAILABLE must fail.

    Distinct from omitting the method from --methods: here the sweep asks
    for it and the partitioner is missing, which is exactly how a two-method
    table gets misread as a three-method one.
    """
    monkeypatch.setattr(mod, "method_available",
                        lambda m: m != "metis")
    out = tmp_path / "unavail.json"
    monkeypatch.setattr(sys, "argv", [
        "bench", "--subdivision", "2", "--rank-counts", "2",
        "--methods", "geometric,metis", "--schedule-cost",
        "--expect-rounds", "metis:2=1", "--out", str(out)])
    assert mod.main() == 1
    payload = json.loads(out.read_text())
    fails = payload["expected_rounds_check"]["failures"]
    assert any("NOT SCORED" in f for f in fails), fails
    # Assert the EXPLICIT unavailable row, not just the failure: without
    # this the test would also pass if metis were silently omitted, which
    # is the very substitution this bench refuses to make.
    metis_rows = [r for r in payload["rows"] if r["method"] == "metis"]
    assert metis_rows and metis_rows[0]["available"] is False, payload["rows"]


def test_lloyd_rejects_negative(monkeypatch):
    """A negative Lloyd count behaves like 0 in the builder (it relaxes only
    for > 0) but is recorded and cached under its own key — false
    provenance, so the CLI must refuse it rather than run."""
    monkeypatch.setattr(sys, "argv", [
        "bench", "--subdivision", "2", "--rank-counts", "2",
        "--methods", "geometric", "--lloyd", "-1", "--out", "/dev/null"])
    with pytest.raises(SystemExit, match=r"--lloyd must be >= 0"):
        mod.main()


def test_empty_methods_refused_instead_of_measuring_nothing(monkeypatch):
    """``--methods " , , "`` used to select nothing, write ``rows: []`` and
    exit 0 — an empty run that reads as a successful one (codex round 4)."""
    monkeypatch.setattr(sys, "argv", [
        "bench", "--subdivision", "2", "--rank-counts", "2",
        "--methods", " , , ", "--out", "/dev/null"])
    with pytest.raises(SystemExit, match="selects NO methods"):
        mod.main()
