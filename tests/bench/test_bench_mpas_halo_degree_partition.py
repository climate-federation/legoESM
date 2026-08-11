"""Unit test for the degree-aware-partition probe
(scripts/bench/bench_mpas_halo_degree_partition.py).

The probe is an INSTRUMENT: it decides whether a production partitioner gets
built, so its arithmetic is checked here on a mesh small enough to count by
hand — a 6-cell chain split into three 2-cell ranks, where the depth-1 contact
graph is 0-1-2 and the degrees are therefore exactly [1, 2, 1].
"""
from __future__ import annotations

import importlib.util
import os

import numpy as np
import pytest

_PROBE = os.path.join(
    os.path.dirname(__file__), "..", "..", "scripts", "bench",
    "bench_mpas_halo_degree_partition.py")


def _load():
    spec = importlib.util.spec_from_file_location("_degree_probe", _PROBE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _ChainMesh:
    """Cells 0..n-1 in a line; ``cellsOnCell`` is (maxEdges, nCells)."""

    def __init__(self, n):
        self.nCells = n
        self.maxEdges = 2
        coc = np.full((2, n), -1, dtype=np.int32)
        coc[0, 1:] = np.arange(n - 1)      # left neighbour
        coc[1, :-1] = np.arange(1, n)      # right neighbour
        self.cellsOnCell = coc


def test_degree_vector_counts_distinct_partners():
    mod = _load()
    deg = mod.degree_vector({(0, 1), (1, 2), (1, 3)}, 4)
    assert deg.tolist() == [1, 3, 1, 1]


def test_padded_bytes_is_round_max_times_active_pairs():
    mod = _load()
    # Round 0: two active pairs, round max 10 cells / 4 edges.
    # Round 1: one active pair, 3 cells / 0 edges.  Device 3 self-maps.
    sched = {
        "n_rounds": 2,
        "ppermute_perms": [[(0, 1), (1, 0), (2, 3), (3, 2)], [(0, 2), (2, 0),
                                                              (3, 3)]],
        "halo_cells_per_round": [10, 3],
        "halo_edges_per_round": [4, 0],
    }
    expect = (4 * (10 * mod.CELL_BYTES + 4 * mod.EDGE_BYTES)
              + 2 * (3 * mod.CELL_BYTES + 0 * mod.EDGE_BYTES))
    assert mod.padded_bytes(sched) == expect


def test_chain_contact_graph_and_degrees_by_hand():
    mod = _load()
    mesh = _ChainMesh(6)
    adj = mod.cell_adjacency(mesh)
    assert adj.shape == (6, 6)
    assert adj[0].toarray().ravel().tolist() == [0, 1, 0, 0, 0, 0]

    owner = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    m = mod.contact_matrix(adj, owner, 3, 1)
    # Rank 0 owns {0,1}; its depth-1 reach adds cell 2 (rank 1) only.
    assert m[0, 1] == 1 and m[0, 2] == 0
    # Rank 1 owns {2,3}; reach adds cells 1 (rank 0) and 4 (rank 2).
    assert m[1, 0] == 1 and m[1, 2] == 1
    assert mod.proxy_degrees(m).tolist() == [1, 2, 1]

    # Depth 2 reaches only cell 3 from rank 0 (still rank 1), so the graph is
    # unchanged; depth 3 finally reaches cell 4 and makes 0 and 2 partners.
    assert mod.proxy_degrees(
        mod.contact_matrix(adj, owner, 3, 2)).tolist() == [1, 2, 1]
    assert mod.proxy_degrees(
        mod.contact_matrix(adj, owner, 3, 3)).tolist() == [2, 2, 2]


def test_incremental_reach_equals_full_recompute():
    """The search only recomputes the reach columns of devices whose owned
    set changed. If that shortcut ever drifts from a full recompute it
    produces a plausible wrong degree, not an error."""
    mod = _load()
    adj = mod.cell_adjacency(_ChainMesh(12))
    owner = np.repeat(np.arange(3), 4).astype(np.int64)
    full0 = mod.reach_matrix(adj, owner, 3, 2)
    cand = owner.copy()
    cand[4] = 0  # one cell moves from device 1 to device 0
    inc = mod.reach_matrix(adj, cand, 3, 2, prev=full0, dirty={0, 1})
    assert np.array_equal(inc, mod.reach_matrix(adj, cand, 3, 2))
    assert np.array_equal(mod.contact_from_reach(cand, inc, 3),
                          mod.contact_matrix(adj, cand, 3, 2))


def test_cell_adjacency_rejects_transposed_connectivity():
    mod = _load()
    mesh = _ChainMesh(6)
    mesh.cellsOnCell = mesh.cellsOnCell.T  # (nCells, maxEdges) — wrong axis
    with pytest.raises(ValueError, match="maxEdges"):
        mod.cell_adjacency(mesh)


def test_rebalance_restores_exactly_equal_counts():
    mod = _load()
    adj = mod.cell_adjacency(_ChainMesh(6))
    owner = np.array([0, 0, 0, 0, 1, 2], dtype=np.int64)  # 4/1/1
    out = mod._rebalance(adj, owner, 3)
    assert out is not None
    assert np.bincount(out, minlength=3).tolist() == [2, 2, 2]


def test_rebalance_handles_indivisible_cell_counts():
    """subdiv-9 has 2 621 442 cells and is NOT divisible by 64 or 128;
    demanding exact equality would reject every candidate at the very
    configuration this probe exists to measure."""
    mod = _load()
    assert mod.balance_bounds(2621442, 128) == (20480, 20481)
    assert mod.balance_bounds(2621442, 64) == (40960, 40961)
    adj = mod.cell_adjacency(_ChainMesh(7))
    owner = np.array([0, 0, 0, 0, 0, 1, 2], dtype=np.int64)  # 5/1/1
    out = mod._rebalance(adj, owner, 3)
    assert out is not None
    assert sorted(np.bincount(out, minlength=3).tolist()) == [2, 2, 3]


def test_rebalance_does_not_livelock_between_two_full_neighbours():
    """A local 'give to the emptiest neighbour' rule traded one boundary cell
    back and forth between two devices at the cap while a distant device
    stayed starved, and the bounded sweep count turned that into a false
    'cannot balance'. The transfer must route along a device path."""
    mod = _load()
    adj = mod.cell_adjacency(_ChainMesh(11))
    owner = np.array([0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3], dtype=np.int64)
    out = mod._rebalance(adj, owner, 4)
    assert out is not None
    lo, hi = mod.balance_bounds(11, 4)
    counts = np.bincount(out, minlength=4)
    assert counts.min() >= lo and counts.max() <= hi


def test_block_mismatch_measures_the_equal_block_recut():
    mod = _load()
    # 3 devices owning 2/2/2 of a 6-cell mesh: blocks of 2 line up exactly.
    st = mod.owner_stats(np.array([0, 0, 1, 1, 2, 2]), 3, 6)
    assert st["block_mismatch"] == 0.0
    # 7 cells padded to 9 -> blocks of 3, but domains are 2/2/3: the block
    # boundaries drift off the domain boundaries and cells change hands.
    st = mod.owner_stats(np.array([0, 0, 1, 1, 2, 2, 2]), 3, 9)
    assert st["block_mismatch"] > 0.0
    assert mod.padded_cells(2621442, 128) == 2621568


def test_domain_components_detects_a_shattered_domain():
    mod = _load()
    adj = mod.cell_adjacency(_ChainMesh(6))
    assert mod.domain_components(adj, np.array([0, 0, 1, 1, 2, 2]), 3) == 3
    # Rank 0 owns cells 0 and 5 — two islands, so 4 components for 3 devices.
    assert mod.domain_components(adj, np.array([0, 1, 1, 2, 2, 0]), 3) == 4


def test_proxy_pair_set_matches_the_hand_counted_chain_graph():
    mod = _load()
    adj = mod.cell_adjacency(_ChainMesh(6))
    owner = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    assert mod.proxy_pair_set(mod.contact_matrix(adj, owner, 3, 1)) == {
        (0, 1), (1, 2)}


def test_greedy_pass_preserves_balance_and_never_worsens_max_degree():
    mod = _load()
    mesh = _ChainMesh(24)
    adj = mod.cell_adjacency(mesh)
    owner = np.repeat(np.arange(4), 6).astype(np.int64)
    before = int(mod.proxy_degrees(mod.contact_matrix(adj, owner, 4, 1)).max())
    out, hist = mod.greedy_degree_pass(adj, owner, 4, 1, max_iters=3,
                                       verbose=False)
    assert np.bincount(out, minlength=4).tolist() == [6, 6, 6, 6]
    assert hist[0] == before
    assert hist[-1] <= before


_GATE_KW = {"n_dev_owner": 4, "n_cells": 8}   # balanced counts are 2..2


def _cand(n_rounds, padded=100, counts=(2, 2), comps=4, missed=0):
    return {"n_rounds": n_rounds, "padded_bytes": padded,
            "owner_counts": {"min": counts[0], "max": counts[1],
                             "imbalance": counts[1] / 2.0,
                             "block_mismatch": 0.0},
            "n_components": comps, "n_components_scored": comps,
            "proxy_fidelity": {"n_missed": missed, "n_prod_pairs": 10}}


def test_verdict_refutes_when_prototype_finds_nothing(capsys):
    mod = _load()
    mod.verdict({"sfc": {"n_rounds": 12, "max_degree": 11, "padded_bytes": 100,
                         "degree_floor_rounds": 11, "degree_mean": 10.6,
                         "n_components_scored": 4,
                         "owner_counts": {"block_mismatch": 0.0}},
                 "greedy": {"status": "no improving move found",
                            "proxy_fidelity": {"n_missed": 0,
                                               "n_prod_pairs": 240}}},
                **_GATE_KW)
    out = capsys.readouterr().out
    assert "REFUTED" in out
    # The refutation must be scoped to the prototype, not to all partitioners.
    assert "not every conceivable partitioner" in out


def test_verdict_gate_thresholds(capsys):
    mod = _load()
    base = {"n_rounds": 12, "padded_bytes": 100, "max_degree": 8,
            "degree_floor_rounds": 8, "degree_mean": 7.1,
            "n_components_scored": 4,
            "owner_counts": {"block_mismatch": 0.0}}
    mod.verdict({"sfc": base, "greedy": _cand(9, padded=105)}, **_GATE_KW)
    assert "CONFIRMED" in capsys.readouterr().out
    # 12 -> 10 is 16.7%: above the 10% refute floor, below the 25% bar.
    mod.verdict({"sfc": base, "greedy": _cand(10)}, **_GATE_KW)
    assert "INTERMEDIATE" in capsys.readouterr().out
    # 12 -> 11 is only 8.3%: below the floor, refuted.
    mod.verdict({"sfc": base, "greedy": _cand(11)}, **_GATE_KW)
    assert "REFUTED" in capsys.readouterr().out
    # A win bought by breaking balance is refuted, however big it is.
    mod.verdict({"sfc": base, "greedy": _cand(6, counts=(1, 3))}, **_GATE_KW)
    assert "balance/connectivity" in capsys.readouterr().out
    # ...as is one bought by shattering the domains.
    mod.verdict({"sfc": base, "greedy": _cand(6, comps=9)}, **_GATE_KW)
    assert "balance/connectivity" in capsys.readouterr().out
    # ...as is one that inflates padded bytes past 10%, per the docstring.
    mod.verdict({"sfc": base, "greedy": _cand(6, padded=120)}, **_GATE_KW)
    assert "padded bytes inflated" in capsys.readouterr().out


def test_verdict_refuses_a_partial_candidate_row():
    mod = _load()
    good_base = {"n_rounds": 12, "padded_bytes": 100, "n_components_scored": 4,
                 "owner_counts": {"block_mismatch": 0.0}}
    # A partial BASELINE cannot produce a verdict either: every gate number
    # is a ratio against it.
    with pytest.raises(ValueError, match="baseline row missing"):
        mod.verdict({"sfc": {"n_rounds": 12, "padded_bytes": 100},
                     "greedy": _cand(8)}, **_GATE_KW)
    with pytest.raises(ValueError, match="missing"):
        mod.verdict({"sfc": good_base, "greedy": {"n_rounds": 8}}, **_GATE_KW)
    # An unmeasured connectivity must NOT be assumed connected: a row with
    # everything but n_components used to be able to reach CONFIRMED.
    row = _cand(8)
    del row["n_components_scored"]
    with pytest.raises(ValueError, match="n_components_scored"):
        mod.verdict({"sfc": good_base, "greedy": row}, **_GATE_KW)


def test_verdict_gates_on_the_scored_blocks_not_the_intended_ownership():
    """The equal-block re-cut can disconnect a domain the partitioner meant to
    be contiguous, so the gate reads n_components_scored."""
    mod = _load()
    base = {"n_rounds": 12, "padded_bytes": 100, "n_components_scored": 4,
            "owner_counts": {"block_mismatch": 0.0}}
    row = _cand(6)                       # a 50% round cut
    row["n_components_scored"] = 9       # ...on a shattered scored partition
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        mod.verdict({"sfc": base, "greedy": row}, **_GATE_KW)
    assert "balance/connectivity" in buf.getvalue()


def test_verdict_discloses_proxy_blind_spots_on_a_scored_refutation(capsys):
    mod = _load()
    base = {"n_rounds": 12, "padded_bytes": 100, "max_degree": 8,
            "degree_floor_rounds": 8, "degree_mean": 7.1,
            "n_components_scored": 4,
            "owner_counts": {"block_mismatch": 0.0}}
    mod.verdict({"sfc": base, "greedy": _cand(12, missed=9)}, **_GATE_KW)
    out = capsys.readouterr().out
    assert "REFUTED" in out and "GATE CAVEAT" in out and "missed 9" in out
