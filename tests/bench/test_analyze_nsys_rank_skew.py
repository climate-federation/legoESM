"""Direct tests for the multi-rank halo-skew analyzer.

The load-bearing logic is the PAIRING: which collective on rank A is the same
collective as which on rank B.  Get that wrong and every duration comparison
is between unrelated kernels — which is exactly how the previous probe in this
family produced a confident wrong number.  So these tests hammer the pairing
and its refusals, not the arithmetic.
"""
from __future__ import annotations

import importlib.util
import sqlite3
import sys
from pathlib import Path

import pytest

_SCRIPT = (Path(__file__).resolve().parents[2]
           / "scripts" / "bench" / "analyze_nsys_rank_skew.py")
_spec = importlib.util.spec_from_file_location("nsys_rank_skew", _SCRIPT)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def _profile(*entries):
    """entries: (round, partner) -> the round_profile shape."""
    return [{"round": r, "partner": p, "halo_cells": 10, "halo_edges": 20}
            for r, p in entries]


# --- shared_round: the pairing and its refusals ---------------------------


def test_shared_round_finds_the_mirrored_entry_and_both_indices():
    # A issues rounds [0(->1), 3(->2)]; B=1 issues [0(->0), 5(->2)].
    a = _profile((0, 1), (3, 2))
    b = _profile((0, 0), (5, 2))
    assert mod.shared_round(a, b, 0, 1) == (0, 0, 0)

    # Same, but the shared round sits at DIFFERENT issue positions on each
    # rank — the case a naive "k-th call on both" pairing gets wrong.
    a2 = _profile((7, 4), (9, 1))
    b2 = _profile((2, 8), (5, 6), (9, 0))
    assert mod.shared_round(a2, b2, 0, 1) == (9, 1, 2)


def test_shared_round_returns_none_for_non_partners():
    """The control. An overlap-based matcher would pair these anyway."""
    a = _profile((0, 1), (3, 2))
    b = _profile((4, 5), (6, 7))
    assert mod.shared_round(a, b, 0, 3) is None


def test_shared_round_refuses_a_one_sided_schedule():
    """A names B, B does not name A: the schedule was misread."""
    a = _profile((0, 1))
    b = _profile((0, 9))          # B thinks it talks to 9 at round 0
    with pytest.raises(SystemExit, match="asymmetric schedule"):
        mod.shared_round(a, b, 0, 1)


def test_shared_round_refuses_an_ambiguous_pairing():
    """Two rounds with the same partner: which collective is which?"""
    a = _profile((0, 1), (4, 1))
    b = _profile((0, 0), (4, 0))
    with pytest.raises(SystemExit, match="not unique"):
        mod.shared_round(a, b, 0, 1)


# --- compare: indexing and the consistency refusals -----------------------


def _k(*start_duration_pairs):
    """Kernels as (start_ns, duration_ns) -> the (start, end) rows."""
    return [(start, start + dur) for start, dur in start_duration_pairs]


def test_compare_indexes_the_same_collective_on_both_ranks():
    """Ranks have DIFFERENT rounds-per-fill, so the stride differs.

    A: 2 rounds/fill, shared round at index 1.
    B: 3 rounds/fill, shared round at index 0.
    Fill k therefore sits at A[2k+1] and B[3k+0]; pairing by raw call index
    would compare A[1] with B[1], a different collective entirely.
    """
    # 3 fills. The collectives of interest get distinctive durations so a
    # mis-stride shows up as the WRONG duration, not merely a wrong index.
    a = _k((0, 1000), (10_000, 5000),       # fill 0: idx1 -> 5000 ns
           (20_000, 1000), (30_000, 6000),   # fill 1: idx1 -> 6000 ns
           (40_000, 1000), (50_000, 7000))   # fill 2: idx1 -> 7000 ns
    b = _k((5_000, 2000), (6_000, 100), (7_000, 100),
           (25_000, 2500), (26_000, 100), (27_000, 100),
           (45_000, 3000), (46_000, 100), (47_000, 100))
    recs = mod.compare(a, b, 2, 3, 1, 0, drop_fills=0)
    assert [r["fill"] for r in recs] == [0, 1, 2]
    assert [r["a_dur_us"] for r in recs] == [5.0, 6.0, 7.0]
    assert [r["b_dur_us"] for r in recs] == [2.0, 2.5, 3.0]
    # wait estimate = |difference| between the two ranks' time in the SAME
    # collective.
    assert [r["wait_estimate_us"] for r in recs] == [3.0, 3.5, 4.0]


def test_compare_drops_warmup_fills():
    a = _k((0, 1000), (10_000, 1000), (20_000, 1000), (30_000, 1000))
    b = _k((0, 1000), (10_000, 1000), (20_000, 1000), (30_000, 1000))
    recs = mod.compare(a, b, 1, 1, 0, 0, drop_fills=2)
    assert [r["fill"] for r in recs] == [2, 3]


def test_compare_refuses_partial_fills():
    """A call count that is not a whole number of fills means the trace and
    the schedule disagree; pairing would silently shear."""
    a = _k((0, 1), (10, 1), (20, 1))       # 3 calls, 2 rounds/fill
    b = _k((0, 1), (10, 1), (20, 1), (30, 1))
    with pytest.raises(SystemExit, match="not whole fills"):
        mod.compare(a, b, 2, 2, 0, 0, drop_fills=0)


def test_compare_refuses_unequal_fill_counts():
    """Ranks step in lockstep; different fill counts means different runs."""
    a = _k((0, 1), (10, 1), (20, 1), (30, 1))   # 2 rounds/fill -> 2 fills
    b = _k((0, 1), (10, 1), (20, 1))            # 1 round/fill  -> 3 fills
    with pytest.raises(SystemExit, match="lockstep|halo fills"):
        mod.compare(a, b, 2, 1, 0, 0, drop_fills=0)


# --- trace loading --------------------------------------------------------


def _make_sqlite(path, names_and_rows):
    con = sqlite3.connect(str(path))
    con.execute("CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT)")
    con.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL "
                "(start INTEGER, end INTEGER, demangledName INTEGER)")
    for sid, (name, rows) in enumerate(names_and_rows):
        con.execute("INSERT INTO StringIds VALUES (?,?)", (sid, name))
        for s, e in rows:
            con.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?)",
                        (s, e, sid))
    con.commit()
    con.close()


def test_load_kernels_reads_only_the_collective_and_keeps_order(tmp_path):
    db = tmp_path / "rank_0.sqlite"
    _make_sqlite(db, [
        ("ncclDevKernel_SendRecv(x)", [(100, 300), (500, 900)]),
        ("loop_multiply_fusion", [(0, 50)]),          # must be ignored
        ("ncclDevKernel_AllGather_RING_LL(y)", [(0, 99999)]),  # ignored
    ])
    assert mod.load_kernels(db) == [(100, 300), (500, 900)]


def test_load_kernels_raises_when_the_name_matches_nothing(tmp_path):
    """A renamed NCCL kernel would otherwise yield zero rows everywhere and
    read as 'no skew' rather than 'no data'."""
    db = tmp_path / "rank_0.sqlite"
    _make_sqlite(db, [("some_other_kernel", [(0, 10)])])
    with pytest.raises(SystemExit, match="no kernel matches"):
        mod.load_kernels(db)


# --- the cross-node refusal ----------------------------------------------


def test_rank_node_map_parsed_and_cross_node_pair_refused(tmp_path,
                                                          monkeypatch):
    (tmp_path / "_rank_nodes.tsv").write_text(
        "rank=0\tnode=l50018\tpid=1\n"
        "rank=1\tnode=l50018\tpid=2\n"
        "rank=9\tnode=l50033\tpid=3\n")
    nodes = mod.read_rank_nodes(tmp_path / "_rank_nodes.tsv")
    assert nodes == {0: "l50018", 1: "l50018", 9: "l50033"}

    # 0 and 9 are on different machines: their CLOCK_MONOTONIC values share
    # no origin, so a "skew" computed from them would be clock drift.
    monkeypatch.setattr(sys, "argv", [
        "skew", "--sqlite-dir", str(tmp_path), "--pairs", "0:9",
        "--subdivision", "2", "--n-devices", "8"])
    with pytest.raises(SystemExit, match="spans nodes"):
        mod.main()


def test_missing_rank_node_map_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "skew", "--sqlite-dir", str(tmp_path), "--pairs", "0:1",
        "--subdivision", "2", "--n-devices", "8"])
    with pytest.raises(SystemExit, match="_rank_nodes.tsv"):
        mod.main()
