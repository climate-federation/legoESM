"""Size-aware halo colouring (LEGOESM_MPAS_SIZE_COLORING, default ON).

Landed on the s9@64 production A/B/A2 receipt (job 26857404, drift
1.7%): ratio 0.803 (10.2 -> 8.11 ms/step) from regrouping halo pairs by
payload size at the SAME round count — the lane is node-NIC-bound on
the PADDED wire bytes (every pair in a round ships the round maximum),
and the legacy colouring inflated padded/actual to 2.926x (job
26856854; size-aware brings it to 2.162).
"""

import os

import pytest


def _schedules(monkeypatch, subdivision=5, n_dev=8):
    from legoesm.grids.voronoi import create_voronoi_mesh
    from legoesm.parallel.voronoi_partition import (
        reorder_voronoi_for_sharding,
    )
    from legoesm.parallel.sharded_dynamics import (
        SPMD_HALO_DEPTH, _build_voronoi_partition_infra,
        _build_ppermute_schedule,
    )
    mesh = create_voronoi_mesh(subdivision_level=subdivision)
    mesh = reorder_voronoi_for_sharding(mesh, n_dev)
    cp, ep = mesh.nCells // n_dev, mesh.nEdges // n_dev
    (_, _, _, _, _, mlc, mle, parts, own) = _build_voronoi_partition_infra(
        mesh, n_dev, halo_depth=SPMD_HALO_DEPTH)
    out = {}
    for env in ("0", "1"):
        monkeypatch.setenv("LEGOESM_MPAS_SIZE_COLORING", env)
        out[env] = _build_ppermute_schedule(
            parts, own, n_dev, cp, ep, mlc, mle)
    return out


def _round_pairs(sched):
    """Set of (round-invariant) undirected transfer pairs, and the full
    transfer multiset independent of round grouping."""
    pairs = set()
    for perm in sched["ppermute_perms"]:
        for (a, b) in perm:
            if a != b:
                pairs.add((min(a, b), max(a, b)))
    return pairs


def _padded(sched):
    tot = 0
    for r in range(sched["n_rounds"]):
        n_pairs = sum(1 for (a, b) in sched["ppermute_perms"][r] if a != b)
        tot += n_pairs * (sched["halo_cells_per_round"][r]
                          + sched["halo_edges_per_round"][r])
    return tot


class TestSizeAwareColoring:

    def test_same_rounds_same_transfers_less_padding(self, monkeypatch):
        s = _schedules(monkeypatch)
        off, on = s["0"], s["1"]
        assert on["coloring_method"] == "size_aware", (
            "size-aware colouring did not engage on a graph where it is "
            "known to win (subdiv5/nd8: -31% padded weight)")
        assert on["n_rounds"] == off["n_rounds"], (
            "round count regressed — the adoption guard is broken")
        assert _round_pairs(on) == _round_pairs(off), (
            "the transfer PAIR SET changed — colouring must only regroup "
            "existing exchanges, never add or drop one")
        assert _padded(on) < _padded(off), (
            "padded weight did not improve although size_aware was adopted")

    def test_deterministic_across_rebuilds(self, monkeypatch):
        """Two independent builds produce the IDENTICAL colouring —
        every multicontroller process derives the schedule on its own,
        so any order nondeterminism is a cross-rank deadlock (codex
        review of 9ff0d5892)."""
        a = _schedules(monkeypatch)["1"]
        b = _schedules(monkeypatch)["1"]
        assert a["ppermute_perms"] == b["ppermute_perms"]
        assert a["halo_cells_per_round"] == b["halo_cells_per_round"]
        assert a["halo_edges_per_round"] == b["halo_edges_per_round"]

    def test_bidirectional_entries_survive(self, monkeypatch):
        """Every undirected pair contributes exactly TWO directed
        entries in its round's perm — the pair-set test alone would not
        catch one dropped direction (codex review)."""
        on = _schedules(monkeypatch)["1"]
        for perm in on["ppermute_perms"]:
            directed = [(a, b) for (a, b) in perm if a != b]
            assert len(directed) % 2 == 0
            as_set = set(directed)
            for (a, b) in directed:
                assert (b, a) in as_set, (
                    f"direction ({b},{a}) missing for pair ({a},{b})")

    def test_default_is_on_and_escape_works(self, monkeypatch):
        from legoesm.parallel.sharded_dynamics import _resolve_size_coloring
        assert _resolve_size_coloring("") is True
        assert _resolve_size_coloring("1") is True
        assert _resolve_size_coloring("0") is False
        with pytest.raises(ValueError, match="LEGOESM_MPAS_SIZE_COLORING"):
            _resolve_size_coloring("auto")
