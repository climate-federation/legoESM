"""Edge-coloring correctness + round-count for the route-B MPAS ppermute
schedule (:func:`legoesm.parallel.sharded_dynamics._build_ppermute_schedule`).

Each color is one bidirectional ppermute ROUND and the route-B lane is
round-latency-bound (#1113), so the schedule builder picks the coloring
with the fewest rounds across several deterministic first-fit orderings
(``multi_ordering_edge_coloring``). These tests pin:

- every candidate coloring is PROPER (no device sends/receives twice in a
  round) — a wrong coloring would collide two exchanges at a device;
- the multi-start coloring never uses MORE rounds than the legacy sorted
  greedy (never-regress), and reaches the chromatic-index floor
  (``max_degree``) on the graphs where sorted greedy overshoots;
- determinism across repeated builds (rank-agreement proxy).
"""

from __future__ import annotations

import random

import pytest
from legoesm.parallel.sharded_dynamics import (
    _check_proper_edge_coloring,
    greedy_edge_coloring,
    multi_ordering_edge_coloring,
)


def _random_graph(rng, n, p):
    return {
        (i, j)
        for i in range(n)
        for j in range(i + 1, n)
        if rng.random() < p
    }


@pytest.mark.parametrize("seed", range(40))
def test_multi_ordering_proper_and_no_regression(seed):
    rng = random.Random(seed)
    n = rng.randint(2, 18)
    p = rng.choice([0.2, 0.4, 0.6, 0.85, 1.0])
    pairs = _random_graph(rng, n, p)
    if not pairs:
        pytest.skip("empty graph")
    colors, max_degree = multi_ordering_edge_coloring(pairs)
    # proper
    assert _check_proper_edge_coloring(colors, pairs)
    rounds = max(colors.values()) + 1
    # chromatic-index lower bound
    assert rounds >= max_degree
    # never worse than the legacy sorted greedy
    greedy = greedy_edge_coloring(pairs)
    assert rounds <= max(greedy.values()) + 1


def test_complete_graph_proper_and_bounded():
    """K_n is the adversarial case for first-fit (chromatic index n-1 for
    even n, n for odd n). Multi-start greedy carries NO Vizing guarantee
    (that needs Misra-Gries) — only that it is PROPER, at least
    max_degree, and within the first-fit bound 2*max_degree-1. Complete
    comm graphs never arise from spatial partitioning; this just pins the
    invariants at high edge density."""
    for n in range(2, 12):
        pairs = {(i, j) for i in range(n) for j in range(i + 1, n)}
        colors, max_degree = multi_ordering_edge_coloring(pairs)
        assert _check_proper_edge_coloring(colors, pairs)
        rounds = max(colors.values()) + 1
        assert max_degree <= rounds <= 2 * max_degree - 1


def test_deterministic_across_builds():
    """Same graph → byte-identical coloring every call (every MPI rank must
    build the same schedule)."""
    rng = random.Random(123)
    pairs = _random_graph(rng, 16, 0.5)
    a, _ = multi_ordering_edge_coloring(pairs)
    b, _ = multi_ordering_edge_coloring(pairs)
    assert a == b


def test_star_graph_single_round_floor():
    """A star K_{1,m} has max_degree m and chromatic index m — every edge
    touches the hub, so all m must be different rounds. Both colorings
    agree here; the point is the proper-coloring invariant holds at high
    degree."""
    m = 10
    pairs = {(0, k) for k in range(1, m + 1)}
    colors, max_degree = multi_ordering_edge_coloring(pairs)
    assert max_degree == m
    assert max(colors.values()) + 1 == m
    assert _check_proper_edge_coloring(colors, pairs)


def test_schedule_reaches_floor_on_real_mesh():
    """End-to-end through the schedule builder on a real reordered MPAS
    mesh where sorted greedy overshoots: the produced schedule must reach
    the max_degree floor and stay proper (the round count feeds the
    ppermute loop directly)."""
    pytest.importorskip("jax")

    from legoesm.grids.voronoi import create_voronoi_mesh
    from legoesm.parallel.sharded_dynamics import (
        _build_ppermute_schedule,
        _build_voronoi_partition_infra,
    )
    from legoesm.parallel.voronoi_partition import (
        reorder_voronoi_for_sharding,
    )

    n_dev = 16
    mesh = create_voronoi_mesh(subdivision_level=3)
    mesh = reorder_voronoi_for_sharding(mesh, n_dev, method="auto")
    if mesh.nCells % n_dev or mesh.nEdges % n_dev:
        pytest.skip("mesh not divisible by n_dev")
    cells_per, edges_per = mesh.nCells // n_dev, mesh.nEdges // n_dev
    (_sm, _gc, _ge, _noc, _noe, max_lc, max_le, partitions,
     cell_owner) = _build_voronoi_partition_infra(mesh, n_dev, halo_depth=3)
    sched = _build_ppermute_schedule(
        partitions, cell_owner, n_dev, cells_per, edges_per, max_lc, max_le,
    )
    # sorted greedy overshoots here (17); multi-start reaches the floor (14).
    assert sched["n_rounds"] == sched["max_degree"], (
        f"expected chromatic-index floor {sched['max_degree']}, "
        f"got {sched['n_rounds']}")
    assert sched["n_rounds"] < sched["n_rounds_greedy"], (
        "multi-start coloring should beat sorted greedy on this mesh")
    assert sched["coloring_method"] == "multi_greedy"
    # ppermute perms are proper: no device appears twice as a source or
    # twice as a destination within a single round.
    for perm in sched["ppermute_perms"]:
        srcs = [s for s, _ in perm]
        dsts = [d for _, d in perm]
        assert len(srcs) == len(set(srcs)), "duplicate source in a round"
        assert len(dsts) == len(set(dsts)), "duplicate dest in a round"
