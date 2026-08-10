"""Direct tests for ``sharded_dynamics.spmd_schedule_cost``.

It scores a mesh split by how many sequential halo exchanges it needs. The
number only means something if it comes from the same builders, the same halo
depth, and the same mesh state production uses -- so that is what these pin.
"""
from __future__ import annotations

import pytest

from legoesm.grids.voronoi import create_voronoi_mesh
from legoesm.parallel import sharded_dynamics as sd
from legoesm.parallel.voronoi_partition import reorder_voronoi_for_sharding


@pytest.fixture(scope="module")
def mesh():
    return create_voronoi_mesh(subdivision_level=4)


def test_returns_sane_cost(mesh):
    c = sd.spmd_schedule_cost(mesh, 4)
    assert c["n_dev"] == 4
    assert c["halo_depth"] == 3
    assert c["resolved_method"] == "sfc", "auto must resolve concretely"
    # A round exchanges data between disjoint device PAIRS, so a proper
    # schedule on n_dev devices needs between 1 and n_dev-1 rounds.
    assert 1 <= c["n_rounds"] <= 3, c
    assert c["n_rounds"] <= c["n_rounds_greedy"]
    assert c["max_degree"] >= 1
    assert c["n_rounds"] >= c["max_degree"], (
        "a proper edge colouring can never use FEWER rounds than max_degree")
    assert c["max_local_cells"] > mesh.nCells // 4, "must include the halo"


def test_reports_max_degree_so_optimality_is_measured_not_assumed(mesh):
    """The colouring is a best-of-a-few-orders search, NOT a proof of the
    max_degree lower bound. Callers must be able to check equality rather than
    assume it, so max_degree is returned alongside n_rounds."""
    c = sd.spmd_schedule_cost(mesh, 8)
    assert "max_degree" in c and c["max_degree"] > 0
    assert c["n_rounds"] >= c["max_degree"]


def test_single_device_is_zero_rounds_and_zero_is_refused(mesh):
    assert sd.spmd_schedule_cost(mesh, 1)["n_rounds"] == 0
    with pytest.raises(ValueError, match="integer >= 1"):
        sd.spmd_schedule_cost(mesh, 0)


def test_flags_allgather_when_production_would_not_use_ppermute(mesh):
    """Production auto-selects allgather below a cells/device threshold; the
    ppermute round count is then counterfactual and must say so."""
    c = sd.spmd_schedule_cost(mesh, 8,
                              ppermute_cells_per_device_threshold=10**9)
    assert c["production_strategy"] == "allgather"
    big = sd.spmd_schedule_cost(mesh, 8,
                                ppermute_cells_per_device_threshold=1)
    assert big["production_strategy"] == "ppermute"


def test_already_reordered_mesh_is_not_reordered_again(mesh):
    """Production holds an already-reordered mesh. Re-splitting it would score
    a mesh no run uses, so that path must be expressible and must agree with
    scoring the raw mesh once."""
    prepared = reorder_voronoi_for_sharding(mesh, 8, method="sfc")
    pre = sd.spmd_schedule_cost(prepared, 8, already_reordered=True)
    raw = sd.spmd_schedule_cost(mesh, 8, method="sfc")
    assert pre["resolved_method"] == "pre-reordered"
    assert pre["reorder_target"] is None, (
        "the target that produced a pre-reordered mesh is not recoverable "
        "from it — reporting n_dev would assert something unverified")
    assert pre["n_rounds"] == raw["n_rounds"], (
        "scoring a pre-reordered mesh must match scoring the raw mesh with "
        "the same ownership")
    with pytest.raises(ValueError, match="reorder_target is meaningless"):
        sd.spmd_schedule_cost(prepared, 8, already_reordered=True,
                              reorder_target=16)


def test_reorder_target_differing_from_run_device_count(mesh):
    """The scaling bench reorders once for a target and then runs at a
    different device count. That combination must be expressible, and must
    differ from reordering for the run count."""
    same = sd.spmd_schedule_cost(mesh, 4, method="sfc")
    split_for_16 = sd.spmd_schedule_cost(mesh, 4, method="sfc",
                                         reorder_target=16)
    assert split_for_16["reorder_target"] == 16
    assert split_for_16["n_dev"] == 4
    # Non-vacuity: a split built for 16 devices really is a different split.
    assert split_for_16["n_rounds"] != same["n_rounds"] or (
        split_for_16["max_local_cells"] != same["max_local_cells"]), (
        "reorder_target had no effect — the argument would be decorative")


def test_repeatable(mesh):
    a = sd.spmd_schedule_cost(mesh, 8, method="sfc")
    b = sd.spmd_schedule_cost(mesh, 8, method="sfc")
    assert a == b, "same inputs must give the same score"


@pytest.mark.slow
def test_sfc_beats_metis_and_geometric_on_rounds():
    """The finding this function exists to make measurable. Census at
    production sizes: subdiv-8 sfc 12/14 rounds at 64/128 devices vs metis
    13/19, geometric 16/21. Same ordering here at the SMALLEST size that can
    still tell the methods apart.

    subdiv-4@8 and subdiv-5@8 score all three methods identically (7 rounds) —
    too coarse to discriminate — so this uses subdiv-5@16 and asserts the
    scores actually differ before asserting their order. ~100 s, hence slow.
    """
    pytest.importorskip("pymetis", reason="metis arm needs pymetis")
    big = create_voronoi_mesh(subdivision_level=5)
    r = {m: sd.spmd_schedule_cost(big, 16, method=m)["n_rounds"]
         for m in ("sfc", "metis", "geometric")}
    assert len(set(r.values())) > 1, (
        f"all methods scored identically ({r}) — the comparison is vacuous "
        f"at this mesh size; use a finer mesh or more devices")
    assert r["sfc"] <= r["metis"] and r["sfc"] <= r["geometric"], r


def test_unknown_method_raises(mesh):
    with pytest.raises(ValueError, match="Unknown partitioning method"):
        sd.spmd_schedule_cost(mesh, 4, method="not_a_method")


def test_already_reordered_does_not_reorder(mesh, monkeypatch):
    """Spy, not inference: an accidental second reorder could still produce a
    coincidentally equal round count, so assert the call never happens."""
    from legoesm.parallel import voronoi_partition as vp

    prepared = reorder_voronoi_for_sharding(mesh, 8, method="sfc")
    calls = []
    monkeypatch.setattr(
        vp, "reorder_voronoi_for_sharding",
        lambda *a, **k: calls.append(1) or prepared)
    sd.spmd_schedule_cost(prepared, 8, already_reordered=True)
    assert not calls, "already_reordered=True must not reorder the mesh"


def test_indivisible_reorder_target_is_refused(mesh):
    """A mesh padded for 3 devices is not divisible by 4. The builder would
    assign residual cells to the last owner while excluding them from every
    owned block — a plausible-looking, wrong number."""
    with pytest.raises(ValueError, match="divisible by n_dev"):
        sd.spmd_schedule_cost(mesh, 4, method="sfc", reorder_target=3)


def test_non_integer_device_count_is_refused(mesh):
    with pytest.raises(ValueError, match="integer >= 1"):
        sd.spmd_schedule_cost(mesh, 3.9)


def test_halo_depth_is_one_shared_constant():
    """Production and the scorer must read the SAME depth. Two independently
    hardcoded 3s let production drift without the score noticing."""
    import inspect
    assert sd.SPMD_HALO_DEPTH == 3
    assert (inspect.signature(sd.spmd_schedule_cost)
            .parameters["halo_depth"].default == sd.SPMD_HALO_DEPTH)
    prod = inspect.getsource(sd.make_voronoi_sharded_step)
    assert "halo_depth=SPMD_HALO_DEPTH" in prod, (
        "production stopped consuming the shared constant")


def test_single_device_reports_no_strategy(mesh):
    """Production returns before choosing a halo strategy at one device."""
    assert sd.spmd_schedule_cost(mesh, 1)["production_strategy"] is None
