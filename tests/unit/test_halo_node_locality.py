"""The halo-locality probe must count real traffic and improve a bad labelling.

It exists to decide, offline, whether relabelling devices so heavy talkers
share a node is worth building. Two ways it could quietly lie: counting the
wrong cells, and reporting an improvement its permutation does not actually
deliver. Both are pinned here on a hand-checkable graph.
"""
import importlib.util
import pathlib

import numpy as np
import pytest

_PATH = (pathlib.Path(__file__).resolve().parents[2]
         / "scripts" / "validate" / "halo_node_locality.py")
_spec = importlib.util.spec_from_file_location("_halo_node_locality", _PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


def _chain(n_cells):
    """A 1-D chain of cells, as a (maxEdges=2, nCells) neighbour table."""
    left = np.arange(-1, n_cells - 1)
    right = np.arange(1, n_cells + 1)
    right[-1] = -1
    return np.stack([left, right])


def test_owner_is_contiguous_and_rejects_a_ragged_mesh():
    owner = _mod.owner_of_each_cell(12, 4)
    assert owner.tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
    with pytest.raises(ValueError, match="not divisible"):
        _mod.owner_of_each_cell(13, 4)


def test_traffic_matrix_on_a_chain_is_exactly_the_halo_depth():
    """Each interior device sees `halo_depth` cells from each neighbour."""
    n_cells, n_dev, depth = 40, 4, 2
    W = _mod.halo_traffic_matrix(_chain(n_cells),
                                 _mod.owner_of_each_cell(n_cells, n_dev),
                                 n_dev, depth)
    # Device 1 owns cells 10..19: it needs 8,9 from device 0 and 20,21 from 2.
    assert W[1, 0] == depth and W[1, 2] == depth
    assert W[1, 3] == 0
    # The ends of the chain have one neighbour only.
    assert W[0, 1] == depth and W[0].sum() == depth
    assert np.all(np.diag(W) == 0)


def test_a_device_never_appears_in_its_own_halo():
    """The guard that would catch a ring walk including owned cells."""
    n_cells, n_dev = 24, 3
    W = _mod.halo_traffic_matrix(_chain(n_cells),
                                 _mod.owner_of_each_cell(n_cells, n_dev),
                                 n_dev, 2)
    assert np.all(np.diag(W) == 0)


def test_relabelling_rescues_a_deliberately_bad_placement():
    """Devices 0,2,4,6 talk only to each other; 1,3,5,7 likewise.

    Under the natural labelling each group of four is split two-and-two across
    the two nodes, so only a third of its traffic stays on a node. A relabelling
    worth building has to find the two groups and take that to all of it.
    """
    n, gpn = 8, 4
    W = np.zeros((n, n), dtype=np.int64)
    for group in ([0, 2, 4, 6], [1, 3, 5, 7]):
        for a in group:
            for b in group:
                if a != b:
                    W[a, b] = 100
    natural = np.arange(n)
    # 12 ordered pairs per group, 4 of them within a node: exactly one third.
    assert _mod.on_node_share(W, natural, gpn) == pytest.approx(1.0 / 3.0)
    place = _mod.group_devices_by_traffic(W, gpn)
    assert sorted(place.tolist()) == list(range(n))
    assert _mod.on_node_share(W, place, gpn) == pytest.approx(1.0)


def test_grouping_always_returns_full_nodes():
    """A permutation that left a node with three devices is not a placement."""
    rng = np.random.default_rng(0)
    W = rng.integers(0, 50, size=(16, 16)).astype(np.int64)
    np.fill_diagonal(W, 0)
    place = _mod.group_devices_by_traffic(W, 4)
    node = place // 4
    counts = np.bincount(node, minlength=4)
    assert counts.tolist() == [4, 4, 4, 4]


def test_empty_traffic_is_an_error_not_a_zero():
    with pytest.raises(ValueError, match="no halo traffic"):
        _mod.on_node_share(np.zeros((4, 4), dtype=np.int64), np.arange(4), 2)
