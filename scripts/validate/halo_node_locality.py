"""How much of an unstructured mesh's halo traffic has to leave the node?

An isolated exchange measurement on Levante (job 27088012, sixteen arms, two
repetitions, all agreeing to under 1%) found that taking a partner off the node
costs a factor of 2.4 in bandwidth, and that moving it from two nodes away to
nine costs nothing measurable. So the question for a decomposition is not how
far apart its neighbours are placed; it is what fraction of the halo has to
cross a node boundary at all.

This answers that offline, on the CPU, before any GPU time is spent. For a given
mesh, device count and partitioner it reports:

  * the halo traffic matrix -- how many halo cells each device has to receive
    from each other device,
  * the share of that traffic that stays inside a four-GPU node under the
    natural device labelling, where device ``r`` sits on node ``r // 4``,
  * the share that would stay inside a node if the devices were RELABELLED so
    that heavily-communicating devices are placed together, which is a pure
    permutation of the device-to-partition map and changes nothing about the
    mesh, the physics or the halo code.

The gap between those two shares is the whole prize of a node-aware placement.
This script does NOT convert it into a predicted speed-up: the exchange time
also depends on how the collective library schedules whatever remains, and
turning a byte fraction into a time here would be a claim the script cannot
support.

Usage
-----
    python scripts/validate/halo_node_locality.py \
        --subdivision 9 --n-devices 64 --partition-method metis --out out.json
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np


def owner_of_each_cell(n_cells: int, n_devices: int) -> np.ndarray:
    """Device that owns each cell of a mesh already reordered for sharding.

    ``reorder_voronoi_for_sharding`` puts every cell owned by device 0 first,
    then device 1, and pads so the count divides; JAX then splits the array into
    equal contiguous chunks. So ownership is position, not a lookup table.
    """
    if n_cells % n_devices:
        raise ValueError(
            f"nCells={n_cells} is not divisible by n_devices={n_devices}; the "
            "reordered mesh is padded for a specific device count and this "
            "script must be run at that count.")
    return np.repeat(np.arange(n_devices), n_cells // n_devices)


def halo_traffic_matrix(cells_on_cell: np.ndarray, owner: np.ndarray,
                        n_devices: int, halo_depth: int) -> np.ndarray:
    """``W[r, s]`` = halo cells device ``r`` must receive from device ``s``.

    Walks the same rings as ``voronoi_partition.compute_halo_cells``: a cell is
    in device ``r``'s halo if it is reachable from ``r``'s owned cells within
    ``halo_depth`` neighbour hops and ``r`` does not own it. Every such cell is
    one column of state that has to arrive from its owner each step, so the
    matrix is proportional to bytes.
    """
    n_cells = owner.shape[0]
    W = np.zeros((n_devices, n_devices), dtype=np.int64)
    nbr = cells_on_cell                      # (maxEdges, nCells), -1 = none
    for r in range(n_devices):
        owned = owner == r
        seen = owned.copy()                  # owned cells are never halo
        frontier = owned
        halo = np.zeros(n_cells, dtype=bool)
        for _ in range(halo_depth):
            idx = np.flatnonzero(frontier)
            cand = nbr[:, idx].ravel()
            cand = cand[cand >= 0]
            cand = np.unique(cand)
            fresh = cand[~seen[cand]]
            if fresh.size == 0:
                break
            halo[fresh] = True
            seen[fresh] = True
            frontier = np.zeros(n_cells, dtype=bool)
            frontier[fresh] = True
        counts = np.bincount(owner[halo], minlength=n_devices)
        W[r] = counts
        # A device is never in its own halo: the ring walk excludes owned cells.
        if counts[r] != 0:
            raise AssertionError(
                f"device {r} appears in its own halo ({counts[r]} cells); the "
                "ring walk is wrong and every number below would be too.")
    return W


def on_node_share(W: np.ndarray, place: np.ndarray, gpus_per_node: int) -> float:
    """Fraction of halo cells whose owner sits on the same node as the receiver.

    ``place[d]`` is the slot device ``d``'s partition occupies, so
    ``place[d] // gpus_per_node`` is its node.
    """
    node = place // gpus_per_node
    same = node[:, None] == node[None, :]
    total = W.sum()
    if total == 0:
        raise ValueError("no halo traffic at all; the matrix is empty")
    return float(W[same].sum()) / float(total)


def group_devices_by_traffic(W: np.ndarray, gpus_per_node: int) -> np.ndarray:
    """Relabel devices so heavy talkers share a node. Returns ``place``.

    The device-adjacency graph is tiny (one vertex per device), so the same
    graph partitioner used on the mesh is asked to cut it into node-sized
    groups with the halo counts as edge weights. The groups are then laid out
    in order, which is what makes ``place // gpus_per_node`` a node.
    """
    import pymetis

    n = W.shape[0]
    if n % gpus_per_node:
        raise ValueError(f"{n} devices does not divide into groups of "
                         f"{gpus_per_node}")
    sym = W + W.T
    np.fill_diagonal(sym, 0)
    # Edge weights need the CSR form: the list-of-lists ``adjacency=`` spelling
    # cannot carry them (pymetis docstring), and the loose xadj/adjncy keywords
    # are deprecated in this version.
    xadj, adjncy, eweights = [0], [], []
    for r in range(n):
        nbrs = np.flatnonzero(sym[r])
        adjncy.extend(int(x) for x in nbrs)
        eweights.extend(int(w) for w in sym[r, nbrs])
        xadj.append(len(adjncy))
    _, membership = pymetis.part_graph(
        n // gpus_per_node,
        adjacency=pymetis.CSRAdjacency(adj_starts=xadj, adjacent=adjncy),
        eweights=eweights)
    membership = np.asarray(membership)

    # METIS balances but does not guarantee exactly gpus_per_node per group.
    # Rebalance greedily, moving devices out of oversized groups into
    # undersized ones, cheapest talker first, so the result is a real placement
    # rather than one that only exists on paper.
    groups = [list(np.flatnonzero(membership == g))
              for g in range(n // gpus_per_node)]
    over = [g for g in groups if len(g) > gpus_per_node]
    under = [g for g in groups if len(g) < gpus_per_node]
    while over and under:
        src, dst = over[0], under[0]
        internal = [(sym[d, [x for x in src if x != d]].sum(), d) for d in src]
        internal.sort()
        _, moved = internal[0]
        src.remove(moved)
        dst.append(moved)
        if len(src) <= gpus_per_node:
            over.pop(0)
        if len(dst) >= gpus_per_node:
            under.pop(0)
    place = np.empty(n, dtype=np.int64)
    slot = 0
    for g in groups:
        for d in g:
            place[d] = slot
            slot += 1
    if sorted(place.tolist()) != list(range(n)):
        raise AssertionError("relabelling is not a permutation")
    return place


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subdivision", type=int, default=9)
    ap.add_argument("--n-devices", type=int, required=True)
    ap.add_argument("--gpus-per-node", type=int, default=4)
    ap.add_argument("--partition-method", default="metis",
                    choices=["metis", "sfc", "geometric"])
    ap.add_argument("--halo-depth", type=int, default=2)
    ap.add_argument("--lloyd", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    from legoesm.grids.voronoi import create_voronoi_mesh
    from legoesm.parallel.voronoi_partition import reorder_voronoi_for_sharding

    t0 = time.time()
    mesh = create_voronoi_mesh(subdivision_level=args.subdivision,
                               lloyd_iterations=args.lloyd)
    mesh = reorder_voronoi_for_sharding(mesh, args.n_devices,
                                        method=args.partition_method)
    cells_on_cell = np.asarray(mesh.cellsOnCell)
    if cells_on_cell.shape[0] != mesh.maxEdges:
        raise AssertionError(
            f"cellsOnCell is {cells_on_cell.shape}; this script indexes it as "
            f"(maxEdges={mesh.maxEdges}, nCells={mesh.nCells})")
    owner = owner_of_each_cell(int(mesh.nCells), args.n_devices)
    t_setup = time.time() - t0

    t0 = time.time()
    W = halo_traffic_matrix(cells_on_cell, owner, args.n_devices,
                            args.halo_depth)
    t_walk = time.time() - t0

    natural = np.arange(args.n_devices, dtype=np.int64)
    grouped = group_devices_by_traffic(W, args.gpus_per_node)
    rec = {
        "component": "halo_node_locality",
        "subdivision": args.subdivision,
        "n_cells": int(mesh.nCells),
        "n_devices": args.n_devices,
        "gpus_per_node": args.gpus_per_node,
        "partition_method": args.partition_method,
        "halo_depth": args.halo_depth,
        "lloyd_iterations": args.lloyd,
        "halo_cells_total": int(W.sum()),
        "halo_cells_per_device_mean": float(W.sum(axis=1).mean()),
        "halo_cells_per_device_max": int(W.sum(axis=1).max()),
        "partner_devices_per_device_mean": float((W > 0).sum(axis=1).mean()),
        "on_node_share_natural": on_node_share(W, natural, args.gpus_per_node),
        "on_node_share_grouped": on_node_share(W, grouped, args.gpus_per_node),
        "placement_grouped": grouped.tolist(),
        "setup_s": round(t_setup, 1),
        "traffic_walk_s": round(t_walk, 1),
    }
    print(json.dumps(rec, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
