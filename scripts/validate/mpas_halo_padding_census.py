"""How much of the icosahedral halo's wire traffic is padding?

The coloured halo schedule pads every pair in a round up to that round's
LARGEST halo, because ``shard_map`` needs one shape for all devices. A device
with a small halo in a round therefore ships the round maximum. That padding
is pure waste on the wire, and how much of it there is decides whether a
better round colouring is worth building.

This probe answers that offline, on a CPU, without an allocation. It rebuilds
the halo maps for a given mesh, device count and halo depth, colours them the
way production does, and reports the real rows, the shipped rows, and the
padding fraction -- per round and in total.

    python scripts/validate/mpas_halo_padding_census.py --devices 64 \
        --subdivision 9 --depth 9

Depth is the number of ghost rings the schedule is built for. It is NOT the
base halo depth when wide halo is on: the wide-halo step multiplies the base
depth by the integrator's tendency evaluations, so the 64-GPU production
configuration builds at depth 9, not 3. Passing the wrong depth silently
censuses a schedule the run never uses.

The mesh build and the neighbour walk are the slow part (~25 s at subdivision
9), so the pair map is cached under the mesh cache directory and reused.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _halo_maps(subdivision, n_dev, depth):
    """Per-pair halo row counts, the same ones the schedule is built from.

    Returns ``(pairs, cell_send, edge_send)`` where the two maps are keyed by
    the DIRECTED pair ``(source, destination)``.
    """
    from legoesm.grids.voronoi import create_voronoi_mesh
    from legoesm.parallel.voronoi_partition import reorder_voronoi_for_sharding

    mesh = reorder_voronoi_for_sharding(
        create_voronoi_mesh(subdivision_level=subdivision, lloyd_iterations=0),
        n_dev, method="sfc")
    n_cells, n_edges = int(mesh.nCells), int(mesh.nEdges)
    cells_per, edges_per = n_cells // n_dev, n_edges // n_dev
    cells_on_cell = np.asarray(mesh.cellsOnCell)
    cells_on_edge = np.asarray(mesh.cellsOnEdge)
    edges_on_cell = np.asarray(mesh.edgesOnCell)
    owner = np.repeat(np.arange(n_dev, dtype=np.int32), cells_per)

    cell_send: dict[tuple[int, int], int] = {}
    edge_send: dict[tuple[int, int], int] = {}
    pairs: set[tuple[int, int]] = set()

    for rank in range(n_dev):
        owned_c = np.arange(rank * cells_per, (rank + 1) * cells_per,
                            dtype=np.int64)
        owned_e = np.arange(rank * edges_per, (rank + 1) * edges_per,
                            dtype=np.int64)
        present = np.zeros(n_cells, dtype=bool)
        present[owned_c] = True
        frontier = owned_c
        for _ in range(depth):
            nb = cells_on_cell[:, frontier].ravel()
            nb = np.unique(nb[nb >= 0])
            new = nb[~present[nb]]
            present[new] = True
            frontier = new
        nb = cells_on_edge[:, owned_e].ravel()
        nb = np.unique(nb[nb >= 0])
        present[nb[~present[nb]]] = True
        for _ in range(2):
            cur = np.flatnonzero(present)
            cand = edges_on_cell[:, cur].ravel()
            cand = np.unique(cand[cand >= 0])
            nb = cells_on_edge[:, cand].ravel()
            nb = np.unique(nb[nb >= 0])
            present[nb[~present[nb]]] = True

        halo_c = np.flatnonzero(present)
        halo_c = halo_c[(halo_c < rank * cells_per)
                        | (halo_c >= (rank + 1) * cells_per)].astype(np.int64)
        local_c = np.concatenate([owned_c, halo_c])
        cand = edges_on_cell[:, local_c].ravel()
        cand = np.unique(cand[cand >= 0])
        local_e = cand[present[cells_on_edge[0, cand]]
                       & present[cells_on_edge[1, cand]]]
        halo_e = np.setdiff1d(local_e, owned_e, assume_unique=True)

        for src in np.unique(owner[halo_c]):
            n = int((owner[halo_c] == src).sum())
            cell_send[(int(src), rank)] = n
            pairs.add(tuple(sorted((rank, int(src)))))
        edge_src = np.minimum(halo_e // edges_per, n_dev - 1)
        for src in np.unique(edge_src):
            n = int((edge_src == src).sum())
            edge_send[(int(src), rank)] = n
            pairs.add(tuple(sorted((rank, int(src)))))

    return sorted(pairs), cell_send, edge_send


def _cached_maps(subdivision, n_dev, depth, cache_dir):
    key = Path(cache_dir) / f"halo_census_s{subdivision}_n{n_dev}_d{depth}.npz"
    if key.exists():
        z = np.load(key, allow_pickle=True)
        return ([tuple(p) for p in z["pairs"]],
                dict(z["cell_send"].item()), dict(z["edge_send"].item()))
    started = time.time()
    pairs, cell_send, edge_send = _halo_maps(subdivision, n_dev, depth)
    print(f"built halo maps in {time.time() - started:.1f}s", flush=True)
    key.parent.mkdir(parents=True, exist_ok=True)
    np.savez(key, pairs=np.array(pairs), cell_send=cell_send,
             edge_send=edge_send)
    return pairs, cell_send, edge_send


def census(colors, pairs, cell_send, edge_send, n_dev):
    """Real versus shipped rows under one colouring."""
    rounds = defaultdict(list)
    for pair, color in colors.items():
        rounds[color].append(pair)
    real_c = sum(cell_send.values())
    real_e = sum(edge_send.values())
    shipped_c = shipped_e = 0
    per_round = []
    for r in sorted(rounds):
        members = rounds[r]
        max_c = max(max(cell_send.get((u, v), 0), cell_send.get((v, u), 0))
                    for u, v in members)
        max_e = max(max(edge_send.get((u, v), 0), edge_send.get((v, u), 0))
                    for u, v in members)
        directed = 2 * len(members)
        shipped_c += directed * max_c
        shipped_e += directed * max_e
        per_round.append((r, directed, max_c, max_e))
    return {
        "rounds": len(rounds),
        "real": real_c + real_e,
        "shipped": shipped_c + shipped_e,
        "padding_fraction": 1.0 - (real_c + real_e) / (shipped_c + shipped_e),
        "per_round": per_round,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", type=int, default=64)
    parser.add_argument("--subdivision", type=int, default=9)
    parser.add_argument("--depth", type=int, default=9,
                        help="ghost rings the schedule is built for; 9 is the "
                             "64-GPU wide-halo production value")
    parser.add_argument("--nlev", type=int, default=26)
    parser.add_argument("--cache-dir", default=os.environ.get(
        "LEGOESM_MESH_CACHE_DIR", ".mesh_cache"))
    parser.add_argument("--per-round", action="store_true")
    args = parser.parse_args()

    from legoesm.parallel import sharded_dynamics as sd

    pairs, cell_send, edge_send = _cached_maps(
        args.subdivision, args.devices, args.depth, args.cache_dir)
    pair_set = set(pairs)

    # Production picks the fewer-rounds colouring, then locally searches for a
    # lighter one at the same round count. The weights are the per-pair wire
    # widths the search minimises.
    pair_w = {}
    for u, v in pairs:
        c = max(cell_send.get((u, v), 0), cell_send.get((v, u), 0))
        e = max(edge_send.get((u, v), 0), edge_send.get((v, u), 0))
        pair_w[(u, v)] = (c * (args.nlev + 2), e * args.nlev)

    greedy = sd.greedy_edge_coloring(pair_set)
    multi, max_degree = sd.multi_ordering_edge_coloring(pair_set)
    baseline = multi if (max(multi.values()) + 1
                         < max(greedy.values()) + 1) else greedy

    print(f"devices {args.devices}  subdivision {args.subdivision}  "
          f"depth {args.depth}  pairs {len(pairs)}  max degree {max_degree}")
    # NOTE: these two are the SEEDS the size-aware search starts from, not
    # what production ships. Size-aware colouring is on by default, and it
    # rewrites the colouring before it is used -- at 64 devices, depth 9, it
    # takes the padding from 75.5% down to 34.4%. Quoting a seed row as "the
    # schedule" overstates the remaining prize by a factor of two.
    for name, colors in (("greedy seed", greedy),
                         ("multi-ordering seed", baseline)):
        c = census(colors, pairs, cell_send, edge_send, args.devices)
        print(f"{name:>16}: {c['rounds']:3d} rounds   "
              f"{c['real']:>10,} real   {c['shipped']:>11,} shipped   "
              f"padding {c['padding_fraction']:6.1%}   "
              f"weight {sd._padded_weight(colors, pair_w):,}")
        if args.per_round:
            for r, directed, mc, me in c["per_round"]:
                print(f"      round {r:2d}: {directed:3d} directed pairs  "
                      f"max cells {mc:6d}  max edges {me:6d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
