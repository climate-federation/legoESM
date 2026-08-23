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




def balanced_edge_coloring(pairs, weights, n_colors_max):
    """Heaviest-first best-fit, then steepest-descent single-pair moves.

    Written by GLM-5.2 against this objective, which it identified as batch
    scheduling on an unbounded machine rather than as graph colouring: a class
    costs its size times its largest member, so a heavy pair prices the whole
    class it sits in and belongs somewhere sparse, while light pairs should
    fill the crowded classes.

    Scored here rather than shipped: it is a candidate against the colouring
    the model already runs, and the comparison is the point.
    """
    pairs = [tuple(p) for p in pairs]
    n = len(pairs)
    w = np.asarray(weights, float)
    n_colors = int(n_colors_max)
    key = w[:, 0] + w[:, 1]
    order = sorted(range(n), key=lambda i: (-key[i], i))

    used = [set() for _ in range(n_colors)]
    items = [[] for _ in range(n_colors)]
    count = [0] * n_colors
    max_c = [-np.inf] * n_colors
    max_e = [-np.inf] * n_colors
    color = {}

    def cost(k):
        return count[k] * (max_c[k] + max_e[k]) if count[k] else 0.0

    for i in order:
        a, b = pairs[i]
        c, e = w[i, 0], w[i, 1]
        best, best_delta = -1, np.inf
        for k in range(n_colors):
            if a in used[k] or b in used[k]:
                continue
            delta = ((count[k] + 1) * (max(max_c[k], c) + max(max_e[k], e))
                     - cost(k))
            if delta < best_delta - 1e-12:
                best, best_delta = k, delta
        if best < 0:
            raise ValueError(
                f"n_colors_max={n_colors} is below the edge-chromatic number")
        used[best].update((a, b))
        items[best].append(i)
        count[best] += 1
        max_c[best] = max(max_c[best], c)
        max_e[best] = max(max_e[best], e)
        color[i] = best

    for _ in range(1000):
        improved = False
        for i in range(n):          # ascending index, so this is deterministic
            a, b = pairs[i]
            c, e = w[i, 0], w[i, 1]
            k0 = color[i]
            used[k0].difference_update((a, b))
            items[k0].remove(i)
            count[k0] -= 1
            if count[k0]:
                max_c[k0] = max(w[j, 0] for j in items[k0])
                max_e[k0] = max(w[j, 1] for j in items[k0])
            else:
                max_c[k0] = max_e[k0] = -np.inf
            back = ((count[k0] + 1) * (max(max_c[k0], c) + max(max_e[k0], e))
                    - cost(k0))
            best_k, best_delta = k0, 0.0
            for k in range(n_colors):
                if a in used[k] or b in used[k]:
                    continue
                delta = ((count[k] + 1) * (max(max_c[k], c) + max(max_e[k], e))
                         - cost(k) - back)
                if delta < best_delta - 1e-9:
                    best_k, best_delta = k, delta
            used[best_k].update((a, b))
            items[best_k].append(i)
            count[best_k] += 1
            max_c[best_k] = max(max_c[best_k], c)
            max_e[best_k] = max(max_e[best_k], e)
            color[i] = best_k
            if best_delta < -1e-9:
                improved = True
        if not improved:
            break
    return {pairs[i]: color[i] for i in range(n)}


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
    parser.add_argument("--anneal", type=float, default=0.0,
                        help="millions of moves to spend searching harder "
                             "than the model does, to bound what is left "
                             "to win")
    parser.add_argument("--anneal-extra-rounds", type=int, default=1,
                        help="colours the anneal may use beyond the shipped "
                             "count")
    args = parser.parse_args()

    from legoesm.parallel import sharded_dynamics as sd
    from legoesm.parallel.sharded_dynamics import anneal_coloring

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
    # The first two rows are the SEEDS the size-aware search starts from; the
    # third is what the model actually ships, produced by calling the model's
    # own colouring rather than a re-derivation of it. Quoting a seed row as
    # "the schedule" overstates what is left to win by about a factor of two.
    seed_rounds = max(baseline.values()) + 1
    adopted, adopted_rounds, method = sd.size_aware_edge_coloring(
        pair_set, baseline, pair_w, seed_rounds)

    for name, colors in (("greedy seed", greedy),
                         ("multi-ordering seed", baseline),
                         (f"SHIPPED ({method})", adopted)):
        c = census(colors, pairs, cell_send, edge_send, args.devices)
        print(f"{name:>16}: {c['rounds']:3d} rounds   "
              f"{c['real']:>10,} real   {c['shipped']:>11,} shipped   "
              f"padding {c['padding_fraction']:6.1%}   "
              f"weight {sd.padded_weight(colors, pair_w):,}")
        if args.per_round:
            for r, directed, mc, me in c["per_round"]:
                print(f"      round {r:2d}: {directed:3d} directed pairs  "
                      f"max cells {mc:6d}  max edges {me:6d}")

    for n_colors in (adopted_rounds - 1, adopted_rounds, adopted_rounds + 1):
        try:
            cand = balanced_edge_coloring(
                pairs, np.array([pair_w[p] for p in pairs], float), n_colors)
        except ValueError as exc:
            print(f"{'batch-greedy ' + str(n_colors):>16}: {exc}")
            continue
        assert sd.check_proper_edge_coloring(cand, pair_set), (
            "batch-greedy returned a colouring that would collide")
        c = census(cand, pairs, cell_send, edge_send, args.devices)
        print(f"{'batch-greedy':>16}: {c['rounds']:3d} rounds   "
              f"{c['real']:>10,} real   {c['shipped']:>11,} shipped   "
              f"padding {c['padding_fraction']:6.1%}   "
              f"weight {sd.padded_weight(cand, pair_w):,}")

    if args.anneal > 0:
        searched = anneal_coloring(
            pair_set, pair_w, adopted,
            adopted_rounds + args.anneal_extra_rounds,
            int(args.anneal * 1e6))
        assert sd.check_proper_edge_coloring(searched, pair_set), (
            "the anneal returned a colouring that would collide at a device")
        c = census(searched, pairs, cell_send, edge_send, args.devices)
        print(f"{'annealed':>16}: {c['rounds']:3d} rounds   "
              f"{c['real']:>10,} real   {c['shipped']:>11,} shipped   "
              f"padding {c['padding_fraction']:6.1%}   "
              f"weight {sd.padded_weight(searched, pair_w):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
