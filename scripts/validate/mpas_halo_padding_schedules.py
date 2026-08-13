"""How much of the MPAS halo payload is padding, and can a better
schedule remove it?  Offline, CPU only, no GPU hours.

WHY THIS EXISTS
---------------
The coloured halo schedule sends, in each round, the ROUND'S MAXIMUM
payload for every participating pair — ``ppermute`` moves a fixed-shape
buffer, so a round costs ``n_pairs_r * max_size_r``. A previous audit
measured the resulting inflation at subdiv-9 on 64 devices:

    actual  175.0 MB/fill
    padded  512.0 MB/fill      inflation 2.93
    with size-aware colouring  inflation 2.16

The exchange on that lane is bandwidth-dominated (fitting three
independent A/B receipts gives roughly 0.57 ms of per-round latency
against 2.64 ms of payload in a 6.98 ms step), so bytes are worth more
than rounds there, and 2.16x of them are padding. The alternative that
has no padding at all — one ``ragged_all_to_all`` — is measured FASTER
below 64 devices (0.686x at 16, 0.735x at 32) and SLOWER at 64
(1.220x), because its offset/size arrays are ``(n_dev, n_dev)``: every
rank carries a slice entry for every other rank, almost all of them
empty, and that cost grows with device count while the neighbour degree
does not.

So the question this script answers is narrow and offline: given the
colouring constraint (a round is a MATCHING — each device in at most
one pair), how far below 2.16 can the inflation be pushed, and at what
cost in rounds?  Only if that number is convincing does a GPU A/B make
sense.

WHAT IT PRINTS
--------------
For each scheduling heuristic: round count, padded bytes, inflation,
and the modelled step cost under the fitted latency/bandwidth split, so
a heuristic that buys bytes by adding rounds is visible as such rather
than looking like a win. No verdict — the numbers are the output.

Usage
-----
    python scripts/validate/mpas_halo_padding_schedules.py \\
        --subdivision 9 --n-devices 64 --reorder-for 64
"""
from __future__ import annotations

import argparse

# Fitted from three independent Levante A/B receipts at subdiv-9 @ 64
# GPUs (wide halo 33->11 rounds -14.1 %; METIS -28.3 % padded bytes
# -10.5 % step; an nsys trace putting SendRecv at 61 % of the narrow
# step).  These convert a schedule into a MODELLED step time so that
# "fewer bytes, more rounds" trades are visible.  They are a model, not
# a measurement of any schedule below.
_LATENCY_PER_ROUND_S = 52e-6
_BANDWIDTH_S_PER_BYTE = 2.64e-3 / (512.0e6 / 1.0)   # 2.64 ms per 512 MB


def _pair_sizes(sched_pairs, cell_send_map, edge_send_map, cell_w, edge_w):
    """Bytes each DIRECTED pair actually has to move."""
    sizes = {}
    for (a, b) in sched_pairs:
        n_c = len(cell_send_map.get((a, b), ()))
        n_e = len(edge_send_map.get((a, b), ()))
        sizes[(a, b)] = n_c * cell_w + n_e * edge_w
    return sizes


def _cost(rounds, sizes):
    """Padded bytes of a schedule given as a list of rounds, each a list
    of directed pairs: every pair in a round ships the round maximum."""
    total = 0
    for rnd in rounds:
        if not rnd:
            continue
        total += len(rnd) * max(sizes[p] for p in rnd)
    return total


def _greedy_first_fit(pairs, sizes, order):
    """Proper edge colouring by first fit in the given pair order: a
    round is a matching, so a pair joins the first round in which
    neither endpoint is already busy."""
    rounds, busy = [], []
    for p in order:
        a, b = p
        for r, used in enumerate(busy):
            if a not in used and b not in used:
                rounds[r].append(p)
                used.add(a)
                used.add(b)
                break
        else:
            rounds.append([p])
            busy.append({a, b})
    return rounds


def _size_banded(pairs, sizes, n_bands):
    """Colour within size BANDS: pairs are sorted by size and cut into
    ``n_bands`` contiguous chunks of equal COUNT, and each chunk is
    coloured on its own, so a round only ever mixes pairs of similar
    size and the round maximum cannot be set by an outlier from another
    band.  More bands = tighter rounds but more of them.

    Ranked cuts, not value quantiles: on a skewed distribution with ties
    (one fat pair, many identical small ones) quantile edges collapse
    and every pair lands in one band, which silently reproduces the
    unbanded schedule.
    """
    ordered = sorted(pairs, key=lambda p: -sizes[p])
    n = len(ordered)
    if n == 0:
        return []
    nb = max(1, min(n_bands, n))
    cuts = [round(i * n / nb) for i in range(nb + 1)]
    rounds = []
    for k in range(nb):
        band = ordered[cuts[k]:cuts[k + 1]]
        if band:
            rounds.extend(_greedy_first_fit(band, sizes, band))
    return rounds


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subdivision", type=int, default=9)
    ap.add_argument("--n-devices", type=int, default=64)
    ap.add_argument("--reorder-for", type=int, default=None,
                    help="mesh reorder target; defaults to --n-devices, "
                         "which is what the production arms pass")
    ap.add_argument("--nlev", type=int, default=26)
    ap.add_argument("--bands", type=int, nargs="+", default=[2, 3, 4, 6, 8])
    args = ap.parse_args()

    from legoesm.grids.voronoi import create_voronoi_mesh
    from legoesm.parallel.voronoi_partition import (
        reorder_voronoi_for_sharding,
    )
    from legoesm.parallel.sharded_dynamics import (
        SPMD_HALO_DEPTH, _build_voronoi_partition_infra,
        _build_halo_send_maps,
    )

    tgt = args.reorder_for or args.n_devices
    mesh = create_voronoi_mesh(subdivision_level=args.subdivision,
                               lloyd_iterations=0)
    mesh = reorder_voronoi_for_sharding(mesh, tgt)
    n_dev = args.n_devices
    nC, nE = int(mesh.nCells), int(mesh.nEdges)
    cells_per, edges_per = nC // n_dev, nE // n_dev

    (_, _, _, _, _, max_lc, max_le, parts, owner) = \
        _build_voronoi_partition_infra(mesh, n_dev,
                                       halo_depth=SPMD_HALO_DEPTH)
    (comm_pairs, cell_send_map, _cr, edge_send_map, _er) = \
        _build_halo_send_maps(parts, owner, n_dev, cells_per, edges_per)

    cell_w = (args.nlev + 2) * 4     # T (nlev) + p_s + phis, f32
    edge_w = args.nlev * 4
    pairs = [p for p in cell_send_map.keys()]
    sizes = _pair_sizes(pairs, cell_send_map, edge_send_map,
                        cell_w, edge_w)
    actual = sum(sizes.values())

    print(f"# subdiv-{args.subdivision} n_dev={n_dev} reorder_for={tgt} "
          f"nlev={args.nlev}")
    print(f"# directed pairs={len(pairs)} comm_pairs={len(comm_pairs)} "
          f"actual={actual / 1e6:.2f} MB/fill")
    print(f"# model: {_LATENCY_PER_ROUND_S * 1e6:.0f} us/round + "
          f"{_BANDWIDTH_S_PER_BYTE * 1e6 * 1e6:.3f} us/MB  (fitted, not "
          f"measured per schedule)")
    print(f"{'schedule':>22} {'rounds':>7} {'padded_MB':>10} "
          f"{'inflation':>10} {'model_ms':>9}")

    def report(name, rounds):
        padded = _cost(rounds, sizes)
        ms = (len(rounds) * _LATENCY_PER_ROUND_S
              + padded * _BANDWIDTH_S_PER_BYTE) * 1e3
        print(f"{name:>22} {len(rounds):7d} {padded / 1e6:10.2f} "
              f"{padded / actual:10.3f} {ms:9.3f}")

    report("first-fit (as listed)", _greedy_first_fit(pairs, sizes, pairs))
    report("first-fit desc size",
           _greedy_first_fit(pairs, sizes,
                             sorted(pairs, key=lambda p: -sizes[p])))
    for nb in args.bands:
        report(f"size-banded x{nb}", _size_banded(pairs, sizes, nb))

    # Floor: no padding at all (what a true ragged exchange would move),
    # priced at the same bandwidth with the round count of the best
    # colouring — the target any schedule is trying to approach.
    best_r = min(len(_size_banded(pairs, sizes, nb)) for nb in args.bands)
    print(f"{'zero-padding floor':>22} {best_r:7d} {actual / 1e6:10.2f} "
          f"{1.0:10.3f} "
          f"{(best_r * _LATENCY_PER_ROUND_S + actual * _BANDWIDTH_S_PER_BYTE) * 1e3:9.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
