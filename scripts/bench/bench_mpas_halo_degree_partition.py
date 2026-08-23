"""Offline measurement of MPAS/Voronoi *ownership* levers (degree, bytes).

WHAT THIS IS
------------
The sharded MPAS atmosphere step fills its halo with an edge-coloured
``ppermute`` schedule.  One colour = one SEQUENTIAL collective ROUND, and the
round count is bounded below by the communication graph's MAXIMUM VERTEX
DEGREE -- the largest number of distinct partner devices any one device must
exchange with.  Edge CUT (what METIS minimises) is a different objective and
measured moves the wrong way (subdiv-8 @128: sfc 14 rounds, metis 19).

A "degree-aware" partitioner was named as a lever with a guessed ~14 % ceiling
and never built.  This probe MEASURES whether it is worth building.  It does
not build it: the greedy pass below is a throwaway prototype that lives HERE,
never in production.

PRE-REGISTERED GATE (written before the first run; do not edit after)
---------------------------------------------------------------------
Baseline = ``sfc`` at subdivision 9, 64 devices, halo depth 3.

* CONFIRM (worth building in production): the degree-aware pass cuts the
  ROUND COUNT by >= 25 % vs sfc, while keeping the partitioner's owned-cell
  imbalance within the balance tolerance (see BALANCE below) and inflating
  padded bytes by no more than ~10 %.
* REFUTE: < 10 % round reduction, or the balance/padding constraint has to be
  violated to get it.
* Anything between 10 % and 25 %, or a win bought with a padding/balance
  cost: report the numbers and let a human decide.  Do NOT build.

THE SECOND LEVER: PADDED BYTES (``--objective bytes``)
------------------------------------------------------
The degree gate above was REFUTED (rounds are already Vizing-optimal at
s9@64), but its prototype cut PADDED BYTES 9.4 % as a side effect, and the
lane is bytes-bound.  So the same machinery is re-pointed at the padding.

WHERE THE PADDING COMES FROM -- it is PER-ROUND MAX, not per-device max and
not a global max.  ``_build_ppermute_schedule`` walks the rounds one at a
time (``sharded_dynamics.py:2151``) and for round *r* takes

    max_c = max over BOTH directions of every pair in the round of
            len(cell_send_map[(src, dst)])            (:2153-2158)

then allocates the per-round index arrays at ``(n_dev, max_c)``
(``sc``/``rc`` at :2175-2179) and ships them with one ``ppermute``.  The
index arrays are shape-uniform across devices, so EVERY active pair in
round *r* moves ``max_c`` cell records and ``max_e`` edge records -- its own
count is irrelevant.  Hence

    padded = sum over rounds r of  n_active_pairs(r) * (max_c(r)*CELL_BYTES
                                                      + max_e(r)*EDGE_BYTES)

which is exactly :func:`padded_bytes` here.  Production's size-aware
colouring minimises the SAME SHAPE of quantity but not the same number
(``padded_weight``, :1917, counts each undirected pair once, works in
element-width units rather than float32 bytes, and does not apply the
schedule's ``max(..., 1)`` floor) -- normally a factor of 8 apart, and not
even proportional for a round whose entity max is 0.  This byte model is
also DRY-STATE: the cell record here is ``(nlev+2)`` float32 while production
also packs ``nlev * n_tracers`` values into the same cell buffer
(``q_flat``, :2288).  Tracer bytes do NOT cancel in a ratio, because a
candidate ownership can shift the cell/edge padding MIX -- so every MB and
every percentage in this probe is a DRY-STATE number, and a moist
production ratio would differ by however much the mix moved.  It is quoted
that way in the verdict and must be quoted that way in any report.
Two consequences drive the objective below: a round costs its WORST pair,
and a thin pair sharing a round with a fat one pays the fat one's price.
An ownership pass therefore wins bytes by (a) deleting thin corner contacts
-- each one is an extra pair paying the round max for almost no payload --
and (b) narrowing the spread of per-pair halo counts.  It does NOT win by
lowering total halo (that is ``actual_bytes``, the denominator).

PRE-REGISTERED GATE -- BYTE LEVER (written before the first byte run; do
not edit after).  Baseline = ``sfc`` at subdivision 9, 64 devices, depth 3.

* CONFIRM (worth building in production): padded bytes cut >= 20 % vs sfc,
  with the round count NOT increased, owned-cell counts inside the same
  floor/ceil tolerance the degree gate uses, and ``n_components_scored`` no
  worse than the sfc baseline's.
* REFUTE: < 10 % byte cut, or any of those constraints violated to get it.
* 10-20 %: report, do NOT build.

The byte lever is scored ENTIRELY by the production builders, one scoring
per trail step.  The first attempt used a cheap proxy padded weight and it
FAILED its control -- see :func:`bytes_trail_pass` for the measurement that
killed it and for what the proxy is still allowed to do.

HEADROOM, and the one thing it does NOT prove
---------------------------------------------
``n_rounds >= max_degree >= ceil(mean_degree)``: the first inequality is the
schedule (a proper edge colouring of the comm graph), the second is
arithmetic.  ``degree_floor_rounds = ceil(mean_degree)`` therefore bounds the
rounds achievable ON THIS GRAPH -- i.e. by RECOLOURING alone.  It is NOT a
bound across ownerships: a different partition is a different graph with a
different mean, which is exactly what the prototype tries to produce.  So a
small ``max_degree - degree_floor_rounds`` gap says recolouring is nearly
exhausted and any win must come from lowering the MEAN degree, which on a
sphere split into compact equal domains is set by geometry.  Read it as
headroom, never as a proof.

BALANCE -- why it is not free
-----------------------------
``_build_voronoi_partition_infra`` gives every device an EXACTLY equal
contiguous block (``cells_per = nCells // n_dev``) of the REORDERED, GHOST-
PADDED mesh.  The partitioner therefore only chooses the ORDER; ownership is
re-cut into equal blocks afterwards.  If a partitioner's own domains are
unequal, the block re-cut does not line up with them and a device's block
straddles two domains -- silently a different (worse) partition than the one
scored.  ``sfc`` is balanced to within one cell by construction (``nCells`` is
generally not divisible by the device count: subdiv-9 has 2 621 442 cells, so
2 of 128 domains carry one extra), and that is the tolerance the prototype
must respect too: every candidate is rebalanced back to the floor/ceil
distribution and its per-device domains are checked for CONNECTEDNESS, since
a count-balanced but shattered ownership is not a partitioner anyone could
ship.  The checks in the output are ``owner_counts`` (counts plus
``block_mismatch``, the fraction of cells the equal-block re-cut hands to a
device that did not claim them) and ``n_components_scored`` -- connectivity of
the BLOCKS production actually runs, not of the domains the partitioner
intended.  Both are reported for the baseline too, so the candidate is judged
against the same yardstick rather than an idealisation.

MEASUREMENT REUSE
-----------------
Round count and padded bytes come from the PRODUCTION builders
(``_build_voronoi_partition_infra`` -> ``_build_halo_send_maps`` ->
``_build_ppermute_schedule``), the same call chain
``spmd_schedule_cost``/``make_voronoi_sharded_step`` use, and the byte model
is the one from the receipted padding audit (job 26865788,
``scripts/cluster/scaling_levante/mpas_padding_audit.sbatch``): nlev=26, cell
record ``(nlev+2)*4`` bytes, edge record ``nlev*4`` bytes, per round every
active pair ships the round MAXIMUM.

One deliberate difference from that receipt: the size-aware colouring takes
per-pair WIDTHS, and production threads the real ones (``cell_width=nlev+2``,
``edge_width=nlev``) while the receipt's heredoc left the estimator's
unit-free 1:1 default.  This probe scores at the PRODUCTION widths by
default; ``--validate`` reproduces the receipt at its unit widths and then
also prints the production-width score, so the reproduction is exact and the
headline numbers are still the ones production would see.  If ``--validate``
does not reproduce the receipt, the instrument is not trusted and nothing
else in the output means anything.

The prototype's SEARCH uses a cheap sparse-matrix reach proxy for the comm
graph (cell halos only, no edge halos).  The proxy is validated against the
production degree vector on the baseline before it is used, and every
candidate it produces is re-scored with the production builders -- the proxy
never reports a headline number.

Run (compute node, CPU only)::

    LEGOESM_MESH_CACHE_DIR=/work/bd1083/b309178/diffESM/legoesm_mesh_cache \\
    JAX_PLATFORMS=cpu python scripts/bench/bench_mpas_halo_degree_partition.py \\
        --validate
    ... --subdivision 9 --n-dev 64 --methods sfc,metis,geometric --greedy
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

# Byte model of the receipted padding audit (job 26865788). nlev is the MPAS
# default column count used there; the dry cell record is (T over nlev, p_s,
# phis) and the edge record is the normal wind over nlev, both float32.
NLEV = 26
CELL_BYTES = (NLEV + 2) * 4
EDGE_BYTES = NLEV * 4


# --------------------------------------------------------------------------
# Production-backed scoring
# --------------------------------------------------------------------------
def score_prepared(prepared, n_dev, *, halo_depth=None, widths=None,
                   n_real_cells=None):
    """Score an ALREADY-REORDERED mesh with the production halo builders.

    Returns rounds, max/mean degree, the per-device degree histogram, the
    DRY-STATE actual and padded wire bytes of one halo fill (cell record =
    ``nlev+2`` float32; production adds ``nlev * n_tracers`` per cell, see
    the module docstring), and the inflation factor.

    *widths* is the ``(cell_width, edge_width)`` pair fed to the size-aware
    colouring.  Default = what production threads (``nlev+2``, ``nlev``);
    ``(1, 1)`` is the estimator's unit-free default, which the receipted
    padding audit used and which ``--validate`` therefore reproduces.
    """
    from legoesm.parallel.sharded_dynamics import (
        SPMD_HALO_DEPTH,
        _build_halo_send_maps,
        _build_ppermute_schedule,
        _build_voronoi_partition_infra,
    )

    if halo_depth is None:
        halo_depth = SPMD_HALO_DEPTH
    cell_width, edge_width = (NLEV + 2, NLEV) if widths is None else widths
    n_cells, n_edges = int(prepared.nCells), int(prepared.nEdges)
    if n_cells % n_dev or n_edges % n_dev:
        raise ValueError(
            f"prepared mesh nCells={n_cells} nEdges={n_edges} not divisible "
            f"by n_dev={n_dev}: the equal-block ownership would mis-slice.")
    cells_per, edges_per = n_cells // n_dev, n_edges // n_dev

    (*_, max_lc, max_le, parts, owner) = _build_voronoi_partition_infra(
        prepared, n_dev, halo_depth=halo_depth)
    pairs, cs, _cr, es, _er = _build_halo_send_maps(
        parts, owner, n_dev, cells_per, edges_per)
    sched = _build_ppermute_schedule(
        parts, owner, n_dev, cells_per, edges_per, max_lc, max_le,
        cell_width=cell_width, edge_width=edge_width)

    deg = degree_vector(pairs, n_dev)
    # Connectivity of the ownership that is ACTUALLY SCORED -- the equal
    # contiguous blocks of the reordered, ghost-padded mesh -- not of the
    # partitioner's intended domains. The re-cut can straddle domain
    # boundaries (see ``block_mismatch``), so checking the intended ownership
    # would certify a partition production never sees. Ghost cells carry no
    # connectivity and would each count as their own component, so they are
    # excluded; *n_real_cells* defaults to every cell (unpadded mesh).
    n_real = n_cells if n_real_cells is None else int(n_real_cells)
    block_owner = np.minimum(np.arange(n_real) // cells_per, n_dev - 1)
    n_comp = domain_components(
        cell_adjacency(prepared, limit=n_real), block_owner, n_dev)

    actual = (sum(len(v) for v in cs.values()) * CELL_BYTES
              + sum(len(v) for v in es.values()) * EDGE_BYTES)
    padded = padded_bytes(sched)
    return {
        "n_dev": n_dev,
        "halo_depth": halo_depth,
        "coloring_widths": [int(cell_width), int(edge_width)],
        "n_components_scored": int(n_comp),
        "comm_pairs": sorted(tuple(map(int, p)) for p in pairs),
        "n_rounds": int(sched["n_rounds"]),
        "n_rounds_greedy": int(sched["n_rounds_greedy"]),
        "coloring_method": sched["coloring_method"],
        "max_degree": int(sched.get("max_degree", -1)),
        "degree_max": int(deg.max()),
        "degree_mean": float(deg.mean()),
        "degree_min": int(deg.min()),
        "degree_hist": {int(k): int(v) for k, v in
                        zip(*np.unique(deg, return_counts=True))},
        # Floor on the rounds achievable ON THIS GRAPH, i.e. by RECOLOURING
        # alone: rounds >= max degree >= mean degree. NOT a bound across
        # ownerships -- a different partition has a different mean.
        "degree_floor_rounds": int(math.ceil(deg.mean())),
        "n_comm_pairs": len(pairs),
        "actual_bytes": int(actual),
        "padded_bytes": int(padded),
        "inflation": float(padded / actual) if actual else float("nan"),
        "max_local_cells": int(max_lc),
        "max_local_edges": int(max_le),
        "cells_per_device": cells_per,
    }


def degree_vector(comm_pairs, n_dev):
    """Per-device count of distinct exchange partners in the comm graph."""
    deg = np.zeros(n_dev, dtype=np.int64)
    for u, v in comm_pairs:
        deg[u] += 1
        deg[v] += 1
    return deg


def padded_bytes(sched):
    """Bytes one halo fill puts on the wire under the coloured schedule.

    ``ppermute`` moves the whole padded buffer, and the per-round index
    arrays are shape-uniform across devices, so every ACTIVE pair in a round
    ships that round's MAXIMUM cell/edge extent -- not its own count.
    """
    total = 0
    for r in range(sched["n_rounds"]):
        n_pairs = sum(1 for (a, b) in sched["ppermute_perms"][r] if a != b)
        total += n_pairs * (sched["halo_cells_per_round"][r] * CELL_BYTES
                            + sched["halo_edges_per_round"][r] * EDGE_BYTES)
    return total


# --------------------------------------------------------------------------
# Cheap search proxy: rank-contact matrix from a sparse depth-k reach
# --------------------------------------------------------------------------
def cell_adjacency(mesh, *, limit=None):
    """Symmetric boolean cell-cell adjacency as a scipy CSR matrix.

    ``mesh.cellsOnCell`` is (maxEdges, nCells) -- axis order is load-bearing.
    *limit* restricts the graph to the first *limit* cells (used to drop the
    appended ghost cells, which have no connectivity at all).
    """
    import scipy.sparse as sp

    coc = np.asarray(mesh.cellsOnCell)
    if coc.shape[0] != int(mesh.maxEdges):
        raise ValueError(
            f"cellsOnCell has shape {coc.shape}; expected "
            f"(maxEdges={int(mesh.maxEdges)}, nCells={int(mesh.nCells)})")
    n = int(mesh.nCells) if limit is None else int(limit)
    coc = coc[:, :n]
    cols = np.repeat(np.arange(n, dtype=np.int64)[None, :], coc.shape[0], 0)
    ok = (coc >= 0) & (coc < n)
    rows, cs = coc[ok].astype(np.int64), cols[ok]
    a = sp.csr_matrix((np.ones(rows.size, np.uint8), (rows, cs)), shape=(n, n))
    return ((a + a.T) > 0).astype(np.uint8).tocsr()


def reach_matrix(adj, owner, n_dev, depth, *, prev=None, dirty=None):
    """``R[c, d]`` = cell *c* lies within *depth* rings of some *d*-owned cell.

    ``dirty`` recomputes only those columns and reuses ``prev`` for the rest.
    A rank's reach depends only on ITS OWN owned set, so after a move that
    touches a handful of devices this is exact, not an approximation -- and it
    is what makes the search affordable at 2.6 M cells (a full recompute per
    candidate would cost minutes each).
    """
    ranks = range(n_dev) if dirty is None else sorted(dirty)
    seed = np.zeros((owner.size, len(ranks)), dtype=bool)
    for j, e in enumerate(ranks):
        seed[owner == e, j] = True
    r = seed
    for _ in range(depth):
        r = ((adj @ r.astype(np.uint8)) > 0) | r
    if dirty is None:
        return r
    out = prev.copy()
    out[:, ranks] = r
    return out


def contact_from_reach(owner, reach, n_dev):
    """``M[d, e]`` = number of *e*-owned cells within the reach of *d*.

    ``M[d, e] > 0`` (d != e) is exactly the cell-halo contact that puts the
    pair into the comm graph, so ``(M > 0)`` off-diagonal is the proxy graph.
    """
    import scipy.sparse as sp

    n = owner.size
    onehot = sp.csr_matrix(
        (np.ones(n, np.uint8), (np.arange(n), owner)), shape=(n, n_dev))
    return (onehot.T @ reach.astype(np.int64)).T


def contact_matrix(adj, owner, n_dev, depth):
    """``contact_from_reach`` on a freshly built reach matrix."""
    return contact_from_reach(
        owner, reach_matrix(adj, owner, n_dev, depth), n_dev)


def _proxy_graph(m):
    g = (m > 0) | (m > 0).T
    np.fill_diagonal(g, False)
    return g


def proxy_degrees(m):
    """Degree vector of the proxy graph (off-diagonal nonzeros, symmetrised)."""
    return _proxy_graph(m).sum(axis=1)


LEVERS = ("degree", "bytes")


def thin_contacts_first(m, n_dev):
    """Contacts of the proxy graph, THINNEST first.

    Thin contacts are what both levers attack: a corner partner is a whole
    extra comm pair for a handful of cells, and under the per-round-max
    padding it pays its round's maximum regardless of how little it carries.
    Ranking them is the ONE job the proxy is fit for -- see
    :func:`bytes_trail_pass` for why it is not fit to SCORE the result.
    """
    pairs = [(d, e) for d in range(n_dev) for e in range(n_dev)
             if d != e and (m[d, e] > 0 or m[e, d] > 0)]
    pairs.sort(key=lambda p: (m[p[0], p[1]] + m[p[1], p[0]], p))
    return pairs


def proxy_pair_set(m):
    """Undirected pair set of the proxy graph, for comparison with the
    production ``comm_pairs``.  The production graph also carries EDGE-halo
    contacts and two extra ``cellsOnEdge`` closure passes, so the proxy is a
    SUBSET in general -- the difference is measured, never assumed away."""
    u, v = np.nonzero(np.triu(_proxy_graph(m), 1))
    return {(int(a), int(b)) for a, b in zip(u, v)}


# --------------------------------------------------------------------------
# THROWAWAY prototype: greedy boundary reassignment against a chosen objective
# --------------------------------------------------------------------------
def greedy_degree_pass(adj, owner, n_dev, depth, *, max_iters=20,
                       max_move_frac=0.02, max_attempts=6, verbose=True):
    """Try to lower the comm graph's MAX degree by dissolving weak contacts.

    A device's degree is inflated by CORNER contacts -- partners that share
    only a handful of cells.  Each iteration takes the worst device, picks its
    weakest partner, and hands the few cells that create that contact to a
    third device, then rebalances the per-device counts.  A move is kept only
    if the sorted degree vector strictly improves.

    ponytail: greedy, single-move lookahead, at most *max_attempts* candidate
    moves per iteration, proxy objective (cell halos, no edge halos).
    Ceiling: it cannot reshape a domain globally, so it only removes contacts
    that are already thin.  Upgrade path if this ever matters: a proper
    multilevel partitioner with a max-degree objective.
    """
    owner = owner.copy()
    reach = reach_matrix(adj, owner, n_dev, depth)
    m = contact_from_reach(owner, reach, n_dev)
    best_key = tuple(sorted(proxy_degrees(m), reverse=True))
    cap = int(max_move_frac * owner.size / n_dev)
    history = [best_key[0]]

    for it in range(max_iters):
        deg = proxy_degrees(m)
        moved, tried = False, 0
        # Worst devices first; within a device, thinnest contact first.
        for d in np.argsort(-deg):
            partners = [e for e in range(n_dev)
                        if e != d and (m[d, e] > 0 or m[e, d] > 0)]
            for p in sorted(partners, key=lambda e: m[d, e] + m[e, d]):
                if tried >= max_attempts:
                    break
                cand = _dissolve_contact(adj, owner, reach, n_dev, d, p, cap)
                if cand is None:
                    continue
                tried += 1
                dirty = set(np.unique(owner[owner != cand]).tolist())
                dirty |= set(np.unique(cand[owner != cand]).tolist())
                reach_new = reach_matrix(adj, cand, n_dev, depth,
                                         prev=reach, dirty=dirty)
                m_new = contact_from_reach(cand, reach_new, n_dev)
                key = tuple(sorted(proxy_degrees(m_new), reverse=True))
                if key < best_key:
                    owner, reach, m = cand, reach_new, m_new
                    best_key, moved = key, True
                    break
            if moved or tried >= max_attempts:
                break
        history.append(best_key[0])
        if verbose:
            print(f"  greedy iter {it}: max proxy degree {best_key[0]} "
                  f"after {tried} attempted move(s) "
                  f"{'(moved)' if moved else '(no improving move; stop)'}",
                  flush=True)
        if not moved:
            break
    return owner, history


def bytes_trail_pass(mesh, adj, owner0, n_dev, depth, *, n_steps,
                     max_move_frac=0.02, n_real_cells=None, verbose=True):
    """ATTEMPT the thinnest contacts in turn; PRODUCTION-score every state.

    "Attempt", not "dissolve": a move hands the cells that create a contact
    to a third device and then rebalances, and the rebalance can hand some
    back, so the targeted contact sometimes survives.  ``attempted`` is the
    skip list, ``dissolved`` records only the contacts verified gone against
    the recomputed contact matrix, and both are on every trail row.

    WHY NO PROXY PRICES A ROW.  The first byte scorer was a
    proxy padded weight (colour the proxy contact graph, charge each pair its
    round maximum) and it FAILED its own control: it mis-ranked
    sfc/metis/geometric, whose production padded bytes are receipted, at BOTH
    s8@16 and s9@64.  The cause was isolated, not guessed -- with the TRUE
    per-pair payloads from ``_build_halo_send_maps`` the same formula ranks
    them correctly, so the formula is fine and the GRAPH is not: the cell-
    reach proxy sees 206 of the 269 production comm pairs at s9@64 (49 of 76
    for geometric at s8@16), because production's EDGE ownership is an
    equal-block cut of the REORDERED edge array and an ownership-space proxy
    cannot reproduce it.  Deepening the reach 3 -> 4 changed the pair set by
    exactly zero, so this is structural.  The proxy is therefore demoted to
    the one job it can do -- ranking contacts by thickness -- and every state
    on the trail is priced by the production builders.

    The proxy still CHOOSES which states get visited (ordering, move
    construction, termination), so a refutation bounds THIS search; it just
    cannot put a wrong number on a reported row.

    Returns ``(trail, owners)``: parallel lists, one entry per scored state,
    starting with the untouched *owner0*.  A TRAIL rather than a hill-climb
    because a production score costs minutes: the curve of padded bytes vs
    contacts attempted says whether the byte cut saturates or keeps going,
    which one accept/reject decision would not.

    ponytail: fixed attempt order, no backtracking, one production score
    per step.  Ceiling: *n_steps* production scorings (minutes each at s9), so
    it explores tens of moves out of hundreds of contacts.
    """
    n_real = int(mesh.nCells) if n_real_cells is None else int(n_real_cells)
    owner = owner0.copy()
    reach = reach_matrix(adj, owner, n_dev, depth)
    m = contact_from_reach(owner, reach, n_dev)
    cap = int(max_move_frac * owner.size / n_dev)

    def _score(own, step, dissolved, attempted):
        row = score_prepared(reorder_with_owner(mesh, n_dev, own), n_dev,
                             n_real_cells=n_real)
        row["owner_counts"] = owner_stats(own, n_dev,
                                          padded_cells(n_real, n_dev))
        row["owner_counts_at_target"] = row["owner_counts"]
        row["reorder_target"] = n_dev
        row["n_components"] = domain_components(adj, own, n_dev)
        row["trail_step"] = step
        row["dissolved_contacts"] = list(dissolved)
        row["attempted_contacts"] = list(attempted)
        if verbose:
            print(f"  [trail {step:2d}] dissolved {len(dissolved):2d}/"
                  f"{len(attempted):2d} attempted "
                  f"contact(s): rounds={row['n_rounds']} "
                  f"max_degree={row['max_degree']} "
                  f"padded={row['padded_bytes']/1e6:.1f} MB "
                  f"actual={row['actual_bytes']/1e6:.1f} MB "
                  f"inflation={row['inflation']:.3f} "
                  f"components={row['n_components']}/"
                  f"{row['n_components_scored']} "
                  f"counts={row['owner_counts']['min']}-"
                  f"{row['owner_counts']['max']}", flush=True)
        return row

    # `attempted` lists the contacts a move was actually MADE on, and doubles
    # as the skip list for those (retrying one livelocks the search); a
    # contact _dissolve_contact REJECTS -- too thick to move within `cap`, no
    # third device -- is not recorded and stays eligible on later steps, when
    # a changed ownership may make it movable. `dissolved` records only the
    # contacts verified GONE against the recomputed contact matrix: a
    # rebalance can hand cells back and leave the contact standing, so
    # conflating the two mislabels the trail.
    dissolved, attempted = [], []
    trail = [_score(owner, 0, dissolved, attempted)]
    owners = [owner.copy()]
    for step in range(1, n_steps + 1):
        moved = None
        for d, p in thin_contacts_first(m, n_dev):
            if (d, p) in attempted or (p, d) in attempted:
                continue
            cand = _dissolve_contact(adj, owner, reach, n_dev, d, p, cap)
            if cand is not None:
                moved = (d, p, cand)
                break
        if moved is None:
            if verbose:
                print(f"  [trail {step:2d}] no dissolvable contact left; stop",
                      flush=True)
            break
        d, p, cand = moved
        dirty = set(np.unique(owner[owner != cand]).tolist())
        dirty |= set(np.unique(cand[owner != cand]).tolist())
        reach = reach_matrix(adj, cand, n_dev, depth, prev=reach, dirty=dirty)
        owner, m = cand, contact_from_reach(cand, reach, n_dev)
        attempted.append((int(d), int(p)))
        if m[d, p] == 0 and m[p, d] == 0:
            dissolved.append((int(d), int(p)))
        trail.append(_score(owner, step, dissolved, attempted))
        owners.append(owner.copy())
    return trail, owners


def _dissolve_contact(adj, owner, reach, n_dev, d, p, cap):
    """Hand *d*'s cells that touch *p* to a third device; rebalance.

    Returns None when the contact is too thick to move within *cap* cells (a
    real shared boundary, not a corner), no third device is available, or the
    result cannot be rebalanced.
    """
    s = np.where((owner == d) & reach[:, p])[0]
    if s.size == 0 or s.size > cap:
        return None
    # Third device: the one owning the most cells adjacent to S.
    nbr = np.asarray(adj[s].sum(axis=0)).ravel() > 0
    counts = np.bincount(owner[nbr], minlength=n_dev).astype(np.int64)
    counts[d] = counts[p] = -1
    q = int(np.argmax(counts))
    if counts[q] <= 0:
        return None
    cand = owner.copy()
    cand[s] = q
    return _rebalance(adj, cand, n_dev)


def balance_bounds(n_cells, n_dev):
    """The floor/ceil counts an exactly-balanced partition can have.

    ``nCells`` is generally NOT divisible by the device count (subdiv-9:
    2 621 442 cells, 128 devices), and the production ``sfc`` partitioner is
    itself only balanced to within one cell, so demanding exact equality
    would reject every real partition -- including the baseline.
    """
    lo, extra = divmod(int(n_cells), int(n_dev))
    return lo, (lo + 1 if extra else lo)


def rank_adjacency(adj, owner, n_dev):
    """Boolean device-adjacency: ``G[a, b]`` iff some *a*-cell touches a
    *b*-cell.  Used to route rebalancing moves through neighbouring devices
    only, so a domain never sheds an island to a far-away device."""
    import scipy.sparse as sp

    # int64 throughout: a shared boundary of 256+ cell edges wraps a uint8
    # accumulator back to 0, and the resulting "these devices do not touch"
    # is a false negative that silently rejects a good candidate.
    onehot = sp.csr_matrix(
        (np.ones(owner.size, np.int64), (np.arange(owner.size), owner)),
        shape=(owner.size, n_dev))
    g = (onehot.T @ adj.astype(np.int64) @ onehot).toarray() > 0
    np.fill_diagonal(g, False)
    return g


def _rank_path(g, src, dst):
    """Shortest device path ``src -> ... -> dst`` in the device graph."""
    from collections import deque

    prev = {src: None}
    q = deque([src])
    while q:
        a = q.popleft()
        if a == dst:
            path = []
            while a is not None:
                path.append(a)
                a = prev[a]
            return path[::-1]
        for b in np.nonzero(g[a])[0]:
            b = int(b)
            if b not in prev:
                prev[b] = a
                q.append(b)
    return None


def _rebalance(adj, owner, n_dev, *, max_sweeps=2048):
    """Restore per-device cell counts to the floor/ceil distribution.

    Balance is a hard constraint, not a preference: the SPMD path re-cuts the
    reordered mesh into equal blocks, so an unbalanced partition is silently
    not the partition that was scored.  Moves are taken from cells of the
    surplus device that already touch the deficit device, so domains stay
    connected where possible; a candidate is checked for connectedness
    afterwards regardless.  Returns None if it cannot balance -- the caller
    then discards the candidate rather than scoring a lie.
    """
    lo, hi = balance_bounds(owner.size, n_dev)
    owner = owner.copy()
    for _ in range(max_sweeps):
        counts = np.bincount(owner, minlength=n_dev)
        if counts.max() <= hi and counts.min() >= lo:
            return owner
        # The over-full and the under-full device are usually NOT neighbours,
        # so route the transfer along a shortest DEVICE path and hand the same
        # amount along each hop: intermediate devices net zero, the source
        # loses and the sink gains, and the total imbalance strictly drops.
        # (A local "give to the emptiest neighbour" rule livelocks here: two
        # adjacent devices at the cap trade the same boundary cell forever
        # while a distant device stays starved.)
        s, f = int(np.argmax(counts)), int(np.argmin(counts))
        n_move = min(int(counts[s] - lo), int(hi - counts[f]))
        if n_move <= 0:
            return None
        path = _rank_path(rank_adjacency(adj, owner, n_dev), s, f)
        if path is None:
            return None
        for a, b in zip(path, path[1:]):
            a_cells = np.where(owner == a)[0]
            pick = a_cells[adj[a_cells].dot((owner == b).astype(np.uint8)) > 0]
            if pick.size == 0:
                return None
            n_move = min(n_move, int(pick.size))
            owner[pick[:n_move]] = b
    # The loop exits on the sweep budget, so re-test: the LAST sweep may well
    # have finished the job, and returning None there would falsely discard a
    # balanced candidate.
    counts = np.bincount(owner, minlength=n_dev)
    return owner if counts.max() <= hi and counts.min() >= lo else None


def domain_components(adj, owner, n_dev):
    """Number of connected components across all per-device domains.

    Equals *n_dev* exactly when every device owns ONE contiguous region.  A
    count-balanced ownership whose domains are shattered is not a partitioner
    anyone could ship, and it would flatter the degree metric.
    """
    from scipy.sparse.csgraph import connected_components

    same = adj.tocoo()
    keep = owner[same.row] == owner[same.col]
    import scipy.sparse as sp

    intra = sp.coo_matrix(
        (same.data[keep], (same.row[keep], same.col[keep])), shape=adj.shape)
    n_comp, _ = connected_components(intra, directed=False)
    return int(n_comp)


# --------------------------------------------------------------------------
# Mesh preparation
# --------------------------------------------------------------------------
def reorder_with_owner(mesh, n_target, owner):
    """Apply the PRODUCTION reorder using a caller-supplied ownership array.

    Harness-only glue: the production reorder chooses ownership from a fixed
    method set, so a prototype ownership is injected by swapping the
    partitioner it calls.  Everything after that -- the permutation, the
    connectivity remap, the ghost padding -- is the production code, so the
    scored mesh is exactly what production would hold.
    """
    from legoesm.parallel import voronoi_partition as vp

    if owner.shape != (int(mesh.nCells),):
        raise ValueError(f"owner shape {owner.shape} != ({int(mesh.nCells)},)")
    orig = vp.partition_cells_sfc
    vp.partition_cells_sfc = lambda m, n, order=None: owner.astype(np.int32)
    try:
        return vp.reorder_voronoi_for_sharding(mesh, n_target, method="sfc")
    finally:
        vp.partition_cells_sfc = orig


def load_mesh(subdivision, lloyd):
    from legoesm.grids.voronoi import create_voronoi_mesh

    return create_voronoi_mesh(subdivision_level=subdivision,
                               lloyd_iterations=lloyd)


def owner_stats(owner, n_dev, n_cells_padded=None):
    """Per-device owned-cell counts, plus how much of the partitioner's
    ownership the equal-block re-cut actually honours.

    The reorder sorts cells by (owner, Hilbert key) and production then slices
    the GHOST-PADDED mesh into ``nCells_padded // n_dev`` equal blocks.  When
    the partitioner's own counts are not exactly that block size -- and they
    never are, because ``nCells`` is not divisible (subdiv-9: 2 621 442 cells)
    -- the block boundaries drift away from the domain boundaries and some
    cells end up on a device that did not claim them.  ``block_mismatch`` is
    that fraction, measured rather than assumed away; it is reported for the
    baseline too, so the candidate is judged against the same yardstick.
    """
    c = np.bincount(owner, minlength=n_dev)
    out = {"min": int(c.min()), "max": int(c.max()),
           "imbalance": float(c.max() / c.mean())}
    if n_cells_padded is not None:
        per = int(n_cells_padded) // n_dev
        pos = np.concatenate([[0], np.cumsum(c)])
        mismatched = 0
        for d in range(n_dev):
            block = np.minimum(np.arange(pos[d], pos[d + 1]) // per, n_dev - 1)
            mismatched += int(np.count_nonzero(block != d))
        out["block_mismatch"] = float(mismatched / owner.size)
    return out


def padded_cells(n_cells, n_devices):
    """``nCells`` after the production ghost padding for *n_devices*."""
    return int(n_cells) + (-int(n_cells)) % int(n_devices)


# --------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--subdivision", type=int, default=9)
    ap.add_argument("--lloyd", type=int, default=0)
    ap.add_argument("--n-dev", type=int, default=64)
    ap.add_argument("--reorder-target", type=int, default=None,
                    help="device count the reorder targets (default: n-dev). "
                         "The receipted audit reorders for 128 and scores 64.")
    ap.add_argument("--methods", default="sfc")
    ap.add_argument("--greedy", action="store_true",
                    help="also run the throwaway ownership prototype")
    ap.add_argument("--objective", default="degree", choices=sorted(LEVERS),
                    help="which lever the prototype attacks: 'degree' (round "
                         "count, proxy-scored hill climb, REFUTED) or 'bytes' "
                         "(padded wire bytes, PRODUCTION-scored trail)")
    ap.add_argument("--greedy-iters", type=int, default=20,
                    help="degree lever: hill-climb iterations. bytes lever: "
                         "trail steps, each costing one production scoring")
    ap.add_argument("--greedy-attempts", type=int, default=6,
                    help="candidate moves tried per iteration; each costs a "
                         "partial reach recompute")
    ap.add_argument("--validate", action="store_true",
                    help="reproduce the receipted s9@64 configuration (job "
                         "26865788: rounds 12, padded 273 MB, inflation "
                         "1.560) and exit non-zero if it does not match")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    if args.validate:
        args.subdivision, args.lloyd, args.n_dev = 9, 0, 64
        args.reorder_target, args.methods = 128, "sfc"

    target = args.reorder_target or args.n_dev
    print(f"mesh: subdiv={args.subdivision} lloyd={args.lloyd} "
          f"reorder_target={target} n_dev={args.n_dev}", flush=True)
    mesh = load_mesh(args.subdivision, args.lloyd)
    print(f"  nCells={int(mesh.nCells)} nEdges={int(mesh.nEdges)} "
          f"cellsOnCell.shape={np.asarray(mesh.cellsOnCell).shape}", flush=True)

    from legoesm.parallel.voronoi_partition import (
        partition_cells_geometric, partition_cells_metis, partition_cells_sfc,
        reorder_voronoi_for_sharding,
    )
    partitioners = {"sfc": partition_cells_sfc,
                    "geometric": partition_cells_geometric,
                    "metis": partition_cells_metis}

    rows = {}
    owners_at_target = {}   # method -> ownership array (kept out of the JSON)
    for method in args.methods.split(","):
        method = method.strip()
        if method not in partitioners:
            raise SystemExit(f"unknown method {method!r}; "
                             f"choose from {sorted(partitioners)}")
        if method == "metis":
            try:
                import pymetis  # noqa: F401
            except ImportError:
                rows[method] = {"status": "unavailable (no pymetis)"}
                print(f"[{method}] unavailable", flush=True)
                continue
        print(f"[{method}] reordering...", flush=True)
        prepared = reorder_voronoi_for_sharding(mesh, target, method=method)
        row = score_prepared(prepared, args.n_dev,
                             n_real_cells=int(mesh.nCells))
        # Ownership is chosen for the REORDER TARGET, which is not always the
        # scored device count (the receipt reorders for 128 and scores 64), so
        # the key says which one these counts describe.
        row["reorder_target"] = target
        owners_at_target[method] = partitioners[method](
            mesh, target).astype(np.int64)
        row["owner_counts_at_target"] = owner_stats(
            owners_at_target[method], target,
            padded_cells(int(mesh.nCells), target))
        if target == args.n_dev:
            row["owner_counts"] = row["owner_counts_at_target"]
        if args.validate:
            # The receipt was taken at the estimator's unit-free colouring
            # widths; reproduce THOSE, and keep the production-width score
            # above as the headline.
            row["receipt_widths_score"] = score_prepared(
                prepared, args.n_dev, widths=(1, 1),
                n_real_cells=int(mesh.nCells))
        rows[method] = row
        print(f"[{method}] rounds={row['n_rounds']} "
              f"max_degree={row['max_degree']} "
              f"mean_degree={row['degree_mean']:.2f} "
              f"floor={row['degree_floor_rounds']} "
              f"padded={row['padded_bytes']/1e6:.1f} MB "
              f"inflation={row['inflation']:.3f} "
              f"imbalance={row['owner_counts_at_target']['imbalance']:.4f} "
              f"block_mismatch="
              f"{row['owner_counts_at_target']['block_mismatch']*100:.2f}% "
              f"(ownership at target={target})", flush=True)

    if args.validate:
        r = rows["sfc"]["receipt_widths_score"]
        ok = (r["n_rounds"] == 12
              and abs(r["padded_bytes"] / 1e6 - 273) < 3
              and abs(r["inflation"] - 1.560) < 0.01)
        print(f"\nINSTRUMENT VALIDATION vs job 26865788 at the receipt's "
              f"unit colouring widths (rounds 12, padded 273 MB, inflation "
              f"1.560): got rounds {r['n_rounds']}, padded "
              f"{r['padded_bytes']/1e6:.1f} MB, inflation {r['inflation']:.3f}"
              f" -> {'REPRODUCED' if ok else 'MISMATCH'}", flush=True)
        pr = rows["sfc"]
        print(f"  at PRODUCTION colouring widths {pr['coloring_widths']}: "
              f"rounds {pr['n_rounds']}, padded "
              f"{pr['padded_bytes']/1e6:.1f} MB, inflation "
              f"{pr['inflation']:.3f}", flush=True)
        if not ok:
            return 1

    if args.greedy:
        base = rows["sfc"]
        if target != args.n_dev:
            raise SystemExit(
                "--greedy requires --reorder-target == --n-dev: the proxy "
                "cannot be validated against a production graph built at a "
                "different device count, and an unvalidated proxy cannot "
                "decide this lever.")
        print("\n[greedy] building sparse adjacency + validating proxy...",
              flush=True)
        adj = cell_adjacency(mesh)
        owner0 = owners_at_target["sfc"]
        # VALIDATE THE PROXY against the production comm graph on the
        # baseline before it is allowed to rank candidates: compare the PAIR
        # SETS, not just the degree summary, and carry the mismatch into the
        # verdict so a proxy blind spot cannot masquerade as a refutation.
        proxy_pairs = proxy_pair_set(
            contact_matrix(adj, owner0, target, base["halo_depth"]))
        prod_pairs = {tuple(p) for p in base["comm_pairs"]}
        missed = sorted(prod_pairs - proxy_pairs)
        spurious = sorted(proxy_pairs - prod_pairs)
        pdeg = proxy_degrees(contact_matrix(adj, owner0, target,
                                            base["halo_depth"]))
        print(f"[greedy] proxy graph: {len(proxy_pairs)} pairs vs production "
              f"{len(prod_pairs)}; misses {len(missed)}, spurious "
              f"{len(spurious)}; proxy max/mean degree {int(pdeg.max())}/"
              f"{pdeg.mean():.2f} vs production {base['degree_max']}/"
              f"{base['degree_mean']:.2f}", flush=True)
        proxy_fidelity = {"n_proxy_pairs": len(proxy_pairs),
                          "n_prod_pairs": len(prod_pairs),
                          "n_missed": len(missed),
                          "n_spurious": len(spurious),
                          "proxy_degree_max": int(pdeg.max()),
                          "proxy_degree_mean": float(pdeg.mean())}

        if args.objective == "bytes":
            # PRODUCTION-SCORED TRAIL. The proxy never PRICES a state -- every
            # number below comes from the production builders. It does still
            # choose which states get visited (ordering, move construction,
            # termination), so its measured blind spot bounds THIS SEARCH; it
            # just cannot put a wrong number on a reported row.
            trail, _owners = bytes_trail_pass(
                mesh, adj, owner0, target, base["halo_depth"],
                n_steps=args.greedy_iters, n_real_cells=int(mesh.nCells))
            # Pick the best row that SATISFIES the constraints, not the
            # lowest-byte row outright: an infeasible 25% row would otherwise
            # hide a feasible 20% row further along and turn a CONFIRM into a
            # constraint REFUTE.
            lo_c, hi_c = balance_bounds(int(mesh.nCells), target)
            best, feasible = select_best_feasible(trail, base, lo_c, hi_c)
            rows["trail"] = [
                {k: r[k] for k in ("trail_step", "n_rounds", "max_degree",
                                   "padded_bytes", "actual_bytes", "inflation",
                                   "n_components", "n_components_scored",
                                   "owner_counts", "dissolved_contacts",
                                   "attempted_contacts")}
                for r in trail]
            # NAMED for what it is: the fidelity of the proxy on the BASELINE
            # ownership. The proxy never priced this row, but it did choose
            # which states were visited, so a refutation bounds THIS search.
            best["proxy_fidelity"] = proxy_fidelity
            best["trail_steps_scored"] = len(trail)
            best["trail_states_feasible"] = len(feasible)
            rows["greedy"] = best
            print(f"[greedy] best of {len(feasible)} feasible / {len(trail)} "
                  f"production-scored trail states: step "
                  f"{best['trail_step']}, padded "
                  f"{best['padded_bytes']/1e6:.1f} MB vs baseline "
                  f"{base['padded_bytes']/1e6:.1f} MB", flush=True)
            verdict(rows, n_dev_owner=target, n_cells=int(mesh.nCells),
                    lever="bytes",
                    config=(args.subdivision, args.lloyd, args.n_dev,
                            base["halo_depth"]))
            if args.out:
                _write(args, rows)
            return 0

        owner1, hist = greedy_degree_pass(
            adj, owner0, target, base["halo_depth"],
            max_iters=args.greedy_iters, max_attempts=args.greedy_attempts)
        print(f"[greedy] proxy max-degree history: {hist}", flush=True)
        if np.array_equal(owner1, owner0):
            rows["greedy"] = {"status": "no improving move found",
                              "proxy_history": hist,
                              "proxy_fidelity": proxy_fidelity}
        else:
            comps = domain_components(adj, owner1, target)
            prepared = reorder_with_owner(mesh, target, owner1)
            row = score_prepared(prepared, args.n_dev,
                                 n_real_cells=int(mesh.nCells))
            row["owner_counts"] = owner_stats(
                owner1, target, padded_cells(int(mesh.nCells), target))
            row["owner_counts_at_target"] = row["owner_counts"]
            row["reorder_target"] = target
            row["n_components"] = comps
            row["proxy_history"] = hist
            # The CANDIDATE's own proxy fidelity, not the baseline's: the
            # search ran on a different ownership, so the baseline miss count
            # does not describe the graph the verdict is about.
            cand_proxy = proxy_pair_set(
                contact_matrix(adj, owner1, target, base["halo_depth"]))
            cand_prod = {tuple(p) for p in row["comm_pairs"]}
            row["proxy_fidelity"] = {
                "baseline": proxy_fidelity,
                "n_proxy_pairs": len(cand_proxy),
                "n_prod_pairs": len(cand_prod),
                "n_missed": len(cand_prod - cand_proxy),
                "n_spurious": len(cand_proxy - cand_prod)}
            rows["greedy"] = row
            print(f"[greedy] rounds={row['n_rounds']} "
                  f"max_degree={row['max_degree']} "
                  f"padded={row['padded_bytes']/1e6:.1f} MB "
                  f"inflation={row['inflation']:.3f} "
                  f"counts={row['owner_counts']['min']}-"
                  f"{row['owner_counts']['max']} "
                  f"components={comps} (want {target})", flush=True)
        verdict(rows, n_dev_owner=target, n_cells=int(mesh.nCells),
                lever=args.objective)

    if args.out:
        _write(args, rows)
    return 0


def _write(args, rows):
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    # Stamp the byte model next to the bytes: `actual_bytes`/`padded_bytes`
    # are DRY-STATE and a reader months later has no other way to know.
    meta = {"byte_model": "dry-state", "nlev": NLEV, "n_tracers": 0,
            "cell_bytes_per_entity": CELL_BYTES,
            "edge_bytes_per_entity": EDGE_BYTES,
            "note": "production also packs nlev*n_tracers cell values "
                    "(sharded_dynamics.py q_flat), which do not cancel in a "
                    "ratio when the cell/edge padding mix changes"}
    with open(args.out, "w") as f:
        json.dump({"args": vars(args), "byte_model": meta, "rows": rows},
                  f, indent=2,
                  default=lambda o: (o.item() if hasattr(o, "item")
                                     else str(o)))
    print(f"wrote {args.out}")


# (subdivision, lloyd_iterations, n_devices, halo_depth). LLOYD IS PART OF
# THE IDENTITY: it changes the mesh, so `--subdivision 9 --n-dev 64 --lloyd 1`
# is a different measurement and must not print an unqualified CONFIRMED.
PREREG_BYTE_CONFIG = (9, 0, 64, 3)


def select_best_feasible(trail, base, lo, hi):
    """Lowest-padded-bytes trail row that SATISFIES :func:`byte_constraints`.

    Falls back to the whole trail when nothing is feasible, so the verdict
    still gets a row and refutes it on the constraint it violates.  Picking
    the global minimum first would let one infeasible 25 % row hide a
    feasible 20 % row further along.
    """
    feasible = [r for r in trail
                if all(byte_constraints(r, base, lo, hi).values())]
    return min(feasible or trail, key=lambda r: r["padded_bytes"]), feasible


def byte_constraints(cand, base, lo, hi):
    """The byte gate's three hard constraints, as booleans.

    Shared by the trail's BEST-ROW SELECTION and by :func:`verdict` so the row
    the gate judges is the best row that actually SATISFIES the constraints.
    Picking the lowest-byte row first and refuting it afterwards would let one
    infeasible 25 % row hide a feasible 20 % row further along the trail.
    """
    return {
        "counts_ok": (cand["owner_counts"]["min"] >= lo
                      and cand["owner_counts"]["max"] <= hi),
        "rounds_ok": cand["n_rounds"] <= base["n_rounds"],
        # Pre-registration says n_components_scored (the blocks production
        # runs) and nothing about the INTENDED ownership -- sfc's own
        # intended ownership is 73 domains at s9@64, so the degree gate's
        # "== n_dev" clause would refute the baseline against itself.
        "conn_scored_ok": (cand["n_components_scored"]
                           <= base["n_components_scored"]),
    }


def verdict(rows, *, n_dev_owner, n_cells, lever="degree", config=None):
    """Apply the pre-registered gate for *lever*. Reports; never builds.

    *config* is the ``(subdivision, lloyd_iterations, n_devices,
    halo_depth)`` actually measured.  The byte thresholds were pre-registered
    for :data:`PREREG_BYTE_CONFIG` only, so any other configuration gets an
    ``(ADVISORY)`` suffix on WHATEVER the outcome is -- an off-registration
    REFUTED is exactly as unquotable as an off-registration CONFIRMED.
    """
    if lever not in LEVERS:
        raise ValueError(f"unknown lever {lever!r}; "
                         f"choose from {sorted(LEVERS)}")
    base, cand = rows.get("sfc"), rows.get("greedy")
    if base is None or cand is None:
        raise ValueError("verdict needs both an 'sfc' baseline and a "
                         "'greedy' candidate row")
    base_missing = [k for k in ("n_rounds", "padded_bytes", "owner_counts",
                                "n_components_scored") if k not in base]
    if base_missing:
        raise ValueError(f"baseline row missing {base_missing}: every gate "
                         f"number is a RATIO against it, so a partial "
                         f"baseline cannot produce a verdict")
    if "n_rounds" not in cand:
        fid = cand.get("proxy_fidelity", {})
        if lever == "bytes":
            print("\nGATE (bytes): REFUTED — the prototype produced no "
                  f"candidate ({cand.get('status')}). Baseline padded "
                  f"{base['padded_bytes']/1e6:.1f} MB over "
                  f"{base['n_rounds']} rounds stands; this refutes the "
                  "PROTOTYPE, not every conceivable partitioner. Search "
                  f"proxy missed {fid.get('n_missed', '?')} of "
                  f"{fid.get('n_prod_pairs', '?')} production comm pairs.")
            return
        print("\nGATE: REFUTED — the prototype produced no candidate "
              f"({cand.get('status')}). Recolouring headroom on the baseline "
              f"graph is {base['max_degree'] - base['degree_floor_rounds']} "
              f"rounds (max degree {base['max_degree']} vs ceil mean "
              f"{base['degree_floor_rounds']}); this refutes the PROTOTYPE, "
              f"not every conceivable partitioner. Search proxy missed "
              f"{fid.get('n_missed', '?')} of {fid.get('n_prod_pairs', '?')} "
              f"production comm pairs.")
        return
    missing = [k for k in ("padded_bytes", "owner_counts", "n_components",
                           "n_components_scored") if k not in cand]
    if missing:
        raise ValueError(f"greedy row scored but missing {missing}: refusing "
                         f"to issue a verdict on a partial row")
    lo, hi = balance_bounds(n_cells, n_dev_owner)
    cut = 1.0 - cand["n_rounds"] / base["n_rounds"]
    pad = cand["padded_bytes"] / base["padded_bytes"] - 1.0
    con = byte_constraints(cand, base, lo, hi)
    counts_ok = con["counts_ok"]
    # Gate on the SCORED blocks (what production runs), and require the
    # candidate to be no worse than the baseline -- sfc's own equal-block
    # re-cut is not guaranteed perfectly connected either, so demanding
    # n_dev exactly would hold the prototype to a bar the shipped
    # partitioner does not clear.  The DEGREE gate additionally demands the
    # INTENDED ownership be exactly n_dev domains; the byte gate does not,
    # because its pre-registration says "n_components_scored no worse than
    # the sfc baseline's" and nothing more -- and sfc's own intended
    # ownership is 73 domains at s9@64, so importing that clause would
    # refute the baseline against itself.
    conn_scored_ok = con["conn_scored_ok"]
    conn_ok = conn_scored_ok and cand["n_components"] == n_dev_owner
    fid = cand.get("proxy_fidelity", {})
    if lever != "bytes":
        print(f"\nGATE: round cut {cut*100:.1f}% "
              f"({base['n_rounds']} -> {cand['n_rounds']}), padded bytes "
              f"{pad*100:+.1f}%, counts {cand['owner_counts']['min']}-"
              f"{cand['owner_counts']['max']} (allowed {lo}-{hi}), "
              f"domains connected: {conn_ok} (scored components "
              f"{cand['n_components_scored']} vs baseline "
              f"{base['n_components_scored']}, want {n_dev_owner}; intended "
              f"ownership {cand['n_components']}), block_mismatch "
              f"{cand['owner_counts'].get('block_mismatch', float('nan'))*100:.2f}%"
              f" (baseline "
              f"{base['owner_counts'].get('block_mismatch', float('nan'))*100:.2f}%)")
    if fid.get("n_missed"):
        # A search that cannot see part of the production graph can only under-
        # report what is reachable, so a NEGATIVE result carries this caveat
        # with it rather than standing as an unqualified refutation.
        print(f"GATE CAVEAT: the search proxy missed {fid['n_missed']} of "
              f"{fid['n_prod_pairs']} production comm pairs (edge-halo "
              f"contacts it does not model), so a negative result bounds THIS "
              f"search, not the lever.")
    if lever == "bytes":
        # BYTE GATE (docstring): >=20% padded-byte cut, rounds not increased,
        # same balance/connectivity constraints. The win and the constraint
        # swap roles vs the degree gate -- there rounds were the win and
        # bytes the constraint.
        # 1 - c/b, not -pad: -(0.0) prints as "-0.0%".
        byte_cut = 1.0 - cand["padded_bytes"] / base["padded_bytes"]
        rounds_ok = con["rounds_ok"]
        off_prereg = config is not None and tuple(config) != PREREG_BYTE_CONFIG
        print(f"\nGATE (bytes, DRY-STATE byte model — no tracers, see the "
              f"module docstring): padded-byte cut {byte_cut*100:.1f}% "
              f"({base['padded_bytes']/1e6:.1f} -> "
              f"{cand['padded_bytes']/1e6:.1f} MB dry), rounds "
              f"{base['n_rounds']} -> {cand['n_rounds']}, counts "
              f"{cand['owner_counts']['min']}-{cand['owner_counts']['max']} "
              f"(allowed {lo}-{hi}), scored components "
              f"{cand['n_components_scored']} vs baseline "
              f"{base['n_components_scored']}")
        advisory = config is None or off_prereg
        if config is None:
            print("GATE: ADVISORY — configuration not supplied, so it cannot "
                  "be checked against the pre-registered "
                  f"{PREREG_BYTE_CONFIG}")
        elif off_prereg:
            print(f"GATE: ADVISORY — measured at {tuple(config)}, and the "
                  f"byte thresholds were pre-registered for "
                  f"{PREREG_BYTE_CONFIG} only")
        # The suffix goes on EVERY outcome, not just CONFIRMED: an
        # off-preregistration REFUTED is just as unquotable as an
        # off-preregistration CONFIRMED.
        sfx = " (ADVISORY)" if advisory else ""
        # INTEGER ratio tests: 800/1000 is 0.19999999999999996 in float, so a
        # float ">= 0.20" silently downgrades an exact 20% cut to
        # INTERMEDIATE and an exact 10% cut to REFUTED.
        c_b, b_b = int(cand["padded_bytes"]), int(base["padded_bytes"])
        confirm = 5 * c_b <= 4 * b_b            # cut >= 20%
        refute = 10 * c_b > 9 * b_b             # cut < 10%
        if not counts_ok or not conn_scored_ok:
            print("GATE: REFUTED — balance/connectivity constraint "
                  f"violated{sfx}")
        elif not rounds_ok:
            print(f"GATE: REFUTED — round count increased{sfx}")
        elif confirm:
            print(f"GATE: CONFIRMED{sfx}")
        elif refute:
            print(f"GATE: REFUTED{sfx}")
        else:
            print(f"GATE: INTERMEDIATE — report, do not build{sfx}")
    elif not counts_ok or not conn_ok:
        # Docstring: a win bought by breaking the balance constraint (or by
        # shattering domains, which is the same cheat one level down) is a
        # refutation however large the round cut is.
        print("GATE: REFUTED — balance/connectivity constraint violated")
    # Integer ratios here too: 1 - 9/12 is 0.25 exactly but 1 - 1100/1000 and
    # 1 - 18/20 are not, so a float comparison can reject an exactly-at-
    # threshold case (same defect the byte gate had).
    elif 10 * int(cand["padded_bytes"]) > 11 * int(base["padded_bytes"]):
        print("GATE: REFUTED — padded bytes inflated more than 10%")
    elif 4 * int(cand["n_rounds"]) <= 3 * int(base["n_rounds"]):
        print("GATE: CONFIRMED")
    elif 10 * int(cand["n_rounds"]) > 9 * int(base["n_rounds"]):
        print("GATE: REFUTED")
    else:
        print("GATE: INTERMEDIATE — report, do not build")


if __name__ == "__main__":
    sys.exit(main())
