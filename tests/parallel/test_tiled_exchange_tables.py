"""Static-table tests for the sub-face tiled SPMD exchange (P1a).

Validates ``_build_tile_connectivity`` / ``_build_tiled_tables`` in
``legoesm.parallel.cubesphere_exchange`` — the pure-numpy schedule layer
under the 6*kt^2-device tiled ppermute halo (task: break the 6-device
cap of the face-only SPMD path).  No MPI, no shard_map: every property
here must hold BEFORE the kernel is trusted.

Independent oracle: a serial (6, n, n) reference array whose value
encodes (face, i, j); for every tile edge the strip THE TABLES say a
device receives is compared against the strip the serial full-face
CONNECTIVITY pad would place in that tile's halo.
"""
from __future__ import annotations

import numpy as np
import pytest

from legoesm.grids.halo import CONNECTIVITY, WEST, EAST, SOUTH, NORTH
from legoesm.parallel.cubesphere_exchange import (
    _build_tile_connectivity,
    _build_tiled_tables,
    _tile_id,
)

KTS = (2, 3, 4)


@pytest.mark.parametrize("kt", KTS)
def test_tile_connectivity_involutive_and_complete(kt):
    conn = _build_tile_connectivity(kt)  # raises on involution break
    n_dev = 6 * kt * kt
    assert set(conn.keys()) == set(range(n_dev))
    for t, ents in conn.items():
        assert len(ents) == 4
        for e in range(4):
            b, e2, rv, xf = ents[e]
            assert 0 <= b < n_dev and b != t
            # cross_face iff the edge sits on the face border.
            f, r = divmod(t, kt * kt)
            ti, tj = divmod(r, kt)
            on_border = (
                (e == WEST and ti == 0) or (e == EAST and ti == kt - 1)
                or (e == SOUTH and tj == 0) or (e == NORTH and tj == kt - 1)
            )
            assert xf == on_border
            if not xf:
                assert rv is False


@pytest.mark.parametrize("kt", KTS)
def test_interior_edges_match_serial_adjacency(kt):
    """Interior tile edges must point at the serially adjacent tile."""
    conn = _build_tile_connectivity(kt)
    for f in range(6):
        for ti in range(kt):
            for tj in range(kt):
                t = _tile_id(f, ti, tj, kt)
                if ti > 0:
                    assert conn[t][WEST][:2] == (
                        _tile_id(f, ti - 1, tj, kt), EAST)
                if tj < kt - 1:
                    assert conn[t][NORTH][:2] == (
                        _tile_id(f, ti, tj + 1, kt), SOUTH)


def _strip_indices(edge, n):
    """(i, j) index arrays of a face's boundary strip for ``edge``,
    ordered along the strip coordinate (j for W/E, i for S/N)."""
    r = np.arange(n)
    if edge == WEST:
        return np.zeros(n, dtype=int), r
    if edge == EAST:
        return np.full(n, n - 1, dtype=int), r
    if edge == SOUTH:
        return r, np.zeros(n, dtype=int)
    return r, np.full(n, n - 1, dtype=int)


@pytest.mark.parametrize("kt", KTS)
def test_cross_face_tile_strips_match_face_connectivity(kt):
    """Tile-level cross-face mapping == face-level CONNECTIVITY.

    For each face-border tile edge, the (sender tile, sender edge,
    reversal) from the tile tables must deliver EXACTLY the cells the
    face-level rule selects for that segment of the face edge:
    receiver face f edge e segment [p*n_loc, (p+1)*n_loc) pulls the
    neighbour face's edge strip (reversed when CONNECTIVITY says so)
    restricted to the same segment positions.
    """
    n_loc = 5
    n = kt * n_loc
    conn = _build_tile_connectivity(kt)

    # Serial oracle: value encodes (face, i, j) uniquely.
    ref = (
        np.arange(6)[:, None, None] * (n * n)
        + np.arange(n)[None, :, None] * n
        + np.arange(n)[None, None, :]
    )

    for f in range(6):
        for e in range(4):
            nf, ne, rv = CONNECTIVITY[f][e]
            si, sj = _strip_indices(ne, n)
            donor_full = ref[nf, si, sj]          # along neighbour strip
            if rv:
                donor_full = donor_full[::-1]      # receiver-side flip
            for p in range(kt):                    # receiver segment
                seg = donor_full[p * n_loc:(p + 1) * n_loc]

                # Receiver tile holding segment p of (f, e):
                if e in (WEST, EAST):
                    ti = 0 if e == WEST else kt - 1
                    tj = p
                else:
                    ti = p
                    tj = 0 if e == SOUTH else kt - 1
                t = _tile_id(f, ti, tj, kt)
                b, e2, rv_t, xf = conn[t][e]
                assert xf, "face-border tile edge must be cross_face"
                assert rv_t == rv, (
                    f"tile reversal flag != face flag at f{f} e{e} p{p}"
                )

                # Sender tile's OWN edge-e2 strip (tile-local, unrev).
                bf, br = divmod(b, kt * kt)
                bti, btj = divmod(br, kt)
                assert bf == nf
                bi, bj = _strip_indices(e2, n_loc)
                tile_vals = ref[
                    bf,
                    bti * n_loc + bi,
                    btj * n_loc + bj,
                ]
                if rv_t:
                    tile_vals = tile_vals[::-1]
                np.testing.assert_array_equal(
                    tile_vals, seg,
                    err_msg=(
                        f"kt={kt} f{f} e{e} segment {p}: tile tables "
                        f"deliver wrong cells (sender tile {b} edge {e2})"
                    ),
                )


@pytest.mark.parametrize("kt", KTS)
def test_tiled_tables_schedule_invariants(kt):
    tabs = _build_tiled_tables(kt)  # build-time asserts: coverage etc.
    n_dev = 6 * kt * kt
    n_rounds = len(tabs.perms)
    # Degree bound: every device sends 4 strips and receives 4 — the
    # König bound says 4 rounds suffice; allow exactly that.
    assert n_rounds == 4, f"expected 4 ppermute rounds, got {n_rounds}"
    # Each round is a partial permutation.
    for rnd in tabs.perms:
        srcs = [s for s, _ in rnd]
        dsts = [d for _, d in rnd]
        assert len(set(srcs)) == len(srcs)
        assert len(set(dsts)) == len(dsts)
    # recv_tgt sentinel only where the device is not a round's dst.
    for r, rnd in enumerate(tabs.perms):
        dsts = {d for _, d in rnd}
        for d in range(n_dev):
            if d in dsts:
                assert 0 <= tabs.recv_tgt[r, d] < 4
            else:
                assert tabs.recv_tgt[r, d] == 4
    # offs_pos within range; cross_face consistent with connectivity.
    assert tabs.offs_pos.min() >= 0 and tabs.offs_pos.max() < kt
    conn = _build_tile_connectivity(kt)
    for t in range(n_dev):
        for e in range(4):
            assert tabs.cross_face[t, e] == int(conn[t][e][3])
            assert tabs.rev[t, e] == int(conn[t][e][2])


def test_kt1_rejected():
    with pytest.raises(ValueError, match="kt >= 2"):
        _build_tiled_tables(1)
