"""Explicit cubed-sphere halo exchange for multi-GPU SPMD.

Replaces the implicit cross-shard reads that ``pad_halo`` generates
under face-axis sharding with **explicit collective operations**
inside ``shard_map``.  Three collective kernels are provided:

* **ppermute multiface** (DEFAULT for every face-sharded device count
  ``n_devices ∈ {1, 2, 3, 6}``, halo 1 and 2): each shard owns
  ``k = 6 / n_devices`` contiguous faces.  Face edges *within* a shard
  are filled shard-locally (same rotation + corner conventions as the
  serial path); edges *crossing* shard boundaries ride a static
  device-pair schedule of ``jax.lax.ppermute`` rounds, one
  concatenated strip buffer per (src, dst) device pair per round.
  No value is ever materialized with full face extent on any device.

* **ppermute one-face** (halo=1, exactly 6 devices): the original
  validated 4-round perfect-matching kernel — kept as the lowest-risk
  path for the 1-face-per-device layout.

* **all_gather** (EXPLICIT DIAGNOSTIC OPT-IN ONLY — ``force_allgather``
  kwarg on :func:`activate_spmd_halo_backend` or env
  ``LEGOESM_SPMD_FORCE_ALLGATHER=1``): each device gathers the
  perimeter strips of all 6 faces.  The HLO probe (job 8456476) proved
  that this variant's true cost is not bandwidth but **full compute
  replication** (per-device/single-device FLOPs ratio 1.00 at 2
  devices, all-gather results with full 6-face extent), so it must
  never be auto-selected.  Use :func:`assert_no_fullcube_allgather`
  as the mechanical tripwire against this bug class returning.

Also supports:
- **Packed multi-field exchange**: stack several fields → one
  collective → unstack.  Reduces collective count from O(fields)
  to O(1).
- **Vector (u, v) exchange**: rotate to geographic frame, exchange
  both components in a single collective, rotate back.
"""

from __future__ import annotations

import logging
import os
import re
from collections import Counter
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from legoesm.grids.halo import (
    CONNECTIVITY, WEST, EAST, SOUTH, NORTH, interp_strip,
)

logger = logging.getLogger("legoesm.parallel.cubesphere_exchange")

try:
    from jax import shard_map  # JAX >= 0.8 exposes it at top level
except ImportError:
    from jax.experimental.shard_map import shard_map  # older JAX fallback

# ---------------------------------------------------------------------------
# Static connectivity tables (built once at import).
#
# iter-93: stored as ``np.array`` at module-top to avoid eager
# JAX device dispatch on ``import legoesm.parallel.cubesphere_exchange``.
# See ``src/legoesm/grids/vertical.py`` for the same pattern + rationale
# (Metal default_memory_space crash before fallback applies). Callers
# convert via ``jnp.asarray(...)`` inside the per-exchange closures.
# ---------------------------------------------------------------------------

_NBR_FACES = np.array([
    [3, 1, 5, 4],   # face 0: W←3, E←1, S←5, N←4
    [0, 2, 5, 4],   # face 1
    [1, 3, 5, 4],   # face 2
    [2, 0, 5, 4],   # face 3
    [3, 1, 0, 2],   # face 4
    [3, 1, 2, 0],   # face 5
], dtype=np.int32)

_NBR_EDGES = np.array([
    [EAST, WEST, NORTH, SOUTH],    # face 0
    [EAST, WEST, EAST,  EAST],     # face 1
    [EAST, WEST, SOUTH, NORTH],    # face 2
    [EAST, WEST, WEST,  WEST],     # face 3
    [NORTH, NORTH, NORTH, NORTH],  # face 4
    [SOUTH, SOUTH, SOUTH, SOUTH],  # face 5
], dtype=np.int32)

_IS_REVERSED = np.array([
    [0, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 1, 1],
    [0, 0, 0, 1],
    [1, 0, 0, 1],
    [0, 1, 1, 0],
], dtype=np.int32)


# ---------------------------------------------------------------------------
# ppermute round tables
# ---------------------------------------------------------------------------
# The 24 face-to-face transfers (6 faces × 4 edges) are partitioned
# into 4 perfect matchings so that each round is one ppermute call
# where every device sends and receives exactly once.

def _build_ppermute_tables():
    """Compute the static ppermute schedule from CONNECTIVITY.

    Returns (PERMS, SEND_EDGES, RECV_EDGES, RECV_REVS).
    PERMS is a list of 4 tuples of (src, dst) pairs.
    SEND/RECV/REV are shape (4, 6) JAX arrays.
    """
    # Build send table: sends[src][dst] = (src_edge, dst_halo_edge, reversed)
    sends = {}
    for f in range(6):
        sends[f] = {}
        for e in range(4):
            nbr_f, _nbr_e, _rev = CONNECTIVITY[f][e]
            # Reverse lookup: which edge of nbr_f connects back to f?
            for ne in range(4):
                nf2, ne2, rev2 = CONNECTIVITY[nbr_f][ne]
                if nf2 == f and ne2 == e:
                    sends[f][nbr_f] = (e, ne, rev2)
                    break

    # Verified perfect matchings covering all 24 directed edges.
    # Each round is a permutation where every face sends and receives once.
    _ROUNDS = [
        ((0, 4), (1, 5), (2, 3), (3, 2), (4, 0), (5, 1)),
        ((0, 5), (1, 4), (2, 1), (3, 0), (4, 3), (5, 2)),
        ((0, 3), (1, 2), (2, 4), (3, 5), (4, 1), (5, 0)),
        ((0, 1), (1, 0), (2, 5), (3, 4), (4, 2), (5, 3)),
    ]

    rounds_perm = []
    rounds_send = []
    rounds_recv = []
    rounds_rev = []
    for rd in _ROUNDS:
        send_e = [0] * 6
        recv_e = [0] * 6
        recv_r = [0] * 6
        for s, d in rd:
            se, re, rv = sends[s][d]
            send_e[s] = se
            recv_e[d] = re
            recv_r[d] = int(rv)
        rounds_perm.append(rd)
        rounds_send.append(tuple(send_e))
        rounds_recv.append(tuple(recv_e))
        rounds_rev.append(tuple(recv_r))

    # iter-94d: return np.array (not jnp.array) so the module-top
    # call ``_PPERMUTE_* = _build_ppermute_tables()`` does NOT
    # eagerly dispatch to the JAX default platform (METAL on
    # macOS, which raises UNIMPLEMENTED). Same pattern as the
    # iter-93b refactor of ``_NBR_FACES``/``_NBR_EDGES``/
    # ``_IS_REVERSED``. Callers convert via ``jnp.asarray`` inside
    # the ``_make_exchange_ppermute._exchange`` closure.
    return (
        rounds_perm,
        np.array(rounds_send, dtype=np.int32),   # (4, 6)
        np.array(rounds_recv, dtype=np.int32),
        np.array(rounds_rev, dtype=np.int32),
    )


_PPERMUTE_PERMS, _PPERMUTE_SEND, _PPERMUTE_RECV, _PPERMUTE_REV = (
    _build_ppermute_tables()
)


# ===================================================================
# Helper: build padded output from strips (shared by both backends)
# ===================================================================

def _fill_halo_and_corners(padded, strips, n_spatial):
    """Place 4 edge strips into the halo region and fill corners.

    Parameters
    ----------
    padded : (n+2, n+2, ...) — already contains interior at [1:-1, 1:-1].
    strips : list of 4 arrays, each (n, ...), in W/E/S/N order.

    Returns
    -------
    padded with halos and corners filled.
    """
    padded = padded.at[0, 1:-1].set(strips[WEST])
    padded = padded.at[-1, 1:-1].set(strips[EAST])
    padded = padded.at[1:-1, 0].set(strips[SOUTH])
    padded = padded.at[1:-1, -1].set(strips[NORTH])
    # Corners: average adjacent edge-halo values.
    padded = padded.at[0, 0].set(0.5 * (padded[0, 1] + padded[1, 0]))
    padded = padded.at[0, -1].set(0.5 * (padded[0, -2] + padded[1, -1]))
    padded = padded.at[-1, 0].set(0.5 * (padded[-1, 1] + padded[-2, 0]))
    padded = padded.at[-1, -1].set(0.5 * (padded[-1, -2] + padded[-2, -1]))
    return padded


def _fill_halo_and_corners_h2_local(padded, strips, n):
    """Place 4 width-2 edge strips into the halo region and fill 2x2 corners.

    Single-face SPMD analogue of :func:`legoesm.grids.halo.fill_corners_h2`.

    Parameters
    ----------
    padded : (n+4, n+4, ...) — interior already at [2:-2, 2:-2].
    strips : list of 4 arrays, each (2, n, ...), in W/E/S/N order.
        Within each strip:
        - index 0 = depth-0 = the cell of the neighbour that touches the
          local face's interior boundary; placed at the inner halo
          position (closest to interior).
        - index 1 = depth-1 = one cell deeper into the neighbour;
          placed at the outer halo position.
    n : int
        Per-face interior resolution.

    Returns
    -------
    padded with both halo layers and 2x2 corners filled.
    """
    # WEST: rows [0:2], cols [2:-2]; depth-0 → row 1, depth-1 → row 0
    padded = padded.at[1, 2:-2].set(strips[WEST][0])
    padded = padded.at[0, 2:-2].set(strips[WEST][1])
    # EAST: rows [n+2:n+4], cols [2:-2]; depth-0 → row n+2, depth-1 → row n+3
    padded = padded.at[n + 2, 2:-2].set(strips[EAST][0])
    padded = padded.at[n + 3, 2:-2].set(strips[EAST][1])
    # SOUTH: rows [2:-2], cols [0:2]; depth-0 → col 1, depth-1 → col 0
    padded = padded.at[2:-2, 1].set(strips[SOUTH][0])
    padded = padded.at[2:-2, 0].set(strips[SOUTH][1])
    # NORTH: rows [2:-2], cols [n+2:n+4]; depth-0 → col n+2, depth-1 → col n+3
    padded = padded.at[2:-2, n + 2].set(strips[NORTH][0])
    padded = padded.at[2:-2, n + 3].set(strips[NORTH][1])
    # Corners: 4 corners × 4 cells per corner.  Inner corner first, then
    # propagate outward; matches ``halo.fill_corners_h2`` semantics.
    # SW corner (rows 0..1, cols 0..1)
    padded = padded.at[1, 1].set(0.5 * (padded[1, 2] + padded[2, 1]))
    padded = padded.at[0, 1].set(0.5 * (padded[0, 2] + padded[1, 1]))
    padded = padded.at[1, 0].set(0.5 * (padded[2, 0] + padded[1, 1]))
    padded = padded.at[0, 0].set(0.5 * (padded[0, 1] + padded[1, 0]))
    # SE corner (rows n+2..n+3, cols 0..1)
    padded = padded.at[-2, 1].set(0.5 * (padded[-2, 2] + padded[-3, 1]))
    padded = padded.at[-1, 1].set(0.5 * (padded[-1, 2] + padded[-2, 1]))
    padded = padded.at[-2, 0].set(0.5 * (padded[-3, 0] + padded[-2, 1]))
    padded = padded.at[-1, 0].set(0.5 * (padded[-1, 1] + padded[-2, 0]))
    # NW corner (rows 0..1, cols n+2..n+3)
    padded = padded.at[1, -2].set(0.5 * (padded[1, -3] + padded[2, -2]))
    padded = padded.at[0, -2].set(0.5 * (padded[0, -3] + padded[1, -2]))
    padded = padded.at[1, -1].set(0.5 * (padded[2, -1] + padded[1, -2]))
    padded = padded.at[0, -1].set(0.5 * (padded[0, -2] + padded[1, -1]))
    # NE corner (rows n+2..n+3, cols n+2..n+3)
    padded = padded.at[-2, -2].set(0.5 * (padded[-2, -3] + padded[-3, -2]))
    padded = padded.at[-1, -2].set(0.5 * (padded[-1, -3] + padded[-2, -2]))
    padded = padded.at[-2, -1].set(0.5 * (padded[-3, -1] + padded[-2, -2]))
    padded = padded.at[-1, -1].set(0.5 * (padded[-1, -2] + padded[-2, -1]))
    return padded


# ===================================================================
# Backend A: all_gather  (EXPLICIT DIAGNOSTIC OPT-IN ONLY)
# ===================================================================
# Reachable only with the module ppermute flag off (force_allgather /
# LEGOESM_SPMD_FORCE_ALLGATHER=1).  The strips all_gather regains full
# 6-face extent on every device and GSPMD then replicates ALL
# downstream compute (HLO probe job 8456476: per-device FLOPs ratio
# 1.00 at 2 devices) — production routing uses the ppermute kernels.

def _make_exchange_allgather(mesh, ndim, with_offsets=False):
    """Build a shard_map exchange using all_gather (diagnostic only).

    When ``with_offsets`` is True the kernel takes a second
    ``interp_offsets`` argument shaped ``(6, 4, n)`` (replicated) and
    applies the per-edge 3-point Lagrange correction (via
    :func:`legoesm.grids.halo.interp_strip`) to each gathered strip.
    This restores numerical equivalence with the local ``pad_halo_local``
    fill, which is the reference path on single device.  Without
    offsets the kernel skips the interp — same behaviour as before
    (and a small numerical drift vs single-device, accepted as the
    SPMD trade-off).
    """
    P = jax.sharding.PartitionSpec
    in_sp_data = P("face", *((None,) * (ndim - 1)))
    out_sp = P("face", *((None,) * (ndim - 1)))

    if with_offsets:
        in_sp = (in_sp_data, P())
    else:
        in_sp = in_sp_data

    # iter-94g: build jnp tables in outer factory body (not inside
    # shard_map body) so they're closure-captured constants. See
    # detailed rationale in `_make_exchange_ppermute` below.
    nbr_faces_j = jnp.asarray(_NBR_FACES)
    nbr_edges_j = jnp.asarray(_NBR_EDGES)
    is_reversed_j = jnp.asarray(_IS_REVERSED)

    @partial(shard_map, mesh=mesh, in_specs=in_sp, out_specs=out_sp,
             check_vma=False)
    def _exchange(*args):
        if with_offsets:
            local_shard, offsets = args
        else:
            (local_shard,) = args
            offsets = None

        n_faces_per_shard = local_shard.shape[0]
        n = local_shard.shape[1]
        # ``mesh.shape["face"]`` is the device count along the face
        # axis (1, 2, 3, or 6 for cubed-sphere) — *not* the global
        # face count (always 6).  Iter-49 fix.
        n_faces = 6

        # Extract 4 perimeter strips from EACH face this shard owns.
        # Iter-49: previously the kernel hard-coded ``my_face = local_shard[0]``
        # and assumed exactly 1 face per shard.  That made the SPMD
        # path safe only at 6 devices; at 2 or 3 devices the kernel
        # silently dropped 1 or 2 faces.  Generalising via per-face
        # loop (n_faces_per_shard ≤ 6) lets the SPMD halo activate at
        # any divisor of 6.  The ppermute multi-face refit has since
        # landed: this all_gather kernel is no longer the default (it
        # replicates compute, see the Backend A header) and survives
        # only as the explicit diagnostic opt-in.
        if ndim == 3:
            my_strips = jnp.stack([
                jnp.stack([
                    local_shard[i, 0, :], local_shard[i, -1, :],
                    local_shard[i, :, 0], local_shard[i, :, -1],
                ], axis=0)
                for i in range(n_faces_per_shard)
            ], axis=0)  # (n_faces_per_shard, 4, n)
        else:
            my_strips = jnp.stack([
                jnp.stack([
                    local_shard[i, 0, :, :], local_shard[i, -1, :, :],
                    local_shard[i, :, 0, :], local_shard[i, :, -1, :],
                ], axis=0)
                for i in range(n_faces_per_shard)
            ], axis=0)  # (n_faces_per_shard, 4, n, C)

        # All-gather along the face axis: contributes (4, n[, C]) per
        # face per device → after gather, shape (n_faces, 4, n[, C]).
        # Reshape from the tiled layout (n_faces_per_shard merges with
        # device axis after concatenation).
        all_strips = jax.lax.all_gather(my_strips, "face", tiled=True)
        # all_strips after tiled gather: (n_faces, 4, n[, C]).
        if ndim == 3:
            all_strips = all_strips.reshape(n_faces, 4, n)
        else:
            all_strips = all_strips.reshape(
                n_faces, 4, n, local_shard.shape[-1],
            )

        my_idx = jax.lax.axis_index("face")
        if with_offsets:
            from legoesm.grids.halo import interp_strip

        padded_faces = []
        for i in range(n_faces_per_shard):
            global_face = my_idx * n_faces_per_shard + i
            nbr_f = nbr_faces_j[global_face]
            nbr_e = nbr_edges_j[global_face]
            rev = is_reversed_j[global_face]
            face = local_shard[i]
            if ndim == 3:
                padded = jnp.pad(face, ((1, 1), (1, 1)))
            else:
                padded = jnp.pad(face, ((1, 1), (1, 1), (0, 0)))
            halo_strips = []
            for e in range(4):
                strip = all_strips[nbr_f[e], nbr_e[e]]
                strip = jnp.where(rev[e], strip[::-1], strip)
                if with_offsets:
                    strip = interp_strip(strip, offsets[global_face, e])
                halo_strips.append(strip)
            padded = _fill_halo_and_corners(padded, halo_strips, n)
            padded_faces.append(padded)

        return jnp.stack(padded_faces, axis=0)

    return _exchange


# ===================================================================
# Backend A2: all_gather for halo=2 (EXPLICIT DIAGNOSTIC OPT-IN ONLY —
# see the Backend A header; production halo=2 routing is the multiface
# ppermute kernel)
# ===================================================================

def _make_exchange_allgather_h2(mesh, ndim, with_offsets=False):
    """Build a shard_map exchange for halo=2 using a single all_gather of
    2-cell-wide perimeter strips (diagnostic only).

    The volume per face is ``8 * n[, * C]`` cells (4 edges × 2 deep)
    versus the previous fall-through path which used the local h2 fill
    on a face-sharded array — a pattern that triggers XLA auto-gather of
    the full ``(6, n, n[, C])`` state.  The 2-strip allgather moves
    ``n/4`` × less data per device for typical ``n``.

    When ``with_offsets`` is True the kernel takes a replicated
    ``interp_offsets`` argument shaped ``(6, 4, 2, n)`` and applies
    :func:`legoesm.grids.halo.interp_strip` per (edge, depth) to each
    gathered strip — restoring bit-exact equivalence with
    ``_pad_halo_local_h2(data, interp_offsets)``.
    """
    P = jax.sharding.PartitionSpec
    in_sp_data = P("face", *((None,) * (ndim - 1)))
    out_sp = P("face", *((None,) * (ndim - 1)))

    if with_offsets:
        in_sp = (in_sp_data, P())
    else:
        in_sp = in_sp_data

    # iter-94g: build jnp tables in outer factory body, not inside
    # shard_map body. See _make_exchange_ppermute rationale.
    nbr_faces_j = jnp.asarray(_NBR_FACES)
    nbr_edges_j = jnp.asarray(_NBR_EDGES)
    is_reversed_j = jnp.asarray(_IS_REVERSED)

    @partial(shard_map, mesh=mesh, in_specs=in_sp, out_specs=out_sp,
             check_vma=False)
    def _exchange(*args):
        if with_offsets:
            local_shard, offsets = args
        else:
            (local_shard,) = args
            offsets = None

        n_faces_per_shard = local_shard.shape[0]
        n = local_shard.shape[1]
        n_faces = 6  # iter-49: cubed-sphere always has 6 faces globally

        # Extract 4 perimeter strips of width 2 from EACH face this
        # shard owns.  Iter-49 generalises the kernel to multi-face
        # shards (n_faces_per_shard ∈ {1, 2, 3, 6} for n_devices ∈
        # {6, 3, 2, 1}).  Depth-0 (closest to the interface = my
        # outermost cell on that edge) is at index 0; depth-1 (one
        # cell inward) is at index 1.
        if ndim == 3:
            my_strips = jnp.stack([
                jnp.stack([
                    jnp.stack([local_shard[i, 0, :], local_shard[i, 1, :]], axis=0),
                    jnp.stack([local_shard[i, -1, :], local_shard[i, -2, :]], axis=0),
                    jnp.stack([local_shard[i, :, 0], local_shard[i, :, 1]], axis=0),
                    jnp.stack([local_shard[i, :, -1], local_shard[i, :, -2]], axis=0),
                ], axis=0)
                for i in range(n_faces_per_shard)
            ], axis=0)  # (n_faces_per_shard, 4, 2, n)
        else:
            my_strips = jnp.stack([
                jnp.stack([
                    jnp.stack([local_shard[i, 0, :, :], local_shard[i, 1, :, :]], axis=0),
                    jnp.stack([local_shard[i, -1, :, :], local_shard[i, -2, :, :]], axis=0),
                    jnp.stack([local_shard[i, :, 0, :], local_shard[i, :, 1, :]], axis=0),
                    jnp.stack([local_shard[i, :, -1, :], local_shard[i, :, -2, :]], axis=0),
                ], axis=0)
                for i in range(n_faces_per_shard)
            ], axis=0)  # (n_faces_per_shard, 4, 2, n, C)

        all_strips = jax.lax.all_gather(my_strips, "face", tiled=True)
        if ndim == 3:
            all_strips = all_strips.reshape(n_faces, 4, 2, n)
        else:
            all_strips = all_strips.reshape(
                n_faces, 4, 2, n, local_shard.shape[-1],
            )

        my_idx = jax.lax.axis_index("face")
        if with_offsets:
            from legoesm.grids.halo import interp_strip

        padded_faces = []
        for i in range(n_faces_per_shard):
            global_face = my_idx * n_faces_per_shard + i
            nbr_f = nbr_faces_j[global_face]
            nbr_e = nbr_edges_j[global_face]
            rev = is_reversed_j[global_face]
            face = local_shard[i]

            if ndim == 3:
                padded = jnp.pad(face, ((2, 2), (2, 2)))
            else:
                padded = jnp.pad(face, ((2, 2), (2, 2), (0, 0)))

            halo_strips = []
            for e in range(4):
                strip = all_strips[nbr_f[e], nbr_e[e]]   # (2, n[, C])
                # Reverse the spatial axis (axis 1 of (2, n[, C])) when
                # the neighbour's edge is oriented opposite to ours.
                # The depth axis is invariant under spatial reversal.
                strip = jnp.where(rev[e], strip[:, ::-1], strip)
                if with_offsets:
                    # Apply 3-point Lagrange correction per depth.  The
                    # spatial axis (n) is axis 1 of ``strip``;
                    # ``interp_strip`` interpolates along axis 0 of its
                    # input, so we slice each depth as a (n[, C]) tensor.
                    strip_d0 = interp_strip(strip[0], offsets[global_face, e, 0])
                    strip_d1 = interp_strip(strip[1], offsets[global_face, e, 1])
                    strip = jnp.stack([strip_d0, strip_d1], axis=0)
                halo_strips.append(strip)

            padded = _fill_halo_and_corners_h2_local(padded, halo_strips, n)
            padded_faces.append(padded)

        return jnp.stack(padded_faces, axis=0)

    return _exchange


# ===================================================================
# Backend B: ppermute  (bandwidth-optimal for high resolution)
# ===================================================================

def _make_exchange_ppermute(mesh, ndim, with_offsets=False):
    """Build a shard_map exchange using 4 rounds of ppermute.

    When ``with_offsets`` is True the kernel applies the per-edge
    3-point Lagrange correction to each gathered strip — same numerics
    as the all_gather ``with_offsets`` variant, restoring bit-exact
    equivalence with ``pad_halo_local(data, interp_offsets)``.  The
    offsets are indexed by the *receiving* face / edge (``my_idx``,
    ``e``), matching the local-pad convention.
    """
    P = jax.sharding.PartitionSpec
    in_sp_data = P("face", *((None,) * (ndim - 1)))
    out_sp = P("face", *((None,) * (ndim - 1)))

    if with_offsets:
        in_sp = (in_sp_data, P())
    else:
        in_sp = in_sp_data

    # iter-94g: build the jnp tables HERE (in the outer factory
    # body, outside the shard_map decorator), so they are regular
    # jax.Array constants captured by reference in the shard_map
    # closure — same semantics as the pre-iter-93b module-top
    # jnp.array. Building them INSIDE the shard_map body (iter-94d
    # initial attempt) caused JAX/XLA to behave differently under
    # multi-device emulation (10x worse numerical drift on 6-device
    # cubed-sphere SPMD tests vs. pre-iter-93b baseline). The
    # outer-scope construction runs ONCE per factory call (when
    # `_make_exchange_ppermute` is called from
    # `activate_spmd_halo_backend`), which is after
    # `ensure_metal_or_fallback()` has run — so no Metal crash.
    ppermute_send_j = jnp.asarray(_PPERMUTE_SEND)
    ppermute_recv_j = jnp.asarray(_PPERMUTE_RECV)
    ppermute_rev_j = jnp.asarray(_PPERMUTE_REV)

    @partial(shard_map, mesh=mesh, in_specs=in_sp, out_specs=out_sp,
             check_vma=False)
    def _exchange(*args):
        if with_offsets:
            local_shard, offsets = args
        else:
            (local_shard,) = args
            offsets = None

        n = local_shard.shape[1]
        my_face = local_shard[0]
        my_idx = jax.lax.axis_index("face")

        if ndim == 3:
            my_strips = jnp.stack([
                my_face[0, :], my_face[-1, :],
                my_face[:, 0], my_face[:, -1],
            ])  # (4, n)
            # Single Pad HLO op replaces alloc-zeros + scatter.
            padded = jnp.pad(my_face, ((1, 1), (1, 1)))
        else:
            my_strips = jnp.stack([
                my_face[0, :, :], my_face[-1, :, :],
                my_face[:, 0, :], my_face[:, -1, :],
            ])  # (4, n, C)
            padded = jnp.pad(my_face, ((1, 1), (1, 1), (0, 0)))
        halo_strips = [None, None, None, None]
        if with_offsets:
            from legoesm.grids.halo import interp_strip

        for r in range(4):
            send_edge = ppermute_send_j[r, my_idx]   # traced int
            to_send = my_strips[send_edge]           # (n,) or (n, C)
            received = jax.lax.ppermute(
                to_send, "face", _PPERMUTE_PERMS[r],
            )
            recv_edge = ppermute_recv_j[r, my_idx]
            rev = ppermute_rev_j[r, my_idx]
            received = jnp.where(rev, received[::-1], received)
            if with_offsets:
                # offsets[my_idx, recv_edge] selects the right per-edge
                # offset for whichever halo slot this round fills.  The
                # offset table is indexed by *receiving* face/edge —
                # same convention the local-pad ``pad_halo_local``
                # uses (halo.py:1078, ``interp_offsets[face, edge_idx]``).
                # ``recv_edge`` is traced, so we ``lax.dynamic_slice``
                # by indexing into the (4, n)-shaped face slice.
                offs_for_face = offsets[my_idx]  # (4, n)
                offs_for_edge = offs_for_face[recv_edge]  # (n,)
                received = interp_strip(received, offs_for_edge)
            # Place in the correct halo slot.  recv_edge is traced,
            # so we use conditional sets.
            for e in range(4):
                is_this = (recv_edge == e)
                if halo_strips[e] is None:
                    halo_strips[e] = jnp.where(is_this, received,
                                               jnp.zeros_like(received))
                else:
                    halo_strips[e] = jnp.where(is_this, received,
                                               halo_strips[e])

        padded = _fill_halo_and_corners(padded, halo_strips, n)
        return padded[None]

    return _exchange


# ===================================================================
# Backend C: multi-face ppermute (k = 6 / n_devices faces per shard)
# ===================================================================
#
# Generalizes the one-face ppermute kernel to n_devices ∈ {1, 2, 3, 6}
# with k = 6/n_devices CONTIGUOUS faces per shard (matching how
# ``Mesh(devices, ("face",))`` + ``P("face")`` block-partitions axis 0:
# device d owns global faces [d*k, (d+1)*k)).
#
#   * intra-shard face edges → shard-local strip copies (no collective);
#   * cross-shard face edges → static schedule of ppermute rounds.
#     The schedule is a proper edge coloring of the DIRECTED DEVICE-PAIR
#     graph (codex design review: coloring on device pairs, not face
#     pairs — at n_devices=3 every device has cross edges to BOTH other
#     devices, so multiple rounds are required).  Each round is a
#     partial permutation (each device sends ≤ 1 buffer and receives
#     ≤ 1 buffer); all strips a device owes a given peer in a round are
#     packed into ONE fixed-shape buffer (slots), so the SPMD program
#     is shape-uniform across devices.
#
# By König's edge-coloring theorem on the bipartite (senders ×
# receivers) multigraph, max(out_degree, in_degree) rounds always
# suffice; the exhaustive backtracking below finds such an optimal
# coloring deterministically.  Resulting round counts:
#   n_devices=1 → 0 rounds (everything intra-shard)
#   n_devices=2 → 1 round   (pairwise swap, 8 strips/direction)
#   n_devices=3 → 2 rounds  (two opposite 3-cycles, ≤ 4 strips/pair)
#   n_devices=6 → 4 rounds  (octahedral face graph, 1 strip/pair)


class _MultifaceTables(NamedTuple):
    """Static per-layout tables for the multi-face ppermute exchange.

    All index tables have a leading device axis so the SPMD kernel can
    select its rows with the traced ``jax.lax.axis_index("face")``.

    Attributes
    ----------
    perms : tuple[tuple[tuple[int, int], ...], ...]
        ``perms[r]`` is the (src_dev, dst_dev) pair list for ppermute
        round ``r``.
    max_slots : int
        Strip-slot count of every round's send/recv buffer (global max
        over all directed device pairs; unused slots carry garbage on
        send and an out-of-bounds target on receive → dropped).
    loc_lf, loc_le : (n_devices, k, 4) int32
        Intra-shard source (local face, edge) for each receiving
        (local face i, edge e).  Cross-shard entries point at (0, 0)
        as a placeholder; they are provably overwritten by a ppermute
        round (coverage asserted at build time).
    rev : (n_devices, k, 4) int32
        ``CONNECTIVITY[g][e].reversed`` for the receiving (face, edge)
        — applied receiver-side to BOTH local and ppermute strips
        (senders always transmit unreversed source-edge strips).
    send_lf, send_le : (n_rounds, n_devices, max_slots) int32
        Strip (local face, edge) this device places in slot ``m`` when
        it is round ``r``'s sender (garbage rows when idle/padding).
    recv_tgt : (n_rounds, n_devices, max_slots) int32
        Flattened ``local_face * 4 + edge`` halo target for slot ``m``,
        or the sentinel ``4 * k`` (out of bounds → ``mode="drop"``
        scatter discards it) when the slot is padding or the device
        does not receive in round ``r``.
    faces_per_shard : int
        k = 6 // n_devices.
    """

    perms: tuple
    max_slots: int
    loc_lf: np.ndarray
    loc_le: np.ndarray
    rev: np.ndarray
    send_lf: np.ndarray
    send_le: np.ndarray
    recv_tgt: np.ndarray
    faces_per_shard: int


def _color_device_pairs(pairs, n_rounds):
    """Exhaustive backtracking edge coloring of directed device pairs.

    Returns a list of ``n_rounds`` lists of (src, dst) pairs where no
    round repeats a src or a dst (each round is a valid ppermute
    partial permutation), or ``None`` if no coloring with ``n_rounds``
    exists.  Deterministic: pairs are processed in sorted order and
    rounds tried in ascending index.
    """
    rounds_src = [set() for _ in range(n_rounds)]
    rounds_dst = [set() for _ in range(n_rounds)]
    assignment = [-1] * len(pairs)

    def _bt(i):
        if i == len(pairs):
            return True
        s, d = pairs[i]
        for r in range(n_rounds):
            if s not in rounds_src[r] and d not in rounds_dst[r]:
                rounds_src[r].add(s)
                rounds_dst[r].add(d)
                assignment[i] = r
                if _bt(i + 1):
                    return True
                rounds_src[r].remove(s)
                rounds_dst[r].remove(d)
                assignment[i] = -1
        return False

    if not _bt(0):
        return None
    out = [[] for _ in range(n_rounds)]
    for i, p in enumerate(pairs):
        out[assignment[i]].append(p)
    return out


def _build_multiface_tables(faces_per_shard: int) -> _MultifaceTables:
    """Build the static multi-face exchange tables for a face layout.

    The tables are halo-depth independent (the schedule moves whole
    (face, edge) strips; only the strip payload shape differs between
    halo=1 and halo=2), so one table set serves both kernels.
    """
    k = int(faces_per_shard)
    if k not in (1, 2, 3, 6):
        raise ValueError(
            f"faces_per_shard must be one of 1, 2, 3, 6 (n_devices must "
            f"divide 6), got {k}."
        )
    n_dev = 6 // k

    def owner(f):
        return f // k

    loc_lf = np.zeros((n_dev, k, 4), dtype=np.int32)
    loc_le = np.zeros((n_dev, k, 4), dtype=np.int32)
    rev = np.zeros((n_dev, k, 4), dtype=np.int32)
    # cross[(src_dev, dst_dev)] = sorted list of
    #   (dst_local_face, dst_edge, src_local_face, src_edge)
    cross: dict[tuple[int, int], list] = {}
    covered_local = set()
    for d in range(n_dev):
        for i in range(k):
            g = d * k + i
            for e in range(4):
                nf, ne, rv = CONNECTIVITY[g][e]
                rev[d, i, e] = int(rv)
                sd = owner(nf)
                if sd == d:
                    loc_lf[d, i, e] = nf - d * k
                    loc_le[d, i, e] = ne
                    covered_local.add((d, i, e))
                else:
                    cross.setdefault((sd, d), []).append(
                        (i, e, nf - sd * k, ne)
                    )

    pairs = sorted(cross.keys())
    for p in pairs:
        # Canonical slot order by (receiving local face, edge) — both
        # the send and recv tables are derived from this single list,
        # so sender slot m and receiver slot m always describe the
        # same strip.
        cross[p].sort()

    if pairs:
        out_deg = Counter(s for s, _ in pairs)
        in_deg = Counter(d for _, d in pairs)
        r_min = max(max(out_deg.values()), max(in_deg.values()))
        rounds = _color_device_pairs(pairs, r_min)
        # König guarantees an r_min coloring exists for the bipartite
        # send/recv multigraph; the exhaustive search must find it.
        if rounds is None:  # pragma: no cover - mathematically unreachable
            raise RuntimeError(
                f"multiface ppermute schedule coloring failed for "
                f"faces_per_shard={k} at the König bound {r_min}."
            )
        max_slots = max(len(v) for v in cross.values())
    else:
        rounds = []
        max_slots = 1  # unused (n_rounds == 0)

    n_rounds = len(rounds)
    t_rounds = max(n_rounds, 1)  # keep arrays non-empty for jnp.asarray
    send_lf = np.zeros((t_rounds, n_dev, max_slots), dtype=np.int32)
    send_le = np.zeros((t_rounds, n_dev, max_slots), dtype=np.int32)
    recv_tgt = np.full((t_rounds, n_dev, max_slots), 4 * k, dtype=np.int32)
    covered_cross = set()
    for r, rnd in enumerate(rounds):
        # Each round must be a partial permutation on devices.
        if (len({s for s, _ in rnd}) != len(rnd)
                or len({d for _, d in rnd}) != len(rnd)):
            raise RuntimeError(
                f"multiface schedule round {r} is not a partial "
                f"permutation: {rnd}"
            )
        for (s, d) in rnd:
            for m, (i, e, slf, sle) in enumerate(cross[(s, d)]):
                send_lf[r, s, m] = slf
                send_le[r, s, m] = sle
                recv_tgt[r, d, m] = i * 4 + e
                covered_cross.add((d, i, e))

    # --- build-time invariants (loud failure beats silent halo junk) ---
    all_entries = {
        (d, i, e) for d in range(n_dev) for i in range(k) for e in range(4)
    }
    if covered_local | covered_cross != all_entries or (
            covered_local & covered_cross):
        raise RuntimeError(
            f"multiface tables for faces_per_shard={k}: (face, edge) "
            f"coverage broken — local={len(covered_local)}, "
            f"cross={len(covered_cross)}, total={len(all_entries)}."
        )
    if sorted(p for rnd in rounds for p in rnd) != pairs:
        raise RuntimeError(
            f"multiface schedule for faces_per_shard={k} does not cover "
            f"every directed device pair exactly once."
        )

    return _MultifaceTables(
        perms=tuple(tuple(rnd) for rnd in rounds),
        max_slots=max_slots,
        loc_lf=loc_lf,
        loc_le=loc_le,
        rev=rev,
        send_lf=send_lf,
        send_le=send_le,
        recv_tgt=recv_tgt,
        faces_per_shard=k,
    )


# Built lazily on first factory call (NOT at import — iter-93 doctrine:
# no eager work at module import).  Keyed by faces_per_shard.
_MULTIFACE_TABLE_CACHE: dict[int, _MultifaceTables] = {}


def _get_multiface_tables(faces_per_shard: int) -> _MultifaceTables:
    if faces_per_shard not in _MULTIFACE_TABLE_CACHE:
        _MULTIFACE_TABLE_CACHE[faces_per_shard] = _build_multiface_tables(
            faces_per_shard,
        )
    return _MULTIFACE_TABLE_CACHE[faces_per_shard]


def _make_exchange_ppermute_multiface(mesh, ndim, halo=1, with_offsets=False):
    """Build the multi-face ppermute shard_map exchange.

    Supports halo=1 and halo=2, 3D ``(6, n, n)`` and 4D ``(6, n, n, C)``
    inputs, with optional per-edge Lagrange ``interp_offsets`` (h1
    offsets ``(6, 4, n)``; h2 offsets ``(6, 4, 2, n)``) — the same
    receiver-side reverse→interpolate→place pipeline as the serial
    local pad, so the output is bit-identical to
    ``pad_halo_local``/``_pad_halo_local_h2`` (avg corner fill).
    """
    if ndim not in (3, 4):
        raise ValueError(f"ndim must be 3 or 4, got {ndim}")
    if halo not in (1, 2):
        raise ValueError(f"multiface ppermute supports halo 1/2, got {halo}")

    n_devices = len(mesh.devices.flat)
    if n_devices < 1 or 6 % n_devices != 0:
        raise ValueError(
            f"multiface ppermute exchange requires a face-axis mesh whose "
            f"device count divides 6; got {n_devices} devices."
        )
    k = 6 // n_devices
    tables = _get_multiface_tables(k)
    n_rounds = len(tables.perms)
    perms = tables.perms  # static python (src, dst) pair tuples

    P = jax.sharding.PartitionSpec
    in_sp_data = P("face", *((None,) * (ndim - 1)))
    out_sp = P("face", *((None,) * (ndim - 1)))
    in_sp = (in_sp_data, P()) if with_offsets else in_sp_data

    # iter-94g pattern: jnp table constants built in the factory body
    # (outside the shard_map closure body, after backend init).
    loc_lf_j = jnp.asarray(tables.loc_lf)
    loc_le_j = jnp.asarray(tables.loc_le)
    rev_j = jnp.asarray(tables.rev)
    send_lf_j = jnp.asarray(tables.send_lf)
    send_le_j = jnp.asarray(tables.send_le)
    recv_tgt_j = jnp.asarray(tables.recv_tgt)

    @partial(shard_map, mesh=mesh, in_specs=in_sp, out_specs=out_sp,
             check_vma=False)
    def _exchange(*args):
        if with_offsets:
            local_shard, offsets = args
        else:
            (local_shard,) = args
            offsets = None

        n = local_shard.shape[1]
        my_idx = jax.lax.axis_index("face")

        # ---- extract all 4k perimeter strips once (static loops) ----
        # halo=1 payload: (4, n[, C]) per face; halo=2 payload:
        # (4, 2, n[, C]) with depth-0 = boundary cell, depth-1 = one
        # cell inward (same convention as _make_exchange_allgather_h2
        # and the serial extract_edge_strip_at_depth).
        def _face_strips(face):
            if halo == 1:
                if ndim == 3:
                    return jnp.stack([
                        face[0, :], face[-1, :], face[:, 0], face[:, -1],
                    ], axis=0)
                return jnp.stack([
                    face[0, :, :], face[-1, :, :],
                    face[:, 0, :], face[:, -1, :],
                ], axis=0)
            if ndim == 3:
                return jnp.stack([
                    jnp.stack([face[0, :], face[1, :]], axis=0),
                    jnp.stack([face[-1, :], face[-2, :]], axis=0),
                    jnp.stack([face[:, 0], face[:, 1]], axis=0),
                    jnp.stack([face[:, -1], face[:, -2]], axis=0),
                ], axis=0)
            return jnp.stack([
                jnp.stack([face[0, :, :], face[1, :, :]], axis=0),
                jnp.stack([face[-1, :, :], face[-2, :, :]], axis=0),
                jnp.stack([face[:, 0, :], face[:, 1, :]], axis=0),
                jnp.stack([face[:, -1, :], face[:, -2, :]], axis=0),
            ], axis=0)

        my_strips = jnp.stack(
            [_face_strips(local_shard[i]) for i in range(k)], axis=0,
        )  # (k, 4, [2,] n[, C])

        # ---- intra-shard fill (cross entries are placeholders that the
        # ppermute rounds provably overwrite — coverage asserted at
        # table-build time) ----
        my_loc_lf = loc_lf_j[my_idx].reshape(-1)   # (4k,) traced
        my_loc_le = loc_le_j[my_idx].reshape(-1)
        halo_flat = my_strips[my_loc_lf, my_loc_le]  # (4k, [2,] n[, C])

        # ---- cross-shard ppermute rounds ----
        for r in range(n_rounds):
            send_buf = my_strips[send_lf_j[r, my_idx],
                                 send_le_j[r, my_idx]]  # (max_slots, ...)
            received = jax.lax.ppermute(send_buf, "face", perms[r])
            tgt = recv_tgt_j[r, my_idx]  # (max_slots,) — 4k sentinel drops
            halo_flat = halo_flat.at[tgt].set(received, mode="drop")

        halo_buf = halo_flat.reshape((k, 4) + halo_flat.shape[1:])

        # ---- receiver-side reversal along the spatial strip axis ----
        # (senders transmit unreversed source-edge strips; reversal is a
        # property of the receiving (face, edge), exactly as the serial
        # tables bake it in and the one-face kernel applies it.)
        flip_axis = 2 if halo == 1 else 3
        my_rev = rev_j[my_idx].reshape(
            (k, 4) + (1,) * (halo_buf.ndim - 2)
        ).astype(bool)
        halo_buf = jnp.where(
            my_rev, jnp.flip(halo_buf, axis=flip_axis), halo_buf,
        )

        if with_offsets:
            # Offsets are indexed by the *receiving* global face/edge —
            # same convention as pad_halo_local (halo.py).  Slice this
            # shard's k rows with the traced device index.
            offs_dev = jax.lax.dynamic_slice_in_dim(
                offsets, my_idx * k, k, axis=0,
            )  # (k, 4, n) for h1; (k, 4, 2, n) for h2

        pad_width = ((halo, halo), (halo, halo))
        if ndim == 4:
            pad_width = pad_width + ((0, 0),)

        padded_faces = []
        for i in range(k):
            padded = jnp.pad(local_shard[i], pad_width)
            strips = []
            for e in range(4):
                s = halo_buf[i, e]
                if with_offsets:
                    if halo == 1:
                        s = interp_strip(s, offs_dev[i, e])
                    else:
                        # Per-depth Lagrange, matching the allgather_h2
                        # kernel and the serial h2 loop (reverse first,
                        # then interpolate each depth).
                        s = jnp.stack([
                            interp_strip(s[0], offs_dev[i, e, 0]),
                            interp_strip(s[1], offs_dev[i, e, 1]),
                        ], axis=0)
                strips.append(s)
            if halo == 1:
                padded = _fill_halo_and_corners(padded, strips, n)
            else:
                padded = _fill_halo_and_corners_h2_local(padded, strips, n)
            padded_faces.append(padded)

        return jnp.stack(padded_faces, axis=0)

    return _exchange


# ===================================================================
# Compiled-HLO tripwire: full-cube all-gather detector
# ===================================================================

# Matches the RESULT type list of a sync or async all-gather HLO op:
#   %ag  = f32[6,4,24,8]{3,2,1,0} all-gather(f32[3,4,24,8] %x), ...
#   %ags = (f32[3,...], f32[6,...]) all-gather-start(...), ...
#   %agd = f32[6,4,24,8]{3,2,1,0} all-gather-done(%ags)
# The -done form is matched too so the guard does not depend on the
# paired -start line surviving textual transformations (duplicate
# reports for a start/done pair are harmless — the result is a flag).
_HLO_ALLGATHER_LINE_RE = re.compile(
    r"=\s*(.+?)\s+all-gather(?:-start|-done)?\("
)
_HLO_SHAPE_RE = re.compile(r"\[([0-9,]*)\]")


def find_fullcube_allgathers(
    hlo_text: str, n_faces: int = 6, n: int | None = None,
) -> list[str]:
    """Scan compiled HLO text for all-gather ops that materialize the cube.

    Mechanical tripwire (NOT a proof) for the compute-replication bug
    class proven by the job-8456476 HLO probe: under face sharding any
    all-gather whose result regains the full 6-face extent means some
    value is replicated across the face axis — and GSPMD then replicates
    the downstream compute (per-device FLOPs ratio 1.00 at 2 devices).

    Flags an all-gather result shape when either
    * its leading dim equals ``n_faces`` and it has more than one
      element per face (catches both full fields ``[6,n,n,...]`` and
      perimeter-strip gathers ``[6,4,n,...]``), or
    * ``n`` is given and the total element count is >= ``n_faces*n*n``
      (catches full-cube volume hidden behind reshapes/slicing — the
      codex-flagged false-pass surface).

    Known false-pass surface: an all-gather whose result drops the
    leading face dim AND stays under the volume bound.  Known
    false-positive surface: a legitimate gather of a non-face axis of
    extent exactly ``n_faces`` — none exist on the ppermute hot path.

    Returns the offending result-shape strings (empty list = clean).
    """
    bad: list[str] = []
    for line in hlo_text.splitlines():
        if "all-gather" not in line:
            continue
        m = _HLO_ALLGATHER_LINE_RE.search(line)
        if m is None:
            continue
        for dims_s in _HLO_SHAPE_RE.findall(m.group(1)):
            dims = [int(d) for d in dims_s.split(",") if d]
            if not dims:
                continue
            total = 1
            for d in dims:
                total *= d
            if (dims[0] == n_faces and total > n_faces) or (
                    n is not None and total >= n_faces * n * n):
                bad.append(f"[{dims_s}]")
    return bad


def assert_no_fullcube_allgather(
    hlo_text: str,
    n_faces: int = 6,
    n: int | None = None,
    context: str = "compiled program",
) -> None:
    """Raise ``RuntimeError`` if :func:`find_fullcube_allgathers` flags
    any op — loud guard for benches/tests on the hot path (every device
    would silently compute the full globe while the timing rows claim
    multi-device scaling)."""
    bad = find_fullcube_allgathers(hlo_text, n_faces=n_faces, n=n)
    if bad:
        raise RuntimeError(
            f"{context}: compiled HLO contains {len(bad)} all-gather "
            f"op(s) with full-cube face extent {bad[:8]} — the SPMD halo "
            f"is materializing all {n_faces} faces per device (compute "
            f"replication, HLO probe job 8456476).  Expected the "
            f"ppermute multiface exchange (collective-permute only).  "
            f"If the all_gather diagnostic backend was intended, set "
            f"LEGOESM_SPMD_FORCE_ALLGATHER=1 explicitly."
        )


# ===================================================================
# Public scalar exchange API
# ===================================================================

_cache: dict[tuple, object] = {}


def _select_variant(use_ppermute, halo, n_devices):
    """Resolve the kernel variant name for routing + cache keying.

    * ``use_ppermute=False`` → the all_gather kernels (explicit
      diagnostic opt-in only — see module docstring).
    * ``use_ppermute=True, halo=1, n_devices=6`` → the original
      validated one-face ppermute kernel (lowest-risk path for the
      1-face-per-device layout; bit-identical to the multiface kernel
      by the parity tests, kept per codex review).
    * every other ppermute combination (halo=2 at any count; halo=1 at
      1/2/3 devices) → the multiface ppermute kernel.
    """
    if not use_ppermute:
        return "allgather_h2" if halo == 2 else "allgather"
    if halo == 1 and n_devices == 6:
        return "ppermute_oneface"
    return "ppermute_multiface"


def _get_exchange(mesh, ndim, use_ppermute, halo=1, with_offsets=False):
    """Get (or build and cache) a SPMD halo-exchange kernel.

    Parameters
    ----------
    mesh : jax.sharding.Mesh
        Face-axis mesh (device count must divide 6).
    ndim : int
        3 for scalar (6, n, n) inputs, 4 for (6, n, n, C) inputs.
    use_ppermute : bool
        When *True* (the production default set by
        :func:`activate_spmd_halo_backend`), route to a ppermute kernel
        — the multiface kernel for any layout, except halo=1 at exactly
        6 devices which keeps the original validated one-face kernel.
        When *False*, route to the all_gather kernels (diagnostic
        opt-in only: they replicate compute, see module docstring).
    halo : int
        Halo depth.  Supported: 1, 2.  Other values fall back to the
        local pad path; see :func:`explicit_pad_halo`.
    with_offsets : bool
        When True, build a kernel that takes a second ``interp_offsets``
        argument and applies the per-edge Lagrange correction.

    Notes
    -----
    The cache key includes the shard layout (``faces_per_shard``) and
    the resolved variant name (codex MINOR 6): ``id(mesh)`` alone can
    collide after garbage collection, and the old
    ``use_ppermute``-boolean key could not distinguish the one-face
    and multiface kernels.
    """
    n_devices = len(mesh.devices.flat)
    if n_devices < 1 or 6 % n_devices != 0:
        raise ValueError(
            f"SPMD cubed-sphere halo exchange requires a face-axis mesh "
            f"whose device count divides 6, got {n_devices}."
        )
    faces_per_shard = 6 // n_devices
    variant = _select_variant(use_ppermute, halo, n_devices)
    key = (id(mesh), ndim, variant, halo, with_offsets, faces_per_shard)
    if key not in _cache:
        if variant == "allgather_h2":
            _cache[key] = _make_exchange_allgather_h2(
                mesh, ndim, with_offsets=with_offsets,
            )
        elif variant == "allgather":
            _cache[key] = _make_exchange_allgather(
                mesh, ndim, with_offsets=with_offsets,
            )
        elif variant == "ppermute_oneface":
            _cache[key] = _make_exchange_ppermute(
                mesh, ndim, with_offsets=with_offsets,
            )
        else:  # ppermute_multiface
            _cache[key] = _make_exchange_ppermute_multiface(
                mesh, ndim, halo=halo, with_offsets=with_offsets,
            )
    return _cache[key]


# Module-level flag: use ppermute by default?
_use_ppermute: bool = False

# Explicit all_gather diagnostic override (documented opt-in).
_FORCE_ALLGATHER_ENV = "LEGOESM_SPMD_FORCE_ALLGATHER"


def _force_allgather_from_env() -> bool:
    return os.environ.get(_FORCE_ALLGATHER_ENV, "") == "1"


def select_exchange_backend(
    n: int = 0,
    nlev: int = 1,
    n_devices: int = 6,
    dtype_bytes: int = 4,
) -> bool:
    """Decide whether to use ppermute (True) or all_gather (False).

    Always returns True (ppermute) unless the explicit all_gather
    diagnostic override ``LEGOESM_SPMD_FORCE_ALLGATHER=1`` is set.

    RETIRED HEURISTIC: this function used to compare estimated
    all_gather data volume against ``LEGOESM_SPMD_HALO_THRESHOLD_MB``.
    That heuristic was structurally wrong — the all_gather variant's
    true cost is not bandwidth but FULL COMPUTE REPLICATION (HLO probe
    job 8456476: per-device/single-device FLOPs ratio 1.00 at 2
    devices, all-gather results with full 6-face extent), which no
    data-volume model can see.  ppermute is therefore the only
    auto-selectable backend; all_gather remains available solely as an
    explicit diagnostic opt-in.

    Parameters
    ----------
    n, nlev, n_devices, dtype_bytes : int
        Retained for call-site compatibility; no longer consulted.

    Returns
    -------
    bool
        True if ppermute is selected, False only under the explicit
        all_gather diagnostic override.
    """
    del n, nlev, n_devices, dtype_bytes  # retired volume heuristic inputs
    return not _force_allgather_from_env()


def set_ppermute_default(enabled: bool) -> None:
    """Switch the module-flag collective backend.

    Parameters
    ----------
    enabled : bool
        ``True`` → ppermute kernels (the production default set by
        :func:`activate_spmd_halo_backend`): the multiface kernel with
        a layout-dependent round count (0/1/2/4 rounds at 1/2/3/6
        devices), or the one-face 4-round kernel at halo=1 with
        exactly 6 devices.
        ``False`` → the all_gather kernels — DIAGNOSTIC ONLY: they
        replicate all compute per device (HLO probe job 8456476), so
        only set this for explicit comparisons, never for production
        or timing runs.
    """
    global _use_ppermute
    _use_ppermute = enabled


def explicit_pad_halo(data, mesh, halo=1, interp_offsets=None):
    """Explicit 3D scalar exchange.  (6,n,n) → (6,n+2h,n+2h).

    Halo=1 and halo=2 both honour the module ppermute flag (the
    production default set by :func:`activate_spmd_halo_backend`):
    ppermute routes to the multiface kernel (one-face kernel at halo=1
    with exactly 6 devices); the all_gather kernels remain reachable
    only with the flag off (diagnostic opt-in).  Other halo depths
    fall back to the local-pad path.

    When ``interp_offsets`` is provided the SPMD kernel applies the
    per-edge 3-point Lagrange correction so the SPMD result matches
    ``pad_halo_local(data, interp_offsets)`` bit-for-bit.
    """
    if halo == 2:
        if interp_offsets is None:
            return _get_exchange(mesh, 3, _use_ppermute, halo=2)(data)
        return _get_exchange(
            mesh, 3, _use_ppermute, halo=2, with_offsets=True,
        )(data, interp_offsets)
    if halo != 1:
        from legoesm.grids.halo import pad_halo_local
        return pad_halo_local(data, interp_offsets)
    if interp_offsets is None:
        return _get_exchange(mesh, 3, _use_ppermute, halo=1)(data)
    return _get_exchange(
        mesh, 3, _use_ppermute, halo=1, with_offsets=True,
    )(data, interp_offsets)


def explicit_pad_halo_4d(data, mesh, halo=1, interp_offsets=None):
    """Explicit 4D scalar exchange.  (6,n,n,C) → (6,n+2h,n+2h,C).

    See :func:`explicit_pad_halo` for the halo support matrix and the
    ``interp_offsets`` semantics.
    """
    # FV3_3D iter-1073 (codex iter-1072 BLOCKER): mirror the iter-1072
    # non-square guard from ``pad_halo_4d`` so SPMD callers don't
    # bypass the silent-corruption check.  ``pad_halo_local_4d``
    # fallback path (``halo != 1, halo != 2``) and the SPMD exchanges
    # all assume square ``(n, n)``.
    if data.shape[1] != data.shape[2]:
        raise ValueError(
            f"explicit_pad_halo_4d expects square (n, n) data on each "
            f"face, got shape {tuple(data.shape)}.  See FV3_3D.md "
            f"iter-1072 for the silent-corruption probe."
        )
    if halo == 2:
        if interp_offsets is None:
            return _get_exchange(mesh, 4, _use_ppermute, halo=2)(data)
        return _get_exchange(
            mesh, 4, _use_ppermute, halo=2, with_offsets=True,
        )(data, interp_offsets)
    if halo != 1:
        from legoesm.grids.halo import pad_halo_local_4d
        return pad_halo_local_4d(data, interp_offsets)
    if interp_offsets is None:
        return _get_exchange(mesh, 4, _use_ppermute, halo=1)(data)
    return _get_exchange(
        mesh, 4, _use_ppermute, halo=1, with_offsets=True,
    )(data, interp_offsets)


# ===================================================================
# Vector (u, v) exchange
# ===================================================================

def explicit_pad_halo_vector_4d(
    u_data, v_data,
    cos_angle, sin_angle,
    cos_angle_padded, sin_angle_padded,
    mesh, halo=1, interp_offsets=None,
):
    """Explicit 4D vector halo exchange.

    Rotates grid-aligned (u, v) to geographic (east, north), exchanges
    BOTH components in a **single** collective (packed along the trailing
    axis), then rotates back using the padded grid angles.

    This halves the collective count compared to two separate scalar
    exchanges.

    When ``interp_offsets`` is provided, the underlying scalar SPMD
    exchange applies the per-edge Lagrange correction so the result
    matches the local-pad vector reference bit-for-bit.  Iter-31 fix:
    without this forwarding, the dycore's hyperdiffusion path
    (`hyperdiffusion_3d` → `divergence_3d` → `pad_halo_vector_4d` →
    here) silently dropped offsets under SPMD, producing ~6e-4 relative
    drift on u/v after one SSP-RK3 step at C24/L8.

    Parameters
    ----------
    u_data, v_data : (6, n, n, nlev)
    cos_angle, sin_angle : (6, n, n)   — grid angles (interior)
    cos_angle_padded, sin_angle_padded : (6, n+2, n+2) — padded angles
    mesh : jax.sharding.Mesh
    halo : int
    interp_offsets : (6, 4, n) or None
        Forwarded to the inner ``explicit_pad_halo_4d``.

    Returns
    -------
    u_padded, v_padded : (6, n+2*halo, n+2*halo, nlev)
    """
    # Step 1: rotate grid-aligned → geographic
    ca = cos_angle[..., None]
    sa = sin_angle[..., None]
    u_east = ca * u_data - sa * v_data
    v_north = sa * u_data + ca * v_data

    # Step 2: pack both into one field → single collective
    packed = jnp.concatenate([u_east, v_north], axis=-1)  # (6, n, n, 2*nlev)
    packed_padded = explicit_pad_halo_4d(
        packed, mesh, halo=halo, interp_offsets=interp_offsets,
    )

    # Step 3: unpack
    nlev = u_data.shape[-1]
    u_east_pad = packed_padded[..., :nlev]
    v_north_pad = packed_padded[..., nlev:]

    # Step 4: rotate back geographic → grid-aligned (using padded angles)
    cap = cos_angle_padded[..., None]
    sap = sin_angle_padded[..., None]
    u_padded = cap * u_east_pad + sap * v_north_pad
    v_padded = -sap * u_east_pad + cap * v_north_pad
    return u_padded, v_padded


# ===================================================================
# Packed multi-field exchange
# ===================================================================

def packed_pad_halo_4d(
    *fields, mesh, duogrid=None, interp_offsets=None, halo=1,
):
    """Exchange halos for multiple 4D fields in a single collective.

    Stacks fields along the trailing axis, performs ONE exchange, then
    splits.  Reduces collective count from ``len(fields)`` to 1.

    All fields must share the same ``(6, n, n)`` spatial prefix.

    When ``duogrid`` is provided, applies the kinked-to-extended remap
    + corner fill to each output field after the packed exchange.
    Without this, packed SPMD halos silently bypass duogrid while the
    unpacked ``pad_halo_4d`` path applies it (see halo.py:559-573).

    When ``interp_offsets`` is provided (and ``duogrid`` is None — the
    canonical Lagrange-corrected halo path), the SPMD exchange kernel
    applies per-edge 3-point Lagrange correction to each received strip
    using the same offsets the unpacked ``pad_halo_4d`` would apply.
    Without this, the packed SPMD path silently bypasses
    ``interp_offsets`` while the unpacked ``pad_halo_4d`` path applies
    them.

    ``halo`` selects the SPMD exchange depth (1 → ``(6, n+2, n+2, C)``,
    2 → ``(6, n+4, n+4, C)``).  Kernel routing is identical to
    :func:`explicit_pad_halo_4d`: the ppermute kernels under the module
    flag (multiface for any face-sharded layout; one-face at halo=1
    with 6 devices), the all_gather kernels only as the explicit
    diagnostic opt-in.
    """
    if not fields:
        return []
    # Mirror pad_halo_4d (halo.py): duogrid (nearest-copy + Lagrange remap) and
    # interp_offsets (in-kernel 3-point Lagrange) are mutually exclusive — applying
    # both would silently run "offsets first, duogrid remap second".  Reject up
    # front rather than corrupt the halo.
    if interp_offsets is not None and duogrid is not None:
        raise ValueError(
            "interp_offsets and duogrid are mutually exclusive in "
            "packed_pad_halo_4d (mirrors pad_halo_4d); pass one or the other."
        )
    # FV3_3D iter-1073 (codex iter-1072 BLOCKER): non-square guard
    # mirroring ``pad_halo_4d``.  Each field must be square (n, n).
    for i, f in enumerate(fields):
        if f.shape[1] != f.shape[2]:
            raise ValueError(
                f"packed_pad_halo_4d field {i}: expects square (n, n) "
                f"data on each face, got shape {tuple(f.shape)}.  See "
                f"FV3_3D.md iter-1072."
            )

    if len(fields) == 1:
        out = explicit_pad_halo_4d(
            fields[0], mesh, halo=halo, interp_offsets=interp_offsets,
        )
        # Single field still owes the duogrid post-remap — the multi-field path
        # below applies it per piece, so a 1-field call must too (otherwise
        # `packed_pad_halo_4d(f, duogrid=dg)` silently returned a nearest-copy
        # halo while the unpacked `pad_halo_4d(f, duogrid=dg)` applied the remap).
        if duogrid is not None:
            out = _apply_duogrid_4d(out, duogrid, halo=halo)
        return [out]

    # Use plain Python ints for split indices so JAX treats them as
    # static constants — passing a traced ``jnp.cumsum`` to ``jnp.split``
    # forces a host evaluation in older JAX and outright errors in newer
    # versions.  This mirrors the MPI-side fix in ``halo_exchange.py``.
    splits = [int(f.shape[-1]) for f in fields]
    split_indices = np.cumsum(splits[:-1]).tolist()
    stacked = jnp.concatenate(fields, axis=-1)
    padded = explicit_pad_halo_4d(
        stacked, mesh, halo=halo, interp_offsets=interp_offsets,
    )
    pieces = list(jnp.split(padded, split_indices, axis=-1))
    if duogrid is not None:
        pieces = [_apply_duogrid_4d(p, duogrid, halo=halo) for p in pieces]
    return pieces


def _apply_duogrid_4d(padded, duogrid, halo):
    """Apply duogrid kinked-to-extended remap + corner fill to a 4D
    padded field, level-by-level via ``jax.vmap``.  Mirrors the
    post-processing loop inside ``halo.pad_halo_4d``.
    """
    from legoesm.grids.duogrid import cube_rmp_vectorized, fill_corner_region
    import jax

    def _remap_level(level_slice):
        level_slice = cube_rmp_vectorized(level_slice, duogrid, halo)
        level_slice = fill_corner_region(level_slice, duogrid, halo)
        return level_slice

    padded_t = jnp.transpose(padded, (3, 0, 1, 2))
    padded_t = jax.vmap(_remap_level)(padded_t)
    return jnp.transpose(padded_t, (1, 2, 3, 0))


# ===================================================================
# SPMD backend activation
# ===================================================================

_spmd_mesh = None


def activate_spmd_halo_backend(
    mesh, n: int = 0, nlev: int = 1, *, force_allgather: bool | None = None,
) -> None:
    """Switch the global halo backend to explicit SPMD exchange.

    The exchange kernel is **ppermute** for every face-sharded device
    count (1, 2, 3, 6 — multiface kernel; one-face kernel at halo=1
    with exactly 6 devices).  The all_gather kernels are an explicit
    diagnostic opt-in ONLY: they were proven to replicate ALL compute
    on every device (HLO probe job 8456476 — per-device FLOPs ratio
    1.00 at 2 devices), so no heuristic may auto-select them.

    Parameters
    ----------
    mesh : jax.sharding.Mesh
        Face-axis mesh; the device count must divide 6.
    n : int
        Per-face resolution.  Retained for call-site compatibility and
        logging; no longer drives backend choice (the retired volume
        heuristic could not see replication cost).
    nlev : int
        Number of vertical levels (logging only, see ``n``).
    force_allgather : bool or None
        ``True`` selects the all_gather diagnostic kernels.  ``None``
        (default) defers to the ``LEGOESM_SPMD_FORCE_ALLGATHER=1``
        environment override, otherwise ppermute.
    """
    global _spmd_mesh, _use_ppermute
    from legoesm.grids import halo
    # The SPMD exchange kernels hard-code AVERAGE corner fill at the 4 cube
    # corners (3-face junctions); they do NOT honor the non-default
    # ``_corner_fill_mode`` the serial/mpi4jax paths apply. Reject non-avg modes
    # so SPMD never silently produces wrong corner cells (codex review). Checked
    # FIRST, before any global mutation, so a raise leaves state untouched.
    cfm = halo.get_corner_fill_mode()
    if cfm != "avg":
        raise NotImplementedError(
            f"SPMD halo backend supports only corner_fill_mode='avg', not "
            f"'{cfm}': the 4 cube-corner cells would silently mismatch the "
            f"serial path. Call set_corner_fill_mode('avg'), or use the mpi4jax "
            f"backend for non-avg corner fills.")
    n_devices = len(mesh.devices.flat)
    if n_devices < 1 or 6 % n_devices != 0:
        raise ValueError(
            f"SPMD halo backend requires a face-axis mesh whose device "
            f"count divides 6 (1, 2, 3 or 6), got {n_devices}."
        )
    _spmd_mesh = mesh
    halo._halo_backend = "spmd"
    halo._spmd_mesh = mesh

    if force_allgather is None:
        use_pp = select_exchange_backend(n, nlev, n_devices)
    else:
        use_pp = not force_allgather
    _use_ppermute = use_pp
    backend_name = "ppermute" if use_pp else "all_gather(DIAGNOSTIC)"
    logger.info(
        "SPMD halo backend activated (mesh=%s, %d devices, "
        "faces/shard=%d, n=%d, nlev=%d, exchange=%s)",
        mesh.axis_names, n_devices, 6 // n_devices, n, nlev, backend_name,
    )

    # Pre-warm the exchange kernel cache so the factories are never
    # called from inside a JIT trace.  The first call to _get_exchange
    # for a given key triggers jnp.asarray in the factory outer body;
    # if that call happens while _step_cell_centre is being traced, the
    # resulting jax.Array gets captured in the _exchange closure and
    # stored in the module-level _cache — an UnexpectedTracerError
    # (production job 25211021 crashed after 5 min).  Calling all
    # relevant (ndim, halo, with_offsets, use_ppermute) combinations
    # here, outside any JIT scope, populates the cache before the first
    # compilation begins.  (Fix from bd072199 on ap/amip_upper_atm.)
    # halo=2 now warms BOTH flag values too (the public APIs honour the
    # module flag at halo=2 since the multiface ppermute kernel landed,
    # so a post-activation ``set_ppermute_default`` flip must still be
    # a cache hit inside JIT).
    for _ndim in (3, 4):
        for _pp in (False, True):
            for _halo in (1, 2):
                _get_exchange(mesh, _ndim, _pp, halo=_halo,
                              with_offsets=False)
                _get_exchange(mesh, _ndim, _pp, halo=_halo,
                              with_offsets=True)


def deactivate_spmd_halo_backend() -> None:
    """Revert to the default local halo backend."""
    global _spmd_mesh
    _spmd_mesh = None
    from legoesm.grids import halo
    halo._halo_backend = "local"
    halo._spmd_mesh = None


def get_spmd_mesh():
    """Return the active SPMD halo mesh (or ``None``).

    Set by :func:`activate_spmd_halo_backend`; cleared by
    :func:`deactivate_spmd_halo_backend`.  Dycore step functions call
    this when ``grids.halo.get_halo_backend() == "spmd"`` to obtain the
    :class:`jax.sharding.Mesh` for the packed halo exchanges.
    """
    return _spmd_mesh
