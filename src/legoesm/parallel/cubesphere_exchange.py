"""Explicit cubed-sphere halo exchange for multi-GPU SPMD.

Replaces the implicit cross-shard reads that ``pad_halo`` generates
under face-axis sharding with **explicit collective operations**
inside ``shard_map``.  Two collective backends are provided:

* **all_gather** (default): each device gathers all 6 faces, then
  locally extracts the 4 neighbor strips it needs.  Simple, correct,
  and sufficient for ≤6 GPUs at moderate resolution.

* **ppermute** (``use_ppermute=True``): 4 rounds of
  ``jax.lax.ppermute``, each moving one edge strip per device.
  Moves ~50× less data than all_gather (edge strips vs full faces),
  which matters at C192+ resolution on bandwidth-limited interconnects.

Both backends produce explicit HLO collectives (``all-gather`` or
``collective-permute``) that NCCL/ICI can schedule and pipeline,
unlike the implicit ``dynamic-slice`` pattern from standard sharding.

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
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from legoesm.grids.halo import CONNECTIVITY, WEST, EAST, SOUTH, NORTH

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

    Single-face SPMD analogue of :func:`legoesm.grids.halo._fill_corners_h2`.

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
    # propagate outward; matches ``halo._fill_corners_h2`` semantics.
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
# Backend A: all_gather  (simple, low-latency for ≤6 devices)
# ===================================================================

def _make_exchange_allgather(mesh, ndim, with_offsets=False):
    """Build a shard_map exchange using all_gather.

    When ``with_offsets`` is True the kernel takes a second
    ``interp_offsets`` argument shaped ``(6, 4, n)`` (replicated) and
    applies the per-edge 3-point Lagrange correction (via
    :func:`legoesm.grids.halo._interp_strip`) to each gathered strip.
    This restores numerical equivalence with the local ``_pad_halo_local``
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
        # any divisor of 6.  Iter-1's restriction in
        # ``make_sharded_step`` is left in place pending a final
        # ppermute multi-face refit (the all_gather kernel is the
        # default and is fully generalised here).
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
            from legoesm.grids.halo import _interp_strip

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
                    strip = _interp_strip(strip, offsets[global_face, e])
                halo_strips.append(strip)
            padded = _fill_halo_and_corners(padded, halo_strips, n)
            padded_faces.append(padded)

        return jnp.stack(padded_faces, axis=0)

    return _exchange


# ===================================================================
# Backend A2: all_gather for halo=2 (gather 2-cell-wide perimeter strips)
# ===================================================================

def _make_exchange_allgather_h2(mesh, ndim, with_offsets=False):
    """Build a shard_map exchange for halo=2 using a single all_gather of
    2-cell-wide perimeter strips.

    The volume per face is ``8 * n[, * C]`` cells (4 edges × 2 deep)
    versus the previous fall-through path which used the local h2 fill
    on a face-sharded array — a pattern that triggers XLA auto-gather of
    the full ``(6, n, n[, C])`` state.  The 2-strip allgather moves
    ``n/4`` × less data per device for typical ``n``.

    When ``with_offsets`` is True the kernel takes a replicated
    ``interp_offsets`` argument shaped ``(6, 4, 2, n)`` and applies
    :func:`legoesm.grids.halo._interp_strip` per (edge, depth) to each
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
            from legoesm.grids.halo import _interp_strip

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
                    # ``_interp_strip`` interpolates along axis 0 of its
                    # input, so we slice each depth as a (n[, C]) tensor.
                    strip_d0 = _interp_strip(strip[0], offsets[global_face, e, 0])
                    strip_d1 = _interp_strip(strip[1], offsets[global_face, e, 1])
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
    equivalence with ``_pad_halo_local(data, interp_offsets)``.  The
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
            from legoesm.grids.halo import _interp_strip

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
                # same convention the local-pad ``_pad_halo_local``
                # uses (halo.py:1078, ``interp_offsets[face, edge_idx]``).
                # ``recv_edge`` is traced, so we ``lax.dynamic_slice``
                # by indexing into the (4, n)-shaped face slice.
                offs_for_face = offsets[my_idx]  # (4, n)
                offs_for_edge = offs_for_face[recv_edge]  # (n,)
                received = _interp_strip(received, offs_for_edge)
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
# Public scalar exchange API
# ===================================================================

_cache: dict[tuple, object] = {}


def _get_exchange(mesh, ndim, use_ppermute, halo=1, with_offsets=False):
    """Get (or build and cache) a SPMD halo-exchange kernel.

    Parameters
    ----------
    mesh : jax.sharding.Mesh
        Face-axis mesh.
    ndim : int
        3 for scalar (6, n, n) inputs, 4 for (6, n, n, C) inputs.
    use_ppermute : bool
        When *True* and ``halo == 1``, use the bandwidth-optimal
        4-round ppermute backend.  Otherwise use the all_gather kernel.
        ``ppermute`` is currently halo=1 only; halo=2 always uses the
        2-strip all_gather kernel.  When ``with_offsets`` is True,
        ``ppermute`` is forced off (only the all_gather kernel applies
        ``interp_offsets``).
    halo : int
        Halo depth.  Supported: 1, 2.  Other values fall back to the
        local pad path; see :func:`explicit_pad_halo`.
    with_offsets : bool
        When True, build a kernel that takes a second ``interp_offsets``
        argument and applies the per-edge Lagrange correction.
    """
    key = (id(mesh), ndim, use_ppermute, halo, with_offsets)
    if key not in _cache:
        if with_offsets:
            if halo == 2:
                # halo=2 ppermute kernel does not exist — only the
                # 2-strip allgather supports halo=2.  ``use_ppermute``
                # is silently downgraded to False on this path; the
                # caller already passes ``False`` when halo=2.
                _cache[key] = _make_exchange_allgather_h2(
                    mesh, ndim, with_offsets=True,
                )
            elif use_ppermute:
                _cache[key] = _make_exchange_ppermute(
                    mesh, ndim, with_offsets=True,
                )
            else:
                _cache[key] = _make_exchange_allgather(
                    mesh, ndim, with_offsets=True,
                )
        elif halo == 2:
            _cache[key] = _make_exchange_allgather_h2(mesh, ndim)
        elif use_ppermute:
            _cache[key] = _make_exchange_ppermute(mesh, ndim)
        else:
            _cache[key] = _make_exchange_allgather(mesh, ndim)
    return _cache[key]


# Module-level flag: use ppermute by default?
_use_ppermute: bool = False

# Auto-selection threshold (bytes).  When the all_gather data volume
# per device exceeds this, ppermute is preferred.  Default 4 MB.
_AUTO_THRESHOLD_BYTES = int(
    float(os.environ.get("LEGOESM_SPMD_HALO_THRESHOLD_MB", "4")) * 1_048_576
)


def select_exchange_backend(
    n: int,
    nlev: int = 1,
    n_devices: int = 6,
    dtype_bytes: int = 4,
) -> bool:
    """Decide whether to use ppermute (True) or all_gather (False).

    Heuristic: all_gather moves O(6 * n^2 * nlev * dtype_bytes) per
    device.  ppermute moves O(4 * n * nlev * dtype_bytes).  When the
    all_gather volume exceeds the threshold, ppermute is better.

    The threshold is configurable via ``LEGOESM_SPMD_HALO_THRESHOLD_MB``
    (default 4 MB).

    Parameters
    ----------
    n : int
        Per-face spatial resolution (e.g. 48 for C48).
    nlev : int
        Number of vertical levels (1 for shallow water).
    n_devices : int
        Number of devices in the mesh.
    dtype_bytes : int
        Bytes per element (4 for float32, 8 for float64).

    Returns
    -------
    bool
        True if ppermute is recommended, False for all_gather.
    """
    allgather_bytes = 6 * n * n * nlev * dtype_bytes
    return allgather_bytes > _AUTO_THRESHOLD_BYTES


def set_ppermute_default(enabled: bool) -> None:
    """Switch the default collective backend.

    Parameters
    ----------
    enabled : bool
        ``True`` → use 4 rounds of ``ppermute`` (bandwidth-optimal).
        ``False`` → use ``all_gather`` (latency-optimal at low N).
    """
    global _use_ppermute
    _use_ppermute = enabled


def explicit_pad_halo(data, mesh, halo=1, interp_offsets=None):
    """Explicit 3D scalar exchange.  (6,n,n) → (6,n+2h,n+2h).

    Halo=1 uses ppermute or all_gather (auto-selected via the module
    flag), with ``interp_offsets`` Lagrange correction applied in
    either kernel when provided.  Halo=2 uses the 2-strip all_gather
    kernel (ppermute is currently halo=1 only).  Other halo depths
    fall back to the local-pad path.

    When ``interp_offsets`` is provided the SPMD kernel applies the
    per-edge 3-point Lagrange correction so the SPMD result matches
    ``_pad_halo_local(data, interp_offsets)`` bit-for-bit.
    """
    if halo == 2:
        if interp_offsets is None:
            return _get_exchange(mesh, 3, False, halo=2)(data)
        return _get_exchange(
            mesh, 3, False, halo=2, with_offsets=True,
        )(data, interp_offsets)
    if halo != 1:
        from legoesm.grids.halo import _pad_halo_local
        return _pad_halo_local(data, interp_offsets)
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
    # bypass the silent-corruption check.  ``_pad_halo_local_4d``
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
            return _get_exchange(mesh, 4, False, halo=2)(data)
        return _get_exchange(
            mesh, 4, False, halo=2, with_offsets=True,
        )(data, interp_offsets)
    if halo != 1:
        from legoesm.grids.halo import _pad_halo_local_4d
        return _pad_halo_local_4d(data, interp_offsets)
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
    canonical Lagrange-corrected halo path), the SPMD allgather kernel
    applies per-edge 3-point Lagrange correction to each gathered strip
    using the same offsets the unpacked ``pad_halo_4d`` would apply.
    Without this, the packed SPMD path silently bypasses
    ``interp_offsets`` while the unpacked ``pad_halo_4d`` path applies
    them.

    ``halo`` selects the SPMD exchange depth.  ``halo=1`` uses the
    1-strip allgather kernel (shape ``(6, n+2, n+2, C)`` per device);
    ``halo=2`` uses the 2-strip allgather kernel (shape
    ``(6, n+4, n+4, C)``) — the same kernel
    :func:`explicit_pad_halo_4d` selects when called with that depth.
    """
    if not fields:
        return []
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


def activate_spmd_halo_backend(mesh, n: int = 0, nlev: int = 1) -> None:
    """Switch the global halo backend to explicit SPMD exchange.

    When *n* (per-face resolution) is provided, auto-selects between
    all_gather and ppermute based on estimated data volume.

    Parameters
    ----------
    mesh : jax.sharding.Mesh
    n : int
        Per-face resolution for auto-selection (0 = skip auto-select).
    nlev : int
        Number of vertical levels.
    """
    global _spmd_mesh, _use_ppermute
    _spmd_mesh = mesh
    from legoesm.grids import halo
    halo._halo_backend = "spmd"
    halo._spmd_mesh = mesh

    # Auto-select ppermute vs all_gather based on data volume.
    # The ppermute kernel still assumes exactly one face per shard
    # (only the all_gather + all_gather_h2 kernels are multi-face,
    # iter-49); force all_gather when n_devices != 6 so 2- and
    # 3-device configs at high resolution stay correct.
    n_devices = len(mesh.devices.flat)
    if n_devices == 6 and n > 0:
        use_pp = select_exchange_backend(n, nlev, n_devices)
    else:
        use_pp = False
    _use_ppermute = use_pp
    backend_name = "ppermute" if use_pp else "all_gather"
    logger.info(
        "SPMD halo backend activated (mesh=%s, %d devices, "
        "n=%d, nlev=%d, exchange=%s)",
        mesh.axis_names, n_devices, n, nlev, backend_name,
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
    for _ndim in (3, 4):
        for _pp in (False, True):
            _get_exchange(mesh, _ndim, _pp, halo=1, with_offsets=False)
            _get_exchange(mesh, _ndim, _pp, halo=1, with_offsets=True)
        _get_exchange(mesh, _ndim, False, halo=2, with_offsets=False)
        _get_exchange(mesh, _ndim, False, halo=2, with_offsets=True)


def deactivate_spmd_halo_backend() -> None:
    """Revert to the default local halo backend."""
    global _spmd_mesh
    _spmd_mesh = None
    from legoesm.grids import halo
    halo._halo_backend = "local"
    halo._spmd_mesh = None
