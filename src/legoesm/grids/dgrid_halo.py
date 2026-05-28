"""Staggered D-grid vector halo for FV3 ``cross_face_du_proj`` (iter-1076).

This module ports the equator-equator portion of FV3's
``mpp_update_domains(u, v, DGRID_NE)`` cubed-sphere vector halo
exchange for staggered D-grid wind tendencies / fields:

- ``u_d`` shape ``(6, n, n+1, nlev)`` — D-grid u at v-edges
  (cell ``j``-edges; ``i`` stagger = ``n`` cells, ``j`` stagger = ``n+1``).
- ``v_d`` shape ``(6, n+1, n, nlev)`` — D-grid v at u-edges
  (cell ``i``-edges; ``i`` stagger = ``n+1``, ``j`` stagger = ``n``).

The cubed-sphere ``CONNECTIVITY`` has two edge-pair classes:

1. **Same-axis pairs** (i-edge ↔ i-edge OR j-edge ↔ j-edge): same
   strip orientation, same component (u→u, v→v).  Strip lengths
   match consistently across all 6 faces.  16 of 24 directed
   edges fall here.
2. **Axis-swap pairs** (i-edge ↔ j-edge): component swap (u↔v)
   required to match strip lengths.  8 of 24 directed edges:
   ``(1, S)↔(5, E)``, ``(1, N)↔(4, E)``, ``(3, S)↔(5, W)``,
   ``(3, N)↔(4, W)``.

iter-1076 implements (1) — the same-axis subset — bit-for-bit
faithful.  Axis-swap edges fall back to ``mode='edge'`` (matches
the legacy iter-1072 default-disable behavior) until iter-1077
adds the proper component swap with FV3 sign conventions.

Coverage:

- Faces 0-3 (equator): all 4 edges are same-axis → fully covered.
- Faces 4-5 (poles): N/S edges are same-axis; W/E edges are
  axis-swap → only N/S covered, W/E falls back to mode='edge'.

Net: 16/24 directed edges get faithful cross-face halos; 8/24 use
mode='edge'.  This improves the iter-370 ``cross_face_du_proj``
result on equator face boundaries (where the cube-imprint is
strongest in baroclinic-wave tests) while leaving pole-face W/E
halos approximate.  Tracked as iter-1077 follow-up.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from legoesm.grids.halo import CONNECTIVITY, WEST, EAST, SOUTH, NORTH


# ---------------------------------------------------------------------
# Classify each (face, edge) pair as same-axis vs axis-swap
# ---------------------------------------------------------------------

_I_EDGES = (WEST, EAST)  # edges that index along j-axis
_J_EDGES = (SOUTH, NORTH)  # edges that index along i-axis


def _is_axis_swap(face: int, edge: int) -> bool:
    """Return True if the (face, edge) connection swaps axes.

    Axis swap = E/W edge connects to N/S edge or vice versa.  At
    these connections, the cube-face geometry rotates 90°, so the
    u-component on one face maps to v on the other.
    """
    nbr_face, nbr_edge, _ = CONNECTIVITY[face][edge]
    same_axis_i = edge in _I_EDGES and nbr_edge in _I_EDGES
    same_axis_j = edge in _J_EDGES and nbr_edge in _J_EDGES
    return not (same_axis_i or same_axis_j)


# ---------------------------------------------------------------------
# Halo table builder for non-square staggered scalar fields
# ---------------------------------------------------------------------

def _build_dgrid_scalar_halo_table_h1(
    n_i: int, n_j: int, *, axis_swap_skip: bool = True,
) -> tuple:
    """Build halo=1 source/destination tables for a non-square staggered
    scalar field with cubed-sphere connectivity.

    The field has shape ``(6, n_i, n_j)``.  Strip extents per edge:

    - WEST/EAST: along j-axis, length ``n_j``
    - SOUTH/NORTH: along i-axis, length ``n_i``

    For the table to be valid, source-edge strip length must equal
    destination-edge strip length.  At axis-swap edges (i-edge ↔
    j-edge) this requires ``n_i == n_j`` (i.e., square data) — the
    same-component halo doesn't preserve strip length.  When
    ``axis_swap_skip=True``, axis-swap edges are skipped (the
    destination halo cells are left at the ``jnp.pad`` default,
    typically zero; caller is expected to overwrite via component
    swap or mode='edge' fallback).

    Parameters
    ----------
    n_i, n_j : int
        Face-local i- and j-axis cell counts.  For u_d: (n, n+1).
        For v_d: (n+1, n).
    axis_swap_skip : bool
        If True, skip axis-swap edges (default).  If False, raise
        when n_i != n_j (preserved for unit-testing the table
        builder against the legacy square-only path).

    Returns
    -------
    (src_f, src_i, src_j, dst_f, dst_i, dst_j) : tuple of numpy arrays
        Each shape ``(n_entries,)``.  ``n_entries`` ≤ ``24 * max(n_i, n_j)``;
        equal when all edges included, less when axis-swap edges
        are skipped.
    """
    entries = []  # list of (sf, si, sj, df, di, dj)

    for face in range(6):
        for edge in (WEST, EAST, SOUTH, NORTH):
            nbr_face, nbr_edge, is_reversed = CONNECTIVITY[face][edge]

            if _is_axis_swap(face, edge):
                if axis_swap_skip:
                    continue
                if n_i != n_j:
                    raise ValueError(
                        f"Axis-swap edge (face={face}, edge={edge}) requires "
                        f"n_i == n_j when axis_swap_skip=False; "
                        f"got n_i={n_i}, n_j={n_j}.  Use component-swap "
                        f"halo for staggered fields with n_i != n_j."
                    )

            # Strip length: matches destination edge's natural extent.
            if edge in _I_EDGES:
                strip_len = n_j
            else:  # j-edge
                strip_len = n_i

            for k in range(strip_len):
                # Source position on neighbor's edge.  k indexes the
                # destination strip; the source-side index is
                # reversed if connectivity says so.
                src_k = (strip_len - 1 - k) if is_reversed else k

                # Source cell from neighbor's edge.  Neighbor edge
                # orientation matches destination orientation when
                # not axis-swap (which we've guaranteed above).
                if nbr_edge == WEST:
                    sf, si, sj = nbr_face, 0, src_k
                elif nbr_edge == EAST:
                    sf, si, sj = nbr_face, n_i - 1, src_k
                elif nbr_edge == SOUTH:
                    sf, si, sj = nbr_face, src_k, 0
                else:  # NORTH
                    sf, si, sj = nbr_face, src_k, n_j - 1

                # Destination position in padded (n_i+2, n_j+2) array.
                if edge == WEST:
                    df, di, dj = face, 0, k + 1
                elif edge == EAST:
                    df, di, dj = face, n_i + 1, k + 1
                elif edge == SOUTH:
                    df, di, dj = face, k + 1, 0
                else:  # NORTH
                    df, di, dj = face, k + 1, n_j + 1

                entries.append((sf, si, sj, df, di, dj))

    total = len(entries)
    arr = np.array(entries, dtype=np.int32).T  # shape (6, total)
    return tuple(arr)  # (src_f, src_i, src_j, dst_f, dst_i, dst_j)


_dgrid_halo_table_cache: dict[tuple[int, int], tuple] = {}


def _get_dgrid_halo_table_h1(n_i: int, n_j: int) -> tuple:
    """Cached non-square staggered halo table (skip axis-swap edges)."""
    n_i = int(n_i)
    n_j = int(n_j)
    key = (n_i, n_j)
    if key not in _dgrid_halo_table_cache:
        _dgrid_halo_table_cache[key] = _build_dgrid_scalar_halo_table_h1(
            n_i, n_j, axis_swap_skip=True,
        )
    return _dgrid_halo_table_cache[key]


# ---------------------------------------------------------------------
# Public halo function for staggered scalar fields
# ---------------------------------------------------------------------

def pad_halo_dgrid_scalar_4d(
    data: jax.Array,
    *,
    axis_swap_fill: str = "edge",
) -> jax.Array:
    """Halo a non-square staggered scalar field on the cubed sphere.

    Handles the same-axis (non-axis-swap) edges with FV3-faithful
    cross-face source.  Axis-swap edges fall back to
    ``axis_swap_fill`` (default ``"edge"`` — replicate from the
    nearest interior cell, matching the legacy iter-370
    ``mode='edge'`` fallback).

    Parameters
    ----------
    data : jax.Array, shape ``(6, n_i, n_j, nlev)``
        Staggered 4D field.  For D-grid u: ``n_i=n, n_j=n+1``.
        For D-grid v: ``n_i=n+1, n_j=n``.  Square data
        (``n_i == n_j``) is rejected — the caller should use
        ``pad_halo_4d`` instead.
    axis_swap_fill : str, default ``"edge"``
        Strategy for axis-swap edges (8 of 24 directed edges):

        - ``"edge"``: replicate the nearest interior cell (faces
          4-5 W/E halos use the adjacent face-interior column).
          Matches the iter-370 legacy ``mode='edge'`` fallback.
        - ``"zero"``: leave at zero (for debugging only).

    Returns
    -------
    padded : jax.Array, shape ``(6, n_i+2, n_j+2, nlev)``
        Same dtype as input.

    Notes
    -----
    iter-1076 (this implementation): covers 16/24 directed edges
    bit-for-bit faithful.  iter-1077 (deferred) adds component
    swap with FV3 sign conventions to handle the remaining 8
    axis-swap edges fully.
    """
    if data.ndim != 4:
        raise ValueError(
            f"pad_halo_dgrid_scalar_4d expects 4D input, got "
            f"{data.ndim}D."
        )
    if data.shape[0] != 6:
        raise ValueError(
            f"pad_halo_dgrid_scalar_4d expects 6 faces on axis 0, "
            f"got shape {tuple(data.shape)}."
        )
    if data.shape[1] == data.shape[2]:
        raise ValueError(
            f"pad_halo_dgrid_scalar_4d is for NON-square staggered "
            f"data; got square shape {tuple(data.shape)}.  Use "
            f"pad_halo_4d for square data."
        )
    if axis_swap_fill not in ("edge", "zero"):
        raise ValueError(
            f"axis_swap_fill must be 'edge' or 'zero', got "
            f"{axis_swap_fill!r}."
        )

    n_i, n_j = data.shape[1], data.shape[2]

    # Start with edge-padded array.  This gives correct values at
    # axis-swap halo positions when axis_swap_fill == 'edge' (cells
    # adjacent to the boundary are replicated outward).  For
    # axis_swap_fill == 'zero', we want zero halos, which a plain
    # constant pad gives.
    if axis_swap_fill == "edge":
        padded = jnp.pad(
            data, ((0, 0), (1, 1), (1, 1), (0, 0)), mode="edge",
        )
    else:  # zero
        padded = jnp.pad(
            data, ((0, 0), (1, 1), (1, 1), (0, 0)),
        )

    # Overwrite the same-axis halos with cubed-sphere cross-face
    # sources via the precomputed table.
    src_f, src_i, src_j, dst_f, dst_i, dst_j = _get_dgrid_halo_table_h1(
        n_i, n_j,
    )
    if src_f.size > 0:
        values = data[src_f, src_i, src_j]  # (n_entries, nlev)
        padded = padded.at[dst_f, dst_i, dst_j].set(values)

    return padded
