"""Staggered D-grid vector halo for FV3 ``cross_face_du_proj``.

Ports FV3's ``mpp_update_domains(u, v, DGRID_NE)`` cubed-sphere
vector halo for staggered D-grid wind components:

- ``u_d`` shape ``(6, n, n+1, nlev)`` — u at v-edges
- ``v_d`` shape ``(6, n+1, n, nlev)`` — v at u-edges

Two edge-pair classes in cubed-sphere ``CONNECTIVITY``:

1. **Same-axis pairs** (16/24 directed): same-component cross-face
   halo (u→u, v→v).  Equator E-W + face 0,2 N/S + face 4,5 S/N.
2. **Axis-swap pairs** (8/24): ``(1, S)↔(5, E)``,
   ``(1, N)↔(4, E)``, ``(3, S)↔(5, W)``, ``(3, N)↔(4, W)``.
   Component swap (u↔v) with face-pair-specific signs derived
   from tangent-vector matching at the boundary corner in 3D
   Cartesian.

Public entries:

- ``pad_halo_dgrid_scalar_4d(data, axis_swap_fill='edge')``
  (iter-1076) — non-square staggered scalar halo.  Same-axis
  edges via precomputed table; axis-swap edges fall back to
  ``axis_swap_fill`` ('edge' replicates, 'zero' leaves at 0).
- ``pad_halo_dgrid_vector_4d(u_d, v_d)``
  (iter-1078; value-level semantics fixed 2026-08-04) — full 24/24
  staggered vector halo with axis-swap component swap (single-device
  input ``(6, ...)``).  Node-axis ghosts source the neighbour line one
  INWARD of the shared seam; the four half-turn seams negate both
  components.  Certified vs ``analytic_swcore_state`` halos
  (fv3_recon/transplant2_9311777.log caught the pre-fix seam-line
  copies and missing half-turn signs).
- ``pad_halo_dgrid_scalar_pair_4d(u_like, v_like, axis_swap_sign=+1)``
  (2026-07-31) — FV3 SCALAR_PAIR/CGRID_NE staggered METRIC halo:
  dyc↔dxc and sina_v↔sina_u with a common ``+1``, cosa_v↔cosa_u
  with ``-1``, plus the C-grid corner fills.
- ``pad_halo_dgrid_sg_slots_4d(sin_sg, cos_sg)``
  (2026-08-01) — raw four-SLOT cell-metric halo for
  ``divergence_corner``, which reads MIXED raw slots at panel
  boundaries.  8 quarter-turn seams permute the slot channel and
  negate ``cos``; 4 half-turn seams permute without negating; the
  other 12 are identity copies.  The two outward-facing slots of
  each diagonal ghost cell stay POISONED at
  ``SG_TINY_NUMBER``/``SG_BIG_NUMBER``, as FV3 leaves them.
  PERF: currently emits ~120 source-level ``.at[].set()`` chains
  (96 side writes + 24 diagonal).  Fine while it is unwired; build
  a cached batched gather/scatter before putting it in a
  per-timestep hot path (codex review P3).
- ``pad_halo_dgrid_vector_4d_mpi(u_d, v_d, topology)``
  (iter-1083) — MPI-aware variant via batched-per-peer
  ``mpi4jax.sendrecv``.  Rank-local input ``(n_local, ...)``.
- ``pad_halo_dgrid_vector_4d_replicated_mpi(u_d, v_d, topology)``
  (iter-1083) — wrapper for full ``(6, ...)`` replicated state
  (canonical cubed-sphere MPI mode).

Validation: ``test_dgrid_halo_iter1076.py`` (27),
``test_dgrid_vector_halo_iter1078.py`` (12),
``tests/distributed/test_mpi_dgrid_vector_halo_iter1083.py`` (1
strict bit-for-bit at np ∈ {2, 3, 6}).
"""
from __future__ import annotations


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


# Same-axis seams whose transverse axis is REVERSED are 180-degree relative
# rotations of the neighbouring face frame (CONNECTIVITY: exactly
# (2,S)<->(5,S) and (2,N)<->(4,N), i.e. dest edges (2,S),(2,N),(4,N),(5,S)).
# There the neighbour's +i/+j basis is the NEGATION of this face's continued
# basis, so COVARIANT vector components cross the seam with sign -1 on BOTH
# components (a quarter turn maps (e1,e2)->(e2,-e1); composing two gives
# (-e1,-e2)).  Scalar metrics (lengths, sina) do NOT flip -- this overlay
# belongs to the VECTOR exchange only, never to the scalar/metric-pair path.
# Certified value-level vs analytic_swcore_state halos
# (fv3_recon/transplant2_9311777.log): F=2 v S got=-2.500000000000004
# want=+2.500000000000004 (exact negation -- position already right, sign
# missing, line 260); F=5 u S got=+5.0000 want=-4.9824 (sign PLUS the
# node-axis source row, line 248).
_SAME_AXIS_FLIP_SEAMS: tuple[tuple[int, int], ...] = tuple(
    (face, edge)
    for face in range(6)
    for edge in (WEST, EAST, SOUTH, NORTH)
    if (not _is_axis_swap(face, edge)) and CONNECTIVITY[face][edge][2]
)


def _negate_side_ghost_line(padded: jax.Array, face: int, edge: int) -> jax.Array:
    """Negate one depth-1 side ghost line (excluding the diagonal cells).

    Covers exactly the cells the same-axis halo table writes (destination
    positions ``k + 1`` over the strip length, i.e. ``[1:-1]`` of the ghost
    line) -- the diagonal corner cells are never table-written and stay
    untouched.
    """
    if edge == WEST:
        return padded.at[face, 0, 1:-1, :].multiply(-1.0)
    if edge == EAST:
        return padded.at[face, -1, 1:-1, :].multiply(-1.0)
    if edge == SOUTH:
        return padded.at[face, 1:-1, 0, :].multiply(-1.0)
    return padded.at[face, 1:-1, -1, :].multiply(-1.0)


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

    # D-staggered depth-1 SOURCE semantics (2026-08-04 transplant fix).
    # On a NODE-staggered axis (n+1 lines vs the transverse n) the OUTERMOST
    # line IS the shared seam -- BOTH faces store that physical line -- so the
    # depth-1 ghost must carry the neighbour's line one INWARD of the seam.
    # On a CELL-staggered axis the outermost line is the correct source.
    # Certified value-level against the phase-4a gridstruct /
    # analytic_swcore_state halos (fv3_recon/transplant2_9311777.log):
    # the old seam-line copy read the SHARED-edge value on every node-axis
    # strip INCLUDING identity seams, e.g. create face 0 u SOUTH ghost
    # got=+3.5355 (shared edge) where FV3 truth holds +3.2270 (one row
    # beyond); same class drove the PAIR dyc/dxc failures (uniform
    # 2.667e-02 rel).  Square input (legacy test path): both flags False,
    # behaviour unchanged.
    i_node = n_i == n_j + 1   # i-axis node-staggered (v-like / dxc-like)
    j_node = n_j == n_i + 1   # j-axis node-staggered (u-like / dyc-like)
    src_i_lo = 1 if i_node else 0
    src_i_hi = n_i - 2 if i_node else n_i - 1
    src_j_lo = 1 if j_node else 0
    src_j_hi = n_j - 2 if j_node else n_j - 1

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
                # not axis-swap (which we've guaranteed above).  The
                # normal-axis source index steps one inward of the seam
                # on a node-staggered axis (see block above).
                if nbr_edge == WEST:
                    sf, si, sj = nbr_face, src_i_lo, src_k
                elif nbr_edge == EAST:
                    sf, si, sj = nbr_face, src_i_hi, src_k
                elif nbr_edge == SOUTH:
                    sf, si, sj = nbr_face, src_k, src_j_lo
                else:  # NORTH
                    sf, si, sj = nbr_face, src_k, src_j_hi

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

    len(entries)
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
    iter-1076 (this implementation): same-axis 16/24 directed
    edges via the cross-face table; axis-swap 8/24 edges use
    ``axis_swap_fill``.  iter-1078's ``pad_halo_dgrid_vector_4d``
    handles all 24 edges (calls this function for same-axis +
    overwrites axis-swap with component swap + half-turn signs).

    2026-08-04 depth fix: on a component's NODE-staggered axis the
    same-axis source is the neighbour's line one INWARD of the shared
    seam (the outermost node line IS the seam).  The pre-fix seam-line
    copy was caught value-level by the transplant probe on every
    node-axis strip including identity seams
    (fv3_recon/transplant2_9311777.log).  NOTE: same-axis strips are
    UNSIGNED here — covariant vectors additionally need the half-turn
    sign, which lives in ``pad_halo_dgrid_vector_4d`` only.
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


# =====================================================================
# iter-1078: DGRID_NE vector halo with axis-swap component swap
# =====================================================================

_AXIS_SWAP_TABLE = {
    (1, NORTH): (4, EAST,  False, +1, -1),
    (4, EAST):  (1, NORTH, False, -1, +1),
    (1, SOUTH): (5, EAST,  True,  -1, +1),
    (5, EAST):  (1, SOUTH, True,  +1, -1),
    (3, NORTH): (4, WEST,  True,  -1, +1),
    (4, WEST):  (3, NORTH, True,  +1, -1),
    (3, SOUTH): (5, WEST,  False, +1, -1),
    (5, WEST):  (3, SOUTH, False, -1, +1),
}


def _build_axis_swap_tables_h1(n):
    n_i_u, n_j_u = n, n + 1
    n_i_v, n_j_v = n + 1, n
    u_dst_f, u_dst_i, u_dst_j = [], [], []
    v_src_f, v_src_i, v_src_j = [], [], []
    u_dst_signs = []
    v_dst_f, v_dst_i, v_dst_j = [], [], []
    u_src_f, u_src_i, u_src_j = [], [], []
    v_dst_signs = []
    # Depth-1 source rows (2026-08-04 transplant fix, same rule as the
    # same-axis table): the SOURCE component's line along the seam steps one
    # INWARD when its normal axis is node-staggered, because the outermost
    # node line is the shared seam both faces already store.
    #   v source (feeds u ghosts): i-axis NODE (n+1) -> W/E inward;
    #                              j-axis cell -> S/N outermost.
    #   u source (feeds v ghosts): j-axis NODE (n+1) -> S/N inward;
    #                              i-axis cell -> W/E outermost.
    # Log cross-checks (fv3_recon/transplant2_9311777.log):
    #   F=1 u S <- (5,E) v: got=+2.5000 (seam node i=n) want=+2.2818
    #     (i=n-1) -- sign already right, row off by one (line 232);
    #   F=4 v W <- (3,N) u: got=-3.5355 want=-3.8192 (line 214);
    #   F=4 u W <- (3,N) v at cell row j=n-1: max_abs=0.0 (line 242) --
    #     cell-axis rows were already correct and are UNCHANGED here.
    v_src_w, v_src_e = 1, n_i_v - 2          # node axis: one inward
    v_src_s, v_src_n = 0, n_j_v - 1          # cell axis: outermost
    u_src_w, u_src_e = 0, n_i_u - 1          # cell axis: outermost
    u_src_s, u_src_n = 1, n_j_u - 2          # node axis: one inward
    for (face, edge), (nbr_face, nbr_edge, is_rev, sign_uv, sign_vu) in _AXIS_SWAP_TABLE.items():
        u_strip_len = n_j_u if edge in _I_EDGES else n_i_u
        v_strip_len = n_j_v if edge in _I_EDGES else n_i_v
        for k in range(u_strip_len):
            k_src = (u_strip_len - 1 - k) if is_rev else k
            if nbr_edge == WEST: sf, si, sj = nbr_face, v_src_w, k_src
            elif nbr_edge == EAST: sf, si, sj = nbr_face, v_src_e, k_src
            elif nbr_edge == SOUTH: sf, si, sj = nbr_face, k_src, v_src_s
            else: sf, si, sj = nbr_face, k_src, v_src_n
            if edge == WEST: df, di, dj = face, 0, k + 1
            elif edge == EAST: df, di, dj = face, n_i_u + 1, k + 1
            elif edge == SOUTH: df, di, dj = face, k + 1, 0
            else: df, di, dj = face, k + 1, n_j_u + 1
            u_dst_f.append(df); u_dst_i.append(di); u_dst_j.append(dj)
            v_src_f.append(sf); v_src_i.append(si); v_src_j.append(sj)
            u_dst_signs.append(sign_uv)
        for k in range(v_strip_len):
            k_src = (v_strip_len - 1 - k) if is_rev else k
            if nbr_edge == WEST: sf, si, sj = nbr_face, u_src_w, k_src
            elif nbr_edge == EAST: sf, si, sj = nbr_face, u_src_e, k_src
            elif nbr_edge == SOUTH: sf, si, sj = nbr_face, k_src, u_src_s
            else: sf, si, sj = nbr_face, k_src, u_src_n
            if edge == WEST: df, di, dj = face, 0, k + 1
            elif edge == EAST: df, di, dj = face, n_i_v + 1, k + 1
            elif edge == SOUTH: df, di, dj = face, k + 1, 0
            else: df, di, dj = face, k + 1, n_j_v + 1
            v_dst_f.append(df); v_dst_i.append(di); v_dst_j.append(dj)
            u_src_f.append(sf); u_src_i.append(si); u_src_j.append(sj)
            v_dst_signs.append(sign_vu)
    return (
        np.array(u_dst_f, dtype=np.int32), np.array(u_dst_i, dtype=np.int32), np.array(u_dst_j, dtype=np.int32),
        np.array(v_src_f, dtype=np.int32), np.array(v_src_i, dtype=np.int32), np.array(v_src_j, dtype=np.int32),
        np.array(u_dst_signs, dtype=np.int32),
        np.array(v_dst_f, dtype=np.int32), np.array(v_dst_i, dtype=np.int32), np.array(v_dst_j, dtype=np.int32),
        np.array(u_src_f, dtype=np.int32), np.array(u_src_i, dtype=np.int32), np.array(u_src_j, dtype=np.int32),
        np.array(v_dst_signs, dtype=np.int32),
    )


_axis_swap_table_cache = {}


def _get_axis_swap_tables_h1(n):
    n = int(n)
    if n not in _axis_swap_table_cache:
        _axis_swap_table_cache[n] = _build_axis_swap_tables_h1(n)
    return _axis_swap_table_cache[n]


# --- FV3 raw sin_sg/cos_sg SLOT halo (2026-08-01) ---------------------------
# ``divergence_corner`` reads MIXED RAW slots at panel boundaries -- Fortran
# ``(j-1,4)+(j,2)`` for ``uf`` and ``(i-1,3)+(i,1)`` for ``vf``
# (sw_core.F90:2187-2207) -- NOT staggered sina/cosa.  So neither the scalar
# helper (which edge-replicates the eight axis-swap seams) nor the metric-pair
# helper (two inputs + one common sign cannot express a four-slot permutation)
# can supply the ghost values it needs.
#
# Slot order is (W, S, E, N) = (0, 1, 2, 3), matching sin_sg[..., k].
# Derived against the oracle; the eight quarter-turn rows below agree row-for-row
# with this package's own CONNECTIVITY table.
#
# CAVEAT (historical) — CLOSED 2026-08-04: the face-ID/slot mapping was
# certified INDEPENDENTLY of CONNECTIVITY by the transplant probe
# (fv3_recon/probe_transplant.py) against ``build_fv3_native_gridstruct``'s
# phase-4a bit-certified sin_sg/cos_sg halos: all 24 seams x 4 slots exact
# to fp roundoff (TRANSPLANT_SG=PASS, fv3_recon/transplant2_9311777.log).

# dest (W,S,E,N) <- source slots, as an index array: dst k takes src perm[k].
_SG_PERM_SENW = (1, 2, 3, 0)     # "(S,E,N,W)"
_SG_PERM_NWSE = (3, 0, 1, 2)     # "(N,W,S,E)"
_SG_PERM_HALF = (2, 3, 0, 1)     # "(E,N,W,S)" -- the four half-turn seams

# (dest_face, dest_edge) -> slot permutation, for the eight quarter-turn seams.
# sin sign +1, cos sign -1 on every one of these.
_SG_QUARTER_TURN: dict[tuple[int, int], tuple[int, ...]] = {
    (1, NORTH): _SG_PERM_SENW,   # <- 4:E
    (4, EAST):  _SG_PERM_NWSE,   # <- 1:N
    (1, SOUTH): _SG_PERM_NWSE,   # <- 5:E   (reversed)
    (5, EAST):  _SG_PERM_SENW,   # <- 1:S   (reversed)
    (3, NORTH): _SG_PERM_NWSE,   # <- 4:W   (reversed)
    (4, WEST):  _SG_PERM_SENW,   # <- 3:N   (reversed)
    (3, SOUTH): _SG_PERM_SENW,   # <- 5:W
    (5, WEST):  _SG_PERM_NWSE,   # <- 3:S
}

# The four half-turn seams: same permutation, and cos does NOT flip sign.
_SG_HALF_TURN: frozenset[tuple[int, int]] = frozenset({
    (2, SOUTH), (2, NORTH), (4, NORTH), (5, SOUTH),
})

# FV3 leaves the two outward-facing slots of each diagonal ghost cell INVALID on
# purpose (fv_grid_utils.F90:51).  We reproduce that rather than inventing
# values, so a consumer that reads them gets an obviously wrong number instead of
# a plausible one.
SG_TINY_NUMBER = 1.0e-8      # sin slots
SG_BIG_NUMBER = 1.0e8        # cos slots


def pad_halo_dgrid_sg_slots_4d(sin_sg, cos_sg):
    """Halo the raw four-slot ``sin_sg``/``cos_sg`` cell metrics.

    Parameters
    ----------
    sin_sg, cos_sg : (6, n, n, 4)
        Raw sub-grid metrics, slot order (W, S, E, N).

    Returns
    -------
    (6, n+2, n+2, 4) each, with physical cells at ``[1:-1, 1:-1]``.

    Notes
    -----
    Identity seams are plain copies; the eight quarter-turn seams permute the
    slot channel and negate ``cos``; the four half-turn seams permute without
    negating.  The four diagonal ghost cells get FV3's two explicit
    inward-facing assignments (fv_grid_utils.F90:580 SW/NW, :609 SE/NE); their
    other two slots are left poisoned at
    :data:`SG_TINY_NUMBER` / :data:`SG_BIG_NUMBER`.
    """
    if sin_sg.ndim != 4 or cos_sg.ndim != 4:
        raise ValueError(
            "pad_halo_dgrid_sg_slots_4d requires (6, n, n, 4) inputs; got "
            f"{tuple(sin_sg.shape)} and {tuple(cos_sg.shape)}")
    if sin_sg.shape != cos_sg.shape:
        raise ValueError(
            f"sin_sg {tuple(sin_sg.shape)} and cos_sg {tuple(cos_sg.shape)} "
            "must have identical shapes")
    if sin_sg.shape[0] != 6 or sin_sg.shape[3] != 4:
        raise ValueError(
            "expected 6 faces and 4 slots; got "
            f"{tuple(sin_sg.shape)}")
    n = sin_sg.shape[1]
    if sin_sg.shape[2] != n:
        raise ValueError(f"cells must be square; got {tuple(sin_sg.shape)}")

    # Identity baseline for the twelve identity seams: pad_halo_4d treats the
    # trailing slot axis as levels, which is exactly a per-slot scalar copy.
    from legoesm.grids.halo import pad_halo_4d
    sin_p = pad_halo_4d(sin_sg)
    cos_p = pad_halo_4d(cos_sg)

    # Override the twelve permuting seams.
    for face in range(6):
        for edge in (WEST, EAST, SOUTH, NORTH):
            key = (face, edge)
            quarter = key in _SG_QUARTER_TURN
            half = key in _SG_HALF_TURN
            if not (quarter or half):
                continue
            perm = _SG_QUARTER_TURN[key] if quarter else _SG_PERM_HALF
            cos_sign = -1.0 if quarter else 1.0
            nbr_face, nbr_edge, reversed_ = CONNECTIVITY[face][edge]

            strip_sin = _sg_neighbour_strip(sin_sg, nbr_face, nbr_edge,
                                            reversed_)
            strip_cos = _sg_neighbour_strip(cos_sg, nbr_face, nbr_edge,
                                            reversed_)
            for dst_slot in range(4):
                src_slot = perm[dst_slot]
                sin_p = _sg_write_ghost(sin_p, face, edge, dst_slot,
                                        strip_sin[..., src_slot])
                cos_p = _sg_write_ghost(cos_p, face, edge, dst_slot,
                                        cos_sign * strip_cos[..., src_slot])

    return _sg_fill_diagonals(sin_p, cos_p, n)


def _sg_neighbour_strip(field, nbr_face, nbr_edge, reversed_):
    """The neighbour's outermost cell line along ``nbr_edge``, (n, 4)."""
    if nbr_edge == WEST:
        strip = field[nbr_face, 0, :, :]
    elif nbr_edge == EAST:
        strip = field[nbr_face, -1, :, :]
    elif nbr_edge == SOUTH:
        strip = field[nbr_face, :, 0, :]
    else:                                    # NORTH
        strip = field[nbr_face, :, -1, :]
    return strip[::-1, :] if reversed_ else strip


def _sg_write_ghost(padded, face, edge, slot, values):
    """Write one ghost line (excluding the diagonal cells) for one slot."""
    if edge == WEST:
        return padded.at[face, 0, 1:-1, slot].set(values)
    if edge == EAST:
        return padded.at[face, -1, 1:-1, slot].set(values)
    if edge == SOUTH:
        return padded.at[face, 1:-1, 0, slot].set(values)
    return padded.at[face, 1:-1, -1, slot].set(values)


def _sg_fill_diagonals(sin_p, cos_p, n):
    """FV3's post-fill_ghost diagonal repair (fv_grid_utils.F90:580, :609).

    Two inward-facing slots per diagonal ghost cell are assigned; the other two
    are left POISONED, as FV3 leaves them, so nobody reads a plausible-looking
    wrong value.
    """
    W, S, E, N = 0, 1, 2, 3
    L = n + 1
    for p_is_sin, p in ((True, sin_p), (False, cos_p)):
        poison = SG_TINY_NUMBER if p_is_sin else SG_BIG_NUMBER
        p = p.at[:, 0, 0, :].set(poison)
        p = p.at[:, L, 0, :].set(poison)
        p = p.at[:, L, L, :].set(poison)
        p = p.at[:, 0, L, :].set(poison)
        # SW
        p = p.at[:, 0, 0, E].set(p[:, 0, 1, S])
        p = p.at[:, 0, 0, N].set(p[:, 1, 0, W])
        # SE
        p = p.at[:, L, 0, W].set(p[:, L, 1, S])
        p = p.at[:, L, 0, N].set(p[:, L - 1, 0, E])
        # NE
        p = p.at[:, L, L, W].set(p[:, L, L - 1, N])
        p = p.at[:, L, L, S].set(p[:, L - 1, L, E])
        # NW
        p = p.at[:, 0, L, E].set(p[:, 0, L - 1, N])
        p = p.at[:, 0, L, S].set(p[:, 1, L, W])
        if p_is_sin:
            sin_p = p
        else:
            cos_p = p
    return sin_p, cos_p


# --- FV3 cell-centre metric halo (2026-08-04) --------------------------------
# ``d2a2c_vect`` computes the A-grid contravariant ua/va over the FULL data
# domain (Fortran ``is-1-id .. ie+1+id`` = TWO ghost rings, sw_core.F90:3513)
# from utmp/vtmp and the CELL-CENTRE metrics cosa_s/rsin2, whose ghost values
# in FV3 come from the gridstruct halo: fv_grid_utils evaluates them locally
# at ghost positions from the mpp-exchanged grid geometry.  For the same
# physical cell seen across a seam, the tangent pair relabels as
# (e_i, e_j) -> (e_j, -e_i) per quarter turn, so cos(angle) picks up a sign
# while sin-even quantities copy through — the phase-2B rot90 remap law
# (cos-type odd, sin/rsin even).  Edge-replicating these pads instead (the
# pre-2026-08-04 behaviour) feeds d2a2c's ghost-ring ua/va with wrong metric
# values, which ``divergence_corner`` then consumes at panel boundaries.


def pad_halo_dgrid_cell_scalar_4d(
    q: jax.Array,
    *,
    cos_type: bool = False,
    halo: int = 2,
) -> jax.Array:
    """Halo a cell-centre scalar metric (``cosa_s``/``rsin2``-like).

    Parameters
    ----------
    q : (6, n, n, nlev)
        Cell-centre metric field.
    cos_type : bool
        ``True`` for quantities that transform like ``e_i . e_j``
        (``cosa_s`` = ``cos_sg[..., 4]``): ghost strips across the eight
        quarter-turn seams are negated.  ``False`` for even quantities
        (``rsin2``, ``sin_sg``-center): plain copies everywhere.
    halo : int
        Ghost rings per side (default 2, matching Fortran's UA/VA span).

    Returns
    -------
    (6, n+2*halo, n+2*halo, nlev) with physical cells at
    ``[halo:-halo, halo:-halo]``.  The four ``halo x halo`` diagonal corner
    blocks are POISONED at :data:`SG_BIG_NUMBER`: FV3 fills cell-centre
    corner regions with its fill_corners AGRID convention, which no
    consumer of THIS helper reads (divergence_corner's boundary forms mask
    them out; edge_interpolate4 stencils stay on physical rows), so an
    obviously wrong number is safer than an unverified convention.
    """
    if q.ndim != 4:
        raise ValueError(
            f"pad_halo_dgrid_cell_scalar_4d requires (6, n, n, nlev); got "
            f"{tuple(q.shape)}")
    if q.shape[0] != 6 or q.shape[1] != q.shape[2]:
        raise ValueError(
            f"expected 6 square faces; got {tuple(q.shape)}")

    from legoesm.grids.halo import pad_halo_4d
    q_pad = pad_halo_4d(q, halo=halo)

    if cos_type:
        # Sign overlay: pad_halo_4d already places the correct source CELLS
        # (identity/reversal per seam); only the quarter-turn seams need the
        # tensorial cos sign.  Half-turn seams keep +1.
        n = q.shape[1]
        m = n + 2 * halo
        sign = np.ones((6, m, m, 1), dtype=np.float64)
        for (face, edge) in _SG_QUARTER_TURN:
            if edge == WEST:
                sign[face, :halo, :, :] = -1.0
            elif edge == EAST:
                sign[face, -halo:, :, :] = -1.0
            elif edge == SOUTH:
                sign[face, :, :halo, :] = -1.0
            else:  # NORTH
                sign[face, :, -halo:, :] = -1.0
        q_pad = q_pad * jnp.asarray(sign, dtype=q_pad.dtype)

    # Poison the diagonal corner blocks (both metrics classes).
    q_pad = q_pad.at[:, :halo, :halo, :].set(SG_BIG_NUMBER)
    q_pad = q_pad.at[:, -halo:, :halo, :].set(SG_BIG_NUMBER)
    q_pad = q_pad.at[:, -halo:, -halo:, :].set(SG_BIG_NUMBER)
    q_pad = q_pad.at[:, :halo, -halo:, :].set(SG_BIG_NUMBER)
    return q_pad


# --- FV3 SCALAR_PAIR / CGRID_NE staggered-metric halo (2026-07-31) ---
# Companion to ``pad_halo_dgrid_vector_4d``.  The corner-divergence routine
# needs the SEAM METRICS (dyc/dxc, sina_v/sina_u, cosa_v/cosa_u) to cross
# panels the same way the winds do; leaving them edge-replicated while the
# winds use a real DGRID_NE halo is an INCONSISTENT pairing that measurably
# broke DCMIP TC1 (PASS -> BLOWUP step 3950).  ``pad_halo_dgrid_scalar_4d``
# cannot serve: it deliberately edge-replicates on exactly the eight
# axis-swapping seams (see its ``axis_swap_fill`` argument).  Upstream FV3
# exchanges dxc/dyc as SCALAR_PAIR, CGRID_NE and then fills C-grid corners at
# grid-construction time (fv_grid_tools.F90:1071-1075); this compact CD-grid
# stores no halo ring, so the exchange is reconstructed here at use time.
# Unlike DGRID_NE winds, a metric pair carries ONE common sign, not
# component-specific ones: +1 for dyc/dxc and sina_v/sina_u, -1 for
# cosa_v/cosa_u.


def pad_halo_dgrid_scalar_pair_4d(
    u_like: jax.Array,
    v_like: jax.Array,
    *,
    axis_swap_sign: float = 1.0,
) -> tuple[jax.Array, jax.Array]:
    """Halo a paired C-grid staggered scalar metric field.

    Parameters
    ----------
    u_like : (6, n, n+1, nlev)
        ``dyc``, ``sina_v``, or ``cosa_v``-like field.
    v_like : (6, n+1, n, nlev)
        ``dxc``, ``sina_u``, or ``cosa_u``-like field.
    axis_swap_sign : {+1.0, -1.0}
        +1 for ``dyc/dxc`` and ``sina_v/sina_u``;
        -1 for ``cosa_v/cosa_u``.

    Returns
    -------
    u_padded : (6, n+2, n+3, nlev)
    v_padded : (6, n+3, n+2, nlev)

    Notes
    -----
    Same-axis edges are scalar copies.  On the eight axis-swapping
    edges, components exchange but use one common metric-pair sign,
    unlike DGRID_NE vector winds' component-specific signs.
    """
    if u_like.ndim != 4 or v_like.ndim != 4:
        raise ValueError(
            "pad_halo_dgrid_scalar_pair_4d requires 4D inputs; "
            f"got u_like.ndim={u_like.ndim}, v_like.ndim={v_like.ndim}"
        )
    if u_like.shape[0] != 6 or v_like.shape[0] != 6:
        raise ValueError("pad_halo_dgrid_scalar_pair_4d requires 6 faces")

    n = u_like.shape[1]
    if (
        u_like.shape[2] != n + 1
        or v_like.shape[1] != n + 1
        or v_like.shape[2] != n
        or u_like.shape[3] != v_like.shape[3]
    ):
        raise ValueError(
            "Expected u_like (6, n, n+1, nlev) and "
            "v_like (6, n+1, n, nlev); got "
            f"{tuple(u_like.shape)} and {tuple(v_like.shape)}"
        )
    if axis_swap_sign not in (-1.0, 1.0):
        raise ValueError(
            f"axis_swap_sign must be +1.0 or -1.0, got {axis_swap_sign!r}"
        )

    # Seed same-axis edges using the existing scalar table.  Its axis-swap
    # edge copies are overwritten below.
    u_padded = pad_halo_dgrid_scalar_4d(
        u_like, axis_swap_fill="edge",
    )
    v_padded = pad_halo_dgrid_scalar_4d(
        v_like, axis_swap_fill="edge",
    )

    (
        u_dst_f, u_dst_i, u_dst_j,
        v_src_f, v_src_i, v_src_j, _u_vector_signs,
        v_dst_f, v_dst_i, v_dst_j,
        u_src_f, u_src_i, u_src_j, _v_vector_signs,
    ) = _get_axis_swap_tables_h1(n)

    if u_dst_f.size > 0:
        u_padded = u_padded.at[u_dst_f, u_dst_i, u_dst_j].set(
            v_like[v_src_f, v_src_i, v_src_j]
            * jnp.asarray(axis_swap_sign, dtype=u_like.dtype)
        )
    if v_dst_f.size > 0:
        v_padded = v_padded.at[v_dst_f, v_dst_i, v_dst_j].set(
            u_like[u_src_f, u_src_i, u_src_j]
            * jnp.asarray(axis_swap_sign, dtype=v_like.dtype)
        )

    # FV3 fill_corners(..., CGRID=.true.), with X=v_like and Y=u_like.
    # Do this after side strips; these are paired C-grid corner values,
    # not mode='edge' diagonal copies.
    q_u = jnp.asarray(axis_swap_sign, dtype=u_like.dtype)
    q_v = jnp.asarray(axis_swap_sign, dtype=v_like.dtype)

    v_padded = v_padded.at[:, 0, 0, :].set(
        u_padded[:, 1, 0, :]
    )
    v_padded = v_padded.at[:, 0, n + 1, :].set(
        q_v * u_padded[:, 1, n + 2, :]
    )
    v_padded = v_padded.at[:, n + 2, 0, :].set(
        q_v * u_padded[:, n, 0, :]
    )
    v_padded = v_padded.at[:, n + 2, n + 1, :].set(
        u_padded[:, n, n + 2, :]
    )

    u_padded = u_padded.at[:, 0, 0, :].set(
        v_padded[:, 0, 1, :]
    )
    u_padded = u_padded.at[:, 0, n + 2, :].set(
        q_u * v_padded[:, 0, n, :]
    )
    u_padded = u_padded.at[:, n + 1, 0, :].set(
        q_u * v_padded[:, n + 2, 1, :]
    )
    u_padded = u_padded.at[:, n + 1, n + 2, :].set(
        v_padded[:, n + 2, n, :]
    )
    return u_padded, v_padded


def pad_halo_dgrid_vector_4d(u_d, v_d):
    """FV3 DGRID_NE staggered covariant vector halo (iter-1078).

    Combines the iter-1076 same-axis cross-face halo (16/24 edges) with the
    iter-1078 DGRID_NE component swap (8 axis-swap edges) and the half-turn
    covariant sign (4 same-axis reversed seams, ``_SAME_AXIS_FLIP_SEAMS``).

    VALUE-LEVEL SEMANTICS (2026-08-04, certified against
    ``analytic_swcore_state`` halos -- fv3_recon/transplant2_9311777.log,
    which caught two defect classes in the pre-fix version):

    - the SOURCE line on a component's NODE-staggered axis (u: j, v: i) is
      the neighbour's line one INWARD of the shared seam (the outermost node
      line IS the seam, already stored by this face); cell axes use the
      outermost line;
    - the four half-turn seams (2,S),(2,N),(4,N),(5,S) negate BOTH
      components (180-degree relative frame rotation of covariant
      components).

    The pre-fix "bit-for-bit" claim rested on constant-per-face stencil
    tests, which are blind to both classes (a constant field makes the seam
    line equal the one-beyond line, and no test probed a half-turn strip).
    """
    if u_d.ndim != 4 or v_d.ndim != 4:
        raise ValueError(f"4D inputs required; got u_d.ndim={u_d.ndim}, v_d.ndim={v_d.ndim}")
    if u_d.shape[0] != 6 or v_d.shape[0] != 6:
        raise ValueError(f"6 faces required")
    n = u_d.shape[1]
    if u_d.shape[2] != n + 1 or v_d.shape[1] != n + 1 or v_d.shape[2] != n:
        raise ValueError(f"u_d expects (6, n, n+1, nlev), v_d (6, n+1, n, nlev); got u_d {tuple(u_d.shape)}, v_d {tuple(v_d.shape)}")
    u_padded = pad_halo_dgrid_scalar_4d(u_d, axis_swap_fill="edge")
    v_padded = pad_halo_dgrid_scalar_4d(v_d, axis_swap_fill="edge")
    (u_dst_f, u_dst_i, u_dst_j,
     v_src_f, v_src_i, v_src_j, u_dst_signs,
     v_dst_f, v_dst_i, v_dst_j,
     u_src_f, u_src_i, u_src_j, v_dst_signs) = _get_axis_swap_tables_h1(n)
    if u_dst_f.size > 0:
        v_vals = v_d[v_src_f, v_src_i, v_src_j]
        u_padded = u_padded.at[u_dst_f, u_dst_i, u_dst_j].set(v_vals * u_dst_signs.astype(u_d.dtype)[:, None])
    if v_dst_f.size > 0:
        u_vals = u_d[u_src_f, u_src_i, u_src_j]
        v_padded = v_padded.at[v_dst_f, v_dst_i, v_dst_j].set(u_vals * v_dst_signs.astype(v_d.dtype)[:, None])
    # Half-turn covariant sign: the four same-axis reversed seams carry the
    # neighbour's components NEGATED (both u and v).  Applied after the
    # same-axis seeding (these seams are never axis-swap, so the overlay
    # negates exactly the table-written strip values).
    for _face, _edge in _SAME_AXIS_FLIP_SEAMS:
        u_padded = _negate_side_ghost_line(u_padded, _face, _edge)
        v_padded = _negate_side_ghost_line(v_padded, _face, _edge)
    return u_padded, v_padded


# =====================================================================
# iter-1083: MPI-aware DGRID vector halo (batched-per-peer sendrecv)
# =====================================================================


def _build_dgrid_mpi_edges(topology):
    """Build per-face per-edge dispatch list with axis-swap metadata."""
    edges = []
    for face in topology.local_face_ids:
        for edge in (WEST, EAST, SOUTH, NORTH):
            nbr_face, nbr_edge, is_reversed = topology.neighbor_info[
                (face, edge)
            ]
            nbr_rank = topology.neighbor_ranks[(face, edge)]
            if (face, edge) in _AXIS_SWAP_TABLE:
                _, _, _, sign_uv, sign_vu = _AXIS_SWAP_TABLE[(face, edge)]
                kind = "swap"
            else:
                sign_uv = sign_vu = 1
                kind = "same"
            edges.append((face, edge, nbr_face, nbr_edge, is_reversed,
                           nbr_rank, kind, sign_uv, sign_vu))
    return edges


def _extract_edge_strip_dgrid(data, face_loc, edge, n_i, n_j):
    """Extract the depth-1 SOURCE strip from staggered (n_i, n_j) data.

    Same rule as the single-device tables (2026-08-04 transplant fix): on a
    NODE-staggered axis (n+1 lines) the outermost line IS the shared seam,
    so the exchanged strip is the line one INWARD; on a CELL axis it is the
    outermost line.  Keeps the MPI twin bit-for-bit with
    ``pad_halo_dgrid_vector_4d`` for both same-axis and axis-swap seams.
    """
    i_node = n_i == n_j + 1
    j_node = n_j == n_i + 1
    if edge == WEST:
        return data[face_loc, 1 if i_node else 0, :, :]
    elif edge == EAST:
        return data[face_loc, n_i - 2 if i_node else n_i - 1, :, :]
    elif edge == SOUTH:
        return data[face_loc, :, 1 if j_node else 0, :]
    else:  # NORTH
        return data[face_loc, :, n_j - 2 if j_node else n_j - 1, :]


def _place_strip_dgrid(padded, face_loc, edge, strip, n_i, n_j):
    """Place a strip into halo=1 position of padded staggered array."""
    if edge == WEST:
        return padded.at[face_loc, 0, 1:n_j + 1, :].set(strip)
    elif edge == EAST:
        return padded.at[face_loc, n_i + 1, 1:n_j + 1, :].set(strip)
    elif edge == SOUTH:
        return padded.at[face_loc, 1:n_i + 1, 0, :].set(strip)
    else:  # NORTH
        return padded.at[face_loc, 1:n_i + 1, n_j + 1, :].set(strip)


def pad_halo_dgrid_vector_4d_mpi(u_d, v_d, topology):
    """MPI-aware DGRID vector halo with batched-per-peer sendrecv.

    Mirrors the deadlock-free pattern from
    ``_pad_halo_mpi_face_only_4d`` (sorted peer iteration, one
    sendrecv per peer with packed multi-edge buffer).  Same-axis +
    axis-swap edges are encoded into the strip pack/unpack and the
    iter-1078 component-swap signs.

    Each rank holds local-only u_d/v_d:
    - u_d shape: ``(n_local, n, n+1, nlev)``
    - v_d shape: ``(n_local, n+1, n, nlev)``

    Returns padded halo arrays with the same per-face staggered
    layout as ``pad_halo_dgrid_vector_4d``.

    Parameters
    ----------
    u_d, v_d : jax.Array
        Rank-local staggered D-grid wind components.
    topology : CommTopology

    Returns
    -------
    u_d_padded : jax.Array, shape ``(n_local, n+2, n+3, nlev)``
    v_d_padded : jax.Array, shape ``(n_local, n+3, n+2, nlev)``
    """
    try:
        import mpi4jax
        from mpi4py import MPI
    except ImportError as exc:
        raise ImportError(
            "pad_halo_dgrid_vector_4d_mpi requires mpi4jax + mpi4py."
        ) from exc

    if u_d.ndim != 4 or v_d.ndim != 4:
        raise ValueError("4D inputs required")
    n_local = u_d.shape[0]
    if len(topology.local_face_ids) != n_local:
        raise ValueError(
            f"u_d.shape[0]={n_local} != len(local_face_ids)={len(topology.local_face_ids)}"
        )
    n = u_d.shape[1]
    _n_j_u, _n_i_v = n + 1, n + 1
    if u_d.shape[1:3] != (n, n + 1) or v_d.shape[1:3] != (n + 1, n):
        raise ValueError(
            f"Expected u_d (n_local, n, n+1, nlev), v_d (n_local, n+1, n, nlev); "
            f"got u_d {tuple(u_d.shape)}, v_d {tuple(v_d.shape)}"
        )
    nlev = u_d.shape[-1]
    g2l = {f: i for i, f in enumerate(topology.local_face_ids)}

    # Initialize padded arrays with edge-replicate.
    u_padded = jnp.pad(u_d, ((0, 0), (1, 1), (1, 1), (0, 0)), mode="edge")
    v_padded = jnp.pad(v_d, ((0, 0), (1, 1), (1, 1), (0, 0)), mode="edge")

    # Classify edges into local (same rank) vs remote.
    edges = _build_dgrid_mpi_edges(topology)
    local_edges = [e for e in edges if e[5] == topology.rank]
    remote_edges = [e for e in edges if e[5] != topology.rank]

    # ---- Local edges: direct read-write ----
    for (face, edge, nbr_face, nbr_edge, is_reversed,
         _, kind, sign_uv, sign_vu) in local_edges:
        f_loc = g2l[face]
        nf_loc = g2l[nbr_face]
        # Extract from neighbor's edge.
        u_strip = _extract_edge_strip_dgrid(u_d, nf_loc, nbr_edge, n, n + 1)
        v_strip = _extract_edge_strip_dgrid(v_d, nf_loc, nbr_edge, n + 1, n)
        if is_reversed:
            u_strip = u_strip[::-1]
            v_strip = v_strip[::-1]
        if kind == "same":
            if is_reversed:
                # Half-turn seam (_SAME_AXIS_FLIP_SEAMS): covariant
                # components cross with sign -1 on BOTH components.
                u_strip = -u_strip
                v_strip = -v_strip
            u_padded = _place_strip_dgrid(
                u_padded, f_loc, edge, u_strip, n, n + 1,
            )
            v_padded = _place_strip_dgrid(
                v_padded, f_loc, edge, v_strip, n + 1, n,
            )
        else:  # axis-swap: u_face_halo ← sign_uv * v_nbr; v_face_halo ← sign_vu * u_nbr
            u_padded = _place_strip_dgrid(
                u_padded, f_loc, edge, v_strip * sign_uv, n, n + 1,
            )
            v_padded = _place_strip_dgrid(
                v_padded, f_loc, edge, u_strip * sign_vu, n + 1, n,
            )

    if not remote_edges:
        return u_padded, v_padded

    # ---- Remote edges: batched-per-peer sendrecv ----
    from collections import defaultdict
    comm = MPI.COMM_WORLD
    rank = topology.rank

    by_nbr = defaultdict(list)
    for entry in remote_edges:
        by_nbr[entry[5]].append(entry)

    # Sorted peer iteration (deadlock-free at np=6).
    for nbr_rank in sorted(by_nbr.keys()):
        entries = by_nbr[nbr_rank]
        # Send-order: sort by (nbr_face, nbr_edge) — peer will recv in
        # the same key order via (face, edge) on their side.
        send_order = sorted(entries, key=lambda e: (e[2], e[3]))
        # Recv-order: sort by (face, edge) — matches peer's send_order.
        recv_order = sorted(entries, key=lambda e: (e[0], e[1]))

        # Build send buffer: for each entry, pack [u_strip, v_strip]
        # of THIS face's edge (the peer needs them to fill THEIR halo).
        send_parts = []
        for face, edge, nbr_face, nbr_edge, is_reversed, _, kind, _, _ in send_order:
            f_loc = g2l[face]
            u_strip = _extract_edge_strip_dgrid(u_d, f_loc, edge, n, n + 1)
            v_strip = _extract_edge_strip_dgrid(v_d, f_loc, edge, n + 1, n)
            send_parts.append(u_strip.reshape(-1))
            send_parts.append(v_strip.reshape(-1))
        send_buf = jnp.concatenate(send_parts)

        # Single sendrecv per neighbor rank.  Use the AD-safe
        # ``_sendrecv_vjp`` wrapper (custom_vjp) so jax.grad flows
        # through MPI sendrecv (raw mpi4jax.sendrecv chokes on the
        # symbolic Zero cotangent JAX emits during backward).
        from legoesm.parallel.halo_exchange import get_sendrecv_vjp
        sendrecv = get_sendrecv_vjp(mpi4jax)
        recv_buf = sendrecv(
            send_buf, jnp.zeros_like(send_buf),
            nbr_rank, nbr_rank,
            rank, nbr_rank, comm,
        )

        # Unpack recv buffer in recv_order.
        offset = 0
        for (face, edge, nbr_face, nbr_edge, is_reversed,
             _, kind, sign_uv, sign_vu) in recv_order:
            # Strip lengths on neighbor's side, sourcing peer's edge
            # at (nbr_face, nbr_edge): same convention as u/v strips
            # extracted in the send loop on the OTHER rank.
            if nbr_edge in _I_EDGES:
                u_strip_len = n + 1
                v_strip_len = n
            else:
                u_strip_len = n
                v_strip_len = n + 1
            u_size = u_strip_len * nlev
            v_size = v_strip_len * nlev
            u_strip = recv_buf[offset:offset + u_size].reshape(
                (u_strip_len, nlev)
            )
            offset += u_size
            v_strip = recv_buf[offset:offset + v_size].reshape(
                (v_strip_len, nlev)
            )
            offset += v_size
            if is_reversed:
                u_strip = u_strip[::-1]
                v_strip = v_strip[::-1]

            f_loc = g2l[face]
            if kind == "same":
                if is_reversed:
                    # Half-turn seam: covariant sign -1, both components
                    # (matches the single-device _SAME_AXIS_FLIP_SEAMS
                    # overlay bit-for-bit).
                    u_strip = -u_strip
                    v_strip = -v_strip
                u_padded = _place_strip_dgrid(
                    u_padded, f_loc, edge, u_strip, n, n + 1,
                )
                v_padded = _place_strip_dgrid(
                    v_padded, f_loc, edge, v_strip, n + 1, n,
                )
            else:  # swap
                u_padded = _place_strip_dgrid(
                    u_padded, f_loc, edge, v_strip * sign_uv, n, n + 1,
                )
                v_padded = _place_strip_dgrid(
                    v_padded, f_loc, edge, u_strip * sign_vu, n + 1, n,
                )

    return u_padded, v_padded


def pad_halo_dgrid_vector_4d_replicated_mpi(u_d_full, v_d_full, topology):
    """MPI dgrid halo on REPLICATED (6, ...) input.

    Wraps ``pad_halo_dgrid_vector_4d_mpi`` for callers that hold full
    ``(6, n, n+1, nlev)`` / ``(6, n+1, n, nlev)`` state on each rank
    (the canonical cubed-sphere MPI mode per
    ``driver/model_driver.py:1206``).  Returns full ``(6, n+2, n+3,
    nlev)`` / ``(6, n+3, n+2, nlev)`` with this rank's OWNED face
    halos correctly filled via cross-rank sendrecv; non-owned face
    halos stay at edge-replicate (consumers should only use their
    owned-face slices).

    Parameters
    ----------
    u_d_full : jax.Array, shape (6, n, n+1, nlev)
    v_d_full : jax.Array, shape (6, n+1, n, nlev)
    topology : CommTopology

    Returns
    -------
    u_d_padded : jax.Array, shape (6, n+2, n+3, nlev)
    v_d_padded : jax.Array, shape (6, n+3, n+2, nlev)
    """
    if u_d_full.shape[0] != 6 or v_d_full.shape[0] != 6:
        raise ValueError(
            f"Expected 6 faces; got u {tuple(u_d_full.shape)}, "
            f"v {tuple(v_d_full.shape)}"
        )
    # Slice to owned faces.
    idx = jnp.asarray(list(topology.local_face_ids), dtype=jnp.int32)
    u_local = u_d_full[idx]
    v_local = v_d_full[idx]
    # Run MPI halo on owned faces.
    u_local_padded, v_local_padded = pad_halo_dgrid_vector_4d_mpi(
        u_local, v_local, topology,
    )
    # Initialize full padded with edge-replicate (gives reasonable
    # values on non-owned faces — consumers will use only owned).
    u_full_padded = jnp.pad(
        u_d_full, ((0, 0), (1, 1), (1, 1), (0, 0)), mode="edge",
    )
    v_full_padded = jnp.pad(
        v_d_full, ((0, 0), (1, 1), (1, 1), (0, 0)), mode="edge",
    )
    # Place owned-face halos.
    u_full_padded = u_full_padded.at[idx].set(u_local_padded)
    v_full_padded = v_full_padded.at[idx].set(v_local_padded)
    return u_full_padded, v_full_padded
