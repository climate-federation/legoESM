"""Inter-face halo exchange for the cubed-sphere grid.

Implements proper communication between the 6 faces of the cubed-sphere,
replacing the incorrect `jnp.roll` (within-face wrapping) with correct
neighbor-face data exchange.

The connectivity table is derived from the gnomonic projection geometry
in `cubed_sphere.py`. Each face edge maps to a specific edge on a
neighbor face, with possible index reversal and axis swaps.

All functions are pure and compatible with jax.jit, jax.grad, jax.vmap.

References
----------
- Putman & Lin (2007): Finite-volume transport on various cubed-sphere grids
- Ronchi, Iacono, Paolucci (1996): The "Cubed Sphere"
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


# ==============================================================================
# Edge constants
# ==============================================================================

WEST = 0   # i = 0 boundary
EAST = 1   # i = n-1 boundary
SOUTH = 2  # j = 0 boundary
NORTH = 3  # j = n-1 boundary


# ==============================================================================
# Face connectivity table
# ==============================================================================
#
# CONNECTIVITY[face][edge] = (neighbor_face, neighbor_edge, reversed)
#
# Face numbering (from cubed_sphere.py _face_to_cartesian):
#   0: +x (front)    1: +y (right)   2: -x (back)
#   3: -y (left)     4: +z (north)   5: -z (south)
#
# Derived by tracing the gnomonic projection at each face boundary.
# All connections are verified to be symmetric:
#   if A_edge -> (B, B_edge, rev), then B_edge -> (A, A_edge, rev)

CONNECTIVITY = {
    0: {
        WEST:  (3, EAST, False),
        EAST:  (1, WEST, False),
        SOUTH: (5, NORTH, False),
        NORTH: (4, SOUTH, False),
    },
    1: {
        WEST:  (0, EAST, False),
        EAST:  (2, WEST, False),
        SOUTH: (5, EAST, True),     # reversed
        NORTH: (4, EAST, False),
    },
    2: {
        WEST:  (1, EAST, False),
        EAST:  (3, WEST, False),
        SOUTH: (5, SOUTH, True),    # reversed
        NORTH: (4, NORTH, True),    # reversed
    },
    3: {
        WEST:  (2, EAST, False),
        EAST:  (0, WEST, False),
        SOUTH: (5, WEST, False),
        NORTH: (4, WEST, True),     # reversed
    },
    4: {
        WEST:  (3, NORTH, True),    # reversed
        EAST:  (1, NORTH, False),
        SOUTH: (0, NORTH, False),
        NORTH: (2, NORTH, True),    # reversed
    },
    5: {
        WEST:  (3, SOUTH, False),
        EAST:  (1, SOUTH, True),    # reversed
        SOUTH: (2, SOUTH, True),    # reversed
        NORTH: (0, SOUTH, False),
    },
}


# ==============================================================================
# Halo interpolation offset precomputation
# ==============================================================================

def _face_to_xyz_np(face: int, ax: float, ay: float) -> np.ndarray:
    """Gnomonic coords → unit-sphere Cartesian (numpy, scalar)."""
    tx, ty = np.tan(ax), np.tan(ay)
    if face == 0:   x, y, z = 1.0, tx, ty
    elif face == 1: x, y, z = -tx, 1.0, ty
    elif face == 2: x, y, z = -1.0, -tx, ty
    elif face == 3: x, y, z = tx, -1.0, ty
    elif face == 4: x, y, z = -ty, tx, 1.0
    elif face == 5: x, y, z = ty, tx, -1.0
    else: raise ValueError(face)
    r = np.sqrt(x**2 + y**2 + z**2)
    return np.array([x / r, y / r, z / r])


def _xyz_to_gnomonic_np(
    face: int, x: float, y: float, z: float,
) -> tuple[float, float]:
    """Unit-sphere Cartesian → gnomonic coords on *face* (numpy, scalar).

    Inverts the projection: project from the sphere centre through
    (x, y, z) onto the face plane and recover (alpha_x, alpha_y).
    """
    if face == 0:   return float(np.arctan(y / x)), float(np.arctan(z / x))
    elif face == 1: return float(np.arctan(-x / y)), float(np.arctan(z / y))
    elif face == 2: return float(np.arctan(y / x)), float(np.arctan(-z / x))
    elif face == 3: return float(np.arctan(-x / y)), float(np.arctan(-z / y))
    elif face == 4: return float(np.arctan(y / z)), float(np.arctan(-x / z))
    elif face == 5: return float(np.arctan(-y / z)), float(np.arctan(-x / z))
    else: raise ValueError(face)


def _extract_edge_strip_at_depth(
    data: jax.Array, face: int, edge: int, depth: int,
) -> jax.Array:
    """Extract strip at given depth from edge (depth=0 is boundary row).

    Parameters
    ----------
    data : jax.Array, shape (6, n, n)
    face : int
    edge : int
        WEST, EAST, SOUTH, NORTH.
    depth : int
        0 = boundary row, 1 = one row inward, etc.

    Returns
    -------
    strip : jax.Array, shape (n,)
    """
    if edge == WEST:
        return data[face, depth, :]
    elif edge == EAST:
        return data[face, -(depth + 1), :]
    elif edge == SOUTH:
        return data[face, :, depth]
    elif edge == NORTH:
        return data[face, :, -(depth + 1)]
    else:
        raise ValueError(f"Invalid edge: {edge}")


def compute_halo_interp_offsets(n: int) -> jnp.ndarray:
    """Precompute fractional-index offsets for the interpolated halo exchange.

    The standard halo exchange copies the j-th cell of the neighbour's
    edge strip into the j-th halo slot.  On a gnomonic cubed sphere the
    physical position of the halo slot (from the face's own extended
    gnomonic grid) does NOT coincide with the neighbour's j-th cell —
    the mismatch grows toward cube corners (up to ~0.5 grid cells at C48).

    This function computes, for every halo cell, the true fractional index
    on the neighbour strip so that ``pad_halo`` can linearly interpolate
    instead of doing a nearest-index copy.

    Parameters
    ----------
    n : int
        Number of cells per face edge.

    Returns
    -------
    offsets : jax.Array, shape (6, 4, n)
        ``offsets[face, edge_idx, j]`` is the correction δ such that
        the true fractional index on the (possibly reversed) neighbour
        strip is ``j + δ``.  Edge indices: 0=WEST, 1=EAST, 2=SOUTH, 3=NORTH.
    """
    dalpha = np.pi / (2 * n)
    alpha = np.linspace(-np.pi / 4, np.pi / 4, n, endpoint=False) + dalpha / 2

    edges = [WEST, EAST, SOUTH, NORTH]
    offsets = np.zeros((6, 4, n), dtype=np.float64)

    for face in range(6):
        for edge_idx, edge in enumerate(edges):
            nbr_face, nbr_edge, is_reversed = CONNECTIVITY[face][edge]

            for j in range(n):
                # ---- halo cell position on extended gnomonic grid ----
                if edge == WEST:
                    ax_h, ay_h = -np.pi / 4 - dalpha / 2, alpha[j]
                elif edge == EAST:
                    ax_h, ay_h = np.pi / 4 + dalpha / 2, alpha[j]
                elif edge == SOUTH:
                    ax_h, ay_h = alpha[j], -np.pi / 4 - dalpha / 2
                else:  # NORTH
                    ax_h, ay_h = alpha[j], np.pi / 4 + dalpha / 2

                xyz_h = _face_to_xyz_np(face, ax_h, ay_h)

                # ---- map to neighbour-face gnomonic coordinates ----
                ax_n, ay_n = _xyz_to_gnomonic_np(
                    nbr_face, xyz_h[0], xyz_h[1], xyz_h[2],
                )

                # ---- fractional index along the strip ----
                # The strip varies along the "transverse" gnomonic axis.
                if nbr_edge in (WEST, EAST):
                    frac = (ay_n - alpha[0]) / dalpha
                else:  # SOUTH, NORTH
                    frac = (ax_n - alpha[0]) / dalpha

                if is_reversed:
                    frac = (n - 1) - frac

                offsets[face, edge_idx, j] = frac - j

    return jnp.array(offsets, dtype=jnp.float32)


def compute_halo_interp_offsets_h2(n: int) -> jnp.ndarray:
    """Precompute fractional-index offsets for halo=2 exchange.

    Like :func:`compute_halo_interp_offsets` but returns offsets for
    two halo depths (depth 0 = adjacent to interior, depth 1 = outer).

    Parameters
    ----------
    n : int
        Number of cells per face edge.

    Returns
    -------
    offsets : jax.Array, shape (6, 4, 2, n)
        ``offsets[face, edge_idx, depth, j]`` is the correction δ such
        that the true fractional index on the neighbour strip at that
        depth is ``j + δ``.
    """
    dalpha = np.pi / (2 * n)
    alpha = np.linspace(-np.pi / 4, np.pi / 4, n, endpoint=False) + dalpha / 2

    edges = [WEST, EAST, SOUTH, NORTH]
    offsets = np.zeros((6, 4, 2, n), dtype=np.float64)

    for face in range(6):
        for edge_idx, edge in enumerate(edges):
            nbr_face, nbr_edge, is_reversed = CONNECTIVITY[face][edge]

            for depth in range(2):
                # Distance from boundary: depth=0 → dα/2, depth=1 → 3dα/2
                d = dalpha / 2 + depth * dalpha

                for j in range(n):
                    if edge == WEST:
                        ax_h, ay_h = -np.pi / 4 - d, alpha[j]
                    elif edge == EAST:
                        ax_h, ay_h = np.pi / 4 + d, alpha[j]
                    elif edge == SOUTH:
                        ax_h, ay_h = alpha[j], -np.pi / 4 - d
                    else:  # NORTH
                        ax_h, ay_h = alpha[j], np.pi / 4 + d

                    xyz_h = _face_to_xyz_np(face, ax_h, ay_h)
                    ax_n, ay_n = _xyz_to_gnomonic_np(
                        nbr_face, xyz_h[0], xyz_h[1], xyz_h[2],
                    )

                    if nbr_edge in (WEST, EAST):
                        frac = (ay_n - alpha[0]) / dalpha
                    else:
                        frac = (ax_n - alpha[0]) / dalpha

                    if is_reversed:
                        frac = (n - 1) - frac

                    offsets[face, edge_idx, depth, j] = frac - j

    return jnp.array(offsets, dtype=jnp.float32)


def _interp_strip(strip: jax.Array, offsets_1d: jax.Array) -> jax.Array:
    """Quadratically interpolate *strip* at positions ``j + offsets_1d[j]``.

    Uses 3-point Lagrange interpolation, which reduces the gradient error
    at face boundaries from O(dx) (linear) to O(dx^2) (quadratic).
    Falls back to linear for very short strips (n < 3).

    Parameters
    ----------
    strip : shape (n,)
    offsets_1d : shape (n,)

    Returns
    -------
    interpolated strip : shape (n,)
    """
    n = strip.shape[0]
    idx = jnp.arange(n, dtype=offsets_1d.dtype) + offsets_1d
    idx = jnp.clip(idx, 0.0, n - 1.0)

    if n < 3:
        # Fall back to linear for very coarse grids
        lo = jnp.clip(jnp.floor(idx).astype(jnp.int32), 0, n - 2)
        w = jnp.clip(idx - lo.astype(offsets_1d.dtype), 0.0, 1.0)
        return ((1.0 - w) * strip[lo] + w * strip[lo + 1]).astype(strip.dtype)

    # 3-point Lagrange: stencil centre clamped to [1, n-2] so all
    # three indices {jc-1, jc, jc+1} are in bounds.
    jc = jnp.clip(jnp.round(idx).astype(jnp.int32), 1, n - 2)
    f = idx - jc.astype(offsets_1d.dtype)
    c_m1 = 0.5 * f * (f - 1.0)
    c_0 = 1.0 - f * f
    c_p1 = 0.5 * f * (f + 1.0)
    interp = c_m1 * strip[jc - 1] + c_0 * strip[jc] + c_p1 * strip[jc + 1]
    return interp.astype(strip.dtype)


# ==============================================================================
# Halo backend dispatch (for MPI)
# ==============================================================================

# Module-level state: "local" for single-node (default), "mpi" for distributed.
_halo_backend: str = "local"
_mpi_topology = None  # CommTopology or None
_spmd_mesh = None     # jax.sharding.Mesh or None


def set_halo_backend(backend: str, topology=None) -> None:
    """Configure the halo exchange backend.

    Parameters
    ----------
    backend : ``"local"`` | ``"mpi"`` | ``"spmd"``
        ``"local"`` uses the default JAX-native implementation
        (works for single-device and single-node multi-device).
        ``"mpi"`` uses ``mpi4jax`` for distributed communication.
        ``"spmd"`` uses ``shard_map`` + ``all_gather`` for explicit
        multi-GPU collectives (set via
        :func:`parallel.cubesphere_exchange.activate_spmd_halo_backend`).
    topology : CommTopology, optional
        Required when ``backend="mpi"``.
    """
    global _halo_backend, _mpi_topology
    if backend not in ("local", "mpi", "spmd"):
        raise ValueError(f"Unknown halo backend: {backend!r}")
    if backend == "mpi" and topology is None:
        raise ValueError("CommTopology is required for MPI halo backend")
    _halo_backend = backend
    _mpi_topology = topology


def get_halo_backend() -> str:
    """Return the current halo exchange backend name."""
    return _halo_backend


# ==============================================================================
# Scalar halo exchange
# ==============================================================================

def _extract_edge_strip(data: jax.Array, face: int, edge: int) -> jax.Array:
    """Extract a 1D strip (length n) from a face edge.

    Parameters
    ----------
    data : jax.Array, shape (6, n, n)
        Field data on all faces.
    face : int
        Face index (0-5).
    edge : int
        Edge constant (WEST, EAST, SOUTH, NORTH).

    Returns
    -------
    strip : jax.Array, shape (n,)
    """
    if edge == WEST:
        return data[face, 0, :]
    elif edge == EAST:
        return data[face, -1, :]
    elif edge == SOUTH:
        return data[face, :, 0]
    elif edge == NORTH:
        return data[face, :, -1]
    else:
        raise ValueError(f"Invalid edge: {edge}")


def _pad_halo_wall(data: jax.Array, halo: int = 1) -> jax.Array:
    """Pad a field with Neumann (zero-gradient) wall boundary conditions.

    For single-face regional panels where inter-face halo exchange is
    not available.  Each boundary is filled by replicating the nearest
    interior row/column (zero-gradient BC), which prevents spurious
    gradients at domain edges.

    Handles both 3D ``(1, n, n)`` and 4D ``(1, n, n, nlev)`` inputs.

    Parameters
    ----------
    data : jax.Array, shape (1, n, n) or (1, n, n, nlev)
    halo : int
        Halo width (1 or 2).

    Returns
    -------
    padded : jax.Array, shape (1, n+2*h, n+2*h) or (1, n+2*h, n+2*h, nlev)
    """
    h = halo
    if data.ndim == 3:
        # (1, n, n)
        padded = jnp.pad(data, ((0, 0), (h, h), (h, h)), mode="edge")
    elif data.ndim == 4:
        # (1, n, n, nlev) — pad only spatial dims
        padded = jnp.pad(data, ((0, 0), (h, h), (h, h), (0, 0)), mode="edge")
    else:
        raise ValueError(f"_pad_halo_wall expects 3D or 4D, got {data.ndim}D")
    return padded


def pad_halo(
    data: jax.Array,
    halo: int = 1,
    interp_offsets: jax.Array | None = None,
) -> jax.Array:
    """Pad a scalar field with inter-face halo data.

    Creates an array of shape (6, n+2*halo, n+2*halo) where the interior
    contains the original data and the halo cells contain data from
    neighboring faces, correctly handling axis swaps and reversals.

    When *interp_offsets* is provided (shape ``(6, 4, n)``), the neighbour
    strip is linearly interpolated to the correct physical position of
    each halo cell instead of using a nearest-index copy.  This removes
    the O(Δα) position mismatch near cube corners that otherwise
    degrades operator accuracy from second to first order.

    Corner cells (4 per face) are left as zero, which is safe for
    halo=1 with 2nd-order stencils (corners are never used).

    When the MPI backend is active, this dispatches to
    :func:`legoesm.parallel.halo_exchange.pad_halo_mpi`.

    Parameters
    ----------
    data : jax.Array, shape (6, n, n)
        Scalar field on the cubed-sphere.
    halo : int
        Halo width (default 1). Only halo=1 is currently supported.
    interp_offsets : jax.Array or None
        Precomputed fractional-index offsets, shape (6, 4, n).
        When ``None``, the standard nearest-index copy is used.

    Returns
    -------
    padded : jax.Array, shape (6, n+2*halo, n+2*halo)
    """
    if halo not in (1, 2):
        raise NotImplementedError(f"Only halo=1 and halo=2 are supported, got {halo}")

    # Single-face (regional panel) dispatch: wall boundary conditions.
    if data.shape[0] == 1:
        return _pad_halo_wall(data, halo)

    # MPI dispatch.
    if _halo_backend == "mpi":
        from legoesm.parallel.halo_exchange import pad_halo_mpi
        return pad_halo_mpi(data, _mpi_topology, halo=halo)

    # SPMD dispatch (explicit all_gather for multi-GPU).
    if _halo_backend == "spmd" and _spmd_mesh is not None:
        from legoesm.parallel.cubesphere_exchange import explicit_pad_halo
        return explicit_pad_halo(data, _spmd_mesh, halo=halo)

    if halo == 1:
        return _pad_halo_local(data, interp_offsets)
    else:
        return _pad_halo_local_h2(data, interp_offsets)


def pad_halo_4d(
    data: jax.Array,
    halo: int = 1,
    interp_offsets: jax.Array | None = None,
) -> jax.Array:
    """Pad a 4D scalar field with inter-face halo data.

    Like :func:`pad_halo` but operates on all vertical levels at once,
    issuing a single communication instead of one per level.  This
    reduces MPI messages by a factor of ``nlev``.

    Parameters
    ----------
    data : jax.Array, shape (6, n, n, nlev)
        Scalar field on the cubed-sphere with a trailing level axis.
    halo : int
        Halo width (1 or 2).
    interp_offsets : jax.Array or None
        Precomputed fractional-index offsets, shape (6, 4, n).

    Returns
    -------
    padded : jax.Array, shape (6, n+2*halo, n+2*halo, nlev)
    """
    if data.ndim != 4:
        raise ValueError(f"pad_halo_4d expects 4D input, got {data.ndim}D")
    if halo not in (1, 2):
        raise NotImplementedError(f"Only halo=1 and halo=2 are supported, got {halo}")

    # Single-face (regional panel) dispatch: wall boundary conditions.
    if data.shape[0] == 1:
        return _pad_halo_wall(data, halo)

    # MPI dispatch.
    if _halo_backend == "mpi":
        from legoesm.parallel.halo_exchange import pad_halo_mpi_4d
        return pad_halo_mpi_4d(data, _mpi_topology, halo=halo)

    # SPMD dispatch (explicit all_gather for multi-GPU).
    if _halo_backend == "spmd" and _spmd_mesh is not None:
        from legoesm.parallel.cubesphere_exchange import explicit_pad_halo_4d
        return explicit_pad_halo_4d(data, _spmd_mesh, halo=halo)

    if halo == 1:
        return _pad_halo_local_4d(data, interp_offsets)
    else:
        return _pad_halo_local_h2_4d(data, interp_offsets)


def _pad_halo_local_4d(
    data: jax.Array,
    interp_offsets: jax.Array | None = None,
) -> jax.Array:
    """Local 4D scalar halo exchange for halo=1.

    Uses the same precomputed index tables as the 2D version.  The
    trailing level axis is preserved via ``data[src_f, src_i, src_j]``
    which yields shape ``(24*n, nlev)`` when data is ``(6, n, n, nlev)``.
    """
    n = data.shape[1]
    nlev = data.shape[3]
    tables = _get_halo_tables_h1(n)
    src_f, src_i, src_j, dst_f, dst_i, dst_j = tables

    padded = jnp.zeros((6, n + 2, n + 2, nlev), dtype=data.dtype)
    padded = padded.at[:, 1:-1, 1:-1, :].set(data)

    if interp_offsets is None:
        # Nearest-neighbor: gather (24*n, nlev) then scatter
        values = data[src_f, src_i, src_j]  # (24*n, nlev)
        padded = padded.at[dst_f, dst_i, dst_j].set(values)
    else:
        # 3-point quadratic Lagrange (same upgrade as 2D path)
        flat_offsets = interp_offsets.reshape(-1)
        n_int = int(n)
        strip_base = jnp.repeat(jnp.arange(24) * n_int, n_int)
        j_local = jnp.tile(jnp.arange(n_int), 24)

        frac = j_local + flat_offsets
        frac = jnp.clip(frac, 0.0, n_int - 1.0)

        jc = jnp.clip(jnp.round(frac).astype(jnp.int32), 1, n_int - 2)
        f = frac - jc.astype(frac.dtype)
        c_m1 = 0.5 * f * (f - 1.0)
        c_0 = 1.0 - f * f
        c_p1 = 0.5 * f * (f + 1.0)

        _sf = jnp.asarray(src_f)
        _si = jnp.asarray(src_i)
        _sj = jnp.asarray(src_j)
        idx_m1 = strip_base + jnp.clip(jc - 1, 0, n_int - 1)
        idx_0 = strip_base + jc
        idx_p1 = strip_base + jnp.clip(jc + 1, 0, n_int - 1)
        vals_m1 = data[_sf[idx_m1], _si[idx_m1], _sj[idx_m1]]  # (24*n, nlev)
        vals_0 = data[_sf[idx_0], _si[idx_0], _sj[idx_0]]
        vals_p1 = data[_sf[idx_p1], _si[idx_p1], _sj[idx_p1]]
        values = (c_m1[:, None] * vals_m1 + c_0[:, None] * vals_0
                  + c_p1[:, None] * vals_p1).astype(data.dtype)
        padded = padded.at[dst_f, dst_i, dst_j].set(values)

    padded = _fill_corners_h1(padded)
    return padded


def _pad_halo_local_h2_4d(
    data: jax.Array,
    interp_offsets: jax.Array | None = None,
) -> jax.Array:
    """Local 4D scalar halo exchange for halo=2.

    Loop-based exchange (same as 2D version) — the scalar indexing
    ``data[face, i, j]`` naturally returns shape ``(nlev,)`` for 4D.
    """
    n = data.shape[1]
    nlev = data.shape[3]
    padded = jnp.zeros((6, n + 4, n + 4, nlev), dtype=data.dtype)
    padded = padded.at[:, 2:-2, 2:-2, :].set(data)

    edges = [WEST, EAST, SOUTH, NORTH]

    for face in range(6):
        for edge_idx, edge in enumerate(edges):
            nbr_face, nbr_edge, is_reversed = CONNECTIVITY[face][edge]

            for depth in range(2):
                strip = _extract_edge_strip_at_depth(
                    data, nbr_face, nbr_edge, depth,
                )  # (n,) for 3D or (n, nlev) for 4D

                if is_reversed:
                    strip = strip[::-1]

                if interp_offsets is not None:
                    strip = _interp_strip(
                        strip, interp_offsets[face, edge_idx, depth],
                    )

                if edge == WEST:
                    padded = padded.at[face, 1 - depth, 2:-2].set(strip)
                elif edge == EAST:
                    padded = padded.at[face, n + 2 + depth, 2:-2].set(strip)
                elif edge == SOUTH:
                    padded = padded.at[face, 2:-2, 1 - depth].set(strip)
                elif edge == NORTH:
                    padded = padded.at[face, 2:-2, n + 2 + depth].set(strip)

    padded = _fill_corners_h2(padded)
    return padded


def pad_halo_vector_4d(
    u_data: jax.Array,
    v_data: jax.Array,
    cos_angle: jax.Array,
    sin_angle: jax.Array,
    cos_angle_padded: jax.Array,
    sin_angle_padded: jax.Array,
    interp_offsets: jax.Array | None = None,
    halo: int = 1,
) -> tuple[jax.Array, jax.Array]:
    """4D vector halo exchange (rotation + pad for all levels at once).

    Same logic as :func:`pad_halo_vector` but using :func:`pad_halo_4d`.
    Inputs are (6, n, n, nlev); rotation angles are (6, n, n) and get
    broadcast over the trailing level axis.

    Returns
    -------
    u_padded, v_padded : jax.Array, shape (6, n+2*halo, n+2*halo, nlev)
    """
    # SPMD dispatch: pack both components into a single collective.
    if _halo_backend == "spmd" and _spmd_mesh is not None and halo == 1:
        from legoesm.parallel.cubesphere_exchange import (
            explicit_pad_halo_vector_4d,
        )
        return explicit_pad_halo_vector_4d(
            u_data, v_data,
            cos_angle, sin_angle,
            cos_angle_padded, sin_angle_padded,
            _spmd_mesh, halo=halo,
        )

    # Broadcast 2D angles to match 4D data
    ca = cos_angle[..., None]
    sa = sin_angle[..., None]
    # Step 1: convert to geographic
    u_east = ca * u_data - sa * v_data
    v_north = sa * u_data + ca * v_data
    # Step 2: pad as scalars (one communication per component)
    u_east_padded = pad_halo_4d(u_east, halo=halo, interp_offsets=interp_offsets)
    v_north_padded = pad_halo_4d(v_north, halo=halo, interp_offsets=interp_offsets)
    # Step 3: convert back using padded angles
    cap = cos_angle_padded[..., None]
    sap = sin_angle_padded[..., None]
    u_padded = cap * u_east_padded + sap * v_north_padded
    v_padded = -sap * u_east_padded + cap * v_north_padded
    return u_padded, v_padded


def _pad_halo_local(
    data: jax.Array,
    interp_offsets: jax.Array | None = None,
) -> jax.Array:
    """Local (single-node) scalar halo exchange implementation.

    Uses precomputed index tables for a vectorized gather+scatter instead
    of 24 sequential Python-loop iterations.  This reduces XLA IR size
    and enables better fusion across the halo operation.
    """
    n = data.shape[1]
    tables = _get_halo_tables_h1(n)
    src_f, src_i, src_j, dst_f, dst_i, dst_j = tables

    padded = jnp.zeros((6, n + 2, n + 2), dtype=data.dtype)
    padded = padded.at[:, 1:-1, 1:-1].set(data)

    if interp_offsets is None:
        # Nearest-neighbor copy: single gather + single scatter
        values = data[src_f, src_i, src_j]
        padded = padded.at[dst_f, dst_i, dst_j].set(values)
    else:
        # Interpolated exchange: 3-point quadratic Lagrange interpolation.
        # Reduces halo interpolation error from O(dx^2) to O(dx^3),
        # which lowers the gradient error at face boundaries from O(dx)
        # to O(dx^2) — matching interior accuracy and reducing (but not
        # eliminating) edge artifacts in the Arakawa-Lamb gradient.
        # Full visual elimination of edge artifacts also requires the
        # boundary_fix option in fv3_sw_tendencies (see iteration 15 in
        # docs/cubed_sphere_edge_artifacts.md).
        flat_offsets = interp_offsets.reshape(-1)
        n_int = int(n)
        strip_base = jnp.repeat(jnp.arange(24) * n_int, n_int)  # (24*n,)
        j_local = jnp.tile(jnp.arange(n_int), 24)  # position within strip

        frac = j_local + flat_offsets
        frac = jnp.clip(frac, 0.0, n_int - 1.0)

        # 3-point Lagrange stencil centered at jc = round(frac)
        jc = jnp.clip(jnp.round(frac).astype(jnp.int32), 1, n_int - 2)
        f = frac - jc.astype(frac.dtype)  # fractional distance from center
        # Lagrange basis polynomials for nodes {jc-1, jc, jc+1}:
        c_m1 = 0.5 * f * (f - 1.0)
        c_0 = 1.0 - f * f
        c_p1 = 0.5 * f * (f + 1.0)

        # Map strip-local indices to global data indices.
        _sf = jnp.asarray(src_f)
        _si = jnp.asarray(src_i)
        _sj = jnp.asarray(src_j)
        idx_m1 = strip_base + jnp.clip(jc - 1, 0, n_int - 1)
        idx_0 = strip_base + jc
        idx_p1 = strip_base + jnp.clip(jc + 1, 0, n_int - 1)
        vals_m1 = data[_sf[idx_m1], _si[idx_m1], _sj[idx_m1]]
        vals_0 = data[_sf[idx_0], _si[idx_0], _sj[idx_0]]
        vals_p1 = data[_sf[idx_p1], _si[idx_p1], _sj[idx_p1]]
        values = (c_m1 * vals_m1 + c_0 * vals_0 + c_p1 * vals_p1).astype(
            data.dtype)
        padded = padded.at[dst_f, dst_i, dst_j].set(values)

    # Fill corner cells (vectorized)
    padded = _fill_corners_h1(padded)
    return padded


def _pad_halo_local_h2(
    data: jax.Array,
    interp_offsets: jax.Array | None = None,
) -> jax.Array:
    """Local (single-node) scalar halo exchange for halo=2.

    Parameters
    ----------
    data : jax.Array, shape (6, n, n)
    interp_offsets : jax.Array or None, shape (6, 4, 2, n)
        Precomputed fractional-index offsets for 2 halo depths.

    Returns
    -------
    padded : jax.Array, shape (6, n+4, n+4)
    """
    n = data.shape[1]
    padded = jnp.zeros((6, n + 4, n + 4), dtype=data.dtype)

    # Place interior data
    padded = padded.at[:, 2:-2, 2:-2].set(data)

    edges = [WEST, EAST, SOUTH, NORTH]

    for face in range(6):
        for edge_idx, edge in enumerate(edges):
            nbr_face, nbr_edge, is_reversed = CONNECTIVITY[face][edge]

            for depth in range(2):
                # Extract neighbour strip at this depth
                strip = _extract_edge_strip_at_depth(
                    data, nbr_face, nbr_edge, depth,
                )

                if is_reversed:
                    strip = strip[::-1]

                # Interpolate to correct physical position if offsets provided
                if interp_offsets is not None:
                    strip = _interp_strip(
                        strip, interp_offsets[face, edge_idx, depth],
                    )

                # Place in halo: depth=0 → adjacent to interior, depth=1 → outer
                # Interior is at [2:-2, 2:-2], so:
                #   WEST halo positions: i=1 (depth=0), i=0 (depth=1)
                #   EAST halo positions: i=n+2 (depth=0), i=n+3 (depth=1)
                pos = 1 - depth  # WEST: depth0→1, depth1→0
                if edge == WEST:
                    padded = padded.at[face, 1 - depth, 2:-2].set(strip)
                elif edge == EAST:
                    padded = padded.at[face, n + 2 + depth, 2:-2].set(strip)
                elif edge == SOUTH:
                    padded = padded.at[face, 2:-2, 1 - depth].set(strip)
                elif edge == NORTH:
                    padded = padded.at[face, 2:-2, n + 2 + depth].set(strip)

    # Fill L-shaped corner regions (4 cells per corner × 4 corners × 6 faces)
    padded = _fill_corners_h2(padded)

    return padded


# ==============================================================================
# Precomputed index tables for vectorized halo exchange
# ==============================================================================

_halo_table_cache_h1: dict[int, tuple] = {}


def _build_halo_tables_h1(n: int) -> tuple:
    """Build source/destination index arrays for vectorized halo=1 exchange.

    For each of the 24*n halo cells (6 faces × 4 edges × n cells per edge),
    stores the (face, i, j) source coordinates in the original data array
    and the (face, i, j) destination coordinates in the padded array.

    Source indices already account for edge geometry and reversal, so the
    runtime exchange is a single gather + scatter.
    """
    edges = [WEST, EAST, SOUTH, NORTH]
    total = 24 * n

    src_f = np.zeros(total, dtype=np.int32)
    src_i = np.zeros(total, dtype=np.int32)
    src_j = np.zeros(total, dtype=np.int32)
    dst_f = np.zeros(total, dtype=np.int32)
    dst_i = np.zeros(total, dtype=np.int32)
    dst_j = np.zeros(total, dtype=np.int32)

    idx = 0
    for face in range(6):
        for edge_idx, edge in enumerate(edges):
            nbr_face, nbr_edge, is_reversed = CONNECTIVITY[face][edge]

            for j in range(n):
                # Position in strip after reversal
                k = (n - 1 - j) if is_reversed else j

                # Source cell from neighbor edge
                if nbr_edge == WEST:
                    src_f[idx], src_i[idx], src_j[idx] = nbr_face, 0, k
                elif nbr_edge == EAST:
                    src_f[idx], src_i[idx], src_j[idx] = nbr_face, n - 1, k
                elif nbr_edge == SOUTH:
                    src_f[idx], src_i[idx], src_j[idx] = nbr_face, k, 0
                else:  # NORTH
                    src_f[idx], src_i[idx], src_j[idx] = nbr_face, k, n - 1

                # Destination in padded array (n+2 × n+2)
                if edge == WEST:
                    dst_f[idx], dst_i[idx], dst_j[idx] = face, 0, j + 1
                elif edge == EAST:
                    dst_f[idx], dst_i[idx], dst_j[idx] = face, n + 1, j + 1
                elif edge == SOUTH:
                    dst_f[idx], dst_i[idx], dst_j[idx] = face, j + 1, 0
                else:  # NORTH
                    dst_f[idx], dst_i[idx], dst_j[idx] = face, j + 1, n + 1

                idx += 1

    # Return numpy arrays (not jnp) so that caching doesn't leak JAX
    # tracers when pad_halo is called inside jax.lax.scan.  JAX treats
    # numpy arrays as static constants during tracing.
    return (src_f, src_i, src_j, dst_f, dst_i, dst_j)


def _get_halo_tables_h1(n: int) -> tuple:
    """Return cached halo index tables for grid size n.

    Safe to call inside JAX transforms: tables are built with pure
    numpy (no JAX tracers).  The ``int(n)`` call ensures concrete
    Python int even if ``n`` comes from a traced shape dimension.
    """
    n = int(n)
    if n not in _halo_table_cache_h1:
        _halo_table_cache_h1[n] = _build_halo_tables_h1(n)
    return _halo_table_cache_h1[n]


def precompute_halo_tables(n: int) -> None:
    """Eagerly populate the halo index cache for grid size n.

    Call this at grid creation time (outside JIT) to avoid cache
    writes during JAX tracing.
    """
    _get_halo_tables_h1(int(n))


def _fill_corners_h1(padded: jax.Array) -> jax.Array:
    """Fill corner cells of halo=1 padded array by averaging adjacent edge halos.

    Vectorized: all 24 corners (6 faces × 4 corners) in a single
    gather + average + scatter.

    Parameters
    ----------
    padded : jax.Array, shape (6, n+2, n+2)

    Returns
    -------
    jax.Array, shape (6, n+2, n+2)
    """
    n2i = padded.shape[1] - 1  # n+1 (last index in padded)

    # All 24 corner cells: (face, i, j) and their two adjacent halo cells
    f_idx = jnp.arange(6)
    # SW(0,0), SE(n+1,0), NW(0,n+1), NE(n+1,n+1) per face
    cf = jnp.repeat(f_idx, 4)
    ci = jnp.tile(jnp.array([0, n2i, 0, n2i]), 6)
    cj = jnp.tile(jnp.array([0, 0, n2i, n2i]), 6)
    # Adjacent cell 1
    a1i = jnp.tile(jnp.array([0, n2i, 0, n2i]), 6)
    a1j = jnp.tile(jnp.array([1, 1, n2i - 1, n2i - 1]), 6)
    # Adjacent cell 2
    a2i = jnp.tile(jnp.array([1, n2i - 1, 1, n2i - 1]), 6)
    a2j = jnp.tile(jnp.array([0, 0, n2i, n2i]), 6)

    corner_vals = 0.5 * (padded[cf, a1i, a1j] + padded[cf, a2i, a2j])
    padded = padded.at[cf, ci, cj].set(corner_vals)
    return padded


def _fill_corners_h2(padded: jax.Array) -> jax.Array:
    """Fill L-shaped corner regions of halo=2 padded array.

    Each face has 4 corner regions of 2×2 = 4 cells that are not
    filled by the edge-strip exchange.  We fill inside-out: the cell
    closest to the interior first (average of its two filled neighbours),
    then propagate outward.

    Parameters
    ----------
    padded : jax.Array, shape (6, n+4, n+4)

    Returns
    -------
    jax.Array, shape (6, n+4, n+4)
    """
    for f in range(6):
        # --- SW corner (rows 0-1, cols 0-1) ---
        # Inner corner (1,1): adjacent cells (1,2) and (2,1) are filled
        padded = padded.at[f, 1, 1].set(
            0.5 * (padded[f, 1, 2] + padded[f, 2, 1])
        )
        # (0,1): adjacent to (0,2) [WEST halo, filled] and (1,1) [just filled]
        padded = padded.at[f, 0, 1].set(
            0.5 * (padded[f, 0, 2] + padded[f, 1, 1])
        )
        # (1,0): adjacent to (2,0) [SOUTH halo, filled] and (1,1) [just filled]
        padded = padded.at[f, 1, 0].set(
            0.5 * (padded[f, 2, 0] + padded[f, 1, 1])
        )
        # Outer corner (0,0)
        padded = padded.at[f, 0, 0].set(
            0.5 * (padded[f, 0, 1] + padded[f, 1, 0])
        )

        # --- SE corner (rows n+2..n+3, cols 0-1) ---
        padded = padded.at[f, -2, 1].set(
            0.5 * (padded[f, -2, 2] + padded[f, -3, 1])
        )
        padded = padded.at[f, -1, 1].set(
            0.5 * (padded[f, -1, 2] + padded[f, -2, 1])
        )
        padded = padded.at[f, -2, 0].set(
            0.5 * (padded[f, -3, 0] + padded[f, -2, 1])
        )
        padded = padded.at[f, -1, 0].set(
            0.5 * (padded[f, -1, 1] + padded[f, -2, 0])
        )

        # --- NW corner (rows 0-1, cols n+2..n+3) ---
        padded = padded.at[f, 1, -2].set(
            0.5 * (padded[f, 1, -3] + padded[f, 2, -2])
        )
        padded = padded.at[f, 0, -2].set(
            0.5 * (padded[f, 0, -3] + padded[f, 1, -2])
        )
        padded = padded.at[f, 1, -1].set(
            0.5 * (padded[f, 2, -1] + padded[f, 1, -2])
        )
        padded = padded.at[f, 0, -1].set(
            0.5 * (padded[f, 0, -2] + padded[f, 1, -1])
        )

        # --- NE corner (rows n+2..n+3, cols n+2..n+3) ---
        padded = padded.at[f, -2, -2].set(
            0.5 * (padded[f, -2, -3] + padded[f, -3, -2])
        )
        padded = padded.at[f, -1, -2].set(
            0.5 * (padded[f, -1, -3] + padded[f, -2, -2])
        )
        padded = padded.at[f, -2, -1].set(
            0.5 * (padded[f, -3, -1] + padded[f, -2, -2])
        )
        padded = padded.at[f, -1, -1].set(
            0.5 * (padded[f, -1, -2] + padded[f, -2, -1])
        )

    return padded


def extrapolate_to_halo(data: jax.Array) -> jax.Array:
    """Extrapolate a field to halo cells using boundary values.

    Unlike pad_halo (which gets neighbor data), this copies each face's
    own boundary values to its halo cells. This is appropriate for
    grid metric terms (dx, dy, area) that are defined in the local
    face coordinate system and should NOT be exchanged between faces
    with different axis orientations.

    Parameters
    ----------
    data : jax.Array, shape (6, n, n)
        Field to extrapolate.

    Returns
    -------
    padded : jax.Array, shape (6, n+2, n+2)
    """
    # jnp.pad with mode='edge' replicates boundary values — equivalent to
    # the previous 9 sequential .at[].set() scatter operations but compiled
    # as a single XLA pad op.
    return jnp.pad(data, ((0, 0), (1, 1), (1, 1)), mode='edge')


# ==============================================================================
# Vector halo exchange
# ==============================================================================

def pad_halo_vector(
    u_data: jax.Array,
    v_data: jax.Array,
    cos_angle: jax.Array,
    sin_angle: jax.Array,
    cos_angle_padded: jax.Array,
    sin_angle_padded: jax.Array,
    interp_offsets: jax.Array | None = None,
    halo: int = 1,
) -> tuple[jax.Array, jax.Array]:
    """Pad vector field components with proper rotation at face boundaries.

    Grid-aligned velocity components (u, v) change meaning across face
    boundaries because each face has different grid axis orientations.
    This function handles the rotation correctly:

    1. Convert (u_grid, v_grid) -> (u_east, v_north) using grid angle
    2. Pad u_east, v_north as scalars (geographic components are continuous)
    3. Convert back to grid-aligned using the padded grid angle

    When the MPI backend is active, the scalar padding step dispatches
    to :func:`legoesm.parallel.halo_exchange.pad_halo_mpi` via
    :func:`pad_halo`.

    Parameters
    ----------
    u_data : jax.Array, shape (6, n, n)
        Grid-aligned x-velocity component.
    v_data : jax.Array, shape (6, n, n)
        Grid-aligned y-velocity component.
    cos_angle : jax.Array, shape (6, n, n)
        Cosine of grid rotation angle (precomputed).
    sin_angle : jax.Array, shape (6, n, n)
        Sine of grid rotation angle (precomputed).
    cos_angle_padded : jax.Array, shape (6, n+2*halo, n+2*halo)
        Cosine of padded grid angle (precomputed).
    sin_angle_padded : jax.Array, shape (6, n+2*halo, n+2*halo)
        Sine of padded grid angle (precomputed).
    interp_offsets : jax.Array or None
        Precomputed fractional-index offsets.
        halo=1: shape (6, 4, n). halo=2: shape (6, 4, 2, n).
    halo : int
        Halo width (1 or 2).

    Returns
    -------
    u_padded : jax.Array, shape (6, n+2*halo, n+2*halo)
        Padded grid-aligned x-velocity.
    v_padded : jax.Array, shape (6, n+2*halo, n+2*halo)
        Padded grid-aligned y-velocity.
    """
    # Step 1: Convert to geographic (east, north)
    u_east = cos_angle * u_data - sin_angle * v_data
    v_north = sin_angle * u_data + cos_angle * v_data

    # Step 2: Pad geographic components as scalars (auto-dispatches to MPI)
    u_east_padded = pad_halo(u_east, halo=halo, interp_offsets=interp_offsets)
    v_north_padded = pad_halo(v_north, halo=halo, interp_offsets=interp_offsets)

    # Step 3: Convert back to grid-aligned using padded angle
    u_padded = cos_angle_padded * u_east_padded + sin_angle_padded * v_north_padded
    v_padded = -sin_angle_padded * u_east_padded + cos_angle_padded * v_north_padded

    return u_padded, v_padded


# ==============================================================================
# Padded grid angle computation
# ==============================================================================

def compute_padded_angle(n: int, halo: int = 1) -> jax.Array:
    """Compute grid angle on extended (n+2*halo) x (n+2*halo) grid per face.

    Uses a further-extended gnomonic grid internally so that centred
    differences are available at every point of the output grid,
    matching the approach in ``compute_padded_half_metrics``.

    Parameters
    ----------
    n : int
        Number of cells per face edge (interior grid).
    halo : int
        Halo width (1 or 2).

    Returns
    -------
    angle_padded : jax.Array, shape (6, n+2*halo, n+2*halo)
    """
    dalpha = jnp.pi / (2 * n)

    # Need (n+2*halo+2) internal grid so centred diffs yield (n+2*halo) output
    n_big = n + 2 * halo + 2
    ext = halo + 1  # cells beyond interior on each side
    alpha_big = jnp.linspace(
        -jnp.pi / 4 - ext * dalpha,
        jnp.pi / 4 + ext * dalpha,
        n_big,
        endpoint=False,
    )
    alpha_big = alpha_big + dalpha / 2.0
    alpha_bx, alpha_by = jnp.meshgrid(alpha_big, alpha_big, indexing='ij')

    all_angle = []
    for face in range(6):
        lon, lat = _face_gnomonic_to_lonlat(face, alpha_bx, alpha_by)

        # Centred differences on (n_big) → (n_big-2) = (n+2*halo) output
        dlon_dx = lon[2:, 1:-1] - lon[:-2, 1:-1]
        dlat_dx = lat[2:, 1:-1] - lat[:-2, 1:-1]

        # Handle longitude wrapping
        dlon_dx = jnp.where(dlon_dx > jnp.pi, dlon_dx - 2 * jnp.pi, dlon_dx)
        dlon_dx = jnp.where(dlon_dx < -jnp.pi, dlon_dx + 2 * jnp.pi, dlon_dx)

        cos_lat_ext = jnp.cos(lat[1:-1, 1:-1])
        face_angle = jnp.arctan2(dlat_dx, dlon_dx * cos_lat_ext)
        all_angle.append(face_angle)

    return jnp.stack(all_angle, axis=0)


def _face_gnomonic_to_lonlat(
    face: int,
    alpha_x: jax.Array,
    alpha_y: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Convert gnomonic coordinates to lon/lat for a single face.

    This is a pure-function version of the face-to-cartesian-to-lonlat
    conversion, usable for extended grids.

    Parameters
    ----------
    face : int
        Face index (0-5).
    alpha_x, alpha_y : jax.Array
        Gnomonic coordinates (can extend beyond [-pi/4, pi/4]).

    Returns
    -------
    lon, lat : jax.Array
        Longitude and latitude in radians.
    """
    tan_x = jnp.tan(alpha_x)
    tan_y = jnp.tan(alpha_y)

    if face == 0:    # +x
        x, y, z = jnp.ones_like(tan_x), tan_x, tan_y
    elif face == 1:  # +y
        x, y, z = -tan_x, jnp.ones_like(tan_x), tan_y
    elif face == 2:  # -x
        x, y, z = -jnp.ones_like(tan_x), -tan_x, tan_y
    elif face == 3:  # -y
        x, y, z = tan_x, -jnp.ones_like(tan_x), tan_y
    elif face == 4:  # +z
        x, y, z = -tan_y, tan_x, jnp.ones_like(tan_x)
    elif face == 5:  # -z
        x, y, z = tan_y, tan_x, -jnp.ones_like(tan_x)
    else:
        raise ValueError(f"Invalid face: {face}")

    r = jnp.sqrt(x**2 + y**2 + z**2)
    x, y, z = x / r, y / r, z / r

    lon = jnp.arctan2(y, x)
    lat = jnp.arcsin(jnp.clip(z, -1.0, 1.0))

    return lon, lat


# ==============================================================================
# Padded metric computation (for divergence / curl operators)
# ==============================================================================

def compute_padded_half_metrics(
    n: int, radius: float, halo: int = 1,
) -> tuple[jax.Array, jax.Array]:
    """Compute half-edge-length metrics on the extended (n+2*halo) × (n+2*halo) grid.

    Each face's gnomonic coordinate range is extended by *halo* cells on
    each side.  Great-circle distances between cell centres two positions
    apart are computed (the 2-cell span), and halved to give the
    single-cell edge lengths ``hx = dx/2`` and ``hy = dy/2``.

    A further-extended grid is used internally so that centred differences
    are available at every point of the output grid.

    Parameters
    ----------
    n : int
        Number of interior cells per face edge.
    radius : float
        Sphere radius [m].
    halo : int
        Halo width (1 or 2).

    Returns
    -------
    hx_ext : jax.Array, shape (6, n+2*halo, n+2*halo)
        Half dx on the extended grid (single-cell x-edge length).
    hy_ext : jax.Array, shape (6, n+2*halo, n+2*halo)
        Half dy on the extended grid (single-cell y-edge length).
    """
    dalpha = jnp.pi / (2 * n)

    # Need (n+2*halo+2) internal grid so centred diffs yield (n+2*halo)
    n_big = n + 2 * halo + 2
    ext = halo + 1  # cells beyond interior on each side
    alpha_big = jnp.linspace(
        -jnp.pi / 4 - ext * dalpha,
        jnp.pi / 4 + ext * dalpha,
        n_big,
        endpoint=False,
    )
    alpha_big = alpha_big + dalpha / 2.0
    alpha_bx, alpha_by = jnp.meshgrid(alpha_big, alpha_big, indexing='ij')

    all_hx: list[jax.Array] = []
    all_hy: list[jax.Array] = []

    for face in range(6):
        lon, lat = _face_gnomonic_to_lonlat(face, alpha_bx, alpha_by)
        cos_lat = jnp.cos(lat)
        x = cos_lat * jnp.cos(lon)
        y = cos_lat * jnp.sin(lon)
        z = jnp.sin(lat)

        # 2-cell chord in x → (n_big-2, n_big-2) = (n+2*halo, n+2*halo)
        dx_chord = jnp.sqrt(
            (x[2:, 1:-1] - x[:-2, 1:-1]) ** 2
            + (y[2:, 1:-1] - y[:-2, 1:-1]) ** 2
            + (z[2:, 1:-1] - z[:-2, 1:-1]) ** 2
        )
        dx_face = radius * 2.0 * jnp.arcsin(
            jnp.clip(dx_chord / 2.0, 0.0, 1.0)
        )

        # 2-cell chord in y
        dy_chord = jnp.sqrt(
            (x[1:-1, 2:] - x[1:-1, :-2]) ** 2
            + (y[1:-1, 2:] - y[1:-1, :-2]) ** 2
            + (z[1:-1, 2:] - z[1:-1, :-2]) ** 2
        )
        dy_face = radius * 2.0 * jnp.arcsin(
            jnp.clip(dy_chord / 2.0, 0.0, 1.0)
        )

        all_hx.append(dx_face * 0.5)
        all_hy.append(dy_face * 0.5)

    hx_ext = jnp.stack(all_hx, axis=0)
    hy_ext = jnp.stack(all_hy, axis=0)

    return hx_ext, hy_ext
