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

import contextlib
import importlib

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


def extract_edge_strip_at_depth(
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
    # halo=1 is the depth-0 slice of the general N-depth precomputation
    # (mirrors compute_halo_interp_offsets_ed → _compute_halo_interp_offsets_ed_hN).
    return _compute_halo_interp_offsets_hN(n, 1)[:, :, 0, :]


def _ed_indomain_boundary_strip(arr, edge, n, ext):
    """In-domain boundary strip (length n, ascending array order) of an
    ``(M, M)`` padded gnomonic_ed face, in-domain block ``[ext:ext+n]``."""
    if edge == WEST:
        return arr[ext, ext:ext + n]
    elif edge == EAST:
        return arr[ext + n - 1, ext:ext + n]
    elif edge == SOUTH:
        return arr[ext:ext + n, ext]
    else:  # NORTH
        return arr[ext:ext + n, ext + n - 1]


def _ed_halo_strip(arr, edge, n, ext, depth):
    """Halo strip at ``depth`` cells beyond ``edge`` (depth 0 = adjacent to the
    in-domain boundary), length n, ascending array order."""
    if edge == WEST:
        return arr[ext - 1 - depth, ext:ext + n]
    elif edge == EAST:
        return arr[ext + n + depth, ext:ext + n]
    elif edge == SOUTH:
        return arr[ext:ext + n, ext - 1 - depth]
    else:  # NORTH
        return arr[ext:ext + n, ext + n + depth]


def _compute_halo_interp_offsets_ed_hN(n: int, halo: int) -> jnp.ndarray:
    """gnomonic_ed cross-face halo interp offsets, ``(6, 4, halo, n)``.

    Position-matching on the actual extended gnomonic_ed cell centres
    (`gnomonic_ed_padded_centers`, which extend cleanly into the halo): each
    halo cell (depth 0..halo-1 beyond an edge) is matched to its neighbour's
    in-domain edge strip via a parabola-vertex fit on great-circle distance,
    giving the true fractional index; ``δ = frac − j``.  gnomonic_ed-specific
    (the equiangular analytic offsets are the wrong geometry — codex gating
    blocker).
    """
    from legoesm.grids.cubed_sphere import gnomonic_ed_padded_centers

    ext = halo + 1
    lon, lat = gnomonic_ed_padded_centers(n, halo)  # (6, M, M), M=n+2*halo+2
    lon = np.asarray(lon)
    lat = np.asarray(lat)
    xyz = np.stack([
        np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)
    ], axis=-1)

    edges = [WEST, EAST, SOUTH, NORTH]
    offsets = np.zeros((6, 4, halo, n), dtype=np.float64)

    for face in range(6):
        for edge_idx, edge in enumerate(edges):
            nbr_face, nbr_edge, is_reversed = CONNECTIVITY[face][edge]
            nbr_strip = _ed_indomain_boundary_strip(xyz[nbr_face], nbr_edge, n, ext)
            for depth in range(halo):
                halo_strip = _ed_halo_strip(xyz[face], edge, n, ext, depth)
                for j in range(n):
                    d2 = np.sum((nbr_strip - halo_strip[j][None, :]) ** 2, axis=1)
                    k = int(np.argmin(d2))
                    if 0 < k < n - 1:
                        dl, dc, dr = d2[k - 1], d2[k], d2[k + 1]
                        denom = dl - 2.0 * dc + dr
                        delta = (0.5 * (dl - dr) / denom
                                 if abs(denom) > 1e-30 else 0.0)
                        delta = float(np.clip(delta, -1.0, 1.0))
                    else:
                        delta = 0.0
                    frac = k + delta
                    if is_reversed:
                        frac = (n - 1) - frac
                    offsets[face, edge_idx, depth, j] = frac - j

    return jnp.array(offsets, dtype=jnp.float64)


def compute_halo_interp_offsets_ed(n: int) -> jnp.ndarray:
    """gnomonic_ed counterpart of :func:`compute_halo_interp_offsets` —
    ``(6, 4, n)`` (halo=1, squeezed).  See :func:`_compute_halo_interp_offsets_ed_hN`."""
    return _compute_halo_interp_offsets_ed_hN(n, 1)[:, :, 0, :]


def compute_halo_interp_offsets_ed_h2(n: int) -> jnp.ndarray:
    """gnomonic_ed counterpart of :func:`compute_halo_interp_offsets_h2` — ``(6, 4, 2, n)``."""
    return _compute_halo_interp_offsets_ed_hN(n, 2)


def compute_halo_interp_offsets_ed_h3(n: int) -> jnp.ndarray:
    """gnomonic_ed counterpart of :func:`compute_halo_interp_offsets_h3` — ``(6, 4, 3, n)``."""
    return _compute_halo_interp_offsets_ed_hN(n, 3)


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
    return _compute_halo_interp_offsets_hN(n, halo=2)


def compute_halo_interp_offsets_h3(n: int) -> jnp.ndarray:
    """Precompute fractional-index offsets for halo=3 exchange.

    Like :func:`compute_halo_interp_offsets_h2` but returns offsets for
    three halo depths (depth 0 = adjacent to interior, depth 2 = outer).

    This supports the FB-path stability work that requires deeper halos
    to reach stable c_sw/d_sw stencils at C36 resolution.

    Parameters
    ----------
    n : int
        Number of cells per face edge.

    Returns
    -------
    offsets : jax.Array, shape (6, 4, 3, n)
        ``offsets[face, edge_idx, depth, j]`` is the correction δ such
        that the true fractional index on the neighbour strip at that
        depth is ``j + δ``.
    """
    return _compute_halo_interp_offsets_hN(n, halo=3)


def _compute_halo_interp_offsets_hN(n: int, halo: int) -> jnp.ndarray:
    """General N-depth halo interp-offset precomputation.

    Shared implementation for :func:`compute_halo_interp_offsets_h2`
    and :func:`compute_halo_interp_offsets_h3` — folded into a single
    function so the per-depth gnomonic→neighbour mapping logic is in
    one place.
    """
    if halo < 1:
        raise ValueError(f"halo must be >= 1, got {halo}")

    dalpha = np.pi / (2 * n)
    alpha = np.linspace(-np.pi / 4, np.pi / 4, n, endpoint=False) + dalpha / 2

    edges = [WEST, EAST, SOUTH, NORTH]
    offsets = np.zeros((6, 4, halo, n), dtype=np.float64)

    for face in range(6):
        for edge_idx, edge in enumerate(edges):
            nbr_face, nbr_edge, is_reversed = CONNECTIVITY[face][edge]

            for depth in range(halo):
                # Distance from boundary: depth=0 → dα/2, depth=k → (2k+1)dα/2
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

    return jnp.array(offsets, dtype=jnp.float64)


def interp_strip(strip: jax.Array, offsets_1d: jax.Array) -> jax.Array:
    """Interpolate *strip* at positions ``j + offsets_1d[j]``.

    Uses 3-point quadratic Lagrange interpolation, giving O(dx^3) value
    accuracy and O(dx^2) gradient accuracy.  Falls back to linear for
    very short strips (n < 3).

    The quadratic stencil uses points {jc-1, jc, jc+1} where jc is the
    nearest integer.  The stencil centre is clamped to [1, n-2] to keep
    all three indices in bounds.

    Parameters
    ----------
    strip : shape (n,) or (n, ...)
        Edge strip.  Trailing axes (e.g. nlev or nlev*n_tracers) are
        carried through passively.
    offsets_1d : shape (n,)

    Returns
    -------
    interpolated strip : shape matching ``strip``.
    """
    n = strip.shape[0]
    idx = jnp.arange(n, dtype=offsets_1d.dtype) + offsets_1d
    idx = jnp.clip(idx, 0.0, n - 1.0)

    # Iter-724: support 4D strips `(n, nlev)` by broadcasting weights
    # along a trailing level axis.  Prior to iter-724 this function
    # assumed `strip.shape == (n,)`; callers passing multi-level
    # strips got a latent broadcast failure when offsets != None
    # (caught by Codex iter-723 review on the new halo=3 4D path).
    extra_dims = strip.ndim - 1
    broadcast_shape = (n,) + (1,) * extra_dims

    if n < 3:
        # Fall back to linear for very coarse grids
        lo = jnp.clip(jnp.floor(idx).astype(jnp.int32), 0, n - 2)
        w = jnp.clip(idx - lo.astype(offsets_1d.dtype), 0.0, 1.0)
        w_b = w.reshape(broadcast_shape)
        return ((1.0 - w_b) * strip[lo] + w_b * strip[lo + 1]).astype(strip.dtype)

    # 3-point Lagrange: stencil centre clamped to [1, n-2] so all
    # three indices {jc-1, jc, jc+1} are in bounds.
    jc = jnp.clip(jnp.round(idx).astype(jnp.int32), 1, n - 2)
    # Promote f to strip dtype so Lagrange weights have full precision.
    # With float32 offsets + float64 data, the float32 weights have
    # sum(w) = 1 ± O(1e-7), causing ~0.06 Pa error for 6e5 Pa fields.
    f = (idx - jc.astype(offsets_1d.dtype)).astype(strip.dtype)
    c_m1 = (0.5 * f * (f - 1.0)).reshape(broadcast_shape)
    c_0 = (1.0 - f * f).reshape(broadcast_shape)
    c_p1 = (0.5 * f * (f + 1.0)).reshape(broadcast_shape)
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
    topology : CommTopology or LatLonBandLayout, optional
        Required when ``backend="mpi"``.  Cubed-sphere passes a
        :class:`~legoesm.parallel.comm.CommTopology`; the lat-lon
        band path passes a
        :class:`~legoesm.parallel.latlon_mpi.LatLonBandLayout`.  The
        per-grid ``pad_halo*`` functions branch on the topology's
        type, so a single backend slot serves both grids and the
        conservation reductions' ``is_distributed()`` gate fires
        uniformly.
    """
    global _halo_backend, _mpi_topology
    if backend not in ("local", "mpi", "spmd"):
        raise ValueError(f"Unknown halo backend: {backend!r}")
    if backend == "mpi" and topology is None:
        raise ValueError(
            "MPI halo backend requires a topology "
            "(CommTopology for cubed-sphere, LatLonBandLayout for lat-lon)."
        )
    _halo_backend = backend
    _mpi_topology = topology


def get_halo_backend() -> str:
    """Return the current halo exchange backend name."""
    return _halo_backend


def get_mpi_topology():
    """Return the active MPI topology object (or ``None``).

    Cubed-sphere callers receive a
    :class:`~legoesm.parallel.comm.CommTopology`; lat-lon callers
    receive a
    :class:`~legoesm.parallel.latlon_mpi.LatLonBandLayout`.  Each
    grid's halo-pad dispatch checks the type before consuming it.
    """
    return _mpi_topology


def get_spmd_mesh():
    """Return the active SPMD ``jax.sharding.Mesh`` (or ``None``).

    Set by the per-grid SPMD activators (cube
    :func:`parallel.cubesphere_exchange.activate_spmd_halo_backend`; lat-lon
    :func:`parallel.latlon_spmd.activate_latlon_spmd_halo`).  The per-grid
    ``pad_halo*`` dispatch reads it when ``get_halo_backend() == "spmd"`` and
    routes to the matching shard_map ppermute body (cube: face/tile axes;
    lat-lon: a ``"lat"`` band axis)."""
    return _spmd_mesh


def set_spmd_mesh(mesh) -> None:
    """Set/clear the active SPMD mesh (``None`` clears).  Public setter so the
    SPMD activators need not write the private module global cross-package."""
    global _spmd_mesh
    _spmd_mesh = mesh


@contextlib.contextmanager
def local_halo_pads():
    """Force every ``pad_halo*`` dispatch to the LOCAL (serial) backend
    within the ``with`` block, then restore the previous backend state.

    This is a TRACE-TIME switch for wide-halo compute regions: after a code
    path has already exchanged a wide halo (width = its full stencil reach),
    its interior operators must run their internal pads as plain local
    ``jnp.pad`` — re-dispatching them to MPI sendrecv / SPMD ppermute would
    re-communicate every substep, defeating the wide exchange (and, on the
    extended arrays, exchange the WRONG rows).  The flag is read when the
    operators trace, so wrap the traced region, not the runtime call.

    Neutralizes all three dispatch signals: backend name, MPI topology, and
    the SPMD mesh.  Re-entrant and exception-safe.
    """
    global _halo_backend, _mpi_topology, _spmd_mesh
    saved = (_halo_backend, _mpi_topology, _spmd_mesh)
    _halo_backend, _mpi_topology, _spmd_mesh = "local", None, None
    try:
        yield
    finally:
        _halo_backend, _mpi_topology, _spmd_mesh = saved


# ==============================================================================
# Scalar halo exchange
# ==============================================================================

def extract_edge_strip(data: jax.Array, face: int, edge: int) -> jax.Array:
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
    duogrid=None,
    monotone_clip: bool = False,
    monotone_clip_slack: float = 0.0,
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

    When *duogrid* is provided (a ``DuoGridData``), the standard copy is
    followed by a kinked-to-extended Lagrange remap (cube_rmp) and corner
    fill. This is mutually exclusive with *interp_offsets*.

    Parameters
    ----------
    data : jax.Array, shape (6, n, n)
        Scalar field on the cubed-sphere.
    halo : int
        Halo width (default 1).
    interp_offsets : jax.Array or None
        Precomputed fractional-index offsets, shape (6, 4, n).
        When ``None``, the standard nearest-index copy is used.
    duogrid : DuoGridData or None
        Duo-Grid remapping data. Mutually exclusive with interp_offsets.

    Returns
    -------
    padded : jax.Array, shape (6, n+2*halo, n+2*halo)
    """
    if interp_offsets is not None and duogrid is not None:
        raise ValueError(
            "interp_offsets and duogrid are mutually exclusive"
        )
    if halo not in (1, 2, 3):
        raise NotImplementedError(
            f"Only halo=1, halo=2, and halo=3 are supported, got {halo}")

    # Validate interp_offsets shape for halo=3 (the new path added in
    # iter-499).  Without this guard, a wrongly-shaped offsets array
    # would fail with a cryptic IndexError deep inside
    # `_pad_halo_local_h3`.  halo=1/2 are intentionally *not* checked
    # here because several existing callers pass the h1 shape
    # `(6, 4, n)` together with `halo=2`; that usage is silently
    # reduced to nearest-neighbour-like interpolation by
    # `_pad_halo_local_h2` and any pure tightening would be a broader
    # refactor outside iter-500's Codex-driven guardrail scope.  The
    # halo=3 path has no legacy callers so we lock it down now.
    #
    # iter-501 (Codex stop-time review of iter-500): the ndim + depth
    # check alone is not strict — it accepts (10, 7, 3, 100) which
    # would then fail cryptically in the loop for face=6..9 / edge=4..6,
    # and even if it survived the indexing loop a strip shape
    # mismatch against `data.shape[1]` (the grid size `n`) would give
    # silently-wrong interpolation.  Now every axis is validated.
    if interp_offsets is not None and halo == 3:
        n = data.shape[1]
        expected = (6, 4, 3, n)
        if tuple(interp_offsets.shape) != expected:
            raise ValueError(
                f"halo=3 expects interp_offsets of shape {expected} "
                f"(6 faces, 4 edges, 3 halo depths, n = data.shape[1]); "
                f"got shape={tuple(interp_offsets.shape)}.")

    # When duogrid is active, suppress interp_offsets (use nearest copy + remap)
    offsets = None if duogrid is not None else interp_offsets

    # Single-face (regional panel) dispatch: wall boundary conditions.
    # Iter 29: gate the wall-BC fast path on the *non-MPI* backends.
    # When MPI is active and a rank owns exactly one face (compact
    # local layout), ``data.shape[0] == 1`` is the *normal* state and
    # we still need to exchange halos with neighbouring ranks — taking
    # the wall-BC branch silently zeroed out the inter-face coupling.
    if data.shape[0] == 1 and _halo_backend != "mpi":
        return _pad_halo_wall(data, halo)

    # MPI dispatch.
    if _halo_backend == "mpi":
        # iter-630: MPI halo=3 is now supported end-to-end.  The 2D scalar
        # (`_pad_halo_mpi_face_only` / `_pad_halo_mpi_tiled`) and 4D tensor
        # (`_pad_halo_mpi_4d_face_only` / `_pad_halo_mpi_4d_tiled`) paths
        # all handle three halo depths, and corner cells are filled by
        # `fill_corners_h3`.
        #
        # FV3_3D 2026-05-27: option (a) implemented — `pad_halo_mpi`
        # now carries `interp_offsets` through to the face-only receive
        # path and applies `interp_strip` strip-by-strip.  This brings
        # the duogrid Lagrange-extrapolated halo to MPI, making the
        # FV3 3D PE/NH cubed-sphere paths bit-for-bit identical to the
        # single-device backend under MPI.  Sub-face tiling still
        # refuses — the `(6, 4, n)` offsets are global-face-indexed,
        # tile-local indexing has not been derived.
        from legoesm.parallel.halo_exchange import pad_halo_mpi
        padded = pad_halo_mpi(
            data, _mpi_topology, halo=halo, interp_offsets=offsets,
        )
    # SPMD dispatch (explicit all_gather for multi-GPU).
    elif _halo_backend == "spmd" and _spmd_mesh is not None:
        if halo == 3:
            raise NotImplementedError(
                "SPMD halo=3 exchange not yet implemented; "
                "halo=3 is only available on the single-node local path.")
        from legoesm.parallel.cubesphere_exchange import explicit_pad_halo
        padded = explicit_pad_halo(
            data, _spmd_mesh, halo=halo, interp_offsets=offsets,
        )
    elif halo == 1:
        padded = pad_halo_local(data, offsets)
    elif halo == 2:
        padded = _pad_halo_local_h2(data, offsets)
    else:  # halo == 3
        padded = _pad_halo_local_h3(data, offsets)

    # Duo-Grid post-processing: kinked→extended remap + corner fill.
    # iter-533: halo=3 is now ALLOWED for the duogrid path:
    # `cube_rmp_vectorized` already loops `for d in range(min(halo,
    # duogrid.ng))` (line 775), and `fill_corner_region` falls back
    # to averaging when `h > duogrid.ng` (line 893).  So:
    #   - duogrid.ng >= 3: full Lagrange remap at all 3 halo depths
    #   - duogrid.ng <  3: outer halo-3 cells get averaging-fallback
    #     instead of Lagrange, matching the iter-114 behaviour for
    #     halo > ng.
    if duogrid is not None:
        from legoesm.grids.duogrid import cube_rmp_vectorized, fill_corner_region
        padded = cube_rmp_vectorized(padded, duogrid, halo)
        padded = fill_corner_region(
            padded, duogrid, halo,
            monotone_clip=monotone_clip,
            monotone_clip_slack=monotone_clip_slack,
        )

    return padded


def pad_halo_pair_h2(
    q1: jax.Array,
    q2: jax.Array,
    interp_offsets: jax.Array | None = None,
    duogrid=None,
    monotone_clip: bool = False,
    monotone_clip_slack: float = 0.0,
) -> tuple[jax.Array, jax.Array]:
    """Halo=2 exchange a pair of independent ``(6, n, n)`` fields.

    Under the SPMD backend, the two fields ride a single
    ``packed_pad_halo_4d(halo=2)`` collective — dropping per-call halo=2
    cross-device collective count from 2 → 1 when both fields can be
    exchanged together (e.g. PPM transport's ``q_i`` and ``q_j``,
    which depend only on the already-padded ``q``).  Under the
    local / MPI backends this falls through to two sequential
    ``pad_halo(halo=2)`` calls — same arithmetic, no extra overhead.

    Parameters
    ----------
    q1, q2 : jax.Array, shape ``(6, n, n)``
    interp_offsets : optional
        Forwarded to the underlying SPMD / local kernels (3D h2
        offsets shaped ``(6, 4, 2, n)`` when ``duogrid is None``).
    duogrid : DuoGridData or None

    Returns
    -------
    (q1_pad, q2_pad) : tuple of jax.Array, each shape ``(6, n+4, n+4)``.
    """
    if _halo_backend == "spmd" and _spmd_mesh is not None:
        from legoesm.parallel.cubesphere_exchange import packed_pad_halo_4d
        offsets = None if duogrid is not None else interp_offsets
        # Add a singleton trailing axis so the SPMD packed kernel —
        # which targets 4D ``(6, n, n, C)`` inputs — can ride the same
        # all_gather.  Squeeze the channel axis off on return.
        q1_4d = q1[..., None]
        q2_4d = q2[..., None]
        q1_pad_4d, q2_pad_4d = packed_pad_halo_4d(
            q1_4d, q2_4d, mesh=_spmd_mesh,
            duogrid=duogrid, interp_offsets=offsets, halo=2,
        )
        return q1_pad_4d[..., 0], q2_pad_4d[..., 0]
    if _halo_backend == "mpi" and _mpi_topology is not None:
        # MPI: ``packed_pad_halo_mpi_4d`` supports halo=2 and halves the
        # MPI message count from 2 → 1 by stacking the two fields along
        # the trailing axis.  FV3_3D iter-1041: also threads
        # ``interp_offsets``, so the single packed exchange handles both
        # the no-offsets path and the duogrid Lagrange-remap path.
        from legoesm.parallel.halo_exchange import packed_pad_halo_mpi_4d
        q1_4d = q1[..., None]
        q2_4d = q2[..., None]
        q1_pad_4d, q2_pad_4d = packed_pad_halo_mpi_4d(
            q1_4d, q2_4d, topology=_mpi_topology,
            halo=2, duogrid=duogrid,
            interp_offsets=interp_offsets,
        )
        return q1_pad_4d[..., 0], q2_pad_4d[..., 0]
    # Local backend (or MPI-with-offsets — handled above): two
    # sequential pad_halo calls with identical arithmetic.
    q1_pad = pad_halo(q1, halo=2, interp_offsets=interp_offsets,
                      duogrid=duogrid,
                      monotone_clip=monotone_clip,
                      monotone_clip_slack=monotone_clip_slack)
    q2_pad = pad_halo(q2, halo=2, interp_offsets=interp_offsets,
                      duogrid=duogrid,
                      monotone_clip=monotone_clip,
                      monotone_clip_slack=monotone_clip_slack)
    return q1_pad, q2_pad


def pad_halo_4d(
    data: jax.Array,
    halo: int = 1,
    interp_offsets: jax.Array | None = None,
    duogrid=None,
    monotone_clip: bool = False,
    monotone_clip_slack: float = 0.0,
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
    duogrid : DuoGridData or None
        Duo-Grid remapping data. Mutually exclusive with interp_offsets.

    Returns
    -------
    padded : jax.Array, shape (6, n+2*halo, n+2*halo, nlev)
    """
    if data.ndim != 4:
        raise ValueError(f"pad_halo_4d expects 4D input, got {data.ndim}D")
    # FV3_3D iter-1072: non-square (n_x, n_y) data is silently
    # corrupted by ``pad_halo_local_4d`` / ``_pad_halo_mpi_face_only_4d``
    # because both use ``_get_halo_tables_h1(n=data.shape[1])`` which
    # assumes square shape — the y-axis halo cells past index n_x are
    # left at the jnp.pad default of 0, and the cells filled past
    # ``data.shape[2]`` index out-of-bounds.  Probe verified at
    # iter-1072: pad_halo_4d on shape ``(6, 5, 6, 1)`` returns
    # face-0 WEST halo with 6th cell corrupted, face-0 NORTH halo
    # all zero.  Loud error here surfaces the silent bug for any
    # caller passing non-square data (notably
    # ``use_fv3_cross_face_du_proj`` paths on ``du_normal`` /
    # ``dv_normal`` D-grid wind increments with shape
    # ``(6, n+1, n, nlev)`` / ``(6, n, n+1, nlev)``).
    if data.shape[0] >= 1 and data.shape[1] != data.shape[2]:
        raise ValueError(
            f"pad_halo_4d expects square (n, n) data on each face, "
            f"got shape {tuple(data.shape)} with data.shape[1] != "
            f"data.shape[2].  Non-square halo is a known iter-1046 "
            f"follow-up — see FV3_3D.md iter-1072 for the silent-"
            f"corruption probe.  Workaround: extend non-square data "
            f"to square via jnp.pad before calling pad_halo_4d, then "
            f"trim back, OR disable use_fv3_cross_face_du_proj."
        )
    if interp_offsets is not None and duogrid is not None:
        raise ValueError(
            "interp_offsets and duogrid are mutually exclusive"
        )
    if halo not in (1, 2, 3):
        raise NotImplementedError(
            f"Only halo=1, halo=2, and halo=3 are supported, got {halo}")
    # iter-723: halo=3 single-node 4D now supported via
    # `_pad_halo_local_h3_4d` (below).  MPI halo=3 4D already supported
    # via `pad_halo_mpi_4d` (iter-630).  SPMD halo=3 4D still not
    # implemented (checked below).

    # Iter-724 (Codex iter-723 finding): validate halo=3 interp_offsets
    # shape at the 4D entry mirroring the scalar `pad_halo` guard
    # (iter-500/501).  Without this, a wrongly-shaped offsets array
    # would fail cryptically in `_pad_halo_local_h3_4d`'s strip
    # loop.  Matches the scalar guard at line 531.  halo=1/2 are
    # NOT checked (legacy callers pass h1-shaped offsets with halo=2
    # per existing precedent in `pad_halo`).
    if interp_offsets is not None and halo == 3:
        n = data.shape[1]
        expected = (6, 4, 3, n)
        if tuple(interp_offsets.shape) != expected:
            raise ValueError(
                f"halo=3 expects interp_offsets of shape {expected} "
                f"(6 faces, 4 edges, 3 halo depths, n = data.shape[1]); "
                f"got shape={tuple(interp_offsets.shape)}.")

    offsets = None if duogrid is not None else interp_offsets

    # Single-face (regional panel) dispatch: wall boundary conditions.
    # Iter 29: gate on non-MPI backends (see pad_halo above).
    if data.shape[0] == 1 and _halo_backend != "mpi":
        return _pad_halo_wall(data, halo)

    # MPI dispatch.
    if _halo_backend == "mpi":
        # FV3_3D 2026-05-27: option (a) implemented — `pad_halo_mpi_4d`
        # now carries `interp_offsets` through and applies `interp_strip`
        # per (face, edge[, depth]) strip on the receive side.  Sibling
        # of the scalar `pad_halo` fix in the same iteration.  Sub-face
        # tiling still refuses (offsets are global-face-indexed).
        from legoesm.parallel.halo_exchange import pad_halo_mpi_4d
        padded = pad_halo_mpi_4d(
            data, _mpi_topology, halo=halo, interp_offsets=offsets,
        )
    # SPMD dispatch (explicit all_gather for multi-GPU).
    elif _halo_backend == "spmd" and _spmd_mesh is not None:
        if halo == 3:
            raise NotImplementedError(
                "SPMD 4D halo=3 exchange not yet implemented; "
                "halo=3 is only available on MPI and single-node paths.")
        from legoesm.parallel.cubesphere_exchange import explicit_pad_halo_4d
        padded = explicit_pad_halo_4d(
            data, _spmd_mesh, halo=halo, interp_offsets=offsets,
        )
    elif halo == 1:
        padded = pad_halo_local_4d(data, offsets)
    elif halo == 2:
        padded = _pad_halo_local_h2_4d(data, offsets)
    else:  # halo == 3 (iter-723)
        padded = _pad_halo_local_h3_4d(data, offsets)

    # Duo-Grid post-processing: apply per-level via vmap
    if duogrid is not None:
        from legoesm.grids.duogrid import cube_rmp_vectorized, fill_corner_region

        def _remap_level(level_slice):
            """Apply cube_rmp + corner fill to one (6, n+2h, n+2h) level."""
            level_slice = cube_rmp_vectorized(level_slice, duogrid, halo)
            level_slice = fill_corner_region(
                level_slice, duogrid, halo,
                monotone_clip=monotone_clip,
                monotone_clip_slack=monotone_clip_slack,
            )
            return level_slice

        # Transpose to (nlev, 6, n+2h, n+2h), vmap, transpose back
        padded.shape[3]
        padded_t = jnp.transpose(padded, (3, 0, 1, 2))  # (nlev, 6, ...)
        padded_t = jax.vmap(_remap_level)(padded_t)
        padded = jnp.transpose(padded_t, (1, 2, 3, 0))

    return padded


def pad_halo_local_4d(
    data: jax.Array,
    interp_offsets: jax.Array | None = None,
) -> jax.Array:
    """Local 4D scalar halo exchange for halo=1.

    Uses the same precomputed index tables as the 2D version.  The
    trailing level axis is preserved via ``data[src_f, src_i, src_j]``
    which yields shape ``(24*n, nlev)`` when data is ``(6, n, n, nlev)``.
    """
    n = data.shape[1]
    data.shape[3]
    tables = _get_halo_tables_h1(n)
    src_f, src_i, src_j, dst_f, dst_i, dst_j = tables

    # Single Pad HLO op replaces alloc-zeros + scatter.
    padded = jnp.pad(data, ((0, 0), (1, 1), (1, 1), (0, 0)))

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
        # Promote f to data dtype so Lagrange weights have full precision.
        f = (frac - jc.astype(frac.dtype)).astype(data.dtype)
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

    padded = fill_corners_h1(padded)
    return padded


def _pad_halo_local_h2_4d(
    data: jax.Array,
    interp_offsets: jax.Array | None = None,
) -> jax.Array:
    """Local 4D scalar halo exchange for halo=2.

    Two paths:

    * ``interp_offsets is None`` — the common fast path: gather all
      ``48 × n`` halo source cells via a precomputed index table and
      scatter them into the padded array in a single ``.at[].set``
      call.  Mirrors the halo=1 vectorisation
      (:func:`pad_halo_local_4d`) and replaces the previous 48-step
      ``.at[].set`` loop body, which produced one XLA scatter per
      step.
    * ``interp_offsets is not None`` — keep the loop-based path because
      :func:`interp_strip` consumes a per-edge offset ``(n,)`` array
      and would require a separate per-edge gather to vectorise; this
      branch is exercised only by Lagrange-corrected halos where the
      runtime cost is dominated by the interpolation itself.
    """
    n = data.shape[1]
    data.shape[3]
    # Single Pad HLO op replaces alloc-zeros + scatter.
    padded = jnp.pad(data, ((0, 0), (2, 2), (2, 2), (0, 0)))

    if interp_offsets is None:
        src_f, src_i, src_j, dst_f, dst_i, dst_j = _get_halo_tables_h2(n)
        values = data[src_f, src_i, src_j]   # (48*n, nlev)
        padded = padded.at[dst_f, dst_i, dst_j].set(values)
        padded = fill_corners_h2(padded)
        return padded

    edges = [WEST, EAST, SOUTH, NORTH]

    for face in range(6):
        for edge_idx, edge in enumerate(edges):
            nbr_face, nbr_edge, is_reversed = CONNECTIVITY[face][edge]

            for depth in range(2):
                strip = extract_edge_strip_at_depth(
                    data, nbr_face, nbr_edge, depth,
                )  # (n,) for 3D or (n, nlev) for 4D

                if is_reversed:
                    strip = strip[::-1]

                strip = interp_strip(
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

    padded = fill_corners_h2(padded)
    return padded


def _pad_halo_local_h3_4d(
    data: jax.Array,
    interp_offsets: jax.Array | None = None,
) -> jax.Array:
    """Local 4D scalar halo exchange for halo=3.

    Iter-723 port of :func:`_pad_halo_local_h3` (3D scalar halo=3) to
    the 4D multi-level case.  The 4D form is required for FB-chain
    production work (review-doc item #2): multi-level scalar fields
    (e.g., 3D height, vorticity snapshots) need halo=3 quality to
    match Fortran's ng=3 halo semantic at cube face boundaries and
    vertices.

    Parameters
    ----------
    data : jax.Array, shape (6, n, n, nlev)
    interp_offsets : jax.Array or None, shape (6, 4, 3, n)
        Precomputed fractional-index offsets for 3 halo depths (see
        :func:`compute_halo_interp_offsets_h3`).

    Returns
    -------
    padded : jax.Array, shape (6, n+6, n+6, nlev)
    """
    n = data.shape[1]
    nlev = data.shape[3]
    padded = jnp.zeros((6, n + 6, n + 6, nlev), dtype=data.dtype)

    # Place interior data
    padded = padded.at[:, 3:-3, 3:-3, :].set(data)

    edges = [WEST, EAST, SOUTH, NORTH]

    for face in range(6):
        for edge_idx, edge in enumerate(edges):
            nbr_face, nbr_edge, is_reversed = CONNECTIVITY[face][edge]

            for depth in range(3):
                # Extract neighbour strip at this depth — for a 4D
                # array `data[face, i, j]` returns (nlev,) so the
                # strip along an edge has shape (n, nlev).
                strip = extract_edge_strip_at_depth(
                    data, nbr_face, nbr_edge, depth,
                )

                if is_reversed:
                    strip = strip[::-1]

                if interp_offsets is not None:
                    strip = interp_strip(
                        strip, interp_offsets[face, edge_idx, depth],
                    )

                # Interior is at [3:-3, 3:-3, :], so:
                #   WEST  halo positions: i = 2 - depth
                #   EAST  halo positions: i = n + 3 + depth
                #   SOUTH halo positions: j = 2 - depth
                #   NORTH halo positions: j = n + 3 + depth
                if edge == WEST:
                    padded = padded.at[face, 2 - depth, 3:-3].set(strip)
                elif edge == EAST:
                    padded = padded.at[face, n + 3 + depth, 3:-3].set(strip)
                elif edge == SOUTH:
                    padded = padded.at[face, 3:-3, 2 - depth].set(strip)
                elif edge == NORTH:
                    padded = padded.at[face, 3:-3, n + 3 + depth].set(strip)

    # Fill 3×3 L-shaped corner regions (9 cells × 4 corners × 6 faces).
    padded = fill_corners_h3(padded)
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
    duogrid=None,
    monotone_clip: bool = False,
    monotone_clip_slack: float = 0.0,
) -> tuple[jax.Array, jax.Array]:
    """4D vector halo exchange (rotation + pad for all levels at once).

    Same logic as :func:`pad_halo_vector` but using :func:`pad_halo_4d`.
    Inputs are (6, n, n, nlev); rotation angles are (6, n, n) and get
    broadcast over the trailing level axis.

    Returns
    -------
    u_padded, v_padded : jax.Array, shape (6, n+2*halo, n+2*halo, nlev)
    """
    # FV3_3D iter-1073 (codex iter-1072 BLOCKER): mirror the iter-1072
    # non-square guard.  Vector halo also assumes square (n, n) per
    # face — all internal kernels and the SPMD/MPI dispatches share
    # the same square-shape assumption.
    if u_data.shape[1] != u_data.shape[2]:
        raise ValueError(
            f"pad_halo_vector_4d u_data expects square (n, n), got "
            f"shape {tuple(u_data.shape)}.  See FV3_3D.md iter-1072."
        )
    if v_data.shape[1] != v_data.shape[2]:
        raise ValueError(
            f"pad_halo_vector_4d v_data expects square (n, n), got "
            f"shape {tuple(v_data.shape)}.  See FV3_3D.md iter-1072."
        )
    # SPMD dispatch: pack both components into a single collective.
    if _halo_backend == "spmd" and _spmd_mesh is not None and halo == 1:
        from legoesm.parallel.cubesphere_exchange import (
            explicit_pad_halo_vector_4d,
        )
        # Iter-31: forward ``interp_offsets`` to the SPMD vector kernel.
        # Without this, the dycore's hyperdiffusion path
        # (``hyperdiffusion_3d`` → ``divergence_3d`` → here) silently
        # drops the Lagrange correction under SPMD, producing ~6e-4
        # relative drift on u/v after a single SSP-RK3 step at C24/L8.
        # ``duogrid``-on-grid runs already handle this through the
        # post-exchange remap; ``interp_offsets``-with-no-duogrid runs
        # need the explicit forwarding.
        offsets = None if duogrid is not None else interp_offsets
        return explicit_pad_halo_vector_4d(
            u_data, v_data,
            cos_angle, sin_angle,
            cos_angle_padded, sin_angle_padded,
            _spmd_mesh, halo=halo, interp_offsets=offsets,
        )

    # Broadcast 2D angles to match 4D data
    ca = cos_angle[..., None]
    sa = sin_angle[..., None]
    # Step 1: convert to geographic
    u_east = ca * u_data - sa * v_data
    v_north = sa * u_data + ca * v_data
    # Step 2: pad as scalars.
    # When MPI is active, pack both components along the level axis and
    # do one exchange instead of two, halving MPI message count.
    if _halo_backend == "mpi":
        # FV3_3D 2026-05-27: `pad_halo_mpi_4d` now honors `interp_offsets`,
        # so the packed (u_east, v_north) exchange can apply the Lagrange
        # remap once for both components.  This was previously a hard
        # `NotImplementedError` blocking the FV3 3D PE step under MPI
        # (the `hydrostatic_to_fv3` cell-center→D-grid corner lift uses
        # this path with `interp_offsets=base.halo_interp_offsets`).
        if duogrid is not None:
            # Iter-634: per-component scalar `pad_halo_4d` fallback.
            u_east_padded = pad_halo_4d(
                u_east, halo=halo, duogrid=duogrid,
                monotone_clip=monotone_clip,
                monotone_clip_slack=monotone_clip_slack,
            )
            v_north_padded = pad_halo_4d(
                v_north, halo=halo, duogrid=duogrid,
                monotone_clip=monotone_clip,
                monotone_clip_slack=monotone_clip_slack,
            )
        else:
            from legoesm.parallel.halo_exchange import pad_halo_mpi_4d
            packed = jnp.concatenate([u_east, v_north], axis=-1)  # (6, n, n, 2*nlev)
            packed_padded = pad_halo_mpi_4d(
                packed, _mpi_topology, halo=halo,
                interp_offsets=interp_offsets,
            )
            nlev = u_data.shape[-1]
            u_east_padded = packed_padded[..., :nlev]
            v_north_padded = packed_padded[..., nlev:]
    else:
        u_east_padded = pad_halo_4d(
            u_east, halo=halo, interp_offsets=interp_offsets,
            duogrid=duogrid, monotone_clip=monotone_clip,
            monotone_clip_slack=monotone_clip_slack,
        )
        v_north_padded = pad_halo_4d(
            v_north, halo=halo, interp_offsets=interp_offsets,
            duogrid=duogrid, monotone_clip=monotone_clip,
            monotone_clip_slack=monotone_clip_slack,
        )
    # Step 3: convert back using padded angles
    cap = cos_angle_padded[..., None]
    sap = sin_angle_padded[..., None]
    u_padded = cap * u_east_padded + sap * v_north_padded
    v_padded = -sap * u_east_padded + cap * v_north_padded
    return u_padded, v_padded


def pad_halo_local(
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

    # Single Pad HLO op replaces alloc-zeros + scatter (the subsequent
    # halo scatter only writes into halo regions, which ``jnp.pad`` has
    # already zeroed).
    padded = jnp.pad(data, ((0, 0), (1, 1), (1, 1)))

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
        # cubed_sphere_edge_artifacts.md).
        flat_offsets = interp_offsets.reshape(-1)
        n_int = int(n)
        strip_base = jnp.repeat(jnp.arange(24) * n_int, n_int)  # (24*n,)
        j_local = jnp.tile(jnp.arange(n_int), 24)  # position within strip

        frac = j_local + flat_offsets
        frac = jnp.clip(frac, 0.0, n_int - 1.0)

        # 3-point Lagrange stencil centered at jc = round(frac)
        jc = jnp.clip(jnp.round(frac).astype(jnp.int32), 1, n_int - 2)
        # Promote f to data dtype so Lagrange weights have full precision.
        f = (frac - jc.astype(frac.dtype)).astype(data.dtype)
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
    padded = fill_corners_h1(padded)
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
    # Single Pad HLO op replaces alloc-zeros + scatter (the subsequent
    # halo scatters only write into halo regions, which ``jnp.pad`` has
    # already zeroed).
    padded = jnp.pad(data, ((0, 0), (2, 2), (2, 2)))

    edges = [WEST, EAST, SOUTH, NORTH]

    for face in range(6):
        for edge_idx, edge in enumerate(edges):
            nbr_face, nbr_edge, is_reversed = CONNECTIVITY[face][edge]

            for depth in range(2):
                # Extract neighbour strip at this depth
                strip = extract_edge_strip_at_depth(
                    data, nbr_face, nbr_edge, depth,
                )

                if is_reversed:
                    strip = strip[::-1]

                # Interpolate to correct physical position if offsets provided
                if interp_offsets is not None:
                    strip = interp_strip(
                        strip, interp_offsets[face, edge_idx, depth],
                    )

                # Place in halo: depth=0 → adjacent to interior, depth=1 → outer
                # Interior is at [2:-2, 2:-2], so:
                #   WEST halo positions: i=1 (depth=0), i=0 (depth=1)
                #   EAST halo positions: i=n+2 (depth=0), i=n+3 (depth=1)
                1 - depth  # WEST: depth0→1, depth1→0
                if edge == WEST:
                    padded = padded.at[face, 1 - depth, 2:-2].set(strip)
                elif edge == EAST:
                    padded = padded.at[face, n + 2 + depth, 2:-2].set(strip)
                elif edge == SOUTH:
                    padded = padded.at[face, 2:-2, 1 - depth].set(strip)
                elif edge == NORTH:
                    padded = padded.at[face, 2:-2, n + 2 + depth].set(strip)

    # Fill L-shaped corner regions (4 cells per corner × 4 corners × 6 faces)
    padded = fill_corners_h2(padded)

    return padded


def _pad_halo_local_h3(
    data: jax.Array,
    interp_offsets: jax.Array | None = None,
) -> jax.Array:
    """Local (single-node) scalar halo exchange for halo=3.

    Generalizes :func:`_pad_halo_local_h2` to 3 halo depths.  This is
    plumbing for the FB-path C36 stability work (review-doc item #2),
    which requires ng=3-equivalent halo quality to reach Fortran-
    faithful stability on the `_c_sw` first-order upwind mass flux.

    Parameters
    ----------
    data : jax.Array, shape (6, n, n)
    interp_offsets : jax.Array or None, shape (6, 4, 3, n)
        Precomputed fractional-index offsets for 3 halo depths (see
        :func:`compute_halo_interp_offsets_h3`).

    Returns
    -------
    padded : jax.Array, shape (6, n+6, n+6)
    """
    n = data.shape[1]
    padded = jnp.zeros((6, n + 6, n + 6), dtype=data.dtype)

    # Place interior data
    padded = padded.at[:, 3:-3, 3:-3].set(data)

    edges = [WEST, EAST, SOUTH, NORTH]

    for face in range(6):
        for edge_idx, edge in enumerate(edges):
            nbr_face, nbr_edge, is_reversed = CONNECTIVITY[face][edge]

            for depth in range(3):
                # Extract neighbour strip at this depth
                strip = extract_edge_strip_at_depth(
                    data, nbr_face, nbr_edge, depth,
                )

                if is_reversed:
                    strip = strip[::-1]

                # Interpolate to correct physical position if offsets provided
                if interp_offsets is not None:
                    strip = interp_strip(
                        strip, interp_offsets[face, edge_idx, depth],
                    )

                # Place in halo: depth=0 → adjacent to interior; depth=2 → outer
                # Interior is at [3:-3, 3:-3], so:
                #   WEST  halo positions: i = 2 - depth (2, 1, 0 for depths 0..2)
                #   EAST  halo positions: i = n + 3 + depth
                #   SOUTH halo positions: j = 2 - depth
                #   NORTH halo positions: j = n + 3 + depth
                if edge == WEST:
                    padded = padded.at[face, 2 - depth, 3:-3].set(strip)
                elif edge == EAST:
                    padded = padded.at[face, n + 3 + depth, 3:-3].set(strip)
                elif edge == SOUTH:
                    padded = padded.at[face, 3:-3, 2 - depth].set(strip)
                elif edge == NORTH:
                    padded = padded.at[face, 3:-3, n + 3 + depth].set(strip)

    # Fill L-shaped 3×3 corner regions (9 cells × 4 corners × 6 faces)
    padded = fill_corners_h3(padded)

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


_halo_table_cache_h2: dict[int, tuple] = {}


def _build_halo_tables_h2(n: int) -> tuple:
    """Build source/destination index arrays for vectorized halo=2 exchange.

    For each of the 48*n halo cells (6 faces × 4 edges × 2 depths × n
    cells per edge), stores the (face, i, j) source coordinates in the
    original data array and the (face, i, j) destination coordinates
    in the padded ``(n+4)×(n+4)`` array.  Mirrors
    :func:`_build_halo_tables_h1` so the runtime exchange becomes a
    single gather + scatter instead of 48 ``.at[].set`` scatters.
    """
    edges = [WEST, EAST, SOUTH, NORTH]
    total = 48 * n

    src_f = np.zeros(total, dtype=np.int32)
    src_i = np.zeros(total, dtype=np.int32)
    src_j = np.zeros(total, dtype=np.int32)
    dst_f = np.zeros(total, dtype=np.int32)
    dst_i = np.zeros(total, dtype=np.int32)
    dst_j = np.zeros(total, dtype=np.int32)

    idx = 0
    for face in range(6):
        for edge in edges:
            nbr_face, nbr_edge, is_reversed = CONNECTIVITY[face][edge]
            for depth in range(2):
                for j in range(n):
                    k = (n - 1 - j) if is_reversed else j

                    # Source cell at this depth on the neighbour's
                    # edge — same convention as
                    # :func:`extract_edge_strip_at_depth`.
                    if nbr_edge == WEST:
                        sf, si, sj = nbr_face, depth, k
                    elif nbr_edge == EAST:
                        sf, si, sj = nbr_face, n - 1 - depth, k
                    elif nbr_edge == SOUTH:
                        sf, si, sj = nbr_face, k, depth
                    else:  # NORTH
                        sf, si, sj = nbr_face, k, n - 1 - depth
                    src_f[idx], src_i[idx], src_j[idx] = sf, si, sj

                    # Destination index in the (n+4)×(n+4) padded
                    # array.  Mirrors the layout used by the loop
                    # version: WEST halo at i = 1-depth (i.e. col 1 then
                    # col 0), EAST at n+2+depth, etc.  Interior occupies
                    # rows/cols [2, n+2).
                    if edge == WEST:
                        di, dj = 1 - depth, j + 2
                    elif edge == EAST:
                        di, dj = n + 2 + depth, j + 2
                    elif edge == SOUTH:
                        di, dj = j + 2, 1 - depth
                    else:  # NORTH
                        di, dj = j + 2, n + 2 + depth
                    dst_f[idx], dst_i[idx], dst_j[idx] = face, di, dj

                    idx += 1

    return (src_f, src_i, src_j, dst_f, dst_i, dst_j)


def _get_halo_tables_h2(n: int) -> tuple:
    """Return cached halo index tables for halo=2 of grid size n."""
    n = int(n)
    if n not in _halo_table_cache_h2:
        _halo_table_cache_h2[n] = _build_halo_tables_h2(n)
    return _halo_table_cache_h2[n]


def precompute_halo_tables(n: int) -> None:
    """Eagerly populate the halo index cache for grid size n.

    Call this at grid creation time (outside JIT) to avoid cache
    writes during JAX tracing.
    """
    _get_halo_tables_h1(int(n))
    _get_halo_tables_h2(int(n))


# FV3_3D iter 7: corner-fill mode toggle.
#
# The 2-point-average ("avg") path is the legacy default (preserves
# all existing tests bit-for-bit).  The "fv3_agrid_xdir" mode applies
# the FV3-faithful diagonal-mirror corner fill from
# fv_mp_mod.F90:fill_corners_2d_r8 AGRID/XDir branch (line 1077): for
# the SW vertex halo at padded (0, 0), use ``q[0, 1]`` (the cell
# immediately to the south of the SW interior corner, also a halo).
# This eliminates the direction-invariant smoothing in the 2-point
# average, restoring the FV3 directional preference.
#
# Empirical effect on Held-Suarez C36 hybrid 30-day:
#   day 30 max abs v: 2.56 m/s (avg) -> 1.35 m/s (fv3_agrid_xdir)  -47%
#   day 30 zonal_std: 0.520 -> 0.274                               -47%
#   day 30 eddy_std:  0.364 -> 0.260                               -29%
#
# Set via the ``LEGOESM_CORNER_FILL`` environment variable or via the
# ``set_corner_fill_mode(...)`` helper.  Default ``avg`` preserves the
# current production behaviour.
import os as _os

_corner_fill_mode = _os.environ.get("LEGOESM_CORNER_FILL", "avg")


def set_corner_fill_mode(mode: str) -> None:
    """Set the cube-vertex halo fill mode.

    Parameters
    ----------
    mode : {"avg", "fv3_agrid_xdir", "fv3_bgrid_xdir"}
        - ``"avg"`` (default): legacy 2-point average — direction-
          invariant; symmetric combination of FV3's XDir and YDir.
        - ``"fv3_agrid_xdir"``: FV3-faithful AGRID-XDir diagonal
          mirror (``fv_mp_mod.F90:1077``).  Each cube-vertex halo
          takes its value from the IMMEDIATELY-ADJACENT halo strip
          (depth-1 mirror).  Reduces HS C36 mid-level cube imprint
          by 47 % at day 30, but increases max-over-all-levels
          max abs v by 56 % (extreme-level jet release).
        - ``"fv3_bgrid_xdir"``: FV3-faithful BGRID-XDir diagonal
          mirror (``fv_mp_mod.F90:1041``).  Each cube-vertex halo
          takes its value from the FACE-INTERIOR cell at depth 2
          along the XDir direction.  Reduces HS C36 mid-level cube
          imprint by 41 % at day 30 AND reduces max-over-all-levels
          max abs v by 15 %.  **Cleanest win** of the three modes.
          See FV3_3D.md iter 10.
    """
    global _corner_fill_mode
    valid_modes = ("avg", "fv3_agrid_xdir", "fv3_bgrid_xdir")
    if mode not in valid_modes:
        raise ValueError(
            f"Unknown corner fill mode: {mode!r}.  "
            f"Choose from {valid_modes}."
        )
    _corner_fill_mode = mode


def get_corner_fill_mode() -> str:
    """Return the current cube-vertex halo fill mode."""
    return _corner_fill_mode


def fill_corners_h1(padded: jax.Array) -> jax.Array:
    """Fill corner cells of halo=1 padded array (cube vertices, 24 cells).

    Two modes via :func:`set_corner_fill_mode`:

    - ``"avg"`` (default): legacy 2-point average of two adjacent halo
      cells.  Direction-invariant; symmetric combination of FV3's
      XDir and YDir.

    - ``"fv3_agrid_xdir"``: FV3-faithful AGRID-XDir diagonal mirror
      from ``fv_mp_mod.F90:fill_corners_2d_r8`` (line 1077).  Each
      cube-vertex halo cell takes its value from the cell immediately
      adjacent in the XDir direction:
          SW (0, 0)         ← (0, 1)
          NW (0, n+1)       ← (0, n)
          SE (n+1, 0)       ← (n+1, 1)
          NE (n+1, n+1)     ← (n+1, n)
      This restores the FV3 directional preference and reduces the 3D
      atmospheric cube imprint on Held-Suarez C36 hybrid by ~47 percent
      (max abs v, day 30).  See FV3_3D.md iter 7.

    Vectorised: all 24 corners (6 faces × 4 corners) in a single
    gather + scatter.

    Parameters
    ----------
    padded : jax.Array, shape (6, n+2, n+2)

    Returns
    -------
    jax.Array, shape (6, n+2, n+2)
    """
    n2i = padded.shape[1] - 1  # n+1 (last index in padded)

    # All 24 corner cells: (face, i, j)
    f_idx = jnp.arange(6)
    cf = jnp.repeat(f_idx, 4)
    ci = jnp.tile(jnp.array([0, n2i, 0, n2i]), 6)
    cj = jnp.tile(jnp.array([0, 0, n2i, n2i]), 6)

    if _corner_fill_mode == "fv3_agrid_xdir":
        # FV3 AGRID-XDir: depth-1 mirror in XDir direction.
        # SW: q[0, 0] = q[0, 1]
        # NW: q[0, -1] = q[0, -2]
        # SE: q[-1, 0] = q[-1, 1]
        # NE: q[-1, -1] = q[-1, -2]
        si = ci
        sj = jnp.tile(jnp.array([1, 1, n2i - 1, n2i - 1]), 6)
        corner_vals = padded[cf, si, sj]
    elif _corner_fill_mode == "fv3_bgrid_xdir":
        # FV3 BGRID-XDir (fv_mp_mod.F90:1041): depth-2 mirror — sample
        # at the FACE-INTERIOR cell two steps in the XDir direction.
        # SW: q[0, 0] = q[0, 2]
        # NW: q[0, -1] = q[0, -3]
        # SE: q[-1, 0] = q[-1, 2]
        # NE: q[-1, -1] = q[-1, -3]
        si = ci
        sj = jnp.tile(jnp.array([2, 2, n2i - 2, n2i - 2]), 6)
        corner_vals = padded[cf, si, sj]
    else:
        # Legacy 2-point average.
        a1i = jnp.tile(jnp.array([0, n2i, 0, n2i]), 6)
        a1j = jnp.tile(jnp.array([1, 1, n2i - 1, n2i - 1]), 6)
        a2i = jnp.tile(jnp.array([1, n2i - 1, 1, n2i - 1]), 6)
        a2j = jnp.tile(jnp.array([0, 0, n2i, n2i]), 6)
        corner_vals = 0.5 * (padded[cf, a1i, a1j] + padded[cf, a2i, a2j])

    padded = padded.at[cf, ci, cj].set(corner_vals)
    return padded


def fill_corners_h2(padded: jax.Array) -> jax.Array:
    """Fill L-shaped corner regions of halo=2 padded array.

    Two modes via :func:`set_corner_fill_mode`:

    - ``"avg"`` (default): legacy inside-out 2-point averaging.  Each
      cube-vertex 2×2 halo block (4 cells per corner × 4 corners ×
      6 faces = 96 cells) is filled inside-out: inner corner first
      (average of its two halo-strip neighbours), then propagate
      outward.

    - ``"fv3_agrid_xdir"``: FV3-faithful AGRID ``XDir`` diagonal
      mirror for ng=2, faithful port of ``fv_mp_mod.F90:1077``
      (``q(1-i, 1-j) = q(1-j, i)`` for i, j in {1, 2}).  In our
      0-based padded representation the SW 2×2 block is::

          (1, 1) ← (1, 2)
          (1, 0) ← (0, 2)
          (0, 1) ← (1, 3)
          (0, 0) ← (0, 3)

      Empirical: in the iter-7 HS C36 hybrid 30-day diagnostic, the
      h2 toggle ALONE produces zero change vs the avg path because
      the Held-Suarez dycore does not exercise any operator that
      reads the cube-vertex 2×2 halo block (PPM 1D sweeps slice to
      keep either i-halo or j-halo, never both — see iter-69 review
      note above).  The h2 toggle is wired here for FV3-fidelity
      symmetry with h1, but is currently a no-op for the 3D HS
      case.  Will become active in iter 9+ once a forward-backward
      operator that uses 2x2 corner halos is added.

    Parameters
    ----------
    padded : jax.Array, shape (6, n+4, n+4)

    Returns
    -------
    jax.Array, shape (6, n+4, n+4)
    """
    if _corner_fill_mode == "fv3_agrid_xdir":
        # Vectorised FV3 AGRID-XDir for ng=2 (4 cells × 4 corners × 6 faces).
        # SW block (0..1, 0..1).
        padded = padded.at[:, 1, 1].set(padded[:, 1, 2])
        padded = padded.at[:, 1, 0].set(padded[:, 0, 2])
        padded = padded.at[:, 0, 1].set(padded[:, 1, 3])
        padded = padded.at[:, 0, 0].set(padded[:, 0, 3])
        # NW block (0..1, -2..-1) — mirror of SW in the j direction.
        padded = padded.at[:, 1, -2].set(padded[:, 1, -3])
        padded = padded.at[:, 1, -1].set(padded[:, 0, -3])
        padded = padded.at[:, 0, -2].set(padded[:, 1, -4])
        padded = padded.at[:, 0, -1].set(padded[:, 0, -4])
        # SE block (-2..-1, 0..1).
        padded = padded.at[:, -2, 1].set(padded[:, -2, 2])
        padded = padded.at[:, -2, 0].set(padded[:, -1, 2])
        padded = padded.at[:, -1, 1].set(padded[:, -2, 3])
        padded = padded.at[:, -1, 0].set(padded[:, -1, 3])
        # NE block (-2..-1, -2..-1).
        padded = padded.at[:, -2, -2].set(padded[:, -2, -3])
        padded = padded.at[:, -2, -1].set(padded[:, -1, -3])
        padded = padded.at[:, -1, -2].set(padded[:, -2, -4])
        padded = padded.at[:, -1, -1].set(padded[:, -1, -4])
        return padded
    elif _corner_fill_mode == "fv3_bgrid_xdir":
        # iter 10 diagnostic showed that applying the FV3 BGRID-XDir
        # depth-2/3-4 mirror to the 2×2 cube-vertex L-block destabilises
        # the 3D HS dycore (NaN around step 600 ≈ 1.4 days).  The
        # mirror reads cells from the face interior at depth 3/4, which
        # under sigma-coord HS produces an exponentially-growing
        # mode incompatible with our (A-L gradient + RK3) chain.
        #
        # The iter-7 review note already established that no operator
        # in the 3D PE path reads the 2×2 cube-vertex halo block (PPM
        # 1D sweeps slice to keep either i-halo or j-halo, never both
        # simultaneously); therefore the h2 BGRID mode is unsafe AND
        # unnecessary.  Fall through to the legacy 2-point-average
        # path to maintain stability for both sigma and hybrid coord.
        # h1 BGRID (the actually load-bearing path for cube imprint
        # reduction) remains active.
        pass  # fall through to legacy avg path

    # Legacy inside-out 2-point average path — VECTORISED over the 6-face axis.
    # Was a ``for f in range(6)`` loop = 96 serialised ``.at[f,i,j]`` scatters;
    # now 16 batched ``.at[:,i,j]`` scatters (bit-identical: each face reads
    # ONLY its own cells so the faces are independent, and the inner->outer
    # dependency WITHIN each corner is preserved by the statement order).
    # Mirrors the already-vectorised fv3_agrid_xdir branch above (per-device
    # kernel-count reduction; deep-dive 2026-06-14 lever #1).
    # --- SW corner (rows 0-1, cols 0-1) --- inner (1,1) first, then outward.
    padded = padded.at[:, 1, 1].set(0.5 * (padded[:, 1, 2] + padded[:, 2, 1]))
    padded = padded.at[:, 0, 1].set(0.5 * (padded[:, 0, 2] + padded[:, 1, 1]))
    padded = padded.at[:, 1, 0].set(0.5 * (padded[:, 2, 0] + padded[:, 1, 1]))
    padded = padded.at[:, 0, 0].set(0.5 * (padded[:, 0, 1] + padded[:, 1, 0]))
    # --- SE corner (rows n+2..n+3, cols 0-1) ---
    padded = padded.at[:, -2, 1].set(0.5 * (padded[:, -2, 2] + padded[:, -3, 1]))
    padded = padded.at[:, -1, 1].set(0.5 * (padded[:, -1, 2] + padded[:, -2, 1]))
    padded = padded.at[:, -2, 0].set(0.5 * (padded[:, -3, 0] + padded[:, -2, 1]))
    padded = padded.at[:, -1, 0].set(0.5 * (padded[:, -1, 1] + padded[:, -2, 0]))
    # --- NW corner (rows 0-1, cols n+2..n+3) ---
    padded = padded.at[:, 1, -2].set(0.5 * (padded[:, 1, -3] + padded[:, 2, -2]))
    padded = padded.at[:, 0, -2].set(0.5 * (padded[:, 0, -3] + padded[:, 1, -2]))
    padded = padded.at[:, 1, -1].set(0.5 * (padded[:, 2, -1] + padded[:, 1, -2]))
    padded = padded.at[:, 0, -1].set(0.5 * (padded[:, 0, -2] + padded[:, 1, -1]))
    # --- NE corner (rows n+2..n+3, cols n+2..n+3) ---
    padded = padded.at[:, -2, -2].set(0.5 * (padded[:, -2, -3] + padded[:, -3, -2]))
    padded = padded.at[:, -1, -2].set(0.5 * (padded[:, -1, -3] + padded[:, -2, -2]))
    padded = padded.at[:, -2, -1].set(0.5 * (padded[:, -3, -1] + padded[:, -2, -2]))
    padded = padded.at[:, -1, -1].set(0.5 * (padded[:, -1, -2] + padded[:, -2, -1]))

    return padded


def fill_corners_h3(padded: jax.Array) -> jax.Array:
    """Fill L-shaped 3×3 corner regions of halo=3 padded array.

    Each face has 4 corner regions of 3×3 = 9 cells that are not
    filled by the edge-strip exchange.  Fill inside-out so each cell
    depends only on already-filled neighbours:

    ``(2,2)`` (diagonal from interior) first, using adjacent edge
    halos; then the axis-aligned cells outward along each arm; then
    the interior-to-outer diagonals; finally the outermost corner
    ``(0,0)``.

    Parameters
    ----------
    padded : jax.Array, shape (6, n+6, n+6)

    Returns
    -------
    jax.Array, shape (6, n+6, n+6)
    """
    for f in range(6):
        # --- SW corner (rows 0..2, cols 0..2) ---
        # Interior-adjacent cells along the two arms:
        #   (2, 3..) is WEST depth=0 halo (set)
        #   (3, 2)  is SOUTH depth=0 halo (set)
        # So (2,2) = avg of those two neighbours.
        padded = padded.at[f, 2, 2].set(
            0.5 * (padded[f, 2, 3] + padded[f, 3, 2])
        )
        # (1,2): WEST depth=1 (padded[f,1,3]) and (2,2) just filled
        padded = padded.at[f, 1, 2].set(
            0.5 * (padded[f, 1, 3] + padded[f, 2, 2])
        )
        # (2,1): SOUTH depth=1 (padded[f,3,1]) and (2,2) just filled
        padded = padded.at[f, 2, 1].set(
            0.5 * (padded[f, 3, 1] + padded[f, 2, 2])
        )
        # (0,2): WEST depth=2 (padded[f,0,3]) and (1,2) just filled
        padded = padded.at[f, 0, 2].set(
            0.5 * (padded[f, 0, 3] + padded[f, 1, 2])
        )
        # (2,0): SOUTH depth=2 (padded[f,3,0]) and (2,1) just filled
        padded = padded.at[f, 2, 0].set(
            0.5 * (padded[f, 3, 0] + padded[f, 2, 1])
        )
        # (1,1): diagonal interior-ward, average of just-filled (1,2) and (2,1)
        padded = padded.at[f, 1, 1].set(
            0.5 * (padded[f, 1, 2] + padded[f, 2, 1])
        )
        # (0,1): average of just-filled (0,2) and (1,1)
        padded = padded.at[f, 0, 1].set(
            0.5 * (padded[f, 0, 2] + padded[f, 1, 1])
        )
        # (1,0): average of just-filled (2,0) and (1,1)
        padded = padded.at[f, 1, 0].set(
            0.5 * (padded[f, 2, 0] + padded[f, 1, 1])
        )
        # (0,0): outermost, average of just-filled (0,1) and (1,0)
        padded = padded.at[f, 0, 0].set(
            0.5 * (padded[f, 0, 1] + padded[f, 1, 0])
        )

        # --- SE corner (rows n+3..n+5, cols 0..2) ---
        padded = padded.at[f, -3, 2].set(
            0.5 * (padded[f, -3, 3] + padded[f, -4, 2])
        )
        padded = padded.at[f, -2, 2].set(
            0.5 * (padded[f, -2, 3] + padded[f, -3, 2])
        )
        padded = padded.at[f, -3, 1].set(
            0.5 * (padded[f, -4, 1] + padded[f, -3, 2])
        )
        padded = padded.at[f, -1, 2].set(
            0.5 * (padded[f, -1, 3] + padded[f, -2, 2])
        )
        padded = padded.at[f, -3, 0].set(
            0.5 * (padded[f, -4, 0] + padded[f, -3, 1])
        )
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

        # --- NW corner (rows 0..2, cols n+3..n+5) ---
        padded = padded.at[f, 2, -3].set(
            0.5 * (padded[f, 2, -4] + padded[f, 3, -3])
        )
        padded = padded.at[f, 1, -3].set(
            0.5 * (padded[f, 1, -4] + padded[f, 2, -3])
        )
        padded = padded.at[f, 2, -2].set(
            0.5 * (padded[f, 3, -2] + padded[f, 2, -3])
        )
        padded = padded.at[f, 0, -3].set(
            0.5 * (padded[f, 0, -4] + padded[f, 1, -3])
        )
        padded = padded.at[f, 2, -1].set(
            0.5 * (padded[f, 3, -1] + padded[f, 2, -2])
        )
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

        # --- NE corner (rows n+3..n+5, cols n+3..n+5) ---
        padded = padded.at[f, -3, -3].set(
            0.5 * (padded[f, -3, -4] + padded[f, -4, -3])
        )
        padded = padded.at[f, -2, -3].set(
            0.5 * (padded[f, -2, -4] + padded[f, -3, -3])
        )
        padded = padded.at[f, -3, -2].set(
            0.5 * (padded[f, -4, -2] + padded[f, -3, -3])
        )
        padded = padded.at[f, -1, -3].set(
            0.5 * (padded[f, -1, -4] + padded[f, -2, -3])
        )
        padded = padded.at[f, -3, -1].set(
            0.5 * (padded[f, -4, -1] + padded[f, -3, -2])
        )
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
    duogrid=None,
    cos_theta: jax.Array | None = None,
    sin_theta: jax.Array | None = None,
    monotone_clip: bool = False,
    monotone_clip_slack: float = 0.0,
) -> tuple[jax.Array, jax.Array]:
    """Pad vector field components with proper rotation at face boundaries.

    Grid-aligned velocity components (u, v) change meaning across face
    boundaries because each face has different grid axis orientations.
    This function handles the rotation correctly:

    1. Convert (u_grid, v_grid) -> (u_east, v_north) using grid angle
    2. Pad u_east, v_north as scalars (geographic components are continuous)
    3. Convert back to grid-aligned using the padded grid angle

    When cos_theta/sin_theta are provided (from cdgrid non-orthogonality
    metrics), the rotation accounts for the non-perpendicular grid axes
    on the cubed sphere.  This eliminates the O(cos_theta) error in the
    standard orthogonal rotation at face boundaries where non-orthogonality
    is largest.

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
    cos_theta, sin_theta : jax.Array or None, shape (6, n, n)
        Non-orthogonality metrics: cos/sin of the angle between grid axes.
        When None (default), uses the orthogonal rotation.

    Returns
    -------
    u_padded : jax.Array, shape (6, n+2*halo, n+2*halo)
        Padded grid-aligned x-velocity.
    v_padded : jax.Array, shape (6, n+2*halo, n+2*halo)
        Padded grid-aligned y-velocity.
    """
    # Iter-595: halo=3 support for the non-MPI single-device path.
    # Requires the caller to pass `cos_angle_padded_h3` /
    # `sin_angle_padded_h3` (shape (6, n+6, n+6)) as the
    # `cos_angle_padded` / `sin_angle_padded` args.  The scalar
    # `pad_halo(halo=3)` path has been validated since iter-499; the
    # vector round-trip just reuses that scalar exchange twice.
    #
    # Iter-630: `pad_halo_mpi_4d(halo=3)` was added in iter-627/628 and
    # the 2D scalar MPI paths gained halo=3 support in iter-630, so the
    # MPI backend now handles halo=3 end-to-end.  The prior
    # `NotImplementedError` guard that lived here has been removed.
    if halo not in (1, 2, 3):
        raise NotImplementedError(
            f"Only halo=1, halo=2 and halo=3 are supported for "
            f"pad_halo_vector, got {halo}")

    _EPS = float(jnp.finfo(jnp.float32).eps)

    if cos_theta is not None and sin_theta is not None:
        # Non-orthogonal rotation (exact for cubed-sphere grids).
        # utmp = V·x_hat, vtmp = V·y_hat, angle(y_hat, x_hat) = theta
        ca, sa = cos_angle, sin_angle
        ct, st = cos_theta, jnp.maximum(sin_theta, _EPS)
        u_east = ca * u_data + sa * (u_data * ct - v_data) / st
        v_north = sa * u_data + ca * (v_data - u_data * ct) / st
    else:
        # Orthogonal rotation (backward compatible)
        u_east = cos_angle * u_data - sin_angle * v_data
        v_north = sin_angle * u_data + cos_angle * v_data

    # Step 2: Pad geographic components as scalars.
    # When MPI is active, pack both into a single 4D exchange to halve
    # the MPI message count (one exchange instead of two).
    if _halo_backend == "mpi":
        # Iter-633 refused both `interp_offsets != None` and
        # `duogrid != None` under MPI because the packed MPI vector
        # path does NOT honor offsets or apply the duogrid
        # kinked→extended remap.  Iter-634 (Codex stop-time follow-up):
        # the duogrid refusal was overcautious — scalar `pad_halo`
        # already runs `cube_rmp_vectorized` + `fill_corner_region`
        # after dispatch regardless of backend, so per-component
        # fallback is correct.  The `interp_offsets` refusal stays
        # because `pad_halo_mpi` (invoked by the scalar fallback) does
        # not carry offsets either.
        # FV3_3D 2026-05-27: `pad_halo_mpi_4d` now honors `interp_offsets`,
        # so the packed (u_east, v_north) MPI exchange threads offsets
        # through.  Previously this was a hard NotImplementedError.
        if duogrid is not None:
            # Iter-634: per-component scalar fallback.  Pays 2 MPI
            # messages instead of 1 packed exchange, but exercises
            # `pad_halo`'s validated duogrid post-processing so the
            # kinked→extended remap actually runs.  Drop-in correct.
            u_east_padded = pad_halo(
                u_east, halo=halo, duogrid=duogrid,
                monotone_clip=monotone_clip,
                monotone_clip_slack=monotone_clip_slack,
            )
            v_north_padded = pad_halo(
                v_north, halo=halo, duogrid=duogrid,
                monotone_clip=monotone_clip,
                monotone_clip_slack=monotone_clip_slack,
            )
        else:
            from legoesm.parallel.halo_exchange import pad_halo_mpi_4d
            packed = jnp.stack([u_east, v_north], axis=-1)  # (6, n, n, 2)
            packed_padded = pad_halo_mpi_4d(
                packed, _mpi_topology, halo=halo,
                interp_offsets=interp_offsets,
            )
            u_east_padded = packed_padded[..., 0]
            v_north_padded = packed_padded[..., 1]
    else:
        u_east_padded = pad_halo(
            u_east, halo=halo, interp_offsets=interp_offsets,
            duogrid=duogrid,
            monotone_clip=monotone_clip,
            monotone_clip_slack=monotone_clip_slack,
        )
        v_north_padded = pad_halo(
            v_north, halo=halo, interp_offsets=interp_offsets,
            duogrid=duogrid,
            monotone_clip=monotone_clip,
            monotone_clip_slack=monotone_clip_slack,
        )

    # Step 3: Convert back to grid-aligned using padded angle
    cap, sap = cos_angle_padded, sin_angle_padded
    if cos_theta is not None and sin_theta is not None:
        # Non-orthogonal back-rotation: vtmp = cos_beta*u_east + sin_beta*v_north
        # Iter-838 (Codex stop-time review): replace `mode='edge'` padding
        # of `cos_theta`/`sin_theta` (same-face extension, loses cross-
        # face metric values at panel boundaries) with proper cross-face
        # halo exchange via `pad_halo`.  The non-orthogonality metrics
        # are scalar cell-centre fields, continuous across panel seams,
        # but their numerical values on face F's halo at a seam with
        # face G should come from G's metric, not a copy of F's.  Matches
        # Fortran's halo-exchanged `gridstruct%sin_sg(:,:,5)` /
        # `cos_sg(:,:,5)` semantics at panel boundaries.
        ct_pad = pad_halo(cos_theta, halo=halo, interp_offsets=interp_offsets,
                          duogrid=duogrid)
        st_pad = pad_halo(sin_theta, halo=halo, interp_offsets=interp_offsets,
                          duogrid=duogrid)
        cos_beta = cap * ct_pad - sap * st_pad
        sin_beta = sap * ct_pad + cap * st_pad
        u_padded = cap * u_east_padded + sap * v_north_padded
        v_padded = cos_beta * u_east_padded + sin_beta * v_north_padded
    else:
        u_padded = cap * u_east_padded + sap * v_north_padded
        v_padded = -sap * u_east_padded + cap * v_north_padded

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
        Halo width (1, 2, or 3 — iter-530 confirmed halo=3 numerically
        consistent with the halo=2 output at the overlapping interior
        region; supports the iter-496..501 ng=3 halo extension).

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
        lon, lat = face_gnomonic_to_lonlat(face, alpha_bx, alpha_by)

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


def face_gnomonic_to_lonlat(
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

    lon = jnp.mod(jnp.arctan2(y, x), 2.0 * jnp.pi)  # [0, 2π)
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
        Halo width (1, 2, or 3 — iter-530 confirmed halo=3
        numerically consistent with halo=2 at overlapping interior
        cells; supports the iter-496..501 ng=3 halo extension).

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
        lon, lat = face_gnomonic_to_lonlat(face, alpha_bx, alpha_by)
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


# ==============================================================================
# CGRID flux synchronization (duogrid face-boundary averaging)
# ==============================================================================

# Iter-808: empirical sign-flip table for `synchronize_cgrid_fluxes`.
# Certain face-to-face adjacencies (all involving polar faces 4 or 5)
# use OPPOSITE sign conventions for the mass flux across the shared
# edge, because the local (i, j) axes on the two faces point in
# different physical directions at the shared edge.  When averaging
# face A's flux with face B's flux, we need to sign-flip B's flux
# at these edges to obtain the physically-correct conservation
# average.
#
# iter-807 measured |A - B| / max(|A|, |B|) for all shared edges on
# W2 IC at C36.  Most edges had rel ~ 0 (both faces agree).  Four
# edges had rel = 2.00, indicating opposite signs: identified below.
#
# Fortran's mpp_get_boundary handles this internally; our Python
# extracts raw neighbor data and must apply the sign-flip explicitly.
_FLUX_SIGN_FLIP_EDGES = frozenset({
    (1, NORTH), (4, EAST),
    (2, SOUTH), (5, SOUTH),
    (2, NORTH), (4, NORTH),
    (3, SOUTH), (5, WEST),
})


def synchronize_cgrid_fluxes(fx, fy, n):
    """Average C-grid fluxes at shared face boundaries (duogrid conservation fix).

    FV3_3D iter-1049: under the MPI backend (``_halo_backend == "mpi"``)
    we dispatch to :func:`_synchronize_cgrid_fluxes_mpi`, which uses
    ``mpi4jax.sendrecv`` to swap boundary flux strips between ranks
    before averaging.  Without that swap, ``fx[nbr_face, ...]`` reads
    of non-owned faces return values computed with zero halos and
    contaminate the averaged result on OWNED faces.

    Implements the duogrid flux averaging from FV3 dyn_core.F90:853-900.
    Each shared face boundary flux is replaced by the average of both
    faces' independently computed boundary fluxes, ensuring that the mass
    flux leaving face A exactly equals the mass flux entering face B.

    This is required for conservation when using duogrid halo exchange,
    because each face computes boundary fluxes independently using its own
    extended grid, producing slightly different values at shared edges.

    Iter-808: polar-adjacent shared edges require a sign flip on the
    neighbor's flux before averaging, because the local (i, j) axes on
    the two faces point in opposite physical directions at those shared
    edges.  See ``_FLUX_SIGN_FLIP_EDGES`` above for the empirical table
    (derived from iter-807 flux-discrepancy measurements).  Without the
    sign flip, the sync reduced DUOGRID W2 t=0 dh/dt by 1172× vs the
    signed version (iter-806b/807/808).

    Parameters
    ----------
    fx : jax.Array, shape (6, n+1, n)
        x-direction flux at cell x-interfaces.
    fy : jax.Array, shape (6, n, n+1)
        y-direction flux at cell y-interfaces.
    n : int
        Number of cells per face edge.

    Returns
    -------
    fx_sync, fy_sync : jax.Array
        Fluxes with averaged boundary values.
    """
    # FV3_3D iter-1049: MPI dispatch.
    if _halo_backend == "mpi" and _mpi_topology is not None:
        return _synchronize_cgrid_fluxes_mpi(fx, fy, n, _mpi_topology)

    # Pre-compute all boundary averages from the ORIGINAL (unsynchronized)
    # fluxes so that we read before writing.
    avgs = {}
    for face in range(6):
        for edge in (WEST, EAST, SOUTH, NORTH):
            nbr_face, nbr_edge, rev = CONNECTIVITY[face][edge]
            local_bdy = _extract_cgrid_boundary(fx, fy, face, edge, n)
            nbr_bdy = _extract_cgrid_boundary(fx, fy, nbr_face, nbr_edge, n)
            if rev:
                nbr_bdy = nbr_bdy[::-1]
            if (face, edge) in _FLUX_SIGN_FLIP_EDGES:
                nbr_bdy = -nbr_bdy
            avgs[(face, edge)] = 0.5 * (local_bdy + nbr_bdy)

    # Write all averaged values back.
    for face in range(6):
        for edge in (WEST, EAST, SOUTH, NORTH):
            avg = avgs[(face, edge)]
            if edge == WEST:
                fx = fx.at[face, 0, :].set(avg)
            elif edge == EAST:
                fx = fx.at[face, n, :].set(avg)
            elif edge == SOUTH:
                fy = fy.at[face, :, 0].set(avg)
            else:  # NORTH
                fy = fy.at[face, :, n].set(avg)

    return fx, fy


def _synchronize_cgrid_fluxes_mpi(fx, fy, n, topology):
    """MPI-aware variant of :func:`synchronize_cgrid_fluxes`.

    For each owned face's edge, locate the neighbour face and its
    cross-face edge.  If both endpoints live on this rank (local
    edge), read directly from ``fx`` / ``fy`` — same as the
    single-device path.  Otherwise issue an ``mpi4jax.sendrecv``
    that swaps the local boundary flux strip with the neighbour
    rank's strip from the corresponding edge.

    The function leaves non-owned face boundary values UNCHANGED in
    the returned arrays — only owned faces get the averaged result.
    Callers comparing across local/MPI must compare owned faces only
    (the MPI-replicated-mode contract).
    """
    from legoesm.parallel.halo_exchange import get_sendrecv_vjp
    from collections import defaultdict
    try:
        import mpi4jax
        from mpi4py import MPI as _MPI
    except ImportError as exc:
        raise ImportError(
            "MPI synchronize_cgrid_fluxes requires mpi4jax + mpi4py."
        ) from exc
    sendrecv = get_sendrecv_vjp(mpi4jax)
    comm = _MPI.COMM_WORLD
    rank = topology.rank

    # Classify each (owned face, edge) as local or remote.
    local_edges = []
    remote_edges = []
    for face in topology.local_face_ids:
        for edge in (WEST, EAST, SOUTH, NORTH):
            nbr_face, nbr_edge, rev = CONNECTIVITY[face][edge]
            nbr_rank = topology.neighbor_ranks[(face, edge)]
            entry = (face, edge, nbr_face, nbr_edge, rev, nbr_rank)
            if nbr_rank == rank:
                local_edges.append(entry)
            else:
                remote_edges.append(entry)

    def _bdy_strip(face, edge):
        return _extract_cgrid_boundary(fx, fy, face, edge, n)

    # FV3_3D iter-1051: batched-per-neighbour sendrecv.  The iter-1049
    # per-edge sendrecv pattern deadlocked at runtime because each
    # ``mpi4jax.sendrecv`` blocks on its own ``recv`` until the peer
    # issues a matching call.  With multiple sendrecvs per neighbour
    # rank, both ranks block on their first call's recv waiting for
    # the other's later send → deadlock.  The pad_halo_mpi pattern
    # packs ALL strips for a given neighbour into ONE contiguous
    # send/recv buffer (sorted canonically: send by ``(nbr_face,
    # nbr_edge)``, recv by ``(face, edge)``).  One sendrecv per
    # peer rank — at most 3 peers in face-only mode, so at most 3
    # MPI calls per sync.

    # Pre-extract every owned-face boundary strip.
    local_strips = {}
    for face in topology.local_face_ids:
        for edge in (WEST, EAST, SOUTH, NORTH):
            local_strips[(face, edge)] = _bdy_strip(face, edge)

    # Holding place for neighbour strips fetched via MPI (remote
    # edges).  Local-edge strips are read directly later.
    nbr_strips_remote = {}

    # Group remote edges by neighbour rank.
    by_nbr_rank: dict[int, list] = defaultdict(list)
    for entry in remote_edges:
        by_nbr_rank[entry[5]].append(entry)

    # Per-neighbour single sendrecv with canonically-ordered batched
    # strips.  Send ordered by ``(nbr_face, nbr_edge)`` = the peer's
    # local-edge identity, so the peer's recv-side canonical order
    # (sorted by ``(face, edge)``) matches.  Tag is the peer rank
    # itself (matches ``pad_halo_mpi_face_only`` pattern).
    #
    # All boundary strips share the same length ``n`` along axis 0
    # (cell axis) and the same trailing shape (e.g., ``(nlev,)`` for
    # 4D).  Pack by stacking on axis 0; unpack by slicing axis 0
    # in chunks of ``n``.
    # FV3_3D iter-1053: iterate peers in ascending rank order so all
    # ranks issue sendrecv calls in the same global peer-sequence
    # (mirror of the iter-1053 fix applied to ``_sync_dgrid_boundary_mpi``).
    for nbr_rank in sorted(by_nbr_rank.keys()):
        entries = by_nbr_rank[nbr_rank]
        send_order = sorted(entries, key=lambda e: (e[2], e[3]))
        recv_order = sorted(entries, key=lambda e: (e[0], e[1]))

        send_parts = [local_strips[(f, e)] for f, e, *_ in send_order]
        send_buf = jnp.concatenate(send_parts, axis=0)

        send_tag = rank
        recv_tag = nbr_rank
        recv_buf = sendrecv(
            send_buf, jnp.zeros_like(send_buf),
            nbr_rank, nbr_rank,
            send_tag, recv_tag, comm,
        )

        # Unpack recv_buf in canonical recv order.  Each strip is
        # ``n`` cells along axis 0; trailing axes match the input.
        offset = 0
        for face, edge, nbr_face, nbr_edge, rev, _ in recv_order:
            chunk = recv_buf[offset:offset + n]
            offset += n
            nbr_strips_remote[(face, edge)] = chunk

    # FV3_3D iter-1051: pre-extract local-edge neighbour strips
    # BEFORE the write loop.  Reading on-demand inside the loop
    # picks up already-averaged values (write-before-read) on edges
    # whose neighbour face was processed earlier in the iteration.
    nbr_strips_local = {}
    for face, edge, nbr_face, nbr_edge, rev, _ in local_edges:
        nbr_strips_local[(face, edge)] = _bdy_strip(nbr_face, nbr_edge)

    # Compute averages and write back to owned faces.
    all_entries = local_edges + remote_edges
    for face, edge, nbr_face, nbr_edge, rev, nbr_rank in all_entries:
        local_bdy = local_strips[(face, edge)]
        if nbr_rank == rank:
            nbr_bdy = nbr_strips_local[(face, edge)]
        else:
            nbr_bdy = nbr_strips_remote[(face, edge)]
        if rev:
            nbr_bdy = nbr_bdy[::-1]
        if (face, edge) in _FLUX_SIGN_FLIP_EDGES:
            nbr_bdy = -nbr_bdy
        avg = 0.5 * (local_bdy + nbr_bdy)
        if edge == WEST:
            fx = fx.at[face, 0, :].set(avg)
        elif edge == EAST:
            fx = fx.at[face, n, :].set(avg)
        elif edge == SOUTH:
            fy = fy.at[face, :, 0].set(avg)
        else:  # NORTH
            fy = fy.at[face, :, n].set(avg)

    return fx, fy


def _extract_cgrid_boundary(fx, fy, face, edge, n):
    """Extract boundary flux from the appropriate array and position.

    WEST/EAST boundaries extract from fx (x-direction fluxes).
    SOUTH/NORTH boundaries extract from fy (y-direction fluxes).
    Returns shape (n,).
    """
    if edge == WEST:
        return fx[face, 0, :]
    elif edge == EAST:
        return fx[face, n, :]
    elif edge == SOUTH:
        return fy[face, :, 0]
    else:  # NORTH
        return fy[face, :, n]


def synchronize_corner_scalar(field, n):
    """Average a scalar corner field at shared face boundaries and cube vertices.

    For a field at D-grid corner positions (6, n+1, n+1):
    1. Average boundary edges between adjacent face pairs.
    2. Average cube-vertex corners where 3 faces meet (8 vertices).

    Parameters
    ----------
    field : jax.Array, shape (6, n+1, n+1)
    n : int — number of cells per face edge

    Returns
    -------
    jax.Array, shape (6, n+1, n+1) — with averaged boundary values
    """
    def _bdy(f, edge):
        if edge == WEST:
            return field[f, 0, :]      # (n+1,)
        elif edge == EAST:
            return field[f, n, :]
        elif edge == SOUTH:
            return field[f, :, 0]
        else:
            return field[f, :, n]

    # --- Save original vertex values before any modification ---
    # Each cube vertex connects 3 faces. We need the ORIGINAL (unaveraged)
    # values to compute the true 3-face mean, not edge-averaged intermediates.
    orig_corners = {}
    for face in range(6):
        for ci in (0, n):
            for cj in (0, n):
                orig_corners[(face, ci, cj)] = field[face, ci, cj]

    # --- Pass 1: edge-pairwise averaging (read all before write) ---
    avgs = {}
    for face in range(6):
        for edge in (WEST, EAST, SOUTH, NORTH):
            nbr_face, nbr_edge, rev = CONNECTIVITY[face][edge]
            local = _bdy(face, edge)
            nbr = _bdy(nbr_face, nbr_edge)
            if rev:
                nbr = nbr[::-1]
            avgs[(face, edge)] = 0.5 * (local + nbr)

    for face in range(6):
        for edge in (WEST, EAST, SOUTH, NORTH):
            avg = avgs[(face, edge)]
            if edge == WEST:
                field = field.at[face, 0, :].set(avg)
            elif edge == EAST:
                field = field.at[face, n, :].set(avg)
            elif edge == SOUTH:
                field = field.at[face, :, 0].set(avg)
            else:
                field = field.at[face, :, n].set(avg)

    # --- Pass 2: cube-vertex averaging (3 faces share each vertex) ---
    # Each face has 4 corners at (i,j) ∈ {0,n}×{0,n}. Each corner is a
    # cube vertex shared by exactly 3 faces. After edge averaging, the 3
    # face values may be inconsistent because each was averaged from a
    # different edge pair. Replace with the 3-face mean.

    def _neighbor_corner(face_a, ci, cj, edge):
        """Find the neighbor face corner reached via the given edge."""
        nbr_f, nbr_e, rev = CONNECTIVITY[face_a][edge]
        # Position along face_a's edge
        if edge in (WEST, EAST):
            pos = cj  # position along vertical edge
        else:
            pos = ci  # position along horizontal edge
        if rev:
            pos = n - pos
        # Map to neighbor face corner
        if nbr_e == WEST:
            return nbr_f, 0, pos
        elif nbr_e == EAST:
            return nbr_f, n, pos
        elif nbr_e == SOUTH:
            return nbr_f, pos, 0
        else:  # NORTH
            return nbr_f, pos, n

    _edge_for_i = {0: WEST, n: EAST}
    _edge_for_j = {0: SOUTH, n: NORTH}

    visited = set()
    for face_a in range(6):
        for ci in (0, n):
            for cj in (0, n):
                fb, bi, bj = _neighbor_corner(face_a, ci, cj, _edge_for_i[ci])
                fc, ci_c, cj_c = _neighbor_corner(face_a, ci, cj, _edge_for_j[cj])

                key = tuple(sorted([(face_a, ci, cj), (fb, bi, bj), (fc, ci_c, cj_c)]))
                if key in visited:
                    continue
                visited.add(key)

                # Use ORIGINAL (pre-edge-averaged) values for unbiased 3-face mean
                avg3 = (orig_corners[(face_a, ci, cj)]
                        + orig_corners[(fb, bi, bj)]
                        + orig_corners[(fc, ci_c, cj_c)]) / 3.0
                field = field.at[face_a, ci, cj].set(avg3)
                field = field.at[fb, bi, bj].set(avg3)
                field = field.at[fc, ci_c, cj_c].set(avg3)

    return field


def synchronize_bgrid_ne_corner_geo(u, v, z11, z12, z21, z22, n):
    """BGRID_NE vector corner sync via the (exact) geographic frame.

    FV3 reference: `dyn_core.F90:968-1009` performs the BGRID_NE corner
    sync as `mpp_get_boundary(gridtype=BGRID_NE)` (exact discrete panel-
    to-panel rotation into the buffers) followed by a plain 0.5 average
    in the grid-local frame.  Because a physical vector has the same
    (east, north) components on every face that shares a corner, that
    discrete-rotation-then-average is mathematically identical to:
    convert each face's local components to the INVARIANT geographic
    frame, average, convert back — PROVIDED the local→geographic
    conversion is exact.

    iter3 FIX (cube-faithfulness): the previous conversion used the
    ORTHOGONAL rotation `u_east = cos·u − sin·v`, which silently assumes
    the two grid tangents are perpendicular.  At the 8 cube vertices the
    inter-tangent angle is non-orthogonal by an O(1) amount that does NOT
    vanish with resolution (|cos θ| ≈ 0.47), so the orthogonal form
    corrupted a constant geographic wind by ≈0.35 there (resolution-
    independent) — the residual W5 vertex mode.  The synced (u, v) are
    FV3's B-grid Courant components ub = V·x̂′, vb = V·ŷ′ with x̂′ ⊥ e2,
    ŷ′ ⊥ e1 (from `vb = dt5·(vc − uc·cosa)·rsina`); equivalently
    V = (u·e1 + v·e2)/sinθ.  The exact non-orthogonal conversion uses the
    corner c2l z-matrix M = [[z11, z21], [z12, z22]] (rows = (ec1,ec2) ·
    (east, north)), det M = sinθ > 0::

        u_east  = (z11·u + z21·v) / detM
        u_north = (z12·u + z22·v) / detM
        # ... average geographic scalars across faces ...
        u_sync  =  z22·u_east − z21·u_north      # adjugate (det folds out)
        v_sync  = −z12·u_east + z11·u_north

    Round-trips to the identity exactly (verified) and reduces to the old
    orthogonal form on perpendicular axes (z21→−sin, z22→cos, detM→1).

    Parameters
    ----------
    u, v : jax.Array, shape (6, n+1, n+1)
        Face-local corner-stagger B-grid components (ubb, vbbtemp).
    z11, z12 : jax.Array, shape (6, n+1, n+1)
        ec1·east, ec1·north = cdgrid.cos_angle_corner / sin_angle_corner.
    z21, z22 : jax.Array, shape (6, n+1, n+1)
        ec2·east, ec2·north = cdgrid.z21_corner / z22_corner.
    n : int

    Returns
    -------
    u_sync, v_sync : jax.Array, shape (6, n+1, n+1)
    """
    # Pointwise local→geo, cross-face scalar sync, pointwise geo→local.  The
    # two pointwise conversions are factored (bgrid_ne_corner_to_geo /
    # _from_geo) so a sub-face tiled corner-sync stage can run them PER TILE
    # (approach-C) around the cross-tile scalar sync (task #3 cube tiling
    # U3e); this global path is bit-identical (pure extraction).
    u_east, u_north = bgrid_ne_corner_to_geo(u, v, z11, z12, z21, z22)
    u_east = synchronize_corner_scalar(u_east, n)
    u_north = synchronize_corner_scalar(u_north, n)
    return bgrid_ne_corner_from_geo(u_east, u_north, z11, z12, z21, z22)


def bgrid_ne_corner_to_geo(u, v, z11, z12, z21, z22):
    """Pointwise exact non-orthogonal local→geographic corner conversion
    (the forward half of :func:`synchronize_bgrid_ne_corner_geo`, factored
    for sub-face tiling — task #3 U3e).  ``u_east = (z11·u + z21·v)/detM``,
    ``u_north = (z12·u + z22·v)/detM`` with ``detM = z11·z22 − z21·z12``.
    Shape-generic (global ``(6,n+1,n+1)`` or a per-tile corner block); each
    corner reads ONLY its own ``(u,v,z*)`` → tiles with no halo."""
    _EPS = float(jnp.finfo(jnp.float32).eps)
    det = z11 * z22 - z21 * z12  # = sin(inter-axis angle) > 0
    inv = 1.0 / jnp.where(jnp.abs(det) > _EPS, det, 1.0)
    return (z11 * u + z21 * v) * inv, (z12 * u + z22 * v) * inv


def bgrid_ne_corner_from_geo(u_east, u_north, z11, z12, z21, z22):
    """Pointwise exact geo→local corner conversion (the inverse half;
    adjugate of M, the det cancels the forward 1/det).  ``u_sync =
    z22·u_east − z21·u_north``, ``v_sync = −z12·u_east + z11·u_north``.
    Shape-generic; no halo (per-corner)."""
    u_sync = z22 * u_east - z21 * u_north
    v_sync = -z12 * u_east + z11 * u_north
    return u_sync, v_sync


@contextlib.contextmanager
def _swap_module_attr(target: str, new_value):
    """Temporarily rebind the attribute named by dotted ``target``.

    Behaviour-preserving stand-in for ``unittest.mock.patch(target,
    new_value)`` so this shippable library carries no test-framework
    dependency at runtime (slopbuster Pass 11).  Like ``mock.patch``, it
    resolves ``target`` by importing the longest importable module prefix
    and then walking any remaining (class/attribute) components with
    ``getattr`` to reach the owner of the final attribute, which is
    swapped on enter and restored on exit.  Raises the same exceptions
    the callers already guard: ``ModuleNotFoundError`` when no module
    prefix imports and ``AttributeError`` when an intermediate or final
    attribute is absent.
    """
    parent_path, _, attr = target.rpartition(".")
    parts = parent_path.split(".")
    # Import the longest importable prefix (mock-style deepest module).
    module = None
    split = len(parts)
    while split > 0:
        try:
            module = importlib.import_module(".".join(parts[:split]))
            break
        except ModuleNotFoundError:
            split -= 1
    if module is None:
        raise ModuleNotFoundError(parent_path)
    parent = module
    for name in parts[split:]:  # walk class/attribute components, if any
        parent = getattr(parent, name)  # AttributeError if absent
    old_value = getattr(parent, attr)  # AttributeError if attr is absent
    setattr(parent, attr, new_value)
    try:
        yield new_value
    finally:
        setattr(parent, attr, old_value)


def monotone_halo_clip_context(slack: float = 0.5):
    """FV3_3D iter 505: context manager that monkey-patches 15
    known halo import aliases in NH/PE/SW dycore + operator
    modules to use ``monotone_clip=True`` with the given ``slack``.

    Per iter-504, enabling this around a dycore step reduces
    the duogrid-induced cube-edge θ′ variance ratio by ~65%
    when combined with iter-466's ``make_legoesm_nh_min_edge_
    config`` factory.  At smooth atmospheric ICs, the iter-553
    ``make_fv3_component_fidelity_nh_config`` factory alone gives 89.5%
    reduction without needing this context.  Use this context
    when iters>2 boost is enabled (iter-562) for additional
    edge suppression.

    See also:
    - ``make_clipped_step(model, state, dt, slack)`` (iter-526):
      JIT-safe variant that bakes the clip into a cached compiled
      function.  Required for ``jax.jit`` users.
    - ``make_clipped_scan_step(model, state, dt, n_steps, slack)``
      (iter-544): multi-step ``jax.lax.scan`` variant for low
      Python-overhead long runs.

    Usage::

        from legoesm.grids.halo import monotone_halo_clip_context

        with monotone_halo_clip_context(slack=0.5):
            new_state = model.step(state, dt)

    Parameters
    ----------
    slack : float, default 0.5
        ``monotone_clip_slack`` value (see ``fill_corner_
        region`` docs).  0.0 = strict clip; 0.5 = optimal per
        iter-504; ≥1.0 = over-relaxed.

    Returns
    -------
    contextlib.ExitStack
        Context manager.  Patches are removed on exit.

    Notes
    -----
    Implementation: ``_swap_module_attr`` rebinds these targets:
    * scalar halo: 3 sites (compressible_euler_cdgrid,
      operators_3d, operators_cdgrid).
    * vector halo: 2 sites (operators_cdgrid, operators_3d).

    May not catch every halo call site in the dycore (e.g.,
    SPMD ``packed_pad_halo_4d`` is not patched); the 6 sites
    cover the dominant single-rank paths.
    """
    import functools

    scalar_targets = [
        "legoesm.atmosphere.dynamics.gcm.compressible_euler_cdgrid."
        "_pad_halo_4d_module",
        "legoesm.core.operators_3d.pad_halo_4d",
        "legoesm.core.operators_cdgrid.pad_halo_4d",
        # FV3_3D iter 527: PE-side import aliases.
        "legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid."
        "_pad_halo_4d",
        "legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid."
        "_pad_halo_4d_module",
    ]
    vector_targets = [
        "legoesm.core.operators_cdgrid.pad_halo_vector_4d",
        "legoesm.core.operators_3d.pad_halo_vector_4d",
        # FV3_3D iter 527: PE-side import alias.
        "legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid."
        "pad_halo_vector_4d",
    ]
    pad_halo_3d_targets = [
        # FV3_3D iter 513: pad_halo (3D) is used by pad_halo_pair_h2
        # in fv_tp_2d transport, an unpatched leak in iter-505/512.
        "legoesm.core.operators_cdgrid.pad_halo",
        "legoesm.core.fv_tp_2d.pad_halo",
        "legoesm.core.fv3_sw_core.pad_halo",
    ]
    pair_h2_targets = [
        "legoesm.core.fv_tp_2d.pad_halo_pair_h2",
    ]
    pad_halo_vector_3d_targets = [
        # FV3_3D iter 514: pad_halo_vector (3D) is used in fv3_sw_core
        # at 3 sites (line 602, 1194, 1430), called from NH dycore
        # via C-D coupling. NH residual leaks through this path.
        "legoesm.core.fv3_sw_core.pad_halo_vector",
        # FV3_3D iter 527: PE-side import alias.
        "legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid."
        "pad_halo_vector",
    ]

    clipped_scalar = functools.partial(
        pad_halo_4d,
        monotone_clip=True,
        monotone_clip_slack=slack,
    )
    clipped_vector = functools.partial(
        pad_halo_vector_4d,
        monotone_clip=True,
        monotone_clip_slack=slack,
    )
    clipped_pad_halo_3d = functools.partial(
        pad_halo,
        monotone_clip=True,
        monotone_clip_slack=slack,
    )
    clipped_pair_h2 = functools.partial(
        pad_halo_pair_h2,
        monotone_clip=True,
        monotone_clip_slack=slack,
    )
    clipped_pad_halo_vector_3d = functools.partial(
        pad_halo_vector,
        monotone_clip=True,
        monotone_clip_slack=slack,
    )

    stack = contextlib.ExitStack()
    for tgt in scalar_targets:
        try:
            stack.enter_context(_swap_module_attr(tgt, clipped_scalar))
        except (AttributeError, ModuleNotFoundError):
            pass
    for tgt in vector_targets:
        try:
            stack.enter_context(_swap_module_attr(tgt, clipped_vector))
        except (AttributeError, ModuleNotFoundError):
            pass
    for tgt in pad_halo_3d_targets:
        try:
            stack.enter_context(_swap_module_attr(tgt, clipped_pad_halo_3d))
        except (AttributeError, ModuleNotFoundError):
            pass
    for tgt in pair_h2_targets:
        try:
            stack.enter_context(_swap_module_attr(tgt, clipped_pair_h2))
        except (AttributeError, ModuleNotFoundError):
            pass
    for tgt in pad_halo_vector_3d_targets:
        try:
            stack.enter_context(_swap_module_attr(tgt, clipped_pad_halo_vector_3d))
        except (AttributeError, ModuleNotFoundError):
            pass
    return stack


def make_clipped_step(model, state_template, dt: float, slack: float = 0.5):
    """FV3_3D iter 526: return a JIT-compiled ``model.step`` with the
    iter-505 clip baked into the compiled graph.

    Solves the iter-525 limitation that ``jax.jit(model.step)`` only
    picks up the clip if traced inside the context.  This helper
    enters the context, JIT-compiles ``model.step``, FORCES TRACING by
    calling the JIT once with ``state_template`` and ``dt``, then exits
    the context and returns the cached-compiled function.  Subsequent
    calls reuse the cached compilation (with clip permanently baked
    into the lowered HLO).

    Parameters
    ----------
    model : object
        Anything with a ``.step(state, dt)`` method (NH or PE dycore).
    state_template : pytree
        Example state used to force trace (shapes/dtypes must match
        all later calls).
    dt : float
        Step size used to force trace.
    slack : float
        ``monotone_clip_slack``.  Default 0.5 (iter-504 optimum).

    Returns
    -------
    callable
        ``step(state, dt) -> new_state``, JIT-compiled with clip baked in.

    Usage
    -----
    ::

        from legoesm.grids.halo import make_clipped_step

        model = CDGridCompressibleEulerModel(grid, hc, tm, cfg)
        step = make_clipped_step(model, state, dt=10.0, slack=0.5)
        for _ in range(100):
            state = step(state, dt=10.0)
    """
    with monotone_halo_clip_context(slack=slack):
        step_jit = jax.jit(model.step)
        # Force trace + compilation NOW, while context is active.
        _ = step_jit(state_template, dt)
    return step_jit


def make_clipped_scan_step(
    model, state_template, dt: float, n_steps: int, slack: float = 0.5,
):
    """FV3_3D iter 544: ``jax.lax.scan``-based multi-step with clip baked in.

    Faster than a Python ``for`` loop over ``make_clipped_step``
    because the n-step loop is JIT-compiled as a single graph
    (no Python overhead per step).

    Parameters
    ----------
    model : object
        Has ``.step(state, dt)`` method.
    state_template : pytree
        Example state used to force trace.
    dt : float
        Per-step size.  Static (baked into the compiled scan).
    n_steps : int
        Number of steps in the scan loop.  Static.
    slack : float
        ``monotone_clip_slack``.  Default 0.5.

    Returns
    -------
    callable
        ``scan_step(state) -> final_state`` after ``n_steps`` steps.

    Usage
    -----
    ::

        scan_step = make_clipped_scan_step(model, state, dt=10.0,
                                           n_steps=100, slack=0.5)
        final_state = scan_step(state)

    Differentiable via ``jax.grad`` end-to-end.
    """
    with monotone_halo_clip_context(slack=slack):
        def _body(s, _):
            return model.step(s, dt), None

        @jax.jit
        def scan_step(s):
            final, _ = jax.lax.scan(_body, s, jnp.arange(n_steps))
            return final

        # Force trace + compilation NOW, while context is active.
        _ = scan_step(state_template)
    return scan_step


def compute_edge_artifact_metric(field_data):
    """FV3_3D iter 581: edge-artifact diagnostic helper.

    Computes the standard edge-vs-interior std and absolute
    edge_std for a 4D field of shape ``(face, x, y, level)``.
    Used throughout iter 466-580 to characterize cube-edge
    artifacts.

    Returns
    -------
    dict with keys ``edge_std``, ``interior_std``, ``ratio``.

    Usage::

        from legoesm.grids.halo import compute_edge_artifact_metric

        metrics = compute_edge_artifact_metric(state.theta_prime.data)
        print(f"edge_std = {metrics['edge_std']:.3e}")
        print(f"ratio = {metrics['ratio']:.3f}x")
    """
    import numpy as np
    arr = np.asarray(field_data)
    n_face, n_x, n_y, n_lev = arr.shape
    edge_mask = np.zeros((n_x, n_y), dtype=bool)
    edge_mask[0, :] = True
    edge_mask[-1, :] = True
    edge_mask[:, 0] = True
    edge_mask[:, -1] = True
    edge_mask_b = np.broadcast_to(
        edge_mask[None, :, :, None], arr.shape,
    )
    interior_mask = ~edge_mask_b
    e = float(arr[edge_mask_b].std())
    i = float(arr[interior_mask].std())
    return {
        "edge_std": e,
        "interior_std": i,
        "ratio": e / max(i, 1e-30),
    }


def compute_cross_face_continuity(field_data, interp_offsets=None):
    """TRUE cross-face seam-continuity diagnostic (iter ~58).

    The reliable edge-artifact metric: pads ``field_data`` with the
    model's *real* inter-face halo (:func:`pad_halo_4d`) and, per face,
    compares the cross-seam first difference (edge interior cell minus
    its physical neighbor on the adjacent face) to the same-face interior
    first difference (the field's natural gradient)::

        ratio = RMS(cross-seam Δ) / RMS(same-face interior Δ)

    Interpretation: ``ratio ≈ 1`` ⇒ the field is as smooth across the
    panel seam as it is in the interior (CONTINUOUS).  A genuine seam
    discontinuity registers 5–50×.

    This SUPERSEDES :func:`compute_edge_artifact_metric` for edge-artifact
    claims.  That helper is a *same-face* edge-vs-interior std ratio: it
    amplifies high-frequency edge curvature and is unreliable in BOTH
    directions (it both over- and under-states — verified iter ~57-58:
    same-face 2nd-diff gave 5.95× on W5 v where the true cross-face
    continuity is 1.30×).  Use this function, applied to a *geographic*
    (seam-continuous) field component, for the real continuity check.

    Parameters
    ----------
    field_data : array, shape ``(6, n, n)`` or ``(6, n, n, nlev)``
        A scalar or geographic-component field on the cubed sphere.  For
        a vector, pass each *geographic* component (north/east) — those
        are continuous across seams; do NOT pass face-local components.
    interp_offsets : array, optional
        Halo interpolation offsets (``grid.halo_interp_offsets``) for the
        corrected cross-face interpolation; ``None`` uses the plain halo.

    Returns
    -------
    dict with keys ``per_face_ratio`` (list, len 6), ``max_ratio``,
    ``mean_ratio``.
    """
    import numpy as np
    arr = np.asarray(field_data)
    if arr.ndim == 3:
        arr = arr[..., None]
    fp = np.asarray(
        pad_halo_4d(jnp.asarray(arr), halo=1, interp_offsets=interp_offsets)
    )
    n_face = fp.shape[0]
    per_face = []
    for f in range(n_face):
        a = fp[f]  # (n+2, n+2, nlev); interior = a[1:-1, 1:-1]
        # Interior first differences — STRICTLY between interior cells, so the
        # halo (cross-seam) rows/cols never enter the denominator.  (Codex
        # iter61: a[2:,...]/a[...,2:] included the N/E halo row a[n+1]-a[n],
        # i.e. a seam jump, self-normalizing the ratio and suppressing
        # detection of N/E-edge artifacts.)
        gx = (a[2:-1, 1:-1] - a[1:-2, 1:-1]).ravel()   # i-diff, interior only
        gy = (a[1:-1, 2:-1] - a[1:-1, 1:-2]).ravel()   # j-diff, interior only
        gi = float(np.sqrt(np.mean(np.concatenate([gx, gy]) ** 2)))
        seam = np.concatenate([
            (a[1, 1:-1] - a[0, 1:-1]).ravel(),
            (a[-1, 1:-1] - a[-2, 1:-1]).ravel(),
            (a[1:-1, 1] - a[1:-1, 0]).ravel(),
            (a[1:-1, -1] - a[1:-1, -2]).ravel(),
        ])
        gs = float(np.sqrt(np.mean(seam ** 2)))
        per_face.append(gs / max(gi, 1e-30))
    per_face = np.asarray(per_face)
    return {
        "per_face_ratio": per_face.tolist(),
        "max_ratio": float(per_face.max()),
        "mean_ratio": float(per_face.mean()),
    }
