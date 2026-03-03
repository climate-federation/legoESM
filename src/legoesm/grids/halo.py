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
# Halo backend dispatch (for MPI)
# ==============================================================================

# Module-level state: "local" for single-node (default), "mpi" for distributed.
_halo_backend: str = "local"
_mpi_topology = None  # CommTopology or None


def set_halo_backend(backend: str, topology=None) -> None:
    """Configure the halo exchange backend.

    Parameters
    ----------
    backend : ``"local"`` | ``"mpi"``
        ``"local"`` uses the default JAX-native implementation
        (works for single-device and single-node multi-device).
        ``"mpi"`` uses ``mpi4jax`` for distributed communication.
    topology : CommTopology, optional
        Required when ``backend="mpi"``.
    """
    global _halo_backend, _mpi_topology
    if backend not in ("local", "mpi"):
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


def pad_halo(data: jax.Array, halo: int = 1) -> jax.Array:
    """Pad a scalar field with inter-face halo data.

    Creates an array of shape (6, n+2*halo, n+2*halo) where the interior
    contains the original data and the halo cells contain data from
    neighboring faces, correctly handling axis swaps and reversals.

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

    Returns
    -------
    padded : jax.Array, shape (6, n+2*halo, n+2*halo)
    """
    if halo != 1:
        raise NotImplementedError("Only halo=1 is supported")

    # MPI dispatch.
    if _halo_backend == "mpi":
        from legoesm.parallel.halo_exchange import pad_halo_mpi
        return pad_halo_mpi(data, _mpi_topology)

    return _pad_halo_local(data)


def _pad_halo_local(data: jax.Array) -> jax.Array:
    """Local (single-node) scalar halo exchange implementation."""
    n = data.shape[1]
    padded = jnp.zeros((6, n + 2, n + 2), dtype=data.dtype)

    # Place interior data
    padded = padded.at[:, 1:-1, 1:-1].set(data)

    # Fill halo strips from neighbors
    for face in range(6):
        for edge in [WEST, EAST, SOUTH, NORTH]:
            nbr_face, nbr_edge, is_reversed = CONNECTIVITY[face][edge]

            # Extract the neighbor's edge strip
            strip = _extract_edge_strip(data, nbr_face, nbr_edge)

            # Reverse if needed
            if is_reversed:
                strip = strip[::-1]

            # Place in halo position
            if edge == WEST:
                padded = padded.at[face, 0, 1:-1].set(strip)
            elif edge == EAST:
                padded = padded.at[face, -1, 1:-1].set(strip)
            elif edge == SOUTH:
                padded = padded.at[face, 1:-1, 0].set(strip)
            elif edge == NORTH:
                padded = padded.at[face, 1:-1, -1].set(strip)

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
    cos_angle_padded : jax.Array, shape (6, n+2, n+2)
        Cosine of padded grid angle (precomputed).
    sin_angle_padded : jax.Array, shape (6, n+2, n+2)
        Sine of padded grid angle (precomputed).

    Returns
    -------
    u_padded : jax.Array, shape (6, n+2, n+2)
        Padded grid-aligned x-velocity.
    v_padded : jax.Array, shape (6, n+2, n+2)
        Padded grid-aligned y-velocity.
    """
    # Step 1: Convert to geographic (east, north)
    u_east = cos_angle * u_data - sin_angle * v_data
    v_north = sin_angle * u_data + cos_angle * v_data

    # Step 2: Pad geographic components as scalars (auto-dispatches to MPI)
    u_east_padded = pad_halo(u_east)
    v_north_padded = pad_halo(v_north)

    # Step 3: Convert back to grid-aligned using padded angle
    u_padded = cos_angle_padded * u_east_padded + sin_angle_padded * v_north_padded
    v_padded = -sin_angle_padded * u_east_padded + cos_angle_padded * v_north_padded

    return u_padded, v_padded


# ==============================================================================
# Padded grid angle computation
# ==============================================================================

def compute_padded_angle(n: int) -> jax.Array:
    """Compute grid angle on extended (n+2) x (n+2) grid per face.

    Extends the gnomonic coordinate range by one cell width on each side,
    then computes the grid angle (rotation of grid x-axis relative to
    geographic east) at the extended cell centers.

    This provides exact angle values at halo cell locations for the
    vector halo exchange.

    Parameters
    ----------
    n : int
        Number of cells per face edge (interior grid).

    Returns
    -------
    angle_padded : jax.Array, shape (6, n+2, n+2)
    """
    # Gnomonic coordinates on the extended grid
    dalpha = jnp.pi / (2 * n)
    alpha = jnp.linspace(
        -jnp.pi / 4 - dalpha,  # one cell before
        jnp.pi / 4 + dalpha,   # one cell after
        n + 2,
        endpoint=False,
    )
    # Shift to cell centers (same offset as main grid but for n+2 cells)
    alpha = alpha + dalpha / 2.0

    alpha_x, alpha_y = jnp.meshgrid(alpha, alpha, indexing='ij')

    all_angle = []
    for face in range(6):
        lon, lat = _face_gnomonic_to_lonlat(face, alpha_x, alpha_y)

        # Grid angle: direction of grid x-axis relative to east
        # Use finite differences of the (n+2) grid
        dlon_dx = jnp.zeros_like(lon)
        dlat_dx = jnp.zeros_like(lat)

        # Interior: centered differences
        dlon_dx = dlon_dx.at[1:-1, :].set(lon[2:, :] - lon[:-2, :])
        dlat_dx = dlat_dx.at[1:-1, :].set(lat[2:, :] - lat[:-2, :])

        # Boundaries: one-sided differences
        dlon_dx = dlon_dx.at[0, :].set(lon[1, :] - lon[0, :])
        dlat_dx = dlat_dx.at[0, :].set(lat[1, :] - lat[0, :])
        dlon_dx = dlon_dx.at[-1, :].set(lon[-1, :] - lon[-2, :])
        dlat_dx = dlat_dx.at[-1, :].set(lat[-1, :] - lat[-2, :])

        # Handle longitude wrapping
        dlon_dx = jnp.where(dlon_dx > jnp.pi, dlon_dx - 2 * jnp.pi, dlon_dx)
        dlon_dx = jnp.where(dlon_dx < -jnp.pi, dlon_dx + 2 * jnp.pi, dlon_dx)

        cos_lat = jnp.cos(lat)
        face_angle = jnp.arctan2(dlat_dx, dlon_dx * cos_lat)
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
    n: int, radius: float,
) -> tuple[jax.Array, jax.Array]:
    """Compute half-edge-length metrics on the extended (n+2) × (n+2) grid.

    Each face's gnomonic coordinate range is extended by one cell on each
    side.  Great-circle distances between cell centres two positions apart
    are computed (the 2-cell span, matching ``_compute_grid_spacing`` in
    ``cubed_sphere.py``), and halved to give the single-cell edge lengths
    ``hx = dx/2`` and ``hy = dy/2``.

    A further-extended (n+4) grid is used internally so that centred
    differences are available at every point of the (n+2) output grid.

    Parameters
    ----------
    n : int
        Number of interior cells per face edge.
    radius : float
        Sphere radius [m].

    Returns
    -------
    hx_ext : jax.Array, shape (6, n+2, n+2)
        Half dx on the extended grid (single-cell x-edge length).
    hy_ext : jax.Array, shape (6, n+2, n+2)
        Half dy on the extended grid (single-cell y-edge length).
    """
    dalpha = jnp.pi / (2 * n)

    # Build an (n+4) gnomonic grid: 2 cells beyond the interior on each
    # side.  Centred differences of this grid yield (n+2) dx/dy values.
    alpha_big = jnp.linspace(
        -jnp.pi / 4 - 2 * dalpha,
        jnp.pi / 4 + 2 * dalpha,
        n + 4,
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

        # 2-cell chord in x: positions [i+1,j] – [i-1,j] on (n+4) grid
        # → (n+2, n+2) result after slicing off the one-cell border
        dx_chord = jnp.sqrt(
            (x[2:, 1:-1] - x[:-2, 1:-1]) ** 2
            + (y[2:, 1:-1] - y[:-2, 1:-1]) ** 2
            + (z[2:, 1:-1] - z[:-2, 1:-1]) ** 2
        )
        dx_face = radius * 2.0 * jnp.arcsin(
            jnp.clip(dx_chord / 2.0, 0.0, 1.0)
        )

        # 2-cell chord in y: positions [i,j+1] – [i,j-1]
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

    hx_ext = jnp.stack(all_hx, axis=0)  # (6, n+2, n+2)
    hy_ext = jnp.stack(all_hy, axis=0)

    return hx_ext, hy_ext
