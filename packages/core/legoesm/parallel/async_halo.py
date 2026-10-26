"""Async halo exchange with compute/communication overlap.

Pattern:
    1. Start non-blocking halo sends/receives
    2. Compute interior points (no halo dependency)
    3. Wait for halo data
    4. Compute boundary points (needs halo)
    5. Merge interior + boundary results

Since ``mpi4jax`` only exposes blocking ``sendrecv`` (no ``isend``/``irecv``),
true non-blocking overlap at the MPI level is not available.  Instead, we
implement the overlap at a *higher level*:

    1. Compute the stencil operator on **interior** points that do NOT depend
       on any halo data (i.e., points at least ``halo_width`` away from
       face/tile edges).
    2. Perform the **blocking** halo exchange (which only moves edge strips).
    3. Compute the stencil operator on **boundary** points that DO depend on
       halo data.
    4. Merge the two results.

This is still beneficial because the interior computation and the
communication data packing/unpacking are separated, enabling the XLA
compiler to overlap the interior kernel with the host-side MPI scheduling.

For the local (single-node) backend the same split is available and useful
for pipelining: interior work is independent of neighbour-face reads, so
XLA can schedule them in parallel when the computation graph is structured
this way.

All functions are pure and compatible with ``jax.jit``, ``jax.grad``, and
``jax.vmap``.

Usage
-----
::

    from legoesm.parallel.async_halo import overlapped_halo_compute

    # compute_fn takes (6, n+2h, n+2h) padded field -> (6, n, n) result
    result = overlapped_halo_compute(field, compute_fn, halo_width=1)

Or use the lower-level utilities directly::

    from legoesm.parallel.async_halo import (
        create_interior_boundary_masks,
        split_interior_boundary,
        merge_interior_boundary,
    )

    masks = create_interior_boundary_masks(n, halo_width=1)
    interior, boundary = split_interior_boundary(field, masks)
"""

from __future__ import annotations

from typing import Callable, NamedTuple, TYPE_CHECKING

import jax
import jax.numpy as jnp

from legoesm.grids.halo import (
    CONNECTIVITY,
    EAST,
    NORTH,
    SOUTH,
    WEST,
    pad_halo,
)
from legoesm.parallel.halo_exchange import pad_halo_mpi
from legoesm.parallel.mesh import get_active_config

if TYPE_CHECKING:
    from legoesm.parallel.comm import CommTopology


# ======================================================================
# JAX-native halo exchange (non-blocking, device-native)
# ======================================================================

def jax_native_halo_exchange(data, grid, mesh=None):
    """JAX-native halo exchange using collective operations.

    .. note::

       This is a **legacy** entry point.  For production multi-device
       halo exchange, use the SPMD backend activated via
       ``activate_spmd_halo_backend()`` in ``cubesphere_exchange.py``
       (ppermute multiface exchange; all_gather only as the explicit
       ``LEGOESM_SPMD_FORCE_ALLGATHER=1`` diagnostic — it replicates
       compute, HLO probe job 8456476).

    When running on multi-GPU/TPU with JAX sharding, this uses
    ``jax.lax.ppermute`` for device-to-device communication instead of MPI.
    Falls back to the local halo pad when mesh is None.

    Parameters
    ----------
    data : jax.Array
        Field to exchange halos for, shape (6, n, n) or (6, n, n, nlev).
    grid : CubedSphereGrid
        Grid with halo metadata and connectivity information.
    mesh : jax.sharding.Mesh, optional
        JAX device mesh for multi-GPU/TPU environments. If provided, uses
        the experimental ``ppermute`` path for device-to-device halos;
        otherwise falls back to the local halo pad. Default None.

    Returns
    -------
    jax.Array
        Halo-padded field with same shape as input but with halo regions filled.

    Notes
    -----
    The ppermute implementation is experimental and:

    1. Uses the grid's face connectivity to define permutation patterns.
    2. Applies ``jax.lax.ppermute`` calls for each edge/halo strip.
    3. Avoids MPI entirely, keeping computation on-device.
    4. Integrates with JAX's collective operations for potential
       compiler-level optimizations.

    This path has not been validated beyond single-node multi-device setups.
    """
    if mesh is not None:
        import warnings
        # ppermute path only works for face-only sharding (6 faces, no tiles).
        # Reject sub-face tiling to prevent silent incorrect results.
        active_cfg = get_active_config()
        if active_cfg is not None and getattr(active_cfg, 'tiling', (1, 1)) != (1, 1):
            warnings.warn(
                "ppermute halo exchange does not support sub-face tiling "
                f"(tiling={active_cfg.tiling}). Falling back to local/MPI-based "
                "exchange. Use mesh=None to suppress this warning.",
                RuntimeWarning,
                stacklevel=2,
            )
            return pad_halo(data)
        warnings.warn(
            "jax_native_halo_exchange: the ppermute-based code path is "
            "experimental and has not been validated at scale. "
            "Use mesh=None for the well-tested local/MPI-based exchange.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _ppermute_halo_exchange(data, grid, mesh)
    # Fall back to the standard local halo pad (no MPI needed for
    # single-node multi-GPU when XLA handles data movement via sharding).
    return pad_halo(data)


def _ppermute_halo_exchange(data, grid, mesh):
    """Implement halo exchange via jax.lax.ppermute (experimental).

    .. warning::

       This function is experimental and has not been validated beyond
       basic single-node multi-device setups.  It may produce incorrect
       halo data for complex connectivity patterns or multi-node runs.

    Uses the cubed-sphere CONNECTIVITY table to build permutation
    patterns that move edge strips directly between devices using
    the interconnect (NCCL on GPU, ICI on TPU), avoiding MPI.

    Parameters
    ----------
    data : jax.Array, shape (6, n, n, ...)
        Field sharded on axis 0 ("face" axis) across devices.
    grid : CubedSphereGrid
        Grid with halo metadata.
    mesh : jax.sharding.Mesh
        Device mesh with a "face" axis.

    Returns
    -------
    jax.Array
        Halo-padded field.
    """
    from legoesm.grids.halo import extract_edge_strip

    n_devices = mesh.shape["face"] if "face" in mesh.axis_names else 1
    if n_devices <= 1:
        return pad_halo(data)

    # Single Pad HLO op replaces alloc-zeros + scatter (the subsequent
    # halo scatter only writes into the zeroed border).
    padded = jnp.pad(data, ((0, 0), (1, 1), (1, 1)))

    edges = (WEST, EAST, SOUTH, NORTH)
    faces_per_device = 6 // n_devices

    for edge in edges:
        # Build permutation: for each face, which device holds the
        # neighbor face for this edge?
        perm = []
        for src_face in range(6):
            nbr_face, _, _ = CONNECTIVITY[src_face][edge]
            src_dev = src_face // faces_per_device
            dst_dev = nbr_face // faces_per_device
            if src_dev != dst_dev and (src_dev, dst_dev) not in perm:
                perm.append((src_dev, dst_dev))

        if not perm:
            # All neighbors are local — use standard pad_halo logic.
            continue

        # Extract edge strips for all faces along this edge.
        strips = jax.vmap(lambda f: extract_edge_strip(data, f, edge))(
            jnp.arange(6)
        )  # (6, n)

        # Permute strips between devices.
        strips_permuted = jax.lax.ppermute(
            strips, axis_name="face", perm=perm,
        )

        # Place received strips into halo positions.
        for src_face in range(6):
            nbr_face, nbr_edge, is_reversed = CONNECTIVITY[src_face][edge]
            strip = strips_permuted[nbr_face]
            if is_reversed:
                strip = strip[::-1]
            if edge == WEST:
                padded = padded.at[src_face, 0, 1:-1].set(strip)
            elif edge == EAST:
                padded = padded.at[src_face, -1, 1:-1].set(strip)
            elif edge == SOUTH:
                padded = padded.at[src_face, 1:-1, 0].set(strip)
            elif edge == NORTH:
                padded = padded.at[src_face, 1:-1, -1].set(strip)

    return padded


# ======================================================================
# Interior / boundary mask creation
# ======================================================================

class InteriorBoundaryMasks(NamedTuple):
    """Masks that split a cubed-sphere face into interior and boundary regions.

    Attributes
    ----------
    interior : jnp.ndarray
        Boolean mask, shape ``(n, n)``.  ``True`` for grid points at least
        ``halo_width`` away from all face edges (these points do NOT depend
        on halo data for a stencil of width ``halo_width``).
    boundary : jnp.ndarray
        Boolean mask, shape ``(n, n)``.  ``True`` for grid points within
        ``halo_width`` of at least one face edge (these points NEED halo
        data).
    halo_width : int
        The halo width used to construct the masks.
    n : int
        Grid dimension per face edge.
    """
    interior: jnp.ndarray
    boundary: jnp.ndarray
    halo_width: int
    n: int


def create_interior_boundary_masks(
    n: int,
    halo_width: int = 1,
) -> InteriorBoundaryMasks:
    """Create masks splitting a cubed-sphere face into interior and boundary.

    Interior points are at least ``halo_width`` cells away from every face
    edge in both the i and j directions.  Boundary points are the
    complement: within ``halo_width`` of at least one edge.

    Parameters
    ----------
    n : int
        Number of cells per face edge.
    halo_width : int
        Width of the halo region (typically 1 or 2).

    Returns
    -------
    InteriorBoundaryMasks
        Named tuple with boolean masks of shape ``(n, n)``.

    Examples
    --------
    >>> masks = create_interior_boundary_masks(8, halo_width=1)
    >>> masks.interior.shape
    (8, 8)
    >>> int(masks.interior.sum())  # 6x6 interior
    36
    >>> int(masks.boundary.sum())  # 8x8 - 6x6 boundary
    28
    """
    if halo_width < 1:
        raise ValueError(f"halo_width must be >= 1, got {halo_width}")
    if n < 2 * halo_width + 1:
        raise ValueError(
            f"Grid dimension n={n} is too small for halo_width={halo_width}. "
            f"Need n >= {2 * halo_width + 1}."
        )

    # Build 1D mask: True for indices in [halo_width, n - halo_width)
    idx = jnp.arange(n)
    interior_1d = (idx >= halo_width) & (idx < n - halo_width)

    # 2D interior = both i and j are interior
    interior = interior_1d[:, None] & interior_1d[None, :]
    boundary = ~interior

    return InteriorBoundaryMasks(
        interior=interior,
        boundary=boundary,
        halo_width=halo_width,
        n=n,
    )


# ======================================================================
# Splitting and merging
# ======================================================================

def split_interior_boundary(
    field: jax.Array,
    masks: InteriorBoundaryMasks,
) -> tuple[jax.Array, jax.Array]:
    """Partition a cubed-sphere field into interior and boundary parts.

    Both returned arrays have the same shape as ``field``; the
    non-selected region is filled with zeros.  This keeps shapes static
    for JAX tracing.

    Parameters
    ----------
    field : jax.Array, shape ``(6, n, n)`` or ``(6, n, n, ...)``
        Cubed-sphere scalar or multi-level field.
    masks : InteriorBoundaryMasks
        Precomputed masks from :func:`create_interior_boundary_masks`.

    Returns
    -------
    interior_field : jax.Array
        Same shape as *field*; boundary points zeroed out.
    boundary_field : jax.Array
        Same shape as *field*; interior points zeroed out.
    """
    # Broadcast the (n, n) mask to match field shape (6, n, n, ...)
    n_extra = field.ndim - 3  # extra trailing dims (e.g., vertical levels)
    mask_shape = (1,) + masks.interior.shape + (1,) * n_extra
    int_mask = masks.interior.reshape(mask_shape)
    bnd_mask = masks.boundary.reshape(mask_shape)

    interior_field = jnp.where(int_mask, field, 0.0)
    boundary_field = jnp.where(bnd_mask, field, 0.0)

    return interior_field, boundary_field


def merge_interior_boundary(
    interior_result: jax.Array,
    boundary_result: jax.Array,
    masks: InteriorBoundaryMasks,
) -> jax.Array:
    """Merge interior and boundary computation results.

    Selects values from ``interior_result`` at interior points and from
    ``boundary_result`` at boundary points.

    Parameters
    ----------
    interior_result : jax.Array, shape ``(6, n, n)`` or ``(6, n, n, ...)``
        Result of stencil computation on interior region.
    boundary_result : jax.Array, shape ``(6, n, n)`` or ``(6, n, n, ...)``
        Result of stencil computation on boundary region (after halo fill).
    masks : InteriorBoundaryMasks
        Precomputed masks.

    Returns
    -------
    merged : jax.Array
        Combined result with same shape.
    """
    n_extra = interior_result.ndim - 3
    mask_shape = (1,) + masks.interior.shape + (1,) * n_extra
    int_mask = masks.interior.reshape(mask_shape)

    return jnp.where(int_mask, interior_result, boundary_result)


# ======================================================================
# Boundary index helpers
# ======================================================================

def boundary_slices(
    n: int,
    halo_width: int = 1,
) -> dict[str, tuple[slice, slice]]:
    """Return index slices for the four boundary strips of a face.

    Each strip is a rectangular region of the ``(n, n)`` face that lies
    within ``halo_width`` of an edge.

    Parameters
    ----------
    n : int
        Grid dimension per face edge.
    halo_width : int
        Halo width.

    Returns
    -------
    dict mapping ``"west"``/``"east"``/``"south"``/``"north"`` to
    ``(i_slice, j_slice)`` tuples.
    """
    h = halo_width
    return {
        "west":  (slice(0, h), slice(0, n)),         # i < h
        "east":  (slice(n - h, n), slice(0, n)),      # i >= n-h
        "south": (slice(0, n), slice(0, h)),          # j < h
        "north": (slice(0, n), slice(n - h, n)),      # j >= n-h
    }


def interior_slice(
    n: int,
    halo_width: int = 1,
) -> tuple[slice, slice]:
    """Return the ``(i_slice, j_slice)`` for the interior region.

    Parameters
    ----------
    n : int
        Grid dimension per face edge.
    halo_width : int
        Halo width.

    Returns
    -------
    tuple of two slices selecting the interior ``(n - 2*h, n - 2*h)``
    block.
    """
    h = halo_width
    return (slice(h, n - h), slice(h, n - h))


# ======================================================================
# Padded-domain interior/boundary extraction
# ======================================================================

def extract_interior_padded(
    padded: jax.Array,
    halo_width: int = 1,
) -> jax.Array:
    """Extract the full interior from a halo-padded array.

    For a padded array of shape ``(6, n + 2*h, n + 2*h)``, returns the
    ``(6, n, n)`` interior (strips the halo padding on all sides).

    Parameters
    ----------
    padded : jax.Array, shape ``(6, n+2h, n+2h)``
    halo_width : int

    Returns
    -------
    jax.Array, shape ``(6, n, n)``
    """
    h = halo_width
    return padded[:, h:-h, h:-h]


def build_boundary_stencil_padded(
    padded: jax.Array,
    masks: InteriorBoundaryMasks,
    halo_width: int = 1,
) -> jax.Array:
    """Zero out interior points in a halo-padded array.

    Returns a copy of *padded* where the interior region (the points
    that do NOT need halo data) is set to zero.  This is useful for
    applying a stencil only to boundary points: the stencil can read
    the full padded domain but only boundary-region outputs are used.

    Parameters
    ----------
    padded : jax.Array, shape ``(6, n+2h, n+2h)``
    masks : InteriorBoundaryMasks
    halo_width : int

    Returns
    -------
    jax.Array, shape ``(6, n+2h, n+2h)``
    """
    h = halo_width
    # Embed the (n, n) boundary mask into the (n+2h, n+2h) padded shape.
    # The halo ring itself is always "boundary" so we start with all-True.
    n = masks.n
    n_padded = n + 2 * h
    full_mask = jnp.ones((n_padded, n_padded), dtype=jnp.bool_)
    # Zero out the interior block
    full_mask = full_mask.at[h:-h, h:-h].set(masks.boundary)
    return padded * full_mask[None, :, :]


# ======================================================================
# Main overlapped compute function
# ======================================================================

def overlapped_halo_compute(
    field: jax.Array,
    compute_fn: Callable[[jax.Array], jax.Array],
    halo_width: int = 1,
    interp_offsets: jax.Array | None = None,
    masks: InteriorBoundaryMasks | None = None,
) -> jax.Array:
    """Execute a stencil with overlapped halo exchange.

    This is the main entry point for compute/communication overlap.  The
    pattern is:

    1. Pad the field with a *partial* halo: copy only local-face data
       (no cross-face exchange yet).  Interior points already have all
       the data they need because their stencil footprint does not
       reach the face boundary.
    2. Compute ``compute_fn`` on the partially-padded field.  The
       result is correct for interior points.
    3. Perform the full halo exchange (blocking).
    4. Compute ``compute_fn`` on the fully-padded field.  The result
       is correct for boundary points.
    5. Merge: take interior values from step 2, boundary values from
       step 4.

    ``compute_fn`` signature
    ~~~~~~~~~~~~~~~~~~~~~~~~
    Takes a padded array of shape ``(6, n+2h, n+2h)`` and returns a
    result array of shape ``(6, n, n)`` (the stencil output on the
    interior grid, having consumed the halo).

    Parameters
    ----------
    field : jax.Array, shape ``(6, n, n)``
        Input cubed-sphere scalar field.
    compute_fn : callable
        Stencil function: ``(6, n+2h, n+2h) -> (6, n, n)``.
    halo_width : int
        Halo width (1 or 2).
    interp_offsets : jax.Array or None
        Interpolation offsets for the halo exchange (forwarded to
        :func:`legoesm.grids.halo.pad_halo`).
    masks : InteriorBoundaryMasks or None
        Precomputed masks.  If ``None``, they are created on the fly.
        Pass precomputed masks to avoid re-creation in tight loops.

    Returns
    -------
    jax.Array, shape ``(6, n, n)``
        The stencil result, with interior computed before the halo
        exchange completes and boundary computed after.
    """
    n = field.shape[1]
    h = halo_width

    if masks is None:
        masks = create_interior_boundary_masks(n, halo_width=h)

    # ------------------------------------------------------------------
    # Step 1: Build a partial padded array using only local face data.
    # For interior points the stencil footprint stays within the face,
    # so extrapolating the boundary is sufficient (the values at boundary
    # stencil outputs will be garbage, but we discard them later).
    # ------------------------------------------------------------------
    padded_partial = _pad_local_only(field, halo_width=h)

    # ------------------------------------------------------------------
    # Step 2: Compute on the partial padded field.
    # Interior result is correct; boundary result is junk.
    # ------------------------------------------------------------------
    interior_result = compute_fn(padded_partial)

    # ------------------------------------------------------------------
    # Step 3: Full (blocking) halo exchange.
    # ------------------------------------------------------------------
    padded_full = pad_halo(field, halo=h, interp_offsets=interp_offsets)

    # ------------------------------------------------------------------
    # Step 4: Compute on the fully-padded field.
    # Boundary result is now correct.
    # ------------------------------------------------------------------
    boundary_result = compute_fn(padded_full)

    # ------------------------------------------------------------------
    # Step 5: Merge — interior from step 2, boundary from step 4.
    # ------------------------------------------------------------------
    return merge_interior_boundary(interior_result, boundary_result, masks)


# ======================================================================
# Local-only padding (no cross-face exchange)
# ======================================================================

def _pad_local_only(
    data: jax.Array,
    halo_width: int = 1,
) -> jax.Array:
    """Pad a field using only local (same-face) boundary extrapolation.

    Creates a ``(6, n+2h, n+2h)`` padded array where halo cells are
    filled by replicating the nearest edge value of each face.  This is
    sufficient for stencil operations on interior points (where the
    stencil footprint does not reach the face edge), and avoids any
    cross-face data movement.

    Parameters
    ----------
    data : jax.Array, shape ``(6, n, n)``
    halo_width : int

    Returns
    -------
    padded : jax.Array, shape ``(6, n+2h, n+2h)``
    """
    # jnp.pad with mode='edge' replicates boundary values
    return jnp.pad(data, ((0, 0), (halo_width, halo_width),
                          (halo_width, halo_width)), mode='edge')


# ======================================================================
# Precomputed overlap context (for tight loops)
# ======================================================================

class OverlapContext(NamedTuple):
    """Precomputed data for repeated overlapped halo computes.

    Create once and reuse across time steps to avoid re-creating masks.

    Attributes
    ----------
    masks : InteriorBoundaryMasks
        Interior/boundary masks.
    halo_width : int
        Halo width.
    n : int
        Grid dimension per face edge.
    interp_offsets : jnp.ndarray or None
        Interpolation offsets for the halo exchange.
    """
    masks: InteriorBoundaryMasks
    halo_width: int
    n: int
    interp_offsets: jnp.ndarray | None


def create_overlap_context(
    n: int,
    halo_width: int = 1,
    interp_offsets: jax.Array | None = None,
) -> OverlapContext:
    """Create a reusable overlap context for repeated calls.

    Parameters
    ----------
    n : int
        Grid dimension per face edge.
    halo_width : int
        Halo width (1 or 2).
    interp_offsets : jax.Array or None
        Precomputed interpolation offsets.

    Returns
    -------
    OverlapContext
    """
    masks = create_interior_boundary_masks(n, halo_width)
    return OverlapContext(
        masks=masks,
        halo_width=halo_width,
        n=n,
        interp_offsets=interp_offsets,
    )


# ======================================================================
# MPI-aware split-compute functions
# ======================================================================

def start_halo_exchange(
    field: jax.Array,
    topology: "CommTopology",
    halo_width: int = 1,
) -> jax.Array:
    """Initiate MPI halo exchange and return the padded result.

    .. warning::
        **This is currently a blocking pass-through to** :func:`pad_halo_mpi`,
        not a non-blocking primitive.  ``mpi4jax`` does not yet expose
        ``Isend`` / ``Irecv``, so the ``start / compute_interior / finish``
        idiom this function suggests cannot deliver real overlap on top of
        it.  The packed 4D halo exchange (``packed_pad_halo_mpi_4d``)
        coalesces multiple fields into one collective and is the path
        production code should use.  This API is preserved for back-
        compat and as a hook for a future ``mpi4jax`` non-blocking
        upgrade — its current performance profile is identical to
        ``pad_halo_mpi``.

    Parameters
    ----------
    field : jax.Array, shape ``(6, n, n)``
        Cubed-sphere scalar field.
    topology : CommTopology
        Pre-computed MPI communication topology.
    halo_width : int
        Halo width (1 or 2).

    Returns
    -------
    jax.Array, shape ``(6, n+2h, n+2h)``
        Halo-padded field with correct neighbor data on all edges.
    """
    return pad_halo_mpi(field, topology, halo=halo_width)


