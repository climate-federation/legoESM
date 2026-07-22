"""Surface albedo variables and initialization for CTSM.

This module defines the data structure for surface albedo calculations,
including cosine of solar zenith angle and various albedo components for
direct and diffuse radiation in multiple spectral bands.

Fortran source: SurfaceAlbedoType.F90, lines 1-74

Key components:
- SurfaceAlbedoState: Immutable state container for albedo variables
- init_surface_albedo_state: Initialize state with NaN values
- update_surface_albedo_state: Create new state with updated fields
- init_allocate: Allocate and initialize arrays based on domain bounds

Note:
    All functions are pure and JIT-compatible. State updates return new
    NamedTuple instances rather than modifying in place.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

# =============================================================================
# Constants and Parameters
# =============================================================================

# Number of radiation bands (visible and near-infrared)
# Fortran source: clm_varpar, line reference in uses
NUMRAD: int = 2


# =============================================================================
# Type Definitions
# =============================================================================


class BoundsType(NamedTuple):
    """Domain decomposition bounds for patches and columns.

    Corresponds to Fortran bounds_type from decompMod.

    Attributes:
        begp: Beginning patch index (1-based in Fortran, 0-based here).
        endp: Ending patch index (inclusive).
        begc: Beginning column index (1-based in Fortran, 0-based here).
        endc: Ending column index (inclusive).
        begg: Beginning gridcell index (optional, for completeness).
        endg: Ending gridcell index (optional, for completeness).
        begl: Beginning landunit index (optional, for completeness).
        endl: Ending landunit index (optional, for completeness).

    Note:
        Fortran source: decompMod module
        JAX uses 0-based indexing, but we preserve Fortran semantics
        where appropriate for array sizing.
    """

    begp: int
    endp: int
    begc: int
    endc: int
    begg: int = 0
    endg: int = 0
    begl: int = 0
    endl: int = 0


class SurfaceAlbedoState(NamedTuple):
    """Surface albedo state variables.

    This corresponds to the Fortran surfalb_type derived type.
    All arrays are immutable and designed for JIT compilation.

    Attributes:
        coszen_col: Cosine of solar zenith angle for each column [ncol].
            Fortran: coszen_col(:) - line 21
        albd_patch: Patch surface albedo (direct) [npatch, numrad].
            Fortran: albd_patch(:,:) - line 22
        albi_patch: Patch surface albedo (diffuse) [npatch, numrad].
            Fortran: albi_patch(:,:) - line 23
        albgrd_col: Column ground albedo (direct) [ncol, numrad].
            Fortran: albgrd_col(:,:) - line 24
        albgri_col: Column ground albedo (diffuse) [ncol, numrad].
            Fortran: albgri_col(:,:) - line 25

    Note:
        numrad is typically 2 (visible and near-infrared bands).
        All arrays initialized to NaN following Fortran convention.
        Fortran source lines 21-25.
    """

    coszen_col: Array  # shape: (ncol,)
    albd_patch: Array  # shape: (npatch, numrad)
    albi_patch: Array  # shape: (npatch, numrad)
    albgrd_col: Array  # shape: (ncol, numrad)
    albgri_col: Array  # shape: (ncol, numrad)


# =============================================================================
# Initialization Functions
# =============================================================================


def init_surface_albedo_state(ncol: int, npatch: int, numrad: int = NUMRAD) -> SurfaceAlbedoState:
    """Initialize surface albedo state with default values.

    Creates a new SurfaceAlbedoState with all arrays initialized to NaN,
    corresponding to the Fortran Init and InitAllocate procedures.

    Args:
        ncol: Number of columns in the grid.
        npatch: Number of patches in the grid.
        numrad: Number of radiation bands (default: 2 for visible and NIR).

    Returns:
        SurfaceAlbedoState: Initialized state with NaN values.

    Note:
        Fortran source lines 21-29, 39-46
        In Fortran, arrays are allocated and initialized to nan (spval).
        Here we use jnp.nan for consistency with the Fortran behavior.
        Uses float32 for memory efficiency while maintaining precision.
    """
    # Size numrad+1 so that 1-based ivis=1, inir=2 indices are valid (slot 0 unused)
    return SurfaceAlbedoState(
        coszen_col=jnp.full((ncol,), jnp.nan, dtype=jnp.float32),
        albd_patch=jnp.full((npatch, numrad + 1), jnp.nan, dtype=jnp.float32),
        albi_patch=jnp.full((npatch, numrad + 1), jnp.nan, dtype=jnp.float32),
        albgrd_col=jnp.full((ncol, numrad + 1), jnp.nan, dtype=jnp.float32),
        albgri_col=jnp.full((ncol, numrad + 1), jnp.nan, dtype=jnp.float32),
    )


def init_allocate(bounds: BoundsType, numrad: int = NUMRAD) -> SurfaceAlbedoState:
    """Initialize and allocate surface albedo data structure.

    Allocates arrays for surface albedo variables and initializes them with NaN.
    This follows the Fortran pattern of allocating arrays with bounds and setting
    initial values.

    Args:
        bounds: Bounds structure containing patch and column indices.
        numrad: Number of radiation bands (default: 2 for visible and near-infrared).

    Returns:
        SurfaceAlbedoState: Initialized surface albedo data structure with NaN values.

    Note:
        Fortran source lines 49-72
        - Line 64: begp = bounds%begp ; endp = bounds%endp
        - Line 65: begc = bounds%begc ; endc = bounds%endc
        - Lines 67-71: Array allocations and NaN initialization

        Fortran uses 1-based indexing with inclusive bounds.
        Array size = end - beg + 1 to match Fortran semantics.
    """
    # Extract bounds (Fortran lines 64-65)
    begp = bounds.begp
    endp = bounds.endp
    begc = bounds.begc
    endc = bounds.endc

    # Calculate array sizes (inclusive bounds in Fortran)
    num_cols = endc - begc + 1
    num_patches = endp - begp + 1

    # Allocate and initialize arrays with NaN (Fortran lines 67-71)
    # Line 67: allocate (this%coszen_col (begc:endc))
    coszen_col = jnp.full((num_cols,), jnp.nan, dtype=jnp.float32)

    # Line 68: allocate (this%albd_patch (begp:endp,1:numrad))
    # Size numrad+1 so that 1-based ivis=1, inir=2 indices are valid (slot 0 unused)
    albd_patch = jnp.full((num_patches, numrad + 1), jnp.nan, dtype=jnp.float32)

    # Line 69: allocate (this%albi_patch (begp:endp,1:numrad))
    albi_patch = jnp.full((num_patches, numrad + 1), jnp.nan, dtype=jnp.float32)

    # Line 70: allocate (this%albgrd_col (begc:endc,1:numrad))
    # Size numrad+1 so that 1-based ivis=1, inir=2 indices are valid (slot 0 unused)
    albgrd_col = jnp.full((num_cols, numrad + 1), jnp.nan, dtype=jnp.float32)

    # Line 71: allocate (this%albgri_col (begc:endc,1:numrad))
    albgri_col = jnp.full((num_cols, numrad + 1), jnp.nan, dtype=jnp.float32)

    return SurfaceAlbedoState(
        coszen_col=coszen_col,
        albd_patch=albd_patch,
        albi_patch=albi_patch,
        albgrd_col=albgrd_col,
        albgri_col=albgri_col,
    )


def init(bounds: BoundsType, numrad: int = NUMRAD) -> SurfaceAlbedoState:
    """Initialize surface albedo type.

    This function initializes the surface albedo data structure by calling
    init_allocate to set up all required arrays and fields.

    Args:
        bounds: Bounds type containing domain decomposition information
            (begg, endg, begl, endl, begc, endc, begp, endp).
        numrad: Number of radiation bands (default: 2).

    Returns:
        SurfaceAlbedoState: Initialized surface albedo type with allocated arrays.

    Note:
        Fortran source: SurfaceAlbedoType.F90, lines 39-46
        This is a wrapper that delegates to init_allocate.
        Line 44: call this%InitAllocate (bounds)
    """
    return init_allocate(bounds, numrad)


# =============================================================================
# State Update Functions
# =============================================================================


def update_surface_albedo_state(
    state: SurfaceAlbedoState,
    coszen_col: Array | None = None,
    albd_patch: Array | None = None,
    albi_patch: Array | None = None,
    albgrd_col: Array | None = None,
    albgri_col: Array | None = None,
) -> SurfaceAlbedoState:
    """Update surface albedo state with new values.

    Creates a new SurfaceAlbedoState with updated fields. This is the
    JAX-idiomatic way to handle state updates (immutable data structures).

    Args:
        state: Current surface albedo state.
        coszen_col: New cosine of solar zenith angle values (optional).
        albd_patch: New patch direct albedo values (optional).
        albi_patch: New patch diffuse albedo values (optional).
        albgrd_col: New column direct ground albedo values (optional).
        albgri_col: New column diffuse ground albedo values (optional).

    Returns:
        SurfaceAlbedoState: New state with updated values.

    Note:
        This function provides a convenient way to update individual fields
        while maintaining immutability, which is required for JAX JIT compilation.
        No direct Fortran equivalent - this is a JAX design pattern.
    """
    return SurfaceAlbedoState(
        coszen_col=coszen_col if coszen_col is not None else state.coszen_col,
        albd_patch=albd_patch if albd_patch is not None else state.albd_patch,
        albi_patch=albi_patch if albi_patch is not None else state.albi_patch,
        albgrd_col=albgrd_col if albgrd_col is not None else state.albgrd_col,
        albgri_col=albgri_col if albgri_col is not None else state.albgri_col,
    )


# Backward compatibility alias (Fortran naming convention)
surfalb_type = SurfaceAlbedoState
