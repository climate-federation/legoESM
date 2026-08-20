"""
Friction velocity module for CLM-JAX.

This module provides data structures and initialization functions for
friction velocity calculations in the Community Land Model.

Translated from FrictionVelocityMod.F90 (lines 1-66).

Key components:
- FrictionVelType: NamedTuple holding friction velocity state variables
- Initialization functions for allocating and setting up state arrays

Reference:
    Original Fortran: FrictionVelocityMod.F90
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

# ============================================================================
# Type Definitions
# ============================================================================


class BoundsType(NamedTuple):
    """Domain bounds for patch indices.

    Attributes:
        begp: Beginning patch index
        endp: Ending patch index
    """

    begp: int
    endp: int


class FrictionVelType(NamedTuple):
    """
    Friction velocity state variables.

    Translated from Fortran type frictionvel_type (lines 18-28).

    Attributes:
        forc_hgt_u_patch: Observational height of wind at patch level [m],
            shape (n_patches,)
        u10_clm_patch: 10-m wind speed at patch level [m/s],
            shape (n_patches,)
        fv_patch: Friction velocity at patch level [m/s],
            shape (n_patches,)

    Reference:
        FrictionVelocityMod.F90:18-28
    """

    forc_hgt_u_patch: Array  # (n_patches,) - patch wind forcing height (m)
    u10_clm_patch: Array  # (n_patches,) - patch 10-m wind (m/s)
    fv_patch: Array  # (n_patches,) - patch friction velocity (m/s)


# ============================================================================
# Initialization Functions
# ============================================================================


def init_frictionvel_type(n_patches: int) -> FrictionVelType:
    """
    Initialize FrictionVelType with NaN values.

    Creates a new friction velocity state with all arrays initialized to NaN.
    This is useful for creating state arrays of a known size.

    Args:
        n_patches: Number of patches in the domain

    Returns:
        FrictionVelType: Initialized state with NaN arrays of shape (n_patches,)

    Reference:
        FrictionVelocityMod.F90:1-34 (module initialization pattern)
    """
    nan_val = jnp.nan

    return FrictionVelType(
        forc_hgt_u_patch=jnp.full(n_patches, nan_val, dtype=jnp.float64),
        u10_clm_patch=jnp.full(n_patches, nan_val, dtype=jnp.float64),
        fv_patch=jnp.full(n_patches, nan_val, dtype=jnp.float64),
    )


def init_allocate(bounds: BoundsType) -> FrictionVelType:
    """
    Initialize friction velocity data structure from bounds.

    Allocates and initializes arrays for friction velocity calculations
    based on the provided domain bounds. All arrays are initialized to NaN
    following the Fortran implementation.

    Translated from FrictionVelocityMod.F90, lines 45-64.

    Args:
        bounds: Domain bounds containing begp and endp patch indices

    Returns:
        FrictionVelType: Initialized friction velocity state with NaN values

    Note:
        Original Fortran allocates arrays from begp:endp and initializes to nan.
        In JAX, we create arrays of size (endp - begp + 1) to represent the
        same index range.

    Reference:
        FrictionVelocityMod.F90:45-64
    """
    # Lines 58-59: Extract bounds
    begp = bounds.begp
    endp = bounds.endp

    # Calculate array size for the patch range
    n_patches = endp - begp + 1

    # Lines 61-63: Allocate and initialize arrays to NaN
    # allocate (this%forc_hgt_u_patch (begp:endp)) ; this%forc_hgt_u_patch (:) = nan
    forc_hgt_u_patch = jnp.full(n_patches, jnp.nan, dtype=jnp.float64)

    # allocate (this%u10_clm_patch    (begp:endp)) ; this%u10_clm_patch    (:) = nan
    u10_clm_patch = jnp.full(n_patches, jnp.nan, dtype=jnp.float64)

    # allocate (this%fv_patch         (begp:endp)) ; this%fv_patch         (:) = nan
    fv_patch = jnp.full(n_patches, jnp.nan, dtype=jnp.float64)

    return FrictionVelType(
        forc_hgt_u_patch=forc_hgt_u_patch, u10_clm_patch=u10_clm_patch, fv_patch=fv_patch
    )


def init_friction_velocity(bounds: BoundsType) -> FrictionVelType:
    """
    Initialize friction velocity state (convenience wrapper).

    This function provides a high-level interface for initializing the
    friction velocity type by delegating to init_allocate.

    Translated from FrictionVelocityMod.F90, lines 35-42.

    Args:
        bounds: Domain bounds containing patch index range (begp:endp)

    Returns:
        FrictionVelType: Initialized friction velocity state with allocated
            arrays for the given bounds

    Note:
        This is a wrapper that calls init_allocate to perform the actual
        allocation. In the JAX translation, we directly return the initialized
        state rather than mutating a class instance.

    Reference:
        FrictionVelocityMod.F90:35-42
    """
    return init_allocate(bounds)


# ============================================================================
# Update Functions
# ============================================================================


def update_frictionvel_type(
    state: FrictionVelType,
    forc_hgt_u_patch: Array | None = None,
    u10_clm_patch: Array | None = None,
    fv_patch: Array | None = None,
) -> FrictionVelType:
    """
    Update FrictionVelType fields immutably.

    Creates a new FrictionVelType with updated field values. Only fields
    provided as arguments are updated; others retain their original values.
    This follows JAX's functional programming paradigm.

    Args:
        state: Current FrictionVelType state
        forc_hgt_u_patch: Optional new patch wind forcing height [m]
        u10_clm_patch: Optional new patch 10-m wind [m/s]
        fv_patch: Optional new patch friction velocity [m/s]

    Returns:
        FrictionVelType: New state with updated values where provided

    Reference:
        FrictionVelocityMod.F90:18-28 (type definition)
    """
    return FrictionVelType(
        forc_hgt_u_patch=(
            forc_hgt_u_patch if forc_hgt_u_patch is not None else state.forc_hgt_u_patch
        ),
        u10_clm_patch=(u10_clm_patch if u10_clm_patch is not None else state.u10_clm_patch),
        fv_patch=(fv_patch if fv_patch is not None else state.fv_patch),
    )


# Backward compatibility alias (Fortran naming convention)
frictionvel_type = FrictionVelType
