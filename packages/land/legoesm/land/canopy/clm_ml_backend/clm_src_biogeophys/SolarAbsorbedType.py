"""Solar absorbed radiation module for CLM.

This module defines the data structure and initialization functions for solar
radiation absorption variables in the Community Land Model (CLM).

Fortran source: SolarAbsorbedType.F90, lines 1-63

Key components:
- SolarAbsState: NamedTuple holding solar absorption state variables
- Initialization functions for allocating and setting up state arrays
- Pure functional interface for state updates

Note:
    All functions are pure and return new state objects rather than modifying
    in place, following JAX best practices for JIT compilation.
"""

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

# ============================================================================
# Type Definitions
# ============================================================================


class BoundsType(NamedTuple):
    """Domain bounds for grid dimensions.

    Attributes:
        begp: Beginning patch index
        endp: Ending patch index (inclusive)
    """

    begp: int
    endp: int


class SolarAbsState(NamedTuple):
    """Solar absorption state variables.

    This corresponds to the solarabs_type in Fortran (lines 19-27).

    Attributes:
        fsa_patch: Patch solar radiation absorbed (total) (W/m²).
                   Shape: (n_patches,)
                   Fortran line 21: real(r8), pointer :: fsa_patch (:)
    """

    fsa_patch: Array  # (n_patches,) - solar radiation absorbed (W/m²)


# Type alias for consistency with Fortran naming
SolarAbsType = SolarAbsState


# ============================================================================
# Initialization Functions
# ============================================================================


def init_solar_abs_state(n_patches: int) -> SolarAbsState:
    """Initialize solar absorption state with NaN values.

    This corresponds to the Init procedure in Fortran (line 24).

    Args:
        n_patches: Number of patches in the domain.

    Returns:
        SolarAbsState: Initialized state with NaN values.

    Note:
        Fortran lines 24-26: procedure declarations for Init and InitAllocate
    """
    return SolarAbsState(fsa_patch=jnp.full(n_patches, jnp.nan, dtype=jnp.float64))


def init_allocate_solar_abs(n_patches: int) -> SolarAbsState:
    """Allocate and initialize solar absorption state.

    This corresponds to the InitAllocate procedure in Fortran (line 25).

    Args:
        n_patches: Number of patches in the domain.

    Returns:
        SolarAbsState: Allocated and initialized state.

    Note:
        Fortran line 25: procedure, private :: InitAllocate
    """
    return init_solar_abs_state(n_patches)


def init_allocate(
    this: SolarAbsType,
    bounds: BoundsType,
) -> SolarAbsType:
    """Initialize solar absorption data structure with allocated arrays.

    Allocates and initializes the fsa_patch array for the given bounds.
    Arrays are initialized with NaN values.

    Fortran source: SolarAbsorbedType.F90, lines 44-61

    Args:
        this: Current solar absorption state (may be partially initialized)
        bounds: Bounds type containing patch index ranges

    Returns:
        Updated SolarAbsType with allocated and initialized fsa_patch array

    Note:
        This corresponds to the Fortran InitAllocate subroutine which allocates
        the fsa_patch array from begp to endp and initializes with NaN.
    """
    # Lines 57-58: Extract patch bounds
    begp = bounds.begp
    endp = bounds.endp

    # Line 60: Allocate and initialize fsa_patch with NaN
    # In JAX, we create arrays with the appropriate size
    # Array size is (endp - begp + 1) to match Fortran inclusive bounds
    n_patches = endp - begp + 1
    fsa_patch = jnp.full(n_patches, jnp.nan, dtype=jnp.float64)

    # Return updated state with allocated array
    return this._replace(fsa_patch=fsa_patch)


def init(
    bounds: BoundsType,
) -> SolarAbsType:
    """Initialize solar absorption state.

    This function initializes the solar absorption data structure by allocating
    arrays based on the provided bounds.

    Fortran source: SolarAbsorbedType.F90, lines 34-41

    Args:
        bounds: Domain bounds containing grid dimensions

    Returns:
        Initialized SolarAbsType state with allocated arrays

    Note:
        This is a pure function that creates and returns a new state object.
        In the Fortran version, this was a class method that modified the object
        in place via InitAllocate.
    """
    # Create empty state to pass to init_allocate
    empty_state = SolarAbsState(fsa_patch=jnp.array([], dtype=jnp.float64))
    return init_allocate(empty_state, bounds)


# ============================================================================
# State Update Functions
# ============================================================================


def update_fsa_patch(state: SolarAbsState, fsa_patch: Array) -> SolarAbsState:
    """Update the patch solar radiation absorbed values.

    Args:
        state: Current solar absorption state.
        fsa_patch: New solar radiation absorbed values (W/m²).
                   Shape: (n_patches,)

    Returns:
        SolarAbsState: Updated state with new fsa_patch values.

    Note:
        This is a helper function for updating the immutable state.
        Fortran line 21: fsa_patch array update
    """
    return state._replace(fsa_patch=fsa_patch)


# Backward compatibility alias (Fortran naming convention)
solarabs_type = SolarAbsState
