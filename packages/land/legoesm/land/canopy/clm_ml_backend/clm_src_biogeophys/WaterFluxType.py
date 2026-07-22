"""
WaterFluxType module - Water flux variables for CTSM

Translated from Fortran source: WaterFluxType.F90, lines 1-63

This module defines water flux state variables and initialization routines
for the Community Terrestrial Systems Model (CTSM). It manages evaporation
and transpiration fluxes at the patch level.

Key components:
- WaterFluxType: NamedTuple holding water flux state variables
- init_waterflux_type: Initialize flux state with default values
- init: Main initialization routine
- init_allocate: Allocate and initialize flux arrays based on domain bounds
- update_qflx_evap_tot_patch: Functional update for evaporation flux

Fortran module dependencies:
- shr_kind_mod: Real kind parameter (r8)
- clm_varcon: Constants (ispval, spval/nan)
- decompMod: Domain decomposition bounds
"""

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

# ============================================================================
# Type Definitions
# ============================================================================


class WaterFluxType(NamedTuple):
    """Water flux variables.

    Translated from Fortran WaterFluxType module (lines 1-33).

    Attributes:
        qflx_evap_tot_patch: Total evaporation flux at patch level
            (qflx_evap_soi + qflx_evap_veg + qflx_tran_veg) [kg H2O/m²/s]
            Shape: (n_patches,)

    Note:
        In the full CTSM implementation, this type would contain many more
        flux variables. This translation includes only qflx_evap_tot_patch
        as shown in the provided Fortran source excerpt.
    """

    qflx_evap_tot_patch: Array  # (n_patches,) [kg H2O/m²/s]


class BoundsType(NamedTuple):
    """Domain decomposition bounds.

    Corresponds to bounds_type from decompMod in Fortran.

    Attributes:
        begp: Beginning patch index (inclusive)
        endp: Ending patch index (inclusive)

    Note:
        Fortran uses 1-based inclusive indexing. In JAX, we maintain
        the same convention for compatibility, but array allocation
        uses 0-based indexing internally.
    """

    begp: int
    endp: int


# ============================================================================
# Initialization Functions
# ============================================================================


def init_waterflux_type(n_patches: int) -> WaterFluxType:
    """Initialize WaterFluxType with default values.

    Corresponds to Init and InitAllocate procedures in Fortran (lines 24-25).

    Args:
        n_patches: Number of patches in the domain

    Returns:
        WaterFluxType: Initialized water flux state with NaN values

    Note:
        Fortran reference: WaterFluxType.F90, lines 1-33
        Arrays are initialized to NaN following Fortran convention (nan => spval)
        This is a convenience function for simple initialization without bounds.
    """
    return WaterFluxType(qflx_evap_tot_patch=jnp.full(n_patches, jnp.nan, dtype=jnp.float64))


def init(waterflux_state: WaterFluxType, bounds: BoundsType) -> WaterFluxType:
    """
    Initialize water flux state by allocating arrays.

    Translated from Fortran source: WaterFluxType.F90, lines 34-41

    This is the main initialization routine that delegates to init_allocate
    to set up all water flux arrays with proper dimensions based on bounds.

    Args:
        waterflux_state: Current water flux state (may be uninitialized)
        bounds: Domain bounds containing grid dimensions

    Returns:
        WaterFluxType: Initialized water flux state with allocated arrays

    Note:
        In the Fortran version, this is a class method that calls InitAllocate.
        In JAX, we maintain functional purity by returning a new state.
        Fortran line 38: call this%InitAllocate(bounds)
    """
    # Delegate to init_allocate to perform the actual allocation
    return init_allocate(waterflux_state, bounds)


def init_allocate(waterflux_state: WaterFluxType, bounds: BoundsType) -> WaterFluxType:
    """
    Initialize water flux state structure by allocating arrays.

    Translated from Fortran source: WaterFluxType.F90, lines 44-61

    This function allocates and initializes the qflx_evap_tot_patch array
    with NaN values for the patch dimension range specified by bounds.

    Args:
        waterflux_state: Current water flux state (may be empty/uninitialized)
        bounds: Bounds type containing patch index ranges (begp, endp)

    Returns:
        WaterFluxType: New state with allocated and initialized qflx_evap_tot_patch array

    Note:
        - Fortran lines 57-58: begp = bounds%begp ; endp = bounds%endp
        - Fortran line 60: allocate and initialize qflx_evap_tot_patch with nan
        - Array is allocated from begp:endp (inclusive in Fortran)
        - In JAX, we use 0-based indexing, so size is (endp - begp + 1)
    """
    # Extract patch bounds (Fortran line 57-58)
    begp = bounds.begp
    endp = bounds.endp

    # Calculate array size (Fortran uses inclusive ranges)
    n_patches = endp - begp + 1

    # Allocate and initialize qflx_evap_tot_patch with NaN (Fortran line 60)
    qflx_evap_tot_patch = jnp.full(n_patches, jnp.nan, dtype=jnp.float64)

    # Return new state with allocated array
    return waterflux_state._replace(qflx_evap_tot_patch=qflx_evap_tot_patch)


# ============================================================================
# Update Functions
# ============================================================================


def update_qflx_evap_tot_patch(
    waterflux: WaterFluxType, qflx_evap_tot_patch: Array
) -> WaterFluxType:
    """Update total evaporation flux.

    Args:
        waterflux: Current water flux state
        qflx_evap_tot_patch: New total evaporation flux values [kg H2O/m²/s]
            Shape: (n_patches,)

    Returns:
        WaterFluxType: Updated water flux state

    Note:
        Fortran reference: WaterFluxType.F90, line 20
        This is a pure functional update preserving immutability.
        In Fortran, this would be a direct assignment to the member variable.
    """
    return waterflux._replace(qflx_evap_tot_patch=qflx_evap_tot_patch)


# Backward compatibility alias (Fortran naming convention)
waterflux_type = WaterFluxType
