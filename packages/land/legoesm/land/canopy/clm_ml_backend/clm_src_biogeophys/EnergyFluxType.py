"""
EnergyFluxType module translation.

Translated from Fortran source: EnergyFluxType.F90 (lines 1-73)
Defines energy flux variables and initialization routines for CTSM.

This module manages energy flux state variables at the patch level, including:
- Sensible and latent heat fluxes
- Outgoing longwave radiation
- Wind stress components

Original Fortran dependencies:
- shr_kind_mod: r8 => shr_kind_r8
- clm_varcon: ispval, nan => spval
- decompMod: bounds_type
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

# =============================================================================
# Type Definitions
# =============================================================================


class BoundsType(NamedTuple):
    """Domain decomposition bounds.

    Translated from decompMod::bounds_type.

    Attributes:
        begp: Beginning patch index
        endp: Ending patch index
        begc: Beginning column index
        endc: Ending column index
        begg: Beginning gridcell index
        endg: Ending gridcell index
    """

    begp: int
    endp: int
    begc: int
    endc: int
    begg: int
    endg: int


class EnergyFluxState(NamedTuple):
    """Energy flux variables.

    Translated from Fortran type energyflux_type (lines 18-34).
    All arrays are indexed by patch.

    Attributes:
        eflx_sh_tot_patch: Total sensible heat flux (W/m2) [+ to atm]
            Shape: (n_patches,)
        eflx_lh_tot_patch: Total latent heat flux (W/m2) [+ to atm]
            Shape: (n_patches,)
        eflx_lwrad_out_patch: Emitted infrared (longwave) radiation (W/m2)
            Shape: (n_patches,)
        taux_patch: Wind (shear) stress: e-w (kg/m/s**2)
            Shape: (n_patches,)
        tauy_patch: Wind (shear) stress: n-s (kg/m/s**2)
            Shape: (n_patches,)
    """

    eflx_sh_tot_patch: Array  # (n_patches,)
    eflx_lh_tot_patch: Array  # (n_patches,)
    eflx_lwrad_out_patch: Array  # (n_patches,)
    taux_patch: Array  # (n_patches,)
    tauy_patch: Array  # (n_patches,)


# =============================================================================
# Initialization Functions
# =============================================================================


def init_allocate(bounds: BoundsType) -> EnergyFluxState:
    """Initialize and allocate energy flux data structure.

    Allocates and initializes energy flux arrays with NaN values
    for all patches in the specified bounds.

    Translated from Fortran lines 50-71 in EnergyFluxType.F90

    Args:
        bounds: Patch bounds containing begp and endp indices

    Returns:
        EnergyFluxState: Initialized energy flux state with NaN-filled arrays

    Note:
        In Fortran, arrays are allocated from begp:endp. In JAX, we create
        arrays of size (endp - begp + 1) and handle indexing separately.
        All arrays are initialized to NaN as in the original Fortran code
        (lines 65-69).
    """
    # Fortran line 63: begp = bounds%begp ; endp = bounds%endp
    begp = bounds.begp
    endp = bounds.endp

    # Calculate array size (Fortran inclusive indexing)
    n_patches = endp - begp + 1

    # Fortran lines 65-69: allocate arrays and initialize to nan
    # All arrays are initialized to NaN to match Fortran behavior
    eflx_sh_tot_patch = jnp.full(n_patches, jnp.nan, dtype=jnp.float64)
    eflx_lh_tot_patch = jnp.full(n_patches, jnp.nan, dtype=jnp.float64)
    eflx_lwrad_out_patch = jnp.full(n_patches, jnp.nan, dtype=jnp.float64)
    taux_patch = jnp.full(n_patches, jnp.nan, dtype=jnp.float64)
    tauy_patch = jnp.full(n_patches, jnp.nan, dtype=jnp.float64)

    return EnergyFluxState(
        eflx_sh_tot_patch=eflx_sh_tot_patch,
        eflx_lh_tot_patch=eflx_lh_tot_patch,
        eflx_lwrad_out_patch=eflx_lwrad_out_patch,
        taux_patch=taux_patch,
        tauy_patch=tauy_patch,
    )


def init(bounds: BoundsType) -> EnergyFluxState:
    """Initialize energy flux type.

    Fortran source lines 40-47 in EnergyFluxType.F90

    This function initializes the energy flux type by calling init_allocate
    to allocate and initialize all required arrays based on the provided bounds.

    Args:
        bounds: Domain decomposition bounds containing patch, column, and
                gridcell index ranges

    Returns:
        EnergyFluxState: Initialized energy flux state with allocated arrays

    Note:
        This is a wrapper that delegates to init_allocate for the actual
        allocation and initialization work. In the Fortran version, this
        was a class method that called another class method.
    """
    # Fortran line 45: call this%InitAllocate(bounds)
    return init_allocate(bounds)


def init_energyflux_state(n_patches: int) -> EnergyFluxState:
    """Initialize energy flux state with zeros.

    Alternative initialization method that creates zero-filled arrays
    when bounds are not available.

    Corresponds to Init and InitAllocate procedures (lines 31-32).

    Args:
        n_patches: Number of patches in the domain

    Returns:
        EnergyFluxState: Initialized state with all arrays set to zero

    Note:
        Fortran source lines 1-39. This is a convenience function for
        cases where zero initialization is preferred over NaN initialization.
    """
    return EnergyFluxState(
        eflx_sh_tot_patch=jnp.zeros(n_patches, dtype=jnp.float64),
        eflx_lh_tot_patch=jnp.zeros(n_patches, dtype=jnp.float64),
        eflx_lwrad_out_patch=jnp.zeros(n_patches, dtype=jnp.float64),
        taux_patch=jnp.zeros(n_patches, dtype=jnp.float64),
        tauy_patch=jnp.zeros(n_patches, dtype=jnp.float64),
    )


# =============================================================================
# State Update Functions
# =============================================================================


def update_energyflux_state(
    state: EnergyFluxState,
    eflx_sh_tot_patch: Array | None = None,
    eflx_lh_tot_patch: Array | None = None,
    eflx_lwrad_out_patch: Array | None = None,
    taux_patch: Array | None = None,
    tauy_patch: Array | None = None,
) -> EnergyFluxState:
    """Update energy flux state with new values.

    Provides immutable update mechanism for the state following JAX
    functional programming paradigm.

    Args:
        state: Current energy flux state
        eflx_sh_tot_patch: New sensible heat flux values (optional)
        eflx_lh_tot_patch: New latent heat flux values (optional)
        eflx_lwrad_out_patch: New longwave radiation values (optional)
        taux_patch: New e-w wind stress values (optional)
        tauy_patch: New n-s wind stress values (optional)

    Returns:
        EnergyFluxState: Updated state with new values

    Note:
        Fortran source lines 1-39. This function enables functional updates
        to the immutable NamedTuple state.
    """
    return EnergyFluxState(
        eflx_sh_tot_patch=(
            eflx_sh_tot_patch if eflx_sh_tot_patch is not None else state.eflx_sh_tot_patch
        ),
        eflx_lh_tot_patch=(
            eflx_lh_tot_patch if eflx_lh_tot_patch is not None else state.eflx_lh_tot_patch
        ),
        eflx_lwrad_out_patch=(
            eflx_lwrad_out_patch if eflx_lwrad_out_patch is not None else state.eflx_lwrad_out_patch
        ),
        taux_patch=(taux_patch if taux_patch is not None else state.taux_patch),
        tauy_patch=(tauy_patch if tauy_patch is not None else state.tauy_patch),
    )


# Backward compatibility alias (Fortran naming convention)
energyflux_type = EnergyFluxState
