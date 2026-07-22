"""
WaterStateType module - Water state variables for CLM.

Translated from Fortran source: WaterStateType.F90 (lines 1-80)

This module defines water state variables and initialization functions for the
Community Land Model (CLM). It manages snow water, soil moisture, ice content,
and related hydrological state variables.

Key components:
- WaterState: NamedTuple containing all water state variables
- init_waterstate: Initialize water state with default values
- init_allocate_water_state: Allocate arrays based on domain bounds
- update_waterstate: Functional update helper

Translation notes:
- Fortran arrays with negative indices (-nlevsno+1:0) mapped to 0-based indexing
- All state is immutable using NamedTuples
- Pure functional approach replaces Fortran's object-oriented style
"""

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

# ============================================================================
# Type Definitions
# ============================================================================


class Bounds(NamedTuple):
    """Domain decomposition bounds.

    Corresponds to bounds_type from decompMod in Fortran.

    Attributes:
        begp: Beginning patch index
        endp: Ending patch index
        begc: Beginning column index
        endc: Ending column index
        begg: Beginning gridcell index (optional)
        endg: Ending gridcell index (optional)
    """

    begp: int
    endp: int
    begc: int
    endc: int
    begg: int = 0
    endg: int = 0


class WaterState(NamedTuple):
    """Water state variables for CLM.

    Translated from Fortran type waterstate_type (lines 20-38).

    Attributes:
        bw_col: Partial density of water in snow pack (ice + liquid) [kg/m3].
            Shape: (ncols, nlevsno) - Fortran line 23
        h2osno_col: Snow water [mm H2O].
            Shape: (ncols,) - Fortran line 24
        h2osoi_liq_col: Liquid water [kg H2O/m2].
            Shape: (ncols, nlevsno + nlevgrnd) - Fortran line 25
            Index range: -nlevsno+1:nlevgrnd in Fortran
        h2osoi_ice_col: Ice lens [kg H2O/m2].
            Shape: (ncols, nlevsno + nlevgrnd) - Fortran line 26
            Index range: -nlevsno+1:nlevgrnd in Fortran
        h2osoi_vol_col: Volumetric soil water (0<=h2osoi_vol<=watsat) [m3/m3].
            Shape: (ncols, nlevgrnd) - Fortran line 27
        h2osfc_col: Surface water [mm H2O].
            Shape: (ncols,) - Fortran line 28
        q_ref2m_patch: 2 m height surface specific humidity [kg/kg].
            Shape: (npatches,) - Fortran line 29
        frac_sno_eff_col: Fraction of ground covered by snow (0 to 1).
            Shape: (ncols,) - Fortran line 30
    """

    bw_col: Array  # (ncols, nlevsno)
    h2osno_col: Array  # (ncols,)
    h2osoi_liq_col: Array  # (ncols, nlevsno + nlevgrnd)
    h2osoi_ice_col: Array  # (ncols, nlevsno + nlevgrnd)
    h2osoi_vol_col: Array  # (ncols, nlevgrnd)
    h2osfc_col: Array  # (ncols,)
    q_ref2m_patch: Array  # (npatches,)
    frac_sno_eff_col: Array  # (ncols,)


# ============================================================================
# Initialization Functions
# ============================================================================


def init_waterstate(
    ncols: int, npatches: int, nlevgrnd: int, nlevsno: int, fill_value: float = 0.0
) -> WaterState:
    """Initialize WaterState with default values.

    Corresponds to InitAllocate procedure in Fortran (lines 32-33).

    Args:
        ncols: Number of columns
        npatches: Number of patches
        nlevgrnd: Number of ground levels
        nlevsno: Number of snow levels
        fill_value: Initial value for all arrays (default: 0.0)

    Returns:
        Initialized WaterState instance

    Note:
        In Fortran, arrays are allocated in InitAllocate procedure.
        Here we create arrays with specified dimensions and fill value.
    """
    # Total vertical levels for soil layers (snow + ground)
    nlevtot = nlevsno + nlevgrnd

    return WaterState(
        bw_col=jnp.full((ncols, nlevsno), fill_value, dtype=jnp.float64),
        h2osno_col=jnp.full((ncols,), fill_value, dtype=jnp.float64),
        h2osoi_liq_col=jnp.full((ncols, nlevtot), fill_value, dtype=jnp.float64),
        h2osoi_ice_col=jnp.full((ncols, nlevtot), fill_value, dtype=jnp.float64),
        h2osoi_vol_col=jnp.full((ncols, nlevgrnd), fill_value, dtype=jnp.float64),
        h2osfc_col=jnp.full((ncols,), fill_value, dtype=jnp.float64),
        q_ref2m_patch=jnp.full((npatches,), fill_value, dtype=jnp.float64),
        frac_sno_eff_col=jnp.full((ncols,), fill_value, dtype=jnp.float64),
    )


def init_allocate_water_state(
    bounds: Bounds, nlevsno: int, nlevgrnd: int, use_nan: bool = True
) -> WaterState:
    """Initialize water state data structure with allocated arrays.

    Allocates and initializes all water state arrays. Corresponds to the
    InitAllocate subroutine in Fortran (lines 52-78).

    Args:
        bounds: Bounds containing patch and column index ranges
        nlevsno: Number of snow layers (positive value, arrays indexed from -nlevsno+1:0)
        nlevgrnd: Number of ground layers
        use_nan: If True, initialize to NaN; otherwise use 0.0 (default: True)

    Returns:
        WaterState: Initialized water state structure

    Note:
        - Fortran arrays with negative indices (e.g., -nlevsno+1:0) are mapped to
          0-based indexing in JAX (e.g., shape (nlevsno,))
        - Default initialization to NaN matches Fortran behavior (lines 69-75)
        - Fortran dimension (-nlevsno+1:nlevgrnd) has size nlevsno + nlevgrnd
    """
    # Extract bounds (lines 67-68)
    begp = bounds.begp
    endp = bounds.endp
    begc = bounds.begc
    endc = bounds.endc

    # Calculate array sizes (Fortran-style: bounds are inclusive)
    num_cols = endc - begc + 1
    num_patches = endp - begp + 1

    # Set fill value
    fill_value = jnp.nan if use_nan else 0.0

    # Allocate and initialize arrays (lines 69-75)
    # Note: Fortran dimension (-nlevsno+1:0) has size nlevsno
    # Note: Fortran dimension (-nlevsno+1:nlevgrnd) has size nlevsno + nlevgrnd
    bw_col = jnp.full((num_cols, nlevsno), fill_value, dtype=jnp.float64)
    h2osno_col = jnp.full((num_cols,), fill_value, dtype=jnp.float64)
    h2osoi_liq_col = jnp.full((num_cols, nlevsno + nlevgrnd), fill_value, dtype=jnp.float64)
    h2osoi_ice_col = jnp.full((num_cols, nlevsno + nlevgrnd), fill_value, dtype=jnp.float64)
    h2osoi_vol_col = jnp.full((num_cols, nlevgrnd), fill_value, dtype=jnp.float64)
    h2osfc_col = jnp.full((num_cols,), fill_value, dtype=jnp.float64)
    q_ref2m_patch = jnp.full((num_patches,), fill_value, dtype=jnp.float64)
    frac_sno_eff_col = jnp.full((num_cols,), fill_value, dtype=jnp.float64)

    return WaterState(
        bw_col=bw_col,
        h2osno_col=h2osno_col,
        h2osoi_liq_col=h2osoi_liq_col,
        h2osoi_ice_col=h2osoi_ice_col,
        h2osoi_vol_col=h2osoi_vol_col,
        h2osfc_col=h2osfc_col,
        q_ref2m_patch=q_ref2m_patch,
        frac_sno_eff_col=frac_sno_eff_col,
    )


def init_water_state_from_bounds(bounds: Bounds, nlevsno: int, nlevgrnd: int) -> WaterState:
    """Initialize water state variables by allocating arrays.

    This function corresponds to the Init subroutine in the Fortran code
    (lines 42-49), which calls InitAllocate to set up the water state structure.

    Args:
        bounds: Bounds object containing domain decomposition information
               (begc, endc, begp, endp, etc.)
        nlevsno: Number of snow layers
        nlevgrnd: Number of ground layers

    Returns:
        WaterState: Initialized water state NamedTuple with allocated arrays

    Note:
        This is a pure functional version that returns a new WaterState
        instead of modifying 'this' in place.
    """
    return init_allocate_water_state(bounds, nlevsno, nlevgrnd, use_nan=False)


# ============================================================================
# Update Functions
# ============================================================================


def update_waterstate(state: WaterState, **updates) -> WaterState:
    """Update WaterState with new values.

    Provides a functional interface for updating water state fields.

    Args:
        state: Current WaterState
        **updates: Keyword arguments for fields to update

    Returns:
        New WaterState with updated fields

    Example:
        >>> new_state = update_waterstate(state, h2osno_col=new_snow_water)
        >>> new_state = update_waterstate(
        ...     state,
        ...     h2osno_col=new_snow,
        ...     frac_sno_eff_col=new_frac
        ... )
    """
    return state._replace(**updates)


# ============================================================================
# Utility Functions
# ============================================================================


def get_total_water_mass(state: WaterState) -> Array:
    """Calculate total water mass (liquid + ice) for each column.

    Args:
        state: Current WaterState

    Returns:
        Array of total water mass per column [kg/m2]
        Shape: (ncols,)
    """
    # Sum liquid and ice over all levels
    total_liq = jnp.sum(state.h2osoi_liq_col, axis=1)
    total_ice = jnp.sum(state.h2osoi_ice_col, axis=1)
    return total_liq + total_ice + state.h2osno_col + state.h2osfc_col


def get_soil_water_mass(state: WaterState, nlevsno: int) -> Array:
    """Calculate soil water mass (excluding snow layers) for each column.

    Args:
        state: Current WaterState
        nlevsno: Number of snow layers

    Returns:
        Array of soil water mass per column [kg/m2]
        Shape: (ncols,)
    """
    # Extract soil layers only (skip snow layers)
    soil_liq = state.h2osoi_liq_col[:, nlevsno:]
    soil_ice = state.h2osoi_ice_col[:, nlevsno:]
    return jnp.sum(soil_liq, axis=1) + jnp.sum(soil_ice, axis=1)


# Backward compatibility alias (Fortran naming convention)
waterstate_type = WaterState
