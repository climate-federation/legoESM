"""
JAX translation of shr_file_mod.F90

This module handles file utility functions, specifically managing logical unit numbers
for file I/O operations. In JAX, we implement this as a pure functional state management
system using NamedTuples.

Original Fortran: shr_file_mod.F90, lines 1-111

Key differences from Fortran:
- Pure functional approach with immutable state
- Returns error codes instead of calling endrun()
- JIT-compatible implementations
- No global mutable state

Public Interface:
- create_initial_file_unit_state(): Initialize file unit tracking
- shr_file_get_unit(): Allocate a unit number
- shr_file_free_unit(): Free a unit number
"""

from typing import NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import Array

# =============================================================================
# Constants (Fortran lines 13-14)
# =============================================================================

SHR_FILE_MIN_UNIT: int = 10  # Min unit number to give
SHR_FILE_MAX_UNIT: int = 99  # Max unit number to give


# =============================================================================
# State Types
# =============================================================================


class FileUnitState(NamedTuple):
    """
    State for tracking file unit usage.

    Attributes:
        unit_tag: Boolean array indicating which units are in use (Fortran line 15)
                  Shape: (SHR_FILE_MAX_UNIT + 1,), indexed 0 to SHR_FILE_MAX_UNIT
    """

    unit_tag: Array  # shape: (100,), dtype: bool


class FreeUnitResult(NamedTuple):
    """
    Result of freeing a unit number.

    Attributes:
        state: Updated FileUnitState with freed unit
        error_code: 0 for success, >0 for error
        error_msg: Description of error if any
    """

    state: FileUnitState
    error_code: int
    error_msg: str


# =============================================================================
# Initialization
# =============================================================================


def create_initial_file_unit_state() -> FileUnitState:
    """
    Create initial file unit state with all units marked as free.

    Corresponds to Fortran line 15: UnitTag(0:shr_file_maxUnit) = .false.

    Returns:
        FileUnitState: Initial state with all units available
    """
    return FileUnitState(unit_tag=jnp.zeros(SHR_FILE_MAX_UNIT + 1, dtype=bool))


# =============================================================================
# Unit Allocation
# =============================================================================


def shr_file_get_unit(
    state: FileUnitState, unit: Optional[int] = None
) -> Tuple[FileUnitState, int, bool]:
    """
    Get the next free FORTRAN unit number.

    Corresponds to Fortran lines 24-78 (shr_file_getUnit function).

    This is a pure functional version that returns updated state rather than
    modifying global state. The function finds an available unit number for
    file I/O operations.

    Args:
        state: Current file unit state
        unit: Optional desired unit number (Fortran line 37)

    Returns:
        Tuple containing:
        - Updated FileUnitState with the allocated unit marked as used
        - Allocated unit number (or -1 if allocation failed)
        - Success flag (True if unit was allocated, False otherwise)

    Notes:
        - Units 5 and 6 are reserved (stdin/stdout) (Fortran lines 51, 67)
        - Valid range is SHR_FILE_MIN_UNIT to SHR_FILE_MAX_UNIT (lines 13-14)
        - If unit is specified, validates and allocates that specific unit (lines 41-58)
        - Otherwise, searches from max to min for first available unit (lines 62-74)
    """

    if unit is not None:
        # Use specified unit number (Fortran lines 41-58)

        # Check if unit is in valid range (lines 48-49)
        valid_range = (unit > 0) & (unit <= SHR_FILE_MAX_UNIT)

        # Check if unit is reserved or already in use (lines 50-51)
        is_reserved = (unit == 5) | (unit == 6)
        is_in_use = state.unit_tag[unit]

        # Unit is available if in valid range, not reserved, and not in use
        is_available = valid_range & (~is_reserved) & (~is_in_use)

        # Allocate unit if available
        new_unit_tag = jnp.where(is_available, state.unit_tag.at[unit].set(True), state.unit_tag)

        allocated_unit = jnp.where(is_available, unit, -1)
        success = is_available

        return FileUnitState(unit_tag=new_unit_tag), int(allocated_unit), bool(success)

    else:
        # Choose first available unit (Fortran lines 62-74)
        # Search from max to min (line 64)

        # Create array of potential units in reverse order
        units = jnp.arange(SHR_FILE_MIN_UNIT, SHR_FILE_MAX_UNIT + 1)[::-1]

        # Check which units are available
        # Not reserved (not 5 or 6) and not in use (lines 66-67, 70)
        is_reserved = (units == 5) | (units == 6)
        is_in_use = state.unit_tag[units]
        is_available = (~is_reserved) & (~is_in_use)

        # Find first available unit
        # Use argmax to find first True value (returns 0 if none found)
        first_available_idx = jnp.argmax(is_available)
        has_available = is_available[first_available_idx]

        allocated_unit = jnp.where(has_available, units[first_available_idx], -1)

        # Update state if unit was allocated
        new_unit_tag = jnp.where(
            has_available, state.unit_tag.at[allocated_unit].set(True), state.unit_tag
        )

        return FileUnitState(unit_tag=new_unit_tag), int(allocated_unit), bool(has_available)


# =============================================================================
# Unit Deallocation
# =============================================================================


def shr_file_free_unit(state: FileUnitState, unit: int) -> FreeUnitResult:
    """
    Free up the given unit number.

    Fortran source: lines 81-109 in shr_file_mod.F90

    This function validates the unit number and marks it as free in the unit_tag array.
    Reserved units (0, 5, 6) and invalid unit numbers are rejected.

    Args:
        state: Current FileUnitState containing unit_tag array
        unit: Unit number to be freed (must be in valid range and not reserved)

    Returns:
        FreeUnitResult containing:
            - Updated state with freed unit
            - error_code: 0 if successful, 1 if invalid unit, 2 if reserved unit,
                         3 if unit was not in use
            - error_msg: Description of any error

    Note:
        In the original Fortran, errors call endrun() to abort execution.
        In JAX, we return error codes to maintain pure function semantics.
        The caller should check error_code and handle appropriately.

    Fortran line references:
        - Lines 95-97: Invalid unit number check
        - Lines 98-100: Reserved unit check
        - Lines 101-104: Unit in use check
        - Line 107: Free the unit
    """
    # Check for invalid unit number (lines 95-97)
    invalid_range = (unit < 0) | (unit > SHR_FILE_MAX_UNIT)

    # Check for reserved units (lines 98-100)
    reserved_unit = (unit == 0) | (unit == 5) | (unit == 6)

    # Check if unit was in use (lines 101-104)
    unit_in_use = jnp.where((unit >= 0) & (unit <= SHR_FILE_MAX_UNIT), state.unit_tag[unit], False)

    # Determine error code
    # Priority: invalid_range > reserved_unit > not_in_use
    error_code = jnp.where(
        invalid_range, 1, jnp.where(reserved_unit, 2, jnp.where(~unit_in_use, 3, 0))
    )

    # Generate error message
    error_msg = jnp.where(
        error_code == 1,
        f"invalid unit number request: {unit}",
        jnp.where(
            error_code == 2,
            "Error: units 0, 5, and 6 must not be freed",
            jnp.where(error_code == 3, f"unit {unit} was not in use", ""),
        ),
    )

    # Update unit_tag array - only if no error and unit was in use (line 107)
    new_unit_tag = jnp.where(
        (error_code == 0) & (jnp.arange(len(state.unit_tag)) == unit), False, state.unit_tag
    )

    new_state = FileUnitState(unit_tag=new_unit_tag)

    return FreeUnitResult(state=new_state, error_code=int(error_code), error_msg=str(error_msg))


# =============================================================================
# JIT-compiled versions for performance
# =============================================================================

shr_file_get_unit_jit = jax.jit(shr_file_get_unit, static_argnames=["unit"])
shr_file_free_unit_jit = jax.jit(shr_file_free_unit)
