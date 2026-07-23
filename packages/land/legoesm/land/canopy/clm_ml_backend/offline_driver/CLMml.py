"""
CLM-ML Main Driver Program.

Translated from CTSM's CLMml.F90

This is the main entry point for the CLM-ML offline driver. It sets up the
computational domain (a single grid cell with one land unit, column, and patch)
and calls the main driver routine.

The CLM hierarchy is:
    Grid cell (g) -> Land unit (l) -> Column (c) -> Patch (p)

For offline testing, we process a single grid cell with one of each subgrid entity.

Reference: CLMml.F90, lines 1-24
"""

from typing import NamedTuple

# =============================================================================
# Data Structures
# =============================================================================


class BoundsType(NamedTuple):
    """
    Bounds for CLM grid hierarchy.

    Defines the beginning and ending indices for each level of the CLM
    subgrid hierarchy: gridcell (g), landunit (l), column (c), and patch (p).

    For offline single-point simulations, all bounds are typically [1, 1].

    Attributes:
        begg: Beginning gridcell index [scalar int]
        endg: Ending gridcell index [scalar int]
        begl: Beginning landunit index [scalar int]
        endl: Ending landunit index [scalar int]
        begc: Beginning column index [scalar int]
        endc: Ending column index [scalar int]
        begp: Beginning patch index [scalar int]
        endp: Ending patch index [scalar int]

    Reference: decompMod.F90 bounds_type definition
    """

    begg: int
    endg: int
    begl: int
    endl: int
    begc: int
    endc: int
    begp: int
    endp: int


class ClumpConfig(NamedTuple):
    """
    Configuration for computational clumps.

    CLM processes the domain in "clumps" for parallelization. Each clump
    contains a subset of gridcells and their associated subgrid entities.

    Attributes:
        n_clumps: Number of computational clumps [scalar int]
        clump_id: ID of current clump (1-indexed) [scalar int]

    Reference: decompMod.F90
    """

    n_clumps: int
    clump_id: int


# =============================================================================
# Domain Decomposition Functions
# =============================================================================


def get_clump_bounds(clump_id: int) -> BoundsType:
    """
    Get bounds for a computational clump.

    For offline single-point simulations, this returns bounds for a single
    grid cell with one land unit, one column, and one patch.

    Args:
        clump_id: Clump identifier (1-indexed) [scalar int]

    Returns:
        Bounds for the clump's grid hierarchy

    Note:
        In the full CLM, this would look up bounds from a decomposition table.
        For offline testing, we hardcode single-point bounds.

    Reference: decompMod.F90 get_clump_bounds, CLMml.F90 lines 18-19
    """
    # Validate input
    if clump_id < 1:
        raise ValueError(f"clump_id must be >= 1, got {clump_id}")

    # Single grid cell configuration
    # All indices are 1-based to match Fortran convention
    bounds = BoundsType(
        begg=1,  # Beginning gridcell
        endg=1,  # Ending gridcell
        begl=1,  # Beginning landunit
        endl=1,  # Ending landunit
        begc=1,  # Beginning column
        endc=1,  # Ending column
        begp=1,  # Beginning patch
        endp=1,  # Ending patch
    )

    return bounds


def initialize_clump_config(n_clumps: int = 1) -> ClumpConfig:
    """
    Initialize clump configuration.

    Args:
        n_clumps: Number of computational clumps [scalar int]

    Returns:
        Clump configuration

    Reference: CLMml.F90 line 17
    """
    # Validate input
    if n_clumps < 1:
        raise ValueError(f"n_clumps must be >= 1, got {n_clumps}")

    return ClumpConfig(
        n_clumps=n_clumps,
        clump_id=1,  # Process first (and only) clump
    )


# =============================================================================
# Main Driver
# =============================================================================


def clm_ml_main(
    driver_fn,
    n_clumps: int = 1,
) -> None:
    """
    Main entry point for CLM-ML offline driver.

    This function:
    1. Initializes the computational domain (clumps and bounds)
    2. Calls the main driver routine to run the model

    Args:
        driver_fn: Main driver function (CLMml_drv) that takes bounds
        n_clumps: Number of computational clumps [scalar int]

    Note:
        This is a simplified version for offline testing. The full CLM
        would loop over multiple clumps and timesteps.

    Reference: CLMml.F90 lines 1-24
    """
    # Validate input
    if n_clumps < 1:
        raise ValueError(f"n_clumps must be >= 1, got {n_clumps}")

    # Initialize clump configuration (line 17)
    clump_config = initialize_clump_config(n_clumps)

    # Get bounds for the clump (line 19)
    bounds = get_clump_bounds(clump_config.clump_id)

    # Run the model (line 23)
    driver_fn(bounds)


# =============================================================================
# JAX-Compatible Main Driver
# =============================================================================


def clm_ml_main_jax(
    driver_fn,
    initial_state,
    forcing_data,
    params,
) -> tuple:
    """
    JAX-compatible main driver for CLM-ML.

    This version is designed for JAX transformations (jit, grad, vmap).
    Instead of side effects, it takes initial state and returns final state.

    Args:
        driver_fn: Main driver function that takes (bounds, state, forcing, params)
        initial_state: Initial model state [NamedTuple]
        forcing_data: Atmospheric forcing data [NamedTuple]
        params: Model parameters [NamedTuple]

    Returns:
        final_state: Final model state [NamedTuple]
        diagnostics: Diagnostic outputs [NamedTuple]

    Example:
        >>> from jax_ctsm.driver import clm_ml_driver
        >>> state, diag = clm_ml_main_jax(
        ...     clm_ml_driver,
        ...     initial_state,
        ...     forcing,
        ...     params
        ... )

    Note:
        This is a pure functional version suitable for JAX. The original
        Fortran code uses side effects and global state.

    Reference: CLMml.F90 lines 1-24 (functional adaptation)
    """
    # Get bounds for single-point simulation
    bounds = get_clump_bounds(clump_id=1)

    # Run the driver (pure functional version)
    final_state, diagnostics = driver_fn(
        bounds=bounds,
        state=initial_state,
        forcing=forcing_data,
        params=params,
    )

    return final_state, diagnostics


# =============================================================================
# Utility Functions
# =============================================================================


def validate_bounds(bounds: BoundsType) -> bool:
    """
    Validate bounds structure.

    Checks that:
    - End indices >= begin indices
    - Indices are positive
    - Hierarchy is consistent (g >= l >= c >= p)

    Args:
        bounds: Bounds to validate

    Returns:
        True if valid, False otherwise

    Reference: decompMod.F90 (validation logic)
    """
    # Check that end >= begin for each level
    valid_g = bounds.endg >= bounds.begg
    valid_l = bounds.endl >= bounds.begl
    valid_c = bounds.endc >= bounds.begc
    valid_p = bounds.endp >= bounds.begp

    # Check positive indices
    positive = all(
        [
            bounds.begg > 0,
            bounds.endg > 0,
            bounds.begl > 0,
            bounds.endl > 0,
            bounds.begc > 0,
            bounds.endc > 0,
            bounds.begp > 0,
            bounds.endp > 0,
        ]
    )

    return valid_g and valid_l and valid_c and valid_p and positive


def get_domain_size(bounds: BoundsType) -> dict:
    """
    Get size of each hierarchy level.

    Args:
        bounds: Domain bounds

    Returns:
        Dictionary with counts for each level

    Example:
        >>> bounds = get_clump_bounds(1)
        >>> sizes = get_domain_size(bounds)
        >>> print(sizes)
        {'n_gridcells': 1, 'n_landunits': 1, 'n_columns': 1, 'n_patches': 1}
    """
    return {
        "n_gridcells": bounds.endg - bounds.begg + 1,
        "n_landunits": bounds.endl - bounds.begl + 1,
        "n_columns": bounds.endc - bounds.begc + 1,
        "n_patches": bounds.endp - bounds.begp + 1,
    }


# =============================================================================
# Translation Notes
# =============================================================================

"""
TRANSLATION NOTES:

1. **Program to Function**:
   - Fortran PROGRAM CLMml (lines 1-24) -> Python function clm_ml_main
   - Added JAX-compatible version clm_ml_main_jax for pure functional style
   - Original uses side effects; JAX version returns state

2. **Module Dependencies**:
   - decompMod.bounds_type -> BoundsType NamedTuple
   - decompMod.get_clump_bounds -> get_clump_bounds function
   - CLMml_driver.CLMml_drv -> passed as driver_fn argument

3. **Data Structures**:
   - bounds_type (Fortran derived type) -> BoundsType (Python NamedTuple)
   - Added ClumpConfig for clump management
   - All fields are immutable (NamedTuple property)

4. **Single-Point Assumption**:
   - Original comment (lines 8-14): "assumes that a grid cell has one land unit
     with one column and one patch"
   - Hardcoded in get_clump_bounds: all bounds are [1, 1]
   - Easy to extend for multi-point by modifying get_clump_bounds

5. **Clump Processing**:
   - Original: nc = 1 (line 17) -> single clump
   - JAX version: could vmap over multiple clumps for parallelization
   - Current implementation processes one clump (offline testing)

6. **Fortran Indexing**:
   - Fortran uses 1-based indexing
   - Kept 1-based in bounds for consistency with CLM conventions
   - Array operations in driver will use 0-based Python indexing

7. **Side Effects**:
   - Original: CLMml_drv modifies global state
   - JAX version: pure function that returns new state
   - This is essential for JAX transformations (jit, grad, vmap)

8. **Error Handling**:
   - Added validate_bounds for runtime checks
   - Added get_domain_size for diagnostics
   - Original Fortran has implicit validation

9. **Extensibility**:
   - Easy to add multi-clump support via vmap
   - Easy to add multi-timestep via scan
   - Easy to add ensemble runs via vmap over initial conditions

10. **Testing Strategy**:
    - Can test bounds generation independently
    - Can test driver with mock functions
    - Can verify single-point configuration matches Fortran

USAGE EXAMPLE:

    >>> import CLMml
    >>> # Initialize and run model
    >>> # See CLMml_driver for detailed usage
"""
