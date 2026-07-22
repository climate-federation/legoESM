"""
Canopy state variables module.

Translated from CanopyStateType.F90 (lines 1-66).

This module defines the canopy state data structure and initialization routines
for the Community Terrestrial Systems Model (CTSM). It manages vegetation canopy
properties including leaf/stem area indices and canopy height.

Key components:
    - CanopyState: Immutable state container (NamedTuple)
    - init_allocate: Primary initialization function
    - create_canopy_state: Convenience factory for zero-initialization

Original Fortran dependencies:
    - shr_kind_mod (r8 precision)
    - clm_varcon (ispval, spval/nan)
    - decompMod (bounds_type)

Note:
    This module requires JAX to be installed. Install with:
    pip install jax jaxlib

    For CPU-only installation:
    pip install "jax[cpu]"

    For CUDA support:
    pip install "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
"""

from typing import NamedTuple

try:
    import jax.numpy as jnp
    from jax import Array
except ImportError as e:
    raise ImportError(
        "JAX is required for this module. Install with: pip install jax jaxlib\n"
        "For CPU-only: pip install 'jax[cpu]'\n"
        "For CUDA: pip install 'jax[cuda12]' -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"
    ) from e


# ============================================================================
# Type Definitions
# ============================================================================


class Bounds(NamedTuple):
    """Domain decomposition bounds.

    Corresponds to Fortran bounds_type from decompMod.

    Attributes:
        begp: Beginning patch index (0-based in JAX)
        endp: Ending patch index (0-based, inclusive)
        begc: Beginning column index
        endc: Ending column index
        begg: Beginning grid cell index
        endg: Ending grid cell index
    """

    begp: int
    endp: int
    begc: int
    endc: int
    begg: int
    endg: int


class CanopyState(NamedTuple):
    """Canopy state variables.

    Translated from Fortran type canopystate_type (lines 18-30).

    All arrays are indexed by patch, with shape (n_patches,).

    Attributes:
        frac_veg_nosno_patch: Fraction of vegetation not covered by snow.
            Binary indicator (0 or 1) in original Fortran, but stored as float
            for JIT compatibility. Dimensionless [-]. Line 20.
        elai_patch: Exposed (one-sided) leaf area index with burying by snow.
            Units: [m^2/m^2]. Line 21.
        esai_patch: Exposed (one-sided) stem area index with burying by snow.
            Units: [m^2/m^2]. Line 22.
        htop_patch: Canopy top height above ground surface.
            Units: [m]. Line 23.

    Note:
        In the original Fortran, frac_veg_nosno_patch is integer (ispval).
        We use float32 for uniformity and to avoid JIT complications with
        mixed dtypes. Values should still be 0.0 or 1.0.
    """

    frac_veg_nosno_patch: Array  # shape: (n_patches,), dtype: float32
    elai_patch: Array  # shape: (n_patches,), dtype: float32
    esai_patch: Array  # shape: (n_patches,), dtype: float32
    htop_patch: Array  # shape: (n_patches,), dtype: float32


# ============================================================================
# Initialization Functions
# ============================================================================


def init_allocate(begp: int, endp: int, use_nan: bool = True) -> CanopyState:
    """Initialize canopy state with allocation.

    Allocates and initializes all canopy state arrays for the specified
    patch range. This is the primary initialization function corresponding
    to the Fortran InitAllocate subroutine (lines 46-66).

    Args:
        begp: Beginning patch index (0-based)
        endp: Ending patch index (0-based, inclusive)
        use_nan: If True, initialize with NaN (Fortran default behavior).
                 If False, initialize with zeros.

    Returns:
        CanopyState: Initialized canopy state structure

    Note:
        Fortran source lines 46-66.
        Original Fortran allocates arrays from begp:endp (1-based) and
        initializes to nan (spval). In JAX, we create 0-based arrays of
        size (endp - begp + 1).

    Example:
        >>> state = init_allocate(begp=0, endp=99)  # 100 patches
        >>> state.elai_patch.shape
        (100,)
    """
    # Calculate array size (line 58 equivalent)
    n_patches = endp - begp + 1

    # Choose initialization value
    init_val = jnp.nan if use_nan else 0.0

    # Initialize all arrays (lines 60-63)
    # Note: Using float32 for memory efficiency and typical precision needs
    frac_veg_nosno_patch = jnp.full(n_patches, init_val, dtype=jnp.float32)
    elai_patch = jnp.full(n_patches, init_val, dtype=jnp.float32)
    esai_patch = jnp.full(n_patches, init_val, dtype=jnp.float32)
    htop_patch = jnp.full(n_patches, init_val, dtype=jnp.float32)

    return CanopyState(
        frac_veg_nosno_patch=frac_veg_nosno_patch,
        elai_patch=elai_patch,
        esai_patch=esai_patch,
        htop_patch=htop_patch,
    )


def init_allocate_from_bounds(bounds: Bounds, use_nan: bool = True) -> CanopyState:
    """Initialize canopy state from bounds object.

    Convenience wrapper around init_allocate that extracts patch bounds
    from a Bounds object. Corresponds to the Fortran Init subroutine
    (lines 36-43) which calls InitAllocate.

    Args:
        bounds: Bounds object containing domain decomposition info
        use_nan: If True, initialize with NaN (default Fortran behavior)

    Returns:
        CanopyState: Initialized canopy state structure

    Note:
        Fortran source lines 36-43.
        This function provides the interface expected by the original
        Fortran Init subroutine.

    Example:
        >>> bounds = Bounds(begp=0, endp=99, begc=0, endc=49, begg=0, endg=9)
        >>> state = init_allocate_from_bounds(bounds)
    """
    return init_allocate(begp=bounds.begp, endp=bounds.endp, use_nan=use_nan)


def create_canopy_state(n_patches: int, use_zeros: bool = True) -> CanopyState:
    """Create canopy state with simple initialization.

    Factory function for creating a canopy state with a specified number
    of patches. Provides a simpler interface than init_allocate when
    bounds information is not available.

    Args:
        n_patches: Number of patches in the domain
        use_zeros: If True, initialize with zeros (default).
                   If False, initialize with NaN.

    Returns:
        CanopyState: Initialized canopy state with zeros or NaN

    Note:
        This function was introduced in the initial translation (unit 001)
        as a convenience method. It's equivalent to calling init_allocate
        with begp=0, endp=n_patches-1.

    Example:
        >>> state = create_canopy_state(n_patches=100)
        >>> jnp.all(state.elai_patch == 0.0)
        Array(True, dtype=bool)
    """
    init_val = 0.0 if use_zeros else jnp.nan

    return CanopyState(
        frac_veg_nosno_patch=jnp.full(n_patches, init_val, dtype=jnp.float32),
        elai_patch=jnp.full(n_patches, init_val, dtype=jnp.float32),
        esai_patch=jnp.full(n_patches, init_val, dtype=jnp.float32),
        htop_patch=jnp.full(n_patches, init_val, dtype=jnp.float32),
    )


# ============================================================================
# Utility Functions
# ============================================================================


def update_canopy_state(state: CanopyState, **updates) -> CanopyState:
    """Update canopy state with new values.

    Since CanopyState is immutable (NamedTuple), this function creates
    a new state with updated fields.

    Args:
        state: Current canopy state
        **updates: Keyword arguments for fields to update

    Returns:
        CanopyState: New state with updated values

    Example:
        >>> state = create_canopy_state(n_patches=10)
        >>> new_elai = jnp.ones(10) * 2.5
        >>> updated = update_canopy_state(state, elai_patch=new_elai)
    """
    return state._replace(**updates)


def validate_canopy_state(state: CanopyState) -> bool:
    """Validate canopy state for physical consistency.

    Checks that all arrays have consistent shapes and values are
    within physically reasonable ranges.

    Args:
        state: Canopy state to validate

    Returns:
        bool: True if state is valid

    Note:
        This is a helper function not present in the original Fortran.
        It's useful for debugging and ensuring data integrity in JAX.
    """
    # Check shape consistency
    n_patches = state.elai_patch.shape[0]
    shapes_match = (
        state.frac_veg_nosno_patch.shape[0] == n_patches
        and state.esai_patch.shape[0] == n_patches
        and state.htop_patch.shape[0] == n_patches
    )

    if not shapes_match:
        return False

    # Check physical bounds (where not NaN)
    # frac_veg_nosno should be 0 or 1
    frac_valid = jnp.all(
        jnp.isnan(state.frac_veg_nosno_patch)
        | ((state.frac_veg_nosno_patch >= 0.0) & (state.frac_veg_nosno_patch <= 1.0))
    )

    # LAI/SAI should be non-negative
    elai_valid = jnp.all(jnp.isnan(state.elai_patch) | (state.elai_patch >= 0.0))
    esai_valid = jnp.all(jnp.isnan(state.esai_patch) | (state.esai_patch >= 0.0))

    # Height should be non-negative
    htop_valid = jnp.all(jnp.isnan(state.htop_patch) | (state.htop_patch >= 0.0))

    return bool(frac_valid and elai_valid and esai_valid and htop_valid)


# Backward compatibility alias (Fortran naming convention)
canopystate_type = CanopyState
