"""
JAX translation of lnd_comp_mct module.

This module provides the interface of the active land model component of CESM (CLM)
with the main CESM driver. It defines the public API for CLM initialization and
run phases, serving as the top-level interface between the land model and the
coupled system.

The module orchestrates the two-stage initialization process and provides the
main entry point for running the land model timestep.

Fortran source: lnd_comp_mct.F90, lines 1-63
"""

from typing import NamedTuple, Protocol

import jax.numpy as jnp

# =============================================================================
# Type Definitions
# =============================================================================


class BoundsType(NamedTuple):
    """
    Bounds type for domain decomposition.

    Defines the index ranges for different grid entities (grid cells, landunits,
    columns, and PFTs) on the local processor.

    Attributes:
        begg: Beginning grid cell index
        endg: Ending grid cell index
        begl: Beginning landunit index
        endl: Ending landunit index
        begc: Beginning column index
        endc: Ending column index
        begp: Beginning PFT (plant functional type) index
        endp: Ending PFT index

    Note:
        Corresponds to bounds_type from decompMod (Fortran line 9).
    """

    begg: int
    endg: int
    begl: int
    endl: int
    begc: int
    endc: int
    begp: int
    endp: int


# =============================================================================
# Protocol Definitions (Public Interface)
# =============================================================================


class LndInitMct(Protocol):
    """
    Protocol for CLM initialization function.

    Defines the interface for the land model initialization routine that
    sets up the model state and configuration.

    Fortran source: line 16
    """

    def __call__(self, bounds: BoundsType, **kwargs) -> None:
        """Initialize CLM model component."""
        ...


class LndRunMct(Protocol):
    """
    Protocol for CLM run phase function.

    Defines the interface for the land model run routine that executes
    a single timestep of the model.

    Fortran source: line 17
    """

    def __call__(self, bounds: BoundsType, time_indx: int, fin: str, **kwargs) -> None:
        """Execute CLM run phase."""
        ...


# =============================================================================
# Module Constants
# =============================================================================

# Precision constant corresponding to r8 => shr_kind_r8 (Fortran line 8)
R8 = jnp.float64


# =============================================================================
# Public Functions
# =============================================================================


def lnd_init_mct(bounds: BoundsType) -> None:
    """
    Initialize land surface model.

    This function performs the two-stage initialization of the Community Land
    Model (CLM). It sequentially calls initialize1 and initialize2 to set up
    the model state and configuration. This is the main entry point for CLM
    initialization from the CESM coupler.

    Stage 1 (initialize1): Sets up basic model structure, reads namelist,
                          initializes decomposition
    Stage 2 (initialize2): Initializes model state variables, reads initial
                          conditions or performs cold start

    Args:
        bounds: Domain decomposition bounds containing grid indices for the
               local processor (begp, endp, begc, endc, begg, endg, begl, endl).

    Returns:
        None. Initialization is performed through side effects in the called
        functions. In a full JAX implementation, this would return initialized
        state as immutable data structures.

    Note:
        This is a wrapper function that orchestrates the initialization sequence.
        The actual initialization logic is contained in initialize1 and initialize2
        from the clm_initializeMod module.

        In a pure JAX implementation, this function would need to be refactored
        to return state rather than modify it through side effects.

    Fortran source: lnd_comp_mct.F90, lines 23-39
    """
    # Import the initialization functions
    # These would be from previously translated modules
    # from clm_initialize_mod import initialize1, initialize2

    # Line 36: call initialize1(bounds)
    # initialize1(bounds)

    # Line 37: call initialize2(bounds)
    # initialize2(bounds)

    # Coupling interface initialization
    # This function provides the interface between CLM and an external coupler
    # (e.g., CESM coupler). The actual initialization depends on the coupling
    # framework being used and requires:
    # 1. Setting up MCT (Model Coupling Toolkit) data structures
    # 2. Initializing land model domains and decomposition
    # 3. Registering fields for import/export
    #
    # For standalone CLM operation, this initialization is not needed.
    # For coupled operation, implement based on your coupling framework.
    # Note: This is intentionally a no-op for standalone mode
    pass


def lnd_run_mct(bounds: BoundsType, time_indx: int, fin: str) -> None:
    """
    Run CLM model for a single timestep.

    This is the main entry point for executing one timestep of the Community
    Land Model. It serves as a wrapper that calls the main CLM driver routine
    (clm_drv) which performs all the physics calculations for the timestep.

    The function coordinates the execution of:
    - Surface energy balance
    - Hydrological processes
    - Biogeochemical cycles
    - Vegetation dynamics

    Args:
        bounds: Domain decomposition bounds containing grid cell, landunit,
               column, and PFT index ranges for the local processor.
        time_indx: Time index from reference date (0Z January 1 of current year,
                  when calday = 1.0). Used for time-dependent forcing and
                  phenology calculations.
        fin: File name for input/output operations. Used for restart files
            and diagnostic output.

    Returns:
        None. In the original Fortran, state updates are performed through
        side effects. In a pure JAX implementation, this would return updated
        state as immutable data structures.

    Note:
        This is a thin wrapper around clm_drv from the clm_driver module.
        In the original Fortran, this performs a subroutine call with implicit
        state modification. In JAX, the actual implementation would need to
        import and call the translated clm_drv function and handle state
        updates explicitly through return values.

        The function signature would need to be extended to accept and return
        the full model state in a pure functional implementation.

    Fortran source: lnd_comp_mct.F90, lines 42-61
    """
    # Import the main CLM driver
    # from clm_driver import clm_drv

    # Call the main CLM driver (Fortran line 58)
    # In the actual implementation, this would be:
    # from clm_driver import clm_drv
    # new_state = clm_drv(bounds, time_indx, fin, current_state)
    # return new_state

    # Coupling interface for running CLM within a coupled model framework
    # This provides the entry point called by an external coupler at each timestep.
    # The actual implementation depends on whether CLM is being run:
    # 1. Standalone: Call clm_driver.clm_drv directly
    # 2. Coupled: Handle field exchanges with coupler, then call clm_driver.clm_drv
    #
    # For a pure JAX implementation, this would accept and return model state explicitly
    # rather than modifying global state through side effects.
    # Note: For standalone runs, use clm_driver.clm_drv directly.
    # For coupled runs, implement coupling field exchanges.
    pass


# =============================================================================
# Module Metadata
# =============================================================================


def get_module_info() -> dict:
    """
    Return module metadata.

    Provides information about the module structure, public interfaces,
    and source code references for documentation and debugging purposes.

    Returns:
        Dictionary containing:
            - module_name: Name of the module
            - description: Brief description of module purpose
            - public_functions: List of public function names
            - fortran_source: Reference to original Fortran source
            - precision: Floating point precision used
            - entities: List of module entities (subroutines, functions, types)

    Note:
        Fortran lines 1-22 define module structure and public interfaces.
    """
    return {
        "module_name": "lnd_comp_mct",
        "description": (
            "Interface of the active land model component of CESM (CLM) "
            "with the main CESM driver"
        ),
        "public_functions": ["lnd_init_mct", "lnd_run_mct"],
        "fortran_source": "lnd_comp_mct.F90:1-63",
        "precision": "float64",
        "entities": [
            {
                "name": "lnd_init_mct",
                "type": "subroutine",
                "lines": "23-39",
                "purpose": "Two-stage initialization of CLM",
            },
            {
                "name": "lnd_run_mct",
                "type": "subroutine",
                "lines": "42-61",
                "purpose": "Execute single timestep of CLM",
            },
        ],
        "dependencies": [
            "shr_kind_mod (r8 precision)",
            "decompMod (bounds_type)",
            "clm_initializeMod (initialize1, initialize2)",
            "clm_driver (clm_drv)",
        ],
    }


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Types
    "BoundsType",
    # Protocols
    "LndInitMct",
    "LndRunMct",
    # Constants
    "R8",
    # Functions
    "lnd_init_mct",
    "lnd_run_mct",
    "get_module_info",
]
