"""
JAX translation of lnd_comp_nuopc Fortran module.

Interface of the active land model component of CESM (CLM, Community Land Model)
with the main CESM driver. Provides initialization and time-stepping entry points
for the land surface model.

Original Fortran module: lnd_comp_nuopc
Fortran lines 1-72
"""

from legoesm.land.canopy.clm_ml_backend.clm_src_main.decompMod import bounds_type

# ---------------------------------------------------------------------------
# Public interface functions
# ---------------------------------------------------------------------------


def InitializeRealize(bounds: bounds_type) -> None:
    """
    Initialize the land surface model (CLM).

    Mirrors Fortran subroutine ``InitializeRealize`` (lines 30-44).
    Calls the two-phase CLM initialization sequence: ``initialize1``
    performs domain/grid setup independent of decomposition bounds, while
    ``initialize2`` completes state-variable allocation using the provided
    decomposition bounds.

    Args:
        bounds: Decomposition bounds for the local MPI task, describing the
            index ranges of gridcells, landunits, columns, and patches owned
            by this task.
    """
    from legoesm.land.canopy.clm_ml_backend.clm_src_main.clm_initializeMod import initialize1, initialize2  # noqa: F401

    initialize1()
    initialize2(bounds)


def ModelAdvance(
    bounds: bounds_type,
    time_indx: int,
    fin1: str,
    fin2: str,
) -> None:
    """
    Advance the CLM model by one time step.

    Mirrors Fortran subroutine ``ModelAdvance`` (lines 46-68).
    Delegates entirely to ``clm_drv``, which orchestrates the full
    physics sequence for a single driver time step. Model state is
    modified through global module-level instances in ``clm_instMod``.

    Args:
        bounds: Decomposition bounds for the local MPI task.
        time_indx: Time index measured from the reference date
            (0Z January 1 of the current year, where ``calday == 1.000``).
            Corresponds to Fortran ``integer, intent(in) :: time_indx``.
        fin1: Path to the first required input file (<=256 characters,
            matching Fortran ``character(len=256) :: fin1``).
        fin2: Path to the second required input file (<=256 characters,
            matching Fortran ``character(len=256) :: fin2``).

    Note:
        ``time_indx`` is a Python ``int`` (or scalar JAX integer array).
        File path strings are passed through unchanged and are not traced
        by JAX; they must not be used inside JIT-compiled kernels.
    """
    from legoesm.land.canopy.clm_ml_backend.clm_src_main.clm_driver import clm_drv  # noqa: F401

    clm_drv(bounds, time_indx, fin1, fin2)
