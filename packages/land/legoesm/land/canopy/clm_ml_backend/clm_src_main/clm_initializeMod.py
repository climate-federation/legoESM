"""
JAX translation of clm_initializeMod Fortran module.

Performs land model initialization in two phases. Phase one initializes
run control variables; phase two reads PFT parameters, builds the
subgrid hierarchy, allocates filters, and initializes all derived-type
instances and time-constant variables.

Original Fortran module: clm_initializeMod
Fortran lines 1-80
"""

from legoesm.land.canopy.clm_ml_backend.clm_src_main import (
    ColumnType,
    GridcellType,
    pftconMod,  # noqa: F401
)
from legoesm.land.canopy.clm_ml_backend.clm_src_main.clm_instMod import clm_instInit  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_main.clm_varpar import clm_varpar_init
from legoesm.land.canopy.clm_ml_backend.clm_src_main.ColumnType import init_column  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_main.decompMod import bounds_type  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_main.filterMod import allocFilters, filter  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_main.GridcellType import gridcell_type_Init  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_main.initGridCellsMod import initGridCells  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_main.PatchType import patch  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_main.pftconMod import pftcon  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.multilayer_canopy import MLpftconMod  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLCanopyTurbulenceMod import LookupPsihatINI
from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLpftconMod import Init, MLpftcon  # noqa: F401

# ---------------------------------------------------------------------------
# Public: phase one initialization
# ---------------------------------------------------------------------------


def initialize1() -> None:
    """
    CLM initialization — first phase.

    Mirrors Fortran subroutine ``initialize1`` (lines 35-44).
    Initializes run control variables by calling ``clm_varpar_init``.
    This phase requires no decomposition bounds and must complete
    before :func:`initialize2` is called.
    """
    # Initialize run control variables — Fortran line 42
    clm_varpar_init()


# ---------------------------------------------------------------------------
# Public: phase two initialization
# ---------------------------------------------------------------------------


def initialize2(bounds: bounds_type) -> None:
    """
    CLM initialization — second phase.

    Mirrors Fortran subroutine ``initialize2`` (lines 46-76).

    Executes the following initialization sequence:

    1. Read CLM PFT parameter list and values (``pftcon.Init``).
    2. Read CLMml PFT parameter list and values
       (``MLpftcon.Init``).
    3. Initialize look-up tables for the CLMml roughness sublayer
       psihat functions (``LookupPsihatINI``).
    4. Allocate memory for subgrid data structures
       (``grc.Init``, ``col.Init``, ``patch.Init``).
    5. Build the subgrid hierarchy of landunit, column, and patch
       (``initGridCells``).
    6. Allocate column and patch filters (``allocFilters``).
    7. Initialize all derived-type instances and time-constant
       variables (``clm_instInit``).

    Args:
        bounds: Decomposition bounds for the local MPI task, supplying
            ``begg``, ``endg``, ``begc``, ``endc``, ``begp``, and
            ``endp``.
    """
    # Read list of PFTs and their parameter values — Fortran lines 54-55
    pftconMod.pftcon = pftconMod.Init()
    MLpftconMod.MLpftcon = MLpftconMod.Init()

    # Initialize CLMml roughness sublayer psihat look-up tables — Fortran line 59
    LookupPsihatINI()  # CLMml

    # Allocate memory for subgrid data structures — Fortran lines 61-63
    GridcellType.grc = gridcell_type_Init(bounds.begg, bounds.endg)
    ColumnType.col = init_column(bounds.begc, bounds.endc)

    patch.Init(bounds.begp, bounds.endp)

    # Build subgrid hierarchy of landunit, column, and patch — Fortran line 65
    initGridCells()

    # Allocate filters — Fortran line 67
    # Note: allocFilters returns a new filter instance (functional style)
    # but we're updating the global filter module variable

    global filter
    filter = allocFilters(bounds.begp, bounds.endp, bounds.begc, bounds.endc)

    # Initialize instances of all derived types and time-constant variables
    # Fortran lines 70-72
    clm_instInit(bounds)
