"""
JAX/Python translation of the CLM model parameters module.

Call :func:`clm_varpar_init` during model initialisation to set the
snow/soil layer counts based on the chosen physics version.

Original Fortran module: clm_varpar
"""

from legoesm.land.canopy.clm_ml_backend.offline_driver.clmSoilOptionMod import clm_phys

# ---------------------------------------------------------------------------
# Mutable layer-count parameters (set by clm_varpar_init)
# ---------------------------------------------------------------------------

nlevsno: int = -1  # Maximum number of snow layers
nlevsoi: int = -1  # Number of hydrologically active soil layers
nlevgrnd: int = -1  # Number of ground layers (including hydrologically inactive)

# ---------------------------------------------------------------------------
# Fixed parameters
# ---------------------------------------------------------------------------

numrad: int = 2  # Number of radiation wavebands
ivis: int = 1  # Visible waveband index
inir: int = 2  # Near-infrared waveband index
mxpft: int = 78  # Maximum number of plant functional types


# ---------------------------------------------------------------------------
# clm_varpar_init
# ---------------------------------------------------------------------------


def clm_varpar_init() -> None:
    """
    Initialise snow and soil layer counts from the chosen physics version.

    Mirrors Fortran subroutine ``clm_varpar_init`` (lines 33-44).

    **CLM5.0** (``clm_phys == 'CLM5_0'``):

    .. code-block:: none

        nlevsno  = 12
        nlevsoi  = 20
        nlevgrnd = nlevsoi + 5   (= 25)

    **CLM4.5** (``clm_phys == 'CLM4_5'``):

    .. code-block:: none

        nlevsno  = 5
        nlevsoi  = 10
        nlevgrnd = 15
    """
    global nlevsno, nlevsoi, nlevgrnd

    if clm_phys == "CLM5_0":
        nlevsno = 12
        nlevsoi = 20
        nlevgrnd = nlevsoi + 5  # 25
    elif clm_phys == "CLM4_5":
        nlevsno = 5
        nlevsoi = 10
        nlevgrnd = 15
