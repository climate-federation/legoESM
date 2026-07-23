"""
JAX/Python translation of the CLM gridcell data type module.

Provides the :class:`gridcell_type` NamedTuple and its factory
function :func:`gridcell_type_Init`, plus the module-level singleton
``grc`` that holds latitude/longitude for each gridcell.

Original Fortran module: GridcellType
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from legoesm.land.canopy.clm_ml_backend.clm_src_main.clm_varcon import spval as nan  # nan => spval

# ---------------------------------------------------------------------------
# gridcell_type
# ---------------------------------------------------------------------------


class GridcellType(NamedTuple):
    """
    Gridcell geographic coordinates.

    Mirrors Fortran derived type ``gridcell_type`` (lines 18-22).

    Arrays are NumPy float64, 0-based storage, sized
    ``(endg - begg + 1,)`` and initialised to ``nan`` (``spval``) by
    :func:`gridcell_type_Init`.

    Attributes:
        latdeg: Latitude  (degrees).  Fortran: ``latdeg(begg:endg)``.
        londeg: Longitude (degrees).  Fortran: ``londeg(begg:endg)``.
    """

    latdeg: np.ndarray  # shape (endg - begg + 1,)
    londeg: np.ndarray  # shape (endg - begg + 1,)


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------


def gridcell_type_Init(begg: int, endg: int) -> GridcellType:
    """
    Allocate and return a :class:`GridcellType` initialised to ``nan``.

    Mirrors Fortran subroutine ``Init`` (lines 36-44).

    Fortran::

        allocate(this%latdeg(begg:endg)) ; this%latdeg(:) = nan
        allocate(this%londeg(begg:endg)) ; this%londeg(:) = nan

    Args:
        begg: First gridcell index (inclusive, 1-based).
        endg: Last  gridcell index (inclusive, 1-based).

    Returns:
        A new :class:`gridcell_type` with both arrays filled with
        ``spval`` (the CLM missing-value sentinel).
    """
    ng = endg - begg + 1
    return GridcellType(
        latdeg=np.full(ng, nan, dtype=np.float64),
        londeg=np.full(ng, nan, dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# Module-level singleton — Fortran: type(gridcell_type), public, target :: grc
# Initialised to a single-gridcell domain (begg=endg=1).
# ---------------------------------------------------------------------------
grc: GridcellType = gridcell_type_Init(begg=1, endg=1)
