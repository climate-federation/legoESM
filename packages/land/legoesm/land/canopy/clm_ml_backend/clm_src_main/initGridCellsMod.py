"""
JAX/Python translation of the CLM sub-grid initialisation module.

Initialises the sub-grid mapping for each land gridcell, establishing
the landunit → column → patch hierarchy.  In standalone mode the
domain is a single naturally-vegetated patch determined by the tower
site's PFT.

Original Fortran module: initGridCellsMod
"""

from __future__ import annotations

import legoesm.land.canopy.clm_ml_backend.offline_driver.TowerDataMod as TowerDataMod
from legoesm.land.canopy.clm_ml_backend.clm_src_main.initSubgridMod import add_patch

# ---------------------------------------------------------------------------
# set_landunit_veg_compete  (private)
# ---------------------------------------------------------------------------


def _set_landunit_veg_compete() -> None:
    """
    Initialise the vegetated landunit with competition.

    Mirrors Fortran subroutine ``set_landunit_veg_compete`` (lines 39-52).

    In standalone mode one patch is registered via :func:`add_patch`
    using the tower site's plant functional type::

        pi = 0
        call add_patch(pi, tower_pft(tower_num))

    The patch counter ``pi`` starts at 0 and is passed by reference in
    Fortran (incremented inside ``add_patch``).  Here :func:`add_patch`
    accepts and returns the updated counter following the pure-function
    convention used throughout the codebase.
    """
    pi: int = -1
    add_patch(pi, int(TowerDataMod.tower_pft[TowerDataMod.tower_num]))


# ---------------------------------------------------------------------------
# initGridcells  (public)
# ---------------------------------------------------------------------------


def initGridCells() -> None:
    """
    Initialise sub-grid mapping and the g/l/c/p derived-type hierarchy.

    Mirrors Fortran subroutine ``initGridcells`` (lines 23-32).

    For each land gridcell this determines landunit, column, and patch
    properties.  In standalone mode the only landunit is the naturally
    vegetated one, delegated to :func:`_set_landunit_veg_compete`.
    """
    _set_landunit_veg_compete()
