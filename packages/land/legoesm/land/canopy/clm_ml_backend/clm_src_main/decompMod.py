"""
JAX/Python translation of the CLM decomposition module.

Provides the :class:`bounds_type` data structure and
:func:`get_clump_bounds` factory, which define the g/l/c/p index
ranges for CLM's gridcell → landunit → column → patch hierarchy.

In the standalone multilayer canopy configuration a single clump
covers exactly one gridcell with one landunit, one column, and one
patch (Fortran comment lines 20-25).

Original Fortran module: decompMod
"""

from __future__ import annotations

from typing import NamedTuple

# ---------------------------------------------------------------------------
# bounds_type
# ---------------------------------------------------------------------------


class bounds_type(NamedTuple):
    """
    Index bounds for CLM's g/l/c/p decomposition hierarchy.

    Mirrors Fortran derived type ``bounds_type`` (lines 17-22).

    All indices are 1-based and inclusive on both ends, matching the
    Fortran convention.  In standalone mode every range spans exactly
    one element (``beg == end == 1``).

    Attributes:
        begg: Beginning gridcell index.
        endg: Ending gridcell index.
        begl: Beginning landunit index.
        endl: Ending landunit index.
        begc: Beginning column index.
        endc: Ending column index.
        begp: Beginning patch index.
        endp: Ending patch index.
    """

    begg: int
    endg: int
    begl: int
    endl: int
    begc: int
    endc: int
    begp: int
    endp: int


# ---------------------------------------------------------------------------
# get_clump_bounds
# ---------------------------------------------------------------------------


def get_clump_bounds(n: int) -> bounds_type:
    """
    Return index bounds for processor clump ``n``.

    Mirrors Fortran subroutine ``get_clump_bounds`` (lines 28-50).

    In standalone mode this always returns a single-element range
    for all levels of the g/l/c/p hierarchy (all ``beg == end == 1``).

    Args:
        n: Processor clump index (unused in standalone mode; present for
           API compatibility with the full CLM decomposition).

    Returns:
        :class:`bounds_type` with all bounds set to 1.
    """
    return bounds_type(
        begg=1,
        endg=1,
        begl=1,
        endl=1,
        begc=1,
        endc=1,
        begp=1,
        endp=1,
    )
