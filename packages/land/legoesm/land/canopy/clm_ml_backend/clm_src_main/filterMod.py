"""
JAX/Python translation of the CLM filter module.

Provides the :class:`clumpfilter` data structure and helper functions
for initialising, setting, and updating the patch/column filters used
throughout CLM processing.

In standalone mode the domain is a single patch and a single column,
so every filter collapses to a one-element array pointing at index 1.

Design: ``clumpfilter`` is an immutable :class:`NamedTuple`; all
"mutation" is expressed by returning a new instance, consistent with
the JAX pure-function convention used throughout the codebase.
Index arrays are NumPy (not JAX) because they serve as Python-level
loop indices and are never passed through ``jit``-compiled kernels.

Original Fortran module: filterMod
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from jax import Array

# ---------------------------------------------------------------------------
# clumpfilter
# ---------------------------------------------------------------------------


class clumpfilter(NamedTuple):
    """
    Immutable container for CLM patch and column filter arrays.

    Mirrors Fortran derived type ``clumpfilter`` (lines 20-33).

    All index arrays are 1-based NumPy integer arrays; counts are plain
    Python ints.  A new instance is constructed by :func:`allocFilters`,
    populated by :func:`setFilters`, and refined by
    :func:`setExposedvegpFilter`.  NumPy is used (rather than JAX) because
    these arrays serve as Python-level loop indices and are never passed
    through ``jit``-compiled kernels.

    Attributes:
        num_exposedvegp:   Number of patches in the exposedvegp filter.
        exposedvegp:       Patch indices where ``frac_veg_nosno > 0``.
        num_nolakeurbanp:  Number of patches in the non-lake/non-urban filter.
        nolakeurbanp:      Non-lake, non-urban patch indices.
        num_nolakec:       Number of columns in the non-lake filter.
        nolakec:           Non-lake column indices.
        num_nourbanc:      Number of columns in the non-urban filter.
        nourbanc:          Non-urban column indices.
        num_hydrologyc:    Number of columns in the hydrology filter.
        hydrologyc:        Hydrology column indices.
    """

    # Counts — Fortran integer scalars inside clumpfilter
    num_exposedvegp: int
    num_nolakeurbanp: int
    num_nolakec: int
    num_nourbanc: int
    num_hydrologyc: int

    # Index arrays — NumPy int32, 0-based storage, 1-based values
    exposedvegp: np.ndarray  # shape (endp - begp + 1,)
    nolakeurbanp: np.ndarray  # shape (endp - begp + 1,)
    nolakec: np.ndarray  # shape (endc - begc + 1,)
    nourbanc: np.ndarray  # shape (endc - begc + 1,)
    hydrologyc: np.ndarray  # shape (endc - begc + 1,)


# ---------------------------------------------------------------------------
# allocFilters
# ---------------------------------------------------------------------------


def allocFilters(
    begp: int,
    endp: int,
    begc: int,
    endc: int,
) -> clumpfilter:
    """
    Allocate and return a zero-initialised :class:`clumpfilter`.

    Mirrors Fortran subroutine ``allocFilters`` (lines 43-58).

    Fortran allocates 1-based slices ``(begp:endp)`` and ``(begc:endc)``
    as pointer components of the derived type.  Here we allocate flat
    NumPy arrays of the same length and return an immutable
    :class:`clumpfilter` NamedTuple — no mutation, no ``intent(inout)``.

    Args:
        begp: First patch index (inclusive, 1-based).
        endp: Last patch index (inclusive, 1-based).
        begc: First column index (inclusive, 1-based).
        endc: Last column index (inclusive, 1-based).

    Returns:
        A new :class:`clumpfilter` with all count fields set to 0 and
        all index arrays zero-initialised.
    """
    np_size = endp - begp + 1  # number of patches
    nc_size = endc - begc + 1  # number of columns

    return clumpfilter(
        num_exposedvegp=0,
        num_nolakeurbanp=0,
        num_nolakec=0,
        num_nourbanc=0,
        num_hydrologyc=0,
        exposedvegp=np.zeros(np_size, dtype=np.int32),  # Fortran: (begp:endp)
        nolakeurbanp=np.zeros(np_size, dtype=np.int32),  # Fortran: (begp:endp)
        nolakec=np.zeros(nc_size, dtype=np.int32),  # Fortran: (begc:endc)
        nourbanc=np.zeros(nc_size, dtype=np.int32),  # Fortran: (begc:endc)
        hydrologyc=np.zeros(nc_size, dtype=np.int32),  # Fortran: (begc:endc)
    )


# ---------------------------------------------------------------------------
# setFilters
# ---------------------------------------------------------------------------


def setFilters(filter: clumpfilter) -> clumpfilter:
    """
    Set CLM filters for the standalone single-patch/single-column case.

    Mirrors Fortran subroutine ``setFilters`` (lines 62-75).

    In the standalone configuration every filter count is 1 and every
    element in the corresponding index array is set to 1 (the single
    valid index), matching the Fortran assignments::

        filter%num_nolakeurbanp = 1 ; filter%nolakeurbanp(:) = 1
        filter%num_nolakec      = 1 ; filter%nolakec(:)      = 1
        filter%num_nourbanc     = 1 ; filter%nourbanc(:)     = 1
        filter%num_hydrologyc   = 1 ; filter%hydrologyc(:)   = 1

    ``num_exposedvegp`` and ``exposedvegp`` are left to
    :func:`setExposedvegpFilter`.

    Args:
        filter: Existing :class:`clumpfilter` (from :func:`allocFilters`).

    Returns:
        A new :class:`clumpfilter` with non-lake/non-urban counts and
        index arrays set to 1; ``exposedvegp`` fields carried over
        unchanged.
    """
    nolakeurbanp = filter.nolakeurbanp.copy()
    nolakeurbanp[:] = 0  # 0-based index for single patch
    nolakec = filter.nolakec.copy()
    nolakec[:] = 0  # 0-based index for single column
    nourbanc = filter.nourbanc.copy()
    nourbanc[:] = 0  # 0-based index for single column
    hydrologyc = filter.hydrologyc.copy()
    hydrologyc[:] = 0  # 0-based index for single column

    return clumpfilter(
        num_exposedvegp=filter.num_exposedvegp,
        num_nolakeurbanp=1,
        num_nolakec=1,
        num_nourbanc=1,
        num_hydrologyc=1,
        exposedvegp=filter.exposedvegp,
        nolakeurbanp=nolakeurbanp,
        nolakec=nolakec,
        nourbanc=nourbanc,
        hydrologyc=hydrologyc,
    )


# ---------------------------------------------------------------------------
# setExposedvegpFilter
# ---------------------------------------------------------------------------


def setExposedvegpFilter(
    filter: clumpfilter,
    frac_veg_nosno: Array,
) -> clumpfilter:
    """
    Build and return an updated filter with the ``exposedvegp`` patch list.

    Mirrors Fortran subroutine ``setExposedvegpFilter`` (lines 79-101).

    Iterates over the non-lake/non-urban patch filter and retains those
    patches for which ``frac_veg_nosno(p) > 0``::

        fe = 0
        do fp = 1, filter%num_nolakeurbanp
            p = filter%nolakeurbanp(fp)
            if (frac_veg_nosno(p) > 0):
                fe = fe + 1
                filter%exposedvegp(fe) = p
        filter%num_exposedvegp = fe

    Args:
        filter:         Existing :class:`clumpfilter` (from :func:`setFilters`).
        frac_veg_nosno: Fraction of vegetation not covered by snow, indexed
                        by patch (1-based values; element 0 unused).

    Returns:
        A new :class:`clumpfilter` identical to ``filter`` except that
        ``exposedvegp`` and ``num_exposedvegp`` reflect the current
        ``frac_veg_nosno`` state.
    """
    exposedvegp = filter.exposedvegp.copy()
    fe: int = 0

    for fp in range(1, filter.num_nolakeurbanp + 1):
        p = int(filter.nolakeurbanp[fp - 1])  # 1-based fp → 0-based array slot
        if frac_veg_nosno[p] > 0:
            exposedvegp[fe] = p  # write into 0-based accumulator slot
            fe += 1

    return clumpfilter(
        num_exposedvegp=fe,
        num_nolakeurbanp=filter.num_nolakeurbanp,
        num_nolakec=filter.num_nolakec,
        num_nourbanc=filter.num_nourbanc,
        num_hydrologyc=filter.num_hydrologyc,
        exposedvegp=exposedvegp,
        nolakeurbanp=filter.nolakeurbanp,
        nolakec=filter.nolakec,
        nourbanc=filter.nourbanc,
        hydrologyc=filter.hydrologyc,
    )


# ---------------------------------------------------------------------------
# Module-level singleton — Fortran: type(clumpfilter), public, target :: filter
# Initialised to a single-patch, single-column domain (begp=endp=begc=endc=1).
# ---------------------------------------------------------------------------
filter: clumpfilter = allocFilters(begp=1, endp=1, begc=1, endc=1)
