"""
JAX translation of TemperatureType Fortran module.

Temperature variables for the Community Land Model (CLM/CTSM).
Replaces Fortran allocatable pointers with immutable JAX arrays
initialised to NaN, mirroring the original InitAllocate semantics.

Original Fortran module: TemperatureType
Fortran lines 1-78
"""

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

from legoesm.land.canopy.clm_ml_backend.clm_src_main.clm_varcon import spval as nan
from legoesm.land.canopy.clm_ml_backend.clm_src_main.clm_varpar import nlevgrnd, nlevsno  # col: nlevgrnd, nlevsno
from legoesm.land.canopy.clm_ml_backend.clm_src_main.decompMod import bounds_type

# ---------------------------------------------------------------------------
# Public data type
# ---------------------------------------------------------------------------


class temperature_type(NamedTuple):
    """
    Temperature state variables, mirroring Fortran ``temperature_type``.

    All arrays are JAX arrays (immutable). NaN sentinel values match the
    Fortran initialisation ``= nan`` (where ``nan => spval`` from clm_varcon).

    Attributes:
        t_soisno_col:  Soil temperature (Kelvin) per column, shape
                       ``(endc-begc+1, nlevgrnd+nlevsno)``.
                       Fortran: ``real(r8), pointer :: t_soisno_col(:,:)``
                       index range ``(-nlevsno+1:nlevgrnd)``. (lines 22-22)
        t_a10_patch:   10-day running mean of 2 m temperature (K) per patch,
                       shape ``(endp-begp+1,)``.
                       Fortran: ``real(r8), pointer :: t_a10_patch(:)`` (lines 23-23)
        t_ref2m_patch: 2 m height surface air temperature (K) per patch,
                       shape ``(endp-begp+1,)``.
                       Fortran: ``real(r8), pointer :: t_ref2m_patch(:)`` (lines 24-24)
    """

    t_soisno_col: Array  # (n_col, nlevsno + nlevgrnd)  -- float64, init NaN
    t_a10_patch: Array  # (n_patch,)                   -- float64, init NaN
    t_ref2m_patch: Array  # (n_patch,)                   -- float64, init NaN


# ---------------------------------------------------------------------------
# Initialization helpers  (mirrors Init -> InitAllocate call chain)
# ---------------------------------------------------------------------------


def InitAllocate(bounds: bounds_type) -> temperature_type:
    """
    Allocate and initialise all temperature arrays to NaN.

    Mirrors Fortran ``InitAllocate`` (lines 51-72). Fortran allocates with
    index range ``(-nlevsno+1:nlevgrnd)`` for the snow/soil axis; the
    equivalent JAX axis length is ``nlevsno + nlevgrnd`` (zero-based).

    Args:
        bounds: Decomposition bounds supplying ``begp/endp`` (patch) and
                ``begc/endc`` (column) index ranges for this MPI task.

    Returns:
        A fully initialised :class:`temperature_type` with every element
        set to ``nan`` (``spval`` from ``clm_varcon``), matching the
        Fortran ``this%field(:) = nan`` initialisations.
    """
    # Fortran lines 57-58: begp/endp, begc/endc from bounds
    begp, endp = bounds.begp, bounds.endp
    begc, endc = bounds.begc, bounds.endc

    n_patch = endp - begp + 1
    n_col = endc - begc + 1
    n_lev = nlevsno + nlevgrnd  # covers index range -nlevsno+1 : nlevgrnd

    # Fortran lines 60-62: allocate + initialise to nan
    t_soisno_col = jnp.full((n_col, n_lev), nan, dtype=jnp.float64)
    t_a10_patch = jnp.full((n_patch,), nan, dtype=jnp.float64)
    t_ref2m_patch = jnp.full((n_patch,), nan, dtype=jnp.float64)

    return temperature_type(
        t_soisno_col=t_soisno_col,
        t_a10_patch=t_a10_patch,
        t_ref2m_patch=t_ref2m_patch,
    )


def Init(bounds: bounds_type) -> temperature_type:
    """
    Initialise the temperature state for this MPI task.

    Mirrors Fortran ``Init`` (lines 42-48), which calls ``InitAllocate``.
    In the Fortran the method is bound to ``temperature_type`` via
    ``this%InitAllocate``; here it is a pure function that returns the
    constructed :class:`temperature_type` instead of mutating ``this``.

    Args:
        bounds: Decomposition bounds for the local MPI task.

    Returns:
        Fully initialised :class:`temperature_type` instance.
    """
    return InitAllocate(bounds)
