"""
JAX translation of WaterType Fortran module.

Water state variables for the CLM land surface model.
Provides data structures and initialization routines for water
quantities at the column level.

Original Fortran module: WaterType
Fortran lines 1-62
"""

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

from legoesm.land.canopy.clm_ml_backend.clm_src_main.clm_varcon import ispval  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_main.clm_varcon import spval as nan
from legoesm.land.canopy.clm_ml_backend.clm_src_main.decompMod import bounds_type  # noqa: F401

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


class water_type(NamedTuple):
    """
    Water state variables mirroring Fortran ``water_type`` (lines 18-28).

    All arrays are immutable JAX arrays. The Fortran offset ``begc:endc``
    becomes a contiguous array of length ``endc - begc + 1``, with
    ``begc`` stored in ``bounds_type`` for downstream index arithmetic.

    Attributes:
        h2osno_col: Snow water per column (mm H2O).
            Shape ``(endc - begc + 1,)``.
            Fortran: ``h2osno_col(begc:endc)``.
    """

    h2osno_col: Array  # (n_col,)


# ---------------------------------------------------------------------------
# Initialization helpers (mirror Init / InitAllocate)
# ---------------------------------------------------------------------------


def InitAllocate(bounds: bounds_type) -> water_type:
    """
    Allocate and initialize a ``water_type`` instance.

    Mirrors Fortran subroutine ``InitAllocate`` (lines 45-58).
    All arrays are filled with ``nan`` (``spval``) matching the Fortran
    initialisation ``this%h2osno_col(:) = nan``.

    Args:
        bounds: Decomposition bounds supplying ``begc`` and ``endc``
            for this MPI task.

    Returns:
        A fully initialised :class:`water_type` with every element
        set to ``nan`` (``spval``).
    """
    begc = bounds.begc
    endc = bounds.endc  # Fortran lines 53-54

    n_col = endc - begc + 1

    # Fortran: allocate(this%h2osno_col(begc:endc)); this%h2osno_col(:) = nan
    h2osno_col = jnp.full((n_col,), nan, dtype=jnp.float64)

    return water_type(
        h2osno_col=h2osno_col,
    )


def Init(bounds: bounds_type) -> water_type:
    """
    Initialize a ``water_type`` instance.

    Mirrors Fortran subroutine ``Init`` (lines 35-41), which is the
    public entry point that delegates to ``InitAllocate``.

    Args:
        bounds: Decomposition bounds for the local MPI task.

    Returns:
        A fully initialised :class:`water_type`.
    """
    return InitAllocate(bounds)
