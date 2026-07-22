"""
JAX translation of wateratm2lndBulkType Fortran module.

Atmosphere-to-land water flux variables for the CLM land surface model.
Provides data structures and initialization routines for downscaled
atmospheric water forcing quantities at the column level.

Original Fortran module: wateratm2lndBulkType
Fortran lines 1-68
"""

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

from legoesm.land.canopy.clm_ml_backend.clm_src_main.clm_varpar import numrad  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_main.decompMod import bounds_type  # noqa: F401

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


class wateratm2lndbulk_type(NamedTuple):
    """
    Atmosphere-to-land bulk water forcing variables mirroring Fortran
    ``wateratm2lndbulk_type`` (lines 19-30).

    All arrays are immutable JAX arrays. The Fortran offset ``begc:endc``
    becomes a contiguous array of length ``endc - begc + 1``, with
    ``begc`` stored in ``bounds_type`` for downstream index arithmetic.

    Attributes:
        forc_q_downscaled_col: Atmospheric specific humidity downscaled
            to column (kg/kg). Shape ``(endc - begc + 1,)``.
            Fortran: ``forc_q_downscaled_col(begc:endc)``.
        forc_rain_downscaled_col: Rainfall rate downscaled to column
            (mm/s). Shape ``(endc - begc + 1,)``.
            Fortran: ``forc_rain_downscaled_col(begc:endc)``.
        forc_snow_downscaled_col: Snowfall rate downscaled to column
            (mm/s). Shape ``(endc - begc + 1,)``.
            Fortran: ``forc_snow_downscaled_col(begc:endc)``.
    """

    forc_q_downscaled_col: Array  # (n_col,)
    forc_rain_downscaled_col: Array  # (n_col,)
    forc_snow_downscaled_col: Array  # (n_col,)


# ---------------------------------------------------------------------------
# Initialization helpers (mirror Init / InitAllocate)
# ---------------------------------------------------------------------------


def InitAllocate(bounds: bounds_type) -> wateratm2lndbulk_type:
    """
    Allocate and initialize a ``wateratm2lndbulk_type`` instance.

    Mirrors Fortran subroutine ``InitAllocate`` (lines 50-65).
    All arrays are filled with ``ival = 0.0`` matching the Fortran
    initialisation ``this%field(:) = ival``.

    Args:
        bounds: Decomposition bounds supplying ``begg``, ``endg``,
            ``begc``, and ``endc`` for this MPI task.

    Returns:
        A fully initialised :class:`wateratm2lndbulk_type` with every
        element set to ``0.0``.
    """
    ival = jnp.float64(0.0)  # Fortran line 52: ival = 0.0_r8

    begc = bounds.begc
    endc = bounds.endc  # Fortran lines 55-56

    n_col = endc - begc + 1

    # Fortran: allocate(this%forc_q_downscaled_col(begc:endc)); ... = ival
    forc_q_downscaled_col = jnp.full((n_col,), ival, dtype=jnp.float64)

    # Fortran: allocate(this%forc_rain_downscaled_col(begc:endc)); ... = ival
    forc_rain_downscaled_col = jnp.full((n_col,), ival, dtype=jnp.float64)

    # Fortran: allocate(this%forc_snow_downscaled_col(begc:endc)); ... = ival
    forc_snow_downscaled_col = jnp.full((n_col,), ival, dtype=jnp.float64)

    return wateratm2lndbulk_type(
        forc_q_downscaled_col=forc_q_downscaled_col,
        forc_rain_downscaled_col=forc_rain_downscaled_col,
        forc_snow_downscaled_col=forc_snow_downscaled_col,
    )


def Init(bounds: bounds_type) -> wateratm2lndbulk_type:
    """
    Initialize a ``wateratm2lndbulk_type`` instance.

    Mirrors Fortran subroutine ``Init`` (lines 37-43), which is the
    public entry point that delegates to ``InitAllocate``.

    Args:
        bounds: Decomposition bounds for the local MPI task.

    Returns:
        A fully initialised :class:`wateratm2lndbulk_type`.
    """
    return InitAllocate(bounds)
