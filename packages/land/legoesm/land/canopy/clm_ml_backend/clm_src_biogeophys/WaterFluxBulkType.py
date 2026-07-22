"""
JAX translation of WaterFluxBulkType Fortran module.

Bulk water flux variables for the CLM land surface model.
Provides data structures and initialization routines for total
evapotranspiration flux at the patch level.

Original Fortran module: WaterFluxBulkType
Fortran lines 1-60
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


class waterfluxbulk_type(NamedTuple):
    """
    Bulk water flux variables mirroring Fortran ``waterfluxbulk_type``
    (lines 19-26).

    All arrays are immutable JAX arrays. The Fortran offset ``begp:endp``
    becomes a contiguous array of length ``endp - begp + 1``, with
    ``begp`` stored in ``bounds_type`` for downstream index arithmetic.

    Attributes:
        qflx_evap_tot_patch: Total evapotranspiration flux per patch
            (kg H2O/m2/s), combining soil evaporation, canopy
            evaporation, and transpiration.
            Shape ``(endp - begp + 1,)``.
            Fortran: ``qflx_evap_tot_patch(begp:endp)``.
    """

    qflx_evap_tot_patch: Array  # (n_patch,)


# ---------------------------------------------------------------------------
# Initialization helpers (mirror Init / InitAllocate)
# ---------------------------------------------------------------------------


def InitAllocate(bounds: bounds_type) -> waterfluxbulk_type:
    """
    Allocate and initialize a ``waterfluxbulk_type`` instance.

    Mirrors Fortran subroutine ``InitAllocate`` (lines 46-56).
    All arrays are filled with ``nan`` (``spval``) matching the Fortran
    initialisation ``this%qflx_evap_tot_patch(:) = nan``.

    Args:
        bounds: Decomposition bounds supplying ``begp`` and ``endp``
            for this MPI task.

    Returns:
        A fully initialised :class:`waterfluxbulk_type` with every
        element set to ``nan`` (``spval``).
    """
    begp = bounds.begp
    endp = bounds.endp  # Fortran line 52

    n_patch = endp - begp + 1

    # Fortran: allocate(this%qflx_evap_tot_patch(begp:endp)); ... = nan
    qflx_evap_tot_patch = jnp.full((n_patch,), nan, dtype=jnp.float64)

    return waterfluxbulk_type(
        qflx_evap_tot_patch=qflx_evap_tot_patch,
    )


def Init(bounds: bounds_type) -> waterfluxbulk_type:
    """
    Initialize a ``waterfluxbulk_type`` instance.

    Mirrors Fortran subroutine ``Init`` (lines 38-43), which is the
    public entry point that delegates to ``InitAllocate``.

    Args:
        bounds: Decomposition bounds for the local MPI task.

    Returns:
        A fully initialised :class:`waterfluxbulk_type`.
    """
    return InitAllocate(bounds)
