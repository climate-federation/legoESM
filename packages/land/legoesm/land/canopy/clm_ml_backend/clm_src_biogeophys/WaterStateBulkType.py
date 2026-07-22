"""
JAX translation of WaterStateBulkType Fortran module.

Bulk water state variables for the CLM land surface model.
Provides data structures and initialization routines for liquid water,
ice, volumetric soil water, and surface water at the column level.

Original Fortran module: WaterStateBulkType
Fortran lines 1-72
"""

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

from legoesm.land.canopy.clm_ml_backend.clm_src_main.clm_varcon import ispval  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_main.clm_varcon import spval as nan
from legoesm.land.canopy.clm_ml_backend.clm_src_main.clm_varpar import nlevgrnd, nlevsno  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_main.decompMod import bounds_type  # noqa: F401

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


class waterstatebulk_type(NamedTuple):
    """
    Bulk water state variables mirroring Fortran ``waterstatebulk_type``
    (lines 19-32).

    All arrays are immutable JAX arrays. Fortran offsets are mapped to
    contiguous zero-based axes; ``begc`` is stored in ``bounds_type``
    for downstream index arithmetic.

    Attributes:
        h2osoi_liq_col: Liquid water per column and layer (kg H2O/m2).
            Shape ``(endc - begc + 1, nlevsno + nlevgrnd)``.
            Fortran: ``h2osoi_liq_col(begc:endc, -nlevsno+1:nlevgrnd)``.
        h2osoi_ice_col: Ice lens per column and layer (kg H2O/m2).
            Shape ``(endc - begc + 1, nlevsno + nlevgrnd)``.
            Fortran: ``h2osoi_ice_col(begc:endc, -nlevsno+1:nlevgrnd)``.
        h2osoi_vol_col: Volumetric soil water per column and soil layer
            (m3/m3), constrained to ``0 <= h2osoi_vol <= watsat``.
            Shape ``(endc - begc + 1, nlevgrnd)``.
            Fortran: ``h2osoi_vol_col(begc:endc, 1:nlevgrnd)``.
        h2osfc_col: Surface water per column (mm H2O).
            Shape ``(endc - begc + 1,)``.
            Fortran: ``h2osfc_col(begc:endc)``.
    """

    h2osoi_liq_col: Array  # (n_col, nlevsno + nlevgrnd)
    h2osoi_ice_col: Array  # (n_col, nlevsno + nlevgrnd)
    h2osoi_vol_col: Array  # (n_col, nlevgrnd)
    h2osfc_col: Array  # (n_col,)


# ---------------------------------------------------------------------------
# Initialization helpers (mirror Init / InitAllocate)
# ---------------------------------------------------------------------------


def InitAllocate(bounds: bounds_type) -> waterstatebulk_type:
    """
    Allocate and initialize a ``waterstatebulk_type`` instance.

    Mirrors Fortran subroutine ``InitAllocate`` (lines 50-66).
    All arrays are filled with ``nan`` (``spval``) matching the Fortran
    initialisation ``this%field(:,:) = nan``.

    The second axis of ``h2osoi_liq_col`` and ``h2osoi_ice_col`` spans
    Fortran indices ``-nlevsno+1:nlevgrnd``, giving a total of
    ``nlevsno + nlevgrnd`` levels. The second axis of
    ``h2osoi_vol_col`` spans Fortran indices ``1:nlevgrnd``, giving
    ``nlevgrnd`` levels.

    Args:
        bounds: Decomposition bounds supplying ``begc`` and ``endc``
            for this MPI task.

    Returns:
        A fully initialised :class:`waterstatebulk_type` with every
        element set to ``nan`` (``spval``).
    """
    begc = bounds.begc
    endc = bounds.endc  # Fortran lines 58-59

    n_col = endc - begc + 1
    n_liqice = nlevsno + nlevgrnd  # Fortran: -nlevsno+1:nlevgrnd

    # Fortran: allocate(this%h2osoi_liq_col(begc:endc,-nlevsno+1:nlevgrnd)); ... = nan
    h2osoi_liq_col = jnp.full((n_col, n_liqice), nan, dtype=jnp.float64)

    # Fortran: allocate(this%h2osoi_ice_col(begc:endc,-nlevsno+1:nlevgrnd)); ... = nan
    h2osoi_ice_col = jnp.full((n_col, n_liqice), nan, dtype=jnp.float64)

    # Fortran: allocate(this%h2osoi_vol_col(begc:endc,1:nlevgrnd)); ... = nan
    h2osoi_vol_col = jnp.full((n_col, nlevgrnd), nan, dtype=jnp.float64)

    # Fortran: allocate(this%h2osfc_col(begc:endc)); ... = nan
    h2osfc_col = jnp.full((n_col,), nan, dtype=jnp.float64)

    return waterstatebulk_type(
        h2osoi_liq_col=h2osoi_liq_col,
        h2osoi_ice_col=h2osoi_ice_col,
        h2osoi_vol_col=h2osoi_vol_col,
        h2osfc_col=h2osfc_col,
    )


def Init(bounds: bounds_type) -> waterstatebulk_type:
    """
    Initialize a ``waterstatebulk_type`` instance.

    Mirrors Fortran subroutine ``Init`` (lines 43-48), which is the
    public entry point that delegates to ``InitAllocate``.

    Args:
        bounds: Decomposition bounds for the local MPI task.

    Returns:
        A fully initialised :class:`waterstatebulk_type`.
    """
    return InitAllocate(bounds)
