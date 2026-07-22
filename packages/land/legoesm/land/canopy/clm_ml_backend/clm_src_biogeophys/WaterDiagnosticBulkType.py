"""
JAX translation of WaterDiagnosticBulkType Fortran module.

Water diagnostic bulk state variables for the CLM land surface model.
Provides data structures and initialization routines for diagnostic
water quantities at the patch and column level.

Original Fortran module: WaterDiagnosticBulkType
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


class waterdiagnosticbulk_type(NamedTuple):
    """
    Water diagnostic bulk state variables mirroring Fortran
    ``waterdiagnosticbulk_type`` (lines 18-28).

    All arrays are immutable JAX arrays. Index conventions follow the
    original Fortran exactly; Python arrays are zero-based so the Fortran
    offset ``begp:endp`` becomes a contiguous array of length
    ``endp - begp + 1``, with ``begp`` / ``begc`` stored separately in
    ``bounds_type`` for downstream index arithmetic.

    Attributes:
        q_ref2m_patch: 2 m height surface specific humidity per patch
            (kg/kg). Shape ``(endp - begp + 1,)``.
            Fortran: ``q_ref2m_patch(begp:endp)``.
        frac_sno_eff_col: Fraction of ground covered by snow per column
            (0 to 1). Shape ``(endc - begc + 1,)``.
            Fortran: ``frac_sno_eff_col(begc:endc)``.
        bw_col: Partial density of water in the snow pack (ice + liquid)
            per column and snow layer (kg/m3).
            Shape ``(endc - begc + 1, nlevsno)``.
            Fortran: ``bw_col(begc:endc, -nlevsno+1:0)``.
    """

    q_ref2m_patch: Array  # (n_patch,)
    frac_sno_eff_col: Array  # (n_col,)
    bw_col: Array  # (n_col, nlevsno)


# ---------------------------------------------------------------------------
# Initialization helpers  (mirror Init / InitAllocate)
# ---------------------------------------------------------------------------


def InitAllocate(bounds: bounds_type) -> waterdiagnosticbulk_type:
    """
    Allocate and initialize a ``waterdiagnosticbulk_type`` instance.

    Mirrors Fortran subroutine ``InitAllocate`` (lines 48-66).
    All arrays are filled with ``nan`` (``spval``) matching the Fortran
    initialisation ``this%field(:) = nan``.

    Args:
        bounds: Decomposition bounds supplying ``begp``, ``endp``,
            ``begc``, and ``endc`` for this MPI task.

    Returns:
        A fully initialised :class:`waterdiagnosticbulk_type` with every
        element set to ``nan`` (``spval``).
    """
    begp = bounds.begp
    endp = bounds.endp  # Fortran lines 55-56
    begc = bounds.begc
    endc = bounds.endc  # Fortran lines 55-56

    n_patch = endp - begp + 1
    n_col = endc - begc + 1

    # Fortran: allocate(this%q_ref2m_patch(begp:endp)); ... = nan
    q_ref2m_patch = jnp.full((n_patch,), nan, dtype=jnp.float64)

    # Fortran: allocate(this%frac_sno_eff_col(begc:endc)); ... = nan
    frac_sno_eff_col = jnp.full((n_col,), nan, dtype=jnp.float64)

    # Fortran: allocate(this%bw_col(begc:endc, -nlevsno+1:0)); ... = nan
    # Second axis spans snow layers -nlevsno+1 .. 0  => nlevsno levels
    bw_col = jnp.full((n_col, nlevsno), nan, dtype=jnp.float64)

    return waterdiagnosticbulk_type(
        q_ref2m_patch=q_ref2m_patch,
        frac_sno_eff_col=frac_sno_eff_col,
        bw_col=bw_col,
    )


def Init(bounds: bounds_type) -> waterdiagnosticbulk_type:
    """
    Initialize a ``waterdiagnosticbulk_type`` instance.

    Mirrors Fortran subroutine ``Init`` (lines 38-44), which is the
    public entry point that delegates to ``InitAllocate``.

    Args:
        bounds: Decomposition bounds for the local MPI task.

    Returns:
        A fully initialised :class:`waterdiagnosticbulk_type`.
    """
    return InitAllocate(bounds)
