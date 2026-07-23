"""
JAX translation of atm2lndType Fortran module.

Atmosphere-to-land forcing variables for the CLM land surface model.
Provides data structures and initialization routines for atmospheric
forcing quantities at the grid cell and column level.

Original Fortran module: atm2lndType
Fortran lines 1-80
"""

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

from legoesm.land.canopy.clm_ml_backend.clm_src_main.clm_varpar import numrad  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_main.decompMod import bounds_type  # noqa: F401

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


class atm2lnd_type(NamedTuple):
    """
    Atmosphere-to-land forcing variables mirroring Fortran
    ``atm2lnd_type`` (lines 19-36).

    All arrays are immutable JAX arrays. Fortran offsets ``begg:endg``
    and ``begc:endc`` become contiguous arrays of length
    ``endg - begg + 1`` and ``endc - begc + 1`` respectively, with
    ``begg`` and ``begc`` stored in ``bounds_type`` for downstream
    index arithmetic.

    Attributes:
        forc_u_grc: Atmospheric wind speed in east direction (m/s).
            Shape ``(endg - begg + 1,)``.
            Fortran: ``forc_u_grc(begg:endg)``.
        forc_v_grc: Atmospheric wind speed in north direction (m/s).
            Shape ``(endg - begg + 1,)``.
            Fortran: ``forc_v_grc(begg:endg)``.
        forc_pco2_grc: Atmospheric CO2 partial pressure (Pa).
            Shape ``(endg - begg + 1,)``.
            Fortran: ``forc_pco2_grc(begg:endg)``.
        forc_po2_grc: Atmospheric O2 partial pressure (Pa).
            Shape ``(endg - begg + 1,)``.
            Fortran: ``forc_po2_grc(begg:endg)``.
        forc_solad_downscaled_col: Atmospheric direct beam radiation
            downscaled to column, per waveband (W/m2).
            Shape ``(endc - begc + 1, numrad)``.
            Fortran: ``forc_solad_downscaled_col(begc:endc, numrad)``.
        forc_solai_grc: Atmospheric diffuse radiation per waveband
            (W/m2). Shape ``(endg - begg + 1, numrad)``.
            Fortran: ``forc_solai_grc(begg:endg, numrad)``.
        forc_t_downscaled_col: Atmospheric temperature downscaled to
            column (K). Shape ``(endc - begc + 1,)``.
            Fortran: ``forc_t_downscaled_col(begc:endc)``.
        forc_pbot_downscaled_col: Atmospheric pressure downscaled to
            column (Pa). Shape ``(endc - begc + 1,)``.
            Fortran: ``forc_pbot_downscaled_col(begc:endc)``.
        forc_lwrad_downscaled_col: Atmospheric longwave radiation
            downscaled to column (W/m2). Shape ``(endc - begc + 1,)``.
            Fortran: ``forc_lwrad_downscaled_col(begc:endc)``.
    """

    forc_u_grc: Array  # (n_grc,)
    forc_v_grc: Array  # (n_grc,)
    forc_pco2_grc: Array  # (n_grc,)
    forc_po2_grc: Array  # (n_grc,)
    forc_solad_downscaled_col: Array  # (n_col, numrad)
    forc_solai_grc: Array  # (n_grc, numrad)
    forc_t_downscaled_col: Array  # (n_col,)
    forc_pbot_downscaled_col: Array  # (n_col,)
    forc_lwrad_downscaled_col: Array  # (n_col,)


# ---------------------------------------------------------------------------
# Initialization helpers (mirror Init / InitAllocate)
# ---------------------------------------------------------------------------


def InitAllocate(bounds: bounds_type) -> atm2lnd_type:
    """
    Allocate and initialize an ``atm2lnd_type`` instance.

    Mirrors Fortran subroutine ``InitAllocate`` (lines 57-74).
    All arrays are filled with ``ival = 0.0`` matching the Fortran
    initialisation ``this%field(:) = ival``.

    Args:
        bounds: Decomposition bounds supplying ``begg``, ``endg``,
            ``begc``, and ``endc`` for this MPI task.

    Returns:
        A fully initialised :class:`atm2lnd_type` with every element
        set to ``0.0``.
    """
    ival = jnp.float64(0.0)  # Fortran line 59: ival = 0.0_r8

    begg = bounds.begg
    endg = bounds.endg  # Fortran lines 62-63
    begc = bounds.begc
    endc = bounds.endc  # Fortran lines 62-63

    n_grc = endg - begg + 1
    n_col = endc - begc + 1

    # Fortran: allocate(this%forc_u_grc(begg:endg)); ... = ival
    forc_u_grc = jnp.full((n_grc,), ival, dtype=jnp.float64)

    # Fortran: allocate(this%forc_v_grc(begg:endg)); ... = ival
    forc_v_grc = jnp.full((n_grc,), ival, dtype=jnp.float64)

    # Fortran: allocate(this%forc_pco2_grc(begg:endg)); ... = ival
    forc_pco2_grc = jnp.full((n_grc,), ival, dtype=jnp.float64)

    # Fortran: allocate(this%forc_po2_grc(begg:endg)); ... = ival
    forc_po2_grc = jnp.full((n_grc,), ival, dtype=jnp.float64)

    # Fortran: allocate(this%forc_solad_downscaled_col(begc:endc,numrad)); ... = ival
    # Size numrad+1 so that 1-based ivis=1, inir=2 indices are valid (slot 0 unused)
    forc_solad_downscaled_col = jnp.full((n_col, numrad + 1), ival, dtype=jnp.float64)

    # Fortran: allocate(this%forc_solai_grc(begg:endg,numrad)); ... = ival
    # Size numrad+1 so that 1-based ivis=1, inir=2 indices are valid (slot 0 unused)
    forc_solai_grc = jnp.full((n_grc, numrad + 1), ival, dtype=jnp.float64)

    # Fortran: allocate(this%forc_t_downscaled_col(begc:endc)); ... = ival
    forc_t_downscaled_col = jnp.full((n_col,), ival, dtype=jnp.float64)

    # Fortran: allocate(this%forc_pbot_downscaled_col(begc:endc)); ... = ival
    forc_pbot_downscaled_col = jnp.full((n_col,), ival, dtype=jnp.float64)

    # Fortran: allocate(this%forc_lwrad_downscaled_col(begc:endc)); ... = ival
    forc_lwrad_downscaled_col = jnp.full((n_col,), ival, dtype=jnp.float64)

    return atm2lnd_type(
        forc_u_grc=forc_u_grc,
        forc_v_grc=forc_v_grc,
        forc_pco2_grc=forc_pco2_grc,
        forc_po2_grc=forc_po2_grc,
        forc_solad_downscaled_col=forc_solad_downscaled_col,
        forc_solai_grc=forc_solai_grc,
        forc_t_downscaled_col=forc_t_downscaled_col,
        forc_pbot_downscaled_col=forc_pbot_downscaled_col,
        forc_lwrad_downscaled_col=forc_lwrad_downscaled_col,
    )


def Init(bounds: bounds_type) -> atm2lnd_type:
    """
    Initialize an ``atm2lnd_type`` instance.

    Mirrors Fortran subroutine ``Init`` (lines 50-55), which is the
    public entry point that delegates to ``InitAllocate``.

    Args:
        bounds: Decomposition bounds for the local MPI task.

    Returns:
        A fully initialised :class:`atm2lnd_type`.
    """
    return InitAllocate(bounds)
