"""
JAX translation of SurfaceAlbedoMod Fortran module.

Calculates ground surface (soil) albedo for the CLM land surface model.
Provides time-constant initialization of soil color lookup tables and
per-timestep soil albedo computation based on volumetric water content.

Original Fortran module: SurfaceAlbedoMod
Fortran lines 1-130
"""

import jax.numpy as jnp
import numpy as np
from jax import Array

from legoesm.land.canopy.clm_ml_backend.clm_src_biogeophys.SurfaceAlbedoType import surfalb_type  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_biogeophys.WaterStateBulkType import waterstatebulk_type  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_main.abortutils import endrun  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_main.clm_varpar import inir, ivis, numrad  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_main.decompMod import bounds_type  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.offline_driver.TowerDataMod import tower_isoicol, tower_num  # noqa: F401

# ---------------------------------------------------------------------------
# Module-level private data (Fortran lines 22-25)
# Wet and dry soil albedo lookup tables, initialised by
# SurfaceAlbedoInitTimeConst and consumed by SoilAlbedo.
# ---------------------------------------------------------------------------

# albsat(mxsoil_color, numrad): Wet soil albedo by color class and waveband
# albdry(mxsoil_color, numrad): Dry soil albedo by color class and waveband
# isoicol(begc:endc):           Column soil color class
albsat: Array = jnp.empty((0, 0), dtype=jnp.float64)
albdry: Array = jnp.empty((0, 0), dtype=jnp.float64)
isoicol: Array = jnp.empty((0,), dtype=jnp.int32)


# ---------------------------------------------------------------------------
# Public: time-constant initialization
# ---------------------------------------------------------------------------


def SurfaceAlbedoInitTimeConst(bounds: bounds_type) -> None:
    """
    Initialize module time-constant variables for soil albedo.

    Populates the module-level ``albsat``, ``albdry``, and ``isoicol``
    arrays. Only ``mxsoil_color == 20`` (the CLM5 default) and
    ``mxsoil_color == 8`` (legacy) are supported; any other value
    triggers an abort.

    Mirrors Fortran subroutine ``SurfaceAlbedoInitTimeConst``
    (lines 35-90).

    Args:
        bounds: CLM column decomposition bounds, supplying ``begc``
            and ``endc``.
    """
    global albsat, albdry, isoicol

    begc = bounds.begc
    endc = bounds.endc

    # Assign soil color class for every column — Fortran lines 45-48
    n_col = endc - begc + 1
    isoicol = jnp.full((n_col,), int(tower_isoicol[tower_num]), dtype=jnp.int32)

    # Saturated and dry soil albedo tables — Fortran lines 51-88
    mxsoil_color = 20

    if mxsoil_color == 8:
        # 8-class tables — Fortran lines 56-60
        # Size numrad+1: slot 0 unused; ivis=1 and inir=2 are valid 1-based indices
        _albsat = jnp.zeros((mxsoil_color, numrad + 1), dtype=jnp.float64)
        _albdry = jnp.zeros((mxsoil_color, numrad + 1), dtype=jnp.float64)

        _albsat = _albsat.at[:, ivis].set(
            jnp.array([0.12, 0.11, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05], dtype=jnp.float64)
        )
        _albsat = _albsat.at[:, inir].set(
            jnp.array([0.24, 0.22, 0.20, 0.18, 0.16, 0.14, 0.12, 0.10], dtype=jnp.float64)
        )
        _albdry = _albdry.at[:, ivis].set(
            jnp.array([0.24, 0.22, 0.20, 0.18, 0.16, 0.14, 0.12, 0.10], dtype=jnp.float64)
        )
        _albdry = _albdry.at[:, inir].set(
            jnp.array([0.48, 0.44, 0.40, 0.36, 0.32, 0.28, 0.24, 0.20], dtype=jnp.float64)
        )

    elif mxsoil_color == 20:
        # 20-class tables (CLM5 default) — Fortran lines 61-77
        # Size numrad+1: slot 0 unused; ivis=1 and inir=2 are valid 1-based indices
        _albsat = jnp.zeros((mxsoil_color, numrad + 1), dtype=jnp.float64)
        _albdry = jnp.zeros((mxsoil_color, numrad + 1), dtype=jnp.float64)

        _albsat = _albsat.at[:, ivis].set(
            jnp.array(
                [
                    0.25,
                    0.23,
                    0.21,
                    0.20,
                    0.19,
                    0.18,
                    0.17,
                    0.16,
                    0.15,
                    0.14,
                    0.13,
                    0.12,
                    0.11,
                    0.10,
                    0.09,
                    0.08,
                    0.07,
                    0.06,
                    0.05,
                    0.04,
                ],
                dtype=jnp.float64,
            )
        )
        _albsat = _albsat.at[:, inir].set(
            jnp.array(
                [
                    0.50,
                    0.46,
                    0.42,
                    0.40,
                    0.38,
                    0.36,
                    0.34,
                    0.32,
                    0.30,
                    0.28,
                    0.26,
                    0.24,
                    0.22,
                    0.20,
                    0.18,
                    0.16,
                    0.14,
                    0.12,
                    0.10,
                    0.08,
                ],
                dtype=jnp.float64,
            )
        )
        _albdry = _albdry.at[:, ivis].set(
            jnp.array(
                [
                    0.36,
                    0.34,
                    0.32,
                    0.31,
                    0.30,
                    0.29,
                    0.28,
                    0.27,
                    0.26,
                    0.25,
                    0.24,
                    0.23,
                    0.22,
                    0.20,
                    0.18,
                    0.16,
                    0.14,
                    0.12,
                    0.10,
                    0.08,
                ],
                dtype=jnp.float64,
            )
        )
        _albdry = _albdry.at[:, inir].set(
            jnp.array(
                [
                    0.61,
                    0.57,
                    0.53,
                    0.51,
                    0.49,
                    0.48,
                    0.45,
                    0.43,
                    0.41,
                    0.39,
                    0.37,
                    0.35,
                    0.33,
                    0.31,
                    0.29,
                    0.27,
                    0.25,
                    0.23,
                    0.21,
                    0.16,
                ],
                dtype=jnp.float64,
            )
        )

    else:
        endrun(msg=" ERROR: SurfaceAlbedoInitTimeConst: maximum color class is not supported")
        # Unreachable; satisfies static type checkers
        _albsat = jnp.empty((0, 0), dtype=jnp.float64)
        _albdry = jnp.empty((0, 0), dtype=jnp.float64)

    albsat = _albsat
    albdry = _albdry


# ---------------------------------------------------------------------------
# Public: per-timestep soil albedo
# ---------------------------------------------------------------------------


def SoilAlbedo(
    bounds: bounds_type,
    num_nourbanc: int,
    filter_nourbanc: np.ndarray,
    waterstatebulk_inst: waterstatebulk_type,
    surfalb_inst: surfalb_type,
) -> surfalb_type:
    """
    Compute ground surface (soil) albedo for non-urban columns.

    Mirrors Fortran subroutine ``SoilAlbedo`` (lines 92-128).

    Soil albedo is computed per waveband ``ib`` (1 = vis, 2 = nir) as:

    .. code-block:: none

        inc        = max(0.11 - 0.40 * h2osoi_vol(c,1), 0)
        albsoib(c,ib) = min(albsat(soilcol,ib) + inc, albdry(soilcol,ib))
        albsoid(c,ib) = albsoib(c,ib)

    where ``soilcol = isoicol(c)`` is the soil color class of the column
    and ``h2osoi_vol(c,1)`` is the top-layer volumetric water content.
    Both direct-beam and diffuse albedos are set equal.

    Args:
        bounds: CLM column decomposition bounds.
        num_nourbanc: Number of non-urban points in the column filter.
        filter_nourbanc: 1-D array of non-urban column indices.
            Fortran: ``integer, intent(in) :: filter_nourbanc(:)``.
        waterstatebulk_inst: Bulk water state container supplying
            ``h2osoi_vol_col`` (read-only).
        surfalb_inst: Surface albedo container; ``albgrd_col`` and
            ``albgri_col`` are updated and returned in a new instance.

    Returns:
        Updated :class:`surfalb_type` with direct-beam and diffuse
        soil albedos populated.
    """
    # Unpack inputs (Fortran associate block, lines 112-117)
    h2osoi_vol = waterstatebulk_inst.h2osoi_vol_col  # Volumetric water content (m3/m3)

    albsoib = surfalb_inst.albgrd_col  # Direct beam albedo of ground (soil)
    albsoid = surfalb_inst.albgri_col  # Diffuse albedo of ground (soil)

    begc = bounds.begc

    # Calculate soil albedo — Fortran lines 120-129
    for ib in range(1, numrad + 1):  # Fortran: do ib = 1, numrad (1-based)
        for f in range(num_nourbanc):  # Fortran: do f = 1, num_nourbanc
            c = int(filter_nourbanc[f])
            ci = c - begc  # 0-based offset into isoicol

            soilcol = int(isoicol[ci])  # Soil color class (1-based into albsat/albdry)

            # Soil water correction — Fortran line 125
            # h2osoi_vol_col uses j=1..nlevgrnd indexing (SoilInit/clmDataMod),
            # so first soil layer is at index 1 (index 0 is unused/nan)
            inc = jnp.maximum(0.11 - 0.40 * h2osoi_vol[ci, 1], 0.0)

            # Albedo bounded by wet (albsat) and dry (albdry) limits
            # Fortran lines 126-127
            alb_val = jnp.minimum(
                albsat[soilcol - 1, ib] + inc,
                albdry[soilcol - 1, ib],
            )
            albsoib = albsoib.at[c, ib].set(alb_val)
            albsoid = albsoid.at[c, ib].set(alb_val)  # diffuse = direct

    return surfalb_inst._replace(
        albgrd_col=albsoib,
        albgri_col=albsoid,
    )
