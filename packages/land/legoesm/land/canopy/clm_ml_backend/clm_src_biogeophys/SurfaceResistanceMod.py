"""
JAX translation of SurfaceResistanceMod Fortran module.

Calculates resistance for soil evaporation using the Swenson-Lawrence
(2014) dry surface layer approach. Derives soil dry surface layer
thickness and soil evaporative resistance from soil hydraulic and
thermal properties.

Original Fortran module: SurfaceResistanceMod
Fortran lines 1-130
"""

import jax.numpy as jnp
import numpy as np

from legoesm.land.canopy.clm_ml_backend.clm_src_biogeophys.SoilStateType import soilstate_type  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_biogeophys.TemperatureType import temperature_type  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_biogeophys.WaterStateBulkType import waterstatebulk_type  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_main.abortutils import endrun  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_main.clm_varcon import denh2o, denice  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_main.clm_varpar import nlevsno  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_main.ColumnType import col  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_main.decompMod import bounds_type  # noqa: F401

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def calc_soilevap_resis(
    bounds: bounds_type,
    num_nolakec: int,
    filter_nolakec: np.ndarray,
    soilstate_inst: soilstate_type,
    waterstatebulk_inst: waterstatebulk_type,
    temperature_inst: temperature_type,
) -> soilstate_type:
    """
    Compute resistance for soil evaporation.

    Mirrors Fortran subroutine ``calc_soilevap_resis`` (lines 30-56).
    Delegates entirely to ``calc_soil_resistance_sl14`` and writes the
    resulting dry surface layer thickness and evaporative resistance back
    into the soil state container.

    Args:
        bounds: CLM column decomposition bounds.
        num_nolakec: Number of non-lake points in the column filter.
        filter_nolakec: 1-D array of non-lake column indices.
            Fortran: ``integer, intent(in) :: filter_nolakec(:)``.
        soilstate_inst: Soil state container; ``dsl_col`` and
            ``soilresis_col`` are updated and returned in a new instance.
        waterstatebulk_inst: Bulk water state container (read-only).
        temperature_inst: Temperature state container (read-only).

    Returns:
        Updated :class:`soilstate_type` with ``dsl_col`` and
        ``soilresis_col`` populated.
    """
    return calc_soil_resistance_sl14(
        bounds,
        num_nolakec,
        filter_nolakec,
        soilstate_inst,
        waterstatebulk_inst,
        temperature_inst,
    )


# ---------------------------------------------------------------------------
# Private: Swenson-Lawrence (2014) dry surface layer resistance
# ---------------------------------------------------------------------------


def calc_soil_resistance_sl14(
    bounds: bounds_type,
    num_nolakec: int,
    filter_nolakec: np.ndarray,
    soilstate_inst: soilstate_type,
    waterstatebulk_inst: waterstatebulk_type,
    temperature_inst: temperature_type,
) -> soilstate_type:
    """
    Compute soil dry surface layer thickness and evaporative resistance.

    Mirrors Fortran subroutine ``calc_soil_resistance_sl14``
    (lines 58-125).

    The following physics are applied per column (Fortran lines 96-112):

    .. code-block:: none

        vwc_liq     = max(h2osoi_liq(c,1), 1e-6) / (dz(c,1) * denh2o)
        eff_por_top = max(0.01, watsat(c,1)
                         - min(watsat(c,1), h2osoi_ice(c,1)/(dz(c,1)*denice)))
        aird        = watsat(c,1) * (sucsat(c,1)/1e7) ** (1/bsw(c,1))
        d0          = 2.12e-5 * (t_soisno(c,1)/273.15) ** 1.75
        eps         = watsat(c,1) - aird
        tort        = eps^2 * (eps/watsat(c,1)) ^ (3/max(3, bsw(c,1)))
        dsl(c)      = clamp(15 * max(0.001, 0.8*eff_por_top - vwc_liq)
                             / max(0.001, 0.8*watsat(c,1) - aird), 0, 200)
        soilresis(c)= min(dsl(c)/(d0*tort*1e3) + 20, 1e6)

    Args:
        bounds: CLM column decomposition bounds.
        num_nolakec: Number of non-lake points in the column filter.
        filter_nolakec: 1-D array of non-lake column indices.
        soilstate_inst: Soil state container (read-only inputs;
            ``dsl_col`` and ``soilresis_col`` written to new instance).
        waterstatebulk_inst: Bulk water state container (read-only).
        temperature_inst: Temperature state container (read-only).

    Returns:
        Updated :class:`soilstate_type` with ``dsl_col`` and
        ``soilresis_col`` populated for all non-lake columns.
    """
    # Unpack inputs (Fortran associate block, lines 82-93)
    watsat = soilstate_inst.watsat_col  # Porosity
    sucsat = soilstate_inst.sucsat_col  # Saturated suction (mm)
    bsw = soilstate_inst.bsw_col  # Clapp-Hornberger b parameter
    t_soisno = temperature_inst.t_soisno_col  # Soil temperature (K)
    h2osoi_ice = waterstatebulk_inst.h2osoi_ice_col  # Soil layer ice lens (kg H2O/m2)
    h2osoi_liq = waterstatebulk_inst.h2osoi_liq_col  # Soil layer liquid water (kg H2O/m2)
    dz = col.dz  # Soil layer thickness (m)

    dsl = soilstate_inst.dsl_col  # Dry surface layer thickness (mm)
    soilresis = soilstate_inst.soilresis_col  # Soil evaporative resistance (s/m)

    begc = bounds.begc

    for f in range(num_nolakec):  # Fortran: do f = 1, num_nolakec
        c = int(filter_nolakec[f])
        ci = c - begc  # 0-based column offset

        # Layer 1 quantities — clmData writes h2osoi_liq/ice at [ci, j-1]
        # for j=1..nlevgrnd, so the first soil layer is at index 0.
        # t_soisno is written by SoilInit at [c, j], so first layer is at index 1.

        # Liquid volumetric water content — Fortran line 97
        vwc_liq = jnp.maximum(h2osoi_liq[ci, 0], 1.0e-6) / (dz[c, 1] * denh2o)

        # Effective porosity of first layer — Fortran line 98
        eff_por_top = jnp.maximum(
            0.01,
            watsat[c, 1] - jnp.minimum(watsat[c, 1], h2osoi_ice[ci, 0] / (dz[c, 1] * denice)),
        )

        # Air-dry soil moisture — Fortran line 99
        aird = watsat[c, 1] * (sucsat[c, 1] / 1.0e7) ** (1.0 / bsw[c, 1])

        # Diffusivity of water vapor (m2/s) — Fortran line 100
        d0 = 2.12e-5 * (t_soisno[ci, 1] / 273.15) ** 1.75

        # Air-filled pore space and tortuosity — Fortran lines 101-102
        eps = watsat[c, 1] - aird
        tort = eps * eps * (eps / watsat[c, 1]) ** (3.0 / jnp.maximum(3.0, bsw[c, 1]))

        # Dry surface layer thickness (mm), clamped to [0, 200] — Fortran lines 103-105
        dsl_c = (
            15.0
            * jnp.maximum(0.001, 0.8 * eff_por_top - vwc_liq)
            / jnp.maximum(0.001, 0.8 * watsat[c, 1] - aird)
        )
        dsl_c = jnp.maximum(dsl_c, 0.0)
        dsl_c = jnp.minimum(dsl_c, 200.0)

        # Soil evaporative resistance (s/m), capped at 1e6 — Fortran lines 106-107
        soilresis_c = dsl_c / (d0 * tort * 1.0e3) + 20.0
        soilresis_c = jnp.minimum(1.0e6, soilresis_c)

        dsl = dsl.at[c].set(dsl_c)
        soilresis = soilresis.at[c].set(soilresis_c)

    return soilstate_inst._replace(
        dsl_col=dsl,
        soilresis_col=soilresis,
    )
