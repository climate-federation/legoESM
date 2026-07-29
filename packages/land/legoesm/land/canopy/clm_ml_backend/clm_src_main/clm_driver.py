"""
JAX translation of clm_driver Fortran module.

Main CLM model driver to calculate surface fluxes. Orchestrates the
full physics sequence for a single driver time step, including soil
albedo, evaporative resistance, thermal conductivity, soil water
movement, multilayer canopy fluxes, and soil temperature updates.

Original Fortran module: clm_driver
Fortran lines 1-110
"""

import jax.numpy as jnp
from jax import Array

import legoesm.land.canopy.clm_ml_backend.clm_src_main.clm_instMod as clm_instMod  # noqa: F401
import legoesm.land.canopy.clm_ml_backend.offline_driver.clmSoilOptionMod as _clmSoilOpt  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_biogeophys.SoilWaterMovementMod import SoilWater  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_biogeophys.SurfaceAlbedoMod import SoilAlbedo  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_biogeophys.SurfaceResistanceMod import calc_soilevap_resis  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_main.abortutils import endrun  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_main.clm_varpar import nlevgrnd, nlevsno  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_main.ColumnType import col  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_main.decompMod import bounds_type  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_main.filterMod import filter, setExposedvegpFilter  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLCanopyFluxesMod import MLCanopyFluxes  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLSoilTemperatureMod import SoilTemperature, SoilThermProp  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.offline_driver.clmDataMod import clmData  # noqa: F401

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def clm_drv(
    bounds: bounds_type,
    time_indx: int,
    fin1: str,
    fin2: str,
) -> None:
    """
    Main CLM model driver to calculate surface fluxes.

    Mirrors Fortran subroutine ``clm_drv`` (lines 22-105).

    Executes the following physics sequence for a single driver
    time step:

    1. Read CLM data for the current time slice (``clmData``).
    2. Set fractional vegetation and its filter
       (``setExposedvegpFilter``).
    3. Compute soil albedo (``SoilAlbedo``).
    4. Compute soil evaporative resistance
       (``calc_soilevap_resis``).
    5. Zero out snow and surface water for non-lake columns.
    6. Compute soil thermal conductivity and heat capacity
       (``SoilThermProp``).
    7. Compute soil hydraulic conductivity and matric potential
       (``SoilWater``).
    8. Compute multilayer canopy and soil fluxes
       (``MLCanopyFluxes``).
    9. Update soil temperatures (``SoilTemperature``).

    .. warning::
        The soil temperature routine (``SoilTemperature``) is specific
        to the multilayer canopy and is **not** the same as in
        standard CLM. Fortran lines 96-100.

    Args:
        bounds: CLM decomposition bounds for the local MPI task,
            supplying ``begp``, ``endp``, ``begc``, and ``endc``.
        time_indx: Time index from the reference date (0Z January 1
            of the current year, where ``calday == 1.000``).
            Fortran: ``integer, intent(in) :: time_indx``.
        fin1: Path to the first required input file (≤256 characters).
            Fortran: ``character(len=256) :: fin1``.
        fin2: Path to the second required input file (≤256 characters).
            Fortran: ``character(len=256) :: fin2``.
    """
    # ------------------------------------------------------------------
    # Read all state instances live from clm_instMod to avoid stale
    # module-level aliases (singletons are rebound during initialize2)
    # ------------------------------------------------------------------

    canopystate_inst = clm_instMod.canopystate_inst
    waterdiagnosticbulk_inst = clm_instMod.waterdiagnosticbulk_inst
    water_inst = clm_instMod.water_inst
    waterstatebulk_inst = clm_instMod.waterstatebulk_inst
    soilstate_inst = clm_instMod.soilstate_inst
    temperature_inst = clm_instMod.temperature_inst
    atm2lnd_inst = clm_instMod.atm2lnd_inst
    waterfluxbulk_inst = clm_instMod.waterfluxbulk_inst
    energyflux_inst = clm_instMod.energyflux_inst
    frictionvel_inst = clm_instMod.frictionvel_inst
    surfalb_inst = clm_instMod.surfalb_inst
    solarabs_inst = clm_instMod.solarabs_inst
    mlcanopy_inst = clm_instMod.mlcanopy_inst
    wateratm2lndbulk_inst = clm_instMod.wateratm2lndbulk_inst

    # ------------------------------------------------------------------
    # Unpack associate aliases (Fortran lines 47-54)
    # ------------------------------------------------------------------
    snl: Array = col.snl  # Number of snow layers
    frac_veg_nosno: Array = (
        canopystate_inst.frac_veg_nosno_patch
    )  # Fraction of veg not covered by snow (0 or 1)
    frac_sno_eff: Array = (
        waterdiagnosticbulk_inst.frac_sno_eff_col
    )  # Effective fraction of ground covered by snow
    h2osno: Array = water_inst.h2osno_col  # Total snow water (kg H2O/m2)
    h2osfc: Array = waterstatebulk_inst.h2osfc_col  # Surface water (kg H2O/m2)

    # Local work arrays for soil thermal properties — Fortran lines 41-43
    # Shape: (n_col, nlevsno + nlevgrnd + 1) — the +1 is needed because
    # SoilThermProp uses jpy = j + nlevsno mapping j from -nlevsno+1..nlevgrnd
    # giving jpy from 1..nlevsno+nlevgrnd (0-based needs size nlevsno+nlevgrnd+1)
    n_col = bounds.endc - bounds.begc + 1
    n_lev = (
        nlevsno + nlevgrnd + 1
    )  # Fortran: -nlevsno+1:nlevgrnd → size nlevsno+nlevgrnd, offset by 1
    cv = jnp.zeros((n_col, n_lev), dtype=jnp.float64)  # Soil heat capacity (J/m2/K)
    tk = jnp.zeros((n_col, n_lev), dtype=jnp.float64)  # Soil thermal conductivity (W/m/K)
    tk_h2osfc = jnp.zeros((n_col,), dtype=jnp.float64)  # Thermal conductivity of h2osfc (W/m/K)

    # ------------------------------------------------------------------
    # 1. Read CLM data for current time slice — Fortran lines 57-59
    # fin2 = CLM history file; fin_soil_adjust read from clmSoilOptionMod
    # ------------------------------------------------------------------
    _fin_soil_adjust = _clmSoilOpt.fin_soil_adjust
    waterstatebulk_inst, canopystate_inst, surfalb_inst = clmData(
        fin2,
        _fin_soil_adjust,
        time_indx,
        bounds.begp,
        bounds.endp,
        bounds.begc,
        bounds.endc,
        soilstate_inst,
        waterstatebulk_inst,
        canopystate_inst,
        surfalb_inst,
    )

    # ------------------------------------------------------------------
    # 2. Set frac_veg_nosno and exposedvegp filter — Fortran lines 61-64
    # ------------------------------------------------------------------
    for p in range(bounds.begp, bounds.endp + 1):  # Fortran: do p = bounds%begp, bounds%endp
        frac_veg_nosno = frac_veg_nosno.at[p].set(1)

    # setExposedvegpFilter returns a new NamedTuple; capture it so
    # the rest of clm_drv() sees num_exposedvegp = 1.
    # Using `global filter` avoids Python's UnboundLocalError (assigning to
    # a name in a function makes it local *throughout* the function).
    global filter
    filter = setExposedvegpFilter(filter, frac_veg_nosno)

    # ------------------------------------------------------------------
    # 3. Soil albedo — Fortran lines 66-67
    # ------------------------------------------------------------------
    assert filter.nourbanc is not None, "filter.nourbanc must be initialized"
    surfalb_inst = SoilAlbedo(
        bounds,
        filter.num_nourbanc,
        filter.nourbanc,
        waterstatebulk_inst,
        surfalb_inst,
    )

    # ------------------------------------------------------------------
    # 4. Soil evaporative resistance — Fortran lines 69-71
    # ------------------------------------------------------------------
    assert filter.nolakec is not None, "filter.nolakec must be initialized"
    soilstate_inst = calc_soilevap_resis(
        bounds,
        filter.num_nolakec,
        filter.nolakec,
        soilstate_inst,
        waterstatebulk_inst,
        temperature_inst,
    )

    # ------------------------------------------------------------------
    # 5. Zero out snow and surface water — Fortran lines 73-79
    # ------------------------------------------------------------------
    for f in range(filter.num_nolakec):  # Fortran: do f = 1, filter%num_nolakec
        c = int(filter.nolakec[f])
        snl = snl.at[c].set(0)
        frac_sno_eff = frac_sno_eff.at[c - bounds.begc].set(0.0)
        h2osno = h2osno.at[c - bounds.begc].set(0.0)
        h2osfc = h2osfc.at[c - bounds.begc].set(0.0)

    # Write zeroed arrays back into their containers
    waterdiagnosticbulk_inst = waterdiagnosticbulk_inst._replace(frac_sno_eff_col=frac_sno_eff)
    water_inst = water_inst._replace(h2osno_col=h2osno)
    waterstatebulk_inst = waterstatebulk_inst._replace(h2osfc_col=h2osfc)

    # ------------------------------------------------------------------
    # 6. Soil thermal conductivity and heat capacity — Fortran lines 83-87
    # ------------------------------------------------------------------
    assert filter.nolakec is not None, "filter.nolakec must be initialized"
    soilstate_inst, temperature_inst, tk, cv, tk_h2osfc = SoilThermProp(
        bounds,
        filter.num_nolakec,
        filter.nolakec,
        tk,
        cv,
        tk_h2osfc,
        temperature_inst,
        waterdiagnosticbulk_inst,
        waterstatebulk_inst,
        water_inst,
        soilstate_inst,
    )

    # ------------------------------------------------------------------
    # 7. Soil hydraulic conductivity and matric potential — Fortran lines 89-91
    # ------------------------------------------------------------------
    assert filter.hydrologyc is not None, "filter.hydrologyc must be initialized"
    soilstate_inst = SoilWater(
        bounds,
        filter.num_hydrologyc,
        filter.hydrologyc,
        soilstate_inst,
        waterstatebulk_inst,
    )

    # ------------------------------------------------------------------
    # 8. Multilayer canopy and soil fluxes — Fortran lines 93-99
    # ------------------------------------------------------------------
    assert filter.exposedvegp is not None, "filter.exposedvegp must be initialized"
    # Convert JAX array to list for MLCanopyFluxes signature
    exposedvegp_list = [int(p) for p in filter.exposedvegp[: filter.num_exposedvegp]]
    mlcanopy_inst = MLCanopyFluxes(
        bounds,
        filter.num_exposedvegp,
        exposedvegp_list,
        atm2lnd_inst,
        canopystate_inst,
        soilstate_inst,
        temperature_inst,
        waterstatebulk_inst,
        waterfluxbulk_inst,
        energyflux_inst,
        frictionvel_inst,
        surfalb_inst,
        solarabs_inst,
        mlcanopy_inst,
        wateratm2lndbulk_inst,
        waterdiagnosticbulk_inst,
    )

    # ------------------------------------------------------------------
    # 9. Update soil temperatures — Fortran lines 101-104
    # WARNING: specific to multilayer canopy, not standard CLM
    # ------------------------------------------------------------------
    assert filter.nolakec is not None, "filter.nolakec must be initialized"
    temperature_inst, soilstate_inst = SoilTemperature(
        bounds,
        filter.num_nolakec,
        filter.nolakec,
        soilstate_inst,
        temperature_inst,
        waterdiagnosticbulk_inst,
        waterstatebulk_inst,
        water_inst,
        mlcanopy_inst,
    )

    # ------------------------------------------------------------------
    # Write all mutated state back to clm_instMod so changes persist
    # across timesteps (local variables are copies of the module refs)
    # ------------------------------------------------------------------
    clm_instMod.canopystate_inst = canopystate_inst
    clm_instMod.waterdiagnosticbulk_inst = waterdiagnosticbulk_inst
    clm_instMod.water_inst = water_inst
    clm_instMod.waterstatebulk_inst = waterstatebulk_inst
    clm_instMod.soilstate_inst = soilstate_inst
    clm_instMod.temperature_inst = temperature_inst
    clm_instMod.waterfluxbulk_inst = waterfluxbulk_inst
    clm_instMod.energyflux_inst = energyflux_inst
    clm_instMod.frictionvel_inst = frictionvel_inst
    clm_instMod.surfalb_inst = surfalb_inst
    clm_instMod.solarabs_inst = solarabs_inst
    clm_instMod.mlcanopy_inst = mlcanopy_inst
