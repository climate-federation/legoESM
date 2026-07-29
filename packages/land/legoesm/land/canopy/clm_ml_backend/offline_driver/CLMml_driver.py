"""
JAX translation of CLMml_driver Fortran module.

Top-level model driver for the CLMml (multilayer canopy) offline
tower-site simulation. Orchestrates the full run sequence: namelist
parsing, CLM initialization, orbital parameters, acclimation
temperature calculation, vegetation and soil initialization, output
file management, and the main time-stepping loop.

Original Fortran module: CLMml_driver
Fortran lines 1-200
"""

from __future__ import annotations

import math
import os
from typing import IO

import jax.numpy as jnp

from legoesm.land.canopy.clm_ml_backend.clm_share.shr_orb_mod import shr_orb_params  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_biogeophys.CanopyStateType import canopystate_type  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_biogeophys.FrictionVelocityMod import frictionvel_type  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_biogeophys.SoilStateType import soilstate_type  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_biogeophys.TemperatureType import temperature_type  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_biogeophys.WaterStateBulkType import waterstatebulk_type  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_cpl.lnd_comp_nuopc import InitializeRealize, ModelAdvance, bounds_type  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_main import clm_instMod  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_main.abortutils import endrun  # noqa: F401

# Type imports for function signatures
from legoesm.land.canopy.clm_ml_backend.clm_src_main.atm2lndType import atm2lnd_type  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_main.clm_instMod import (  # noqa: F401
    atm2lnd_inst,
    canopystate_inst,
    frictionvel_inst,
    mlcanopy_inst,
    soilstate_inst,
    temperature_inst,
    wateratm2lndbulk_inst,
    waterstatebulk_inst,
)
from legoesm.land.canopy.clm_ml_backend.clm_src_main.clm_varctl import iulog  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_main.filterMod import filter, setFilters  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_main.PatchType import patch  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_main.wateratm2lndBulkType import wateratm2lndbulk_type  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_utils import clm_time_manager  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_utils.clm_time_manager import (  # noqa: F401
    curr_date_tod,
    dtstep,
    get_curr_calday,
    get_curr_date,
    get_curr_time,
    itim,
    start_date_tod,
    start_date_ymd,
)
from legoesm.land.canopy.clm_ml_backend.clm_src_utils.clm_varorb import eccen, lambm0, mvelpp, obliqr  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLCanopyFluxesType import mlcanopy_type  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLclm_varctl import flux_profile_type, met_type  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.offline_driver.controlMod import control  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.offline_driver.TowerDataMod import tower_id, tower_num  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.offline_driver.TowerMetMod import TowerMetCurr, TowerMetNext  # noqa: F401

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def CLMml_drv(bounds: bounds_type) -> None:
    """
    Top-level model driver: process one tower site and year.

    Mirrors Fortran subroutine ``CLMml_drv`` (lines 30-185).

    Execution sequence
    ------------------
    1. Parse namelist run control variables via :func:`controlMod.control`.
    2. Extract year/month/day from ``start_date_ymd`` and log the
       tower site being processed.
    3. Initialize CLM via :func:`InitializeRealize` and build filters.
    4. Compute orbital parameters for the current year.
    5. Read tower meteorology once to derive the acclimation temperature
       (:func:`init_acclim`).
    6. Initialize tower vegetation (:func:`TowerVeg`).
    7. Compute the correct CLM history file time slice index and
       initialize soil temperature and moisture (:func:`SoilInit`).
    8. Open six ASCII output files (flux, auxiliary, profile, sun/shade
       flux, vertical flux profile, soil temperature) and optionally
       an ASCII profile input file.
    9. Run the main time-stepping loop (``itim = 1, ntim``):
       - Update date/time bookkeeping.
       - Recalculate CLM history file slice index.
       - Read current-step tower meteorology (:func:`TowerMetCurr`).
       - Optionally read next-step meteorology for 3-point interpolation
         (:func:`TowerMetNext`, active when ``met_type == 3``).
       - Optionally read canopy T/Q/U profile data.
       - Advance the model (:func:`ModelAdvance`).
       - Write output (:func:`output`).
    10. Close and release all file handles.

    Args:
        bounds: Decomposition bounds supplying ``begp``, ``endp``,
            ``begc``, ``endc`` for the local task.
    """
    # ------------------------------------------------------------------
    # Initialize namelist run control variables — Fortran line 55
    # ------------------------------------------------------------------
    ntim, clm_start_ymd, clm_start_tod, fin_tower, fin_clm, fin_soil_adjust, dirout = control()

    # ------------------------------------------------------------------
    # Get current date from start_date_ymd — Fortran lines 63-65
    # ------------------------------------------------------------------
    clm_time_manager.itim = 1
    yr, mon, day, _ = get_curr_date()

    print(f"{iulog}: Processing: {tower_id[tower_num]} {yr} {mon}")

    # ------------------------------------------------------------------
    # Initialize CLM and build filters — Fortran lines 73-77
    # ------------------------------------------------------------------
    InitializeRealize(bounds)
    setFilters(filter)

    # ------------------------------------------------------------------
    # Orbital parameters for this year — Fortran lines 80-81
    # ------------------------------------------------------------------

    obliq, mvelp = shr_orb_params(yr)  # obliq, mvelp are local (not used further)

    # ------------------------------------------------------------------
    # Acclimation temperature from full tower record — Fortran lines 84-87
    # ------------------------------------------------------------------
    init_acclim(
        fin_tower,
        tower_num,
        ntim,
        bounds.begp,
        bounds.endp,
        clm_instMod.atm2lnd_inst,
        clm_instMod.wateratm2lndbulk_inst,
        clm_instMod.temperature_inst,
        clm_instMod.frictionvel_inst,
        clm_instMod.mlcanopy_inst,
    )

    # ------------------------------------------------------------------
    # Initialize tower vegetation — Fortran lines 91-92
    # ------------------------------------------------------------------
    TowerVeg(
        tower_num,
        bounds.begp,
        bounds.endp,
        clm_instMod.canopystate_inst,
        clm_instMod.mlcanopy_inst,
    )

    # ------------------------------------------------------------------
    # Compute CLM history file calendar day and time slice index
    # Fortran lines 95-120
    #
    # NOTE: This is described as a "hack" in the Fortran source
    # (lines 97-100). start_date_ymd/tod are temporarily overwritten
    # with the CLM history file start date so that get_curr_calday
    # returns the CLM file's calendar day rather than the run start.
    # ------------------------------------------------------------------

    # Save run start date/time — Fortran lines 103-104
    run_start_date = clm_time_manager.start_date_ymd
    run_start_tod = clm_time_manager.start_date_tod

    # Temporarily set to CLM history start to compute start_calday_clm — Fortran lines 106-109
    clm_time_manager.start_date_ymd = clm_start_ymd
    clm_time_manager.start_date_tod = clm_start_tod
    clm_time_manager.itim = 1
    start_calday_clm = get_curr_calday(offset=0)

    # Restore run start date/time — Fortran lines 111-112
    clm_time_manager.start_date_ymd = run_start_date
    clm_time_manager.start_date_tod = run_start_tod
    clm_time_manager.itim = 1
    curr_calday = get_curr_calday(offset=0)

    # Time slice index into CLM history file — Fortran line 116
    time_indx = round((curr_calday - start_calday_clm) * 86400.0 / float(dtstep)) + 1

    # Initialize soil temperature and moisture — Fortran lines 118-120
    SoilInit(
        fin_clm,
        time_indx,
        bounds.begc,
        bounds.endc,
        clm_instMod.soilstate_inst,
        clm_instMod.waterstatebulk_inst,
        clm_instMod.temperature_inst,
    )

    # ------------------------------------------------------------------
    # Open ASCII output files — Fortran lines 123-153
    # Output file names follow the pattern:
    #   {tower_id}_{yyyy}-{mm}_{tag}.out
    # ------------------------------------------------------------------
    tid = tower_id[tower_num]

    def _outpath(tag: str) -> str:
        """Build full output file path matching Fortran write/format."""
        fname = f"{tid}_{yr:04d}-{mon:02d}_{tag}.out"
        return os.path.join(dirout.strip(), fname)

    fout1 = open(_outpath("flux"), "w")  # Fluxes
    fout2 = open(_outpath("aux"), "w")  # Auxiliary data
    fout3 = open(_outpath("profile"), "w")  # Profile data
    fout4 = open(_outpath("fsun"), "w")  # Sun/shade fluxes
    fout5 = open(_outpath("fluxprofile"), "w")  # Vertical flux profiles
    fout6 = open(_outpath("soiltemp"), "w")  # Soil temperature

    # ------------------------------------------------------------------
    # Optionally open ASCII profile input file — Fortran lines 155-162
    # ------------------------------------------------------------------
    fin1: IO | None = None
    if flux_profile_type == -1:
        endrun(msg=" ERROR: flux_profile_type not supported")
        # Lines below are unreachable; preserved from Fortran lines 157-161
        fin1_path = "set_file_name"
        fin1 = open(fin1_path, "r")

    # ------------------------------------------------------------------
    # Main time-stepping loop — Fortran lines 165-183
    # ------------------------------------------------------------------
    print(f"{iulog}: Starting time stepping loop .....")

    for _itim in range(1, ntim + 1):  # Fortran: do itim = 1, ntim
        clm_time_manager.itim = _itim

        # Date, time, and calendar day — Fortran lines 171-173
        yr, mon, day, _ = get_curr_date()
        curr_time_day, curr_time_sec = get_curr_time()
        curr_calday = get_curr_calday(offset=0)

        # Time slice into CLM history file — Fortran line 176
        time_indx = round((curr_calday - start_calday_clm) * 86400.0 / float(dtstep)) + 1

        # Read tower meteorology for current time step — Fortran lines 178-180
        (
            clm_instMod.atm2lnd_inst,
            clm_instMod.wateratm2lndbulk_inst,
            clm_instMod.frictionvel_inst,
        ) = TowerMetCurr(
            fin_tower,
            _itim,
            tower_num,
            bounds.begp,
            bounds.endp,
            clm_instMod.atm2lnd_inst,
            clm_instMod.wateratm2lndbulk_inst,
            clm_instMod.frictionvel_inst,
        )

        # Read next-step meteorology for 3-point interpolation — Fortran lines 182-185
        if met_type == 3:
            itim_next = min(_itim + 1, ntim)  # Fortran: itim_next = min(itim+1, ntim)
            clm_instMod.mlcanopy_inst = TowerMetNext(
                fin_tower,
                itim_next,
                bounds.begp,
                bounds.endp,
                clm_instMod.mlcanopy_inst,
            )

        # Read canopy profile data if required — Fortran line 188
        if flux_profile_type == -1:
            if fin1 is None:
                endrun(msg=" ERROR: flux_profile_type -1 requires profile input file")
            assert fin1 is not None  # Type narrowing for static analysis
            clm_instMod.mlcanopy_inst = ReadCanopyProfiles(
                _itim, curr_calday, fin1, clm_instMod.mlcanopy_inst
            )

        # Advance the model — Fortran lines 190-192
        ModelAdvance(bounds, time_indx, fin_clm, fin_soil_adjust)
        if _itim == 1:
            print(f"{iulog}: Executing model .....")

        # Write output — Fortran lines 194-196
        output(
            curr_calday,
            tower_num,
            fout1,
            fout2,
            fout3,
            fout4,
            fout5,
            fout6,
            clm_instMod.mlcanopy_inst,
            clm_instMod.temperature_inst,
        )

    # ------------------------------------------------------------------
    # Close output and input files — Fortran lines 198-209
    # ------------------------------------------------------------------
    for fh in (fout1, fout2, fout3, fout4, fout5, fout6):
        fh.close()

    if flux_profile_type == -1 and fin1 is not None:
        fin1.close()

    print(f"{iulog}: Successfully finished simulation")


def init_acclim(
    fin: str,
    tower_num: int,
    ntim: int,
    begp: int,
    endp: int,
    atm2lnd_inst: atm2lnd_type,  # noqa: F811
    wateratm2lndbulk_inst: wateratm2lndbulk_type,  # noqa: F811
    temperature_inst: temperature_type,  # noqa: F811
    frictionvel_inst: frictionvel_type,  # noqa: F811
    mlcanopy_inst: mlcanopy_type,  # noqa: F811
) -> tuple[atm2lnd_type, wateratm2lndbulk_type, temperature_type, frictionvel_type, mlcanopy_type]:
    """
    Read tower meteorology data once over all time slices to derive
    the mean acclimation temperature for each patch.

    Mirrors Fortran subroutine ``init_acclim`` (private to
    ``CLMml_driver``).

    For each patch ``p`` in ``[begp, endp]``, ``t_a10_patch`` is
    initialised to zero, then accumulated with ``forc_t_downscaled_col``
    at the patch's column for every time slice, and finally divided by
    ``ntim`` to yield the time-mean air temperature used by the
    acclimation scheme.

    Additionally, ``pref_forcing`` in the multilayer canopy container is
    set to ``forc_pbot_downscaled_col`` at the first time slice for each
    patch. This pressure value is used only when reading Q vertical
    profiles from an external dataset.

    Args:
        fin: Path to the tower meteorology netCDF file.
        tower_num: Tower site index into ``TowerDataMod`` arrays.
        ntim: Total number of time slices to process.
        begp: First patch index.
        endp: Last patch index.
        atm2lnd_inst: Atmosphere-to-land forcing container.
            ``forc_t_downscaled_col`` and ``forc_pbot_downscaled_col``
            are read (updated internally by :func:`TowerMetCurr`).
        wateratm2lndbulk_inst: Bulk atm-to-land water forcing container
            (passed through to :func:`TowerMetCurr`).
        temperature_inst: Temperature state container;
            ``t_a10_patch`` is accumulated and averaged in-place.
        frictionvel_inst: Friction velocity container (passed through
            to :func:`TowerMetCurr`).
        mlcanopy_inst: Multilayer canopy container;
            ``pref_forcing`` is set from the first time slice.

    Returns:
        Tuple of updated ``(atm2lnd_inst, wateratm2lndbulk_inst,
        temperature_inst, frictionvel_inst, mlcanopy_inst)``.
    """
    # Unpack mutable arrays (Fortran associate block)
    t10 = temperature_inst.t_a10_patch  # Acclimation temperature (K)
    pref = mlcanopy_inst.pref_forcing  # Air pressure at reference height (Pa)

    # ------------------------------------------------------------------
    # Initialize accumulator to zero — Fortran lines: do p = begp, endp; t10(p) = 0
    # ------------------------------------------------------------------
    for p in range(begp, endp + 1):
        t10 = t10.at[p].set(0.0)

    # ------------------------------------------------------------------
    # Loop over all time slices — Fortran: do itim = 1, ntim
    # ------------------------------------------------------------------
    for _itim in range(1, ntim + 1):

        # Read temperature for this time slice — Fortran lines: call TowerMetCurr(...)
        atm2lnd_inst, wateratm2lndbulk_inst, frictionvel_inst = TowerMetCurr(
            fin,
            _itim,
            tower_num,
            begp,
            endp,
            atm2lnd_inst,
            wateratm2lndbulk_inst,
            frictionvel_inst,
        )

        # Re-bind after TowerMetCurr returns updated instances
        forc_t = atm2lnd_inst.forc_t_downscaled_col
        forc_pbot = atm2lnd_inst.forc_pbot_downscaled_col

        for p in range(begp, endp + 1):  # Fortran: do p = begp, endp
            c = int(patch.column[p])

            # Accumulate temperature — Fortran: t10(p) = t10(p) + forc_t(c)
            t10 = t10.at[p].add(float(forc_t[c]))

            # Save pressure from first time slice — Fortran: if (itim == 1) pref(p) = forc_pbot(c)
            if _itim == 1:
                pref = pref.at[p].set(float(forc_pbot[c]))

    # ------------------------------------------------------------------
    # Average over all time slices — Fortran: t10(p) = t10(p) / float(ntim)
    # ------------------------------------------------------------------
    for p in range(begp, endp + 1):
        t10 = t10.at[p].set(float(t10[p]) / float(ntim))

    return (
        atm2lnd_inst,
        wateratm2lndbulk_inst,
        temperature_inst._replace(t_a10_patch=t10),
        frictionvel_inst,
        mlcanopy_inst._replace(pref_forcing=pref),
    )


def TowerVeg(
    it: int,
    begp: int,
    endp: int,
    canopystate_inst: canopystate_type,  # noqa: F811
    mlcanopy_inst: mlcanopy_type,  # noqa: F811
) -> tuple[canopystate_type, mlcanopy_type]:
    """
    Initialize tower vegetation properties for each patch.

    Mirrors Fortran subroutine ``TowerVeg`` (private to
    ``CLMml_driver``).

    For each patch ``p`` in ``[begp, endp]``:

    - ``patch.itype`` is set to the CLM PFT for the tower site.
    - ``htop_patch`` (canopy height) is set from ``tower_canht`` when
      positive; otherwise falls back to the PFT-default table
      ``htop_pft``.
    - ``root_biomass_canopy`` is set from ``tower_root`` when positive;
      missing root biomass is a fatal error.
    - Beta-distribution shape parameters for leaf (``pbeta_lai_canopy``)
      and stem (``pbeta_sai_canopy``) area density profiles are set from
      ``tower_pbeta_lai`` / ``tower_pbeta_sai`` when all four values are
      positive. If any are non-positive the assignment is skipped and
      the parameters are filled later by ``getPADparameters`` from the
      PFT defaults.

    CLM top canopy height by PFT (``htop_pft``, Fortran local ``data``
    statements, lines 40-42):

    .. code-block:: none

        0        => 0.0   (not_vegetated)
        1-3      => 17, 17, 14  (needleleaf trees)
        4-8      => 35, 35, 18, 20, 20  (broadleaf trees)
        9-16     => 0.5  (shrubs, grasses, crops)
        17-mxpft => 0.0  (additional crop PFTs)

    Args:
        it: Tower site index into ``TowerDataMod`` arrays (1-based).
        begp: First patch index.
        endp: Last patch index.
        canopystate_inst: Canopy state container; ``htop_patch`` is
            updated.
        mlcanopy_inst: Multilayer canopy container; ``root_biomass_canopy``,
            ``pbeta_lai_canopy``, and ``pbeta_sai_canopy`` are updated.

    Returns:
        Tuple of updated ``(canopystate_inst, mlcanopy_inst)``.
    """
    from legoesm.land.canopy.clm_ml_backend.clm_src_main.abortutils import endrun  # noqa: F401
    from legoesm.land.canopy.clm_ml_backend.clm_src_main.clm_varpar import mxpft  # noqa: F401
    from legoesm.land.canopy.clm_ml_backend.clm_src_main.PatchType import patch  # noqa: F401
    from legoesm.land.canopy.clm_ml_backend.offline_driver.TowerDataMod import (  # noqa: F401
        tower_canht,
        tower_pbeta_lai,
        tower_pbeta_sai,
        tower_pft,
        tower_root,
    )

    # ------------------------------------------------------------------
    # CLM canopy top height by PFT — Fortran local data htop_pft(0:mxpft)
    # Fortran lines 40-42
    # ------------------------------------------------------------------
    _htop_pft: list[float] = (
        [0.0]  # index 0: not_vegetated
        + [
            17.0,
            17.0,
            14.0,  # 1-3:  needleleaf trees
            35.0,
            35.0,
            18.0,
            20.0,
            20.0,  # 4-8:  broadleaf trees
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
        ]  # 9-16: shrubs/grasses/crops
        + [0.0] * (mxpft - 16)  # 17-mxpft: additional crop PFTs
    )

    # Unpack mutable arrays (Fortran associate block)
    htop = canopystate_inst.htop_patch
    root_biomass = mlcanopy_inst.root_biomass_canopy
    pbeta_lai = mlcanopy_inst.pbeta_lai_canopy
    pbeta_sai = mlcanopy_inst.pbeta_sai_canopy

    for p in range(begp, endp + 1):  # Fortran: do p = begp, endp

        # PFT assignment — Fortran: patch%itype(p) = tower_pft(it)
        patch.itype = patch.itype.at[p].set(int(tower_pft[it]))

        # Canopy height — Fortran lines 52-56
        if float(tower_canht[it]) > 0.0:
            htop = htop.at[p].set(float(tower_canht[it]))
        else:
            htop = htop.at[p].set(_htop_pft[int(patch.itype[p])])

        # Fine root biomass — Fortran lines 58-62
        if float(tower_root[it]) > 0.0:
            root_biomass = root_biomass.at[p].set(float(tower_root[it]))
        else:
            endrun(msg=" TowerVeg ERROR: invalid root biomass")

        # Beta distribution parameters — Fortran lines 64-71
        # Use tower values only when all four are positive; otherwise
        # deferred to getPADparameters using PFT defaults.
        if (
            float(tower_pbeta_lai[it, 0]) > 0.0
            and float(tower_pbeta_lai[it, 1]) > 0.0
            and float(tower_pbeta_sai[it, 0]) > 0.0
            and float(tower_pbeta_sai[it, 1]) > 0.0
        ):
            pbeta_lai = pbeta_lai.at[p, 1].set(float(tower_pbeta_lai[it, 0]))
            pbeta_lai = pbeta_lai.at[p, 2].set(float(tower_pbeta_lai[it, 1]))
            pbeta_sai = pbeta_sai.at[p, 1].set(float(tower_pbeta_sai[it, 0]))
            pbeta_sai = pbeta_sai.at[p, 2].set(float(tower_pbeta_sai[it, 1]))

    return (
        canopystate_inst._replace(htop_patch=htop),
        mlcanopy_inst._replace(
            root_biomass_canopy=root_biomass,
            pbeta_lai_canopy=pbeta_lai,
            pbeta_sai_canopy=pbeta_sai,
        ),
    )


def SoilInit(
    ncfilename: str,
    strt: int,
    begc: int,
    endc: int,
    soilstate_inst: soilstate_type,  # noqa: F811
    waterstatebulk_inst: waterstatebulk_type,  # noqa: F811
    temperature_inst: temperature_type,  # noqa: F811
) -> tuple[waterstatebulk_type, temperature_type]:
    """
    Initialize soil temperature and volumetric soil moisture profiles
    from a CLM netCDF history file.

    Mirrors Fortran subroutine ``SoilInit`` (private to
    ``CLMml_driver``).

    Reads ``TSOI`` (soil temperature, ``nlevgrnd`` layers) and
    ``H2OSOI`` (volumetric soil moisture) at time slice ``strt`` from
    the CLM history file and copies the values into the model state
    containers for every column in ``[begc, endc]``.

    Soil moisture handling mirrors :func:`clmDataMod.clmData`:

    - **CLM4.5**: ``nlevgrnd`` layers read directly.
    - **CLM5.0**: ``nlevsoi`` layers read; bedrock layers
      (``nlevsoi+1`` to ``nlevgrnd``) set to zero; active layers capped
      at ``watsat`` because the model porosity may differ from the CLM5
      history file.

    Liquid water and ice are derived as:

    .. code-block:: none

        h2osoi_liq(c,j) = h2osoi_vol(c,j) * dz(c,j) * denh2o
        h2osoi_ice(c,j) = 0

    Args:
        ncfilename: Path to the CLM netCDF history file.
        strt: 1-based time slice index into the netCDF file.
        begc: First column index.
        endc: Last column index.
        soilstate_inst: Soil state container supplying ``watsat_col``
            (read-only).
        waterstatebulk_inst: Bulk water state container;
            ``h2osoi_vol_col``, ``h2osoi_liq_col``, and
            ``h2osoi_ice_col`` are updated.
        temperature_inst: Temperature state container;
            ``t_soisno_col`` is updated.

    Returns:
        Tuple of updated ``(waterstatebulk_inst, temperature_inst)``.
    """
    import netCDF4 as nc  # noqa: F401

    from legoesm.land.canopy.clm_ml_backend.clm_src_main.abortutils import handle_err  # noqa: F401
    from legoesm.land.canopy.clm_ml_backend.clm_src_main.clm_varcon import denh2o, spval  # noqa: F401
    from legoesm.land.canopy.clm_ml_backend.clm_src_main.clm_varpar import nlevgrnd, nlevsno, nlevsoi  # noqa: F401
    from legoesm.land.canopy.clm_ml_backend.clm_src_main.ColumnType import col  # noqa: F401
    from legoesm.land.canopy.clm_ml_backend.offline_driver.clmSoilOptionMod import clm_phys  # noqa: F401

    # Unpack read-only inputs (Fortran associate block)
    dz = col.dz  # Soil layer thickness (m)
    nbedrock = col.nbedrock  # Depth to bedrock index
    watsat = soilstate_inst.watsat_col  # Porosity

    t_soisno = temperature_inst.t_soisno_col  # Soil temperature (K)
    h2osoi_vol = waterstatebulk_inst.h2osoi_vol_col  # Volumetric soil water (m3/m3)
    h2osoi_liq = waterstatebulk_inst.h2osoi_liq_col  # Liquid water (kg H2O/m2)
    h2osoi_ice = waterstatebulk_inst.h2osoi_ice_col  # Ice lens (kg H2O/m2)

    # ------------------------------------------------------------------
    # Read soil temperature and moisture from netCDF — Fortran lines 55-82
    # Fortran: start3=(1,1,strt), count3=(1,nlevgrnd,1)
    # Python/C row-major: ds['VAR'][t, :nlev, 0]
    # ------------------------------------------------------------------
    t = strt - 1  # Convert 1-based Fortran index to 0-based Python

    with nc.Dataset(ncfilename, "r") as ds:

        # TSOI(nlndgrid, nlevgrnd, ntime) — Fortran lines 60-64
        if "TSOI" not in ds.variables:
            handle_err(-1, "TSOI")
        tsoi_loc = ds.variables["TSOI"][t, :nlevgrnd, 0]  # (nlevgrnd,)

        # H2OSOI — Fortran lines 66-77
        if "H2OSOI" not in ds.variables:
            handle_err(-1, "H2OSOI")

        if clm_phys == "CLM4_5":
            # count3=(1,nlevgrnd,1) — Fortran lines 68-70
            h2osoi_loc_clm45 = ds.variables["H2OSOI"][t, :nlevgrnd, 0]  # (nlevgrnd,)
            h2osoi_loc_clm50 = None
        elif clm_phys == "CLM5_0":
            # count3=(1,nlevsoi,1) — Fortran lines 71-74
            h2osoi_loc_clm45 = None
            h2osoi_loc_clm50 = ds.variables["H2OSOI"][t, :nlevsoi, 0]  # (nlevsoi,)

    # ------------------------------------------------------------------
    # Copy data to model variables — Fortran lines 82-107
    # ------------------------------------------------------------------
    for c in range(begc, endc + 1):  # Fortran: do c = begc, endc

        # Soil temperature — Fortran lines 84-86
        for j in range(1, nlevgrnd + 1):  # Fortran: do j = 1, nlevgrnd
            t_soisno = t_soisno.at[c, j].set(float(tsoi_loc[j - 1]))

        # Volumetric soil moisture — Fortran lines 88-97
        if clm_phys == "CLM4_5":
            assert h2osoi_loc_clm45 is not None  # Type narrowing
            for j in range(1, nlevgrnd + 1):
                h2osoi_vol = h2osoi_vol.at[c, j].set(float(h2osoi_loc_clm45[j - 1]))

        elif clm_phys == "CLM5_0":
            assert h2osoi_loc_clm50 is not None  # Type narrowing
            for j in range(1, nlevsoi + 1):
                h2osoi_vol = h2osoi_vol.at[c, j].set(float(h2osoi_loc_clm50[j - 1]))
            for j in range(nlevsoi + 1, nlevgrnd + 1):  # Bedrock layers = 0
                h2osoi_vol = h2osoi_vol.at[c, j].set(0.0)

        # Cap soil moisture at porosity for CLM5.0 — Fortran lines 99-103
        if clm_phys == "CLM5_0":
            nb = int(nbedrock[c])
            for j in range(1, nb + 1):  # Fortran: do j = 1, nbedrock(c)
                h2osoi_vol = h2osoi_vol.at[c, j].set(
                    float(jnp.minimum(h2osoi_vol[c, j], watsat[c, j]))
                )

        # Liquid water and ice — Fortran lines 105-108
        for j in range(1, nlevgrnd + 1):  # Fortran: do j = 1, nlevgrnd
            h2osoi_liq = h2osoi_liq.at[c, j].set(float(h2osoi_vol[c, j]) * float(dz[c, j]) * denh2o)
            h2osoi_ice = h2osoi_ice.at[c, j].set(0.0)

    return (
        waterstatebulk_inst._replace(
            h2osoi_vol_col=h2osoi_vol,
            h2osoi_liq_col=h2osoi_liq,
            h2osoi_ice_col=h2osoi_ice,
        ),
        temperature_inst._replace(
            t_soisno_col=t_soisno,
        ),
    )


def output(
    curr_calday: float,
    it: int,
    nout1: IO,
    nout2: IO,
    nout3: IO,
    nout4: IO,
    nout5: IO,
    nout6: IO,
    mlcan: mlcanopy_type,
    temperature_inst: temperature_type,  # noqa: F811
) -> None:
    """
    Write per-timestep model output to six ASCII files.

    Mirrors Fortran subroutine ``output`` (private to ``CLMml_driver``).

    Output files and their contents
    --------------------------------
    - ``nout1`` (``*_flux.out``): Canopy and soil energy fluxes,
      GPP, wind, albedo, and storage terms (18 columns).
    - ``nout2`` (``*_aux.out``): Leaf water potential and soil
      moisture stress (6 columns).
    - ``nout3`` (``*_profile.out``): Vertical profiles of leaf and
      air-layer state variables; above-canopy layers use
      ``missing_value`` for leaf quantities; within-canopy layers
      with ``dpai > 0`` include per-leaf-area fluxes (28 columns).
    - ``nout4`` (``*_fsun.out``): Sunlit/shaded canopy fluxes and
      bulk canopy properties (32 columns).
    - ``nout5`` (``*_fluxprofile.out``): Vertical flux profiles of
      sensible heat, latent heat (converted from mol H2O/m2/s),
      momentum, and shortwave/longwave radiation (13 columns).
    - ``nout6`` (``*_soiltemp.out``): Soil layer depths and
      temperatures for the first 10 layers (21 columns).

    Time stamp adjustment (Fortran lines 64-72):
    - ``met_type == 0``: time at end of time step — ``curr_calday``.
    - ``met_type == 3``: time centred in time step —
      ``curr_calday - 0.5 * dtstep / 86400``.
    - ``met_type == 2``: fatal error (not supported).

    All patch-level quantities are read at ``p = 1`` (single-patch
    tower site).

    Args:
        curr_calday: Current calendar day (1.000 = 0Z 1 January).
        it: Tower site index into ``TowerDataMod`` arrays.
        nout1–nout6: Open Python file objects for the six output
            streams (replace Fortran unit numbers).
        mlcan: Multilayer canopy state container (read-only).
        temperature_inst: Temperature state container (read-only).
    """
    from legoesm.land.canopy.clm_ml_backend.clm_src_main.abortutils import endrun  # noqa: F401
    from legoesm.land.canopy.clm_ml_backend.clm_src_main.clm_varcon import tfrz  # noqa: F401
    from legoesm.land.canopy.clm_ml_backend.clm_src_main.clm_varpar import inir, ivis, nlevsno  # noqa: F401
    from legoesm.land.canopy.clm_ml_backend.clm_src_main.ColumnType import col  # noqa: F401
    from legoesm.land.canopy.clm_ml_backend.clm_src_utils.clm_time_manager import dtstep  # noqa: F401
    from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLclm_varcon import mmdry, mmh2o  # noqa: F401
    from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLclm_varctl import met_type  # noqa: F401
    from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLclm_varpar import isha, isun  # noqa: F401
    from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLWaterVaporMod import LatVap  # noqa: F401

    missing_value: float = -999.0
    zero_value: float = 0.0

    p: int = 0  # Single-patch tower site — 0-based Python indexing (Fortran used 1)

    # ------------------------------------------------------------------
    # Time stamp — Fortran lines 66-74
    # ------------------------------------------------------------------
    if met_type == 0:
        # Time at end of timestep — Fortran line 68
        time_stamp = curr_calday
    elif met_type == 3:
        # Time centred in timestep — Fortran line 71
        time_stamp = curr_calday - 0.5 * dtstep / 86400.0
    elif met_type == 2:
        # Time at end of timestep (not supported) — Fortran lines 73-74
        time_stamp = curr_calday
        endrun(msg=" ERROR: met_type not valid")
    else:
        time_stamp = curr_calday

    # ------------------------------------------------------------------
    # nout1: flux.out — canopy and soil fluxes — Fortran lines 77-95
    # write(nout1,'(f12.7,17f10.3)') time_stamp, 17 variables
    # ------------------------------------------------------------------
    swup = float(mlcan.albcan_canopy[p, ivis]) * (
        float(mlcan.swskyb_forcing[p, ivis]) + float(mlcan.swskyd_forcing[p, ivis])
    ) + float(mlcan.albcan_canopy[p, inir]) * (
        float(mlcan.swskyb_forcing[p, inir]) + float(mlcan.swskyd_forcing[p, inir])
    )

    lhflx_tr = float(mlcan.trveg_canopy[p]) * LatVap(float(mlcan.tref_forcing[p]))
    lhflx_ev = float(mlcan.evveg_canopy[p]) * LatVap(float(mlcan.tref_forcing[p]))

    ic_top = int(mlcan.ntop_canopy[p])
    tair = float(mlcan.tair_profile[p, ic_top])

    nout1.write(
        f"{time_stamp:12.7f}"
        + _fmt10(mlcan.rnet_canopy[p])
        + _fmt10(mlcan.stflx_air_canopy[p])
        + _fmt10(mlcan.shflx_canopy[p])
        + _fmt10(mlcan.lhflx_canopy[p])
        + _fmt10(mlcan.gppveg_canopy[p])
        + _fmt10(mlcan.ustar_canopy[p])
        + _fmt10(swup)
        + _fmt10(mlcan.lwup_canopy[p])
        + _fmt10(tair)
        + _fmt10(mlcan.gsoi_soil[p])
        + _fmt10(mlcan.rnsoi_soil[p])
        + _fmt10(mlcan.shsoi_soil[p])
        + _fmt10(mlcan.lhsoi_soil[p])
        + _fmt10(lhflx_tr)
        + _fmt10(lhflx_ev)
        + _fmt10(mlcan.beta_canopy[p])
        + _fmt10(mlcan.stflx_veg_canopy[p])
        + "\n"
    )

    # ------------------------------------------------------------------
    # nout4: fsun.out — sunlit/shaded fluxes — Fortran lines 98-113
    # write(nout4,'(32f10.3)') 32 variables
    # ------------------------------------------------------------------
    nout4.write(
        _fmt10(float(mlcan.solar_zen_forcing[p]) * 180.0 / math.pi)
        + _fmt10(float(mlcan.swskyb_forcing[p, ivis]) + float(mlcan.swskyd_forcing[p, ivis]))
        + _fmt10(float(mlcan.lai_canopy[p]) + float(mlcan.sai_canopy[p]))
        + _fmt10(mlcan.laisun_canopy[p])
        + _fmt10(mlcan.laisha_canopy[p])
        + _fmt10(mlcan.swveg_canopy[p, ivis])
        + _fmt10(mlcan.swvegsun_canopy[p, ivis])
        + _fmt10(mlcan.swvegsha_canopy[p, ivis])
        + _fmt10(mlcan.gppveg_canopy[p])
        + _fmt10(mlcan.gppvegsun_canopy[p])
        + _fmt10(mlcan.gppvegsha_canopy[p])
        + _fmt10(mlcan.lhveg_canopy[p])
        + _fmt10(mlcan.lhvegsun_canopy[p])
        + _fmt10(mlcan.lhvegsha_canopy[p])
        + _fmt10(mlcan.shveg_canopy[p])
        + _fmt10(mlcan.shvegsun_canopy[p])
        + _fmt10(mlcan.shvegsha_canopy[p])
        + _fmt10(mlcan.vcmax25veg_canopy[p])
        + _fmt10(mlcan.vcmax25sun_canopy[p])
        + _fmt10(mlcan.vcmax25sha_canopy[p])
        + _fmt10(mlcan.gsveg_canopy[p])
        + _fmt10(mlcan.gsvegsun_canopy[p])
        + _fmt10(mlcan.gsvegsha_canopy[p])
        + _fmt10(mlcan.windveg_canopy[p])
        + _fmt10(mlcan.windvegsun_canopy[p])
        + _fmt10(mlcan.windvegsha_canopy[p])
        + _fmt10(mlcan.tlveg_canopy[p])
        + _fmt10(mlcan.tlvegsun_canopy[p])
        + _fmt10(mlcan.tlvegsha_canopy[p])
        + _fmt10(mlcan.taveg_canopy[p])
        + _fmt10(mlcan.tavegsun_canopy[p])
        + _fmt10(mlcan.tavegsha_canopy[p])
        + "\n"
    )

    # ------------------------------------------------------------------
    # nout2: aux.out — leaf water potential / soil stress — Fortran lines 116-120
    # write(nout2,'(f10.4,5f10.3)') 6 variables
    # ------------------------------------------------------------------
    top = int(mlcan.ntop_canopy[p])
    mid = max(
        1,
        int(mlcan.nbot_canopy[p])
        + (int(mlcan.ntop_canopy[p]) - int(mlcan.nbot_canopy[p]) + 1) // 2
        - 1,
    )

    nout2.write(
        f"{float(mlcan.btran_soil[p]):10.4f}"
        + _fmt10(mlcan.lsc_profile[p, top])
        + _fmt10(mlcan.psis_soil[p])
        + _fmt10(mlcan.lwp_mean_profile[p, top])
        + _fmt10(mlcan.lwp_mean_profile[p, mid])
        + _fmt10(mlcan.fracminlwp_canopy[p])
        + "\n"
    )

    # ------------------------------------------------------------------
    # nout3: profile.out — vertical profiles — Fortran lines 122-176
    # write(nout3,'(f12.7,27f10.3)') time_stamp + 27 variables
    # ------------------------------------------------------------------

    def _qair(ic_: int) -> float:
        """Specific humidity (g/kg) from vapour pressure profile."""
        e = float(mlcan.eair_profile[p, ic_])
        pr = float(mlcan.pref_forcing[p])
        return 1000.0 * (mmh2o / mmdry) * e / (pr - (1.0 - mmh2o / mmdry) * e)

    def _eair_kpa(ic_: int) -> float:
        """Vapour pressure (kPa)."""
        return float(mlcan.eair_profile[p, ic_]) / 1000.0

    def _ra(ic_: int) -> float:
        """Aerodynamic resistance (s/m)."""
        return float(mlcan.rhomol_forcing[p]) / float(mlcan.gac_profile[p, ic_])

    def _lad(ic_: int) -> float:
        """Plant area density (m2/m3)."""
        return float(mlcan.dpai_profile[p, ic_]) / float(mlcan.dz_profile[p, ic_])

    def _profile_line(ic_: int, is_above: bool) -> str:
        """Build one nout3 record (28 columns)."""
        tair_ = float(mlcan.tair_profile[p, ic_])
        qair_ = _qair(ic_)
        ra_ = _ra(ic_)
        mv = missing_value
        zv = zero_value

        if is_above:
            # Above-canopy: all leaf quantities are missing — Fortran lines 131-149
            return (
                f"{time_stamp:12.7f}"
                + _fmt10(mlcan.zs_profile[p, ic_])
                + _fmt10(zv)
                + _fmt10(zv)
                + _fmt10(zv)
                + _fmt10(zv)
                + (_fmt10(mv) * 18)
                + _fmt10(float(mlcan.wind_profile[p, ic_]))
                + _fmt10(tair_)
                + _fmt10(qair_)
                + _fmt10(ra_)
                + "\n"
            )
        else:
            dpai_ = float(mlcan.dpai_profile[p, ic_])
            lad_ = _lad(ic_)
            frac_ = float(mlcan.fracsun_profile[p, ic_])
            if dpai_ > 0.0:
                # Leaf layer — Fortran lines 155-170
                return (
                    f"{time_stamp:12.7f}"
                    + _fmt10(mlcan.zs_profile[p, ic_])
                    + _fmt10(frac_)
                    + _fmt10(lad_)
                    + _fmt10(lad_ * frac_)
                    + _fmt10(lad_ * (1.0 - frac_))
                    + _fmt10(mlcan.rnleaf_leaf[p, ic_, isun])
                    + _fmt10(mlcan.rnleaf_leaf[p, ic_, isha])
                    + _fmt10(mlcan.shleaf_leaf[p, ic_, isun])
                    + _fmt10(mlcan.shleaf_leaf[p, ic_, isha])
                    + _fmt10(mlcan.lhleaf_leaf[p, ic_, isun])
                    + _fmt10(mlcan.lhleaf_leaf[p, ic_, isha])
                    + _fmt10(mlcan.anet_leaf[p, ic_, isun])
                    + _fmt10(mlcan.anet_leaf[p, ic_, isha])
                    + _fmt10(mlcan.apar_leaf[p, ic_, isun])
                    + _fmt10(mlcan.apar_leaf[p, ic_, isha])
                    + _fmt10(mlcan.gs_leaf[p, ic_, isun])
                    + _fmt10(mlcan.gs_leaf[p, ic_, isha])
                    + _fmt10(mlcan.lwp_hist_leaf[p, ic_, isun])
                    + _fmt10(mlcan.lwp_hist_leaf[p, ic_, isha])
                    + _fmt10(mlcan.tleaf_hist_leaf[p, ic_, isun])
                    + _fmt10(mlcan.tleaf_hist_leaf[p, ic_, isha])
                    + _fmt10(mlcan.vcmax25_leaf[p, ic_, isun])
                    + _fmt10(mlcan.vcmax25_leaf[p, ic_, isha])
                    + _fmt10(float(mlcan.wind_profile[p, ic_]))
                    + _fmt10(tair_)
                    + _fmt10(qair_)
                    + _fmt10(ra_)
                    + "\n"
                )
            else:
                # Non-leaf within-canopy layer — Fortran lines 172-185
                return (
                    f"{time_stamp:12.7f}"
                    + _fmt10(mlcan.zs_profile[p, ic_])
                    + _fmt10(frac_)
                    + _fmt10(zv)
                    + _fmt10(zv)
                    + _fmt10(zv)
                    + (_fmt10(mv) * 18)
                    + _fmt10(float(mlcan.wind_profile[p, ic_]))
                    + _fmt10(tair_)
                    + _fmt10(qair_)
                    + _fmt10(ra_)
                    + "\n"
                )

    # Above-canopy layers — Fortran: do ic = ncan, ntop+1, -1
    # Python range stop is exclusive, so use ntop (not ntop+1) to include ic=ntop+1
    for ic in range(int(mlcan.ncan_canopy[p]), int(mlcan.ntop_canopy[p]), -1):
        nout3.write(_profile_line(ic, is_above=True))

    # Within-canopy layers — Fortran: do ic = ntop, 1, -1
    for ic in range(int(mlcan.ntop_canopy[p]), 0, -1):
        nout3.write(_profile_line(ic, is_above=False))

    # ------------------------------------------------------------------
    # nout5: fluxprofile.out — vertical flux profiles — Fortran lines 188-202
    # write(nout5,'(f12.7,26f10.3)') time_stamp + 12 variables (13 cols total)
    # ------------------------------------------------------------------
    _ntop_fp = int(mlcan.ntop_canopy[p])
    for ic in range(int(mlcan.ncan_canopy[p]), 0, -1):  # Fortran: do ic = ncan, 1, -1
        shf = float(mlcan.shair_profile[p, ic])
        lhf = float(mlcan.etair_profile[p, ic]) * LatVap(float(mlcan.tref_forcing[p]))
        mflx = float(mlcan.mflx_profile[p, ic])
        # LW profiles are only computed for within-canopy layers (0..ntop).
        # Above-canopy layers retain spval; write 0.0 to match Fortran behavior.
        lwdwn_ic = float(mlcan.lwdwn_profile[p, ic]) if ic <= _ntop_fp else 0.0
        lwupw_ic = float(mlcan.lwupw_profile[p, ic]) if ic <= _ntop_fp else 0.0
        nout5.write(
            f"{time_stamp:12.7f}"
            + _fmt10(mlcan.zw_profile[p, ic])
            + _fmt10(shf)
            + _fmt10(lhf)
            + _fmt10(mflx)
            + _fmt10(mlcan.swbeam_profile[p, ic, ivis])
            + _fmt10(mlcan.swbeam_profile[p, ic, inir])
            + _fmt10(mlcan.swdwn_profile[p, ic, ivis])
            + _fmt10(mlcan.swdwn_profile[p, ic, inir])
            + _fmt10(mlcan.swupw_profile[p, ic, ivis])
            + _fmt10(mlcan.swupw_profile[p, ic, inir])
            + _fmt10(lwdwn_ic)
            + _fmt10(lwupw_ic)
            + "\n"
        )

    # ------------------------------------------------------------------
    # nout6: soiltemp.out — soil temperature — Fortran lines 204-207
    # write(nout6,'(f12.7,20f10.3)') time_stamp, (z(1,ic), t_soisno(1,ic), ic=1,10)
    # ------------------------------------------------------------------
    soiltemp_cols = ""
    for ic in range(1, 11):  # Fortran: ic=1,10
        soiltemp_cols += _fmt10(col.z[1, ic])
        soiltemp_cols += _fmt10(temperature_inst.t_soisno_col[1, ic])
    nout6.write(f"{time_stamp:12.7f}" + soiltemp_cols + "\n")


# ---------------------------------------------------------------------------
# Private formatting helper
# ---------------------------------------------------------------------------


def _fmt10(val) -> str:
    """
    Format a scalar as a 10-character field with 3 decimal places,
    matching Fortran ``f10.3``.
    """
    return f"{float(val):10.3f}"


def ReadCanopyProfiles(
    itim: int,  # noqa: F811
    curr_calday: float,
    nin1: IO,
    mlcanopy_inst: mlcanopy_type,  # noqa: F811
) -> mlcanopy_type:
    """
    Read T, Q, U vertical profile data for the current time step from
    an ASCII profile input file.

    Mirrors Fortran subroutine ``ReadCanopyProfiles`` (private to
    ``CLMml_driver``; final routine in the module, lines 1-80).

    On the **first time step** (``itim == 1``) the file is scanned from
    the beginning to count the number of vertical levels by reading
    records that share the same calendar day, then the file is rewound.
    On all subsequent time steps the file is read sequentially,
    continuing from where the previous call left off.

    Each record in the file has the format ``(f10.4, 26f10.3)``
    (Fortran), providing:

    .. code-block:: none

        curr_calday_data   (f10.4)   calendar day
        zs_data            (f10.3)   layer height (m)
        x(1:22)            (22f10.3) dummy variables (discarded)
        wind               (f10.3)   wind speed (m/s)
        tair               (f10.3)   air temperature (K)
        qair               (f10.3)   specific humidity (g/kg)

    Layers are stored top-to-bottom in the file (``ic = ncan, ..., 1``)
    matching the Fortran ``do ic = ncan(p), 1, -1`` loop.

    After reading, specific humidity is converted from g/kg to kg/kg
    and then to vapour pressure (Pa):

    .. code-block:: none

        qair [kg/kg] = qair_file / 1000
        eair [Pa]    = qair * pref / (mmh2o/mmdry + (1 - mmh2o/mmdry) * qair)

    Two error checks are applied:

    - Calendar day mismatch: ``|curr_calday_data - curr_calday| >= 1e-4``
      → fatal error.
    - Height profile mismatch (``itim > 1``):
      ``|zs_data - zs(p,ic)| >= 1e-3`` → fatal error.

    Args:
        itim: Current time step index (1-based).
        curr_calday: Current calendar day (1.000 = 0Z 1 January).
        nin1: Open Python file object for the ASCII profile input file
            (replaces Fortran unit ``nin1``).
        mlcanopy_inst: Multilayer canopy container; ``ncan_canopy``,
            ``wind_data_profile``, ``tair_data_profile``, and
            ``eair_data_profile`` are updated.

    Returns:
        Updated :class:`mlcanopy_type`.
    """
    from legoesm.land.canopy.clm_ml_backend.clm_src_main.abortutils import endrun  # noqa: F401
    from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLclm_varcon import mmdry, mmh2o  # noqa: F401

    # Unpack mutable arrays (Fortran associate block)
    ncan = mlcanopy_inst.ncan_canopy  # Number of aboveground layers
    pref = mlcanopy_inst.pref_forcing  # Air pressure at reference height (Pa)
    zs = mlcanopy_inst.zs_profile  # Layer height for scalar conc. and source (m)
    wind_data = mlcanopy_inst.wind_data_profile  # Wind speed FROM DATASET (m/s)
    tair_data = mlcanopy_inst.tair_data_profile  # Air temperature FROM DATASET (K)
    eair_data = mlcanopy_inst.eair_data_profile  # Vapour pressure FROM DATASET (Pa)

    p: int = 0  # 0-based single-patch tower site (Fortran used 1)

    # ------------------------------------------------------------------
    # Private helper: parse one record from the profile file
    # Format: (f10.4, 26f10.3) = 1 + 26 = 27 values per line
    # Fields: curr_calday_data, zs_data, x(1:22), wind, tair, qair
    # ------------------------------------------------------------------
    def _read_record(fh: IO) -> tuple[float, float, float, float, float] | None:
        """
        Read and parse one profile record.

        Returns ``(curr_calday_data, zs_data, wind, tair, qair)`` or
        ``None`` on end-of-file.
        """
        line = fh.readline()
        if not line:
            return None
        vals = [float(line[i * 10 : (i + 1) * 10]) for i in range(27)]
        curr_calday_data = vals[0]
        zs_data = vals[1]
        # vals[2:24] are 22 dummy variables x(1:22) — discarded
        wind = vals[24]
        tair = vals[25]
        qair = vals[26]
        return curr_calday_data, zs_data, wind, tair, qair

    # ------------------------------------------------------------------
    # First time step: scan file to count vertical levels — Fortran lines 47-57
    # ------------------------------------------------------------------
    if itim == 1:
        nrec: int = 0
        check: float = 0.0

        while True:
            rec = _read_record(nin1)
            if rec is None:  # Fortran: end=100
                break
            calday_rec, _, _, _, _ = rec
            if nrec == 0:
                check = calday_rec  # Fortran: check = curr_calday_data
            if calday_rec == check:
                nrec += 1
            else:
                break  # Fortran: exit

        # Fortran label 100: ncan(p) = nrec; rewind(nin1)
        ncan = ncan.at[p].set(nrec)
        nin1.seek(0)  # Fortran: rewind(nin1)

    # ------------------------------------------------------------------
    # Read profile data for the current time slice — Fortran lines 60-77
    # Fortran: do ic = ncan(p), 1, -1
    # ------------------------------------------------------------------
    for ic in range(int(ncan[p]), 0, -1):

        rec = _read_record(nin1)
        if rec is None:
            endrun(msg=" ERROR: ReadCanopyProfiles: unexpected end of file")
            break

        curr_calday_data, zs_data, wind, tair, qair_gkg = rec

        qair = qair_gkg / 1000.0  # g/kg -> kg/kg — Fortran line 63

        # Calendar day error check — Fortran lines 65-68
        err = curr_calday_data - curr_calday
        if abs(err) >= 1.0e-4:
            endrun(msg=" ERROR: ReadCanopyProfiles: calendar error")

        # Height profile error check (skip on first time step) — Fortran lines 70-74
        if itim > 1:
            err = zs_data - float(zs[p, ic])
            if abs(err) >= 1.0e-3:
                endrun(msg=" ERROR: ReadCanopyProfiles: height profile error")

        # Store wind and temperature — Fortran lines 76-77
        wind_data = wind_data.at[p, ic].set(wind)
        tair_data = tair_data.at[p, ic].set(tair)

        # Specific humidity -> vapour pressure (Pa) — Fortran line 78
        eair_val = qair * float(pref[p]) / (mmh2o / mmdry + (1.0 - mmh2o / mmdry) * qair)
        eair_data = eair_data.at[p, ic].set(eair_val)

    return mlcanopy_inst._replace(
        ncan_canopy=ncan,
        wind_data_profile=wind_data,
        tair_data_profile=tair_data,
        eair_data_profile=eair_data,
    )
