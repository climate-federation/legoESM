"""
JAX translation of controlMod Fortran module.

Initialize namelist run control variables for the CLM offline
multilayer canopy model. Reads simulation settings from a namelist
file (stdin), matches the tower site identifier to the TowerDataMod
index arrays, and derives the number of time steps to execute.

Original Fortran module: controlMod
Fortran lines 1-110
"""

from typing import Tuple

from legoesm.land.canopy.clm_ml_backend.clm_src_main.abortutils import endrun  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_main.clm_varctl import iulog  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_utils import clm_time_manager  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_utils.clm_time_manager import dtstep, start_date_tod, start_date_ymd  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.multilayer_canopy import MLclm_varctl  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.offline_driver import (
    TowerDataMod,  # noqa: F401
    clmSoilOptionMod,  # noqa: F401
)
from legoesm.land.canopy.clm_ml_backend.offline_driver.TowerDataMod import ntower, tower_id, tower_num, tower_time  # noqa: F401

# ---------------------------------------------------------------------------
# Module-level control variables for offline executable
# ---------------------------------------------------------------------------
# These are set by the main driver (offline_executable/main.py) after reading
# the namelist and are used throughout the simulation run

tower_site: str = ""
iyear: int = -1
imonth: int = -1


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def control(
    fin_tower_default: str = " ", fin_clm_default: str = " "
) -> Tuple[int, int, int, str, str, str, str]:
    """
    Initialize run control variables from a namelist file.

    Mirrors Fortran subroutine ``control`` (lines 22-105).

    Reads the ``clmML_inparm`` namelist group from ``stdin`` (Fortran
    unit 5), sets global time-manager and tower-index state, and
    returns the derived simulation parameters to the caller.

    The following module-level globals are updated as side effects,
    mirroring Fortran's use of ``use``-associated module variables:

    - ``clm_time_manager.start_date_ymd``
    - ``clm_time_manager.start_date_tod``
    - ``clm_time_manager.dtstep``
    - ``TowerDataMod.tower_num``
    - ``clmSoilOptionMod.clm_phys``
    - ``clmSoilOptionMod.nlev_soil_adjust``
    - ``MLclm_varctl.met_type``
    - ``MLclm_varctl.dpai_min``
    - ``MLclm_varctl.pftcon_val``

    Namelist variables and their defaults (Fortran lines 55-76):

    .. code-block:: none

        tower_name      = ' '   ! Flux tower site to process
        start_ymd       = 0     ! Run start date (yyyymmdd)
        start_tod       = 0     ! Time-of-day of start date (s past 0Z UTC)
        stop_option     = ' '   ! 'ndays' or 'nsteps'
        stop_n          = 0     ! Run length in days or timesteps
        clm_start_ymd   = 0     ! CLM file start date (yyyymmdd)
        clm_start_tod   = 0     ! CLM file start time-of-day (s past 0Z UTC)
        fin_tower       = ' '   ! Tower meteorology file name
        fin_clm         = ' '   ! CLM file name
        clm_phys        = ' '   ! 'CLM4_5' or 'CLM5_0'
        fin_soil_adjust = ' '   ! Soil moisture adjustment file name
        nlev_soil_adjust = 0    ! Layers for soil moisture adjustment (0 = off)
        dirout          = ' '   ! Output directory path
        met_type  (default in MLclm_varctl)
        dpai_min  (default in MLclm_varctl)
        pftcon_val (default in MLclm_varctl)

    Returns:
        Tuple of ``(ntim, clm_start_ymd, clm_start_tod, fin_tower,
        fin_clm, fin_soil_adjust, dirout)`` matching the Fortran
        ``intent(out)`` arguments (lines 33-40).

        - ``ntim``: Number of time steps to execute.
        - ``clm_start_ymd``: CLM file start date (yyyymmdd).
        - ``clm_start_tod``: CLM file start time-of-day (s past 0Z UTC).
        - ``fin_tower``: Tower meteorology file path.
        - ``fin_clm``: CLM file path.
        - ``fin_soil_adjust``: Soil moisture adjustment file path.
        - ``dirout``: Model output directory path.
    """
    # ------------------------------------------------------------------
    # Default namelist variables — Fortran lines 55-76
    # ------------------------------------------------------------------
    tower_name = " "  # Flux tower site to process
    start_ymd = 0  # Run start date (yyyymmdd)
    start_tod = 0  # Time-of-day of start date (s past 0Z UTC)
    stop_option = " "  # 'ndays' or 'nsteps'
    stop_n = 0  # Run length (days or timesteps)
    clm_start_ymd = 0  # CLM file start date (yyyymmdd)
    clm_start_tod = 0  # CLM file start time-of-day (s past 0Z UTC)
    fin_tower = " "  # Tower meteorology file name
    fin_clm = " "  # CLM file name
    clm_phys = " "  # CLM snow/soil layers: 'CLM4_5' or 'CLM5_0'
    fin_soil_adjust = " "  # Soil moisture adjustment factor file name
    nlev_soil_adjust = 0  # Layers for soil moisture adjustment (0 = off)
    dirout = " "  # Output file directory path

    # met_type, dpai_min, pftcon_val: set to defaults in MLclm_varctl;
    # overridden by namelist if provided. (Fortran lines 78-84)
    #
    # met_type   = 0       ! 0 = no interp, 2 = 2-point, 3 = 3-point (CHATS)
    # dpai_min   = 0.01    ! Min plant area index treated as veg layer (m2/m2)
    # pftcon_val = 0       ! 0 = default PFT params, 1 = CHATS override

    # ------------------------------------------------------------------
    # Read namelist from stdin (Fortran unit 5) — Fortran lines 86-88
    # Python equivalent: parse a simple key=value text block from stdin.
    # ------------------------------------------------------------------
    print(f"{iulog}: Attempting to read namelist file .....")

    import sys

    namelist_vars = {
        "tower_name": tower_name,
        "start_ymd": start_ymd,
        "start_tod": start_tod,
        "stop_option": stop_option,
        "stop_n": stop_n,
        "clm_start_ymd": clm_start_ymd,
        "clm_start_tod": clm_start_tod,
        "fin_tower": fin_tower,
        "fin_clm": fin_clm,
        "clm_phys": clm_phys,
        "fin_soil_adjust": fin_soil_adjust,
        "nlev_soil_adjust": nlev_soil_adjust,
        "dirout": dirout,
        "met_type": MLclm_varctl.met_type,
        "dpai_min": MLclm_varctl.dpai_min,
        "pftcon_val": MLclm_varctl.pftcon_val,
    }

    # Read and evaluate the namelist block from stdin
    for line in sys.stdin:
        line = line.strip()
        if not line or line.startswith("!") or line in ("&clmML_inparm", "/"):
            continue
        if "=" in line:
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.rstrip(",").strip().strip("'\"")
            if key in namelist_vars:
                ref = namelist_vars[key]
                if isinstance(ref, int):
                    namelist_vars[key] = int(val)
                elif isinstance(ref, float):
                    namelist_vars[key] = float(val)
                else:
                    namelist_vars[key] = str(val)

    print(f"{iulog}: Successfully read namelist file")

    # Unpack parsed values
    tower_name = str(namelist_vars["tower_name"])
    start_ymd = int(namelist_vars["start_ymd"])
    start_tod = int(namelist_vars["start_tod"])
    stop_option = str(namelist_vars["stop_option"])
    stop_n = int(namelist_vars["stop_n"])
    clm_start_ymd = int(namelist_vars["clm_start_ymd"])
    clm_start_tod = int(namelist_vars["clm_start_tod"])
    fin_tower = str(namelist_vars["fin_tower"])
    fin_clm = str(namelist_vars["fin_clm"])
    fin_soil_adjust = str(namelist_vars["fin_soil_adjust"])
    nlev_soil_adjust = int(namelist_vars["nlev_soil_adjust"])
    dirout = str(namelist_vars["dirout"])

    # Write parsed namelist values back to their owning modules
    clmSoilOptionMod.clm_phys = str(namelist_vars["clm_phys"])
    clmSoilOptionMod.nlev_soil_adjust = nlev_soil_adjust
    MLclm_varctl.met_type = int(namelist_vars["met_type"])
    MLclm_varctl.dpai_min = float(namelist_vars["dpai_min"])
    MLclm_varctl.pftcon_val = int(namelist_vars["pftcon_val"])

    # ------------------------------------------------------------------
    # Set calendar variables — Fortran lines 90-91
    # ------------------------------------------------------------------
    clm_time_manager.start_date_ymd = start_ymd
    clm_time_manager.start_date_tod = start_tod

    # ------------------------------------------------------------------
    # Match tower site to TowerDataMod index — Fortran lines 93-101
    # ------------------------------------------------------------------
    _tower_num = 0
    for i in range(1, ntower + 1):  # Fortran: do i = 1, ntower
        if tower_name == tower_id[i]:
            _tower_num = i
            break

    if _tower_num == 0:
        print(f"{iulog}: control error: tower site = {tower_name} not found")
        endrun()

    TowerDataMod.tower_num = _tower_num

    # ------------------------------------------------------------------
    # Time step of forcing data (seconds) — Fortran line 103
    # tower_time stores the time step in minutes
    # ------------------------------------------------------------------
    _dtstep = int(tower_time[_tower_num]) * 60
    clm_time_manager.dtstep = _dtstep

    # ------------------------------------------------------------------
    # Set length of simulation — Fortran lines 105-109
    # ------------------------------------------------------------------
    ntim: int
    if stop_option == "nsteps":
        ntim = stop_n  # Fortran line 106
    elif stop_option == "ndays":
        steps_per_day = 86400 // _dtstep  # Fortran line 108
        ntim = steps_per_day * stop_n  # Fortran line 109
    else:
        print(f"{iulog}: control error: stop_option = {stop_option} not recognized")
        endrun()
        ntim = 0  # Unreachable; satisfies type checker

    return ntim, clm_start_ymd, clm_start_tod, fin_tower, fin_clm, fin_soil_adjust, dirout
