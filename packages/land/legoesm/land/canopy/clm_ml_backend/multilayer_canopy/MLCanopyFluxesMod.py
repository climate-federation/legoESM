"""
JAX translation of MLCanopyFluxesMod Fortran module.

Orchestrator for all multilayer canopy physics within a single CLM
timestep.  Manages the filter of vegetated patches, initialisation,
atmospheric-forcing interpolation, the Runge-Kutta loop, and the
pass-back to CLM output variables.

Public routine
--------------
- :func:`MLCanopyFluxes`: main driver.

Private helpers
---------------
- :func:`_GetCLMVar`: stub — copy CLM variables to mlcanopy container.
- :func:`_MLTimeStepFluxIntegration`: stub — accumulate ML-step fluxes.
- :func:`_CanopyFluxesDiagnostics`: stub — canopy-wide diagnostics.

Original Fortran module: MLCanopyFluxesMod
Fortran lines 1-400
"""

from __future__ import annotations

from functools import partial
from typing import Any, List, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from legoesm.land.canopy.clm_ml_backend.clm_src_main.abortutils import endrun  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_main.clm_varcon import grav, spval  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_main.clm_varctl import iulog  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_main.clm_varpar import inir, ivis  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_main.PatchType import patch  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_utils.spmdMod import masterproc  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLCanopyFluxesType import mlcanopy_type  # noqa: F401

# Physics modules (translated elsewhere)
from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLCanopyNitrogenProfileMod import CanopyNitrogenProfile  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLCanopyTurbulenceMod import CanopyTurbulence  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLCanopyWaterMod import (  # noqa: F401
    CanopyEvaporation,
    CanopyInterception,
    CanopyWettedFraction,
)
from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLclm_varcon import mmh2o, rgas  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLclm_varctl import (  # noqa: F401
    GridInfo,
    dtime_ml,
    met_type,
    ml_vert_init,
    mlcan_to_clm,
    nrk,
    runge_kutta_type,
)
from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLclm_varpar import isha, isun, nleaf, nlevmlcan  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLFluxProfileSolutionMod import FluxProfileSolution  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLGetAtmForcingMod import GetAtmForcing  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLinitVerticalMod import (  # noqa: F401
    getPADparameters,
    initVerticalProfiles,
    initVerticalStructure,
)
from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLLeafBoundaryLayerMod import (
    LeafBoundaryLayerBoth,
)  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLLeafHeatCapacityMod import LeafHeatCapacity  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLLeafPhotosynthesisMod import LeafPhotosynthesis  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLLongwaveRadiationMod import LongwaveRadiation  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLPlantHydraulicsMod import (  # noqa: F401
    LeafWaterPotential,
    PlantResistance,
    SoilResistance,
)
from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLRungeKuttaMod import RungeKuttaIni, RungeKuttaUpdate  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLSolarRadiationMod import SolarRadiation  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLWaterVaporMod import LatVap  # noqa: F401

# ---------------------------------------------------------------------------
# Module-level cache for Runge-Kutta coefficients (computed once on first
# call, reused on all subsequent calls — mirrors Fortran module state).
# ---------------------------------------------------------------------------
_rk_ark = None
_rk_brk = None
_rk_crk = None


# ---------------------------------------------------------------------------
# JIT-compiled bef-state copy helper
# ---------------------------------------------------------------------------


@partial(jax.jit, static_argnums=(0, 1, 4))
def _copy_bef_state(
    filter_mlcan: tuple,
    ncan_vals: tuple,
    mlcanopy_inst: mlcanopy_type,
    grid_p: "int | jnp.ndarray | None" = None,
    grid_ncan: "int | None" = None,
) -> mlcanopy_type:
    """Copy current-state arrays to their ``*_bef`` counterparts.

    Replaces a Python for-loop of 7 eager ``.at[].set()`` dispatches with a
    single JIT-compiled kernel.  ``filter_mlcan`` and ``ncan_vals`` are
    static so slice bounds are concrete at trace time; recompilation only
    occurs when the canopy structure changes (never for a fixed site).

    S3 single-patch grid path: pass ``grid_p`` (the patch index — a DYNAMIC
    scalar that may be a TRACED lax.scan-over-columns index, so it is NOT a
    static arg) and ``grid_ncan`` (the concrete canopy-layer count — STATIC,
    arg 4, so the ``1:ncan+1`` slice bound stays concrete at trace time).  The
    ``.at[grid_p].set`` scatters take the dynamic index.  ``filter_mlcan`` /
    ``ncan_vals`` are then dummy statics.  Value-identical to the filter loop
    when ``grid_p`` is concrete (the per-column-call path).  A NamedTuple
    ``GridInfo`` cannot be used here: a non-static pytree arg would trace its
    ``ncan`` leaf too, breaking the static slice — hence the split scalars.

    Mirrors Fortran lines 272-285 of ``MLCanopyFluxes``.
    """
    inst = mlcanopy_inst
    _patch_iter = ([(grid_p, grid_ncan)] if grid_p is not None
                   else zip(filter_mlcan, ncan_vals))
    for p, ncan in _patch_iter:
        _sl = slice(1, ncan + 1)
        inst = inst._replace(
            tg_bef_soil=inst.tg_bef_soil.at[p].set(inst.tg_soil[p]),
            tair_bef_profile=inst.tair_bef_profile.at[p, _sl].set(inst.tair_profile[p, _sl]),
            eair_bef_profile=inst.eair_bef_profile.at[p, _sl].set(inst.eair_profile[p, _sl]),
            cair_bef_profile=inst.cair_bef_profile.at[p, _sl].set(inst.cair_profile[p, _sl]),
            h2ocan_bef_profile=inst.h2ocan_bef_profile.at[p, _sl].set(inst.h2ocan_profile[p, _sl]),
            tleaf_bef_leaf=inst.tleaf_bef_leaf.at[p, _sl, :].set(inst.tleaf_leaf[p, _sl, :]),
            lwp_bef_leaf=inst.lwp_bef_leaf.at[p, _sl, :].set(inst.lwp_leaf[p, _sl, :]),
        )
    return inst


# ---------------------------------------------------------------------------
# JIT-compiled helper: save current forcing as *_bef for next CLM timestep
# ---------------------------------------------------------------------------


@partial(jax.jit, static_argnums=(0,))
def _save_bef_forcing(
    filter_mlcan: tuple,
    mlcanopy_inst: mlcanopy_type,
    grid_p: "int | jnp.ndarray | None" = None,
) -> mlcanopy_type:
    """Copy ``*_cur_forcing`` fields to their ``*_bef_forcing`` counterparts.

    Fuses 8 separate scatter operations (uref, tref, qref, pref, co2ref,
    swskyb, swskyd, lwsky) into one JIT-compiled XLA program — reducing
    Python→GPU dispatch from 8+ roundtrips to 1.

    ``filter_mlcan`` is static so the patch-index loop is unrolled at trace
    time; recompilation only occurs when the active-patch set changes
    (never for a fixed site).  S3 single-patch grid path: pass ``grid_p`` (the
    patch index — a DYNAMIC scalar that may be a TRACED scan-over-columns
    index) and this fuses to a single dynamic scatter.  No slice bounds here,
    so only the index is needed (no static ``ncan``).

    Mirrors Fortran lines 332-343 of ``MLCanopyFluxes``.
    """
    inst = mlcanopy_inst
    for p in ([grid_p] if grid_p is not None else filter_mlcan):
        inst = inst._replace(
            uref_bef_forcing=inst.uref_bef_forcing.at[p].set(inst.uref_cur_forcing[p]),
            tref_bef_forcing=inst.tref_bef_forcing.at[p].set(inst.tref_cur_forcing[p]),
            qref_bef_forcing=inst.qref_bef_forcing.at[p].set(inst.qref_cur_forcing[p]),
            pref_bef_forcing=inst.pref_bef_forcing.at[p].set(inst.pref_cur_forcing[p]),
            co2ref_bef_forcing=inst.co2ref_bef_forcing.at[p].set(inst.co2ref_cur_forcing[p]),
            lwsky_bef_forcing=inst.lwsky_bef_forcing.at[p].set(inst.lwsky_cur_forcing[p]),
            # Copy full band slice for swskyb/swskyd (ivis + inir) in one op
            swskyb_bef_forcing=inst.swskyb_bef_forcing.at[p, :].set(inst.swskyb_cur_forcing[p, :]),
            swskyd_bef_forcing=inst.swskyd_bef_forcing.at[p, :].set(inst.swskyd_cur_forcing[p, :]),
        )
    return inst


# ---------------------------------------------------------------------------
# Public: MLCanopyFluxes
# ---------------------------------------------------------------------------


def MLCanopyFluxes(
    bounds: Any,
    num_exposedvegp: int,
    filter_exposedvegp: Sequence[int],
    atm2lnd_inst: Any,
    canopystate_inst: Any,
    soilstate_inst: Any,
    temperature_inst: Any,
    waterstatebulk_inst: Any,
    waterfluxbulk_inst: Any,
    energyflux_inst: Any,
    frictionvel_inst: Any,
    surfalb_inst: Any,
    solarabs_inst: Any,
    mlcanopy_inst: mlcanopy_type,
    wateratm2lndbulk_inst: Any,
    waterdiagnosticbulk_inst: Any,
    grid: "GridInfo | None" = None,
    _o2ref_py: "float | None" = None,
    vcmaxpft_jax=None,
    g1_MED_jax=None,
    cos_zenith_device=None,
) -> mlcanopy_type:
    """
    Compute all multilayer canopy fluxes for one CLM timestep.

    Mirrors Fortran subroutine ``MLCanopyFluxes`` (lines 48-400).

    This is the top-level driver.  It:

    1. Resolves CLM time information (``nstep``, ``dtime_clm``,
       calendar days).
    2. Builds the ``filter_mlcan`` patch filter from the exposed-veg
       filter (Fortran lines 152-160).
    3. On the first CLM timestep (``zref == spval``), initialises PAD
       parameters, vertical structure, vertical profiles, and
       Runge-Kutta coefficients (Fortran lines 162-198).
    4. Copies CLM state into ``mlcanopy_inst`` via ``_GetCLMVar``
       (Fortran line 201).
    5. Seeds the ``*_bef`` forcing fields on the first timestep
       (Fortran lines 204-218).
    6. Updates the leaf/stem area profiles for current LAI/SAI and
       checks total PAI conservation (Fortran lines 220-243).
    7. Computes soil hydraulic resistance, plant resistance, leaf heat
       capacity, and soil relative humidity (Fortran lines 245-258).
    8. Loops over ``num_ml_steps`` sub-steps (Fortran lines 261-330):

       a. Saves previous-step state variables.
       b. Interpolates atmospheric forcing to the ML timestep.
       c. Calls physics in order:

          - :func:`SolarRadiation`
          - :func:`CanopyNitrogenProfile`
          - Inner Runge-Kutta loop (``irk = 1, nrk_steps+1``):

            - :func:`CanopyWettedFraction`
            - :func:`LongwaveRadiation`
            - Net radiation (inline, Fortran lines 301-307)
            - :func:`CanopyTurbulence`
            - :func:`LeafBoundaryLayer` (sun + shade)
            - :func:`LeafPhotosynthesis` (sun + shade)
            - :func:`FluxProfileSolution`
            - :func:`LeafWaterPotential` (sun + shade)
            - :func:`CanopyInterception` + :func:`CanopyEvaporation`
            - :func:`RungeKuttaUpdate` (for non-Euler steps)

       d. Accumulates fluxes over ML sub-steps.

    9. Saves current forcing as ``*_bef`` for next CLM timestep
       (Fortran lines 332-343).
    10. Calls :func:`_CanopyFluxesDiagnostics`.
    11. Merges sun/shade leaf temperature and water potential to a
        layer-mean prognostic state (Fortran lines 355-375).
    12. Copies selected fields back to CLM output variables when
        ``mlcan_to_clm == 1`` (Fortran lines 378-396).

    **Atmospheric forcing interpolation** (Fortran lines 105-130):

    .. code-block:: none

        met_type == 0: no interpolation; calday = 0
        met_type == 2: 2-point (end of interval); NOT VALID in this
                       implementation (endrun)
        met_type == 3: 3-point (centred); calday_cur = 0.5*(end+beg)

    **Runge-Kutta sub-stepping** (Fortran lines 265-271):

    .. code-block:: none

        runge_kutta_type == 10: Euler, nrk_steps = 0
        runge_kutta_type >= 20: nrk_steps = runge_kutta_type // 10

    **Sun/shade merge** (Fortran lines 361-375):

    .. code-block:: none

        if dpai(ic) > 0:
            tleaf_sun_merged = tleaf_sun*fracsun + tleaf_sha*(1-fracsun)
            tleaf_sha        = tleaf_sun_merged
            lwp_sun_merged   = lwp_sun*fracsun  + lwp_sha*(1-fracsun)
            lwp_sha          = lwp_sun_merged

    Args:
        bounds: CLM decomposition bounds (``begp``, ``endp``).
        num_exposedvegp: Number of non-snow-covered veg patches.
        filter_exposedvegp: CLM patch filter (1-based values).
        atm2lnd_inst: CLM atmosphere-to-land instance.
        canopystate_inst: CLM canopy state instance (LAI/SAI).
        soilstate_inst: CLM soil state instance.
        temperature_inst: CLM temperature instance.
        waterstatebulk_inst: CLM bulk water state instance.
        waterfluxbulk_inst: CLM bulk water flux instance.
        energyflux_inst: CLM energy flux instance (output receiver).
        frictionvel_inst: CLM friction velocity instance (output receiver).
        surfalb_inst: CLM surface albedo instance.
        solarabs_inst: CLM solar absorbed instance (output receiver).
        mlcanopy_inst: Multilayer canopy container; updated in-place
            over the timestep and returned.
        wateratm2lndbulk_inst: CLM water forcing instance.
        waterdiagnosticbulk_inst: CLM water diagnostic instance
            (output receiver).

    Returns:
        Updated :class:`mlcanopy_type` holding end-of-timestep state
        and CLM-timestep-mean fluxes.
    """
    import legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLclm_varctl as _ml_ctl  # for ml_vert_init mutation
    from legoesm.land.canopy.clm_ml_backend.clm_src_utils.clm_time_manager import (
        get_curr_calday,
        get_nstep,
        get_step_size,
    )  # noqa: F401

    _diff_mode = grid is not None

    # grid/diff mode (esp. the S3 column scan, where grid.p is TRACED) MUST use the
    # DEVICE solar zenith (cos_zenith_device): the host fallback in _GetCLMVar
    # recomputes zenith from per-step orbital globals via shr_orb_cosz and scatters
    # it at the (dummy) filter index — which under a traced grid.p writes patch 1 for
    # EVERY scanned column (silent wrong numerics).  Require it up front, loudly.
    if _diff_mode and cos_zenith_device is None:
        raise ValueError(
            "CLM-ML grid/diff mode requires cos_zenith_device (the per-column device "
            "solar zenith): the host shr_orb_cosz path cannot run under a traced patch "
            "index and would silently write the wrong column. Supply cos_zenith_device "
            "(the interface does this on the traceable path).")

    # Declare module-level RK coefficient cache as global so assignments
    # inside the if-block persist across calls.
    global _rk_ark, _rk_brk, _rk_crk

    # ------------------------------------------------------------------
    # CLM time information — Fortran lines 155-165
    # ------------------------------------------------------------------
    nstep = get_nstep()
    dtime_clm = float(get_step_size())

    curr_calday_end = float(get_curr_calday(offset=0))
    curr_calday_beg = float(get_curr_calday(offset=-int(dtime_clm)))

    # Interpolation calendar days — Fortran lines 105-130
    if met_type == 0:
        calday_interp_cur = 0.0
        calday_interp_bef = 0.0
        calday_interp_next = 0.0
    elif met_type == 2:
        endrun(msg=" ERROR: met_type not valid")
        calday_interp_cur = curr_calday_end
        calday_interp_bef = calday_interp_cur - dtime_clm / 86400.0
        calday_interp_next = 0.0
    elif met_type == 3:
        calday_interp_cur = 0.5 * (curr_calday_end + curr_calday_beg)
        calday_interp_bef = calday_interp_cur - dtime_clm / 86400.0
        calday_interp_next = calday_interp_cur + dtime_clm / 86400.0
    else:
        endrun(msg=" ERROR: MLCanopyFluxes: met_type not valid")
        calday_interp_cur = calday_interp_bef = calday_interp_next = 0.0

    # Number of ML sub-steps — Fortran line 132
    # Re-read from module so test code can override MLclm_varctl.dtime_ml at runtime.
    num_ml_steps: int = int(dtime_clm / _ml_ctl.dtime_ml)

    # ------------------------------------------------------------------
    # Build filter for patches to process — Fortran lines 152-160
    # ------------------------------------------------------------------
    num_mlcan: int = 0
    _filter_mlcan_list: List[int] = []
    for fp in range(1, num_exposedvegp + 1):
        p = int(filter_exposedvegp[fp - 1])
        num_mlcan += 1
        _filter_mlcan_list.append(p)
    filter_mlcan: tuple = tuple(_filter_mlcan_list)  # tuple required for jax.jit static_argnums

    # ------------------------------------------------------------------
    # First-timestep initialisation — Fortran lines 162-198
    # Triggered when any zref == spval (unset sentinel)
    # ------------------------------------------------------------------
    _ml_vert_init = 0
    if not _diff_mode:
        for p in filter_mlcan:
            if float(mlcanopy_inst.zref_forcing[p]) == spval:
                _ml_vert_init = 1
                break
    _ml_ctl.ml_vert_init = _ml_vert_init  # expose to other modules

    if _ml_vert_init == 1:

        if masterproc:
            print("Attempting to initialize multilayer canopy pad parameters .....")
        mlcanopy_inst = getPADparameters(num_mlcan, filter_mlcan, mlcanopy_inst)
        if masterproc:
            print("Successfuly initialized multilayer canopy pad parameters")

        if masterproc:
            print("Attempting to initialize multilayer canopy vertical structure .....")
        mlcanopy_inst = initVerticalStructure(
            bounds, num_mlcan, filter_mlcan, canopystate_inst, frictionvel_inst, mlcanopy_inst
        )
        mlcanopy_inst = initVerticalProfiles(
            num_mlcan, filter_mlcan, atm2lnd_inst, wateratm2lndbulk_inst, mlcanopy_inst
        )
        if masterproc:
            print("Successfully initialized multilayer canopy vertical structure")

        # Runge-Kutta coefficients — Fortran line 197
        _rk_ark, _rk_brk, _rk_crk = RungeKuttaIni()

    # Retrieve cached Runge-Kutta coefficients (valid after first call)
    ark, brk, crk = _rk_ark, _rk_brk, _rk_crk

    # ------------------------------------------------------------------
    # Copy CLM variables to mlcanopy container — Fortran line 201
    # ------------------------------------------------------------------
    mlcanopy_inst = _GetCLMVar(
        nstep,
        dtime_clm,
        num_mlcan,
        filter_mlcan,
        atm2lnd_inst,
        soilstate_inst,
        temperature_inst,
        surfalb_inst,
        wateratm2lndbulk_inst,
        mlcanopy_inst,
        cos_zenith_device=cos_zenith_device,
        grid=grid,
    )
    # ------------------------------------------------------------------
    # Seed *_bef forcing on first timestep — Fortran lines 204-218
    # ------------------------------------------------------------------
    if _ml_vert_init == 1:
        uref_bef = mlcanopy_inst.uref_bef_forcing
        tref_bef = mlcanopy_inst.tref_bef_forcing
        qref_bef = mlcanopy_inst.qref_bef_forcing
        pref_bef = mlcanopy_inst.pref_bef_forcing
        co2ref_bef = mlcanopy_inst.co2ref_bef_forcing
        swskyb_bef = mlcanopy_inst.swskyb_bef_forcing
        swskyd_bef = mlcanopy_inst.swskyd_bef_forcing
        lwsky_bef = mlcanopy_inst.lwsky_bef_forcing

        for p in filter_mlcan:
            uref_bef = uref_bef.at[p].set(mlcanopy_inst.uref_cur_forcing[p])
            tref_bef = tref_bef.at[p].set(mlcanopy_inst.tref_cur_forcing[p])
            qref_bef = qref_bef.at[p].set(mlcanopy_inst.qref_cur_forcing[p])
            pref_bef = pref_bef.at[p].set(mlcanopy_inst.pref_cur_forcing[p])
            co2ref_bef = co2ref_bef.at[p].set(mlcanopy_inst.co2ref_cur_forcing[p])
            for ib in (ivis, inir):
                swskyb_bef = swskyb_bef.at[p, ib].set(mlcanopy_inst.swskyb_cur_forcing[p, ib])
                swskyd_bef = swskyd_bef.at[p, ib].set(mlcanopy_inst.swskyd_cur_forcing[p, ib])
            lwsky_bef = lwsky_bef.at[p].set(mlcanopy_inst.lwsky_cur_forcing[p])

        mlcanopy_inst = mlcanopy_inst._replace(
            uref_bef_forcing=uref_bef,
            tref_bef_forcing=tref_bef,
            qref_bef_forcing=qref_bef,
            pref_bef_forcing=pref_bef,
            co2ref_bef_forcing=co2ref_bef,
            swskyb_bef_forcing=swskyb_bef,
            swskyd_bef_forcing=swskyd_bef,
            lwsky_bef_forcing=lwsky_bef,
        )

    # ------------------------------------------------------------------
    # Update leaf/stem area profiles for current LAI/SAI — Fortran lines 220-243
    # ------------------------------------------------------------------
    lai = mlcanopy_inst.lai_canopy
    sai = mlcanopy_inst.sai_canopy
    dlai = mlcanopy_inst.dlai_profile
    dsai = mlcanopy_inst.dsai_profile
    dpai = mlcanopy_inst.dpai_profile

    # grid.p may be a TRACED lax.scan-over-columns index (S3): iterate the single
    # patch grid.p under the map, else the concrete filter.  grid.ncan is concrete
    # (uniform structure) so the 1:ncan+1 slice bound below stays static.
    for p in ([grid.p] if _diff_mode else filter_mlcan):
        lai_val = canopystate_inst.elai_patch[p]
        sai_val = canopystate_inst.esai_patch[p]
        lai = lai.at[p].set(lai_val)
        sai = sai.at[p].set(sai_val)

        _ncan = grid.ncan if _diff_mode else int(mlcanopy_inst.ncan_canopy[p])
        # Vectorized slice assignment replaces a per-layer Python for-loop.
        # 3 slice scatter ops instead of 3×ncan scalar scatters → smaller
        # XLA trace and better kernel fusion inside lax.scan.
        _sl_lai = slice(1, _ncan + 1)
        _dlai_sl = mlcanopy_inst.dlai_frac_profile[p, _sl_lai] * lai_val
        _dsai_sl = mlcanopy_inst.dsai_frac_profile[p, _sl_lai] * sai_val
        dlai = dlai.at[p, _sl_lai].set(_dlai_sl)
        dsai = dsai.at[p, _sl_lai].set(_dsai_sl)
        dpai = dpai.at[p, _sl_lai].set(_dlai_sl + _dsai_sl)

        # PAI conservation check — Fortran lines 239-242
        if not _diff_mode:
            totpai = sum(float(dpai[p, ic]) for ic in range(1, _ncan + 1))
            if abs(totpai - float(lai_val + sai_val)) > 1.0e-6:
                endrun(msg=" ERROR: MLCanopyFluxes: plant area index not updated correctly")

    mlcanopy_inst = mlcanopy_inst._replace(
        lai_canopy=lai,
        sai_canopy=sai,
        dlai_profile=dlai,
        dsai_profile=dsai,
        dpai_profile=dpai,
    )

    # ------------------------------------------------------------------
    # Plant hydraulics and leaf heat capacity — Fortran lines 245-257
    # ------------------------------------------------------------------
    mlcanopy_inst = SoilResistance(
        num_mlcan, filter_mlcan, soilstate_inst, waterstatebulk_inst, mlcanopy_inst,
        grid=grid,
    )
    mlcanopy_inst = PlantResistance(num_mlcan, filter_mlcan, mlcanopy_inst, grid=grid)
    mlcanopy_inst = LeafHeatCapacity(num_mlcan, filter_mlcan, mlcanopy_inst, grid=grid)

    # Soil surface relative humidity — Fortran lines 259-262
    # patch.column as a DEVICE (jnp) array indexed by dynamic gather — valid for a
    # concrete p (identity map c==p single-site, value-identical) AND a traced scan
    # index (S3).  (Was np.asarray + int(), which breaks on a traced p.)
    _patch_col_rhg = jnp.asarray(patch.column)
    rhg = mlcanopy_inst.rhg_soil
    for p in ([grid.p] if _diff_mode else filter_mlcan):  # grid.p may be TRACED (S3)
        c = _patch_col_rhg[p]
        smp1 = soilstate_inst.smp_l_col[c, 1]  # mm
        tsoi1 = temperature_inst.t_soisno_col[c, 1]  # K
        rhg_val = jnp.exp(grav * mmh2o * smp1 * 1.0e-3 / (rgas * tsoi1))
        rhg = rhg.at[p].set(rhg_val)
    mlcanopy_inst = mlcanopy_inst._replace(rhg_soil=rhg)

    # ------------------------------------------------------------------
    # Flux accumulators — Fortran lines 57-64 (local arrays)
    # Skipped in differentiable mode (diagnostics only).
    # ------------------------------------------------------------------
    # Flux accumulators are needed in BOTH modes.  Previously init'd only in the
    # eager path ("Skipped in differentiable mode (diagnostics only)"): the scan/
    # diff path then returned only the FINAL sub-step's instantaneous state, which
    # drops the canopy-air storage term (stflx_air/stair have equilibrated to ~0 by
    # the last sub-step) and shifts that energy (~5 W/m^2) into sensible heat —
    # diverging ~40% in shflx from the eager Fortran-faithful sub-step MEAN.  The
    # scan branch below now carries these accumulators and time-averages too.
    nvar1d = 23
    nvar2d = 14
    nvar3d = 12
    endp = bounds.endp
    flux_accumulator = jnp.zeros((endp + 1, nvar1d))
    flux_accumulator_profile = jnp.zeros((endp + 1, nlevmlcan + 2, nvar2d))
    flux_accumulator_leaf = jnp.zeros((endp + 1, nlevmlcan + 1, nleaf + 1, nvar3d))

    # ------------------------------------------------------------------
    # Runge-Kutta configuration — Fortran lines 265-271
    # Re-read from module so test code can override MLclm_varctl.runge_kutta_type at runtime.
    # ------------------------------------------------------------------
    _runge_kutta_type = _ml_ctl.runge_kutta_type
    if _runge_kutta_type == 10:
        nrk_steps = 0
    elif _runge_kutta_type >= 20:
        nrk_steps = _runge_kutta_type // 10
    else:
        nrk_steps = 0

    if masterproc and nstep == 1:
        print(
            f"MLCanopyFluxes: starting ML sub-steps (num_ml_steps={num_ml_steps}, "
            f"nrk_steps={nrk_steps}, num_mlcan={num_mlcan})",
            flush=True,
        )

    # Keep heartbeat output bounded while still proving forward progress.
    progress_stride = max(1, num_ml_steps // 10)

    # Pre-compute ncan for each active patch once — avoids int() device-to-host
    # sync inside the hot ML sub-step loop.  ncan is constant for a fixed site
    # so this tuple is computed only once per CLM timestep.
    if _diff_mode:
        ncan_vals: tuple = tuple(grid.ncan for _ in filter_mlcan)
    else:
        ncan_vals: tuple = tuple(int(mlcanopy_inst.ncan_canopy[p]) for p in filter_mlcan)

    # Pre-extract o2ref_p as a Python float before jax.checkpoint tracing.
    # o2ref_forcing is atmospheric O2 — constant for the entire simulation.
    # Used by LeafPhotosynthesis as a lru_cache key (must be a Python float).
    # In diff mode, mlcanopy_inst may be abstract under jax.grad tracing,
    # so we accept it as a pre-extracted parameter (_o2ref_py) from make_clm_ml_forward.
    if _diff_mode:
        if _o2ref_py is not None:
            _o2ref_py_val: float = _o2ref_py
        else:
            # grid.p may be TRACED (S3 column scan), so float(o2ref[grid.p]) would
            # fail at trace time — fail loud instead of a cryptic tracer error. The
            # interface always supplies _o2ref_py; a direct grid-mode caller must too.
            raise ValueError(
                "CLM-ML grid/diff mode requires a concrete _o2ref_py float: under the "
                "column scan grid.p is a tracer, so float(mlcanopy_inst.o2ref_forcing"
                "[grid.p]) cannot be evaluated at trace time. "
                "Pass _o2ref_py=float(canopy_config.o2ref).")
    else:
        _o2ref_py_val = None  # non-diff mode: LeafPhotosynthesis reads it directly

    # ==================================================================
    # Multilayer canopy time-stepping loop — Fortran lines 261-330
    # ==================================================================
    # DIFF MODE: uses jax.lax.scan over a pre-computed calday array.
    #   - Single Python→XLA dispatch (vs num_ml_steps separate traces).
    #   - XLA sees the loop body once; can fuse/optimize across sub-steps.
    #   - jax.checkpoint on the scan body gives O(step_mem) backward memory.
    #   - Works because all physics functions are JAX-traceable when grid
    #     is not None (no int()/float() device-host syncs in that path).
    #   - TimeInterpolation3 already uses jnp.where for traced calday_ml.
    #
    # NON-DIFF MODE: Python for loop.
    #   - Physics functions with grid=None perform int(arr[p])/float(arr[p])
    #     device-host syncs that prevent XLA tracing via lax.fori_loop.
    #   - Future path to lax.fori_loop: add grid-like static caching to
    #     each physics module so all paths are XLA-traceable.
    # ==================================================================

    # nstep_ml = 0 before _physics_step_fn is defined so the closure is
    # valid in diff mode (lax.scan path has no Python for-loop to set it).
    # In non-diff mode the for-loop below updates this variable each iteration;
    # the closure in _physics_step_fn always sees the current value.
    nstep_ml = 0

    # ------------------------------------------------------------------
    # Per-step physics function.
    # Closes over nstep_ml: 0 in diff mode (ignored by _HF2008_diff),
    # current loop value in non-diff mode (used only for HF2008 warning).
    # calday_ml may be a Python float (non-diff) or JAX traced array (diff).
    # ------------------------------------------------------------------
    def _physics_step_fn(inst, calday_ml):
        """Pure (inst, calday_ml) → inst for one ML sub-step."""
        # Save previous-step state — Fortran lines 272-285
        inst = _copy_bef_state(
            filter_mlcan, ncan_vals, inst,
            grid_p=(grid.p if grid is not None else None),
            grid_ncan=(grid.ncan if grid is not None else None),
        )
        # Atmospheric forcing — Fortran line 287
        # calday_ml may be a JAX traced array in lax.scan diff mode;
        # TimeInterpolation3 uses jnp.where so this is safe.
        inst = GetAtmForcing(
            calday_interp_bef,
            calday_interp_cur,
            calday_interp_next,
            calday_ml,
            num_mlcan,
            filter_mlcan,
            inst,
            grid=grid,
        )
        # Solar radiation — Fortran line 290
        inst = SolarRadiation(bounds, num_mlcan, filter_mlcan, inst, grid=grid)
        # Nitrogen profile — Fortran line 293
        inst = CanopyNitrogenProfile(num_mlcan, filter_mlcan, inst, vcmaxpft_jax, grid=grid)
        # Runge-Kutta inner loop — Fortran lines 310-328
        for _irk in range(1, nrk_steps + 2):
            inst = CanopyWettedFraction(num_mlcan, filter_mlcan, inst, grid=grid)
            inst = LongwaveRadiation(bounds, num_mlcan, filter_mlcan, inst, grid=grid)
            inst = inst._replace(
                rnleaf_leaf=(
                    inst.swleaf_leaf[..., ivis] + inst.swleaf_leaf[..., inir] + inst.lwleaf_leaf
                ),
                rnsoi_soil=(inst.swsoi_soil[:, ivis] + inst.swsoi_soil[:, inir] + inst.lwsoi_soil),
            )
            # nstep_ml is 0 in diff mode; _HF2008_diff ignores it entirely.
            # In non-diff mode it carries the current sub-step index (warning only).
            inst = CanopyTurbulence(nstep_ml, num_mlcan, filter_mlcan, inst, grid=grid)
            # Both sun and shade in one fused GPU dispatch (2× fewer round-trips)
            inst = LeafBoundaryLayerBoth(num_mlcan, filter_mlcan, inst, grid=grid)
            inst = LeafPhotosynthesis(
                num_mlcan,
                filter_mlcan,
                isun,
                inst,
                grid=grid,
                _o2ref_py=_o2ref_py_val,
                g1_MED_jax=g1_MED_jax,
            )
            inst = LeafPhotosynthesis(
                num_mlcan,
                filter_mlcan,
                isha,
                inst,
                grid=grid,
                _o2ref_py=_o2ref_py_val,
                g1_MED_jax=g1_MED_jax,
            )
            inst = FluxProfileSolution(num_mlcan, filter_mlcan, inst, grid=grid)
            inst = LeafWaterPotential(num_mlcan, filter_mlcan, isun, inst, grid=grid)
            inst = LeafWaterPotential(num_mlcan, filter_mlcan, isha, inst, grid=grid)
            inst = CanopyInterception(num_mlcan, filter_mlcan, inst, grid=grid)
            inst = CanopyEvaporation(num_mlcan, filter_mlcan, inst, grid=grid)
            if nrk_steps > 0 and _irk <= nrk_steps:
                inst = RungeKuttaUpdate(
                    _irk,
                    ark,
                    brk,
                    crk,
                    num_mlcan,
                    filter_mlcan,
                    inst,
                    grid=grid,
                )
        return inst

    # Pre-compute dtime_ml once (used by both mode branches below)
    _dtime_ml = _ml_ctl.dtime_ml

    if _diff_mode:
        # ------------------------------------------------------------------
        # DIFF MODE: lax.scan over pre-computed calday array
        # ------------------------------------------------------------------
        # Build a small (num_ml_steps,) array of calendar days — one per
        # sub-step.  These are Python floats at construction time; inside
        # lax.scan each element is a JAX traced scalar.
        if met_type in (0, 2):
            _calday_arr = jnp.array(
                [
                    curr_calday_beg + float(k + 1) * (_dtime_ml / 86400.0)
                    for k in range(num_ml_steps)
                ],
                dtype=jnp.float64,
            )
        else:  # met_type == 3
            _calday_arr = jnp.array(
                [
                    curr_calday_beg + (float(k + 1) - 0.5) * (_dtime_ml / 86400.0)
                    for k in range(num_ml_steps)
                ],
                dtype=jnp.float64,
            )

        import os as _os

        _use_checkpoint = _os.environ.get("CLM_ML_NO_CHECKPOINT", "0") != "1"

        def _scan_body(carry, calday_ml_x):
            """lax.scan body: one ML sub-step + flux accumulation.

            carry = (inst, flux_accumulator, flux_accumulator_profile,
            flux_accumulator_leaf).  Accumulating INSIDE the scan (instead of
            returning only the final sub-step) is what reproduces the eager path's
            sub-step-MEAN fluxes — including the stiff canopy-air storage term.
            """
            inst, fa, fap, fal = carry
            inst = _physics_step_fn(inst, calday_ml_x)
            fa, fap, fal = _MLAccumulateFluxes(
                num_mlcan, filter_mlcan, ncan_vals, fa, fap, fal, inst, grid=grid)
            return (inst, fa, fap, fal), None

        # jax.checkpoint on the scan body: recomputes activations during
        # backward instead of storing them — O(step_mem) not O(N*step_mem).
        _scan_body_fn = jax.checkpoint(_scan_body) if _use_checkpoint else _scan_body

        _init_carry = (mlcanopy_inst, flux_accumulator,
                       flux_accumulator_profile, flux_accumulator_leaf)
        (mlcanopy_inst, flux_accumulator, flux_accumulator_profile,
         flux_accumulator_leaf), _ = jax.lax.scan(
            _scan_body_fn, _init_carry, _calday_arr)

        # Time-average the accumulated sub-step fluxes (sum / num_ml_steps) and
        # write them back into mlcanopy_inst — the SAME reduction the eager path
        # applies (Fortran line 372), so the differentiable/jit path reproduces the
        # Fortran-faithful sub-step-mean fluxes (incl. the canopy-air storage term).
        (flux_accumulator, flux_accumulator_profile, flux_accumulator_leaf,
         mlcanopy_inst) = _MLScaleAndWriteBack(
            num_ml_steps, num_mlcan, filter_mlcan, ncan_vals,
            flux_accumulator, flux_accumulator_profile, flux_accumulator_leaf,
            mlcanopy_inst, grid=grid)

    else:
        # ------------------------------------------------------------------
        # NON-DIFF MODE: Python for loop
        # ------------------------------------------------------------------
        for nstep_ml in range(1, num_ml_steps + 1):
            if (
                masterproc
                and nstep == 1
                and (nstep_ml == 1 or nstep_ml == num_ml_steps or (nstep_ml % progress_stride == 0))
            ):
                print(
                    f"MLCanopyFluxes: sub-step {nstep_ml}/{num_ml_steps}",
                    flush=True,
                )

            # Calendar day for this ML sub-step — Fortran lines 263-270
            if met_type in (0, 2):
                calday_interp_ml = curr_calday_beg + float(nstep_ml) * (_dtime_ml / 86400.0)
            else:  # met_type == 3
                calday_interp_ml = curr_calday_beg + (float(nstep_ml) - 0.5) * (_dtime_ml / 86400.0)

            mlcanopy_inst = _physics_step_fn(mlcanopy_inst, calday_interp_ml)

            # Accumulate fluxes over ML sub-steps — Fortran line 372
            flux_accumulator, flux_accumulator_profile, flux_accumulator_leaf = _MLAccumulateFluxes(
                num_mlcan,
                filter_mlcan,
                ncan_vals,
                flux_accumulator,
                flux_accumulator_profile,
                flux_accumulator_leaf,
                mlcanopy_inst,
                grid=grid,
            )

        # Scale accumulated sums by 1/num_ml_steps and write back to inst
        flux_accumulator, flux_accumulator_profile, flux_accumulator_leaf, mlcanopy_inst = (
            _MLScaleAndWriteBack(
                num_ml_steps,
                num_mlcan,
                filter_mlcan,
                ncan_vals,
                flux_accumulator,
                flux_accumulator_profile,
                flux_accumulator_leaf,
                mlcanopy_inst,
                grid=grid,
            )
        )

    # End multilayer time-stepping loop
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Save current forcing as *_bef for next CLM timestep — Fortran lines 332-343
    # JIT-compiled helper fuses 8 scatter ops into one XLA dispatch.
    # ------------------------------------------------------------------
    mlcanopy_inst = _save_bef_forcing(
        filter_mlcan, mlcanopy_inst,
        grid_p=(grid.p if grid is not None else None),
    )

    # ------------------------------------------------------------------
    # Canopy-level diagnostics — Fortran line 346
    # ------------------------------------------------------------------
    # Runs in BOTH modes: this routine populates the canopy-integrated flux
    # outputs (shflx/lhflx/etflx/gpp/rnet/swveg/albcan/taveg/stflx_air/stflx_veg
    # …) that downstream consumers read.  In diff mode it takes ``grid=`` so its
    # structural ints and energy-balance ``endrun`` checks stay trace-safe.
    mlcanopy_inst = _CanopyFluxesDiagnostics(
        num_mlcan, filter_mlcan, mlcanopy_inst, grid=grid
    )

    # ------------------------------------------------------------------
    # Merge sun/shade leaf temperature and water potential — Fortran lines 355-375
    # State is prognostic; sun/shade fraction changes between CLM steps,
    # so merge to a layer-mean before the next timestep.  Runs in both modes:
    # the merge is pure JAX (``ncan_vals`` slices + ``jnp.where``) so it stays on
    # the grad tape, and it keeps the prognostic ``tleaf``/``lwp`` state correct
    # for a differentiable multi-step warm-start carry.
    # ------------------------------------------------------------------
    tleaf = mlcanopy_inst.tleaf_leaf
    tleaf_hist = mlcanopy_inst.tleaf_hist_leaf
    lwp = mlcanopy_inst.lwp_leaf
    lwp_hist = mlcanopy_inst.lwp_hist_leaf

    # ncan_vals is pre-computed above as a tuple of concrete ints
    # (one per active patch), avoiding int() D->H syncs here.  Under S3 the patch
    # index grid.p may be TRACED (lax.scan over columns); grid.ncan stays concrete
    # (uniform structure) so the 1:ncan+1 slice bound is static.
    for p, _ncan in ([(grid.p, grid.ncan)] if _diff_mode else zip(filter_mlcan, ncan_vals)):
        _sl = slice(1, _ncan + 1)

        # Save sun/shade values for history files (bulk copy, uses JAX .at[])
        tleaf_hist = tleaf_hist.at[p, _sl, :].set(tleaf[p, _sl, :])
        lwp_hist = lwp_hist.at[p, _sl, :].set(lwp[p, _sl, :])

        # Vectorized merge to layer mean — Fortran lines 367-374
        # Pure JAX: no np.asarray() D->H sync; all ops stay on device.
        dpai_p = mlcanopy_inst.dpai_profile[p, _sl]  # shape (ncan,)
        fracsun_p = mlcanopy_inst.fracsun_profile[p, _sl]  # shape (ncan,)
        tleaf_p = tleaf[p, _sl, :]  # shape (ncan, nleaf+1)
        lwp_p = lwp[p, _sl, :]  # shape (ncan, nleaf+1)

        has_pai = dpai_p > 0.0  # (ncan,)

        t_sun = tleaf_p[:, isun]
        t_sha = tleaf_p[:, isha]
        t_mean = t_sun * fracsun_p + t_sha * (1.0 - fracsun_p)
        tleaf_p_new = tleaf_p.at[:, isun].set(jnp.where(has_pai, t_mean, t_sun))
        tleaf_p_new = tleaf_p_new.at[:, isha].set(jnp.where(has_pai, t_mean, t_sha))

        l_sun = lwp_p[:, isun]
        l_sha = lwp_p[:, isha]
        l_mean = l_sun * fracsun_p + l_sha * (1.0 - fracsun_p)
        lwp_p_new = lwp_p.at[:, isun].set(jnp.where(has_pai, l_mean, l_sun))
        lwp_p_new = lwp_p_new.at[:, isha].set(jnp.where(has_pai, l_mean, l_sha))

        tleaf = tleaf.at[p, _sl, :].set(tleaf_p_new)
        lwp = lwp.at[p, _sl, :].set(lwp_p_new)

    mlcanopy_inst = mlcanopy_inst._replace(
        tleaf_leaf=tleaf,
        tleaf_hist_leaf=tleaf_hist,
        lwp_leaf=lwp,
        lwp_hist_leaf=lwp_hist,
    )

    # ------------------------------------------------------------------
    # Copy to CLM output variables (when mlcan_to_clm == 1) — Fortran lines 378-396
    # ------------------------------------------------------------------
    if mlcan_to_clm == 1:
        for p in filter_mlcan:
            for ib in (ivis, inir):
                surfalb_inst.albd_patch[p, ib] = 0.0
                surfalb_inst.albi_patch[p, ib] = 0.0
            energyflux_inst.taux_patch[p] = 0.0
            energyflux_inst.tauy_patch[p] = 0.0
            energyflux_inst.eflx_lh_tot_patch[p] = mlcanopy_inst.lhflx_canopy[p]
            energyflux_inst.eflx_sh_tot_patch[p] = mlcanopy_inst.shflx_canopy[p]
            energyflux_inst.eflx_lwrad_out_patch[p] = mlcanopy_inst.lwup_canopy[p]
            waterfluxbulk_inst.qflx_evap_tot_patch[p] = mlcanopy_inst.etflx_canopy[p] * mmh2o
            frictionvel_inst.fv_patch[p] = mlcanopy_inst.ustar_canopy[p]
            frictionvel_inst.u10_clm_patch[p] = 0.0
            temperature_inst.t_ref2m_patch[p] = 0.0
            waterdiagnosticbulk_inst.q_ref2m_patch[p] = 0.0
            solarabs_inst.fsa_patch[p] = (
                mlcanopy_inst.swveg_canopy[p, ivis]
                + mlcanopy_inst.swveg_canopy[p, inir]
                + mlcanopy_inst.swsoi_soil[p, ivis]
                + mlcanopy_inst.swsoi_soil[p, inir]
            )

    return mlcanopy_inst


def _GetCLMVar(
    nstep: int,
    dtime_clm: float,
    num_filter: int,
    filter: List[int],
    atm2lnd_inst: Any,
    soilstate_inst: Any,
    temperature_inst: Any,
    surfalb_inst: Any,
    wateratm2lndbulk_inst: Any,
    mlcanopy_inst: mlcanopy_type,
    cos_zenith_device: Any = None,
    grid: "GridInfo | None" = None,
) -> mlcanopy_type:
    """
    Copy CLM variables into the multilayer canopy container.

    Mirrors Fortran subroutine ``GetCLMVar`` (lines 1-120).

    All CLM variables are copied verbatim into the corresponding
    ``mlcanopy_inst`` fields.  The only arithmetic beyond indexing is:

    - Wind speed: ``uref_cur(p) = sqrt(forc_u(g)^2 + forc_v(g)^2)``
      (resultant from east and north components).
    - CO2 unit conversion:
      ``co2ref_cur(p) = forc_pco2(g) / forc_pbot(c) * 1e6``
      (Pa → umol/mol).
    - O2 unit conversion:
      ``o2ref(p) = forc_po2(g) / forc_pbot(c) * 1e3``
      (Pa → mmol/mol).
    - First snow/soil layer depth:
      ``soil_dz(p) = z(c, snl(c)+1) - zi(c, snl(c))``
      (layer mid-point depth minus interface depth above it).

    Solar zenith angle (Fortran lines 92-110):

    The zenith angle is computed for the *beginning* of the CLM
    timestep, i.e. at calendar day ``caldaym1 = calday(nstep-1)``.
    The call chain is::

        caldaym1  → shr_orb_decl  → declinm1
        (caldaym1, lat, lon, declinm1) → shr_orb_cosz → coszen
        solar_zen(p) = acos(max(0.01, coszen))

    The ``max(0.01, coszen)`` floor prevents ``acos`` from returning
    values beyond ``pi/2`` (i.e. clamps the sun no lower than ~89.4°
    below horizontal), matching the Fortran exactly.

    .. warning::
        ``soil_t``, ``soil_dz``, and ``soil_tk`` (Fortran lines 84-88)
        are required for the soil heat flux calculation.  When run
        **coupled to CLM**, ``t_soisno`` is updated by CLM's own
        ``SoilTemperatureMod``, *not* by the multilayer canopy's
        ``MLSoilTemperatureMod``.  Therefore the soil temperature seen
        here is independent of the multilayer canopy soil energy
        balance.

    Args:
        nstep: Current CLM timestep number.
        dtime_clm: CLM timestep duration (s).
        num_filter: Number of patches in filter.
        filter: Patch index filter (1-based values).
        atm2lnd_inst: CLM atmosphere-to-land instance.  Required
            fields:

            - ``forc_u_grc[g]``, ``forc_v_grc[g]`` — wind components (m/s)
            - ``forc_pco2_grc[g]`` — CO2 partial pressure (Pa)
            - ``forc_po2_grc[g]`` — O2 partial pressure (Pa)
            - ``forc_solai_grc[g, ib]`` — diffuse SW by waveband (W/m2)
            - ``forc_solad_downscaled_col[c, ib]`` — direct SW (W/m2)
            - ``forc_t_downscaled_col[c]`` — air temperature (K)
            - ``forc_pbot_downscaled_col[c]`` — air pressure (Pa)
            - ``forc_lwrad_downscaled_col[c]`` — longwave (W/m2)

        soilstate_inst: CLM soil state instance.  Required fields:

            - ``soilresis_col[c]`` — soil evaporative resistance (s/m)
            - ``thk_col[c, j]`` — thermal conductivity (W/m/K)

        temperature_inst: CLM temperature instance.  Required fields:

            - ``t_a10_patch[p]`` — 10-day mean 2-m air temperature (K)
            - ``t_soisno_col[c, j]`` — soil/snow temperature (K)

        surfalb_inst: CLM surface albedo instance.  Required fields:

            - ``albgrd_col[c, ib]`` — direct beam ground albedo (-)
            - ``albgri_col[c, ib]`` — diffuse ground albedo (-)

        wateratm2lndbulk_inst: CLM water forcing instance.  Required
            fields:

            - ``forc_q_downscaled_col[c]`` — specific humidity (kg/kg)
            - ``forc_rain_downscaled_col[c]`` — rainfall (mm/s)
            - ``forc_snow_downscaled_col[c]`` — snowfall (mm/s)

        mlcanopy_inst: Canopy container to update.

    Returns:
        Updated :class:`mlcanopy_type` with all ``*_cur_forcing``,
        ``*_soil``, and ``solar_zen_forcing`` fields set for the
        current CLM timestep.
    """
    from legoesm.land.canopy.clm_ml_backend.clm_share.shr_orb_mod import shr_orb_cosz, shr_orb_decl  # noqa: F401
    from legoesm.land.canopy.clm_ml_backend.clm_src_main.clm_varcon import pi  # noqa: F401 (pi = rpi)
    from legoesm.land.canopy.clm_ml_backend.clm_src_main.clm_varpar import (  # noqa: F401
        inir,
        ivis,
        nlevsno,  # noqa: F401
    )
    from legoesm.land.canopy.clm_ml_backend.clm_src_main.ColumnType import col  # noqa: F401
    from legoesm.land.canopy.clm_ml_backend.clm_src_main.GridcellType import grc  # noqa: F401
    from legoesm.land.canopy.clm_ml_backend.clm_src_main.PatchType import patch  # noqa: F401
    from legoesm.land.canopy.clm_ml_backend.clm_src_utils.clm_time_manager import get_curr_calday  # noqa: F401
    from legoesm.land.canopy.clm_ml_backend.clm_src_utils.clm_varorb import eccen, lambm0, mvelpp, obliqr  # noqa: F401

    # ------------------------------------------------------------------
    # Unpack CLM inputs (mirrors Fortran associate block)
    # ------------------------------------------------------------------
    forc_u = atm2lnd_inst.forc_u_grc
    forc_v = atm2lnd_inst.forc_v_grc
    forc_pco2 = atm2lnd_inst.forc_pco2_grc
    forc_po2 = atm2lnd_inst.forc_po2_grc
    forc_solad_col = atm2lnd_inst.forc_solad_downscaled_col
    forc_solai = atm2lnd_inst.forc_solai_grc
    forc_t = atm2lnd_inst.forc_t_downscaled_col
    forc_pbot = atm2lnd_inst.forc_pbot_downscaled_col
    forc_lwrad = atm2lnd_inst.forc_lwrad_downscaled_col
    forc_q = wateratm2lndbulk_inst.forc_q_downscaled_col
    forc_rain = wateratm2lndbulk_inst.forc_rain_downscaled_col
    forc_snow = wateratm2lndbulk_inst.forc_snow_downscaled_col
    albgrd = surfalb_inst.albgrd_col
    albgri = surfalb_inst.albgri_col
    soilresis = soilstate_inst.soilresis_col
    thk = soilstate_inst.thk_col
    t_a10_patch = temperature_inst.t_a10_patch
    t_soisno = temperature_inst.t_soisno_col
    snl = col.snl
    z = col.z
    zi = col.zi

    # Per-column CLM topology maps read by the patch index.  Kept as DEVICE (jnp)
    # arrays and indexed by ``p`` via dynamic gather, so this copy is valid whether
    # ``p`` is a concrete Python int (the per-column-call path) OR a TRACED scan
    # index (the S3 lax.scan-over-columns path) — a value-identical no-op for the
    # concrete case.  (Previously np.asarray + int(), which breaks on a traced p.)
    _snl_j = jnp.asarray(snl)
    _patch_column_j = jnp.asarray(patch.column)
    _patch_gridcell_j = jnp.asarray(patch.gridcell)

    # ------------------------------------------------------------------
    # Mutable copies of the mlcanopy arrays to update
    # ------------------------------------------------------------------
    tref_cur = mlcanopy_inst.tref_cur_forcing
    qref_cur = mlcanopy_inst.qref_cur_forcing
    uref_cur = mlcanopy_inst.uref_cur_forcing
    pref_cur = mlcanopy_inst.pref_cur_forcing
    co2ref_cur = mlcanopy_inst.co2ref_cur_forcing
    o2ref = mlcanopy_inst.o2ref_forcing
    solar_zen = mlcanopy_inst.solar_zen_forcing
    swskyb_cur = mlcanopy_inst.swskyb_cur_forcing
    swskyd_cur = mlcanopy_inst.swskyd_cur_forcing
    lwsky_cur = mlcanopy_inst.lwsky_cur_forcing
    qflx_rain = mlcanopy_inst.qflx_rain_forcing
    qflx_snow = mlcanopy_inst.qflx_snow_forcing
    tacclim = mlcanopy_inst.tacclim_forcing
    albsoib = mlcanopy_inst.albsoib_soil
    albsoid = mlcanopy_inst.albsoid_soil
    soilres = mlcanopy_inst.soilres_soil
    soil_t = mlcanopy_inst.soil_t_soil
    soil_dz = mlcanopy_inst.soil_dz_soil
    soil_tk = mlcanopy_inst.soil_tk_soil

    # ------------------------------------------------------------------
    # Copy CLM variables to multilayer canopy — Fortran lines 63-90
    # ------------------------------------------------------------------
    for fp in range(1, 2 if grid is not None else num_filter + 1):  # Fortran: do fp = 1, num_filter
        p = grid.p if grid is not None else int(filter[fp - 1])  # grid.p may be TRACED (S3)
        # Dynamic gather (traced-p-safe): identity maps in single-site mode
        # (c == p, g == p) so the values are unchanged for a concrete p.
        c = _patch_column_j[p]
        g = _patch_gridcell_j[p]

        # Wind: resultant from east/north components — Fortran line 67
        uref_cur = uref_cur.at[p].set(jnp.sqrt(forc_u[g] ** 2 + forc_v[g] ** 2))

        # Diffuse solar (grid cell g → patch p) — Fortran lines 68-69
        swskyd_cur = swskyd_cur.at[p, ivis].set(forc_solai[g, ivis])
        swskyd_cur = swskyd_cur.at[p, inir].set(forc_solai[g, inir])

        # Column (c) → patch (p) scalars — Fortran lines 72-80
        tref_cur = tref_cur.at[p].set(forc_t[c])
        qref_cur = qref_cur.at[p].set(forc_q[c])
        pref_cur = pref_cur.at[p].set(forc_pbot[c])
        lwsky_cur = lwsky_cur.at[p].set(forc_lwrad[c])
        qflx_rain = qflx_rain.at[p].set(forc_rain[c])
        qflx_snow = qflx_snow.at[p].set(forc_snow[c])
        swskyb_cur = swskyb_cur.at[p, ivis].set(forc_solad_col[c, ivis])
        swskyb_cur = swskyb_cur.at[p, inir].set(forc_solad_col[c, inir])

        # CO2 and O2 unit conversions — Fortran lines 83-84
        pbot_val = forc_pbot[c]
        co2ref_cur = co2ref_cur.at[p].set(forc_pco2[g] / pbot_val * 1.0e6)  # Pa -> umol/mol
        o2ref = o2ref.at[p].set(forc_po2[g] / pbot_val * 1.0e3)  # Pa -> mmol/mol

        # Acclimation temperature — Fortran line 87
        tacclim = tacclim.at[p].set(t_a10_patch[p])

        # Ground albedos: column (c) → patch (p) — Fortran lines 89-90
        albsoib = albsoib.at[p, ivis].set(albgrd[c, ivis])
        albsoib = albsoib.at[p, inir].set(albgrd[c, inir])
        albsoid = albsoid.at[p, ivis].set(albgri[c, ivis])
        albsoid = albsoid.at[p, inir].set(albgri[c, inir])

        # Soil evaporative resistance — Fortran line 93
        soilres = soilres.at[p].set(soilresis[c])

        # First snow/soil layer properties — Fortran lines 99-101
        # snl(c) is the number of snow layers (negative value in CLM);
        # snl(c)+1 gives the Fortran index of the first snow or soil layer.
        # In standalone mode snl=0, so j=1 (first soil layer).
        # All arrays use direct j indexing (1:nlevgrnd), no nlevsno offset.
        _snl_c = _snl_j[c]
        j = _snl_c + 1  # Fortran index = Python index (snl==0 -> j==1 standalone)

        soil_t = soil_t.at[p].set(t_soisno[c, j])
        soil_dz = soil_dz.at[p].set(z[c, j] - zi[c, _snl_c])
        soil_tk = soil_tk.at[p].set(thk[c, j])

    # ------------------------------------------------------------------
    # Solar zenith angle — Fortran lines 92-110
    #
    # Computed for the beginning of the CLM timestep, i.e. at
    # calday(nstep-1):  caldaym1 = get_curr_calday(offset=-dtime_clm)
    # ------------------------------------------------------------------
    if cos_zenith_device is not None:
        # DEVICE-NATIVE solar zenith (jax.jit / global path).  The caller supplies
        # cos(zenith) as a per-column jnp array aligned with ``filter`` (entry k ->
        # patch filter[k]).  This replaces the host shr_orb_cosz recompute below,
        # which reads per-step orbital globals (caldaym1/declinm1) and pre-
        # materialised numpy lat/lon via float(_latdeg_np[g]) — both of which break
        # a jax.jit trace of the forcing step (the zenith changes every step, so the
        # host recompute cannot be hoisted to trace time).  ``filter`` values are
        # concrete 1-based patch indices, so the scatter indices stay static while
        # the cos(zenith) values flow as tracers.  Same 0.01 floor as the host path.
        # Scatter cos(zenith) to the patch(es) under compute.  Under S3 the single
        # column is grid.p (possibly TRACED); else the concrete 1-based filter.
        _fidx = (jnp.asarray([grid.p], dtype=jnp.int32) if grid is not None
                 else jnp.asarray([int(filter[k]) for k in range(num_filter)],
                                  dtype=jnp.int32))
        # Clip to [0.01, 1.0]: the 0.01 floor matches the host path (avoid acos(<=0)
        # at the horizon), and the 1.0 CEIL guards against a forcing producer that
        # emits cos(zenith) slightly above 1 (float round-off / f32->f64 overshoot),
        # which would make arccos(>1) NaN.  The host shr_orb_cosz path cannot exceed
        # 1 by construction, so only the device path needs the upper clip.
        _cz = jnp.clip(
            jnp.asarray(cos_zenith_device, dtype=solar_zen.dtype), 0.01, 1.0)
        solar_zen = solar_zen.at[_fidx].set(jnp.arccos(_cz))
    else:
        caldaym1 = float(get_curr_calday(offset=-int(dtime_clm)))

        # Orbital declination for caldaym1 — Fortran line 105
        declinm1, eccf = shr_orb_decl(caldaym1, eccen, mvelpp, lambm0, obliqr)

        # Pre-materialise grc lat/lon as numpy — these are JAX arrays set via
        # TowerMetMod, so indexing them during jax.grad tracing would fail.
        _latdeg_np = np.asarray(grc.latdeg)
        _londeg_np = np.asarray(grc.londeg)

        for fp in range(1, num_filter + 1):  # Fortran: do fp = 1, num_filter
            p = int(filter[fp - 1])
            # HOST solar path (concrete p only — it reads float(grc.latdeg[g]));
            # the traced-p scan path uses cos_zenith_device instead of this branch.
            g = int(_patch_gridcell_j[p])

            # Latitude and longitude in radians — Fortran lines 107-108
            lat = float(_latdeg_np[g]) * pi / 180.0
            lon = float(_londeg_np[g]) * pi / 180.0

            # Cosine of solar zenith angle — Fortran line 109
            coszen = shr_orb_cosz(caldaym1, lat, lon, declinm1)

            # Zenith angle: floor coszen at 0.01 to avoid acos(≤0) — Fortran line 110
            solar_zen_val = jnp.arccos(jnp.maximum(jnp.asarray(0.01), jnp.asarray(coszen)))
            solar_zen = solar_zen.at[p].set(solar_zen_val)

    # ------------------------------------------------------------------
    # Return updated container
    # ------------------------------------------------------------------
    return mlcanopy_inst._replace(
        tref_cur_forcing=tref_cur,
        qref_cur_forcing=qref_cur,
        uref_cur_forcing=uref_cur,
        pref_cur_forcing=pref_cur,
        co2ref_cur_forcing=co2ref_cur,
        o2ref_forcing=o2ref,
        solar_zen_forcing=solar_zen,
        swskyb_cur_forcing=swskyb_cur,
        swskyd_cur_forcing=swskyd_cur,
        lwsky_cur_forcing=lwsky_cur,
        qflx_rain_forcing=qflx_rain,
        qflx_snow_forcing=qflx_snow,
        tacclim_forcing=tacclim,
        albsoib_soil=albsoib,
        albsoid_soil=albsoid,
        soilres_soil=soilres,
        soil_t_soil=soil_t,
        soil_dz_soil=soil_dz,
        soil_tk_soil=soil_tk,
    )


def _MLAccumulateFluxes(
    num_filter: int,
    filter: List[int],
    ncan_vals: tuple,
    flux_accumulator: jnp.ndarray,
    flux_accumulator_profile: jnp.ndarray,
    flux_accumulator_leaf: jnp.ndarray,
    mlcanopy_inst: mlcanopy_type,
    grid: "GridInfo | None" = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Accumulate ML-step fluxes into running-sum accumulators.

    Called once per ML sub-step.  The accumulators are initialised to
    zero before the loop (``jnp.zeros`` in MLCanopyFluxes), so the
    Fortran ``nstep_ml == 1`` zero-init branch is not needed here.

    Split from the original ``_MLTimeStepFluxIntegration`` so the
    accumulation body has no Python conditionals on ``nstep_ml`` —
    a prerequisite for future ``jax.lax.fori_loop`` usage once all
    physics functions are ported to XLA-traceable paths.

    Args:
        num_filter: Number of patches in filter.
        filter: Patch index filter (1-based values).
        ncan_vals: Pre-computed tuple of ``ncan`` per active patch,
            avoiding ``int(mlcanopy_inst.ncan_canopy[p])`` D→H syncs.
        flux_accumulator: JAX array ``(endp+1, nvar1d)`` — running sums
            of single-level fluxes.
        flux_accumulator_profile: JAX array ``(endp+1, nlevmlcan+2, nvar2d)``.
        flux_accumulator_leaf: JAX array
            ``(endp+1, nlevmlcan+1, nleaf+1, nvar3d)``.
        mlcanopy_inst: Canopy container (read-only in this function).

    Returns:
        Updated ``(flux_accumulator, flux_accumulator_profile,
        flux_accumulator_leaf)`` tuple.
    """
    from legoesm.land.canopy.clm_ml_backend.clm_src_main.clm_varpar import inir, ivis  # noqa: F401

    nvar1d = 23
    nvar2d = 14
    nvar3d = 12

    for fp in range(1 if grid is not None else num_filter):
        if grid is not None:
            p = grid.p            # possibly a TRACED scan-over-columns index (S3)
            _ncan = grid.ncan
        else:
            p = int(filter[fp])
            _ncan = ncan_vals[fp]  # pre-computed; avoids int(arr[p]) D→H sync

        # ------------------------------------------------------------------
        # Accumulate single-level (1-D) fluxes — Fortran lines 65-87
        # ------------------------------------------------------------------
        i = -1

        def _acc1(val):
            nonlocal i, flux_accumulator
            i += 1
            flux_accumulator = flux_accumulator.at[p, i].add(val)

        _acc1(mlcanopy_inst.ustar_canopy[p])
        _acc1(mlcanopy_inst.beta_canopy[p])
        _acc1(mlcanopy_inst.obu_canopy[p])
        _acc1(mlcanopy_inst.z0m_canopy[p])
        _acc1(mlcanopy_inst.zdisp_canopy[p])
        _acc1(mlcanopy_inst.lwup_canopy[p])
        _acc1(mlcanopy_inst.swsoi_soil[p, ivis])
        _acc1(mlcanopy_inst.swsoi_soil[p, inir])
        _acc1(mlcanopy_inst.lwsoi_soil[p])
        _acc1(mlcanopy_inst.rnsoi_soil[p])
        _acc1(mlcanopy_inst.shsoi_soil[p])
        _acc1(mlcanopy_inst.lhsoi_soil[p])
        _acc1(mlcanopy_inst.etsoi_soil[p])
        _acc1(mlcanopy_inst.gsoi_soil[p])
        _acc1(mlcanopy_inst.gac0_soil[p])
        _acc1(mlcanopy_inst.qflx_intr_canopy[p])
        _acc1(mlcanopy_inst.qflx_tflrain_canopy[p])
        _acc1(mlcanopy_inst.qflx_tflsnow_canopy[p])
        _acc1(mlcanopy_inst.swskyb_forcing[p, ivis])
        _acc1(mlcanopy_inst.swskyb_forcing[p, inir])
        _acc1(mlcanopy_inst.swskyd_forcing[p, ivis])
        _acc1(mlcanopy_inst.swskyd_forcing[p, inir])
        _acc1(mlcanopy_inst.lwsky_forcing[p])

        assert (i + 1) == nvar1d, f"_MLAccumulateFluxes: nvar1d mismatch ({i+1} != {nvar1d})"

        # ------------------------------------------------------------------
        # Accumulate profile (2-D) fluxes — Fortran lines 89-102
        # ------------------------------------------------------------------
        j = -1

        def _acc2_layers(arr_slice):
            nonlocal j, flux_accumulator_profile
            j += 1
            flux_accumulator_profile = flux_accumulator_profile.at[p, 1 : _ncan + 1, j].add(
                arr_slice
            )

        def _acc2_rad(arr_slice):
            nonlocal j, flux_accumulator_profile
            j += 1
            flux_accumulator_profile = flux_accumulator_profile.at[p, 1 : _ncan + 2, j].add(
                arr_slice
            )

        _acc2_layers(mlcanopy_inst.shair_profile[p, 1 : _ncan + 1])
        _acc2_layers(mlcanopy_inst.etair_profile[p, 1 : _ncan + 1])
        _acc2_layers(mlcanopy_inst.stair_profile[p, 1 : _ncan + 1])
        _acc2_layers(mlcanopy_inst.mflx_profile[p, 1 : _ncan + 1])
        _acc2_layers(mlcanopy_inst.kc_eddy_profile[p, 1 : _ncan + 1])
        _acc2_layers(mlcanopy_inst.gac_profile[p, 1 : _ncan + 1])
        _acc2_rad(mlcanopy_inst.swupw_profile[p, 0 : _ncan + 1, ivis])
        _acc2_rad(mlcanopy_inst.swupw_profile[p, 0 : _ncan + 1, inir])
        _acc2_rad(mlcanopy_inst.swdwn_profile[p, 0 : _ncan + 1, ivis])
        _acc2_rad(mlcanopy_inst.swdwn_profile[p, 0 : _ncan + 1, inir])
        _acc2_rad(mlcanopy_inst.swbeam_profile[p, 0 : _ncan + 1, ivis])
        _acc2_rad(mlcanopy_inst.swbeam_profile[p, 0 : _ncan + 1, inir])
        _acc2_rad(mlcanopy_inst.lwupw_profile[p, 0 : _ncan + 1])
        _acc2_rad(mlcanopy_inst.lwdwn_profile[p, 0 : _ncan + 1])

        assert (j + 1) == nvar2d, f"_MLAccumulateFluxes: nvar2d mismatch ({j+1} != {nvar2d})"

        # ------------------------------------------------------------------
        # Accumulate leaf (3-D) fluxes — Fortran lines 104-117
        # ------------------------------------------------------------------
        k = -1

        def _acc3(arr_slice):
            nonlocal k, flux_accumulator_leaf
            k += 1
            flux_accumulator_leaf = flux_accumulator_leaf.at[p, :, :, k].add(arr_slice)

        _acc3(mlcanopy_inst.swleaf_leaf[p, :, :, ivis])
        _acc3(mlcanopy_inst.swleaf_leaf[p, :, :, inir])
        _acc3(mlcanopy_inst.lwleaf_leaf[p, :, :])
        _acc3(mlcanopy_inst.rnleaf_leaf[p, :, :])
        _acc3(mlcanopy_inst.shleaf_leaf[p, :, :])
        _acc3(mlcanopy_inst.lhleaf_leaf[p, :, :])
        _acc3(mlcanopy_inst.trleaf_leaf[p, :, :])
        _acc3(mlcanopy_inst.evleaf_leaf[p, :, :])
        _acc3(mlcanopy_inst.stleaf_leaf[p, :, :])
        _acc3(mlcanopy_inst.anet_leaf[p, :, :])
        _acc3(mlcanopy_inst.agross_leaf[p, :, :])
        _acc3(mlcanopy_inst.gs_leaf[p, :, :])

        assert (k + 1) == nvar3d, f"_MLAccumulateFluxes: nvar3d mismatch ({k+1} != {nvar3d})"

    return flux_accumulator, flux_accumulator_profile, flux_accumulator_leaf


def _MLScaleAndWriteBack(
    num_ml_steps: int,
    num_filter: int,
    filter: List[int],
    ncan_vals: tuple,
    flux_accumulator: jnp.ndarray,
    flux_accumulator_profile: jnp.ndarray,
    flux_accumulator_leaf: jnp.ndarray,
    mlcanopy_inst: mlcanopy_type,
    grid: "GridInfo | None" = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, mlcanopy_type]:
    """
    Scale accumulated flux sums by ``1/num_ml_steps`` and write the
    time-averaged values back into ``mlcanopy_inst``.

    Called once after the ML sub-step loop completes.  Split from
    ``_MLTimeStepFluxIntegration`` so the per-step accumulation body
    (``_MLAccumulateFluxes``) has no loop-index conditionals.

    Args:
        num_ml_steps: Total number of ML sub-steps (divisor for average).
        num_filter: Number of patches in filter.
        filter: Patch index filter (1-based values).
        ncan_vals: Pre-computed tuple of ``ncan`` per active patch.
        flux_accumulator: Running sums (single-level).
        flux_accumulator_profile: Running sums (profile).
        flux_accumulator_leaf: Running sums (leaf).
        mlcanopy_inst: Canopy container to update with averaged fluxes.

    Returns:
        ``(flux_accumulator, flux_accumulator_profile,
        flux_accumulator_leaf, mlcanopy_inst)`` with scaled accumulators
        and updated canopy container.
    """
    from legoesm.land.canopy.clm_ml_backend.clm_src_main.abortutils import endrun  # noqa: F401
    from legoesm.land.canopy.clm_ml_backend.clm_src_main.clm_varpar import inir, ivis  # noqa: F401

    nvar1d = 23
    nvar2d = 14
    nvar3d = 12

    scale = 1.0 / float(num_ml_steps)

    for fp in range(1 if grid is not None else num_filter):
        p = grid.p if grid is not None else int(filter[fp])  # grid.p may be TRACED (S3)
        flux_accumulator = flux_accumulator.at[p, :].mul(scale)
        flux_accumulator_profile = flux_accumulator_profile.at[p, :, :].mul(scale)
        flux_accumulator_leaf = flux_accumulator_leaf.at[p, :, :, :].mul(scale)

    # Read all arrays we need to update
    ustar = mlcanopy_inst.ustar_canopy
    beta = mlcanopy_inst.beta_canopy
    obu = mlcanopy_inst.obu_canopy
    z0m = mlcanopy_inst.z0m_canopy
    zdisp = mlcanopy_inst.zdisp_canopy
    lwup = mlcanopy_inst.lwup_canopy
    swsoi = mlcanopy_inst.swsoi_soil
    lwsoi = mlcanopy_inst.lwsoi_soil
    rnsoi = mlcanopy_inst.rnsoi_soil
    shsoi = mlcanopy_inst.shsoi_soil
    lhsoi = mlcanopy_inst.lhsoi_soil
    etsoi = mlcanopy_inst.etsoi_soil
    gsoi = mlcanopy_inst.gsoi_soil
    gac0 = mlcanopy_inst.gac0_soil
    qflx_intr = mlcanopy_inst.qflx_intr_canopy
    qflx_tflrain = mlcanopy_inst.qflx_tflrain_canopy
    qflx_tflsnow = mlcanopy_inst.qflx_tflsnow_canopy
    swskyb = mlcanopy_inst.swskyb_forcing
    swskyd = mlcanopy_inst.swskyd_forcing
    lwsky = mlcanopy_inst.lwsky_forcing
    shair = mlcanopy_inst.shair_profile
    etair = mlcanopy_inst.etair_profile
    stair = mlcanopy_inst.stair_profile
    mflx = mlcanopy_inst.mflx_profile
    kc_eddy = mlcanopy_inst.kc_eddy_profile
    gac = mlcanopy_inst.gac_profile
    swupw = mlcanopy_inst.swupw_profile
    swdwn = mlcanopy_inst.swdwn_profile
    swbeam = mlcanopy_inst.swbeam_profile
    lwupw = mlcanopy_inst.lwupw_profile
    lwdwn = mlcanopy_inst.lwdwn_profile
    swleaf = mlcanopy_inst.swleaf_leaf
    lwleaf = mlcanopy_inst.lwleaf_leaf
    rnleaf = mlcanopy_inst.rnleaf_leaf
    shleaf = mlcanopy_inst.shleaf_leaf
    lhleaf = mlcanopy_inst.lhleaf_leaf
    trleaf = mlcanopy_inst.trleaf_leaf
    evleaf = mlcanopy_inst.evleaf_leaf
    stleaf = mlcanopy_inst.stleaf_leaf
    anet = mlcanopy_inst.anet_leaf
    agross = mlcanopy_inst.agross_leaf
    gs = mlcanopy_inst.gs_leaf

    for fp in range(1 if grid is not None else num_filter):
        if grid is not None:
            p = grid.p            # possibly a TRACED scan-over-columns index (S3)
            _ncan = grid.ncan
        else:
            p = int(filter[fp])
            _ncan = ncan_vals[fp]  # pre-computed; avoids int(arr[p]) D→H sync

        i = -1

        def _wb1():
            nonlocal i
            i += 1
            return flux_accumulator[p, i]

        ustar = ustar.at[p].set(_wb1())
        beta = beta.at[p].set(_wb1())
        obu = obu.at[p].set(_wb1())
        z0m = z0m.at[p].set(_wb1())
        zdisp = zdisp.at[p].set(_wb1())
        lwup = lwup.at[p].set(_wb1())
        swsoi = swsoi.at[p, ivis].set(_wb1())
        swsoi = swsoi.at[p, inir].set(_wb1())
        lwsoi = lwsoi.at[p].set(_wb1())
        rnsoi = rnsoi.at[p].set(_wb1())
        shsoi = shsoi.at[p].set(_wb1())
        lhsoi = lhsoi.at[p].set(_wb1())
        etsoi = etsoi.at[p].set(_wb1())
        gsoi = gsoi.at[p].set(_wb1())
        gac0 = gac0.at[p].set(_wb1())
        qflx_intr = qflx_intr.at[p].set(_wb1())
        qflx_tflrain = qflx_tflrain.at[p].set(_wb1())
        qflx_tflsnow = qflx_tflsnow.at[p].set(_wb1())
        swskyb = swskyb.at[p, ivis].set(_wb1())
        swskyb = swskyb.at[p, inir].set(_wb1())
        swskyd = swskyd.at[p, ivis].set(_wb1())
        swskyd = swskyd.at[p, inir].set(_wb1())
        lwsky = lwsky.at[p].set(_wb1())

        assert (i + 1) == nvar1d, "_MLScaleAndWriteBack: nvar1d mismatch"

        j = -1

        def _wb2_layers():
            nonlocal j
            j += 1
            return flux_accumulator_profile[p, 1 : _ncan + 1, j]

        def _wb2_rad():
            nonlocal j
            j += 1
            return flux_accumulator_profile[p, 1 : _ncan + 2, j]

        shair = shair.at[p, 1 : _ncan + 1].set(_wb2_layers())
        etair = etair.at[p, 1 : _ncan + 1].set(_wb2_layers())
        stair = stair.at[p, 1 : _ncan + 1].set(_wb2_layers())
        mflx = mflx.at[p, 1 : _ncan + 1].set(_wb2_layers())
        kc_eddy = kc_eddy.at[p, 1 : _ncan + 1].set(_wb2_layers())
        gac = gac.at[p, 1 : _ncan + 1].set(_wb2_layers())
        swupw = swupw.at[p, 0 : _ncan + 1, ivis].set(_wb2_rad())
        swupw = swupw.at[p, 0 : _ncan + 1, inir].set(_wb2_rad())
        swdwn = swdwn.at[p, 0 : _ncan + 1, ivis].set(_wb2_rad())
        swdwn = swdwn.at[p, 0 : _ncan + 1, inir].set(_wb2_rad())
        swbeam = swbeam.at[p, 0 : _ncan + 1, ivis].set(_wb2_rad())
        swbeam = swbeam.at[p, 0 : _ncan + 1, inir].set(_wb2_rad())
        lwupw = lwupw.at[p, 0 : _ncan + 1].set(_wb2_rad())
        lwdwn = lwdwn.at[p, 0 : _ncan + 1].set(_wb2_rad())

        assert (j + 1) == nvar2d, "_MLScaleAndWriteBack: nvar2d mismatch"

        k = -1

        def _wb3():
            nonlocal k
            k += 1
            return flux_accumulator_leaf[p, :, :, k]

        swleaf = swleaf.at[p, :, :, ivis].set(_wb3())
        swleaf = swleaf.at[p, :, :, inir].set(_wb3())
        lwleaf = lwleaf.at[p, :, :].set(_wb3())
        rnleaf = rnleaf.at[p, :, :].set(_wb3())
        shleaf = shleaf.at[p, :, :].set(_wb3())
        lhleaf = lhleaf.at[p, :, :].set(_wb3())
        trleaf = trleaf.at[p, :, :].set(_wb3())
        evleaf = evleaf.at[p, :, :].set(_wb3())
        stleaf = stleaf.at[p, :, :].set(_wb3())
        anet = anet.at[p, :, :].set(_wb3())
        agross = agross.at[p, :, :].set(_wb3())
        gs = gs.at[p, :, :].set(_wb3())

        assert (k + 1) == nvar3d, "_MLScaleAndWriteBack: nvar3d mismatch"

    mlcanopy_inst = mlcanopy_inst._replace(
        ustar_canopy=ustar,
        beta_canopy=beta,
        obu_canopy=obu,
        z0m_canopy=z0m,
        zdisp_canopy=zdisp,
        lwup_canopy=lwup,
        swsoi_soil=swsoi,
        lwsoi_soil=lwsoi,
        rnsoi_soil=rnsoi,
        shsoi_soil=shsoi,
        lhsoi_soil=lhsoi,
        etsoi_soil=etsoi,
        gsoi_soil=gsoi,
        gac0_soil=gac0,
        qflx_intr_canopy=qflx_intr,
        qflx_tflrain_canopy=qflx_tflrain,
        qflx_tflsnow_canopy=qflx_tflsnow,
        swskyb_forcing=swskyb,
        swskyd_forcing=swskyd,
        lwsky_forcing=lwsky,
        shair_profile=shair,
        etair_profile=etair,
        stair_profile=stair,
        mflx_profile=mflx,
        kc_eddy_profile=kc_eddy,
        gac_profile=gac,
        swupw_profile=swupw,
        swdwn_profile=swdwn,
        swbeam_profile=swbeam,
        lwupw_profile=lwupw,
        lwdwn_profile=lwdwn,
        swleaf_leaf=swleaf,
        lwleaf_leaf=lwleaf,
        rnleaf_leaf=rnleaf,
        shleaf_leaf=shleaf,
        lhleaf_leaf=lhleaf,
        trleaf_leaf=trleaf,
        evleaf_leaf=evleaf,
        stleaf_leaf=stleaf,
        anet_leaf=anet,
        agross_leaf=agross,
        gs_leaf=gs,
    )

    return flux_accumulator, flux_accumulator_profile, flux_accumulator_leaf, mlcanopy_inst


# _MLTimeStepFluxIntegration has been replaced by the two-function split:
#   _MLAccumulateFluxes  — per-step accumulation (no conditionals on nstep_ml)
#   _MLScaleAndWriteBack — post-loop scale + write-back (called once)
# This separation is a prerequisite for future lax.fori_loop usage once all
# physics functions are ported to XLA-traceable paths.


def _CanopyFluxesDiagnostics(
    num_filter: int,
    filter: List[int],
    mlcanopy_inst: mlcanopy_type,
    grid: "GridInfo | None" = None,
) -> mlcanopy_type:
    """
    Compute canopy-integrated fluxes, diagnostics, and energy balance checks.

    Mirrors Fortran subroutine ``CanopyFluxesDiagnostics`` (lines 1-270).

    Differentiable mode
    -------------------
    When ``grid`` is not ``None`` the routine runs on the ``jax.grad`` tape:
    structural ints (``ncan``/``ntop``) are read from ``grid`` (concrete Python
    ints extracted before tracing) instead of ``int(mlcanopy_inst.*)`` (which
    would raise ``ConcretizationTypeError`` on a tracer), and the host-syncing
    ``abs(err)`` energy-balance ``endrun`` checks are skipped (they force a
    device→host read of a traced scalar). The flux arithmetic is identical in
    both modes, so the populated output fields match to floating-point.

    **Algorithm** (one pass per patch ``p``):

    1. **Layer-mean leaf fluxes** (Fortran lines 80-130):
       For each layer ``ic`` with ``dpai > 0`` compute sun/shade
       weighted means::

           X_mean(ic) = X(ic,isun)*fracsun(ic) + X(ic,isha)*(1-fracsun(ic))

       and convert to per-ground-area source terms::

           Xsrc(ic) = X_mean(ic) * dpai(ic)

       Layers with ``dpai == 0`` are zeroed.

    2. **Vegetation totals** (Fortran lines 132-158):
       Sum source terms over all layers to produce ``swveg``, ``lwveg``,
       ``shveg``, ``lhveg``, ``etveg``, ``trveg``, ``evveg``,
       ``gppveg``, ``vcmax25veg``, ``gsveg``, ``stflx_veg``.

    3. **Vegetation energy balance check** (Fortran lines 160-163):
       ``|swveg(vis) + swveg(nir) + lwveg - shveg - lhveg - stflx_veg| < 1e-3 W/m²``

    4. **Albedo** (Fortran lines 165-171):
       ``albcan(ib) = swupw(ntop, ib) / (swskyb(ib) + swskyd(ib))``
       (0 when incoming == 0).

    5. **Turbulent fluxes** (Fortran lines 173-182):
       ``flux_profile_type == 0 or -1``: ``shflx = shveg + shsoi``,
       ``etflx = etveg + etsoi``, ``lhflx = lhveg + lhsoi``.
       ``flux_profile_type == 1``: take top-layer air flux directly.

    6. **Air storage flux** (Fortran lines 184-187):
       ``stflx_air = sum(stair(1:ncan))``.

    7. **Overall energy balance checks** (Fortran lines 189-211):
       Three independent checks with tolerances 0.001, 0.001, and
       0.01 W/m².

    8. **Sun/shade canopy totals** (Fortran lines 213-250):
       Accumulate ``laisun``, ``laisha``, per-population fluxes
       ``swvegsun/sha``, ``lwvegsun/sha``, ``shvegsun/sha``,
       ``lhvegsun/sha``, ``etvegsun/sha``, ``gppvegsun/sha``,
       ``vcmax25sun/sha``, ``gsvegsun/sha``.

    9. **PAI-weighted temperatures and wind** (Fortran lines 252-268):
       ``windveg``, ``windvegsun``, ``windvegsha``, ``tlveg``,
       ``tlvegsun``, ``tlvegsha``, ``taveg``, ``tavegsun``,
       ``tavegsha``.

    10. **Water-stress fraction** (Fortran lines 270-278):
        ``fracminlwp = sum(dpai where lwp_mean <= -2 MPa) / (lai+sai)``

    Args:
        num_filter: Number of patches in filter.
        filter: Patch index filter (1-based values).
        mlcanopy_inst: Canopy container; all fields read; output fields
            written and returned.

    Returns:
        Updated :class:`mlcanopy_type`.
    """
    from legoesm.land.canopy.clm_ml_backend.clm_src_main.abortutils import endrun  # noqa: F401
    from legoesm.land.canopy.clm_ml_backend.clm_src_main.clm_varpar import inir, ivis, numrad  # noqa: F401
    from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLclm_varctl import flux_profile_type  # noqa: F401
    from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLclm_varpar import isha, isun  # noqa: F401

    _diff_mode = grid is not None

    # ------------------------------------------------------------------
    # Extract mutable arrays (all output fields)
    # ------------------------------------------------------------------
    rnet = mlcanopy_inst.rnet_canopy
    stflx_air = mlcanopy_inst.stflx_air_canopy
    stflx_veg = mlcanopy_inst.stflx_veg_canopy
    shflx = mlcanopy_inst.shflx_canopy
    lhflx = mlcanopy_inst.lhflx_canopy
    etflx = mlcanopy_inst.etflx_canopy
    albcan = mlcanopy_inst.albcan_canopy
    swveg = mlcanopy_inst.swveg_canopy
    swvegsun = mlcanopy_inst.swvegsun_canopy
    swvegsha = mlcanopy_inst.swvegsha_canopy
    lwveg = mlcanopy_inst.lwveg_canopy
    lwvegsun = mlcanopy_inst.lwvegsun_canopy
    lwvegsha = mlcanopy_inst.lwvegsha_canopy
    shveg = mlcanopy_inst.shveg_canopy
    shvegsun = mlcanopy_inst.shvegsun_canopy
    shvegsha = mlcanopy_inst.shvegsha_canopy
    lhveg = mlcanopy_inst.lhveg_canopy
    lhvegsun = mlcanopy_inst.lhvegsun_canopy
    lhvegsha = mlcanopy_inst.lhvegsha_canopy
    etveg = mlcanopy_inst.etveg_canopy
    etvegsun = mlcanopy_inst.etvegsun_canopy
    etvegsha = mlcanopy_inst.etvegsha_canopy
    trveg = mlcanopy_inst.trveg_canopy
    evveg = mlcanopy_inst.evveg_canopy
    gppveg = mlcanopy_inst.gppveg_canopy
    gppvegsun = mlcanopy_inst.gppvegsun_canopy
    gppvegsha = mlcanopy_inst.gppvegsha_canopy
    vcmax25veg = mlcanopy_inst.vcmax25veg_canopy
    vcmax25sun = mlcanopy_inst.vcmax25sun_canopy
    vcmax25sha = mlcanopy_inst.vcmax25sha_canopy
    gsveg = mlcanopy_inst.gsveg_canopy
    gsvegsun = mlcanopy_inst.gsvegsun_canopy
    gsvegsha = mlcanopy_inst.gsvegsha_canopy
    windveg = mlcanopy_inst.windveg_canopy
    windvegsun = mlcanopy_inst.windvegsun_canopy
    windvegsha = mlcanopy_inst.windvegsha_canopy
    tlveg = mlcanopy_inst.tlveg_canopy
    tlvegsun = mlcanopy_inst.tlvegsun_canopy
    tlvegsha = mlcanopy_inst.tlvegsha_canopy
    taveg = mlcanopy_inst.taveg_canopy
    tavegsun = mlcanopy_inst.tavegsun_canopy
    tavegsha = mlcanopy_inst.tavegsha_canopy
    laisun = mlcanopy_inst.laisun_canopy
    laisha = mlcanopy_inst.laisha_canopy
    fracminlwp = mlcanopy_inst.fracminlwp_canopy
    swsrc = mlcanopy_inst.swsrc_profile
    lwsrc = mlcanopy_inst.lwsrc_profile
    rnsrc = mlcanopy_inst.rnsrc_profile
    stsrc = mlcanopy_inst.stsrc_profile
    shsrc = mlcanopy_inst.shsrc_profile
    lhsrc = mlcanopy_inst.lhsrc_profile
    etsrc = mlcanopy_inst.etsrc_profile
    trsrc = mlcanopy_inst.trsrc_profile
    evsrc = mlcanopy_inst.evsrc_profile
    fco2src = mlcanopy_inst.fco2src_profile
    swleaf_mean = mlcanopy_inst.swleaf_mean_profile
    lwleaf_mean = mlcanopy_inst.lwleaf_mean_profile
    rnleaf_mean = mlcanopy_inst.rnleaf_mean_profile
    stleaf_mean = mlcanopy_inst.stleaf_mean_profile
    shleaf_mean = mlcanopy_inst.shleaf_mean_profile
    lhleaf_mean = mlcanopy_inst.lhleaf_mean_profile
    etleaf_mean = mlcanopy_inst.etleaf_mean_profile
    trleaf_mean = mlcanopy_inst.trleaf_mean_profile
    evleaf_mean = mlcanopy_inst.evleaf_mean_profile
    fco2_mean = mlcanopy_inst.fco2_mean_profile
    apar_mean = mlcanopy_inst.apar_mean_profile
    gs_mean = mlcanopy_inst.gs_mean_profile
    tleaf_mean = mlcanopy_inst.tleaf_mean_profile
    lwp_mean = mlcanopy_inst.lwp_mean_profile

    # ==================================================================
    # Main patch loop — Fortran: do fp = 1, num_filter
    # ==================================================================
    for fp in range(1, 2 if _diff_mode else num_filter + 1):
        p = grid.p if _diff_mode else int(filter[fp - 1])  # grid.p may be TRACED (S3)
        # Structural ints: from grid (concrete, pre-traced) in diff mode; from a
        # host read otherwise. Single-site diff mode uses one shared (ncan, ntop).
        _ncan = grid.ncan if _diff_mode else int(mlcanopy_inst.ncan_canopy[p])
        _ntop = grid.ntop if _diff_mode else int(mlcanopy_inst.ntop_canopy[p])
        _sl = slice(1, _ncan + 1)

        # ----------------------------------------------------------------
        # Per-layer inputs — kept as JAX arrays (no D→H sync until needed
        # by Python-level energy balance checks below).
        # ----------------------------------------------------------------
        _dpai = mlcanopy_inst.dpai_profile[p]
        _fracsun = mlcanopy_inst.fracsun_profile[p]
        _fwet = mlcanopy_inst.fwet_profile[p]
        _fdry = mlcanopy_inst.fdry_profile[p]
        _wind = mlcanopy_inst.wind_profile[p]
        _tair = mlcanopy_inst.tair_profile[p]
        _stair = mlcanopy_inst.stair_profile[p]
        _vcmax25_prof = mlcanopy_inst.vcmax25_profile[p]
        _lwleaf_sun = mlcanopy_inst.lwleaf_leaf[p, :, isun]
        _lwleaf_sha = mlcanopy_inst.lwleaf_leaf[p, :, isha]
        _swleaf_sun_vis = mlcanopy_inst.swleaf_leaf[p, :, isun, ivis]
        _swleaf_sun_nir = mlcanopy_inst.swleaf_leaf[p, :, isun, inir]
        _swleaf_sha_vis = mlcanopy_inst.swleaf_leaf[p, :, isha, ivis]
        _swleaf_sha_nir = mlcanopy_inst.swleaf_leaf[p, :, isha, inir]
        _rnleaf_sun = mlcanopy_inst.rnleaf_leaf[p, :, isun]
        _rnleaf_sha = mlcanopy_inst.rnleaf_leaf[p, :, isha]
        _stleaf_sun = mlcanopy_inst.stleaf_leaf[p, :, isun]
        _stleaf_sha = mlcanopy_inst.stleaf_leaf[p, :, isha]
        _shleaf_sun = mlcanopy_inst.shleaf_leaf[p, :, isun]
        _shleaf_sha = mlcanopy_inst.shleaf_leaf[p, :, isha]
        _lhleaf_sun = mlcanopy_inst.lhleaf_leaf[p, :, isun]
        _lhleaf_sha = mlcanopy_inst.lhleaf_leaf[p, :, isha]
        _evleaf_sun = mlcanopy_inst.evleaf_leaf[p, :, isun]
        _evleaf_sha = mlcanopy_inst.evleaf_leaf[p, :, isha]
        _trleaf_sun = mlcanopy_inst.trleaf_leaf[p, :, isun]
        _trleaf_sha = mlcanopy_inst.trleaf_leaf[p, :, isha]
        _anet_sun = mlcanopy_inst.anet_leaf[p, :, isun]
        _anet_sha = mlcanopy_inst.anet_leaf[p, :, isha]
        _agross_sun = mlcanopy_inst.agross_leaf[p, :, isun]
        _agross_sha = mlcanopy_inst.agross_leaf[p, :, isha]
        _apar_sun = mlcanopy_inst.apar_leaf[p, :, isun]
        _apar_sha = mlcanopy_inst.apar_leaf[p, :, isha]
        _gs_sun = mlcanopy_inst.gs_leaf[p, :, isun]
        _gs_sha = mlcanopy_inst.gs_leaf[p, :, isha]
        _tleaf_sun = mlcanopy_inst.tleaf_leaf[p, :, isun]
        _tleaf_sha = mlcanopy_inst.tleaf_leaf[p, :, isha]
        _lwp_sun = mlcanopy_inst.lwp_leaf[p, :, isun]
        _lwp_sha = mlcanopy_inst.lwp_leaf[p, :, isha]
        _vcmax25_sun = mlcanopy_inst.vcmax25_leaf[p, :, isun]
        _vcmax25_sha = mlcanopy_inst.vcmax25_leaf[p, :, isha]
        _swskyb = mlcanopy_inst.swskyb_forcing[p]
        _swskyd = mlcanopy_inst.swskyd_forcing[p]
        _swbeam_ntop = mlcanopy_inst.swbeam_profile[p, _ntop]
        _swdwn_ntop = mlcanopy_inst.swdwn_profile[p, _ntop]
        _swupw_ntop = mlcanopy_inst.swupw_profile[p, _ntop]
        _lwdwn_ntop = mlcanopy_inst.lwdwn_profile[p, _ntop]
        _lwupw_ntop = mlcanopy_inst.lwupw_profile[p, _ntop]
        _swsoi_vis = mlcanopy_inst.swsoi_soil[p, ivis]
        _swsoi_nir = mlcanopy_inst.swsoi_soil[p, inir]
        _lwsoi_p = mlcanopy_inst.lwsoi_soil[p]
        _lwsky_p = mlcanopy_inst.lwsky_forcing[p]
        _lwup_p = mlcanopy_inst.lwup_canopy[p]
        _gsoi_p = mlcanopy_inst.gsoi_soil[p]
        lai_p = mlcanopy_inst.lai_canopy[p]
        sai_p = mlcanopy_inst.sai_canopy[p]

        # ----------------------------------------------------------------
        # Vectorized computation over ic=1.._ncan  (pure JAX — no D→H)
        # ----------------------------------------------------------------
        ics = np.arange(1, _ncan + 1)  # concrete int array for indexing
        dpai_v = _dpai[ics]
        fs_v = _fracsun[ics]
        fsh_v = 1.0 - fs_v
        fwet_v = _fwet[ics]
        fdry_v = _fdry[ics]
        has_pai = dpai_v > 0.0
        dpai_s = jnp.where(has_pai, dpai_v, 0.0)

        # fracgreen = fdry / (1 - fwet), guard against fwet == 1
        fwet_safe = jnp.where(fwet_v < 1.0, fwet_v, 0.0)
        fracgreen_v = jnp.where(has_pai, fdry_v / (1.0 - fwet_safe), 0.0)

        # --- 1. Layer-mean leaf fluxes (per leaf area) ---
        def _blend(a_sun, a_sha):
            return jnp.where(has_pai, a_sun[ics] * fs_v + a_sha[ics] * fsh_v, 0.0)

        lw_m = _blend(_lwleaf_sun, _lwleaf_sha)
        sw_m_vis = _blend(_swleaf_sun_vis, _swleaf_sha_vis)
        sw_m_nir = _blend(_swleaf_sun_nir, _swleaf_sha_nir)
        rn_m = _blend(_rnleaf_sun, _rnleaf_sha)
        st_m = _blend(_stleaf_sun, _stleaf_sha)
        sh_m = _blend(_shleaf_sun, _shleaf_sha)
        lh_m = _blend(_lhleaf_sun, _lhleaf_sha)
        ev_m = _blend(_evleaf_sun, _evleaf_sha)
        tr_m = _blend(_trleaf_sun, _trleaf_sha)
        et_m = ev_m + tr_m
        fco2_m = _blend(_anet_sun, _anet_sha)
        apar_m = _blend(_apar_sun, _apar_sha)
        gs_m = _blend(_gs_sun, _gs_sha)
        tl_m = _blend(_tleaf_sun, _tleaf_sha)
        lp_m = _blend(_lwp_sun, _lwp_sha)

        # --- 1. Source terms (per ground area) ---
        lw_src = lw_m * dpai_s
        sw_src_vis = sw_m_vis * dpai_s
        sw_src_nir = sw_m_nir * dpai_s
        rn_src = rn_m * dpai_s
        st_src = st_m * dpai_s
        sh_src = sh_m * dpai_s
        lh_src = lh_m * dpai_s
        et_src = et_m * dpai_s
        tr_src = tr_m * dpai_s
        ev_src = ev_m * dpai_s
        fco2_src = fco2_m * dpai_s * fracgreen_v

        # --- 2. Vegetation totals ---
        swveg_vis = jnp.sum(sw_src_vis)
        swveg_nir = jnp.sum(sw_src_nir)
        lwveg_p = jnp.sum(lw_src)
        stflx_veg_p = jnp.sum(st_src)
        shveg_p = jnp.sum(sh_src)
        lhveg_p = jnp.sum(lh_src)
        etveg_p = jnp.sum(et_src)
        trveg_p = jnp.sum(tr_src)
        evveg_p = jnp.sum(ev_src)
        vcmax25veg_p = jnp.sum(_vcmax25_prof[ics] * dpai_s)
        gppveg_p = jnp.sum(
            (_agross_sun[ics] * fs_v + _agross_sha[ics] * fsh_v) * dpai_s * fracgreen_v
        )
        gsveg_p = jnp.sum(gs_m * dpai_s)

        swveg = swveg.at[p, ivis].set(swveg_vis)
        swveg = swveg.at[p, inir].set(swveg_nir)
        lwveg = lwveg.at[p].set(lwveg_p)
        stflx_veg = stflx_veg.at[p].set(stflx_veg_p)
        shveg = shveg.at[p].set(shveg_p)
        lhveg = lhveg.at[p].set(lhveg_p)
        etveg = etveg.at[p].set(etveg_p)
        trveg = trveg.at[p].set(trveg_p)
        evveg = evveg.at[p].set(evveg_p)
        gppveg = gppveg.at[p].set(gppveg_p)
        vcmax25veg = vcmax25veg.at[p].set(vcmax25veg_p)
        gsveg = gsveg.at[p].set(gsveg_p)

        # --- 3. Vegetation energy balance check ---
        # (host-syncing abs() on a traced scalar — skip on the grad tape)
        if not _diff_mode:
            err = swveg_vis + swveg_nir + lwveg_p - shveg_p - lhveg_p - stflx_veg_p
            if abs(err) >= 1.0e-3:
                endrun(msg=" ERROR: CanopyFluxesDiagnostics: energy conservation error (1)")

        # --- 4. Albedo ---
        _inc_vis = _swskyb[ivis] + _swskyd[ivis]
        _inc_nir = _swskyb[inir] + _swskyd[inir]
        _alb_vis = jnp.where(_inc_vis > 0.0, _swupw_ntop[ivis] / jnp.maximum(_inc_vis, 1e-30), 0.0)
        _alb_nir = jnp.where(_inc_nir > 0.0, _swupw_ntop[inir] / jnp.maximum(_inc_nir, 1e-30), 0.0)
        albcan = albcan.at[p, ivis].set(_alb_vis)
        albcan = albcan.at[p, inir].set(_alb_nir)

        # --- 5. Turbulent fluxes ---
        if flux_profile_type in (0, -1):
            shflx_val = shveg_p + mlcanopy_inst.shsoi_soil[p]
            etflx_val = etveg_p + mlcanopy_inst.etsoi_soil[p]
            lhflx_val = lhveg_p + mlcanopy_inst.lhsoi_soil[p]
        elif flux_profile_type == 1:
            shflx_val = mlcanopy_inst.shair_profile[p, _ncan]
            etflx_val = mlcanopy_inst.etair_profile[p, _ncan]
            # LatVap accepts a traced jnp scalar (jnp.where internally); the
            # former float() cast broke the tape and is unnecessary in both modes.
            lhflx_val = etflx_val * LatVap(mlcanopy_inst.tref_forcing[p])
        else:
            endrun(msg=" ERROR: CanopyFluxesDiagnostics: turbulence type not valid")
            shflx_val = etflx_val = lhflx_val = 0.0  # Unreachable
        shflx = shflx.at[p].set(shflx_val)
        etflx = etflx.at[p].set(etflx_val)
        lhflx = lhflx.at[p].set(lhflx_val)

        # --- 6. Air heat storage ---
        stflx_air_p = jnp.sum(_stair[ics])
        stflx_air = stflx_air.at[p].set(stflx_air_p)

        # --- 7. Net radiation (output) + energy balance checks (diagnostic) ---
        rnet_p = swveg_vis + swveg_nir + _swsoi_vis + _swsoi_nir + lwveg_p + _lwsoi_p
        rnet = rnet.at[p].set(rnet_p)

        # The three closure checks below only read state to form host-syncing
        # abs(err) comparisons; they produce no output field, so they are skipped
        # on the grad tape (concretising a traced scalar would raise).
        if not _diff_mode:
            radin = _swskyb[ivis] + _swskyd[ivis] + _swskyb[inir] + _swskyd[inir] + _lwsky_p
            radout = (
                _alb_vis * (_swskyb[ivis] + _swskyd[ivis])
                + _alb_nir * (_swskyb[inir] + _swskyd[inir])
                + _lwup_p
            )

            err = rnet_p - (radin - radout)
            if abs(err) > 0.001:
                endrun(msg=" ERROR: CanopyFluxesDiagnostics: energy conservation error (2)")

            avail = radin - radout - _gsoi_p
            flux_p = shflx_val + lhflx_val + stflx_air_p + stflx_veg_p
            err = avail - flux_p
            if abs(err) > 0.01:
                endrun(msg=" ERROR: CanopyFluxesDiagnostics: energy conservation error (3)")

            radin_top = (
                _swbeam_ntop[ivis]
                + _swbeam_ntop[inir]
                + _swdwn_ntop[ivis]
                + _swdwn_ntop[inir]
                + _lwdwn_ntop
            )
            radout_top = _swupw_ntop[ivis] + _swupw_ntop[inir] + _lwupw_ntop
            err = (radin_top - radout_top) - rnet_p
            if abs(err) > 0.001:
                endrun(msg=" ERROR: CanopyFluxesDiagnostics: energy conservation error (4)")

        # --- 8. Sun/shade canopy totals ---
        laisun_p = jnp.sum(fs_v * dpai_s)
        laisha_p = jnp.sum(fsh_v * dpai_s)
        swvegsun_vis = jnp.sum(_swleaf_sun_vis[ics] * fs_v * dpai_s)
        swvegsun_nir = jnp.sum(_swleaf_sun_nir[ics] * fs_v * dpai_s)
        swvegsha_vis = jnp.sum(_swleaf_sha_vis[ics] * fsh_v * dpai_s)
        swvegsha_nir = jnp.sum(_swleaf_sha_nir[ics] * fsh_v * dpai_s)
        lwvegsun_p = jnp.sum(_lwleaf_sun[ics] * fs_v * dpai_s)
        lwvegsha_p = jnp.sum(_lwleaf_sha[ics] * fsh_v * dpai_s)
        shvegsun_p = jnp.sum(_shleaf_sun[ics] * fs_v * dpai_s)
        shvegsha_p = jnp.sum(_shleaf_sha[ics] * fsh_v * dpai_s)
        lhvegsun_p = jnp.sum(_lhleaf_sun[ics] * fs_v * dpai_s)
        lhvegsha_p = jnp.sum(_lhleaf_sha[ics] * fsh_v * dpai_s)
        etvegsun_p = jnp.sum((_evleaf_sun[ics] + _trleaf_sun[ics]) * fs_v * dpai_s)
        etvegsha_p = jnp.sum((_evleaf_sha[ics] + _trleaf_sha[ics]) * fsh_v * dpai_s)
        gppvegsun_p = jnp.sum(_agross_sun[ics] * fs_v * dpai_s * fracgreen_v)
        gppvegsha_p = jnp.sum(_agross_sha[ics] * fsh_v * dpai_s * fracgreen_v)
        vcmax25sun_p = jnp.sum(_vcmax25_sun[ics] * fs_v * dpai_s)
        vcmax25sha_p = jnp.sum(_vcmax25_sha[ics] * fsh_v * dpai_s)
        gsvegsun_p = jnp.sum(_gs_sun[ics] * fs_v * dpai_s)
        gsvegsha_p = jnp.sum(_gs_sha[ics] * fsh_v * dpai_s)

        laisun = laisun.at[p].set(laisun_p)
        laisha = laisha.at[p].set(laisha_p)
        swvegsun = swvegsun.at[p, ivis].set(swvegsun_vis)
        swvegsun = swvegsun.at[p, inir].set(swvegsun_nir)
        swvegsha = swvegsha.at[p, ivis].set(swvegsha_vis)
        swvegsha = swvegsha.at[p, inir].set(swvegsha_nir)
        lwvegsun = lwvegsun.at[p].set(lwvegsun_p)
        lwvegsha = lwvegsha.at[p].set(lwvegsha_p)
        shvegsun = shvegsun.at[p].set(shvegsun_p)
        shvegsha = shvegsha.at[p].set(shvegsha_p)
        lhvegsun = lhvegsun.at[p].set(lhvegsun_p)
        lhvegsha = lhvegsha.at[p].set(lhvegsha_p)
        etvegsun = etvegsun.at[p].set(etvegsun_p)
        etvegsha = etvegsha.at[p].set(etvegsha_p)
        gppvegsun = gppvegsun.at[p].set(gppvegsun_p)
        gppvegsha = gppvegsha.at[p].set(gppvegsha_p)
        vcmax25sun = vcmax25sun.at[p].set(vcmax25sun_p)
        vcmax25sha = vcmax25sha.at[p].set(vcmax25sha_p)
        gsvegsun = gsvegsun.at[p].set(gsvegsun_p)
        gsvegsha = gsvegsha.at[p].set(gsvegsha_p)

        # --- 9. PAI-weighted temperatures and wind (pure JAX, no D→H) ---
        total_pai = laisun_p + laisha_p
        wind_v = _wind[ics]
        tair_v = _tair[ics]
        # Guard against zero PAI with jnp.where — avoids D→H for the if/else
        total_safe = jnp.maximum(total_pai, 1e-30)
        laisun_safe = jnp.maximum(laisun_p, 1e-30)
        laisha_safe = jnp.maximum(laisha_p, 1e-30)
        windveg_p = jnp.where(total_pai > 0.0, jnp.sum(wind_v * dpai_s) / total_safe, 0.0)
        windvegsun_p = jnp.where(laisun_p > 0.0, jnp.sum(wind_v * fs_v * dpai_s) / laisun_safe, 0.0)
        windvegsha_p = jnp.where(
            laisha_p > 0.0, jnp.sum(wind_v * fsh_v * dpai_s) / laisha_safe, 0.0
        )
        tlveg_p = jnp.where(total_pai > 0.0, jnp.sum(tl_m * dpai_s) / total_safe, 0.0)
        tlvegsun_p = jnp.where(
            laisun_p > 0.0, jnp.sum(_tleaf_sun[ics] * fs_v * dpai_s) / laisun_safe, 0.0
        )
        tlvegsha_p = jnp.where(
            laisha_p > 0.0, jnp.sum(_tleaf_sha[ics] * fsh_v * dpai_s) / laisha_safe, 0.0
        )
        taveg_p = jnp.where(total_pai > 0.0, jnp.sum(tair_v * dpai_s) / total_safe, 0.0)
        tavegsun_p = jnp.where(laisun_p > 0.0, jnp.sum(tair_v * fs_v * dpai_s) / laisun_safe, 0.0)
        tavegsha_p = jnp.where(laisha_p > 0.0, jnp.sum(tair_v * fsh_v * dpai_s) / laisha_safe, 0.0)

        windveg = windveg.at[p].set(windveg_p)
        windvegsun = windvegsun.at[p].set(windvegsun_p)
        windvegsha = windvegsha.at[p].set(windvegsha_p)
        tlveg = tlveg.at[p].set(tlveg_p)
        tlvegsun = tlvegsun.at[p].set(tlvegsun_p)
        tlvegsha = tlvegsha.at[p].set(tlvegsha_p)
        taveg = taveg.at[p].set(taveg_p)
        tavegsun = tavegsun.at[p].set(tavegsun_p)
        tavegsha = tavegsha.at[p].set(tavegsha_p)

        # --- 10. Water-stress fraction ---
        total_pai_lai_sai = lai_p + sai_p
        fracminlwp_p = jnp.sum(jnp.where(lp_m <= -2.0, dpai_s, 0.0))
        fracminlwp_p = jnp.where(
            total_pai_lai_sai > 0.0, fracminlwp_p / total_pai_lai_sai, fracminlwp_p
        )
        fracminlwp = fracminlwp.at[p].set(fracminlwp_p)

        # ----------------------------------------------------------------
        # Bulk JAX write-backs for per-layer profile fields
        # ----------------------------------------------------------------
        swleaf_mean = swleaf_mean.at[p, _sl, ivis].set(sw_m_vis)
        swleaf_mean = swleaf_mean.at[p, _sl, inir].set(sw_m_nir)
        lwleaf_mean = lwleaf_mean.at[p, _sl].set(lw_m)
        rnleaf_mean = rnleaf_mean.at[p, _sl].set(rn_m)
        stleaf_mean = stleaf_mean.at[p, _sl].set(st_m)
        shleaf_mean = shleaf_mean.at[p, _sl].set(sh_m)
        lhleaf_mean = lhleaf_mean.at[p, _sl].set(lh_m)
        etleaf_mean = etleaf_mean.at[p, _sl].set(et_m)
        trleaf_mean = trleaf_mean.at[p, _sl].set(tr_m)
        evleaf_mean = evleaf_mean.at[p, _sl].set(ev_m)
        fco2_mean = fco2_mean.at[p, _sl].set(fco2_m)
        apar_mean = apar_mean.at[p, _sl].set(apar_m)
        gs_mean = gs_mean.at[p, _sl].set(gs_m)
        tleaf_mean = tleaf_mean.at[p, _sl].set(tl_m)
        lwp_mean = lwp_mean.at[p, _sl].set(lp_m)
        swsrc = swsrc.at[p, _sl, ivis].set(sw_src_vis)
        swsrc = swsrc.at[p, _sl, inir].set(sw_src_nir)
        lwsrc = lwsrc.at[p, _sl].set(lw_src)
        rnsrc = rnsrc.at[p, _sl].set(rn_src)
        stsrc = stsrc.at[p, _sl].set(st_src)
        shsrc = shsrc.at[p, _sl].set(sh_src)
        lhsrc = lhsrc.at[p, _sl].set(lh_src)
        etsrc = etsrc.at[p, _sl].set(et_src)
        trsrc = trsrc.at[p, _sl].set(tr_src)
        evsrc = evsrc.at[p, _sl].set(ev_src)
        fco2src = fco2src.at[p, _sl].set(fco2_src)

    # ==================================================================
    # Commit all updates — single ._replace() call
    # ==================================================================
    return mlcanopy_inst._replace(
        rnet_canopy=rnet,
        stflx_air_canopy=stflx_air,
        stflx_veg_canopy=stflx_veg,
        shflx_canopy=shflx,
        lhflx_canopy=lhflx,
        etflx_canopy=etflx,
        albcan_canopy=albcan,
        swveg_canopy=swveg,
        swvegsun_canopy=swvegsun,
        swvegsha_canopy=swvegsha,
        lwveg_canopy=lwveg,
        lwvegsun_canopy=lwvegsun,
        lwvegsha_canopy=lwvegsha,
        shveg_canopy=shveg,
        shvegsun_canopy=shvegsun,
        shvegsha_canopy=shvegsha,
        lhveg_canopy=lhveg,
        lhvegsun_canopy=lhvegsun,
        lhvegsha_canopy=lhvegsha,
        etveg_canopy=etveg,
        etvegsun_canopy=etvegsun,
        etvegsha_canopy=etvegsha,
        trveg_canopy=trveg,
        evveg_canopy=evveg,
        gppveg_canopy=gppveg,
        gppvegsun_canopy=gppvegsun,
        gppvegsha_canopy=gppvegsha,
        vcmax25veg_canopy=vcmax25veg,
        vcmax25sun_canopy=vcmax25sun,
        vcmax25sha_canopy=vcmax25sha,
        gsveg_canopy=gsveg,
        gsvegsun_canopy=gsvegsun,
        gsvegsha_canopy=gsvegsha,
        windveg_canopy=windveg,
        windvegsun_canopy=windvegsun,
        windvegsha_canopy=windvegsha,
        tlveg_canopy=tlveg,
        tlvegsun_canopy=tlvegsun,
        tlvegsha_canopy=tlvegsha,
        taveg_canopy=taveg,
        tavegsun_canopy=tavegsun,
        tavegsha_canopy=tavegsha,
        laisun_canopy=laisun,
        laisha_canopy=laisha,
        fracminlwp_canopy=fracminlwp,
        swsrc_profile=swsrc,
        lwsrc_profile=lwsrc,
        rnsrc_profile=rnsrc,
        stsrc_profile=stsrc,
        shsrc_profile=shsrc,
        lhsrc_profile=lhsrc,
        etsrc_profile=etsrc,
        trsrc_profile=trsrc,
        evsrc_profile=evsrc,
        fco2src_profile=fco2src,
        swleaf_mean_profile=swleaf_mean,
        lwleaf_mean_profile=lwleaf_mean,
        rnleaf_mean_profile=rnleaf_mean,
        stleaf_mean_profile=stleaf_mean,
        shleaf_mean_profile=shleaf_mean,
        lhleaf_mean_profile=lhleaf_mean,
        etleaf_mean_profile=etleaf_mean,
        trleaf_mean_profile=trleaf_mean,
        evleaf_mean_profile=evleaf_mean,
        fco2_mean_profile=fco2_mean,
        apar_mean_profile=apar_mean,
        gs_mean_profile=gs_mean,
        tleaf_mean_profile=tleaf_mean,
        lwp_mean_profile=lwp_mean,
    )


# ---------------------------------------------------------------------------
# Differentiable forward pass factory
# ---------------------------------------------------------------------------


def make_clm_ml_forward(
    mlcanopy_inst_template: mlcanopy_type,
    bounds: Any,
    num_exposedvegp: int,
    filter_exposedvegp: Sequence[int],
    atm2lnd_inst: Any,
    canopystate_inst: Any,
    soilstate_inst: Any,
    temperature_inst: Any,
    waterstatebulk_inst: Any,
    waterfluxbulk_inst: Any,
    energyflux_inst: Any,
    frictionvel_inst: Any,
    surfalb_inst: Any,
    solarabs_inst: Any,
    wateratm2lndbulk_inst: Any,
    waterdiagnosticbulk_inst: Any,
):
    """Create a differentiable forward function with baked-in structural params.

    Usage::

        forward_fn = make_clm_ml_forward(mlcanopy_inst, bounds, ...)
        grads = jax.grad(forward_fn)(mlcanopy_inst)

    Structural parameters (ncan, ntop, nbot, p) are extracted from the
    template instance as concrete Python ints *before* ``jax.grad`` tracing.
    All other CLM instances are captured via closure and treated as constants
    (not differentiated).  Only ``mlcanopy_inst`` is the differentiation
    variable.

    Returns:
        A function ``forward(mlcanopy_inst) -> scalar`` suitable for
        ``jax.grad``.
    """
    _p = int(filter_exposedvegp[0])
    grid = GridInfo(
        p=_p,
        ncan=int(mlcanopy_inst_template.ncan_canopy[_p]),
        ntop=int(mlcanopy_inst_template.ntop_canopy[_p]),
        nbot=int(mlcanopy_inst_template.nbot_canopy[_p]),
        pft=int(patch.itype[_p]),
    )
    # Pre-extract o2ref as a concrete Python float from the template instance.
    # mlcanopy_inst inside forward() is abstract under jax.grad tracing.
    _o2ref_py_fwd: float = float(mlcanopy_inst_template.o2ref_forcing[_p])

    def forward(mlcanopy_inst: mlcanopy_type) -> jnp.ndarray:
        """Differentiable CLM-ML forward pass. Returns scalar loss."""
        inst = MLCanopyFluxes(
            bounds=bounds,
            num_exposedvegp=num_exposedvegp,
            filter_exposedvegp=filter_exposedvegp,
            atm2lnd_inst=atm2lnd_inst,
            canopystate_inst=canopystate_inst,
            soilstate_inst=soilstate_inst,
            temperature_inst=temperature_inst,
            waterstatebulk_inst=waterstatebulk_inst,
            waterfluxbulk_inst=waterfluxbulk_inst,
            energyflux_inst=energyflux_inst,
            frictionvel_inst=frictionvel_inst,
            surfalb_inst=surfalb_inst,
            solarabs_inst=solarabs_inst,
            mlcanopy_inst=mlcanopy_inst,
            wateratm2lndbulk_inst=wateratm2lndbulk_inst,
            waterdiagnosticbulk_inst=waterdiagnosticbulk_inst,
            grid=grid,
            _o2ref_py=_o2ref_py_fwd,
        )
        p = grid.p
        n = grid.ncan
        # Scalar loss: sum of sensible + latent heat flux profiles
        loss = jnp.sum(inst.shair_profile[p, 1 : n + 1]) + jnp.sum(inst.etair_profile[p, 1 : n + 1])
        return loss

    return forward
