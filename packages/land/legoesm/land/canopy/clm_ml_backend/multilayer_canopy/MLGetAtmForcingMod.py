"""
JAX translation of MLGetAtmForcingMod Fortran module.

Atmospheric forcing for the current multilayer canopy timestep.
Provides time interpolation and derived variable computation for
atmospheric boundary conditions passed to the multilayer canopy model.

Original Fortran module: MLGetAtmForcingMod
Fortran lines 1-175
"""

import jax.numpy as jnp
from jax import Array

from legoesm.land.canopy.clm_ml_backend.clm_src_main.abortutils import endrun  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_main.clm_varctl import iulog  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_main.clm_varpar import inir, ivis  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLCanopyFluxesType import mlcanopy_type  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLclm_varcon import (  # noqa: F401
    cpd,
    cpw,
    lapse_rate,
    mmdry,
    mmh2o,
    rgas,
    wind_forc_min,
)
from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLclm_varctl import GridInfo, met_type  # noqa: F401

# ---------------------------------------------------------------------------
# Private: 2-point linear interpolation
# ---------------------------------------------------------------------------


def TimeInterpolation2(
    x0: float,
    x1: float,
    t0: float,
    t1: float,
    tx: float,
) -> float:
    """
    2-point linear interpolation of atmospheric forcing from a CLM
    timestep to a multilayer canopy (ML) sub-timestep.

    Forcing is linear from ``x0`` at ``t0`` to ``x1`` at ``t1``.

    Mirrors Fortran function ``TimeInterpolation2`` (lines 30-55).

    Args:
        x0: Value at the preceding CLM timestep (at time ``t0``).
        x1: Value at the current CLM timestep (at time ``t1``).
        t0: Calendar day for the preceding CLM timestep.
        t1: Calendar day for the current CLM timestep.
        tx: Calendar day for the ML sub-timestep interpolation target.

    Returns:
        Interpolated value at time ``tx``.
    """
    # Fortran lines 50-52
    b1 = (x1 - x0) / (t1 - t0)  # Slope for linear interpolation
    b0 = x1 - b1 * t1  # Intercept for linear interpolation
    return b0 + b1 * tx


# ---------------------------------------------------------------------------
# Private: 3-point linear interpolation
# ---------------------------------------------------------------------------


def TimeInterpolation3(
    x0: float,
    x1: float,
    x2: float,
    t0: float,
    t1: float,
    t2: float,
    tx: float,
) -> float:
    """
    3-point piecewise linear interpolation of atmospheric forcing from
    CLM timesteps to a multilayer canopy (ML) sub-timestep.

    Uses the ``(t0, x0)``-to-``(t1, x1)`` segment when ``tx < t1``,
    and the ``(t1, x1)``-to-``(t2, x2)`` segment when ``tx > t1``.

    Mirrors Fortran function ``TimeInterpolation3`` (lines 57-88).

    Args:
        x0: Value at the preceding CLM timestep (at time ``t0``).
        x1: Value at the current CLM timestep (at time ``t1``).
        x2: Value at the next CLM timestep (at time ``t2``).
        t0: Calendar day for the preceding CLM timestep.
        t1: Calendar day for the current CLM timestep.
        t2: Calendar day for the next CLM timestep.
        tx: Calendar day for the ML sub-timestep interpolation target.

    Returns:
        Interpolated value at time ``tx``.
    """
    # Fortran lines 80-86: piecewise linear, branch on tx vs t1
    # Use jnp.where instead of Python if so tx can be a JAX traced value
    # (required when called from inside lax.fori_loop in diff mode).
    b1_lo = (x1 - x0) / (t1 - t0)  # Slope for lower segment
    b0_lo = x0 - b1_lo * t0  # Intercept for lower segment
    b1_hi = (x2 - x1) / (t2 - t1)  # Slope for upper segment
    b0_hi = x1 - b1_hi * t1  # Intercept for upper segment
    b1 = jnp.where(tx < t1, b1_lo, b1_hi)
    b0 = jnp.where(tx < t1, b0_lo, b0_hi)
    return b0 + b1 * tx


# ---------------------------------------------------------------------------
# Public: atmospheric forcing for current ML timestep
# ---------------------------------------------------------------------------


def GetAtmForcing(
    time_bef: float,
    time_cur: float,
    time_next: float,
    time_ml: float,
    num_filter: int,
    filter: Array,
    mlcanopy_inst: mlcanopy_type,
    grid: "GridInfo | None" = None,
) -> mlcanopy_type:
    """
    Interpolate and derive atmospheric forcing for the current multilayer
    canopy sub-timestep.

    Mirrors Fortran subroutine ``GetAtmForcing`` (lines 90-175).

    Depending on ``met_type``, one of three strategies is applied to
    each patch in ``filter``:

    - ``met_type == 0``: No interpolation; use current CLM timestep values.
    - ``met_type == 2``: 2-point linear interpolation from ``bef`` to ``cur``.
    - ``met_type == 3``: 3-point piecewise linear interpolation across
      ``bef``, ``cur``, and ``next``.

    After interpolation the following derived quantities are computed
    (Fortran lines 158-166):

    - ``eref``:   Vapor pressure at reference height (Pa).
    - ``rhomol``: Molar density at reference height (mol/m3).
    - ``rhoair``: Air density at reference height (kg/m3).
    - ``mmair``:  Molecular mass of air (kg/mol).
    - ``cpair``:  Specific heat of air at constant pressure (J/mol/K).
    - ``thref``:  Potential temperature at reference height (K).
    - ``thvref``: Virtual potential temperature at reference height (K).

    Args:
        time_bef: Calendar day for the preceding CLM timestep.
        time_cur: Calendar day for the current CLM timestep.
        time_next: Calendar day for the next CLM timestep.
        time_ml: Calendar day for the ML sub-timestep interpolation target.
        num_filter: Number of patches in the filter.
        filter: 1-D array of patch indices to process.
        mlcanopy_inst: Multilayer canopy state container.

    Returns:
        Updated :class:`mlcanopy_type` with interpolated and derived
        atmospheric forcing fields populated.
    """
    # ------------------------------------------------------------------
    # Unpack input forcing arrays (Fortran associate block, lines 108-153)
    # ------------------------------------------------------------------
    tref_bef = mlcanopy_inst.tref_bef_forcing
    tref_cur = mlcanopy_inst.tref_cur_forcing
    tref_next = mlcanopy_inst.tref_next_forcing
    qref_bef = mlcanopy_inst.qref_bef_forcing
    qref_cur = mlcanopy_inst.qref_cur_forcing
    qref_next = mlcanopy_inst.qref_next_forcing
    uref_bef = mlcanopy_inst.uref_bef_forcing
    uref_cur = mlcanopy_inst.uref_cur_forcing
    uref_next = mlcanopy_inst.uref_next_forcing
    pref_bef = mlcanopy_inst.pref_bef_forcing
    pref_cur = mlcanopy_inst.pref_cur_forcing
    pref_next = mlcanopy_inst.pref_next_forcing
    co2ref_bef = mlcanopy_inst.co2ref_bef_forcing
    co2ref_cur = mlcanopy_inst.co2ref_cur_forcing
    co2ref_next = mlcanopy_inst.co2ref_next_forcing
    swskyb_bef = mlcanopy_inst.swskyb_bef_forcing
    swskyb_cur = mlcanopy_inst.swskyb_cur_forcing
    swskyb_next = mlcanopy_inst.swskyb_next_forcing
    swskyd_bef = mlcanopy_inst.swskyd_bef_forcing
    swskyd_cur = mlcanopy_inst.swskyd_cur_forcing
    swskyd_next = mlcanopy_inst.swskyd_next_forcing
    lwsky_bef = mlcanopy_inst.lwsky_bef_forcing
    lwsky_cur = mlcanopy_inst.lwsky_cur_forcing
    lwsky_next = mlcanopy_inst.lwsky_next_forcing
    zref = mlcanopy_inst.zref_forcing

    # Mutable output arrays — written per patch then stored back
    tref = mlcanopy_inst.tref_forcing
    qref = mlcanopy_inst.qref_forcing
    uref = mlcanopy_inst.uref_forcing
    pref = mlcanopy_inst.pref_forcing
    co2ref = mlcanopy_inst.co2ref_forcing
    swskyb = mlcanopy_inst.swskyb_forcing
    swskyd = mlcanopy_inst.swskyd_forcing
    lwsky = mlcanopy_inst.lwsky_forcing
    thref = mlcanopy_inst.thref_forcing
    thvref = mlcanopy_inst.thvref_forcing
    eref = mlcanopy_inst.eref_forcing
    rhoair = mlcanopy_inst.rhoair_forcing
    rhomol = mlcanopy_inst.rhomol_forcing
    mmair = mlcanopy_inst.mmair_forcing
    cpair = mlcanopy_inst.cpair_forcing

    # Pre-compute constant — avoids repeated division in the loop
    _eps = mmh2o / mmdry  # ratio of molecular masses
    _one_minus_eps = 1.0 - _eps

    # ------------------------------------------------------------------
    # Per-patch loop — Fortran lines 156-167
    # ------------------------------------------------------------------
    for fp in range(num_filter):
        if grid is not None:
            p = grid.p
        else:
            p = int(filter[fp])

        # --------------------------------------------------------------
        # Atmospheric forcing interpolation — values stay as JAX scalars
        # when grid is provided (differentiable mode).
        # --------------------------------------------------------------
        if met_type == 0:
            # No interpolation: use current CLM timestep values
            # Fortran lines 122-131
            _uref_p = uref_cur[p]
            _tref_p = tref_cur[p]
            _qref_p = qref_cur[p]
            _pref_p = pref_cur[p]
            _co2ref_p = co2ref_cur[p]
            _swskyb_vis = swskyb_cur[p, ivis]
            _swskyd_vis = swskyd_cur[p, ivis]
            _swskyb_nir = swskyb_cur[p, inir]
            _swskyd_nir = swskyd_cur[p, inir]
            _lwsky_p = lwsky_cur[p]

        elif met_type == 2:
            # 2-point linear interpolation from bef to cur
            # Fortran lines 133-145  (endrun guard preserved)
            endrun(msg=" ERROR: met_type not valid")
            _uref_p = TimeInterpolation2(uref_bef[p], uref_cur[p], time_bef, time_cur, time_ml)
            _tref_p = TimeInterpolation2(tref_bef[p], tref_cur[p], time_bef, time_cur, time_ml)
            _qref_p = TimeInterpolation2(qref_bef[p], qref_cur[p], time_bef, time_cur, time_ml)
            _pref_p = TimeInterpolation2(pref_bef[p], pref_cur[p], time_bef, time_cur, time_ml)
            _co2ref_p = TimeInterpolation2(
                co2ref_bef[p], co2ref_cur[p], time_bef, time_cur, time_ml
            )
            _swskyb_vis = TimeInterpolation2(
                swskyb_bef[p, ivis], swskyb_cur[p, ivis], time_bef, time_cur, time_ml
            )
            _swskyd_vis = TimeInterpolation2(
                swskyd_bef[p, ivis], swskyd_cur[p, ivis], time_bef, time_cur, time_ml
            )
            _swskyb_nir = TimeInterpolation2(
                swskyb_bef[p, inir], swskyb_cur[p, inir], time_bef, time_cur, time_ml
            )
            _swskyd_nir = TimeInterpolation2(
                swskyd_bef[p, inir], swskyd_cur[p, inir], time_bef, time_cur, time_ml
            )
            _lwsky_p = TimeInterpolation2(lwsky_bef[p], lwsky_cur[p], time_bef, time_cur, time_ml)

        elif met_type == 3:
            # 3-point piecewise linear interpolation across bef, cur, next
            # Fortran lines 147-158
            _uref_p = TimeInterpolation3(
                uref_bef[p], uref_cur[p], uref_next[p], time_bef, time_cur, time_next, time_ml
            )
            _tref_p = TimeInterpolation3(
                tref_bef[p], tref_cur[p], tref_next[p], time_bef, time_cur, time_next, time_ml
            )
            _qref_p = TimeInterpolation3(
                qref_bef[p], qref_cur[p], qref_next[p], time_bef, time_cur, time_next, time_ml
            )
            _pref_p = TimeInterpolation3(
                pref_bef[p], pref_cur[p], pref_next[p], time_bef, time_cur, time_next, time_ml
            )
            _co2ref_p = TimeInterpolation3(
                co2ref_bef[p], co2ref_cur[p], co2ref_next[p], time_bef, time_cur, time_next, time_ml
            )
            _swskyb_vis = TimeInterpolation3(
                swskyb_bef[p, ivis],
                swskyb_cur[p, ivis],
                swskyb_next[p, ivis],
                time_bef,
                time_cur,
                time_next,
                time_ml,
            )
            _swskyd_vis = TimeInterpolation3(
                swskyd_bef[p, ivis],
                swskyd_cur[p, ivis],
                swskyd_next[p, ivis],
                time_bef,
                time_cur,
                time_next,
                time_ml,
            )
            _swskyb_nir = TimeInterpolation3(
                swskyb_bef[p, inir],
                swskyb_cur[p, inir],
                swskyb_next[p, inir],
                time_bef,
                time_cur,
                time_next,
                time_ml,
            )
            _swskyd_nir = TimeInterpolation3(
                swskyd_bef[p, inir],
                swskyd_cur[p, inir],
                swskyd_next[p, inir],
                time_bef,
                time_cur,
                time_next,
                time_ml,
            )
            _lwsky_p = TimeInterpolation3(
                lwsky_bef[p], lwsky_cur[p], lwsky_next[p], time_bef, time_cur, time_next, time_ml
            )

        # --------------------------------------------------------------
        # Minimum wind speed restriction — Fortran line 161
        # jnp.maximum works on both JAX scalars and Python floats
        # --------------------------------------------------------------
        _uref_p = jnp.maximum(wind_forc_min, _uref_p)

        # --------------------------------------------------------------
        # Derived atmospheric variables — Fortran lines 164-170
        # --------------------------------------------------------------
        _zref_p = zref[p]

        # Vapor pressure at reference height (Pa)
        _eref_p = _qref_p * _pref_p / (_eps + _one_minus_eps * _qref_p)

        # Molar density at reference height (mol/m3): p / (R * T)
        _rhomol_p = _pref_p / (rgas * _tref_p)

        # Air density at reference height (kg/m3)
        _rhoair_p = _rhomol_p * mmdry * (1.0 - _one_minus_eps * _eref_p / _pref_p)

        # Molecular mass of air (kg/mol)
        _mmair_p = _rhoair_p / _rhomol_p

        # Specific heat of moist air at constant pressure (J/mol/K)
        _cpair_p = cpd * (1.0 + (cpw / cpd - 1.0) * _qref_p) * _mmair_p

        # Potential temperature at reference height (K)
        _thref_p = _tref_p + lapse_rate * _zref_p

        # Virtual potential temperature at reference height (K)
        _thvref_p = _thref_p * (1.0 + 0.61 * _qref_p)

        # Write all fields back — all use Python scalars so XLA can
        # dispatch these independently (no dependency chain).
        uref = uref.at[p].set(_uref_p)
        tref = tref.at[p].set(_tref_p)
        qref = qref.at[p].set(_qref_p)
        pref = pref.at[p].set(_pref_p)
        co2ref = co2ref.at[p].set(_co2ref_p)
        swskyb = swskyb.at[p, ivis].set(_swskyb_vis)
        swskyd = swskyd.at[p, ivis].set(_swskyd_vis)
        swskyb = swskyb.at[p, inir].set(_swskyb_nir)
        swskyd = swskyd.at[p, inir].set(_swskyd_nir)
        lwsky = lwsky.at[p].set(_lwsky_p)
        eref = eref.at[p].set(_eref_p)
        rhomol = rhomol.at[p].set(_rhomol_p)
        rhoair = rhoair.at[p].set(_rhoair_p)
        mmair = mmair.at[p].set(_mmair_p)
        cpair = cpair.at[p].set(_cpair_p)
        thref = thref.at[p].set(_thref_p)
        thvref = thvref.at[p].set(_thvref_p)

    # ------------------------------------------------------------------
    # Write all updated arrays back into the immutable state container
    # ------------------------------------------------------------------
    mlcanopy_inst = mlcanopy_inst._replace(
        tref_forcing=tref,
        qref_forcing=qref,
        uref_forcing=uref,
        pref_forcing=pref,
        co2ref_forcing=co2ref,
        swskyb_forcing=swskyb,
        swskyd_forcing=swskyd,
        lwsky_forcing=lwsky,
        thref_forcing=thref,
        thvref_forcing=thvref,
        eref_forcing=eref,
        rhoair_forcing=rhoair,
        rhomol_forcing=rhomol,
        mmair_forcing=mmair,
        cpair_forcing=cpair,
    )

    return mlcanopy_inst
