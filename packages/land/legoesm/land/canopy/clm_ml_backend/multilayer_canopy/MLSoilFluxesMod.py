"""
JAX translation of MLSoilFluxesMod Fortran module.

Calculate soil surface temperature and energy balance for the
multilayer canopy model.

Original Fortran module: MLSoilFluxesMod
Fortran lines 1-105

Differentiability notes
-----------------------
All ``float()`` wrappers removed — JAX scalar arithmetic works directly
on traced values.  The energy balance error check uses ``jnp.abs`` and
is left as a Python ``if`` on a JAX scalar (diagnostic only; skipped
when running under ``jax.jit``; use ``jax.debug.callback`` for
JIT-compatible checking if needed).
"""

import jax

from legoesm.land.canopy.clm_ml_backend.clm_src_main.abortutils import endrun  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLCanopyFluxesType import mlcanopy_type  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLWaterVaporMod import LatVap, SatVap  # noqa: F401


def SoilFluxes(
    p: int,
    mlcanopy_inst: mlcanopy_type,
) -> mlcanopy_type:
    """
    Compute soil surface temperature and energy balance fluxes.

    Mirrors Fortran subroutine ``SoilFluxes`` (lines 22-105).

    The soil surface temperature ``tg`` is solved analytically by
    linearising the surface energy balance around the previous
    timestep's temperature ``tg_bef``.  See Bonan et al. (2018)
    *Geosci. Model Dev.*, 11, 1467-1496,
    doi:10.5194/gmd-11-1467-2018, eqs. (14)-(15).

    The soil conductance for water vapour is assembled from a series
    combination of the aerodynamic conductance ``gac0`` and a soil
    evaporative resistance converted to molar units:

    .. code-block:: none

        gws = (1 / soilres) * rhomol       [mol H2O/m2/s]
        gw  = gac0 * gws / (gac0 + gws)    [total conductance]

    Surface temperature solution (Fortran lines 71-74):

    .. code-block:: none

        num1 = cpair * gac0
        num2 = lambda * gw
        num3 = soil_tk / soil_dz
        num4 = rnsoi - num2*rhg*(qsat - dqsat*tg_bef) + num3*soil_t
        den  = num1 + num2*dqsat*rhg + num3
        tg   = (num1*tair(1) + num2*(eair(1)/pref) + num4) / den

    Derived fluxes (Fortran lines 76-88):

    .. code-block:: none

        shsoi = cpair * (tg - tair(1)) * gac0          [W/m2]
        eg    = rhg * (esat + desat*(tg - tg_bef))     [Pa]
        lhsoi = lambda * (eg - eair(1)) / pref * gw    [W/m2]
        gsoi  = soil_tk * (tg - soil_t) / soil_dz      [W/m2]
        etsoi = lhsoi / lambda                          [mol H2O/m2/s]

    An energy balance closure check is applied; imbalance
    ``|rnsoi - shsoi - lhsoi - gsoi| > 0.001 W/m2`` is fatal.

    Args:
        p: Patch index into all canopy arrays (1-based).
        mlcanopy_inst: Multilayer canopy container.  All input fields
            are read at index ``p``; the following output fields are
            updated: ``shsoi_soil``, ``lhsoi_soil``, ``gsoi_soil``,
            ``etsoi_soil``, ``tg_soil``, ``eg_soil``.

    Returns:
        Updated :class:`mlcanopy_type`.
    """
    # Unpack inputs (Fortran associate block, lines 43-63)
    tref = mlcanopy_inst.tref_forcing[p]
    pref_p = mlcanopy_inst.pref_forcing[p]
    rhomol_p = mlcanopy_inst.rhomol_forcing[p]
    cpair_p = mlcanopy_inst.cpair_forcing[p]
    rnsoi_p = mlcanopy_inst.rnsoi_soil[p]
    rhg_p = mlcanopy_inst.rhg_soil[p]
    soilres_p = mlcanopy_inst.soilres_soil[p]
    gac0_p = mlcanopy_inst.gac0_soil[p]
    soil_t_p = mlcanopy_inst.soil_t_soil[p]
    soil_dz_p = mlcanopy_inst.soil_dz_soil[p]
    soil_tk_p = mlcanopy_inst.soil_tk_soil[p]
    tg_bef_p = mlcanopy_inst.tg_bef_soil[p]
    tair_1 = mlcanopy_inst.tair_profile[p, 1]
    eair_1 = mlcanopy_inst.eair_profile[p, 1]

    # ------------------------------------------------------------------
    # Latent heat of vaporization — Fortran line 66
    # ------------------------------------------------------------------
    lam = LatVap(tref)  # J/mol

    # ------------------------------------------------------------------
    # Soil conductance for water vapour — Fortran lines 68-70
    # ------------------------------------------------------------------
    gws = (1.0 / soilres_p) * rhomol_p  # s/m → mol H2O/m2/s
    gw = gac0_p * gws / (gac0_p + gws)  # total conductance

    # ------------------------------------------------------------------
    # Saturation vapour pressure at tg_bef — Fortran lines 72-73
    # ------------------------------------------------------------------
    esat, desat = SatVap(tg_bef_p)  # Pa, Pa/K
    qsat = esat / pref_p  # mol/mol
    dqsat = desat / pref_p  # mol/mol/K

    # ------------------------------------------------------------------
    # Soil surface temperature — Fortran lines 75-78
    # ------------------------------------------------------------------
    num1 = cpair_p * gac0_p
    num2 = lam * gw
    num3 = soil_tk_p / soil_dz_p
    num4 = rnsoi_p - num2 * rhg_p * (qsat - dqsat * tg_bef_p) + num3 * soil_t_p
    den = num1 + num2 * dqsat * rhg_p + num3
    tg_p = (num1 * tair_1 + num2 * (eair_1 / pref_p) + num4) / den

    # ------------------------------------------------------------------
    # Sensible heat flux — Fortran line 80
    # ------------------------------------------------------------------
    shsoi_p = cpair_p * (tg_p - tair_1) * gac0_p  # W/m2

    # ------------------------------------------------------------------
    # Latent heat flux — Fortran lines 82-83
    # ------------------------------------------------------------------
    eg_p = rhg_p * (esat + desat * (tg_p - tg_bef_p))  # Pa
    lhsoi_p = lam * (eg_p - eair_1) / pref_p * gw  # W/m2

    # ------------------------------------------------------------------
    # Soil heat flux — Fortran line 85
    # ------------------------------------------------------------------
    gsoi_p = soil_tk_p * (tg_p - soil_t_p) / soil_dz_p  # W/m2

    # ------------------------------------------------------------------
    # Energy balance error check — Fortran lines 87-90
    # Diagnostic only: the jnp.abs call is differentiable; the Python
    # ``if`` on the JAX scalar is skipped under jax.jit.
    # ------------------------------------------------------------------
    err = rnsoi_p - shsoi_p - lhsoi_p - gsoi_p
    # JIT-compatible diagnostic: callback runs host-side with concrete value
    jax.debug.callback(
        lambda e: (
            endrun(msg=" ERROR: SoilFluxes: energy balance error")
            if abs(float(e)) > 0.001
            else None
        ),
        err,
    )

    # ------------------------------------------------------------------
    # Water vapour flux: W/m2 → mol H2O/m2/s — Fortran line 92
    # ------------------------------------------------------------------
    etsoi_p = lhsoi_p / lam

    return mlcanopy_inst._replace(
        shsoi_soil=mlcanopy_inst.shsoi_soil.at[p].set(shsoi_p),
        lhsoi_soil=mlcanopy_inst.lhsoi_soil.at[p].set(lhsoi_p),
        gsoi_soil=mlcanopy_inst.gsoi_soil.at[p].set(gsoi_p),
        etsoi_soil=mlcanopy_inst.etsoi_soil.at[p].set(etsoi_p),
        tg_soil=mlcanopy_inst.tg_soil.at[p].set(tg_p),
        eg_soil=mlcanopy_inst.eg_soil.at[p].set(eg_p),
    )
