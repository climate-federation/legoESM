"""
JAX translation of MLLeafFluxesMod Fortran module.

Leaf temperature and energy fluxes for the multilayer canopy model.

Original Fortran module: MLLeafFluxesMod
Fortran lines 1-115

Differentiability notes
-----------------------
* All ``float()`` wrappers removed — JAX scalar arithmetic works on
  traced values.
* ``if dpai_ic > 0:`` replaced by ``jnp.where`` masks.  Both branches
  are always computed; safe denominators (``jnp.where(..., denom, 1.0)``)
  prevent NaN/Inf when conductances are zero on empty layers.
* ``SatVap`` and ``LatVap`` are now fully differentiable (see
  MLWaterVaporMod).
* Energy balance check uses ``jnp.abs``; the Python ``if`` on the
  result is a diagnostic-only path skipped under ``jax.jit``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm.land.canopy.clm_ml_backend.clm_src_main.abortutils import endrun  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLCanopyFluxesType import mlcanopy_type  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLclm_varctl import dtime_ml  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLWaterVaporMod import LatVap, SatVap  # noqa: F401


def LeafFluxes(
    p: int,
    ic: int,
    il: int,
    mlcanopy_inst: mlcanopy_type,
) -> mlcanopy_type:
    """
    Calculate leaf temperature and energy fluxes.

    Mirrors Fortran subroutine ``LeafFluxes`` (lines 22-115).

    Reference: Bonan et al. (2018) *Geosci. Model Dev.*, 11, 1467-1496,
    doi:10.5194/gmd-11-1467-2018, eqs. (10)-(13).

    The leaf temperature is obtained by linearising the energy balance
    around ``tleaf_bef`` (the previous timestep's leaf temperature):

    .. code-block:: none

        gleaf = gs * gbv / (gs + gbv)                   [transpiration conductance]
        gw    = gleaf * fdry + gbv * fwet               [total conductance]

        num1  = 2 * cpair * gbh
        num2  = lambda * gw
        num3  = rnleaf - lambda*gw*(qsat - dqsat*tleaf_bef)
                + (cpleaf/dtime) * tleaf_bef
        den   = cpleaf/dtime + num1 + num2*dqsat
        tleaf = (num1*tair + num2*(eair/pref) + num3) / den

    Derived fluxes (Fortran lines 87-100):

    .. code-block:: none

        stleaf = (tleaf - tleaf_bef) * cpleaf / dtime          [W/m2 leaf]
        shleaf = 2 * cpair * (tleaf - tair) * gbh              [W/m2 leaf]
        num1   = qsat + dqsat*(tleaf - tleaf_bef) - eair/pref
        trleaf = gleaf * fdry * num1                            [mol H2O/m2/s]
        evleaf = gbv   * fwet * num1                            [mol H2O/m2/s]
        lhleaf = (trleaf + evleaf) * lambda                     [W/m2 leaf]

    Energy balance closure is checked to within 1e-3 W/m2.

    Layers with ``dpai == 0`` have all fluxes set to zero and
    ``tleaf = tair(p, ic)``.

    Args:
        p: Patch index.
        ic: Canopy layer index.
        il: Sunlit (``isun``) or shaded (``isha``) leaf index.
        mlcanopy_inst: Canopy container; ``tleaf_leaf``,
            ``stleaf_leaf``, ``shleaf_leaf``, ``lhleaf_leaf``,
            ``evleaf_leaf``, and ``trleaf_leaf`` are updated.

    Returns:
        Updated :class:`mlcanopy_type`.
    """
    dtime = dtime_ml  # Multilayer canopy timestep (s) — Python float constant

    # Latent heat of vaporization — Fortran line 65
    lam = LatVap(mlcanopy_inst.tref_forcing[p])  # J/mol

    dpai_ic = mlcanopy_inst.dpai_profile[p, ic]
    active = dpai_ic > 0.0  # JAX bool scalar

    # Unpack scalars (used in both branches)
    tair_ic = mlcanopy_inst.tair_profile[p, ic]

    # -----------------------------------------------------------------------
    # Active-layer physics — computed unconditionally; masked below
    # Fortran lines 67-96
    # -----------------------------------------------------------------------
    pref_p = mlcanopy_inst.pref_forcing[p]
    cpair_p = mlcanopy_inst.cpair_forcing[p]
    eair_ic = mlcanopy_inst.eair_profile[p, ic]
    cpleaf_ic = mlcanopy_inst.cpleaf_profile[p, ic]
    fwet_ic = mlcanopy_inst.fwet_profile[p, ic]
    fdry_ic = mlcanopy_inst.fdry_profile[p, ic]
    gbh_ic = mlcanopy_inst.gbh_leaf[p, ic, il]
    gbv_ic = mlcanopy_inst.gbv_leaf[p, ic, il]
    gs_ic = mlcanopy_inst.gs_leaf[p, ic, il]
    rnleaf_ic = mlcanopy_inst.rnleaf_leaf[p, ic, il]
    tleaf_bef_ic = mlcanopy_inst.tleaf_bef_leaf[p, ic, il]

    # Saturation vapour pressure at tleaf_bef — Fortran lines 69-70
    esat, desat = SatVap(tleaf_bef_ic)
    qsat = esat / pref_p  # mol/mol
    dqsat = desat / pref_p  # mol/mol/K

    # Leaf transpiration conductance — Fortran line 72
    # Safe denominator: avoid 0/0 when gs = gbv = 0 on empty layers
    # jnp.maximum avoids select op → prevents XLA select_divide_fusion bug
    gleaf_denom = jnp.maximum(gs_ic + gbv_ic, 1.0e-30)
    gleaf = gs_ic * gbv_ic / gleaf_denom

    # Total conductance — Fortran line 74
    gw = gleaf * fdry_ic + gbv_ic * fwet_ic

    # Linearised leaf temperature — Fortran lines 76-81
    num1 = 2.0 * cpair_p * gbh_ic
    num2 = lam * gw
    num3 = rnleaf_ic - lam * gw * (qsat - dqsat * tleaf_bef_ic) + cpleaf_ic / dtime * tleaf_bef_ic
    # Safe denominator: avoid /0 when cpleaf = gbh = gw = 0 on empty layers
    # jnp.maximum avoids select op → prevents XLA select_divide_fusion bug
    tleaf_denom = jnp.maximum(cpleaf_ic / dtime + num1 + num2 * dqsat, 1.0e-30)
    tleaf_active = (num1 * tair_ic + num2 * (eair_ic / pref_p) + num3) / tleaf_denom

    # Storage heat flux — Fortran line 83
    stleaf_active = (tleaf_active - tleaf_bef_ic) * cpleaf_ic / dtime

    # Sensible heat flux — Fortran line 85
    shleaf_active = 2.0 * cpair_p * (tleaf_active - tair_ic) * gbh_ic

    # Transpiration and evaporation — Fortran lines 87-89
    vapour_flux = qsat + dqsat * (tleaf_active - tleaf_bef_ic) - eair_ic / pref_p
    trleaf_active = gleaf * fdry_ic * vapour_flux
    evleaf_active = gbv_ic * fwet_ic * vapour_flux

    # Latent heat flux — Fortran line 91
    lhleaf_active = (trleaf_active + evleaf_active) * lam

    # -----------------------------------------------------------------------
    # Select active vs. inactive values — Fortran lines 98-104
    # -----------------------------------------------------------------------
    tleaf_val = jnp.where(active, tleaf_active, tair_ic)
    stleaf_val = jnp.where(active, stleaf_active, 0.0)
    shleaf_val = jnp.where(active, shleaf_active, 0.0)
    lhleaf_val = jnp.where(active, lhleaf_active, 0.0)
    evleaf_val = jnp.where(active, evleaf_active, 0.0)
    trleaf_val = jnp.where(active, trleaf_active, 0.0)

    # Energy balance error check — JIT-compatible via debug.callback
    # Mask inactive layers (dpai=0, rnleaf may hold spval=1e36) to avoid false errors
    err = jnp.where(active, rnleaf_ic - shleaf_val - lhleaf_val - stleaf_val, 0.0)
    jax.debug.callback(
        lambda e: (
            endrun(msg=" ERROR: LeafFluxes: energy balance error")
            if abs(float(e)) > 1.0e-3
            else None
        ),
        err,
    )

    return mlcanopy_inst._replace(
        tleaf_leaf=mlcanopy_inst.tleaf_leaf.at[p, ic, il].set(tleaf_val),
        stleaf_leaf=mlcanopy_inst.stleaf_leaf.at[p, ic, il].set(stleaf_val),
        shleaf_leaf=mlcanopy_inst.shleaf_leaf.at[p, ic, il].set(shleaf_val),
        lhleaf_leaf=mlcanopy_inst.lhleaf_leaf.at[p, ic, il].set(lhleaf_val),
        evleaf_leaf=mlcanopy_inst.evleaf_leaf.at[p, ic, il].set(evleaf_val),
        trleaf_leaf=mlcanopy_inst.trleaf_leaf.at[p, ic, il].set(trleaf_val),
    )
