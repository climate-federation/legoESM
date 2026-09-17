"""OpenIFS main updraught ascent (cuascn.F90 + cuentr.F90, MAIN updraught).

This module ports the IFS "cuascn" updraught ascent for the ordinary-Earth,
deterministic (SPP perturbations OFF), non-ARPEGE branch with the switch
combination::

    LDTDKMF  = .FALSE.
    LPHYLIN  = .FALSE.
    LSCVLIQ  = .TRUE.
    RMFSOLTQ = 1   (implicit moist correction with RMFCFL = 3)

Scope and declared departures
-----------------------------
* KTYPE 1 (deep) and KTYPE 2 (shallow) convection only.  Mid-level
  convection (KTYPE 3, cubasmcn) is OUT OF SCOPE.
* LIQUID-ONLY thermodynamics: the FOEALFCU latent-heat blend, the ZLGLAC
  (LMFGLAC) glaciation term and the RALFDCP terms are declared departures
  (PLGLAC = 0).
* No precipitation formation inside the plume: the in-updraught
  conversion (cuascn.F90:730-800) lives in the repo module
  ``_ifs_inplume_precip_conversion`` and is a later coupling step.  Inside
  this module PLRAIN = 0 and PDMFUP = 0, declared.
* Specific humidity INSIDE (the caller converts from any external
  moisture variable).

Conventions (shared with ``_ifs_test_ascent``)
----------------------------------------------
* Surface-last arrays with shape ``(ncol, nlev)``; half levels
  ``(ncol, nlev+1)`` with half index ``nlev`` at the surface.
* IFS half level ``JK`` <-> our half index ``JK - 1``; IFS full level
  ``JK`` <-> our full index ``JK - 1``.  The IFS loop ascends with
  decreasing ``JK``; our ``lax.scan`` descends with increasing index.
* Constants only from ``legoesm.constants``; saturation only through the
  helpers of ``legoesm.atmosphere.physics.convection._ifs_test_ascent``.
* All empirical/numerical coefficients carry provenance
  (sucumf.F90 / cuascn.F90 line numbers of the source excerpt).
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import lax

from legoesm.constants import g, R_d, R_v, c_pd, L_v, epsilon, T_freeze
from legoesm.atmosphere.physics._shared import virtual_temperature
from legoesm.atmosphere.physics.convection._ifs_test_ascent import (
    cuadjtq_condense,
    dqsat_dT,
    half_level_env,
    TINY,
    T_PHYS_MIN,
    T_PHYS_MAX,
    RLMIN,
)

__all__ = [
    "IFSAscentConfig",
    "cuentr_turbulent_detrainment",
    "layer_increments",
    "ke_step",
    "organized_entrainment_next",
    "organized_detrainment",
]

# --------------------------------------------------------------------------
# Derived constants (from legoesm.constants only)
# --------------------------------------------------------------------------
_RCPD = c_pd
_RETV = 1.0 / epsilon - 1.0  # IFS RETV
_RG = g


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
class IFSAscentConfig(NamedTuple):
    """Scheme parameters for the IFS main updraught ascent.

    Defaults reproduce the OpenIFS source (sucumf.F90 / cuascn.F90).
    Tier-0 "excluded" fields are numerics / switch-defining constants and
    are not exposed as tunable parameters (see the module-level
    ``__param_spec__``).
    """

    entrorg: float = 1.75e-3          # 1/m, sucumf.F90:136
    entr_rh: float = 1.0              # -, sucumf.F90:140
    entshalp: float = 2.0             # -, sucumf.F90:144
    detrpen: float = 7.5e-5           # 1/m, sucumf.F90:129
    rmfcfl: float = 3.0               # -, implicit branch, sucumf.F90:225/247-251
    rmfcmin: float = 1.0e-8           # kg m-2 s-1, sucumf.F90:160
    entr_org_cap: float = 0.4         # -, cuascn.F90:691
    detr_early_cap: float = 0.75      # -, cuascn.F90:487
    rh_detr_offset: float = 1.6       # -, cuascn.F90:514/672
    rh_entr_offset: float = 0.3       # -, cuascn.F90:686
    buoy_reinterp_K: float = -0.1     # K, cuascn.F90:640
    buoy_entr_shutoff_K: float = -0.2 # K, cuascn.F90:679
    buoy_accept_K: float = -2.0       # K, cuascn.F90:700
    lapse_accept_K_per_m: float = -3.0e-3  # K/m, cuascn.F90:701
    ke_buoy_factor: float = 1.0 / 3.0      # ZFACBUO, cuascn.F90:282
    ke_drag_cw: float = 0.94875            # Z_CWDRAG, cuascn.F90:288
    ke_floor: float = -1000.0              # m2/s2, cuascn.F90:667
    ke_ratio_floor: float = 1.0e-3         # -, cuascn.F90:670
    ptu_min_K: float = 100.0               # K, cuascn.F90:539
    ptu_max_K: float = 400.0               # K, cuascn.F90:540
    # ---- precipitation formation (cuascn.F90:730-806) ----
    rprcon: float = 1.4e-3            # 1/m, sucumf.F90:175 (RPRCON*RPLRG = ZPRCDGW)
    dnoprc: float = 3.0e-4            # kg/kg, ZDNOPRC cuascn.F90:284
    cldmax: float = 5.0e-3            # kg/kg, Z_CLDMAX cuascn.F90:285 (excluded cap)
    cwifrac: float = 0.5              # -, Z_CWIFRAC cuascn.F90:286 (excluded; unused,
                                      #   liquid-only branch => ZALFAW = 1)
    cprc2: float = 0.5                # -, Z_CPRC2 cuascn.F90:287 (excluded)
    ke_wu_floor_conv: float = 0.5     # m2/s2, cuascn.F90:736 (excluded numerics)
    ke_wu_floor_fall: float = 0.1     # m2/s2, cuascn.F90:792 (excluded numerics)
    wu_max: float = 15.0              # m/s, cuascn.F90:736, 792 (excluded numerics)
    rain_fall_coef: float = 21.18     # -, cuascn.F90:780 (fall-speed law)
    rain_fall_exp: float = 0.2        # -, cuascn.F90:780 (fall-speed law)


__param_spec__ = {
    "IFSAscentConfig": {
        "scheme_key": "atm.conv.IFSAscentConfig",
        "excluded": {
            "rmfcfl": "switch-defining constant of the implicit moist-correction branch; faithful-port source constant",
            "rmfcmin": "mass-flux numerics floor; iteration-coupled stability guard",
            "entr_org_cap": "numerics floor capping organized entrainment",
            "detr_early_cap": "numerics floor capping early detrainment",
            "buoy_reinterp_K": "buoyancy numerics threshold, iteration-coupled re-interpretation trigger",
            "buoy_entr_shutoff_K": "buoyancy numerics threshold for entrainment shutoff",
            "buoy_accept_K": "numerics floor for plume-termination acceptance test",
            "lapse_accept_K_per_m": "numerics floor of the lapse-rate acceptance test",
            "ke_buoy_factor": "source constant of a faithful port (ZFACBUO, cuascn.F90:282)",
            "ke_drag_cw": "source constant of a faithful port (Z_CWDRAG, cuascn.F90:288)",
            "ke_floor": "kinetic-energy numerics floor",
            "ke_ratio_floor": "kinetic-energy-ratio numerics floor",
            "ptu_min_K": "temperature clamp numerics floor",
            "ptu_max_K": "temperature clamp numerics floor",
            "cldmax": "numerics floor: cloud-water clip cap",
            "cwifrac": "unused in the liquid-only branch (ZALFAW = 1)",
            "cprc2": "source constant of a faithful port (Z_CPRC2, cuascn.F90:287)",
            "ke_wu_floor_conv": "numerics floor on ZWU inside the conversion term",
            "ke_wu_floor_fall": "numerics floor on ZWU inside the fallout term",
            "wu_max": "numerics floor on the updraught speed cap",
        },
        "params": {
            "entrorg": {
                "units": "1/m", "bounds": (1.0e-4, 6.0e-3),
                "tunable_tier": 1, "transform": "none",
                "category": "convection",
                "reference": "sucumf.F90:136", "shape": None,
            },
            "entr_rh": {
                "units": "-", "bounds": (0.5, 1.5),
                "tunable_tier": 2, "transform": "none",
                "category": "convection",
                "reference": "sucumf.F90:140", "shape": None,
            },
            "entshalp": {
                "units": "-", "bounds": (0.5, 4.0),
                "tunable_tier": 2, "transform": "none",
                "category": "convection",
                "reference": "sucumf.F90:144", "shape": None,
            },
            "detrpen": {
                "units": "1/m", "bounds": (1.0e-5, 3.0e-4),
                "tunable_tier": 1, "transform": "none",
                "category": "convection",
                "reference": "sucumf.F90:129", "shape": None,
            },
            "rh_detr_offset": {
                "units": "-", "bounds": (1.0, 2.5),
                "tunable_tier": 2, "transform": "none",
                "category": "convection",
                "reference": "cuascn.F90:514/672", "shape": None,
            },
            "rh_entr_offset": {
                "units": "-", "bounds": (0.0, 1.0),
                "tunable_tier": 2, "transform": "none",
                "category": "convection",
                "reference": "cuascn.F90:686", "shape": None,
            },
            "rprcon": {
                "units": "1/m", "bounds": (5.0e-4, 5.0e-3),
                "tunable_tier": 2, "transform": "none",
                "category": "convection",
                "reference": "sucumf.F90:175 (ZPRCDGW = RPRCON*RPLRG)",
                "shape": None,
            },
            "dnoprc": {
                "units": "kg/kg", "bounds": (1.0e-4, 1.0e-3),
                "tunable_tier": 2, "transform": "none",
                "category": "convection",
                "reference": "cuascn.F90:284 (ZDNOPRC)", "shape": None,
            },
            "rain_fall_coef": {
                "units": "m s-1 (kg/kg)^-0.2", "bounds": (10.0, 35.0),
                "tunable_tier": 2, "transform": "none",
                "category": "convection",
                "reference": "cuascn.F90:780 (ZVW = 21.18*PLRAIN**0.2)",
                "shape": None,
            },
            "rain_fall_exp": {
                "units": "-", "bounds": (0.1, 0.3),
                "tunable_tier": 2, "transform": "none",
                "category": "convection",
                "reference": "cuascn.F90:780 (ZVW exponent)", "shape": None,
            },
        },
    }
}


__physics_contract__ = {
    "summary": (
        "OpenIFS main updraught ascent (cuascn.F90 + cuentr.F90), liquid-only "
        "thermodynamics, deep (KTYPE 1) and shallow (KTYPE 2) convection; "
        "advances the plume one full level per scan step and emits the layer "
        "mass-flux, moisture, liquid and dry-static-energy fluxes. "
        "Surface-last arrays; IFS level JK maps to our j = JK-1; k_* are "
        "0-based surface-last level indices."
    ),
    "inputs": {
        "T": "K", "q": "kg kg-1", "qs": "kg kg-1",
        "p_full": "Pa", "p_half": "Pa",
        "geo_full": "m2 s-2", "geo_half": "m2 s-2",
        "T_h": "K", "q_h": "kg kg-1", "plitot": "kg kg-1",
        "ldcum": "1 (bool)", "ktype": "1 (1 deep, 2 shallow)",
        "k_dpl": "1 (level index)", "k_cbot": "1 (level index)",
        "M_b": "kg m-2 s-1", "w_base": "m s-1", "dt": "s",
    },
    "outputs": {
        "M": "kg m-2 s-1", "PMFUS": "W m-2", "PMFUQ": "kg m-2 s-1",
        "PMFUL": "kg m-2 s-1", "PLUDE": "kg m-2 s-1",
        "PDMFEN": "kg m-2 s-1", "PMFUDE_RATE": "kg m-2 s-1",
        "PDMFUP": "kg m-2 s-1", "PLRAIN": "kg kg-1", "PKINEU": "m2 s-2",
        "T_u": "K", "q_u": "kg kg-1", "l_u": "kg kg-1",
        "klab": "1 (0/1/2)", "k_ctop": "1 (level index)",
        "pwmean": "m2 s-2 Pa", "zdpmean": "Pa",
    },
    "sign_convention": 'Fluxes are positive upward (mass flux M > 0 for the updraught); entrainment is a positive source to the plume, detrainment a positive loss to the environment. Buoyancy acceptance thresholds are in K and negative (buoy_accept_K = -2 K). Arrays are surface-last; the IFS source loop ascends with decreasing JK, our lax.scan descends with increasing j (IFS JK <-> our j = JK-1).',
    "conserves": ["none"],
    "differentiable": False,
    "reference": 'OpenIFS cuascn.F90 (updraught ascent) and cuentr.F90:158-168 (turbulent exchange below cloud base); tunables default to sucumf.F90. Switches: LDTDKMF=.FALSE., LPHYLIN=.FALSE., LSCVLIQ=.TRUE., RMFSOLTQ=1 (RMFCFL=3).',
    "idealized_test": "tests/unit/test_ifs_ascent.py",
}


# --------------------------------------------------------------------------
# Additional module constants (YDTHF, suphec.F90 via sucumf.F90:272)
# --------------------------------------------------------------------------


def cuentr_turbulent_detrainment(M_below, dz, detrpen, in_cloud):
    """cuentr.F90:158-168, turbulent exchange below the cloud base.

    IFS symbols: ``ZDZ`` (layer geopotential thickness / g), ``ZMF =
    PMFU(JL,KK+1) * ZDZ``, ``LLO1 = KK < KCBOT``.  Entrainment ZENTR = 0
    in this branch, so E = 0 always; detrainment D = DETRPEN * ZMF on
    in-cloud layers, else 0.

    Args:
      M_below: PMFU at the half level below (surface side), (ncol,).
      dz: layer thickness ZDZ in m, (ncol,).
      detrpen: DETRPEN in 1/m, scalar.
      in_cloud: True where KK < KCBOT (full level inside the cloud),
        (ncol,) bool.

    Returns:
      (E, D): turbulent entrainment (0) and detrainment, (ncol,).
    """
    ZMF = M_below * dz
    E = jnp.zeros_like(ZMF)
    D = jnp.where(in_cloud, detrpen * ZMF, jnp.zeros_like(ZMF))
    return E, D


def layer_increments(E_org_prev, D_turb, M_below, rh_env, is_shallow,
                     dp_layer, dt, cfg):
    """cuascn.F90:485-520 organized/turbulent mass-flux increments, in order.

    IFS symbols: ZDMFEN, ZDMFDE, ZMFMAX, ZMFTEST, ZCHANGE, ZXS, ZXE;
    ZCONS2 = RMFCFL / (g * dt) (cuascn.F90:281).

    Steps:
      1. D = min(D_turb, 0.75 * M_below)                                   (:487)
      2. E = E_org_prev (organized entrainment diagnosed at the layer
         below; zero at the first layer above cloud base)                  (:503)
      3. shallow (KTYPE >= 2): E = ENTSHALP * E, D = E                     (:504-510)
      4. D = D * (1.6 - min(1, RH))                                        (:514)
      5. CFL redistribution (:499-520):
         ZMFMAX = dp * RMFCFL / (g dt); ZXS = max(M_below - ZMFMAX, 0);
         ZMFTEST = M_below + E - D; ZCHANGE = max(ZMFTEST - ZMFMAX, 0);
         ZXE = max(ZCHANGE - ZXS, 0); E -= ZXE; ZCHANGE -= ZXE; D += ZCHANGE.

    Args:
      E_org_prev: ZOENTR from the layer below, (ncol,).
      D_turb: turbulent detrainment from ``cuentr_turbulent_detrainment``.
      M_below: PMFU(JK+1), (ncol,).
      rh_env: PQEN/PQSEN at the current full level, (ncol,).
      is_shallow: KTYPE >= 2, (ncol,) bool.
      dp_layer: pressure thickness of the layer, Pa, (ncol,).
      dt: PTSPHY physics timestep, scalar.
      cfg: IFSAscentConfig.

    Returns:
      (E, D, M_new, PDMFEN) with M_new = M_below + E - D and
      PDMFEN = E - D (:522-523); all (ncol,).
    """
    rh1 = jnp.minimum(1.0, rh_env)

    # (:487) early detrainment cap
    D = jnp.minimum(D_turb, cfg.detr_early_cap * jnp.maximum(M_below, TINY))
    D = jnp.where(D_turb <= cfg.detr_early_cap * jnp.maximum(M_below, TINY),
                  D_turb, D)

    # (:503) organized entrainment from below
    E = E_org_prev

    # (:504-510) shallow convection: entr_shalp * E, D = E
    E = jnp.where(is_shallow, cfg.entshalp * E, E)
    D = jnp.where(is_shallow, E, D)

    # (:514) RH-modulated detrainment
    D = D * (cfg.rh_detr_offset - rh1)

    # (:499-520) CFL redistribution
    ZMFMAX = dp_layer * cfg.rmfcfl / (g * dt)
    ZXS = jnp.maximum(M_below - ZMFMAX, 0.0)
    ZMFTEST = M_below + E - D
    ZCHANGE = jnp.maximum(ZMFTEST - ZMFMAX, 0.0)
    ZXE = jnp.maximum(ZCHANGE - ZXS, 0.0)
    E = E - ZXE
    ZCHANGE = ZCHANGE - ZXE
    D = D + ZCHANGE

    M_new = M_below + E - D
    return E, D, M_new, E - D


def ke_step(K_below, E, D, M_below, buoy_mean_over_Tv, dphi, cfg):
    """cuascn.F90:647-668 kinetic-energy update (LDTDKMF=.FALSE. branch).

    IFS symbols: ZBUOC, ZDKBUO, ZDKEN, PKINEU.

      ZDKBUO = dphi * ZFACBUO * ZBUOC                              (:647-648)
        where dphi = (PGEOH(JK) - PGEOH(JK+1)) is the geopotential
        thickness of the layer and ZBUOC the buoyancy averaged over the
        two half levels divided by the environmental virtual temperature
        (``buoy_mean_over_Tv``, already divided by Tv by the caller).
      ZDKEN = min(1, (1 + Z_CWDRAG) * (E if ZDMFEN > 0 else D)
                 / max(RMFCMIN, M_below))                          (:656-661)
        the source branch ``IF (ZDMFEN > 0)`` is a selection between E
        and D, implemented with ``jnp.where`` on the signed net flux.
      K = max(ke_floor, (K_below * (1 - ZDKEN) + ZDKBUO) / (1 + ZDKEN))
                                                                  (:667)

    Args:
      K_below: PKINEU at the half level below, m2/s2, (ncol,).
      E, D: layer mass-flux increments (not net), (ncol,).
      M_below: PMFU(JK+1), (ncol,).
      buoy_mean_over_Tv: ZBUOC, K, (ncol,).
      dphi: geopotential thickness of the layer, m2/s2, (ncol,).
      cfg: IFSAscentConfig.

    Returns:
      K: updated kinetic energy, m2/s2, (ncol,).  Signed: a negative K is
      the plume-termination signal and is NEVER clipped to zero here.
    """
    ZDKBUO = dphi * cfg.ke_buoy_factor * buoy_mean_over_Tv

    ZDMFEN = E - D
    mixing_rate = jnp.where(ZDMFEN > 0.0, E, D)
    ZDKEN = jnp.minimum(
        1.0,
        (1.0 + cfg.ke_drag_cw) * mixing_rate
        / jnp.maximum(cfg.rmfcmin, jnp.maximum(M_below, TINY)),
    )

    K = (K_below * (1.0 - ZDKEN) + ZDKBUO) / (1.0 + ZDKEN)
    return jnp.maximum(cfg.ke_floor, K)


def organized_entrainment_next(M_new, rh_env_above, qs_env, qs_base,
                               dphi_next, buoy_K, cfg):
    """cuascn.F90:679-691 organized entrainment diagnosed for the NEXT layer.

    IFS symbols: ZOENTR, ZXENTRORG.

    Where the layer buoyancy ZBUO > -0.2 K (buoy_entr_shutoff_K)::

        ZOENTR = ENTRORG * (0.3 - (min(1, RH) - ENTR_RH))
                 * (PGEOH(JK-1) - PGEOH(JK)) / g
                 * min(1, PQSEN(JK) / PQSEN(IKB))**3
        ZOENTR = min(0.4, ZOENTR) * PMFU(JK)

    Indexing note: the source uses ``PQEN(JL,JK-1)`` and
    ``PGEOH(JL,JK-1)``.  IFS is top-first, so IFS level JK-1 is the full
    level *above* JK; equivalently our surface-last index k-1 is the level
    above k.  The relative humidity used here is therefore that of the
    level ABOVE the current one (``rh_env_above``), consistent with the
    look-ahead character of the diagnosis; dphi_next is the geopotential
    thickness of the layer above.  qs_env / qs_base enter as
    min(1, PQSEN(JK)/PQSEN(IKB))**3 with IKB = KCBOT.

    Args:
      M_new: PMFU at the current level, (ncol,).
      rh_env_above: PQEN(JK-1)/PQSEN(JK-1), (ncol,).
      qs_env: PQSEN at the current full level, (ncol,).
      qs_base: PQSEN at cloud base (IKB), (ncol,).
      dphi_next: geopotential thickness of the layer above, m2/s2, (ncol,).
      buoy_K: ZBUO at the current level, K, (ncol,).
      cfg: IFSAscentConfig.

    Returns:
      ZOENTR for the next layer, (ncol,); 0 where the buoyancy shutoff
      applies.
    """
    rh1 = jnp.minimum(1.0, rh_env_above)
    qs_ratio = jnp.minimum(
        1.0, qs_env / jnp.maximum(qs_base, TINY))
    ZOENTR = (
        cfg.entrorg * (cfg.rh_entr_offset - (rh1 - cfg.entr_rh))
        * dphi_next / g * qs_ratio**3
    )
    ZOENTR = jnp.minimum(cfg.entr_org_cap, ZOENTR) * M_new
    return jnp.where(
        buoy_K > cfg.buoy_entr_shutoff_K,
        ZOENTR,
        jnp.zeros_like(ZOENTR),
    )


def organized_detrainment(D, M_below, K, K_below, rh_env, cfg):
    """cuascn.F90:669-676 negative-buoyancy organized detrainment.

    IFS symbols: ZKEDKE, ZOCUDET, ZMFUN.  Where the layer-mean buoyancy
    ZBUOC < 0::

        ZKEDKE  = clip(K / max(1e-3, K_below), 0, 1)
        ZOCUDET = 1.6 - min(1, RH)
        ZMFUN   = ZOCUDET * sqrt(ZKEDKE)
        D       = max(D, M_below * (1 - ZMFUN))

    The sqrt argument ZKEDKE is non-negative by construction (ratio of a
    floored positive denominator; the numerator is clipped inside the
    selected branch).  The caller afterwards recomputes PLUDE =
    PLU_below * D and M = M_below + E - D.

    Args:
      D: current detrainment, (ncol,).
      M_below: PMFU(JK+1), (ncol,).
      K: PKINEU at the current level (from ``ke_step``), (ncol,).
      K_below: PKINEU at the half level below, (ncol,).
      rh_env: PQEN/PQSEN at the current full level, (ncol,).
      cfg: IFSAscentConfig.

    Returns:
      Updated detrainment D, (ncol,).  Where ZBUOC >= 0 the input D is
      returned unchanged; the caller supplies the sign of ZBUOC through
      the selection performed on the returned array's twin (see driver);
      here the negative-buoyancy update is gated by the caller passing a
      where-selected K so that this helper stays pure per layer.

      NOTE: this helper expects the caller to have already applied the
      ``ZBUOC < 0`` gate to ``K`` (K is passed as the where-selected,
      in-branch value); the sqrt dummy ``ZKEDKE`` is floored to >= 0
      before sqrt on the evaluated branch.
    """
    rh1 = jnp.minimum(1.0, rh_env)
    ZKEDKE = K / jnp.maximum(cfg.ke_ratio_floor, jnp.maximum(K_below, TINY))
    ZKEDKE = jnp.clip(ZKEDKE, 0.0, 1.0)
    ZOCUDET = cfg.rh_detr_offset - rh1
    ZMFUN = ZOCUDET * jnp.sqrt(jnp.maximum(ZKEDKE, 0.0))
    D_new = jnp.maximum(D, M_below * (1.0 - ZMFUN))
    return D_new


# ==========================================================================
# ### REPLACE IFSAscentConfig (A2: adds precipitation-formation constants)
# ==========================================================================
_RTBERCU = T_freeze - 5.0    # K, RTBERCU = RTT - 5  (Bergeron-Findeisen onset)
_RTICECU = T_freeze - 38.0   # K, RTICECU = RTT - 38 (mixed-phase range end)
_ZWU_DENOM = 0.75   # -, cuascn.F90:734 ZPRCON denominator, ZWU floor in the conversion term
_ZZCO_LIQ = 1.3     # -, cuascn.F90:735-736 ZZCO = 1 + 0.3*FOEALFCU, FOEALFCU = 1 (liquid-only)


# --------------------------------------------------------------------------
# Scan body: one parent layer j (surface-last; IFS JK <-> j = JK-1)
# --------------------------------------------------------------------------
class _AscentCarry(NamedTuple):
    M: jnp.ndarray            # PMFU
    PMFUS: jnp.ndarray
    PMFUQ: jnp.ndarray
    PMFUL: jnp.ndarray
    T_u: jnp.ndarray
    q_u: jnp.ndarray
    l_u: jnp.ndarray
    K: jnp.ndarray            # PKINEU
    PLRAIN: jnp.ndarray
    PDMFUP: jnp.ndarray
    PLUDE: jnp.ndarray
    PMFUDE_RATE: jnp.ndarray
    PDMFEN: jnp.ndarray
    klab: jnp.ndarray
    k_ctop: jnp.ndarray
    ZOENTR: jnp.ndarray       # (ncol,) organized entrainment for next layer
    ZBUO: jnp.ndarray         # (ncol, nlev) buoyancy profile
    ZLUOLD: jnp.ndarray       # (ncol,) pre-condensation condensate
    pwmean: jnp.ndarray       # (ncol,)
    zdpmean: jnp.ndarray      # (ncol,)



def _ascent_layer(carry, j, T, q, qs, p_full, p_half, geo_full, geo_half,
                  T_h, q_h, plitot, ldcum, ktype, k_dpl, k_cbot, qs_base,
                  dt, cfg):
    """One cuascn layer for all columns, cuascn.F90:415-816 (KTYPE 1/2 only).

    ``j`` is the surface-last full-level index; the IFS half level below
    (surface side) is ``j+1``.  All writes are masked by
    ``flag = ldcum & (klab[j+1] == 2)``; no Python control flow on traced
    values.
    """
    ncol = T.shape[0]
    ar = jnp.arange(ncol)

    flag = ldcum & (carry.klab[:, j + 1] == 2)
    below = j > k_dpl                       # below the departure level
    in_cloud = j < k_cbot

    # -- cuentr turbulent detrainment (helper 1) -------------------------
    dphi = geo_half[:, j] - geo_half[:, j + 1]           # m2/s2, > 0
    dz = dphi / g
    E_turb, D_turb = cuentr_turbulent_detrainment(
        carry.M[:, j + 1], dz, cfg.detrpen, in_cloud)

    # -- mass-flux increments (helper 2) ---------------------------------
    rh_env = q[:, j] / jnp.maximum(qs[:, j], TINY)
    # cuascn.F90:499: ZMFMAX uses PAPH(JK)-PAPH(JK-1)  <->  our
    # p_half[j] - p_half[j-1]  (IFS half JK <-> j, JK-1 <-> j-1).
    dp_layer = jnp.maximum(p_half[:, j] - p_half[:, j - 1], 0.0)
    E, D, M_step, pdmfen_j = layer_increments(
        carry.ZOENTR, D_turb, carry.M[:, j + 1], rh_env,
        ktype >= 2, dp_layer, dt, cfg)

    M_new = M_step

    # -- closure mean-velocity accumulators (cuascn.F90:501-502) ---------
    acc = flag & in_cloud
    dpf = p_full[:, j + 1] - p_full[:, j]
    pwmean = carry.pwmean + jnp.where(acc, carry.K[:, j + 1] * dpf, 0.0)
    zdpmean = carry.zdpmean + jnp.where(acc, dpf, 0.0)

    # -- transport (cuascn.F90:522-549) ----------------------------------
    ZQEEN = q_h[:, j + 1] * E
    ZSEEN = (c_pd * T_h[:, j + 1] + geo_half[:, j + 1]) * E
    ZLEEN = jnp.where(plitot[:, j] > RLMIN, plitot[:, j] * E, 0.0)
    ZSCDE = (c_pd * carry.T_u[:, j + 1] + geo_half[:, j + 1]) * D
    ZQUDE = carry.q_u[:, j + 1] * D
    PLUDE_j = carry.l_u[:, j + 1] * D

    ZMFUSK = carry.PMFUS[:, j + 1] + ZSEEN - ZSCDE
    ZMFUQK = carry.PMFUQ[:, j + 1] + ZQEEN - ZQUDE
    ZMFULK = carry.PMFUL[:, j + 1] + ZLEEN - PLUDE_j
    ZFAC = 1.0 / jnp.maximum(cfg.rmfcmin, M_new)
    l_u_j = ZMFULK * ZFAC
    q_u_j = ZMFUQK * ZFAC
    T_u_j = jnp.clip(
        (ZMFUSK * ZFAC - geo_half[:, j]) / c_pd,
        cfg.ptu_min_K, cfg.ptu_max_K)
    ZQOLD = q_u_j
    PLRAIN_j = carry.PLRAIN[:, j + 1] * jnp.maximum(
        0.0, carry.M[:, j + 1] - D) * ZFAC
    ZLUOLD = l_u_j

    # reset to the environment below the departure level (:544-549)
    reset = flag & below
    T_u_j = jnp.where(reset, T_h[:, j], T_u_j)
    q_u_j = jnp.where(reset, q_h[:, j], q_u_j)
    l_u_j = jnp.where(reset, 0.0, l_u_j)
    ZLUOLD = jnp.where(reset, 0.0, ZLUOLD)

    # -- moist correction (CUADJTQ, KCALL = 1) at p_half[j] --------------
    T_adj, q_adj = cuadjtq_condense(T_u_j, q_u_j, p_half[:, j])
    cond = flag & (q_adj < ZQOLD - TINY)
    T_u_j = jnp.where(cond, T_adj, T_u_j)
    q_u_j = jnp.where(cond, q_adj, q_u_j)
    l_u_j = l_u_j + jnp.where(cond, ZQOLD - q_u_j, 0.0)

    # -- buoyancy (:607-646) ---------------------------------------------
    ZBC = T_u_j * (1.0 + _RETV * q_u_j
                   - carry.l_u[:, j + 1] - carry.PLRAIN[:, j + 1])
    T_h_j = T_h[:, j]
    q_h_j = q_h[:, j]
    ZBE = T_h_j * (1.0 + _RETV * q_h_j)
    ZBUO_j = ZBC - ZBE
    # re-interpolation (:637-643); PLGLAC = 0 (liquid-only departure)
    reinterp = cond & (ZBUO_j < cfg.buoy_reinterp_K)
    T_h_ri = 0.5 * (T[:, j] + T[:, j - 1])
    q_h_ri = 0.5 * (q[:, j] + q[:, j - 1])
    T_h_j = jnp.where(reinterp, T_h_ri, T_h_j)
    q_h_j = jnp.where(reinterp, q_h_ri, q_h_j)
    ZBUO_j = jnp.where(reinterp,
                       ZBC - T_h_j * (1.0 + _RETV * q_h_j), ZBUO_j)
    ZBUO_j = jnp.where(cond, ZBUO_j, jnp.zeros_like(ZBUO_j))
    ZBUOC = 0.5 * (ZBUO_j + carry.ZBUO[:, j + 1]) / (
        T_h_j * (1.0 + _RETV * q_h_j))

    # -- kinetic energy (helper 3) ---------------------------------------
    K_j = ke_step(carry.K[:, j + 1], E, D, carry.M[:, j + 1], ZBUOC,
                  dphi, cfg)
    K_j = jnp.where(cond, K_j, carry.K[:, j + 1])

    # -- organized detrainment on negative layer buoyancy (:669-676) -----
    negb = cond & (ZBUOC < 0.0)
    K_gated = jnp.where(negb, K_j, jnp.ones_like(K_j))  # positive dummy
    D_org = organized_detrainment(D, carry.M[:, j + 1], K_gated,
                                  carry.K[:, j + 1], rh_env, cfg)
    D = jnp.where(negb, D_org, D)
    PLUDE_j = jnp.where(negb, carry.l_u[:, j + 1] * D, PLUDE_j)
    M_new = jnp.where(negb, carry.M[:, j + 1] + E - D, M_new)

    # -- organized entrainment for the NEXT layer (:679-691) -------------
    rh_above = q[:, j - 1] / jnp.maximum(qs[:, j - 1], TINY)
    dphi_next = geo_half[:, j - 1] - geo_half[:, j]
    ZOENTR_next = organized_entrainment_next(
        M_new, rh_above, qs[:, j], qs_base, dphi_next, ZBUO_j, cfg)
    ZOENTR_new = jnp.where(
        cond & (ZBUO_j > cfg.buoy_entr_shutoff_K),
        ZOENTR_next, jnp.zeros_like(ZOENTR_next))

    # -- erase below the departure level (:694-697) ----------------------
    erase = flag & below
    M_new = jnp.where(erase, carry.M[:, j + 1], M_new)
    K_j = jnp.where(erase, 0.5, K_j)

    # -- acceptance (:698-715) --------------------------------------------
    lapse = (T[:, j - 1] - T[:, j]) / (
        jnp.maximum(geo_full[:, j - 1] - geo_full[:, j], TINY) / g)
    accept = ((K_j > 0.0) & (M_new > 0.0)
              & ((ZBUO_j > cfg.buoy_accept_K)
                 | (lapse < cfg.lapse_accept_K_per_m)))
    llo1 = cond & accept
    not_accept = cond & ~accept
    klab_j = jnp.where(llo1, 2, 0).astype(carry.klab.dtype)
    M_new = jnp.where(not_accept, 0.0, M_new)
    K_j = jnp.where(not_accept, 0.0, K_j)
    D = jnp.where(not_accept, carry.M[:, j + 1], D)
    PLUDE_j = jnp.where(not_accept, carry.l_u[:, j + 1] * D, PLUDE_j)

    # -- no-condensation stop for KTYPE <= 2 (:723-728) ------------------
    nocond = flag & ~cond
    klab_j = jnp.where(nocond, 0, klab_j).astype(carry.klab.dtype)
    M_new = jnp.where(nocond, 0.0, M_new)
    K_j = jnp.where(nocond, 0.0, K_j)
    D = jnp.where(nocond, carry.M[:, j + 1], D)
    PLUDE_j = jnp.where(nocond, carry.l_u[:, j + 1] * D, PLUDE_j)
    # unflagged columns: the source only clears KLAB(JK) where KLAB(JK+1) == 0
    # (cuascn.F90:437); every other label is left untouched
    klab_j = jnp.where(flag, klab_j,
                       jnp.where(carry.klab[:, j + 1] == 0, 0, carry.klab[:, j])
                       ).astype(carry.klab.dtype)

    # -- detrainment-rate store (:718-720) --------------------------------
    rate_set = (cond & (carry.M[:, j + 1] > 0.0)) | nocond
    PMFUDE_RATE_j = jnp.where(rate_set, D, carry.PMFUDE_RATE[:, j])

    # -- precipitation formation (:733-758) --------------------------------
    # Liquid-only: FOEALFCU = 1 => ZZCO = 1 + 0.3*1 = 1.3 (LDTDKMF=.FALSE.)
    prc = llo1 & (l_u_j > cfg.dnoprc)
    l_u_safe = jnp.maximum(l_u_j, TINY)
    ZWU = jnp.minimum(cfg.wu_max,
                      jnp.sqrt(2.0 * jnp.maximum(cfg.ke_wu_floor_conv,
                                                  carry.K[:, j + 1])))
    ZPRCON = (cfg.rprcon / g) / (_ZWU_DENOM * ZWU) * _ZZCO_LIQ
    ZDT = jnp.minimum(_RTBERCU - _RTICECU,
                      jnp.maximum(_RTBERCU - T_u_j, 0.0))
    ZCBF = 1.0 + cfg.cprc2 * jnp.sqrt(ZDT)
    ZZCO2 = ZPRCON * ZCBF
    ZLCRIT = cfg.dnoprc / ZCBF
    ZC = l_u_j - ZLUOLD
    ZD = ZZCO2 * (1.0 - jnp.exp(-(l_u_safe / ZLCRIT) ** 2)) * dphi
    ZINT = jnp.exp(-jnp.clip(ZD, 0.0, 50.0))  # coeff-ok: exp() overflow guard on a dimensionless exponent
    ZLNEW = jnp.where(
        ZD > TINY,
        ZLUOLD * ZINT + ZC / jnp.maximum(ZD, TINY) * (1.0 - ZINT),
        ZC)  # series limit ZD -> 0
    ZLNEW = jnp.clip(ZLNEW, 0.0, jnp.minimum(l_u_j, cfg.cldmax))
    ZPRECIP = jnp.maximum(0.0, ZLUOLD + ZC - ZLNEW)
    PDMFUP_j = jnp.where(prc, ZPRECIP * M_new, carry.PDMFUP[:, j])
    PLRAIN_j = PLRAIN_j + jnp.where(prc, ZPRECIP, 0.0)
    l_u_j = jnp.where(prc, ZLNEW, l_u_j)

    # -- rain fallout (:777-806, non-LPHYLIN, liquid: ZALFAW = 1) ---------
    fall = llo1 & (PLRAIN_j > 0.0)
    ZVW = cfg.rain_fall_coef * jnp.maximum(PLRAIN_j, TINY) ** cfg.rain_fall_exp
    ZVV = ZVW                      # ZALFAW = 1 (liquid-only)
    ZROLD = PLRAIN_j - ZPRECIP
    ZWU2 = jnp.minimum(cfg.wu_max,
                       jnp.sqrt(2.0 * jnp.maximum(cfg.ke_wu_floor_fall, K_j)))
    ZDr = ZVV / jnp.maximum(ZWU2, TINY)
    ZINTr = jnp.exp(-jnp.clip(ZDr, 0.0, 50.0))  # coeff-ok: exp() overflow guard, as the ZINT site above
    ZRNEW = jnp.where(
        ZDr > TINY,
        ZROLD * ZINTr + ZC / jnp.maximum(ZDr, TINY) * (1.0 - ZINTr),
        ZROLD + ZC)
    ZRNEW = jnp.clip(ZRNEW, 0.0, jnp.maximum(PLRAIN_j, 0.0))
    PLRAIN_j = jnp.where(fall, ZRNEW, PLRAIN_j)

    # -- fluxes (:808-816) --------------------------------------------------
    PMFUL_j = jnp.where(flag, l_u_j * M_new, carry.PMFUL[:, j])
    PMFUS_j = jnp.where(flag, (c_pd * T_u_j + geo_half[:, j]) * M_new,
                        carry.PMFUS[:, j])
    PMFUQ_j = jnp.where(flag, q_u_j * M_new, carry.PMFUQ[:, j])
    PDMFEN_j = jnp.where(flag, pdmfen_j, carry.PDMFEN[:, j])

    # -- assemble the new carry ---------------------------------------------
    new = _AscentCarry(
        # unflagged columns keep PMFU (the source only touches JLX columns)
        M=carry.M.at[:, j].set(jnp.where(flag, M_new, carry.M[:, j])),
        PMFUS=carry.PMFUS.at[:, j].set(PMFUS_j),
        PMFUQ=carry.PMFUQ.at[:, j].set(PMFUQ_j),
        PMFUL=carry.PMFUL.at[:, j].set(PMFUL_j),
        T_u=carry.T_u.at[:, j].set(jnp.where(flag, T_u_j,
                                             carry.T_u[:, j])),
        q_u=carry.q_u.at[:, j].set(jnp.where(flag, q_u_j,
                                             carry.q_u[:, j])),
        l_u=carry.l_u.at[:, j].set(jnp.where(flag, l_u_j,
                                             carry.l_u[:, j])),
        K=carry.K.at[:, j].set(jnp.where(flag, K_j, carry.K[:, j])),
        PLRAIN=carry.PLRAIN.at[:, j].set(jnp.where(flag, PLRAIN_j,
                                                   carry.PLRAIN[:, j])),
        PDMFUP=carry.PDMFUP.at[:, j].set(jnp.where(flag, PDMFUP_j,
                                                   carry.PDMFUP[:, j])),
        PLUDE=carry.PLUDE.at[:, j].set(jnp.where(flag, PLUDE_j,
                                                 carry.PLUDE[:, j])),
        PMFUDE_RATE=carry.PMFUDE_RATE.at[:, j].set(
            jnp.where(flag, PMFUDE_RATE_j, carry.PMFUDE_RATE[:, j])),
        PDMFEN=carry.PDMFEN.at[:, j].set(PDMFEN_j),
        klab=carry.klab.at[:, j].set(klab_j),
        k_ctop=jnp.where(llo1, jnp.full_like(carry.k_ctop, j),
                         carry.k_ctop),
        ZOENTR=ZOENTR_new,
        ZBUO=carry.ZBUO.at[:, j].set(jnp.where(flag, ZBUO_j, carry.ZBUO[:, j])),
        ZLUOLD=jnp.where(flag, ZLUOLD, carry.ZLUOLD),
        pwmean=pwmean,
        zdpmean=zdpmean,
    )
    return new, None


class UpdraughtAscent(NamedTuple):
    """Result of the main updraught ascent (specific humidity inside)."""
    M: jnp.ndarray             # PMFU, kg m-2 s-1
    PMFUS: jnp.ndarray         # dry static energy flux
    PMFUQ: jnp.ndarray         # specific-humidity flux
    PMFUL: jnp.ndarray         # liquid-water flux
    PLUDE: jnp.ndarray         # detrained liquid
    PDMFEN: jnp.ndarray        # net env -> plume mass-flux source
    PMFUDE_RATE: jnp.ndarray   # plume -> env detrainment rate
    PDMFUP: jnp.ndarray        # precipitation flux source
    PLRAIN: jnp.ndarray        # in-plume rain water
    PKINEU: jnp.ndarray        # updraught kinetic energy
    T_u: jnp.ndarray
    q_u: jnp.ndarray
    l_u: jnp.ndarray
    klab: jnp.ndarray
    k_ctop: jnp.ndarray
    pwmean: jnp.ndarray        # closure mean-velocity numerator
    zdpmean: jnp.ndarray       # closure mean-velocity denominator


def ifs_updraught_ascent(T, q, qs, p_full, p_half, geo_full, geo_half,
                         T_h, q_h, plitot, ldcum, ktype, k_dpl, k_cbot,
                         klab0, T_u0, q_u0, l_u0, w_base, M_b, dt, cfg):
    """Driver for the cuascn main updraught ascent.

    Initialisation follows cuascn.F90:357-405: at the cloud base
    IKB = k_cbot, K = 0.5*w_base^2, M = M_b and the base fluxes from the
    test-ascent (T_u0, q_u0, l_u0) profiles; the scan then runs
    j = nlev-2 .. 2 (IFS JK = KLEV-1 .. 3, cuascn.F90:415), each layer
    masked by ``ldcum & (klab[j+1] == 2)``.

    Humidity variables are SPECIFIC humidity throughout; the caller
    converts any external moisture variable.
    """
    ncol, nlev = T.shape
    dtype = T.dtype
    ar = jnp.arange(ncol)
    zeros = lambda: jnp.zeros((ncol, nlev), dtype=dtype)

    qs_base = jnp.take_along_axis(qs, k_cbot[:, None], axis=1)[:, 0]

    # base initialisation only where LDCUM (cuascn.F90:397-405)
    ld = jnp.asarray(ldcum, bool)
    M = zeros().at[ar, k_cbot].set(jnp.where(ld, M_b, 0.0))
    PMFUS = zeros()
    PMFUQ = zeros()
    PMFUL = zeros()
    K = zeros().at[ar, k_cbot].set(jnp.where(ld, 0.5 * w_base**2, 0.0))
    PLRAIN = zeros()
    PDMFUP = zeros()
    PLUDE = zeros()
    PMFUDE_RATE = zeros()
    PDMFEN = zeros()
    klab = klab0.astype(jnp.int32)
    k_ctop = k_cbot.astype(jnp.int32)

    # base fluxes from the test-ascent cloud-base values
    geo_h_b = jnp.take_along_axis(geo_half, k_cbot[:, None], axis=1)[:, 0]
    M_b_ld = jnp.where(ld, M_b, 0.0)
    PMFUS = PMFUS.at[ar, k_cbot].set(
        M_b_ld * (c_pd * jnp.take_along_axis(T_u0, k_cbot[:, None], 1)[:, 0]
                  + geo_h_b))
    PMFUQ = PMFUQ.at[ar, k_cbot].set(
        M_b_ld * jnp.take_along_axis(q_u0, k_cbot[:, None], 1)[:, 0])
    PMFUL = PMFUL.at[ar, k_cbot].set(
        M_b_ld * jnp.take_along_axis(l_u0, k_cbot[:, None], 1)[:, 0])

    carry = _AscentCarry(
        M=M, PMFUS=PMFUS, PMFUQ=PMFUQ, PMFUL=PMFUL,
        T_u=jnp.array(T_u0, dtype=dtype),
        q_u=jnp.array(q_u0, dtype=dtype),
        l_u=jnp.array(l_u0, dtype=dtype),
        K=K, PLRAIN=PLRAIN, PDMFUP=PDMFUP, PLUDE=PLUDE,
        PMFUDE_RATE=PMFUDE_RATE, PDMFEN=PDMFEN,
        klab=klab, k_ctop=k_ctop,
        ZOENTR=jnp.zeros(ncol, dtype=dtype),
        ZBUO=zeros(),
        ZLUOLD=jnp.zeros(ncol, dtype=dtype),
        pwmean=jnp.zeros(ncol, dtype=dtype),
        zdpmean=jnp.zeros(ncol, dtype=dtype),
    )

    # IFS loop JK = KLEV-1 .. 3  <->  our j = nlev-2 .. 2 (descending)
    js = jnp.arange(nlev - 2, 1, -1, dtype=jnp.int32)

    def step(c, j):
        return _ascent_layer(c, j, T, q, qs, p_full, p_half, geo_full,
                             geo_half, T_h, q_h, plitot, ldcum, ktype,
                             k_dpl, k_cbot, qs_base, dt, cfg)

    carry, _ = lax.scan(step, carry, xs=js)

    return UpdraughtAscent(
        M=carry.M, PMFUS=carry.PMFUS, PMFUQ=carry.PMFUQ, PMFUL=carry.PMFUL,
        PLUDE=carry.PLUDE, PDMFEN=carry.PDMFEN,
        PMFUDE_RATE=carry.PMFUDE_RATE, PDMFUP=carry.PDMFUP,
        PLRAIN=carry.PLRAIN, PKINEU=carry.K,
        T_u=carry.T_u, q_u=carry.q_u, l_u=carry.l_u,
        klab=carry.klab, k_ctop=carry.k_ctop,
        pwmean=carry.pwmean, zdpmean=carry.zdpmean,
    )
