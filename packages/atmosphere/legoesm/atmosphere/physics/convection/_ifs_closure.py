"""IFS cumulus closure and profile scaling, ported from OpenIFS cumastrn.F90
(source main 8f6f722), branch Earth, deterministic configuration
(SPP / LLPERT_RTAU off).

Ported sections
---------------
* first_guess_mass_flux      cumastrn.F90:540-592   (initial cloud-base mass flux)
* subcloud_mse_supply        cumastrn.F90:484-496   (ZDHPBL integral)
* deep_cape_closure          cumastrn.F90:715-866   (CAPE/heat closure, KTYPE=1)
* shallow_closure            cumastrn.F90:880-916   (sub-cloud MSE closure, KTYPE=2)
* scale_profile              cumastrn.F90:955-1018  (final closure = scaling)

Conventions (as _ifs_ascent.py / _ifs_tendencies.py)
---------------------------------------------------
* surface-last level arrays; IFS half/full level JK maps to our index j=JK-1,
  the trigger/ascent modules already return k_cbot as OUR half index of the
  cloud-base interface (IFS IKB <-> k_cbot), so PAPH(IKB) is p_half[k_cbot]
  and PAPH(IKB-1) is p_half[k_cbot-1].  Hence
  ZMFMAX = (PAPH(IKB)-PAPH(IKB-1))*RMFCFL/(RG*PTSPHY)
        = (p_half[k_cbot] - p_half[k_cbot-1]) * RMFCFL / (g * dt).
* specific humidity (not mixing ratio); fixed source coefficients appear as
  module constants with sucumf.F90 provenance comments; tunables/switches live
  in IFSClosureConfig.
* KTYPE restricted to 1 (deep) and 2 (shallow); KTYPE=3 branch not ported.

IMPORTANT source behaviour
--------------------------
The source runs the entraining ascent (CUASC) ONCE with the first-guess
mass flux ZMFUB and then only RESCALES the resulting flux profiles by
ZMFS = ZMFUB1/max(RMFCMIN,ZMFUB) (:963) -- there is no second ascent.

The downdraught (cudlfsn/cuddrafn) is out of scope: PMFD = 0 everywhere,
so the ZHEAT integrand uses PMFU only, and ZEPS (shallow downdraught
ratio, :884-889) is zero.

Deviations from the letter of the source (documented limitations):
* RCAPDCYCL==2 ocean branch (cumastrn.F90:820-825) needs the environmental
  winds PUEN/PVEN, which are not part of this module's contract; only the
  land branch ZCAPDCYCL = ZCAPPBL*ZTAU/ZTAURES (:818-819) is applied,
  weighted by land_frac.
* The `PVERVEL < -100` alternative of the ZSATFR test (:846) needs the
  subsidence velocity, not in the contract; only ZSATFR <= 0.94 is used.
* PLUDELI (ice/liquid split of detrainment, :1012-1013) is not rescaled
  (not part of the ascent-output contract).
"""


from typing import NamedTuple

import jax.numpy as jnp

from legoesm import constants as C

from .mass_flux import ifs_ztaures

# ---------------------------------------------------------------------------
# Fixed source constants (sucumf.F90, main 8f6f722) -- do not tune here.
# ---------------------------------------------------------------------------
RTAUA = 1.0          # sucumf.F90:193  '' RTAUA = 1 s-1
RCAPDCYCL = 2.0      # sucumf.F90:234  (default; config switch)
RCAPQADV = 0.8       # sucumf.F90:237  (default; config switch)
RMINCAPE = 0.05      # sucumf.F90:239  fraction of CAPE always adjusted
RMFLIA = 2.0         # sucumf.F90:241  RMFLIA=2.0*RPLDARE (RPLDARE=1)
RMFCFL = 3.0         # sucumf.F90:248  implicit scheme CFL factor
RMFCMIN = 1.0e-8     # sucumf.F90 (RMFCMIN), floor in ZMFS

# Physical constants / derived quantities used verbatim in the source
RG = C.g                      # RG
RCPD = C.c_pd                 # RCPD
RLVTT = C.L_v                 # RLVTT (latent heat of vaporisation)
RETV = 1.0 / C.epsilon - 1.0  # RETV = RV/RD - 1
ZORCPD = 1.0 / RCPD           # cumastrn.F90:411  ZORCPD=1.0/RCPD (PGEO is already
                              # a geopotential difference -- no factor of RG)

_EPS = 1.0e-20   # AD-safe denominator floor (numerics, not physics)

# ---------------------------------------------------------------------------
# Additional fixed source constants (cumastrn.F90, main 8f6f722) -- do not
# tune here; verbatim source literals hoisted out of function bodies.
# ---------------------------------------------------------------------------
ZMFUB_FRAC_OF_ZMFMAX = 0.1   # cumastrn.F90:560 ZMFUB=ZMFMAX*0.1 (deep), :580 (shallow fallback)
ZDQMIN_FRAC = 0.01           # cumastrn.F90:567 ZDQMIN=MAX(0.01*ZQENH,...)
ZDQMIN_ABS_FLOOR = 1.0e-10   # cumastrn.F90:567 ...MAX(...,1.E-10)
ZDH_DQMIN_SCALE = 1.0e5      # cumastrn.F90:569 ZDH=RG*MAX(ZDH,1.E5*ZDQMIN)
ZDPL_SURF_PDIFF_MAX = 50.0e2   # cumastrn.F90:813 surface-to-departure-level depth limit (Pa)
ZCAPE_MAX = 5000.0           # cumastrn.F90:858 ZCAPE=MIN(ZCAPE,5000.0)
ZHEAT_MIN = 1.0e-4           # cumastrn.F90:859 ZHEAT=MAX(1.E-4,ZHEAT)
ZXTAU_MIN = 3.6e3 / 5.0      # cumastrn.F90:861 ZXTAU=MAX(3.6E3/5.0,...) (PPLRG*PPLDARE=1)
ZXTAU_MAX = 3.0 * 3.6e3      # cumastrn.F90:861 ...MIN(3.0*3.6E3,ZXTAU))
ZMFUB1_MIN = 0.001           # cumastrn.F90:863 ZMFUB1=MAX(ZMFUB1,0.001)
ZTAURES_DQCV_FLOOR = 1.25    # cumastrn.F90:853 .../MAX(1.25,ZTAURES)
ZDH_FLOOR_FRAC_RCPD = 0.1    # cumastrn.F90:899 ZDH=RG*MAX(ZDH,0.1*RCPD)
ZMFUB1_WEAK_FRAC = 0.9       # cumastrn.F90:897 ZMFUB1>0.9*ZMFMAX/RMFCFL
ZDH_WEAK_FRAC = 0.5          # cumastrn.F90:897 ...AND.ZDH<0.5*RG*RCPD
ZDH_WEAK_FLOOR_FRAC = 0.75   # cumastrn.F90:900 ZDH=0.75*RCPD*RG
ZSATFR_MAX = 0.94            # cumastrn.F90:846 IF(ZSATFR<=0.94 .OR. PVERVEL<-100)
ZTAU_W_OFFSET = 2.0          # cumastrn.F90:803 (2.0+MIN(15.,PWMEAN)) in ZTAU
ZTAU_W_MAX = 15.0            # cumastrn.F90:803 MIN(15.0_JPRB,PWMEAN(JL))
PWMEAN_MIN = 1.0e-2          # cuascn.F90:863 MAX(1.E-2,PWMEAN/MAX(1.,ZDPMEAN))
ZDPMEAN_MIN = 1.0            # cuascn.F90:863 MAX(1.0_JPRB,ZDPMEAN(JL))


def _f(x) -> jnp.ndarray:
    """Guard zero denominators (AD-safe floor); keeps the input dtype."""
    x = jnp.asarray(x)
    return jnp.maximum(x, jnp.asarray(_EPS, x.dtype))


class IFSClosureConfig(NamedTuple):
    """Tunables and switches of the cumastrn closure (sucumf.F90)."""
    lmfwstar: bool = False       # sucumf.F90:232  LMFWSTAR=.FALSE. (Grant w*)
    rcapdcycl: float = RCAPDCYCL  # sucumf.F90:234  land/PBL diurnal-cycle corr.
    rcapqadv: float = RCAPQADV    # sucumf.F90:237  moisture advection term
    rmincape: float = RMINCAPE    # sucumf.F90:239
    rtaua: float = RTAUA          # sucumf.F90:193
    rmfcfl: float = RMFCFL        # sucumf.F90:248
    rmfcmin: float = RMFCMIN
    rmflia: float = RMFLIA        # sucumf.F90:241
    # NJKT2 threshold (sucumf.F90:280-285): the closure integrals start at
    # NJKT2, the highest level whose reference (standard-atmosphere) pressure
    # exceeds 60 hPa.  Kept in the source's pressure convention (Pa) and
    # derived per column from p_full; excluded from calibration.
    njkt2_pa: float = 60.0e2


__param_spec__ = {
    "IFSClosureConfig": {
        "scheme_key": "atm.conv.IFSClosureConfig",
        "excluded": {
            "rcapdcycl": "sucumf.F90:234 source constant (default 2.0) of a faithful OpenIFS port",
            "rcapqadv": "sucumf.F90:237 source constant (default 0.8) of a faithful OpenIFS port",
            "rmincape": "sucumf.F90:239 source constant (default 0.05) of a faithful OpenIFS port",
            "rtaua": "sucumf.F90:193 source constant (default 1.0 s-1) of a faithful OpenIFS port",
            "rmfcfl": "sucumf.F90:248 source constant (default 3.0) of a faithful OpenIFS port",
            "rmfcmin": "sucumf.F90 source constant (default 1e-8) of a faithful OpenIFS port",
            "rmflia": "sucumf.F90:241 source constant (default 2.0 = 2*RPLDARE) of a faithful OpenIFS port",
            "njkt2_pa": "sucumf.F90:280-285 source constant (60 hPa) defining the NJKT2 integral window; a vertical-grid property, not a tunable",
        },
        "params": {},  # every field is a source constant/switch of a faithful port -- nothing is genuinely tunable
    },
}



__physics_contract__ = {
    "summary": (
        "cumastrn closure: first-guess cloud-base mass flux, deep CAPE "
        "closure (KTYPE=1) and shallow sub-cloud MSE closure (KTYPE=2), then "
        "a single rescaling of the once-computed updraught flux profiles by "
        "ZMFS = ZMFUB1/max(RMFCMIN, ZMFUB). Surface-last arrays; IFS level "
        "JK maps to our j = JK-1; k_cbot/k_ctop/k_dpl are OUR 0-based "
        "surface-last indices (PAPH(IKB) = p_half[k_cbot])."
    ),
    "inputs": {
        "M_u": "kg m-2 s-1", "PMFUS": "W m-2", "PMFUQ": "kg m-2 s-1",
        "PMFUL": "kg m-2 s-1", "PLUDE": "kg m-2 s-1", "PDMFUP": "kg m-2 s-1",
        "PMFUDE_RATE": "kg m-2 s-1", "PDMFEN": "kg m-2 s-1",
        "T": "K", "q": "kg kg-1", "qs": "kg kg-1", "T_h": "K", "q_h": "kg kg-1",
        "T_u": "K", "q_u": "kg kg-1", "l_u": "kg kg-1",
        "p_full": "Pa", "p_half": "Pa", "geo_full": "m2 s-2", "geo_half": "m2 s-2",
        # total physics tendencies (PTENT/PTENQ) -- ZDHPBL / ZCAPPBL integrands
        "dT_dt_other": "K s-1", "dq_dt_other": "kg kg-1 s-1",
        # advective (dynamics) tendencies (PTENTA/PTENQA) -- ZCAPE2 and ZDQCV
        # integrands; same name/meaning as the trigger module's dq_dt_adv
        "dT_dt_adv": "K s-1", "dq_dt_adv": "kg kg-1 s-1",
        "pwmean": "m2 s-2 Pa", "zdpmean": "Pa",
        "land_frac": "1", "dx_m": "m", "dt": "s",
        "ldcum": "1 (bool)", "ktype": "1 (1 deep, 2 shallow)",
        "k_cbot": "1 (level index)", "k_ctop": "1 (level index)",
        "k_dpl": "1 (level index)",
    },
    "outputs": {
        "M_u": "kg m-2 s-1", "PMFUS": "W m-2", "PMFUQ": "kg m-2 s-1",
        "PMFUL": "kg m-2 s-1", "PLUDE": "kg m-2 s-1", "PDMFUP": "kg m-2 s-1",
        "PMFUDE_RATE": "kg m-2 s-1", "PDMFEN": "kg m-2 s-1",
        "M_b1": "kg m-2 s-1", "zcape": "Pa", "zheat": "kg m-2 s-1",
        "zxtau": "s", "zmfs": "1", "ldcum": "1 (bool)",
    },
    "sign_convention": (
        "updraught mass flux M_u non-negative; positive zcape drives the "
        "updraught; zmfs non-negative; zheat non-negative after the 1e-4 "
        "floor; pressure increases with the array index (surface last)."
    ),
    "conserves": ["none"],
    "differentiable": True,
    "reference": (
        "OpenIFS cumastrn.F90 (main 8f6f722), branch Earth, deterministic "
        "configuration (SPP / LLPERT_RTAU off); fixed coefficients from "
        "sucumf.F90, provenance in the module constants"
    ),
    "idealized_test": "tests/unit/test_ifs_closure.py",
}


# ---------------------------------------------------------------------------
# (1) sub-cloud moist-static-energy supply  cumastrn.F90:484-496
# ---------------------------------------------------------------------------
def subcloud_mse_supply(dT_dt_other, dq_dt_other, p_half, k_cbot, k_start):
    """ZDHPBL, cumastrn.F90:484-496::

        IF(LDCUM(JL).AND.JK >= KCBOT(JL)) THEN
          ZDZ=(PAPH(JL,JK+1)-PAPH(JL,JK))
          ZDHPBL(JL)=ZDHPBL(JL)+(RLVTT*PTENQ(JL,JK)+RCPD*PTENT(JL,JK))*ZDZ

    Integrated from the surface (PAPH(KLEV+1)) up to the cloud-base
    interface.  The outer loop starts at NJKT2 (sucumf.F90:280-285, the
    highest level whose reference pressure exceeds 60 hPa), not at 1:
    k_start is that cutoff expressed as OUR 0-based full-level index
    (IFS JK = k_start + 1), derived from p_full by the caller.

    Tendency mapping: PTENT/PTENQ are the TOTAL physics tendencies -- pass
    them as dT_dt_other / dq_dt_other (NOT the advective pair, which the
    source reserves for ZCAPE2 / ZDQCV).

    Arrays are surface-last and k_cbot is OUR surface-last 0-based half
    index (IFS IKB <-> k_cbot), so the mask is the raw 0-based full-level
    index j >= k_cbot.
    """
    nlev = p_half.shape[-1] - 1
    j = jnp.arange(nlev)[None, :]                             # our 0-based j
    mask = (j >= jnp.asarray(k_cbot)[:, None]) & (j >= jnp.asarray(k_start)[:, None])
    dp = (p_half[..., 1:] - p_half[..., :-1]) * mask
    return jnp.sum((RLVTT * dq_dt_other + RCPD * dT_dt_other) * dp, axis=-1)


# ---------------------------------------------------------------------------
# (1) first guess cloud-base mass flux  cumastrn.F90:540-592
# ---------------------------------------------------------------------------
def first_guess_mass_flux(p_half, k_cbot, ldcum, ktype, zdhpbl, zdh_shal, dt, cfg):
    """ZMFUB, cumastrn.F90:540-592.

    Deep (KTYPE==1), :559-564::

        ZMFMAX=(PAPH(JL,IKB)-PAPH(JL,IKB-1))*ZCONS2
        IF (KTYPE(JL) == 1) THEN
          ZMFUB(JL)=ZMFMAX*0.1_JPRB

    Shallow (KTYPE==2), :570-581, with ZDH from :566-572::

        ZQUMQE=PQU(JL,IKB)+PLU(JL,IKB)-ZQENH(JL,IKB)
        ZDQMIN=MAX(0.01_JPRB*ZQENH(JL,IKB),1.E-10_JPRB)
        ZDH=RCPD*(PTU(JL,IKB)-ZTENH(JL,IKB))+RLVTT*ZQUMQE
        ZDH=RG*MAX(ZDH,1.E5_JPRB*ZDQMIN)
        IF (ZDHPBL(JL) > 0.0_JPRB) THEN
          ZMFUB(JL)=ZDHPBL(JL)/ZDH
          ZMFUB(JL)=MIN(ZMFUB(JL),ZMFMAX)
        ELSE
          ZMFUB(JL)=ZMFMAX*0.1_JPRB
          LDCUM(JL)=.FALSE.
        ENDIF

    Not LDCUM (:590): ``ZMFUB(JL)=0.0_JPRB``.

    zdh_shal is the ZDH above (with the :570 floor RG*MAX(ZDH,1.E5*ZDQMIN)
    already applied) supplied by the caller from the plume/environment
    values at the cloud base.

    Returns (zmfub, zmfmax, ldcum) where ldcum is the updated convection
    mask carrying the source's ``LDCUM(JL)=.FALSE.`` on the shallow
    ZDHPBL <= 0 branch (:578).
    """
    kb = jnp.asarray(k_cbot).astype(jnp.int32)                # our half index
    idx = jnp.arange(p_half.shape[0])
    dp_base = p_half[idx, kb] - p_half[idx, jnp.maximum(kb - 1, 0)]
    zmfmax = dp_base * cfg.rmfcfl / (RG * _f(dt))             # ZCONS2 = RMFCFL/(RG*PTSPHY)

    deep = ldcum & (ktype == 1)
    shallow = ldcum & (ktype == 2)
    shal_off = shallow & ~(zdhpbl > 0.0)                      # :578 LDCUM=.FALSE.

    mf_deep = zmfmax * ZMFUB_FRAC_OF_ZMFMAX                   # :560
    mf_shal = jnp.where(
        zdhpbl > 0.0,
        jnp.minimum(zdhpbl / _f(zdh_shal), zmfmax),
        zmfmax * ZMFUB_FRAC_OF_ZMFMAX,                        # :580
    )
    zmfub = jnp.where(deep, mf_deep, jnp.where(shallow, mf_shal, 0.0))
    ldcum = jnp.where(shal_off, jnp.zeros_like(jnp.asarray(ldcum, bool)),
                      jnp.asarray(ldcum, bool))
    return zmfub, zmfmax, ldcum


# ---------------------------------------------------------------------------
# (2) deep-convective CAPE closure  cumastrn.F90:715-866
# ---------------------------------------------------------------------------
def deep_cape_closure(
    T, q, qs, T_h, q_h, T_u, q_u, l_u, M_u,
    p_full, p_half, geo_full, geo_half,
    dT_dt_other, dq_dt_other,
    dT_dt_adv, dq_dt_adv,
    ldcum, ktype, k_cbot, k_ctop, k_dpl,
    pwmean_raw, zdpmean,
    land_frac, dx_m, dt, cfg,
    zmfub, zmfmax,
):
    """CAPE closure for KTYPE=1, cumastrn.F90:715-866.

    Tendency mapping (cumastrn.F90):
      * dT_dt_other / dq_dt_other are the TOTAL physics tendencies
        PTENT / PTENQ -- used ONLY by the ZCAPPBL integrand (:494-496).
      * dT_dt_adv / dq_dt_adv are the ADVECTIVE (dynamics) tendencies
        PTENTA / PTENQA -- used by ZCAPE2 (:757-758) and ZDQCV (:767).
        Same name and meaning as the trigger module's dq_dt_adv.

    NJKT2 window (sucumf.F90:280-285): every integral loop starts at
    ``DO JK=NJKT2,KLEV`` (:490, :732), NJKT2 being the highest level whose
    reference pressure exceeds 60 hPa (cfg.njkt2_pa).  The cutoff is
    derived per column from p_full as the highest OUR index j with
    p_full[j] > njkt2_pa, floored at j = 1 (IFS default NJKT2 = 2).

    Integrand masks: ``LLO1 = LDCUM .AND. KTYPE==1``,
    ``LLO3 = LLO1 .AND. JK <= KCBOT .AND. JK > KCTOP`` -- evaluated here on
    OUR surface-last 0-based indices j (k_cbot/k_ctop are those indices,
    so llo3 = llo1 & (j <= k_cbot) & (j > k_ctop)).

    ZHEAT integrand (:735-741, LDTDKMF = .FALSE. branch)::

        ZHEAT(JL)=ZHEAT(JL) +MAX(0.0_JPRB,&
         & (  (PTEN(JL,JK-1)-PTEN(JL,JK) + ZDZ*ZORCPD)/ZTENH(JL,JK)&
         & +  RETV*(PQEN(JL,JK-1)-PQEN(JL,JK))  ) *&
         & (RG*(PMFU(JL,JK)+PMFD(JL,JK))) )

    with ZDZ = PGEO(JK-1)-PGEO(JK), ZORCPD = 1/RCPD (cumastrn.F90:411 --
    PGEO is already a geopotential difference, so no factor of RG), and
    PMFD = 0 here (downdraught out of scope).

    ZCAPE integrand (:743-747, virtual terms + condensate loading;
    ``-PLRAIN not added`` per the source comment)::

        ZCAPE(JL)=ZCAPE(JL) +&
         & ((PTU(JL,JK)-ZTENH(JL,JK))/ZTENH(JL,JK)&
         & +RETV*(PQU(JL,JK)-ZQENH(JL,JK))&
         & -PLU(JL,JK) ) * ZDZ

    with ZDZ = PAP(JK)-PAP(JK-1) (pressure integral).

    ZCAPE2 (:757-765) uses the ADVective-tendency-corrected environment::

        ZTENH2(JL,JK)=ZTENH(JL,JK)-0.5_JPRB*(PTENTA(JL,JK)+PTENTA(JL,JK-1))*PTSPHY
        ZQENH2(JL,JK)=ZQENH(JL,JK)-0.5_JPRB*(PTENQA(JL,JK)+PTENQA(JL,JK-1))*PTSPHY

    ZDQCV advection term (:767, :853) -- PTENQA, i.e. dq_dt_adv::

        ZDQCV(JL)=ZDQCV(JL)+PTENQA(JL,JK)*ZDZ*(PQEN(JL,JK)/PQSEN(JL,JK))
        ...
        ZDQCV(JL)=ZDQCV(JL)*RLVTT/PGEOH(JL,IK)*ZXTAU(JL)/MAX(1.25_JPRB,ZTAURES)*RCAPQADV

    ZSATFR (:773-776, :826)::

        ZSATFR(JL)=ZSATFR(JL)+(PQEN(JL,JK)/PQSEN(JL,JK))*ZDZ   ! JK >= KCTOP
        ZSATFR(JL)=ZSATFR(JL)/(PAPH(JL,KLEV+1)-PAPH(JL,IK))

    Time scale (:803)::

        ZTAU(JL)=(PGEOH(JL,IK)-PGEOH(JL,IKB))/((2.0_JPRB+MIN(15.0_JPRB,PWMEAN(JL)))*RG)*ZTAURES*RTAUA

    with IK = KCTOP and PWMEAN the mean updraught velocity converted from
    the raw CUASC accumulators exactly as cuascn.F90:863-864::

        PWMEAN(JL)=MAX(1.E-2_JPRB,PWMEAN(JL)/MAX(1.0_JPRB,ZDPMEAN(JL)))
        PWMEAN(JL)=SQRT(2.0_JPRB*PWMEAN(JL))

    (both floors reproduced: the MAX(1,ZDPMEAN) denominator floor and the
    1e-2 numerator floor, which also keeps sqrt away from zero).

    RCAPDCYCL == 2 correction (:815-819).  ZCAPPBL (:494-496, integrated
    in the same NJKT2 loop as ZDHPBL, PTENT/PTENQ = total tendencies)::

        ZCAPPBL(JL)=ZCAPPBL(JL)+(PTENT(JL,JK)+RETV*PTEN(JL,JK)*PTENQ(JL,JK))*ZDZ

    Land branch::

        IF(LLO1.AND.RCAPDCYCL==2.0_JPRB) THEN
          IF(LDLAND(JL)) THEN
            ZCAPDCYCL(JL)=ZCAPPBL(JL)*ZTAU(JL)/ZTAURES

    (ocean branch needs PUEN/PVEN, out of contract; applied on land_frac.)

    Final assembly (:844-865, LMFCUCA off so ZFACCA = 1, LDTDKMF off)::

        ZCAPE2(JL)=RCAPQADV*ZCAPE2(JL)+(1.0_JPRB-RCAPQADV)*ZCAPE(JL)
        ZCAPDCYCL(JL)=MAX(ZCAPDCYCL(JL),-2*ZCAPE2(JL))
        IF(ZSATFR(JL)<=0.94.OR.PVERVEL(JL,NJKT5)<-100._JPRB) THEN
          ZCAPE(JL)=MAX(RMINCAPE*ZCAPE(JL),ZCAPE2(JL)-ZCAPDCYCL(JL)+ZDQCV(JL))
        ELSE
          ZCAPE(JL)=MAX(RMINCAPE*ZCAPE(JL),ZCAPE(JL)-ZCAPDCYCL(JL))
        ENDIF
        ZCAPE(JL)=MIN(ZCAPE(JL),5000.0_JPRB)
        ZHEAT(JL)=MAX(1.E-4_JPRB,ZHEAT(JL))
        ZXTAU(JL)=MAX(3.6E3/(5.0),MIN(3.0*3.6E3,ZXTAU(JL)))     ! PPLRG*PPLDARE = 1
        ZMFUB1(JL)=(ZCAPE(JL)*ZMFUB(JL))/(ZHEAT(JL)*ZXTAU(JL))
        ZMFUB1(JL)=MAX(ZMFUB1(JL),0.001_JPRB)*ZFACCA(JL)        ! ZFACCA = 1 (LMFCUCA=.F.)
        ZMFMAX=(PAPH(JL,IKB)-PAPH(JL,IKB-1))*ZCONS2
        ZMFUB1(JL)=MIN(ZMFUB1(JL),ZMFMAX)
    """
    ncol = T.shape[0]
    nlev = T.shape[1]
    j = jnp.arange(nlev)[None, :]                             # our 0-based j
    kb = jnp.asarray(k_cbot)[:, None]
    kt = jnp.asarray(k_ctop)[:, None]

    # NJKT2 window (sucumf.F90:280-285): highest OUR index with p > cfg.njkt2_pa,
    # floored at j = 1 (IFS default NJKT2 = 2 -> our 1).
    # sucumf.F90:280-285 scans JLEV = NFLEVG..2 and keeps overwriting NJKT2, so the
    # surviving value is the SMALLEST (highest-altitude) index whose pressure still
    # exceeds the threshold: a MINIMUM over our surface-last index, not a maximum.
    j_njkt2 = jnp.min(
        jnp.where(p_full > cfg.njkt2_pa,
                  jnp.broadcast_to(j, p_full.shape),
                  jnp.full_like(p_full, p_full.shape[-1] - 1)),
        axis=-1)
    j_njkt2 = jnp.maximum(j_njkt2, 1)
    m_top = j >= j_njkt2[:, None]

    llo1 = ldcum[:, None] & (jnp.asarray(ktype) == 1)[:, None] & m_top
    llo3 = llo1 & (j <= kb) & (j > kt)

    # ---- ZHEAT (environmental stability x g*(PMFU+PMFD), PMFD=0) -------
    dz_geo = geo_full[:, :-1] - geo_full[:, 1:]               # PGEO(JK-1)-PGEO(JK)
    stable = ((T[:, :-1] - T[:, 1:] + dz_geo * ZORCPD) / _f(T_h[:, 1:])
              + RETV * (q[:, :-1] - q[:, 1:])) * (RG * M_u[:, 1:])
    zheat = jnp.sum(jnp.maximum(0.0, stable) * llo3[:, 1:], axis=-1)

    # ---- ZCAPE / ZCAPE2 (pressure integral, virtual + loading) ---------
    dp_full = (p_full - jnp.concatenate(
        [p_full[:, :1], p_full[:, :-1]], axis=-1))            # PAP(JK)-PAP(JK-1)
    buoy = ((T_u - T_h) / _f(T_h) + RETV * (q_u - q_h) - l_u)
    zcape = jnp.sum(buoy * dp_full * llo3, axis=-1)

    # ZCAPE2 uses the ADVECTIVE tendencies PTENTA/PTENQA (:757-758)
    dT2 = 0.5 * (dT_dt_adv
                 + jnp.concatenate([dT_dt_adv[:, :1], dT_dt_adv[:, :-1]], -1)) * dt
    dq2 = 0.5 * (dq_dt_adv
                 + jnp.concatenate([dq_dt_adv[:, :1], dq_dt_adv[:, :-1]], -1)) * dt
    T_h2 = T_h - dT2
    q_h2 = q_h - dq2
    buoy2 = ((T_u - T_h2) / _f(T_h2) + RETV * (q_u - q_h2) - l_u)
    zcape2 = jnp.sum(buoy2 * dp_full * llo3, axis=-1)

    # ---- ZDQCV / ZSATFR -------------------------------------------------
    dp_half = p_half[:, 1:] - p_half[:, :-1]                  # PAPH(JK+1)-PAPH(JK)
    rh_frac = q / _f(qs)
    # ZDQCV integrand uses PTENQA (:767), i.e. dq_dt_adv
    zdqcv_raw = jnp.sum(dq_dt_adv * dp_half * rh_frac * llo1, axis=-1)
    m_sat = llo1 & (j >= kt)
    zsatfr_int = jnp.sum(rh_frac * dp_half * m_sat, axis=-1)
    p_top_if = jnp.take_along_axis(p_half, kt.astype(jnp.int32), axis=1)[:, 0]          # PAPH(KCTOP)
    ps = p_half[:, -1]                                        # PAPH(KLEV+1)
    zsatfr = zsatfr_int / _f(ps - p_top_if)

    # ---- ZCAPPBL (:494-496), sub-cloud, JK >= KCBOT; PTENT/PTENQ (total) --
    m_pbl = (j >= kb) & m_top
    zcappbl = jnp.sum((dT_dt_other + RETV * T * dq_dt_other)
                      * dp_half * m_pbl, axis=-1)

    # ---- timescale ------------------------------------------------------
    # cuascn.F90:863-864, both floors reproduced
    w_mean = jnp.sqrt(2.0 * jnp.maximum(
        PWMEAN_MIN, pwmean_raw / jnp.maximum(ZDPMEAN_MIN, zdpmean)))
    ztr = ifs_ztaures(dx_m)
    geo_top = jnp.take_along_axis(geo_half, kt.astype(jnp.int32), axis=1)[:, 0]        # PGEOH(KCTOP)
    geo_base = jnp.take_along_axis(geo_half, kb.astype(jnp.int32), axis=1)[:, 0]       # PGEOH(IKB)
    ztau = (geo_top - geo_base) / ((ZTAU_W_OFFSET + jnp.minimum(ZTAU_W_MAX, w_mean)) * RG) \
        * ztr * cfg.rtaua                                     # :803
    zxtau = ztau                                             # deterministic (LLPERT_RTAU off)

    # ---- RCAPDCYCL == 2 land/PBL correction (:815-819) ------------------
    p_dpl = jnp.take_along_axis(p_half, jnp.asarray(k_dpl)[:, None].astype(jnp.int32), axis=1)[:, 0]   # PAPH(KDPL)
    llo1_surf = (ps - p_dpl) < ZDPL_SURF_PDIFF_MAX            # :813
    zcapdcycl = jnp.where(
        llo1_surf & (jnp.asarray(cfg.rcapdcycl) == 2.0) & (land_frac > 0.0),
        land_frac * zcappbl * ztau / _f(ztr),
        0.0,
    )

    zdqcv = zdqcv_raw * RLVTT / _f(geo_top) * zxtau \
        / jnp.maximum(ZTAURES_DQCV_FLOOR, ztr) * cfg.rcapqadv  # :853

    # ---- final assembly (:844-865) ---------------------------------------
    deep = ldcum & (jnp.asarray(ktype) == 1)
    zcape2_mix = cfg.rcapqadv * zcape2 + (1.0 - cfg.rcapqadv) * zcape
    zcapdcycl = jnp.maximum(zcapdcycl, -2.0 * zcape2_mix)
    zcape_new = jnp.where(
        zsatfr <= ZSATFR_MAX,                                       # PVERVEL term out of contract
        jnp.maximum(cfg.rmincape * zcape, zcape2_mix - zcapdcycl + zdqcv),
        jnp.maximum(cfg.rmincape * zcape, zcape - zcapdcycl),
    )
    zcape = jnp.minimum(zcape_new, ZCAPE_MAX)                 # :858
    zheat = jnp.maximum(ZHEAT_MIN, zheat)                     # :859
    zxtau = jnp.clip(zxtau, ZXTAU_MIN, ZXTAU_MAX)             # :861 (PPLRG*PPLDARE=1)
    zfacca = jnp.ones_like(zmfub)                             # LMFCUCA off, :801
    zmfub1 = (zcape * zmfub) / _f(zheat * zxtau)              # :862
    zmfub1 = jnp.maximum(zmfub1, ZMFUB1_MIN) * zfacca         # :863
    zmfub1 = jnp.minimum(zmfub1, zmfmax)                      # :864-865

    return (jnp.where(deep, zmfub1, zmfub),
            zcape * deep, zheat * deep, zxtau * deep)


# ---------------------------------------------------------------------------
# (3) shallow-convective closure  cumastrn.F90:880-916
# ---------------------------------------------------------------------------
def shallow_closure(p_half, k_cbot, ldcum, ktype, zdhpbl,
                    T_u, q_u, l_u, T_h, q_h, zmfub, zmfmax, dt, cfg):
    """KTYPE=2 closure, cumastrn.F90:880-916.

    With PMFD = 0 the downdraught ratio ZEPS (:884-889) is zero, so
    (:894-905)::

        ZQUMQE=PQU(JL,IKB)+PLU(JL,IKB)-ZQENH(JL,IKB)
        ZDH=RCPD*(PTU(JL,IKB)-ZTENH(JL,IKB))+RLVTT*ZQUMQE
        ZDH2=RCPD*(PTU(JL,IKB-1)-ZTENH(JL,IKB-1))+RLVTT*ZQUMQE
        ZDH=0.5_JPRB*(ZDH+ZDH2)
        ZDH=RG*MAX(ZDH,0.1_JPRB*RCPD)
        ZMFUB1(JL)=ZDHPBL(JL)/ZDH
        IF (.NOT. LDTDKMF) THEN
          IF(ZMFUB1(JL)>0.9_JPRB*ZMFMAX/RMFCFL.AND.ZDH<0.5_JPRB*RG*RCPD) THEN
            ZDH=0.75_JPRB*RCPD*RG
            ZMFUB1(JL)=ZDHPBL(JL)/ZDH
          ENDIF
        ENDIF

    Note ZDH2 uses the plume/environment values at IKB-1 (the interface
    above the cloud base), i.e. our index k_cbot-1.
    (:912) ``ZMFUB1(JL)=ZMFUB(JL)`` when ZDHPBL <= 0;
    (:915) ``ZMFUB1(JL)=MIN(ZMFUB1(JL),ZMFMAX)``.
    LMFWSTAR = .FALSE. (sucumf.F90:232), so ZMF_SHAL is never used.
    k_cbot is OUR surface-last 0-based half index: PAPH(IKB)=p_half[k_cbot],
    PAPH(IKB-1)=p_half[k_cbot-1].
    """
    ncol = p_half.shape[0]
    idx = jnp.arange(ncol)
    kb = jnp.asarray(k_cbot).astype(jnp.int32)
    kb_m1 = jnp.maximum(kb - 1, 0)

    t_u_b = jnp.take_along_axis(T_u, kb[:, None], axis=1)[:, 0]
    t_u_b1 = jnp.take_along_axis(T_u, kb_m1[:, None], axis=1)[:, 0]
    q_u_b = jnp.take_along_axis(q_u, kb[:, None], axis=1)[:, 0]
    l_u_b = jnp.take_along_axis(l_u, kb[:, None], axis=1)[:, 0]
    th_b = jnp.take_along_axis(T_h, kb[:, None], axis=1)[:, 0]
    th_b1 = jnp.take_along_axis(T_h, kb_m1[:, None], axis=1)[:, 0]
    qh_b = jnp.take_along_axis(q_h, kb[:, None], axis=1)[:, 0]

    zqumqe = q_u_b + l_u_b - qh_b                             # ZEPS = 0 (no downdraught)
    zdh = RCPD * (t_u_b - th_b) + RLVTT * zqumqe
    zdh2 = RCPD * (t_u_b1 - th_b1) + RLVTT * zqumqe            # :896 uses IKB-1 values
    zdh = 0.5 * (zdh + zdh2)
    zdh = RG * jnp.maximum(zdh, ZDH_FLOOR_FRAC_RCPD * RCPD)   # :899

    dp_base = p_half[idx, kb] - p_half[idx, kb_m1]
    zmfmax_rmfcfl = dp_base / (RG * _f(dt))                   # ZMFMAX/RMFCFL
    zmfub1 = jnp.where(zdhpbl > 0.0, zdhpbl / _f(zdh), zmfub)  # :905 / :912
    weak = (zmfub1 > ZMFUB1_WEAK_FRAC * zmfmax_rmfcfl) \
        & (zdh < ZDH_WEAK_FRAC * RG * RCPD)                   # :897
    zmfub1 = jnp.where(weak, zdhpbl / _f(ZDH_WEAK_FLOOR_FRAC * RCPD * RG),
                       zmfub1)                                # :899-900
    shallow = ldcum & (jnp.asarray(ktype) == 2)
    zmfub1 = jnp.minimum(zmfub1, zmfmax)                      # :915
    return jnp.where(shallow, zmfub1, zmfub)


# ---------------------------------------------------------------------------
# (4) final closure = scaling  cumastrn.F90:955-1018
# ---------------------------------------------------------------------------
def scale_profile(M_u, PMFUS, PMFUQ, PMFUL, PLUDE, PDMFUP, PMFUDE_RATE,
                  PDMFEN, M_b0, M_b1, p_half, k_cbot, k_ctop, ldcum, dt, cfg):
    """Rescale the once-computed ascent profiles, cumastrn.F90:955-1018.

    (:963) ``ZMFS(JL)=ZMFUB1(JL)/MAX(RMFCMIN,ZMFUB(JL))``.

    Limit loop (:967-979)::

        DO JK=2,KLEV
          IF(LDCUM(JL).AND.JK>=KCTOP(JL)-1) THEN
            IKB=KCBOT(JL)
            IF(JK>IKB) THEN
              ZDZ=((PAPH(JL,KLEV+1)-PAPH(JL,JK))/(PAPH(JL,KLEV+1)-PAPH(JL,IKB)))
              PMFU(JL,JK)=PMFU(JL,IKB)*ZDZ
            ENDIF
            ZMFMAX=(PAPH(JL,JK)-PAPH(JL,JK-1))*ZCONS2
            IF (.NOT. LDTDKMF) ZMFMAX=MIN(ZMFMAX,RMFLIA)
            IF(PMFU(JL,JK)*ZMFS(JL)>ZMFMAX) THEN
               ZMFS(JL)=MIN(ZMFS(JL),ZMFMAX/PMFU(JL,JK))
               ZMFS(JL)=MAX(ZMFS(JL),1.E-10_JPRB)
            ENDIF
          ENDIF

    The 1e-10 floor applies ONLY inside the binding branch (:975-979), i.e.
    only when some level binds; it is NOT applied to columns whose ratio is
    legitimately below 1e-10 with no binding level.

    Implemented as a per-column minimum of the allowed factor over levels,
    then the single ZMFS applied to exactly the arrays rescaled at
    :1008-1018: PMFU, ZMFUS, ZMFUQ, ZMFUL, ZDMFUP, ZDMFEN, PLUDE and
    PMFUDE_RATE (PLUDELI of :1012-1013 is out of contract).
    """
    ncol, nlev = M_u.shape
    # our surface-last indices: j <-> IFS JK-1; p_half has nlev+1 interfaces
    j = jnp.arange(nlev)[None, :]
    kb = jnp.asarray(k_cbot)[:, None].astype(jnp.int32)
    kt = jnp.asarray(k_ctop)[:, None].astype(jnp.int32)
    ld = jnp.asarray(ldcum, bool)[:, None]
    ph = p_half[:, :nlev]                                     # PAPH(JK)
    ps = p_half[:, -1:]                                       # PAPH(KLEV+1)

    zmfs = M_b1 / jnp.maximum(cfg.rmfcmin, M_b0) * ldcum      # :957

    # (:963-967) below the base (JK > IKB): PMFU(JK) = PMFU(IKB)*ZDZ
    pb = jnp.take_along_axis(p_half, kb, axis=1)              # PAPH(IKB)
    mfu_base = jnp.take_along_axis(M_u, kb, axis=1)           # PMFU(IKB)
    window = ld & (j >= 1) & (j >= kt - 1)                    # JK = 2..KLEV, JK >= KCTOP-1
    below = window & (j > kb)
    zdz_taper = (ps - ph) / _f(ps - pb)
    M_u_eff = jnp.where(below, mfu_base * zdz_taper, M_u)

    # (:968-979) one column factor against the per-level CFL / RMFLIA bound.
    # The 1e-10 floor sits on the binding path only (:975-979): a non-binding
    # level contributes +inf, not a floor.
    dp_lev = ph - jnp.concatenate([ph[:, :1], ph[:, :-1]], -1)  # PAPH(JK)-PAPH(JK-1)
    zmfmax_lev = jnp.minimum(dp_lev * cfg.rmfcfl / (RG * _f(dt)), cfg.rmflia)
    binding = window & (M_u_eff * zmfs[:, None] > zmfmax_lev)
    allowed = jnp.where(binding,
                        jnp.maximum(1.0e-10, zmfmax_lev / _f(M_u_eff)),
                        jnp.inf)
    zmfs = jnp.minimum(zmfs, jnp.min(allowed, axis=-1))
    zmfs = jnp.where(ldcum, zmfs, 0.0)

    # (:1003-1018) scale the profiles on JK <= KCBOT and JK >= KCTOP-1
    m_scale = ld & (j >= 1) & (j <= kb) & (j >= kt - 1)
    s = jnp.where(m_scale, zmfs[:, None], 1.0)
    M_u_s = M_u_eff * s
    (PMFUS, PMFUQ, PMFUL, PDMFUP, PDMFEN, PLUDE, PMFUDE_RATE) = (
        arr * s for arr in (PMFUS, PMFUQ, PMFUL, PDMFUP, PDMFEN, PLUDE, PMFUDE_RATE))
    return (M_u_s, PMFUS, PMFUQ, PMFUL, PLUDE, PDMFUP, PMFUDE_RATE, PDMFEN, zmfs)


# ---------------------------------------------------------------------------
# top-level composition
# ---------------------------------------------------------------------------
def ifs_closure(
    M_u, PMFUS, PMFUQ, PMFUL, PLUDE, PDMFUP, PMFUDE_RATE, PDMFEN,
    T_u, q_u, l_u, k_ctop, pwmean, zdpmean,
    ldcum, ktype, k_cbot, k_dpl,
    T, q, qs, p_full, p_half, geo_full, geo_half, T_h, q_h,
    dT_dt_other, dq_dt_other,
    dT_dt_adv, dq_dt_adv,
    land_frac, dx_m, dt,
    cfg: IFSClosureConfig = IFSClosureConfig(),
    ktype_first_guess=None,
):
    """Compose (1)-(4) over a column set (leading dimension = columns).

    Tendency mapping (cumastrn.F90):
      * dT_dt_other / dq_dt_other -- TOTAL physics tendencies PTENT/PTENQ,
        used by the ZDHPBL and ZCAPPBL integrals (:484-496).
      * dT_dt_adv / dq_dt_adv -- ADVECTIVE (dynamics) tendencies
        PTENTA/PTENQA, used by ZCAPE2 (:757-758) and ZDQCV (:767); same
        name and meaning as the trigger module's dq_dt_adv.

    The source (cumastrn) runs the entraining ascent ONCE with the
    first-guess mass flux ZMFUB and then rescales all updraught flux
    profiles by ZMFS -- there is no second ascent.  This routine
    therefore takes the ascent output computed with M_b0 and returns it
    rescaled by ZMFS together with M_b1 and the closure diagnostics
    (ZCAPE, ZHEAT, ZXTAU, ZMFS).

    CALLER CONTRACT: the ascent handed in must have been run with the
    first-guess base mass flux, i.e.
    ``M_u[k_cbot] == first_guess_mass_flux(...)[0]`` (PMFU(IKB) == ZMFUB
    in cumastrn).  ZHEAT is proportional to PMFU and ZMFS is the ratio of
    the closure mass flux to that base, so a caller whose ascent used a
    different cloud-base flux (too small, in particular) gets a ZHEAT that
    is too small and hence a closure that saturates at ZMFMAX.  Rescale
    the updraught profiles by ``M_b0 / M_u[k_cbot]`` before calling if the
    ascent was run at another amplitude.

    The source's ``LDCUM(JL)=.FALSE.`` on the shallow ZDHPBL <= 0 branch
    (cumastrn.F90:578) is applied here: the updated ldcum is threaded
    through the later stages and returned to the caller.

    The downdraught (cudlfsn/cuddrafn) is out of scope: M_d = 0 in the
    ZHEAT integral and ZEPS = 0 in the shallow closure.
    """
    # half-level inputs carry nlev+1 interfaces (index nlev = surface); the
    # IFS half arrays have KLEV entries (JK <-> our j = JK-1): drop the surface
    nlev = T.shape[1]
    T_h = T_h[:, :nlev]
    q_h = q_h[:, :nlev]
    geo_half = geo_half[:, :nlev]

    # NJKT2 window (sucumf.F90:280-285): highest OUR index j with
    # p_full[j] > cfg.njkt2_pa, floored at j = 1 (IFS default NJKT2 = 2).
    jj = jnp.arange(nlev)[None, :]
    # NJKT2 = SMALLEST surface-last index with p > njkt2_pa (see deep_cape_closure).
    k_start = jnp.maximum(
        jnp.min(jnp.where(p_full > cfg.njkt2_pa,
                          jnp.broadcast_to(jj, p_full.shape),
                          jnp.full_like(p_full, p_full.shape[-1] - 1)), axis=-1),
            1)

    # (1) first guess -----------------------------------------------------
    zdhpbl = subcloud_mse_supply(dT_dt_other, dq_dt_other, p_half, k_cbot, k_start)
    idx = jnp.arange(T.shape[0])
    kb = jnp.asarray(k_cbot).astype(jnp.int32)
    t_u_b = T_u[idx, kb]
    th_b = T_h[idx, kb]
    q_u_b = q_u[idx, kb]
    l_u_b = l_u[idx, kb]
    qh_b = q_h[idx, kb]
    zqumqe = q_u_b + l_u_b - qh_b
    zdqmin = jnp.maximum(ZDQMIN_FRAC * qh_b, ZDQMIN_ABS_FLOOR)   # :567
    # cumastrn.F90:569:  ZDH=RG*MAX(ZDH,1.E5_JPRB*ZDQMIN)
    zdh_base = RG * jnp.maximum(
        RCPD * (t_u_b - th_b) + RLVTT * zqumqe, ZDH_DQMIN_SCALE * zdqmin)
    # The first guess is cumastrn's ZMFUB at :563-576, computed BEFORE the
    # ascent and therefore with the PRE-reclassification type; the final
    # scaling ZMFS = ZMFUB1/ZMFUB (:963) must divide by that same ZMFUB.  A
    # caller that reclassifies KTYPE after the ascent (cumastrn.F90:635-641)
    # passes the earlier type here; everything below uses the new one.
    ktype_fg = ktype if ktype_first_guess is None else ktype_first_guess
    M_b0, zmfmax, ldcum = first_guess_mass_flux(
        p_half, k_cbot, ldcum, ktype_fg, zdhpbl, zdh_base, dt, cfg)

    # (2) deep closure ----------------------------------------------------
    M_b1, zcape, zheat, zxtau = deep_cape_closure(
        T, q, qs, T_h, q_h, T_u, q_u, l_u, M_u,
        p_full, p_half, geo_full, geo_half,
        dT_dt_other, dq_dt_other,
        dT_dt_adv, dq_dt_adv,
        ldcum, ktype, k_cbot, k_ctop, k_dpl,
        pwmean, zdpmean, land_frac, dx_m, dt, cfg, M_b0, zmfmax)

    # (3) shallow closure: overrides M_b1 on KTYPE 2 columns only; deep
    # columns keep the CAPE-closure value (cumastrn.F90:880-916 is a
    # separate loop over shallow points)
    M_b1 = shallow_closure(
        p_half, k_cbot, ldcum, ktype, zdhpbl,
        T_u, q_u, l_u, T_h, q_h, M_b1, zmfmax, dt, cfg)

    # (4) scaling ---------------------------------------------------------
    (M_u, PMFUS, PMFUQ, PMFUL, PLUDE, PDMFUP, PMFUDE_RATE, PDMFEN,
     zmfs) = scale_profile(
        M_u, PMFUS, PMFUQ, PMFUL, PLUDE, PDMFUP, PMFUDE_RATE, PDMFEN,
        M_b0, M_b1, p_half, k_cbot, k_ctop, ldcum, dt, cfg)

    return {
        "M_u": M_u, "PMFUS": PMFUS, "PMFUQ": PMFUQ, "PMFUL": PMFUL,
        "PLUDE": PLUDE, "PDMFUP": PDMFUP, "PMFUDE_RATE": PMFUDE_RATE,
        "PDMFEN": PDMFEN, "M_b0": M_b0, "M_b1": M_b1, "ldcum": ldcum,
        "zcape": zcape, "zheat": zheat, "zxtau": zxtau, "zmfs": zmfs,
    }
