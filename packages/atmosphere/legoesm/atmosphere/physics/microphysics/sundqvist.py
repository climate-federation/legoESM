"""Sundqvist large-scale microphysics: SBK89 precipitation via the Zhao-Carr port.

Two stages run in sequence:

1. **Condensation (NOT-YET-FAITHFUL surrogate).** A sigmoid-gated removal of
   supersaturation, ``sigmoid(k·(RH-RH_crit))·max(q_v-q_sat,0)/dt``, optionally
   overlaid by the shared hard saturation-adjustment guard
   (``SundqvistConfig.hard_saturation_adjustment``, default False).  This is NOT
   SBK89's condensation closure and has NOT been compared with oracle code (the
   Zhao-Carr condensation routine, ``zhaocarr_gscond``, has not been ported).

2. **Precipitation (FAITHFUL port).** :func:`zhao_carr_precpd` is a literal JAX
   port of ``zhaocarr_precpd_run`` from NCAR/ccpp-physics v7.0.1 (commit
   da75531bf90a799e4c0dcb988cc591d82bd024de,
   ``physics/MP/Zhao_Carr/zhaocarr_precpd.f``), the Sundqvist et al. (1989)
   precipitation scheme as implemented by Zhao & Carr (1997).  It reproduces the
   oracle, run unmodified with legoESM's Tetens curve as ``fpvs`` (the audit's
   stub), to <= 1e-10 relative in fp64 at the hard-switch limit of
   ``sigmoid_sharpness`` (``tests/atmosphere/microphysics/unit/
   test_sundqvist_zhao_carr_oracle.py``).  Contents, oracle line numbers in
   brackets:

   * cloud cover ``b = 1 - sqrt((1-RH)/(1-RH_crit))`` (0 below RH_crit, 1 at
     saturation) [409-418]; no release where ``b = 0`` [467];
   * liquid release (SBK89) ``P = c_0·dt·F1·F2·m·(1-exp(-min(50,(m·F1·F2/
     (m_r·max(b,0.01)))²)))`` with ``m = max(0, cwm - wmin)``, coalescence
     ``F1 = 1 + c1·sqrt(P_above/dt)`` and Bergeron ``F2 = 1 + c2·sqrt(min(max(0,
     268-T),20))`` [495-507];
   * ice route: ice-water id (T < -15 °C, or -15..0 °C under an ice layer)
     [380-395]; ice autoconversion ``psaut`` and accretion by snow ``psaci``
     (Lin et al. 1983) instead of the liquid release [472-482];
   * rain evaporation ``evpco·sqrt(1/dt)·(RH_crit-RH)·sqrt(P_r)``, snow
     sublimation, and the saturation-deficit cap [562-599];
   * snow melting below the freezing level [633-653]; T and q update with
     L_v / L_v+L_f / L_f [664-665];
   * the negative-condensate fix (vapour fills negative ``cwm``) [686-699];
   * the column-eligibility rule (a column with no condensate above
     ``min(wmin, wmini)`` below the model top is skipped) [297-309];
   * the frozen fraction of the surface flux, ``sr`` (0 when the total is below
     1e-13 kg m-2) [729-734].

   Mapping (legoESM -> oracle): ``cwm`` is ``q_c`` (plus ``q_i`` on a lane that
   carries it, re-split after precpd by :func:`split_condensate`); the layer mass ``del/g`` is
   ``dp/g`` from ``p_half`` (``rho``/``dz`` are not read);
   ``rh_crit`` = ``u00k``; ``auto_rate`` = ``prautco``; ``qc_crit`` = ``1/cmr``
   (oracle hard-codes 3e-4); ``coalescence_enh_coeff`` = ``c1`` (300);
   ``bergeron_enh_coeff`` = ``c2`` (0.5); ``evap_coeff`` = ``evpco``;
   ``psautco_fine``/``psautco_coarse`` = the oracle's ``psautco(1:2)``,
   blended by grid-cell area with GFS's ``wk1`` (:func:`gfs_area_weight`, GFS's
   hard clip); ``wminco_liquid``, ``wminco_ice`` = the oracle's
   ``wminco(1:2)``.  ``prautco`` is a single value (GFS's pair is equal).  The saturation humidity is the oracle's
   ``eps·e/(p + (eps-1)·e)`` with ``e = min(p, e_s)`` on legoESM's Tetens
   ``e_s``; the real GFS ``fpvs`` table (ice-blended below 0 °C) is NOT used.
   The only smoothing is the existing ``sigmoid_sharpness``: it multiplies the
   rain-evaporation and snow-sublimation rates by ``sigmoid(k·(RH_crit-RH))``
   (the gate the pre-port evaporation carried), which is exactly the oracle at
   the hard limit (the oracle's ``max(0, RH_crit-RH)`` already vanishes where
   the sigmoid switches).  Every other oracle switch is ported as a hard
   ``jnp.where`` (piecewise differentiable).  Outside the oracle, the
   liquid/ice split's -15..0 C ramp (user decision) is smoothly clamped with
   ``liquid_fraction_clamp_sharpness`` (``math.inf`` = hard clip; RULE 5).

   ``bergeron_T_peak_K`` / ``bergeron_T_width_K`` parameterised the pre-port
   Gaussian F2 and have no counterpart in the oracle: setting either to a
   non-default value raises.

The precipitation is diagnostic (falls to the surface within the step).  Any
``q_r`` passed in is drained to the surface in one step (pre-port behaviour,
not part of the oracle).  The frozen surface flux (the oracle's ``sr``) is
published as ``precip_frozen``.

References
----------
- Sundqvist, H., Berge, E. & Kristjansson, J. E. (1989): Mon. Wea. Rev., 117,
  1641-1657.
- Zhao, Q. & Carr, F. H. (1997): Mon. Wea. Rev., 125, 1931-1953.
- Lin, Y.-L., Farley, R. D. & Orville, H. D. (1983): J. Clim. Appl. Meteor.,
  22, 1065-1092 (ice autoconversion form used by Zhao-Carr).
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from legoesm import constants
from legoesm.core.smooth import smooth_clamp
from legoesm.thermo import saturation_specific_humidity, saturation_vapor_pressure
from legoesm.atmosphere.physics.microphysics._warm_rain import (
    hard_saturation_blend,
    safe_pow,
)
from legoesm.atmosphere.physics.microphysics.config import SundqvistConfig
from legoesm.atmosphere.physics.microphysics.output import (
    HydrometeorState,
    MicrophysicsOutput,
)

# --- Zhao-Carr precpd fixed coefficients (NCAR/ccpp-physics v7.0.1,
# zhaocarr_precpd.f; line numbers in brackets) ---
_ZC_CLIMIT = 1.0e-20            # [kg/kg] condensate presence threshold [223]
_ZC_EPSQ = 2.0e-12              # [kg/kg] humidity floor [155]
_ZC_CWS = 0.025                 # [-] melting by collection of cloud water [224]
_ZC_ZAODT_S = 800.0             # [s] zaodt = 800/dt time scale [226]
_ZC_CSM1 = 5.0e-8               # snow melting coefficient (x zaodt) [228]
_ZC_CRS1 = 5.0e-6               # snow sublimation coefficient (x zaodt) [229]
_ZC_CRS2 = 6.666e-10            # snow sublimation T coefficient (x zaodt) [230]
_ZC_CR = 5.0e-4                 # collection coefficient (x zaodt) [231]
_ZC_AA2 = 1.25e-3               # ice accretion by snow (x zaodt) [232]
_ZC_ICE_EXP_COEFF = 0.025       # [1/K] exp(0.025*(T-T0)) in psaut/psaci [474]
_ZC_T_ICE_C = -15.0             # [degC] ice-water id threshold [380]
_ZC_TMT0K_MIN_C = -30.0         # [degC] floor of T in the snow sublimation [566]
_ZC_QC_MIN = 1.0e-10            # [kg/kg] saturation humidity below which RH = 0 [404]
_ZC_RQ_MIN = 1.0e-10            # [-] RH above which the deficit cap uses RH [582]
_ZC_CCR_FLOOR = 0.01            # [-] cloud-cover floor in the release [501]
_ZC_RELEASE_EXP_CAP = 50.0      # [-] cap of the release exponent [502]
_ZC_F2_T_K = 268.0              # [K] Bergeron factor onset temperature [498]
_ZC_F2_DT_CAP_K = 20.0          # [K] cap of 268-T in the Bergeron factor [498]
_ZC_WMIN_P_SCALE = 1.0e-5       # [1/Pa] wmin = wminco * p * 1e-5 [255,262-263]
_ZC_SR_MIN = 1.0e-13            # [kg/m^2] total surface precip below which sr = 0 [729-734]

# --- GFS grid-area weight wk1 (fv3atm ufs-srw-v2.0.0 GFS_typedefs.F90:3218,
# 4656-4659; GFS_suite_interstitial_1.F90:43-44) ---
_GFS_MAX_LON, _GFS_MAX_LAT = 5000, 2000     # finest grid: wk1 = 1
_GFS_MIN_LON, _GFS_MIN_LAT = 192, 94        # coarsest grid: wk1 = 0
# Liquid fraction of the total condensate when the lane carries q_i: a linear
# ramp from 0 at -15 C to 1 at 0 C (Zhao-Carr's ice/water window, user
# 2026-10-27), smoothly clamped.
_LIQ_RAMP_COLD_C = -15.0


def gfs_area_weight(cell_area):
    """GFS ``wk1 = max(0, min(1, (log(area) - dxmin) * dxinv))``, the clip
    exactly as GFS (cell area is grid geometry, not traced state: RULE 5 out of
    scope; codex + GLM 2026-10-29): 1 on the finest grid GFS knows
    (5000 x 2000) and finer, 0 on the coarsest (192 x 94) and coarser."""
    tem = 2.0 * jnp.pi ** 2 * constants.R_earth ** 2
    dxmax = jnp.log(tem / (_GFS_MAX_LON * _GFS_MAX_LAT))
    dxmin = jnp.log(tem / (_GFS_MIN_LON * _GFS_MIN_LAT))
    work1 = (jnp.log(cell_area) - dxmin) / (dxmax - dxmin)
    return jnp.clip(work1, 0.0, 1.0)


def liquid_fraction(T, sharpness):
    """Liquid share of the total condensate: 0 at -15 C, 1 at 0 C, linear
    between, smoothly clamped with ``sharpness``
    (``SundqvistConfig.liquid_fraction_clamp_sharpness``; ``math.inf`` is the
    hard clip of the ramp, RULE 5).

    KNOWN DIFFERENCE at the default sharpness 100 vs the hard clip (pinned per
    temperature in test_sundqvist_area_blend_and_split.py): ln(2)/100 = 6.9e-3
    in liquid fraction exactly at -15 C and 0 C, 4.1e-3 at -14.9 C and
    -0.1 C, 0 at mid-ramp (-7.5 C) and beyond 5 C of either end (< 1e-16 at
    -20 C).  Measured 2026-10-29 (#2251).
    """
    x = (T - (constants.T_freeze + _LIQ_RAMP_COLD_C)) / -_LIQ_RAMP_COLD_C
    return smooth_clamp(x, 0.0, 1.0, sharpness)


def split_condensate(T, cwm, ice_share_in, sharpness):
    """Split precpd's total condensate ``cwm`` into ``(q_c, q_i)`` by
    :func:`liquid_fraction` of ``T`` and return the heating [K] of that
    re-partition: ``L_f/c_pd`` per kg of ice it adds over the input ice share,
    so ``c_pd T + L_v q_v - L_f q_i`` is unchanged by it (user 2026-10-27)."""
    q_i = (1.0 - liquid_fraction(T, sharpness)) * cwm
    dT = constants.L_f / constants.c_pd * (q_i - ice_share_in * cwm)  # latent-ok: Zhao-Carr oracle constant (hfus)
    return cwm - q_i, q_i, dT


class SundqvistProcessRates(NamedTuple):
    """Intermediate Sundqvist process rates used to assemble tendencies.

    ``autoconversion`` is the condensate converted to precipitation (liquid
    release, or ice autoconversion + accretion by snow) and ``evaporation`` the
    precipitation returned to vapour (rain evaporation + snow sublimation), both
    [kg/kg/s]; ``precipitation`` is the surface flux [kg/m^2/s] excluding any
    ``q_r`` drain.
    """

    condensation: jax.Array
    autoconversion: jax.Array
    evaporation: jax.Array
    precipitation: jax.Array


class ZhaoCarrPrecpdResult(NamedTuple):
    """Updated state and diagnostics of :func:`zhao_carr_precpd`."""

    T: jax.Array               # [K] (ncol, nlev)
    q_v: jax.Array             # [kg/kg]
    cwm: jax.Array             # [kg/kg] condensate
    rain_sfc: jax.Array        # [kg/m^2/s] (ncol,) total minus the frozen part
    snow_sfc: jax.Array        # [kg/m^2/s] (ncol,) the oracle sr*rn/dt
    autoconversion: jax.Array  # [kg/kg/s] condensate -> precipitation
    evaporation: jax.Array     # [kg/kg/s] precipitation -> vapour


def _check_no_gaussian_f2_override(config: SundqvistConfig) -> None:
    """The pre-port Gaussian-F2 fields have no oracle counterpart: refuse a
    non-default value (concrete values only; a traced leaf cannot be read)."""
    defaults = SundqvistConfig._field_defaults
    for name in ("bergeron_T_peak_K", "bergeron_T_width_K"):
        value = getattr(config, name)
        if isinstance(value, jax.core.Tracer):
            continue
        v = np.asarray(value)
        if v != np.asarray(defaults[name]).astype(v.dtype):
            raise ValueError(
                f"SundqvistConfig.{name}={value!r}: this field parameterised the "
                "pre-port Gaussian Bergeron factor and has no counterpart in the "
                "Zhao-Carr/SBK89 port (F2 = 1 + c2*sqrt(min(max(0, 268-T), 20)))."
            )


def _sqrt(x):
    """sqrt with a finite (zero) gradient at x <= 0; value-identical to sqrt for
    x >= 0 (x < 0 never reaches it: every caller passes a max(0, .) result)."""
    return safe_pow(x, 0.5)


def zhao_carr_precpd(
    T: jax.Array,
    q_v: jax.Array,
    cwm: jax.Array,
    p_full: jax.Array,
    layer_mass: jax.Array,
    dt: float,
    config: SundqvistConfig,
    wk1: jax.Array,
) -> ZhaoCarrPrecpdResult:
    """Literal JAX port of ccpp-physics v7.0.1 ``zhaocarr_precpd_run``.

    ``wk1`` (ncol,) is GFS's grid-area weight (:func:`gfs_area_weight`).

    Columns are (ncol, nlev) with level 0 = model top (the oracle runs bottom ->
    top; the scan here runs top -> surface, the oracle's ``k = km, 1, -1``).
    ``layer_mass`` [kg/m^2] plays the oracle's ``del/g``.  Returns the updated
    state (the oracle's in/out ``t``, ``q``, ``cwm``) and the surface rain and
    snow fluxes (the oracle's ``rn`` and ``sr``).

    Sign conventions: precipitation fluxes are positive downward; ``evaporation``
    > 0 moistens and cools; ``autoconversion`` > 0 removes condensate.
    """
    elwv = constants.L_v                       # latent-ok: Zhao-Carr oracle constant (hvap)
    eliv = constants.L_v + constants.L_f       # latent-ok: Zhao-Carr oracle constant (hvap+hfus [210])
    eliw = eliv - elwv                         # [211]
    rcp = 1.0 / constants.c_pd
    eps = constants.R_d / constants.R_v
    epsm1 = eps - 1.0
    rdt = 1.0 / dt
    zaodt = _ZC_ZAODT_S * rdt
    csm1 = _ZC_CSM1 * zaodt
    crs1 = _ZC_CRS1 * zaodt
    crs2 = _ZC_CRS2 * zaodt
    cr = _ZC_CR * zaodt
    aa2 = _ZC_AA2 * zaodt
    ke = config.evap_coeff * jnp.sqrt(rdt)     # [220, 234]
    dtcp = dt * rcp
    # Pre-port guard on q_c,crit -> 0 (the linear no-threshold limit), kept.
    cmr = 1.0 / jnp.maximum(config.qc_crit, 1e-12)
    c1 = config.coalescence_enh_coeff
    c2 = config.bergeron_enh_coeff
    u00k = config.rh_crit
    prautco = config.auto_rate
    # [194-198] psautco_l = psautco(1)*wk1 + psautco(2)*(1-wk1)
    psautco = config.psautco_fine * wk1 + config.psautco_coarse * (1.0 - wk1)
    sharpness = config.sigmoid_sharpness

    dtype = jnp.result_type(T, q_v, cwm)
    T = T.astype(dtype)
    q_v = q_v.astype(dtype)
    cwm = cwm.astype(dtype)
    p_full = p_full.astype(dtype)
    layer_mass = layer_mass.astype(dtype)

    # [253-264] wmin / wmini; [297-309] columns where precipitation can form
    # (levels 1..km-1 of the bottom->top oracle = every level but the top).
    tem_p = p_full * _ZC_WMIN_P_SCALE
    wmin = config.wminco_liquid * tem_p
    wmini = config.wminco_ice * tem_p
    eligible = jnp.any(
        (cwm > jnp.minimum(wmin, wmini))[:, 1:], axis=1)

    # Saturation specific humidity, oracle formula on the shared Tetens e_s [357-361].
    qw = jnp.minimum(p_full, saturation_vapor_pressure(T))
    qw = eps * qw / (p_full + epsm1 * qw)
    qw = jnp.maximum(qw, _ZC_EPSQ)

    def level(carry, x):
        precrl1, precsl1, iwl1 = carry
        tt, qq, ww, qw_k, wmink, wminik, mass = x
        precrk = jnp.maximum(0.0, precrl1)
        precsk = jnp.maximum(0.0, precsl1)
        wwn = jnp.maximum(ww, _ZC_CLIMIT)
        comput = eligible & ((wwn > _ZC_CLIMIT) | ((precrk + precsk) > 0.0))  # [329-337]
        has_c = wwn > _ZC_CLIMIT

        conde = dt * mass                      # (dt/g)*del  [344]
        condt = conde * rdt
        # The oracle forms 1/conde only in computed cells [342-346]; the guard
        # keeps a skipped zero-mass cell's gradient finite (value-identical).
        rconde = 1.0 / jnp.where(comput, conde, 1.0)
        qk = jnp.maximum(_ZC_EPSQ, qq)
        tmt0 = tt - constants.T_zero_zhao_carr

        # Ice-water id [380-395]; qi = qint = qw [373-374].
        fi = qk - u00k * qw_k
        iwl = jnp.where(
            tmt0 < _ZC_T_ICE_C, (fi > 0.0) | has_c,
            jnp.where(tmt0 >= 0.0, False, iwl1 & has_c))
        iwl = iwl & comput
        qc = qw_k                              # (1-fiw)*qint + fiw*qi with qi = qint
        rq_ok = qc > _ZC_QC_MIN
        rq = jnp.where(rq_ok, qk / jnp.where(rq_ok, qc, 1.0), 0.0)
        # Cloud cover b [411-418].
        rqkll = jnp.minimum(1.0, rq)
        partial = (rq >= u00k) & (rq < 1.0)
        # 1-u00k is only formed where the oracle forms it (value-identical; keeps
        # the gradient finite at rh_crit = 1).
        ccr = jnp.where(
            rq < u00k, 0.0,
            jnp.where(rq >= 1.0, 1.0, 1.0 - _sqrt(
                (1.0 - rqkll) / jnp.where(partial, 1.0 - u00k, 1.0))))

        # Precipitation production [466-536].
        do_auto = comput & (ccr > 0.0)
        wws = ww
        cwmk = jnp.maximum(0.0, wws)
        #   ice phase
        amaxcm_i = jnp.maximum(0.0, cwmk - wminik)
        expf = dt * jnp.exp(_ZC_ICE_EXP_COEFF * tmt0)
        psaut = jnp.minimum(cwmk, psautco * expf * amaxcm_i)
        ww_i = ww - psaut
        cwmk_i = jnp.maximum(0.0, ww_i)
        psaci = jnp.minimum(cwmk_i, aa2 * expf * precsl1 * cwmk_i)
        ww_i = ww_i - psaci
        #   liquid water (SBK89 release)
        amaxcm = jnp.maximum(0.0, cwmk - wmink)
        tem1 = precsl1 + precrl1
        tem2 = jnp.minimum(jnp.maximum(0.0, _ZC_F2_T_K - tt), _ZC_F2_DT_CAP_K)
        tem = (1.0 + c1 * _sqrt(tem1 * rdt)) * (1.0 + c2 * _sqrt(tem2))
        tem2 = amaxcm * cmr * tem / jnp.maximum(ccr, _ZC_CCR_FLOOR)
        tem2 = jnp.minimum(_ZC_RELEASE_EXP_CAP, tem2 * tem2)
        praut = (prautco * dt) * tem * amaxcm * (1.0 - jnp.exp(-tem2))
        praut = jnp.minimum(praut, cwmk)
        ww_l = ww - praut
        ice_auto = do_auto & iwl
        liq_auto = do_auto & ~iwl
        ww = jnp.where(ice_auto, ww_i, jnp.where(liq_auto, ww_l, ww))
        precsl = jnp.where(ice_auto, precsl1 + (wws - ww) * condt, precsl1)
        precrl = jnp.where(liq_auto, precrl1 + (wws - ww) * condt, precrl1)
        auto = (wws - ww) * rdt

        # Evaporation of precipitation [562-599].
        tmt0k = jnp.maximum(_ZC_TMT0K_MIN_C, tmt0)
        precrk = jnp.maximum(0.0, precrl)
        precsk = jnp.maximum(0.0, precsl)
        amaxrq = jnp.maximum(0.0, u00k - rq) * conde
        # Pre-port RH_crit gate (sigmoid_sharpness); 1 or 0 at the hard limit,
        # where amaxrq is already 0 whenever the gate is 0.
        gate = jax.nn.sigmoid(sharpness * (u00k - rq))
        ppr = ke * amaxrq * _sqrt(precrk) * gate
        pps = jnp.where(
            tmt0 >= 0.0, 0.0,
            (crs1 + crs2 * tmt0k) * amaxrq * precsk / u00k * gate)
        rq_cap = rq >= _ZC_RQ_MIN
        erk = jnp.where(
            rq_cap, amaxrq * qk * rdt / jnp.where(rq_cap, rq, 1.0), precrk + precsk)
        over = (ppr + pps) > jnp.abs(erk)
        psum = precrk + precsk
        rprs = erk / jnp.where(over, psum, 1.0)
        ppr = jnp.where(over, precrk * rprs, ppr)
        pps = jnp.where(over, precsk * rprs, pps)
        ppr = jnp.minimum(ppr, precrk)
        pps = jnp.minimum(pps, precsk)
        err = jnp.where(comput, ppr * rconde, 0.0)
        ers = jnp.where(comput, pps * rconde, 0.0)
        precrl = jnp.where(comput, precrl - ppr, precrl)
        precsl = jnp.where(comput, precsl - pps, precsl)

        # Melting of snow [633-653].
        melt = comput & (tmt0 > 0.0)
        amaxps = jnp.maximum(0.0, precsl)
        psm1 = csm1 * tmt0 * tmt0 * amaxps
        psm2 = _ZC_CWS * cr * jnp.maximum(0.0, ww) * amaxps
        pmelt = (psm1 + psm2) * conde
        capped = pmelt > amaxps
        pmelt = jnp.where(capped, amaxps, pmelt)
        psm1 = jnp.where(capped, amaxps * rconde, psm1)
        precrl = jnp.where(melt, precrl + pmelt, precrl)
        precsl = jnp.where(melt, precsl - pmelt, precsl)
        psm1 = jnp.where(melt, psm1, 0.0)

        # Update t and q [664-665].
        tt = jnp.where(
            comput, tt - dtcp * (elwv * err + eliv * ers + eliw * psm1), tt)
        qq = jnp.where(comput, qq + dt * (err + ers), qq)

        carry = (jnp.maximum(0.0, precrl), jnp.maximum(0.0, precsl), iwl)  # [670-672]
        return carry, (tt, qq, ww, auto, err + ers)

    ncol = T.shape[0]
    zeros = jnp.zeros((ncol,), dtype)
    xs = tuple(jnp.moveaxis(a, 1, 0) for a in (
        T, q_v, cwm, qw, wmin, wmini, layer_mass))
    (precrl1, precsl1, _), ys = jax.lax.scan(
        level, (zeros, zeros, jnp.zeros((ncol,), bool)), xs)
    t_new, q_new, w_new, auto, evap = (jnp.moveaxis(a, 0, 1) for a in ys)

    # Negative condensate filled from vapour [686-699] (every column).
    neg = w_new < 0.0
    tem = q_new + w_new
    fill = neg & (tem >= 0.0)
    drain = neg & (tem < 0.0) & (q_new > 0.0)
    t_fix = jnp.where(fill, t_new - elwv * rcp * w_new,
                      jnp.where(drain, t_new + elwv * rcp * q_new, t_new))
    q_fix = jnp.where(fill, tem, jnp.where(drain, 0.0, q_new))
    w_fix = jnp.where(fill, 0.0, jnp.where(drain, tem, w_new))

    # Frozen fraction sr = 0 for trace precipitation [729-734].
    snow = jnp.where(precrl1 + precsl1 < _ZC_SR_MIN, 0.0, precsl1)
    return ZhaoCarrPrecpdResult(
        T=t_fix, q_v=q_fix, cwm=w_fix,
        rain_sfc=(precrl1 + precsl1 - snow) * rdt, snow_sfc=snow * rdt,
        autoconversion=auto, evaporation=evap,
    )


def _sundqvist_step(T, q_v, hydrometeors, p_full, p_half, dt, config,
                    cell_area, carries_q_i):
    """Condensation surrogate followed by the Zhao-Carr precipitation port.

    Returns ``(condensation, zc, layer_mass)``; ``zc`` is the precpd state
    update computed on the PRE-condensation T and q_v (the oracle's gscond ->
    precpd order waits for the gscond port: with this surrogate's overshoot the
    post-condensation state is sub-rh_crit and precpd re-evaporates its own
    production); its ``cwm`` is q_c + q_i where the lane carries q_i, and
    ``layer_mass`` is the oracle's ``del/g``."""
    _check_no_gaussian_f2_override(config)
    if cell_area is None:
        raise ValueError(
            "sundqvist needs the lane's grid-cell area (GFS blends the ice "
            "autoconversion coefficient by it); pass cell_area, or declare the "
            "GCM cell area a single-column case stands for in its case config")
    sharpness = config.sigmoid_sharpness

    # 1. Condensation (NOT-YET-FAITHFUL surrogate; unchanged by the port).
    # Removes only the supersaturation ``q_v - q_sat``, with the smooth
    # RH_crit onset gate.  Sign: POSITIVE = vapour -> cloud water.
    q_sat = saturation_specific_humidity(T, p_full)
    RH = q_v / jnp.clip(q_sat, 1e-10)
    f = jax.nn.sigmoid(sharpness * (RH - config.rh_crit))
    condensation = f * jnp.maximum(q_v - q_sat, 0.0) / dt  # [kg/kg/s]
    # OPT-IN hard (iterated) saturation-adjustment guard (shared with the bulk
    # warm-rain schemes).  STATIC Python ``if`` on a compile-time bool.
    if config.hard_saturation_adjustment:
        condensation = hard_saturation_blend(
            condensation, T, q_v, p_full, dt, q_sat,
            config.hard_sat_adjust_threshold,
            config.hard_sat_max_heating_K,
        )

    # 2. Precipitation: the condensate after condensation, on the pre-step
    # T and q_v, with the oracle's layer mass del/g = dp/g (R9/C12: the column
    # water budget closes in the model's pressure coordinate).
    q_c_in = hydrometeors.q_c + hydrometeors.q_i if carries_q_i else hydrometeors.q_c
    cwm = q_c_in + condensation * dt
    layer_mass = (p_half[:, 1:] - p_half[:, :-1]) / constants.g
    wk1 = gfs_area_weight(jnp.broadcast_to(jnp.asarray(cell_area, T.dtype), T.shape[:1]))
    zc = zhao_carr_precpd(T, q_v, cwm, p_full, layer_mass, dt, config, wk1)
    return condensation, zc, layer_mass


def diagnose_sundqvist_process_rates(
    T: jax.Array,
    q_v: jax.Array,
    hydrometeors: HydrometeorState,
    p_full: jax.Array,
    p_half: jax.Array,
    rho: jax.Array,
    dz: jax.Array,
    dt: float,
    config: SundqvistConfig = SundqvistConfig(),
    cell_area: jax.Array | None = None,
    carries_q_i: bool = False,
) -> SundqvistProcessRates:
    """Diagnose the Sundqvist condensation, precipitation-production and
    precipitation-evaporation rates (see :class:`SundqvistProcessRates`)."""
    del rho, dz  # Signature parity with ``sundqvist_microphysics``.
    condensation, zc, _ = _sundqvist_step(
        T, q_v, hydrometeors, p_full, p_half, dt, config, cell_area, carries_q_i)
    return SundqvistProcessRates(
        condensation=condensation,
        autoconversion=zc.autoconversion,
        evaporation=zc.evaporation,
        precipitation=zc.rain_sfc + zc.snow_sfc,
    )


__physics_contract__ = {
    "summary": (
        "Sundqvist large-scale microphysics: a NOT-YET-FAITHFUL RH-gated "
        "condensation surrogate followed by a literal port of the Zhao-Carr "
        "precipitation scheme (ccpp-physics v7.0.1 zhaocarr_precpd_run): SBK89 "
        "liquid release with cloud cover b, coalescence F1 and Bergeron F2; ice "
        "autoconversion and accretion by snow; rain evaporation, snow "
        "sublimation and melting; diagnostic (instant) precipitation."
    ),
    "inputs": {
        "T": "K", "q_v": "kg/kg", "hydrometeors.q_c": "kg/kg",
        "hydrometeors.q_r": "kg/kg", "p_full": "Pa",
        "p_half": "Pa (interfaces, top-down; layer mass dp/g, R9)",
        "rho": "kg/m^3 (unused since R9; signature parity)",
        "dz": "m (unused since R9; signature parity)", "dt": "s",
    },
    "outputs": {
        "dT_dt": "K/s", "dq_v_dt": "kg/kg/s", "dq_c_dt": "kg/kg/s",
        "dq_r_dt": "kg/kg/s", "precipitation": "kg/m^2/s",
        "precip_frozen": "kg/m^2/s",
    },
    "sign_convention": (
        "z up; level 0 = model top, surface at [:, -1]. Precipitation is "
        "positive downward and leaves the column: int (dq_v + dq_c + dq_r) "
        "dp/g = -precipitation. Condensation heats with L_v; rain "
        "evaporation cools with L_v, snow sublimation with L_v+L_f, melting "
        "with L_f (Zhao-Carr); conversion of condensate to precipitation "
        "carries no latent heat."
    ),
    "conserves": ["none"],
    "differentiable": True,
    "reference": (
        "Sundqvist, Berge & Kristjansson (1989), Mon. Wea. Rev. 117, 1641-1657; "
        "Zhao & Carr (1997), Mon. Wea. Rev. 125, 1931-1953; oracle "
        "NCAR/ccpp-physics v7.0.1 zhaocarr_precpd.f"
    ),
    "idealized_test": (
        "tests/atmosphere/microphysics/unit/test_sundqvist_zhao_carr_oracle.py "
        "-- the precipitation port against the running Zhao-Carr oracle; "
        "tests/unit/test_physics_microphysics.py -- supersaturation removed, "
        "column water budget closes, q_r drained."
    ),
}


def sundqvist_microphysics(
    T: jax.Array,
    q_v: jax.Array,
    hydrometeors: HydrometeorState,
    p_full: jax.Array,
    p_half: jax.Array,
    rho: jax.Array,
    dz: jax.Array,
    dt: float,
    config: SundqvistConfig = SundqvistConfig(),
    cell_area: jax.Array | None = None,
    carries_q_i: bool = False,
) -> MicrophysicsOutput:
    """Compute Sundqvist tendencies (condensation surrogate + Zhao-Carr precipitation).

    Parameters
    ----------
    T, q_v, hydrometeors, p_full, p_half, rho, dz, dt, config
        Same interface as all microphysics backends.
    cell_area : (ncol,) or scalar grid-cell area [m^2]; required (raises if None).
    carries_q_i : static bool, whether the lane carries q_i.  If so, precpd
        runs on q_c + q_i and its condensate returns as q_c and q_i by
        :func:`liquid_fraction` of the post-step temperature; otherwise all
        condensate stays in q_c and q_i is not read.

    Returns
    -------
    MicrophysicsOutput
    """
    del rho, dz  # the layer mass is the oracle's dp/g (from p_half)
    ncol, nlev = T.shape
    condensation, zc, layer_mass = _sundqvist_step(
        T, q_v, hydrometeors, p_full, p_half, dt, config, cell_area, carries_q_i)

    # Tendencies = condensation + the Zhao-Carr state update over the step.
    dT_dt = constants.L_v * condensation / constants.c_pd + (zc.T - T) / dt
    dq_v_dt = -condensation + (zc.q_v - q_v) / dt
    z = jnp.zeros((ncol, nlev), dtype=T.dtype)
    if carries_q_i:
        cwm_in = hydrometeors.q_c + hydrometeors.q_i + condensation * dt
        pos = cwm_in > 0.0
        share = jnp.where(pos, hydrometeors.q_i / jnp.where(pos, cwm_in, 1.0), 0.0)
        q_c_new, q_i_new, dT_fus = split_condensate(
            T + dT_dt * dt, zc.cwm, share, config.liquid_fraction_clamp_sharpness)
        dT_dt = dT_dt + dT_fus / dt
        dq_c_dt = (q_c_new - hydrometeors.q_c) / dt
        dq_i_dt = (q_i_new - hydrometeors.q_i) / dt
    else:
        dq_c_dt = (zc.cwm - hydrometeors.q_c) / dt
        dq_i_dt = z

    # Diagnostic-rain semantics (pre-port, not part of the oracle): any q_r
    # passed in is treated as already-falling rain and drained to the surface
    # in one step, with the same dp/g mass weighting as the tendencies so
    #     int (dq_v + dq_c + dq_r) dp/g  =  -precipitation.
    dt_safe = jnp.maximum(dt, 1e-10)
    q_r_in = jnp.clip(hydrometeors.q_r, 0.0, None)
    dq_r_dt = -q_r_in / dt_safe
    q_r_drain_flux = jnp.sum(q_r_in * layer_mass, axis=1) / dt_safe
    precipitation = zc.rain_sfc + zc.snow_sfc + q_r_drain_flux

    return MicrophysicsOutput(
        dT_dt=dT_dt,
        dq_v_dt=dq_v_dt,
        dq_c_dt=dq_c_dt,
        dq_r_dt=dq_r_dt,
        dq_i_dt=dq_i_dt,
        dq_s_dt=z,
        dq_g_dt=z,
        dN_c_dt=z,
        dN_r_dt=z,
        dN_i_dt=z,
        precipitation=precipitation,
        precip_frozen=zc.snow_sfc,   # the oracle's sr * rn
    )
