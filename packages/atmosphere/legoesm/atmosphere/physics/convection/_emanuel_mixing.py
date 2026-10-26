"""Emanuel CONVECT v4.3c updraught, closure and buoyancy-sorting mixing.

Faithful, vectorised, differentiable JAX port of Kerry Emanuel's
``convect43c.f`` (CONVECT version 4.3c, 20 May 2002) from the column setup
through the mixing-matrix tendencies (lines 338-938, ``IPBL = 0``,
``MINORIG = 1``), plus the TLIFT subroutine (lines 1004-1075).  The
precipitating downdraught (lines 713-830) is :mod:`._emanuel_downdraft`; the
cloud-top ``FRAC`` adjustment and the ``ENTS`` enthalpy pass (lines 940-984)
are applied by :mod:`.emanuel` after the downdraught tendencies are added, as
in the oracle.

The oracle is the CONVECT 4.3c source verified by sha256 in
``scripts/validate/oracle/emanuel_convect43c/build.sh``; the committed test
``tests/atmosphere/test_emanuel_convect43c_oracle.py`` runs this port against
the fixture produced by RUNNING that oracle.

What is ported, in oracle order:

1. ``GZ``, ``CPN``, ``H`` (dry static energy with a moist heat capacity),
   ``LV``, frozen moist static energy ``HM``, virtual temperature ``TV``.
2. ``IHMIN`` (lowest local minimum of ``HM``) and the parcel source ``NK``
   (first level of maximum ``HM`` at or below ``IHMIN``).
3. The rejection gates: ``T(NK) < 250 K``, ``Q(NK) <= 0``,
   ``IHMIN = NL-1`` (IFLAG 0); ``PLCL`` outside [200, 2000) hPa (IFLAG 2);
   ``ICB >= NL-1`` (IFLAG 3); the cold-start stable exit
   (``CBMF = 0`` and ``TVP(ICB) <= TV(ICB) - DTMAX``); and
   ``CBMF = CBMFOLD = 0``.  A rejected column returns zero tendencies; IFLAG
   0/2/3 also reset the carried cloud-base mass flux to zero.
4. ``PLCL`` (the chi form, lines 392-394), ``ICB``, TLIFT with the oracle's
   cold-branch vapour pressure, ``TVP`` from the Newton ``QG``.
5. ``EP``, ``INB``/``INB1`` from the running CAPE, ``FRAC``, ``HP``.
6. The ``DTMA`` closure and the prognostic ``CBMF`` relaxation.
7. ``M(I)`` with ``DBO`` at ``MIN(I, INB1)``; ``SIJ``/``ELIJ``/``MENT``/
   ``QENT`` with the saturated re-solve; the no-entrainment local detrainment;
   the ``DO 200`` equal-probability normalisation (running ``SMIN`` test,
   ``SJMIN = 0`` at the lowest level, ``BSUM < 1e-18`` fallback).
8. The ``DO 410-500`` mixing tendencies (``AMP1``, ``AD``, local
   detrainment, ``AWAT``), with the undilute updraught flux counted only at
   and above ``NK``.

Differentiability
-----------------
Integer indices (``NK``, ``ICB``, ``INB``, ``INB1``) are discrete in the oracle
and stay discrete here (``argmax`` / ``take_along_axis``); gradients flow
through the gathered values.  Every oracle switch that the shipped code had
already smoothed through a config field keeps that field, and the field's
large-sharpness limit IS the oracle switch:

* ``level_window_sharpness`` -- the ``ICB``/``INB`` level windows, evaluated at
  half-integer offsets (``sigmoid(s*(level - ICB + 1/2))``) so they tend to
  exactly 0/1;
* ``sij_gate_sharpness`` -- the ``0 < SIJ < sij_upper_gate`` entrainment band;
* ``sat_branch_sharpness`` -- the saturated re-solve switch;
* ``strict_index_sharpness`` -- the strict ``J > I`` / ``K < I`` inequalities;
* ``lcl_pressure_sharpness`` -- the sub-cloud ``DTPBL`` layer above ``NK``.

Every other oracle switch is ported hard (``jnp.where``), including the
``ABS(DENOM) < 0.01`` guards.  Where a hard branch can produce a non-finite
value on the unselected side, the input is replaced before the operation
(``where(ok, x, 1)``) so the forward value is unchanged and no NaN reaches
the gradient.

Conventions
-----------
Internally SURFACE-FIRST oracle ordering (index 0 = lowest level), 0-based
indices (oracle level ``I`` is index ``I-1``), pressures in hPa as in the
oracle.  ``NL = nlev - 1``.  The public entry accepts and returns the model's
surface-LAST arrays in Pa.  Thermodynamic constants are legoESM's
(``legoesm.constants``); CONVECT asks the host for them.  ``QS`` (a CONVECT
input) is :func:`_oracle_qsat`.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from legoesm import constants
from legoesm.thermo import saturation_vapor_pressure


__all__ = (
    "EmanuelMixingOutput",
    "emanuel_mixing_tendencies",
)


__physics_contract__ = {
    "summary": (
        "CONVECT v4.3c (Emanuel 1991) updraught, source/cloud-base/cloud-top "
        "diagnosis, DTMA cloud-base mass-flux closure and (i,j) buoyancy-sort "
        "mixing matrix; returns the mixing temperature/vapour tendencies, the "
        "retained mixture condensate and the oracle intermediates the "
        "downdraught and the FRAC/ENTS passes consume."
    ),
    "inputs": {
        "T": "K", "q_v": "kg/kg", "p_full": "Pa", "p_half": "Pa",
        "cbmf_old": "kg/m^2/s (carried cloud-base mass flux, oracle CBMF in)",
        "dt": "s",
        "c_l": "J/kg/K (CONVECT liquid-water heat capacity CL)",
    },
    "outputs": {
        "dT_dt": "K/s", "dq_v_dt": "kg/kg/s", "dq_c_conv_dt": "kg/kg/s",
        "cbmf": "kg/m^2/s (oracle CBMF out)",
        "ment": "kg/m^2/s ((i,j) mixing matrix, diagnostic)",
        "m_profile": "kg/m^2/s (per-level updraught mixing mass flux)",
        "convective_layer_mask": "1 (levels 1..INB)",
    },
    "sign_convention": (
        "Heats and dries/moistens the environment by the oracle's mass-flux "
        "convergence; the mixture condensate AWAT detrained below its origin "
        "is a non-negative cloud-water source (dq_c_conv_dt>=0); the "
        "precipitating fraction EP*CLW leaves the column; column enthalpy is "
        "closed by the caller's ENTS pass; surface at the last vertical index."
    ),
    "conserves": ["none"],
    "differentiable": True,
    "reference": (
        "Emanuel (1991), J. Atmos. Sci. 48, 2313-2335; CONVECT v4.3c "
        "convect43c.f lines 338-938, TLIFT 1004-1075"
    ),
    "idealized_test": (
        "tests/atmosphere/test_emanuel_convect43c_oracle.py: FT, FQ and CBMF "
        "match the running convect43c.f oracle to <= 1e-10 (fp64) at the hard "
        "limit of the smoothing fields; a stable column returns zero."
    ),
}


# --- unit conversion (oracle P/PH in hPa) ---
_HPA_PER_PA = 0.01
# --- source-level / LCL rejection gates (convect43c.f l.384, 395) ---
_T_SOURCE_MIN_K = 250.0
_PLCL_MIN_HPA = 200.0
_PLCL_MAX_HPA = 2000.0
# --- LCL "within 0.2% of Bolton (1980)" chi form (convect43c.f l.393) ---
_EMANUEL_CHI_A = 1669.0
_EMANUEL_CHI_B = 122.0
# --- TLIFT (convect43c.f l.1056-1065) ---
_TLIFT_TG_FLOOR_K = 35.0
# Cold-branch vapour pressure ES = EXP(A - B/TG + C*LOG(TG)) [hPa], TC < 0.
# satcurve-ok: CONVECT's own TLIFT curve, ported because the oracle uses it.
_TLIFT_ES_COLD_A = 23.33086
_TLIFT_ES_COLD_B = 6111.72784
_TLIFT_ES_COLD_C = 0.15215
_TLIFT_NEWTON_ITERS = 2
# --- precipitation efficiency (convect43c.f l.458) ---
_CLW_EP_FLOOR = 1.0e-8
# --- mixing rate M(I) (convect43c.f l.580): ENTP*0.02*(PH(K)-PH(K+1)) [hPa] ---
_DBO_ENTRAIN_COEFF_PER_HPA = 0.02
# --- FRAC (convect43c.f l.527) ---
_DEFRAC_FLOOR = 0.001
# --- DO 200 normalisation floors (convect43c.f l.690, 698) ---
_ASIJ_FLOOR = 1.0e-21
_BSUM_MIN = 1.0e-18
# --- closure (convect43c.f l.563-566): CBMF += 0.1*ALPHA*DTMA, DELT0=300 s ---
_CBMF_RELAX = 0.1
_DELT0_S = 300.0
# _oracle_qsat AD floor (pre-existing; Tetens pole near 29.65 K)
_TETENS_MIN_VALID_K = 180.0
_EPMAX_DEFAULT = 0.999
_STRICT_INDEX_SHARPNESS = 20.0


class EmanuelMixingOutput(NamedTuple):
    """Output of :func:`emanuel_mixing_tendencies`.

    ``dT_dt``, ``dq_v_dt``, ``dq_c_conv_dt`` and ``convective_layer_mask`` are
    surface-LAST; every oracle intermediate is SURFACE-FIRST, 0-based.
    ``dT_dt``/``dq_v_dt`` are the oracle's DO 410-500 mixing terms only: the
    downdraught terms, ``FRAC`` and ``ENTS`` are added by the caller.
    """

    dT_dt: jax.Array
    dq_v_dt: jax.Array
    dq_c_conv_dt: jax.Array
    ment: jax.Array
    m_profile: jax.Array
    convective_layer_mask: jax.Array
    elij: jax.Array = None      # ELIJ(i,j)
    clw: jax.Array = None       # CLW(i)
    ep: jax.Array = None        # EP(i)
    qs: jax.Array = None        # QS(i)
    # H(i) = T*CPN + GZ (oracle line 353): DRY static energy with a
    # moisture-weighted heat capacity, no LV*Q term.
    h_dry_static: jax.Array = None
    gz: jax.Array = None        # GZ(i)
    lv: jax.Array = None        # LV(i)
    cpn: jax.Array = None       # CPN(i)
    cbmf: jax.Array = None      # oracle CBMF out, (ncol,)
    inb: jax.Array = None       # INB, 0-based surface-first, (ncol,) int
    frac: jax.Array = None      # FRAC, (ncol,)


def _oracle_qsat(T, p):
    """Saturation specific humidity [kg/kg], ``eps*es/(p - es*(1-eps))``.

    This is the host-supplied ``QS`` CONVECT takes as an input (the warm
    Tetens curve of :func:`legoesm.thermo.saturation_vapor_pressure`); the
    oracle fixture was produced with the same ``QS``.  The 180 K floor and the
    softplus on the denominator are AD guards that do not act on atmospheric
    columns.
    """
    eps = constants.epsilon
    T_safe = jnp.maximum(T, _TETENS_MIN_VALID_K)
    es = saturation_vapor_pressure(T_safe)
    denom = jax.nn.softplus(p - es * (1.0 - eps) - 1.0) + 1.0
    return eps * es / denom


def _tlift_es_hpa(tg):
    """TLIFT vapour pressure [hPa] (convect43c.f l.1058-1065): warm Tetens for
    ``TC >= 0``, the oracle's cold-branch fit below."""
    es_warm = saturation_vapor_pressure(tg) * _HPA_PER_PA
    es_cold = jnp.exp(
        _TLIFT_ES_COLD_A - _TLIFT_ES_COLD_B / tg + _TLIFT_ES_COLD_C * jnp.log(tg))
    return jnp.where(tg - constants.T_freeze >= 0.0, es_warm, es_cold)


def _tlift_moist(T, qs, p_hpa, gz, T_nk, q_nk, gz_nk, *, c_l):
    """TLIFT saturated-parcel branch at every level (convect43c.f l.1042-1073).

    Returns ``(TPK, QG)``.  ``S`` and ``ALV`` use the ENVIRONMENT ``T(I)``;
    ``QG`` starts from the host ``QS(I)``; two Newton steps.
    """
    cpd = constants.c_pd
    rv = constants.R_v
    eps = constants.epsilon
    cpvmcl = c_l - constants.c_pv
    ah0 = ((cpd * (1.0 - q_nk) + c_l * q_nk) * T_nk
           + q_nk * (constants.L_v - cpvmcl * (T_nk - constants.T_freeze)) + gz_nk)
    alv = constants.L_v - cpvmcl * (T - constants.T_freeze)
    tg = T
    qg = qs
    for _ in range(_TLIFT_NEWTON_ITERS):
        s = 1.0 / (cpd + alv * alv * qg / (rv * T * T))
        ahg = cpd * tg + (c_l - cpd) * q_nk * T + alv * qg + gz
        tg = jnp.maximum(tg + s * (ah0 - ahg), _TLIFT_TG_FLOOR_K)
        es = _tlift_es_hpa(tg)
        qg = eps * es / (p_hpa - es * (1.0 - eps))
    tpk = (ah0 - (c_l - cpd) * q_nk * T - gz - alv * qg) / cpd
    return tpk, qg


def _hard_denom(d, floor):
    """Oracle ``IF(ABS(D).LT.floor) D = floor`` (positive floor)."""
    return jnp.where(jnp.abs(d) < floor, floor, d)


def _last_true(mask, idx):
    """(any, last index where ``mask``) along the last axis."""
    return jnp.any(mask, axis=-1), jnp.max(jnp.where(mask, idx, -1), axis=-1)


def _at(x, i):
    """``x[c, i[c]]`` for a (ncol, n) array and (ncol,) int index."""
    return jnp.take_along_axis(x, i[:, None], axis=-1)[:, 0]


def emanuel_mixing_tendencies(
    T: jax.Array,
    q_v: jax.Array,
    p_full: jax.Array,
    p_half: jax.Array,
    cbmf_old: jax.Array,
    dt: float,
    *,
    c_l: float,
    elcrit: float,
    tlcrit: float,
    entp: float,
    dtmax: float,
    alpha_closure: float,
    damp_coefficient: float,
    M_b_max: float,
    level_window_sharpness: float,
    sij_gate_sharpness: float,
    sij_upper_gate: float,
    denom_floor: float,
    sat_branch_sharpness: float,
    lcl_pressure_sharpness: float,
    strict_index_sharpness: float = _STRICT_INDEX_SHARPNESS,
    epmax: float = _EPMAX_DEFAULT,
) -> EmanuelMixingOutput:
    """CONVECT v4.3c from the column setup through the mixing tendencies.

    Parameters
    ----------
    T, q_v : (ncol, nlev) temperature [K], specific humidity [kg/kg],
        surface-LAST.
    p_full, p_half : (ncol, nlev), (ncol, nlev+1) pressures [Pa], surface-LAST.
    cbmf_old : (ncol,) carried cloud-base mass flux [kg/m^2/s] (oracle CBMF in).
    dt : time step [s] (oracle DELT).
    c_l, elcrit, tlcrit, entp, dtmax, alpha_closure, damp_coefficient,
    epmax, sij_upper_gate, denom_floor : oracle CL, ELCRIT, TLCRIT, ENTP,
        DTMAX, ALPHA, DAMP, EPMAX, the 0.9 band edge and the 0.01 floor.
    M_b_max : cap on the TRANSPORT mass flux (no oracle counterpart; the
        carried ``cbmf`` is returned uncapped).
    level_window_sharpness, sij_gate_sharpness, sat_branch_sharpness,
    strict_index_sharpness, lcl_pressure_sharpness : smoothing of the oracle
        switches named in the module docstring; large values are the oracle.
    """
    ncol, nlev = T.shape
    dtype = T.dtype
    nl = nlev - 1                       # oracle NL (1-based) = ND - 1

    cpd = constants.c_pd
    cpv = constants.c_pv
    rd = constants.R_d
    rv = constants.R_v
    epsi = 1.0 / constants.epsilon
    g = constants.g
    cpvmcl = c_l - cpv

    # ---- surface-FIRST, hPa -------------------------------------------------
    Tf = T[:, ::-1]
    qf = q_v[:, ::-1]
    pf = p_full[:, ::-1] * _HPA_PER_PA
    phf = p_half[:, ::-1] * _HPA_PER_PA             # (ncol, nlev+1), PH(1)=sfc
    qs = _oracle_qsat(T, p_full)[:, ::-1]
    idx = jnp.arange(nlev)
    lvl = idx.astype(dtype)[None, :]
    dph = phf[:, :-1] - phf[:, 1:]                  # PH(I)-PH(I+1) > 0 [hPa]

    # ---- DO 40: GZ, CPN, H, LV, HM, TV --------------------------------------
    tv = Tf * (1.0 + qf * epsi - qf)
    dgz = 0.5 * rd * (tv[:, 1:] + tv[:, :-1]) * (pf[:, :-1] - pf[:, 1:]) / phf[:, 1:-1]
    gz = jnp.concatenate([jnp.zeros((ncol, 1), dtype), jnp.cumsum(dgz, axis=-1)], axis=-1)
    cpn = cpd * (1.0 - qf) + cpv * qf
    h = Tf * cpn + gz
    lv = constants.L_v - cpvmcl * (Tf - constants.T_freeze)
    hm = (cpd * (1.0 - qf) + c_l * qf) * (Tf - Tf[:, :1]) + lv * qf + gz

    # IHMIN: first level of the minimum HM among levels where HM decreases
    # upward (I = 2..NL+1); NL if none; then MIN(IHMIN, NL-1).
    cand = jnp.concatenate([jnp.zeros((ncol, 1), bool), hm[:, 1:] < hm[:, :-1]], axis=-1)
    ihmin = jnp.where(jnp.any(cand, axis=-1),
                      jnp.argmin(jnp.where(cand, hm, jnp.inf), axis=-1), nl - 1)
    ihmin = jnp.minimum(ihmin, nl - 2)
    # NK: first level of maximum HM in MINORIG..IHMIN.
    nk = jnp.argmax(jnp.where(idx[None, :] <= ihmin[:, None], hm, -jnp.inf), axis=-1)

    T_nk = _at(Tf, nk)
    q_nk = _at(qf, nk)
    gz_nk = _at(gz, nk)
    gate_a = (T_nk < _T_SOURCE_MIN_K) | (q_nk <= 0.0) | (ihmin == nl - 2)

    # ---- LCL (l.392-394) and ICB (l.402-408) -------------------------------
    rh = q_nk / _at(qs, nk)
    rh = jnp.where(q_nk > 0.0, rh, 1.0)             # IFLAG-0 column: avoid 0**chi
    chi = T_nk / (_EMANUEL_CHI_A - _EMANUEL_CHI_B * rh - T_nk)
    plcl = _at(pf, nk) * rh ** chi
    gate_b = (plcl < _PLCL_MIN_HPA) | (plcl >= _PLCL_MAX_HPA)
    above_lcl = (idx[None, :] > nk[:, None]) & (idx[None, :] <= nl - 1) & (pf < plcl[:, None])
    icb = jnp.where(jnp.any(above_lcl, axis=-1), jnp.argmax(above_lcl, axis=-1), nl - 2)
    gate_c = icb >= nl - 2

    # ---- TLIFT (KK=1 and KK=2) ---------------------------------------------
    tpk_m, qg = _tlift_moist(Tf, qs, pf, gz, T_nk[:, None], q_nk[:, None],
                             gz_nk[:, None], c_l=c_l)
    q_nk2 = q_nk[:, None]
    cpp = cpd * (1.0 - q_nk2) + q_nk2 * cpv
    tpk_d = T_nk[:, None] - (gz - gz_nk[:, None]) / cpp
    # TVP after the -TP*Q(NK) correction (l.423, 469).
    tvp_m = tpk_m * (1.0 + qg / (1.0 - q_nk2) * epsi) - tpk_m * q_nk2
    tvp_d = tpk_d * (1.0 + q_nk2 * epsi) - tpk_d * q_nk2
    icb_f = icb.astype(dtype)[:, None]
    above_icb = jax.nn.sigmoid(level_window_sharpness * (lvl - icb_f + 0.5))
    tp = above_icb * tpk_m + (1.0 - above_icb) * tpk_d
    tvp = above_icb * tvp_m + (1.0 - above_icb) * tvp_d
    clw = jnp.where(idx[None, :] >= icb[:, None], jnp.maximum(q_nk2 - qg, 0.0), 0.0)

    # Cold-start stable exit (l.429-432).
    stable = (cbmf_old == 0.0) & (_at(tvp, icb) <= _at(tv, icb) - dtmax)

    # ---- EP (l.446-464) ----------------------------------------------------
    tca = tp - constants.T_freeze
    elacrit = jnp.maximum(jnp.where(tca >= 0.0, elcrit, elcrit * (1.0 - tca / tlcrit)), 0.0)
    ep = jnp.clip(epmax * (1.0 - elacrit / jnp.maximum(clw, _CLW_EP_FLOOR)), 0.0, epmax)
    ep = jnp.where(idx[None, :] > nk[:, None], ep, 0.0)

    # ---- INB, INB1, FRAC (DO 82, l.511-533) --------------------------------
    by = jnp.concatenate([
        (tvp[:, :-1] - tv[:, :-1]) * dph[:, :-1] / pf[:, :-1],
        jnp.zeros((ncol, 1), dtype)], axis=-1)       # BY(I), I <= NL
    rng = (idx[None, :] >= icb[:, None] + 1) & (idx[None, :] <= nl - 2)
    cape_run = jnp.cumsum(jnp.where(rng, by, 0.0), axis=-1)
    any1, last1 = _last_true(rng & (by >= 0.0), idx[None, :])
    inb1 = jnp.where(any1, last1 + 1, icb + 1)
    anyc, lastc = _last_true(rng & (cape_run > 0.0), idx[None, :])
    lastc_s = jnp.maximum(lastc, 0)
    inb = jnp.where(anyc, lastc + 1, icb + 1)
    byp = jnp.where(anyc, _at(by, jnp.minimum(lastc_s + 1, nlev - 1)), 0.0)
    capem = jnp.where(anyc, _at(cape_run, lastc_s), 0.0)
    inb = jnp.maximum(inb, inb1)
    cape = capem + byp
    defrac = jnp.maximum(capem - cape, _DEFRAC_FLOOR)
    frac = jnp.clip(-cape / defrac, 0.0, 1.0)
    inb_f = inb.astype(dtype)[:, None]

    # ---- HP (DO 95): H(NK)+(LV+(CPD-CPV)T)*EP*CLW on ICB..INB --------------
    hp_c = _at(h, nk)[:, None] + (lv + (cpd - cpv) * Tf) * ep * clw
    hp = above_icb * hp_c + (1.0 - above_icb) * h

    # ---- closure DTMA (l.549-567) ------------------------------------------
    i0 = icb - 1
    tvp0 = _at(tvp, i0)
    p0 = _at(pf, i0)
    tvpplcl = tvp0 - rd * tvp0 * (p0 - plcl) / (_at(cpn, i0) * p0)
    p_icb = _at(pf, icb)
    tvaplcl = _at(tv, icb) + (_at(tvp, icb) - _at(tvp, icb + 1)) * (plcl - p_icb) / (
        p_icb - _at(pf, icb + 1))
    pbl = jnp.where(
        idx[None, :] == nk[:, None], 1.0,
        jnp.where(idx[None, :] > nk[:, None],
                  jax.nn.sigmoid(lcl_pressure_sharpness * (pf - plcl[:, None]) / _HPA_PER_PA),
                  0.0))                               # levels NK..ICB-1
    dtpbl = jnp.sum(pbl * (tvp - tv) * dph, axis=-1) / jnp.maximum(
        jnp.sum(pbl * dph, axis=-1), _HPA_PER_PA)
    dtma = tvpplcl - tvaplcl + dtmax + dtpbl
    damps = damp_coefficient * dt / _DELT0_S
    cbmf_new = jnp.maximum((1.0 - damps) * cbmf_old + _CBMF_RELAX * alpha_closure * dtma, 0.0)
    passed = ~(gate_a | gate_b | gate_c | stable)
    cbmf = jnp.where(passed, cbmf_new, 0.0)
    M_b = jnp.clip(cbmf, 0.0, M_b_max)

    # ---- M(I) (DO 103/110) -------------------------------------------------
    dbo_full = jnp.abs(tv - tvp) + entp * _DBO_ENTRAIN_COEFF_PER_HPA * dph
    dbo = jnp.take_along_axis(dbo_full, jnp.minimum(idx[None, :], inb1[:, None]), axis=-1)
    in_updraft_i = (jax.nn.sigmoid(level_window_sharpness * (lvl - icb_f - 0.5))
                    * jax.nn.sigmoid(level_window_sharpness * (inb_f - lvl + 0.5)))
    in_cloud_j = above_icb * jax.nn.sigmoid(level_window_sharpness * (inb_f - lvl + 0.5))
    dbo_w = dbo * in_updraft_i
    m_i = M_b[:, None] * dbo_w / jnp.sum(dbo_w, axis=-1, keepdims=True)

    # ---- SIJ / ELIJ / MENT / QENT (DO 160/170) -----------------------------
    H_i, H_j = h[:, :, None], h[:, None, :]
    HP_i = hp[:, :, None]
    Q_i, Q_j = qf[:, :, None], qf[:, None, :]
    QS_j, T_j, LV_j = qs[:, None, :], Tf[:, None, :], lv[:, None, :]
    QTI = q_nk[:, None, None] - (ep * clw)[:, :, None]
    BF2 = 1.0 + LV_j * LV_j * QS_j / (rv * T_j * T_j * cpd)
    eye = jnp.eye(nlev, dtype=bool)[None, :, :]

    ANUM = H_j - HP_i + (cpv - cpd) * T_j * (QTI - Q_j)
    DENOM = H_i - HP_i + (cpd - cpv) * (Q_i - QTI) * T_j
    SIJ = jnp.where(eye, 1.0, ANUM / _hard_denom(DENOM, denom_floor))
    ALTEM = (SIJ * Q_i + (1.0 - SIJ) * QTI - QS_j) / BF2
    CWAT = (clw * (1.0 - ep))[:, None, :]
    j_gt_i = jax.nn.sigmoid(
        strict_index_sharpness * (lvl[:, None, :] - lvl[:, :, None] - 0.5))
    cond = jnp.clip(
        jax.nn.sigmoid(-sat_branch_sharpness * SIJ)
        + jax.nn.sigmoid(sat_branch_sharpness * (SIJ - 1.0))
        + jax.nn.sigmoid(sat_branch_sharpness * (ALTEM - CWAT)), 0.0, 1.0) * j_gt_i
    ANUM2 = ANUM - LV_j * (QTI - QS_j - CWAT * BF2)
    DENOM2 = DENOM + LV_j * (Q_i - QTI)
    SIJ2 = ANUM2 / _hard_denom(DENOM2, denom_floor)
    ALTEM2 = SIJ2 * Q_i + (1.0 - SIJ2) * QTI - QS_j - (BF2 - 1.0) * CWAT
    # The oracle's diagonal is exactly 1 (SIJ(I,I)=1, never re-solved: J>I).
    SIJ_eff = jnp.where(eye, 1.0, (1.0 - cond) * SIJ + cond * SIJ2)
    ALTEM_eff = (1.0 - cond) * ALTEM + cond * ALTEM2

    # Entrainment band 0 < SIJ < sij_upper_gate (smooth, sij_gate_sharpness).
    eps_low = 3.0 / sij_gate_sharpness
    upper_center = sij_upper_gate - 2.0 / sij_gate_sharpness
    sij_active = (jax.nn.sigmoid(sij_gate_sharpness * (SIJ_eff - eps_low))
                  * jax.nn.sigmoid(sij_gate_sharpness * (upper_center - SIJ_eff)))
    sij_active = sij_active * in_updraft_i[:, :, None] * in_cloud_j[:, None, :]
    # Hard oracle band and windows (NENT, SMIN control flow).
    win_i = (idx[None, :] >= icb[:, None] + 1) & (idx[None, :] <= inb[:, None])
    win_j = (idx[None, :] >= icb[:, None]) & (idx[None, :] <= inb[:, None])
    act = ((SIJ_eff > 0.0) & (SIJ_eff < sij_upper_gate) & ~eye
           & win_i[:, :, None] & win_j[:, None, :])

    QENT = SIJ_eff * Q_i + (1.0 - SIJ_eff) * QTI
    ELIJ = jnp.maximum(ALTEM_eff, 0.0) * sij_active
    one_minus_sij = jnp.clip(1.0 - SIJ_eff, 1.0 - sij_upper_gate, None)
    MENT_raw = (m_i[:, :, None] / one_minus_sij) * sij_active
    nent = jnp.sum(act, axis=-1)

    # ---- SCRIT and the DO 200 normalisation --------------------------------
    qp1 = q_nk[:, None] - ep * clw
    scrit = (h - hp - lv * (qp1 - qs)) / _hard_denom(h - hp + lv * (qf - qp1), denom_floor)
    scrit = jnp.where(qp1 - qs + scrit * (qf - qp1) < 0.0, 1.0, scrit)
    scrit = jnp.maximum(scrit, 0.0)                   # (ncol, i)

    # Clipped SIJ as stored by the oracle: J window only, 1 on the diagonal.
    S = jnp.where(win_j[:, None, :], jnp.clip(SIJ_eff, 0.0, 1.0), 0.0)
    S = jnp.where(eye & win_i[:, :, None], 1.0, S)
    Sp = jnp.concatenate([S[:, :, 1:], jnp.zeros_like(S[:, :, :1])], axis=-1)
    Sm = jnp.concatenate([jnp.zeros_like(S[:, :, :1]), S[:, :, :-1]], axis=-1)

    def _norm_step(smin, xs):
        j, s_j, sp_j, sm_j, a_j = xs
        gt = j > idx[None, :]
        smid_u = jnp.minimum(s_j, scrit)
        upd = a_j & gt & (smid_u < smin) & (sp_j < smid_u)
        sjmax_u = jnp.where(upd, jnp.minimum(jnp.minimum(sp_j, s_j), scrit), smid_u)
        sjmin_u = jnp.where(upd, jnp.minimum(jnp.maximum(sm_j, s_j), scrit), smid_u)
        smin = jnp.where(upd, smid_u, smin)
        smid_d = jnp.maximum(s_j, scrit)
        sjmax_d = jnp.maximum(sp_j, scrit)
        sjmin_d = jnp.maximum(sm_j, scrit)            # Sm = 0 at the lowest level
        smid = jnp.where(gt, smid_u, smid_d)
        w = (jnp.abs(jnp.where(gt, sjmax_u, sjmax_d) - smid)
             + jnp.abs(jnp.where(gt, sjmin_u, sjmin_d) - smid))
        return smin, w

    xs = (idx, jnp.moveaxis(S, 2, 0), jnp.moveaxis(Sp, 2, 0),
          jnp.moveaxis(Sm, 2, 0), jnp.moveaxis(act, 2, 0))
    _, dw = lax.scan(_norm_step, jnp.ones_like(scrit), xs)
    prob_w = jnp.moveaxis(dw, 0, 2) * dph[:, None, :] * sij_active
    # ASIJ = MAX(1e-21, ASIJ), written so a floored row divides by the
    # constant: same value, but no derivative through 1/ASIJ**2, which
    # overflows in fp32 (1e-21**-2) and turns gradients/tangents into NaN.
    asij = jnp.sum(prob_w, axis=-1, keepdims=True)
    ok = asij > _ASIJ_FLOOR
    MENT = jnp.where(ok, MENT_raw * prob_w / jnp.where(ok, asij, 1.0),
                     MENT_raw * prob_w / _ASIJ_FLOOR)
    bsum = jnp.sum(MENT, axis=-1)
    # Local detrainment: NENT = 0 (DO 170) or BSUM < 1e-18 (DO 190).
    local = (nent == 0) | (bsum < _BSUM_MIN)
    diag_local = eye & local[:, :, None]
    MENT = MENT + jnp.where(diag_local, m_i[:, :, None], 0.0)
    QENT = jnp.where(diag_local, QTI, QENT)
    ELIJ = jnp.where(diag_local, clw[:, :, None], ELIJ)

    # ---- tendencies (DO 410-500, mixing terms) -----------------------------
    m_above = jnp.maximum(jnp.cumsum(m_i[:, ::-1], axis=-1)[:, ::-1] - m_i, 0.0)
    m_above = jnp.where(idx[None, :] >= nk[:, None], m_above, 0.0)
    F_up = jnp.sum(jnp.cumsum(MENT, axis=1) * j_gt_i, axis=-1)
    AMP1 = m_above + F_up
    k_below_t = jax.nn.sigmoid(
        strict_index_sharpness * (lvl[:, :, None] - lvl[:, None, :] - 0.5))
    AD = jnp.sum(jnp.cumsum(MENT[:, ::-1], axis=1)[:, ::-1] * k_below_t, axis=-1)

    def up(x):
        return jnp.concatenate([x[:, 1:], x[:, -1:]], axis=-1)

    def dn(x):
        return jnp.concatenate([x[:, :1], x[:, :-1]], axis=-1)

    # oracle DPINV = 0.01/(PH(I)-PH(I+1)) [1/Pa]; the floor (1e-6 Pa) is the
    # pre-existing guard and never binds on a valid grid.
    dpinv = _HPA_PER_PA / jnp.maximum(dph, 1.0e-8)
    cpn_inv = 1.0 / cpn
    ft = g * dpinv * (AMP1 * (up(Tf) - Tf + (up(gz) - gz) * cpn_inv)
                      - AD * (Tf - dn(Tf) + (gz - dn(gz)) * cpn_inv))
    fq = g * dpinv * (AMP1 * (up(qf) - qf) - AD * (qf - dn(qf)))
    ment_ii = jnp.sum(jnp.where(eye, MENT, 0.0), axis=-1)
    qent_ii = jnp.sum(jnp.where(eye, QENT, 0.0), axis=-1)
    ft = ft + g * dpinv * ment_ii * (hp - h + Tf * (cpv - cpd) * (qf - qent_ii)) * cpn_inv
    AWAT = jnp.maximum(ELIJ - ((1.0 - ep) * clw)[:, None, :], 0.0)
    k_lt_j = j_gt_i                                   # origin k (axis 1) < level j
    detr = MENT * (QENT - k_lt_j * AWAT - Q_j)
    fq = fq + g * dpinv * jnp.sum(detr, axis=1)
    # Mixture condensate AWAT detrained below its origin's level is kept as
    # cloud water (the oracle sends it to WDTRAIN; routing unchanged here).
    dqc = g * dpinv * jnp.sum(k_lt_j * MENT * AWAT, axis=1)

    ents_mask = jax.nn.sigmoid(level_window_sharpness * (inb_f - lvl + 0.5))

    return EmanuelMixingOutput(
        dT_dt=ft[:, ::-1],
        dq_v_dt=fq[:, ::-1],
        dq_c_conv_dt=dqc[:, ::-1],
        ment=MENT,
        m_profile=m_i,
        convective_layer_mask=ents_mask[:, ::-1],
        elij=ELIJ,
        clw=clw,
        ep=ep,
        qs=qs,
        h_dry_static=h,
        gz=gz,
        lv=lv,
        cpn=cpn,
        cbmf=cbmf,
        inb=inb,
        frac=frac,
    )
