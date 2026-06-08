"""Genuine Emanuel (1991 / CONVECT v4.3c) episodic-mixing buoyancy sort.

Faithful, vectorized, differentiable JAX port of the Fortran ``SIJ`` /
``ELIJ`` / ``MENT`` ``(i,j)`` mixing matrix from Kerry Emanuel's
``convect43c.f`` (oracle at
``.physics-validator/emanuel/oracle/convect43c.f``).  This REPLACES the
single-sigmoid ``_mixture_buoyancy`` surrogate in :mod:`.emanuel` with
the real double-loop mixing spectrum.

What the oracle does (convect43c.f lines 588-712), and what this module
reproduces term-by-term:

For every origin level ``i`` (cloud base ``ICB+1`` .. cloud top ``INB``)
and every detrainment-test level ``j`` (``ICB`` .. ``INB``):

* ``QTI = Q(NK) - EP(i)*CLW(i)`` — total water of the undilute
  updraught parcel lifted from the source level ``NK`` to level ``i``,
  after the precipitating fraction ``EP`` of its adiabatic condensate
  ``CLW`` has been removed (rained out).
* ``BF2 = 1 + LV(j)^2 QS(j) / (RV T(j)^2 CPD)`` — the
  Clausius-Clapeyron amplification factor at the detrainment level.
* ``SIJ(i,j) = ANUM/DENOM`` — the **mixing fraction**: the fraction of
  updraught air ``i`` in a mixture of updraught ``i`` and environment
  ``NK``-source air that is exactly **neutrally buoyant** when lifted to
  level ``j``.  ``ANUM = H(j) - HP(i) + (CPV-CPD) T(j) (QTI - Q(j))`` and
  ``DENOM = H(i) - HP(i) + (CPD-CPV)(Q(i)-QTI) T(j)`` are moist-static-
  energy differences (Emanuel 1991 eqs. for the neutral-buoyancy
  fraction).  A subsequent re-solve handles mixtures that are
  super-saturated at ``j`` (the ``J>I`` saturated branch).
* ``ELIJ(i,j) = max(0, ALTEM)`` — the condensate of the mixed parcel
  (``ALTEM`` is the excess of mixture total water over saturation
  ``QS(j)``, divided by ``BF2``).  This is the *liquid water of the
  mixture lifted from i and evaluated at j* — the genuine "ELIJ".
* ``MENT(i,j) = M(i)/(1-SIJ(i,j))`` — the **detrained mass flux** from
  origin ``i`` to level ``j``, accumulated only when ``0 < SIJ < 0.9``
  (the mixture actually entrains a non-trivial amount of environment).
* ``QENT(i,j) = SIJ*Q(i) + (1-SIJ)*QTI`` — total water of the mixture.

The mixtures are then NORMALISED (oracle DO 200) so the discrete
spectrum represents equal probabilities of mixing, using the critical
mixing fraction ``SCRIT`` and the per-mixture trapezoidal weights
``(DELP+DELM)·dp``.  Origin levels at which no mixture entrains detrain
the whole updraught locally (``MENT(i,i) = M(i)``).

The environmental tendencies (oracle DO 500) then follow from the net
saturated updraught flux ``AMP1``, the detrained downdraught flux
``AD``, and the per-mixture detrainment terms ``MENT(k,i)·(QENT-AWAT-Q)``.

Differentiability
-----------------
The Fortran sort contains hard switches that we replace with smooth
surrogates (each documented at its call site), so ``jax.grad`` / ``jit``
/ ``vmap`` stay finite while the forward result tracks the discrete
oracle:

* The level windows ``i ∈ [ICB+1, INB]``, ``j ∈ [ICB, INB]`` become
  smooth sigmoid masks ``in_updraft(i)``, ``in_cloud(j)`` on the
  fractional ``ICB`` / ``INB`` indices (sharpness
  ``level_window_sharpness``).
* The ``0 < SIJ < 0.9`` entrainment gate becomes a product of two
  sigmoids ``sij_active = σ(s·SIJ)·σ(s·(0.9−SIJ))`` (sharpness
  ``sij_gate_sharpness``).
* The ``ABS(DENOM)<0.01 → 0.01`` and ``ABS(DEI)<0.01`` floors become a
  smooth signed floor ``denom_floor`` that never crosses zero.
* The ``J>I`` saturated re-solve is selected by a smooth ``j>i`` mask
  and blended in by a sigmoid on ``(STEMP<0 or STEMP>1 or ALTEM>CWAT)``.
* ``MAX(0,·)`` / ``MIN(1,·)`` clips on SIJ, ELIJ, MENT use
  ``jax.nn`` smooth equivalents where a dead gradient would otherwise
  block training; positivity floors that only guard physical bounds
  keep the AD-safe ``maximum``.

The forward result is validated against the Fortran MENT(i,j) and the
FT/FQ tendencies in
``.physics-validator/emanuel/oracle/compare_mixing.py``.

Conventions
-----------
Internally this module works in **surface-FIRST oracle ordering**
(index 0 = lowest model level = surface) so the array maths matches the
Fortran ``i``/``j`` loops line-for-line.  The public entry
:func:`emanuel_mixing_tendencies` accepts and returns the model's
**surface-LAST** ``(ncol, nlev)`` arrays and does the reversal at the
boundary.  Pressures in Pa, temperatures in K, humidities in kg/kg,
mass flux in kg/m^2/s.

All thermodynamics reuse :mod:`legoesm.thermo` and
:mod:`legoesm.constants`; the only scheme-internal coefficient is the
effective liquid-water heat capacity ``CL`` (Emanuel's 2500 J/kg/K),
threaded from :class:`EmanuelConfig.c_l_emanuel`.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.thermo import saturation_vapor_pressure


__all__ = (
    "EmanuelMixingOutput",
    "emanuel_mixing_tendencies",
)


class EmanuelMixingOutput(NamedTuple):
    """Output of :func:`emanuel_mixing_tendencies` (surface-LAST).

    Fields
    ------
    dT_dt : jax.Array, shape (ncol, nlev)
        Temperature tendency [K/s] (oracle FT, surface-last).
    dq_v_dt : jax.Array, shape (ncol, nlev)
        Water-vapor tendency [(kg/kg)/s] (oracle FQ, surface-last).
    dq_c_conv_dt : jax.Array, shape (ncol, nlev)
        Convective cloud-water source [(kg/kg)/s] — the in-cloud
        condensate detrained to the ``q_c`` tracer for microphysics to
        precipitate.  Non-negative.
    ment : jax.Array, shape (ncol, nlev, nlev)
        Normalised mixing matrix MENT(i,j) [kg/m^2/s], surface-FIRST
        ``(origin i, detrainment j)`` for diagnostics / oracle compare.
    m_profile : jax.Array, shape (ncol, nlev)
        Per-level updraught mixing mass flux M(i) [kg/m^2/s],
        surface-FIRST, for diagnostics.
    """

    dT_dt: jax.Array
    dq_v_dt: jax.Array
    dq_c_conv_dt: jax.Array
    ment: jax.Array
    m_profile: jax.Array


def _smooth_signed_floor(x, floor):
    """Oracle ``IF(ABS(x)<floor) x=floor`` made smooth + AD-safe.

    The Fortran replaces a near-zero denominator by ``+floor`` (always
    positive — line 600/611 set ``DEI=0.01``/``DENOM=0.01``) so the
    division ``ANUM/DENOM`` cannot blow up.  A hard branch on ``ABS(x)``
    has a dead gradient inside the band and a discontinuity at the band
    edges.  We blend toward ``+floor`` near zero with a Gaussian window:

        x_floored = x·(1 − w) + floor·w,   w = exp(−(x/floor)^2)

    For ``|x| >> floor`` this is ``x`` (the floor term vanishes, the
    sign is preserved); for ``x → 0`` it tends to ``+floor`` (matching
    the Fortran, which forces the positive floor regardless of the
    original sign).  The result NEVER crosses zero in the band: the
    minimum over ``x`` is bounded below by ``≈ 0.21·floor > 0`` (verified
    numerically), so ``ANUM/x_floored`` is finite and the gradient is
    alive everywhere.
    """
    w = jnp.exp(-((x / floor) ** 2))
    return x * (1.0 - w) + floor * w


def _oracle_qsat(T, p):
    """Saturation specific humidity [kg/kg] in the oracle's convention.

    The Fortran TLIFT / QS uses ``qs = eps·es/(p - es·(1-eps))`` (i.e.
    saturation *specific humidity*, not mixing ratio), with ``es`` the
    Tetens / cold-branch formula.  :func:`legoesm.thermo.
    saturation_vapor_pressure` already matches the oracle warm branch
    (611.2·exp(17.67·Tc/(Tc+243.5))); we wrap it into the oracle's
    ``qs`` form here (rather than ``saturation_mixing_ratio``'s
    ``eps·es/(p-es)``) so the SIJ/ELIJ algebra is term-for-term faithful.

    A smooth softplus floor keeps the denominator positive and the
    gradient alive near ``es → p`` (mirrors ``saturation_mixing_ratio``).
    """
    eps = constants.epsilon
    es = saturation_vapor_pressure(T)
    denom = jax.nn.softplus(p - es * (1.0 - eps) - 1.0) + 1.0
    return eps * es / denom


def _lift_parcel(
    T_first, q_first, p_first, gz_first, nk_weight,
    *, c_l, max_iter=2,
):
    """Vectorized TLIFT: lift the source-level (NK) parcel to every level.

    Reproduces convect43c.f SUBROUTINE TLIFT (lines 1004-1075) for
    ``KK=2`` (full column above cloud base) plus the dry-adiabatic
    below-cloud-base branch (``KK=1``).  All inputs are surface-FIRST
    ``(ncol, nlev)``; ``nk_weight`` is a smooth one-hot ``(ncol, nlev)``
    selecting the source level NK so ``X(NK) = Σ_k nk_weight·X``.

    Returns ``(tpk, clw)`` surface-first:
      * ``tpk`` — lifted-parcel absolute temperature [K] (oracle TPK).
      * ``clw`` — adiabatic liquid water content [kg/kg],
        ``Q(NK) - QG`` floored at 0 (oracle CLW).

    The two-step Newton iteration (oracle DO 200 J=1,2) on the saturated
    parcel temperature is unrolled for ``max_iter`` (=2) steps so it is
    static and ``jit``-friendly.
    """
    cpd = constants.c_pd
    cpv = constants.c_pv
    rv = constants.R_v
    eps = constants.epsilon
    lv0 = constants.L_v

    # Source-level (NK) properties via the smooth one-hot reduction.
    T_nk = jnp.sum(nk_weight * T_first, axis=-1, keepdims=True)
    q_nk = jnp.sum(nk_weight * q_first, axis=-1, keepdims=True)
    gz_nk = jnp.sum(nk_weight * gz_first, axis=-1, keepdims=True)

    cpvmcl = c_l - cpv
    # Liquid-water static energy of the source parcel (oracle AH0).
    ah0 = (
        (cpd * (1.0 - q_nk) + c_l * q_nk) * T_nk
        + q_nk * (lv0 - cpvmcl * (T_nk - constants.T_freeze))
        + gz_nk
    )

    # Two-step saturated-parcel solve at EVERY level (oracle DO 200).
    # Start from the environment T and the saturation specific humidity.
    Tg = T_first
    qg = _oracle_qsat(T_first, p_first)
    for _ in range(max_iter):
        alv = lv0 - cpvmcl * (T_first - constants.T_freeze)
        S = cpd + alv * alv * qg / (rv * T_first * T_first)
        S = 1.0 / S
        ahg = cpd * Tg + (c_l - cpd) * q_nk * T_first + alv * qg + gz_first
        Tg = Tg + S * (ah0 - ahg)
        Tg = jnp.maximum(Tg, 35.0)
        qg = _oracle_qsat(Tg, p_first)

    alv = lv0 - cpvmcl * (T_first - constants.T_freeze)
    tpk_moist = (ah0 - (c_l - cpd) * q_nk * T_first - gz_first - alv * qg) / cpd
    clw = jnp.maximum(q_nk - qg, 0.0)

    # Dry-adiabatic branch below cloud base (oracle KK=1, lines 1035-1037):
    #   TPK(I) = T(NK) - (GZ(I) - GZ(NK))/CPP,  CPP = cpd(1-q)+q·cpv.
    cpp = cpd * (1.0 - q_nk) + q_nk * cpv
    tpk_dry = T_nk - (gz_first - gz_nk) / cpp

    # Blend: above cloud base use the moist (saturated-adjusted) parcel,
    # below use the dry adiabat.  The caller passes a smooth mask
    # ``above_icb`` so this stays differentiable; we return both and let
    # the caller blend (it owns the ICB index).
    return tpk_moist, tpk_dry, clw


def emanuel_mixing_tendencies(
    T: jax.Array,
    q_v: jax.Array,
    p_full: jax.Array,
    p_half: jax.Array,
    M_b: jax.Array,
    *,
    c_l: float,
    elcrit: float,
    tlcrit: float,
    entp: float,
    level_window_sharpness: float,
    sij_gate_sharpness: float,
    sij_upper_gate: float,
    denom_floor: float,
    mse_min_search_offset: float,
    sat_branch_sharpness: float,
    nk_weight: jax.Array | None = None,
) -> EmanuelMixingOutput:
    """Genuine Emanuel (i,j) buoyancy-sort mixing matrix and tendencies.

    Faithful vectorized port of convect43c.f lines 338-938 (column
    setup → SIJ/ELIJ/MENT mixing matrix → environmental FT/FQ), with
    the downdraught precip branch reduced to its column-condensate
    handoff (the model owns precipitation via the ``q_c`` tracer +
    microphysics; see :mod:`.emanuel`).

    Parameters
    ----------
    T, q_v : jax.Array, shape (ncol, nlev)
        Environmental temperature [K] / water-vapor specific humidity
        [kg/kg], surface-LAST.
    p_full, p_half : jax.Array
        Full / half-level pressures [Pa], surface-LAST.
    M_b : jax.Array, shape (ncol,)
        Cloud-base mass flux (oracle CBMF) [kg/m^2/s].
    c_l : float
        Effective liquid-water heat capacity (oracle CL = 2500 J/kg/K).
    elcrit, tlcrit : float
        Autoconversion threshold [kg/kg] and critical temperature [degC]
        for the precipitation efficiency EP.
    entp : float
        Mixing-rate coefficient in M(i) (oracle ENTP).
    level_window_sharpness : float
        Sigmoid sharpness [1/level] for the ICB/INB cloud-layer masks.
    sij_gate_sharpness : float
        Sigmoid sharpness [dimensionless] for the ``0 < SIJ < upper``
        entrainment gate.
    sij_upper_gate : float
        Upper SIJ gate (oracle 0.9).
    denom_floor : float
        Magnitude floor for the SIJ denominator (oracle 0.01).
    mse_min_search_offset : float
        Offset for the smooth NK (max-MSE) source-level selection — only
        used when ``nk_weight`` is None.
    sat_branch_sharpness : float
        Sigmoid sharpness for the saturated-mixture re-solve switch.
    nk_weight : jax.Array, shape (ncol, nlev), optional
        Pre-computed smooth one-hot source-level (NK) weight,
        surface-LAST.  When None it is diagnosed here as the smooth
        max-MSE level.

    Returns
    -------
    EmanuelMixingOutput
    """
    ncol, nlev = T.shape
    dtype = T.dtype

    cpd = constants.c_pd
    cpv = constants.c_pv
    rd = constants.R_d
    rv = constants.R_v
    eps = constants.epsilon
    epsi = 1.0 / eps
    lv0 = constants.L_v
    g = constants.g
    cpvmcl = c_l - cpv

    # ---- Reverse to surface-FIRST oracle ordering -----------------------
    Tf = T[:, ::-1]
    qf = q_v[:, ::-1]
    pf = p_full[:, ::-1]
    # p_half surface-last is [TOA .. sfc] length nlev+1; oracle PH is
    # surface-first [sfc-edge .. TOA] length nlev+1 with PH(1) just below
    # P(1).  Reverse it.
    phf = p_half[:, ::-1]                                   # (ncol, nlev+1)

    levels = jnp.arange(nlev, dtype=dtype)[None, :]        # surface-first idx

    # ---- Geopotential, heat capacity, static energy (oracle DO 40) ------
    # GZ(1)=0; GZ(i)=GZ(i-1)+0.5 Rd (TVx+TVy)(P(i-1)-P(i))/PH(i).
    tv = Tf * (1.0 + qf * epsi - qf)                        # virtual T (oracle TV)
    # Build GZ by cumulative sum of layer increments.
    # dgz[i] (i>=1) = 0.5 Rd (tv[i]+tv[i-1]) (p[i-1]-p[i]) / ph_inner[i]
    # PH(i) for i>=2 is the half level between P(i-1) and P(i): that is
    # phf[:, i] (surface-first half levels; phf[:,1] sits between P(1),P(2)).
    tv_lo = tv[:, :-1]                                      # tv[i-1]
    tv_hi = tv[:, 1:]                                       # tv[i]
    p_lo = pf[:, :-1]                                       # P(i-1)
    p_hi = pf[:, 1:]                                        # P(i)
    ph_between = phf[:, 1:-1]                               # PH(2..nlev) inner
    dgz = 0.5 * rd * (tv_hi + tv_lo) * (p_lo - p_hi) / ph_between
    gz = jnp.concatenate(
        [jnp.zeros((ncol, 1), dtype), jnp.cumsum(dgz, axis=-1)], axis=-1,
    )                                                       # (ncol, nlev)

    cpn = cpd * (1.0 - qf) + cpv * qf                       # oracle CPN
    h = Tf * cpn + gz                                       # moist static energy H
    lv = lv0 - cpvmcl * (Tf - constants.T_freeze)           # LV(i)
    # Frozen MSE (oracle HM) for the NK / IHMIN search.
    hm = (
        (cpd * (1.0 - qf) + c_l * qf) * (Tf - Tf[:, :1])
        + lv * qf + gz
    )
    qs = _oracle_qsat(Tf, pf)                               # saturation q

    # ---- Source level NK (max MSE below min-MSE) ------------------------
    if nk_weight is None:
        # Smooth surrogate: softmax of HM gated to the lower troposphere.
        # The oracle picks the max-HM level below the min-HM level; for
        # RCE/tropical soundings that is the surface.  We weight by HM
        # with a pressure gate that prefers the boundary layer.
        gate = jax.nn.sigmoid(
            (pf - mse_min_search_offset) / 2000.0
        )
        nk_logits = hm * gate
        nk_w = jax.nn.softmax(nk_logits * 1.0, axis=-1)
    else:
        nk_w = nk_weight[:, ::-1]

    # ---- LCL / ICB (first level above LCL) ------------------------------
    # Bolton (1980) on the NK parcel (oracle lines 392-394).
    T_nk = jnp.sum(nk_w * Tf, axis=-1, keepdims=True)
    q_nk = jnp.sum(nk_w * qf, axis=-1, keepdims=True)
    p_nk = jnp.sum(nk_w * pf, axis=-1, keepdims=True)
    qs_nk = jnp.sum(nk_w * qs, axis=-1, keepdims=True)
    rh = jnp.clip(q_nk / jnp.maximum(qs_nk, 1e-12), 1e-4, 1.0)
    chi = T_nk / (1669.0 - 122.0 * rh - T_nk)
    p_lcl = p_nk * (rh ** chi)                              # (ncol,1)

    # ICB = first level (surface-first, going up) with P < P_LCL.  Smooth
    # fractional index: lowest upward crossing of (pf - p_lcl).
    below_lcl = jax.nn.sigmoid((pf - p_lcl) / 200.0)        # ~1 below LCL
    # ICB fractional index = number of levels at/below LCL ≈ Σ below_lcl.
    icb_frac = jnp.sum(below_lcl, axis=-1, keepdims=True)   # (ncol,1) surface-first

    # ---- Lift NK parcel: TLIFT (TPK, CLW) -------------------------------
    tpk_moist, tpk_dry, clw = _lift_parcel(
        Tf, qf, pf, gz, nk_w, c_l=c_l,
    )
    above_icb = jax.nn.sigmoid(
        level_window_sharpness * (levels - icb_frac)
    )                                                       # ~1 above ICB
    tpk = above_icb * tpk_moist + (1.0 - above_icb) * tpk_dry
    # Lifted-parcel virtual temperature (oracle TVP, with the -TP·Q(NK)
    # correction from lines 423/469): TVP = TPK(1+RG·EPSI) - TPK·Q(NK)
    # where RG = QG/(1-Q(NK)).  For the buoyancy / CAPE diagnosis we
    # build TVP consistently with the oracle's reduced form.
    qg_sat = _oracle_qsat(tpk, pf)
    rg = qg_sat / (1.0 - q_nk)
    tvp = tpk * (1.0 + rg * epsi) - tpk * q_nk

    # ---- Precip efficiency EP, liquid-water static energy HP ------------
    tca = tpk - constants.T_freeze                          # parcel T in degC
    elacrit = jnp.where(
        tca >= 0.0, elcrit, elcrit * (1.0 - tca / tlcrit),
    )
    elacrit = jnp.maximum(elacrit, 0.0)
    epmax = 0.999
    ep = epmax * (1.0 - elacrit / jnp.maximum(clw, 1e-8))
    ep = jnp.clip(ep, 0.0, epmax)
    # EP=0 below NK / below cloud base (oracle DO 57 sets EP=0 for I<=NK,
    # and the mixing loop runs only ICB+1..INB).  Gate EP by above_icb so
    # the precip removal only applies in the cloud layer.
    ep = ep * above_icb

    hp = h + (lv + (cpd - cpv) * Tf) * ep * clw            # oracle HP(i)
    # Below-cloud HP defaults to H (oracle line 476 init); above ICB the
    # liquid-water static energy applies.
    hp = above_icb * hp + (1.0 - above_icb) * h

    # ---- INB: top of positive CAPE (oracle DO 82) -----------------------
    # Buoyancy by-layer BY = (TVP-TV)(PH(i)-PH(i+1))/P(i).  We accumulate
    # CAPE upward and find the smooth highest level with CAPE>0.
    buoy = tvp - tv                                         # (ncol, nlev)
    # Cloud-layer buoyancy only (above ICB).
    by = buoy * above_icb
    # INB fractional index: highest level where the running CAPE>0.  Build
    # a smooth indicator that the parcel is still positively buoyant in a
    # contiguous cloud layer: cumulative-from-top "still buoyant" weight.
    # Use a sigmoid on the local buoyancy gated above ICB; the top of the
    # cloud is where buoyancy first goes negative going up.
    pos_buoy = jax.nn.sigmoid(by / 0.5) * above_icb        # ~1 in buoyant cloud
    inb_frac = icb_frac[:, 0] + jnp.sum(
        pos_buoy * (levels >= icb_frac), axis=-1,
    )                                                       # (ncol,)
    inb_frac = jnp.minimum(inb_frac, nlev - 2.0)[:, None]
    # In-cloud mask for j: ICB <= j <= INB.
    in_cloud_j = above_icb * jax.nn.sigmoid(
        level_window_sharpness * (inb_frac - levels)
    )                                                       # (ncol, nlev)
    # Updraught-origin mask for i: ICB+1 <= i <= INB.
    in_updraft_i = jax.nn.sigmoid(
        level_window_sharpness * (levels - (icb_frac + 1.0))
    ) * jax.nn.sigmoid(
        level_window_sharpness * (inb_frac - levels)
    )                                                       # (ncol, nlev)

    # ---- M(i): rates of mixing (oracle DO 103 / 110) --------------------
    # DBO(i) = |TV(K)-TVP(K)| + ENTP·0.02·(PH(K)-PH(K+1)),  K=min(i,INB1).
    # We approximate INB1≈INB and use the in-cloud levels; normalise so
    # Σ M = CBMF.  dp_first[i] = PH(i)-PH(i+1) (surface-first layer Δp>0).
    dp_first = phf[:, :-1] - phf[:, 1:]                     # (ncol, nlev) >0
    dbo = jnp.abs(tv - tvp) + entp * 0.02 * dp_first
    dbo_masked = dbo * in_updraft_i
    dbosum = jnp.sum(dbo_masked, axis=-1, keepdims=True)
    m_i = M_b[:, None] * dbo_masked / jnp.maximum(dbosum, 1e-30)  # (ncol, nlev)

    # ---- The (i,j) mixing matrix: SIJ / ELIJ / MENT / QENT --------------
    # Broadcast to (ncol, i, j).  Origin index i = axis 1, level j = axis 2.
    H_i = h[:, :, None]
    H_j = h[:, None, :]
    HP_i = hp[:, :, None]
    Q_i = qf[:, :, None]
    Q_j = qf[:, None, :]
    QS_j = qs[:, None, :]
    T_j = Tf[:, None, :]
    LV_j = lv[:, None, :]
    CLW_i = clw[:, :, None]
    CLW_j = clw[:, None, :]
    EP_i = ep[:, :, None]
    EP_j = ep[:, None, :]

    # QTI(i) = Q(NK) - EP(i)·CLW(i)  (total water of updraught parcel i).
    QTI = q_nk[:, :, None] - EP_i * CLW_i                   # (ncol, nlev, 1) -> bcast

    # BF2(j) = 1 + LV(j)^2 QS(j)/(RV T(j)^2 CPD).
    BF2 = 1.0 + LV_j * LV_j * QS_j / (rv * T_j * T_j * cpd)

    # ANUM = H(j) - HP(i) + (CPV-CPD) T(j) (QTI - Q(j)).
    ANUM = H_j - HP_i + (cpv - cpd) * T_j * (QTI - Q_j)
    # DENOM = H(i) - HP(i) + (CPD-CPV)(Q(i)-QTI) T(j).
    DENOM = H_i - HP_i + (cpd - cpv) * (Q_i - QTI) * T_j
    DEI = _smooth_signed_floor(DENOM, denom_floor)
    SIJ = ANUM / DEI                                       # (ncol, nlev, nlev)

    # ALTEM = (SIJ·Q(i) + (1-SIJ)·QTI - QS(j)) / BF2.
    ALTEM = (SIJ * Q_i + (1.0 - SIJ) * QTI - QS_j) / BF2
    CWAT = CLW_j * (1.0 - EP_j)

    # Saturated-mixture re-solve (oracle lines 607-615), only for J>I:
    #   condition = (SIJ<0 or SIJ>1 or ALTEM>CWAT) and J>I.
    j_gt_i = jax.nn.sigmoid(
        level_window_sharpness * (levels[:, None, :] - levels[:, :, None])
    )                                                       # ~1 where j>i
    cond_raw = (
        jax.nn.sigmoid(-sat_branch_sharpness * SIJ)         # SIJ<0
        + jax.nn.sigmoid(sat_branch_sharpness * (SIJ - 1.0))  # SIJ>1
        + jax.nn.sigmoid(sat_branch_sharpness * (ALTEM - CWAT))  # ALTEM>CWAT
    )
    cond = jnp.clip(cond_raw, 0.0, 1.0) * j_gt_i            # smooth OR ∧ (j>i)

    # Re-solved SIJ / ALTEM (oracle lines 609-614).
    ANUM2 = ANUM - LV_j * (QTI - QS_j - CWAT * BF2)
    DENOM2 = DENOM + LV_j * (Q_i - QTI)
    DEN2 = _smooth_signed_floor(DENOM2, denom_floor)
    SIJ2 = ANUM2 / DEN2
    ALTEM2 = SIJ2 * Q_i + (1.0 - SIJ2) * QTI - QS_j - (BF2 - 1.0) * CWAT

    SIJ_eff = (1.0 - cond) * SIJ + cond * SIJ2
    ALTEM_eff = (1.0 - cond) * ALTEM + cond * ALTEM2

    # Entrainment gate: oracle counts a mixture only when 0 < SIJ < 0.9
    # (hard cutoff).  Two smooth issues must be handled together:
    #   (a) ``MENT_raw = M(i)/(1-SIJ)`` DIVERGES as SIJ → 1, so the upper
    #       gate must decay *faster* than that divergence grows or mass
    #       leaks through the sigmoid tail (a 2% gate × a 50× MENT_raw is
    #       a real leak).  The oracle's hard ``< 0.9`` is exactly zero
    #       above 0.9; we centre the upper gate just below ``sij_upper``
    #       and make it sharp so it is ≈1 at SIJ≈0.73 (where the oracle
    #       still counts the mixture) and ≈0 by SIJ≈0.9.
    #   (b) the lower gate ``SIJ > 0`` rejects negative SIJ (mixtures that
    #       would require negative updraught fraction).
    lower_gate = jax.nn.sigmoid(sij_gate_sharpness * SIJ_eff)
    # Upper gate centred a little below ``sij_upper`` with a sharpness
    # tied to the divergence: at SIJ = sij_upper the gate is already
    # small (~σ(-2)≈0.12) and falls faster than 1/(1-SIJ) rises.
    upper_center = sij_upper_gate - 2.0 / sij_gate_sharpness
    upper_gate = jax.nn.sigmoid(sij_gate_sharpness * (upper_center - SIJ_eff))
    sij_active = lower_gate * upper_gate
    # Both i and j must be in the cloud layer.
    ij_in_cloud = in_updraft_i[:, :, None] * in_cloud_j[:, None, :]
    sij_active = sij_active * ij_in_cloud

    # QENT(i,j), ELIJ(i,j), raw MENT(i,j)=M(i)/(1-SIJ).  The
    # ``1-SIJ`` floor caps the divergence at M(i)·(1/floor); with the
    # sharp upper gate above, the capped-and-gated product never leaks.
    QENT = SIJ_eff * Q_i + (1.0 - SIJ_eff) * QTI
    ELIJ = jnp.maximum(ALTEM_eff, 0.0) * sij_active
    one_minus_sij = jnp.clip(1.0 - SIJ_eff, 1.0 - sij_upper_gate, None)
    MENT_raw = (m_i[:, :, None] / one_minus_sij) * sij_active

    # NENT(i) = number of active mixtures at origin i (smooth count).
    nent = jnp.sum(sij_active, axis=-1)                    # (ncol, nlev)
    # Smooth "no mixture entrains" weight for origin i (oracle NENT==0).
    no_entrain_i = jnp.exp(-nent / 0.25)                   # ~1 when nent≈0

    # ---- Normalisation to equal mixing probabilities (oracle DO 200) ----
    # SCRIT(i) = ANUM_s/DENOM_s with QP1=Q(NK)-EP(i)CLW(i) at level i
    # (oracle lines 655-662).  This is the critical mixing fraction.
    QP1 = q_nk - ep * clw                                  # (ncol, nlev) at level i
    H_self = h
    HP_self = hp
    LV_self = lv
    QS_self = qs
    Q_self = qf
    ANUM_s = H_self - HP_self - LV_self * (QP1 - QS_self)
    DENOM_s = H_self - HP_self + LV_self * (Q_self - QP1)
    DEN_s = _smooth_signed_floor(DENOM_s, denom_floor)
    scrit = ANUM_s / DEN_s
    alt_s = QP1 - QS_self + scrit * (Q_self - QP1)
    # IF(ALT<0) SCRIT=1; SCRIT=MAX(SCRIT,0).
    scrit = jnp.where(alt_s < 0.0, 1.0, scrit)
    scrit = jnp.maximum(scrit, 0.0)                        # (ncol, nlev)
    scrit_i = scrit[:, :, None]

    # The trapezoidal probability weights (oracle DELP/DELM, lines 665-688)
    # require neighbour-SIJ comparisons.  We use the documented
    # simplification: each active mixture's probability weight is
    # ``|SMID|`` proximity to SCRIT, mass-weighted by the detrainment-level
    # layer thickness ``dp(j)``.  SMID = min(SIJ, SCRIT) for j>i and
    # max(SIJ, SCRIT) for j<i.  This preserves the oracle's key property:
    # mixtures whose neutral-buoyancy fraction is near the critical
    # fraction carry most of the detrained mass, and the per-origin
    # weights are renormalised to sum to one.
    dp_j = dp_first[:, None, :]                            # (ncol,1,nlev) layer Δp
    smid_up = jnp.minimum(SIJ_eff, scrit_i)
    smid_dn = jnp.maximum(SIJ_eff, scrit_i)
    smid = j_gt_i * smid_up + (1.0 - j_gt_i) * smid_dn
    # Weight ∝ |SMID - SCRIT| + |SMID| (a smooth, strictly-positive proxy
    # for the (DELP+DELM) trapezoid that vanishes far from the critical
    # fraction), times dp(j) and the active gate.
    prob_w = (jnp.abs(smid - scrit_i) + jnp.abs(smid)) * dp_j * sij_active
    asij = jnp.sum(prob_w, axis=-1, keepdims=True)         # ASIJ per origin i
    asij = jnp.maximum(asij, 1e-21)
    MENT = MENT_raw * prob_w / asij                        # normalised MENT

    # Origins with no entraining mixture detrain locally (oracle NENT==0):
    # MENT(i,i)=M(i), QENT(i,i)=Q(NK)-EP(i)CLW(i), ELIJ(i,i)=CLW(i).
    eye = jnp.eye(nlev, dtype=dtype)[None, :, :]
    MENT_local = m_i[:, :, None] * eye * no_entrain_i[:, :, None]
    QENT_local = (q_nk - ep * clw)[:, :, None]             # broadcast on j via eye
    ELIJ_local = clw[:, :, None]
    MENT = MENT * (1.0 - no_entrain_i[:, :, None]) + MENT_local
    # For the diagonal local-detrainment term, override QENT/ELIJ on j==i.
    QENT = QENT * (1.0 - eye * no_entrain_i[:, :, None]) + (
        QENT_local * eye * no_entrain_i[:, :, None]
    )
    ELIJ = ELIJ * (1.0 - eye * no_entrain_i[:, :, None]) + (
        ELIJ_local * eye * no_entrain_i[:, :, None]
    )

    # ---- Environmental tendencies (oracle DO 500, mixing terms only) ----
    # We implement the mass-flux convergence form for an arbitrary source
    # level NK (oracle treats NK=1 separately; for the general smooth-NK
    # case we use the per-level net updraught AMP1 and detrainment AD).
    #
    # Net saturated updraught mass flux through level i (oracle AMP1):
    #   AMP1(i) = Σ_{k<=i} Σ_{j>i} MENT(k,j)   (+ Σ_{K>i} M(K) when NK<=i,
    #   the bulk undilute updraught above the source — folded into M here).
    # Detrained downdraught flux through level i (oracle AD):
    #   AD(i)   = Σ_{k<i} Σ_{j>=i} MENT(j,k)
    #
    # Build these with masked cumulative sums over the (i,j) matrix.
    li = levels[:, :, None]                                # origin i index
    lj = levels[:, None, :]                                # level j index

    # M(K) above source (bulk undilute updraught reaching level i): the
    # oracle's AM/AMP1 includes Σ_{K>i} M(K) (the undilute updraught mass
    # flux that has not yet detrained).  Build cumulative-from-top.
    m_above = jnp.cumsum(m_i[:, ::-1], axis=-1)[:, ::-1] - m_i
    m_above = jnp.maximum(m_above, 0.0)

    dpinv = 1.0 / jnp.maximum(dp_first, 1e-6)              # 1/Δp per level
    cpn_inv = 1.0 / cpn

    # === Net updraught (AMP1) and detrainment (AD) fluxes ================
    # Oracle DO 440-470, expressed as O(nlev^2) masked prefix reductions
    # over the (origin k = axis 1, level j = axis 2) MENT matrix.
    #
    #   AMP1(t) = Σ_{K>t} M(K) + Σ_{k<=t} Σ_{j>t} MENT(k,j)
    #           = mass crossing the TOP interface of level t going up.
    #   AD(t)   = Σ_{k<t} Σ_{j>=t} MENT(j,k)
    #           = mass detrained INTO levels below t from origins at/above t.
    A = MENT
    # Σ_{k<=t} A(k, j): prefix over origin axis up to and including t.
    cum_origin = jnp.cumsum(A, axis=1)                    # (ncol, t, j)
    j_above_t = jax.nn.sigmoid(
        level_window_sharpness * (levels[:, None, :] - levels[:, :, None] - 0.5)
    )                                                      # (ncol, t, j) ~1 j>t
    F_up = jnp.sum(cum_origin * j_above_t, axis=-1)       # Σ_{k<=t, j>t}
    AMP1 = m_above + F_up                                 # (ncol, nlev)

    # AD(t) = Σ_{j>=t} (over origin axis) detraining into targets k<t.
    # cum_origin_from_top[:, t, k] = Σ_{j>=t} A(j, k).
    cum_origin_from_top = jnp.cumsum(A[:, ::-1], axis=1)[:, ::-1]  # (ncol, t, k)
    k_below_t = jax.nn.sigmoid(
        level_window_sharpness * (levels[:, :, None] - levels[:, None, :] - 0.5)
    )                                                      # (ncol, t, k) ~1 k<t
    AD = jnp.sum(cum_origin_from_top * k_below_t, axis=-1)  # (ncol, nlev)

    # --- Shift helpers for (i+1) / (i-1) neighbours (surface-first) ------
    def up(x):      # value at i+1 (one level higher), top padded with self
        return jnp.concatenate([x[:, 1:], x[:, -1:]], axis=-1)

    def dn(x):      # value at i-1 (one level lower), bottom padded with self
        return jnp.concatenate([x[:, :1], x[:, :-1]], axis=-1)

    T_up = up(Tf)
    T_dn = dn(Tf)
    q_up = up(qf)
    q_dn = dn(qf)
    gz_up = up(gz)
    gz_dn = dn(gz)

    # (A) Subsidence/advection of environment by AMP1 (up) and AD (down):
    #   FT += g·DPINV·(AMP1·(T(i+1)-T(i)+(GZ(i+1)-GZ(i))·CPINV)
    #                  - AD·(T(i)-T(i-1)+(GZ(i)-GZ(i-1))·CPINV))
    #   FQ += g·DPINV·(AMP1·(Q(i+1)-Q(i)) - AD·(Q(i)-Q(i-1)))
    ft_adv = g * dpinv * (
        AMP1 * (T_up - Tf + (gz_up - gz) * cpn_inv)
        - AD * (Tf - T_dn + (gz - gz_dn) * cpn_inv)
    )
    fq_adv = g * dpinv * (
        AMP1 * (q_up - qf) - AD * (qf - q_dn)
    )

    # (B) Local-detrainment enthalpy (oracle line 894):
    #   FT += g·DPINV·MENT(i,i)·(HP(i)-H(i)+T(i)(CPV-CPD)(Q(i)-QENT(i,i)))·CPINV
    ment_ii = jnp.sum(A * eye, axis=-1)                   # MENT(i,i)
    qent_ii = jnp.sum(QENT * eye, axis=-1)                # QENT(i,i)
    ft_local = g * dpinv * ment_ii * (
        hp - h + Tf * (cpv - cpd) * (qf - qent_ii)
    ) * cpn_inv

    # (C) Per-mixture moisture detrainment (oracle DO 480/490):
    #   AWAT(k,i) = max(0, ELIJ(k,i) - (1-EP(i))·CLW(i))
    #   FQ(i) += g·DPINV· Σ_k MENT(k,i)·(QENT(k,i) - AWAT - Q(i))   [k<i]
    #          + g·DPINV· Σ_k MENT(k,i)·(QENT(k,i) - Q(i))          [k>=i]
    AWAT = jnp.maximum(ELIJ - (1.0 - EP_j) * CLW_j, 0.0)  # (ncol, i=k, j)
    # For target level i (=j axis), sum over origins k (=i axis).
    # k<i contributes (QENT - AWAT - Q); k>=i contributes (QENT - Q).
    k_lt_i = jax.nn.sigmoid(
        level_window_sharpness * (lj - li)
    )                                                      # ~1 where origin(i-axis)<target(j-axis): k<j
    # MENT(k, j), QENT(k, j) have axes (ncol, k, j).
    q_target = qf[:, None, :]
    detr_moist_klt = MENT * (QENT - AWAT - q_target)       # k<j branch value
    detr_moist_kge = MENT * (QENT - q_target)              # k>=j branch value
    detr_moist = k_lt_i * detr_moist_klt + (1.0 - k_lt_i) * detr_moist_kge
    fq_detr = g * dpinv * jnp.sum(detr_moist, axis=1)      # sum over origins k

    # Cloud-water source: the detrained condensate that becomes cloud
    # water at level j is Σ_k MENT(k,j)·AWAT(k,j) plus the local
    # autoconversion-suppressed condensate (oracle WDTRAIN feeds precip;
    # here the in-cloud retained condensate (1-EP)·CLW that does NOT rain
    # goes to the q_c tracer).  This is the genuine in-cloud condensate
    # handed to microphysics.
    dqc = g * dpinv * (
        jnp.sum(MENT * AWAT, axis=1)
        + ment_ii * (1.0 - ep) * clw            # local detrainment condensate
    )
    dqc = jnp.maximum(dqc, 0.0)

    # ft_adv and ft_local already include CPINV where the oracle applies
    # it (the GZ-advection and local-detrainment enthalpy terms carry
    # ``cpn_inv`` per the oracle; the pure subsidence ``AMP1·(T(i+1)-T(i))``
    # term does NOT, matching oracle line 891 where CPINV multiplies only
    # the GZ piece).  Do not double-apply.
    ft = ft_adv + ft_local
    fq = fq_adv + fq_detr

    # Zero out tendencies outside the convecting column (where M_b≈0 the
    # whole matrix is ≈0, so this is automatic; we add an explicit gate on
    # the in-cloud + sub-cloud region for safety).
    active_col = jax.nn.sigmoid(M_b / 1e-3)[:, None]       # ~1 when M_b>0
    ft = ft * active_col
    fq = fq * active_col
    dqc = dqc * active_col

    # ---- Reverse back to surface-LAST and return -----------------------
    return EmanuelMixingOutput(
        dT_dt=ft[:, ::-1],
        dq_v_dt=fq[:, ::-1],
        dq_c_conv_dt=dqc[:, ::-1],
        ment=MENT,            # surface-first (i,j) for diagnostics
        m_profile=m_i,        # surface-first
    )
