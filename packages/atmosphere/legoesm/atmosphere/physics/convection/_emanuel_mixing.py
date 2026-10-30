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

Differentiability (RULE 5)
--------------------------
Integer indices (``NK``, ``IHMIN``, ``ICB``, ``INB``, ``INB1``) are index
picks: discrete here as in the oracle (``argmax`` / ``take_along_axis``),
gradients flow through the gathered values, and the two index rejection
gates (``IHMIN = NL-1``, ``ICB >= NL-1``) stay hard.  Every switch on traced
state is smooth, its sharpness an :class:`EmanuelConfig` field whose
``math.inf`` value (``1e3``/``1e12`` for the five older sigmoid fields) makes
that switch the oracle's (to <= 1 ulp: e.g. ``ANUM*(1/D)`` for ``ANUM/D``);
all of them at those values reproduce the oracle (<= 1e-10):

* ``level_window_sharpness`` -- the ``ICB``/``INB`` level windows, evaluated at
  half-integer offsets (``sigmoid(s*(level - ICB + 1/2))``) so they tend to
  exactly 0/1;
* ``sij_gate_sharpness`` -- the ``0 < SIJ < sij_upper_gate`` entrainment band;
* ``sat_branch_sharpness`` -- the saturated re-solve switch;
* ``strict_index_sharpness`` -- the strict ``J > I`` / ``K < I`` inequalities;
* ``lcl_pressure_sharpness`` -- the sub-cloud ``DTPBL`` layer above ``NK``;
* ``source_temperature_gate_sharpness``, ``source_humidity_gate_sharpness``,
  ``plcl_gate_sharpness`` -- the IFLAG 0/2 rejections ``T(NK) < 250 K``,
  ``Q(NK) <= 0``, ``PLCL`` outside [200, 2000) hPa (weights on CBMF); the
  humidity field also smooths the ``RH = 1`` guard for ``Q(NK) <= 0``;
* ``stable_exit_sharpness``, ``cold_start_sharpness`` -- the cold-start
  stable exit (``TVP(ICB) <= TV(ICB) - DTMAX`` and ``CBMF = 0``, the latter as
  ``exp(-s*CBMF_old)``);
* ``cbmf_sharpness`` -- ``CBMF = MAX(..., 0)`` (softplus);
* ``precip_efficiency_sharpness`` -- the ``ELACRIT`` and ``EP`` clips;
* ``condensate_sharpness`` -- ``CLW``, ``ELIJ`` and ``AWAT`` positivity and
  the ``SCRIT = 1`` humidity test;
* ``scrit_floor_sharpness`` -- ``MAX(SCRIT, 0)``;
* ``frac_sharpness``, ``defrac_floor_sharpness`` -- the ``FRAC`` clip and its
  ``MAX(.., 0.001)`` denominator;
* ``tlift_floor_sharpness``, ``tlift_freeze_sharpness`` -- TLIFT's
  ``MAX(TG, 35)`` and its warm/cold vapour-pressure switch at 0 C;
* ``qsat_floor_sharpness`` -- the 180 K floor of the host ``QS``;
* ``denom_floor_sharpness`` -- the ``ABS(DENOM) < 0.01`` floors, blended in
  the reciprocal (see :func:`_floor_recip`);
* ``nent_count_sharpness``, ``bsum_sharpness`` -- the ``NENT = 0`` /
  ``BSUM < 1e-18`` local-detrainment switch (the BSUM sigmoid is centred
  ``_BSUM_CENTRE_WIDTHS/bsum_sharpness`` above 1e-18, so a row with BSUM
  exactly 0 and entrainment counted gets local-detrainment weight
  ``1 - sigmoid(-12)``, not one half);
* ``asij_floor_sharpness`` -- ``MAX(ASIJ, 1e-21)``;
* ``sort_sharpness`` -- the ``DO 200`` SMIN march (its min/max/abs and the
  strict SMIN tests) and the ``[0, 1]`` clip of the stored ``SIJ``.

KNOWN DIFFERENCE: at the default sharpnesses the port does not reproduce the
oracle; the per-column gap (FT, FQ, CBMF) is pinned in
``tests/atmosphere/oracle_fixtures/emanuel_convect43c_default_gaps.json``.
Setting every field above to its hard value recovers the oracle (<= 1e-10).
Invariants the smoothing breaks: a column the oracle rejects on a STATE gate
(IFLAG 0 temperature/humidity, IFLAG 2, the stable exit) carries a small
positive CBMF and so small nonzero tendencies (softplus CBMF and sigmoid
weights; in fp32 a weight can underflow to exactly 0); the index gates
still give exactly 0; ``CLW``/``ELIJ``/
``AWAT`` carry a softplus offset of up to ``ln2/condensate_sharpness``
(7e-7 kg/kg) where the oracle has exactly 0.  The ENTS enthalpy closure
still holds exactly; column water closes exactly where the column's net
vapour loss is >= ``1/rain_column_sharpness`` and the rain falls short by
at most ``4/(27 rain_column_sharpness)`` below that (see
:func:`legoesm.core.smooth.positive_part_c1`, used by :mod:`.emanuel`).

Kept hard: the index picks above and their consequences -- ``CAPEM`` and
``BYP`` are gathered at the last level of positive running CAPE, so they
jump when that level jumps (same class as ``INB``); and, as division guards
on values that never select a branch on an atmospheric column,
``MAX(CLW, 1e-8)`` in ``EP``, the ``1e-12`` floor on the smooth ``RH`` (an
``RH**chi`` singularity guard), the non-negativity of the running ``M`` sum
above a level (rounding only), and the ``DPINV`` and ``DTPBL`` denominator
floors.

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

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from legoesm import constants
from legoesm.core.smooth import sigmoid_switch, smooth_relu
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
        "dT_dt": "K/s", "dq_v_dt": "kg/kg/s",
        "cbmf": "kg/m^2/s (oracle CBMF out)",
        "ment": "kg/m^2/s ((i,j) mixing matrix, diagnostic)",
        "m_profile": "kg/m^2/s (per-level updraught mixing mass flux)",
        "convective_layer_mask": "1 (levels 1..INB)",
    },
    "sign_convention": (
        "Heats and dries/moistens the environment by the oracle's mass-flux "
        "convergence; the mixture condensate AWAT detrained below its origin "
        "and the precipitating fraction EP*CLW leave the vapour/cloud budget "
        "as precipitation (the oracle's WDTRAIN); column enthalpy is "
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
        "limit of the smoothing fields; a stable column returns zero there."
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
# BSUM sigmoid centre offset in transition widths (reviewer pick 2026-10-30,
# codex + GLM): centred on 1e-18 a row with BSUM exactly 0 got sigmoid(0) =
# 1/2 at every finite sharpness (measured: one-field FT gap 0.71 on fixture
# column 12 for 1e4 <= s <= 1e16); 12 widths above it gives 8.7e-6.
_BSUM_CENTRE_WIDTHS = 12.0
# Rows with ASIJ above this many floors divide by ASIJ directly (the smooth
# floor equals ASIJ there to < 1e-30 relative at any sharpness >= 1/10).
_ASIJ_SPLIT = 1.0e3
# |DENOM| below FLOOR - _RECIP_GUARD/s: the smooth weight on 1/DENOM is
# < sigmoid(-_RECIP_GUARD) ~ 1e-26 and 1/DENOM is not formed (no pole).
_RECIP_GUARD = 60.0
# --- closure (convect43c.f l.563-566): CBMF += 0.1*ALPHA*DTMA, DELT0=300 s ---
_CBMF_RELAX = 0.1
_DELT0_S = 300.0
# _oracle_qsat AD floor (pre-existing; Tetens pole near 29.65 K)
_TETENS_MIN_VALID_K = 180.0
# RH floor under the smooth RH = 1 guard (RH**chi singularity guard)
_RH_MIN = 1.0e-12
_EPMAX_DEFAULT = 0.999
_STRICT_INDEX_SHARPNESS = 20.0


class EmanuelMixingOutput(NamedTuple):
    """Output of :func:`emanuel_mixing_tendencies`.

    ``dT_dt``, ``dq_v_dt`` and ``convective_layer_mask`` are
    surface-LAST; every oracle intermediate is SURFACE-FIRST, 0-based.
    ``dT_dt``/``dq_v_dt`` are the oracle's DO 410-500 mixing terms only: the
    downdraught terms, ``FRAC`` and ``ENTS`` are added by the caller.
    """

    dT_dt: jax.Array
    dq_v_dt: jax.Array
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


def _oracle_qsat(T, p, floor_sharpness=math.inf):
    """Saturation specific humidity [kg/kg], ``eps*es/(p - es*(1-eps))``.

    This is the host-supplied ``QS`` CONVECT takes as an input (the warm
    Tetens curve of :func:`legoesm.thermo.saturation_vapor_pressure`); the
    oracle fixture was produced with the same ``QS`` (``floor_sharpness =
    inf``; the scheme passes ``EmanuelConfig.qsat_floor_sharpness``).  The
    180 K floor (softplus at finite ``floor_sharpness`` [1/K]) and the
    softplus on the denominator are AD guards that do not act on atmospheric
    columns.
    """
    eps = constants.epsilon
    T_safe = _smax(T, _TETENS_MIN_VALID_K, floor_sharpness)
    es = saturation_vapor_pressure(T_safe)
    denom = jax.nn.softplus(p - es * (1.0 - eps) - 1.0) + 1.0
    return eps * es / denom


def _tlift_es_hpa(tg, freeze_sharpness):
    """TLIFT vapour pressure [hPa] (convect43c.f l.1058-1065): warm Tetens for
    ``TC >= 0``, the oracle's cold-branch fit below.  The two curves differ by
    ~0.7 % at 0 C, so the switch is a jump: blended by a sigmoid of sharpness
    ``freeze_sharpness`` [1/K] (``inf`` = the oracle's switch)."""
    es_warm = saturation_vapor_pressure(tg) * _HPA_PER_PA
    es_cold = jnp.exp(
        _TLIFT_ES_COLD_A - _TLIFT_ES_COLD_B / tg + _TLIFT_ES_COLD_C * jnp.log(tg))
    w = sigmoid_switch(tg - constants.T_freeze, freeze_sharpness)
    return _blend(w, es_warm, es_cold, freeze_sharpness)


def _tlift_moist(T, qs, p_hpa, gz, T_nk, q_nk, gz_nk, *, c_l, tlift_floor_sharpness,
                 tlift_freeze_sharpness):
    """TLIFT saturated-parcel branch at every level (convect43c.f l.1042-1073).

    Returns ``(TPK, QG)``.  ``S`` and ``ALV`` use the ENVIRONMENT ``T(I)``;
    ``QG`` starts from the host ``QS(I)``; two Newton steps.  The oracle's
    ``MAX(TG, 35)`` is smoothed by ``tlift_floor_sharpness`` [1/K], its
    warm/cold vapour-pressure switch by ``tlift_freeze_sharpness`` [1/K].
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
        tg = _smax(tg + s * (ah0 - ahg), _TLIFT_TG_FLOOR_K, tlift_floor_sharpness)
        es = _tlift_es_hpa(tg, tlift_freeze_sharpness)
        qg = eps * es / (p_hpa - es * (1.0 - eps))
    tpk = (ah0 - (c_l - cpd) * q_nk * T - gz - alv * qg) / cpd
    return tpk, qg


def _hard(s):
    return s == math.inf


def _relu(x, s):
    """``max(x, 0)``; softplus at finite ``s``."""
    return smooth_relu(x, s)


def _smax(a, b, s):
    """``max(a, b)``; ``b + softplus`` at finite ``s``."""
    return jnp.maximum(a, b) if _hard(s) else b + smooth_relu(a - b, s)


def _smin(a, b, s):
    """``min(a, b)``; ``a - softplus`` at finite ``s``."""
    return jnp.minimum(a, b) if _hard(s) else a - smooth_relu(a - b, s)


def _sclip(x, lo, hi, s):
    """``clip(x, lo, hi)``; nested softplus max/min at finite ``s``."""
    return jnp.clip(x, lo, hi) if _hard(s) else _smin(_smax(x, lo, s), hi, s)


def _sabs(x, s):
    """``|x|`` = ``max(x, -x)``; softplus at finite ``s``."""
    return jnp.abs(x) if _hard(s) else smooth_relu(2.0 * x, s) - x


def _lt(a, b, s):
    """Indicator of the STRICT ``a < b`` (exact at ``s = inf``)."""
    return 1.0 - sigmoid_switch(a - b, s)


def _blend(w, a, b, s):
    """``w*a + (1-w)*b``; at ``s = inf`` the exact ``where(w, a, b)``."""
    return jnp.where(w > 0.5, a, b) if _hard(s) else w * a + (1.0 - w) * b


def _floor_recip(d, floor, s):
    """``1/D`` after the oracle's ``IF(ABS(D).LT.floor) D = floor``.

    The floor is discontinuous (``D = -1.01 floor`` stays, ``-0.99 floor``
    becomes ``+floor``), and any continuous blend of the DENOMINATORS passes
    through 0.  So the RECIPROCAL is blended: ``w/D + (1-w)/floor`` with
    ``w = sigmoid(s(|D| - floor))``; ``1/D`` is replaced by 0 where
    ``|D| < floor - 60/s`` (there ``w < 1e-26``), so no pole is formed.
    Exactly the oracle at ``s = inf``.  The caller validates
    ``s * floor >= 120`` so that guard band stays above ``floor/2``.
    """
    if _hard(s):
        return 1.0 / jnp.where(jnp.abs(d) < floor, floor, d)
    ad = jnp.abs(d)
    w = sigmoid_switch(ad - floor, s)
    far = ad > floor - _RECIP_GUARD / s
    inv = jnp.where(far, 1.0 / jnp.where(far, d, 1.0), 0.0)
    return w * inv + (1.0 - w) / floor


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
    level_window_sharpness: float,
    sij_gate_sharpness: float,
    sij_upper_gate: float,
    denom_floor: float,
    sat_branch_sharpness: float,
    lcl_pressure_sharpness: float,
    source_temperature_gate_sharpness: float,
    source_humidity_gate_sharpness: float,
    plcl_gate_sharpness: float,
    stable_exit_sharpness: float,
    cold_start_sharpness: float,
    cbmf_sharpness: float,
    precip_efficiency_sharpness: float,
    condensate_sharpness: float,
    scrit_floor_sharpness: float,
    frac_sharpness: float,
    defrac_floor_sharpness: float,
    tlift_floor_sharpness: float,
    denom_floor_sharpness: float,
    nent_count_sharpness: float,
    bsum_sharpness: float,
    asij_floor_sharpness: float,
    sort_sharpness: float,
    tlift_freeze_sharpness: float,
    qsat_floor_sharpness: float,
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
    level_window_sharpness, sij_gate_sharpness, sat_branch_sharpness,
    strict_index_sharpness, lcl_pressure_sharpness, and the remaining
    ``*_sharpness`` arguments : smoothing of the oracle switches named in the
        module docstring (fields of :class:`EmanuelConfig`, documented there);
        large values are the oracle and ``math.inf`` is exactly the oracle
        for every field read through :func:`legoesm.core.smooth`.
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
    qs = _oracle_qsat(T, p_full, qsat_floor_sharpness)[:, ::-1]
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
    # IFLAG 0 (l.384): T(NK) < 250 K, Q(NK) <= 0 smoothed; IHMIN = NL-1 is an
    # index pick and stays hard.
    gate_idx = ihmin == nl - 2
    pass_a = (sigmoid_switch(T_nk - _T_SOURCE_MIN_K, source_temperature_gate_sharpness)
              * (1.0 - sigmoid_switch(-q_nk, source_humidity_gate_sharpness)))

    # ---- LCL (l.392-394) and ICB (l.402-408) -------------------------------
    if _hard(source_humidity_gate_sharpness):
        rh = jnp.where(q_nk > 0.0, q_nk / _at(qs, nk), 1.0)   # IFLAG-0: avoid 0**chi
    else:
        # Softplus of Q(NK) (the Q(NK) <= 0 column is rejected by the same
        # field's CBMF weight); the softplus underflows to 0 for Q(NK) below
        # ~-1e-4 kg/kg, so RH is floored at 1e-12 (singularity guard: RH**chi,
        # chi < 1, has an infinite derivative at 0).
        rh = jnp.maximum(smooth_relu(q_nk, source_humidity_gate_sharpness) / _at(qs, nk),
                         _RH_MIN)
    chi = T_nk / (_EMANUEL_CHI_A - _EMANUEL_CHI_B * rh - T_nk)
    plcl = _at(pf, nk) * rh ** chi
    # IFLAG 2 (l.395): PLCL < 200 or >= 2000 hPa, smoothed [1/Pa].
    pass_b = (sigmoid_switch((plcl - _PLCL_MIN_HPA) / _HPA_PER_PA, plcl_gate_sharpness)
              * _lt(plcl / _HPA_PER_PA, _PLCL_MAX_HPA / _HPA_PER_PA, plcl_gate_sharpness))
    above_lcl = (idx[None, :] > nk[:, None]) & (idx[None, :] <= nl - 1) & (pf < plcl[:, None])
    icb = jnp.where(jnp.any(above_lcl, axis=-1), jnp.argmax(above_lcl, axis=-1), nl - 2)
    gate_idx = gate_idx | (icb >= nl - 2)            # IFLAG 3: index, hard

    # ---- TLIFT (KK=1 and KK=2) ---------------------------------------------
    tpk_m, qg = _tlift_moist(Tf, qs, pf, gz, T_nk[:, None], q_nk[:, None],
                             gz_nk[:, None], c_l=c_l,
                             tlift_floor_sharpness=tlift_floor_sharpness,
                             tlift_freeze_sharpness=tlift_freeze_sharpness)
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
    clw = jnp.where(idx[None, :] >= icb[:, None],
                    _relu(q_nk2 - qg, condensate_sharpness), 0.0)

    # Cold-start stable exit (l.429-432): CBMF = 0 and TVP(ICB) <= TV(ICB)-DTMAX.
    if _hard(cold_start_sharpness):
        cold = (cbmf_old == 0.0).astype(dtype)
    else:
        cold = jnp.exp(-cold_start_sharpness * cbmf_old)
    stable_w = cold * sigmoid_switch(_at(tv, icb) - dtmax - _at(tvp, icb),
                                     stable_exit_sharpness)

    # ---- EP (l.446-464) ----------------------------------------------------
    tca = tp - constants.T_freeze
    # MAX(WHERE(TCA >= 0, ELCRIT, ELCRIT*(1 - TCA/TLCRIT)), 0) is
    # ELCRIT*CLIP(1 - TCA/TLCRIT, 0, 1) for TLCRIT < 0 (validated).
    elacrit = elcrit * _sclip(1.0 - tca / tlcrit, 0.0, 1.0, precip_efficiency_sharpness)
    # MAX(CLW, 1e-8) is the oracle's division guard, kept exact.
    ep = _sclip(epmax * (1.0 - elacrit / jnp.maximum(clw, _CLW_EP_FLOOR)), 0.0, epmax,
                precip_efficiency_sharpness)
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
    defrac = _smax(capem - cape, _DEFRAC_FLOOR, defrac_floor_sharpness)
    frac = _sclip(-cape / defrac, 0.0, 1.0, frac_sharpness)
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
    cbmf_new = _relu((1.0 - damps) * cbmf_old + _CBMF_RELAX * alpha_closure * dtma,
                     cbmf_sharpness)
    # Rejected column (IFLAG 0/2/3, the stable exit): CBMF = 0, zero
    # tendencies.  The state gates weight CBMF smoothly; index gates are hard.
    cbmf = jnp.where(gate_idx, 0.0, pass_a * pass_b * (1.0 - stable_w) * cbmf_new)
    M_b = cbmf

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
    SIJ = jnp.where(eye, 1.0, ANUM * _floor_recip(DENOM, denom_floor, denom_floor_sharpness))
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
    SIJ2 = ANUM2 * _floor_recip(DENOM2, denom_floor, denom_floor_sharpness)
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
    ELIJ = _relu(ALTEM_eff, condensate_sharpness) * sij_active
    one_minus_sij = jnp.clip(1.0 - SIJ_eff, 1.0 - sij_upper_gate, None)
    MENT_raw = (m_i[:, :, None] / one_minus_sij) * sij_active
    nent = jnp.sum(act, axis=-1)
    # Smooth NENT for the soft local-detrainment weight: the same band and
    # windows, smoothed, off the diagonal.
    nent_soft = jnp.sum(jnp.where(eye, 0.0, sij_active), axis=-1)

    # ---- SCRIT and the DO 200 normalisation --------------------------------
    qp1 = q_nk[:, None] - ep * clw
    scrit = (h - hp - lv * (qp1 - qs)) * _floor_recip(
        h - hp + lv * (qf - qp1), denom_floor, denom_floor_sharpness)
    # IF(QP1 - QS + SCRIT*(Q - QP1) < 0) SCRIT = 1 [kg/kg], then MAX(SCRIT, 0).
    sat_w = _lt(qp1 - qs + scrit * (qf - qp1), 0.0, condensate_sharpness)
    scrit = _blend(sat_w, 1.0, scrit, condensate_sharpness)
    scrit = _relu(scrit, scrit_floor_sharpness)       # (ncol, i)

    # Clipped SIJ as stored by the oracle: J window only, 1 on the diagonal.
    S = jnp.where(win_j[:, None, :], _sclip(SIJ_eff, 0.0, 1.0, sort_sharpness), 0.0)
    S = jnp.where(eye & win_i[:, :, None], 1.0, S)
    Sp = jnp.concatenate([S[:, :, 1:], jnp.zeros_like(S[:, :, :1])], axis=-1)
    Sm = jnp.concatenate([jnp.zeros_like(S[:, :, :1]), S[:, :, :-1]], axis=-1)

    # The march's min/max/abs and its strict SMIN tests are smoothed by
    # ``sort_sharpness``; ``J > I`` is an index test and stays hard.  The
    # band indicator is the hard NENT band at ``sort_sharpness = inf`` and
    # the smooth ``sij_active`` (off the diagonal) otherwise.
    ss = sort_sharpness
    act_w = (act.astype(dtype) if _hard(ss)
             else jnp.where(eye, 0.0, sij_active))

    def _norm_step(smin, xs):
        j, s_j, sp_j, sm_j, a_j = xs
        gt = j > idx[None, :]
        smid_u = _smin(s_j, scrit, ss)
        upd = a_j * gt * _lt(smid_u, smin, ss) * _lt(sp_j, smid_u, ss)
        sjmax_u = _blend(upd, _smin(_smin(sp_j, s_j, ss), scrit, ss), smid_u, ss)
        sjmin_u = _blend(upd, _smin(_smax(sm_j, s_j, ss), scrit, ss), smid_u, ss)
        smin = _blend(upd, smid_u, smin, ss)
        smid_d = _smax(s_j, scrit, ss)
        sjmax_d = _smax(sp_j, scrit, ss)
        sjmin_d = _smax(sm_j, scrit, ss)              # Sm = 0 at the lowest level
        smid = jnp.where(gt, smid_u, smid_d)
        w = (_sabs(jnp.where(gt, sjmax_u, sjmax_d) - smid, ss)
             + _sabs(jnp.where(gt, sjmin_u, sjmin_d) - smid, ss))
        return smin, w

    xs = (idx, jnp.moveaxis(S, 2, 0), jnp.moveaxis(Sp, 2, 0),
          jnp.moveaxis(Sm, 2, 0), jnp.moveaxis(act_w, 2, 0))
    _, dw = lax.scan(_norm_step, jnp.ones_like(scrit), xs)
    prob_w = jnp.moveaxis(dw, 0, 2) * dph[:, None, :] * sij_active
    # ASIJ = MAX(1e-21, ASIJ), written so a floored row divides by the
    # constant: same value, but no derivative through 1/ASIJ**2, which
    # overflows in fp32 (1e-21**-2) and turns gradients/tangents into NaN.
    asij = jnp.sum(prob_w, axis=-1, keepdims=True)
    if _hard(asij_floor_sharpness):
        ok = asij > _ASIJ_FLOOR
        MENT = jnp.where(ok, MENT_raw * prob_w / jnp.where(ok, asij, 1.0),
                         MENT_raw * prob_w / _ASIJ_FLOOR)
    else:
        # Smooth MAX(ASIJ, 1e-21) in units of the floor, so no 1/1e-21**2
        # (fp32 overflow) is formed; rows far above the floor divide by ASIJ.
        big = asij > _ASIJ_SPLIT * _ASIJ_FLOOR
        a_small = jnp.where(big, 1.0, asij / _ASIJ_FLOOR)
        MENT = jnp.where(
            big, MENT_raw * prob_w / jnp.where(big, asij, 1.0),
            MENT_raw * (prob_w / _ASIJ_FLOOR) / _smax(a_small, 1.0, asij_floor_sharpness))
    bsum = jnp.sum(MENT, axis=-1)
    # Local detrainment: NENT = 0 (DO 170) or BSUM < 1e-18 (DO 190), as the
    # weight w_local = 1 - [NENT > 0]*[BSUM >= 1e-18], each factor smoothed.
    has_ent = ((nent > 0).astype(dtype) if _hard(nent_count_sharpness)
               else sigmoid_switch(nent_soft - 0.5, nent_count_sharpness))
    w_local = 1.0 - has_ent * sigmoid_switch(
        bsum - _BSUM_MIN - _BSUM_CENTRE_WIDTHS / bsum_sharpness, bsum_sharpness)
    # Blended on the diagonal; with both fields at inf, wl is exactly 0/1 and
    # the blend is the oracle's assignment.
    wl = jnp.where(eye, w_local[:, :, None], 0.0)
    MENT = MENT + wl * m_i[:, :, None]
    QENT = wl * QTI + (1.0 - wl) * QENT
    ELIJ = wl * clw[:, :, None] + (1.0 - wl) * ELIJ

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
    AWAT = _relu(ELIJ - ((1.0 - ep) * clw)[:, None, :], condensate_sharpness)
    k_lt_j = j_gt_i                                   # origin k (axis 1) < level j
    detr = MENT * (QENT - k_lt_j * AWAT - Q_j)
    fq = fq + g * dpinv * jnp.sum(detr, axis=1)
    # Mixture condensate AWAT detrained below its origin's level is NOT a
    # cloud-water source: as in the oracle it goes to WDTRAIN (precipitation;
    # the downdraught receives it there), so it leaves through the caller's
    # column rain residual.

    ents_mask = jax.nn.sigmoid(level_window_sharpness * (inb_f - lvl + 0.5))

    return EmanuelMixingOutput(
        dT_dt=ft[:, ::-1],
        dq_v_dt=fq[:, ::-1],
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
