"""Double-diffusive mixing ORACLE-FAITHFULNESS tests (Large 1994 / CVMix / NEMO).

The ocean double-diffusion scheme
(``ocean/physics/vertical_mixing/double_diffusion.py`` ``compute_ddm_diffusivity``)
is a HYBRID of three oracles; the existing ``test_double_diffusion.py`` only
checks directional / regime / bounds properties.  These pin every closed form to
round-off (rel 1e-12) against an INDEPENDENT reimplementation and canary each
departure.

Oracles (constants typed from the docs/source as local ``_O_*`` literals, NOT
copied from the module):
- CVMix Aug-2012 documentation (Griffies et al.) Sec. 6.2-6.3:
    Eq. 6.6  salt-fingering salt diffusivity kappa = kappa0*(1-((R-1)/(R0-1))^2)^3
    Eq. 6.10 diffusive-convection heat kappa = 1.5e-6 * 0.909 * exp(4.6*exp(-0.54*(1/R-1)))
    Eq. 6.12 diffusive-convection salt factor (1.85-0.85/R)*R [0.5<=R<1] else 0.15*R
- NEMO ``zdfddm.F90`` (the module's DECLARED reference):
    heat/salt FLUX ratio gamma = 0.7  ->  avt = 0.7*avs/R  (``zavft = 0.7*zavfs*zinr``)
    diffusive-convection prefactor literal 1.3635e-6 (== 1.5e-6*0.909)

What is pinned, in order of authority:
1. Salt fingering: avs (Eq. 6.6 cubic) and avt (= 0.7*avs/R) vs the independent
   oracle across a grid of R_rho in (1, R_c), for the default AND a non-default cfg.
2. Diffusive convection: avt (Eq. 6.10) and the two-branch avs (Eq. 6.12) vs the
   oracle across R_rho in (0, 1), including the R_rho = 0.5 branch join (continuous).
3. DEPARTURE from the declared NEMO oracle: NEMO's salt-fingering avfs is the
   RATIONAL rn_avts/(1+(R/rn_hsbfr)^6); this module uses the Large/CVMix cubic.
   Canaried: at R_rho = R_c the module gives 0 while NEMO's rational gives
   rn_avts/2 (they differ), and the module's hard cutoff zeroes R_rho >= R_c.
4. Regime gating: N^2 <= 0 -> zero; single-signed (R_rho <= 0) -> zero.
5. Coefficient canaries (module privates == oracle literals) + the k_max cap +
   AD-finiteness at the fingering and convection points.

Precision: the form carries no precision-policy promotion; the autouse fixture
enables x64 for the round-off pins and restores the entry state.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import pytest
from legoesm.ocean.physics.vertical_mixing.double_diffusion import (
    DoubleDiffusionConfig,
    compute_ddm_diffusivity,
)


@pytest.fixture(autouse=True)
def _force_x64():
    entry_x64 = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", entry_x64)


# Oracle constants typed from CVMix Eqs. 6.6/6.10-6.12 + NEMO zdfddm.F90.
_O_FINGER_GAMMA = 0.7        # NEMO salt-finger heat/salt FLUX ratio (zavft=0.7*zavfs/R)
_O_DC_PREF = 1.5e-6         # CVMix Eq. 6.11 nu_molecular [m^2/s]
_O_DC_C0 = 0.909           # CVMix Eq. 6.10
_O_DC_C1 = 4.6
_O_DC_C2 = -0.54
_O_DC_HI_A = 1.85          # CVMix Eq. 6.12 (0.5<=R<1 salt factor slope)
_O_DC_HI_B = 0.85
_O_DC_LO = 0.15            # CVMix Eq. 6.12 (R<0.5 salt factor slope)
_O_DC_SPLIT = 0.5
_O_DC_PREF_NEMO = 1.3635e-6  # NEMO zavdt literal (== _O_DC_PREF*_O_DC_C0)
# float32 machine epsilon, typed independently (the module's AD-guard clip point).
# Used as the EXPECTED clip value; the module's _EPS is canaried == this separately,
# so a regression changing _EPS (e.g. -> 1e-7) is caught by the canary, not hidden.
_O_EPS = 1.1920928955078125e-7

_CFG = DoubleDiffusionConfig(enabled=True, rn_avts=1e-4, rn_hsbfr=1.6, k_max=1e-2)


def _finger_oracle(R, rn_avts, Rc):
    """Independent salt-fingering (avt, avs), plain Python (CVMix Eq. 6.6 cubic +
    NEMO 0.7/R heat ratio).  Assumes 1 < R < Rc (in-window)."""
    frac = (R - 1.0) / (Rc - 1.0)
    avs = rn_avts * (1.0 - frac * frac) ** 3
    avt = _O_FINGER_GAMMA * avs / R
    return avt, avs


def _dc_oracle(R):
    """Independent diffusive-convection (avt, avs), plain Python
    (CVMix Eq. 6.10-6.12).  Assumes 0 < R < 1 (in-window)."""
    avt = _O_DC_PREF * _O_DC_C0 * math.exp(_O_DC_C1 * math.exp(
        _O_DC_C2 * (1.0 / R - 1.0)))
    if R >= _O_DC_SPLIT:
        factor = (_O_DC_HI_A - _O_DC_HI_B / R) * R
    else:
        factor = _O_DC_LO * R
    avs = avt * factor
    return avt, avs


def _nemo_finger_avfs(R, rn_avts, Rc):
    """NEMO zdfddm.F90 salt-fingering ``zavfs = rn_avts/(1+(R/Rc)^6) * masks`` —
    the RATIONAL core the module DEPARTS from.  NEMO's masks are zmsks (N^2>0),
    zmskf (R>1), zmskr (R<1000); for the departure-test R values (1 < R < 1000,
    N^2 > 0) ALL masks are 1, so this rational core IS NEMO's actual avfs there."""
    return rn_avts / (1.0 + (R / Rc) ** 6)


def _inputs(R, N2=1e-4, beta_dSdz=1e-4):
    """(N2, alpha_dTdz, beta_dSdz) that give density ratio R at a stable
    interface: R = (alpha dT/dz)/(beta dS/dz), beta_dSdz > 0."""
    return (jnp.array([float(N2)]),
            jnp.array([float(R) * beta_dSdz]),
            jnp.array([float(beta_dSdz)]))


def _ddm(R, cfg=_CFG, **kw):
    N2, a, b = _inputs(R, **kw)
    avt, avs = compute_ddm_diffusivity(N2, a, b, cfg)
    return float(avt[0]), float(avs[0])


# --- salt fingering: cubic amplitude + 0.7/R heat ratio ---------------------

_FINGER_R = [1.05, 1.1, 1.2, 1.3, 1.45, 1.55, 1.599]  # inside (1, Rc=1.6)


@pytest.mark.parametrize("R", _FINGER_R)
def test_salt_fingering_matches_oracle(R):
    avt, avs = _ddm(R)
    avt_o, avs_o = _finger_oracle(R, _CFG.rn_avts, _CFG.rn_hsbfr)
    assert avs == pytest.approx(avs_o, rel=1e-12, abs=0.0)
    assert avt == pytest.approx(avt_o, rel=1e-12, abs=0.0)


@pytest.mark.parametrize("R", [1.1, 1.5, 2.4])
def test_salt_fingering_matches_oracle_nondefault_cfg(R):
    """A non-default (rn_avts, rn_hsbfr) still matches the oracle to round-off, so
    a regression hard-coding the classic defaults is caught."""
    cfg = DoubleDiffusionConfig(enabled=True, rn_avts=3e-4, rn_hsbfr=2.55, k_max=1e-2)
    if not (1.0 < R < cfg.rn_hsbfr):
        pytest.skip("R outside this cfg's fingering window")
    avt, avs = _ddm(R, cfg=cfg)
    avt_o, avs_o = _finger_oracle(R, cfg.rn_avts, cfg.rn_hsbfr)
    assert avs == pytest.approx(avs_o, rel=1e-12, abs=0.0)
    assert avt == pytest.approx(avt_o, rel=1e-12, abs=0.0)


def test_finger_heat_ratio_is_flux_ratio_over_R():
    """avt/avs = 0.7/R (NEMO flux-ratio), NOT the constant 0.7 of CVMix/Large.
    Holds for the UNCAPPED diffusivities (default k_max=1e-2 >> the ~1e-4
    magnitudes so the cap is inactive here); a small k_max breaks it (see the
    k_max cap test)."""
    for R in (1.1, 1.3, 1.55):
        avt, avs = _ddm(R)
        assert avt / avs == pytest.approx(_O_FINGER_GAMMA / R, rel=1e-12)
        assert avt / avs < _O_FINGER_GAMMA          # strictly below the CVMix constant 0.7


# --- fingering hard cutoff (Large/CVMix cubic) vs the NEMO rational ----------

def test_fingering_hard_cutoff_at_Rc():
    """The Large/CVMix cubic zeroes AT R_c (inclusive): ON just below, OFF at/above."""
    Rc = _CFG.rn_hsbfr
    _, avs_below = _ddm(Rc - 1e-3)
    assert avs_below > 0.0
    _, avs_at = _ddm(Rc)
    _, avs_above = _ddm(Rc + 1e-3)
    assert avs_at == 0.0 and avs_above == 0.0


@pytest.mark.parametrize("R", [1.2, 1.4, 1.6])
def test_departs_from_nemo_rational_avfs(R):
    """DEPARTURE canary: the module's salt-fingering amplitude is the Large/CVMix
    cubic, NOT NEMO zdfddm's rational rn_avts/(1+(R/Rc)^6).  All test R are inside
    NEMO's mask (1 < R < 1000, N^2>0 -> masks=1), so the rational core IS NEMO's
    actual avfs.  At R = R_c the cubic is 0 while NEMO's rational is rn_avts/2 —
    WITHIN the mask NEMO applies no R_c cutoff (its cutoff is the far R>=1000)."""
    Rc = _CFG.rn_hsbfr
    assert 1.0 < R < 1000.0                                 # inside NEMO's mask -> masks=1
    _, avs_code = _ddm(R)
    avs_nemo = _nemo_finger_avfs(R, _CFG.rn_avts, Rc)
    assert avs_nemo > 0.0                    # NEMO has no R_c cutoff (its mask cut is R>=1000)
    if R >= Rc:
        assert avs_code == 0.0                              # cubic cutoff
        assert avs_nemo == pytest.approx(_CFG.rn_avts / 2.0, rel=1e-12)  # rn_avts/(1+1)
        assert abs(avs_code - avs_nemo) > 1e-6              # the departure is real
    else:
        assert avs_code != pytest.approx(avs_nemo, rel=1e-6)  # differ inside the window too


# --- diffusive convection: Eq. 6.10 + two-branch Eq. 6.12 -------------------

_DC_R = [0.1, 0.3, 0.49, 0.51, 0.7, 0.9, 0.99]


@pytest.mark.parametrize("R", _DC_R)
def test_diffusive_convection_matches_oracle(R):
    avt, avs = _ddm(R)
    avt_o, avs_o = _dc_oracle(R)
    assert avt == pytest.approx(avt_o, rel=1e-12, abs=0.0)
    assert avs == pytest.approx(avs_o, rel=1e-12, abs=0.0)


def test_dc_salt_branches_continuous_at_half():
    """The two salt branches agree AT R_rho = 0.5 (constructed at the join):
    (1.85-0.85/0.5)*0.5 == 0.15*0.5 == 0.075 (so avs is continuous)."""
    hi = (_O_DC_HI_A - _O_DC_HI_B / 0.5) * 0.5
    lo = _O_DC_LO * 0.5
    assert hi == pytest.approx(lo, rel=1e-12) == pytest.approx(0.075, rel=1e-12)
    # and the scheme output is continuous across the join
    _, avs_lo = _ddm(0.5 - 1e-6)
    _, avs_hi = _ddm(0.5 + 1e-6)
    assert avs_lo == pytest.approx(avs_hi, rel=1e-4)
    # EXACTLY at R_rho = 0.5 the scheme (Rd>=0.5 -> hi branch) matches the CVMix
    # oracle.  DEPARTURE from NEMO: NEMO's dual-strict masks (zmskd2 R<0.5,
    # zmskd3 0.5<R<1) BOTH vanish at R=0.5, so NEMO's avds=0 there, whereas this
    # module returns the hi branch 0.075*avt (nonzero) — a measure-zero departure.
    avt_half, avs_half = _ddm(0.5)
    avt_o, avs_o = _dc_oracle(0.5)
    assert avt_half == pytest.approx(avt_o, rel=1e-12, abs=0.0)
    assert avs_half == pytest.approx(avs_o, rel=1e-12, abs=0.0)
    assert avs_half > 0.0                              # module: hi branch; NEMO here = 0
    assert avs_half == pytest.approx(0.075 * avt_half, rel=1e-12)


def test_dc_branch_predicate_is_ge_half_via_gradient():
    """The value at R_rho=0.5 CANNOT distinguish the R>=0.5 (hi) vs R>0.5 (lo)
    predicate (both = 0.075*avt).  The DERIVATIVE can: pin d(avs)/dR at R=0.5 to
    the HI-branch oracle derivative.  A regression to `Rd > 0.5` (lo branch at 0.5)
    changes this gradient, so this canaries the exact branch predicate."""
    beta = 1e-4

    def avs_of_R(R):                                   # module avs as a fn of R (alpha=R*beta)
        N2 = jnp.array([1e-4])
        a = jnp.array([R * beta])
        b = jnp.array([beta])
        return compute_ddm_diffusivity(N2, a, b, _CFG)[1][0]

    def _avt_jnp(R):
        return _O_DC_PREF * _O_DC_C0 * jnp.exp(
            _O_DC_C1 * jnp.exp(_O_DC_C2 * (1.0 / R - 1.0)))

    def avs_hi(R):
        return _avt_jnp(R) * (_O_DC_HI_A * R - _O_DC_HI_B)   # (1.85-0.85/R)*R = 1.85R-0.85

    def avs_lo(R):
        return _avt_jnp(R) * _O_DC_LO * R

    g = jax.grad(avs_of_R)(0.5)
    g_hi = jax.grad(avs_hi)(0.5)
    g_lo = jax.grad(avs_lo)(0.5)
    assert abs(float(g_hi) - float(g_lo)) > 1e-9        # non-vacuous: branches DO differ in slope
    assert float(g) == pytest.approx(float(g_hi), rel=1e-9)   # module picks HI (R>=0.5)
    assert float(g) != pytest.approx(float(g_lo), rel=1e-6)


def test_dc_prefactor_equals_nemo_literal():
    """CVMix's 1.5e-6*0.909 equals NEMO's pre-multiplied zavdt literal 1.3635e-6."""
    assert _O_DC_PREF * _O_DC_C0 == pytest.approx(_O_DC_PREF_NEMO, rel=1e-9)


# --- regime gating (incl. the exact <= / boundary corners) ------------------

def test_no_ddm_when_statically_unstable():
    avt, avs = _ddm(1.3, N2=-1e-5)
    assert avt == 0.0 and avs == 0.0


def test_gating_exact_zero_boundaries():
    """The claims use N^2 <= 0 and R_rho <= 0; pin the EXACT-zero boundaries.
    N^2 == 0 -> off (stable is STRICT >0 here — a DEPARTURE from NEMO, whose mask
    rn2+1e-12<=0 treats N^2==0 as stable/active); alpha_dTdz==0 -> R_rho==0 -> off."""
    # N^2 exactly 0 (fingering R would otherwise fire)
    avt0, avs0 = compute_ddm_diffusivity(
        jnp.array([0.0]), jnp.array([1.3e-4]), jnp.array([1e-4]), _CFG)
    assert float(avt0[0]) == 0.0 and float(avs0[0]) == 0.0
    # R_rho exactly 0 (alpha_dTdz == 0) at a stable interface
    avt1, avs1 = compute_ddm_diffusivity(
        jnp.array([1e-4]), jnp.array([0.0]), jnp.array([1e-4]), _CFG)
    assert float(avt1[0]) == 0.0 and float(avs1[0]) == 0.0


def test_no_ddm_single_signed_stratification():
    # alpha_dTdz < 0 with beta_dSdz > 0 -> R_rho < 0 -> neither regime -> 0.
    # DEPARTURE from NEMO: NEMO clamps zrau=MAX(1e-20, ratio), so a single-signed
    # (ratio<0) interface becomes R_rho=1e-20 and enters NEMO's DC regime
    # (avt ~ 1.3635e-6); this module sign-preserves R_rho<0 and returns 0.
    N2 = jnp.array([1e-4])
    avt, avs = compute_ddm_diffusivity(
        N2, jnp.array([-1e-4]), jnp.array([1e-4]), _CFG)
    assert float(avt[0]) == 0.0 and float(avs[0]) == 0.0


# --- AD-guard clip layers (R_rho clipped to (1+eps, Rc-eps) / (eps, 1-eps)) --

def test_finger_clip_layer_near_Rc():
    """In the fingering eps-clip zone (Rc-eps < R_rho < Rc) the scheme evaluates
    the cubic at the CLIPPED Rf = Rc-eps (avs), but the heat ratio uses the RAW
    R_rho — pin that exact clipped behaviour (a documented AD-guard, not the raw
    oracle).  Uses the INDEPENDENT _O_EPS literal (float32 machine eps)."""
    Rc = _CFG.rn_hsbfr
    R = Rc - 1e-9                       # inside (Rc-eps, Rc): clips to Rc-eps, finger mask ON
    assert Rc - _O_EPS < R < Rc
    avt, avs = _ddm(R)
    _, avs_exp = _finger_oracle(Rc - _O_EPS, _CFG.rn_avts, Rc)  # avs at the CLIPPED Rf
    avt_exp = _O_FINGER_GAMMA * avs_exp / R                     # ratio uses the RAW R
    assert avs == pytest.approx(avs_exp, rel=1e-12, abs=0.0)
    assert avt == pytest.approx(avt_exp, rel=1e-12, abs=0.0)


def test_finger_lower_clamp_is_defensive_noop():
    """The fingering R_rho LOWER clamp (Rf >= 1+eps) is a DEFENSIVE no-op, NOT a
    materially-observable clip: the cubic is flat-MAXIMAL at R_rho->1 (frac->0,
    (1-frac^2)^3 -> 1), so across the entire (1, 1+eps) zone avs changes by only
    O((eps/(R_c-1))^2) ~ 1e-13 relative — below the 1e-12 pins.  So we canary the
    FLAT-MAX behaviour (avs == rn_avts to ~1e-11 whether clamped or not) + finite
    grad, rather than a clamp value (which is unobservable here).  The MATERIAL,
    discriminating clip canaries are test_finger_clip_layer_near_Rc (upper) and the
    two convection tests (Rd -> eps / 1-eps)."""
    R = 1.0 + 1e-9                      # inside (1, 1+eps): finger mask ON
    assert 1.0 < R < 1.0 + _O_EPS
    avt, avs = _ddm(R)
    # avs is the flat cubic maximum rn_avts (clamped Rf=1+eps and raw R both give
    # this to ~1e-13 — the clamp is immaterial); avt = 0.7*avs/R (raw R).
    assert avs == pytest.approx(_CFG.rn_avts, rel=1e-11)
    assert avt == pytest.approx(_O_FINGER_GAMMA * _CFG.rn_avts / R, rel=1e-11)
    assert bool(jnp.isfinite(jnp.array(avs)))
    # AD-safe in the lower clamp zone (the clip guards the power): grad finite.

    def avs_of_alpha(a_val):
        return compute_ddm_diffusivity(
            jnp.array([1e-4]), jnp.array([a_val]), jnp.array([1e-4]), _CFG)[1][0]

    assert bool(jnp.isfinite(jax.grad(avs_of_alpha)(R * 1e-4)))


def test_dc_clip_layer_near_zero():
    """In the convection eps-clip zone (0 < R_rho < eps) the scheme evaluates at
    the CLIPPED Rd = eps; pin that (finite, not a 1/R_rho blow-up).  Uses the
    INDEPENDENT _O_EPS literal."""
    R = _O_EPS / 100.0                  # inside (0, eps): clips up to eps, dc mask ON
    assert 0.0 < R < _O_EPS
    avt, avs = _ddm(R)
    avt_exp, avs_exp = _dc_oracle(_O_EPS)                      # evaluated at the clipped Rd
    assert bool(jnp.isfinite(jnp.array(avt)))
    assert avt == pytest.approx(avt_exp, rel=1e-12, abs=0.0)
    assert avs == pytest.approx(avs_exp, rel=1e-12, abs=0.0)


def test_dc_clip_layer_near_one():
    """Convection UPPER clip: R_rho in (1-eps, 1) clips Rd DOWN to 1-eps (Rd>=0.5 ->
    hi branch).  Pins the upper endpoint layer (a regression only-lower-clamping Rd
    would pass the near-zero test but fail here)."""
    R = 1.0 - 1e-9                      # inside (1-eps, 1): dc mask ON, clips to 1-eps
    assert 1.0 - _O_EPS < R < 1.0
    avt, avs = _ddm(R)
    avt_exp, avs_exp = _dc_oracle(1.0 - _O_EPS)               # evaluated at the clipped Rd
    assert avt == pytest.approx(avt_exp, rel=1e-12, abs=0.0)
    assert avs == pytest.approx(avs_exp, rel=1e-12, abs=0.0)


def test_module_eps_is_float32_machine_epsilon():
    """Independence canary for the clip-layer tests: the module's _EPS IS float32
    machine epsilon (== the local _O_EPS literal).  A regression changing _EPS
    (e.g. to 1e-7) trips HERE, so the _O_EPS-based clip pins above stay honest."""
    from legoesm.ocean.physics.vertical_mixing.double_diffusion import _EPS
    assert _EPS == _O_EPS
    assert _EPS == float(jnp.finfo(jnp.float32).eps)


def test_momentum_not_enhanced_departs_from_nemo():
    """DEPARTURE: NEMO adds MAX(avt,avs) to avm; this leaf returns ONLY the tracer
    (avt, avs) — no momentum enhancement.  Structural canary on the return arity."""
    out = compute_ddm_diffusivity(*_inputs(1.3), _CFG)
    assert len(out) == 2                # (avt, avs) only — NEMO's avm term is omitted


def test_beta_dSdz_floor_is_sign_preserving():
    """DEPARTURE canary (AD-guard): the |beta_dSdz| >= eps denominator floor
    PRESERVES SIGN (denom = min(beta,-eps) if beta<0 else max(beta,eps)).  With a
    tiny NEGATIVE beta_dSdz the floor keeps R_rho < 0 -> no double diffusion.  A
    regression using max(|beta|,eps) (dropping the sign) would give denom=+eps,
    R_rho = alpha/eps in (1, R_c) -> SPURIOUS salt fingering.  Chosen so that flip
    matters: alpha=1.5e-7, |beta|<eps -> |R_rho|=1.26 in the fingering window."""
    N2 = jnp.array([1e-4])
    # tiny NEGATIVE beta -> sign-preserving floor -> R_rho < 0 -> zero output
    avt, avs = compute_ddm_diffusivity(
        N2, jnp.array([1.5e-7]), jnp.array([-1e-10]), _CFG)
    assert float(avt[0]) == 0.0 and float(avs[0]) == 0.0
    # tiny POSITIVE beta -> denom floored to +eps -> R_rho=1.26 finger -> finite, >0
    avt2, avs2 = compute_ddm_diffusivity(
        N2, jnp.array([1.5e-7]), jnp.array([1e-10]), _CFG)
    assert bool(jnp.isfinite(avt2[0])) and bool(jnp.isfinite(avs2[0]))
    assert float(avs2[0]) > 0.0          # fingering DID fire (positive branch, finite)


# --- coefficient canaries (module privates == oracle literals) --------------

def test_module_constants_match_oracle():
    from legoesm.ocean.physics.vertical_mixing import double_diffusion as ddm
    assert ddm._FINGER_HEAT_RATIO == _O_FINGER_GAMMA
    assert ddm._DC_AVT_PREFACTOR == _O_DC_PREF
    assert ddm._DC_AVT_C0 == _O_DC_C0
    assert ddm._DC_AVT_C1 == _O_DC_C1
    assert ddm._DC_AVT_C2 == _O_DC_C2
    assert ddm._DC_AVS_HI_A == _O_DC_HI_A
    assert ddm._DC_AVS_HI_B == _O_DC_HI_B
    assert ddm._DC_AVS_LO == _O_DC_LO
    assert ddm._DC_RRHO_SPLIT == _O_DC_SPLIT


def test_config_defaults_are_nemo_namelist():
    c = DoubleDiffusionConfig()
    assert c.rn_avts == 1e-4       # NEMO namzdf_ddm default
    assert c.rn_hsbfr == 1.6       # NEMO default (NOT CVMix R0=2.55)
    assert c.enabled is False      # additive scheme is opt-in


# --- k_max cap departure ----------------------------------------------------

def test_kmax_cap_departs_from_raw_oracle():
    """A tiny k_max caps the raw oracle diffusivity (a documented safety-cap
    departure); the default k_max=1e-2 >> the fingering/DC magnitudes so it is
    inactive on the pins above.  When BOTH avt and avs cap, the 0.7/R heat ratio
    is broken toward 1 (avt/avs -> k_max/k_max = 1)."""
    cfg = DoubleDiffusionConfig(enabled=True, rn_avts=1e-4, rn_hsbfr=1.6, k_max=1e-5)
    avt, avs = _ddm(1.05, cfg=cfg)
    avt_raw, avs_raw = _finger_oracle(1.05, cfg.rn_avts, cfg.rn_hsbfr)
    assert avs_raw > cfg.k_max                       # raw exceeds the cap
    assert avt_raw > cfg.k_max                        # (avt too, at R=1.05)
    assert avs == pytest.approx(cfg.k_max, rel=1e-12)  # output is capped
    assert avt == pytest.approx(cfg.k_max, rel=1e-12)
    assert avt / avs == pytest.approx(1.0, rel=1e-12)  # ratio broken by the cap (not 0.7/R)


# --- differentiability ------------------------------------------------------

def test_ddm_grads_finite_at_both_regimes():
    def avs_of_alpha(a_val, R_target, beta=1e-4):
        N2 = jnp.array([1e-4])
        a = jnp.array([a_val])
        b = jnp.array([beta])
        return compute_ddm_diffusivity(N2, a, b, _CFG)[1][0]
    # a_val = R*beta; fingering R=1.3, convection R=0.7
    for R in (1.3, 0.7):
        g = jax.grad(avs_of_alpha)(R * 1e-4, R)
        assert bool(jnp.isfinite(g))
