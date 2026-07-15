"""Faithful Thompson et al. (2008) SNOW parameterization.

Thompson, G., P. R. Field, R. M. Rasmussen, W. D. Hall (2008): Explicit
forecasts of winter precipitation using an improved bulk microphysics scheme.
Part II: Implementation of a new snow parameterization. Mon. Wea. Rev., 136,
5095-5115.  Reference implementation: WRF ``phys/module_mp_thompson.F``.

The distinctive Thompson-2008 snow assumes a NON-spherical mass-size relation
``m(D) = am_s D^bm_s`` (``bm_s = 2`` ⇒ mass ∝ D², Cox 1988) and a BIMODAL
(exponential + generalized-gamma) size distribution whose shape is fixed and
whose 2nd/3rd moments are set by the snow mass content and temperature via the
Field et al. (2005) universal moment relation:

    N(D) = M2^4 / M3^3 · [ Kap0·exp(-Λ0 D)
                           + Kap1·(M2/M3)^μ_s · D^μ_s · exp(-Λ1 D) ],
    Λ0 = Lam0·M2/M3,   Λ1 = Lam1·M2/M3.

Because the normalization is M2^4/M3^3 the distribution reproduces the 2nd and
3rd moments EXACTLY (∫D²N=M2, ∫D³N=M3); every other moment used by the bulk
process rates is the analytic gamma-function integral

    I(p) = ∫ D^p N(D) dD
         = M2^4/M3^3 · [ Kap0·Γ(p+1)/Λ0^(p+1)
                         + Kap1·(M2/M3)^μ_s·Γ(p+μ_s+1)/Λ1^(p+μ_s+1) ].

This module provides the mass-weighted snow FALL SPEED and the vapour
DEPOSITION/sublimation rate built from those exact moment integrals with the
Thompson-2008 fall-speed (``av_s/bv_s/fv_s``) and ventilation (``Sc``,
``t1_qs_sd``, ``t2_qs_sd``, capacitance ``C_sqrd``) constants.  The rates are
finite and finite-gradient (AD-safe) everywhere, including at q_s→0; they are
NOT globally smooth — the activation gate, speed cap, and availability clamps
are non-differentiable ``where``/``clip`` guards by design.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.atmosphere.physics.microphysics._warm_rain import safe_pow

# --- Thompson-2008 snow constants (WRF module_mp_thompson.F) ----------------
_AM_S = 0.069            # mass-size prefactor  m = am_s D^bm_s   [kg m^-bm_s]
_BM_S = 2.0              # mass-size exponent
_AV_S = 40.0             # fall-speed prefactor  v = av_s D^bv_s exp(-fv_s D)
_BV_S = 0.55             # fall-speed exponent
_FV_S = 100.0            # fall-speed exponential cutoff [1/m]
_MU_S = 0.6357           # gamma-mode shape
_KAP0 = 490.6            # exponential-mode amplitude
_KAP1 = 17.46            # gamma-mode amplitude
_LAM0 = 20.78            # exponential-mode slope factor
_LAM1 = 3.29             # gamma-mode slope factor
_SC = 0.632              # Schmidt number
_C_SQRD = 0.15           # snow capacitance shape factor (plates/aggregates)
# Reference density for the (rho0/rho)^1/2 fall-speed correction.
# (298 K is the Thompson reference temperature; p and R_d from constants.)
_RHO_NOT = constants.p_atm_std / (constants.R_d * 298.0)

# Field et al. (2005) universal moment-relation coefficients (sa, sb), exactly
# as in the WRF reference. ``log10(a) = Σ sa·{1,tc,p,tc·p,tc²,p²,tc²·p,tc·p²,
# tc³,p³}`` and likewise ``b`` from sb; the p-th moment is ``Mp = 10^log10(a) ·
# M2^b`` with the 2nd moment M2 as the reference and ``tc`` the (clamped)
# temperature in Celsius.
_SA = (5.065339, -0.062659, -3.032362, 0.029469, -0.000285,
       0.31255, 0.000204, 0.003199, 0.0, -0.015952)
_SB = (0.476221, -0.015896, 0.165977, 0.007468, -0.000141,
       0.060366, 0.000079, 0.000594, 0.0, -0.003577)

_QS_SMALL = 1.0e-12      # snow mass floor below which rates vanish


# Hall-Pruppacher diffusivity / conductivity / Sutherland viscosity + Thompson
# snow ventilation constants (fixed published).
_RHO_FLOOR = 0.1
_RHO_VISC_FLOOR = 0.01
_DV_PREFACTOR = 8.794e-5
_DV_T_EXP = 1.81
_KA_A = 2.3971e-2
_KA_SLOPE = 7.078e-5
_MU_PREFACTOR = 1.458e-6
_MU_T_EXP = 1.5
_MU_SUTHERLAND_T = 110.4
_SNOW_VENT_F0 = 0.86
_SNOW_VENT_F1 = 0.28
_VT_CLIP_SNOW = 5.0

def _field_moment(p, smo2, tc):
    """Field-2005 p-th snow moment from the 2nd moment ``smo2`` and ``tc`` [°C].

    ``p`` is a Python float (static); ``smo2`` and ``tc`` are arrays.  For
    ``p == 2`` returns ``smo2`` unchanged (self-consistency).
    """
    if abs(p - 2.0) < 1.0e-6:
        return smo2
    loga = (_SA[0] + _SA[1] * tc + _SA[2] * p + _SA[3] * tc * p
            + _SA[4] * tc * tc + _SA[5] * p * p + _SA[6] * tc * tc * p
            + _SA[7] * tc * p * p + _SA[8] * tc * tc * tc + _SA[9] * p * p * p)
    b = (_SB[0] + _SB[1] * tc + _SB[2] * p + _SB[3] * tc * p
         + _SB[4] * tc * tc + _SB[5] * p * p + _SB[6] * tc * tc * p
         + _SB[7] * tc * p * p + _SB[8] * tc * tc * tc + _SB[9] * p * p * p)
    a = 10.0 ** loga
    return a * safe_pow(jnp.clip(smo2, 1.0e-30), b)


def _snow_moments(q_s, rho, T):
    """Return ``(M2, M3, ratio)`` for the snow PSD where ``ratio = M2/M3``.

    ``M2 = ρ q_s / am_s`` is the snow mass content expressed as the 2nd moment
    (bm_s = 2). ``M3`` follows from the Field relation. ``ratio`` is the PSD
    slope normalization ``M2/M3`` used to build Λ0, Λ1 and every integral.
    """
    # Clamp to the Field-2005 fit's validity range. The cubic-in-tc polynomial
    # is fit for roughly −55…0 °C; the RCEMIP cold-point tropopause reaches
    # ~−83 °C, where the unclamped cubic extrapolates to wild moments (→ NaN
    # within a step). WRF clamps the UPPER end (min(-0.1,...)); we also clamp
    # the lower end so cold-tropopause snow stays physical.
    tc = jnp.clip(T - constants.T_freeze, -55.0, -0.1)  # coeff-ok: snow-T physical clip [degC]
    smo2 = jnp.clip(q_s, 0.0) * rho / _AM_S          # = M2 (bm_s = 2)
    M3 = _field_moment(3.0, smo2, tc)
    ratio = smo2 / jnp.clip(M3, 1.0e-30)             # M2/M3  [1/m]
    return smo2, M3, ratio


def _psd_integral(p, M2, M3, ratio):
    """Analytic bimodal-PSD moment integral ``I(p) = ∫ D^p N(D) dD``.

    ``Λ0 = Lam0·ratio``, ``Λ1 = Lam1·ratio``; the ``M2^4/M3^3`` normalization
    is applied so ``I(2)=M2`` and ``I(3)=M3`` exactly.
    """
    lam0 = _LAM0 * ratio
    lam1 = _LAM1 * ratio
    # M2^4/M3^3 written as M2·(M2/M3)^3 = M2·ratio^3 — algebraically identical
    # but float32-safe: forming M2^4 (~1e-42 for thin snow) underflows float32
    # to 0 and M3^3 likewise, giving 0/0 = NaN; the ratio form keeps every
    # intermediate in range (the f32 NaN found in the SCM cold mixed-phase
    # thompson run).
    norm = jnp.clip(M2, 1.0e-30) * safe_pow(jnp.clip(ratio, 1.0e-30), 3.0)
    g0 = jnp.exp(jax.lax.lgamma(p + 1.0))
    g1 = jnp.exp(jax.lax.lgamma(p + _MU_S + 1.0))
    term0 = _KAP0 * g0 / safe_pow(jnp.clip(lam0, 1.0e-30), p + 1.0)
    term1 = (_KAP1 * safe_pow(jnp.clip(ratio, 1.0e-30), _MU_S) * g1
             / safe_pow(jnp.clip(lam1, 1.0e-30), p + _MU_S + 1.0))
    return norm * (term0 + term1)


def snow_fall_speed(q_s, rho, T):
    """Mass-weighted Thompson-2008 snow fall speed [m/s] (positive downward).

    ``V_s = av_s·ρ_f · ∫ D^(bm_s+bv_s) e^(-fv_s D) N dD / ∫ D^bm_s N dD`` with
    the density correction ``ρ_f = √(ρ0/ρ)``.  The ``exp(-fv_s D)`` shifts each
    PSD-mode slope (Λ0→Λ0+fv_s, Λ1→Λ1+fv_s) and is folded into the integral
    analytically.

    FAITHFUL (closed-form algebra): in the active, uncapped, un-floored regime
    this is a term-for-term transcription of the gSAM/WRF Thompson ``vts`` block
    (module_mp_thompson.f90:2751-2762) — oracle-pinned at coefficient level
    (av_s/bv_s/fv_s/mu_s/Kap0/Kap1/Lam0/Lam1 + the cse gamma exponents) by
    ``tests/unit/test_thompson_snow_fall_speed_faithful.py``.  The surrounding
    JAX numerical guards are DELIBERATE departures, not the oracle: the
    q_s>_QS_SMALL activation gate (vs gSAM's R1=1e-18 threshold + next-level
    inheritance), the [0, _VT_CLIP_SNOW] speed clip, the [-55, -0.1] °C Field-fit
    tc clamp (gSAM clamps only the −0.1 upper end), and the safe_pow/clip floors.
    """
    # Evaluate the PSD moments on a floored q_s so the zero-snow branch cannot
    # drive lam0->0: den ~ lam0^-(bm_s+1) then underflows to +inf in float32 and
    # poisons the VJP (grad -> NaN). The `active` gate below still returns V=0
    # there, so the active-regime value is unchanged (maximum picks q_s when
    # q_s>_QS_SMALL). float32 is the default finite-volume dtype, so this matters.
    q_pos = jnp.clip(q_s, 0.0)
    active = q_pos > _QS_SMALL
    M2, M3, ratio = _snow_moments(jnp.maximum(q_pos, _QS_SMALL), rho, T)
    lam0 = _LAM0 * ratio
    lam1 = _LAM1 * ratio
    p_v = _BM_S + _BV_S
    # Numerator integral with the fall-speed exp(-fv_s D) folded in.
    g0 = jnp.exp(jax.lax.lgamma(p_v + 1.0))
    g1 = jnp.exp(jax.lax.lgamma(p_v + _MU_S + 1.0))
    num = (_KAP0 * g0 / safe_pow(jnp.clip(lam0 + _FV_S, 1.0e-30), p_v + 1.0)
           + _KAP1 * safe_pow(jnp.clip(ratio, 1.0e-30), _MU_S) * g1
           / safe_pow(jnp.clip(lam1 + _FV_S, 1.0e-30), p_v + _MU_S + 1.0))
    # Denominator integral ∫ D^bm_s N dD (mass-normalized; M2^4/M3^3 cancels).
    gd0 = jnp.exp(jax.lax.lgamma(_BM_S + 1.0))
    gd1 = jnp.exp(jax.lax.lgamma(_BM_S + _MU_S + 1.0))
    den = (_KAP0 * gd0 / safe_pow(jnp.clip(lam0, 1.0e-30), _BM_S + 1.0)
           + _KAP1 * safe_pow(jnp.clip(ratio, 1.0e-30), _MU_S) * gd1
           / safe_pow(jnp.clip(lam1, 1.0e-30), _BM_S + _MU_S + 1.0))
    rhof = jnp.sqrt(_RHO_NOT / jnp.clip(rho, _RHO_FLOOR))
    V_s = _AV_S * rhof * num / jnp.clip(den, 1.0e-30)
    # Gate on actual snow; cap at a realistic aggregate fall speed.
    return jnp.where(active, jnp.clip(V_s, 0.0, _VT_CLIP_SNOW), 0.0)


def snow_deposition(q_v, q_s, q_sat_i, T, p_full, rho, dt):
    """Thompson-2008 snow vapour deposition / sublimation rate [kg/kg/s].

    Ventilated capacitance growth (Pruppacher-Klett) with Thompson's snow
    ventilation constants:

        PRDS = 4π·C_sqrd·(S_i−1)/(A+B)
               · [ t1_qs_sd·I(1) + t2_qs_sd·ρ_f^¼·I(c_vent) ] / ρ,

    driven by the DIMENSIONLESS ice supersaturation ratio ``S_i−1 =
    q_v/q_sat_i − 1`` (WRF ``ssati``) — the numerator that pairs with the
    ``A+B`` thermodynamic-resistance denominator (Rogers-Yau 9.4 /
    Pruppacher-Klett 13-76).  ``t1_qs_sd = 0.86``, ``t2_qs_sd =
    0.28·Sc^⅓·√av_s``, ventilation moment order ``c_vent = 1 + (1+bv_s)/2``
    (= ``cse(16)`` in WRF), with the half-slope ventilation exp folded into
    ``I``; the ventilation density correction is WRF ``rhof2 = √rhof =
    (ρ0/ρ)^¼`` (ventilation ∝ √Re, Re carries ONE fall-speed factor ρ_f).
    Dimensional closure to [kg/kg/s]: 4πC(S_i−1)/(A+B) is a per-particle
    growth rate [kg/s]; the moment sum ``vent`` [1/m²] integrates it over the
    per-volume PSD (M2 = ρ·q_s/am_s) giving [kg m⁻³ s⁻¹]; the trailing 1/ρ
    converts to mixing ratio.  WRF's diffusion form carries the same closure
    implicitly — its prefactor ``rvs = ρ·qvsi`` cancels the 1/ρ.  Deposition
    (S_i>1) is capped at the available supersaturation EXCESS ``q_v−q_sat_i``
    [kg/kg]; sublimation (S_i<1) is donor-clamped to the snow mass.

    DEPARTURE (documented, not pinned): the snow capacitance ``_C_SQRD = 0.15``
    is FIXED, whereas gSAM ramps it with temperature
    ``C_snow = clip(C_sqrd + (tc+15)(C_cube-C_sqrd)/(-15), C_sqrd, C_cube)`` over
    ``[C_sqrd, C_cube] = [0.3, 0.5]`` (module_mp_thompson.f90:2032-2033). legoESM's
    0.15 is 50-70% lower than that ramp (gSAM's capacitance is 2-3⅓× larger),
    so it SCALES DOWN the uncapped C-dependent raw deposition rate by that same
    factor -- an identified follow-up (adopting the ramped C would change snow
    growth and needs its own validation), which is why PRDS is not
    coefficient-pinned like the fall speed above.  (The final PRDS is not
    categorically smaller: the availability cap and donor clamp can bind first.)
    """
    # Floor q_s for the PSD moments (see snow_fall_speed): at zero snow lam0->0
    # so _psd_integral's bare lam0^-(p+1) underflows to +inf while norm underflows
    # to 0, giving 0*inf = NaN in the PRIMAL that poisons the final where's VJP.
    # The q_s>_QS_SMALL gate below still returns 0, so the active value is intact.
    q_pos = jnp.clip(q_s, 0.0)
    active = q_pos > _QS_SMALL
    M2, M3, ratio = _snow_moments(jnp.maximum(q_pos, _QS_SMALL), rho, T)
    lam0 = _LAM0 * ratio
    lam1 = _LAM1 * ratio
    # Thermodynamic resistance A+B (ice): A = L_s²/(K_a R_v T²),
    # B = R_v T/(e_si·D_v); e_si ≈ q_sat_i·p/ε. D_v vapour diffusivity, K_a air
    # conductivity (standard Pruppacher-Klett forms).
    dv = _DV_PREFACTOR * safe_pow(T, _DV_T_EXP) / jnp.clip(p_full, 1.0)
    ka = _KA_A + _KA_SLOPE * (T - constants.T_freeze)
    e_si = jnp.clip(q_sat_i, 1.0e-12) * p_full / constants.epsilon
    A = constants.L_s ** 2 / (jnp.clip(ka, 1.0e-6) * constants.R_v * T ** 2)
    B = constants.R_v * T / (jnp.clip(e_si, 1.0) * jnp.clip(dv, 1.0e-12))
    abi = jnp.clip(A + B, 1.0e-6)
    # DIMENSIONLESS ice supersaturation ratio (WRF ``ssati = qv/qvsi − 1``).
    # The A+B resistances above pair with S_i−1, NOT the mixing-ratio excess
    # q_v−q_sat_i = q_sat_i·(S_i−1): using the excess made snow deposition
    # weaker by a factor q_sat_i (~1e3-1e4 at cold upper-tropospheric T),
    # effectively disabling the stated major upper-tropospheric vapour sink.
    s_i = q_v / jnp.clip(q_sat_i, 1.0e-12) - 1.0
    # Mixing-ratio excess [kg/kg] — used only for the deposition availability cap.
    excess_i = q_v - q_sat_i
    # Ventilation integrals: t1 term ~ ∫ D N dD = I(1); t2 term ~ ventilation
    # moment with the half-slope ventilation exp(-fv_s D/2).
    I1 = _psd_integral(1.0, M2, M3, ratio)
    c_vent = 1.0 + (1.0 + _BV_S) / 2.0
    g0 = jnp.exp(jax.lax.lgamma(c_vent + 1.0))
    g1 = jnp.exp(jax.lax.lgamma(c_vent + _MU_S + 1.0))
    norm = jnp.clip(M2, 1.0e-30) * safe_pow(jnp.clip(ratio, 1.0e-30), 3.0)  # = M2^4/M3^3, f32-safe
    I_vent = norm * (
        _KAP0 * g0 / safe_pow(jnp.clip(lam0 + 0.5 * _FV_S, 1.0e-30), c_vent + 1.0)
        + _KAP1 * safe_pow(jnp.clip(ratio, 1.0e-30), _MU_S) * g1
        / safe_pow(jnp.clip(lam1 + 0.5 * _FV_S, 1.0e-30), c_vent + _MU_S + 1.0))
    sc3 = _SC ** (1.0 / 3.0)
    t1 = _SNOW_VENT_F0
    t2 = _SNOW_VENT_F1 * sc3 * jnp.sqrt(_AV_S)
    rhof = jnp.sqrt(_RHO_NOT / jnp.clip(rho, _RHO_FLOOR))
    # Ventilation Reynolds factor (WRF ``rhof2*vsc2``): the dynamic-viscosity
    # term ``vsc2 = √(ρ/μ)`` was missing — WRF's t2 ventilation term is
    # ``t2_qs_sd·rhof2·vsc2·smof`` (codex/WRF audit). μ via Sutherland.
    mu_air = _MU_PREFACTOR * safe_pow(T, _MU_T_EXP) / (T + _MU_SUTHERLAND_T)
    vsc2 = jnp.sqrt(jnp.clip(rho, _RHO_VISC_FLOOR) / jnp.clip(mu_air, 1.0e-8))
    # WRF ``rhof2(k) = SQRT(rhof(k)) = (ρ0/ρ)^¼`` — the VENTILATION density
    # correction (ventilation ∝ √Re, Re ∝ fall speed ∝ rhof), distinct from
    # the fall-speed correction rhof used in ``snow_fall_speed``.
    rhof2 = jnp.sqrt(rhof)
    vent = t1 * I1 + t2 * rhof2 * vsc2 * I_vent
    # Sign convention: prds > 0 = DEPOSITION (q_v sink, snow source, +L_s
    # heating upstream); prds < 0 = SUBLIMATION (q_v source, snow sink).
    # 1/ρ closes the per-volume PSD integral to [kg/kg/s] (see docstring).
    prds = (4.0 * jnp.pi * _C_SQRD * s_i / abi * vent
            / jnp.clip(rho, _RHO_FLOOR))
    # Cap deposition at the available supersaturation excess [kg/kg];
    # donor-clamp sublimation to q_s.
    dep_pos = jnp.minimum(jnp.maximum(prds, 0.0),
                          jnp.maximum(excess_i, 0.0) / jnp.clip(dt, 1.0))
    subl_neg = jnp.maximum(jnp.minimum(prds, 0.0),
                           -jnp.clip(q_s, 0.0) / jnp.clip(dt, 1.0))
    out = dep_pos + subl_neg
    return jnp.where(active, out, 0.0)
