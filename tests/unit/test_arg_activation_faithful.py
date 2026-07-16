"""ARG2000 aerosol-activation ORACLE-FAITHFULNESS tests.

`arg_activation` implements the Abdul-Razzak & Ghan (2000) modal-activation
closed form.  The oracle is the ARG2000 paper, cross-checked TERM-FOR-TERM
against the on-disk gSAM/M2005 reference code
``MICRO_M2005/module_mp_graupel.f90:2307-2340`` (``IACT=2``): AACT/ALPHA/GAMM/GG/
PSI/ETA/SM/F11/F21/DUM1/SMAX/UU/erfc.  legoESM generalizes gSAM's fixed two-mode
block to the ARG2000 Part-2 ``sum_i`` over N modes.

The existing tests/unit/test_arg_activation.py checks only bounds/monotonicity/
dispatch/grad — never the closed FORM or its coefficients.  These pin the S_max
balance + activated-fraction against an INDEPENDENT NumPy transcription (rel
1e-9), canary the f/g/exponent constants against their literals, and canary the
SIGVL and 3*sqrt(2) departures specifically (the constant-vs-T-ramped surface
tension and the exact-3*sqrt(2)-vs-gSAM-4.242 erfc denominator).  The other
documented differences (DV/KAP, e_s curve, molar constants, kappa-vs-BACT,
velocity gating, per-mass vs per-volume, tendency-vs-diagnostic) are NOT
canaried here.

legoESM follows the ARG2000 PAPER / CliMA constant-transport-coefficient form;
gSAM/M2005 substitutes empirical fits.  The paper-form oracle below therefore
uses the SAME constant coefficients legoESM uses (constants.sigma_water /
D_vapor / k_air) and the SAME shared saturation curve, so it pins legoESM's
algebra exactly; the SIGVL and 3*sqrt(2) departures are canaried separately.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm import constants                                       # noqa: E402
from legoesm.thermo import saturation_vapor_pressure                # noqa: E402
from legoesm.atmosphere.physics.microphysics.arg_activation import (  # noqa: E402
    arg_cdnc,
    arg_activated_fraction,
    kohler_curvature_A,
    _ARG_F_PREFACTOR,
    _ARG_F_LOGSQ_COEFF,
    _ARG_G_LOG_COEFF,
    _ARG_SMAX_POW_1,
    _ARG_SMAX_POW_2,
)

_PER_CM3 = 1.0e6  # cm^-3 -> m^-3


def _arg_oracle(w, T, p, N, r_g, sigma, kappa,
                pow1=1.5, pow2=0.75, denom=None):
    """Independent ARG2000 paper-form (S_max, per-mode fraction, CDNC).

    Transcribes the gSAM M2005 module_mp_graupel.f90:2307-2340 algebra with the
    ARG/CliMA CONSTANT transport coefficients legoESM uses (constant sigma_w,
    D_v, k_a; exact 3*sqrt(2)).  Scalars ``w, T, p``; 1-D per-mode arrays.  The
    saturation vapour pressure ``e_s`` comes from production thermo (a shared
    upstream input, NOT part of the ARG algebra under test).  ``pow1/pow2/denom``
    are exposed only so the non-vacuity canaries can perturb the exponents /
    erfc denominator.
    """
    Rv, rho_w, cpd, Rd = (constants.R_v, constants.rho_water,
                          constants.c_pd, constants.R_d)
    g, Lv, eps = constants.g, constants.L_v, constants.epsilon
    Dv, ka, sig_w = constants.D_vapor, constants.k_air, constants.sigma_water
    if denom is None:
        denom = 3.0 * math.sqrt(2.0)
    e_s = float(saturation_vapor_pressure(jnp.asarray(T, dtype=float)))
    N = np.asarray(N, float)
    r_g = np.asarray(r_g, float)
    sigma = np.asarray(sigma, float)
    kappa = np.asarray(kappa, float)

    A = 2.0 * sig_w / (rho_w * Rv * T)                       # AACT (const sigma)
    alpha = g * Lv / (cpd * Rv * T ** 2) - g / (Rd * T)      # ALPHA
    gamma = Rv * T / e_s + eps * Lv ** 2 / (cpd * p * T)     # GAMM
    G = 1.0 / (rho_w * Rv * T / (e_s * Dv)                   # GG
               + rho_w * Lv / (ka * T) * (Lv / (Rv * T) - 1.0))
    aw_over_G = alpha * w / G
    zeta = (2.0 * A / 3.0) * math.sqrt(aw_over_G)            # PSI
    S_m = (2.0 / np.sqrt(kappa)) * (A / (3.0 * r_g)) ** 1.5  # SM
    eta = aw_over_G ** 1.5 / (2.0 * math.pi * rho_w * gamma * N)  # ETA
    ln_sig = np.log(sigma)
    f = 0.5 * np.exp(2.5 * ln_sig ** 2)                      # F11
    gg = 1.0 + 0.25 * ln_sig                                 # F21
    inv_smax2 = np.sum((1.0 / S_m ** 2) * (                  # DUM1 summed
        f * (zeta / eta) ** pow1
        + gg * (S_m ** 2 / (eta + 3.0 * zeta)) ** pow2))
    S_max = math.sqrt(1.0 / inv_smax2)                       # SMAX
    u = 2.0 * np.log(S_m / S_max) / (denom * ln_sig)         # UU
    frac = 0.5 * np.array([math.erfc(x) for x in np.atleast_1d(u)])  # (1-erf)/2
    cdnc = float(np.sum(N * frac))
    return S_max, frac, cdnc


def _prod(w, T, p, N, r_g, sigma, kappa):
    """Production arg_cdnc(return_diagnostics) at spatial shape (1,)."""
    return arg_cdnc(
        jnp.array([w]), jnp.array([T]), jnp.array([p]),
        jnp.asarray(N, dtype=float), jnp.asarray(r_g, dtype=float),
        jnp.asarray(sigma, dtype=float), jnp.asarray(kappa, dtype=float),
        return_diagnostics=True)


# --- Full closed-form pin vs the independent ARG2000 paper oracle --------------


@pytest.mark.parametrize("w,T,p", [(0.3, 285.0, 9.5e4),
                                   (1.0, 270.0, 8.0e4),
                                   (0.1, 290.0, 1.0e5)])
def test_arg_single_mode_matches_paper_oracle(w, T, p):
    """S_max, activated fraction, and CDNC match the independent ARG2000 closed
    form to round-off for a single accumulation mode (all inputs above the AD
    floors, so the production clamps are inactive)."""
    N, r_g, sig, kap = [100.0 * _PER_CM3], [0.05e-6], [2.0], [0.6]
    S_max_o, frac_o, cdnc_o = _arg_oracle(w, T, p, N, r_g, sig, kap)
    cdnc_p, S_max_p, frac_p = _prod(w, T, p, N, r_g, sig, kap)
    assert float(S_max_p[0]) == pytest.approx(S_max_o, rel=1e-9)
    assert float(frac_p[0, 0]) == pytest.approx(frac_o[0], rel=1e-9)
    assert float(cdnc_p[0]) == pytest.approx(cdnc_o, rel=1e-9)


def test_arg_two_mode_matches_paper_oracle():
    """The multi-mode 1/S_max^2 = sum_i (…) generalization of gSAM's fixed
    NANEW1/NANEW2 block matches the independent oracle for a 2-mode (accumulation
    + coarse) population — pins the per-mode sum, both S_m,i and both frac_i."""
    w, T, p = 0.3, 285.0, 9.5e4
    N = [100.0 * _PER_CM3, 5.0 * _PER_CM3]
    r_g, sig, kap = [0.05e-6, 0.5e-6], [2.0, 2.2], [0.6, 0.6]
    S_max_o, frac_o, cdnc_o = _arg_oracle(w, T, p, N, r_g, sig, kap)
    cdnc_p, S_max_p, frac_p = _prod(w, T, p, N, r_g, sig, kap)
    assert float(S_max_p[0]) == pytest.approx(S_max_o, rel=1e-9)
    assert float(frac_p[0, 0]) == pytest.approx(frac_o[0], rel=1e-9)
    assert float(frac_p[1, 0]) == pytest.approx(frac_o[1], rel=1e-9)
    assert float(cdnc_p[0]) == pytest.approx(cdnc_o, rel=1e-9)


# --- Mode-factor / exponent constants match the gSAM F11/F21/DUM1 literals ----


def test_arg_mode_factor_constants_match_gsam():
    """f_i = 0.5 exp(2.5 ln^2 sigma) (gSAM F11), g_i = 1 + 0.25 ln sigma (F21),
    and the S_max-sum exponents 1.5 / 0.75 (gSAM DUM1) equal their literals."""
    assert _ARG_F_PREFACTOR == 0.5
    assert _ARG_F_LOGSQ_COEFF == 2.5
    assert _ARG_G_LOG_COEFF == 0.25
    assert _ARG_SMAX_POW_1 == 1.5
    assert _ARG_SMAX_POW_2 == 0.75


def test_arg_smax_exponents_are_load_bearing():
    """Non-vacuity of the closed-form pin: perturbing either S_max-sum exponent
    (1.5 -> 1.4 or 0.75 -> 0.7) moves S_max by >1%, so the rel-1e-9 form match is
    genuinely sensitive to the gSAM DUM1 structure."""
    args = (0.3, 285.0, 9.5e4, [100.0 * _PER_CM3], [0.05e-6], [2.0], [0.6])
    s_ref = _arg_oracle(*args)[0]
    s_p1 = _arg_oracle(*args, pow1=1.4)[0]
    s_p2 = _arg_oracle(*args, pow2=0.7)[0]
    assert abs(s_p1 - s_ref) / s_ref > 0.01
    assert abs(s_p2 - s_ref) / s_ref > 0.01


# --- Departure canaries (legoESM = ARG2000 paper; gSAM/M2005 refines) ---------


def test_arg_erfc_denominator_is_exact_3sqrt2_not_gsam_4242():
    """The activated-fraction erfc denominator is the EXACT 3*sqrt(2) (ARG2000
    paper), not gSAM's rounded literal 4.242.  Construct S_m/S_max = e and
    sigma = e so u = 2/(denom); production must match the 3*sqrt(2) value and
    DIFFER from the 4.242 value (the departure is real, the pin non-vacuous)."""
    S_m = jnp.array([[math.e]])          # (n_modes=1,) + F=(1,)
    S_max = jnp.array([1.0])             # F=(1,)
    sigma = jnp.array([[math.e]])        # broadcastable per-mode sigma
    frac = arg_activated_fraction(S_m, S_max, sigma)
    u_exact = 2.0 / (3.0 * math.sqrt(2.0))
    u_gsam = 2.0 / 4.242
    assert float(frac[0, 0]) == pytest.approx(0.5 * math.erfc(u_exact), rel=1e-12)
    # gSAM's rounded 4.242 would give a measurably different (larger u) fraction.
    assert 0.5 * math.erfc(u_exact) != pytest.approx(
        0.5 * math.erfc(u_gsam), rel=1e-9)


def test_arg_kelvin_uses_constant_sigma_not_gsam_sigvl_ramp():
    """legoESM's Kelvin A uses the CONSTANT constants.sigma_water; gSAM ramps the
    surface tension SIGVL = 0.0761 - 1.55e-4*(T - T_freeze) [N/m] inside AACT.
    This ISOLATES the surface-tension ramp: it holds the gas constants fixed at
    legoESM's values (gSAM's true AACT also uses its own MW/RR molar constants,
    a separate documented difference), so ``A_with_gsam_sigvl`` is NOT gSAM's
    literal AACT — it differs from A_lego by EXACTLY the sigma ratio."""
    T = 285.0
    A_lego = float(kohler_curvature_A(jnp.array(T)))
    A_const = 2.0 * constants.sigma_water / (
        constants.rho_water * constants.R_v * T)
    assert A_lego == pytest.approx(A_const, rel=1e-12)
    # gSAM SIGVL(T) surface-tension ramp (module_mp_graupel.f90:2307), with the
    # gas constants held at legoESM's values to isolate the SIGVL effect alone.
    sigvl_gsam = 0.0761 - 1.55e-4 * (T - float(constants.T_freeze))
    A_with_gsam_sigvl = 2.0 * sigvl_gsam / (
        constants.rho_water * constants.R_v * T)
    assert A_lego != pytest.approx(A_with_gsam_sigvl, rel=1e-6)   # departure is real
    assert A_lego / A_with_gsam_sigvl == pytest.approx(
        constants.sigma_water / sigvl_gsam, rel=1e-9)            # exactly sigma ratio


# --- AD-safety (x64 and float32) ----------------------------------------------


def _cdnc_scalar(w, n, r, k, sig, T=285.0, p=9.5e4):
    return arg_cdnc(
        jnp.array([w]), jnp.array([T]), jnp.array([p]),
        jnp.array([n]), jnp.array([r]), jnp.array([sig]), jnp.array([k]))[0]


def test_arg_cdnc_grad_finite_below_input_floors_x64():
    """grad of CDNC wrt all five ARG inputs (w, N_a, r_g, kappa, sigma) is finite
    at a normal point AND STRICTLY BELOW every input floor (w=0, N_a=0, r_g=0,
    kappa=0, sigma=1 — below _W_MIN/_N_A_FLOOR/_R_G_FLOOR/_KAPPA_FLOOR/
    _SIGMA_G_FLOOR), where the maximum(...) input clamps engage.  The clamps must
    not create a NaN VJP (a sqrt/log/1-over of a zeroed input would).  This
    exercises the INPUT floors only, not the internal 1/S_max^2 / S_m-ratio
    floors."""
    for w, n, r, k, sig in ((0.3, 1.0e8, 5.0e-8, 0.6, 2.0),
                            (0.0, 0.0, 0.0, 0.0, 1.0)):
        gs = jax.grad(_cdnc_scalar, argnums=(0, 1, 2, 3, 4))(w, n, r, k, sig)
        assert all(bool(jnp.isfinite(g)) for g in gs), (w, n, r, k, sig, gs)


def test_arg_cdnc_grad_finite_float32():
    """CDNC gradient wrt updraft is finite in float32 (the default finite-volume
    dtype) — the sqrt/pow/log/erfc chain must not underflow to a NaN VJP the way
    x64 would mask (cf. the Thompson snow fall-speed float32 trap)."""
    _was = getattr(jax.config, "jax_enable_x64", False)
    jax.config.update("jax_enable_x64", False)
    try:
        def cdnc_f32(w):
            return arg_cdnc(
                jnp.asarray([w], jnp.float32), jnp.asarray([285.0], jnp.float32),
                jnp.asarray([9.5e4], jnp.float32),
                jnp.asarray([1.0e8], jnp.float32), jnp.asarray([5.0e-8], jnp.float32),
                jnp.asarray([2.0], jnp.float32), jnp.asarray([0.6], jnp.float32))[0]
        g = jax.grad(cdnc_f32)(jnp.float32(0.3))
        assert g.dtype == jnp.float32
        assert bool(jnp.isfinite(g))
    finally:
        jax.config.update("jax_enable_x64", _was)
