"""Two-stream layer kernels of ecRad 49r1 (radiation_two_stream.F90), JAX port.

Line-by-line port of ``calc_ref_trans_lw`` (lines 108-160 of the compiled
file), ``calc_no_scattering_transmittance_lw`` (161-196) and
``calc_ref_trans_sw`` (260-330) of OpenIFS 49r1 (8f6f722), in fp64 with the
Fortran evaluation order kept (left-to-right, no contraction: the oracle is
built with ``-O1 -ffp-contract=off``).

Not reused from ``rrtmgp/rte/monochromatic_two_stream.py`` (RULE 4 check): that
module evaluates the same Meador-Weaver closures in a different algebraic
arrangement (denominator ``k(1+e2)+g1(1-e2)``, ``g1 = D(1-ssa(1+g)/2)``, no
thin-layer branch, direct beam through ``cos(zenith)``, different clamps), so it
cannot reproduce the 49r1 oracle to 1e-12.

Smoothing (user RULE 5, votes M1-M4 of the mcica-solver section): every traced
clamp/switch takes one static ``sharpness`` (default 100 = band of 0.01 x the
threshold scale; ``math.inf`` = the oracle's hard form, exactly):

* one-sided clamps/floors are "inactive-exact": ``max(a, lo) = a + pp(lo - a)``
  and ``min(a, hi) = a - pp(a - hi)`` with ``pp = smooth.positive_part_c1`` on
  the violation in units of the scale, and the bound itself returned exactly
  once the violation exceeds the band.  Where the oracle clamp does not bind
  the value is the oracle value; inside the band the bound can be passed by at
  most 4/27 of the band;
* formula switches are a centred cubic smoothstep in ``(x - thr)/thr``
  (exactly the oracle branch outside the band ``thr*(1 +- 0.5/sharpness)``),
  blended ``w*A + (1-w)*B`` with both branches evaluated;
* the singularity guard on 1 - k^2 mu0^2 is a smoothstep blend in |x| at
  eps (vote N3); the thick-od branch argument is an inactive-exact floor
  at 0.5e-3 where that branch's weight is exactly zero (vote N4).

KNOWN DIFFERENCE at the default sharpness: inputs inside a band (k^2 within
1% above 1e-12, od within 0.5% of 1e-3, clamp violations smaller than 1% of
the scale) get the blended value, not the oracle's; outside every band the
result is the hard form.  ``sharpness=math.inf`` recovers the oracle.
"""

from __future__ import annotations

import math

import jax.numpy as jnp

from legoesm.atmosphere.physics.radiation.ecrad_ifs._libm import exp_libm
from legoesm.core import smooth

# --- ecRad 49r1 two-stream constants (radiation_two_stream.F90) ---
# LwDiffusivityWP, radiation_two_stream.F90:6 (compiled file).
_LW_DIFFUSIVITY = 1.66
# od threshold between the thin- and thick-layer formulas, lines 136, 180.
_OD_THIN_THRESHOLD = 1.0e-3
# Floor on k^2 (Meador & Weaver eq. 18), lines 134-135 and 299-300 (fp64 build).
_K_SQUARED_FLOOR = 1.0e-12
# SW Eddington gammas, lines 288-291.
_SW_GAMMA_G_FACTOR = 0.75
_SW_GAMMA1_SSA_COEFF = 1.25
_SW_GAMMA2_SSA_COEFF = 0.75
# Floor on 1 - ssa*g in the LW thick-branch source divisor (vote N9; port-only, the oracle has none).
_ONE_MINUS_SSA_G_FLOOR = 1.0e-6
# Cap on the direct-beam optical path od/mu0 before exp, line 285.
_OD_OVER_MU0_CAP = 1000.0

__physics_contract__ = {
    "summary": (
        "ecRad 49r1 two-stream layer kernels: LW Meador-Weaver reflectance/"
        "transmittance with linear-in-optical-depth Planck sources, LW "
        "no-scattering transmittance and sources, SW Eddington (Meador-Weaver) "
        "diffuse and direct reflectance/transmittance. Port of "
        "radiation_two_stream.F90 (OpenIFS 49r1 8f6f722)."
    ),
    "inputs": {
        "od": "1 (layer optical depth per g-point)",
        "ssa": "1", "asymmetry": "1",
        "planck_top": "W/m^2 (g-point Planck flux at layer top)",
        "planck_bot": "W/m^2", "mu0": "1 (cosine solar zenith angle, >0)",
    },
    "outputs": {
        "reflectance": "1", "transmittance": "1",
        "source_up": "W/m^2", "source_dn": "W/m^2",
        "ref_dir": "1 (per unit incoming direct flux normal to the beam)",
        "trans_dir_diff": "1", "trans_dir_dir": "1",
    },
    "sign_convention": (
        "Optical depth >= 0; reflectances/transmittances in [0,1]; source_up "
        "is emitted upward from the layer top, source_dn downward from the "
        "layer base; ref_dir/trans_dir_diff are bounded by mu0*(1-trans_dir_dir) "
        "(oracle clamps). Energy: per layer R + T <= 1 for ssa <= 1."
    ),
    "conserves": ["energy"],
    "differentiable": True,
    "reference": (
        "Meador & Weaver (1980) J. Atmos. Sci. 37, 630-643; Hogan & Bozzo "
        "(2018) JAMES 10, 1990-2008; oracle radiation_two_stream.F90 (49r1)."
    ),
    "idealized_test": (
        "tests/atmosphere/test_ecrad_ifs_mcica_solver_oracle.py (oracle match "
        "1e-12 hard form, conservative-scattering column, gradient checks)."
    ),
}


def _hard(sharpness) -> bool:
    return sharpness == math.inf


def lower_clamp(x, lo, scale, sharpness):
    """Inactive-exact ``max(x, lo)`` (vote M1a), band ``scale/sharpness`` below ``lo``."""
    if _hard(sharpness):
        return jnp.maximum(x, lo)
    v = (lo - x) / scale
    return jnp.where(v * sharpness >= 1.0, lo, x + scale * smooth.positive_part_c1(v, sharpness))


def upper_clamp(x, hi, scale, sharpness):
    """Inactive-exact ``min(x, hi)`` (vote M1a), band ``scale/sharpness`` above ``hi``."""
    if _hard(sharpness):
        return jnp.minimum(x, hi)
    v = (x - hi) / scale
    return jnp.where(v * sharpness >= 1.0, hi, x - scale * smooth.positive_part_c1(v, sharpness))


def step(x, thr, scale, sharpness, strict):
    """Weight of the ``x >= thr`` (``x > thr`` if strict) branch; centred cubic
    smoothstep of half-width ``0.5*scale/sharpness`` (votes M2a, M3a)."""
    if _hard(sharpness):
        x = jnp.asarray(x)
        return ((x > thr) if strict else (x >= thr)).astype(x.dtype)
    return smooth.smoothstep_cubic((x - thr) * (sharpness / scale) + 0.5)


def blend(w, a, b):
    """``w*a + (1-w)*b``, returning ``a``/``b`` exactly where ``w`` is 1/0."""
    return jnp.where(w >= 1.0, a, jnp.where(w <= 0.0, b, w * a + (1.0 - w) * b))


def _k_exponent(gamma1, gamma2, sharpness):
    k2 = (gamma1 - gamma2) * (gamma1 + gamma2)
    return jnp.sqrt(lower_clamp(k2, _K_SQUARED_FLOOR, _K_SQUARED_FLOOR, sharpness))


def _od_thick_weight_and_safe_od(od, sharpness):
    # Fortran ``if (od > 1.0e-3)``: strict.  The thick branch is evaluated at an
    # od held away from 0 (vote M4a) where its weight is exactly zero.
    w = step(od, _OD_THIN_THRESHOLD, _OD_THIN_THRESHOLD, sharpness, strict=True)
    floor = 0.5 * _OD_THIN_THRESHOLD
    od_safe = lower_clamp(od, floor, floor, sharpness)  # vote N4
    return w, od_safe


def calc_ref_trans_lw(od, ssa, asymmetry, planck_top, planck_bot, sharpness=100.0):
    """radiation_two_stream.F90 calc_ref_trans_lw (49r1 lines 108-160).

    Returns (reflectance, transmittance, source_up, source_dn), elementwise.
    """
    factor = (_LW_DIFFUSIVITY * 0.5) * ssa
    gamma1 = _LW_DIFFUSIVITY - factor * (1.0 + asymmetry)
    gamma2 = factor * (1.0 - asymmetry)
    k = _k_exponent(gamma1, gamma2, sharpness)
    w, od_t = _od_thick_weight_and_safe_od(od, sharpness)

    # Thick branch (od > 1e-3).
    e = exp_libm(-k * od_t)
    e2 = e * e
    rtf = 1.0 / (k + gamma1 + (k - gamma1) * e2)
    r_thick = gamma2 * (1.0 - e2) * rtf
    t_thick = 2.0 * k * e * rtf
    # gamma1 + gamma2 = 1.66 (1 - ssa g) is floored at 1.66e-6 (vote N9):
    # oracle-exact wherever ssa*g <= 1 - 1e-6; at ssa = g = 1 the oracle
    # divides by zero (NaN), here the thick source stays finite.
    gamma_sum_floor = _LW_DIFFUSIVITY * _ONE_MINUS_SSA_G_FLOOR
    gamma_sum = lower_clamp(gamma1 + gamma2, gamma_sum_floor, gamma_sum_floor, sharpness)
    coeff = (planck_bot - planck_top) / (od_t * gamma_sum)
    c_up_top = coeff + planck_top
    c_up_bot = coeff + planck_bot
    c_dn_top = -coeff + planck_top
    c_dn_bot = -coeff + planck_bot
    su_thick = c_up_top - r_thick * c_dn_top - t_thick * c_up_bot
    sd_thick = c_dn_bot - r_thick * c_up_bot - t_thick * c_dn_top

    # Thin branch.
    r_thin = gamma2 * od
    # Dead-branch guard as for the thick divisor: 1 + od*(gamma1 - k) can be 0
    # at large od, where the thin weight 1 - w is exactly 0.
    t_thin = (1.0 - k * od) / jnp.where(w < 1.0, 1.0 + od * (gamma1 - k), 1.0)
    s_thin = (1.0 - r_thin - t_thin) * 0.5 * (planck_top + planck_bot)

    return (
        blend(w, r_thick, r_thin),
        blend(w, t_thick, t_thin),
        blend(w, su_thick, s_thin),
        blend(w, sd_thick, s_thin),
    )


def calc_no_scattering_transmittance_lw(od, planck_top, planck_bot, sharpness=100.0):
    """radiation_two_stream.F90 calc_no_scattering_transmittance_lw (161-196).

    Returns (transmittance, source_up, source_dn), elementwise.
    """
    transmittance = exp_libm(-_LW_DIFFUSIVITY * od)
    coeff_thin = _LW_DIFFUSIVITY * od
    w, od_t = _od_thick_weight_and_safe_od(od, sharpness)

    coeff = (planck_bot - planck_top) / (_LW_DIFFUSIVITY * od_t)
    c_up_top = coeff + planck_top
    c_up_bot = coeff + planck_bot
    c_dn_top = -coeff + planck_top
    c_dn_bot = -coeff + planck_bot
    su_thick = c_up_top - transmittance * c_up_bot
    sd_thick = c_dn_bot - transmittance * c_dn_top

    s_thin = coeff_thin * 0.5 * (planck_top + planck_bot)
    return transmittance, blend(w, su_thick, s_thin), blend(w, sd_thick, s_thin)


def calc_ref_trans_sw(mu0, od, ssa, asymmetry, sharpness=100.0):
    """radiation_two_stream.F90 calc_ref_trans_sw (49r1 lines 260-330).

    ``mu0`` > 0 broadcastable to ``od``.  Returns (ref_diff, trans_diff,
    ref_dir, trans_dir_diff, trans_dir_dir).
    """
    eps = jnp.finfo(jnp.result_type(od)).eps
    # trans_dir_dir = exp(max(-max(od*(1/mu0), 0), -1000)), line 285-286.
    od_over_mu0 = lower_clamp(od * (1.0 / mu0), 0.0, 1.0, sharpness)
    trans_dir_dir = exp_libm(
        lower_clamp(-od_over_mu0, -_OD_OVER_MU0_CAP, _OD_OVER_MU0_CAP, sharpness)
    )

    factor = _SW_GAMMA_G_FACTOR * asymmetry
    gamma1 = 2.0 - ssa * (_SW_GAMMA1_SSA_COEFF + factor)
    gamma2 = ssa * (_SW_GAMMA2_SSA_COEFF - factor)
    gamma3 = 0.5 - mu0 * factor
    gamma4 = 1.0 - gamma3
    alpha1 = gamma1 * gamma4 + gamma2 * gamma3
    alpha2 = gamma1 * gamma3 + gamma2 * gamma4
    k = _k_exponent(gamma1, gamma2, sharpness)
    exponential = exp_libm(-k * od)

    k_mu0 = k * mu0
    one_minus_kmu0_sqr = 1.0 - k_mu0 * k_mu0
    k_gamma3 = k * gamma3
    k_gamma4 = k * gamma4
    exponential2 = exponential * exponential
    k_2_exponential = 2.0 * k * exponential
    rtf = 1.0 / (k + gamma1 + (k - gamma1) * exponential2)
    ref_diff = gamma2 * (1.0 - exponential2) * rtf
    trans_diff = lower_clamp(
        upper_clamp(k_2_exponential * rtf, 1.0 - ref_diff, 1.0, sharpness), 0.0, 1.0, sharpness
    )
    # merge(x, eps, abs(x) > eps): removable-singularity guard (vote N3).
    # Reciprocals are blended (a denominator blend would cross 0 at x = -eps);
    # 1/x is evaluated away from 0 only where its weight is exactly zero.
    x = one_minus_kmu0_sqr
    w_den = step(jnp.abs(x), eps, eps, sharpness, strict=True)
    x_safe = jnp.where(jnp.abs(x) >= 0.5 * eps, x, eps)
    num = mu0 * ssa * rtf
    rtf = blend(w_den, num / x_safe, num / eps)
    ref_dir = rtf * (
        (1.0 - k_mu0) * (alpha2 + k_gamma3)
        - ((1.0 + k_mu0) * (alpha2 - k_gamma3)) * exponential2
        - (k_2_exponential * (gamma3 - alpha2 * mu0)) * trans_dir_dir
    )
    trans_dir_diff = rtf * (
        k_2_exponential * (gamma4 + alpha1 * mu0)
        - trans_dir_dir * ((1.0 + k_mu0) * (alpha1 + k_gamma4) - ((1.0 - k_mu0) * (alpha1 - k_gamma4)) * exponential2)
    )
    beam_scattered = mu0 * (1.0 - trans_dir_dir)
    ref_dir = lower_clamp(upper_clamp(ref_dir, beam_scattered, 1.0, sharpness), 0.0, 1.0, sharpness)
    trans_dir_diff = lower_clamp(
        upper_clamp(trans_dir_diff, beam_scattered - ref_dir, 1.0, sharpness), 0.0, 1.0, sharpness
    )
    return ref_diff, trans_diff, ref_dir, trans_dir_diff, trans_dir_dir
