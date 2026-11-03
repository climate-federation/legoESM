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

import jax
import jax.numpy as jnp

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

# --- Correctly-rounded fp64 exp (vote M8) ---
# The oracle's exp is glibc libm, correctly rounded on every sample measured;
# XLA's CPU exp differs from it by 1 ulp on ~14% of inputs, which the SW
# kernel amplifies to 1.5e-10 through 1 - exp(-2 k od) at od ~ 1e-6.
# Three-part ln 2 (42 + 42 + 53 bits; k*_LN2_HI and k*_LN2_MID exact for |k| < 2^11).
_LN2_HI = float.fromhex("0x1.62e42fefa3800p-1")
_LN2_MID = float.fromhex("0x1.ef35793c76000p-45")
_LN2_LO = float.fromhex("0x1.cc01f97b57a08p-87")
_INV_LN2 = float.fromhex("0x1.71547652b82fep+0")
# Double-double 1/i!, i = 0..24 (truncation 1.5e-35 for |r| <= ln2/2).
_INV_FACTORIAL_DD = (
    (float.fromhex("0x1.0000000000000p+0"), float.fromhex("0x0.0p+0")),  # 1/0!
    (float.fromhex("0x1.0000000000000p+0"), float.fromhex("0x0.0p+0")),  # 1/1!
    (float.fromhex("0x1.0000000000000p-1"), float.fromhex("0x0.0p+0")),  # 1/2!
    (float.fromhex("0x1.5555555555555p-3"), float.fromhex("0x1.5555555555555p-57")),  # 1/3!
    (float.fromhex("0x1.5555555555555p-5"), float.fromhex("0x1.5555555555555p-59")),  # 1/4!
    (float.fromhex("0x1.1111111111111p-7"), float.fromhex("0x1.1111111111111p-63")),  # 1/5!
    (float.fromhex("0x1.6c16c16c16c17p-10"), float.fromhex("-0x1.f49f49f49f49fp-65")),  # 1/6!
    (float.fromhex("0x1.a01a01a01a01ap-13"), float.fromhex("0x1.a01a01a01a01ap-73")),  # 1/7!
    (float.fromhex("0x1.a01a01a01a01ap-16"), float.fromhex("0x1.a01a01a01a01ap-76")),  # 1/8!
    (float.fromhex("0x1.71de3a556c734p-19"), float.fromhex("-0x1.c154f8ddc6c00p-73")),  # 1/9!
    (float.fromhex("0x1.27e4fb7789f5cp-22"), float.fromhex("0x1.cbbc05b4fa99ap-76")),  # 1/10!
    (float.fromhex("0x1.ae64567f544e4p-26"), float.fromhex("-0x1.c062e06d1f209p-80")),  # 1/11!
    (float.fromhex("0x1.1eed8eff8d898p-29"), float.fromhex("-0x1.2aec959e14c06p-83")),  # 1/12!
    (float.fromhex("0x1.6124613a86d09p-33"), float.fromhex("0x1.f28e0cc748ebep-87")),  # 1/13!
    (float.fromhex("0x1.93974a8c07c9dp-37"), float.fromhex("0x1.05d6f8a2efd1fp-92")),  # 1/14!
    (float.fromhex("0x1.ae7f3e733b81fp-41"), float.fromhex("0x1.1d8656b0ee8cbp-97")),  # 1/15!
    (float.fromhex("0x1.ae7f3e733b81fp-45"), float.fromhex("0x1.1d8656b0ee8cbp-101")),  # 1/16!
    (float.fromhex("0x1.952c77030ad4ap-49"), float.fromhex("0x1.ac981465ddc6cp-103")),  # 1/17!
    (float.fromhex("0x1.6827863b97d97p-53"), float.fromhex("0x1.eec01221a8b0bp-107")),  # 1/18!
    (float.fromhex("0x1.2f49b46814157p-57"), float.fromhex("0x1.2650f61dbdcb4p-112")),  # 1/19!
    (float.fromhex("0x1.e542ba4020225p-62"), float.fromhex("0x1.ea72b4afe3c2fp-120")),  # 1/20!
    (float.fromhex("0x1.71b8ef6dcf572p-66"), float.fromhex("-0x1.d043ae40c4647p-120")),  # 1/21!
    (float.fromhex("0x1.0ce396db7f853p-70"), float.fromhex("-0x1.aebcdbd20331cp-124")),  # 1/22!
    (float.fromhex("0x1.761b41316381ap-75"), float.fromhex("-0x1.3423c7d91404fp-130")),  # 1/23!
    (float.fromhex("0x1.f2cf01972f578p-80"), float.fromhex("-0x1.9ada5fcc1ab14p-135")),  # 1/24!
)
# --- glibc 2.28 __ieee754_exp_fma small-|x| branches (vote M10) ---
# Read from the oracle's /lib64/libm-2.28.so (RHEL 8): disassembly of
# __ieee754_exp_fma at 0x771e0 and its .rodata constants (0xac8b0, 0xad0a8,
# 0xc55c8-0xc55e0).  Branch limits on the high word of |x|.
_GLIBC_EXP_HI_TINY = 0x3E2FFFFF     # |x| < 2^-28: 1 + x
_GLIBC_EXP_HI_SMALL = 0x3ECFFFFF    # |x| < 2^-18: fma(fma(x, 1/2, 1), x, 1)
_GLIBC_EXP_HI_POLY = 0x3F862E41     # |x| <= ~0.01083: degree-6 polynomial
_GLIBC_EXP_C3 = float.fromhex("0x1.5555555555555p-3")   # 0xc55e0
_GLIBC_EXP_C4 = float.fromhex("0x1.5555555555555p-5")   # 0xc55d0
_GLIBC_EXP_C5 = float.fromhex("0x1.1111111111111p-7")   # 0xc55c8
_GLIBC_EXP_C6 = float.fromhex("0x1.6c16c16c16c17p-10")  # 0xc55d8
_SPLIT_MASK_26 = 0xFFFFFFFFF8000000  # keep sign, exponent and top 25+1 mantissa bits
# IEEE-754 binary64 layout.
_F64_EXPONENT_BIAS = 1023
_F64_MANTISSA_BITS = 52
_F64_HI_WORD_SHIFT = 32
_F64_HI_ABS_MASK = 0x7FFFFFFF
_EXP_UNDERFLOW_X = -746.0   # exp(x) rounds to 0 below this
_EXP_OVERFLOW_X = 709.79    # exp(x) overflows above this
_MIN_SUBNORMAL_EXPONENT = -1074
_MIN_NORMAL_EXPONENT = -1022

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


# --------------------------------------------------------------------------
# Correctly-rounded exp (double-double), vote M8
# --------------------------------------------------------------------------

def _two_sum(a, b):
    s = a + b
    bb = s - a
    return s, (a - (s - bb)) + (b - bb)


def _quick_two_sum(a, b):
    s = a + b
    return s, b - (s - a)


def _two_prod(a, b):
    p = a * b
    ah, al = _split_bits(a)
    bh, bl = _split_bits(b)
    return p, ((ah * bh - p) + ah * bl + al * bh) + al * bl


def _dd_mul(a, b):
    p, e = _two_prod(a[0], b[0])
    return _quick_two_sum(p, e + (a[0] * b[1] + a[1] * b[0]))


def _dd_add(a, b):
    s, e = _two_sum(a[0], b[0])
    return _quick_two_sum(s, e + (a[1] + b[1]))


def _pow2(m):
    """Exact 2.0**m for integer m in [-1022, 1023] (int64 array)."""
    return jax.lax.bitcast_convert_type((m + _F64_EXPONENT_BIAS) << _F64_MANTISSA_BITS, jnp.float64)


def _exp_cr_f64(x):
    xs = jnp.clip(x, _EXP_UNDERFLOW_X, _EXP_OVERFLOW_X)
    kf = jnp.rint(xs * _INV_LN2)
    r_hi = xs - kf * _LN2_HI
    r = _two_sum(r_hi, -(kf * _LN2_MID))
    r = (r[0], r[1] - kf * _LN2_LO)
    r = _quick_two_sum(*r)
    p = _INV_FACTORIAL_DD[-1]
    for c in reversed(_INV_FACTORIAL_DD[:-1]):
        p = _dd_add(_dd_mul(p, r), c)
    h, lo = p
    k = kf.astype(jnp.int64)
    k1 = k // 2
    normal = _pow2(k1) * h * _pow2(k - k1)
    # Subnormal result: round h + lo once onto the 2^-1074 grid.
    m = jnp.clip(k - _MIN_SUBNORMAL_EXPONENT, _MIN_NORMAL_EXPONENT, _F64_EXPONENT_BIAS)
    q = h * _pow2(m)
    ql = lo * _pow2(m)
    n = jnp.rint(q)
    tie = (q - jnp.floor(q)) == 0.5
    n = jnp.where(tie & (ql > 0.0), jnp.ceil(q), jnp.where(tie & (ql < 0.0), jnp.floor(q), n))
    sub = n * _pow2(jnp.full_like(k, _MIN_NORMAL_EXPONENT)) * _pow2(
        jnp.full_like(k, _MIN_SUBNORMAL_EXPONENT - _MIN_NORMAL_EXPONENT))
    tiny = _pow2(jnp.full_like(k, _MIN_NORMAL_EXPONENT))
    y = jnp.where(normal < tiny, sub, normal)
    y = jnp.where(x < _EXP_UNDERFLOW_X, 0.0, y)
    y = jnp.where(x > _EXP_OVERFLOW_X, jnp.inf, y)
    return jnp.where(jnp.isnan(x), x, y)


def _split_bits(a):
    """Exact split a = hi + lo with hi holding the top 26 significant bits."""
    bits = jax.lax.bitcast_convert_type(a, jnp.uint64) & jnp.uint64(_SPLIT_MASK_26)
    hi = jax.lax.bitcast_convert_type(bits, jnp.float64)
    return hi, a - hi


def _fma(a, b, c):
    """round(a*b + c) emulated in fp64 (exact product by Dekker with a bit-mask
    split, so XLA's own FMA contraction cannot break it); can differ from a
    hardware FMA only when a*b + c lies within ~2^-104 relative of a tie."""
    p = a * b
    ah, al = _split_bits(a)
    bh, bl = _split_bits(b)
    pe = ((ah * bh - p) + ah * bl + al * bh) + al * bl
    s, se = _two_sum(p, c)
    return s + (pe + se)


def _exp_glibc_small(x):
    """glibc 2.28 __ieee754_exp_fma for |x| <= 0.01083 (bitwise port)."""
    one = 1.0
    tiny = x + one
    small = _fma(_fma(x, 0.5, one), x, one)
    x2 = x * x
    p2 = _fma(x2, _GLIBC_EXP_C6, _fma(x, _GLIBC_EXP_C5, _GLIBC_EXP_C4))
    p0 = _fma(x, _GLIBC_EXP_C3, 0.5)
    poly = (_fma(p0, x2, (x2 * x2) * p2) + x) + one
    hi_word = (jax.lax.bitcast_convert_type(x, jnp.uint64) >> _F64_HI_WORD_SHIFT).astype(jnp.int64) & _F64_HI_ABS_MASK
    return jnp.where(hi_word <= _GLIBC_EXP_HI_TINY, tiny,
                     jnp.where(hi_word <= _GLIBC_EXP_HI_SMALL, small, poly)), hi_word


@jax.custom_jvp
def exp_libm(x):
    """fp64 exp reproducing the oracle's libm: glibc 2.28 __ieee754_exp_fma
    bitwise for |x| <= 0.01083 (where 1 - exp(-2 k od) amplifies 1 ulp by up
    to 1/|x|), correctly rounded elsewhere (glibc's 0.5x-ulp algorithm then
    differs by 1 ulp on ~0.1% of inputs, amplified by < 92).  XLA CPU flushes
    subnormal results to zero.  Other dtypes fall back to jnp.exp."""
    x = jnp.asarray(x)
    if x.dtype != jnp.float64:
        return jnp.exp(x)
    small, hi_word = _exp_glibc_small(x)
    return jnp.where(hi_word <= _GLIBC_EXP_HI_POLY, small, _exp_cr_f64(x))


@exp_libm.defjvp
def _exp_libm_jvp(primals, tangents):
    (x,), (t,) = primals, tangents
    y = exp_libm(x)
    return y, y * t


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
