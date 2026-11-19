"""Correctly-rounded / glibc-exact fp64 ``exp`` for the ecRad 49r1 port (votes M8,
M10; moved out of ``two_stream.py`` by vote V45 so every section can share it).

``exp_libm`` reproduces the oracle's libm (glibc 2.28 ``__ieee754_exp_fma``)
bitwise for |x| <= 0.01083 and is correctly rounded elsewhere; other dtypes
fall back to ``jnp.exp``.  Numerics helper only (no physics).
"""

import jax
import jax.numpy as jnp

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
