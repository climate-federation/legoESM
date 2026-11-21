"""Smooth differentiable approximations to discontinuous operations.

In a differentiable model, all operations must have well-defined gradients.
These functions replace common discontinuities (Heaviside steps, max, min,
clamping) with smooth approximations parameterized by a sharpness factor.

Higher sharpness -> closer to the true discontinuous function but with
steeper gradients. Lower sharpness -> smoother gradients but less accurate.
A static ``sharpness = math.inf`` returns the discontinuous function exactly
(``x >= 0``, ``jnp.maximum``, ...): the hard limit of a RULE 5 smooth switch.
"""

import functools
import math
import numbers

import jax
import jax.numpy as jnp


def _hard(sharpness) -> bool:
    return isinstance(sharpness, numbers.Real) and sharpness == math.inf


def sigmoid_switch(x: jax.Array, sharpness: float = 100.0) -> jax.Array:
    """Smooth approximation to the Heaviside step function.

    H(x) ~ sigmoid(sharpness * x)

    Parameters
    ----------
    x : array
        Input values. Output ~ 0 where x < 0, ~ 1 where x > 0.
    sharpness : float
        Controls transition steepness. Default 100.

    Returns
    -------
    array : Values in (0, 1); exactly ``x >= 0`` at ``sharpness = inf``.
    """
    if _hard(sharpness):
        x = jnp.asarray(x)
        return (x >= 0).astype(x.dtype)
    return jax.nn.sigmoid(sharpness * x)


def smoothstep_cubic(t: jax.Array) -> jax.Array:
    """C1 step: 0 for t <= 0, 1 for t >= 1, t^2 (3 - 2t) between.

    Zero first derivative at both ends, so a ramp built on it has no kink.
    """
    t = jnp.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def smoothstep_quintic(t: jax.Array) -> jax.Array:
    """C2 step: 0 for t <= 0, 1 for t >= 1, 6t^5 - 15t^4 + 10t^3 between.

    Unlike :func:`sigmoid_switch` it is EXACTLY 0 below the threshold, with
    zero first and second derivatives there.
    """
    t = jnp.clip(t, 0.0, 1.0)
    return 6.0 * t**5 - 15.0 * t**4 + 10.0 * t**3


def smooth_max(a: jax.Array, b: jax.Array, sharpness: float = 100.0) -> jax.Array:
    """Smooth differentiable approximation to max(a, b).

    Uses the LogSumExp trick: max(a, b) ~ log(exp(s*a) + exp(s*b)) / s

    Parameters
    ----------
    a, b : arrays
        Input values (will be broadcast to the same shape).
    sharpness : float
        Controls approximation quality. Default 100.

    Returns
    -------
    array : Smooth approximation to element-wise max(a, b).
    """
    if _hard(sharpness):
        return jnp.maximum(a, b)
    a = jnp.asarray(a)
    b = jnp.asarray(b)
    # Broadcast to same shape before stacking
    a_s, b_s = jnp.broadcast_arrays(a * sharpness, b * sharpness)
    stacked = jnp.stack([a_s, b_s], axis=0)
    return jax.nn.logsumexp(stacked, axis=0) / sharpness


def smooth_min(a: jax.Array, b: jax.Array, sharpness: float = 100.0) -> jax.Array:
    """Smooth differentiable approximation to min(a, b).

    min(a, b) = -max(-a, -b)
    """
    if _hard(sharpness):
        return jnp.minimum(a, b)
    return -smooth_max(-a, -b, sharpness)


def pnorm_smooth_min(a: jax.Array, b: jax.Array, p: float, eps: float) -> jax.Array:
    """p-norm smooth min of two NON-NEGATIVE arrays: ``(a^-p + b^-p)^(-1/p)``.

    Written overflow-free as ``lo*(1 + (lo/hi)^p)^(-1/p)`` (``lo``/``hi`` =
    elementwise min/max); the ratio is 0 where ``hi <= eps`` (where-before-
    divide, AD-safe), so there the result is the exact ``min`` and it jumps
    (from ``eps`` to ``2^(-1/p)*eps`` where ``a = b``) as ``hi`` crosses
    ``eps``: callers with ``eps > 0`` zero inputs ``<= eps`` first (the GWD
    stress caps zero stresses ``<= taumin = eps``).  Above ``eps``: scale-free (unlike the
    additive :func:`smooth_min`), C-infinity, never above ``min(a, b)``,
    equal to ``2^(-1/p)*a`` where ``a = b``; exact ``min`` as ``p -> inf``.
    Used for the GWD stress cap ``min(taudmp, tausat)`` (Lindzen, McFarlane,
    E3SM-CAM), the GWD tendency limiters and the launch Froude cap; callers
    with ``eps = 0`` pass non-negative inputs (``hi = 0`` then means both are
    0 and the exact 0 is returned with finite gradients).  ``lo``/``hi`` are
    picked with one ``where`` (not ``minimum``/``maximum``, whose AD splits
    the tangent at a tie and zeroes the curvature there); the formula is
    symmetric, so either pick is the same smooth function.
    """
    first = a <= b
    return _pnorm_lo_hi(jnp.where(first, a, b), jnp.where(first, b, a), p, eps)


@jax.custom_jvp
def _div(x, y):
    return x / y


@_div.defjvp
def _div_jvp(primals, tangents):
    # d(x/y) = (dx - (x/y) dy) / y through _div again, so forward-mode AD never
    # forms y^-2 (it underflows to 0 for tiny y in fp32: 0*inf = NaN).  Measured
    # in fp32 at y ~ 1e-20: first and second derivatives (fwd-fwd, fwd-rev,
    # rev-rev) finite; third-order reverse is NaN (transposition divides plainly).
    x, y = primals
    dx, dy = tangents
    out = _div(x, y)
    return out, _div(dx - out * dy, y)


def _pnorm_ratio(lo, hi, eps):
    ok = hi > eps
    return ok, jnp.where(ok, _div(lo, jnp.where(ok, hi, jnp.ones_like(hi))), jnp.zeros_like(lo))


@functools.partial(jax.custom_jvp, nondiff_argnums=(2, 3))
def _pnorm_lo_hi(lo, hi, p, eps):
    _, ratio = _pnorm_ratio(lo, hi, eps)
    return lo * (1.0 + ratio ** p) ** (-1.0 / p)


@_pnorm_lo_hi.defjvp
def _pnorm_lo_hi_jvp(p, eps, primals, tangents):
    # Analytic, bounded partials: d/dlo = (1+r^p)^(-1/p-1), d/dhi =
    # r^(p+1) (1+r^p)^(-1/p-1) with r = lo/hi <= 1.  Differentiating lo/hi
    # instead overflows hi^-2 in fp32 for tiny hi (NaN gradient at lo = 0).
    lo, hi = primals
    dlo, dhi = tangents
    ok, ratio = _pnorm_ratio(lo, hi, eps)
    rp = ratio ** p
    w = (1.0 + rp) ** (-1.0 / p - 1.0)
    out = lo * (1.0 + rp) ** (-1.0 / p)
    return out, w * dlo + jnp.where(ok, ratio ** (p + 1.0) * w, 0.0) * dhi


def smooth_clamp(
    x: jax.Array, lo: float, hi: float, sharpness: float = 100.0
) -> jax.Array:
    """Smooth differentiable clamp to [lo, hi].

    Parameters
    ----------
    x : array
        Input values.
    lo, hi : float
        Lower and upper bounds.
    sharpness : float
        Controls transition steepness.

    Returns
    -------
    array : Values smoothly clamped to approximately [lo, hi].
    """
    return smooth_min(smooth_max(x, jnp.asarray(lo), sharpness), jnp.asarray(hi), sharpness)


def smooth_relu(x: jax.Array, sharpness: float = 100.0) -> jax.Array:
    """Smooth approximation to ReLU: max(x, 0).

    Uses softplus: log(1 + exp(s*x)) / s
    """
    if _hard(sharpness):
        return jnp.maximum(x, 0.0)
    return jax.nn.softplus(x * sharpness) / sharpness


def positive_part_c1(x: jax.Array, sharpness: float) -> jax.Array:
    """C1 positive part, EXACTLY 0 for ``x <= 0`` and exactly ``x`` above ``1/s``.

    ``0`` for ``x <= 0``; ``t**2 (2 - t)/s`` with ``t = s*x`` on
    ``0 < x < 1/s`` (value and slope match at both joins); ``x`` above.
    Monotone and never above ``max(x, 0)`` (the shortfall is at most
    ``4/(27 s)``, at ``x = 1/(3s)``).  Unlike :func:`smooth_relu` (softplus,
    positive everywhere, offset everywhere) it creates nothing from a
    non-positive input and is exact for ``x >= 1/s``; exactly ``max(x, 0)``
    at ``sharpness = inf``, static or traced (a traced ``inf`` selects the hard
    form; the smooth branch then runs at sharpness 1 so it holds no ``inf * 0``).
    """
    if _hard(sharpness):
        return jnp.maximum(x, 0.0)
    hard = jnp.isinf(sharpness)
    s = jnp.where(hard, 1.0, sharpness)
    t = jnp.clip(x * s, 0.0, 1.0)
    smooth = jnp.where(t >= 1.0, x, t * t * (2.0 - t) / s)
    return jnp.where(hard, jnp.maximum(x, 0.0), smooth)


def smooth_min_c1(a: jax.Array, b: jax.Array, width: float) -> jax.Array:
    """C1 min(a, b), EXACT outside ``|a - b| >= width/2``; ``width = 0`` is ``jnp.minimum``.

    Inside the band ``y = a - b + width/2`` in ``(0, width)`` it is
    ``a - y**2 / (2 width)``: monotone in ``a`` (slope ``1 - y/width``), never
    above ``min(a, b)`` and at most ``width/8`` below it (at ``a = b``).
    Unlike ``x - positive_part_c1(x - b)`` it does not overshoot ``b``.
    """
    if width == 0:
        return jnp.minimum(a, b)
    y = a - b + 0.5 * width
    inside = a - y * y / (2.0 * width)
    return jnp.where(y <= 0.0, a, jnp.where(y >= width, b + 0.0 * a, inside))


def validate_smoothing_width(name: str, w) -> None:
    """One legal range for a RULE 5 width that selects a static branch: 0 (the
    reference's hard switch) or in [sqrt(t), 1/sqrt(t)], t = finfo(float32).tiny
    (below it the ramp's slope overflows float32 and the gradient turns NaN).
    Booleans and non-numbers are refused (a YAML ``false`` would read as 0)."""
    lo = math.sqrt(float(jnp.finfo(jnp.float32).tiny))
    if (isinstance(w, bool) or not isinstance(w, (int, float))
            or not (w == 0 or lo <= w <= 1.0 / lo)):
        raise ValueError(
            f"{name} must be 0 (the reference's hard switch) or in "
            f"[{lo:.3e}, {1.0 / lo:.3e}]; got {w!r}.")


def abs_tanh(x: jax.Array, width: float) -> jax.Array:
    """``|x|`` as ``x*tanh(x/width)`` (RULE 5): C-infinity, 0 at 0, never above
    ``|x|``, within ``0.28*width`` of it.  ``width = 0`` is ``jnp.abs``."""
    return jnp.abs(x) if width == 0 else x * jnp.tanh(x / width)


def sign_tanh(x: jax.Array, width: float) -> jax.Array:
    """``sign(x)`` as ``tanh(x/width)`` (RULE 5).  ``width = 0`` is ``jnp.sign``."""
    return jnp.sign(x) if width == 0 else jnp.tanh(x / width)


def zero_at_or_below(x: jax.Array, floor: float, width: float) -> jax.Array:
    """``x`` zeroed where ``x <= floor`` (RULE 5): ``x*smoothstep_quintic((x -
    floor)/width)``, exactly 0 at and below ``floor`` and exactly ``x`` from
    ``floor + width``, C2.  ``width = 0`` is the hard ``where(x <= floor, 0, x)``."""
    if width == 0:
        return jnp.where(x <= floor, 0.0, x)
    return x * smoothstep_quintic((x - floor) / width)


def smooth_max_c1(a: jax.Array, b: jax.Array, width: float) -> jax.Array:
    """C1 max(a, b): ``-smooth_min_c1(-a, -b, width)``; ``width = 0`` is ``jnp.maximum``."""
    if width == 0:
        return jnp.maximum(a, b)
    return -smooth_min_c1(-a, -b, width)


def max_c1(a: jax.Array, b: jax.Array, sharpness: float) -> jax.Array:
    """C1 ``max(a, b)`` that is EXACTLY ``a`` for ``a - b >= 1/s`` and ``b`` for ``a <= b``.

    ``b + positive_part_c1(a - b, s)`` inside the band ``0 < a - b < 1/s``;
    outside it the hard operand is returned unchanged (bitwise), so a floor
    or cap that is inactive by more than ``1/s`` costs nothing.  Never above
    ``max(a, b)``.  Exactly ``jnp.maximum`` at ``sharpness = inf``.
    """
    if _hard(sharpness):
        return jnp.maximum(a, b)
    d = a - b
    return jnp.where(d * sharpness >= 1.0, a, b + positive_part_c1(d, sharpness))


def min_c1(a: jax.Array, b: jax.Array, sharpness: float) -> jax.Array:
    """C1 ``min(a, b)``, exact outside the band ``0 < b - a < 1/s``: ``-max_c1(-a, -b, s)``."""
    if _hard(sharpness):
        return jnp.minimum(a, b)
    return -max_c1(-a, -b, sharpness)


@jax.custom_jvp
def straight_through(hard: jax.Array, soft: jax.Array) -> jax.Array:
    """Forward value ``hard`` EXACTLY, derivative of ``soft`` (straight-through estimator).

    A ``custom_jvp``: the primal is ``hard`` itself, so ``soft`` is dead code in
    the forward pass and the compiled forward is the hard one bit for bit,
    under jit too.  (The arithmetic form ``hard - (sg(soft) - soft)`` is exact
    eagerly, but under jit on CPU it was measured to differ from ``hard`` at the
    smallest subnormal snow depth and in the sign of zero downstream, and it
    propagates a NaN ``soft`` into the forward value.)  The tangent is that of
    ``soft``; ``hard`` carries none.
    """
    return hard


@straight_through.defjvp
def _straight_through_jvp(primals, tangents):
    return primals[0], tangents[1]


def sqrt_c1(x: jax.Array, width: float) -> jax.Array:
    """C1 ``sqrt(max(x, 0))``: exactly 0 for ``x <= 0``, exactly ``sqrt(x)`` for
    ``x >= width``, and the monotone cubic Hermite bridge
    ``sqrt(width) * (2.5 t**2 - 1.5 t**3)``, ``t = x/width``, between (value and
    slope match at both joins).  ``width = 0`` is the hard ``sqrt(max(x, 0))``
    with a zero gradient at ``x <= 0``.  The square root is evaluated only on
    ``x >= width`` (masked argument), so its gradient stays finite."""
    if width == 0:
        ok = x > 0.0
        return jnp.where(ok, jnp.sqrt(jnp.where(ok, x, 1.0)), 0.0)
    t = jnp.clip(x / width, 0.0, 1.0)
    bridge = math.sqrt(width) * t * t * (2.5 - 1.5 * t)
    above = x >= width
    return jnp.where(above, jnp.sqrt(jnp.where(above, x, width)), bridge)


def step_rel(x: jax.Array, scale: jax.Array, width: float) -> jax.Array:
    """C1 step ``x > 0`` (RULE 5), one-sided: exactly 0 for ``x <= 0`` (ties go
    to the hard form's side), a ramp over ``0 < x < width*scale``, exactly 1
    above.  ``scale > 0`` is the magnitude ``x`` is measured against (traced is
    fine); ``width`` is dimensionless and static, ``width = 0`` is the hard
    ``x > 0``."""
    if width == 0:
        return (x > 0).astype(jnp.result_type(x, float))
    return smoothstep_cubic(x / (width * scale))


def max_rel(a: jax.Array, b: jax.Array, width: float) -> jax.Array:
    """C1 ``max(a, b)`` with the band ``|a - b| < width*(|a| + |b|)/2`` measured
    relative to the operands (dimensionless static ``width``): exact outside it,
    :func:`smooth_max_c1` on the normalised operands inside.  ``width = 0`` is
    ``jnp.maximum``.  Against an operand of 0 the band is empty (the kink stays):
    use :func:`positive_part_c1` on a scaled argument there."""
    if width == 0:
        return jnp.maximum(a, b)
    a, b = jnp.broadcast_arrays(jnp.asarray(a), jnp.asarray(b))
    s = jnp.abs(a) + jnp.abs(b)
    ok = s > 0
    s1 = jnp.where(ok, s, 1.0)           # double where: no 0/0 in the value or its gradient
    return jnp.where(ok, smooth_max_c1(a / s1, b / s1, width) * s1, jnp.maximum(a, b))


def min_rel(a: jax.Array, b: jax.Array, width: float) -> jax.Array:
    """C1 ``min(a, b)``, relative band as :func:`max_rel`; ``width = 0`` is ``jnp.minimum``."""
    if width == 0:
        return jnp.minimum(a, b)
    return -max_rel(-a, -b, width)


def smooth_abs(x: jax.Array, epsilon: float = 1e-7) -> jax.Array:
    """Smooth approximation to |x|.

    Uses sqrt(x^2 + epsilon) which is differentiable everywhere.
    """
    return jnp.sqrt(x * x + epsilon)


def smooth_round(x: jax.Array, width: float) -> jax.Array:
    """Nearest-integer rounding as a C2 staircase (RULE 5): equal to ``n``
    except where ``frac(x)`` is within ``width/2`` of 1/2, where it rises to
    ``n + 1`` by ``smoothstep_quintic``.  ``width`` in (0, 1].  ``width = 0`` is
    Fortran ``NINT`` (half away from zero), exactly; as ``width -> 0`` the
    staircase tends to ``floor(x + 1/2)``, which differs from ``NINT`` only at
    exact negative half-integers."""
    if width == 0:
        return jnp.sign(x) * jnp.floor(jnp.abs(x) + 0.5)
    n = jnp.floor(x)
    return n + smoothstep_quintic((x - n - 0.5) / width + 0.5)


def smooth_trunc(x: jax.Array, width: float) -> jax.Array:
    """Truncation toward zero as a C2 staircase (RULE 5): ``sign(x)*m`` with
    ``m = floor(|x|)`` except over the last ``width`` of each unit, where it
    rises to ``m + 1`` by ``smoothstep_quintic`` (so exactly 0, with zero
    derivatives, for ``|x| < 1 - width``).  ``width`` in (0, 1].  ``width = 0`` is ``jnp.trunc``, exactly."""
    if width == 0:
        return jnp.trunc(x)
    a = jnp.abs(x)
    m = jnp.floor(a)
    return jnp.sign(x) * (m + smoothstep_quintic((a - m - 1.0) / width + 1.0))


def blend_integer_index(r: jax.Array, take):
    """``take(n)`` at an integer-valued ``r``; between integers the linear
    blend ``(1-t)*take(floor(r)) + t*take(floor(r)+1)``, ``t = r - floor(r)``,
    so a smooth staircase ``r`` (:func:`smooth_round`) gives a smooth lookup.
    ``take`` receives an int32 index array of ``r``'s shape and returns an
    array whose leading dims are ``r``'s."""
    lo = jnp.floor(r)
    t = r - lo
    lo = lo.astype(jnp.int32)
    a, b = take(lo), take(lo + 1)
    t = t.reshape(t.shape + (1,) * (a.ndim - t.ndim))
    return (1.0 - t) * a + t * b
