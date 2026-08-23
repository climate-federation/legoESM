"""Smooth (differentiable) trigger and level-membership primitives.

Shared helpers used by every convection scheme that needs a smooth
replacement for a hard ``if``/``where`` branch in the reference paper.
The legoESM model is trained via :func:`jax.grad` and
``eqx.filter_value_and_grad`` — every trigger threshold therefore must
have a non-zero gradient on both sides of the threshold so that
:class:`legoesm.training.physics_params.TrainablePhysicsParams` and the
neural-physics blend can flow gradients through the convection step.

The conceptual split with :mod:`._plume`:

* This module is **sigmoid math**: scalar-to-scalar smooth
  approximations of step / max / positive-part / lowest-crossing-index
  primitives, plus a thin convenience wrapper for the CAPE > threshold
  trigger reused in four (now nine) convection backends.
* :mod:`._plume` is **column physics math**: LCL / LFC / LNB / CIN
  diagnostics, the entraining-detraining plume integrator,
  unsaturated-downdraft thermodynamics, the Emanuel buoyancy-sorting
  step, and the Gregory et al. 1997 convective momentum-transport
  closure.

The level-membership and lowest-crossing helpers consume profiles in
the canonical legoESM convention: shape ``(ncol, nlev)`` with the
**surface at the last index** ``[:, -1]`` and the model top at
``[:, 0]``.  Indices returned by :func:`smooth_lowest_crossing_index`
are in the same surface-last convention (``nlev - 1`` is the surface,
``0`` is the top).

References
----------
- Pattern for the soft-crossing × log-sum-exp softmin transcribed from
  ``legoesm.atmosphere.physics.turbulence.pbl_height.diagnose_pbl_height_interp``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


__all__ = (
    "smooth_step",
    "smooth_positive_part",
    "smooth_level_indicator",
    "smooth_lowest_crossing_index",
    "cape_trigger",
)


# ---------------------------------------------------------------------------
# Scalar / elementwise smooth primitives
# ---------------------------------------------------------------------------

# --- pspec autoblock
_FIRST_CROSS_SHARPNESS = 20.0

def smooth_step(x: jax.Array, sharpness: float = 1.0) -> jax.Array:
    """Differentiable approximation of the unit step function.

    ``smooth_step(x, s) = sigmoid(s * x) ∈ (0, 1)``.

    The result tends to ``Heaviside(x)`` as ``sharpness → ∞`` while
    remaining strictly differentiable for any finite ``sharpness``.
    Gradient at ``x = 0`` is ``sharpness / 4``.

    Parameters
    ----------
    x : jax.Array
        Argument (typically a difference ``value - threshold``).
    sharpness : float
        Inverse-width of the transition.  Larger values approach a
        sharp step; smaller values broaden the transition.

    Returns
    -------
    jax.Array
        Same shape and dtype family as ``x``, values in ``(0, 1)``.
    """
    return jax.nn.sigmoid(sharpness * x)


def smooth_positive_part(
    x: jax.Array,
    sharpness: float = 1.0,
) -> jax.Array:
    """Differentiable approximation of ``max(x, 0)``.

    Returns ``softplus(sharpness * x) / sharpness`` — a differentiable
    soft upper bound on ``max(x, 0)``.  The gradient at ``x = 0`` is
    ``0.5``, transitioning smoothly to ``1`` for ``x ≫ 0`` and ``0``
    for ``x ≪ 0``.

    Used wherever the reference paper writes ``(... )_+`` — most
    notably the CAPE-relaxation cloud-base mass-flux closure
    ``M_b ∝ (CAPE - CAPE_threshold)_+ / tau``.

    Parameters
    ----------
    x : jax.Array
        Argument.  Any shape.
    sharpness : float
        Larger values approach the kink at ``x = 0`` more sharply.

    Returns
    -------
    jax.Array
        Strictly positive everywhere; tends to ``max(x, 0)`` as
        ``sharpness → ∞``.
    """
    return jax.nn.softplus(sharpness * x) / sharpness


# ---------------------------------------------------------------------------
# Per-level membership and crossing diagnostics
# ---------------------------------------------------------------------------

def smooth_level_indicator(
    profile: jax.Array,
    threshold: jax.Array | float,
    sharpness: float,
    direction: str = "above",
) -> jax.Array:
    """Per-level smooth membership in ``[0, 1]``.

    Returns a per-level weight that smoothly transitions from 0 to 1
    where the profile crosses the threshold.  Useful for
    mass-weighted averages over a column subset (e.g. PBL-mean
    parcel for the Bechtold/IFS departure-CAPE closure).

    Parameters
    ----------
    profile : jax.Array, shape ``(..., nlev)``
        Profile to test against the threshold.  Surface at last index.
    threshold : jax.Array or float
        Threshold value.  Broadcastable against ``profile``.
    sharpness : float
        Sigmoid sharpness on the threshold transition.
    direction : str
        ``"above"`` returns ``sigmoid(sharpness * (profile - threshold))``
        (1 where profile is above threshold, 0 below).
        ``"below"`` is the complement: 1 where profile is below
        threshold (used to weight a PBL-depth average).

    Returns
    -------
    jax.Array
        Same shape as ``profile``; values in ``(0, 1)``.
    """
    if direction == "above":
        return jax.nn.sigmoid(sharpness * (profile - threshold))
    if direction == "below":
        return jax.nn.sigmoid(sharpness * (threshold - profile))
    raise ValueError(
        f"direction must be 'above' or 'below', got {direction!r}"
    )


def smooth_lowest_crossing_index(
    profile: jax.Array,
    threshold: jax.Array | float,
    sharpness: float,
) -> jax.Array:
    """Differentiable fractional index of the lowest upward crossing.

    Scans each column from the surface upward (last index → first
    index) and returns the fractional level index of the lowest
    altitude where ``profile`` crosses ``threshold`` from below to
    above.  The fractional component is a linear interpolation
    between adjacent levels; the choice of which crossing to return
    when several exist is made by a log-sum-exp softmin so that
    columns with multiple candidate crossings still produce a smooth
    (differentiable) answer.

    Used for LFC / LCL / LNB localization where the reference paper
    would normally take the first satisfying integer index.

    Parameters
    ----------
    profile : jax.Array, shape ``(ncol, nlev)``
        Per-level profile.  Surface at last index ``[:, -1]``.
    threshold : jax.Array or float
        Threshold value.  Scalar, ``(ncol,)``, or
        ``(ncol, nlev-1)``-broadcastable.
    sharpness : float
        Sigmoid sharpness on the soft crossing indicator AND the
        scale parameter for the softmin (larger = closer to a hard
        argmin).

    Returns
    -------
    jax.Array, shape ``(ncol,)``
        Fractional index in surface-last coordinates: a return value
        of ``ncol_index = nlev - 1`` means the surface, and ``0``
        means the model top.  When no crossing is detected the
        returned value falls back to ``nlev - 1`` (surface) — callers
        that need a "no-crossing" sentinel should use
        :func:`smooth_level_indicator` to compute a column-wide
        confidence first.
    """
    *_, nlev = profile.shape
    if nlev < 2:
        raise ValueError(
            "smooth_lowest_crossing_index requires nlev >= 2; got "
            f"profile.shape={profile.shape}"
        )

    # Reverse so that surface is first; matches the convention used in
    # turbulence/pbl_height.py.
    profile_rev = profile[..., ::-1]      # (ncol, nlev), surface-first

    profile_below = profile_rev[..., :-1]  # (ncol, nlev-1)
    profile_above = profile_rev[..., 1:]

    # Soft upward-crossing indicator: simultaneously below threshold
    # at the lower level and above at the upper level.
    cross_weight = (
        jax.nn.sigmoid(sharpness * (threshold - profile_below))
        * jax.nn.sigmoid(sharpness * (profile_above - threshold))
    )

    # Linear interpolation fraction within each pair (in [0, 1]).
    dprofile = profile_above - profile_below
    # Numerical floor avoids 0/0 when adjacent levels are equal.
    safe_d = jnp.where(jnp.abs(dprofile) > 1e-30, dprofile, 1e-30)
    frac = jnp.clip((threshold - profile_below) / safe_d, 0.0, 1.0)

    # Surface-first fractional indices of each candidate crossing.
    # Pair k connects level k (below) and level k+1 (above), so the
    # interpolated index is k + frac.
    k_pairs = jnp.arange(nlev - 1, dtype=profile.dtype)
    idx_surface_first = k_pairs + frac    # (ncol, nlev-1)

    # First-crossing probability at each pair, computed as the
    # sequential survival product
    #
    #     first_cross[k] = cross_weight[k] * Π_{j<k}(1 - cross_weight[j])
    #
    # This is the probability (under the soft-crossing model) that
    # pair ``k`` is the *first* upward crossing — it naturally
    # suppresses higher-altitude crossings even when they are
    # individually strong.  Unlike a global log-sum-exp softmin,
    # this formulation is robust to multiple actual crossings and to
    # vanishingly-small but nonzero cross_weight at non-crossing
    # levels (those contribute ``cross_weight ≈ 0`` and drop out
    # multiplicatively).
    not_yet_crossed = jnp.concatenate(
        [
            jnp.ones(profile.shape[:-1] + (1,), dtype=profile.dtype),
            jnp.cumprod(1.0 - cross_weight, axis=-1)[..., :-1],
        ],
        axis=-1,
    )
    first_cross_weight = cross_weight * not_yet_crossed

    # Weighted average of fractional indices.  ``maximum`` with a
    # tiny floor protects against ``0/0`` when ``total_first_cross``
    # is exactly zero; the no-crossing branch below replaces this
    # value with the surface fallback when the gate fires.
    # ``total_first_cross`` and ``idx_min_naive`` numerator share the
    # ``first_cross_weight`` weight on the level axis — fuse into one
    # stacked column reduction.
    _pair = jnp.sum(
        jnp.stack([first_cross_weight, first_cross_weight * idx_surface_first], axis=-1),
        axis=-2,
    )
    total_first_cross = _pair[..., 0]
    safe_total = jnp.maximum(total_first_cross, 1e-12)
    idx_min_naive = _pair[..., 1] / safe_total

    # No-crossing fallback: blend toward the surface index (which is
    # ``0`` in surface-first coordinates) when ``total_first_cross``
    # is small.  Uses a sharpness independent of the user-supplied
    # ``sharpness`` so the gate is consistent across calls — gate
    # sharpness ``20`` around midpoint ``0.5`` gives ``≈ 0%`` trust at
    # ``total = 0`` and ``≈ 100%`` trust at ``total = 1``.  At total
    # values typical of a single firm crossing (~0.8–1.0) the gate
    # fully selects the weighted-average answer.
    no_cross_blend = jax.nn.sigmoid(_FIRST_CROSS_SHARPNESS * (total_first_cross - 0.5))
    fallback = jnp.full_like(idx_min_naive, 0.0)  # surface-first 0 = surface-last nlev-1
    idx_min_surface_first = (
        no_cross_blend * idx_min_naive
        + (1.0 - no_cross_blend) * fallback
    )

    # Convert to surface-last (canonical legoESM) coordinates:
    #   surface-first index 0       -> surface-last index nlev - 1
    #   surface-first index nlev-1  -> surface-last index 0
    return float(nlev - 1) - idx_min_surface_first


# ---------------------------------------------------------------------------
# Convenience wrapper used at convection-trigger call sites
# ---------------------------------------------------------------------------

#: Factor by which the BACKWARD pass widens the CAPE-trigger sigmoid.
#:
#: WHY THIS EXISTS.  ``sigmoid`` saturates to EXACTLY 1.0 in float64 once its
#: argument exceeds ~36.7, so its derivative ``s*(1-s)`` is exactly 0 there
#: (measured: x=36 -> 2.220e-16, x=37 -> 0.0).  A deep-tropical column carries
#: CAPE of a few thousand J/kg against a threshold of tens, so the trigger sits
#: far inside that dead zone and ``d(output)/d(cape_threshold)`` is exactly
#: zero — the parameter is invisible to any gradient-based trainer.  That is
#: why the CAPE thresholds are declared ``tunable_tier 0`` (#1417).
#:
#: They are NOT physically inert, which is what makes the zero a defect rather
#: than a fact: in the 2026-08-16 SCM-RCE campaign, `dca` improved its
#: temperature-and-humidity score by 65 % (6.06 -> 2.13) by tuning its CAPE
#: threshold alone, found by a DERIVATIVE-FREE search precisely because the
#: gradient could not see it.
#:
#: The forward value is unchanged — the trigger still saturates, so the physics
#: and every existing answer are bit-identical — while the backward pass uses a
#: sigmoid widened by this factor, which is non-zero out to CAPE differences of
#: order ``sharpness**-1 * _TRIGGER_GRADIENT_WIDENING * 36``.  With the shipped
#: sharpnesses that covers the whole physical CAPE range.
_TRIGGER_GRADIENT_WIDENING = 1.0e3


def _straight_through_step(x: jax.Array, sharpness: float) -> jax.Array:
    """``smooth_step`` forward, a WIDER sigmoid's derivative backward.

    ``soft + stop_gradient(hard - soft)`` evaluates to ``hard`` exactly (the
    two ``soft`` terms cancel in the forward pass, bit for bit) while the only
    term carrying a derivative is ``soft``.  A straight-through estimator in
    the standard sense: the answer is the hard gate, the gradient is a usable
    surrogate rather than zero.

    This does not make the surrogate gradient EQUAL to the true derivative of
    the saturated sigmoid — that derivative is genuinely zero.  It supplies a
    descent direction with the correct SIGN, which is what a trainer needs and
    what a hard threshold cannot provide.
    """
    hard = smooth_step(x, sharpness)
    soft = smooth_step(x, sharpness / _TRIGGER_GRADIENT_WIDENING)
    return soft + jax.lax.stop_gradient(hard - soft)


def cape_trigger(
    cape: jax.Array,
    threshold: jax.Array | float,
    sharpness: float,
) -> jax.Array:
    """Smooth ``CAPE > threshold`` indicator for convection triggers.

    Equivalent to ``smooth_step(cape - threshold, sharpness)``.  The
    same soft-trigger formula is duplicated inline in four legacy
    schemes (``sbm``, ``dca``, ``kuo``, ``mass_flux``) and in every
    new scheme added by PRs 1 – 5; this convenience wrapper centralizes
    the semantics.

    Parameters
    ----------
    cape : jax.Array
        CAPE diagnostic [J/kg], any shape.
    threshold : jax.Array or float
        CAPE threshold below which convection is suppressed.
    sharpness : float
        Sigmoid sharpness in units of [1/(J/kg)].  Smaller values
        produce a wider transition; larger values approach a hard
        ``CAPE > threshold`` step.

    Returns
    -------
    jax.Array
        Same shape as ``cape``; values in ``(0, 1)``.
    """
    return _straight_through_step(cape - threshold, sharpness)
