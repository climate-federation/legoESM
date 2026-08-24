"""Counter-gradient diagnostic — the Q1 *structural* ceiling on local closures.

LES_SUITE.md Q1 asks where a *local* down-gradient closure must be abandoned for a
*nonlocal* one. Part (a) of the answer is **structural** and tuning-independent: a
down-gradient (K-theory) closure models the turbulent flux as

    <w'θ'> = -K d<θ>/dz ,   K >= 0

so it can ONLY produce a flux **opposite** in sign to the mean gradient. Wherever
the LES total turbulent flux is instead the *same* sign as the mean gradient — a
**counter-gradient** (up-gradient) transport region — *no* non-negative eddy
diffusivity can reproduce it, regardless of tuning. The height/regime at which a
finite counter-gradient layer first appears is therefore a hard ceiling on any
local closure, independent of calibration (§7 Q1a).

This module computes that diagnostic from an :class:`~...bridge.LESTruth` snapshot
(or an explicit flux+profile pair). It is pure array math — no LES integration, no
scoring. Sign convention (matches ``bridge`` / ``large_scale_forcing``): heights
increase with index (surface-first), flux positive upward, gradient positive when
θ increases with height.

Definition used here:
    counter-gradient at level k  ⇔  flux[k] * dθ/dz[k] > gate
where ``gate`` masks out numerically-zero product noise (both a near-zero flux and
a near-zero gradient region are *not* counter-gradient — the classic CBL
mixed-layer signature is a small **positive** (weakly stable) gradient carrying an
**upward** (positive) flux, i.e. a robustly positive product). A region is only
reported when it persists over ``min_layer_levels``
contiguous levels, so a single-cell sign blip near the surface or inversion does
not masquerade as the structural ceiling.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from .bridge import LESTruth

Array = jnp.ndarray

# Product threshold [K^2 m / (s m)] = [K^2 / s]. A flux·gradient product below this
# in magnitude is treated as "no definite transport direction" (mixed-layer core or
# quiescent air), not counter-gradient. Small relative to a convective BL's
# ~0.06 K m/s flux against a ~1e-3 K/m gradient (~6e-5) yet well above round-off.
_DEFAULT_PRODUCT_GATE = 1.0e-7
# Minimum contiguous counter-gradient levels to call a *layer* (reject single-cell
# blips at the surface layer / entrainment interface).
_DEFAULT_MIN_LAYER_LEVELS = 2


def centered_dtheta_dz(theta: Array, heights_m: Array) -> Array:
    """Centered ``d<θ>/dz`` on a surface-first (increasing-height) grid.

    Second-order centered in the interior; one-sided at the two endpoints. Height
    spacing may be non-uniform (LES grids stretch toward the inversion), so the
    non-uniform three-point formula is used rather than assuming constant dz.
    Positive when θ increases with height (statically stable).
    """
    theta = jnp.asarray(theta)
    z = jnp.asarray(heights_m, dtype=theta.dtype)
    if theta.shape != z.shape or theta.ndim != 1:
        raise ValueError("theta and heights_m must be matching 1-D arrays")
    if theta.shape[0] < 2:
        raise ValueError("need >=2 levels for a vertical gradient")
    return jnp.gradient(theta, z)


@dataclass(frozen=True)
class CounterGradientResult:
    """Per-level counter-gradient diagnostic for one LES snapshot.

    heights_m : (nz,) the levels the diagnostic is evaluated on.
    dtheta_dz : (nz,) mean potential-temperature gradient [K/m].
    flux : (nz,) total turbulent heat flux <w'θ'> [K m/s].
    product : (nz,) flux * dθ/dz [K^2/s]; > gate ⇒ counter-gradient.
    is_counter_gradient : (nz,) bool per-level mask (product > gate).
    has_counter_gradient_layer : bool — a contiguous run of length
        >= min_layer_levels exists (the structural-ceiling verdict).
    layer_base_m, layer_top_m : height bounds of the *first* (lowest) qualifying
        contiguous layer, or ``None`` when there is none.
    counter_gradient_fraction : fraction of levels flagged (diagnostic scalar).
    """

    heights_m: Array
    dtheta_dz: Array
    flux: Array
    product: Array
    is_counter_gradient: Array
    has_counter_gradient_layer: bool
    layer_base_m: float | None
    layer_top_m: float | None
    counter_gradient_fraction: float


def _first_contiguous_run(mask_bool: list[bool], min_len: int) -> tuple[int, int] | None:
    """First (lowest-index) contiguous run of ``True`` with length >= min_len.

    Returns ``(start, end_inclusive)`` index bounds or ``None``. Pure Python on a
    host-materialized bool list — this runs on a diagnostic snapshot, not in a
    traced hot loop.
    """
    start = None
    for i, v in enumerate(mask_bool):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= min_len:
                return (start, i - 1)
            start = None
    if start is not None and len(mask_bool) - start >= min_len:
        return (start, len(mask_bool) - 1)
    return None


def counter_gradient_diagnostic(
    flux: Array,
    theta: Array,
    heights_m: Array,
    *,
    product_gate: float = _DEFAULT_PRODUCT_GATE,
    min_layer_levels: int = _DEFAULT_MIN_LAYER_LEVELS,
) -> CounterGradientResult:
    """Counter-gradient diagnostic from an explicit flux + θ profile.

    ``flux``, ``theta``, ``heights_m`` are matching ``(nz,)`` arrays on a
    surface-first grid. A level is counter-gradient when ``flux*dθ/dz >
    product_gate``; the structural-ceiling verdict requires a contiguous run of at
    least ``min_layer_levels`` such levels.
    """
    if min_layer_levels < 1:
        raise ValueError("min_layer_levels must be >= 1")
    flux = jnp.asarray(flux)
    theta = jnp.asarray(theta, dtype=flux.dtype)
    heights = jnp.asarray(heights_m, dtype=flux.dtype)
    if not (flux.shape == theta.shape == heights.shape) or flux.ndim != 1:
        raise ValueError("flux, theta, heights_m must be matching 1-D arrays")
    dtdz = centered_dtheta_dz(theta, heights)
    product = flux * dtdz
    is_cg = product > product_gate
    mask_list = [bool(x) for x in is_cg.tolist()]
    run = _first_contiguous_run(mask_list, min_layer_levels)
    if run is None:
        base_m = top_m = None
        has_layer = False
    else:
        i0, i1 = run
        base_m = float(heights[i0])
        top_m = float(heights[i1])
        has_layer = True
    frac = float(jnp.mean(is_cg.astype(flux.dtype)))
    return CounterGradientResult(
        heights_m=heights,
        dtheta_dz=dtdz,
        flux=flux,
        product=product,
        is_counter_gradient=is_cg,
        has_counter_gradient_layer=has_layer,
        layer_base_m=base_m,
        layer_top_m=top_m,
        counter_gradient_fraction=frac,
    )


def diagnose_truth(
    truth: LESTruth,
    *,
    product_gate: float = _DEFAULT_PRODUCT_GATE,
    min_layer_levels: int = _DEFAULT_MIN_LAYER_LEVELS,
) -> CounterGradientResult:
    """Counter-gradient diagnostic for a single-time :class:`LESTruth` snapshot.

    Consumes the diagnostic-mode truth (``bridge.diagnostic_truth``): a ``(nz,)``
    θ profile and the total (resolved+SGS) heat flux ``wtheta``. Raises if handed a
    multi-time (prognostic) truth — the structural ceiling is a per-snapshot notion.
    """
    theta = jnp.asarray(truth.theta)
    if theta.ndim != 1:
        raise ValueError(
            "diagnose_truth expects a single-time snapshot (1-D theta); "
            "pass bridge.diagnostic_truth(...) output, not prognostic_truth"
        )
    return counter_gradient_diagnostic(
        truth.wtheta,
        theta,
        truth.heights_m,
        product_gate=product_gate,
        min_layer_levels=min_layer_levels,
    )
