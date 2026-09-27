"""Chaos / long-window adjoint guardrails.

Gradients through a chaotic dynamical system grow ~``exp(t / T_lyap)`` with the
rollout window: beyond the predictability horizon the adjoint is dominated by
exponentially amplified noise and is useless (often harmful) for optimisation.
DJ4Earth (Moses et al., JAMES 2026) names this explicitly as a pitfall of
differentiating long ESM trajectories.

These helpers (1) MEASURE adjoint-norm growth versus window length so a model's
usable training horizon is known, (2) estimate the empirical growth rate, and
(3) WARN (or raise) when a configured rollout exceeds a gradient-norm ceiling.
They are eager diagnostics — call them outside ``jax.jit`` — and are grid- and
component-agnostic (they only see a scalar-loss function of a control pytree).
"""
from __future__ import annotations

import logging
from collections.abc import Callable, Iterable

import jax
import jax.numpy as jnp

logger = logging.getLogger(__name__)


def global_grad_norm(grads) -> jax.Array:
    """L2 norm over a gradient pytree (matches ``optax.global_norm``; complex-safe)."""
    leaves = jax.tree.leaves(grads)
    if not leaves:
        return jnp.asarray(0.0)
    sq = sum(jnp.vdot(g, g).real for g in leaves)
    return jnp.sqrt(sq)


def grad_norm_vs_horizon(
    loss_for_horizon: Callable[[int, jax.Array], jax.Array],
    horizons: Iterable[int],
    x,
) -> dict[int, float]:
    """Adjoint norm of ``loss_for_horizon(n, x)`` w.r.t. ``x`` for each ``n`` in ``horizons``.

    ``loss_for_horizon(n_steps, x) -> scalar`` runs an ``n_steps`` rollout and
    returns a scalar loss.  Returns ``{n: ||d loss / d x||_2}`` — sweep it to
    characterise where the adjoint blows up (the usable training horizon).
    """
    out: dict[int, float] = {}
    for n in horizons:
        n = int(n)
        g = jax.grad(lambda z, _n=n: loss_for_horizon(_n, z))(x)
        out[n] = float(global_grad_norm(g))
    return out


def blowup_horizon(norms_by_horizon: dict[int, float]) -> int | None:
    """Smallest horizon whose adjoint norm is NON-FINITE (``inf`` or ``NaN``).

    ``None`` when every sample is a number.  This is the headline result of a
    horizon sweep: the step count at which the adjoint stops being a number.
    Read it BEFORE any growth rate — a growth rate fitted across a blow-up is
    meaningless.

    A norm of exactly zero is NOT a blow-up.  It is a perfectly finite answer
    that merely has no logarithm, so it is unusable for the growth fit and
    nothing more; calling it a blow-up would both invent a failure and discard
    every longer horizon behind it.  Those samples are reported separately by
    :func:`unfittable_horizons`.
    """
    bad = [int(n) for n, v in norms_by_horizon.items() if not jnp.isfinite(v)]
    return min(bad) if bad else None


def unfittable_horizons(norms_by_horizon: dict[int, float]) -> list[int]:
    """Finite horizons whose norm has no logarithm (``<= 0``), sorted.

    Distinct from :func:`blowup_horizon`: a zero adjoint means the control has
    no influence at that horizon, which is a finding of its own — usually a
    severed gradient path — and never an overflow.
    """
    return sorted(int(n) for n, v in norms_by_horizon.items()
                  if jnp.isfinite(v) and float(v) <= 0.0)


def estimate_growth_rate(norms_by_horizon: dict[int, float]) -> float:
    """Least-squares slope of ``log(grad_norm)`` vs horizon = growth rate [1/step].

    A positive slope means the adjoint grows exponentially (chaotic regime); its
    reciprocal is an order-of-magnitude empirical Lyapunov time in time steps.
    Needs at least two finite, positive samples.

    NON-FINITE SAMPLES ARE NOT SILENTLY DROPPED.  A blow-up is the very thing
    this sweep exists to find, and an earlier version filtered ``inf``/``NaN``
    out of the fit — so the sweep discarded its own finding and returned a
    reassuring slope measured on the healthy prefix alone, with nothing in the
    return value to say so.  The rule now:

    * the fit uses only horizons STRICTLY BELOW the first blow-up, because
      samples past a blow-up are not on the same exponential branch;
    * the dropped horizons are logged at WARNING naming the blow-up horizon;
    * fewer than two usable samples below the blow-up returns ``nan`` rather
      than a slope fitted through scattered survivors.

    Call :func:`blowup_horizon` alongside this to get the blow-up itself — the
    slope alone cannot express "and then it stopped being a number".
    """
    cut = blowup_horizon(norms_by_horizon)
    usable = {int(n): float(v) for n, v in norms_by_horizon.items()
              if jnp.isfinite(v) and float(v) > 0.0
              and (cut is None or int(n) < cut)}
    if cut is not None:
        dropped = sorted(int(n) for n in norms_by_horizon if int(n) >= cut)
        logger.warning(
            "adjoint blow-up at horizon %d: dropping horizons %s from the "
            "growth-rate fit; the slope below is the PRE-blow-up branch only, "
            "and the blow-up itself is the finding (see blowup_horizon).",
            cut, dropped)
    items = sorted(usable.items())
    if len(items) < 2:
        return float("nan")
    ns = jnp.asarray([n for n, _ in items], dtype=jnp.result_type(float))
    logv = jnp.log(jnp.asarray([v for _, v in items], dtype=jnp.result_type(float)))
    A = jnp.stack([ns, jnp.ones_like(ns)], axis=1)
    coef = jnp.linalg.lstsq(A, logv, rcond=None)[0]
    return float(coef[0])


def check_grad_horizon(
    grad_norm,
    n_steps: int,
    ceiling: float,
    *,
    raise_on_exceed: bool = False,
) -> bool:
    """Warn (or raise) when an adjoint norm exceeds ``ceiling`` at ``n_steps``.

    Returns ``True`` if within budget (and finite), ``False`` if exceeded.  A
    non-finite norm always counts as exceeded.
    """
    gn = float(grad_norm)
    if (not jnp.isfinite(gn)) or gn > ceiling:
        msg = (
            f"adjoint grad-norm {gn:.3e} at rollout horizon {n_steps} exceeds "
            f"ceiling {ceiling:.3e}: likely past the predictability horizon — "
            f"shorten the assimilation/training window, add gradient clipping, "
            f"or use a shadowing-based sensitivity."
        )
        if raise_on_exceed:
            raise RuntimeError(msg)
        logger.warning(msg)
        return False
    return True


def leaf_grad_report(grads) -> dict[str, tuple[float, bool]]:
    """Per-leaf ``(max|g|, is_finite)`` keyed by its pytree path.

    A global norm collapses to ``inf``/``NaN`` as soon as ONE leaf does, so it
    cannot say whether the adjoint grew until it overflowed or whether a single
    kernel handed back a NaN while everything else stayed small.  That is the
    distinction the horizon sweep has to make, so record the leaves.

    ``max|g|`` of a non-finite leaf is reported as ``inf`` when any element is
    infinite and ``nan`` when any element is NaN, so the two failure modes stay
    distinguishable in the log.
    """
    report: dict[str, tuple[float, bool]] = {}
    for path, leaf in jax.tree_util.tree_flatten_with_path(grads)[0]:
        key = jax.tree_util.keystr(path)
        x = jnp.asarray(leaf)
        finite = bool(jnp.all(jnp.isfinite(x)))
        if finite:
            val = float(jnp.max(jnp.abs(x))) if x.size else 0.0
        elif bool(jnp.any(jnp.isnan(x))):
            val = float("nan")
        else:
            val = float("inf")
        report[key] = (val, finite)
    return report


def diagnose_blowup(
    report: dict[str, tuple[float, bool]],
    reference_max: float | None = None,
    *,
    growth_factor: float = 1e6,
) -> str:
    """Label a leaf report with the HYPOTHESIS it is consistent with.

    This is a triage label, never a proof.  A cotangent is only observed at the
    ENDS of the chain, so no endpoint report can establish where along the
    backward sweep a value was born; a kernel that returns ``inf`` and an
    overflow that later becomes ``NaN`` through ``inf - inf`` are genuinely
    indistinguishable from the endpoints alone.  Localising the birth site needs
    per-step cotangents or a NaN trap, which is a separate job.

    ``reference_max`` is the largest finite per-leaf magnitude from a SHORT
    horizon in the same sweep — the only thing that makes "small" and "large"
    mean anything here.  Without it the amplitude evidence is unavailable and
    every failure is reported as ``"unclassified"`` rather than guessed at.

    ``"finite"``        nothing blew up.
    ``"growth"``        surviving finite leaves have already grown past
                        ``growth_factor`` times the reference, so the adjoint was
                        being amplified whatever the non-finite leaves mean.
    ``"kernel"``        a leaf is non-finite while every surviving finite leaf is
                        still near the reference.  Amplification cannot produce a
                        non-finite value while the rest of the adjoint is small,
                        so this points at a kernel returning ``NaN``/``inf`` in
                        its own right.
    ``"unclassified"``  a failure with no reference to judge amplitude against,
                        or one where NO finite leaf survives (nothing left to
                        compare, so a total overflow is not mislabelled).
    """
    bad = [v for v, ok in report.values() if not ok]
    if not bad:
        return "finite"
    if reference_max is None or not (reference_max > 0.0):
        return "unclassified"
    finite = [v for v, ok in report.values() if ok]
    if not finite:
        # Every leaf is non-finite: there is no surviving amplitude to judge,
        # and "kernel" would have been reached only because max() of nothing
        # defaulted to zero -- a total overflow must not be labelled a kernel.
        return "unclassified"
    return "growth" if max(finite) > growth_factor * reference_max else "kernel"


def grad_max_norm(report: dict[str, tuple[float, bool]]) -> float:
    """Overflow-free magnitude of a gradient: ``max |g|`` over every element.

    Deliberately the max-norm and not the L2 norm.  ``sum(x**2)`` overflows in
    fp32 once any element exceeds ~1.8e19, so an L2 norm manufactures the very
    ``inf`` a horizon sweep would then report as a blow-up — a measurement that
    fails at exactly the magnitudes it exists to measure.  A max-norm never
    overflows, and for deciding whether an adjoint is growing or has gone
    non-finite it carries the same information (the two norms agree to within a
    factor of sqrt(#elements), which is constant across horizons and therefore
    cancels out of a growth rate).

    Returns ``inf``/``nan`` only when a LEAF is genuinely non-finite.
    """
    bad = [v for v, ok in report.values() if not ok]
    if bad:
        return float("nan") if any(v != v for v in bad) else float("inf")
    return max((v for v, ok in report.values() if ok), default=0.0)
