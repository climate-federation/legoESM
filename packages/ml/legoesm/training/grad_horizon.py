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


def estimate_growth_rate(norms_by_horizon: dict[int, float]) -> float:
    """Least-squares slope of ``log(grad_norm)`` vs horizon = growth rate [1/step].

    A positive slope means the adjoint grows exponentially (chaotic regime); its
    reciprocal is an order-of-magnitude empirical Lyapunov time in time steps.
    Needs at least two finite, positive samples.
    """
    items = sorted((n, v) for n, v in norms_by_horizon.items()
                   if jnp.isfinite(v) and v > 0.0)
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
