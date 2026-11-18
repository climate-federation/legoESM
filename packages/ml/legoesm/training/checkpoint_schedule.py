"""Adjoint-memory checkpoint scheduling for long differentiable rollouts.

Three schedules over a fixed-length loop, all yielding the SAME gradient:

- ``"none"``     plain ``lax.scan``; the backward pass stores every step's
                 residuals — O(N) device memory, fastest for short loops.
- ``"uniform"``  wrap the step in ``jax.checkpoint`` — recompute each step in
                 the backward pass; O(1) extra tape, O(N) recompute (the
                 current legoESM default).
- ``"binomial"`` Griewank-Walther / Treeverse optimal checkpointing via
                 ``equinox.internal.while_loop(kind="checkpointed")`` — O(K)
                 stored states and O(N log_K N) recompute, peak memory ~K
                 snapshots.  With ``K ~ sqrt(N)`` this is the square-root
                 schedule the DJ4Earth paper (Moses et al., JAMES 2026) uses
                 for Oceananigans/ShallowWaters long rollouts.

A ``storage`` axis is orthogonal to recompute: ``"host"`` offloads the
dominant (matmul) residuals to pinned host (CPU) DRAM via a JAX offload
checkpoint policy, trading PCIe bandwidth for device (GPU/TPU) HBM.

Everything here operates on an opaque ``body``/carry, so it is grid- and
component-agnostic (atmosphere, ocean, land, sea ice; any grid).
"""
from __future__ import annotations

import math
from collections.abc import Callable
from typing import TypeVar

import jax
import jax.numpy as jnp
from jax.ad_checkpoint import checkpoint_policies as _cp

Carry = TypeVar("Carry")

VALID_SCHEDULES = ("none", "uniform", "binomial")
VALID_STORAGE = ("recompute", "host")


def sqrt_checkpoints(n_steps: int) -> int:
    """Square-root checkpoint budget ``K = ceil(sqrt(N))`` (>= 1).

    The memory/recompute optimum of binomial checkpointing for a single-level
    schedule (Griewank & Walther 2000); the DJ4Earth default.
    """
    if n_steps <= 1:
        return 1
    return math.isqrt(n_steps - 1) + 1  # == ceil(sqrt(n_steps)) for n_steps >= 1


def offload_policy(storage: str):
    """Return a JAX remat policy for ``storage`` (or None for plain recompute)."""
    if storage == "recompute":
        return None
    if storage == "host":
        # ponytail: offloads matmul (dot) residuals device->pinned-host DRAM — the
        # dominant residual in spectral/dense ESM steps. Full-state offload would
        # need per-component jax.ad_checkpoint.checkpoint_name tags; add that when
        # a run is HBM-bound on non-dot residuals. Host memory kinds are a
        # GPU/TPU feature; on a plain CPU backend this lowers to recompute.
        return _cp.offload_dot_with_no_batch_dims("device", "pinned_host")
    raise ValueError(f"unknown storage {storage!r}; expected one of {VALID_STORAGE}")


def checkpointed_loop(
    body: Callable[[Carry], Carry],
    init: Carry,
    n_steps: int,
    *,
    schedule: str = "uniform",
    checkpoints: int | None = None,
    base: int = 16,
    storage: str = "recompute",
) -> Carry:
    """Run ``carry = body(carry)`` ``n_steps`` times with a chosen adjoint-memory schedule.

    ``body`` takes and returns the loop carry (which holds any accumulators the
    caller needs — there is no per-step stacked output, so the ``"binomial"``
    schedule never has to materialise an O(N) trajectory).  The returned final
    carry and its gradient are identical (to floating-point) across schedules.

    Parameters
    ----------
    body : callable
        ``carry -> carry`` single iteration.
    init : pytree
        Initial carry.
    n_steps : int
        Static number of iterations.
    schedule : {"none", "uniform", "binomial"}
        Adjoint-memory strategy (see module docstring).
    checkpoints : int, optional
        Binomial checkpoint budget K; defaults to ``sqrt_checkpoints(n_steps)``.
    base : int
        Treeverse fan-out for the binomial schedule (``equinox`` default 16).
    storage : {"recompute", "host"}
        ``"host"`` additionally offloads matmul residuals to pinned host DRAM;
        requires an active checkpoint (``"uniform"`` or ``"binomial"``) — pairing
        it with ``"none"`` raises (there is no checkpoint to offload).
    """
    # Validate dispatch BEFORE the zero-step early return so an invalid config
    # is rejected regardless of window length (consistency with the other entry
    # points and with positive n_steps).
    if schedule not in VALID_SCHEDULES:
        raise ValueError(
            f"unknown schedule {schedule!r}; expected one of {VALID_SCHEDULES}"
        )
    policy = offload_policy(storage)  # validates storage; None for recompute
    # Host offload modifies an ACTIVE checkpoint; "none" stores every step and
    # has nothing to offload, so reject the contradiction (mirrors
    # differentiable_rollout) rather than silently re-enabling remat.
    if schedule == "none" and policy is not None:
        raise ValueError(
            f"storage {storage!r} requires an active checkpoint schedule "
            "('uniform' or 'binomial'); 'none' has no checkpoint to offload."
        )
    if n_steps <= 0:
        return init

    if schedule in ("none", "uniform"):
        def scan_body(c, _):
            return body(c), None

        step = scan_body
        if schedule == "uniform":
            step = (
                jax.checkpoint(scan_body, policy=policy, prevent_cse=False)
                if policy is not None
                else jax.checkpoint(scan_body, prevent_cse=False)
            )
        final, _ = jax.lax.scan(step, init, xs=None, length=n_steps)
        return final

    if schedule == "binomial":
        import equinox.internal as eqxi

        K = checkpoints if checkpoints is not None else sqrt_checkpoints(n_steps)
        K = int(max(1, min(K, n_steps)))

        def cond(state):
            i, _c = state
            return i < n_steps

        def step(state):
            i, c = state
            return (i + 1, body(c))

        if policy is not None:
            step = jax.checkpoint(step, policy=policy, prevent_cse=False)

        _, out = eqxi.while_loop(
            cond,
            step,
            (jnp.asarray(0), init),
            kind="checkpointed",
            checkpoints=K,
            base=base,
            max_steps=n_steps,
        )
        return out

    raise ValueError(
        f"unknown schedule {schedule!r}; expected one of {VALID_SCHEDULES}"
    )
