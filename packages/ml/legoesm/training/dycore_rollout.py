"""Differentiable multi-day rollout through the compiled dycore.

Wraps ``build_segment_fn`` into a higher-level loop that chains
segments together with gradient checkpointing, supporting 1-day to
14-day rollouts for training via backpropagation.

The key function ``differentiable_rollout`` runs N segments through
``jax.lax.scan``, optionally collecting intermediate states for
multi-day loss computation.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from legoesm.driver.compiled_segments import (
    SegmentCarry,
    SegmentForcing,
)
from legoesm.training.checkpoint_schedule import VALID_STORAGE, offload_policy


class RolloutConfig(NamedTuple):
    """Configuration for a differentiable rollout."""
    n_days: int = 1
    dt: float = 600.0             # timestep [s]
    segment_steps: int = 144      # steps per segment (= 1 day at dt=600)
    gradient_checkpoint: bool = True
    # Adjoint-memory schedule for the per-segment checkpoint.  Only "none" and
    # "uniform" are valid here because this rollout returns the full per-segment
    # trajectory (saved_states); for "binomial" endpoint-loss rollouts use
    # legoesm.training.checkpoint_schedule.checkpointed_loop directly.
    checkpoint_schedule: str = "uniform"
    storage: str = "recompute"   # "recompute" | "host" (offload matmul residuals)


class RolloutOutput(NamedTuple):
    """Output of a differentiable rollout."""
    final_carry: SegmentCarry
    saved_states: jax.Array   # (n_saves, ...) stacked prognostic fields
    saved_days: jax.Array     # (n_saves,) day indices of saved states


def differentiable_rollout(
    initial_carry: SegmentCarry,
    forcing: SegmentForcing,
    run_segment_fn,
    config: RolloutConfig,
) -> RolloutOutput:
    """Run a multi-day differentiable rollout through the compiled dycore.

    Chains ``config.n_days`` segments together using ``jax.lax.scan``.
    Each segment runs ``config.segment_steps`` time steps.  EVERY segment's
    state is stacked into ``saved_states`` for multi-day loss computation
    (one save per day; there is no save stride).

    Gradient checkpointing is applied per segment so that the memory
    cost is O(segment_steps) rather than O(total_steps).

    Parameters
    ----------
    initial_carry : SegmentCarry
        Initial atmospheric state packed as a SegmentCarry.
    forcing : SegmentForcing
        External forcing (SST, SIC, solar, ozone) — held constant
        across the rollout.  For multi-day training this is an
        acceptable approximation at 2.5-degree resolution.
    run_segment_fn : callable
        The compiled segment function from ``build_segment_fn``.
        Signature: ``run_segment(carry, n_steps, forcing) -> carry``.
    config : RolloutConfig
        Rollout configuration.

    Returns
    -------
    RolloutOutput
        Final carry, saved intermediate states, and save day indices.
    """
    n_segments = config.n_days
    seg_steps = config.segment_steps

    def _one_segment(carry, seg_idx):
        """Execute one segment and optionally save state."""
        new_carry = run_segment_fn(carry, seg_steps, forcing)
        # Extract T for saving (representative prognostic field)
        # Save a compact representation: global-mean T profile
        mean_T = jnp.mean(new_carry.T, axis=tuple(range(new_carry.T.ndim - 1)))
        return new_carry, (new_carry, mean_T)

    # Validate the CONFIGURED schedule before deriving the effective one, so a
    # typo or 'binomial' is rejected even when gradient_checkpoint is False (which
    # would otherwise coerce it to "none" and hide the bad value).
    if config.checkpoint_schedule not in ("none", "uniform"):
        raise ValueError(
            f"RolloutConfig.checkpoint_schedule={config.checkpoint_schedule!r} is not "
            "supported by differentiable_rollout (it returns the full per-segment "
            "trajectory); use legoesm.training.checkpoint_schedule.checkpointed_loop "
            "for 'binomial' endpoint-loss rollouts."
        )
    # Effective schedule is "none" when gradient_checkpoint is off, else configured.
    schedule = config.checkpoint_schedule if config.gradient_checkpoint else "none"
    if config.storage not in VALID_STORAGE:
        raise ValueError(
            f"unknown storage {config.storage!r}; expected one of {VALID_STORAGE}"
        )
    # Host offload only modifies an ACTIVE checkpoint.  If checkpointing is
    # disabled (schedule "none") a non-default storage is contradictory — reject
    # it rather than silently re-enabling jax.checkpoint behind gradient_checkpoint=False.
    if schedule == "none" and config.storage != "recompute":
        raise ValueError(
            f"RolloutConfig.storage={config.storage!r} requires gradient "
            "checkpointing; set gradient_checkpoint=True and "
            "checkpoint_schedule='uniform'."
        )
    step_fn = _one_segment
    if schedule == "uniform":
        policy = offload_policy(config.storage)  # None for recompute, offload for host
        step_fn = (
            jax.checkpoint(_one_segment, prevent_cse=False, policy=policy)
            if policy is not None
            else jax.checkpoint(_one_segment, prevent_cse=False)
        )

    final_carry, (all_carries, all_mean_T) = jax.lax.scan(
        step_fn, initial_carry, jnp.arange(n_segments),
    )

    # Save days (1-indexed)
    saved_days = jnp.arange(1, n_segments + 1).astype(jnp.float32)

    return RolloutOutput(
        final_carry=final_carry,
        saved_states=all_carries,
        saved_days=saved_days,
    )


def single_day_rollout(
    initial_carry: SegmentCarry,
    forcing: SegmentForcing,
    run_segment_fn,
    dt: float = 600.0,
    hours: float = 24.0,
) -> SegmentCarry:
    """Run a fixed-length rollout and return the final state.

    ``hours`` is the supervision horizon (default 24 h).  SHORTER
    horizons are strongly preferred for gradient-based training: the
    adjoint through a long primitive-equation rollout grows with the
    dynamics' positive Lyapunov exponents, so a 24 h (144-step) rollout
    explodes the gradient to NaN even when the forward is finite — a
    6 h horizon (the NeuralGCM / AIMIP convention) keeps the adjoint
    well-conditioned.  The target must be loaded at the same ``hours``
    lead.  ``run_segment_fn`` advances one step per ``dt``.
    """
    n_steps = int(round(hours * 3600.0 / dt))
    return run_segment_fn(initial_carry, n_steps, forcing)
