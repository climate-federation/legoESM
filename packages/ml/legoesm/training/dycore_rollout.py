"""Differentiable multi-day rollout through the compiled dycore.

Wraps ``build_segment_fn`` into a higher-level loop that chains
segments together with gradient checkpointing, supporting 1-day to
14-day rollouts for training via backpropagation.

The key function ``differentiable_rollout`` runs N segments through
``jax.lax.scan``, optionally collecting intermediate states for
multi-day loss computation.
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm.driver.compiled_segments import (
    SegmentCarry,
    SegmentForcing,
    build_segment_fn,
    pack_forcing,
)


class RolloutConfig(NamedTuple):
    """Configuration for a differentiable rollout."""
    n_days: int = 1
    dt: float = 600.0             # timestep [s]
    segment_steps: int = 144      # steps per segment (= 1 day at dt=600)
    save_every_n_segments: int = 1  # save state every N segments for loss
    gradient_checkpoint: bool = True


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
    Each segment runs ``config.segment_steps`` time steps.  Intermediate
    states are saved at intervals for multi-day loss computation.

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

    # Wrap with gradient checkpointing if requested
    step_fn = _one_segment
    if config.gradient_checkpoint:
        step_fn = jax.checkpoint(_one_segment, prevent_cse=False)

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
) -> SegmentCarry:
    """Convenience: run a 1-day rollout and return the final state.

    Useful for Mode 1 (physics param tuning) where only the endpoint
    matters for loss computation.
    """
    steps_per_day = int(86400 / dt)
    return run_segment_fn(initial_carry, steps_per_day, forcing)


def make_rollout_loss_fn(
    run_segment_fn,
    loss_fn,
    config: RolloutConfig,
    forcing: SegmentForcing,
):
    """Create a differentiable loss function for training.

    Returns a function ``loss(params, initial_carry, targets) -> scalar``
    that can be differentiated with ``jax.grad`` or
    ``eqx.filter_value_and_grad``.

    Parameters
    ----------
    run_segment_fn : callable
        Compiled segment function.
    loss_fn : callable
        Loss function: ``loss(predicted_carry, target, config) -> scalar``.
    config : RolloutConfig
        Rollout configuration.
    forcing : SegmentForcing
        External forcing for the rollout period.

    Returns
    -------
    callable
        ``loss(initial_carry, target_carry) -> scalar``
    """
    def _loss(initial_carry, target_carry):
        output = differentiable_rollout(
            initial_carry, forcing, run_segment_fn, config,
        )
        return loss_fn(output.final_carry, target_carry)

    return _loss
