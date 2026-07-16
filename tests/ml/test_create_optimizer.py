"""Direct unit test for ``legoesm.ml.training.create_optimizer`` schedule guard.

``create_optimizer`` builds an ``optax.warmup_cosine_decay_schedule`` whose cosine
leg is ``decay_steps - warmup_steps``; ``optax.cosine_decay_schedule`` raises
``requires positive decay_steps`` when that is <= 0.  A tiny-steps smoke run (e.g.
the carbon calibration launched with ``--steps 6`` under the sbatch default
``WARMUP=10``, or a 1-step unit smoke) would otherwise crash at optimizer
construction.  The guard clamps ``total_steps>=1`` and
``warmup_steps<=total_steps-1`` so the cosine leg is always >=1 step.

These build only the optax schedule/optimizer (no model, no compile) and run in a
fraction of a second.
"""
from __future__ import annotations

import optax
import pytest

from legoesm.ml.training import TrainingConfig, create_optimizer


@pytest.mark.parametrize(
    "total_steps,warmup_steps",
    [
        (1, 1),      # the mandated tiny case: warmup == total
        (6, 10),     # the real sbatch scenario: --steps 6, default WARMUP=10
        (1, 1000),   # warmup hugely exceeds a 1-step run
        (2, 2),      # warmup == total, > 1
    ],
)
def test_tiny_steps_does_not_crash(total_steps, warmup_steps):
    """warmup_steps >= total_steps must NOT raise (pre-guard: negative cosine leg)."""
    opt = create_optimizer(
        TrainingConfig(lr=1e-3, warmup_steps=warmup_steps,
                       total_steps=total_steps, optimizer="adam"))
    # A real optax transformation with an init/update pair.
    assert isinstance(opt, optax.GradientTransformation)
    # Exercise it on a trivial scalar-leaf pytree so the (guarded) schedule is
    # actually evaluated inside the update, proving it produces finite values.
    import jax.numpy as jnp
    params = {"w": jnp.ones((2, 2)), "b": jnp.zeros((2,))}
    state = opt.init(params)
    grads = {"w": jnp.ones((2, 2)), "b": jnp.ones((2,))}
    updates, _ = opt.update(grads, state, params)
    assert jnp.all(jnp.isfinite(updates["w"]))
    assert jnp.all(jnp.isfinite(updates["b"]))


def test_zero_total_steps_is_clamped():
    """Defensive: total_steps=0 clamps to 1 (>=1 cosine step) rather than crash."""
    opt = create_optimizer(
        TrainingConfig(lr=1e-3, warmup_steps=0, total_steps=0, optimizer="adamw"))
    assert isinstance(opt, optax.GradientTransformation)


def test_normal_schedule_still_builds():
    """A well-posed (warmup < total) config is unaffected by the guard."""
    opt = create_optimizer(
        TrainingConfig(lr=5e-4, warmup_steps=10, total_steps=100, optimizer="adam"))
    assert isinstance(opt, optax.GradientTransformation)
