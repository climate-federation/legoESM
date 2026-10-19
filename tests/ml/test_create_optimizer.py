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

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from legoesm.ml.training import TrainingConfig, create_optimizer


@pytest.fixture(scope="module", autouse=True)
def _x64():
    previous = bool(jax.config.jax_enable_x64)
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


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


def _run(optimizer, params, gradients):
    state = optimizer.init(params)
    updates = []
    for grads in gradients:
        update, state = optimizer.update(grads, state, params)
        params = optax.apply_updates(params, update)
        updates.append(update)
    return updates, state, params


def test_historical_defaults_match_previous_transform_bit_for_bit():
    """The historical update, written out with its literal values.

    Three steps so the warmup rate is nonzero after step one; a gradient norm
    above the 1.0 clip so clipping acts; and elements near 1e-7 so a changed
    epsilon moves the update.
    """
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=5.0e-4, warmup_steps=1000,
        decay_steps=100_000, end_value=0.0)
    previous = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=schedule, weight_decay=1.0e-5))
    params = {"w": jnp.asarray([1.0, -2.0]), "b": jnp.asarray([0.5])}
    gradients = [{"w": jnp.asarray([2.0, -1.0e-7]), "b": jnp.asarray([0.3])},
                 {"w": jnp.asarray([-1.5, 2.0e-7]), "b": jnp.asarray([0.9])},
                 {"w": jnp.asarray([0.7, -3.0e-7]), "b": jnp.asarray([-1.2])}]
    factory = _run(create_optimizer(TrainingConfig()), params, gradients)
    reference = _run(previous, params, gradients)
    assert np.all(np.asarray(factory[0][1]["w"]) != 0.0)
    for left, right in zip(jax.tree.leaves(factory),
                           jax.tree.leaves(reference), strict=True):
        assert np.array_equal(np.asarray(left), np.asarray(right))


def test_constant_unclipped_adamw_matches_hand_computed_first_step():
    """At |g| = epsilon the first step is half the rate; 1e-8 would give 0.91."""
    config = TrainingConfig(
        lr=1.0e-3, weight_decay=0.004, optimizer="adamw",
        schedule="constant", clip_gradients=False, adam_epsilon=1.0e-7)
    optimizer = create_optimizer(config)
    params = {"w": jnp.asarray([2.0])}
    grads = {"w": jnp.asarray([1.0e-7])}
    updates, _ = optimizer.update(grads, optimizer.init(params), params)
    expected = -1.0e-3 * (1.0e-7 / (1.0e-7 + 1.0e-7) + 0.004 * 2.0)
    assert float(updates["w"][0]) == pytest.approx(expected, rel=1.0e-12)


def _keras_adamw_reference(params, gradients, *, lr, weight_decay, b1, b2,
                           epsilon):
    """NumPy transcription of Keras 3.6.0 AdamW (no clipping, no AMSGrad).

    keras/src/optimizers/base_optimizer.py ``_apply_weight_decay``:
    ``variable - variable * wd * lr`` before the step;
    keras/src/optimizers/adam.py ``update_step``: ``m += (g - m)(1 - b1)``,
    ``v += (g^2 - v)(1 - b2)``, ``alpha = lr sqrt(1 - b2^t) / (1 - b1^t)``,
    ``variable -= m alpha / (sqrt(v) + epsilon)``.
    """
    p = np.asarray(params, dtype=np.float64).copy()
    m = np.zeros_like(p)
    v = np.zeros_like(p)
    for step, g in enumerate(gradients, start=1):
        g = np.asarray(g, dtype=np.float64)
        p = p - p * weight_decay * lr
        m = m + (g - m) * (1.0 - b1)
        v = v + (g * g - v) * (1.0 - b2)
        alpha = lr * np.sqrt(1.0 - b2**step) / (1.0 - b1**step)
        p = p - m * alpha / (np.sqrt(v) + epsilon)
    return p


def test_keras_epsilon_hat_adamw_matches_the_keras_update():
    """Keras-default AdamW over six steps, at |g| from 1 down to 1e-7."""
    config = TrainingConfig(
        lr=1.0e-3, weight_decay=0.004, optimizer="adamw",
        schedule="constant", clip_gradients=False, adam_epsilon=1.0e-7,
        adam_b1=0.9, adam_b2=0.999, adam_epsilon_convention="keras_epsilon_hat")
    rng = np.random.default_rng(4)
    params = rng.normal(size=8)
    magnitudes = 10.0 ** -np.arange(8, dtype=np.float64)
    gradients = [rng.normal(size=8) * magnitudes for _ in range(6)]

    def factory(settings):
        return np.asarray(_run(create_optimizer(settings),
                               {"w": jnp.asarray(params)},
                               [{"w": jnp.asarray(g)} for g in gradients])[2]["w"])

    expected = _keras_adamw_reference(
        params, gradients, lr=1.0e-3, weight_decay=0.004, b1=0.9, b2=0.999,
        epsilon=1.0e-7)
    keras = factory(config)
    # Same arithmetic up to operation order: rounding of O(1) parameters.
    np.testing.assert_allclose(keras, expected, rtol=0.0, atol=1.0e-14)
    # The Optax convention is a different optimizer at small gradients ...
    small = magnitudes <= 1.0e-6
    optax_form = factory(config._replace(adam_epsilon_convention="optax"))
    assert np.all(np.abs(optax_form - expected)[small]
                  > 0.1 * np.abs(expected - params)[small])
    # ... and the Keras result depends on epsilon itself: 1e-8 is not Keras.
    wrong = factory(config._replace(adam_epsilon=1.0e-8))
    assert np.all(np.abs(wrong - expected)[small]
                  > 0.1 * np.abs(expected - params)[small])


def test_optimizer_explicit_options_fail_closed():
    with pytest.raises(ValueError, match="Unknown schedule"):
        create_optimizer(TrainingConfig(schedule="typo"))
    with pytest.raises(ValueError, match="epsilon"):
        create_optimizer(TrainingConfig(adam_epsilon=0.0))
    with pytest.raises(ValueError, match="adam_epsilon_convention"):
        create_optimizer(TrainingConfig(adam_epsilon_convention="keras"))
    with pytest.raises(ValueError, match="'adam' and 'adamw'"):
        create_optimizer(TrainingConfig(
            optimizer="muon", adam_epsilon_convention="keras_epsilon_hat"))
    with pytest.raises(ValueError, match="b1"):
        create_optimizer(TrainingConfig(adam_b2=1.0))
