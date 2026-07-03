"""Gradient-identity + budget tests for the adjoint checkpoint schedules.

The contract: ``none`` / ``uniform`` / ``binomial`` (and the ``host`` storage
tier) must all yield the SAME gradient — only the adjoint memory footprint
differs.  This is the correctness gate for the long-rollout checkpointing.
"""
import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.training.checkpoint_schedule import (  # noqa: E402
    VALID_SCHEDULES,
    checkpointed_loop,
    sqrt_checkpoints,
)


def _body(A, b):
    def body(c):
        return jnp.tanh(A @ c + b)
    return body


def _loss(x0, A, b, n, schedule, **kw):
    final = checkpointed_loop(_body(A, b), x0, n, schedule=schedule, **kw)
    return jnp.sum(final ** 2)


def _problem(d=5, seed=0):
    k = jax.random.PRNGKey(seed)
    A = 0.5 * jax.random.normal(k, (d, d))
    b = 0.1 * jax.random.normal(jax.random.PRNGKey(seed + 1), (d,))
    x0 = jax.random.normal(jax.random.PRNGKey(seed + 2), (d,))
    return A, b, x0


@pytest.mark.parametrize("n", [1, 7, 32])
def test_grad_identity_wrt_initial_state(n):
    A, b, x0 = _problem()
    g_none = jax.grad(_loss)(x0, A, b, n, "none")
    g_unif = jax.grad(_loss)(x0, A, b, n, "uniform")
    g_bino = jax.grad(_loss)(x0, A, b, n, "binomial")
    assert jnp.allclose(g_none, g_unif, rtol=1e-10, atol=1e-12)
    assert jnp.allclose(g_none, g_bino, rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("n", [4, 25])
def test_grad_identity_wrt_parameter(n):
    A, b, x0 = _problem(seed=3)
    g_none = jax.grad(_loss, argnums=1)(x0, A, b, n, "none")
    g_bino = jax.grad(_loss, argnums=1)(x0, A, b, n, "binomial")
    assert jnp.allclose(g_none, g_bino, rtol=1e-8, atol=1e-10)


def test_pytree_carry_binomial_matches():
    def body(c):
        a, s = c
        return (jnp.tanh(a + s), 0.99 * s)

    def loss(x0, n, schedule):
        a, s = checkpointed_loop(body, (x0, x0), n, schedule=schedule)
        return jnp.sum(a ** 2) + jnp.sum(s ** 2)

    x0 = jnp.linspace(-1.0, 1.0, 6)
    g_none = jax.grad(loss)(x0, 20, "none")
    g_bino = jax.grad(loss)(x0, 20, "binomial")
    assert jnp.allclose(g_none, g_bino, rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("schedule", ["uniform", "binomial"])
def test_host_storage_matches_grad(schedule):
    A, b, x0 = _problem(seed=5)
    n = 12
    g_dev = jax.grad(_loss)(x0, A, b, n, schedule, storage="recompute")
    try:
        g_host = jax.grad(_loss)(x0, A, b, n, schedule, storage="host")
    except Exception as e:  # offload unsupported on this backend (e.g. plain CPU)
        pytest.skip(f"host offload unsupported here: {e}")
    assert jnp.allclose(g_dev, g_host, rtol=1e-6, atol=1e-8)


def test_host_storage_rejected_for_none_schedule():
    # "none" has no checkpoint to offload — host storage must raise, not silently
    # re-enable rematerialisation (consistency with differentiable_rollout).
    with pytest.raises(ValueError, match="requires an active checkpoint"):
        checkpointed_loop(lambda c: c, jnp.zeros(3), 4, schedule="none", storage="host")


def test_value_matches_plain_scan():
    A, b, x0 = _problem(seed=7)
    n = 9
    body = _body(A, b)
    # reference: explicit python rollout
    ref = x0
    for _ in range(n):
        ref = body(ref)
    for sch in VALID_SCHEDULES:
        out = checkpointed_loop(body, x0, n, schedule=sch)
        assert jnp.allclose(out, ref, rtol=1e-9, atol=1e-11), sch


def test_sqrt_checkpoints_is_ceil_sqrt():
    assert sqrt_checkpoints(1) == 1
    assert sqrt_checkpoints(2) == 2
    assert sqrt_checkpoints(4) == 2
    assert sqrt_checkpoints(5) == 3
    assert sqrt_checkpoints(16) == 4
    assert sqrt_checkpoints(17) == 5


def test_zero_steps_is_identity():
    x = jnp.arange(4.0)
    assert jnp.allclose(checkpointed_loop(lambda c: c + 1.0, x, 0), x)


def test_zero_steps_still_validates_dispatch():
    # config errors must surface even on an empty window (codex round-5):
    # validation precedes the n_steps<=0 early return.
    with pytest.raises(ValueError, match="unknown schedule"):
        checkpointed_loop(lambda c: c, jnp.zeros(3), 0, schedule="bogus")
    with pytest.raises(ValueError, match="unknown storage"):
        checkpointed_loop(lambda c: c, jnp.zeros(3), 0, storage="bogus")
    with pytest.raises(ValueError, match="requires an active checkpoint"):
        checkpointed_loop(lambda c: c, jnp.zeros(3), 0, schedule="none", storage="host")


def test_unknown_schedule_raises():
    with pytest.raises(ValueError, match="unknown schedule"):
        checkpointed_loop(lambda c: c, jnp.zeros(3), 4, schedule="bogus")


def test_unknown_storage_raises():
    with pytest.raises(ValueError, match="unknown storage"):
        checkpointed_loop(lambda c: c, jnp.zeros(3), 4, storage="bogus")
