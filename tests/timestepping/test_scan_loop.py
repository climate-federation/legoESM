"""Shared integrate_scan_generic loop (ponytail dedup 2026-06-17): the SSP-RK
integrators delegate their scan/checkpoint/trajectory plumbing to it."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from legoesm.timestepping.scan_loop import integrate_scan_generic


def _decay_step(s):  # one explicit-Euler decay step, dt baked in
    return s - 0.1 * 0.3 * s


def test_trajectory_and_final():
    x0 = jnp.asarray(np.linspace(1.0, 5.0, 6))
    final, traj = integrate_scan_generic(x0, _decay_step, 5)
    assert traj.shape == (5, 6)
    # final == last trajectory entry == 5 applications of the step
    ref = x0
    for _ in range(5):
        ref = _decay_step(ref)
    np.testing.assert_allclose(final, ref)
    np.testing.assert_allclose(traj[-1], final)


def test_no_trajectory_returns_none():
    x0 = jnp.ones(4)
    final, traj = integrate_scan_generic(x0, _decay_step, 3, return_trajectory=False)
    assert traj is None
    assert final.shape == (4,)


def test_checkpoint_matches_plain_and_is_differentiable():
    x0 = jnp.asarray(np.linspace(1.0, 2.0, 4))
    f_plain, _ = integrate_scan_generic(x0, _decay_step, 6, return_trajectory=False)
    f_ckpt, _ = integrate_scan_generic(
        x0, _decay_step, 6, checkpoint_interval=1, return_trajectory=False)
    np.testing.assert_array_equal(f_plain, f_ckpt)
    g = jax.grad(lambda x: integrate_scan_generic(
        x, _decay_step, 6, checkpoint_interval=1, return_trajectory=False)[0].sum())(x0)
    assert jnp.all(jnp.isfinite(g))


def test_host_storage_matches_recompute_grad():
    x0 = jnp.asarray(np.linspace(1.0, 2.0, 4))

    def _grad(storage):
        return jax.grad(lambda x: integrate_scan_generic(
            x, _decay_step, 6, checkpoint_interval=1,
            return_trajectory=False, storage=storage)[0].sum())(x0)

    g_dev = _grad("recompute")
    try:
        g_host = _grad("host")  # offloads matmul residuals; no-op fallback on CPU
    except Exception as e:  # pragma: no cover - backend without host memory kind
        import pytest
        pytest.skip(f"host offload unsupported here: {e}")
    np.testing.assert_allclose(g_dev, g_host, rtol=1e-9, atol=1e-12)


def test_unknown_storage_raises():
    import pytest
    x0 = jnp.ones(4)
    with pytest.raises(ValueError, match="unknown storage"):
        integrate_scan_generic(x0, _decay_step, 3, checkpoint_interval=1,
                               storage="bogus")


def test_host_storage_requires_active_checkpoint():
    # storage='host' with no checkpoint (interval<=0) is contradictory -> raise
    import pytest
    x0 = jnp.ones(4)
    with pytest.raises(ValueError, match="requires checkpoint_interval"):
        integrate_scan_generic(x0, _decay_step, 3, checkpoint_interval=0,
                               storage="host")
