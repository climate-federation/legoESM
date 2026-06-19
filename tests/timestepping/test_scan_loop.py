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
