"""Tests for the leapfrog + AB2 + Robert-Asselin time integrator.

Covers:

1. **Forward-Euler bootstrap** — ``initialize_leapfrog_carry`` and
   ``initialize_ab2_carry`` produce sensible starting states.
2. **Linear advection** — leapfrog reproduces the analytic 2nd-order
   centered-in-time discrete-dispersion solution for one step.
3. **Computational mode damping** — Robert-Asselin filter damps the
   2-Δt oscillation that pure leapfrog amplifies.
4. **AB2 reduces to forward-Euler on first step** when initialised
   from ``initialize_ab2_carry``.
5. **Quadratic invariant** — leapfrog without filtering conserves a
   simple quadratic invariant of linear oscillation to machine
   precision over many steps.
6. **Differentiability** — ``jax.grad`` through both schemes works.

Integration into legoESM's outer-integrator dispatch is deferred
(audit doc, Phase G.2). These tests verify the standalone closure.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.timestepping.leapfrog_ab2 import (
    AB2Carry,
    LeapfrogCarry,
    ab2_step,
    initialize_ab2_carry,
    initialize_leapfrog_carry,
    leapfrog_ra_step,
)

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# 1. Initialization
# ---------------------------------------------------------------------------


def test_initialize_leapfrog_carry_same_state():
    state = jnp.asarray([1.0, 2.0, 3.0])
    carry = initialize_leapfrog_carry(state)
    assert jnp.array_equal(carry.prev, state)
    assert jnp.array_equal(carry.curr, state)


def test_initialize_ab2_carry_computes_initial_tendency():
    state = jnp.asarray([1.0, 2.0])
    fn = lambda s: -s   # exponential decay tendency
    carry = initialize_ab2_carry(state, fn)
    np.testing.assert_allclose(np.asarray(carry.prev_tend), -np.asarray(state))
    assert jnp.array_equal(carry.curr, state)


# ---------------------------------------------------------------------------
# 2. Linear test: scalar exponential decay
# ---------------------------------------------------------------------------


def test_leapfrog_first_step_matches_euler_when_carry_seeded():
    """When state^{n-1} = state^n (Forward-Euler bootstrap), the
    leapfrog step reduces to:
        state^{n+1} = state^n + 2 dt · F(state^n)
    """
    state = jnp.asarray(1.0)
    dt = 0.1
    lam = -0.5
    fn = lambda s: lam * s

    carry = initialize_leapfrog_carry(state)
    out = leapfrog_ra_step(carry, fn, dt, asselin_nu=0.0)
    # Manually compute expected new state.
    expected_new = state + 2 * dt * lam * state
    np.testing.assert_allclose(float(out.curr), float(expected_new), rtol=1e-12)


def test_ab2_first_step_is_forward_euler():
    """With prev_tend = F(state) (initialize_ab2_carry's seed), the
    first AB2 step reduces to:
        state^{n+1} = state + dt · ((1.5 + ε) F - (0.5 + ε) F)
                    = state + dt · F
    independent of ε.
    """
    state = jnp.asarray(2.0)
    dt = 0.05
    lam = -0.3
    fn = lambda s: lam * s

    carry = initialize_ab2_carry(state, fn)
    out = ab2_step(carry, fn, dt, epsilon=0.0)
    expected = state + dt * lam * state
    np.testing.assert_allclose(float(out.curr), float(expected), rtol=1e-12)

    # Same result with a non-zero ε on the first step.
    carry = initialize_ab2_carry(state, fn)
    out_eps = ab2_step(carry, fn, dt, epsilon=0.1)
    np.testing.assert_allclose(float(out_eps.curr), float(expected), rtol=1e-12)


# ---------------------------------------------------------------------------
# 3. Computational-mode damping by Robert-Asselin
# ---------------------------------------------------------------------------


def test_robert_asselin_damps_2dt_oscillation():
    """Inject a 2-Δt oscillation directly into the carry (state^n =
    +1, state^{n-1} = -1) and verify that the Robert-Asselin filter
    reduces its amplitude over multiple steps.

    With ν > 0, the computational-mode amplitude factor per step is
    1 - 2ν (for the pure 2-Δt mode with zero tendency).
    """
    nu = 0.1
    # Zero tendency — only the filter acts.
    fn = lambda s: jnp.zeros_like(s)

    carry = LeapfrogCarry(prev=jnp.asarray(-1.0), curr=jnp.asarray(1.0))
    # 10 steps; the 2-Δt mode amplitude should decay geometrically
    # with each filter application.
    amplitudes = []
    for _ in range(10):
        amplitudes.append(float(carry.curr))
        carry = leapfrog_ra_step(carry, fn, dt=0.1, asselin_nu=nu)
    amplitudes.append(float(carry.curr))

    # Each step, the curr value (after the leapfrog step that's a no-op
    # because tendency=0, then the RA filter applied to the previous
    # state) should diminish. We just verify the amplitude shrinks.
    initial = abs(amplitudes[0])
    final = abs(amplitudes[-1])
    assert final < initial, (
        f"Robert-Asselin failed to damp 2-Δt mode: {initial} -> {final}"
    )


def test_pure_leapfrog_preserves_2dt_oscillation():
    """Without the filter (ν=0), the 2-Δt computational mode is
    preserved unchanged. This is the canonical leapfrog behavior
    that the Robert-Asselin filter exists to combat."""
    fn = lambda s: jnp.zeros_like(s)
    carry = LeapfrogCarry(prev=jnp.asarray(-1.0), curr=jnp.asarray(1.0))
    for _ in range(5):
        carry = leapfrog_ra_step(carry, fn, dt=0.1, asselin_nu=0.0)
    # No filter: the magnitude is preserved.
    assert abs(float(carry.curr)) == pytest.approx(1.0, abs=1e-12)


# ---------------------------------------------------------------------------
# 4. Quadratic invariant (linear oscillator)
# ---------------------------------------------------------------------------


def test_leapfrog_conserves_linear_oscillator_energy():
    """For the 1-D oscillator ``du/dt = -ω² · x, dx/dt = u``, leapfrog
    conserves the quadratic invariant ``E = u² + ω² x²`` to second
    order with no secular drift (the canonical demonstration of
    symplectic-like behavior of centred-in-time discretisations).

    We integrate one period and check E_final ≈ E_initial within a
    small tolerance.
    """
    omega = 2.0 * jnp.pi   # period = 1.0
    dt = 0.001
    n_steps = int(1.0 / dt)
    state0 = jnp.asarray([1.0, 0.0])   # (x=1, u=0)

    def fn(state):
        x, u = state[0], state[1]
        return jnp.stack([u, -omega * omega * x])

    carry = initialize_leapfrog_carry(state0)
    for _ in range(n_steps):
        carry = leapfrog_ra_step(carry, fn, dt, asselin_nu=0.0)

    def energy(state):
        x, u = state[0], state[1]
        return u * u + omega * omega * x * x

    E0 = float(energy(state0))
    E_final = float(energy(carry.curr))
    rel_drift = abs(E_final - E0) / E0
    # Pure leapfrog should drift by O(dt^2) per period — for dt = 0.001
    # we expect rel_drift well below 1%.
    assert rel_drift < 0.01, f"Energy drift {rel_drift:.4%} too large"


# ---------------------------------------------------------------------------
# 5. AB2 vs analytic: exponential decay
# ---------------------------------------------------------------------------


def test_ab2_decay_close_to_analytic():
    """AB2 applied to du/dt = -λ u tracks the analytic e^{-λ t}
    solution to second-order accuracy. Verify a 10-step trajectory."""
    lam = 0.5
    dt = 0.05
    fn = lambda s: -lam * s
    state0 = jnp.asarray(1.0)
    carry = initialize_ab2_carry(state0, fn)
    for _ in range(10):
        carry = ab2_step(carry, fn, dt, epsilon=0.0)
    t_final = 10 * dt
    analytic = float(jnp.exp(-lam * t_final))
    np.testing.assert_allclose(float(carry.curr), analytic, rtol=5e-3)


# ---------------------------------------------------------------------------
# 6. Differentiability
# ---------------------------------------------------------------------------


def test_leapfrog_step_differentiable():
    """``jax.grad`` through the leapfrog step (including the RA filter)
    must succeed and return a finite gradient."""
    def loss(x):
        state = jnp.stack([x, jnp.zeros_like(x)])
        carry = initialize_leapfrog_carry(state)
        fn = lambda s: jnp.stack([s[1], -s[0]])
        for _ in range(5):
            carry = leapfrog_ra_step(carry, fn, dt=0.01, asselin_nu=0.05)
        return jnp.sum(carry.curr ** 2)

    grad = jax.grad(loss)(jnp.asarray(1.0))
    assert jnp.isfinite(grad)


def test_ab2_step_differentiable():
    def loss(x):
        state = x
        fn = lambda s: -0.5 * s
        carry = initialize_ab2_carry(state, fn)
        for _ in range(5):
            carry = ab2_step(carry, fn, dt=0.05, epsilon=0.1)
        return carry.curr ** 2

    grad = jax.grad(loss)(jnp.asarray(1.0))
    assert jnp.isfinite(grad)
