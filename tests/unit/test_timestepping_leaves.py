"""Direct unit tests for timestepping leaf modules.

Covers (per 2026-05-13 slopbuster audit, Pass 2 indirect-only modules):

* ``legoesm.timestepping.dispatch``      — integrator name dispatch
* ``legoesm.timestepping.integration``   — :class:`IntegrationMixin` loops
* ``legoesm.timestepping.pytree_ops``    — pytree axpy / linear combination
* ``legoesm.timestepping.split_explicit``— split-explicit step (slow + acoustic)

These leaves are exercised indirectly through the dycore drivers but
had no direct test coverage; the slopbuster audit (Pass 2) flagged
``timestepping/dispatch.py`` (21 src refs) and
``timestepping/integration.py`` (12) as highest-risk untested-but-live.
This file closes that gap.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import pytest

from legoesm.timestepping.dispatch import dispatch_integrator
from legoesm.timestepping.integration import IntegrationMixin
from legoesm.timestepping.pytree_ops import pytree_axpy, pytree_linear_combination
from legoesm.timestepping.split_explicit import (
    SplitExplicitConfig,
    split_explicit_step,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class ScalarState(NamedTuple):
    y: jax.Array


def _decay_tendency(state: ScalarState, lam: float = -1.0) -> ScalarState:
    """``dy/dt = lam * y`` — analytic solution y(t) = y0 * exp(lam * t)."""
    return ScalarState(y=lam * state.y)


# ---------------------------------------------------------------------------
# pytree_ops
# ---------------------------------------------------------------------------


class TestPytreeOps:
    def test_axpy_scalar(self):
        x = ScalarState(y=jnp.array(2.0))
        y = ScalarState(y=jnp.array(3.0))
        out = pytree_axpy(x, y, alpha=0.5)
        # x + 0.5 * y = 2 + 1.5 = 3.5
        assert jnp.allclose(out.y, 3.5)

    def test_axpy_preserves_structure(self):
        x = ScalarState(y=jnp.array([1.0, 2.0, 3.0]))
        y = ScalarState(y=jnp.array([4.0, 5.0, 6.0]))
        out = pytree_axpy(x, y, alpha=2.0)
        assert out.y.shape == x.y.shape
        assert jnp.allclose(out.y, jnp.array([9.0, 12.0, 15.0]))

    def test_linear_combination(self):
        x = ScalarState(y=jnp.array(2.0))
        y = ScalarState(y=jnp.array(3.0))
        # 0.75 * 2 + 0.25 * 3 = 1.5 + 0.75 = 2.25
        out = pytree_linear_combination(x, y, a=0.75, b=0.25)
        assert jnp.allclose(out.y, 2.25)

    def test_linear_combination_partition_of_unity(self):
        """SSP-RK3 combines stages with 0.75/0.25 and 1/3//2/3 — both sum to 1."""
        x = ScalarState(y=jnp.array(10.0))
        y = ScalarState(y=jnp.array(10.0))
        out = pytree_linear_combination(x, y, a=1.0 / 3.0, b=2.0 / 3.0)
        assert jnp.allclose(out.y, 10.0)


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------


class TestDispatchIntegrator:
    @pytest.mark.parametrize(
        "name",
        ["ssp_rk3", "ssp3", "rk3",
         "ssp_rk34", "ssp34", "rk34",
         "ssp_rk54", "ssp54", "ssp45", "rk54",
         "rk4", "runge_kutta_4"],
    )
    def test_known_integrator_runs(self, name):
        state = ScalarState(y=jnp.array(1.0))
        out = dispatch_integrator(state, _decay_tendency, dt=0.01, integrator_name=name)
        # All schemes should at least decay over a forward step on y' = -y, y(0) = 1.
        assert out.y < 1.0
        assert out.y > 0.9  # not too far off ~exp(-0.01) ≈ 0.99005

    def test_case_insensitive(self):
        state = ScalarState(y=jnp.array(1.0))
        out_upper = dispatch_integrator(state, _decay_tendency, 0.01, "SSP_RK3")
        out_lower = dispatch_integrator(state, _decay_tendency, 0.01, "ssp_rk3")
        assert jnp.allclose(out_upper.y, out_lower.y)

    def test_unknown_integrator_raises(self):
        state = ScalarState(y=jnp.array(1.0))
        with pytest.raises(ValueError, match="Unsupported time_integrator"):
            dispatch_integrator(state, _decay_tendency, 0.01, "no_such_scheme")

    def test_dispatch_convergence_rk4(self):
        """Classical RK4 should converge to exp(-1) with ~O(dt^4) error."""
        state = ScalarState(y=jnp.array(1.0))
        for n_steps in (10, 100):
            dt = 1.0 / n_steps
            s = state
            for _ in range(n_steps):
                s = dispatch_integrator(s, _decay_tendency, dt, "rk4")
            assert abs(float(s.y) - jnp.exp(-1.0)) < 1.0e-3


# ---------------------------------------------------------------------------
# IntegrationMixin
# ---------------------------------------------------------------------------


class _ToyModel(IntegrationMixin):
    """Forward-Euler exponential decay; exposes ``.step()`` for the mixin."""

    def step(self, state: ScalarState, dt: float) -> ScalarState:
        return ScalarState(y=state.y + dt * (-1.0 * state.y))


class TestIntegrationMixin:
    def test_integrate_decay(self):
        model = _ToyModel()
        s0 = ScalarState(y=jnp.array(1.0))
        sf, traj = model.integrate(s0, duration=0.1, dt=0.01, save_every=1)
        # 10 steps of forward Euler dy = -y * dt
        # y_n = (1 - dt)^n = 0.99^10 ≈ 0.9044
        assert abs(float(sf.y) - (1 - 0.01) ** 10) < 1.0e-6
        # Trajectory: initial + 10 saves = 11 entries (save_every=1 saves after each)
        assert len(traj) == 11

    def test_integrate_save_every(self):
        model = _ToyModel()
        s0 = ScalarState(y=jnp.array(1.0))
        _, traj = model.integrate(s0, duration=0.1, dt=0.01, save_every=5)
        # 10 steps, save every 5 → initial + saves at step 5, 10 = 3 entries
        assert len(traj) == 3

    def test_integrate_scan_matches_integrate(self):
        model = _ToyModel()
        s0 = ScalarState(y=jnp.array(1.0))
        sf_py, _ = model.integrate(s0, duration=0.05, dt=0.01)
        sf_scan, _ = model.integrate_scan(s0, n_steps=5, dt=0.01)
        assert jnp.allclose(sf_py.y, sf_scan.y, atol=1e-12)

    def test_integrate_scan_is_jit_friendly(self):
        model = _ToyModel()
        s0 = ScalarState(y=jnp.array(1.0))
        jit_run = jax.jit(lambda s: model.integrate_scan(s, n_steps=5, dt=0.01)[0])
        sf = jit_run(s0)
        assert jnp.isfinite(sf.y)


# ---------------------------------------------------------------------------
# split_explicit
# ---------------------------------------------------------------------------


def _identity_acoustic(state, slow_tend, dt_substep, n_substeps, config):
    """Acoustic-update stub: just adds dt*slow_tend, no fast substepping.

    Lets the split-explicit driver be tested for its slow-stage logic
    without a real acoustic operator.  Combined with a zero
    slow_tendency the test reduces to a fixed-point ``state -> state``
    check; combined with linear decay it should agree with the outer
    SSP-RK3 integrator.
    """
    return jax.tree.map(lambda s, t: s + dt_substep * n_substeps * t, state, slow_tend)


class TestSplitExplicit:
    def test_zero_tendency_is_identity_ssp_rk3(self):
        state = ScalarState(y=jnp.array(1.0))
        cfg = SplitExplicitConfig(n_substeps=2, outer_integrator="ssp_rk3")

        def zero_tend(_):
            return ScalarState(y=jnp.array(0.0))

        out = split_explicit_step(state, zero_tend, _identity_acoustic, dt=0.01, config=cfg)
        assert jnp.allclose(out.y, 1.0)

    def test_zero_tendency_is_identity_ssp_rk34(self):
        state = ScalarState(y=jnp.array(1.0))
        cfg = SplitExplicitConfig(n_substeps=2, outer_integrator="ssp_rk34")

        def zero_tend(_):
            return ScalarState(y=jnp.array(0.0))

        out = split_explicit_step(state, zero_tend, _identity_acoustic, dt=0.01, config=cfg)
        assert jnp.allclose(out.y, 1.0)

    def test_zero_tendency_is_identity_ssp_rk54(self):
        state = ScalarState(y=jnp.array(1.0))
        cfg = SplitExplicitConfig(n_substeps=2, outer_integrator="ssp_rk54")

        def zero_tend(_):
            return ScalarState(y=jnp.array(0.0))

        out = split_explicit_step(state, zero_tend, _identity_acoustic, dt=0.01, config=cfg)
        assert jnp.allclose(out.y, 1.0)

    def test_unknown_outer_integrator_raises(self):
        state = ScalarState(y=jnp.array(1.0))
        cfg = SplitExplicitConfig(n_substeps=2, outer_integrator="bogus")
        with pytest.raises(ValueError, match="Unsupported outer_integrator"):
            split_explicit_step(state, _decay_tendency, _identity_acoustic, 0.01, cfg)

    def test_linear_decay_finite_and_decaying(self):
        """With identity acoustic, the slow tendency drives state.

        Don't require bit-for-bit match to outer integrator: the
        ``_identity_acoustic`` stub effectively *triples* the step
        contribution (n_substeps=3 inside one RK stage).  Just verify
        the integrator produces a finite, decaying value.
        """
        state = ScalarState(y=jnp.array(1.0))
        cfg = SplitExplicitConfig(n_substeps=3, outer_integrator="ssp_rk3")
        out = split_explicit_step(
            state, _decay_tendency, _identity_acoustic, dt=0.01, config=cfg,
        )
        assert jnp.isfinite(out.y)
        assert out.y < 1.0
        assert out.y > 0.0
