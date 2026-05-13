"""Differentiability tests for JAX transforms (jit / vmap / scan / checkpoint).

Verifies that ``jax.grad`` interacts correctly with the standard JAX
transformations on a representative dynamical core (CD-grid shallow
water).  Catches:
* JIT tracing errors (Python side-effects leaking through ``jax.jit``)
* vmap shape errors
* scan-carry dtype mismatches over long horizons
* checkpointing-induced gradient drift
* mixed-precision regressions in ``lax.cond`` / ``lax.scan`` branches

Categories:
  10a) JIT compilation of gradients
  10b) vmap over ensemble members
  10c) lax.scan gradient accumulation N=1, 5, 20, 50
  10d) Gradient checkpointing
  10e) Mixed precision (fp32 vs fp64)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def assert_gradient_ok(grad_array, name="", min_nonzero_frac=0.1):
    assert jnp.all(jnp.isfinite(grad_array)), f"{name}: gradient has NaN/Inf"
    nonzero_frac = jnp.mean(jnp.abs(grad_array) > 0).item()
    assert nonzero_frac >= min_nonzero_frac, (
        f"{name}: only {nonzero_frac*100:.1f}% non-zero "
        f"(need {min_nonzero_frac*100:.0f}%)"
    )


@pytest.fixture(scope="module")
def sw_model_state():
    """Build a small CD-grid shallow water model + state.

    Module-scoped so the ~200 ms grid construction is amortised across
    every test.
    """
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import (
        CDGridShallowWaterModel, CDGridShallowWaterConfig,
        CDGridShallowWaterState,
    )
    n = 4
    base = create_cubed_sphere(n)
    config = CDGridShallowWaterConfig()
    model = CDGridShallowWaterModel(base, config)
    h0 = 1000.0 * jnp.ones((6, n, n))
    h0 = h0 + 10.0 * jax.random.normal(jax.random.PRNGKey(0), (6, n, n))
    state = CDGridShallowWaterState(
        h=h0,
        u_d=jnp.zeros((6, n + 1, n + 1)),
        v_d=jnp.zeros((6, n + 1, n + 1)),
        h_s=jnp.zeros((6, n, n)),
    )
    return model, state, 60.0


# ===========================================================================
# 10a  JIT compilation of gradients
# ===========================================================================

class TestJitGrad:
    """``jax.jit(jax.grad(loss))`` must produce the same gradient as
    ``jax.grad(loss)`` — catches tracing errors and Python side-effects
    that leak through the JIT boundary."""

    def test_jit_grad_matches_eager(self, sw_model_state):
        model, state, dt = sw_model_state

        def loss(h_init):
            s = state._replace(h=h_init)
            return jnp.sum(model.step(s, dt).h ** 2)

        grad_eager = jax.grad(loss)(state.h)
        grad_jit = jax.jit(jax.grad(loss))(state.h)

        assert_gradient_ok(grad_eager, "eager grad")
        assert_gradient_ok(grad_jit, "jit grad")
        assert jnp.allclose(grad_jit, grad_eager, atol=1e-12, rtol=1e-12), (
            "jit(grad) and grad disagree — JIT tracing is altering the "
            "gradient computation"
        )

    def test_value_and_grad_under_jit(self, sw_model_state):
        model, state, dt = sw_model_state

        def loss(h_init):
            s = state._replace(h=h_init)
            return jnp.sum(model.step(s, dt).h ** 2)

        val_e, grad_e = jax.value_and_grad(loss)(state.h)
        val_j, grad_j = jax.jit(jax.value_and_grad(loss))(state.h)

        assert jnp.isfinite(val_e) and jnp.isfinite(val_j)
        assert jnp.allclose(val_j, val_e, atol=1e-12, rtol=1e-12)
        assert jnp.allclose(grad_j, grad_e, atol=1e-12, rtol=1e-12)


# ===========================================================================
# 10b  vmap over ensemble members
# ===========================================================================

class TestVmapEnsemble:
    """``jax.vmap(jax.grad(loss))`` over a batch of states must give a
    finite gradient for every member with the right shape."""

    def test_vmap_over_initial_h_ensemble(self, sw_model_state):
        model, state, dt = sw_model_state

        def loss(h_init):
            s = state._replace(h=h_init)
            return jnp.sum(model.step(s, dt).h ** 2)

        n_ens = 4
        # Make 4 perturbed copies of the initial h field.
        keys = jax.random.split(jax.random.PRNGKey(1), n_ens)
        batch_h = jnp.stack([
            state.h + 5.0 * jax.random.normal(k, state.h.shape) for k in keys
        ])  # shape (n_ens, 6, n, n)

        batched_grad = jax.vmap(jax.grad(loss))(batch_h)

        assert batched_grad.shape == (n_ens,) + state.h.shape
        for i in range(n_ens):
            assert_gradient_ok(batched_grad[i], f"ensemble member {i}")

    def test_vmap_grads_distinct(self, sw_model_state):
        """Distinct initial conditions must yield distinct gradients —
        a sanity check that vmap is actually batching, not broadcasting
        a single computation."""
        model, state, dt = sw_model_state

        def loss(h_init):
            s = state._replace(h=h_init)
            return jnp.sum(model.step(s, dt).h ** 2)

        h1 = state.h + 5.0 * jax.random.normal(jax.random.PRNGKey(2), state.h.shape)
        h2 = state.h + 5.0 * jax.random.normal(jax.random.PRNGKey(3), state.h.shape)
        batch = jnp.stack([h1, h2])
        gs = jax.vmap(jax.grad(loss))(batch)
        assert not jnp.allclose(gs[0], gs[1]), (
            "vmap collapsed distinct ensemble members to identical gradients"
        )


# ===========================================================================
# 10c  lax.scan gradient accumulation N=1, 5, 20, 50
# ===========================================================================

class TestScanAccumulation:
    """Gradient through ``jax.lax.scan`` must remain finite over long
    horizons.  Catches gradient explosion / vanishing in dycore
    composition + scan-carry dtype mismatches."""

    @pytest.mark.parametrize("n_steps", [1, 5, 20, 50])
    def test_grad_finite_for_n_steps(self, sw_model_state, n_steps):
        model, state, dt = sw_model_state

        def loss(h_init):
            s = state._replace(h=h_init)
            def body(carry, _):
                return model.step(carry, dt), None
            s_final, _ = jax.lax.scan(body, s, None, length=n_steps)
            return jnp.sum(s_final.h ** 2)

        grad = jax.grad(loss)(state.h)
        assert_gradient_ok(grad, f"scan N={n_steps}")

    def test_grad_changes_with_horizon(self, sw_model_state):
        """Gradients at N=1 and N=20 must differ — catches the silent
        ``jax.lax.cond`` / ``jax.lax.stop_gradient`` failure where the
        gradient gets truncated past step 1.

        Magnitude alone is a bad signal here: the rest-state SW
        gradient is dominated by ``2·h_mean`` (the constant background)
        which scan does not change.  The *spatial pattern*, however,
        must evolve over the 20-step window.
        """
        model, state, dt = sw_model_state

        def loss_at(n_steps):
            def loss(h_init):
                s = state._replace(h=h_init)
                def body(carry, _):
                    return model.step(carry, dt), None
                s_final, _ = jax.lax.scan(body, s, None, length=n_steps)
                return jnp.sum(s_final.h ** 2)
            return jax.grad(loss)(state.h)

        g1 = loss_at(1)
        g20 = loss_at(20)
        # Subtract the constant-background component (gradient is
        # dominated by ``2·h_mean`` which is identical across horizons)
        # and check that the residual perturbation pattern is non-trivially
        # different between N=1 and N=20.
        g1_pert = g1 - jnp.mean(g1)
        g20_pert = g20 - jnp.mean(g20)
        rel_diff = (
            jnp.linalg.norm(g20_pert - g1_pert)
            / (jnp.linalg.norm(g1_pert) + 1e-30)
        )
        # Threshold is loose because rest-state SW is highly stable —
        # the test only needs to distinguish ``scan propagates a real
        # (if small) chain-rule update`` from ``scan silently
        # collapses to step 1``.  ``rel_diff = 0`` would mean the
        # latter; anything above the fp64 round-off floor proves scan
        # is doing real work.
        assert rel_diff > 1e-3, (
            f"scan-N=20 perturbation gradient identical to N=1 "
            f"(rel_diff={float(rel_diff):.3e}) — scan may be silently "
            f"truncating the gradient chain"
        )


# ===========================================================================
# 10d  Gradient checkpointing
# ===========================================================================

class TestCheckpointing:
    """``jax.checkpoint`` must not change the *value* of the gradient,
    only the memory footprint."""

    def test_checkpoint_grad_matches_unckpt(self, sw_model_state):
        model, state, dt = sw_model_state
        n_steps = 8

        def loss_plain(h_init):
            s = state._replace(h=h_init)
            def body(carry, _):
                return model.step(carry, dt), None
            s_final, _ = jax.lax.scan(body, s, None, length=n_steps)
            return jnp.sum(s_final.h ** 2)

        def loss_ckpt(h_init):
            s = state._replace(h=h_init)
            ckpt_step = jax.checkpoint(lambda c: model.step(c, dt))
            def body(carry, _):
                return ckpt_step(carry), None
            s_final, _ = jax.lax.scan(body, s, None, length=n_steps)
            return jnp.sum(s_final.h ** 2)

        g_plain = jax.grad(loss_plain)(state.h)
        g_ckpt = jax.grad(loss_ckpt)(state.h)
        assert_gradient_ok(g_plain, "plain")
        assert_gradient_ok(g_ckpt, "checkpointed")
        assert jnp.allclose(g_plain, g_ckpt, atol=1e-10, rtol=1e-10), (
            "checkpointing changed the gradient values — should only affect "
            "memory, not numerical result"
        )


# ===========================================================================
# 10e  Mixed precision (fp32 vs fp64)
# ===========================================================================

class TestMixedPrecision:
    """fp32 and fp64 gradients must both be finite.  Catches dtype
    promotion bugs in ``lax.cond`` branches (e.g., one branch returns
    fp32 while the other returns fp64, breaking the cond's same-shape
    invariant under AD)."""

    def _grad_at_dtype(self, sw_model_state, dtype):
        model, state, dt = sw_model_state

        h32 = state.h.astype(dtype)
        u32 = state.u_d.astype(dtype)
        v32 = state.v_d.astype(dtype)
        hs32 = state.h_s.astype(dtype)
        s_dtype = state._replace(h=h32, u_d=u32, v_d=v32, h_s=hs32)

        def loss(h_init):
            s = s_dtype._replace(h=h_init)
            return jnp.sum(model.step(s, dt).h ** 2)

        return jax.grad(loss)(h32)

    def test_grad_finite_in_fp64(self, sw_model_state):
        g64 = self._grad_at_dtype(sw_model_state, jnp.float64)
        assert g64.dtype == jnp.float64
        assert_gradient_ok(g64, "fp64")

    def test_grad_finite_in_fp32(self, sw_model_state):
        g32 = self._grad_at_dtype(sw_model_state, jnp.float32)
        assert g32.dtype == jnp.float32
        assert_gradient_ok(g32, "fp32")

    def test_fp32_fp64_signs_agree(self, sw_model_state):
        """Different precisions may differ in magnitude, but the sign
        of each gradient entry should agree with the fp64 reference
        for the dominant entries."""
        g64 = self._grad_at_dtype(sw_model_state, jnp.float64)
        g32 = self._grad_at_dtype(sw_model_state, jnp.float32).astype(jnp.float64)
        # Top 25 % of |g64| entries — those above the noise floor.
        thresh = jnp.quantile(jnp.abs(g64), 0.75)
        mask = jnp.abs(g64) > thresh
        sign_match = jnp.mean(jnp.sign(g32[mask]) == jnp.sign(g64[mask])).item()
        assert sign_match > 0.95, (
            f"fp32 vs fp64 gradient signs disagree on {(1-sign_match)*100:.1f}% "
            f"of dominant entries — possible dtype promotion bug in cond/scan"
        )
