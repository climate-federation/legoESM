"""Differentiability tests for JAX transform compatibility.

Categories:
  10a) JIT compilation of gradients
  10b) vmap over ensemble members
  10c) lax.scan gradient accumulation
  10d) Gradient checkpointing
  10e) Mixed precision (float32 vs float64)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from legoesm.core.field import Field


def assert_gradient_ok(grad_array, name="", min_nonzero_frac=0.1):
    assert jnp.all(jnp.isfinite(grad_array)), f"{name}: gradient has NaN/Inf"
    nonzero_frac = jnp.mean(jnp.abs(grad_array) > 0).item()
    assert nonzero_frac >= min_nonzero_frac, (
        f"{name}: only {nonzero_frac*100:.1f}% non-zero (need {min_nonzero_frac*100:.0f}%)"
    )


# ============================================================================
# Shared fixture: lat-lon shallow water model
# ============================================================================

@pytest.fixture
def sw_latlon():
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.atmosphere.dynamics.shallow_water_fv_latlon import (
        FVShallowWaterLatLonModel,
    )
    from legoesm.core.state import ShallowWaterState

    grid = create_latlon_grid(8, 16)
    dt = 120.0
    model = FVShallowWaterLatLonModel(grid, dt=dt)

    key = jax.random.PRNGKey(0)
    h_data = 1000.0 + 10.0 * jax.random.normal(key, (8, 16))
    state = ShallowWaterState(
        h=Field(h_data, name="h"),
        u=Field(jnp.zeros((8, 16)), name="u"),
        v=Field(jnp.zeros((8, 16)), name="v"),
        h_s=Field(jnp.zeros((8, 16)), name="h_s"),
    )
    return model, state, dt


# ============================================================================
# 10a  JIT compilation of gradients
# ============================================================================

class TestJITGrad:

    def test_jit_grad_matches_grad(self, sw_latlon):
        model, state, dt = sw_latlon

        def loss(h_data):
            s = state._replace(h=state.h.replace(data=h_data))
            out = model.step(s, dt)
            return jnp.sum(out.h.data ** 2)

        grad_eager = jax.grad(loss)(state.h.data)
        grad_jit = jax.jit(jax.grad(loss))(state.h.data)

        assert jnp.allclose(grad_eager, grad_jit, rtol=1e-5), (
            "JIT grad differs from eager grad"
        )


# ============================================================================
# 10b  vmap over ensemble members
# ============================================================================

class TestVmapGrad:

    def test_vmap_grad(self, sw_latlon):
        model, state, dt = sw_latlon
        n_ens = 3

        def loss(h_data):
            s = state._replace(h=state.h.replace(data=h_data))
            out = model.step(s, dt)
            return jnp.sum(out.h.data ** 2)

        grad_fn = jax.vmap(jax.grad(loss))

        # Create batch of initial conditions
        key = jax.random.PRNGKey(10)
        batch_h = jnp.stack([
            state.h.data + jax.random.normal(jax.random.PRNGKey(i), state.h.data.shape)
            for i in range(n_ens)
        ])

        grads = grad_fn(batch_h)
        assert grads.shape == (n_ens, 8, 16), f"vmap grad shape: {grads.shape}"
        assert jnp.all(jnp.isfinite(grads)), "vmap grad has NaN/Inf"


# ============================================================================
# 10c  lax.scan gradient accumulation
# ============================================================================

class TestScanGrad:

    @pytest.mark.parametrize("n_steps", [1, 5, 20])
    def test_scan_grad_finite(self, sw_latlon, n_steps):
        model, state, dt = sw_latlon

        def loss(h_data):
            s = state._replace(h=state.h.replace(data=h_data))
            def body(carry, _):
                return model.step(carry, dt), None
            s_final, _ = jax.lax.scan(body, s, None, length=n_steps)
            return jnp.sum(s_final.h.data ** 2)

        grad = jax.grad(loss)(state.h.data)
        assert jnp.all(jnp.isfinite(grad)), f"Scan grad NaN/Inf at N={n_steps}"
        assert jnp.any(grad != 0), f"Scan grad all zero at N={n_steps}"


# ============================================================================
# 10d  Gradient checkpointing
# ============================================================================

class TestCheckpointGrad:

    def test_checkpoint_matches(self, sw_latlon):
        model, state, dt = sw_latlon
        n_steps = 5

        def loss_no_ckpt(h_data):
            s = state._replace(h=state.h.replace(data=h_data))
            def body(carry, _):
                return model.step(carry, dt), None
            s_final, _ = jax.lax.scan(body, s, None, length=n_steps)
            return jnp.sum(s_final.h.data ** 2)

        def loss_ckpt(h_data):
            s = state._replace(h=state.h.replace(data=h_data))
            @jax.checkpoint
            def step_ckpt(carry, _):
                return model.step(carry, dt), None
            s_final, _ = jax.lax.scan(step_ckpt, s, None, length=n_steps)
            return jnp.sum(s_final.h.data ** 2)

        grad_no = jax.grad(loss_no_ckpt)(state.h.data)
        grad_ck = jax.grad(loss_ckpt)(state.h.data)

        assert jnp.allclose(grad_no, grad_ck, rtol=1e-4), (
            "Checkpoint gradient differs from non-checkpoint"
        )


# ============================================================================
# 10e  Mixed precision (float64 gradients)
# ============================================================================

class TestMixedPrecision:

    def test_float64_grad(self, sw_latlon):
        """Verify gradients in float64 are finite and non-zero."""
        model, state, dt = sw_latlon

        def loss(h_data):
            s = state._replace(h=state.h.replace(data=h_data))
            out = model.step(s, dt)
            return jnp.sum(out.h.data ** 2)

        h64 = state.h.data.astype(jnp.float64)
        grad = jax.grad(loss)(h64)
        assert grad.dtype == jnp.float64, f"Expected float64, got {grad.dtype}"
        assert_gradient_ok(grad, "float64 gradient")
