"""Regression guard for the RRTMGP g-point accumulation strategies.

``RRTMGPConfig.gpoint_batch_size`` defaults to 16 (the vmap-block path) instead
of the legacy per-g-point scan (0).  These tests pin the two contracts that the
default change relies on:

1. the block path is answer-equivalent to the scan path (only summation
   re-association differs), and
2. the block path is reverse-mode-AD safe (finite gradients) when checkpointed
   — the property that lets it be the training default.

They exercise ``_accumulate_over_gpoints`` directly with a synthetic per-g-point
``step_fn`` so no gas-optics tables / GPU are needed.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from legoesm.atmosphere.physics.radiation.rrtmgp.rte.two_stream import (
    _accumulate_over_gpoints,
)


def _make_step_fn(weights):
    """step_fn(igpt, carry) = carry + weights[igpt] * (1, 2) — additive per the
    accumulator contract (``step_fn(ig, zeros)`` == that g-point's flux)."""
    def step_fn(igpt, carry):
        w = weights[igpt]
        return jax.tree.map(lambda c, m: c + w * m, carry,
                            {"a": jnp.asarray(1.0), "b": jnp.asarray(2.0)})
    return step_fn


def _init():
    return {"a": jnp.asarray(0.0), "b": jnp.asarray(0.0)}


@pytest.mark.parametrize("n_gpt", [256, 224, 250])  # divisible + non-divisible
def test_block_matches_scan(n_gpt):
    weights = jnp.sin(jnp.arange(n_gpt, dtype=jnp.float64) * 0.1) + 1.5
    step = _make_step_fn(weights)
    scan = _accumulate_over_gpoints(step, n_gpt, _init(), 0, True)
    block = _accumulate_over_gpoints(step, n_gpt, _init(), 16, True)
    # Answer-equivalent up to float re-association; and equal to the closed form.
    total = float(jnp.sum(weights))
    for key, mult in (("a", 1.0), ("b", 2.0)):
        assert jnp.allclose(scan[key], block[key], rtol=1e-10, atol=1e-8)
        assert jnp.allclose(block[key], total * mult, rtol=1e-10, atol=1e-8)


def test_non_divisible_drops_padding():
    # 250 g-points, block 16 -> 16 blocks, last block has 6 padding lanes that
    # must NOT contribute (masked). If padding leaked, the sum would overshoot.
    n_gpt = 250
    weights = jnp.ones(n_gpt, dtype=jnp.float64)
    block = _accumulate_over_gpoints(_make_step_fn(weights), n_gpt, _init(), 16, True)
    assert jnp.allclose(block["a"], float(n_gpt), rtol=0, atol=1e-9)


def test_block_path_is_reverse_ad_safe():
    # grad of a scalar of the accumulated flux w.r.t. the per-g-point weights,
    # through the checkpointed block scan, must be finite and correct (d/dw a = 1).
    n_gpt = 256
    def loss(weights):
        out = _accumulate_over_gpoints(
            _make_step_fn(weights), n_gpt, _init(), 16, True)
        return out["a"] + out["b"]
    w = jnp.linspace(0.5, 2.0, n_gpt)
    g = jax.grad(loss)(w)
    assert bool(jnp.all(jnp.isfinite(g))), "block-path gradient not finite"
    # da/dw_i = 1, db/dw_i = 2 -> d(a+b)/dw_i = 3 for every g-point.
    assert jnp.allclose(g, 3.0, rtol=1e-10, atol=1e-8)
