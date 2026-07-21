"""Unit tests for training EMA (packages/ml/legoesm/training/ema.py)."""

import equinox as eqx
import jax.numpy as jnp
import pytest

from legoesm.training.ema import EMA_DECAY, ema_update, init_ema


class _TinyModel(eqx.Module):
    w: jnp.ndarray
    b: jnp.ndarray
    n_static: int  # static leaf: must never be blended

    def __init__(self, w, b, n_static=3):
        self.w = jnp.asarray(w, dtype=jnp.float64)
        self.b = jnp.asarray(b, dtype=jnp.float64)
        self.n_static = n_static


def test_init_ema_is_identity_copy():
    m = _TinyModel([1.0, 2.0], [0.5])
    ema = init_ema(m)
    assert jnp.array_equal(ema.w, m.w)
    assert ema.n_static == 3


def test_ema_update_exact_convex_combination():
    ema = _TinyModel([1.0, 1.0], [1.0])
    m = _TinyModel([2.0, 0.0], [3.0])
    d = 0.9
    out = ema_update(ema, m, decay=d)
    assert jnp.allclose(out.w, jnp.array([d * 1.0 + 0.1 * 2.0, d * 1.0]))
    assert jnp.allclose(out.b, jnp.array([d * 1.0 + 0.1 * 3.0]))


def test_ema_fixed_point_when_equal():
    m = _TinyModel([1.5, -2.5], [0.25])
    out = ema_update(init_ema(m), m, decay=EMA_DECAY)
    assert jnp.allclose(out.w, m.w)
    assert jnp.allclose(out.b, m.b)


def test_static_leaves_untouched():
    ema = _TinyModel([1.0], [1.0], n_static=7)
    m = _TinyModel([2.0], [2.0], n_static=9)
    out = ema_update(ema, m, decay=0.5)
    assert out.n_static == 7  # taken from the EMA model, never blended


def test_default_decay_is_ucast_convention():
    assert EMA_DECAY == 0.9999


@pytest.mark.parametrize("bad", [-0.1, 1.0, 1.5])
def test_rejects_bad_decay(bad):
    m = _TinyModel([1.0], [1.0])
    with pytest.raises(ValueError, match="EMA decay"):
        ema_update(m, m, decay=bad)


def test_many_updates_converge_toward_model():
    ema = _TinyModel([0.0], [0.0])
    m = _TinyModel([1.0], [1.0])
    for _ in range(200):
        ema = ema_update(ema, m, decay=0.97)
    # 1 - 0.97^200 ~ 0.9977
    assert float(ema.w[0]) == pytest.approx(1.0 - 0.97**200, rel=1e-9)
