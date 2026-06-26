"""Direct test for ``training_driver.multi_step_rollout_loss``.

The non-trivial logic is the TRUNCATED backprop-through-time: the forward
chains autoregressively across segments, but ``stop_gradient`` between
segments means each loss term's gradient flows ONLY through its own
segment (the lat-lon adjoint NaNs past ~6h, so full BPTT is unusable).
This test pins that gradient property with a closed-form stub rollout,
plus the single-step equivalence and the validation guards.

``single_day_rollout`` and ``combined_loss`` are module-level names in
``training_driver``, so they are monkeypatched to a closed-form
``state -> mult*state`` rollout and a ``sum(pred)`` loss — no model,
grid, or ERA5 needed.
"""

import jax
import jax.numpy as jnp
import pytest

from legoesm.training import training_driver as td
from legoesm.training.losses import LossConfig


def _patch(monkeypatch, mult=2.0):
    monkeypatch.setattr(
        td, "single_day_rollout",
        lambda state, forcing, seg, dt, hours: state * mult)
    monkeypatch.setattr(
        td, "combined_loss",
        lambda pred, target, sigma_full, grid, config: jnp.sum(pred))


def test_single_step_path(monkeypatch):
    """Empty multi_step_hours -> one rollout vs a single target carry."""
    _patch(monkeypatch, mult=2.0)
    cfg = LossConfig(multi_step_hours=(), multi_step_weights=())

    def f(ic):
        return td.multi_step_rollout_loss(
            ic, None, None, dt=600.0, rollout_hours=6, target=None,
            sigma_full=None, grid=None, loss_config=cfg)

    ic = jnp.asarray(3.0)
    assert float(f(ic)) == pytest.approx(6.0)        # 3 * 2
    assert float(jax.grad(f)(ic)) == pytest.approx(2.0)


def test_multi_step_truncated_vs_full_grad(monkeypatch):
    """Forward chains (state1=2, state2=4); truncated grad sees only the
    first segment's ic-dependence, full BPTT sees both."""
    _patch(monkeypatch, mult=2.0)
    cfg = LossConfig(multi_step_hours=(6, 12), multi_step_weights=(1.0, 1.0))
    target = (None, None)  # one per lead

    def f(ic, trunc):
        return td.multi_step_rollout_loss(
            ic, None, None, dt=600.0, rollout_hours=6, target=target,
            sigma_full=None, grid=None, loss_config=cfg, truncated_bptt=trunc)

    ic = jnp.asarray(1.0)
    # total = (w0*state1 + w1*state2)/wsum = (1*2 + 1*4)/2 = 3
    assert float(f(ic, True)) == pytest.approx(3.0)
    assert float(f(ic, False)) == pytest.approx(3.0)   # forward identical
    # Truncated: only seg-0 term depends on ic -> d/dic = w0*2 / wsum = 1.
    assert float(jax.grad(lambda x: f(x, True))(ic)) == pytest.approx(1.0)
    # Full BPTT: (w0*2 + w1*4)/2 = 3.
    assert float(jax.grad(lambda x: f(x, False))(ic)) == pytest.approx(3.0)


def test_weights_default_uniform(monkeypatch):
    """Empty multi_step_weights -> uniform, normalised by the weight sum."""
    _patch(monkeypatch, mult=3.0)
    cfg = LossConfig(multi_step_hours=(6, 12), multi_step_weights=())
    target = (None, None)
    ic = jnp.asarray(1.0)
    # state1=3, state2=9; uniform weights -> (3+9)/2 = 6.
    val = td.multi_step_rollout_loss(
        ic, None, None, dt=600.0, rollout_hours=6, target=target,
        sigma_full=None, grid=None, loss_config=cfg)
    assert float(val) == pytest.approx(6.0)


def test_target_length_mismatch_raises(monkeypatch):
    _patch(monkeypatch)
    cfg = LossConfig(multi_step_hours=(6, 12, 18), multi_step_weights=())
    with pytest.raises(ValueError, match="target carries"):
        td.multi_step_rollout_loss(
            jnp.asarray(1.0), None, None, dt=600.0, rollout_hours=6,
            target=(None, None), sigma_full=None, grid=None, loss_config=cfg)


def test_weights_length_mismatch_raises(monkeypatch):
    _patch(monkeypatch)
    cfg = LossConfig(multi_step_hours=(6, 12), multi_step_weights=(1.0,))
    with pytest.raises(ValueError, match="multi_step_weights length"):
        td.multi_step_rollout_loss(
            jnp.asarray(1.0), None, None, dt=600.0, rollout_hours=6,
            target=(None, None), sigma_full=None, grid=None, loss_config=cfg)


def test_non_increasing_leads_raise(monkeypatch):
    _patch(monkeypatch)
    # Duplicate lead -> zero-length segment -> error.
    cfg = LossConfig(multi_step_hours=(6, 6), multi_step_weights=())
    with pytest.raises(ValueError, match="strictly increasing"):
        td.multi_step_rollout_loss(
            jnp.asarray(1.0), None, None, dt=600.0, rollout_hours=6,
            target=(None, None), sigma_full=None, grid=None, loss_config=cfg)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
