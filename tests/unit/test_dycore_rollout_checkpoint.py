"""Dispatch/validation tests for the new RolloutConfig checkpoint knobs.

The gradient-identity behaviour of the schedules themselves is covered by
tests/unit/test_checkpoint_schedule.py and tests/da/test_cost_function.py; here
we only assert that differentiable_rollout validates its new fields and refuses
the trajectory-incompatible 'binomial' schedule loudly (dispatch hardening).
"""
import pytest
from legoesm.training.dycore_rollout import RolloutConfig, differentiable_rollout


def _noop_segment(carry, n_steps, forcing):  # never reached before the guards
    return carry


def test_rollout_rejects_binomial_schedule():
    cfg = RolloutConfig(n_days=1, checkpoint_schedule="binomial")
    with pytest.raises(ValueError, match="binomial"):
        differentiable_rollout(None, None, _noop_segment, cfg)


def test_rollout_rejects_unknown_schedule():
    cfg = RolloutConfig(n_days=1, checkpoint_schedule="bogus")
    with pytest.raises(ValueError, match="not supported by"):
        differentiable_rollout(None, None, _noop_segment, cfg)


def test_rollout_rejects_unknown_storage():
    cfg = RolloutConfig(n_days=1, storage="bogus")
    with pytest.raises(ValueError, match="unknown storage"):
        differentiable_rollout(None, None, _noop_segment, cfg)


def test_rollout_rejects_host_storage_when_checkpoint_disabled():
    """gradient_checkpoint=False must truly disable checkpointing — host storage
    cannot silently re-enable a jax.checkpoint wrap (codex round-3)."""
    cfg = RolloutConfig(n_days=1, gradient_checkpoint=False, storage="host")
    with pytest.raises(ValueError, match="requires gradient"):
        differentiable_rollout(None, None, _noop_segment, cfg)


@pytest.mark.parametrize("bad", ["binomial", "bogus"])
def test_rollout_rejects_bad_schedule_even_when_checkpoint_disabled(bad):
    """An invalid/unsupported checkpoint_schedule must be rejected regardless of
    gradient_checkpoint (codex round-6): with gradient_checkpoint=False the value
    would otherwise be silently coerced to 'none'."""
    cfg = RolloutConfig(n_days=1, gradient_checkpoint=False, checkpoint_schedule=bad)
    with pytest.raises(ValueError, match="not supported"):
        differentiable_rollout(None, None, _noop_segment, cfg)


def test_rollout_config_defaults():
    cfg = RolloutConfig()
    assert cfg.checkpoint_schedule == "uniform"
    assert cfg.storage == "recompute"
    assert cfg.gradient_checkpoint is True
