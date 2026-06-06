"""Tests for AIMIP epoch-resume helpers.

Validates ``find_latest_epoch_checkpoint`` and ``maybe_resume_model``
in ``legoesm.training.neural_gcm_spectral`` -- the two functions that
back ``scripts/run/run_aimip.py --resume`` for chained-resubmission SLURM
training.

These helpers underpin the per-variant continue-from-walltime-kill
behavior used by ``scripts/run_aimip_headtohead_t106_*.sbatch``.  If
they regress, a 12-hour-walltime job that survives long enough to
write ``epoch_NNNN.eqx`` would still restart from epoch 0 on its next
link, wasting compute.
"""

from __future__ import annotations

from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp

from legoesm.ml.training import save_checkpoint
from legoesm.training.neural_gcm_spectral import (
    find_latest_epoch_checkpoint,
    maybe_resume_model,
)


class _TinyModel(eqx.Module):
    """Minimal pytree for round-trip checkpoint testing."""

    w: jnp.ndarray
    b: jnp.ndarray

    def __init__(self, key, dim=4):
        kw, kb = jax.random.split(key)
        self.w = jax.random.normal(kw, (dim, dim))
        self.b = jax.random.normal(kb, (dim,))


def test_find_latest_returns_none_for_missing_dir(tmp_path):
    assert find_latest_epoch_checkpoint(tmp_path / "does-not-exist") is None


def test_find_latest_returns_none_for_empty_dir(tmp_path):
    (tmp_path / "ckpts").mkdir()
    assert find_latest_epoch_checkpoint(tmp_path / "ckpts") is None


def test_find_latest_picks_highest_numbered(tmp_path):
    ck = tmp_path / "ckpts"
    ck.mkdir()
    # Touch a handful of dummy checkpoints in non-monotonic order.
    for ep in (0, 7, 2, 12, 5):
        (ck / f"epoch_{ep:04d}.eqx").write_bytes(b"")
    # Throw in a non-conforming file that must be ignored.
    (ck / "epoch_notanumber.eqx").write_bytes(b"")
    (ck / "params.eqx").write_bytes(b"")

    result = find_latest_epoch_checkpoint(ck)
    assert result is not None
    epoch, path = result
    assert epoch == 12
    assert path.name == "epoch_0012.eqx"


def test_maybe_resume_returns_template_when_no_dir():
    template = _TinyModel(jax.random.PRNGKey(0))
    model, start_epoch = maybe_resume_model(template, None)
    assert model is template
    assert start_epoch == 0


def test_maybe_resume_returns_template_when_no_checkpoints(tmp_path):
    (tmp_path / "ckpts").mkdir()
    template = _TinyModel(jax.random.PRNGKey(0))
    model, start_epoch = maybe_resume_model(template, tmp_path / "ckpts")
    assert model is template
    assert start_epoch == 0


def test_maybe_resume_loads_and_advances_epoch(tmp_path):
    """End-to-end: save a checkpoint, then resume into a fresh template.

    The loaded model must carry the saved weights, and start_epoch
    must be ``last_saved + 1``.
    """
    ck = tmp_path / "ckpts"
    ck.mkdir()

    saved = _TinyModel(jax.random.PRNGKey(42))
    save_checkpoint(saved, ck / "epoch_0003.eqx")

    fresh = _TinyModel(jax.random.PRNGKey(0))
    loaded, start_epoch = maybe_resume_model(fresh, ck)

    assert start_epoch == 4
    # Weights must match the saved model, not the fresh template.
    assert jnp.allclose(loaded.w, saved.w)
    assert jnp.allclose(loaded.b, saved.b)
    assert not jnp.allclose(loaded.w, fresh.w)
