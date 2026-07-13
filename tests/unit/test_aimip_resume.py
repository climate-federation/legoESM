"""Tests for AIMIP epoch-resume + mid-epoch (per-chunk) resume helpers.

Validates ``find_latest_epoch_checkpoint`` and ``maybe_resume_model``
in ``legoesm.training.neural_gcm_spectral`` -- the two functions that
back ``scripts/run/run_aimip.py --resume`` for chained-resubmission SLURM
training.

These helpers underpin the per-variant continue-from-walltime-kill
behavior used by ``scripts/run_aimip_headtohead_t106_*.sbatch``.  If
they regress, a 12-hour-walltime job that survives long enough to
write ``epoch_NNNN.eqx`` would still restart from epoch 0 on its next
link, wasting compute.

Also validates the mid-epoch (per-chunk) checkpoint (#942):
``_save_midepoch_checkpoint`` / ``_load_midepoch_checkpoint`` (model +
optimizer state + resume position round-trip) and the atomic-write
guarantee of ``save_checkpoint`` (a torn write can never corrupt the
prior checkpoint).  The end-to-end "resume mid-epoch reproduces the
uninterrupted run bit-for-bit" proof lives in
``tests/unit/test_neural_gcm_spectral.py::TestMidEpochResume``.
"""

from __future__ import annotations

from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import pytest

from legoesm.ml.training import save_checkpoint
from legoesm.training.neural_gcm_spectral import (
    find_latest_epoch_checkpoint,
    maybe_resume_model,
    _save_midepoch_checkpoint,
    _load_midepoch_checkpoint,
    MIDEPOCH_CHECKPOINT_NAME,
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


# ===========================================================================
# Mid-epoch (per-chunk) checkpoint helpers (#942)
# ===========================================================================


def _stepped_opt_state(model, opt):
    """An optimizer state that has actually taken a step, so its moments
    and step counter are non-trivial (a fresh ``init`` state would round-
    trip even if the code dropped everything but structure)."""
    grads = eqx.filter(model, eqx.is_array)  # reuse weights as fake grads
    opt_state = opt.init(grads)
    updates, opt_state = opt.update(grads, opt_state, grads)
    stepped_model = eqx.apply_updates(model, updates)
    return stepped_model, opt_state


def _leaves_equal(a, b) -> bool:
    la = jax.tree_util.tree_leaves(a)
    lb = jax.tree_util.tree_leaves(b)
    return len(la) == len(lb) and all(
        jnp.array_equal(x, y) for x, y in zip(la, lb)
    )


def test_load_midepoch_returns_none_when_absent(tmp_path):
    assert _load_midepoch_checkpoint(None, None, None) is None
    (tmp_path / "ck").mkdir()
    tmpl = _TinyModel(jax.random.PRNGKey(0))
    opt = optax.adam(1e-3)
    assert _load_midepoch_checkpoint(
        tmp_path / "ck", tmpl, opt.init(eqx.filter(tmpl, eqx.is_array))
    ) is None


def test_midepoch_roundtrip_preserves_model_optstate_and_position(tmp_path):
    """The mid-epoch checkpoint must restore EXACTLY: model weights, the
    full optimizer state (Adam moments + step count), and the resume
    position (epoch, next_chunk).  Missing optimizer state is precisely
    what makes the epoch-only resume restart the LR schedule (#942)."""
    ck = tmp_path / "ck"
    ck.mkdir()
    opt = optax.adam(1e-3)
    model, opt_state = _stepped_opt_state(_TinyModel(jax.random.PRNGKey(1)), opt)

    _save_midepoch_checkpoint(ck, model, opt_state, epoch=2, next_chunk=3)

    # File landed; no temp file left behind (atomic os.replace).
    assert (ck / MIDEPOCH_CHECKPOINT_NAME).exists()
    assert not list(ck.glob("*.tmp-*"))

    # Deserialise into FRESH templates (different seed) so a pass proves
    # the bytes came from disk, not the template.
    tmpl = _TinyModel(jax.random.PRNGKey(999))
    tmpl_opt_state = opt.init(eqx.filter(tmpl, eqx.is_array))
    loaded = _load_midepoch_checkpoint(ck, tmpl, tmpl_opt_state)
    assert loaded is not None
    lm, lo, epoch, next_chunk = loaded

    assert (epoch, next_chunk) == (2, 3)
    assert jnp.array_equal(lm.w, model.w) and jnp.array_equal(lm.b, model.b)
    assert not jnp.array_equal(lm.w, tmpl.w)          # not the template
    assert _leaves_equal(lo, opt_state)               # moments + count restored
    assert not _leaves_equal(lo, tmpl_opt_state)      # not the fresh template


def test_save_checkpoint_is_atomic_under_torn_write(tmp_path, monkeypatch):
    """A walltime kill mid-serialise must never corrupt the prior
    checkpoint: ``save_checkpoint`` writes a temp file then os.replace,
    so a crash leaves the old file intact and no partial temp behind."""
    from legoesm.ml import training as tr

    ck = tmp_path / "ck"
    ck.mkdir()
    good = _TinyModel(jax.random.PRNGKey(3))
    path = ck / "epoch_0000.eqx"
    save_checkpoint(good, path)
    before = path.read_bytes()

    def _torn_write(p, tree):
        # Simulate a kill after a partial temp write.
        with open(p, "wb") as fh:
            fh.write(b"partial-garbage")
        raise RuntimeError("killed during write")

    monkeypatch.setattr(tr.eqx, "tree_serialise_leaves", _torn_write)
    with pytest.raises(RuntimeError, match="killed during write"):
        save_checkpoint(_TinyModel(jax.random.PRNGKey(4)), path)

    # The previous complete checkpoint is untouched, and the partial temp
    # was cleaned up (never left where a reader could mistake it for real).
    assert path.read_bytes() == before
    assert not list(ck.glob("*.tmp-*"))
