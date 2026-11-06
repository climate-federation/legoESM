"""EMA of the trained weights in the WeatherBench trainer (deck ``ema_decay``).

The loop keeps ``ema <- d * ema + (1 - d) * params`` after every ACCEPTED
step; a mid-epoch save carries the EMA and a resume continues it exactly.
"""
from __future__ import annotations

import ast
import pathlib

import pytest
import yaml

import scripts.run.train_weatherbench_scale as mod
from tests.unit.test_wb_train_resume import (
    _FP,
    _content,
    _load,
    _manifest,
    _mid_epoch_setup,
)

D = 0.9


def _run(samples, n_epochs, loss_fn, opt, p0, ema, **kw):
    from legoesm.training.data_parallel import mpi_data_parallel_training_loop

    seen = []

    def on_step(e, n, losses, skipped, p, s):
        seen.append((float(p["w"]), float(ema["tree"]["w"])))
    out = mpi_data_parallel_training_loop(
        loss_fn, p0, opt.init(p0), opt, samples, n_epochs, 1, on_step=on_step,
        ema=ema, ema_decay=D, **kw)
    return out, seen


def test_the_ema_follows_the_recurrence_over_accepted_steps():
    samples, n_epochs, loss_fn, fresh, p0 = _mid_epoch_setup()
    ema = {"tree": p0}
    _, seen = _run(samples, n_epochs, loss_fn, fresh(), p0, ema)
    want = float(p0["w"])
    for w, e in seen:
        want = D * want + (1.0 - D) * w
        assert e == pytest.approx(want, rel=1e-12)
    assert seen[-1][1] != seen[-1][0]      # an average, not the parameters


def test_a_skipped_step_leaves_the_ema_alone():
    import jax.numpy as jnp

    samples, _n, loss_fn, fresh, p0 = _mid_epoch_setup()
    samples = samples[:2] + [jnp.asarray(jnp.nan)] + samples[2:]
    ema = {"tree": p0}
    _, seen = _run(samples, 1, loss_fn, fresh(), p0, ema)
    assert seen[2] == seen[1]              # the NaN step moved neither


def test_a_run_cut_mid_epoch_continues_the_same_ema():
    from legoesm.training.data_parallel import mpi_data_parallel_training_loop

    samples, n_epochs, loss_fn, fresh, p0 = _mid_epoch_setup()
    straight = {"tree": p0}
    _run(samples, n_epochs, loss_fn, fresh(), p0, straight)

    got, cut = {}, {"tree": p0}

    def on_step(e, n, losses, skipped, p, s):
        if (e, n) == (1, 2):
            got.update(p=p, s=s, losses=list(losses), skipped=skipped,
                       ema=cut["tree"])
    opt = fresh()
    mpi_data_parallel_training_loop(loss_fn, p0, opt.init(p0), opt, samples, 2,
                                    1, on_step=on_step, ema=cut, ema_decay=D)
    resumed = {"tree": got["ema"]}
    mpi_data_parallel_training_loop(
        loss_fn, got["p"], got["s"], opt, samples, n_epochs, 1, start_epoch=1,
        start_step=2, start_losses=got["losses"], start_skipped=got["skipped"],
        ema=resumed, ema_decay=D)
    assert float(resumed["tree"]["w"]) == float(straight["tree"]["w"])


def _save(d, value, ema=None, **kw):
    import equinox as eqx
    import jax.numpy as jnp

    tree = {"w": jnp.asarray(float(value))}
    etree = None if ema is None else {"w": jnp.asarray(float(ema))}
    mod._save_partial(
        str(d), _manifest(**kw),
        lambda t: eqx.tree_serialise_leaves(t, tree),
        lambda t: eqx.tree_serialise_leaves(t, tree),
        None if etree is None else (lambda t: eqx.tree_serialise_leaves(t, etree)))


def test_a_partial_carries_the_ema_and_keeps_one_generation(tmp_path):
    _save(tmp_path, 1.5, ema=0.5, step=3)
    _save(tmp_path, 2.5, ema=0.7, step=5, losses=(0.1, 0.2, 0.3, 0.4), skipped=1)
    obj, _pp, _op = mod._read_partial(str(tmp_path), 2, 10, _FP, _content())
    assert _load(str(tmp_path / obj["ema_file"])) == 0.7
    assert len(list(tmp_path.glob("partial.e*.eqx"))) == 3   # params, opt, ema


def test_a_partial_missing_its_ema_file_fails_loudly(tmp_path):
    _save(tmp_path, 1.5, ema=0.5)
    for f in tmp_path.glob("partial.e*.ema.eqx"):
        f.unlink()
    with pytest.raises(RuntimeError):
        mod._read_partial(str(tmp_path), 2, 10, _FP, _content())


def test_the_epoch_ema_file_is_the_campaign_s_ema_sibling(tmp_path):
    from legoesm.training.campaign_driver import latest_checkpoint

    pp = pathlib.Path(mod._checkpoint_paths(str(tmp_path), 3)[0])
    pp.write_bytes(b"x")
    pathlib.Path(mod._ema_path(str(tmp_path), 3)).write_bytes(b"y")
    assert mod._latest_complete_checkpoint(str(tmp_path)) is None  # no manifest
    assert latest_checkpoint(tmp_path, prefer_ema=True) == mod._ema_path(
        str(tmp_path), 3)


@pytest.mark.parametrize("v,ok", [(0.995, True), (0, True), (1.0, False),
                                  (-0.1, False), (True, False), ("0.9", False)])
def test_the_deck_decay_is_validated(v, ok):
    from legoesm.training.scale_build import wb_ema_decay

    if ok:
        assert wb_ema_decay({"ema_decay": v}) == float(v)
    else:
        with pytest.raises(ValueError):
            wb_ema_decay({"ema_decay": v})


def test_the_trainer_hands_the_ema_to_every_training_loop():
    src = pathlib.Path(mod.__file__).read_text()
    calls = [n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Call)
             and getattr(n.func, "id", None) == "mpi_data_parallel_training_loop"]
    assert calls and all({"ema", "ema_decay"} <= {k.arg for k in c.keywords}
                         for c in calls)


@pytest.mark.parametrize("deck,decay", [
    ("spectral_t63_ngcm_v9", 0.995), ("spectral_t63_ngcm_nolearnedembed_v9", 0.995),
    ("spectral_t63_sfno_v9", 0.995), ("spectral_t63_sfno_nolearnedembed_v9", 0.995),
    ("spectral_t63_cam6_tier3_v9", 0.0)])
def test_the_v9_decks_state_their_ema(deck, decay):
    p = pathlib.Path(mod.__file__).parents[2] / "config/wb/campaign" / f"{deck}.yaml"
    assert yaml.safe_load(p.read_text())["ema_decay"] == decay


@pytest.mark.parametrize("decay,ckpt", [(0.995, "epoch_NNNN_ema.eqx"),
                                        (0.0, "epoch_NNNN.eqx")])
def test_the_wb2_command_scores_the_ema_when_the_run_keeps_one(decay, ckpt, caplog):
    import logging
    from types import SimpleNamespace

    from legoesm.training.scale_build import evaluate_wb2

    cfg = SimpleNamespace(training_core="spectral", out_dir="/o",
                          config_path="d.yaml", mode="neural_gcm")
    with caplog.at_level(logging.INFO, logger="wb_scale"):
        evaluate_wb2(cfg, {"ema_decay": decay, "eval_years": [2020]},
                     None, None, None, None, None)
    assert f"--checkpoint /o/{ckpt} " in caplog.text
