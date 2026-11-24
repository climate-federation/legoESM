"""Dycore options (pressure-surface T damping) recorded with every training
checkpoint and checked on resume, fine-tune and inference.

A checkpoint written before the record existed counts as all-off and loads
bit-identically; a mismatch with the run's options is refused.  The AIMIP
ablation drivers, which do not honour the options, refuse the keys.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
import yaml

_ROOT = Path(__file__).resolve().parents[2]
_LL, _SP = "latlon_ah_pressure_surfaces", "spectral_t_damping_pressure_surfaces"
OFF = {_LL: False, _SP: False}
ON = {_LL: False, _SP: True}


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, _ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _model(seed=0.0):
    return {"w": jnp.arange(6.0).reshape(2, 3) + seed, "b": jnp.ones(3) * seed}


def _bytes(tree):
    import jax
    return [np.asarray(x).tobytes() for x in jax.tree.leaves(tree)]


def test_record_round_trip_and_mismatch(tmp_path):
    from legoesm.ml.checkpoint_io import (
        CheckpointDycoreMismatch, check_dycore_record, dycore_record_path,
        read_dycore_record,
    )
    from legoesm.ml.training import save_checkpoint
    p = tmp_path / "epoch_0000.eqx"
    save_checkpoint(_model(), p, dycore_options=ON)
    assert dycore_record_path(p).exists() and read_dycore_record(p) == ON
    check_dycore_record(p, ON)
    with pytest.raises(CheckpointDycoreMismatch,
                       match=rf"{_SP}: checkpoint True, this run False"):
        check_dycore_record(p, OFF, what="resume")
    # a later save without options removes the stale record (= all off)
    save_checkpoint(_model(), p)
    assert not dycore_record_path(p).exists() and read_dycore_record(p) == OFF
    with pytest.raises(ValueError, match=".eqx path"):
        save_checkpoint(_model(), tmp_path / "params", dycore_options=ON)
    # a suffix-less None save does not touch the separate .eqx's record
    q = tmp_path / "params.eqx"
    save_checkpoint(_model(), q, dycore_options=ON)
    save_checkpoint(_model(9.0), tmp_path / "params")
    assert read_dycore_record(q) == ON
    # a rejected save (bad options) leaves payload and record untouched
    save_checkpoint(_model(5.0), p, dycore_options=ON)
    before = (p.read_bytes(), dycore_record_path(p).read_bytes())
    with pytest.raises(ValueError, match="as bools"):
        save_checkpoint(_model(), p, dycore_options={_SP: "true", _LL: False})
    assert (p.read_bytes(), dycore_record_path(p).read_bytes()) == before


@pytest.mark.parametrize("kill_at", ["payload_rename", "commit"])
@pytest.mark.parametrize("same_weights", [False, True])
def test_interrupted_save_is_refused_not_read_as_off(tmp_path, monkeypatch,
                                                     kill_at, same_weights):
    """Kill the save of an ON checkpoint over an OFF one, before the payload
    rename or before the record commit (also with byte-identical weights, the
    relabelling case): every reader refuses instead of trusting the record or
    reading the payload as all off."""
    import os
    from legoesm.ml import checkpoint_io
    from legoesm.ml.checkpoint_io import CheckpointStructureError, check_dycore_record
    from legoesm.ml.training import save_checkpoint
    p = tmp_path / "epoch_0000.eqx"
    save_checkpoint(_model(1.0), p, dycore_options=OFF)
    real_replace, real_write = os.replace, checkpoint_io.write_dycore_record

    def _replace(src, dst):   # records publish; the payload rename dies
        if kill_at == "payload_rename" and not str(dst).endswith(".json"):
            raise KeyboardInterrupt("killed before the payload rename")
        return real_replace(src, dst)

    def _write(path, options, *, committed=True, **kw):
        if kill_at == "commit" and committed:
            raise KeyboardInterrupt("killed before the record commit")
        return real_write(path, options, committed=committed, **kw)
    monkeypatch.setattr(os, "replace", _replace)
    monkeypatch.setattr(checkpoint_io, "write_dycore_record", _write)
    with pytest.raises(KeyboardInterrupt):
        save_checkpoint(_model(1.0 if same_weights else 2.0), p,
                        dycore_options=ON)
    monkeypatch.undo()
    for opts in (OFF, ON):
        with pytest.raises(CheckpointStructureError, match="interrupted save"):
            check_dycore_record(p, opts)


def test_load_checked_refuses_before_loading(tmp_path):
    from legoesm.ml.checkpoint_io import CheckpointDycoreMismatch, load_checked
    from legoesm.ml.training import load_checkpoint, save_checkpoint
    p = tmp_path / "params.eqx"
    save_checkpoint(_model(3.0), p, dycore_options=ON)
    calls = []
    with pytest.raises(CheckpointDycoreMismatch):
        load_checked(p, OFF, lambda: calls.append(1))
    assert calls == []
    assert _bytes(load_checked(p, ON, lambda: load_checkpoint(_model(), p))) \
        == _bytes(_model(3.0))


def test_legacy_checkpoint_is_off_and_loads_bit_identically(tmp_path):
    """A checkpoint with no record (written by equinox directly, as before
    the record existed) reads as all off, passes an off run's check, loads
    byte-identically, and is refused by an on run."""
    from legoesm.ml.checkpoint_io import CheckpointDycoreMismatch, check_dycore_record
    from legoesm.ml.training import load_checkpoint
    p = tmp_path / "legacy.eqx"
    m = _model(3.0)
    eqx.tree_serialise_leaves(str(p), m)
    check_dycore_record(p, OFF)
    assert _bytes(load_checkpoint(_model(), p)) == _bytes(m)
    with pytest.raises(CheckpointDycoreMismatch):
        check_dycore_record(p, ON)


def test_spectral_resume_paths_check_the_record(tmp_path):
    from legoesm.ml.checkpoint_io import CheckpointDycoreMismatch
    from legoesm.ml.training import save_checkpoint
    from legoesm.training.neural_gcm_spectral import (
        _load_midepoch_checkpoint, _save_midepoch_checkpoint, maybe_resume_model,
    )
    on_dir, legacy_dir = tmp_path / "on", tmp_path / "legacy"
    save_checkpoint(_model(1.0), on_dir / "epoch_0000.eqx", dycore_options=ON)
    legacy_dir.mkdir()
    eqx.tree_serialise_leaves(str(legacy_dir / "epoch_0000.eqx"), _model(2.0))
    m, ep = maybe_resume_model(_model(), on_dir, dycore_options=ON)
    assert ep == 1 and _bytes(m) == _bytes(_model(1.0))
    with pytest.raises(CheckpointDycoreMismatch, match="resume"):
        maybe_resume_model(_model(), on_dir, dycore_options=OFF)
    m, _ = maybe_resume_model(_model(), legacy_dir, dycore_options=OFF)
    assert _bytes(m) == _bytes(_model(2.0))
    with pytest.raises(CheckpointDycoreMismatch):
        maybe_resume_model(_model(), legacy_dir, dycore_options=ON)
    # mid-epoch payload
    opt = {"m": jnp.zeros(3)}
    _save_midepoch_checkpoint(on_dir, _model(1.0), opt, 0, 1, dycore_options=ON)
    assert _load_midepoch_checkpoint(on_dir, _model(), opt,
                                     dycore_options=ON)[2:] == (0, 1)
    with pytest.raises(CheckpointDycoreMismatch, match="mid-epoch"):
        _load_midepoch_checkpoint(on_dir, _model(), opt, dycore_options=OFF)


def test_spectral_dycore_options():
    from legoesm.atmosphere.dynamics.gcm.spectral_pe import SpectralPEConfig
    from legoesm.training.neural_gcm_spectral import spectral_dycore_options
    assert spectral_dycore_options(SpectralPEConfig()) == OFF
    assert spectral_dycore_options(
        SpectralPEConfig(T_damping_pressure_surfaces=True)) == ON


def _suite(tmp_path, **base_extra):
    base = {"n_max": 21, "nlev": 8, "n_levels": 8, "dt": 1800.0,
            "aimip_radiation": "gray", "smoke": True,
            "aimip_convection": "tiedtke", "aimip_turbulence": "louis",
            **base_extra}
    (tmp_path / "base.yaml").write_text(yaml.safe_dump(base))
    suite = tmp_path / "suite.yaml"
    suite.write_text(yaml.safe_dump({"base": str(tmp_path / "base.yaml")}))
    for v in ("classical", "column_nn", "sfno_physics"):
        (tmp_path / f"variant_{v}.yaml").write_text("{}\n")
    return suite


@pytest.mark.parametrize("script, flag", [
    ("scripts/run/run_aimip_amip_finetune.py", "--init-ckpt"),
    ("scripts/run/run_aimip_amip_inference.py", "--ckpt"),
])
def test_finetune_and_inference_refuse_a_mismatched_checkpoint(
        tmp_path, monkeypatch, script, flag):
    """The check runs before any data is touched: an ON checkpoint under an
    OFF suite (and the reverse) is refused with the mismatch error."""
    from legoesm.ml.checkpoint_io import CheckpointDycoreMismatch
    from legoesm.ml.training import save_checkpoint
    mod = _load("_ckdy_" + Path(script).stem, script)
    ck = tmp_path / "params.eqx"
    for rec, suite_extra in ((ON, {}), (OFF, {_SP: True})):
        save_checkpoint(_model(), ck, dycore_options=rec)
        suite = _suite(tmp_path, **suite_extra)
        monkeypatch.setattr(sys, "argv", ["prog", "--suite", str(suite), flag,
                                          str(ck)])
        with pytest.raises(CheckpointDycoreMismatch, match=_SP):
            mod.main()


@pytest.mark.parametrize("script, extra", [
    ("scripts/run/run_aimip_ablation.py", []),
    ("scripts/run/_aimip_ablation_single.py", ["--label", "x", "--out", "o"]),
])
@pytest.mark.parametrize("where", ["base", "cfg_overrides", "suite"])
def test_ablation_drivers_refuse_the_keys(tmp_path, monkeypatch, script, extra,
                                          where):
    import jax
    prev = bool(jax.config.jax_enable_x64)
    jax.config.update("jax_enable_x64", True)
    try:
        _ablation_refuses(tmp_path, monkeypatch, script, extra, where)
    finally:
        jax.config.update("jax_enable_x64", prev)


def _ablation_refuses(tmp_path, monkeypatch, script, extra, where):
    mod = _load("_ckdy_" + Path(script).stem, script)
    # any present value refused, explicit false too
    for key, val in [(k, v) for k in (_LL, _SP) for v in (True, False)]:
        suite = _suite(tmp_path, **({key: val} if where == "base" else {}))
        if where != "base":
            s = yaml.safe_load(suite.read_text())
            if where == "suite":
                s[key] = val
            else:
                s["cfg_overrides"] = {key: val}
            suite.write_text(yaml.safe_dump(s))
        monkeypatch.setattr(sys, "argv", ["prog", "--suite", str(suite), *extra])
        with pytest.raises(ValueError, match=key):
            mod.main()


def test_refuse_helper_refuses_any_present_key():
    from legoesm.training.campaign_driver import refuse_pressure_surface_flags
    refuse_pressure_surface_flags({}, where="t")
    for v in (True, False, "false", None):
        with pytest.raises(ValueError, match=_SP):
            refuse_pressure_surface_flags({_SP: v}, where="t")


def test_wb_partial_generation_keeps_only_its_dycore_record(tmp_path):
    from legoesm.ml.checkpoint_io import dycore_record_path, write_dycore_record
    tws = _load("_ckdy_tws", "scripts/run/train_weatherbench_scale.py")
    stamp = [None]
    for step in (1, 2):
        tws._save_partial(
            str(tmp_path), {"epoch": 0, "step": step},
            lambda t: eqx.tree_serialise_leaves(t, _model()),
            lambda t: eqx.tree_serialise_leaves(t, {"m": jnp.zeros(1)}),
            lambda p: write_dycore_record(p, ON, stamp=stamp[0]),
            before_params_replace=lambda p, tmp: stamp.__setitem__(
                0, write_dycore_record(p, ON, payload=tmp, committed=False)))
    left = sorted(f.name for f in tmp_path.glob("partial.e*.dycore.json"))
    assert left == [dycore_record_path(
        tmp_path / "partial.e0000.s000002.eqx").name]


def test_inference_restart_options_check():
    mod = _load("_ckdy_inf_restart", "scripts/run/run_aimip_amip_inference.py")
    f = mod._check_restart_dycore_options
    f({}, OFF, "r")                              # legacy restart = off
    f({"dycore_options": ON}, ON, "r")
    with pytest.raises(SystemExit, match="different dynamics"):
        f({}, ON, "r")
    with pytest.raises(SystemExit, match="different dynamics"):
        f({"dycore_options": ON}, OFF, "r")
    for bad in (None, {}, {_LL: 0, _SP: 1}):    # present but malformed
        with pytest.raises(ValueError, match="as bools"):
            f({"dycore_options": bad}, OFF, "r")


def test_different_suffixes_do_not_share_a_record(tmp_path):
    from legoesm.ml.checkpoint_io import read_dycore_record
    from legoesm.ml.training import save_checkpoint
    save_checkpoint(_model(), tmp_path / "params.eqx", dycore_options=ON)
    save_checkpoint(_model(), tmp_path / "params.bin")
    assert read_dycore_record(tmp_path / "params.eqx") == ON


def test_wb_partial_interrupted_overwrite_is_refused(tmp_path, monkeypatch):
    """Re-saving the same (epoch, step) generation with byte-identical weights
    but other options, killed after the payload rename: the record is left
    uncommitted, so resume refuses instead of accepting the old options."""
    from legoesm.ml.checkpoint_io import (
        CheckpointStructureError, check_dycore_record, write_dycore_record,
    )
    tws = _load("_ckdy_tws2", "scripts/run/train_weatherbench_scale.py")
    stamp = [None]

    def save(opts, commit=True):
        def rec(p):
            if not commit:
                raise KeyboardInterrupt("killed before the record commit")
            write_dycore_record(p, opts, stamp=stamp[0])
        tws._save_partial(
            str(tmp_path), {"epoch": 0, "step": 1},
            lambda t: eqx.tree_serialise_leaves(t, _model()),
            lambda t: eqx.tree_serialise_leaves(t, {"m": jnp.zeros(1)}),
            rec,
            before_params_replace=lambda p, tmp: stamp.__setitem__(
                0, write_dycore_record(p, opts, payload=tmp, committed=False)))
    save(OFF)
    with pytest.raises(KeyboardInterrupt):
        save(ON, commit=False)
    for opts in (OFF, ON):
        with pytest.raises(CheckpointStructureError, match="interrupted save"):
            check_dycore_record(tmp_path / "partial.e0000.s000001.eqx", opts)


def test_symlinked_checkpoint_uses_the_real_files_record(tmp_path):
    from legoesm.ml.checkpoint_io import CheckpointDycoreMismatch, load_checked
    from legoesm.ml.training import save_checkpoint
    real = tmp_path / "run" / "params.eqx"
    save_checkpoint(_model(), real, dycore_options=ON)
    (tmp_path / "eval").mkdir()
    link = tmp_path / "eval" / "model.eqx"
    link.symlink_to(real)
    with pytest.raises(CheckpointDycoreMismatch):
        load_checked(link, OFF, lambda: pytest.fail("loaded unchecked"))
    load_checked(link, ON, lambda: None)


def test_symlink_to_a_suffixless_payload(tmp_path):
    import shutil
    from legoesm.ml.checkpoint_io import CheckpointDycoreMismatch, check_dycore_record
    from legoesm.ml.training import save_checkpoint
    src = tmp_path / "params.eqx"
    save_checkpoint(_model(), src, dycore_options=ON)
    shutil.move(src, tmp_path / "weights")
    shutil.move(tmp_path / "params.eqx.dycore.json",
                tmp_path / "weights.dycore.json")
    link = tmp_path / "model.eqx"
    link.symlink_to(tmp_path / "weights")
    with pytest.raises(CheckpointDycoreMismatch):
        check_dycore_record(link, OFF)


def test_record_writes_follow_the_real_file_and_symlink_saves_refused(tmp_path):
    from legoesm.ml.checkpoint_io import (
        CheckpointDycoreMismatch, check_dycore_record, write_dycore_record,
    )
    from legoesm.ml.training import save_checkpoint
    real = tmp_path / "params.eqx"
    eqx.tree_serialise_leaves(str(real), _model())
    link = tmp_path / "model.eqx"
    link.symlink_to(real)
    write_dycore_record(link, ON)            # record-only write via the link
    with pytest.raises(CheckpointDycoreMismatch):
        check_dycore_record(link, OFF)
    check_dycore_record(real, ON)
    with pytest.raises(ValueError, match="symlink"):
        save_checkpoint(_model(), link, dycore_options=ON)


def test_record_write_through_a_symlink_to_a_suffixless_payload(tmp_path):
    from legoesm.ml.checkpoint_io import (
        CheckpointDycoreMismatch, check_dycore_record, read_dycore_record,
        write_dycore_record,
    )
    from legoesm.ml.training import save_checkpoint
    eqx.tree_serialise_leaves(str(tmp_path / "weights.tmp"), _model())
    (tmp_path / "weights.tmp").rename(tmp_path / "weights")
    other = tmp_path / "weights.eqx"
    save_checkpoint(_model(4.0), other, dycore_options=OFF)
    link = tmp_path / "model.eqx"
    link.symlink_to(tmp_path / "weights")
    write_dycore_record(link, ON)
    with pytest.raises(CheckpointDycoreMismatch):
        check_dycore_record(link, OFF)
    assert read_dycore_record(other) == OFF      # untouched


def test_phase1_refuses_symlink_destination_and_dangling_record(tmp_path):
    from legoesm.ml.checkpoint_io import (
        CheckpointStructureError, dycore_record_path, read_dycore_record,
        write_dycore_record,
    )
    from legoesm.ml.training import save_checkpoint
    real = tmp_path / "real.eqx"
    save_checkpoint(_model(), real, dycore_options=OFF)
    link = tmp_path / "params.eqx"
    link.symlink_to(real)
    tmp = tmp_path / "new.tmp"
    eqx.tree_serialise_leaves(str(tmp), _model(1.0))
    with pytest.raises(ValueError, match="symlink"):   # WB / lat-lon phase 1
        write_dycore_record(link, ON, payload=tmp, committed=False)
    side = dycore_record_path(real)
    side.unlink()
    side.symlink_to(tmp_path / "gone.json")
    with pytest.raises(CheckpointStructureError, match="dangling"):
        read_dycore_record(real)
    save_checkpoint(_model(), real)             # None save clears it
    assert not side.is_symlink() and read_dycore_record(real) == OFF
    with pytest.raises(FileNotFoundError):
        read_dycore_record(tmp_path / "missing.eqx")


def test_none_save_over_a_symlink_with_a_stale_record(tmp_path):
    from legoesm.ml.checkpoint_io import dycore_record_path, read_dycore_record
    from legoesm.ml.training import save_checkpoint
    real = tmp_path / "real.eqx"
    save_checkpoint(_model(), real, dycore_options=ON)
    link = tmp_path / "params.eqx"
    link.symlink_to(real)
    dycore_record_path(link).write_text("{}")      # stale, beside the link
    save_checkpoint(_model(2.0), link)             # legacy None save works
    assert not link.is_symlink() and read_dycore_record(link) == OFF
    assert read_dycore_record(real) == ON          # target untouched


def test_none_save_of_a_suffixless_payload_clears_its_record(tmp_path):
    from legoesm.ml.checkpoint_io import read_dycore_record, write_dycore_record
    from legoesm.ml.training import save_checkpoint
    weights = tmp_path / "weights"
    save_checkpoint(_model(), weights)
    other = tmp_path / "weights.eqx"
    save_checkpoint(_model(4.0), other, dycore_options=ON)
    link = tmp_path / "model.eqx"
    link.symlink_to(weights)
    write_dycore_record(link, ON)                  # -> weights.dycore.json
    save_checkpoint(_model(), weights)             # byte-identical, None
    assert read_dycore_record(link) == OFF
    assert read_dycore_record(other) == ON         # separate file untouched


def test_resume_helpers_without_options_check_as_all_off(tmp_path):
    from legoesm.ml.checkpoint_io import CheckpointDycoreMismatch
    from legoesm.ml.training import save_checkpoint
    from legoesm.training.neural_gcm_spectral import (
        _load_midepoch_checkpoint, _save_midepoch_checkpoint, maybe_resume_model,
    )
    save_checkpoint(_model(), tmp_path / "epoch_0000.eqx", dycore_options=ON)
    with pytest.raises(CheckpointDycoreMismatch):
        maybe_resume_model(_model(), tmp_path)
    opt = {"m": jnp.zeros(3)}
    _save_midepoch_checkpoint(tmp_path, _model(), opt, 0, 1, dycore_options=ON)
    with pytest.raises(CheckpointDycoreMismatch):
        _load_midepoch_checkpoint(tmp_path, _model(), opt)
    with pytest.raises(ValueError, match="as bools"):   # explicit {} invalid
        maybe_resume_model(_model(), tmp_path, dycore_options={})


_HDR = '"format": "legoesm-dycore-options", "version": 1, "committed": true'


@pytest.mark.parametrize("body", [
    "[]", "{not json",
    '{"format": "legoesm-dycore-options", "version": true}',
    "{" + _HDR + ', "latlon_ah_pressure_surfaces": false}',          # missing
    "{" + _HDR + ', "latlon_ah_pressure_surfaces": false, '
    '"spectral_t_damping_pressure_surfaces": 1}',                     # non-bool
])
def test_malformed_record_is_refused(tmp_path, body):
    from legoesm.ml.checkpoint_io import (
        CheckpointStructureError, dycore_record_path, read_dycore_record,
    )
    import json
    from legoesm.ml.checkpoint_io import write_dycore_record
    p = tmp_path / "params.eqx"
    eqx.tree_serialise_leaves(str(p), _model())
    if body.startswith("{" + _HDR):   # valid stamp, so the options are reached
        write_dycore_record(p, OFF)
        good = json.loads(dycore_record_path(p).read_text())
        stamp = {k: good[k] for k in ("payload_crc32", "payload_size")}
        body = json.dumps({**json.loads(body), **stamp})
    dycore_record_path(p).write_text(body)
    with pytest.raises(CheckpointStructureError, match="not a dycore record"):
        read_dycore_record(p)


def test_none_save_is_fenced_until_complete(tmp_path, monkeypatch):
    from legoesm.ml import checkpoint_io
    from legoesm.ml.checkpoint_io import (
        CheckpointStructureError, dycore_record_path, read_dycore_record,
    )
    from legoesm.ml.training import save_checkpoint
    p = tmp_path / "params.eqx"
    save_checkpoint(_model(), p)
    assert not dycore_record_path(p).exists()      # completed: no record

    def _killed(*_a, **_k):
        raise KeyboardInterrupt("killed before the knob record")
    monkeypatch.setattr(checkpoint_io, "write_knob_record", _killed)
    with pytest.raises(KeyboardInterrupt):
        save_checkpoint(_model(2.0), p)
    monkeypatch.undo()
    with pytest.raises(CheckpointStructureError, match="interrupted save"):
        read_dycore_record(p)
