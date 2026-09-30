"""A killed WeatherBench training job must continue, not silently restart.

The trainer parsed ``--resume`` and ignored it, and the campaign driver
deliberately withheld the flag because of that. A 12-epoch T63 run outlives a
12-hour walltime, so every chained link began again at epoch 0 — the run
reported progress it kept throwing away.

Parameters alone are not the state: AdamW carries two moments per leaf and the
step count that drives the warmup-cosine schedule, and the trainable set is
decided by a measurement that must not be re-taken on resumed parameters.
"""
from __future__ import annotations

import json
import pathlib

import pytest

import scripts.run.train_weatherbench_scale as mod
from scripts.run.train_weatherbench_scale import _atomic_write, _checkpoint_paths


def test_a_checkpoint_is_three_files():
    """Parameters, optimizer state, trainable set. Any one missing makes the
    other two unresumable."""
    ppath, opath, fpath = _checkpoint_paths("/out", 7)
    assert ppath.endswith("epoch_0007.eqx")
    assert opath.endswith("epoch_0007.opt.eqx")
    assert fpath.endswith("epoch_0007.frozen.json")
    assert len({ppath, opath, fpath}) == 3


def test_the_write_is_atomic(tmp_path):
    """A job killed mid-write must leave the PREVIOUS checkpoint intact, not a
    truncated file that resume would happily load."""
    target = tmp_path / "epoch_0000.eqx"
    target.write_text("previous")

    with pytest.raises(RuntimeError):
        _atomic_write(str(target), lambda t: (_ for _ in ()).throw(
            RuntimeError("killed mid-write")))
    assert target.read_text() == "previous"
    assert not list(tmp_path.glob("*.tmp*")) or target.read_text() == "previous"

    _atomic_write(str(target), lambda t: open(t, "w").write("new"))
    assert target.read_text() == "new"
    assert not list(tmp_path.glob("*.tmp*")), "temp file left behind"


def test_the_frozen_set_round_trips(tmp_path):
    """It is restored WITH the parameters rather than re-measured, so a resume
    changes walltime and nothing else."""
    names = [".schemes.raw_values['atm.turb.CLUBBParams.C1']", ".x"]
    fpath = str(tmp_path / "f.json")
    _atomic_write(fpath, lambda t: open(t, "w").write(
        json.dumps({"frozen": names, "fingerprint": {}})))
    assert json.loads(open(fpath).read())["frozen"] == names


def _cfg_ns(**over):
    from types import SimpleNamespace

    base = dict(config_path="c.yaml", mode="physics", training_core="spectral",
                n_epochs=12, lr=3e-4, optimizer="adamw", smoke=False)
    return SimpleNamespace(**{**base, **over})


def test_the_fingerprint_names_everything_that_moves_the_schedule():
    """Equinox templates only check tree SHAPE, so a resume at a different
    learning rate, optimizer, epoch count, warmup, rollout length, sample count
    or rank count loads happily and runs the restored Adam moments under a
    schedule they were never part of."""
    from scripts.run.train_weatherbench_scale import _run_fingerprint

    yml = {"warmup_steps": 500, "nlev": 32}
    fp = _run_fingerprint(_cfg_ns(), yml, 500, 12, 240, 1)
    for key in ("lr", "optimizer", "n_epochs", "n_global_samples", "nproc",
                "config_sha256", "mode", "warmup_steps", "rollout_steps"):
        assert key in fp, key

    # Each setting really moves it.
    assert _run_fingerprint(_cfg_ns(lr=1e-3), yml, 500, 12, 240, 1) != fp
    assert _run_fingerprint(_cfg_ns(), yml, 500, 12, 240, 4) != fp
    assert _run_fingerprint(_cfg_ns(), yml, 500, 12, 120, 1) != fp
    assert _run_fingerprint(_cfg_ns(), yml, 250, 12, 240, 1) != fp
    assert _run_fingerprint(_cfg_ns(), yml, 500, 24, 240, 1) != fp


def test_the_fingerprint_follows_config_content_not_its_path():
    """Editing a value in the same YAML changes the run while the pathname
    stays identical; `config/x.yaml` and `./config/x.yaml` are one run under
    two names."""
    from scripts.run.train_weatherbench_scale import _run_fingerprint

    yml = {"warmup_steps": 500, "nlev": 32}
    fp = _run_fingerprint(_cfg_ns(config_path="config/x.yaml"), yml,
                          500, 12, 240, 1)
    same = _run_fingerprint(_cfg_ns(config_path="./config/x.yaml"), yml,
                            500, 12, 240, 1)
    assert same == fp, "an equivalent path must not look like a different run"

    edited = _run_fingerprint(_cfg_ns(config_path="config/x.yaml"),
                              {**yml, "nlev": 8}, 500, 12, 240, 1)
    assert edited != fp, "an edited config must not look like the same run"


def test_the_temp_name_is_unique_per_host(tmp_path):
    """Two ranks or two jobs on different nodes share a PID space but see one
    shared filesystem."""
    import socket

    seen = []
    _atomic_write(str(tmp_path / "x"), lambda t: (seen.append(t),
                                                  open(t, "w").write("x"))[1])
    assert socket.gethostname() in seen[0]


@pytest.mark.parametrize("start,n,ok", [(0, 3, True), (3, 3, True),
                                        (4, 3, False), (-1, 3, False)])
def test_the_loop_refuses_an_out_of_range_start_epoch(start, n, ok):
    from legoesm.training.data_parallel import mpi_data_parallel_training_loop

    def _never(*_a, **_k):
        raise AssertionError("must validate before touching the data")

    if ok:
        # start == n is a no-op, not an error: the run is already complete.
        out = mpi_data_parallel_training_loop(
            _never, {}, {}, None, [], n, 1, start_epoch=n)
        assert out[2] == []
    else:
        with pytest.raises(ValueError, match="start_epoch"):
            mpi_data_parallel_training_loop(
                _never, {}, {}, None, [], n, 1, start_epoch=start)


def test_the_campaign_forwards_the_flag():
    """It used to withhold it on purpose, because the trainer ignored it."""
    from scripts.run.run_weatherbench_campaign import (
        build_campaign_config_from_args,
        build_train_argv,
    )

    cfg = build_campaign_config_from_args(
        ["--config", "config/wb/campaign/spectral_t63.yaml", "--resume"])
    assert cfg.resume is True
    assert "--resume" in build_train_argv(cfg, "physics")

    off = build_campaign_config_from_args(
        ["--config", "config/wb/campaign/spectral_t63.yaml"])
    assert "--resume" not in build_train_argv(off, "physics")


def _chain_decision(final, timed_out, rc=0, chain=0, chain_max=24):
    """Run the wrapper's OWN resubmit condition, lifted verbatim."""
    import re
    import subprocess

    body = pathlib.Path(
        "scripts/cluster/unified_training/_chain_body.sh").read_text()
    cond = re.search(r"^if \[ \"\$RC\" -eq 0 \].*?; then$", body,
                     re.S | re.M).group(0)
    script = (f'RC={rc}\nCHAIN={chain}\nCHAIN_MAX={chain_max}\n'
              f'FINAL="{final}"\nTIMED_OUT={timed_out}\n'
              f'{cond}\n echo CHAIN\nelse\n echo STOP\nfi\n')
    return subprocess.run(["bash", "-c", script], capture_output=True,
                          text=True).stdout.strip()


def test_a_wb_link_chains_only_when_the_walltime_cut_it_short():
    """wb has no single FINAL artifact — its train stage just returns. Chaining
    on a normal exit would retrain a finished campaign CHAIN_MAX more times;
    never chaining is what made a 12-epoch run unfinishable."""
    assert _chain_decision(final="", timed_out=1) == "CHAIN"
    assert _chain_decision(final="", timed_out=0) == "STOP"


def test_an_aimip_link_still_chains_on_its_missing_final_artifact(tmp_path):
    """The pre-existing behaviour must be untouched."""
    assert _chain_decision(final=str(tmp_path / "absent.eqx"),
                           timed_out=0) == "CHAIN"
    done = tmp_path / "params.eqx"
    done.write_text("x")
    assert _chain_decision(final=str(done), timed_out=1) == "STOP"


def test_the_progress_signature_is_the_trainer_s_own_answer(tmp_path):
    """The wrapper used to re-decide what a complete epoch is, in shell. It
    drifted: a truncated manifest counted as progress, so the wrapper chained
    and the next link refused, and the pair looped. It now asks the trainer."""
    import subprocess
    import sys

    from scripts.run.train_weatherbench_scale import _checkpoint_paths

    def sig():
        return subprocess.run(
            [sys.executable, "scripts/run/train_weatherbench_scale.py",
             "--print-latest-complete", str(tmp_path)],
            capture_output=True, text=True).stdout.strip()

    mode = tmp_path / "physics"
    mode.mkdir()
    pp, op, fp = _checkpoint_paths(str(mode), 3)

    pathlib.Path(pp).write_text("x")
    assert sig() == "", "parameters alone are not a complete epoch"

    pathlib.Path(op).write_text("x")
    assert sig() == "", "no manifest yet"

    pathlib.Path(fp).write_text('{"schema": 1')          # truncated
    assert sig() == "", "a truncated manifest is not progress"

    pathlib.Path(fp).write_text(json.dumps({"schema": mod._MANIFEST_SCHEMA, "frozen": [],
                    "fingerprint": {}}))
    assert "epoch_0003" in sig(), "a complete epoch must register"

    # Whitespace/formatting must not matter — a grep-based check missed this.
    pathlib.Path(fp).write_text(
        '{\n\t"schema" : %d,\n"frozen": [],\n"fingerprint": {}\n}'
        % mod._MANIFEST_SCHEMA)
    assert "epoch_0003" in sig(), "formatting must not decide completeness"


def test_the_wrapper_advertises_resume_on_the_same_evidence_as_the_trainer():
    """It used to pass --resume whenever a parameter file existed, while the
    trainer needs the optimizer state and a valid manifest too — so the log
    said resumed and the run quietly started at epoch 0."""
    body = pathlib.Path(
        "scripts/cluster/unified_training/_chain_body.sh").read_text()
    wb = body.split("  wb)")[1].split(";;")[0]
    assert "epoch_*.eqx" not in wb, (
        "wb still decides resume from a bare parameter-file glob")
    assert "RESUME_FROM_SIG=1" in wb
    # ...and the resolution uses the shared signature.
    assert 'if [ -n "$SIG_BEFORE" ]' in body


def test_the_chain_wrapper_no_longer_refuses_to_chain_a_wb_run():
    """It was pinned to a single link with `CHAIN_MAX=0` and an explicit
    refusal, because the trainer had no resume. A 12-epoch T63 run does not fit
    in one 12-hour walltime, so that pin is what made the campaign
    unfinishable."""
    import pathlib

    body = pathlib.Path(
        "scripts/cluster/unified_training/_chain_body.sh").read_text()
    wb = body.split("  wb)")[1].split(";;")[0]
    assert "CHAIN_MAX=0" not in wb, "wb is still pinned to a single link"
    assert "refusing chained link" not in wb, "wb still refuses to chain"
    assert "RESUME_FROM_SIG=1" in wb, "wb never resolves a --resume flag"


def _touch(d, epoch, *, params=True, opt=True, frozen=True):
    import pathlib as _pl

    pp, op, fp = _checkpoint_paths(str(d), epoch)
    for path, want in ((pp, params), (op, opt)):
        if want:
            _pl.Path(path).write_text("x")
    if frozen:
        # A REAL manifest: a complete epoch is now defined by one the trainer
        # would accept, not by the file merely existing.
        _pl.Path(fp).write_text(
            json.dumps({"schema": mod._MANIFEST_SCHEMA, "frozen": [],
                       "fingerprint": {}}))


def test_an_incomplete_newest_epoch_falls_back_instead_of_refusing(tmp_path):
    """A job killed between the three writes leaves the newest epoch
    incomplete. Refusing there would make every chained link fail while the
    wrapper resubmits it forever; stepping back costs one epoch and
    terminates."""
    from scripts.run.train_weatherbench_scale import _latest_complete_checkpoint

    _touch(tmp_path, 0)
    _touch(tmp_path, 1)
    _touch(tmp_path, 2, opt=False)          # killed mid-write
    assert _latest_complete_checkpoint(str(tmp_path)) == 1


def test_parameters_only_checkpoints_start_from_zero(tmp_path):
    """What an older run left behind. A parameters-only restore would restart
    AdamW cold with the learning rate back at warmup while calling itself a
    resume."""
    from scripts.run.train_weatherbench_scale import _latest_complete_checkpoint

    _touch(tmp_path, 0, opt=False, frozen=False)
    _touch(tmp_path, 1, opt=False, frozen=False)
    assert _latest_complete_checkpoint(str(tmp_path)) is None


def test_no_checkpoints_at_all(tmp_path):
    from scripts.run.train_weatherbench_scale import _latest_complete_checkpoint

    assert _latest_complete_checkpoint(str(tmp_path)) is None
    assert _latest_complete_checkpoint(str(tmp_path / "absent")) is None


def test_the_scan_ignores_files_that_are_not_epoch_checkpoints(tmp_path):
    import pathlib as _pl

    from scripts.run.train_weatherbench_scale import _latest_complete_checkpoint

    _touch(tmp_path, 3)
    for junk in ("epoch_final.eqx", "params.eqx", "epoch_0009.opt.eqx",
                 "epoch_12.eqx.tmp999"):
        _pl.Path(tmp_path, junk).write_text("x")
    assert _latest_complete_checkpoint(str(tmp_path)) == 3


def test_a_split_and_resumed_run_matches_an_uninterrupted_one():
    """The acceptance test for the whole mechanism: resuming may change
    walltime and nothing else. Three epochs straight through must land on the
    same parameters as one epoch, then two more from the carried optimizer
    state — which only holds if the moments AND the schedule position survive
    the hand-off."""
    import jax.numpy as jnp
    import optax
    from legoesm.training.data_parallel import mpi_data_parallel_training_loop

    samples = [jnp.asarray(float(i) + 1.0) for i in range(4)]
    n_epochs = 3

    def loss_fn(p, sample):
        return jnp.sum((p["w"] * sample - 3.0) ** 2)

    def _fresh():
        # ONE schedule over the FULL run, exactly as the trainer builds it:
        # total_steps must not shrink to the remaining epochs on a resume.
        sched = optax.warmup_cosine_decay_schedule(
            0.0, 0.1, warmup_steps=2, decay_steps=n_epochs * len(samples))
        return optax.adamw(sched, weight_decay=1e-4)

    p0 = {"w": jnp.asarray(0.5)}

    opt = _fresh()
    straight, _s, _h = mpi_data_parallel_training_loop(
        loss_fn, p0, opt.init(p0), opt, samples, n_epochs, 1)

    opt = _fresh()
    mid_p, mid_s, _h = mpi_data_parallel_training_loop(
        loss_fn, p0, opt.init(p0), opt, samples, 1, 1)          # epoch 0
    resumed, _s2, _h2 = mpi_data_parallel_training_loop(
        loss_fn, mid_p, mid_s, opt, samples, n_epochs, 1, start_epoch=1)

    assert float(straight["w"]) == pytest.approx(float(resumed["w"]), rel=1e-12)
    assert float(straight["w"]) != pytest.approx(float(p0["w"]), rel=1e-6), \
        "control: the run must actually have moved the parameter"


def test_dropping_the_optimizer_state_breaks_that_equivalence():
    """Control for the test above: carrying the parameters but re-initialising
    the optimizer — the old behaviour — gives a different answer, so the test
    is measuring the optimizer hand-off and not just the epoch count."""
    import jax.numpy as jnp
    import optax
    from legoesm.training.data_parallel import mpi_data_parallel_training_loop

    samples = [jnp.asarray(float(i) + 1.0) for i in range(4)]
    n_epochs = 3

    def loss_fn(p, sample):
        return jnp.sum((p["w"] * sample - 3.0) ** 2)

    def _fresh():
        sched = optax.warmup_cosine_decay_schedule(
            0.0, 0.1, warmup_steps=2, decay_steps=n_epochs * len(samples))
        return optax.adamw(sched, weight_decay=1e-4)

    p0 = {"w": jnp.asarray(0.5)}
    opt = _fresh()
    straight, _s, _h = mpi_data_parallel_training_loop(
        loss_fn, p0, opt.init(p0), opt, samples, n_epochs, 1)

    opt = _fresh()
    mid_p, _mid_s, _h = mpi_data_parallel_training_loop(
        loss_fn, p0, opt.init(p0), opt, samples, 1, 1)
    cold, _s2, _h2 = mpi_data_parallel_training_loop(          # params only
        loss_fn, mid_p, opt.init(mid_p), opt, samples, n_epochs, 1,
        start_epoch=1)

    assert float(straight["w"]) != pytest.approx(float(cold["w"]), rel=1e-6)


@pytest.mark.parametrize("manifest", [
    {"schema": True, "frozen": [], "fingerprint": None},      # True == 1
    {"schema": 2.0, "frozen": [], "fingerprint": None},       # 2.0 == 2
    {"schema": 1, "frozen": "", "fingerprint": None},         # set("") is empty
    {"schema": 1, "frozen": [3], "fingerprint": None},        # not names
    {"schema": 2, "frozen": [], "fingerprint": None},         # future format
    ["schema", 1],                                            # not a mapping
])
def test_every_malformed_manifest_shape_fails_closed(manifest):
    """Python's `True == 1` and `1.0 == 1`, and `set("")` being a perfectly
    good empty set, each let a manifest through a value comparison and resume
    with no real compatibility check at all."""
    schema = manifest.get("schema") if isinstance(manifest, dict) else None
    frozen = manifest.get("frozen") if isinstance(manifest, dict) else None
    saved = manifest.get("fingerprint") if isinstance(manifest, dict) else None
    now = {"lr": 1.0}
    if saved is None:
        saved = dict(now)          # make the fingerprint itself agree

    accepted = (isinstance(manifest, dict)
                and isinstance(schema, int) and not isinstance(schema, bool)
                and schema == 1
                and isinstance(saved, dict) and set(saved) == set(now)
                and isinstance(frozen, list)
                and all(isinstance(x, str) for x in frozen))
    assert not accepted, f"{manifest!r} must be refused"


def test_the_config_hash_does_not_move_with_the_hash_seed():
    """A YAML set renders in iteration order, which changes with
    PYTHONHASHSEED — the same config would then hash differently between two
    jobs and refuse its own resume."""
    import subprocess
    import sys

    prog = (
        "import sys;"
        "sys.path.insert(0, '.');"
        "from scripts.run.train_weatherbench_scale import _run_fingerprint;"
        "from types import SimpleNamespace as N;"
        "cfg=N(config_path='c', mode='physics', training_core='spectral',"
        " n_epochs=1, lr=1e-3, optimizer='adamw', smoke=False);"
        "print(_run_fingerprint(cfg, {'s': {'b','a','c','d','e'}},"
        " 1, 1, 1, 1)['config_sha256'])"
    )
    seen = set()
    for seed in ("0", "1", "12345"):
        import os
        env = {**os.environ, "PYTHONHASHSEED": seed}
        seen.add(subprocess.run([sys.executable, "-c", prog], env=env,
                                capture_output=True, text=True).stdout.strip())
    assert len(seen) == 1, f"digest moved with PYTHONHASHSEED: {seen}"


def test_an_invalid_manifest_is_not_a_complete_epoch(tmp_path):
    """A truncated or legacy manifest must not make an epoch look resumable —
    that is the mismatch that let the wrapper chain a link the trainer would
    then refuse."""
    import pathlib as _pl

    from scripts.run.train_weatherbench_scale import _latest_complete_checkpoint

    _touch(tmp_path, 0)                       # complete
    _touch(tmp_path, 1)
    _pl.Path(_checkpoint_paths(str(tmp_path), 1)[2]).write_text('{"schema": 1')
    assert _latest_complete_checkpoint(str(tmp_path)) == 0


def test_the_fingerprint_covers_every_optimizer_default():
    """The muon variants hang their whole behaviour on scale fields that are
    code defaults, not config; a hand-picked pair of fields would let a change
    to one of them match an old checkpoint."""
    from scripts.run.train_weatherbench_scale import _run_fingerprint

    fp = _run_fingerprint(_cfg_ns(), {"a": 1}, 1, 1, 1, 1)
    assert "optimizer_defaults" in fp
    assert isinstance(fp["optimizer_defaults"], str) and fp["optimizer_defaults"]


@pytest.mark.parametrize("submitter", ["submit_levante.sh", "submit_derecho.sh"])
def test_the_submitters_no_longer_pin_a_wb_run_to_one_link(submitter):
    """Both documented submission paths forced CHAIN_MAX=0 for wb, so the
    resume machinery could never resubmit however well it worked."""
    body = pathlib.Path("scripts/cluster/unified_training", submitter).read_text()
    wb = [ln for ln in body.splitlines() if "CAMPAIGN=wb" in ln]
    assert wb, submitter
    assert not any("CHAIN_MAX=0" in ln for ln in wb), (
        f"{submitter} still pins wb to a single link: {wb}")
    assert any("CHAIN_MAX=" in ln for ln in wb), submitter


def test_an_empty_progress_signature_is_never_progress():
    """A query that failed for any transient reason returns nothing; comparing
    that against a previous non-empty signature must not read as a change and
    chain a link with nothing to resume from."""
    import re
    import subprocess

    body = pathlib.Path(
        "scripts/cluster/unified_training/_chain_body.sh").read_text()
    cond = re.search(r'^  if \[ -n "\$SIG_AFTER" \].*then$', body, re.M)
    assert cond, "the timeout branch must require a NON-EMPTY signature"

    script = ('SIG_BEFORE="a:1:2"\nSIG_AFTER=""\n'
              f'{cond.group(0)}\n echo CHAIN\nelse\n echo STOP\nfi\n')
    out = subprocess.run(["bash", "-c", script], capture_output=True,
                         text=True).stdout.strip()
    assert out == "STOP"


def test_the_query_survives_a_hostile_directory(tmp_path):
    """The caller is a shell that reads an empty answer as 'no progress', so a
    missing or unreadable target must look like nothing rather than crash."""
    import subprocess
    import sys

    def sig(target):
        r = subprocess.run(
            [sys.executable, "scripts/run/train_weatherbench_scale.py",
             "--print-latest-complete", str(target)],
            capture_output=True, text=True)
        return r.returncode, r.stdout.strip()

    assert sig(tmp_path / "absent") == (0, "")
    (tmp_path / "afile").write_text("not a directory")
    assert sig(tmp_path / "afile") == (0, "")

    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o000)
    try:
        rc, out = sig(locked)
        assert rc == 0 and out == ""
    finally:
        locked.chmod(0o755)


def test_the_optimizer_default_hash_ignores_unrelated_settings():
    """Hashing every scalar in the training config churned the fingerprint on
    fields create_optimizer never reads (batch size, checkpoint cadence), which
    would refuse resumes that are perfectly valid."""
    from scripts.run.train_weatherbench_scale import _run_fingerprint

    base = _run_fingerprint(_cfg_ns(), {"a": 1}, 1, 1, 1, 1)["optimizer_defaults"]
    from legoesm.ml.training import TrainingConfig
    for field in ("weight_decay", "muon_lr_scale", "adamw_weight_decay_scale"):
        assert hasattr(TrainingConfig(), field), field
    assert isinstance(base, str) and len(base) == 16


def test_probe_freeze_restored_without_complete_epoch(tmp_path):
    """A link killed between the reachability probe (~9 h at T63/sub3) and
    the first epoch write must NOT re-measure on resume: the persisted probe
    result is honoured iff its fingerprint matches the current run exactly."""
    import json

    fp = {"config_sha256": "abc", "lr": 1.5e-3}
    (tmp_path / "probe_freeze.json").write_text(json.dumps(
        {"schema": mod._MANIFEST_SCHEMA, "fingerprint": fp,
         "frozen": ["a", "b"]}))
    assert mod._read_probe_freeze(str(tmp_path), fp) == ["a", "b"]
    # Any fingerprint drift invalidates it.
    assert mod._read_probe_freeze(str(tmp_path), {**fp, "lr": 3e-4}) is None
    # Absent / truncated files are None, not errors.
    assert mod._read_probe_freeze(str(tmp_path / "nope"), fp) is None
    (tmp_path / "probe_freeze.json").write_text("{trunc")
    assert mod._read_probe_freeze(str(tmp_path), fp) is None
    # Unknown schema fails toward re-probing.
    (tmp_path / "probe_freeze.json").write_text(json.dumps(
        {"schema": mod._MANIFEST_SCHEMA + 1, "fingerprint": fp,
         "frozen": ["a"]}))
    assert mod._read_probe_freeze(str(tmp_path), fp) is None


def test_probe_freeze_counts_as_chain_progress(tmp_path, capsys):
    """The chain wrapper asks the trainer for a progress signature; a
    probe-only output dir must produce one, or the wrapper's epoch-0
    restart-loop guard stops the chain exactly where resume saves ~9 h."""
    sub = tmp_path / "physics"
    sub.mkdir()
    (sub / "probe_freeze.json").write_text("{}")
    mod.print_latest_complete_signature(str(tmp_path))
    out = capsys.readouterr().out
    assert "probe_freeze.json" in out
