"""`legoesm reproduce` for component-runner manifests, the continuation rule
for a run into a directory that already holds a run, and the restart drift
gate (docs/architecture/reproducibility.md, P0).  Land (run_lmip) is the
runner exercised here."""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import pytest

import legoesm.driver.restart as restart
from legoesm.io.git_provenance import GitProvenance

SHA_A = "a" * 40
SHA_B = "b" * 40


def _pin(monkeypatch, commit: str) -> None:
    """Make the IMPORTED package look like a clean checkout at *commit*."""
    monkeypatch.setattr(
        restart, "git_provenance",
        lambda anchor: GitProvenance(anchor=str(anchor), root="/repo",
                                     commit=commit, ref="", dirty=False))


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in (restart.ALLOW_VERSION_DRIFT_ENV, restart.REPRODUCE_REFERENCE_ENV,
                restart.REPRODUCE_OUTPUT_ENV):
        monkeypatch.delenv(var, raising=False)
    _pin(monkeypatch, SHA_A)


class _Reached(Exception):
    """The run got past its manifest write (state initialisation)."""


def _lmip(monkeypatch, out: Path, *extra: str):
    """Run run_lmip.main up to state initialisation (sentinel)."""
    from scripts.run import run_lmip

    def _stop(*a, **kw):
        raise _Reached()

    monkeypatch.setattr(run_lmip, "init_multilayer_land_state", _stop)
    monkeypatch.setattr(run_lmip, "_load_restart", _stop)
    monkeypatch.setattr("sys.argv", [
        "run_lmip.py", "--lat", "45.0", "--days", "1", "--output", str(out),
        *extra])
    run_lmip.main()


def _reference(monkeypatch, tmp_path: Path) -> Path:
    """An in-process reference run, made to record the runner script's
    content as a real launch of scripts/run/run_lmip.py would."""
    ref = tmp_path / "ref"
    with pytest.raises(_Reached):
        _lmip(monkeypatch, ref)
    path = ref / restart.RUN_MANIFEST_FILENAME
    _edit(path, _verifiable)
    return path


def _under_reproduce(monkeypatch, ref_manifest: Path, out_root: Path) -> None:
    monkeypatch.setenv(restart.REPRODUCE_REFERENCE_ENV, str(ref_manifest))
    monkeypatch.setenv(restart.REPRODUCE_OUTPUT_ENV, str(out_root))


# --- rerun closure (the rerun's own manifest writer) -----------------------

def test_identical_rerun_passes_the_closure(tmp_path, monkeypatch):
    ref = _reference(monkeypatch, tmp_path)
    new = tmp_path / "new"
    _under_reproduce(monkeypatch, ref, new)
    with pytest.raises(_Reached):
        _lmip(monkeypatch, new)
    assert (new / restart.RUN_MANIFEST_FILENAME).is_file()


def test_rerun_resolving_differently_is_refused_before_the_run(tmp_path,
                                                               monkeypatch):
    from scripts.run import run_lmip

    ref = _reference(monkeypatch, tmp_path)
    new = tmp_path / "new"
    _under_reproduce(monkeypatch, ref, new)
    build = run_lmip.build_config_from_args
    monkeypatch.setattr(run_lmip, "build_config_from_args",
                        lambda a: build(a)._replace(seed=7))
    with pytest.raises(RuntimeError, match="run_config.seed") as exc:
        _lmip(monkeypatch, new)
    assert "does not resolve to the reference" in str(exc.value)
    assert not (new / restart.RUN_MANIFEST_FILENAME).exists()


def test_rerun_with_a_changed_flag_is_refused(tmp_path, monkeypatch):
    ref = _reference(monkeypatch, tmp_path)
    new = tmp_path / "new"
    _under_reproduce(monkeypatch, ref, new)
    with pytest.raises(RuntimeError, match="cli_args.precip_rate"):
        _lmip(monkeypatch, new, "--precip-rate", "3e-5")


def test_rerun_outside_the_reproduce_output_is_refused(tmp_path, monkeypatch):
    ref = _reference(monkeypatch, tmp_path)
    _under_reproduce(monkeypatch, ref, tmp_path / "intended")
    with pytest.raises(RuntimeError, match="output flag was not rewritten"):
        _lmip(monkeypatch, tmp_path / "elsewhere")


@pytest.mark.parametrize("only", ["reference", "output"])
def test_half_set_reproduce_environment_is_refused(tmp_path, monkeypatch, only):
    ref = _reference(monkeypatch, tmp_path)
    if only == "reference":
        monkeypatch.setenv(restart.REPRODUCE_REFERENCE_ENV, str(ref))
    else:
        monkeypatch.setenv(restart.REPRODUCE_OUTPUT_ENV, str(tmp_path / "new"))
    with pytest.raises(RuntimeError, match="or neither"):
        _lmip(monkeypatch, tmp_path / "new")


def test_unreadable_reference_is_refused(tmp_path, monkeypatch):
    bad = tmp_path / "bad.json"
    bad.write_text("{}")
    _under_reproduce(monkeypatch, bad, tmp_path / "new")
    with pytest.raises(RuntimeError, match="unreadable or invalid"):
        _lmip(monkeypatch, tmp_path / "new")


def test_reference_of_another_runner_is_refused(tmp_path, monkeypatch):
    ref = restart.write_run_manifest(tmp_path / "other", {"runner": "x"},
                                     config_kind="land", runner_tag="other")
    _under_reproduce(monkeypatch, ref, tmp_path / "new")
    with pytest.raises(RuntimeError, match="'other' runner"):
        _lmip(monkeypatch, tmp_path / "new")


def test_reference_from_other_code_is_refused_unless_allowed(tmp_path,
                                                             monkeypatch):
    ref = _reference(monkeypatch, tmp_path)
    _pin(monkeypatch, SHA_B)
    _under_reproduce(monkeypatch, ref, tmp_path / "new")
    with pytest.raises(RuntimeError, match=restart.ALLOW_VERSION_DRIFT_ENV):
        _lmip(monkeypatch, tmp_path / "new")
    monkeypatch.setenv(restart.ALLOW_VERSION_DRIFT_ENV, "1")
    with pytest.raises(_Reached):
        _lmip(monkeypatch, tmp_path / "new")


def test_closure_compares_input_path_and_size():
    def m(size, path="/d/x.npz"):
        return {"config": {"resolved_config": {"a": 1}},
                "result": {"dataset_provenance": [
                    {"id": "restart_from", "path": path, "size_bytes": size}]}}
    assert restart.manifest_closure_rows(m(1), m(1)) == []
    rows = restart.manifest_closure_rows(m(1), m(2))
    assert [r[0] for r in rows] == ["inputs.restart_from.size_bytes"]
    rows = restart.manifest_closure_rows(m(1), m(1, "/d/y.npz"))
    assert [r[0] for r in rows] == ["inputs.restart_from.path"]
    assert restart.manifest_closure_rows(
        m(1), m(2), ignore_dataset_ids=("restart_from",)) == []
    dup = m(1)
    dup["result"]["dataset_provenance"].append(
        {"id": "restart_from", "path": "/d/z.npz", "size_bytes": 1})
    rows = restart.manifest_closure_rows(m(1), dup)    # a second input is seen
    assert [r[0] for r in rows] == ["inputs.restart_from"]


# --- continuation into a directory that already holds a run ----------------

def test_continuation_may_change_only_its_controls(tmp_path, monkeypatch):
    from scripts.run import run_lmip

    out = tmp_path / "run"
    with pytest.raises(_Reached):
        _lmip(monkeypatch, out)
    with pytest.raises(_Reached):                  # longer run, new cadence
        _lmip(monkeypatch, out, "--days", "3", "--checkpoint-days", "5")
    assert len(list(out.glob("run_manifest.prev-*.json"))) == 1
    build = run_lmip.build_config_from_args
    monkeypatch.setattr(run_lmip, "build_config_from_args",
                        lambda a: build(a)._replace(seed=7))
    with pytest.raises(RuntimeError, match="run_config.seed"):
        _lmip(monkeypatch, out, "--days", "3", "--checkpoint-days", "5")
    monkeypatch.setattr(run_lmip, "build_config_from_args", build)
    with pytest.raises(RuntimeError, match="cli_args.precip_rate"):
        _lmip(monkeypatch, out, "--days", "3", "--precip-rate", "3e-5")
    with pytest.raises(_Reached):                  # every continuation control
        _lmip(monkeypatch, out, "--days", "4", "--diag-interval", "2",
              "--max-wallclock-seconds", "100", "--restart-buffer-seconds", "5",
              "--checkpoint-days", "2")


def test_refused_continuation_leaves_the_record_in_place(tmp_path,
                                                        monkeypatch):
    out = tmp_path / "run"
    with pytest.raises(_Reached):
        _lmip(monkeypatch, out)
    before = (out / restart.RUN_MANIFEST_FILENAME).read_text()
    with pytest.raises(RuntimeError, match="one output directory holds one run"):
        _lmip(monkeypatch, out, "--precip-rate", "3e-5")
    assert (out / restart.RUN_MANIFEST_FILENAME).read_text() == before
    assert not list(out.glob("run_manifest.prev-*.json"))


def test_failed_write_after_archiving_restores_the_record(tmp_path,
                                                         monkeypatch):
    def write():
        restart.write_run_manifest_on_root(tmp_path, {"runner": "a"},
                                           config_kind="land",
                                           runner_tag="run_lmip")
    write()
    before = (tmp_path / restart.RUN_MANIFEST_FILENAME).read_text()
    real = restart.write_run_manifest

    def flaky(directory, config, **kw):
        if not kw.get("exclusive"):
            raise OSError("disk full")       # the replace, after the archive
        return real(directory, config, **kw)

    monkeypatch.setattr(restart, "write_run_manifest", flaky)
    with pytest.raises(RuntimeError, match="disk full"):
        write()
    assert (tmp_path / restart.RUN_MANIFEST_FILENAME).read_text() == before
    assert not list(tmp_path.glob("run_manifest.prev-*.json"))


@pytest.mark.parametrize("sha, message", [
    (None, "does not record the launched script"),
    ("0" * 64, "differs from the"),
])
def test_reproduce_refuses_an_unverifiable_or_different_script(
        tmp_path, monkeypatch, caplog, sha, message):
    ref = _reference(monkeypatch, tmp_path)
    _edit(ref, lambda m: m["run"].update(script_sha256=sha))
    caplog.set_level(logging.INFO)
    assert _reproduce(ref, tmp_path / "new") == 2
    assert message in caplog.text


def test_manifest_records_the_launched_script_hash(tmp_path, monkeypatch):
    import hashlib
    script = restart.runner_script_path("run_lmip.py")
    monkeypatch.setattr("sys.argv", [str(script)])
    m = restart.build_run_manifest({"runner": "x"}, config_kind="land")
    assert m["run"]["script_sha256"] == hashlib.sha256(
        script.read_bytes()).hexdigest()
    monkeypatch.setattr("sys.argv", ["not-a-file"])
    assert restart.build_run_manifest(
        {"runner": "x"}, config_kind="land")["run"]["script_sha256"] is None


def test_continuation_of_another_runner_is_refused(tmp_path):
    restart.write_run_manifest_on_root(tmp_path, {"runner": "a"},
                                       config_kind="land", runner_tag="other")
    with pytest.raises(RuntimeError, match="'other' runner"):
        restart.write_run_manifest_on_root(tmp_path, {"runner": "a"},
                                           config_kind="land",
                                           runner_tag="run_lmip")
    # and the other way round: a classified record, an unclassified rerun
    other = tmp_path / "b"
    restart.write_run_manifest_on_root(other, {"runner": "a"},
                                       config_kind="land",
                                       runner_tag="run_lmip")
    with pytest.raises(RuntimeError, match="'run_lmip' runner"):
        restart.write_run_manifest_on_root(other, {"runner": "a"},
                                           config_kind="land",
                                           runner_tag="other")


def test_resume_from_a_restart_of_a_completed_run_continues(tmp_path,
                                                            monkeypatch):
    import numpy as np

    out = tmp_path / "run"
    with pytest.raises(_Reached):
        _lmip(monkeypatch, out)
    restart.record_final_state_digest(out / restart.RUN_MANIFEST_FILENAME,
                                      {"T": np.ones(2)})       # run completed
    ckpt = out / "restart_day000001.npz"
    ckpt.write_bytes(b"x")
    with pytest.raises(_Reached):         # reached the restart load
        _lmip(monkeypatch, out, "--days", "2", "--restart-from", str(ckpt))
    m = restart.read_run_manifest(out)
    assert m["result"]["dataset_provenance"][0]["id"] == "restart_from"


def test_pre_argv_manifest_gets_only_the_drift_gate(tmp_path, monkeypatch):
    out = tmp_path / "run"
    with pytest.raises(_Reached):
        _lmip(monkeypatch, out, "--precip-rate", "3e-5")
    path = out / restart.RUN_MANIFEST_FILENAME
    old = json.loads(path.read_text())
    del old["run"]["argv"]                       # written before this change
    path.write_text(json.dumps(old))
    with pytest.raises(_Reached):
        _lmip(monkeypatch, out)


def test_rerun_into_same_dir_with_the_same_record_archives(tmp_path):
    def write(rec, tag="run_lmip"):
        restart.write_run_manifest_on_root(tmp_path, rec, config_kind="land",
                                           runner_tag=tag)
    write({"runner": "a"})
    write({"runner": "a"})
    assert len(list(tmp_path.glob("run_manifest.prev-*.json"))) == 1
    with pytest.raises(RuntimeError, match="resolved_config.runner"):
        write({"runner": "b"})
    # an unclassified runner may not take over a classified run's directory
    with pytest.raises(RuntimeError, match="one output directory holds one run"):
        write({"runner": "a"}, tag="run_unclassified")


def test_unclassified_runners_keep_the_drift_rule_only(tmp_path):
    for rec in ({"runner": "c"}, {"runner": "d"}):
        restart.write_run_manifest_on_root(tmp_path, rec, config_kind="land",
                                           runner_tag="run_unclassified")
    assert len(list(tmp_path.glob("run_manifest.prev-*.json"))) == 1


# --- restart drift gate ----------------------------------------------------

def test_lmip_restart_copied_alone_is_refused_before_the_manifest_write(
        tmp_path, monkeypatch):
    src = tmp_path / "old"
    with pytest.raises(_Reached):
        _lmip(monkeypatch, src)
    ckpt = src / "restart_day000001.npz"
    ckpt.write_bytes(b"x")
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    lone = fresh / ckpt.name
    lone.write_bytes(b"x")
    with pytest.raises(RuntimeError, match="code that wrote it is unknown"):
        _lmip(monkeypatch, fresh, "--restart-from", str(lone))
    assert not (fresh / restart.RUN_MANIFEST_FILENAME).exists()
    # beside its writing run's manifest: same code continues ...
    with pytest.raises(_Reached):
        _lmip(monkeypatch, tmp_path / "next", "--restart-from", str(ckpt))
    # ... other code is refused, unless explicitly allowed
    _pin(monkeypatch, SHA_B)
    with pytest.raises(RuntimeError, match=restart.ALLOW_VERSION_DRIFT_ENV):
        _lmip(monkeypatch, tmp_path / "next2", "--restart-from", str(ckpt))
    monkeypatch.setenv(restart.ALLOW_VERSION_DRIFT_ENV, "1")
    with pytest.raises(_Reached):
        _lmip(monkeypatch, tmp_path / "next3", "--restart-from", str(ckpt))


def test_restart_version_drift_searches_the_given_dirs(tmp_path):
    ckpt = tmp_path / "restarts" / "r.npz"
    ckpt.parent.mkdir()
    ckpt.write_bytes(b"")
    reasons = restart.restart_version_drift(ckpt, [ckpt.parent, tmp_path])
    assert reasons and "unknown" in reasons[0] and str(tmp_path) in reasons[0]
    restart.write_run_manifest(tmp_path, {"runner": "x"}, config_kind="land")
    assert restart.restart_version_drift(ckpt, [ckpt.parent, tmp_path]) == []
    assert restart.restart_version_drift(tmp_path / "missing.npz", [tmp_path]) == []


# --- multi-process records and the collective vote -------------------------

class _FakeComm:
    def __init__(self, rank, size, root_failed=False):
        self._rank, self._size, self._failed = rank, size, root_failed

    def Get_rank(self):
        return self._rank

    def Get_size(self):
        return self._size

    def bcast(self, value, root=0):
        return self._failed


def _fake_mpi(monkeypatch, comm):
    mod = types.SimpleNamespace(Is_initialized=lambda: True, COMM_WORLD=comm)
    monkeypatch.setitem(sys.modules, "mpi4py.MPI", mod)


def test_manifest_records_the_mpi_world_size(tmp_path, monkeypatch):
    _fake_mpi(monkeypatch, _FakeComm(0, 2))
    m = restart.build_run_manifest({"runner": "x"}, config_kind="land")
    assert m["reproducibility"]["n_processes"] == 2
    assert m["run"]["argv"] == list(sys.argv)
    import os
    assert m["run"]["cwd"] == os.getcwd()


def test_mpi_root_failure_is_broadcast(tmp_path, monkeypatch):
    sent = []

    class _Root(_FakeComm):
        def bcast(self, value, root=0):
            sent.append(value)
            return value

    _fake_mpi(monkeypatch, _Root(0, 2))

    def _boom(*a, **kw):
        raise OSError("disk full")

    monkeypatch.setattr(restart, "write_run_manifest", _boom)
    with pytest.raises(RuntimeError, match="disk full"):
        restart.write_run_manifest_on_root(tmp_path, {"runner": "x"},
                                           config_kind="land")
    assert sent == [True]


def test_mpi_non_root_rank_does_not_write_and_raises(tmp_path, monkeypatch):
    _fake_mpi(monkeypatch, _FakeComm(1, 2, root_failed=True))
    with pytest.raises(RuntimeError, match="failed on process 0"):
        restart.write_run_manifest_on_root(tmp_path, {"runner": "x"},
                                           config_kind="land")
    assert not (tmp_path / restart.RUN_MANIFEST_FILENAME).exists()


# --- argv rewrite ----------------------------------------------------------

def test_rewrite_flag_argv_forms():
    f = ("--output", "-o")
    assert restart.rewrite_flag_argv(["--output", "a", "--x"], f, "N") == [
        "--output", "N", "--x"]
    assert restart.rewrite_flag_argv(["--output=a"], f, "N") == ["--output=N"]
    assert restart.rewrite_flag_argv(["-o", "a", "--output=b"], f, "N") == [
        "-o", "N", "--output=N"]
    assert restart.rewrite_flag_argv(["--x", "1"], f, "N") == [
        "--x", "1", "--output", "N"]
    assert restart.rewrite_flag_argv(["--x", "--", "--output"], f, "N") == [
        "--x", "--output", "N", "--", "--output"]
    for bad in (["--out", "a"], ["--outp=a"], ["-oa"], ["-o=a"], ["--output"]):
        with pytest.raises(ValueError):
            restart.rewrite_flag_argv(bad, f, "N")


# --- legoesm reproduce -----------------------------------------------------

def _reproduce(manifest, out, *, check=False, allow=False):
    from legoesm.cli import cmd_reproduce
    with pytest.raises(SystemExit) as exc:
        cmd_reproduce(argparse.Namespace(manifest=str(manifest), check=check,
                                         output=str(out) if out else None,
                                         allow_version_drift=allow))
    return exc.value.code


def _edit(path: Path, fn) -> None:
    m = json.loads(path.read_text())
    fn(m)
    path.write_text(json.dumps(m))


def _verifiable(m) -> None:
    """Record the runner script's content, as a real launch of it would."""
    m["run"]["script_sha256"] = restart.file_sha256(
        restart.runner_script_path("run_lmip.py"))


def test_reproduce_refuses_a_land_manifest_across_version_drift(
        tmp_path, monkeypatch, caplog):
    ref = _reference(monkeypatch, tmp_path)
    _pin(monkeypatch, SHA_B)
    caplog.set_level(logging.INFO)
    # --output inside the reference: if the drift gate let this through the
    # output guard would exit instead, without the drift refusal line.
    assert _reproduce(ref, ref.parent / "x") == 2
    assert "Refusing to reproduce across version drift" in caplog.text
    assert f"recorded commit {SHA_A[:12]}" in caplog.text
    caplog.clear()
    assert _reproduce(ref, ref.parent / "x", allow=True) == 2
    assert "overlaps the reference run's output" in caplog.text
    caplog.clear()
    assert _reproduce(ref, tmp_path, allow=True) == 2     # an ancestor too
    assert "overlaps the reference run's output" in caplog.text


@pytest.mark.parametrize("edit, message", [
    (lambda m: m["run"].pop("argv"), "does not record the command line"),
    (lambda m: m["run"].update(cwd="/nonexistent/dir"), "launch directory"),
    (lambda m: m["reproducibility"].update(n_processes=2), "used 2 processes"),
    (lambda m: m["reproducibility"].pop("n_processes"),
     "does not record the process count"),
    (lambda m: m["run"]["argv"].__setitem__(0, "wrapper.py"),
     "not as the run_lmip.py runner script"),
])
def test_reproduce_refusals(tmp_path, monkeypatch, caplog, edit, message):
    ref = _reference(monkeypatch, tmp_path)
    _edit(ref, edit)
    caplog.set_level(logging.INFO)
    assert _reproduce(ref, tmp_path / "new") == 2
    assert message in caplog.text


def test_reproduce_refuses_an_output_holding_a_manifest(tmp_path, monkeypatch,
                                                       caplog):
    ref = _reference(monkeypatch, tmp_path)
    busy = tmp_path / "busy"
    restart.write_run_manifest(busy, {"runner": "x"}, config_kind="land")
    caplog.set_level(logging.INFO)
    assert _reproduce(ref, busy) == 2
    assert "already holds a run manifest" in caplog.text


def test_renamed_reference_still_guards_the_runner_manifest(tmp_path,
                                                           monkeypatch, caplog):
    """The rerun writes run_manifest.json whatever the reference is called."""
    ref = _reference(monkeypatch, tmp_path)
    renamed = ref.parent / "land_ref.json"
    shutil.copyfile(ref, renamed)
    busy = tmp_path / "busy"
    restart.write_run_manifest(busy, {"runner": "x"}, config_kind="land")
    caplog.set_level(logging.INFO)
    assert _reproduce(renamed, busy) == 2
    assert "already holds a run manifest" in caplog.text


def test_automatic_output_inside_the_reference_is_refused(tmp_path,
                                                          monkeypatch, caplog):
    ref = _reference(monkeypatch, tmp_path)
    inside = ref.parent / "tmp"
    inside.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(inside))
    caplog.set_level(logging.INFO)
    assert _reproduce(ref, None) == 2
    assert "overlaps the reference run's output" in caplog.text
    assert not any(inside.iterdir())


def test_reproduce_check_without_a_reference_digest_refuses_first(
        tmp_path, monkeypatch, caplog):
    ref = _reference(monkeypatch, tmp_path)       # stopped before the digest
    caplog.set_level(logging.INFO)
    assert _reproduce(ref, tmp_path / "new", check=True) == 2
    assert "Refusing to reproduce --check" in caplog.text
    assert not (tmp_path / "new").exists()        # nothing was launched


def test_runner_without_replay_support_is_validated_only(tmp_path, monkeypatch,
                                                         caplog):
    from legoesm.cli import cmd_reproduce
    ref = restart.write_run_manifest(tmp_path / "c2", {"runner": "x"},
                                     config_kind="ocean",
                                     runner_tag="run_omip_core2")
    caplog.set_level(logging.INFO)
    ns = dict(manifest=str(ref), output=None, allow_version_drift=False)
    cmd_reproduce(argparse.Namespace(check=False, **ns))     # exits normally
    assert "VALIDATED" in caplog.text
    with pytest.raises(SystemExit) as exc:
        cmd_reproduce(argparse.Namespace(check=True, **ns))
    assert exc.value.code == 2


def test_multiprocess_hint_carries_the_drift_override(tmp_path, monkeypatch,
                                                      caplog):
    ref = _reference(monkeypatch, tmp_path)
    _edit(ref, lambda m: m["reproducibility"].update(n_processes=2))
    caplog.set_level(logging.INFO)
    assert _reproduce(ref, tmp_path / "new", allow=True) == 2
    assert f"{restart.ALLOW_VERSION_DRIFT_ENV}=1" in caplog.text


def test_multiprocess_refusal_comes_after_the_script_check(tmp_path,
                                                           monkeypatch, caplog):
    ref = _reference(monkeypatch, tmp_path)
    _edit(ref, lambda m: (m["reproducibility"].update(n_processes=2),
                          m["run"].update(script_sha256=None)))
    caplog.set_level(logging.INFO)
    assert _reproduce(ref, tmp_path / "new") == 2
    assert "does not record the launched script" in caplog.text


def test_atmosphere_manifest_with_a_runner_tag_is_refused(tmp_path, caplog):
    from legoesm.driver.config import ExperimentConfig
    path = restart.write_run_manifest(tmp_path / "a", ExperimentConfig(),
                                      runner_tag="run_lmip")
    caplog.set_level(logging.INFO)
    assert _reproduce(path, tmp_path / "new") == 2
    assert "ambiguous" in caplog.text


def test_record_replaced_during_the_check_is_checked_again(tmp_path,
                                                          monkeypatch):
    """A concurrent start that puts a different run's record in place between
    the check and the archive is not archived unchecked."""
    import os
    out = tmp_path / "run"
    with pytest.raises(_Reached):
        _lmip(monkeypatch, out)
    other = tmp_path / "other"
    with pytest.raises(_Reached):
        _lmip(monkeypatch, other, "--precip-rate", "3e-5")
    intruder = (other / restart.RUN_MANIFEST_FILENAME).read_bytes()
    real_link = os.link
    swapped = []

    def link(src, dst, *a, **kw):
        if not swapped and str(src).endswith(restart.RUN_MANIFEST_FILENAME):
            Path(src).write_bytes(intruder)          # the concurrent start
            swapped.append(True)
        return real_link(src, dst, *a, **kw)
    monkeypatch.setattr(os, "link", link)
    with pytest.raises(RuntimeError, match="cli_args.precip_rate"):
        _lmip(monkeypatch, out, "--days", "2")
    assert swapped
    assert (out / restart.RUN_MANIFEST_FILENAME).read_bytes() == intruder
    assert not list(out.glob("run_manifest.prev-*.json"))


def test_failed_archive_read_puts_the_record_back(tmp_path, monkeypatch):
    out = tmp_path / "run"
    with pytest.raises(_Reached):
        _lmip(monkeypatch, out)
    live = out / restart.RUN_MANIFEST_FILENAME
    before = live.read_bytes()
    real = Path.read_bytes

    def read_bytes(self):
        if ".prev-" in self.name:
            raise OSError("archive unreadable")
        return real(self)
    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    with pytest.raises(RuntimeError, match="archive unreadable"):
        _lmip(monkeypatch, out, "--days", "2")
    monkeypatch.setattr(Path, "read_bytes", real)
    assert live.read_bytes() == before
    assert not list(out.glob("run_manifest.prev-*.json"))


def test_copied_reference_still_guards_the_original_output(tmp_path,
                                                           monkeypatch, caplog):
    """A manifest copied away from its run still refuses an output inside the
    directory the run wrote."""
    ref = _reference(monkeypatch, tmp_path)
    copy = tmp_path / "archive" / "land.json"
    copy.parent.mkdir()
    shutil.copyfile(ref, copy)
    # two processes: without the guard this exits at the process-count refusal
    _edit(copy, lambda m: m["reproducibility"].update(n_processes=2))
    caplog.set_level(logging.INFO)
    assert _reproduce(copy, ref.parent / "inside") == 2
    assert "overlaps the reference run's output" in caplog.text


def _run_lmip_subprocess(out: Path, config: Path, cwd: Path, *extra) -> None:
    script = restart.runner_script_path("run_lmip.py")
    subprocess.run([sys.executable, str(script), "--config", config.name,
                    "--lat", "45.0", "--days", "1", "--output", str(out),
                    *extra], cwd=cwd, check=True)


@pytest.mark.slow
def test_reproduce_lmip_end_to_end(tmp_path, monkeypatch, caplog):
    """Real rerun through the script: digests match; a changed file the rerun
    reads (its --config) is refused.  Both legs allow version drift so a
    dirty developer tree does not refuse them; the closure does not depend on
    it."""
    monkeypatch.undo()                      # real provenance in both processes
    cfg = tmp_path / "lmip.yaml"
    cfg.write_text("precip_rate: 2.0e-5\n")
    _run_lmip_subprocess(tmp_path / "ref", cfg, tmp_path)
    ref = tmp_path / "ref" / restart.RUN_MANIFEST_FILENAME
    caplog.set_level(logging.INFO)
    assert _reproduce(ref, tmp_path / "rerun", check=True, allow=True) == 0
    assert "MATCH" in caplog.text
    assert _reproduce(ref, tmp_path / "plain", allow=True) == 0
    renamed = ref.parent / "land_ref.json"          # a shared/renamed copy
    shutil.copyfile(ref, renamed)
    assert _reproduce(renamed, tmp_path / "renamed", check=True,
                      allow=True) == 0
    cfg.write_text("precip_rate: 3.0e-5\n")
    assert _reproduce(ref, tmp_path / "rerun2", check=True, allow=True) != 0
    assert not (tmp_path / "rerun2" / restart.RUN_MANIFEST_FILENAME).exists()


@pytest.mark.slow
def test_plain_reproduce_fails_when_the_rerun_did_not_complete(tmp_path,
                                                               monkeypatch,
                                                               caplog):
    """A wallclock-budget exit returns 0 without a final digest; plain
    reproduce must not report it as reproduced."""
    monkeypatch.undo()
    cfg = tmp_path / "lmip.yaml"
    cfg.write_text("precip_rate: 2.0e-5\n")
    _run_lmip_subprocess(tmp_path / "ref", cfg, tmp_path,
                         "--max-wallclock-seconds", "1e-9",
                         "--restart-buffer-seconds", "0")
    ref = tmp_path / "ref" / restart.RUN_MANIFEST_FILENAME
    caplog.set_level(logging.INFO)
    assert _reproduce(ref, tmp_path / "rerun", allow=True) == 1
    assert "did not complete" in caplog.text


def test_write_run_manifest_refuses_kwargs_it_would_ignore(tmp_path):
    with pytest.raises(TypeError, match="ignored"):
        restart.write_run_manifest(tmp_path, {}, manifest={"x": 1},
                                   config_kind="land")
