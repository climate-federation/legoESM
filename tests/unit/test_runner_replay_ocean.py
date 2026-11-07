"""`legoesm reproduce` / continuation / restart drift gate for the ocean
runners: run_omip, run_dino, and the CORE2 restart source-drift refusal
(docs/architecture/reproducibility.md, P0)."""
from __future__ import annotations

import argparse
import inspect
import logging

import numpy as np
import pytest

import legoesm.driver.restart as restart
from legoesm.io.git_provenance import GitProvenance

SHA_A = "a" * 40
SHA_B = "b" * 40


def _pin(monkeypatch, commit: str) -> None:
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
    pass


def _stop(*a, **kw):
    raise _Reached()


# --- run_omip ---------------------------------------------------------------

def _omip(monkeypatch, out, *extra, grid_flag="latlon"):
    from scripts.run import run_omip
    monkeypatch.setattr(run_omip, "_run_omip_loop", _stop)
    monkeypatch.setattr(run_omip, "_load_restart", _stop)
    args = run_omip.parse_args([
        "--grid", grid_flag, "--resolution", "4x8", "--nlev", "4",
        "--dt", "10800", "--days", "1", "--output", str(out), *extra])
    run_omip.run_omip_single("latlon", args)


def _omip_manifest(out):
    return out / "latlon" / "4x8" / restart.RUN_MANIFEST_FILENAME


def test_omip_manifest_records_the_command_line_options(tmp_path, monkeypatch):
    with pytest.raises(_Reached):
        _omip(monkeypatch, tmp_path / "ref")
    rc = restart.read_run_manifest(_omip_manifest(tmp_path / "ref"))[
        "config"]["resolved_config"]
    assert rc["cli_args"]["nlev"] == 4 and rc["cli_args"]["grid"] == "latlon"


def test_omip_all_grids_reference_replays_one_grid(tmp_path, monkeypatch):
    with pytest.raises(_Reached):
        _omip(monkeypatch, tmp_path / "ref", grid_flag="all")
    monkeypatch.setenv(restart.REPRODUCE_REFERENCE_ENV,
                       str(_omip_manifest(tmp_path / "ref")))
    monkeypatch.setenv(restart.REPRODUCE_OUTPUT_ENV, str(tmp_path))
    with pytest.raises(_Reached):           # --grid swapped to latlon: passes
        _omip(monkeypatch, tmp_path / "new")
    with pytest.raises(RuntimeError, match="cli_args.H_max"):
        _omip(monkeypatch, tmp_path / "new2", "--H-max", "4000")


def test_omip_rerun_resolving_differently_is_refused(tmp_path, monkeypatch):
    with pytest.raises(_Reached):
        _omip(monkeypatch, tmp_path / "ref")
    monkeypatch.setenv(restart.REPRODUCE_REFERENCE_ENV,
                       str(_omip_manifest(tmp_path / "ref")))
    monkeypatch.setenv(restart.REPRODUCE_OUTPUT_ENV, str(tmp_path / "new"))
    with pytest.raises(RuntimeError, match="does not resolve to the reference"):
        _omip(monkeypatch, tmp_path / "new", "--seed", "3")
    assert not _omip_manifest(tmp_path / "new").exists()


def test_omip_continuation_controls(tmp_path, monkeypatch):
    out = tmp_path / "run"
    with pytest.raises(_Reached):
        _omip(monkeypatch, out)
    with pytest.raises(_Reached):
        _omip(monkeypatch, out, "--days", "5", "--checkpoint-days", "1",
              "--max-wallclock-seconds", "100",
              "--restart-buffer-seconds", "5", "--no-final-snapshot")
    # a restart from this run's own restarts/ continues it
    rdir = _omip_manifest(out).parent / "restarts"
    rdir.mkdir()
    (rdir / "restart_day000001.npz").write_bytes(b"x")
    with pytest.raises(_Reached):
        _omip(monkeypatch, out, "--restart", str(rdir / "restart_day000001.npz"))
    with pytest.raises(RuntimeError, match="one output directory holds one run"):
        _omip(monkeypatch, out, "--H-max", "4000")
    # --diag-every also sets the JRA55 forcing block size: not a control
    with pytest.raises(RuntimeError, match="cli_args.diag_every"):
        _omip(monkeypatch, out, "--diag-every", "2")


def test_omip_restart_gate(tmp_path, monkeypatch):
    with pytest.raises(_Reached):
        _omip(monkeypatch, tmp_path / "old")
    rdir = _omip_manifest(tmp_path / "old").parent / "restarts"
    rdir.mkdir()
    ckpt = rdir / "restart_day000001.npz"
    ckpt.write_bytes(b"x")
    lone = tmp_path / "copied" / ckpt.name
    lone.parent.mkdir()
    lone.write_bytes(b"x")
    with pytest.raises(RuntimeError, match="code that wrote it is unknown"):
        _omip(monkeypatch, tmp_path / "a", "--restart", str(lone))
    with pytest.raises(_Reached):           # native layout: manifest above
        _omip(monkeypatch, tmp_path / "b", "--restart", str(ckpt))
    _pin(monkeypatch, SHA_B)
    with pytest.raises(RuntimeError, match=restart.ALLOW_VERSION_DRIFT_ENV):
        _omip(monkeypatch, tmp_path / "c", "--restart", str(ckpt))


def test_omip_real_manifest_replay_command(tmp_path, monkeypatch, caplog):
    """A manifest written by run_omip itself, launched without --grid: the
    replay command appends --grid <recorded grid> and the new output."""
    monkeypatch.setattr("sys.argv", ["run_omip.py", "--resolution", "4x8"])
    with pytest.raises(_Reached):
        _omip(monkeypatch, tmp_path / "ref", grid_flag="all")
    path = _omip_manifest(tmp_path / "ref")
    m = restart.read_run_manifest(path)
    m["run"]["script_sha256"] = restart.file_sha256(
        restart.runner_script_path("run_omip.py"))
    m["reproducibility"]["n_processes"] = 2
    path.write_text(__import__("json").dumps(m))
    caplog.set_level(logging.INFO)
    assert _reproduce(path, tmp_path / "new") == 2
    assert f"--resolution 4x8 --output {tmp_path / 'new'} --grid latlon" in (
        caplog.text)


def test_omip_final_digest_uses_the_digest_trees(tmp_path, monkeypatch):
    """run_omip_single digests what _final_digest_trees returns (the ocean
    plus, when carried, the final sea ice) after a real loop."""
    from scripts.run import run_omip
    seen = {}
    monkeypatch.setattr(run_omip, "_final_digest_trees",
                        lambda state, js: (seen.setdefault("js", js),
                                           [state, "ICE"])[1])
    monkeypatch.setattr(restart, "record_final_state_digest",
                        lambda path, *trees: seen.setdefault("trees", trees))
    args = run_omip.parse_args([
        "--grid", "latlon", "--resolution", "4x8", "--nlev", "4",
        "--dt", "10800", "--days", "1", "--output", str(tmp_path),
        "--no-final-snapshot", "--checkpoint-days", "0"])
    run_omip.run_omip_single("latlon", args)
    assert "js" in seen and seen["trees"][1] == "ICE"


def test_dino_replay_relocates_the_timing_file(tmp_path, monkeypatch, caplog):
    path = _synthetic(monkeypatch, tmp_path, "run_dino", "run_dino.py",
                      {"cli_args": {"timing_jsonl": "/ref/t.jsonl"}},
                      ["--timing-jsonl", "/ref/t.jsonl", "--output-dir", "o"])
    caplog.set_level(logging.INFO)
    assert _reproduce(path, tmp_path / "new") == 2      # multi-process refusal
    assert f"--timing-jsonl {tmp_path / 'new' / 't.jsonl'}" in caplog.text
    assert "/ref/t.jsonl" not in caplog.text.split("\n", 1)[-1].split(
        "run_dino.py", 1)[-1]


def test_sharded_final_ice_is_gathered_for_the_digest():
    import inspect
    from scripts.run import run_omip
    js = {"ice_state_final": "SHARDED"}
    run_omip._gather_final_ice(js, lambda ice: f"gathered {ice}")
    assert js["ice_state_final"] == "gathered SHARDED"
    for none_js in ({}, None):
        run_omip._gather_final_ice(none_js, lambda ice: 1 / 0)  # not called
    run_omip._gather_final_ice(js, None)                        # serial run
    src = inspect.getsource(run_omip.run_omip_single)
    after = src.split("state = spmd_gather(state)", 1)[1]
    assert after.lstrip().startswith(
        "_gather_final_ice(jra55_state, spmd_gather_ice)")


def test_dino_timing_flag_without_its_record_is_refused(tmp_path, monkeypatch,
                                                       caplog):
    path = _synthetic(monkeypatch, tmp_path, "run_dino", "run_dino.py",
                      {"cli_args": {}},
                      ["--timing-jsonl=/ref/t.jsonl", "--output-dir", "o"])
    caplog.set_level(logging.INFO)
    assert _reproduce(path, tmp_path / "new") == 2
    assert "does not record cli_args.timing_jsonl" in caplog.text


def test_omip_digest_covers_the_final_sea_ice():
    from scripts.run import run_omip
    assert run_omip._final_digest_trees("S", None) == ["S"]
    assert run_omip._final_digest_trees("S", {}) == ["S"]
    assert run_omip._final_digest_trees("S", {"ice_state_final": "I"}) == [
        "S", "I"]


def _reproduce(manifest, out, *, check=False):
    from legoesm.cli import cmd_reproduce
    with pytest.raises(SystemExit) as exc:
        cmd_reproduce(argparse.Namespace(manifest=str(manifest), check=check,
                                         output=str(out) if out else None,
                                         allow_version_drift=False))
    return exc.value.code


def _synthetic(monkeypatch, tmp_path, tag, script, resolved, argv_tail,
               n_processes=2, subdir="."):
    monkeypatch.setattr("sys.argv", [f"/x/{script}", *argv_tail])
    path = restart.write_run_manifest(tmp_path / "ref" / subdir, resolved,
                                      config_kind="ocean", runner_tag=tag)
    m = restart.read_run_manifest(path)
    m["reproducibility"]["n_processes"] = n_processes
    # as a real launch of the runner script would record
    m["run"]["script_sha256"] = restart.file_sha256(
        restart.runner_script_path(script))
    path.write_text(__import__("json").dumps(m))
    return path


def test_omip_reproduce_swaps_the_grid(tmp_path, monkeypatch, caplog):
    path = _synthetic(monkeypatch, tmp_path, "run_omip", "run_omip.py",
                      {"grid_type": "latlon"},
                      ["--grid", "all", "--output", "o"], subdir="latlon/4x8")
    caplog.set_level(logging.INFO)
    assert _reproduce(path, tmp_path / "new") == 2      # multi-process refusal
    assert "--grid latlon" in caplog.text and "--grid all" not in caplog.text
    assert f"--output {tmp_path / 'new'}" in caplog.text


# --- run_dino ---------------------------------------------------------------

def _dino(monkeypatch, out, *extra):
    from scripts.run import run_dino
    monkeypatch.setattr(run_dino, "_save_snapshot", _stop)
    monkeypatch.setattr(run_dino, "_load_restart", _stop)
    monkeypatch.setattr("sys.argv", [
        "run_dino.py", "--n-lon", "10", "--days", "1",
        "--output-dir", str(out), *extra])
    run_dino.main()


def test_dino_restart_gate(tmp_path, monkeypatch):
    with pytest.raises(_Reached):
        _dino(monkeypatch, tmp_path / "old")
    snap = tmp_path / "old" / "snapshots" / "snapshot_00001.npz"
    snap.write_bytes(b"x")
    lone = tmp_path / "copied" / snap.name
    lone.parent.mkdir()
    lone.write_bytes(b"x")
    with pytest.raises(RuntimeError, match="code that wrote it is unknown"):
        _dino(monkeypatch, tmp_path / "a", "--restart-from", str(lone))
    with pytest.raises(_Reached):
        _dino(monkeypatch, tmp_path / "b", "--restart-from", str(snap))


def test_dino_rerun_closure_and_continuation(tmp_path, monkeypatch):
    with pytest.raises(_Reached):
        _dino(monkeypatch, tmp_path / "ref")
    with pytest.raises(_Reached):                 # continuation controls
        _dino(monkeypatch, tmp_path / "ref", "--days", "3",
              "--snapshot-every-days", "2", "--allow-multiyear")
    meta = next((tmp_path / "ref").glob("run_metadata*.json"))
    before = meta.read_bytes()
    with pytest.raises(RuntimeError, match="one output directory holds one run"):
        _dino(monkeypatch, tmp_path / "ref", "--seed", "4")
    assert meta.read_bytes() == before      # the refused run wrote nothing
    monkeypatch.setenv(restart.REPRODUCE_REFERENCE_ENV,
                       str(tmp_path / "ref" / restart.RUN_MANIFEST_FILENAME))
    monkeypatch.setenv(restart.REPRODUCE_OUTPUT_ENV, str(tmp_path / "new"))
    with pytest.raises(RuntimeError, match="cli_args.seed"):
        _dino(monkeypatch, tmp_path / "new", "--days", "3",
              "--snapshot-every-days", "2", "--allow-multiyear", "--seed", "4")
    with pytest.raises(_Reached):
        _dino(monkeypatch, tmp_path / "new", "--days", "3",
              "--snapshot-every-days", "2", "--allow-multiyear")


@pytest.mark.parametrize("check", [True, False])
def test_dino_multicontroller_replay_is_refused(tmp_path, monkeypatch, caplog,
                                                check):
    path = _synthetic(monkeypatch, tmp_path, "run_dino", "run_dino.py",
                      {"cli_args": {"multicontroller": True}}, [],
                      n_processes=1)
    caplog.set_level(logging.INFO)
    assert _reproduce(path, tmp_path / "new", check=check) == 2
    assert "records no final digest" in caplog.text
    assert not (tmp_path / "new").exists()      # nothing was launched


# --- run_omip_core2 ---------------------------------------------------------

def test_core2_restart_source_drift_is_refused_unless_allowed(tmp_path,
                                                              monkeypatch):
    from scripts.run import run_omip_core2 as c2
    clean = "c" * 40
    c2._refuse_restart_source_drift(clean, clean, tmp_path)       # same code
    for parent in ("d" * 40, None, c2._SOURCE_REV_UNAVAILABLE,
                   clean + "-dirty"):
        this = clean + "-dirty" if parent == clean + "-dirty" else clean
        with pytest.raises(RuntimeError, match=restart.ALLOW_VERSION_DRIFT_ENV):
            c2._refuse_restart_source_drift(parent, this, tmp_path)
    monkeypatch.setenv(restart.ALLOW_VERSION_DRIFT_ENV, "1")
    c2._refuse_restart_source_drift("d" * 40, clean, tmp_path)    # warns only


def test_core2_main_routes_restart_drift_through_the_refusal():
    from scripts.run import run_omip_core2 as c2
    src = inspect.getsource(c2.main)
    assert "_refuse_restart_source_drift(_rs_meta.get(\"sha\")" in src
    assert "print(_drift_note" not in src
