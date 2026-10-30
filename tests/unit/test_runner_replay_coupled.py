"""`legoesm reproduce` / continuation / resume gate for the coupled runner
(run_coupled; docs/architecture/reproducibility.md, P0).  Setup is stubbed:
these tests exercise the record, its timing and the refusals, not physics."""
from __future__ import annotations

import argparse
import json
import logging

import pytest
from pathlib import Path

import legoesm.driver.restart as restart
from legoesm.driver.coupled_esm_driver import CoupledESMDriver
from legoesm.io.git_provenance import GitProvenance

SHA_A = "a" * 40


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in (restart.ALLOW_VERSION_DRIFT_ENV, restart.REPRODUCE_REFERENCE_ENV,
                restart.REPRODUCE_OUTPUT_ENV):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(
        restart, "git_provenance",
        lambda anchor: GitProvenance(anchor=str(anchor), root="/repo",
                                     commit=SHA_A, ref="", dirty=False))


class _Ran(Exception):
    """CoupledESMDriver.run was reached (after setup and the manifest)."""


class _SetupCalled(Exception):
    pass


def _stub_setup(monkeypatch, atm_seed=None):
    """Setup writes only the atmosphere manifest, as ModelDriver's would."""
    def setup(self):
        cfg = self.atm_config if atm_seed is None else self.atm_config._replace(
            seed=atm_seed)
        restart.write_run_manifest(self.output_dir, cfg)
    monkeypatch.setattr(CoupledESMDriver, "setup", setup)


def _stop_at_run(monkeypatch, seen=None):
    def run(self, *a, **kw):
        if seen is not None:
            seen.append(self)
        raise _Ran()
    monkeypatch.setattr(CoupledESMDriver, "run", run)


def _coupled(out, *extra):
    from scripts.run import run_coupled
    run_coupled.main(["--output", str(out), "--days", "1", *extra])


def _complete_physics(out):
    """Give the atmosphere manifest at *out* a complete physics record."""
    restart.record_resolved_physics(
        out / restart.RUN_MANIFEST_FILENAME,
        restart.resolved_physics_record(
            dict.fromkeys(restart.RESOLVED_PHYSICS_KEYS)))


def _reference(monkeypatch, tmp_path):
    _stub_setup(monkeypatch)
    _stop_at_run(monkeypatch)
    ref = tmp_path / "ref"
    with pytest.raises(_Ran):
        _coupled(ref)
    _complete_physics(ref)
    return ref


def _under_reproduce(monkeypatch, ref, out_root):
    monkeypatch.setenv(restart.REPRODUCE_REFERENCE_ENV,
                       str(ref / "coupled" / restart.RUN_MANIFEST_FILENAME))
    monkeypatch.setenv(restart.REPRODUCE_OUTPUT_ENV, str(out_root))


def test_coupled_manifest_is_written_after_setup_with_component_configs(
        tmp_path, monkeypatch):
    def setup(self):
        assert not (self.output_dir / "coupled" /
                    restart.RUN_MANIFEST_FILENAME).exists()
        raise _SetupCalled()
    monkeypatch.setattr(CoupledESMDriver, "setup", setup)
    with pytest.raises(_SetupCalled):
        _coupled(tmp_path / "a")
    ref = _reference(monkeypatch, tmp_path)
    m = restart.read_run_manifest(ref / "coupled")
    assert set(m["config"]["resolved_physics"]) == {
        "ocean", "land", "tile", "coupler", "sea_ice", "lake"}
    assert m["config"]["resolved_physics"]["lake"]["__type__"].endswith(
        ":LakeConfig")


def test_coupled_record_carries_the_resolved_ocean_flux_auto(tmp_path,
                                                           monkeypatch):
    """#1320: the coupled record's atmosphere config records the boolean the
    atmosphere's setup resolved for ocean_flux_on_ocean_surface AUTO, the
    same value as the atmosphere's own manifest (never null)."""
    def setup(self):
        self._atm._resolve_ocean_flux_surface()
        restart.write_run_manifest(self.output_dir, self._atm._input_config)
    monkeypatch.setattr(CoupledESMDriver, "setup", setup)
    _stop_at_run(monkeypatch)
    out = tmp_path / "r"
    with pytest.raises(_Ran):
        _coupled(out)
    atm = restart.read_run_manifest(out)["config"]["resolved_config"]
    cpl = restart.read_run_manifest(out / "coupled")["config"][
        "resolved_config"]["atmosphere"]
    assert isinstance(atm["ocean_flux_on_ocean_surface"], bool)
    assert (cpl["ocean_flux_on_ocean_surface"]
            == atm["ocean_flux_on_ocean_surface"])


def test_clean_coupled_rerun_passes_and_forwards_the_atmosphere_record(
        tmp_path, monkeypatch):
    ref = _reference(monkeypatch, tmp_path)
    _under_reproduce(monkeypatch, ref, tmp_path / "new")
    seen = []
    _stop_at_run(monkeypatch, seen)
    with pytest.raises(_Ran):
        _coupled(tmp_path / "new")
    phys = restart.read_run_manifest(ref)["config"]["resolved_physics"]
    assert seen[0]._atm._expected_resolved_physics == phys


def test_coupled_rerun_with_other_component_configs_is_refused(tmp_path,
                                                               monkeypatch):
    from legoesm.coupler.lake.config import LakeConfig

    ref = _reference(monkeypatch, tmp_path)
    _under_reproduce(monkeypatch, ref, tmp_path / "new")
    real = CoupledESMDriver.resolved_component_configs
    field = next(f for f in LakeConfig._fields
                 if isinstance(getattr(LakeConfig(), f), float))

    def changed(self):
        cfgs = real(self)
        lake = cfgs["lake"]
        return {**cfgs, "lake": lake._replace(**{field: 2 * getattr(lake, field)
                                                 + 1})}
    monkeypatch.setattr(CoupledESMDriver, "resolved_component_configs", changed)
    with pytest.raises(RuntimeError, match=f"resolved_physics.lake.{field}"):
        _coupled(tmp_path / "new")


def test_coupled_rerun_with_another_atmosphere_is_refused(tmp_path,
                                                          monkeypatch):
    ref = _reference(monkeypatch, tmp_path)
    _under_reproduce(monkeypatch, ref, tmp_path / "new")
    _stub_setup(monkeypatch, atm_seed=99)
    with pytest.raises(RuntimeError, match="atmosphere configuration differs"):
        _coupled(tmp_path / "new")


def test_reference_without_an_atmosphere_physics_record_is_refused(
        tmp_path, monkeypatch):
    _stub_setup(monkeypatch)
    _stop_at_run(monkeypatch)
    ref = tmp_path / "ref"
    with pytest.raises(_Ran):
        _coupled(ref)                       # no physics record added
    _under_reproduce(monkeypatch, ref, tmp_path / "new")
    with pytest.raises(RuntimeError, match="no complete resolved-physics"):
        _coupled(tmp_path / "new")


def test_coupled_continuation_controls(tmp_path, monkeypatch):
    _stub_setup(monkeypatch)
    _stop_at_run(monkeypatch)
    out = tmp_path / "run"
    with pytest.raises(_Ran):
        _coupled(out)
    with pytest.raises(_Ran):
        _coupled(out, "--days", "3", "--checkpoint-days", "1",
                 "--max-wallclock-hours", "2")
    with pytest.raises(RuntimeError, match="one output directory holds one run"):
        _coupled(out, "--co2-init", "400")
    # the coupling interval is science, not a continuation control
    with pytest.raises(RuntimeError, match="diag_days"):
        _coupled(out, "--diag-days", "2")


def test_resume_checkpoint_code_is_checked_before_setup(tmp_path, monkeypatch):
    def setup(self):
        raise _SetupCalled()
    monkeypatch.setattr(CoupledESMDriver, "setup", setup)
    out = tmp_path / "run"
    out.mkdir()
    (out / "checkpoint_day_0001.npz").write_bytes(b"x")
    with pytest.raises(RuntimeError, match="code that wrote it is unknown"):
        _coupled(out, "--resume")


def test_resume_without_the_coupled_checkpoint_is_refused(tmp_path,
                                                          monkeypatch):
    from legoesm.driver.model_driver import ModelDriver

    _stub_setup(monkeypatch)
    _stop_at_run(monkeypatch)
    out = tmp_path / "run"
    with pytest.raises(_Ran):
        _coupled(out)
    (out / "checkpoint_day_0001.npz").write_bytes(b"x")
    monkeypatch.setattr(ModelDriver, "load_checkpoint",
                        lambda self, p: (8, 1.0))
    record = out / "coupled" / restart.RUN_MANIFEST_FILENAME
    before = record.read_bytes()
    with pytest.raises(RuntimeError, match="No coupled checkpoint"):
        _coupled(out, "--resume")
    assert record.read_bytes() == before        # the record is untouched
    assert not list(record.parent.glob("run_manifest.prev-*.json"))
    # with it present the resume goes on to run
    monkeypatch.setattr(CoupledESMDriver, "load_coupled_checkpoint",
                        lambda self, *a, **kw: None)
    with pytest.raises(_Ran):
        _coupled(out, "--resume")


@pytest.mark.parametrize("status, digested", [
    ("COMPLETED", True), ("WALLCLOCK_EXIT", False), ("BLOWUP", False)])
def test_coupled_digest_is_recorded_only_for_a_completed_run(
        tmp_path, monkeypatch, status, digested):
    _stub_setup(monkeypatch)
    order = []

    def run(self, **kw):
        order.append("run")        # the atmosphere records its own digest
        self._atm.final_state_digest = restart.record_final_state_digest(
            self.output_dir / restart.RUN_MANIFEST_FILENAME,
            {"a": __import__("numpy").ones(3)})
        return status
    monkeypatch.setattr(CoupledESMDriver, "run", run)
    monkeypatch.setattr(CoupledESMDriver, "coupled_checkpoint_arrays",
                        lambda self: (order.append("digest"),
                                      {"x": __import__("numpy").ones(2)})[1])

    def stop(self):                         # the summary after the digest
        raise _Ran()
    monkeypatch.setattr(CoupledESMDriver, "_ocean_surface_KuvC", stop)
    with pytest.raises(_Ran):
        _coupled(tmp_path / "run")
    m = restart.read_run_manifest(tmp_path / "run" / "coupled")
    assert bool(m.get("result", {}).get("state_digest")) is digested
    assert order == (["run", "digest"] if digested else ["run"])
    atm = restart.read_run_manifest(tmp_path / "run")["result"]["state_digest"]
    assert (m["result"].get("companion_state_digests") ==
            ({restart.RUN_MANIFEST_FILENAME: atm} if digested else None))


def test_coupled_digest_never_pairs_an_older_atmosphere_digest(
        tmp_path, monkeypatch):
    """The atmosphere record already holds a digest from an earlier run (a
    same-config manifest is preserved); this run's atmosphere recorded none,
    so the pair is empty, not the stale digest."""
    import numpy as np
    _stub_setup(monkeypatch)
    out = tmp_path / "run"
    monkeypatch.setattr(CoupledESMDriver, "run",
                        lambda self, **kw: "COMPLETED")

    def stop(self):
        raise _Ran()
    monkeypatch.setattr(CoupledESMDriver, "_ocean_surface_KuvC", stop)

    def setup(self):                 # the earlier run's digest is there
        restart.write_run_manifest(self.output_dir, self.atm_config)
        restart.record_final_state_digest(
            self.output_dir / restart.RUN_MANIFEST_FILENAME, {"a": np.ones(3)})
    monkeypatch.setattr(CoupledESMDriver, "setup", setup)
    with pytest.raises(_Ran):
        _coupled(out)
    m = restart.read_run_manifest(out / "coupled")
    assert m["result"]["state_digest"]
    assert m["result"]["companion_state_digests"] == {}


def test_component_config_attributes_are_the_ones_setup_assigns():
    """resolved_component_configs reads these by name with a None fallback
    (a component a configuration leaves out); a rename must not turn the
    record into null on both sides of a comparison."""
    import inspect
    src = inspect.getsource(CoupledESMDriver)
    for attr in ("_ocean_model", "_coupler_cfg", "_land_cfg", "_tile_config"):
        assert f"self.{attr} = " in src, attr
        assert f'"{attr}"' in inspect.getsource(
            CoupledESMDriver.resolved_component_configs), attr


def test_resume_from_a_real_deck_is_recorded_and_refused(tmp_path,
                                                         monkeypatch, caplog):
    """A deck that sets resume: the recorded resolved value refuses replay."""
    _stub_setup(monkeypatch)
    _stop_at_run(monkeypatch)
    deck = tmp_path / "deck.yaml"
    deck.write_text("resume: true\n")
    ref = tmp_path / "ref"
    with pytest.raises(_Ran):
        _coupled(ref, "--config", str(deck))
    _complete_physics(ref)
    path = ref / "coupled" / restart.RUN_MANIFEST_FILENAME
    m = restart.read_run_manifest(path)
    assert m["config"]["resolved_config"]["cli_args"]["resume"] is True
    assert not restart.matching_argv_flags(m["run"]["argv"][1:], ("--resume",))
    caplog.set_level(logging.INFO)
    assert _reproduce(path, tmp_path / "new") == 2
    assert "cli_args.resume=True" in caplog.text


# --- legoesm reproduce -----------------------------------------------------

def _reproduce(manifest, out, *, check=False):
    from legoesm.cli import cmd_reproduce
    with pytest.raises(SystemExit) as exc:
        cmd_reproduce(argparse.Namespace(manifest=str(manifest), check=check,
                                         output=str(out) if out else None,
                                         allow_version_drift=False))
    return exc.value.code


def _set_argv(path, argv):
    m = json.loads(path.read_text())
    m["run"]["argv"] = argv
    path.write_text(json.dumps(m))


@pytest.mark.parametrize("flag", ["--resume", "--resum"])
def test_reproduce_refuses_a_resumed_coupled_run(
        tmp_path, monkeypatch, caplog, flag):
    ref = _reference(monkeypatch, tmp_path)
    path = ref / "coupled" / restart.RUN_MANIFEST_FILENAME
    _set_argv(path, ["run_coupled.py", "--output", str(ref), flag])
    caplog.set_level(logging.INFO)
    assert _reproduce(path, tmp_path / "new") == 2
    assert "cannot be replayed into a fresh directory" in caplog.text


def test_resume_set_by_a_config_file_is_refused(tmp_path, monkeypatch,
                                               caplog):
    """--resume set by the deck, not the command line: the recorded resolved
    value refuses at both layers."""
    _stub_setup(monkeypatch)
    _stop_at_run(monkeypatch)
    ref = tmp_path / "ref"
    with pytest.raises(_Ran):
        _coupled(ref, "--resume")       # fresh directory: starts from day 0
    _complete_physics(ref)
    path = ref / "coupled" / restart.RUN_MANIFEST_FILENAME
    # as if --resume came from the deck: the command line does not show it
    _set_argv(path, ["run_coupled.py", "--output", str(ref)])
    caplog.set_level(logging.INFO)
    assert _reproduce(path, tmp_path / "new") == 2
    assert "cli_args.resume=True" in caplog.text
    _under_reproduce(monkeypatch, ref, tmp_path / "new")
    with pytest.raises(RuntimeError, match="cannot be replayed"):
        _coupled(tmp_path / "new")


@pytest.mark.parametrize("flag", ["--resume", "--resu"])
def test_closure_refuses_a_resumed_reference(tmp_path, monkeypatch, flag):
    ref = _reference(monkeypatch, tmp_path)
    path = ref / "coupled" / restart.RUN_MANIFEST_FILENAME
    _set_argv(path, ["run_coupled.py", flag])
    _under_reproduce(monkeypatch, ref, tmp_path / "new")
    with pytest.raises(RuntimeError, match="cannot be replayed"):
        _coupled(tmp_path / "new")


def test_reproduce_refuses_the_atmosphere_record_of_a_coupled_run(
        tmp_path, monkeypatch, caplog):
    ref = _reference(monkeypatch, tmp_path)
    caplog.set_level(logging.INFO)
    assert _reproduce(ref / restart.RUN_MANIFEST_FILENAME, tmp_path / "x") == 2
    assert "atmosphere record of a coupled run" in caplog.text


def test_reproduce_check_needs_the_atmosphere_digest_too(tmp_path, monkeypatch,
                                                        caplog):
    import numpy as np
    ref = _reference(monkeypatch, tmp_path)
    path = ref / "coupled" / restart.RUN_MANIFEST_FILENAME
    restart.record_final_state_digest(path, {"x": np.ones(2)})
    caplog.set_level(logging.INFO)
    assert _reproduce(path, tmp_path / "new", check=True) == 2
    assert "Refusing to reproduce --check" in caplog.text


def test_tagged_atmosphere_record_is_refused_wherever_it_lies(tmp_path,
                                                             caplog):
    from legoesm.driver.config import ExperimentConfig
    path = restart.write_run_manifest(tmp_path / "copy", ExperimentConfig(),
                                      runner_tag="run_coupled")
    caplog.set_level(logging.INFO)
    assert _reproduce(path, tmp_path / "x") == 2
    assert "atmosphere record of a coupled run" in caplog.text


def test_coupled_driver_tags_its_atmosphere_manifest(tmp_path, monkeypatch):
    import types
    from legoesm.driver.config import ExperimentConfig
    from legoesm.driver.model_driver import ModelDriver
    _stub_setup(monkeypatch)
    seen = []
    _stop_at_run(monkeypatch, seen)
    with pytest.raises(_Ran):
        _coupled(tmp_path / "run")
    assert seen[0]._atm._manifest_runner_tag == "run_coupled"
    # and ModelDriver writes the tag it was given
    fake = types.SimpleNamespace(
        _mpi_rank=None, _output_dir=tmp_path / "atm",
        _input_config=ExperimentConfig(), _params_applied=None,
        _manifest_runner_tag="run_coupled")
    ModelDriver._write_run_manifest(fake)
    m = restart.read_run_manifest(tmp_path / "atm")
    assert m["reproducibility"]["runner_tag"] == "run_coupled"


def test_copied_coupled_record_finds_its_atmosphere_record(tmp_path,
                                                          monkeypatch, caplog):
    import shutil
    import numpy as np
    ref = _reference(monkeypatch, tmp_path)
    copy = tmp_path / "archive" / "coupled" / restart.RUN_MANIFEST_FILENAME
    copy.parent.mkdir(parents=True)
    shutil.copyfile(ref / "coupled" / restart.RUN_MANIFEST_FILENAME, copy)
    restart.record_final_state_digest(copy, {"x": np.ones(2)})
    caplog.set_level(logging.INFO)
    assert _reproduce(copy, tmp_path / "new", check=True) == 2
    assert "of this run found" not in caplog.text   # found the original
    assert "Refusing to reproduce --check" in caplog.text   # its digest: none


@pytest.mark.parametrize("atm_same, expect", [(True, 0), (False, 1)])
def test_reproduce_check_compares_both_records(tmp_path, monkeypatch, caplog,
                                               atm_same, expect):
    """--check compares the coupled AND the atmosphere digests (the rerun is
    stubbed: it writes the reference's records, the atmosphere's digest
    changed or not)."""
    import shutil
    import subprocess
    import numpy as np
    from legoesm import cli
    ref = _reference(monkeypatch, tmp_path)
    path = ref / "coupled" / restart.RUN_MANIFEST_FILENAME
    atm = restart.record_final_state_digest(
        ref / restart.RUN_MANIFEST_FILENAME, {"a": np.ones(3)})
    restart.record_final_state_digest(
        path, {"x": np.ones(2)},
        companion_digests={restart.RUN_MANIFEST_FILENAME: atm})
    # a later run updating the atmosphere record does not change the pair
    restart.record_final_state_digest(ref / restart.RUN_MANIFEST_FILENAME,
                                      {"a": np.full(3, 7.0)})
    script = restart.runner_script_path("run_coupled.py")
    m = json.loads(path.read_text())
    m["run"].update(argv=["run_coupled.py", "--output", str(ref)],
                    cwd=str(tmp_path),
                    script_sha256=restart.file_sha256(script))
    path.write_text(json.dumps(m))

    def rerun(cmd, cwd=None, env=None):
        out = Path(env[restart.REPRODUCE_OUTPUT_ENV])
        (out / "coupled").mkdir(parents=True)
        new = out / "coupled" / restart.RUN_MANIFEST_FILENAME
        shutil.copyfile(path, new)
        shutil.copyfile(ref / restart.RUN_MANIFEST_FILENAME,
                        out / restart.RUN_MANIFEST_FILENAME)
        a = restart.record_final_state_digest(
            out / restart.RUN_MANIFEST_FILENAME,
            {"a": np.ones(3) if atm_same else np.zeros(3)})
        restart.record_final_state_digest(
            new, {"x": np.ones(2)},
            companion_digests={restart.RUN_MANIFEST_FILENAME: a})
        return subprocess.CompletedProcess(cmd, 0)
    monkeypatch.setattr(cli.subprocess if hasattr(cli, "subprocess")
                        else subprocess, "run", rerun)
    caplog.set_level(logging.INFO)
    assert _reproduce(path, tmp_path / "new", check=True) == expect
    assert "MATCH — bit-identical coupled" in caplog.text
    assert ("MATCH — bit-identical run_manifest.json" in caplog.text) is atm_same


def test_companion_beside_an_archive_must_be_this_runs(tmp_path, monkeypatch):
    import shutil
    ref = _reference(monkeypatch, tmp_path)
    copy = tmp_path / "archive" / "coupled" / restart.RUN_MANIFEST_FILENAME
    copy.parent.mkdir(parents=True)
    shutil.copyfile(ref / "coupled" / restart.RUN_MANIFEST_FILENAME, copy)
    m = restart.read_run_manifest(copy)
    spec = restart.RUNNER_REPLAY_SPECS["run_coupled"]
    # a same-configuration record beside the copy: the run's own (under the
    # output directory it recorded) is preferred
    other = restart.read_run_manifest(ref)["config"]["resolved_config"]
    shutil.copyfile(ref / restart.RUN_MANIFEST_FILENAME,
                    tmp_path / "archive" / restart.RUN_MANIFEST_FILENAME)
    path, atm = restart.read_companion_manifest(
        copy, m, spec, restart.RUN_MANIFEST_FILENAME, "atmosphere")
    assert path == ref / restart.RUN_MANIFEST_FILENAME
    assert atm["config"]["resolved_config"] == other
    shutil.rmtree(ref)       # gone: the copy beside the archive is taken ...
    path, _ = restart.read_companion_manifest(
        copy, m, spec, restart.RUN_MANIFEST_FILENAME, "atmosphere")
    assert path == tmp_path / "archive" / restart.RUN_MANIFEST_FILENAME
    beside = path                # ... unless other code or config wrote it
    w = json.loads(beside.read_text())
    w["legoESM"]["commit"] = "b" * 40
    beside.write_text(json.dumps(w))
    with pytest.raises(ValueError, match="written by other code"):
        restart.read_companion_manifest(
            copy, m, spec, restart.RUN_MANIFEST_FILENAME, "atmosphere")
    from legoesm.driver.config import ExperimentConfig
    beside.unlink()
    restart.write_run_manifest(tmp_path / "archive",
                               ExperimentConfig()._replace(seed=12345))
    with pytest.raises(ValueError, match="another configuration"):
        restart.read_companion_manifest(
            copy, m, spec, restart.RUN_MANIFEST_FILENAME, "atmosphere")


def test_model_driver_remembers_the_digest_it_recorded(tmp_path):
    import types
    import numpy as np
    from legoesm.driver.config import ExperimentConfig
    from legoesm.driver.model_driver import ModelDriver
    restart.write_run_manifest(tmp_path, ExperimentConfig())
    fake = types.SimpleNamespace(
        state={"a": np.ones(2)}, tracers={}, _carry_aux=None,
        _is_spmd_multiprocess=lambda: False, _mpi_rank=None,
        _output_dir=tmp_path, final_state_digest=None)
    ModelDriver._record_final_state_digest(fake)
    assert fake.final_state_digest == restart.read_run_manifest(
        tmp_path)["result"]["state_digest"]


def test_reproduce_check_ignores_an_atmosphere_digest_the_rerun_did_not_pair(
        tmp_path, monkeypatch, caplog):
    """The output directory's atmosphere file holds the matching digest, but
    the rerun's record pairs none: --check fails, it does not MATCH."""
    import shutil
    import subprocess
    import numpy as np
    from legoesm import cli
    ref = _reference(monkeypatch, tmp_path)
    path = ref / "coupled" / restart.RUN_MANIFEST_FILENAME
    atm = restart.record_final_state_digest(
        ref / restart.RUN_MANIFEST_FILENAME, {"a": np.ones(3)})
    restart.record_final_state_digest(
        path, {"x": np.ones(2)},
        companion_digests={restart.RUN_MANIFEST_FILENAME: atm})
    m = json.loads(path.read_text())
    m["run"].update(argv=["run_coupled.py", "--output", str(ref)],
                    cwd=str(tmp_path), script_sha256=restart.file_sha256(
                        restart.runner_script_path("run_coupled.py")))
    path.write_text(json.dumps(m))

    def rerun(cmd, cwd=None, env=None):
        out = Path(env[restart.REPRODUCE_OUTPUT_ENV])
        (out / "coupled").mkdir(parents=True)
        new = out / "coupled" / restart.RUN_MANIFEST_FILENAME
        shutil.copyfile(path, new)
        shutil.copyfile(ref / restart.RUN_MANIFEST_FILENAME,      # stale file
                        out / restart.RUN_MANIFEST_FILENAME)
        restart.record_final_state_digest(new, {"x": np.ones(2)},
                                          companion_digests={})
        return subprocess.CompletedProcess(cmd, 0)
    monkeypatch.setattr(cli.subprocess if hasattr(cli, "subprocess")
                        else subprocess, "run", rerun)
    caplog.set_level(logging.INFO)
    assert _reproduce(path, tmp_path / "new", check=True) == 1
    assert "pairs none" in caplog.text


def _atm_driver(tmp_path, cfg, tag="run_coupled"):
    from legoesm.driver.model_driver import ModelDriver
    d = object.__new__(ModelDriver)
    d._mpi_rank, d._output_dir, d._input_config = None, tmp_path, cfg
    d._params_applied, d._manifest_runner_tag = None, tag
    return d


def test_coupled_atmosphere_record_continues_with_its_controls(tmp_path):
    """Choice 8 for run_coupled: its atmosphere's record may change run
    length, checkpoint frequency and wallclock budget (old one archived);
    any other change, or an untagged record, is refused as before."""
    from legoesm.driver.config import ExperimentConfig
    cfg = ExperimentConfig()
    _atm_driver(tmp_path, cfg)._write_run_manifest()
    live = tmp_path / restart.RUN_MANIFEST_FILENAME
    first = live.read_bytes()
    longer = cfg._replace(days=cfg.days + 3, output=cfg.output._replace(
        checkpoint_days=7.0, max_wallclock_seconds=3600.0))
    _atm_driver(tmp_path, longer)._write_run_manifest()
    assert restart.read_run_manifest(live)["config"]["resolved_config"][
        "days"] == cfg.days + 3
    archived = list(tmp_path.glob("run_manifest.prev-*.json"))
    assert len(archived) == 1 and archived[0].read_bytes() == first
    now = live.read_bytes()
    with pytest.raises(RuntimeError, match="DIFFERENT config"):
        _atm_driver(tmp_path, longer._replace(seed=99))._write_run_manifest()
    with pytest.raises(RuntimeError, match="DIFFERENT config"):
        _atm_driver(tmp_path, cfg, tag=None)._write_run_manifest()
    assert live.read_bytes() == now
    assert len(list(tmp_path.glob("run_manifest.prev-*.json"))) == 1
    # a record some other driver wrote (no run_coupled tag) is not continued
    other = tmp_path / "plain"
    _atm_driver(other, cfg, tag=None)._write_run_manifest()
    with pytest.raises(RuntimeError, match="DIFFERENT config"):
        _atm_driver(other, longer)._write_run_manifest()


def test_archived_coupled_record_finds_its_archived_atmosphere_record(
        tmp_path, monkeypatch):
    """After an allowed continuation replaced both records, the archived
    coupled record pairs with the archived atmosphere record."""
    import shutil
    ref = _reference(monkeypatch, tmp_path)
    coupled = ref / "coupled" / restart.RUN_MANIFEST_FILENAME
    old = restart.read_run_manifest(coupled)
    old_atm = ref / restart.RUN_MANIFEST_FILENAME
    archived = ref / "run_manifest.prev-20261019T000000000000Z-1.json"
    shutil.copyfile(old_atm, archived)           # the continuation's archive
    _stub_setup(monkeypatch, atm_seed=7)          # live record: another config
    _stop_at_run(monkeypatch)
    old_atm.unlink()
    with pytest.raises(_Ran):
        _coupled(ref.parent / "elsewhere")
    shutil.copyfile(ref.parent / "elsewhere" / restart.RUN_MANIFEST_FILENAME,
                    old_atm)
    path, _ = restart.read_companion_manifest(
        coupled, old, restart.RUNNER_REPLAY_SPECS["run_coupled"],
        restart.RUN_MANIFEST_FILENAME, "atmosphere")
    assert path == archived


def test_coupled_run_does_not_adopt_a_standalone_atmosphere_record(tmp_path):
    from legoesm.driver.config import ExperimentConfig
    cfg = ExperimentConfig()
    _atm_driver(tmp_path, cfg, tag=None)._write_run_manifest()  # standalone
    with pytest.raises(RuntimeError, match="one output directory holds one run"):
        _atm_driver(tmp_path, cfg)._write_run_manifest()
    # ... and a standalone run does not adopt a coupled run's record
    other = tmp_path / "c"
    _atm_driver(other, cfg)._write_run_manifest()
    with pytest.raises(RuntimeError, match="one output directory holds one run"):
        _atm_driver(other, cfg, tag=None)._write_run_manifest()
    # an untagged record beside a coupled record (written before the tag
    # existed) is still the coupled run's own
    old = tmp_path / "old"
    _atm_driver(old, cfg, tag=None)._write_run_manifest()
    restart.write_run_manifest(old / "coupled", {"runner": "run_coupled"},
                               config_kind="coupled", runner_tag="run_coupled")
    _atm_driver(old, cfg)._write_run_manifest()      # preserved, no raise


def test_companion_with_the_paired_digest_wins(tmp_path, monkeypatch):
    """Two identical-configuration atmosphere records (a later segment came
    back to the same configuration): the one whose digest the coupled record
    paired is taken, not merely the first."""
    import shutil
    import numpy as np
    ref = _reference(monkeypatch, tmp_path)
    coupled = ref / "coupled" / restart.RUN_MANIFEST_FILENAME
    live = ref / restart.RUN_MANIFEST_FILENAME
    archived = ref / "run_manifest.prev-20261019T000000000000Z-1.json"
    shutil.copyfile(live, archived)
    mine = restart.record_final_state_digest(archived, {"a": np.ones(3)})
    restart.record_final_state_digest(live, {"a": np.zeros(3)})
    restart.record_final_state_digest(
        coupled, {"x": np.ones(2)},
        companion_digests={restart.RUN_MANIFEST_FILENAME: mine})
    m = restart.read_run_manifest(coupled)
    path, _ = restart.read_companion_manifest(
        coupled, m, restart.RUNNER_REPLAY_SPECS["run_coupled"],
        restart.RUN_MANIFEST_FILENAME, "atmosphere")
    assert path == archived


def test_standalone_atmosphere_record_is_not_a_companion(tmp_path,
                                                         monkeypatch):
    """With the original output gone, an untagged atmosphere record of equal
    configuration beside a copied coupled record, but with no coupled record
    beside IT, is a standalone run's: refused."""
    import shutil
    ref = _reference(monkeypatch, tmp_path)
    coupled = ref / "coupled" / restart.RUN_MANIFEST_FILENAME
    m = restart.read_run_manifest(coupled)
    copy = tmp_path / "archive" / "coupled" / "kept.json"
    copy.parent.mkdir(parents=True)
    shutil.copyfile(coupled, copy)
    shutil.copyfile(ref / restart.RUN_MANIFEST_FILENAME,
                    tmp_path / "archive" / restart.RUN_MANIFEST_FILENAME)
    shutil.rmtree(ref)
    with pytest.raises(ValueError, match="not written for a run_coupled run"):
        restart.read_companion_manifest(
            copy, m, restart.RUNNER_REPLAY_SPECS["run_coupled"],
            restart.RUN_MANIFEST_FILENAME, "atmosphere")


def test_pre_tag_coupled_atmosphere_record_continues_too(tmp_path):
    from legoesm.driver.config import ExperimentConfig
    cfg = ExperimentConfig()
    _atm_driver(tmp_path, cfg, tag=None)._write_run_manifest()
    restart.write_run_manifest(tmp_path / "coupled", {"runner": "run_coupled"},
                               config_kind="coupled", runner_tag="run_coupled")
    _atm_driver(tmp_path, cfg._replace(days=cfg.days + 1))._write_run_manifest()
    assert len(list(tmp_path.glob("run_manifest.prev-*.json"))) == 1


def test_standalone_run_does_not_adopt_a_pre_tag_coupled_record(tmp_path):
    """An untagged atmosphere record beside a coupled record is the coupled
    run's: a standalone run at equal config may not adopt it."""
    import pytest
    from legoesm.driver.config import ExperimentConfig
    cfg = ExperimentConfig()
    _atm_driver(tmp_path, cfg, tag=None)._write_run_manifest()
    restart.write_run_manifest(tmp_path / "coupled", {"runner": "run_coupled"},
                               config_kind="coupled", runner_tag="run_coupled")
    with pytest.raises(RuntimeError, match="'run_coupled' runner"):
        _atm_driver(tmp_path, cfg, tag=None)._write_run_manifest()


def test_coupled_record_carries_the_atmosphere_resolved_snapshot(
        tmp_path, monkeypatch):
    """The coupled record's atmosphere config is the atmosphere's resolved
    snapshot, the same record its own manifest holds: the ice skin and
    sea-ice albedo "auto" settings appear resolved, never as "auto"."""
    def setup(self):
        self._atm._resolve_ice_skin()
        restart.write_run_manifest(self.output_dir, self._atm._input_config)
    monkeypatch.setattr(CoupledESMDriver, "setup", setup)
    seen = []
    _stop_at_run(monkeypatch, seen)
    out = tmp_path / "r"
    with pytest.raises(_Ran):
        _coupled(out)
    # Non-vacuous: the runner hands the atmosphere both settings unresolved
    # (None is the ice skin's "auto").
    assert seen[0].atm_config.ice_skin_prognostic is None
    assert seen[0].atm_config.sea_ice_albedo_scheme == "auto"
    atm = restart.read_run_manifest(out)["config"]["resolved_config"]
    cpl = restart.read_run_manifest(out / "coupled")["config"][
        "resolved_config"]["atmosphere"]
    assert isinstance(cpl["ice_skin_prognostic"], bool)
    assert cpl["sea_ice_albedo_scheme"] != "auto"

    def untyped(v):   # nested records carry a type tag only in the coupled one
        if isinstance(v, dict):
            return {k: untyped(x) for k, x in v.items() if k != "__type__"}
        return v
    assert untyped(cpl) == untyped(atm)
