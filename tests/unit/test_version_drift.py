"""Version-drift gate: an old run must not be continued or reproduced on code
that may differ from the code that wrote it, unless explicitly allowed
(docs/architecture/reproducibility.md, P0 step 2)."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import pytest

import legoesm.driver.restart as restart_mod
from legoesm.driver.config import DycoreConfig, ExperimentConfig, GridConfig
from legoesm.driver.restart import (
    ALLOW_VERSION_DRIFT_ENV,
    RUN_MANIFEST_FILENAME,
    build_run_manifest,
    config_hash_matches,
    version_drift,
    write_run_manifest,
)
from legoesm.io.git_provenance import GitProvenance

SHA_A = "a" * 40
SHA_B = "b" * 40


def _pin(monkeypatch, commit: str, dirty: bool = False) -> None:
    """Make the IMPORTED package look like it is at *commit*."""
    monkeypatch.setattr(
        restart_mod, "git_provenance",
        lambda anchor: GitProvenance(anchor=str(anchor), root="/repo",
                                     commit=commit, ref="", dirty=dirty))


def _config() -> ExperimentConfig:
    return ExperimentConfig(
        grid=GridConfig(grid_type="cubed_sphere", resolution=48, nlev=40),
        dycore=DycoreConfig(model_type="shallow_water", discretization="cdgrid",
                            dt=600.0),
        days=5,
    )


def test_version_drift_reasons(monkeypatch) -> None:
    _pin(monkeypatch, SHA_A)
    assert version_drift(SHA_A) == []
    assert any("!=" in r for r in version_drift(SHA_B))
    assert version_drift("")                       # unknown recorded commit
    assert version_drift(SHA_A, recorded_dirty=True)
    assert version_drift(SHA_A, recorded_patches=["001-x.patch"])
    _pin(monkeypatch, SHA_A, dirty=True)
    assert version_drift(SHA_A)                    # current tree dirty
    _pin(monkeypatch, "")
    assert version_drift(SHA_A)                    # imported commit unknown


def _reproduce(manifest: Path, out: Path, allow: bool) -> None:
    from legoesm.cli import cmd_reproduce
    cmd_reproduce(argparse.Namespace(manifest=str(manifest), check=True,
                                     output=str(out),
                                     allow_version_drift=allow))


def test_reproduce_refuses_across_version_drift(tmp_path, monkeypatch,
                                                caplog) -> None:
    monkeypatch.delenv(ALLOW_VERSION_DRIFT_ENV, raising=False)
    _pin(monkeypatch, SHA_A)
    write_run_manifest(tmp_path, _config())
    _pin(monkeypatch, SHA_B)
    caplog.set_level(logging.INFO)
    with pytest.raises(SystemExit) as exc:
        # --output = reference dir: if the drift gate let this through, the
        # later output-dir guard would exit instead, without the refusal line.
        _reproduce(tmp_path / RUN_MANIFEST_FILENAME, tmp_path, allow=False)
    assert exc.value.code == 2
    assert "Refusing to reproduce across version drift" in caplog.text
    assert "does NOT mean the same science" in caplog.text
    assert "is the reference run's directory" not in caplog.text  # exited first


def test_reproduce_allow_version_drift_passes_the_gate(tmp_path, monkeypatch,
                                                       caplog) -> None:
    monkeypatch.delenv(ALLOW_VERSION_DRIFT_ENV, raising=False)
    _pin(monkeypatch, SHA_A)
    write_run_manifest(tmp_path, _config())
    _pin(monkeypatch, SHA_B)
    caplog.set_level(logging.INFO)
    with pytest.raises(SystemExit):   # stopped by the output-dir guard instead
        _reproduce(tmp_path / RUN_MANIFEST_FILENAME, tmp_path, allow=True)
    assert "Version drift" in caplog.text
    assert "Refusing to reproduce across version drift" not in caplog.text
    assert "is the reference run's directory" in caplog.text


def _driver(tmp_path: Path):
    from legoesm.driver.model_driver import ModelDriver
    d = ModelDriver(ExperimentConfig())
    d._output_dir = tmp_path
    d._mpi_rank = None
    return d


def test_resume_refuses_across_version_drift(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv(ALLOW_VERSION_DRIFT_ENV, raising=False)
    _pin(monkeypatch, SHA_A)
    _driver(tmp_path)._check_resume_version_drift()  # no manifest yet: fine
    _driver(tmp_path)._write_run_manifest()
    _driver(tmp_path)._check_resume_version_drift()  # same code: resume is fine
    _pin(monkeypatch, SHA_B)
    with pytest.raises(RuntimeError, match=ALLOW_VERSION_DRIFT_ENV):
        _driver(tmp_path)._check_resume_version_drift()
    monkeypatch.setenv(ALLOW_VERSION_DRIFT_ENV, "1")
    _driver(tmp_path)._check_resume_version_drift()  # explicitly allowed


def test_setup_runs_the_resume_gate_first(tmp_path, monkeypatch) -> None:
    """The gate is collective, so it must run before setup's first rank-local
    step; prove setup reaches it right after the runtime bootstrap."""
    from legoesm.driver.model_driver import ModelDriver
    monkeypatch.delenv(ALLOW_VERSION_DRIFT_ENV, raising=False)
    _pin(monkeypatch, SHA_A)
    _driver(tmp_path)._write_run_manifest()
    _pin(monkeypatch, SHA_B)
    monkeypatch.setattr(ModelDriver, "_bootstrap_runtime", lambda self: None)
    monkeypatch.setattr(ModelDriver, "_preflight_land_inputs",
                        lambda self: pytest.fail("gate did not run first"))
    with pytest.raises(RuntimeError, match=ALLOW_VERSION_DRIFT_ENV):
        _driver(tmp_path).setup()


def test_manifest_without_dirty_or_patch_state_is_drift(monkeypatch) -> None:
    from legoesm.driver.restart import manifest_version_drift
    _pin(monkeypatch, SHA_A)
    assert manifest_version_drift({"legoESM": {"commit": SHA_A},
                                   "reproducibility": {}})
    assert manifest_version_drift({"legoESM": {"commit": SHA_A},
                                   "reproducibility": {"git_dirty": None,
                                                       "patches": []}})
    assert manifest_version_drift({"legoESM": {"commit": SHA_A},
                                   "reproducibility": {"git_dirty": False,
                                                       "patches": None}})
    assert manifest_version_drift({"legoESM": {"commit": SHA_A},
                                   "reproducibility": {"git_dirty": False,
                                                       "patches": []}}) == []


def test_checkpoint_continuation_drift_gate(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv(ALLOW_VERSION_DRIFT_ENV, raising=False)
    _pin(monkeypatch, SHA_A)
    ckpt = tmp_path / "checkpoint_day_0001.npz"
    ckpt.write_bytes(b"")
    d = _driver(tmp_path)
    with pytest.raises(RuntimeError, match="code that wrote it is unknown"):
        d._check_checkpoint_version_drift(ckpt)  # no manifest beside it
    write_run_manifest(tmp_path, _config())
    d._check_checkpoint_version_drift(ckpt)      # same code: fine
    _pin(monkeypatch, SHA_B)
    with pytest.raises(RuntimeError, match=ALLOW_VERSION_DRIFT_ENV):
        d._check_checkpoint_version_drift(ckpt)


def test_load_checkpoint_runs_the_drift_gate(tmp_path, monkeypatch) -> None:
    """The gate sits in front of every lane of load_checkpoint."""
    monkeypatch.delenv(ALLOW_VERSION_DRIFT_ENV, raising=False)
    _pin(monkeypatch, SHA_A)
    ckpt = tmp_path / "checkpoint_day_0001.npz"
    ckpt.write_bytes(b"")
    with pytest.raises(RuntimeError, match="code that wrote it is unknown"):
        _driver(tmp_path).load_checkpoint(ckpt)


def test_checkpoint_gate_refuses_a_manifest_without_full_provenance(
        tmp_path, monkeypatch) -> None:
    """A JSON file holding only a matching commit proves nothing about
    uncommitted changes or patches."""
    monkeypatch.delenv(ALLOW_VERSION_DRIFT_ENV, raising=False)
    _pin(monkeypatch, SHA_A)
    ckpt = tmp_path / "checkpoint_day_0001.npz"
    ckpt.write_bytes(b"")
    (tmp_path / RUN_MANIFEST_FILENAME).write_text(
        json.dumps({"legoESM": {"commit": SHA_A}}))
    with pytest.raises(RuntimeError, match="not a valid run manifest"):
        _driver(tmp_path)._check_checkpoint_version_drift(ckpt)


def test_missing_checkpoint_is_left_to_the_loader(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv(ALLOW_VERSION_DRIFT_ENV, raising=False)
    _pin(monkeypatch, SHA_A)
    _driver(tmp_path)._check_checkpoint_version_drift(tmp_path / "typo.npz")


def test_reproduce_honours_the_env_opt_out(tmp_path, monkeypatch,
                                            caplog) -> None:
    _pin(monkeypatch, SHA_A)
    write_run_manifest(tmp_path, _config())
    _pin(monkeypatch, SHA_B)
    monkeypatch.setenv(ALLOW_VERSION_DRIFT_ENV, "1")
    caplog.set_level(logging.INFO)
    with pytest.raises(SystemExit):   # stopped by the output-dir guard instead
        _reproduce(tmp_path / RUN_MANIFEST_FILENAME, tmp_path, allow=False)
    assert "Refusing to reproduce across version drift" not in caplog.text


def test_schema_growth_acceptance_names_the_new_fields(caplog) -> None:
    m = build_run_manifest(_config())
    resolved = m["config"]["resolved_config"]
    del resolved["morrison_flavor"]
    stored_hash = hashlib.sha256(
        json.dumps(resolved, sort_keys=True).encode("utf-8")).hexdigest()
    caplog.set_level(logging.WARNING)
    assert config_hash_matches(stored_hash, resolved, _config())
    assert "morrison_flavor" in caplog.text


def test_manifest_writer_rerun_refuses_across_version_drift(tmp_path,
                                                            monkeypatch) -> None:
    """The run_omip/run_lmip/run_dino/run_coupled writer archives an existing
    manifest on a rerun -- only after the same drift check."""
    from legoesm.driver.restart import write_run_manifest_on_root
    monkeypatch.delenv(ALLOW_VERSION_DRIFT_ENV, raising=False)
    _pin(monkeypatch, SHA_A)
    write_run_manifest_on_root(tmp_path, _config())
    write_run_manifest_on_root(tmp_path, _config())   # same code: archives
    assert len(list(tmp_path.glob("run_manifest.prev-*.json"))) == 1
    _pin(monkeypatch, SHA_B)
    with pytest.raises(RuntimeError, match=ALLOW_VERSION_DRIFT_ENV):
        write_run_manifest_on_root(tmp_path, _config())
    assert len(list(tmp_path.glob("run_manifest.prev-*.json"))) == 1
    monkeypatch.setenv(ALLOW_VERSION_DRIFT_ENV, "1")
    write_run_manifest_on_root(tmp_path, _config())
    assert len(list(tmp_path.glob("run_manifest.prev-*.json"))) == 2


def test_invalid_existing_manifest_is_refused_by_the_collective_gate(
        tmp_path, monkeypatch) -> None:
    """Refused at the top of setup (collectively), not later on rank 0 alone."""
    _pin(monkeypatch, SHA_A)
    (tmp_path / RUN_MANIFEST_FILENAME).write_text("{not json")
    with pytest.raises(RuntimeError, match="is invalid"):
        _driver(tmp_path)._check_resume_version_drift()
