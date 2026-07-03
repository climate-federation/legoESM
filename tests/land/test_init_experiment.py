"""Tests for scripts/run/init_experiment.py — the template instantiator."""

import configparser
import importlib.util
import subprocess
from pathlib import Path

import pytest
import yaml

_ROOT = Path(__file__).resolve().parents[2]
_INIT = _ROOT / "scripts" / "run" / "init_experiment.py"


def _load_init():
    spec = importlib.util.spec_from_file_location("init_experiment", _INIT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_smoke_surfdata(tmp_path):
    """Create a stand-in file so surfdata.path validates."""
    p = tmp_path / "sd.nc"; p.write_text("stub")
    return p


def _run_init(argv):
    """Run the init CLI in-process."""
    import sys as _sys
    saved = _sys.argv
    _sys.argv = ["init_experiment.py"] + argv
    try:
        _load_init().main()
    finally:
        _sys.argv = saved


def test_pbs_headers_raises_on_unknown_scheduler():
    # 'local'/absent -> no directives; a typo or unsupported scheduler must RAISE
    # rather than emit a header-less run.sh the user submits to run on the login
    # node (dispatch hardening).
    mod = _load_init()
    assert mod._pbs_headers({}) == ""                        # absent -> local
    assert mod._pbs_headers({"scheduler": "local"}) == ""
    with pytest.raises(ValueError, match="scheduler"):
        mod._pbs_headers({"scheduler": "slurm"})
    with pytest.raises(ValueError, match="scheduler"):
        mod._pbs_headers({"scheduler": "PBS"})               # case typo != 'pbs'


def test_init_creates_expected_files(tmp_path):
    sd = _write_smoke_surfdata(tmp_path)
    out = tmp_path / "expt"
    _run_init([
        "biophysics/smoke_test",
        "--name", "test_run",
        "--output-dir", str(out),
        "-o", f"surfdata.path={sd}",
    ])
    for f in ("run.yaml", "config.yaml", "experiment.tag", "run.sh"):
        assert (out / f).exists(), f"missing {f}"

    # run.yaml records the template + overrides
    run_yaml = yaml.safe_load((out / "run.yaml").read_text())
    assert run_yaml["template"] == "biophysics/smoke_test"
    assert run_yaml["name"] == "test_run"
    assert any("surfdata.path=" in o for o in run_yaml["overrides"])

    # config.yaml is the resolved config; override took effect
    cfg = yaml.safe_load((out / "config.yaml").read_text())
    assert cfg["surfdata"]["path"] == str(sd)
    assert cfg["forcing"]["source"] == "synthetic"

    # experiment.tag records provenance
    tag = configparser.ConfigParser()
    tag.read(out / "experiment.tag")
    assert tag["experiment"]["template"] == "biophysics/smoke_test"
    assert tag["experiment"]["name"] == "test_run"
    assert tag["legoESM"]["commit"]                       # non-empty (git or "unknown")
    assert tag["reproducibility"]["config_hash"].startswith("sha256:")
    assert tag["reproducibility"]["python_version"]

    # run.sh is executable and points at the config
    run_sh = (out / "run.sh").read_text()
    assert "--config" in run_sh and "config.yaml" in run_sh
    assert (out / "run.sh").stat().st_mode & 0o111


def test_init_rejects_bad_override(tmp_path):
    sd = _write_smoke_surfdata(tmp_path)
    out = tmp_path / "expt"
    with pytest.raises(ValueError, match="simple_seb"):    # config validator catches it
        _run_init([
            "biophysics/smoke_test",
            "--name", "bad",
            "--output-dir", str(out),
            "-o", f"surfdata.path={sd}",
            "-o", "physics.bulk_scheme=most",              # simple_seb + MOST = illegal
        ])
    assert not out.exists() or not (out / "config.yaml").exists()


def test_init_unknown_template_reports_available(tmp_path):
    with pytest.raises(SystemExit, match="template"):
        _run_init([
            "biophysics/nonexistent",
            "--name", "x",
            "--output-dir", str(tmp_path / "x"),
        ])


def test_machine_derecho_emits_pbs_headers_but_no_account(tmp_path):
    """--machine derecho generates PBS directives in run.sh but must NEVER
    include `-A` (project account) — that stays on the qsub line so run.sh is
    portable across projects."""
    sd = _write_smoke_surfdata(tmp_path)
    out = tmp_path / "expt"
    _run_init([
        "biophysics/smoke_test",
        "--name", "derecho_run",
        "--output-dir", str(out),
        "--machine", "derecho",
        "-o", f"surfdata.path={sd}",
    ])
    run_sh = (out / "run.sh").read_text()

    # PBS headers present
    assert "#PBS -N lmip" in run_sh
    assert "#PBS -l select=" in run_sh
    assert "#PBS -l walltime=" in run_sh
    assert "#PBS -q main" in run_sh
    assert "#PBS -j oe" in run_sh

    # Account NEVER emitted
    assert "#PBS -A" not in run_sh
    assert "-A UYAL" not in run_sh
    assert "-A YOUR_ACCOUNT" not in run_sh          # not even a placeholder

    # Env-setup block reached from configs/machines/derecho.yaml
    assert "module load conda" in run_sh
    assert "conda activate" in run_sh
    assert 'export LEGOESM_PYTHON="$(which python)"' in run_sh

    # experiment.tag records the profile
    tag = configparser.ConfigParser()
    tag.read(out / "experiment.tag")
    assert tag["machine"]["profile"] == "derecho"


def test_machine_derecho_gpu_emits_gpu_select_and_jax_env(tmp_path):
    """--machine derecho_gpu requests 1 A100 in the PBS select clause AND sets
    JAX GPU environment variables; still no project account."""
    sd = _write_smoke_surfdata(tmp_path)
    out = tmp_path / "expt"
    _run_init([
        "biophysics/smoke_test",
        "--name", "gpu_run",
        "--output-dir", str(out),
        "--machine", "derecho_gpu",
        "-o", f"surfdata.path={sd}",
    ])
    run_sh = (out / "run.sh").read_text()

    # PBS resource request includes a GPU
    assert "#PBS -l select=" in run_sh and "ngpus=1" in run_sh
    assert "#PBS -N lmip_gpu" in run_sh

    # JAX GPU env — the whole point of a GPU profile
    assert "XLA_PYTHON_CLIENT_PREALLOCATE=false" in run_sh
    assert "JAX_PLATFORM_NAME=gpu" in run_sh
    assert "legoesm-gpu" in run_sh                    # default env name

    # Project account still NEVER emitted
    assert "#PBS -A" not in run_sh
    assert "-A UYAL" not in run_sh

    # Provenance
    tag = configparser.ConfigParser()
    tag.read(out / "experiment.tag")
    assert tag["machine"]["profile"] == "derecho_gpu"


def test_no_machine_flag_keeps_run_sh_plain(tmp_path):
    """No --machine → NO PBS headers, NO env_setup — pure bash wrapper."""
    sd = _write_smoke_surfdata(tmp_path)
    out = tmp_path / "expt"
    _run_init([
        "biophysics/smoke_test",
        "--name", "plain",
        "--output-dir", str(out),
        "-o", f"surfdata.path={sd}",
    ])
    run_sh = (out / "run.sh").read_text()
    assert "#PBS" not in run_sh
    assert "conda activate" not in run_sh
    assert "module load" not in run_sh
    tag = configparser.ConfigParser()
    tag.read(out / "experiment.tag")
    assert tag["machine"]["profile"] == "none"


def test_unknown_machine_reports_available(tmp_path):
    sd = _write_smoke_surfdata(tmp_path)
    with pytest.raises(SystemExit, match="machine"):
        _run_init([
            "biophysics/smoke_test",
            "--name", "bad",
            "--output-dir", str(tmp_path / "x"),
            "--machine", "nonexistent",
            "-o", f"surfdata.path={sd}",
        ])


def test_smoke_4deg_template_validates_and_inits(tmp_path):
    """The 4° smoke template loads, validates, and init produces a config
    with the expected simple_seb + constant + resolution=45 combination."""
    import yaml
    out = tmp_path / "expt"
    _run_init([
        "biophysics/smoke_4deg",
        "--name", "smoke_4deg_test",
        "--output-dir", str(out),
        # No override needed; template already declares a real-forcing path.
        # Override only the surfdata to point at a stub so validate_config passes.
        "-o", f"surfdata.path={_write_smoke_surfdata(tmp_path)}",
    ])
    cfg = yaml.safe_load((out / "config.yaml").read_text())
    assert cfg["grid"]["resolution"] == 45                     # 4° x 4°
    assert cfg["physics"]["surface_scheme"] == "simple_seb"
    assert cfg["physics"]["bulk_scheme"] == "constant"         # required paired
    assert cfg["time"]["n_steps"] == 720                        # 30 days at 1 h
    # Monthly tape shape survives — same schema as production template.
    tape_names = [t["name"] for t in cfg["output"]["tapes"]]
    assert tape_names == ["monthly", "monthly_state"]


def test_run_sh_uses_pbs_o_workdir_when_set(tmp_path):
    """PBS copies the batch script to a spool dir before executing, so
    `dirname $0` no longer points at the experiment directory.  The generated
    run.sh must use $PBS_O_WORKDIR (which PBS sets to the qsub directory)
    when it is set, and fall back to `dirname` only for local bash runs."""
    sd = _write_smoke_surfdata(tmp_path)
    out = tmp_path / "expt"
    _run_init([
        "biophysics/smoke_test",
        "--name", "workdir_test",
        "--output-dir", str(out),
        "--machine", "derecho",
        "-o", f"surfdata.path={sd}",
    ])
    run_sh = (out / "run.sh").read_text()
    # $PBS_O_WORKDIR must be referenced with :-fallback to dirname $0
    assert "cd \"${PBS_O_WORKDIR:-$(dirname \"$0\")}\"" in run_sh
    # Naive dirname-only form must NOT appear (would silently break on PBS)
    assert "cd \"$(dirname \"$0\")\"\n" not in run_sh


def test_init_dry_run_writes_nothing(tmp_path, capsys):
    sd = _write_smoke_surfdata(tmp_path)
    out = tmp_path / "expt"
    _run_init([
        "biophysics/smoke_test",
        "--name", "dry",
        "--output-dir", str(out),
        "-o", f"surfdata.path={sd}",
        "--dry-run",
    ])
    captured = capsys.readouterr().out
    assert "surfdata" in captured                          # printed resolved config
    assert not out.exists()                                 # nothing written
