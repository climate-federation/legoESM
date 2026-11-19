"""Portable preflight tests for the committed NEMO SWG_DG shell tools."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
TOOLS = ROOT / "scripts/data/nemo_swg_double_gyre"
ACQUISITION = ROOT / "scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_swg"
FIXTURES = ROOT / "tests/ocean/fixtures/swg_dg"


def _fake_nemo(tmp_path: Path) -> Path:
    nemo = tmp_path / "nemo"
    shipped = nemo / "tests/SWG"
    variant = nemo / "tests/SWG_DG"
    for path in (
        nemo / "arch", shipped / "EXPREF", shipped / "MY_SRC",
        variant / "BLD/bin", variant / "EXP00",
    ):
        path.mkdir(parents=True, exist_ok=True)
    for executable in (nemo / "makenemo", variant / "BLD/bin/nemo.exe"):
        executable.write_text("#!/usr/bin/env bash\nexit 0\n")
        executable.chmod(0o755)
    (shipped / "EXPREF/namelist_cfg").write_bytes(
        (FIXTURES / "namelist_cfg.shipped").read_bytes())
    (shipped / "MY_SRC/usrdef_zgr.F90").write_bytes(
        (FIXTURES / "usrdef_zgr.F90.shipped").read_bytes())
    (variant / "EXP00/namelist_cfg").write_bytes(
        (FIXTURES / "namelist_cfg.shipped").read_bytes())
    (variant / "EXP00/namelist_ref").write_text("&dummy\n/\n")
    (nemo / "arch/arch-swg-dg-scalarmath.fcm").write_text("reference\n")
    return nemo


def _fake_prefix(tmp_path: Path) -> Path:
    """A conda prefix whose MPI and compiler wrappers only exit 0."""
    prefix = tmp_path / "nemo-build-prefix"
    (prefix / "bin").mkdir(parents=True)
    for name in ("mpirun", "mpif90", "gfortran"):
        tool = prefix / "bin" / name
        tool.write_text("#!/usr/bin/env bash\nexit 0\n")
        tool.chmod(0o755)
    return prefix


@pytest.mark.parametrize(("directory", "name"), (
    (TOOLS, "build_swg_dg.sh"), (TOOLS, "check_swg_dg_identity.sh"),
    (TOOLS, "swg_md_run.sh"), (TOOLS, "swg_dg_rankcheck.sh"),
    (ACQUISITION, "run.sh"), (ACQUISITION, "run_30d.sh"),
))
def test_tool_help_and_source_have_no_personal_path_defaults(directory, name):
    path = directory / name
    result = subprocess.run(
        ["bash", str(path), "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "Usage:" in result.stdout
    source = path.read_text()
    assert "/data/abyssal" not in source
    assert "/home/dbalwada" not in source


def test_reference_arch_keeps_the_measured_scalar_math_flags():
    source = (TOOLS / "arch-conda-scalarmath.fcm.in").read_text()
    assert source.count("@CONDA_PREFIX@") == 2  # prose plus the one value token
    assert "%CPP                 cpp -Dkey_nosignedzero" in source
    assert ("-fdefault-real-8 -O3 -funroll-all-loops -fcray-pointer "
            "-ffree-line-length-none -fallow-argument-mismatch "
            "-fno-tree-vectorize") in source
    assert "%CFLAGS              -O0" in source
    gpu_wrapper = (ROOT / "scripts/run/run_swg_double_gyre_gpu.sh").read_text()
    assert "/data/abyssal" not in gpu_wrapper
    assert "/home/dbalwada" not in gpu_wrapper
    deck_helper = (ACQUISITION / "deck.sh").read_text()
    assert "/data/abyssal" not in deck_helper
    assert "/home/dbalwada" not in deck_helper
    assert "python3" not in deck_helper


def test_nemo_run_and_controls_preflight_without_mpi_or_makenemo(tmp_path):
    nemo = _fake_nemo(tmp_path)
    prefix = _fake_prefix(tmp_path)
    cfg = nemo / "tests/SWG_DG"
    run = subprocess.run([
        "bash", str(TOOLS / "swg_md_run.sh"),
        "--nemo-cfg", str(cfg), "--output-root", str(tmp_path),
        "--python", sys.executable, "--conda-prefix", str(prefix),
        "--three-year", "portable", "32",
    ], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    assert "steps=1892160" in run.stdout
    assert "nn_stock=630720" in run.stdout
    # The launcher takes MPI from the named prefix, not from the shell's PATH.
    assert f"WOULD_RUN: {prefix}/bin/mpirun -np 32 --oversubscribe" in run.stdout
    assert run.stdout.endswith("PREFLIGHT_OK\n")
    assert not (tmp_path / "portable").exists()

    escaped = subprocess.run([
        "bash", str(TOOLS / "swg_md_run.sh"),
        "--nemo-cfg", str(cfg), "--output-root", str(tmp_path),
        "--python", sys.executable, "--conda-prefix", str(prefix),
        "--smoke", "../escaped", "1",
    ], capture_output=True, text=True)
    assert escaped.returncode != 0
    assert "REFUSE_LABEL" in escaped.stderr
    assert not (tmp_path.parent / "escaped").exists()

    identity = subprocess.run([
        "bash", str(TOOLS / "check_swg_dg_identity.sh"),
        "--nemo-root", str(nemo),
        "--arch-name", "swg-dg-scalarmath",
        "--output", str(tmp_path / "identity"), "--python", sys.executable,
        "--conda-prefix", str(prefix),
    ], capture_output=True, text=True)
    assert identity.returncode == 0, identity.stderr
    assert "steps=10 ranks=1" in identity.stdout
    assert f"mpiexec={prefix}/bin/mpirun" in identity.stdout
    assert identity.stdout.endswith("PREFLIGHT_OK\n")

    rank = subprocess.run([
        "bash", str(TOOLS / "swg_dg_rankcheck.sh"),
        "--nemo-cfg", str(cfg), "--output", str(tmp_path / "rank"),
        "--python", sys.executable, "--conda-prefix", str(prefix),
    ], capture_output=True, text=True)
    assert rank.returncode == 0, rank.stderr
    assert f"mpiexec={prefix}/bin/mpirun" in rank.stdout
    assert "rank-control=1-versus-32" in rank.stdout
    assert rank.stdout.endswith("PREFLIGHT_OK\n")

    ladder = subprocess.run([
        "bash", str(TOOLS / "swg_md_run.sh"),
        "--nemo-cfg", str(cfg), "--output-root", str(tmp_path),
        "--python", sys.executable, "--conda-prefix", str(prefix),
        "--short-ladder", "ladder",
    ], capture_output=True, text=True)
    assert ladder.returncode == 0, ladder.stderr
    assert "steps=10 ranks=1 nn_stock=1" in ladder.stdout
    assert ladder.stdout.endswith("PREFLIGHT_OK\n")


@pytest.mark.parametrize("name", (
    "swg_md_run.sh", "check_swg_dg_identity.sh", "swg_dg_rankcheck.sh"))
def test_mpi_steps_refuse_a_prefix_without_its_mpi(tmp_path, name):
    """Every MPI step names its prefix; a prefix lacking mpirun refuses."""
    nemo = _fake_nemo(tmp_path)
    empty_prefix = tmp_path / "empty-prefix"
    (empty_prefix / "bin").mkdir(parents=True)
    common = ["--python", sys.executable, "--conda-prefix", str(empty_prefix)]
    commands = {
        "swg_md_run.sh": [
            "--nemo-cfg", str(nemo / "tests/SWG_DG"),
            "--output-root", str(tmp_path), "--short-ladder", "ladder"],
        "check_swg_dg_identity.sh": [
            "--nemo-root", str(nemo), "--arch-name", "swg-dg-scalarmath",
            "--output", str(tmp_path / "identity")],
        "swg_dg_rankcheck.sh": [
            "--nemo-cfg", str(nemo / "tests/SWG_DG"),
            "--output", str(tmp_path / "rank")],
    }
    result = subprocess.run(
        ["bash", str(TOOLS / name), *commands[name], *common],
        capture_output=True, text=True)
    assert result.returncode == 66
    assert "REFUSE_TOOLCHAIN" in result.stderr
    missing = subprocess.run(
        ["bash", str(TOOLS / name), *commands[name], "--python", sys.executable],
        capture_output=True, text=True)
    assert missing.returncode == 64
    assert "--conda-prefix" in missing.stderr


def test_build_and_identity_name_the_host_archiver_or_refuse(tmp_path):
    """The arch file's plain `ar` comes from the host's binutils."""
    nemo = _fake_nemo(tmp_path)
    prefix = _fake_prefix(tmp_path)
    tools = tmp_path / "tools_without_ar"
    tools.mkdir()
    for tool in ("dirname", "git"):
        (tools / tool).symlink_to(shutil.which(tool))
    bash = shutil.which("bash")
    build = [bash, str(TOOLS / "build_swg_dg.sh"), "--nemo-root", str(nemo),
             "--conda-prefix", str(prefix),
             "--manifest-dir", str(tmp_path / "manifest"),
             "--python", sys.executable]
    without_ar = subprocess.run(
        build, env=os.environ | {"PATH": str(tools)},
        capture_output=True, text=True)
    assert without_ar.returncode == 66, without_ar.stderr
    assert "REFUSE_ARCHIVER" in without_ar.stderr
    with_ar = subprocess.run(build, capture_output=True, text=True)
    assert "REFUSE_ARCHIVER" not in with_ar.stderr
    assert "REFUSE_MISSING_INPUT" in with_ar.stderr


def test_shell_entry_points_bind_this_checkout_ahead_of_decoy(tmp_path):
    decoy = tmp_path / "decoy/legoesm/ocean"
    decoy.mkdir(parents=True)
    (decoy / "__init__.py").write_text("DECOY = True\n")
    env = os.environ | {"PYTHONPATH": str(tmp_path / "decoy"),
                        "PYTHON": sys.executable}
    commands = (
        ["bash", str(TOOLS / "build_swg_dg.sh"),
         "--nemo-root", str(tmp_path / "missing"),
         "--conda-prefix", str(tmp_path / "missing-prefix"),
         "--manifest-dir", str(tmp_path / "manifest"),
         "--python", sys.executable],
        ["bash", str(TOOLS / "check_swg_dg_identity.sh"),
         "--nemo-root", str(tmp_path / "missing"), "--arch-name", "x",
         "--output", str(tmp_path / "identity"), "--python", sys.executable,
         "--conda-prefix", str(tmp_path / "missing-prefix")],
        ["bash", str(TOOLS / "swg_dg_rankcheck.sh"),
         "--nemo-cfg", str(tmp_path / "missing"),
         "--output", str(tmp_path / "rank"), "--python", sys.executable,
         "--conda-prefix", str(tmp_path / "missing-prefix")],
        ["bash", str(TOOLS / "swg_md_run.sh"),
         "--nemo-cfg", str(tmp_path / "missing"),
         "--output-root", str(tmp_path / "run"), "--python", sys.executable,
         "--conda-prefix", str(tmp_path / "missing-prefix"),
         "--short-ladder", "ladder"],
        ["bash", str(ROOT / "scripts/run/run_swg_double_gyre_gpu.sh"),
         "--help"],
        ["bash", str(ACQUISITION / "run.sh"),
         "--nemo-root", str(tmp_path / "missing"),
         "--evidence", str(tmp_path / "stage"), "--python", sys.executable,
         "--deck", "maddison"],
        ["bash", str(ACQUISITION / "run_30d.sh"),
         "--nemo-root", str(tmp_path / "missing"),
         "--evidence", str(tmp_path / "month"), "--python", sys.executable,
         "--deck", "maddison"],
    )
    for command in commands:
        result = subprocess.run(command, env=env, capture_output=True, text=True)
        if command[1] == str(ROOT / "scripts/run/run_swg_double_gyre_gpu.sh"):
            # The wrapper reaches driver --help on a clean tree and refuses
            # tracked/untracked dirt after binding on a development tree.
            assert result.returncode in {0, 66}, result.stderr
        else:
            assert result.returncode != 0
        assert f"CHECKOUT_IMPORT_OK root={ROOT}" in result.stdout
        assert str(ROOT / "packages/ocean/legoesm/ocean/__init__.py") in result.stdout


@pytest.mark.parametrize("relative", (
    "scripts/run/run_swg_double_gyre.py",
    "scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_swg/check_records.py",
    "scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_swg_dg_climate.py",
    "scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_swg_dg_composition.py",
    "scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_swg_dg_deck_guard.py",
    "scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_swg_dg_restart_identity.py",
    "scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_swg_dg_round6.py",
    "scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_swg_dg_round7.py",
    "scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_swg_dg_short_ladder.py",
    "scripts/plot/swg_double_gyre_movie.py",
))
def test_python_entry_points_bind_this_checkout_ahead_of_decoy(tmp_path, relative):
    decoy = tmp_path / "decoy/legoesm/ocean"
    decoy.mkdir(parents=True)
    (decoy / "__init__.py").write_text("DECOY = True\n")
    env = os.environ | {"PYTHONPATH": str(tmp_path / "decoy"),
                        "JAX_PLATFORMS": "cpu"}
    result = subprocess.run(
        [sys.executable, str(ROOT / relative), "--help"], env=env,
        cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert f"CHECKOUT_IMPORT_OK root={ROOT}" in result.stdout
    assert "JAX_PLATFORM_OK requested=cpu selected=cpu" in result.stdout
    assert str(ROOT / "packages/ocean/legoesm/ocean/__init__.py") in result.stdout
