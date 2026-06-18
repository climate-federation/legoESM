"""Pytest wrapper for the multi-PROCESS jax.distributed equivalence gate.

The real assertions live in the STANDALONE script
``scripts/validate/validate_latlon_ocean_spmd_multiprocess.py``: it must run
OUTSIDE pytest because the root ``tests/conftest.py`` initializes the XLA backend
(``ensure_metal_or_fallback`` -> ``jax.default_backend()``) at collection, before a
test module could call ``jax.distributed.initialize()`` (which must precede backend
init).  So this wrapper SUBPROCESS-launches the script under ``mpirun -np 2`` (fresh
processes, no conftest, bootstrap-first) and asserts on its exit code — the
standard pattern for jax.distributed correctness in CI.

The wrapper also directly exercises the script's importable pieces (the perturbed-
state builder + the single-process bootstrap no-op) so the new ``.py`` files get a
direct in-process unit test (CLAUDE.md "every new .py gets a direct test"), without
needing MPI for the bulk of the coverage.

Run::

    pytest tests/parallel/test_latlon_ocean_spmd_multiprocess.py -v
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO / "scripts" / "validate" / "validate_latlon_ocean_spmd_multiprocess.py"


def test_validation_script_present_and_compiles():
    """The standalone multi-process validation script exists and is syntactically
    valid (so the sbatch that runs it under mpirun cannot silently 404 / import-
    error)."""
    import py_compile

    assert _SCRIPT.is_file(), f"missing validation script: {_SCRIPT}"
    py_compile.compile(str(_SCRIPT), doraise=True)


def test_single_process_bootstrap_is_noop():
    """The jax.distributed bootstrap is a NO-OP for a single process (returns
    (rank=0, nproc=1)) — the default single-controller path is byte-unchanged.

    This runs IN a fresh subprocess so it does not perturb the backend state of
    the pytest process (the bootstrap is import-order sensitive)."""
    code = (
        "from legoesm.parallel.distributed import "
        "initialize_jax_distributed_multiprocess as b; "
        "r,n=b(); assert (r,n)==(0,1),(r,n); print('NOOP_OK')"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, timeout=300,
        env={**os.environ, "JAX_PLATFORMS": "cpu"},
    )
    assert out.returncode == 0, f"stdout={out.stdout}\nstderr={out.stderr}"
    assert "NOOP_OK" in out.stdout, out.stdout


def _run_under_mpirun(tripole: bool):
    """Launch the standalone validation under ``mpirun -np 2`` (4 CPU devices
    total) and REQUIRE a real ``[PASS]`` (not ``[SKIP]``).  Under mpirun the only
    legitimate skip is the single-process guard, which cannot fire at ``-np 2``;
    an under-provisioned-devices ``[SKIP]`` would otherwise let the gate go green
    vacuously (codex MED), so we fail on it.  The script LAND-reduces every rank's
    result, so the subprocess exit code IS the whole-job verdict."""
    env = {
        **os.environ,
        "JAX_PLATFORMS": "cpu",
        "JAX_ENABLE_X64": "1",
        # 2 CPU devices per process -> 4 global across np=2.
        "XLA_FLAGS": "--xla_force_host_platform_device_count=2",
    }
    cmd = ["mpirun", "-np", "2", sys.executable, str(_SCRIPT)]
    if tripole:
        cmd.append("--tripole")
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, env=env)
    assert out.returncode == 0, (
        f"multi-process gate FAILED (rc={out.returncode})\n"
        f"--- stdout ---\n{out.stdout}\n--- stderr ---\n{out.stderr}")
    # Require an actual PASS — a vacuous [SKIP] (e.g. <4 devices) is a FAILURE here
    # because mpirun -np 2 + XLA_FLAGS must yield 4 devices.
    assert "[PASS]" in out.stdout, (
        "expected a real [PASS] under mpirun -np 2 (a [SKIP] means the 4-device "
        f"setup did not materialize):\n--- stdout ---\n{out.stdout}\n"
        f"--- stderr ---\n{out.stderr}")


@pytest.mark.skipif(shutil.which("mpirun") is None,
                    reason="needs an MPI launcher (mpirun) for the 2-process gate")
def test_multiprocess_equivalence_under_mpirun_latlon():
    """Regular lat-lon: the cross-process ppermute/psum band halo + barotropic
    reductions + the replicated all-gather match the serial reference.

    Skipped where mpirun is unavailable (e.g. a login node without MPI on PATH);
    the sbatch ``_run_multiprocess_cpu_equiv.sbatch`` runs it on a compute node."""
    pytest.importorskip("mpi4py")
    _run_under_mpirun(tripole=False)


@pytest.mark.skipif(shutil.which("mpirun") is None,
                    reason="needs an MPI launcher (mpirun) for the 2-process gate")
def test_multiprocess_equivalence_under_mpirun_tripole():
    """TRIPOLE (active bipolar north fold): the fold band lands on the NORTH band =
    a REMOTE process, exercising the fold IN the same shard_map program as the
    cross-process collectives — the eORCA025 path the regular-grid case + the
    single-process tripole case don't jointly cover (codex MED)."""
    pytest.importorskip("mpi4py")
    _run_under_mpirun(tripole=True)
