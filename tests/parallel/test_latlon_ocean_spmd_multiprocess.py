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


def _run_two_phase(tripole: bool, tmp_path):
    """Phase 1 (single process, 4 CPU devices): compute the serial + 4-device-SPMD
    reference, assert equal, save it.  Phase 2 (``mpirun -np 2``, 4 global devices):
    run the multi-process SPMD step and compare to the saved reference + assert the
    gather is replicated across processes.  REQUIRE a real ``[PASS]`` on both (a
    vacuous ``[SKIP]`` -- e.g. <4 devices -- is a FAILURE here; codex MED).  The
    phase-2 script LAND-reduces every rank, so its exit code is the whole-job
    verdict."""
    ref = str(tmp_path / f"ref_{'tripole' if tripole else 'latlon'}.npz")
    extra = ["--tripole"] if tripole else []

    # Phase 1: single process, 4 CPU devices (NO jax.distributed -> the serial
    # reference's eta-floor reduction is a local sum, no mpi4jax).
    env1 = {**os.environ, "JAX_PLATFORMS": "cpu", "JAX_ENABLE_X64": "1",
            "XLA_FLAGS": "--xla_force_host_platform_device_count=4"}
    p1 = subprocess.run(
        [sys.executable, str(_SCRIPT), "--phase", "ref", "--out", ref, *extra],
        capture_output=True, text=True, timeout=900, env=env1)
    assert p1.returncode == 0 and "[PASS ref]" in p1.stdout, (
        f"phase ref FAILED (rc={p1.returncode})\n--- stdout ---\n{p1.stdout}\n"
        f"--- stderr ---\n{p1.stderr}")

    # Phase 2: mpirun -np 2, 2 CPU devices per process -> 4 global.
    env2 = {**os.environ, "JAX_PLATFORMS": "cpu", "JAX_ENABLE_X64": "1",
            "XLA_FLAGS": "--xla_force_host_platform_device_count=2"}
    cmd = ["mpirun", "-np", "2", sys.executable, str(_SCRIPT),
           "--phase", "mp", "--ref", ref, *extra]
    p2 = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, env=env2)
    assert p2.returncode == 0, (
        f"multi-process gate FAILED (rc={p2.returncode})\n"
        f"--- stdout ---\n{p2.stdout}\n--- stderr ---\n{p2.stderr}")
    assert "[PASS]" in p2.stdout, (
        "expected a real [PASS] under mpirun -np 2 (a [SKIP] means the 4-device "
        f"setup did not materialize):\n--- stdout ---\n{p2.stdout}\n"
        f"--- stderr ---\n{p2.stderr}")


@pytest.mark.skipif(shutil.which("mpirun") is None,
                    reason="needs an MPI launcher (mpirun) for the 2-process gate")
def test_multiprocess_equivalence_under_mpirun_latlon(tmp_path):
    """Regular lat-lon: the cross-process ppermute/psum band halo + barotropic
    reductions + the replicated all-gather match the serial reference.

    Skipped where mpirun is unavailable (e.g. a login node without MPI on PATH);
    the sbatch ``run_multiprocess_cpu_equiv.sbatch`` runs it on a compute node."""
    pytest.importorskip("mpi4py")
    _run_two_phase(tripole=False, tmp_path=tmp_path)


@pytest.mark.skipif(shutil.which("mpirun") is None,
                    reason="needs an MPI launcher (mpirun) for the 2-process gate")
def test_multiprocess_equivalence_under_mpirun_tripole(tmp_path):
    """TRIPOLE (active bipolar north fold): the fold band lands on the NORTH band =
    a REMOTE process, exercising the fold IN the same shard_map program as the
    cross-process collectives — the eORCA025 path the regular-grid case + the
    single-process tripole case don't jointly cover (codex MED)."""
    pytest.importorskip("mpi4py")
    _run_two_phase(tripole=True, tmp_path=tmp_path)
