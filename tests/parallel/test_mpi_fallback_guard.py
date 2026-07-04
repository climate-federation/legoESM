"""Fail-fast guard against a silent single-process MPI fallback (roadmap #10).

The dangerous case: a job launched multi-rank (``mpirun``/``srun``) where the
MPI stack silently initializes only 1 process (mpi4py absent/broken,
jax.distributed not up) — every rank then runs the FULL global domain
independently, producing "fake scaling" numbers / redundant compute with NO
error.  ``check_no_silent_mpi_fallback`` must turn that into a loud RuntimeError.
"""

from __future__ import annotations

import pytest

from legoesm.parallel.runtime import (
    check_no_silent_mpi_fallback,
    launcher_declared_world_size,
)


# --- launcher_declared_world_size -------------------------------------------

@pytest.mark.parametrize(
    "var",
    ["OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "MV2_COMM_WORLD_SIZE", "SLURM_NTASKS"],
)
def test_declared_reads_each_launcher_var(var):
    assert launcher_declared_world_size({var: "4"}) == 4


def test_declared_zero_when_no_launcher_env():
    assert launcher_declared_world_size({}) == 0


def test_declared_precedence_mpi_over_slurm():
    # An explicit MPI-launcher var wins over the srun fallback.
    env = {"SLURM_NTASKS": "1", "OMPI_COMM_WORLD_SIZE": "8"}
    assert launcher_declared_world_size(env) == 8


def test_declared_skips_non_numeric():
    env = {"OMPI_COMM_WORLD_SIZE": "notanint", "PMI_SIZE": "6"}
    assert launcher_declared_world_size(env) == 6


# --- check_no_silent_mpi_fallback -------------------------------------------

def test_raises_on_silent_single_process_under_mpi_launch():
    with pytest.raises(RuntimeError, match="silent single-process fallback"):
        check_no_silent_mpi_fallback(4, 1)


def test_ok_when_mpi_actually_initialized():
    # declared == detected == 4: the stack came up correctly.
    check_no_silent_mpi_fallback(4, 4)


def test_ok_for_genuine_single_process():
    # No launcher (declared 0) or a 1-rank launch: nothing to guard.
    check_no_silent_mpi_fallback(0, 1)
    check_no_silent_mpi_fallback(1, 1)


def test_allow_override_permits_deliberate_non_mpi_multilaunch():
    check_no_silent_mpi_fallback(4, 1, allow=True)  # must not raise


def test_error_message_includes_mpi4py_reason():
    with pytest.raises(RuntimeError, match="ModuleNotFoundError"):
        check_no_silent_mpi_fallback(
            2, 1, reason="ModuleNotFoundError('mpi4py')")


def test_partial_fallback_also_raises():
    # Launched with 8 ranks but only 2 initialized is still a silent demotion.
    with pytest.raises(RuntimeError):
        check_no_silent_mpi_fallback(8, 1)
    # (detected>1 is treated as "MPI is up"; only detected==1 under a multi-rank
    # launch is the unambiguous silent-fallback signature this guard targets.)
    check_no_silent_mpi_fallback(8, 2)


# --- integration: the guard is actually WIRED into ParallelRuntime.create() --

def test_create_raises_under_faked_mpi_launch(monkeypatch):
    # Fake a 4-rank OpenMPI launch env while running as ONE ordinary process
    # (jax.process_count()==1, mpi4py COMM size 1): create() must refuse rather
    # than silently build a single-process runtime.  Exercises the real
    # detection block + env-allow parsing (not just the pure helper).
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "4")
    monkeypatch.delenv("LEGOESM_ALLOW_SINGLE_PROCESS_UNDER_MPI", raising=False)
    from legoesm.parallel.runtime import ParallelRuntime
    with pytest.raises(RuntimeError, match="silent single-process fallback"):
        ParallelRuntime.create()


def test_create_allow_override_skips_fallback_guard(monkeypatch):
    # With the override set, create() must get PAST the guard (any non-fallback
    # outcome — success or an unrelated error — proves the guard was skipped).
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "4")
    monkeypatch.setenv("LEGOESM_ALLOW_SINGLE_PROCESS_UNDER_MPI", "1")
    from legoesm.parallel.runtime import ParallelRuntime
    try:
        ParallelRuntime.create()
    except RuntimeError as e:
        assert "silent single-process fallback" not in str(e)
