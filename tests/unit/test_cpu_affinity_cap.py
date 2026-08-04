"""The conftest CPU-affinity cap that keeps XLA's thread pool under ``ulimit -u``.

Non-vacuity is the point of this file: each test drives the cap through a
distinct branch with the OS calls stubbed, so a future edit that silently turns
it into a no-op (or into an unconditional clamp) goes red.  The end-to-end proof
that the cap prevents the abort is ``test_many_clients_do_not_exhaust_threads``,
which is opt-in because it is a host stress test, not a unit test.
"""

from __future__ import annotations

import os
import resource
import subprocess
import sys
import time

import pytest

from tests.conftest import (
    _DEFAULT_TEST_CPU_CAP,
    _MPI_RANK_ENV,
    _cap_cpu_affinity_for_thread_rlimit,
    _effective_cap,
    _is_mpi_rank,
    _worker_slice,
)

# The branch tests stub the affinity syscalls, which do not exist on macOS --
# ``raising=False`` so monkeypatch can install them there instead of erroring
# before the branch under test is ever reached.
_SET = dict(raising=False)
_XDIST_ENV = ("PYTEST_XDIST_WORKER", "PYTEST_XDIST_WORKER_COUNT")


@pytest.fixture
def affinity_spy(monkeypatch):
    """Record sched_setaffinity calls instead of performing them."""
    calls: list[set[int]] = []
    monkeypatch.setattr(
        os, "sched_setaffinity", lambda pid, mask: calls.append(set(mask)), **_SET
    )
    return calls


@pytest.fixture
def wide_machine(monkeypatch):
    """A 256-CPU host, 2048-thread ceiling, no MPI and no xdist in the environment."""
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(256)), **_SET)
    monkeypatch.setattr(resource, "getrlimit", lambda which: (2048, 2048))
    for var in ("LEGOESM_TEST_CPU_CAP", *_MPI_RANK_ENV, *_XDIST_ENV):
        monkeypatch.delenv(var, raising=False)


# --- the cap is derived, not hardcoded -------------------------------------
def test_cap_is_the_documented_default_on_the_measured_host(wide_machine):
    """256 CPUs, ulimit -u 2048, one worker -> the documented 32."""
    assert _effective_cap(256, 2048) == _DEFAULT_TEST_CPU_CAP


def test_cap_shrinks_when_the_budget_cannot_afford_the_default(wide_machine, monkeypatch):
    """A tight ceiling shared by many workers must NOT keep handing out 32 CPUs.

    32 CPUs is ~96 threads per client, so 32 is genuinely affordable for ONE
    worker even at a 512 ceiling -- the cap only has to shrink once the budget
    per worker drops below what 32 CPUs costs.  8 workers on a 512 ceiling gives
    each 32 threads, which buys ~5 CPUs.
    """
    monkeypatch.setenv("PYTEST_XDIST_WORKER_COUNT", "8")
    cap = _effective_cap(256, 512)
    assert cap < _DEFAULT_TEST_CPU_CAP
    assert cap >= 4, "never shrink below a workable XLA pool"


def test_cap_shrinks_as_xdist_workers_multiply(wide_machine, monkeypatch):
    """The ceiling is shared across workers, so more workers -> a smaller cap.

    Uses a ceiling tight enough that the _DEFAULT_TEST_CPU_CAP clamp is not the
    binding constraint, otherwise both sides just saturate at 32.
    """
    monkeypatch.setenv("PYTEST_XDIST_WORKER_COUNT", "2")
    solo = _effective_cap(256, 512)
    monkeypatch.setenv("PYTEST_XDIST_WORKER_COUNT", "16")
    assert _effective_cap(256, 512) < solo


def test_default_cap_is_affordable_for_a_lone_worker(wide_machine):
    """Sanity on the formula: 32 CPUs must fit the budget it is granted."""
    from tests.conftest import _RLIMIT_SHARE, _THREADS_BASE, _THREADS_PER_CPU

    cost = _THREADS_PER_CPU * _DEFAULT_TEST_CPU_CAP + _THREADS_BASE
    assert cost <= 2048 * _RLIMIT_SHARE


def test_explicit_env_cap_wins(wide_machine, monkeypatch):
    monkeypatch.setenv("LEGOESM_TEST_CPU_CAP", "6")
    assert _effective_cap(256, 2048) == 6


def test_garbage_env_var_falls_back_to_the_derived_cap(wide_machine, monkeypatch):
    monkeypatch.setenv("LEGOESM_TEST_CPU_CAP", "not-a-number")
    assert _effective_cap(256, 2048) == _DEFAULT_TEST_CPU_CAP


@pytest.mark.parametrize("disabled", ["0", "-1"])
def test_non_positive_env_cap_disables(wide_machine, affinity_spy, monkeypatch, disabled):
    monkeypatch.setenv("LEGOESM_TEST_CPU_CAP", disabled)
    _cap_cpu_affinity_for_thread_rlimit()
    assert affinity_spy == []


# --- xdist workers get DISJOINT slices --------------------------------------
def test_xdist_workers_get_disjoint_cpu_slices(monkeypatch):
    """Every worker inheriting the same low CPUs would pile 4 XLA pools on 32 cores."""
    cpus = list(range(256))
    slices = []
    for worker in ("gw0", "gw1", "gw2", "gw3"):
        monkeypatch.setenv("PYTEST_XDIST_WORKER", worker)
        slices.append(_worker_slice(cpus, 32))
    assert all(len(s) == 32 for s in slices)
    assert set.union(*slices) == set(range(128)), "slices must tile the mask"
    for i, a in enumerate(slices):
        for b in slices[i + 1:]:
            assert not (a & b), "worker slices must not overlap"


def test_worker_slice_wraps_when_the_mask_is_too_narrow(monkeypatch):
    """Fewer CPUs than workers*cap: wrap rather than hand out an empty set."""
    monkeypatch.setenv("PYTEST_XDIST_WORKER", "gw3")
    picked = _worker_slice(list(range(8)), 4)
    assert len(picked) == 4 and picked <= set(range(8))


def test_non_xdist_process_takes_the_first_slice(wide_machine, affinity_spy):
    _cap_cpu_affinity_for_thread_rlimit()
    assert affinity_spy == [set(range(_DEFAULT_TEST_CPU_CAP))]


# --- the no-op branches ------------------------------------------------------
def test_no_op_when_rlimit_is_unlimited(wide_machine, affinity_spy, monkeypatch):
    monkeypatch.setattr(
        resource, "getrlimit", lambda which: (resource.RLIM_INFINITY, resource.RLIM_INFINITY)
    )
    _cap_cpu_affinity_for_thread_rlimit()
    assert affinity_spy == []


def test_no_op_when_mask_already_small(wide_machine, affinity_spy, monkeypatch):
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(8)), **_SET)
    _cap_cpu_affinity_for_thread_rlimit()
    assert affinity_spy == []


@pytest.mark.parametrize("rank_var", _MPI_RANK_ENV)
def test_mpi_ranks_are_exempt(wide_machine, affinity_spy, monkeypatch, rank_var):
    """Never fight the MPI launcher's binding.

    Clamping every rank to the same low-numbered CPUs would pile the whole job
    onto one core set and serialise it -- worse than the problem being solved.
    """
    monkeypatch.setenv(rank_var, "0")
    _cap_cpu_affinity_for_thread_rlimit()
    assert affinity_spy == []


def test_denied_affinity_syscall_is_not_fatal(wide_machine, monkeypatch):
    """A cpuset/container that forbids sched_setaffinity must not break collection."""

    def _denied(pid, mask):
        raise OSError(1, "Operation not permitted")

    monkeypatch.setattr(os, "sched_setaffinity", _denied, **_SET)
    _cap_cpu_affinity_for_thread_rlimit()  # must not raise


# --- host stress test (opt-in) ----------------------------------------------
_CLIENT_SNIPPET = """
import jax, jax.numpy as jnp
jax.jit(lambda x: (x @ x.T).sum())(jnp.ones((32, 32))).block_until_ready()
"""
_N_CLIENTS = 5
_TOTAL_DEADLINE_S = 300.0


def _stress_precondition_skip() -> str | None:
    """Why this host cannot demonstrate the cap, or None when it can."""
    if not hasattr(os, "sched_getaffinity"):
        return "no CPU affinity API on this platform"
    if _is_mpi_rank():
        return "MPI rank: the cap is deliberately not applied here"
    soft, _hard = resource.getrlimit(resource.RLIMIT_NPROC)
    if soft == resource.RLIM_INFINITY:
        return "no thread ceiling on this host"
    n_cpus = len(os.sched_getaffinity(0))
    cap = _effective_cap(n_cpus, soft)
    if cap <= 0:
        return "cap disabled via LEGOESM_TEST_CPU_CAP"
    if n_cpus <= cap:
        return f"mask already {n_cpus} CPUs (<= cap {cap}); nothing to demonstrate"
    return None


@pytest.mark.slow
def test_many_clients_do_not_exhaust_threads():
    """End-to-end: several concurrent JAX CPU clients must all survive.

    Uncapped on a 256-core node this aborts every child with
    ``pthread_create() failed`` (EAGAIN); the children inherit this process's
    already-capped affinity mask, so they fit.  Marked slow: it spawns real
    interpreters and is a property of the HOST, not of any code under test.
    """
    skip = _stress_precondition_skip()
    if skip:
        pytest.skip(skip)
    env = {**os.environ, "JAX_PLATFORMS": "cpu"}  # the CPU client is what EAGAINs
    procs: list[subprocess.Popen] = []
    failures: list[str] = []
    try:
        for _ in range(_N_CLIENTS):
            procs.append(
                subprocess.Popen(
                    [sys.executable, "-c", _CLIENT_SNIPPET],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    env=env,
                )
            )
        # ONE deadline for the whole fan-out, not one per child: five sequential
        # per-child timeouts would let a hung run take 5x as long as intended.
        deadline = time.monotonic() + _TOTAL_DEADLINE_S
        for p in procs:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                failures.append("deadline exceeded before this client was reaped")
                break
            try:
                _out, err = p.communicate(timeout=remaining)
            except subprocess.TimeoutExpired:
                failures.append("client timed out")
                break
            if p.returncode != 0:
                failures.append(err.decode(errors="replace")[-400:])
    finally:
        # Never leave orphaned interpreters behind -- leaking processes is the
        # exact failure class this module exists to prevent.
        for p in procs:
            if p.poll() is None:
                p.kill()
                p.wait()
    assert not failures, (
        "concurrent JAX CPU clients died -- the conftest affinity cap is not "
        "holding XLA's Eigen thread pool under ``ulimit -u``:\n" + "\n".join(failures)
    )
