"""Shared test fixtures for legoESM."""

import os
import resource
from pathlib import Path

#: Cap the CPU affinity mask BEFORE JAX is imported (below).  XLA sizes its
#: Eigen thread pool from the affinity mask, so on a many-core login node a
#: single JAX process opens far more threads than ``ulimit -u`` allows once a
#: few run at once, and the pool constructor aborts the interpreter:
#:
#:   F env.cc:93] Check failed: ret == 0 (11 vs. 0) Thread tf_XLAPjRtCpuClient
#:   creation via pthread_create() failed.
#:     @ Eigen::ThreadPoolTempl<>::ThreadPoolTempl()
#:
#: (11 = EAGAIN.)  It surfaces as "Fatal Python error: Aborted" mid-run, which
#: reads like a test failure but is the process running out of thread slots.
#:
#: Measured on a 256-core Levante login node (``ulimit -u`` = 2048), threads
#: opened by one client for a trivial jitted matmul:
#:
#:     CPUs    4     8    16    32    64   256
#:     threads 29    45    77   141   237   621
#:
#: Uncapped that is 621 threads each, so FOUR concurrent clients (2484) are the
#: first count over the 2048 ceiling; five were observed to fail 5/5.  At 32
#: CPUs a client takes ~141, so ~14 fit.  Capping costs nothing measurable --
#: these tests are XLA-compile-bound, not intra-op-parallelism-bound
#: (tests/test_corner_div_damp_nh.py: 166.9 s on 256 CPUs, 166.3 s on 8,
#: 152.7 s on 32).
#:
#: Set ``LEGOESM_TEST_CPU_CAP`` to a positive integer to force a cap, or to 0
#: (or any value <= 0) to disable and inherit the machine's full affinity mask.
_DEFAULT_TEST_CPU_CAP = 32

#: Threads one XLA CPU client opens ~= _THREADS_PER_CPU * ncpu + _THREADS_BASE,
#: fitted to the table above (slope (621-29)/(256-4) = 2.35).  Used to size the
#: cap against the ACTUAL thread ceiling rather than trusting a fixed 32, which
#: is only evidenced safe for ulimit -u = 2048.
_THREADS_PER_CPU = 2.4
_THREADS_BASE = 20.0
#: Fraction of the thread ceiling the test session may claim, leaving room for
#: the user's other processes (a login node already carries ~50 threads).
_RLIMIT_SHARE = 0.5

#: Env vars every common launcher sets in a rank's environment.  Under MPI the
#: launcher (or the batch scheduler) owns placement: each rank already gets its
#: own binding, and clamping every rank to the SAME low-numbered CPUs here would
#: pile all of them onto one core set and serialise the run.  So the cap is
#: skipped for MPI ranks -- they are separate processes with a launcher-assigned
#: mask, which is exactly the placement this function would otherwise destroy.
#: MPI does not standardise a rank env var, so detection is BEST-EFFORT: this
#: covers the launchers in use here.  For an unlisted launcher, disable the cap
#: explicitly with ``LEGOESM_TEST_CPU_CAP=0``.
_MPI_RANK_ENV = (
    "OMPI_COMM_WORLD_RANK",     # Open MPI
    "PMI_RANK",                 # MPICH / Intel MPI
    "PMIX_RANK",                # PMIx
    "SLURM_PROCID",             # srun-launched
    "MV2_COMM_WORLD_RANK",      # MVAPICH2
    "MPI_RANKID",               # IBM Platform MPI
    "LAMRANK",                  # legacy LAM/MPI
)


def _is_mpi_rank() -> bool:
    """True when this interpreter was launched as a rank of an MPI job."""
    return any(v in os.environ for v in _MPI_RANK_ENV)


def _requested_cap() -> int | None:
    """The user's explicit ``LEGOESM_TEST_CPU_CAP``, or None when unset/garbage.

    A value <= 0 disables capping entirely and is returned as 0.
    """
    raw = os.environ.get("LEGOESM_TEST_CPU_CAP")
    if raw is None:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return max(0, value)


def _xdist_worker_count() -> int:
    """How many pytest-xdist workers share this machine (1 when not under xdist)."""
    try:
        return max(1, int(os.environ.get("PYTEST_XDIST_WORKER_COUNT", "1")))
    except ValueError:
        return 1


def _effective_cap(n_cpus: int, soft_rlimit: int) -> int:
    """CPUs this process may use so every worker's XLA pool fits the ceiling.

    Budgets ``_RLIMIT_SHARE`` of the thread ceiling across the concurrent xdist
    workers and inverts the measured threads-per-CPU fit.  An explicit
    ``LEGOESM_TEST_CPU_CAP`` wins outright -- if a user names a number, honour
    it rather than second-guessing their machine.
    """
    requested = _requested_cap()
    if requested is not None:
        return requested
    budget = (soft_rlimit * _RLIMIT_SHARE) / _xdist_worker_count()
    allowed = int((budget - _THREADS_BASE) / _THREADS_PER_CPU)
    # Never below 4 (XLA needs a workable pool) nor above the documented default.
    return max(4, min(_DEFAULT_TEST_CPU_CAP, allowed))


def _worker_slice(cpus: list[int], cap: int) -> set[int]:
    """Pick this xdist worker's DISJOINT slice of *cpus*.

    Without this every worker inherits the controller's identical low-numbered
    mask and four XLA clients pile onto the same cores.  ``PYTEST_XDIST_WORKER``
    is ``gw0``, ``gw1``, ...; the slice wraps if the mask is too narrow to give
    everyone a private one.
    """
    worker = os.environ.get("PYTEST_XDIST_WORKER", "")
    index = 0
    if worker.startswith("gw") and worker[2:].isdigit():
        index = int(worker[2:])
    start = (index * cap) % len(cpus)
    picked = [cpus[(start + i) % len(cpus)] for i in range(cap)]
    return set(picked)


def _cap_cpu_affinity_for_thread_rlimit() -> None:
    """Shrink this process's CPU affinity so XLA's thread pool fits ``ulimit -u``.

    No-op when the platform has no affinity API (macOS), when running as an MPI
    rank (the launcher owns placement), when the thread rlimit is unlimited,
    when the mask is already small enough, or when the cap is disabled via
    ``LEGOESM_TEST_CPU_CAP=0``.
    """
    if not (hasattr(os, "sched_getaffinity") and hasattr(os, "sched_setaffinity")):
        return  # macOS / non-Linux: no affinity mask to shrink
    if _is_mpi_rank():
        return  # never fight the MPI launcher's binding
    try:
        soft, _hard = resource.getrlimit(resource.RLIMIT_NPROC)
    except (ValueError, OSError):  # pragma: no cover - platform dependent
        return
    if soft == resource.RLIM_INFINITY:
        return  # no thread ceiling to run into
    try:
        cpus = sorted(os.sched_getaffinity(0))
        cap = _effective_cap(len(cpus), soft)
        if cap <= 0 or len(cpus) <= cap:
            return
        os.sched_setaffinity(0, _worker_slice(cpus, cap))
    except OSError:  # pragma: no cover - affinity syscalls can be denied
        return  # a container/cpuset that forbids this is not our problem to fix


_cap_cpu_affinity_for_thread_rlimit()

import pytest  # noqa: E402  - must follow the affinity cap
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from legoesm.grids.cubed_sphere import create_cubed_sphere, CubedSphereGrid
from legoesm.core.field import Field
from legoesm.core.state import ShallowWaterState


def _neutralize_broken_mps_gelu_patch() -> None:
    """Undo ``jax-mps``'s broken monkeypatch of ``jax.nn.gelu`` / ``dot_product_attention``.

    The installed ``jax-mps`` plugin (Apple Metal) monkeypatches ``jax.nn.gelu``
    and ``jax.nn.dot_product_attention`` to route through fused ``mps.*``
    primitives — but those primitives ship NO vmap batching rule, so ANY
    ``vmap``-ed network (every ML-emulator: microphysics, GWD, SFNO) dies with
    ``NotImplementedError: Batching rule for 'mps.gelu' not implemented`` even
    under ``JAX_PLATFORMS=cpu`` (Metal is unused/broken here — see the
    metal-backend memory).  The fused kernel is worthless on CPU, so restore the
    stock JAX implementations the plugin saved.  No-op when the plugin is absent
    (e.g. Linux CI) or has not patched.
    """
    try:
        from jax_plugins.mps import ops as _mps_ops
    except Exception:
        return
    # The patch is applied LAZILY on first backend init; force it so the plugin
    # has saved the originals (``_gelu_original`` / ``_sdpa_original``) before we
    # read them back.
    jnp.zeros(())
    import jax.nn as _jnn
    from jax._src.nn import functions as _nnf

    original_gelu = getattr(_mps_ops, "_gelu_original", None)
    if original_gelu is not None and getattr(_jnn.gelu, "_mps_patched", False):
        _jnn.gelu = original_gelu
        _nnf.gelu = original_gelu
    original_sdpa = getattr(_mps_ops, "_sdpa_original", None)
    if original_sdpa is not None and getattr(
        _jnn.dot_product_attention, "_mps_patched", False
    ):
        _jnn.dot_product_attention = original_sdpa
        _nnf.dot_product_attention = original_sdpa


_neutralize_broken_mps_gelu_patch()


RESULTS_SUBDIRS = (
    "atmosphere/shallow_water",
    "atmosphere/hydrostatic",
    "atmosphere/nonhydrostatic",
    "ocean",
    "land",
    "sea_ice",
)


@pytest.fixture(scope="session", autouse=True)
def _ensure_results_tree() -> Path:
    """Ensure structured component result directories exist."""
    root = Path("results")
    for rel in RESULTS_SUBDIRS:
        (root / rel).mkdir(parents=True, exist_ok=True)
    return root


@pytest.fixture(scope="session")
def small_grid() -> CubedSphereGrid:
    """A small C8 grid for fast unit tests."""
    return create_cubed_sphere(8)


@pytest.fixture(scope="session")
def medium_grid() -> CubedSphereGrid:
    """A medium C24 grid for integration tests."""
    return create_cubed_sphere(24)


@pytest.fixture
def random_field(small_grid) -> Field:
    """A random scalar field on the small grid."""
    key = jax.random.PRNGKey(42)
    data = jax.random.normal(key, shape=(6, 8, 8))
    return Field(data=data, name="test", dims=("face", "x", "y"), units="1")


@pytest.fixture
def constant_state(small_grid) -> ShallowWaterState:
    """A constant state for testing conservation."""
    shape = (6, 8, 8)
    return ShallowWaterState(
        h=Field(data=jnp.ones(shape) * 1e4, name="h", dims=("face", "x", "y"), units="m"),
        u=Field(data=jnp.zeros(shape), name="u", dims=("face", "x", "y"), units="m/s"),
        v=Field(data=jnp.zeros(shape), name="v", dims=("face", "x", "y"), units="m/s"),
        h_s=Field(data=jnp.zeros(shape), name="h_s", dims=("face", "x", "y"), units="m"),
    )


@pytest.fixture
def conservation_gate():
    """The shared matrix conservation gates, so pytest tests assert PASS/FAIL
    with the SAME gates the test-matrix runners use (one source of truth).

    Returns the ``legoesm.experiments.matrix.gates`` module:

        def test_mass_conserved(conservation_gate):
            ok, notes = conservation_gate.mass_gate(
                True, "", mass_series, component="atmosphere")
            assert ok, notes
    """
    from legoesm.experiments.matrix import gates
    return gates
