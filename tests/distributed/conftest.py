"""Distributed test fixtures — initialize MPI layout once per session."""

import pytest

mpi4jax = pytest.importorskip("mpi4jax")
MPI = pytest.importorskip("mpi4py.MPI")


@pytest.fixture(autouse=True, scope="session")
def _init_mpi_layout():
    """Initialize the MPI distributed layout for the test session.

    Creates a DistributedLayout that maps MPI ranks to cubed-sphere
    faces.  Tests that need ``scatter_to_local`` / ``gather_to_global``
    rely on this active layout.
    """
    from legoesm.parallel.distributed import (
        initialize_distributed,
        get_active_layout,
    )

    # Only initialize if not already active (e.g., from a parent conftest)
    if get_active_layout() is not None:
        return

    comm = MPI.COMM_WORLD
    n_procs = comm.Get_size()

    # Use the smallest grid that still tests face decomposition
    # global_n must be divisible by n_procs for even face distribution
    global_n = max(n_procs, 2)
    try:
        initialize_distributed(global_n=global_n)
    except Exception:
        # If initialization fails (e.g., single rank), tests will skip
        # via the scatter_to_local ValueError.
        pass


@pytest.fixture(autouse=True)
def _mpi_test_isolation():
    """Resynchronize ranks and drain stray messages around every test.

    Each distributed test builds its own halo-exchange scenario (different
    grids, layouts, and ``sendrecv`` buffer sizes).  Without an explicit
    barrier between tests the ranks can drift out of lock-step, and a
    message posted by one test that is not consumed before the next test
    starts is later matched — by source/tag — against a *different-sized*
    receive buffer, raising ``MPI_ERR_TRUNCATE`` and aborting the whole
    job.  This was reproducible as ``test_latlon_mpi_step`` poisoning
    ``test_latlon_mpi_tripole`` (order-dependent, version-independent).

    The fix is standard MPI test hygiene: barrier-sync before and after
    each test, and drain any straggler messages (as raw bytes, since
    mpi4jax uses the buffer protocol) so they cannot truncate the next
    test's receive.
    """
    comm = MPI.COMM_WORLD
    if comm.Get_size() < 2:
        # Single-rank runs have no peer messages to drain or sync.
        yield
        return

    def _drain() -> None:
        status = MPI.Status()
        while comm.Iprobe(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status):
            count = status.Get_count(MPI.BYTE)
            buf = bytearray(count)
            comm.Recv([buf, MPI.BYTE],
                      source=status.Get_source(), tag=status.Get_tag())

    comm.Barrier()
    yield
    comm.Barrier()
    _drain()
    comm.Barrier()
