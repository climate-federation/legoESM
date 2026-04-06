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
