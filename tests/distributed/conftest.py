"""Distributed test fixtures.

The MPI face layout is (re-)initialized PER TEST via ``cube_face_layout``
— a once-per-session init cannot work here because the autouse
``_isolate_distributed_state`` teardown resets the process-global
topology/layout/halo-backend after every test.
"""

import pytest

mpi4jax = pytest.importorskip("mpi4jax")
MPI = pytest.importorskip("mpi4py.MPI")


@pytest.fixture()
def cube_face_layout():
    """(Re-)initialize the active MPI face layout + arm the MPI halo backend.

    ``_isolate_distributed_state`` (autouse below) resets the
    process-global topology, layout and halo backend AFTER EVERY test —
    correct leak protection, but it also tears down any module- or
    session-scoped initialization after the first test.  Any test that
    relies on the ACTIVE layout (bare ``scatter_to_local`` /
    ``gather_to_global``) or on an armed ``'mpi'`` halo backend must
    therefore re-establish them at test SETUP through this fixture.
    (This mismatch was the long-standing 'broken local MPI stack':
    deterministic ``No layout provided and no active layout is set``
    failures that looked like an mpi4jax/jax version problem — the
    mpi4jax primitives were fine all along.)

    Safe to call repeatedly: after the reset the topology is None so
    ``initialize_distributed`` runs the clean first-init path;
    ``jax.distributed`` bootstrap is idempotent (single-node skips it).
    """
    from legoesm.parallel.distributed import (
        initialize_distributed,
        get_active_layout,
    )

    if get_active_layout() is None:
        global_n = max(MPI.COMM_WORLD.Get_size(), 2)
        initialize_distributed(global_n=global_n)
    layout = get_active_layout()
    if layout is None:
        pytest.skip("MPI layout unavailable on this rank configuration")
    return layout


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


@pytest.fixture(autouse=True)
def _isolate_distributed_state():
    """Reset the armed halo backend + active topology AFTER every test.

    A leftover ``set_halo_backend('mpi', layout)`` is a process global:
    it leaks into every pad of every later test in the same pytest
    process — a later test's 'serial reference' then silently
    dispatches through the stale band layout (merge gate 8460566: a
    deterministic 1e-5 step-parity failure that vanished when the test
    ran alone).  Mirrors tests/ocean/distributed/conftest.py.
    """
    yield
    from legoesm.parallel.distributed import reset_distributed_topology
    reset_distributed_topology()
