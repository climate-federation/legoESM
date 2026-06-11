"""Per-test isolation for the distributed ocean suites.

An armed MPI halo backend (set_halo_backend('mpi', layout)) is a
process GLOBAL: it leaks into every pad of every subsequent test in
the same pytest process — a later test's 'serial reference' then
silently dispatches through the leftover band layout (merge gate
8460566: a deterministic 1e-5 step-parity failure that vanished when
the test ran alone).  Reset after every test; tests that arm mid-test
are unaffected.
"""

import pytest


@pytest.fixture(autouse=True)
def _isolate_distributed_state():
    yield
    from legoesm.parallel.distributed import reset_distributed_topology
    reset_distributed_topology()
