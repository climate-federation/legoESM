"""Real multi-rank gates for the #1516 per-rank GPU binding.

Run with::

    mpirun -np 2 .venv/bin/python -m pytest \\
        tests/distributed/test_gpu_pin_lockstep_mpi.py -q

Why a REAL launch and not the stubbed ``COMM_WORLD`` in
``tests/parallel/test_early_init_hardening.py``: a stub answers every
collective locally, so it cannot distinguish "both ranks entered the gather"
from "only the healthy rank did" — the second is the hang this code exists to
prevent, and the stubbed test would pass either way (codex).  Here the ranks
really have to meet.
"""
from __future__ import annotations

import os

import pytest

MPI = pytest.importorskip("mpi4py.MPI")

from legoesm.parallel import early_init as ei  # noqa: E402

pytestmark = pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() < 2,
    reason="needs mpirun -np 2 (single-rank launch has nothing to lockstep)")


@pytest.fixture
def fresh(monkeypatch):
    for var in ("CUDA_VISIBLE_DEVICES", "LEGOESM_HOST_CUDA_VISIBLE_DEVICES",
                "SLURM_LOCALID", "PALS_LOCAL_RANKID",
                "OMPI_COMM_WORLD_LOCAL_RANK", "MV2_COMM_WORLD_LOCAL_RANK",
                "SLURM_STEP_NUM_TASKS", "LEGOESM_ALLOW_SHARED_GPU",
                "CUDA_DEVICE_ORDER"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(ei, "_INITIALIZED", False)
    monkeypatch.setenv("SLURM_NTASKS", str(MPI.COMM_WORLD.Get_size()))
    yield
    MPI.COMM_WORLD.Barrier()


def test_node_local_split_agrees_with_hostnames(fresh):
    """``Split_type(COMM_TYPE_SHARED)`` is the node-local grouping we assume."""
    comm = MPI.COMM_WORLD
    node = comm.Split_type(MPI.COMM_TYPE_SHARED)
    n_local, node_rank = node.Get_size(), node.Get_rank()
    node.Free()
    assert 0 <= node_rank < n_local
    # Every rank on one host must get a DISTINCT node-local rank — the whole
    # premise of indexing the device list by it.
    peers = comm.allgather((MPI.Get_processor_name(), node_rank))
    mine = [nr for host, nr in peers if host == MPI.Get_processor_name()]
    assert len(set(mine)) == len(mine)


@pytest.mark.timeout(60)
def test_pin_failure_aborts_every_rank_not_just_the_bad_one(fresh,
                                                            monkeypatch):
    """A pin that fails on ONE rank must raise on ALL of them.

    Non-vacuous and deadlock-sensitive: make the failing rank return early
    instead of joining the gather and this test hangs until the timeout, or
    MPI aborts the job — either way it does not pass.
    """
    comm = MPI.COMM_WORLD
    bad_rank = comm.Get_size() - 1

    def _fake_pin(local_rank, n_local):
        if comm.Get_rank() == bad_rank:
            raise RuntimeError("simulated: more local ranks than GPUs")
        return None

    monkeypatch.setattr(ei, "pin_local_gpu", _fake_pin)
    with pytest.raises(RuntimeError, match="aborting every rank together"):
        ei.maybe_init_jax_distributed()


@pytest.mark.timeout(60)
def test_production_path_gives_each_rank_a_DISTINCT_device(fresh, monkeypatch):
    """End-to-end through ``maybe_init_jax_distributed``, not through a
    helper: after the call, no two ranks on this host hold the same
    ``CUDA_VISIBLE_DEVICES``.  That is the property #1516 lacked.

    The device list is faked so this runs on CPU-only test nodes; what is
    exercised for real is the node-local rank derivation and the collectives
    around it.
    """
    comm = MPI.COMM_WORLD
    n = comm.Get_size()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", ",".join(str(i) for i in range(n)))
    monkeypatch.setattr(ei, "openable_nvidia_nodes",
                        lambda: [f"/dev/nvidia{i}" for i in range(n)])
    if len(set(comm.allgather(MPI.Get_processor_name()))) > 1:
        pytest.skip("multi-node launch: would call jax.distributed.initialize")
    ei.maybe_init_jax_distributed()
    mine = os.environ["CUDA_VISIBLE_DEVICES"]
    all_pins = comm.allgather(mine)
    assert len(set(all_pins)) == n, f"ranks share a device: {all_pins}"
    assert os.environ["LEGOESM_HOST_CUDA_VISIBLE_DEVICES"].count(",") == n - 1


@pytest.mark.timeout(60)
def test_healthy_multi_rank_launch_still_returns(fresh, monkeypatch):
    """The collective added for the failure path must not stall the happy one.

    ``pin_local_gpu`` is neutralised so this runs identically on CPU-only
    test nodes; what is exercised is the gather sequencing around it.
    """
    monkeypatch.setattr(ei, "pin_local_gpu", lambda local_rank, n_local: None)
    # Single-node launch -> False (no jax.distributed); multi-node -> the real
    # initialize() would run, which a test must not do, so assert only the
    # single-node case and skip otherwise.
    comm = MPI.COMM_WORLD
    if len(set(comm.allgather(MPI.Get_processor_name()))) > 1:
        pytest.skip("multi-node launch: would call jax.distributed.initialize")
    assert ei.maybe_init_jax_distributed() is False


@pytest.mark.timeout(60)
def test_duplicate_binding_is_detected_AND_collective(fresh, monkeypatch):
    """Every rank sees one device called '0' and the SAME /dev set: they are
    all on one GPU.  Must be caught, and caught on EVERY rank.

    This is the gate the unit test cannot cover — the unit stub answers the
    fingerprint gather locally, so it cannot show the failure is collective
    (codex).  Here the ranks really meet in ``COMM_WORLD.allgather``.
    """
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(ei, "openable_nvidia_nodes", lambda: ["/dev/nvidia0"])
    with pytest.raises(RuntimeError, match="resolve to the SAME GPU"):
        ei.maybe_init_jax_distributed()


@pytest.mark.timeout(60)
def test_duplicate_gate_survives_an_undergrouping_shm_split(fresh, monkeypatch):
    """The reason the fingerprint gather is on COMM_WORLD, not node_comm.

    Model a container launch where the shared-memory split puts every rank in
    its own group.  A node-communicator gather would then see one entry, find
    no duplicate, and go silent exactly when it matters (codex).  The
    hostname-filtered world gather still catches it.
    """
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(ei, "openable_nvidia_nodes", lambda: ["/dev/nvidia0"])
    monkeypatch.setattr(MPI, "COMM_WORLD", _UnderGroupingComm(MPI.COMM_WORLD))
    with pytest.raises(RuntimeError, match="resolve to the SAME GPU"):
        ei.maybe_init_jax_distributed()


class _Singleton:
    """A shared-memory communicator that under-groups: one rank per group."""

    Get_rank = staticmethod(lambda: 0)
    Get_size = staticmethod(lambda: 1)
    Free = staticmethod(lambda: None)


class _UnderGroupingComm:
    """The real COMM_WORLD, but its shared-memory split under-groups."""

    def __init__(self, real):
        self._real = real

    def __getattr__(self, name):
        return getattr(self._real, name)

    def Split_type(self, *_a, **_kw):
        return _Singleton()


@pytest.mark.timeout(60)
def test_shared_gpu_override_lets_the_duplicate_launch_through(fresh,
                                                               monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("LEGOESM_ALLOW_SHARED_GPU", "1")
    monkeypatch.setattr(ei, "openable_nvidia_nodes", lambda: ["/dev/nvidia0"])
    comm = MPI.COMM_WORLD
    if len(set(comm.allgather(MPI.Get_processor_name()))) > 1:
        pytest.skip("multi-node launch: would call jax.distributed.initialize")
    assert ei.maybe_init_jax_distributed() is False
