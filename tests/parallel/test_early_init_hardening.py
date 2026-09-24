"""Route-B hardening gates for ``legoesm.parallel.early_init`` (audit item 6).

Pure-env helpers tested directly (no cluster needed); the silent-fallback
guard tested against a stubbed ``jax`` module.  The REAL launch paths stay
covered by the multicontroller self-spawn suites.
"""
from __future__ import annotations

import os
import sys
import types

import pytest

from legoesm.parallel import early_init as ei

_ALL_ENV = (
    "SLURM_STEP_NUM_TASKS", "SLURM_NTASKS", "OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "PALS_LOCAL_RANKID",
    "OMPI_COMM_WORLD_LOCAL_RANK", "MV2_COMM_WORLD_LOCAL_RANK",
    "SLURM_LOCALID",
    "CUDA_VISIBLE_DEVICES", "NCCL_NET_PLUGIN", "LD_LIBRARY_PATH",
    "LD_PRELOAD", "NCCL_SOCKET_IFNAME", "NCCL_DEBUG", "SLURM_NNODES",
    "SLURM_JOB_NUM_NODES", "PALS_NNODES",
    "LEGOESM_ALLOW_SINGLE_PROCESS_UNDER_MPI",
    "LEGOESM_HOST_CUDA_VISIBLE_DEVICES", "LEGOESM_ALLOW_SHARED_GPU",
    "CUDA_DEVICE_ORDER",
    # Platform selection gates the post-MPI-pin escalation; the suite itself
    # often runs under JAX_PLATFORMS=cpu, which must not leak into tests.
    "JAX_PLATFORMS", "JAX_PLATFORM_NAME",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for v in _ALL_ENV:
        monkeypatch.delenv(v, raising=False)
    # The escalation probe reads /proc/self/fd of the PYTEST process, which
    # may legitimately hold nvidia fds on a GPU dev box; default it to the
    # no-driver answer so tests are deterministic.  Escalation tests
    # re-patch it to True explicitly.
    monkeypatch.setattr(ei, "_cuda_driver_fds_open", lambda: False)
    yield


def test_launcher_world_size_priority(monkeypatch):
    assert ei.launcher_world_size() == 0
    monkeypatch.setenv("SLURM_NTASKS", "16")
    assert ei.launcher_world_size() == 16  # allocation-wide fallback
    monkeypatch.setenv("PMI_SIZE", "8")
    assert ei.launcher_world_size() == 8   # PMI (actual launcher) wins
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "4")
    assert ei.launcher_world_size() == 4   # OMPI outranks PMI
    # Inner srun step: STEP size outranks everything.
    monkeypatch.setenv("SLURM_STEP_NUM_TASKS", "2")
    assert ei.launcher_world_size() == 2
    monkeypatch.setenv("SLURM_STEP_NUM_TASKS", "garbage")
    assert ei.launcher_world_size() == 4   # non-numeric skipped, not crash


def test_pals_local_device_ids(monkeypatch):
    # Unpinned, no PALS local rank: default [0].
    assert ei._pals_local_device_ids() == [0]
    # Unpinned + PALS local rank: index by it (node-sharing ranks bind
    # DIFFERENT GPUs — the contended-GPU-0 hazard).
    monkeypatch.setenv("PALS_LOCAL_RANKID", "3")
    assert ei._pals_local_device_ids() == [3]
    # Shim-pinned CUDA_VISIBLE_DEVICES: exactly one visible device -> [0]
    # regardless of the local rank.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    assert ei._pals_local_device_ids() == [0]
    # MULTI-device visible list is NOT a pin: index within it by local
    # rank (codex: '0,1,2,3' must not bind every rank to GPU 0); an
    # out-of-range index RAISES below, it is not clamped.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    assert ei._pals_local_device_ids() == [3]
    # More local ranks than visible devices = LAUNCH error, loud — a clamp
    # would silently oversubscribe the last GPU (codex).
    monkeypatch.setenv("PALS_LOCAL_RANKID", "7")
    with pytest.raises(RuntimeError, match="more local ranks"):
        ei._pals_local_device_ids()


def test_local_rank_launcher_families(monkeypatch):
    # Open MPI local rank serves the explicit-coordinator path too (codex:
    # multi-GPU CVD + OMPI multi-rank nodes must not all bind GPU 0).
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    monkeypatch.setenv("OMPI_COMM_WORLD_LOCAL_RANK", "2")
    assert ei._pals_local_device_ids() == [2]
    monkeypatch.delenv("OMPI_COMM_WORLD_LOCAL_RANK")
    # SLURM_LOCALID honored only on a genuine multi-task launch.
    monkeypatch.setenv("SLURM_LOCALID", "1")
    assert ei._pals_local_device_ids() == [0]  # single-task: ignored
    monkeypatch.setenv("SLURM_NTASKS", "4")
    assert ei._pals_local_device_ids() == [1]
    # srun -n1 inside a larger allocation: STEP size 1 wins -> ignored.
    monkeypatch.setenv("SLURM_STEP_NUM_TASKS", "1")
    assert ei._pals_local_device_ids() == [0]


def _stub_jax(monkeypatch, process_count: int):
    stub = types.ModuleType("jax")
    stub.process_count = lambda: process_count
    monkeypatch.setitem(sys.modules, "jax", stub)
    return stub


def test_fallback_guard_raises_on_mismatch(monkeypatch):
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "4")
    _stub_jax(monkeypatch, process_count=1)
    with pytest.raises(RuntimeError, match="un-federated copies"):
        ei.check_no_silent_process_fallback()


def test_fallback_guard_passes_when_federated(monkeypatch):
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "4")
    _stub_jax(monkeypatch, process_count=4)
    ei.check_no_silent_process_fallback()  # no raise


def test_fallback_guard_noop_without_launcher(monkeypatch):
    _stub_jax(monkeypatch, process_count=1)
    ei.check_no_silent_process_fallback()  # declared=0 -> no-op


def test_fallback_guard_override_env(monkeypatch):
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "4")
    monkeypatch.setenv("LEGOESM_ALLOW_SINGLE_PROCESS_UNDER_MPI", "1")
    _stub_jax(monkeypatch, process_count=1)
    ei.check_no_silent_process_fallback()  # override honored


def test_nccl_report_missing_plugin_flag(tmp_path, monkeypatch):
    # Single node: never flagged, plugin or not.
    r = ei.nccl_transport_report()
    assert r["missing_net_plugin_multi_node"] is False
    # Multi-node without a plugin: flagged.
    monkeypatch.setenv("SLURM_NNODES", "2")
    r = ei.nccl_transport_report()
    assert r["missing_net_plugin_multi_node"] is True
    assert r["n_nodes_declared"] == 2
    # Plugin on LD_LIBRARY_PATH clears the flag.
    lib = tmp_path / "libnccl-net-ofi.so"
    lib.write_bytes(b"")
    monkeypatch.setenv("LD_LIBRARY_PATH", str(tmp_path))
    r = ei.nccl_transport_report()
    assert r["missing_net_plugin_multi_node"] is False
    assert r["nccl_net_plugin"] == str(lib)
    # Explicit NCCL_NET_PLUGIN also clears it.
    monkeypatch.delenv("LD_LIBRARY_PATH")
    monkeypatch.setenv("NCCL_NET_PLUGIN", "/opt/nccl/libnccl-net.so")
    r = ei.nccl_transport_report()
    assert r["missing_net_plugin_multi_node"] is False
    # SPACE-separated LD_PRELOAD (the canonical ld.so form) clears it too —
    # a colon-only split missed it and falsely flagged socket fallback
    # (pre-merge codex finding).
    monkeypatch.delenv("NCCL_NET_PLUGIN")
    monkeypatch.setenv(
        "LD_PRELOAD", f"/other/lib.so {lib}")
    r = ei.nccl_transport_report()
    assert r["missing_net_plugin_multi_node"] is False
    assert r["nccl_net_plugin"] == str(lib)


def test_nccl_report_records_knobs(monkeypatch):
    monkeypatch.setenv("NCCL_SOCKET_IFNAME", "hsn")
    monkeypatch.setenv("NCCL_DEBUG", "INFO")
    r = ei.nccl_transport_report()
    assert r["nccl_socket_ifname"] == "hsn"
    assert r["nccl_debug"] == "INFO"


# --- #1516: single-node multi-rank must not pile every rank onto GPU 0 ---


def test_host_visible_gpus_prefers_cuda_visible_devices(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2, 3")
    assert ei._host_visible_gpus() == ["2", "3"]
    # Empty string is an explicit "no GPUs", not "unset".
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert ei._host_visible_gpus() == []


def test_pin_local_gpu_narrows_visible_devices(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    assert ei.pin_local_gpu(local_rank=2, n_local=4) == "2"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "2"
    # Non-contiguous / renamed device lists are indexed, not assumed.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,7")
    assert ei.pin_local_gpu(local_rank=1, n_local=2) == "7"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "7"


def test_pin_local_gpu_noop_cases(monkeypatch):
    # Already shim-pinned to one device (#693): leave it alone.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    assert ei.pin_local_gpu(local_rank=0, n_local=2) is None
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "3"
    # One rank on the host: it may legitimately use every GPU it sees.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    assert ei.pin_local_gpu(local_rank=0, n_local=1) is None
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0,1"
    # CPU-only host: nothing to pin.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert ei.pin_local_gpu(local_rank=0, n_local=2) is None


def test_pin_local_gpu_more_ranks_than_gpus_raises(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    with pytest.raises(RuntimeError, match="more local ranks than GPUs"):
        ei.pin_local_gpu(local_rank=0, n_local=4)


def _stub_mpi(monkeypatch, rank: int, hosts: list[str], peer_errs=None,
              node_fps=None):
    """Minimal ``mpi4py.MPI.COMM_WORLD`` for the single-node early return.

    ``allgather`` echoes THIS rank's contribution from every rank, except the
    hostname gather (which returns ``hosts``) and the pin-error gather when
    ``peer_errs`` models other ranks failing while this one succeeds.
    """
    import socket

    def _allgather(x):
        if x == socket.gethostname():
            return hosts
        if isinstance(x, tuple) and len(x) == 2 and x[0] in hosts:
            # (hostname, fingerprint) — the duplicate-binding gather.  By
            # default every peer reports a DISTINCT device set (correct
            # pinning); ``node_fps`` overrides it per test.
            fps = node_fps if node_fps is not None else [
                x[1] if i == rank else ((f"/dev/peer{i}",), None, str(i))
                for i in range(len(hosts))]
            return list(zip(hosts, fps))
        if peer_errs is not None and x is None:
            return peer_errs
        return [x] * len(hosts)

    # Node-local sub-communicator: all the stubbed ranks share one node.
    node_comm = types.SimpleNamespace(
        Get_rank=lambda: rank, Get_size=lambda: len(hosts),
        Free=lambda: None,
    )
    comm = types.SimpleNamespace(
        Get_rank=lambda: rank,
        Get_size=lambda: len(hosts),
        allgather=_allgather,
        Split_type=lambda _t: node_comm,
    )
    mpi = types.ModuleType("mpi4py.MPI")
    mpi.COMM_WORLD = comm
    mpi.COMM_TYPE_SHARED = 0
    pkg = types.ModuleType("mpi4py")
    pkg.MPI = mpi
    monkeypatch.setitem(sys.modules, "mpi4py", pkg)
    monkeypatch.setitem(sys.modules, "mpi4py.MPI", mpi)


@pytest.mark.parametrize("rank,expect", [(0, "0"), (1, "1")])
def test_single_node_multi_rank_pins_before_early_return(
        monkeypatch, rank, expect):
    """The #1516 gate: the single-node path binds rank -> device.

    Non-vacuous by construction — deleting the ``pin_local_gpu`` call in
    ``maybe_init_jax_distributed`` leaves CUDA_VISIBLE_DEVICES at the full
    list and this assertion fails.
    """
    monkeypatch.setattr(ei, "_INITIALIZED", False)
    monkeypatch.setenv("SLURM_NTASKS", "2")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    _stub_mpi(monkeypatch, rank=rank, hosts=["node01", "node01"])
    # Single node -> returns False (no jax.distributed), but pinned first.
    assert ei.maybe_init_jax_distributed() is False
    assert os.environ["CUDA_VISIBLE_DEVICES"] == expect


def test_host_visible_gpus_counts_only_openable_dev_nodes(monkeypatch, tmp_path):
    """Unset CVD: enumerate from /dev nodes this process can OPEN.

    Codex finding: /proc/driver/nvidia/gpus stays host-global under a device
    cgroup, so a 4-GPU host with only GPU 0 granted counted 4 and ranks were
    pinned to devices they cannot open.
    """
    import glob as _glob
    granted = tmp_path / "nvidia0"
    granted.write_bytes(b"")
    denied = tmp_path / "nvidia1"  # never created: not openable by us
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(_glob, "glob",
                        lambda _pat: [str(granted), str(denied)])
    # CUDA renumbers what it can see from 0 -> one usable device -> ["0"].
    assert ei._host_visible_gpus() == ["0"]
    # Both openable -> two ordinals, so a 2-rank node pins 0 and 1.
    granted2 = tmp_path / "nvidia1"
    granted2.write_bytes(b"")
    assert ei._host_visible_gpus() == ["0", "1"]


def test_pin_local_gpu_stashes_the_host_device_list(monkeypatch):
    """Overwriting CVD must not destroy the host's real GPU count (GLM-5.2)."""
    monkeypatch.delenv("LEGOESM_HOST_CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    ei.pin_local_gpu(local_rank=1, n_local=4)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"
    assert os.environ["LEGOESM_HOST_CUDA_VISIBLE_DEVICES"] == "0,1,2,3"


def test_pin_local_gpu_out_of_range_local_rank_raises(monkeypatch):
    # n_local FITS the GPU count, but this rank's launcher-reported local
    # index does not (a stale SLURM_LOCALID).  Distinct message from the
    # too-few-GPUs case: they point at different knobs (codex).
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    with pytest.raises(RuntimeError, match="inconsistent launcher rank"):
        ei.pin_local_gpu(local_rank=3, n_local=2)


def test_launcher_local_rank_outranks_mpi_rank_order(monkeypatch):
    """`srun --distribution=cyclic`: SLURM_LOCALID, not the MPI rank order.

    MPI rank 1 is node-local #0 here; a hostname-order derivation would pin
    it to GPU 1 and collide with whichever rank SLURM calls local #1.
    """
    monkeypatch.setattr(ei, "_INITIALIZED", False)
    monkeypatch.setenv("SLURM_NTASKS", "2")
    monkeypatch.setenv("SLURM_LOCALID", "0")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    _stub_mpi(monkeypatch, rank=1, hosts=["node01", "node01"])
    assert ei.maybe_init_jax_distributed() is False
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"


def test_pin_failure_on_a_PEER_rank_raises_here_too(monkeypatch):
    """Lockstep abort (GLM-5.2).

    A heterogeneous allocation fails the pin on one node only.  Without the
    collective gather those ranks die and the healthy ones block forever in
    the jax.distributed rendezvous — a hang, not a failure.  This rank's own
    pin SUCCEEDS; it must still raise.
    """
    monkeypatch.setattr(ei, "_INITIALIZED", False)
    monkeypatch.setenv("SLURM_NTASKS", "2")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    _stub_mpi(monkeypatch, rank=0, hosts=["node01", "node01"],
              peer_errs=[None, "4 ranks share this host but only 2 GPU(s)"])
    with pytest.raises(RuntimeError, match="aborting every rank together"):
        ei.maybe_init_jax_distributed()


def test_openable_requires_write_access(tmp_path):
    """CUDA needs R/W on the device node; an O_RDONLY probe would pass a
    read-only cgroup grant and fail later inside backend creation (codex).

    Non-vacuous: reverting ``_openable`` to O_RDONLY makes this fail.
    """
    rw = tmp_path / "rw"
    rw.write_bytes(b"")
    ro = tmp_path / "ro"
    ro.write_bytes(b"")
    ro.chmod(0o444)
    if os.access(str(ro), os.W_OK):  # running as root: the probe cannot fail
        pytest.skip("root can open a 0444 file O_RDWR")
    assert ei._openable(str(rw)) is True
    assert ei._openable(str(ro)) is False


def test_two_ranks_resolving_to_one_gpu_is_a_hard_error(monkeypatch):
    """`CUDA_VISIBLE_DEVICES=0 mpirun -np 2`: one visible device looks exactly
    like a correct --gpu-bind shim from a single rank's environment.  The
    device SET the kernel exposes is what separates them (codex).

    Non-vacuous: drop the duplicate-fingerprint gate and this returns False.
    """
    monkeypatch.setattr(ei, "_INITIALIZED", False)
    monkeypatch.setenv("SLURM_NTASKS", "2")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(ei, "openable_nvidia_nodes",
                        lambda: ["/dev/nvidia0", "/dev/nvidia1"])
    same = (("/dev/nvidia0", "/dev/nvidia1"), None, "0")
    _stub_mpi(monkeypatch, rank=0, hosts=["node01", "node01"],
              node_fps=[same, same])
    with pytest.raises(RuntimeError, match="resolve to the SAME GPU"):
        ei.maybe_init_jax_distributed()


def test_gpu_bind_shim_with_distinct_device_sets_is_accepted(monkeypatch):
    """The legitimate #693 shim: each rank sees ONE device called '0', but a
    DIFFERENT physical node.  Must not trip the duplicate gate."""
    monkeypatch.setattr(ei, "_INITIALIZED", False)
    monkeypatch.setenv("SLURM_NTASKS", "2")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(ei, "openable_nvidia_nodes", lambda: ["/dev/nvidia0"])
    _stub_mpi(monkeypatch, rank=0, hosts=["node01", "node01"],
              node_fps=[(("/dev/nvidia0",), None, "0"), (("/dev/nvidia1",), None, "0")])
    assert ei.maybe_init_jax_distributed() is False
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"  # untouched


@pytest.mark.parametrize("masked", ["", "-1", "NoDevFiles"])
def test_cpu_masked_allocation_is_not_a_gpu(monkeypatch, masked):
    """SLURM/NVIDIA mask GPUs with non-empty STRINGS on a CPU-only
    allocation.  Reading them as one visible device pinned ranks to device
    '-1' and tripped the duplicate gate on a CPU run (codex).
    """
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", masked)
    assert ei._host_visible_gpus() == []
    assert ei.pin_local_gpu(local_rank=0, n_local=4) is None
    # ...and the whole multi-rank path stays quiet, even where /dev nodes
    # are still openable (a CPU job on a GPU node).
    monkeypatch.setattr(ei, "_INITIALIZED", False)
    monkeypatch.setenv("SLURM_NTASKS", "2")
    monkeypatch.setattr(ei, "openable_nvidia_nodes",
                        lambda: ["/dev/nvidia0", "/dev/nvidia1"])
    same = (("/dev/nvidia0", "/dev/nvidia1"), None, masked)
    _stub_mpi(monkeypatch, rank=0, hosts=["node01", "node01"],
              node_fps=[same, same])
    assert ei.maybe_init_jax_distributed() is False


@pytest.mark.parametrize("value,allowed", [("1", True), ("0", False)])
def test_shared_gpu_override_only_on_exact_1(monkeypatch, value, allowed):
    """A deliberate MPS/shared-GPU launch needs an escape hatch (codex) — but
    `=0` must NOT open it, matching LEGOESM_ALLOW_SINGLE_PROCESS_UNDER_MPI.
    """
    monkeypatch.setattr(ei, "_INITIALIZED", False)
    monkeypatch.setenv("SLURM_NTASKS", "2")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("LEGOESM_ALLOW_SHARED_GPU", value)
    monkeypatch.setattr(ei, "openable_nvidia_nodes", lambda: ["/dev/nvidia0"])
    same = (("/dev/nvidia0",), None, "0")
    _stub_mpi(monkeypatch, rank=0, hosts=["node01", "node01"],
              node_fps=[same, same])
    if allowed:
        assert ei.maybe_init_jax_distributed() is False
    else:
        with pytest.raises(RuntimeError, match="resolve to the SAME GPU"):
            ei.maybe_init_jax_distributed()


def test_shm_split_undergrouping_falls_back_to_hostnames(monkeypatch):
    """If MPI puts co-located ranks in SEPARATE shared-memory groups (a
    container without a shared /dev/shm), n_local would be 1, the pin a
    silent no-op, and every rank back on GPU 0 — the original defect
    (GLM-5.2).  The larger of the two groupings wins.

    Non-vacuous: take the shared-memory size unconditionally and this leaves
    CUDA_VISIBLE_DEVICES at '0,1'.
    """
    monkeypatch.setattr(ei, "_INITIALIZED", False)
    monkeypatch.setenv("SLURM_NTASKS", "2")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    _stub_mpi(monkeypatch, rank=1, hosts=["node01", "node01"])
    # Model the under-grouping: every rank alone in its shm communicator.
    import sys as _sys
    _sys.modules["mpi4py"].MPI.COMM_WORLD.Split_type = lambda _t: (
        types.SimpleNamespace(Get_rank=lambda: 0, Get_size=lambda: 1,
                              Free=lambda: None))
    assert ei.maybe_init_jax_distributed() is False
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"


def test_shared_gpu_flag_permits_oversubscription(monkeypatch):
    """The flag now means what its name says: 4 ranks on 2 GPUs share them
    round-robin instead of aborting (GLM-5.2)."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setenv("LEGOESM_ALLOW_SHARED_GPU", "1")
    assert ei.pin_local_gpu(local_rank=2, n_local=4) == "0"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    assert ei.pin_local_gpu(local_rank=3, n_local=4) == "1"


# --- #1516 follow-up: the pin must precede MPI_Init (jobs 26829100/26846811,
# both measured on Levante: a pin applied after `from mpi4py import MPI` is
# silently ignored by the already-initialised CUDA driver while
# CUDA_VISIBLE_DEVICES reads as correctly narrowed per rank).  The ordering
# gate itself lives in tests/unit/test_early_init.py::
# test_maybe_init_pins_before_mpi_import; here: the n_local=None early-pin
# contract, the deferred-collective failure, and the fd escalation.


def test_pin_local_gpu_unknown_n_local_narrows(monkeypatch):
    """Pre-MPI early pin: n_local=None narrows by launcher local rank, and
    rank 0 must NOT hit the single-process no-op (the caller has already
    established a multi-rank launch)."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    assert ei.pin_local_gpu(local_rank=0, n_local=None) == "0"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    assert ei.pin_local_gpu(local_rank=1, n_local=None) == "1"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"


def test_pin_local_gpu_unknown_n_local_oversubscription(monkeypatch):
    """n_local=None: the rank index is the oversubscription witness — local
    rank i implies >= i+1 ranks on this host."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    with pytest.raises(RuntimeError, match="more local ranks than GPUs"):
        ei.pin_local_gpu(local_rank=2, n_local=None)
    # The shared-GPU opt-in spreads round-robin instead (same as n_local>n).
    monkeypatch.setenv("LEGOESM_ALLOW_SHARED_GPU", "1")
    assert ei.pin_local_gpu(local_rank=2, n_local=None) == "0"


def test_early_pin_failure_defers_to_the_collective_raise(monkeypatch):
    """A stale SLURM_LOCALID fails the PRE-MPI pin; the raise must still ride
    the collective gather (lockstep abort), never fire before MPI_Init."""
    monkeypatch.setattr(ei, "_INITIALIZED", False)
    monkeypatch.setenv("SLURM_NTASKS", "2")
    monkeypatch.setenv("SLURM_LOCALID", "5")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    _stub_mpi(monkeypatch, rank=0, hosts=["node01", "node01"])
    with pytest.raises(RuntimeError, match="aborting every rank together"):
        ei.maybe_init_jax_distributed()
    # The failed early pin must not have half-narrowed the visibility.
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0,1"


def test_post_mpi_pin_with_live_cuda_driver_is_refused(monkeypatch):
    """No launcher local-rank var -> the pin falls back to after MPI_Init.
    When the process already holds /dev/nvidia* fds the driver has
    snapshotted the pre-pin CUDA_VISIBLE_DEVICES and the narrowing would be
    silently ignored (#1516, jobs 26829100/26846811): refuse loudly, in
    lockstep."""
    monkeypatch.setattr(ei, "_INITIALIZED", False)
    monkeypatch.setenv("SLURM_NTASKS", "2")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setattr(ei, "_cuda_driver_fds_open", lambda: True)
    _stub_mpi(monkeypatch, rank=0, hosts=["node01", "node01"])
    with pytest.raises(RuntimeError, match="AFTER MPI_Init"):
        ei.maybe_init_jax_distributed()


def test_post_mpi_pin_cpu_platform_is_not_refused(monkeypatch):
    """JAX_PLATFORMS=cpu: no CUDA backend will ever be created, so the
    post-MPI fallback pin must stay a working no-risk narrowing even with
    nvidia fds open (a CPU MPI test run on a GPU node)."""
    monkeypatch.setattr(ei, "_INITIALIZED", False)
    monkeypatch.setenv("SLURM_NTASKS", "2")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.setattr(ei, "_cuda_driver_fds_open", lambda: True)
    _stub_mpi(monkeypatch, rank=1, hosts=["node01", "node01"])
    assert ei.maybe_init_jax_distributed() is False
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"


def test_early_pin_applies_before_the_post_mpi_machinery(monkeypatch):
    """With a launcher local-rank var the pin happens pre-MPI; the post-MPI
    pin then sees a single visible device and no-ops, and the fd escalation
    must NOT fire (the early pin preceded any driver snapshot)."""
    monkeypatch.setattr(ei, "_INITIALIZED", False)
    monkeypatch.setenv("SLURM_NTASKS", "2")
    monkeypatch.setenv("OMPI_COMM_WORLD_LOCAL_RANK", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    # Driver fds "open": irrelevant, the early pin came first.
    monkeypatch.setattr(ei, "_cuda_driver_fds_open", lambda: True)
    _stub_mpi(monkeypatch, rank=1, hosts=["node01", "node01"])
    assert ei.maybe_init_jax_distributed() is False
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"
    assert os.environ["LEGOESM_HOST_CUDA_VISIBLE_DEVICES"] == "0,1"


def test_hostname_undergrouping_falls_back_to_the_shm_split(monkeypatch):
    """The opposite direction of the previous test (codex): ranks on ONE node
    reporting different hostnames (UTS namespaces, short vs FQDN).  The
    hostname grouping says 1 rank per node; the shared-memory split knows
    better and must win.

    Non-vacuous: use the hostname grouping unconditionally and the pin
    becomes a no-op, leaving CUDA_VISIBLE_DEVICES at '0,1'.
    """
    monkeypatch.setattr(ei, "_INITIALIZED", False)
    monkeypatch.setenv("SLURM_NTASKS", "2")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    # Two DIFFERENT hostname strings for the same physical node.  Note the
    # aliasing ALSO defeats the (pre-existing, untouched) single-node
    # decision, so this run takes the multi-node branch — hence the jax stub.
    _stub_mpi(monkeypatch, rank=1, hosts=["node01", "node01.cluster.local"])
    jax = _stub_jax(monkeypatch, process_count=2)
    jax.distributed = types.SimpleNamespace(initialize=lambda **kw: None)
    assert ei.maybe_init_jax_distributed() is True
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"


@pytest.mark.parametrize(
    "jax_platforms, jax_platform_name, expect_non_gpu",
    [
        # A GPU selection, in every spelling JAX accepts.  `gpu` is an alias
        # that expands to ['cuda', 'rocm'], and `rocm` IS a GPU platform:
        # calling either "non-GPU" would suppress the pin refusal on exactly
        # the hardware it protects (the dangerous direction).
        ("cuda", None, False),
        ("CUDA", None, False),
        ("gpu", None, False),
        ("rocm", None, False),
        ("cuda,cpu", None, False),
        ("cpu,cuda", None, False),
        # A genuinely non-GPU selection.
        ("cpu", None, True),
        (" cpu ", None, True),
        ("tpu", None, True),
        # Unset / empty / punctuation-only: GPU is still possible, so the
        # refusal must stay armed.  A bare "," used to read as a non-GPU
        # selection and disarm it.
        (None, None, False),
        ("", None, False),
        (",", None, False),
        # The legacy singular variable is honoured when JAX_PLATFORMS is unset
        # AND when it is set-but-empty -- `os.environ.get(A, get(B))` does not
        # fall back on an empty string, which made a CPU-only run look like a
        # GPU run and could fire the refusal on it.
        (None, "cpu", True),
        ("", "cpu", True),
        ("", "cuda", False),
    ],
)
def test_non_gpu_platform_selected_classifies_every_real_spelling(
    monkeypatch, jax_platforms, jax_platform_name, expect_non_gpu
):
    for var, val in (("JAX_PLATFORMS", jax_platforms),
                     ("JAX_PLATFORM_NAME", jax_platform_name)):
        if val is None:
            monkeypatch.delenv(var, raising=False)
        else:
            monkeypatch.setenv(var, val)
    assert ei._non_gpu_platform_selected() is expect_non_gpu


def test_pin_local_gpu_skips_on_explicit_cpu_platform(monkeypatch):
    """JAX_PLATFORMS=cpu on a GPU node with more ranks than GPUs: no pin,
    no refusal -- the run never creates a CUDA backend (measured: the six-
    process CPU duo parity died here on g[097,271], job 9902131)."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    assert ei.pin_local_gpu(local_rank=2, n_local=3) is None
    # a pin that FITS still narrows on the cpu platform (existing contract)
    assert ei.pin_local_gpu(local_rank=1, n_local=2) == "1"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    # and the refusal is still live when a GPU platform is selected
    monkeypatch.setenv("JAX_PLATFORMS", "cuda")
    with pytest.raises(RuntimeError, match="more local ranks than GPUs"):
        ei.pin_local_gpu(local_rank=2, n_local=3)
