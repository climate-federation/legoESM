"""Route-B hardening gates for ``legoesm.parallel.early_init`` (audit item 6).

Pure-env helpers tested directly (no cluster needed); the silent-fallback
guard tested against a stubbed ``jax`` module.  The REAL launch paths stay
covered by the multicontroller self-spawn suites.
"""
from __future__ import annotations

import sys
import types

import pytest

from legoesm.parallel import early_init as ei

_ALL_ENV = (
    "SLURM_NTASKS", "OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "PALS_LOCAL_RANKID",
    "CUDA_VISIBLE_DEVICES", "NCCL_NET_PLUGIN", "LD_LIBRARY_PATH",
    "LD_PRELOAD", "NCCL_SOCKET_IFNAME", "NCCL_DEBUG", "SLURM_NNODES",
    "SLURM_JOB_NUM_NODES", "PALS_NNODES",
    "LEGOESM_ALLOW_SINGLE_PROCESS_UNDER_MPI",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for v in _ALL_ENV:
        monkeypatch.delenv(v, raising=False)
    yield


def test_launcher_world_size_priority(monkeypatch):
    assert ei.launcher_world_size() == 0
    monkeypatch.setenv("PMI_SIZE", "8")
    assert ei.launcher_world_size() == 8
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "4")
    assert ei.launcher_world_size() == 4  # OMPI outranks PMI
    monkeypatch.setenv("SLURM_NTASKS", "16")
    assert ei.launcher_world_size() == 16  # SLURM outranks all
    monkeypatch.setenv("SLURM_NTASKS", "garbage")
    assert ei.launcher_world_size() == 4  # non-numeric skipped, not crash


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
    # rank (codex: '0,1,2,3' must not bind every rank to GPU 0), clamped.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    assert ei._pals_local_device_ids() == [3]
    # More local ranks than visible devices = LAUNCH error, loud — a clamp
    # would silently oversubscribe the last GPU (codex).
    monkeypatch.setenv("PALS_LOCAL_RANKID", "7")
    with pytest.raises(RuntimeError, match="more local ranks"):
        ei._pals_local_device_ids()


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


def test_nccl_report_records_knobs(monkeypatch):
    monkeypatch.setenv("NCCL_SOCKET_IFNAME", "hsn")
    monkeypatch.setenv("NCCL_DEBUG", "INFO")
    r = ei.nccl_transport_report()
    assert r["nccl_socket_ifname"] == "hsn"
    assert r["nccl_debug"] == "INFO"
