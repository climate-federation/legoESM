"""SELF-SPAWNING multi-controller gate for the icosahedral/MPAS SPMD bench.

The Voronoi sibling of
tests/parallel/test_latlon_ocean_spmd_multicontroller_selfspawn.py: spawns
two worker processes and drives the FULL production bench
(``bench_mpas_spmd_scaling.py --multicontroller``) end-to-end with the
parity + conservation gates armed — pinning the multi-controller pieces a
Derecho/Levante multi-node NCCL icosahedral run exercises:
``shard_pytree`` onto a ("device",) mesh spanning non-addressable devices,
the cell-partition reorder computed independently per process (+ the
cross-process partition checksum), cross-process ppermute halo rounds and
the mass-fix psum inside the jitted step, the
``gather_voronoi_state_spmd`` replication gather, rank-0-gated output, and
the ``sync_global_devices`` barriers.

The workers use the bench's ``--coordinator`` path with the OMPI env vars
the bench reads for rank/size; port races retry via multihost_harness.
Single-process numerics of the sharded step itself:
tests/parallel/test_voronoi_sharded_equivalence.py.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

_BENCH = (
    Path(__file__).resolve().parents[2]
    / "scripts" / "bench" / "bench_mpas_spmd_scaling.py"
)
N_PROC = 2


@pytest.mark.timeout(600)
def test_mpas_spmd_two_process_selfspawn_bench_parity(tmp_path):
    from multihost_harness import run_federated

    out = tmp_path / "mpas_spmd.jsonl"
    base_env = dict(os.environ)
    base_env["JAX_PLATFORMS"] = "cpu"
    base_env["JAX_ENABLE_X64"] = "1"
    base_env.pop("XLA_FLAGS", None)  # 1 real CPU device per process

    def build_cmd(rank, port):
        cmd = [
            sys.executable, str(_BENCH),
            "--multicontroller", "--coordinator", f"localhost:{port}",
            "--n-devices", str(N_PROC),
            "--subdivision", "3", "--nlev", "4",
            "--steps", "4", "--warmup", "1",
            "--partition-method", "sfc",  # deterministic, no pymetis dep
            "--parity-gate", "--check-conservation",
            "--out", str(out),
        ]
        # The bench reads rank/size from OMPI env when --coordinator is
        # given; run_federated only varies argv, so vary env via `env`.
        return [
            "env",
            f"OMPI_COMM_WORLD_SIZE={N_PROC}",
            f"OMPI_COMM_WORLD_RANK={rank}",
        ] + cmd

    rcs, outs = run_federated(build_cmd, N_PROC, base_env, timeout_s=540)
    for rank, (rc, txt) in enumerate(zip(rcs, outs)):
        assert rc == 0, (
            f"rank {rank} exited {rc}\n--- output ---\n{txt[-4000:]}"
        )

    # Rank 0 owns the JSONL; the record carries the multicontroller fields.
    assert out.exists(), outs[0][-2000:]
    rec = json.loads(out.read_text().strip().splitlines()[-1])
    assert rec["component"] == "mpas_atm"
    assert rec["n_devices"] == N_PROC
    assert rec["n_processes"] == N_PROC
    assert rec["multicontroller"] is True
    # Parity gate ran and passed (rank-0-gated banner).
    assert "parity" in outs[0] and "MISMATCH" not in outs[0]


@pytest.mark.timeout(900)
def test_mpas_spmd_two_process_selfspawn_moist_kessler(tmp_path):
    """Two-process FULL-PRODUCTION-state gate for the native step
    (scaling-M3c increment 1): the moist baroclinic wave + Kessler
    microphysics drive the packed tracer halo exchange, the RK tracer
    advection and the tracer parity gate across a real process
    boundary.  ``--halo-strategy ppermute`` is FORCED (auto would pick
    allgather at this per-device cell count — codex M3c-1 MAJOR), so
    the cross-process ppermute rounds and their P("device")-sharded
    schedule arguments are exactly what runs here.  The dry twin above
    keeps the auto/allgather branch covered."""
    from multihost_harness import run_federated

    out = tmp_path / "mpas_spmd_moist.jsonl"
    base_env = dict(os.environ)
    base_env["JAX_PLATFORMS"] = "cpu"
    base_env["JAX_ENABLE_X64"] = "1"
    base_env.pop("XLA_FLAGS", None)  # 1 real CPU device per process

    def build_cmd(rank, port):
        cmd = [
            sys.executable, str(_BENCH),
            "--multicontroller", "--coordinator", f"localhost:{port}",
            "--n-devices", str(N_PROC),
            "--subdivision", "3", "--nlev", "4",
            "--steps", "4", "--warmup", "1",
            "--partition-method", "sfc",  # deterministic, no pymetis dep
            "--physics", "kessler",
            "--halo-strategy", "ppermute",
            "--parity-gate", "--check-conservation",
            "--out", str(out),
        ]
        return [
            "env",
            f"OMPI_COMM_WORLD_SIZE={N_PROC}",
            f"OMPI_COMM_WORLD_RANK={rank}",
        ] + cmd

    rcs, outs = run_federated(build_cmd, N_PROC, base_env, timeout_s=840)
    for rank, (rc, txt) in enumerate(zip(rcs, outs)):
        assert rc == 0, (
            f"rank {rank} exited {rc}\n--- output ---\n{txt[-4000:]}"
        )

    assert out.exists(), outs[0][-2000:]
    rec = json.loads(out.read_text().strip().splitlines()[-1])
    assert rec["component"] == "mpas_atm"
    assert rec["n_processes"] == N_PROC
    assert rec["physics"] == "kessler"
    assert rec["halo_strategy"] == "ppermute"
    # Tracer parity lines printed by the extended gate (q_v/q_c/q_r) and
    # no field mismatched.
    assert "q_v" in outs[0], "tracer parity lines missing from the gate"
    assert "parity" in outs[0] and "MISMATCH" not in outs[0]
