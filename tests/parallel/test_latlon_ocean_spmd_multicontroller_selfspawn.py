"""SELF-SPAWNING multi-controller gate for the ocean lat-band SPMD bench.

Companion to tests/parallel/test_latlon_ocean_spmd_multicontroller.py (the
launcher-gated variant that needs an external `srun`/`mpirun` +
LEGOESM_JAX_DISTRIBUTED_TEST=1 and therefore never runs in plain pytest CI):
THIS variant spawns its own two worker processes and drives the FULL
production bench (``bench_ocean_latlon_spmd_scaling.py --multicontroller``)
end-to-end with the ported parity + conservation gates armed — pinning the
multi-controller pieces a Derecho/Levante multi-node NCCL ocean run
exercises: ``shard_state_latlon`` onto a mesh spanning non-addressable
devices, cross-process ppermute/psum inside the jitted step, the replication
gather, rank-0-gated output, and the ``sync_global_devices`` barriers.

The workers use the bench's ``--coordinator`` path with the OMPI env vars
the bench reads for rank/size; port races retry via multihost_harness.
Numerics of the sharded step itself:
tests/parallel/test_latlon_ocean_spmd_step.py.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

_BENCH = (
    Path(__file__).resolve().parents[2]
    / "scripts" / "bench" / "bench_ocean_latlon_spmd_scaling.py"
)
N_PROC = 2


@pytest.mark.timeout(600)
def test_ocean_spmd_two_process_selfspawn_bench_parity(tmp_path):
    from multihost_harness import run_federated

    out = tmp_path / "ocean_spmd.jsonl"
    base_env = dict(os.environ)
    base_env["JAX_PLATFORMS"] = "cpu"
    base_env["JAX_ENABLE_X64"] = "1"
    base_env.pop("XLA_FLAGS", None)  # 1 real CPU device per process

    def build_cmd(rank, port):
        return [
            sys.executable, str(_BENCH),
            "--multicontroller", "--coordinator", f"localhost:{port}",
            "--n-devices", str(N_PROC),
            "--n-lat", "16", "--n-lon", "32", "--nlev", "4",
            "--steps", "4", "--warmup", "1", "--dt", "600",
            "--parity-gate", "--check-conservation", "--cons-rtol", "1e-6",
            "--out", str(out),
        ]

    # The bench reads rank/size from OMPI env when --coordinator is given;
    # run_federated only varies argv, so vary the env via a per-rank wrapper.
    def build_cmd_with_env(rank, port):
        cmd = build_cmd(rank, port)
        # Encode env in the command via `env` so each worker gets its rank.
        return [
            "env",
            f"OMPI_COMM_WORLD_SIZE={N_PROC}",
            f"OMPI_COMM_WORLD_RANK={rank}",
        ] + cmd

    rcs, outs = run_federated(build_cmd_with_env, N_PROC, base_env,
                              timeout_s=540)
    for rank, (rc, txt) in enumerate(zip(rcs, outs)):
        assert rc == 0, (
            f"rank {rank} exited {rc}\n--- output ---\n{txt[-4000:]}"
        )

    # Rank 0 owns the JSONL; the record carries the multicontroller fields.
    assert out.exists(), outs[0][-2000:]
    rec = json.loads(out.read_text().strip().splitlines()[-1])
    assert rec["component"] == "ocean"
    assert rec["n_devices"] == N_PROC
    assert rec["n_processes"] == N_PROC
    assert rec["multicontroller"] is True
    # Parity gate ran and passed (rank-0-gated banner).
    assert "parity" in outs[0] and "MISMATCH" not in outs[0]
