"""MPI transport probes (``probe_mpi_transport``, ``roofline_probe``
alltoall sweep) under an even rank count. Run::

    mpirun -np 2 python -m pytest tests/distributed/test_probe_mpi_transport_mpi.py
"""
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("mpi4py")
from mpi4py import MPI  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "bench"))
NPROC = MPI.COMM_WORLD.Get_size()
pytestmark = pytest.mark.skipif(NPROC < 2 or NPROC % 2,
                                reason="needs an even rank count >= 2")


def test_transport_probe_rows(tmp_path_factory):
    import probe_mpi_transport as pmt

    out = MPI.COMM_WORLD.bcast(
        str(tmp_path_factory.mktemp("t") / "t.jsonl"), root=0)
    rc = pmt.main(["--log2-bytes", "3,12", "--n-warmup", "2",
                   "--n-timing", "3", "--chain", "1,2,3",
                   "--chain-bytes", "8,4096",
                   "--max-rank-bytes", str(NPROC * 8), "--output", out])
    assert rc == 0
    MPI.COMM_WORLD.Barrier()
    if MPI.COMM_WORLD.Get_rank():
        return
    rec = json.loads(Path(out).read_text().splitlines()[-1])
    rows = rec["rows"]
    assert rec["n_ranks"] == NPROC and len(rec["placement"]) == NPROC
    assert rec["all_ok"] is True
    timed = [r for r in rows if "us" in r]
    assert all(r["us"] > 0 for r in timed)
    got = sorted((r["op"], r["transport"], r.get("bytes", r.get(
        "bytes_per_peer")), r.get("k")) for r in timed
        if r["op"] in ("sendrecv", "alltoall") or r["op"].endswith("_chain"))
    want = sorted(
        [("sendrecv", t, b, None) for b in (8, 4096)
         for t in ("mpi4py", "jit_raw", "jit_vjp", "eager_raw")]
        + [("alltoall", t, 8, None) for t in ("mpi4py", "jit_raw")]
        + [(op + "_chain", "jit_raw", b, k) for op in ("sendrecv", "allreduce")
           for b in (8, 4096) for k in (1, 2, 3)]
        + [("alltoall_chain", "jit_raw", 8, k) for k in (1, 2, 3)])
    assert got == want
    # 4096 B per peer exceeds the cap (sweep row recorded as skipped)
    assert [r["bytes_per_peer"] for r in rows if "skipped" in r] == [4096]
    assert all(r["ok"] is True for r in rows if "ok" in r)


def test_roofline_alltoall_sweep_caps_buffer():
    import roofline_probe as rp

    res = rp.measure_alltoall_collective(
        log2_lo=0, log2_hi=10, n_warmup=1, n_timing=2,
        max_rank_bytes=NPROC * 4 * 8)  # float32: n <= 8 elements kept
    assert res.transport == "mpi4jax_alltoall" and res.n_participants == NPROC
    assert [p.n_elements for p in res.points] == [1, 2, 4, 8]
    assert all(p.time_per_call_ms > 0 for p in res.points)
