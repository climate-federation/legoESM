"""JAX-free BSP jitter probe: fixed spin-work + MPI barrier, per-rank.

Measures OS-noise/tail-latency amplification in a bulk-synchronous loop
with NO model code and NO JAX: each iteration burns a fixed spin-loop
(calibrated once on rank 0 to ~--segment-ms of wall time) and then hits
``MPI.COMM_WORLD.Barrier()``. The per-iteration wall time distribution
at rank counts 32 vs 128 gives the pure jitter-amplification curve —
the discriminator for the ocean-MPAS rank-count term (GLM-5.2 consult
2026-08-09: BSP tail-latency model reproduces the measured 30-41% scan
share; this probe tests that model without JAX/mpi4jax in the loop).

Interpretation (written before first receipt):
* If median per-iteration time at 128 ranks exceeds the 32-rank median
  by a factor comparable to the model's ~1.5x rank-count term (at
  matching segment length ~2-3 ms), OS jitter amplification is
  CONFIRMED as quantitatively sufficient.
* If the 128-rank overhead is small (<<10%), jitter is REFUTED as the
  main term and the remaining suspect is the mpi4jax host-callback
  pipeline itself (JAX-side, not OS-side).

Usage
-----
    srun --ntasks=128 python scripts/bench/bench_bsp_jitter.py \
        --segment-ms 2.5 --iters 400 --out probe.jsonl
"""
from __future__ import annotations

import argparse
import json
import statistics
import time


def calibrate_spin(target_ms: float) -> int:
    """Spin-loop count whose wall time is ~target_ms on THIS core."""
    n = 100_000
    while True:
        t0 = time.perf_counter_ns()
        _spin(n)
        ms = (time.perf_counter_ns() - t0) / 1e6
        if ms >= target_ms:
            return int(n * target_ms / ms)
        n *= 2


def _spin(n: int) -> float:
    x = 0.0
    for i in range(n):
        x += i * 1e-9
    return x


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--segment-ms", type=float, default=2.5,
                   help="Target compute-segment length between barriers "
                        "(~the ocean step's 2-3 ms inter-sync gap).")
    p.add_argument("--iters", type=int, default=400)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()

    # Calibrate on rank 0, broadcast, so all ranks burn the SAME count
    # (per-rank calibration would hide frequency skew, which is part of
    # the phenomenon being measured only if it differs by node — the
    # fixed count exposes it).
    n_spin = calibrate_spin(args.segment_ms) if rank == 0 else None
    n_spin = comm.bcast(n_spin, root=0)

    for _ in range(args.warmup):
        _spin(n_spin)
        comm.Barrier()

    times_ms = []
    for _ in range(args.iters):
        t0 = time.perf_counter_ns()
        _spin(n_spin)
        comm.Barrier()
        times_ms.append((time.perf_counter_ns() - t0) / 1e6)

    # Solo segment time (no barrier), measured after, same count
    solo = []
    for _ in range(50):
        t0 = time.perf_counter_ns()
        _spin(n_spin)
        solo.append((time.perf_counter_ns() - t0) / 1e6)

    local = {
        "rank": rank,
        "solo_median_ms": statistics.median(solo),
        "bsp_median_ms": statistics.median(times_ms),
        "bsp_p95_ms": sorted(times_ms)[int(0.95 * len(times_ms))],
        "bsp_max_ms": max(times_ms),
    }
    rows = comm.gather(local, root=0)
    if rank == 0:
        solo_med = statistics.median([r["solo_median_ms"] for r in rows])
        bsp_med = statistics.median([r["bsp_median_ms"] for r in rows])
        rec = {
            "component": "bsp_jitter_probe",
            "n_ranks": size,
            "segment_ms_target": args.segment_ms,
            "n_spin": n_spin,
            "iters": args.iters,
            "solo_median_ms": round(solo_med, 4),
            "bsp_median_ms": round(bsp_med, 4),
            "amplification": round(bsp_med / solo_med, 4),
            "bsp_p95_ms": round(statistics.median(
                [r["bsp_p95_ms"] for r in rows]), 4),
            "worst_rank_max_ms": round(max(r["bsp_max_ms"] for r in rows), 4),
        }
        print(json.dumps(rec))
        if args.out:
            import os
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            with open(args.out, "a") as f:
                f.write(json.dumps(rec) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
