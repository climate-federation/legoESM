#!/usr/bin/env python
"""Ocean lat-lon barotropic PCG — single-controller SPMD strong scaling (1 vs N device).

Audit lever #1 (ocean lat-lon 2-GPU SPMD), the FIRST plottable ocean-GPU
scaling number.  Times the barotropic implicit-Helmholtz fixed-iteration
PCG (``ocean.dynamics.barotropic_common._fixed_iteration_pcg``) — the
ocean weak/strong-scaling WALL (audit: M~60 iters x 2 reductions/iter,
allreduce/psum-latency-bound) — run under the lat-band SPMD backend
(``shard_map`` over a 1-D ``"lat"`` mesh; halos via local lon-wrap +
``lax.ppermute`` + pole fold; CG dots via ``jax.lax.psum``).  Pure jax, NO
mpi4jax — runs on route-B (cuda-jax, the RTX8000 PCIe pair) where mpi4jax
is unavailable.

STRONG scaling: a FIXED global grid solved on 1 device (whole grid) then N
devices (each a latitude band).  ``efficiency(N) = t(1) / (N * t(N))``;
ideal 1.0.  On a PCIe pair (no NVLink) the psum collective-permute is the
expected ceiling — this bench MEASURES that ceiling.

The operator is the uniform-coefficient 5-point Helmholtz ``A = I +
c(-Delta)`` (built on the backend-oblivious ``pad_halo_latlon``, so the
SAME closure runs serial and sharded).  Its COMMUNICATION pattern (band
halo + 2 psum/iter) is IDENTICAL to the production variable-metric ocean
Helmholtz; only the per-cell LOCAL arithmetic differs — so the measured
strong-scaling EFFICIENCY (comm-vs-compute ratio) transfers, while the
absolute ms is a lower bound (the real solver has more local compute,
which HIDES the same comm => equal-or-better efficiency).  The full
variable-metric sharded ocean step (sharded LatLonGrid metrics) is the
next increment; this isolates and measures the scaling-critical kernel.

Usage (compute node only; never the login node):
  GPU (route-B, the real target):
    JAX_ENABLE_X64=1 python scripts/bench/bench_ocean_latlon_spmd_pcg.py \\
        --device gpu --device-counts 1,2 --n-lat 360 --n-lon 720
  CPU smoke (virtual devices):
    XLA_FLAGS=--xla_force_host_platform_device_count=4 JAX_ENABLE_X64=1 \\
      python scripts/bench/bench_ocean_latlon_spmd_pcg.py \\
        --device cpu --device-counts 1,2,4 --n-lat 64 --n-lon 128
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time


def _configure_jax(device: str, precision: str, n_cpu_devices: int) -> None:
    """Set platform/precision BEFORE importing jax (single process, ALL GPUs).

    Route-B is single-controller: ONE process drives every GPU on the node
    (no per-rank CUDA_VISIBLE_DEVICES pinning — that is the mpi4jax path).
    """
    if precision == "float64":
        os.environ["JAX_ENABLE_X64"] = "1"
    if device == "gpu":
        os.environ["JAX_PLATFORMS"] = "cuda"
        # Two devices share the node; do not let the allocator grab it all.
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    else:
        os.environ["JAX_PLATFORMS"] = "cpu"
        os.environ.setdefault(
            "XLA_FLAGS",
            f"--xla_force_host_platform_device_count={n_cpu_devices}",
        )


def _build_ops(coeff: float):
    """Backend-oblivious ``(A_op, M_inv)`` for ``A = I + c(-Delta)``."""
    import jax.numpy as jnp
    from legoesm.grids.halo_latlon import pad_halo_latlon

    diag = 1.0 + 4.0 * coeff

    def A_op(x):
        p = pad_halo_latlon(x, halo=1)
        interior = p[1:-1, 1:-1]
        neigh = (p[2:, 1:-1] + p[:-2, 1:-1] + p[1:-1, 2:] + p[1:-1, :-2])
        return interior + coeff * (4.0 * interior - neigh)

    def M_inv(x):
        return x / diag

    return A_op, M_inv


def _time_fn(fn, *args, reps: int, warmup: int) -> float:
    """Median wall-time (ms) of ``fn(*args)``, block_until_ready'd."""
    import jax

    for _ in range(warmup):
        jax.block_until_ready(fn(*args))
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn(*args)
        jax.block_until_ready(out)
        samples.append((time.perf_counter() - t0) * 1.0e3)
    return statistics.median(samples)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    ap.add_argument("--device-counts", default="1,2",
                    help="comma list of mesh sizes to bench, e.g. 1,2")
    ap.add_argument("--n-lat", type=int, default=360)
    ap.add_argument("--n-lon", type=int, default=720)
    ap.add_argument("--max-iter", type=int, default=60,
                    help="fixed PCG iterations (production implicit_cn M)")
    ap.add_argument("--variant", choices=("standard", "single_reduce"),
                    default="standard",
                    help="standard = 2 psum/iter; single_reduce = "
                         "Chronopoulos-Gear, 1 psum/iter (lever #1 on PCIe)")
    ap.add_argument("--coeff", type=float, default=1.0e7,
                    help="Helmholtz coeff ~ theta^2 dt^2 g H")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--precision", choices=("float64", "float32"),
                    default="float64")
    ap.add_argument("--csv", default=None,
                    help="append a scaling_indicators row (ocean SPMD eff)")
    ap.add_argument("--tag", default="ocean_latlon_spmd_pcg")
    ap.add_argument("--commit", default="")
    ap.add_argument("--job", default="")
    ap.add_argument("--date", default="",
                    help="YYYY-MM-DD for the CSV row (host clock unavailable "
                         "inside jax timing); required with --csv")
    args = ap.parse_args()

    counts = [int(c) for c in args.device_counts.split(",") if c.strip()]
    _configure_jax(args.device, args.precision, max(counts))

    import numpy as np
    import jax
    import jax.numpy as jnp
    from functools import partial
    try:
        from jax import shard_map
    except ImportError:  # pragma: no cover
        from jax.experimental.shard_map import shard_map
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    from legoesm.ocean.dynamics.barotropic_common import (
        _fixed_iteration_pcg, _fixed_iteration_pcg_single_reduce,
    )
    from legoesm.parallel.latlon_spmd import (
        activate_latlon_spmd_halo, deactivate_latlon_spmd_halo,
    )

    devs = jax.devices()
    print(f"[bench] device={args.device} jax sees {len(devs)} device(s): "
          f"{devs[0].platform}", flush=True)
    print(f"[bench] grid={args.n_lat}x{args.n_lon} max_iter={args.max_iter} "
          f"precision={args.precision} reps={args.reps}", flush=True)

    A_op, M_inv = _build_ops(args.coeff)
    rng = np.random.default_rng(0)
    dt = jnp.float64 if args.precision == "float64" else jnp.float32
    b_host = jnp.asarray(rng.standard_normal((args.n_lat, args.n_lon)), dtype=dt)
    x0_host = jnp.zeros((args.n_lat, args.n_lon), dtype=dt)

    results = {}   # n_dev -> ms
    for n_dev in counts:
        if len(devs) < n_dev:
            print(f"[bench] SKIP n_dev={n_dev}: only {len(devs)} device(s)",
                  flush=True)
            continue
        if args.n_lat % n_dev != 0:
            print(f"[bench] SKIP n_dev={n_dev}: n_lat {args.n_lat} not "
                  f"divisible", flush=True)
            continue
        mesh = Mesh(np.array(devs[:n_dev]), axis_names=("lat",))
        isp = P("lat", None)
        b_sh = jax.device_put(b_host, NamedSharding(mesh, isp))
        x0_sh = jax.device_put(x0_host, NamedSharding(mesh, isp))

        activate_latlon_spmd_halo(mesh)
        try:
            if args.variant == "single_reduce":
                # Chronopoulos-Gear: 1 psum/iter. Uniform-area proxy => the
                # operator is self-adjoint in the EUCLIDEAN dot, so the
                # required area-weight W is unity (per band).
                @partial(shard_map, mesh=mesh, in_specs=(isp, isp),
                         out_specs=isp, check_vma=False)
                def _solve(bb, xx0):
                    W = jnp.ones_like(bb)
                    x, _rr = _fixed_iteration_pcg_single_reduce(
                        A_op, bb, M_inv, xx0, max_iter=args.max_iter,
                        dot_weight=W)
                    return x
            else:
                @partial(shard_map, mesh=mesh, in_specs=(isp, isp),
                         out_specs=isp, check_vma=False)
                def _solve(bb, xx0):
                    x, _rr = _fixed_iteration_pcg(
                        A_op, bb, M_inv, xx0, max_iter=args.max_iter)
                    return x
            jsolve = jax.jit(_solve)
            ms = _time_fn(jsolve, b_sh, x0_sh,
                          reps=args.reps, warmup=args.warmup)
        finally:
            deactivate_latlon_spmd_halo()
        results[n_dev] = ms
        print(f"[bench] n_dev={n_dev}: {ms:.3f} ms/solve", flush=True)

    # Strong-scaling efficiency relative to the smallest benched device count.
    if results:
        base_n = min(results)
        base_ms = results[base_n]
        print("\n[bench] strong scaling (vs n_dev=%d):" % base_n, flush=True)
        rows = []
        for n_dev in sorted(results):
            ms = results[n_dev]
            speedup = base_ms / ms
            eff = speedup / (n_dev / base_n)
            print(f"  n_dev={n_dev}: {ms:8.3f} ms  speedup={speedup:5.2f}x  "
                  f"efficiency={eff:5.3f}", flush=True)
            rows.append((n_dev, ms, speedup, eff))

        if args.csv:
            if not args.date:
                print("[bench] --csv requires --date YYYY-MM-DD", flush=True)
                return 2
            _append_csv(args, rows)
    else:
        print("[bench] no device counts ran", flush=True)
        return 1
    return 0


def _append_csv(args, rows) -> None:
    """Append strong-scaling efficiency rows to the scaling ledger.

    Columns (existing schema): date,commit,tag,grid,backend,mode,value,
    metric,job,note.  Notes are comma-free (the plotter splits on comma)."""
    import csv as _csv
    path = args.csv
    new = not os.path.exists(path)
    with open(path, "a", newline="") as fh:
        w = _csv.writer(fh)
        if new:
            w.writerow(["date", "commit", "tag", "grid", "backend", "mode",
                        "value", "metric", "job", "note"])
        for (n_dev, ms, speedup, eff) in rows:
            note = (f"latlon {args.n_lat}x{args.n_lon} M{args.max_iter} "
                    f"{args.precision} {args.variant} ndev{n_dev} {ms:.2f}ms")
            w.writerow([args.date, args.commit, args.tag, "latlon", "spmd",
                        "strong", f"{eff:.4f}",
                        f"ocean_spmd_pcg_{args.variant}_eff_ndev{n_dev}",
                        args.job, note])
    print(f"[bench] appended {len(rows)} row(s) -> {path}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
