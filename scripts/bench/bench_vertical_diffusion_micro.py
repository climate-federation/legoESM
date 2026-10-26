"""Vertical-diffusion microbenchmark: the ocean's backward-Euler solve in isolation.

Times ``implicit_vertical_diffusion_ocean`` (bands built inside the level sweep)
against the prebuilt-band reference ``thomas_solve(*_build_implicit_tridiag(...))``
on the SAME inputs in the SAME process, forward and gradient. The ratio between
the two is machine-independent enough to gate on; absolute milliseconds are
reported for the record only.

Default size = one rank's edge count at subdivision 7 on 16 ranks (39135 columns
x 40 levels, float64), the shape of the MPAS ocean's velocity solve.

Exit codes: 0 ok; 1 a ratio exceeds ``--max-ratio``; 2 production and reference
disagree beyond rounding, or any timing/agreement number is not finite (the
timing would compare different answers); 3 production no longer runs the
in-sweep solve (a silent fallback to the reference would otherwise pass a
ratio gate near 1).

Run:
  JAX_ENABLE_X64=1 python scripts/bench/bench_vertical_diffusion_micro.py \
      --out results/vdiff_micro.json --max-ratio 1.0
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from metadata import annotate_incomplete, scaling_metadata  # noqa: E402

_REL_TOL = 1e-10


def gate(row: dict, max_ratio: float | None) -> int:
    nums = [row[k] for k in ("value_rel_diff", "grad_rel_diff", "fwd_ratio", "grad_ratio")]
    if not all(math.isfinite(v) for v in nums):
        return 2
    if row["value_rel_diff"] > _REL_TOL or row["grad_rel_diff"] > _REL_TOL:
        return 2
    if not row["in_sweep_path"]:
        return 3
    if max_ratio is not None and max(row["fwd_ratio"], row["grad_ratio"]) > max_ratio:
        return 1
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cols", type=int, default=39135)
    p.add_argument("--nlev", type=int, default=40)
    p.add_argument("--dt", type=float, default=600.0)
    p.add_argument("--repeats", type=int, default=30)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--max-ratio", type=float, default=None,
                   help="fail (exit 1) when production/reference time exceeds "
                        "this, forward or gradient. No default: it must cover "
                        "the run-to-run spread on the machine in use.")
    p.add_argument("--out", type=str, default="results/vdiff_micro.json")
    args = p.parse_args(argv)
    if args.repeats < 1 or args.warmup < 1:
        p.error("--repeats and --warmup must be >= 1")
    if args.max_ratio is not None and not (math.isfinite(args.max_ratio) and args.max_ratio > 0):
        p.error("--max-ratio must be a finite positive number")

    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    from legoesm.ocean.physics.vertical_mixing.implicit_solver import (
        _build_implicit_tridiag,
        implicit_vertical_diffusion_ocean,
    )
    from legoesm.timestepping.tridiagonal import thomas_solve

    n, L, dt = args.cols, args.nlev, args.dt
    rng = np.random.default_rng(0)
    field = jnp.asarray(rng.standard_normal((n, L)))
    K = jnp.asarray(1e-4 + 1e-2 * rng.random((n, L - 1)))
    dz = jnp.asarray(10.0 + 90.0 * rng.random((n, L)))
    dzh = 0.5 * (dz[:, 1:] + dz[:, :-1])
    w = jnp.asarray(rng.standard_normal((n, L)))

    def prod(f, k):
        return implicit_vertical_diffusion_ocean(f, k, dz, dzh, dt)

    def ref(f, k):
        return thomas_solve(*_build_implicit_tridiag(f, k, dz, dzh, dt))

    def grad_of(fn):
        return jax.jit(jax.grad(lambda f, k: jnp.sum(w * fn(f, k)), argnums=(0, 1)))

    def timed_pair(g1, g2):
        """Median ms of g1 and g2; which runs first alternates every repeat
        so drift hits both."""
        for _ in range(args.warmup):
            jax.block_until_ready(g1(field, K))
            jax.block_until_ready(g2(field, K))
        t1, t2 = [], []
        for i in range(args.repeats):
            pair = ((g1, t1), (g2, t2))
            for g, ts in (pair if i % 2 == 0 else pair[::-1]):
                t0 = time.perf_counter()
                jax.block_until_ready(g(field, K))
                ts.append(time.perf_counter() - t0)
        return 1e3 * float(np.median(t1)), 1e3 * float(np.median(t2))

    def rel(a, b):
        den = max(float(jnp.max(jnp.abs(b))), float(np.finfo(np.float64).tiny))
        return float(jnp.max(jnp.abs(a - b))) / den

    fp, fr, gp, gr = jax.jit(prod), jax.jit(ref), grad_of(prod), grad_of(ref)
    xp, xr = fp(field, K), fr(field, K)
    (gpf, gpk), (grf, grk) = gp(field, K), gr(field, K)
    g_errs = [rel(gpf, grf), rel(gpk, grk)]
    pf, rf = timed_pair(fp, fr)
    pg, rg = timed_pair(gp, gr)
    row = {
        "cols": n, "nlev": L, "backend": jax.default_backend(),
        "cpus": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "in_sweep_path": "diffusion_thomas_solve" in str(jax.make_jaxpr(prod)(field, K)),
        "prod_fwd_ms": pf, "ref_fwd_ms": rf, "prod_grad_ms": pg, "ref_grad_ms": rg,
        "value_rel_diff": rel(xp, xr),
        "grad_rel_diff": (max(g_errs) if all(math.isfinite(e) for e in g_errs)
                          else float("nan")),
    }
    row["fwd_ratio"] = row["prod_fwd_ms"] / row["ref_fwd_ms"]
    row["grad_ratio"] = row["prod_grad_ms"] / row["ref_grad_ms"]
    row["max_ratio"] = args.max_ratio
    row["exit"] = gate(row, args.max_ratio)
    payload = {"rows": [row], "metadata": annotate_incomplete(scaling_metadata(
        grid="column", component="ocean", resolution=n, n_levels=L,
        precision="float64", decomposition="none",
        solver_variant="vertical_diffusion_in_sweep_vs_prebuilt",
        scaling_kind="throughput", transport="none",
        extra={"repeats": args.repeats, "warmup": args.warmup, "dt": dt}))}
    if os.path.dirname(args.out):
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(row))
    return row["exit"]


if __name__ == "__main__":
    raise SystemExit(main())
