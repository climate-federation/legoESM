"""Settle the wet-cell-compaction sign: slice-stencil vs gather-stencil.

Wet-cell compaction packs the ~71%-wet cells of a real-bathymetry lat-lon
grid into a dense 1-D array, trading a factor ~1/wet_fraction of compute
for INDIRECT ADDRESSING on every stencil access. Whether that trade wins
on an A100 is a one-number question: how much slower is the same 5-point
Laplacian when neighbours come from precomputed gather indices instead of
array slices?

Times, at bench-relevant tile sizes, per application:
  * slice   — jnp roll/slice Laplacian on the dense (n_lat, n_lon) field
              (what the model does today), on ALL cells;
  * gather  — identical arithmetic on a packed 1-D wet-cell array with
              4 neighbour index vectors (what compaction would do), on
              wet cells only, at the SAME wet fraction.

Compaction wins iff  t_gather < t_slice  (the wet-fraction saving is
already inside t_gather via the shorter array). Prints the ratio and the
break-even wet fraction. Correctness self-check: both paths must produce
identical interior values on a shared random field.
"""
from __future__ import annotations

import argparse
import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np


def _laplacian_slice(f):
    # 5-point Laplacian via shifts; interior-correct, edges wrap (edge
    # handling is identical between arms and cancels in the comparison).
    return (jnp.roll(f, 1, 0) + jnp.roll(f, -1, 0)
            + jnp.roll(f, 1, 1) + jnp.roll(f, -1, 1) - 4.0 * f)


def _build_gather(mask):
    """Packed representation: values[wet], 4 neighbour index vectors.

    Dry neighbours map to self (Neumann-ish; same arithmetic count as the
    slice arm, which is what is being priced — not the BC physics).
    """
    n_lat, n_lon = mask.shape
    idx_of = -np.ones(mask.shape, dtype=np.int32)
    wet = np.argwhere(mask)
    idx_of[wet[:, 0], wet[:, 1]] = np.arange(len(wet), dtype=np.int32)
    nbrs = []
    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        ni = (wet[:, 0] + di) % n_lat
        nj = (wet[:, 1] + dj) % n_lon
        k = idx_of[ni, nj]
        nbrs.append(np.where(k >= 0, k, np.arange(len(wet), dtype=np.int32)))
    return wet, [jnp.asarray(v) for v in nbrs]


def _laplacian_gather(v, nbrs):
    return v[nbrs[0]] + v[nbrs[1]] + v[nbrs[2]] + v[nbrs[3]] - 4.0 * v


def _time_ms(fn, x, n_warmup, n_iters, n_reps):
    # n_reps chained applications inside one jit call: cancels dispatch,
    # same trick as bench_ppermute_microbench.
    @jax.jit
    def run(f):
        def body(_, g):
            return fn(g)
        return jax.lax.fori_loop(0, n_reps, body, f)

    for _ in range(n_warmup):
        out = run(x)
    jax.block_until_ready(out)
    ts = []
    for _ in range(n_iters):
        t0 = time.perf_counter_ns()
        out = run(x)
        jax.block_until_ready(out)
        ts.append((time.perf_counter_ns() - t0) / 1e6 / n_reps)
    return statistics.median(ts)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-lat", type=int, default=576)
    p.add_argument("--n-lon", type=int, default=1152)
    p.add_argument("--wet-fraction", type=float, default=0.71,
                   help="Target wet fraction (real bathymetry ~0.71).")
    p.add_argument("--dtype", choices=["float32", "float64"],
                   default="float32")
    p.add_argument("--n-warmup", type=int, default=3)
    p.add_argument("--n-iters", type=int, default=30)
    p.add_argument("--n-reps", type=int, default=64)
    args = p.parse_args()

    dtype = jnp.float64 if args.dtype == "float64" else jnp.float32
    rng = np.random.default_rng(0)
    # Banded land (continent-like): whole longitude blocks dry, closer to
    # real cache behaviour than salt-and-pepper.
    mask = np.ones((args.n_lat, args.n_lon), dtype=bool)
    n_dry_cols = int(round((1.0 - args.wet_fraction) * args.n_lon))
    mask[:, :n_dry_cols] = False
    f_np = rng.standard_normal(mask.shape)
    f = jnp.asarray(f_np, dtype)

    wet, nbrs = _build_gather(mask)
    v = jnp.asarray(f_np[wet[:, 0], wet[:, 1]], dtype)

    # correctness: identical interior values (wet cells with 4 wet nbrs)
    lap_d = np.asarray(_laplacian_slice(f))
    lap_g = np.asarray(_laplacian_gather(v, nbrs))
    interior = np.ones(len(wet), dtype=bool)
    idx_of = -np.ones(mask.shape, dtype=np.int64)
    idx_of[wet[:, 0], wet[:, 1]] = np.arange(len(wet))
    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        interior &= idx_of[(wet[:, 0] + di) % args.n_lat,
                           (wet[:, 1] + dj) % args.n_lon] >= 0
    np.testing.assert_allclose(
        lap_g[interior], lap_d[wet[interior, 0], wet[interior, 1]],
        rtol=1e-6 if args.dtype == "float32" else 1e-12)

    t_slice = _time_ms(_laplacian_slice, f, args.n_warmup, args.n_iters,
                       args.n_reps)
    t_gather = _time_ms(lambda x: _laplacian_gather(x, nbrs), v,
                        args.n_warmup, args.n_iters, args.n_reps)
    # gather cost per wet cell vs slice cost per (all) cell
    per_slice = t_slice / mask.size
    per_gather = t_gather / int(mask.sum())
    breakeven = per_gather / per_slice  # wet fraction below which gather wins
    print(f"grid {args.n_lat}x{args.n_lon} {args.dtype} "
          f"wet={args.wet_fraction:.2f} backend={jax.default_backend()}")
    print(f"slice : {t_slice:8.4f} ms/apply  (all {mask.size} cells)")
    print(f"gather: {t_gather:8.4f} ms/apply  ({int(mask.sum())} wet cells)")
    print(f"gather/slice time ratio: {t_gather / t_slice:.3f} "
          f"(<1 = compaction wins at this wet fraction)")
    print(f"per-cell gather penalty: {breakeven:.3f}x -> compaction wins "
          f"only when wet_fraction < {1.0 / breakeven:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
