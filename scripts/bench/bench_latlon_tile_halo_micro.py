"""Does packing the tile halo actually buy TIME, or only a smaller count?

The tiled lat-lon lane loses to latitude strips by 27% at 64 GPUs while
moving 5.3x fewer bytes, and a census of the compiled step says 71 of its
108 messages carry 5% of the data — pure per-message cost at about 26
microseconds each. A packed exchange body exists and cuts six fields from
60 collectives to 10.

That is a COUNT. The mechanism review's objection is that a count is not a
time: if a packed message still costs 26 microseconds, packing saves
nothing and the cost is somewhere else entirely. This microbenchmark times
both bodies on the same devices, same fields, same tiling, so the answer
is a measurement rather than an inference.

It deliberately times ONLY the halo exchange, not a model step: the whole
question is the per-message cost, and burying it in a step would hide it.

Usage
-----
    python scripts/bench/bench_latlon_tile_halo_micro.py \
        --p-lat 2 --p-lon 2 --n-lat 512 --n-lon 1024 --nlev 26 --fields 6
"""
from __future__ import annotations

import argparse
import re
import statistics
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, PartitionSpec as P

from legoesm.parallel.latlon_spmd import (
    make_latlon_2d_packed_pad_body,
    make_latlon_2d_pad_body,
)
from legoesm.parallel.shard_map_compat import shard_map


def _count(text):
    return len(re.findall(r"collective-permute(?:-start)?\(", text))


def _time(fn, args, reps, warmup):
    for _ in range(warmup):
        jax.block_until_ready(fn(*args))
    out = []
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        out.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(out), min(out), max(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--p-lat", type=int, default=2)
    ap.add_argument("--p-lon", type=int, default=2)
    ap.add_argument("--n-lat", type=int, default=512)
    ap.add_argument("--n-lon", type=int, default=1024)
    ap.add_argument("--nlev", type=int, default=26)
    ap.add_argument("--fields", type=int, default=6)
    ap.add_argument("--halo", type=int, default=1)
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    args = ap.parse_args()

    need = args.p_lat * args.p_lon
    if len(jax.devices()) < need:
        raise SystemExit(f"need {need} devices, have {len(jax.devices())}")
    mesh = Mesh(np.array(jax.devices()[:need]).reshape(args.p_lat, args.p_lon),
                axis_names=("lat", "lon"))
    rng = np.random.default_rng(7)
    fields = [jnp.asarray(
        rng.standard_normal((args.n_lat, args.n_lon, args.nlev)),
        dtype=jnp.float32) for _ in range(args.fields)]
    specs = tuple(("fold", args.halo, i % 2 == 1)
                  for i in range(args.fields))
    isp = tuple(P("lat", "lon", None) for _ in fields)

    def packed(*fs):
        body = make_latlon_2d_packed_pad_body(mesh, specs)
        return partial(shard_map, mesh=mesh, in_specs=isp, out_specs=isp,
                       check_vma=False)(lambda *x: body(*x))(*fs)

    def per_field(*fs):
        outs = []
        for sp, f in zip(specs, fs):
            b = make_latlon_2d_pad_body(mesh, halo=sp[1], negate=sp[2])
            outs.append(partial(
                shard_map, mesh=mesh, in_specs=P("lat", "lon", None),
                out_specs=P("lat", "lon", None), check_vma=False)(
                    lambda x, _b=b: _b(x))(f))
        return tuple(outs)

    print(f"mesh {args.p_lat}x{args.p_lon} on {jax.default_backend()}, "
          f"{args.fields} fields of {args.n_lat}x{args.n_lon}x{args.nlev}")
    rows = {}
    for name, fn in (("per-field", per_field), ("packed", packed)):
        jitted = jax.jit(fn)
        n = _count(jitted.lower(*fields).compile().as_text())
        med, lo, hi = _time(jitted, fields, args.reps, args.warmup)
        rows[name] = (n, med)
        print(f"  {name:>9}: {n:3d} collectives   {med:8.3f} ms "
              f"(min {lo:.3f}, max {hi:.3f})   "
              f"{1000 * med / max(n, 1):7.1f} us per collective")

    npf, tpf = rows["per-field"]
    npk, tpk = rows["packed"]
    print(f"\ncount {npf} -> {npk} ({npf / npk:.1f}x fewer), "
          f"time {tpf:.3f} -> {tpk:.3f} ms ({tpf / tpk:.2f}x faster)")
    if tpf / tpk >= 0.8 * (npf / npk):
        print("VERDICT: time follows the count — the cost IS per-message, "
              "so packing the remaining exchanges is the right build")
    elif tpf / tpk <= 1.2:
        print("VERDICT: packing buys almost no time — the cost is NOT "
              "per-message dispatch, and packing the rest would be wasted "
              "work. Find where the time actually goes first.")
    else:
        print("VERDICT: partial — packing helps but by much less than the "
              "count suggests; reprice the remaining build before doing it")


if __name__ == "__main__":
    main()
