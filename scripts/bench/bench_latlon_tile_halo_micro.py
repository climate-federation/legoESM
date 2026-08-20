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

WHAT THIS INSTRUMENT GOT WRONG ONCE, and how it is fixed. The first version
timed one call to the body per measurement. On four devices driven from one
process, a single call is dominated by HOST DISPATCH -- it reported about 680
microseconds "per collective" for a body whose collectives move 0.8 MB in
total, and its own spread was min 2.2 ms against max 5.0 ms while the effect
under test was 5%. It could not see the body at all, and the conclusion drawn
from it -- that packing buys no time -- was a statement about dispatch. The fix
is the one the ppermute microbenchmark already uses: run the body on several
INDEPENDENT field groups inside ONE jit call and take the difference between
one group and many, which cancels the constant per-call overhead. The groups
carry different data so the compiler cannot fold them into one.

Usage
-----
    python scripts/bench/bench_latlon_tile_halo_micro.py \
        --p-lat 2 --p-lon 2 --n-lat 512 --n-lon 1024 --nlev 26 --fields 6
"""
from __future__ import annotations

import argparse
import json
import os
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


def _median_ms(fn, args, reps, warmup):
    for _ in range(warmup):
        jax.block_until_ready(fn(*args))
    out = []
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        out.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(out), min(out), max(out)


def _time_one_body(make_fn, groups, reps, warmup):
    """Per-application cost of a body, with host dispatch subtracted.

    ``make_fn(n)`` returns a jitted function applying the body to the first
    ``n`` field groups. One call to it costs ``dispatch + n * body``, so the
    difference between ``n`` groups and one group divided by ``n - 1`` is the
    body alone. Without this the measurement is a dispatch measurement: on four
    devices from one process, dispatch is milliseconds and the body is not.

    Returns ``(per_group_ms, single_call_ms, spread_ms)`` so the contaminated
    number and the run-to-run spread stay visible next to the corrected one.
    """
    n = len(groups)
    if n < 2:
        raise ValueError("need >= 2 field groups to subtract dispatch")
    t1, lo1, hi1 = _median_ms(make_fn(1), groups[:1], reps, warmup)
    tn, lon, hin = _median_ms(make_fn(n), groups, reps, warmup)
    return (tn - t1) / (n - 1), t1, max(hi1 - lo1, hin - lon)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--p-lat", type=int, default=2)
    ap.add_argument("--p-lon", type=int, default=2)
    ap.add_argument("--n-lat", type=int, default=512)
    ap.add_argument("--n-lon", type=int, default=1024)
    ap.add_argument("--nlev", type=int, default=26)
    ap.add_argument("--fields", type=int, default=6)
    ap.add_argument("--halo", type=int, default=1)
    ap.add_argument("--groups", type=int, default=8,
                    help="Independent field groups inside ONE jit call. The "
                         "one-group versus many-groups difference is what "
                         "cancels host dispatch, which otherwise IS the "
                         "measurement on four devices from one process.")
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--out", default=None, help="Append one JSON line here.")
    args = ap.parse_args()

    if args.groups < 2:
        raise SystemExit("--groups must be >= 2: with one group there is "
                         "nothing to subtract and the number is dispatch")
    need = args.p_lat * args.p_lon
    if len(jax.devices()) < need:
        raise SystemExit(f"need {need} devices, have {len(jax.devices())}")
    mesh = Mesh(np.array(jax.devices()[:need]).reshape(args.p_lat, args.p_lon),
                axis_names=("lat", "lon"))
    rng = np.random.default_rng(7)
    # Independent data per group: identical inputs would let the compiler fold
    # the groups into one and the subtraction would measure nothing.
    groups = [tuple(jnp.asarray(
        rng.standard_normal((args.n_lat, args.n_lon, args.nlev)),
        dtype=jnp.float32) for _ in range(args.fields))
        for _ in range(args.groups)]
    specs = tuple(("fold", args.halo, i % 2 == 1)
                  for i in range(args.fields))
    isp = tuple(P("lat", "lon", None) for _ in range(args.fields))

    def packed_once(fs):
        body = make_latlon_2d_packed_pad_body(mesh, specs)
        return partial(shard_map, mesh=mesh, in_specs=isp, out_specs=isp,
                       check_vma=False)(lambda *x: body(*x))(*fs)

    def per_field_once(fs):
        outs = []
        for sp, f in zip(specs, fs):
            b = make_latlon_2d_pad_body(mesh, halo=sp[1], negate=sp[2])
            outs.append(partial(
                shard_map, mesh=mesh, in_specs=P("lat", "lon", None),
                out_specs=P("lat", "lon", None), check_vma=False)(
                    lambda x, _b=b: _b(x))(f))
        return tuple(outs)

    def make(one):
        def maker(n):
            return jax.jit(lambda *gs: tuple(one(g) for g in gs[:n]))
        return maker

    print(f"mesh {args.p_lat}x{args.p_lon} on {jax.default_backend()}, "
          f"{args.fields} fields of {args.n_lat}x{args.n_lon}x{args.nlev}, "
          f"{args.groups} groups")
    rows = {}
    for name, one in (("per-field", per_field_once), ("packed", packed_once)):
        maker = make(one)
        n_coll = _count(maker(1).lower(*groups[:1]).compile().as_text())
        per, single, spread = _time_one_body(maker, groups, args.reps,
                                             args.warmup)
        rows[name] = {"collectives": n_coll, "per_group_ms": round(per, 4),
                      "single_call_ms": round(single, 4),
                      "spread_ms": round(spread, 4)}
        print(f"  {name:>9}: {n_coll:3d} collectives   {per:8.4f} ms/group "
              f"(single call {single:7.3f} ms incl. dispatch, "
              f"spread {spread:6.3f} ms)")

    rec = {
        "component": "latlon_tile_halo_micro",
        "p_lat": args.p_lat, "p_lon": args.p_lon,
        "n_lat": args.n_lat, "n_lon": args.n_lon, "nlev": args.nlev,
        "fields": args.fields, "halo": args.halo, "groups": args.groups,
        "reps": args.reps, "warmup": args.warmup,
        "backend": jax.default_backend(),
        "dispatch_subtracted": True,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "git_sha": os.environ.get("LEGOESM_GIT_SHA"),
        "arms": rows,
    }
    # No verdict is printed here on purpose. The interpretation belongs in the
    # analysis, after the controls have been checked; a probe that prints its
    # own conclusion gets that conclusion quoted back as evidence, which is
    # exactly how the first version of this benchmark misled.
    print(json.dumps(rec))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "a") as f:
            f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
