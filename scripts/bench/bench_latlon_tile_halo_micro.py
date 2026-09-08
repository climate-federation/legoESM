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
    """Marginal cost of ONE more application of a body.

    ``make_fn(n)`` returns a jitted function that applies the body to the first
    ``n`` of the field groups it is handed. EVERY arm is called with ALL the
    groups, so the number of buffers the host marshals is identical and only
    the device work differs; ``gs[:n]`` drops the rest at trace time. Calling
    the one-group arm with one group instead would leave a per-buffer host cost
    growing with ``n`` inside the slope -- both reviews of this benchmark
    flagged exactly that, and it is the same class of contamination the first
    version of this file died of.

    What this returns is a MARGINAL cost, not an isolated one. Groups inside a
    single executable can overlap on the device, so the slope is a throughput
    figure; the midpoint is measured too, and a slope that differs between the
    midpoint and the full count means the applications are overlapping and the
    number should not be read as one body's latency. Comparing two bodies at
    identical shapes through identical arms is what it is for.

    Returns a dict with the marginal cost at the midpoint and at the full
    count, the single-application call time (which still carries dispatch), and
    the worst run-to-run spread seen.
    """
    n = len(groups)
    if n < 4:
        raise ValueError("need >= 4 field groups: a midpoint arm is what "
                         "shows whether the applications overlap")
    mid = n // 2
    t1, lo1, hi1 = _median_ms(make_fn(1), groups, reps, warmup)
    tm, lom, him = _median_ms(make_fn(mid), groups, reps, warmup)
    tn, lon, hin = _median_ms(make_fn(n), groups, reps, warmup)
    return {
        "marginal_ms": round((tn - t1) / (n - 1), 4),
        "marginal_ms_midpoint": round((tm - t1) / (mid - 1), 4),
        "single_call_ms": round(t1, 4),
        "spread_ms": round(max(hi1 - lo1, him - lom, hin - lon), 4),
        "n_groups": n,
        "midpoint_groups": mid,
    }


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

    if args.groups < 4:
        raise SystemExit("--groups must be >= 4: the midpoint arm is what "
                         "shows whether the applications overlap on device")
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
        n_coll = _count(maker(1).lower(*groups).compile().as_text())
        # If the compiler merged the groups' collectives, the many-group
        # program would not carry N times the messages and the slope would not
        # be N applications of the body. Recorded rather than assumed.
        n_coll_all = _count(maker(args.groups).lower(*groups).compile().as_text())
        r = _time_one_body(maker, groups, args.reps, args.warmup)
        r["collectives"] = n_coll
        r["collectives_all_groups"] = n_coll_all
        r["collectives_scale"] = (round(n_coll_all / n_coll, 3)
                                  if n_coll else None)
        rows[name] = r
        print(f"  {name:>9}: {n_coll:3d} collectives "
              f"({n_coll_all} for {args.groups} groups)   "
              f"{r['marginal_ms']:8.4f} ms/application "
              f"(midpoint {r['marginal_ms_midpoint']:.4f}, "
              f"single call {r['single_call_ms']:7.3f} incl. dispatch, "
              f"spread {r['spread_ms']:6.3f})")

    rec = {
        "component": "latlon_tile_halo_micro",
        "p_lat": args.p_lat, "p_lon": args.p_lon,
        "n_lat": args.n_lat, "n_lon": args.n_lon, "nlev": args.nlev,
        "fields": args.fields, "halo": args.halo, "groups": args.groups,
        "reps": args.reps, "warmup": args.warmup,
        "backend": jax.default_backend(),
        # Not "dispatch subtracted": the arms are called with the same
        # number of buffers so the HOST marshalling cancels, but the output
        # allocation still scales with the number of applications, and the
        # applications may overlap on the device. This is a marginal cost for
        # comparing two bodies at identical shapes, not one body's latency.
        "estimator": "marginal cost per application, same buffer count in "
                     "every arm; see _time_one_body",
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
