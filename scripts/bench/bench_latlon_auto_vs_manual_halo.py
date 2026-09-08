"""Would letting the compiler place the halo beat writing it by hand?

The lat-lon lane's halo exchange is hand-written: a ``shard_map`` over the
latitude axis with explicit ``jax.lax.ppermute`` calls. JAX can instead be given
globally-shaped arrays with a sharding annotation and left to insert whatever
communication the program needs. That is less code to own. Whether it is faster,
slower, or a different program entirely is the question, and it is answerable
without porting the model.

WHAT IS AND IS NOT MEASURED. This is the halo PATTERN, not the model: a
five-point stencil, periodic in longitude, sharded across latitude, iterated a
fixed number of times. Nothing here touches the model, the production halo body,
or any configuration; it is a new file and reverting it is deleting it. The
model's real step adds a pole fold and thirteen exchanges per step, so a result
here bounds the shape of the answer rather than predicting the step time.

The two arms compute the SAME VALUES, which is checked rather than assumed:

  manual  shard_map over ("lat",) with ppermute for the two neighbour rows --
          the same MECHANISM the production lane uses, in its simplest case.
  auto    the identical stencil written on the global array under jit with a
          NamedSharding on the latitude axis, no shard_map and no ppermute --
          the compiler decides what to move.

The thing to look at first is not the time but the COLLECTIVES. Automatic
partitioning is documented to lower a sharded neighbour access to
collective-permute when it recognises the pattern, and to fall back to
all-gather when it does not. An all-gather of a latitude-sharded field is the
whole field on every device: that would be a different program with a different
scaling law, not a slower version of the same one. The census in the receipt
says which happened, and it is reported whether or not the timing is favourable.

Usage
-----
    srun ... python scripts/bench/bench_latlon_auto_vs_manual_halo.py \\
        --n-devices 64 --n-lat 2048 --n-lon 4096 --nlev 26 --multicontroller
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
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from legoesm.parallel.shard_map_compat import shard_map

AXIS = "lat"


#: Everything in this module that can move data BETWEEN devices. Point-to-point
#: send/recv and the collective-shaped custom calls are here because a census
#: that omits them can report "no communication" for a program that
#: communicates. ``dynamic-slice`` is deliberately absent: it is a layout
#: operation, and counting it among the collectives was inflating the total.
_COMM_KINDS = ("collective-permute", "all-gather", "all-reduce", "all-to-all",
               "reduce-scatter", "collective-broadcast", "send", "recv")


def _census(text):
    """Count communication SITES in a compiled module, by kind.

    Sites, not executions: a loop body appears once in the text however many
    times it runs. The per-application slope is what says an exchange is
    actually executed per iteration.
    """
    out = {}
    for kind in _COMM_KINDS:
        n = len(re.findall(rf"(?<![\w-]){kind}(?:-start)?\(", text))
        if n:
            out[kind] = n
    nccl = len(re.findall(r'custom-call\([^)]*\),\s*custom_call_target="[^"]*'
                          r'(?:nccl|collective)', text, re.I))
    if nccl:
        out["custom-call(collective)"] = nccl
    return out


def stencil(x, north, south):
    """Five-point stencil body, shared by both arms so they cannot drift.

    ``north``/``south`` are the neighbouring rows however they were obtained.
    Longitude is periodic and local to every device, so it uses a roll either
    way and is not part of what is being compared.
    """
    east = jnp.roll(x, 1, axis=1)
    west = jnp.roll(x, -1, axis=1)
    return 0.2 * (x + north + south + east + west)


def make_manual(mesh, n_dev, n_iters):
    """Hand-written halo: shard_map plus an explicit neighbour exchange."""
    # Row i's north neighbour is row i+1. For the LAST row that lives on the
    # device above, and what is needed from it is its FIRST row -- so every
    # device sends its first row DOWN, which is the pair (s, s-1). The mirror
    # holds going south. Getting this backwards produces a program that still
    # runs and still exchanges the right number of bytes, which is why the two
    # arms are checked against each other before anything is timed.
    send_first_down = [(s, (s - 1) % n_dev) for s in range(n_dev)]
    send_last_up = [(s, (s + 1) % n_dev) for s in range(n_dev)]

    def body(xl):
        def one(_, v):
            first_from_above = jax.lax.ppermute(v[:1], AXIS, send_first_down)
            last_from_below = jax.lax.ppermute(v[-1:], AXIS, send_last_up)
            nrow = jnp.concatenate([v[1:], first_from_above], axis=0)
            srow = jnp.concatenate([last_from_below, v[:-1]], axis=0)
            return stencil(v, nrow, srow)

        return jax.lax.fori_loop(0, n_iters, one, xl)

    try:
        sm = shard_map(body, mesh=mesh, in_specs=P(AXIS), out_specs=P(AXIS),
                       check_vma=False)
    except TypeError:  # pragma: no cover - older JAX spelling
        sm = shard_map(body, mesh=mesh, in_specs=P(AXIS), out_specs=P(AXIS),
                       check_rep=False)
    return jax.jit(sm)


def make_auto(mesh, n_iters):
    """Same stencil on the global array; the compiler places the halo."""
    sharding = NamedSharding(mesh, P(AXIS))

    def body(x):
        def one(_, v):
            # The global roll crosses the sharded axis. What the compiler emits
            # for that is the entire question.
            return stencil(v, jnp.roll(v, -1, axis=0), jnp.roll(v, 1, axis=0))

        return jax.lax.fori_loop(0, n_iters, one, x)

    return jax.jit(body, in_shardings=sharding, out_shardings=sharding)


def median_ms(fn, x, n_reps, warmup):
    out = fn(x)                      # bind `out` even when warmup is 0
    for _ in range(warmup):
        out = fn(x)
    jax.block_until_ready(out)
    ts = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        out = fn(x)
        jax.block_until_ready(out)
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts)


def slope_and_intercept(make_fn, x, iters, n_reps, warmup):
    """Per-application cost and fixed per-call cost, from a sweep.

    Dividing one call's time by the iteration count charges the whole of the
    host's dispatch to the loop body. Timing several iteration counts and
    fitting a line separates them: the SLOPE is what one application of the
    stencil-plus-exchange costs and the INTERCEPT is everything paid once per
    call.

    It also settles a question a static count of collectives cannot. Counting
    instructions in the compiled module says how many exchange SITES exist, not
    how many times they run, so an exchange hoisted out of the loop would look
    identical. If the slope of an arm collapses toward the cost of the stencil
    alone, its exchange is not being executed per iteration.

    Returns ``(slope_ms, intercept_ms, [(iters, ms)])``.
    """
    pts = []
    for n in iters:
        pts.append((n, median_ms(make_fn(n), x, n_reps, warmup)))
    ns = np.array([p[0] for p in pts], dtype=np.float64)
    ts = np.array([p[1] for p in pts], dtype=np.float64)
    slope, intercept = np.polyfit(ns, ts, 1)
    # An exchange hoisted out of the loop would leave the compiled module
    # looking identical and the time nearly flat in the iteration count. Doubling
    # the applications has to roughly double the work above the fixed cost, or
    # this arm is not running what it claims per iteration.
    lo, hi = pts[0], pts[-1]
    grew = (hi[1] - intercept) / max(lo[1] - intercept, 1e-9)
    expected = hi[0] / lo[0]
    if not (0.6 * expected <= grew <= 1.6 * expected):
        raise SystemExit(
            f"time did not scale with the application count: {lo[0]} -> "
            f"{hi[0]} applications grew {grew:.2f}x above the fixed cost, "
            f"expected about {expected:.2f}x. Something is being hoisted out "
            f"of the loop and the per-application number is meaningless.")
    return float(slope), float(intercept), pts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-devices", type=int, default=0)
    ap.add_argument("--n-lat", type=int, default=2048)
    ap.add_argument("--n-lon", type=int, default=4096)
    ap.add_argument("--nlev", type=int, default=26)
    ap.add_argument("--n-iters", type=int, default=10,
                    help="Stencil applications inside one jit call.")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--multicontroller", action="store_true")
    ap.add_argument("--coordinator", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.multicontroller:
        from legoesm.parallel.early_init import init_multicontroller_distributed
        init_multicontroller_distributed(args.coordinator)

    if args.n_iters < 1 or args.reps < 1 or args.warmup < 0:
        raise SystemExit("--n-iters and --reps must be >= 1 and --warmup >= 0; "
                         "zero iterations would benchmark no halo at all")
    devs = jax.devices()
    nd = args.n_devices or len(devs)
    if nd > len(devs):
        raise SystemExit(f"--n-devices {nd} > {len(devs)} visible devices")
    if nd < 2:
        raise SystemExit("needs >= 2 devices; a halo between one device and "
                         "itself measures nothing")
    if args.n_lat % nd:
        raise SystemExit(f"--n-lat {args.n_lat} must divide {nd} devices")
    mesh = Mesh(np.array(devs[:nd]), (AXIS,))
    sharding = NamedSharding(mesh, P(AXIS))

    shape = (args.n_lat, args.n_lon, args.nlev)
    # Built per addressable shard so no process ever holds the global array.
    # The generator is seeded FROM THE SHARD'S FIRST ROW rather than drawn from
    # one running stream: a single stream hands out values in whatever order
    # the runtime happens to ask for local shards, so the same global row would
    # get different data under a different process layout and no reported
    # number could be reproduced.
    def shard_of(index):
        sl = index[0]
        start = sl.start or 0
        stop = sl.stop if sl.stop is not None else args.n_lat
        g = np.random.default_rng((3, start)).standard_normal(
            (stop - start, args.n_lon, args.nlev))
        return jnp.asarray(g, dtype=jnp.float32)

    x = jax.make_array_from_callback(shape, sharding, shard_of)

    if x.sharding != sharding:
        raise SystemExit(
            f"input sharding is {x.sharding}, expected {sharding}. The manual "
            f"arm maps over it without checking, so a mismatch would compute "
            f"wrong values quietly.")

    manual = make_manual(mesh, nd, args.n_iters)
    auto = make_auto(mesh, args.n_iters)
    # If a refactor ever made the two arms the same construction, every number
    # below would still look fine and the identical communication counts would
    # read as corroboration when they were tautology.
    if manual.lower(x).compile().as_text() == auto.lower(x).compile().as_text():
        raise SystemExit(
            "the two arms compiled to an identical module; there is nothing "
            "being compared")

    # Same values, or the comparison is between two different programs. Both
    # arms call the same `stencil` and sum its five terms in the same order, so
    # the only source of disagreement is where the neighbour rows came from and
    # the expected difference is EXACTLY zero; the tolerances below exist to
    # survive a compiler choosing a different fusion, not a re-association
    # argument. Absolute and relative errors are reported separately because a
    # single relative number with a floor in its denominator hides which of the
    # two it is.
    #
    # The reduction runs ON DEVICE with a replicated output, so it is a maximum
    # over EVERY shard and not just the ones this process happens to hold.
    # Comparing local shards only would pass while a remote shard disagreed.
    a, m = auto(x), manual(x)

    @partial(jax.jit, out_shardings=NamedSharding(mesh, P()))
    def _worst(u, v):
        d = jnp.abs(u - v)
        return jnp.stack([jnp.max(d),
                          jnp.max(d / jnp.maximum(jnp.abs(v), 1e-30))])

    abs_err, rel_err = (float(t) for t in np.asarray(_worst(a, m)))
    if not (abs_err <= 1e-5 or rel_err <= 1e-5):
        raise SystemExit(
            f"the two arms do not agree after {args.n_iters} applications: "
            f"worst absolute difference {abs_err:.3e}, worst relative "
            f"{rel_err:.3e}. They are not the same program and no timing "
            f"comparison between them means anything.")

    rec = {
        "component": "latlon_auto_vs_manual_halo",
        "n_devices": nd, "n_processes": jax.process_count(),
        "n_lat": args.n_lat, "n_lon": args.n_lon, "nlev": args.nlev,
        "n_iters": args.n_iters, "reps": args.reps,
        "backend": jax.default_backend(),
        "worst_absolute_difference": abs_err,
        "worst_relative_difference": rel_err,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "arms": {},
    }
    sweep = sorted({1, max(2, args.n_iters // 2), args.n_iters,
                    2 * args.n_iters})
    for name, maker in (("manual", lambda n: make_manual(mesh, nd, n)),
                        ("auto", lambda n: make_auto(mesh, n))):
        fn = maker(args.n_iters)
        text = fn.lower(x).compile().as_text()
        ms = median_ms(fn, x, args.reps, args.warmup)
        slope, intercept, pts = slope_and_intercept(
            maker, x, sweep, args.reps, args.warmup)
        rec["arms"][name] = {
            "ms": round(ms, 4),
            "ms_per_application": round(slope, 5),
            "fixed_ms_per_call": round(intercept, 5),
            "sweep": [{"n_iters": n, "ms": round(t, 4)} for n, t in pts],
            # A count of exchange SITES in the module, not of executions: a
            # loop body appears once however many times it runs. The slope
            # above is what says the exchange is executed per application.
            "collective_sites": _census(text),
        }
    if jax.process_index() == 0:
        for name, r in rec["arms"].items():
            print(f"  {name:>6}: {r['ms_per_application']:9.5f} ms per "
                  f"application + {r['fixed_ms_per_call']:.4f} ms fixed  "
                  f"(one {args.n_iters}-application call: {r['ms']:.4f} ms)   "
                  f"sites={r['collective_sites']}")
        print(f"  agreement: worst absolute {abs_err:.3e}, "
              f"worst relative {rel_err:.3e}")
        print(json.dumps(rec))
        if args.out:
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            with open(args.out, "a") as f:
                f.write(json.dumps(rec) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
