"""How much of a shard_map ``ppermute`` can XLA:GPU hide behind
independent compute?

This is the capability gate for an interior/rim split — the thing
MPAS-Fortran, ICON and FV3 all do by hand: run the cells that need no
halo while the halo is in flight. legoESM cannot do it today (its
compiled step leaves 1-3 instructions between every
``collective-permute-start`` and its ``-done``), but that may be because
nothing in the step is independent of the exchange rather than because
XLA refuses. Building the split is multi-week work, so measure the
compiler's capability first, on a model-free microbenchmark.

WHAT IS MEASURED
----------------
One ``shard_map`` region issues a ``ppermute`` of a fixed payload and
runs ``n_flop`` rounds of a dependency-chained FMA on a second array the
collective never touches. The two results are returned SEPARATELY, so
no final reduction joins the collective's output to the independent
work. The control arm is the same program with the ``ppermute``
replaced by the identity, keeping the payload live and both reductions
present.

``delta = t(with collective) - t(same compute, no collective)`` is the
collective's UNHIDDEN cost. Sweeping ``n_flop`` from 0 upward moves the
independent compute from "much cheaper than the collective" to "much
more expensive"; how ``delta`` behaves across that sweep is the
measurement. Interpretation is deliberately NOT printed — this script
emits numbers, configuration and an HLO census, nothing else.

PROTOCOL (each one earned by a failure in this campaign)
--------------------------------------------------------
* The two arms are INTERLEAVED sample by sample, not run as two blocks.
  Block A followed by block B aliases node warm-up and clock drift into
  the subtraction, and this campaign has measured 7.7 % spread between
  arms that were byte-identical.
* Both arms are compiled and warmed BEFORE any timed sample.
* The paired delta is reported as a median over paired samples together
  with its inter-quartile range, so a delta smaller than the noise is
  visible as such instead of being read as a signal.
* The sweep must start at ``n_flop = 0`` — a custom ``--flops`` list
  that does not is refused rather than silently renormalised against an
  arbitrary row. Note what that row is and is NOT: it is the MINIMUM-
  independent-work reference, not a zero-independent-work one. The
  region still reduces the whole independent buffer there, so the sweep
  measures how much ADDITIONAL collective time is hidden as the FMA
  chain grows past that floor. It cannot support a claim about a step
  that contains no independent work at all.
* Every effective flag, version and device string is printed with the
  numbers, and the compiled module's async-collective census is printed
  next to them: a run whose ``XLA_FLAGS`` did not select the
  latency-hiding scheduler measures a different question, and the flags
  must be set BEFORE this process imports JAX.

Usage
-----
    # The scheduler A/B must move ONE flag. Hold pipelined-p2p FIXED in
    # both arms and toggle only the latency-hiding scheduler; replace
    # XLA_FLAGS, never append, and set it BEFORE this process starts.
    F="--xla_gpu_enable_pipelined_p2p=true"
    XLA_FLAGS="$F --xla_gpu_enable_latency_hiding_scheduler=true" \\
        srun -p gpu-devel --gpus-per-node=2 \\
        python scripts/validate/collective_compute_overlap.py
    XLA_FLAGS="$F --xla_gpu_enable_latency_hiding_scheduler=false" \\
        srun -p gpu-devel --gpus-per-node=2 \\
        python scripts/validate/collective_compute_overlap.py
"""
from __future__ import annotations

import argparse
import os
import re
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, PartitionSpec as P

from legoesm.parallel.shard_map_compat import shard_map


def _build(n_dev: int, payload: int, work: int, n_flop: int, comm: bool,
           n_col: int = 1):
    """Return ``(f, x, y)``.

    ``f`` returns the payload reduction and the independent-work
    reduction as SEPARATE outputs. Joining them in a final add would let
    XLA fuse the independent reduction under a collective-dependent
    instruction and report a false "no overlap".

    ``comm=False`` keeps the identical program with the ``ppermute``
    replaced by the identity — same payload, same two reductions, same
    FMA chain — so the difference between the arms is the collective and
    whatever codegen the collective itself forces.

    ``n_col`` splits the SAME total payload and the SAME total
    independent work into ``n_col`` collectives with one chunk of work
    between consecutive ones. This is the production shape: the MPAS
    step issues 11-13 coloured rounds that all carry ``channel_id=1``,
    so NCCL serialises them on one communicator stream and each window
    can only be filled by the work that is independent of THAT round.
    One fat collective with unlimited work behind it is the best case
    for the scheduler; ``n_col`` sweeps toward the real one.
    """
    mesh = Mesh(np.array(jax.devices()[:n_dev]), ("device",))

    def _perm(k):
        # Round k shifts by a DIFFERENT stride. Identical permutations
        # let XLA's collective combiner merge the rounds into one
        # collective (observed at n_dev=2, where the only non-trivial
        # shift is 1: the census reported n_start=1 for every n_col).
        # Production's coloured rounds each have their own
        # source_target_pairs, so distinct strides are the faithful
        # shape. Needs n_dev >= 3 to produce more than one stride.
        shift = 1 + (k % max(1, n_dev - 1))
        return [(i, (i + shift) % n_dev) for i in range(n_dev)]

    per_pay = payload // n_col
    per_work = work // n_col
    if per_pay == 0 or per_work == 0:
        raise SystemExit(
            f"n_col={n_col} leaves an empty chunk (payload {payload}, "
            f"work {work}); the sweep would not hold the totals fixed.")

    def body(x, y):
        pay_sum = jnp.zeros((), x.dtype)
        acc_sum = jnp.zeros((), y.dtype)
        for k in range(n_col):
            xk = x[k]
            if comm:
                xk = jax.lax.ppermute(xk, "device", perm=_perm(k))
            acc = y[k]
            for _ in range(n_flop):
                acc = acc * 1.0000001 + 1e-7
            pay_sum = pay_sum + xk.sum()
            acc_sum = acc_sum + acc.sum()
        # SEPARATE outputs: nothing downstream joins the collective's
        # result to the independent work.  Both are PER-DEVICE (shape
        # (1,) inside the region): the payload is device-distinct, so
        # promising a replica-equal P() output would be a false contract
        # that only survives because the checker is off.
        return pay_sum[None], acc_sum[None]

    f = jax.jit(shard_map(
        body, mesh=mesh, in_specs=(P("device"), P("device")),
        out_specs=(P("device"), P("device"))))
    # Device-distinct payload, so a misrouted permutation is visible in
    # the value rather than hidden by a uniform fill.
    x = (jnp.arange(n_dev, dtype=jnp.float32)[:, None, None]
         * jnp.ones((n_dev, n_col, per_pay), jnp.float32))
    y = jnp.ones((n_dev, n_col, per_work), jnp.float32)
    return f, x, y


def _hlo_census(f, x, y) -> dict:
    """Async-collective census of the COMPILED module: how many
    ``collective-permute-start`` it holds, and how many instructions sit
    between each start and its matching done (the window the scheduler
    actually left)."""
    txt = f.lower(x, y).compile().as_text()
    lines = txt.split("\n")
    starts, gaps = {}, []
    for i, line in enumerate(lines):
        m = re.match(r"\s*(%?[\w.\-]+) = .*collective-permute-start\(", line)
        if m:
            starts[m.group(1).lstrip("%")] = i
        m2 = re.match(
            r"\s*(%?[\w.\-]+) = .*collective-permute-done\((%?[\w.\-]+)\)",
            line)
        if m2:
            src = m2.group(2).lstrip("%")
            if src in starts:
                # instructions strictly BETWEEN the pair
                gaps.append(i - starts[src] - 1)
    return {
        "n_start": len(starts),
        "n_plain": len(re.findall(r"collective-permute(?!-start|-done)", txt)),
        "start_done_gaps": gaps,
    }


def _paired_times(f1, f0, x, y, reps: int, warmup: int):
    """Interleaved paired timing. Returns ``(t_comm, t_ctrl)`` arrays of
    equal length; sample ``k`` of each was taken back to back, so drift
    between them is bounded by one sample rather than by one block."""
    for _ in range(max(1, warmup)):
        jax.block_until_ready(f1(x, y))
        jax.block_until_ready(f0(x, y))
    t1, t0 = [], []
    for k in range(reps):
        # Alternate which arm is measured first so any residual
        # first-in-pair bias cancels between even and odd samples.
        order = (f1, t1, f0, t0) if k % 2 == 0 else (f0, t0, f1, t1)
        for fn, dst in ((order[0], order[1]), (order[2], order[3])):
            s = time.perf_counter()
            jax.block_until_ready(fn(x, y))
            dst.append((time.perf_counter() - s) * 1e3)
    return np.array(t1), np.array(t0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-devices", type=int, default=2)
    ap.add_argument("--payload", type=int, default=2_000_000,
                    help="floats exchanged per device (8 MB at 2e6)")
    ap.add_argument("--work", type=int, default=2_000_000,
                    help="floats in the independent FMA chain")
    ap.add_argument("--flops", type=int, nargs="+",
                    default=[0, 64, 256, 512, 1024, 2048, 4096],
                    help="FMA-chain lengths; MUST start at 0 (the "
                         "minimum-independent-work reference row)")
    ap.add_argument("--n-collectives", type=int, nargs="+", default=[1],
                    help="split the SAME total payload and work into this "
                         "many collectives (production is 11-13, all on "
                         "one channel)")
    ap.add_argument("--reps", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()

    if not args.flops or args.flops[0] != 0:
        raise SystemExit(
            f"--flops must start at 0 (the minimum-independent-work "
            f"reference row); got {args.flops}. Renormalising against an "
            f"arbitrary row would make every reported fraction "
            f"meaningless.")

    n = min(args.n_devices, jax.device_count())
    if n < 2:
        raise SystemExit(f"need >= 2 devices, have {jax.device_count()}")

    dev = jax.devices()[0]
    print(f"# jax={jax.__version__} device={dev.device_kind} "
          f"n_devices={n} platform={dev.platform}")
    print(f"# XLA_FLAGS={os.environ.get('XLA_FLAGS', '')!r}")
    print(f"# payload={args.payload} floats "
          f"({args.payload * 4 / 1e6:.1f} MB/device) work={args.work} "
          f"reps={args.reps} warmup={args.warmup} (arms interleaved)")

    print(f"{'n_col':>6} {'n_flop':>7} {'t_comm_ms':>10} {'t_ctrl_ms':>10} "
          f"{'delta_ms':>9} {'delta_iqr':>10} {'cp_start':>9} {'gaps':>20}")
    for nc in args.n_collectives:
      for nf in args.flops:
        f1, x, y = _build(n, args.payload, args.work, nf, comm=True,
                          n_col=nc)
        f0, _, _ = _build(n, args.payload, args.work, nf, comm=False,
                          n_col=nc)
        cen1 = _hlo_census(f1, x, y)
        cen0 = _hlo_census(f0, x, y)
        if cen0["n_start"] or cen0["n_plain"]:
            raise SystemExit(
                f"control arm compiled WITH a collective "
                f"({cen0}); the subtraction would not isolate the "
                f"collective. Refusing to report a number.")
        if not (cen1["n_start"] or cen1["n_plain"]):
            raise SystemExit(
                f"collective arm compiled WITHOUT a collective "
                f"({cen1}); the delta would be pure noise. Refusing to "
                f"report a number.")
        t1, t0 = _paired_times(f1, f0, x, y, args.reps, args.warmup)
        d = t1 - t0
        iqr = float(np.percentile(d, 75) - np.percentile(d, 25))
        print(f"{nc:6d} {nf:7d} {np.median(t1):10.3f} "
              f"{np.median(t0):10.3f} {np.median(d):9.3f} {iqr:10.3f} "
              f"{cen1['n_start']:9d} "
              f"{str(cen1['start_done_gaps'])[:20]:>20}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
